#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Ablation study: Direct vs Self-Consistency vs Direct+Code Extraction vs QTree QA.

Four generation strategies are compared at three sampling settings:
  - 1 stage  : a single sampled stage
  - 2 stages : two sampled stages (multi-stage comparison)
  - 3 stages : three sampled stages (multi-stage comparison)

Strategies:
  - Direct generation
      One LLM call. The model is given the *raw EDA log text* for each
      sampled stage (read from ``<design>_<stage>_extract.txt`` next to the
      aggregate metrics JSON) and is asked to output a list of {question,
      answer} pairs directly. When a raw extract is missing on disk, the
      Direct strategy transparently falls back to the structured metrics
      JSON for that stage. No code is generated, no extraction is executed,
      and no cross-validation is done.

  - Self-Consistency generation (consistency-vote filter)
      Takes the QA pairs produced by the Direct strategy in the SAME cell
      as its input pool — it does NOT generate new candidates and it does
      NOT regenerate answers. For each Direct QA pair the LLM is asked M
      times whether the proposed answer is consistent with the *raw EDA
      log text* for the same stages (the same input Direct used) — a
      YES/NO judgment. A pair is kept iff the number of YES votes is >=
      `--sc-min-consistent-votes` (default = strict majority of M, i.e.
      floor(M/2)+1 — for M=3 that means at least 2 of 3 YES votes). The
      kept pair's answer is Direct's answer, unchanged. SC therefore
      depends on Direct: requesting it without Direct in the same run
      will auto-include Direct.

  - Direct + Code Extraction (DCE) ablation
      Hybrid of Direct and QTree. Takes Direct's *questions* and, for each
      one, generates Python extraction code against the structured *metrics
      JSON* (not raw logs). The code is executed and the code-extracted
      answer is (optionally) validated against Direct's original answer via
      an LLM judge. This isolates how much of QTree's quality comes from
      the code-extraction step alone, vs QTree's own question generation.
      Like SC, DCE depends on Direct and will auto-include it.

  - QTree-driven generation
      The existing multi-step pipeline:
          1. LLM generates questions over the metrics.
          2. LLM generates Python extraction code per question.
          3. Code is executed against the metrics to produce a code answer.
          4. LLM generates a direct answer; the code answer is validated against
             it and low-similarity pairs are rejected.
      This is `single_stage_qa_pair` for the 1-stage setting and
      `multi_stage_qa_pair` for 2/3-stage settings.

For each (strategy, num_stages) cell we track:
  - elapsed wall-clock time
  - number of LLM calls and prompt/output tokens
  - number of QA pairs returned (valid only)
  - rejection statistics (QTree and DCE)
  - the QA pairs themselves (for spot-checking)
  - Vendi Score on questions / answers / Q+A for semantic diversity
    (Friedman & Dieng 2023; defined as exp of Shannon entropy of the
    eigenvalues of the n x n normalized cosine-similarity matrix of the
    sentence embeddings; bounded in [1, n], higher = more diverse)

A per-run JSON dump and a console summary table are produced.

Re-score saved JSON (no new LLM calls)::

    python src/score_experiment_results.py experiment_results/direct_vs_qtree/<summary>.json

Example
-------
    python src/experiment_direct_vs_qtree_qa.py \
        --metrics extracted_logs/ariane133_nan45_all_stages_metrics.json \
        --n-runs 1 \
        --temperature 0.7 \
        --output-dir experiment_results/direct_vs_qtree

Defaults pick the first existing `*_all_stages_metrics.json` under
`extracted_logs/` so the script can be invoked with no arguments.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import dataclass, field, asdict, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import qa_generation as _qa_generation  # noqa: E402
from qa_generation import (  # noqa: E402
    OpenRoadBackground,
    QAGenerationError,
    RejectionStats,
    _extract_code_block,
    _normalize_aggregate,
    _read_json,
    _tracked_llm_call,
    generate_answer_directly_through_llm,
    get_qa_generation_stats,
    multi_stage_qa_pair,
    reset_qa_generation_stats,
    sample_stages,
    sanitize_for_json,
    single_stage_qa_pair,
    validate_direct_vs_code_answer,
    validate_extract_answer,
)
from embedding_utils import (  # noqa: E402
    filter_json_by_similarity,
    get_embedding_model,
)
from stage_parsing_examples import (  # noqa: E402
    get_multi_stage_parsing_examples,
    get_parsing_examples,
)


DEFAULT_STAGE_COUNTS = (1, 2, 3)
DEFAULT_N_QUESTIONS = 20
DEFAULT_DUPLICATE_COSINE_THRESHOLD = 0.98

# Stage types this study focuses on. Synthesis, floorplan, and final are
# excluded by default: synthesis is structurally very different from the
# physical-design stages, and floorplan/final are noisier as comparison
# anchors. Placement / CTS / Routing form the core PPA-progression window.
DEFAULT_FOCUS_STAGE_TYPES: tuple[str, ...] = ("placement", "cts", "routing")

# Fixed stage compositions used by the default `stage_selection="fixed"` mode.
# This removes the seeded-random variance from the experiment so all that's
# being measured is the QA-generation method. Override per `num_stages` from
# the CLI with `--fixed-1stage`, `--fixed-2stages`, `--fixed-3stages`.
DEFAULT_FIXED_STAGE_SETS: Dict[int, List[str]] = {
    1: ["routing"],
    2: ["placement", "routing"],
    3: ["placement", "cts", "routing"],
}


# ---------------------------------------------------------------------------
# Vendi Score for semantic diversity.
# ---------------------------------------------------------------------------
#
# Vendi Score (Friedman & Dieng, 2023, https://arxiv.org/abs/2210.02410):
#
#     VS(K) = exp( H(eigvals(K / n)) )
#
# where K is an n x n positive semi-definite similarity matrix with 1's on the
# diagonal (here, cosine similarity of L2-normalized sentence embeddings) and
# H is Shannon entropy with the 0 log 0 = 0 convention. Bounds:
#     1 <= VS <= n
# All items identical -> VS = 1; all items pairwise orthogonal -> VS = n.
# We also report VS / n in (0, 1] to make cells with different counts
# comparable.


def _vendi_score_from_similarity(sim: np.ndarray) -> float:
    """Compute Vendi Score given an n x n PSD similarity matrix with 1's on
    the diagonal."""
    n = sim.shape[0]
    if n <= 1:
        return float(n)

    sim = (sim + sim.T) / 2.0
    eigvals = np.linalg.eigvalsh(sim) / float(n)
    eigvals = np.clip(eigvals, 0.0, None)
    total = eigvals.sum()
    if total <= 0.0:
        return 1.0
    eigvals = eigvals / total

    entropy = 0.0
    for lam in eigvals:
        if lam > 1e-12:
            entropy -= lam * math.log(lam)
    return float(math.exp(entropy))


def compute_vendi(texts: List[str], embedding_model) -> Dict[str, Any]:
    """Compute Vendi Score for a list of texts via sentence embeddings.

    Returns a dict with `count`, `vendi`, and `vendi_normalized` (= vendi / n).
    Empty/singleton inputs degenerate to vendi = count.
    """
    cleaned = [t for t in (texts or []) if isinstance(t, str) and t.strip()]
    n = len(cleaned)
    if n == 0:
        return {"count": 0, "vendi": 0.0, "vendi_normalized": 0.0}
    if n == 1:
        return {"count": 1, "vendi": 1.0, "vendi_normalized": 1.0}

    try:
        emb = embedding_model.encode(cleaned)
    except Exception as exc:  # noqa: BLE001
        return {
            "count": n,
            "vendi": None,
            "vendi_normalized": None,
            "error": f"encode_failed: {type(exc).__name__}: {exc}",
        }

    if emb is None:
        return {
            "count": n,
            "vendi": None,
            "vendi_normalized": None,
            "error": "embedding_model returned None (keyword-only backend)",
        }

    emb = np.asarray(emb, dtype=np.float64)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    emb = emb / norms

    sim = emb @ emb.T
    np.fill_diagonal(sim, 1.0)
    sim = np.clip(sim, -1.0, 1.0)

    vs = _vendi_score_from_similarity(sim)
    return {"count": n, "vendi": vs, "vendi_normalized": vs / n}


def _answer_to_text(answer: Any) -> str:
    """Flatten any answer value to a single string for embedding."""
    if answer is None:
        return ""
    if isinstance(answer, str):
        return answer
    try:
        return json.dumps(answer, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(answer)


def compute_qa_diversity(
    qa_pairs: List[Dict[str, Any]],
    embedding_model,
) -> Dict[str, Any]:
    """Compute Vendi Scores for questions, answers, and the combined Q+A text."""
    questions = [_clean_question_text(qa.get("question", "")) for qa in qa_pairs]
    answers = [_answer_to_text(qa.get("answer")) for qa in qa_pairs]
    combined = [
        f"Q: {q}\nA: {a}".strip()
        for q, a in zip(questions, answers)
        if (q or a)
    ]
    return {
        "questions": compute_vendi(questions, embedding_model),
        "answers": compute_vendi(answers, embedding_model),
        "qa_combined": compute_vendi(combined, embedding_model),
    }


# Stage keywords for question-text detection (case-insensitive).
# Canonical names match experiment input stages: placement, routing, cts,
# floorplan, final.  OpenROAD sub-phrases map to the parent canonical name:
#   global route / detailed route / global routing / detailed routing → routing
#   global placement / detailed placement → placement
_STAGE_DETECTORS: List[tuple[str, re.Pattern[str]]] = [
    ("floorplan", re.compile(r"\bfloorplan\b", re.IGNORECASE)),
    (
        "placement",
        re.compile(
            r"\b(?:global\s+)?placement\b|\bdetailed\s+placement\b|\bplacement\b",
            re.IGNORECASE,
        ),
    ),
    (
        "cts",
        re.compile(r"\bcts\b|\bclock\s+tree(?:\s+synthesis)?\b", re.IGNORECASE),
    ),
    (
        "routing",
        re.compile(
            r"\b(?:global\s+|detailed\s+)?(?:route|routing)\b"
            r"|\bglobal\s+route\b|\bdetailed\s+route\b"
            r"|\bglobal\s+routing\b|\bdetailed\s+routing\b",
            re.IGNORECASE,
        ),
    ),
    (
        "final",
        re.compile(
            r"\bfinal(?:\s+stage|\s+design)?\b"
            r"|\b(?:in|at|after)\s+final\b"
            r"|\bfinal\b",
            re.IGNORECASE,
        ),
    ),
]

_COMPARATIVE_RE = re.compile(
    r"\b(?:compare|comparison|compared|change[ds]?|progression|progress(?:ed|ing)?|"
    r"improv(?:e|ed|ement)|degrad(?:e|ed|ation)|evolution|evolve[ds]?|"
    r"from\s+.+\s+to\s+|between\s+.+\s+and\s+|vs\.?|versus|across\s+(?:the\s+)?stages?)\b",
    re.IGNORECASE,
)

_QUESTION_TAG_RE = re.compile(r"^\s*<Q\d+>\s*", re.IGNORECASE)


def _clean_question_text(question: Any) -> str:
    """Normalize question strings (strip Q-tree tags / difficulty lines)."""
    if not isinstance(question, str):
        return str(question or "")
    text = question.strip()
    text = _QUESTION_TAG_RE.sub("", text)
    if "\nDifficulty:" in text:
        text = text.split("\nDifficulty:")[0].strip()
    return text


def _detect_stages_in_text(text: str) -> set[str]:
    """Return canonical stage names mentioned in ``text``."""
    found: set[str] = set()
    for stage_name, pattern in _STAGE_DETECTORS:
        if pattern.search(text):
            found.add(stage_name)
    return found


def _effective_input_stage_count(n_input: int) -> int:
    """Map detected input-stage count for scoring.

    If no stage keyword is found (``n_input == 0``), count as **1**: the
    question is still generated from a single-stage log context and is
    treated as implicitly about that stage.  Explicit mentions keep their
    count (1, 2, or 3).
    """
    return 1 if n_input == 0 else n_input


def compute_qa_quality(
    qa_pairs: List[Dict[str, Any]],
    *,
    input_stage_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Structural QA quality metrics (complements embedding Vendi score).

  Stage coverage counts how many of the **selected input stages** for this
  cell (1, 2, or 3 types from ``selected_stages_by_num_stages``) appear in
  each question, detected by regex on canonical names (placement, cts, …).

  Primary scalar: ``mean_input_stages_mentioned`` — average **effective**
  count per question.  Effective count = detected keyword count, except
  **0 keywords → 1** (implicit single-stage question).  Action space is
  therefore 1 … num_input_stages for the cell.  ``multi_%`` still uses
  **raw** detected count ≥ 2 (must explicitly name two input stages).

  Raw counts are stored in ``num_input_stages_mentioned`` / ``pct_zero_*``
  for auditing.
    """
    input_stages = [s.lower() for s in (input_stage_types or []) if s]
    input_set = set(input_stages)
    num_input = len(input_set)

    per_pair: List[Dict[str, Any]] = []
    lengths_chars: List[int] = []
    lengths_words: List[int] = []
    input_stages_per_q: List[int] = []
    effective_stages_per_q: List[int] = []

    for qa in qa_pairs or []:
        qtext = _clean_question_text(qa.get("question", ""))
        mentioned_any = _detect_stages_in_text(qtext)
        mentioned_input = mentioned_any & input_set if input_set else mentioned_any
        n_input = len(mentioned_input)
        n_effective = _effective_input_stage_count(n_input)
        input_stages_per_q.append(n_input)
        effective_stages_per_q.append(n_effective)
        lengths_chars.append(len(qtext))
        lengths_words.append(len(qtext.split()) if qtext else 0)

        coverage_frac = n_input / num_input if num_input > 0 else None
        per_pair.append(
            {
                "stages_mentioned_any": sorted(mentioned_any),
                "stages_mentioned_input": sorted(mentioned_input),
                "num_input_stages_mentioned": n_input,
                "effective_input_stages_mentioned": n_effective,
                "num_stages_mentioned_any": len(mentioned_any),
                "input_stage_coverage": coverage_frac,
                "is_multi_stage": n_input >= 2,
                "is_comparative": bool(_COMPARATIVE_RE.search(qtext)),
                "question_chars": len(qtext),
                "question_words": len(qtext.split()) if qtext else 0,
            }
        )

    n = len(per_pair)
    if n == 0:
        return {
            "count": 0,
            "input_stage_types": input_stages,
            "num_input_stages": num_input,
            "mean_input_stages_mentioned": 0.0,
            "mean_stages_mentioned": 0.0,
            "pct_multi_stage": 0.0,
            "pct_comparative": 0.0,
            "mean_input_stage_coverage": None,
            "pct_zero_input_stages": 0.0,
            "mean_question_chars": 0.0,
            "mean_question_words": 0.0,
            "per_stage_mention_rate": {},
            "histogram_input_stages_mentioned": {},
        }

    # Histogram of effective counts (1 … num_input; no 0 bucket after zero→1 rule).
    hist_raw: Dict[str, int] = {}
    for k in input_stages_per_q:
        key = str(k)
        hist_raw[key] = hist_raw.get(key, 0) + 1
    hist: Dict[str, int] = {}
    for k in effective_stages_per_q:
        key = str(min(k, num_input)) if num_input else str(k)
        hist[key] = hist.get(key, 0) + 1

    per_stage_rates: Dict[str, float] = {}
    for stage in input_stages:
        hits = sum(1 for p in per_pair if stage in p["stages_mentioned_input"])
        per_stage_rates[stage] = hits / n

    coverages = [p["input_stage_coverage"] for p in per_pair if p["input_stage_coverage"] is not None]
    mean_cov = float(sum(coverages) / len(coverages)) if coverages else None
    mean_input_count = float(sum(effective_stages_per_q) / n)

    return {
        "count": n,
        "input_stage_types": input_stages,
        "num_input_stages": num_input,
        "zero_keywords_count_as": 1,
        "mean_input_stages_mentioned": mean_input_count,
        "mean_stages_mentioned": mean_input_count,
        "pct_multi_stage": 100.0 * sum(1 for p in per_pair if p["is_multi_stage"]) / n,
        "pct_comparative": 100.0 * sum(1 for p in per_pair if p["is_comparative"]) / n,
        "mean_input_stage_coverage": mean_cov,
        "pct_zero_input_stages": 100.0 * sum(1 for k in input_stages_per_q if k == 0) / n,
        "pct_implicit_single_stage": 100.0 * sum(1 for k in input_stages_per_q if k == 0) / n,
        "pct_using_all_input_stages": (
            100.0 * sum(1 for k in input_stages_per_q if k == num_input) / n
            if num_input > 0
            else None
        ),
        "histogram_input_stages_mentioned_raw": hist_raw,
        "mean_question_chars": float(sum(lengths_chars) / n),
        "mean_question_words": float(sum(lengths_words) / n),
        "min_question_chars": int(min(lengths_chars)),
        "max_question_chars": int(max(lengths_chars)),
        "per_stage_mention_rate": per_stage_rates,
        "histogram_input_stages_mentioned": hist,
        "histogram_stages_mentioned": hist,
        "per_pair": per_pair,
    }


def compute_qa_metrics(
    qa_pairs: List[Dict[str, Any]],
    embedding_model,
    *,
    input_stage_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Vendi diversity + structural quality metrics for one QA list."""
    out = compute_qa_diversity(qa_pairs, embedding_model)
    out["quality"] = compute_qa_quality(qa_pairs, input_stage_types=input_stage_types)
    return out


def _pairs_for_vendi_scoring(
    qa_pairs: List[Dict[str, Any]],
    strategy: str,
    *,
    sc_use_kept_only: bool,
) -> List[Dict[str, Any]]:
    """Pick which QA rows to use for Vendi (SC may filter by ``kept``)."""
    if strategy != "self_consistency":
        return list(qa_pairs)
    if not sc_use_kept_only:
        return list(qa_pairs)
    if not any("kept" in qa for qa in qa_pairs):
        return list(qa_pairs)
    kept = [qa for qa in qa_pairs if qa.get("kept")]
    return kept if kept else list(qa_pairs)


def score_run_cell(
    run: Dict[str, Any],
    embedding_model,
    *,
    input_stage_types: Optional[List[str]] = None,
    recompute_vendi: bool = True,
    sc_use_kept_only: bool = True,
) -> Dict[str, Any]:
    """Recompute ``diversity`` (+ ``quality``) for one ``runs[]`` entry from JSON."""
    strategy = run.get("strategy", "")
    qa_pairs = run.get("qa_pairs") or []

    stages = input_stage_types
    if not stages:
        stages = [s.get("stage") for s in run.get("stages_info", []) if s.get("stage")]

    vendi_pairs = _pairs_for_vendi_scoring(
        qa_pairs, strategy, sc_use_kept_only=sc_use_kept_only,
    )

    if recompute_vendi and embedding_model is not None and vendi_pairs:
        diversity = compute_qa_metrics(
            vendi_pairs, embedding_model, input_stage_types=stages,
        )
    elif recompute_vendi and not vendi_pairs:
        diversity = {"quality": compute_qa_quality([], input_stage_types=stages)}
    else:
        diversity = dict(run.get("diversity") or {})
        diversity["quality"] = compute_qa_quality(qa_pairs, input_stage_types=stages)

    if strategy == "self_consistency" and any("kept" in qa for qa in qa_pairs):
        kept_pairs = [qa for qa in qa_pairs if qa.get("kept")]
        diversity["quality_kept"] = compute_qa_quality(
            kept_pairs, input_stage_types=stages,
        )
        diversity["quality_all_candidates"] = diversity.get("quality")

    run["diversity"] = diversity
    if strategy == "self_consistency" and run.get("sc_stats"):
        run["sc_stats"] = _enrich_sc_stats(dict(run["sc_stats"]))
    return run


def score_experiment_summary(
    summary: Dict[str, Any],
    embedding_model=None,
    *,
    recompute_vendi: bool = True,
    sc_use_kept_only: bool = True,
) -> Dict[str, Any]:
    """Score every cell in a saved experiment summary JSON."""
    selected = summary.get("selected_stages_by_num_stages") or {}
    for run in summary.get("runs") or []:
        ns = str(run.get("num_stages", ""))
        stage_entries = selected.get(ns) or selected.get(int(run.get("num_stages", 0)), [])
        input_stages = [s.get("stage") for s in stage_entries if s.get("stage")]
        if not input_stages:
            input_stages = [s.get("stage") for s in run.get("stages_info", []) if s.get("stage")]
        score_run_cell(
            run,
            embedding_model,
            input_stage_types=input_stages,
            recompute_vendi=recompute_vendi,
            sc_use_kept_only=sc_use_kept_only,
        )
    run_metrics = [run_dict_to_metrics(r) for r in summary.get("runs") or []]
    summary["strategy_averages"] = aggregate_strategy_averages(
        run_metrics,
        embedding_model,
        duplicate_report=summary.get("qa_duplicate_report"),
        sc_kept_only=sc_use_kept_only,
    )
    summary["scoring"] = {
        "recomputed_at": datetime.now().isoformat(timespec="seconds"),
        "recompute_vendi": recompute_vendi,
        "sc_use_kept_only": sc_use_kept_only,
        "metrics": ["vendi", "quality", "strategy_averages"],
    }
    return summary


def run_dict_to_metrics(run: Dict[str, Any]) -> RunMetrics:
    """Hydrate a ``RunMetrics`` row from a JSON ``runs[]`` dict."""
    known = {f.name for f in fields(RunMetrics)}
    return RunMetrics(**{k: v for k, v in run.items() if k in known})


# ---------------------------------------------------------------------------
# Direct QA generation (single-shot, no code, no validation).
# ---------------------------------------------------------------------------

_DIRECT_SYSTEM_PROMPT = (
    "You are an expert in chip design and EDA tools. You generate concise, "
    "specific question/answer pairs about OpenROAD stage metrics. You answer "
    "ONLY from the context provided and always return strict JSON."
)


def _design_stem_for(aggregate_metrics_json_path: str | Path) -> str:
    """Return the design stem used to locate sibling extract / metric files."""
    return Path(aggregate_metrics_json_path).stem.replace(
        "_all_stages_metrics", ""
    )


def _load_raw_stage_text(
    aggregate_metrics_json_path: str | Path,
    stage_name: str,
    *,
    max_chars: Optional[int] = None,
) -> Optional[str]:
    """Read the raw EDA log extract for one stage.

    Looks for ``<design>_<stage>_extract.txt`` next to the aggregate metrics
    JSON. Returns ``None`` when the file is missing or unreadable.
    """
    if not stage_name:
        return None
    folder = Path(aggregate_metrics_json_path).parent
    candidate = folder / f"{_design_stem_for(aggregate_metrics_json_path)}_{stage_name.lower()}_extract.txt"
    if not candidate.exists():
        return None
    try:
        text = candidate.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if max_chars is not None and len(text) > max_chars:
        # Keep the head and tail; logs are usually most informative at both
        # ends (command + final summary) so middle-truncation is safer than
        # tail truncation alone.
        head = text[: max_chars // 2]
        tail = text[-max_chars // 2 :]
        text = (
            f"{head}\n"
            f"... [truncated {len(text) - max_chars} chars from the middle] ...\n"
            f"{tail}"
        )
    return text


def _gather_stage_raw_texts(
    aggregate_metrics_json_path: str | Path,
    stages: List[Dict[str, Any]],
    *,
    max_chars_per_stage: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """For each selected stage, attach its raw log text (or note missing)."""
    out: List[Dict[str, Any]] = []
    for s in stages:
        stage_name = s.get("stage") or ""
        raw = _load_raw_stage_text(
            aggregate_metrics_json_path, stage_name, max_chars=max_chars_per_stage,
        )
        out.append(
            {
                "stage": stage_name,
                "occurrence": s.get("occurrence", 1),
                "sub_design_name": s.get("sub_design_name"),
                "raw_text": raw,
                "raw_text_available": raw is not None,
                "metrics_fallback": s.get("metrics"),
            }
        )
    return out


def _build_direct_prompt(
    stages_with_raw: List[Dict[str, Any]],
    design_name: str,
    n_questions: int,
) -> str:
    """Prompt for the direct (single-shot) baseline using raw stage logs.

    Each item in ``stages_with_raw`` is expected to carry ``raw_text`` (the
    extracted EDA log lines for that stage). When ``raw_text`` is missing we
    fall back to the structured metrics JSON for that stage so the run never
    silently degrades.
    """
    stage_label = ", ".join(
        f"{s.get('stage')} (occurrence {s.get('occurrence', 1)})"
        for s in stages_with_raw
    )

    sections: List[str] = []
    for s in stages_with_raw:
        stage = s.get("stage", "?")
        occ = s.get("occurrence", 1)
        if s.get("raw_text"):
            sections.append(
                f"## Stage: {stage} (occurrence {occ}) — raw EDA log\n"
                f"```text\n{s['raw_text']}\n```"
            )
        else:
            fallback = s.get("metrics_fallback") or {}
            metrics_str = json.dumps(fallback, indent=2)
            sections.append(
                f"## Stage: {stage} (occurrence {occ}) — metrics JSON "
                f"(fallback; raw log not found on disk)\n"
                f"```json\n{metrics_str}\n```"
            )
    body = "\n\n".join(sections) if sections else "(no stages provided)"

    return f"""You are generating question-answer pairs for an OpenROAD EDA design flow benchmark.

# OpenROAD background:
{OpenRoadBackground}

# Design: {design_name}
# Stages: {stage_label}

# Raw stage logs (one section per stage):
{body}

Task: Produce up to {n_questions} high-quality, SPECIFIC question/answer pairs
that can be answered DIRECTLY from the raw stage logs above. Each answer must
be a concrete value (number with unit when applicable, boolean, short string,
list, or small dict). Do NOT speculate. If a fact is not present in the raw
logs, do not ask about it.

Rules:
- Use natural stage names ("synthesis", "floorplan", "placement", "CTS",
  "routing", "final"). Never use raw substage ids like "3_5_place_dp".
- One topic per question.
- Prefer quantitative answers (numbers with units) over qualitative ones.
- Quote units exactly as they appear in the raw logs (e.g. "u", "um", "ns",
  "ps", "%", "MHz").
- If multiple stages are provided, at least half of the questions should be
  comparative across them.

Output STRICT JSON (no markdown fences, no commentary) of the form:
{{
  "qa_pairs": [
    {{"question": "...", "answer": <value>}},
    ...
  ]
}}
"""


def _parse_direct_response(response: str) -> List[Dict[str, Any]]:
    """Best-effort parse of the direct LLM response into qa_pairs."""
    if not response:
        return []

    text = response.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return []
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []

    raw_pairs = data.get("qa_pairs") if isinstance(data, dict) else None
    if not isinstance(raw_pairs, list):
        return []

    cleaned: List[Dict[str, Any]] = []
    for item in raw_pairs:
        if not isinstance(item, dict):
            continue
        q = item.get("question")
        a = item.get("answer")
        if not isinstance(q, str) or not q.strip():
            continue
        if a is None or (isinstance(a, str) and a.strip().lower() in {"", "n/a", "none", "null"}):
            continue
        cleaned.append({"question": q.strip(), "answer": sanitize_for_json(a)})
    return cleaned


# ---------------------------------------------------------------------------
# Centralized stage selection.
# ---------------------------------------------------------------------------
#
# All three strategies must see the SAME sampled stage(s) for a given
# (metrics_file, num_stages, seed). Doing the sampling once at the runner
# level — and threading the result into each strategy — removes the silent
# divergence that would otherwise occur (e.g. `single_stage_qa_pair` does
# `rng.choice(candidates)` rather than calling `sample_stages`, so without
# pinning it would pick a different stage than direct/SC).
#
# Two selection modes are supported:
#   - "random"  : delegate to `sample_stages` (seeded random). Default,
#                 reproducible, but stage richness is uncontrolled.
#   - "richest" : pick the top-K stages by serialized `metrics` size — a
#                 fast and robust proxy for "how much context_data is in
#                 this stage". Useful when you want to guarantee the LLM
#                 has substantial material to ask about, and to remove
#                 luck-of-the-seed variance from the experiment.


def _stage_richness(stage_entry: Dict[str, Any]) -> int:
    """Score a stage by the serialized size of its `metrics` field (bytes).

    Larger == more 'context_data'. We serialize to JSON once; the cost is
    negligible for the small number of candidate stages.
    """
    metrics = stage_entry.get("metrics")
    if not metrics:
        return 0
    try:
        return len(json.dumps(metrics, ensure_ascii=False, default=str))
    except (TypeError, ValueError):
        return len(str(metrics))


def _select_richest_stages(
    aggregate_metrics_json_path: str | Path,
    *,
    num_stages: int,
    allowed_stage_types: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Top-K stages by `_stage_richness` (largest metrics blob wins).

    When `allowed_stage_types` is provided, only stages whose `stage` field
    is in that set are eligible.
    """
    payload = _normalize_aggregate(_read_json(aggregate_metrics_json_path))
    allowed_set = (
        {st.lower() for st in allowed_stage_types}
        if allowed_stage_types
        else None
    )

    candidates: List[Dict[str, Any]] = []
    for s in payload.get("stages") or []:
        if not isinstance(s, dict):
            continue
        if not s.get("metrics"):
            continue
        if allowed_set is not None:
            stype = (s.get("stage") or "").lower()
            if stype not in allowed_set:
                continue
        candidates.append(s)

    if len(candidates) < num_stages:
        raise QAGenerationError(
            f"Not enough stages with metrics for richness selection. "
            f"Need {num_stages}, have {len(candidates)} "
            f"(allowed_stage_types={sorted(allowed_set) if allowed_set else 'any'})."
        )
    ranked = sorted(candidates, key=_stage_richness, reverse=True)
    chosen = ranked[:num_stages]
    chosen.sort(key=lambda s: s.get("sequence_number", 0))
    return chosen


def _select_fixed_stages(
    aggregate_metrics_json_path: str | Path,
    *,
    stage_types: List[str],
) -> List[Dict[str, Any]]:
    """Pick exactly one occurrence per stage type in `stage_types`.

    Picks within a single sub-design when possible (so the comparison is
    intra-design, not cross-design). Tie-breaks on combined richness. Within
    the chosen sub-design, the richest occurrence per stage type wins. The
    result is sorted by `sequence_number` to preserve flow order.
    """
    from collections import defaultdict

    payload = _normalize_aggregate(_read_json(aggregate_metrics_json_path))
    all_stages = [
        s
        for s in (payload.get("stages") or [])
        if isinstance(s, dict) and s.get("metrics")
    ]
    if not all_stages:
        raise QAGenerationError("No stages with metrics found in payload.")

    desired = [t.lower() for t in stage_types]
    n = len(desired)

    # Bucket by sub-design (None / empty -> common bucket "").
    by_subdesign: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in all_stages:
        by_subdesign[s.get("sub_design_name") or ""].append(s)

    def _score(stages: List[Dict[str, Any]]) -> tuple:
        types_present = {(s.get("stage") or "").lower() for s in stages}
        match_count = sum(1 for t in desired if t in types_present)
        richness = sum(
            _stage_richness(s)
            for s in stages
            if (s.get("stage") or "").lower() in desired
        )
        return (match_count, richness)

    ranked_subdesigns = sorted(
        by_subdesign.items(), key=lambda kv: _score(kv[1]), reverse=True,
    )
    best_sd, best_pool = ranked_subdesigns[0]
    match_count, _ = _score(best_pool)

    if match_count < n:
        # Fall back: search the global pool across sub-designs to fill gaps.
        chosen: List[Dict[str, Any]] = []
        used_ids = set()
        for t in desired:
            candidates = [s for s in all_stages if (s.get("stage") or "").lower() == t]
            candidates.sort(key=_stage_richness, reverse=True)
            picked = next(
                (c for c in candidates if id(c) not in used_ids), None,
            )
            if picked is None:
                raise QAGenerationError(
                    f"No candidate stage with metrics for required type '{t}'."
                )
            used_ids.add(id(picked))
            chosen.append(picked)
        chosen.sort(key=lambda s: s.get("sequence_number", 0))
        return chosen

    # Pick richest occurrence per type within best sub-design.
    chosen = []
    for t in desired:
        candidates = [
            s for s in best_pool if (s.get("stage") or "").lower() == t
        ]
        candidates.sort(key=_stage_richness, reverse=True)
        chosen.append(candidates[0])
    chosen.sort(key=lambda s: s.get("sequence_number", 0))
    return chosen


def _select_stages(
    aggregate_metrics_json_path: str | Path,
    *,
    num_stages: int,
    seed: Optional[int],
    mode: str = "fixed",
    allowed_stage_types: Optional[List[str]] = None,
    fixed_stage_sets: Optional[Dict[int, List[str]]] = None,
) -> List[Dict[str, Any]]:
    """Canonical, shared stage selection used by all three strategies.

    Modes
    -----
    fixed   (default): pick exactly the stage types declared for `num_stages`
            in `fixed_stage_sets`. Removes seed-driven variance. Use this
            when you want the experiment to compare *methods* on a fixed
            set of stages, not seeded random samples.
    richest : top-K stages by serialized metrics size, restricted by
              `allowed_stage_types`.
    random  : seeded random via `sample_stages`, restricted by
              `allowed_stage_types`. Subject to `min_stage_separation`
              feasibility (which is what just blew up on ariane133).
    """
    if mode == "fixed":
        sets = fixed_stage_sets if fixed_stage_sets is not None else DEFAULT_FIXED_STAGE_SETS
        if num_stages not in sets:
            raise ValueError(
                f"No fixed_stage_sets entry for num_stages={num_stages}; "
                f"have {sorted(sets.keys())}."
            )
        return _select_fixed_stages(
            aggregate_metrics_json_path,
            stage_types=list(sets[num_stages]),
        )
    if mode == "richest":
        return _select_richest_stages(
            aggregate_metrics_json_path,
            num_stages=num_stages,
            allowed_stage_types=allowed_stage_types,
        )
    if mode != "random":
        raise ValueError(f"Unknown stage_selection mode: {mode!r}")
    stage_types = list(allowed_stage_types) if allowed_stage_types else None
    if num_stages == 1:
        return sample_stages(
            aggregate_metrics_json_path,
            num_stages=2,
            seed=seed,
            stage_types=stage_types,
        )[:1]
    return sample_stages(
        aggregate_metrics_json_path,
        num_stages=num_stages,
        seed=seed,
        stage_types=stage_types,
    )


def _call_multi_stage_pinned(
    aggregate_metrics_json_path: str | Path,
    *,
    pinned_stages: List[Dict[str, Any]],
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run `multi_stage_qa_pair` pinned to `pinned_stages`.

    `multi_stage_qa_pair` calls `sample_stages` internally to choose what to
    compare. To force it to use the *exact* stages the runner already chose
    (especially in "richest" mode), we temporarily replace the module-level
    `sample_stages` reference inside `qa_generation`. The patch is scoped to
    a single call via try/finally.

    This is preferable to forking `multi_stage_qa_pair` itself, since it
    keeps `src/qa_generation.py` untouched.
    """
    original = _qa_generation.sample_stages

    def _patched_sample_stages(*_args, **_kw):
        return list(pinned_stages)

    _qa_generation.sample_stages = _patched_sample_stages
    try:
        return multi_stage_qa_pair(aggregate_metrics_json_path, **kwargs)
    finally:
        _qa_generation.sample_stages = original


def _stages_to_payload(stages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "stage": s.get("stage"),
            "occurrence": s.get("occurrence"),
            "sub_design_name": s.get("sub_design_name"),
            "metrics": s.get("metrics"),
        }
        for s in stages
    ]


def _stages_to_info(stages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "stage": s.get("stage"),
            "occurrence": s.get("occurrence"),
            "sub_design_name": s.get("sub_design_name"),
            "sequence_number": s.get("sequence_number"),
        }
        for s in stages
    ]


def direct_qa_pair(
    aggregate_metrics_json_path: str | Path,
    *,
    num_stages: int,
    n_questions: int = DEFAULT_N_QUESTIONS,
    seed: Optional[int] = None,
    temperature: float = 0.7,
    stages: Optional[List[Dict[str, Any]]] = None,
    max_chars_per_stage: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Direct baseline: ask the LLM to emit QA pairs in a single call, with no
    intermediate question/code/validation steps.

    The Direct strategy uses the raw per-stage EDA log text (the
    ``<design>_<stage>_extract.txt`` files that sit next to the aggregate
    metrics JSON). When a stage's raw extract is missing on disk the
    strategy transparently falls back to the structured metrics JSON for
    that stage so the run never silently degrades.

    If `stages` is provided, it is used as-is (the canonical path from the
    runner). Otherwise stages are sampled internally for backwards
    compatibility.
    """
    payload = _normalize_aggregate(_read_json(aggregate_metrics_json_path))
    design_name = payload.get("design_name") or _design_stem_for(
        aggregate_metrics_json_path
    )

    if stages is None:
        stages = _select_stages(
            aggregate_metrics_json_path, num_stages=num_stages, seed=seed,
        )

    stages_with_raw = _gather_stage_raw_texts(
        aggregate_metrics_json_path,
        stages,
        max_chars_per_stage=max_chars_per_stage,
    )
    missing = [s["stage"] for s in stages_with_raw if not s["raw_text_available"]]
    if missing:
        print(
            "[Direct] WARNING: raw extract files missing for stage(s) "
            f"{missing}; falling back to metrics JSON for those stages."
        )
    raw_char_summary = ", ".join(
        f"{s['stage']}={len(s['raw_text']) if s['raw_text'] else 0} chars"
        for s in stages_with_raw
    )
    print(f"[Direct] Input source = raw EDA logs ({raw_char_summary})")
    
    prompt = _build_direct_prompt(stages_with_raw, design_name, n_questions)
    
    response = _tracked_llm_call(
        system_prompt=_DIRECT_SYSTEM_PROMPT,
        user_prompt=prompt,
        temperature=temperature,
    )

    qa_pairs = _parse_direct_response(response)

    return {
        "design_name": design_name,
        "strategy": "direct",
        "num_stages": num_stages,
        "stages_info": _stages_to_info(stages),
        "input_source": {
            "kind": "raw_extract_text",
            "stages": [
                {
                    "stage": s["stage"],
                    "occurrence": s["occurrence"],
                    "raw_text_available": s["raw_text_available"],
                    "raw_text_chars": len(s["raw_text"]) if s["raw_text"] else 0,
                }
                for s in stages_with_raw
            ],
        },
        "qa_pairs": qa_pairs,
        "raw_response": response,
    }


# ---------------------------------------------------------------------------
# QTree-driven wrappers (delegate to the existing pipeline).
# ---------------------------------------------------------------------------


def qtree_qa_pair(
    aggregate_metrics_json_path: str | Path,
    *,
    num_stages: int,
    seed: Optional[int] = None,
    temperature: float = 0.7,
    rejection_stats: Optional[RejectionStats] = None,
    stages: Optional[List[Dict[str, Any]]] = None,
    n_questions: int = DEFAULT_N_QUESTIONS,
) -> Dict[str, Any]:
    """Run the existing QTree-driven pipeline for the requested stage count.

    When `stages` is provided, the underlying pipeline is pinned to the same
    selection used by the direct / self-consistency strategies:

    - For `num_stages == 1` we pass `stage=` and `occurrence=` to
      `single_stage_qa_pair` (which does not use `sample_stages`).
    - For `num_stages >= 2` we pass `stage_types` and `seed` to
      `multi_stage_qa_pair`; combined with the deterministic `sample_stages`,
      this reproduces the same selection the runner just made.
    """
    if num_stages == 1:
        kwargs: Dict[str, Any] = {
            "seed": seed,
            "temperature": temperature,
            "rejection_stats": rejection_stats,
            "n_questions": n_questions,
        }
        if stages:
            chosen = stages[0]
            kwargs["stage"] = chosen.get("stage")
            occ = chosen.get("occurrence")
            if occ is not None:
                try:
                    kwargs["occurrence"] = int(occ)
                except (TypeError, ValueError):
                    pass
        result = single_stage_qa_pair(aggregate_metrics_json_path, **kwargs)
        stages_info = [
            {
                "stage": result.get("stage"),
                "occurrence": result.get("occurrence"),
                "sub_design_name": result.get("sub_design_name"),
            }
        ]
    else:
        kwargs = {
            "num_stages": num_stages,
            "seed": seed,
            "temperature": temperature,
            "rejection_stats": rejection_stats,
            "n_questions": n_questions,
        }
        if stages:
            kwargs["stage_types"] = list({s.get("stage") for s in stages if s.get("stage")})
            result = _call_multi_stage_pinned(
                aggregate_metrics_json_path,
                pinned_stages=stages,
                **kwargs,
            )
        else:
            result = multi_stage_qa_pair(aggregate_metrics_json_path, **kwargs)
        stages_info = result.get("stages_info", [])

    return {
        "design_name": result.get("design_name") or result.get("sub_design_name"),
        "strategy": "qtree",
        "num_stages": num_stages,
        "stages_info": stages_info,
        "qa_pairs": result.get("qa_pairs", []),
        "raw_result": result,
    }


# ---------------------------------------------------------------------------
# Direct + Code Extraction (DCE) ablation.
# ---------------------------------------------------------------------------
#
# Takes Direct's **questions** and, for each one, generates Python code to
# extract the answer from the structured stage *metrics JSON* (not raw logs).
# Optionally validates the code-extracted answer against Direct's original
# answer using an LLM judge.
#
# This is the hybrid between Direct (question generation) and QTree (code-
# based answer extraction + validation).  It isolates how much of QTree's
# quality comes from the code-extraction step alone.


_DCE_CODE_SYSTEM_PROMPT = (
    "You are an expert Python programmer specializing in data extraction "
    "from JSON structures."
)


def _build_dce_code_prompt_single(
    question: str,
    stage_name: str,
    stage_json_str: str,
    stage_parsing_examples: str,
) -> str:
    """Code-gen prompt for one Direct question (single-stage DCE)."""
    return f"""You are generating Python code to extract an answer from stage metrics JSON.

Stage: {stage_name}
Question: {question}

Metrics (available as variable `stage_entry` - already a Python dict):
```json
{stage_json_str}
```

Write Python code that extracts the answer to the question from the `stage_entry` dict.

Requirements:
- The variable `stage_entry` is already loaded as a Python dict (NOT a string)
- Access substages: substages = stage_entry['metrics'].get('substages', {{}})
- Access optimization_process: for opt in substage_data.get('optimization_process', [])
- Access design_metrics_final: design_metrics_final = stage_entry['metrics'].get('design_metrics_final', {{}})
- Return answers with appropriate units (e.g., "-0.234 ns", "12345.67 um")

Define the extract_answer function with stage-specific parsing guidance:
```python
def extract_answer(stage_entry: Dict[str, Any], questions: list[str]) -> list:
    \"\"\"
    Extract answers from stage metrics.

    Stage-specific parsing patterns:
{stage_parsing_examples}
    \"\"\"
    metrics = stage_entry.get('metrics', {{}})
    substages = metrics.get('substages', {{}})
    design_metrics_final = metrics.get('design_metrics_final', {{}})

    qa_pairs = []
    for question in questions:
        answer = None
        question_lower = question.lower()

        # Add extraction logic based on the question
        # Use the parsing patterns from the docstring above

        qa_pairs.append({{"question": question, "answer": answer}})

    return qa_pairs
```

CRITICAL RULES:
1. Read the question carefully to determine what type of answer is expected
2. For "how many" questions, return an integer count
3. For "what is the value" questions, return the value with units (e.g., "-0.234 ns")
4. For "what are" or "list" questions, return a list or dict
5. For comparison/progression questions, return a dict with all relevant values
6. Never return None, True/False (unless yes/no question), or empty values

IMPORTANT METRIC EXTRACTION HINT:
When a question mentions a stage WITHOUT specifying a substage:
- Look for the LAST substage within that stage that reports the relevant metrics
- Prefer metrics from "rpt_metrics" field if available (more accurate for timing)
- If "rpt_metrics" not available, use "metrics" field
- The last substage typically represents the final state of that stage

Output ONLY the Python code, nothing else (no markdown, no explanations)."""


def _build_dce_code_prompt_multi(
    question: str,
    stage_desc: str,
    stages_json_str: str,
    multi_stage_parsing_examples: str,
) -> str:
    """Code-gen prompt for one Direct question (multi-stage DCE)."""
    return f"""You are generating Python code to extract answers from multiple stage metrics.

Stages: {stage_desc}
Question: {question}

Metrics (available as variable `stages` - already a Python list of dicts):
```json
{stages_json_str}
```

Write Python code that extracts answers to the questions from the `stages` list.

Requirements:
- The variable `stages` is a list of stage dicts, each with 'stage', 'occurrence', 'sub_design_name', and 'metrics'
- Access substages: substages = stage['metrics'].get('substages', {{}})
- Access optimization_process: for opt in substage_data.get('optimization_process', [])
- Compare metrics across stages as needed
- Include units for numeric values
- IMPORTANT: Use stage names (e.g., 'placement', 'cts', 'routing') as dictionary keys, NOT 'stage1', 'stage2'

Define the extract_answer function with multi-stage parsing guidance:
```python
def extract_answer(stages: List[Dict[str, Any]], questions: list[str]) -> list:
    \"\"\"
    Extract answers from multiple stage metrics.

    ### Multi-stage parsing example patterns:
{multi_stage_parsing_examples}
    \"\"\"
    qa_pairs = []

    for question in questions:
        answer = None
        question_lower = question.lower()

        # Add extraction logic based on the question
        # Use the parsing patterns from the docstring above
        # Always use stage names as dictionary keys in answers

        qa_pairs.append({{"question": question, "answer": answer}})

    return qa_pairs
```

CRITICAL RULES:
1. Read the question carefully to determine what type of answer is expected
2. For "how many" questions, return an integer count
3. For "what is the value" questions, return the value with units (e.g., "-0.234 ns")
4. For "what are" or "list" questions, return a list or dict
5. For comparison/progression questions, return a dict with all relevant values
6. Never return None, True/False (unless yes/no question), or empty values
7. Use stage names from stage['stage'] as dictionary keys, NOT 'stage1', 'stage2'

Output ONLY the Python code, nothing else (no markdown, no explanations)."""


def _exec_extract_code(
    code: str,
    context_vars: Dict[str, Any],
    question: str,
) -> Optional[Dict[str, Any]]:
    """Execute LLM-generated extraction code; return the first QA pair or None."""
    local_vars = {**context_vars, "json": json}
    try:
        exec(code, {"json": json}, local_vars)  # noqa: S102
    except Exception as exc:
        print(f"    [DCE exec ERROR] {exc}")
        return None
    fn = local_vars.get("extract_answer")
    if fn is None:
        print("    [DCE exec ERROR] code did not define extract_answer()")
        return None
    try:
        data_arg = context_vars.get("stage_entry") or context_vars.get("stages")
        qa_pairs = fn(data_arg, [question])
    except Exception as exc:
        print(f"    [DCE exec ERROR] extract_answer() raised: {exc}")
        return None
    if not qa_pairs:
        return None
    return qa_pairs[0]


def direct_code_extraction_qa_pair(
    aggregate_metrics_json_path: str | Path,
    *,
    num_stages: int,
    direct_qa_pairs: List[Dict[str, Any]],
    stages: Optional[List[Dict[str, Any]]] = None,
    seed: Optional[int] = None,
    temperature: float = 0.7,
    validate_against_direct: bool = True,
    validation_threshold: float = 0.8,
    rejection_stats: Optional[RejectionStats] = None,
    embedding_model=None,
) -> Dict[str, Any]:
    """Ablation: Direct's questions + code-based answer extraction on metrics JSON.

    For each Direct QA pair:
      1. Generate Python code to extract the answer from the structured metrics.
      2. Execute the code.
      3. (Optional) Validate the code answer against Direct's original answer
         via an LLM judge (``validate_against_direct=True``).
      4. Record per-pair diagnostics.

    Returns a result dict compatible with the experiment runner.
    """
    payload = _normalize_aggregate(_read_json(aggregate_metrics_json_path))
    design_name = payload.get("design_name") or _design_stem_for(
        aggregate_metrics_json_path
    )
    all_stages = payload.get("stages", [])

    if stages is None:
        stages = _select_stages(
            aggregate_metrics_json_path, num_stages=num_stages, seed=seed,
        )

    # Build context for code generation: stage metrics JSON
    selected_stage_entries: List[Dict[str, Any]] = []
    for s in stages:
        stage_name = s.get("stage")
        occ = s.get("occurrence", 1)
        for entry in all_stages:
            if (
                entry.get("stage") == stage_name
                and entry.get("occurrence") == occ
            ):
                selected_stage_entries.append(entry)
                break
        else:
            for entry in all_stages:
                if entry.get("stage") == stage_name:
                    selected_stage_entries.append(entry)
                    break

    if not selected_stage_entries:
        raise QAGenerationError(
            f"DCE: could not find metrics for stages {stages} in {aggregate_metrics_json_path}"
        )

    is_single = num_stages == 1

    if is_single:
        stage_entry = selected_stage_entries[0]
        st = stage_entry.get("stage", "unknown")
        stage_json_str = json.dumps(stage_entry, indent=2, default=str)
        include_rpt = "rpt_metric" in stage_json_str
        parsing_examples = get_parsing_examples(st, include_rpt_metrics=include_rpt)
    else:
        stages_json = [
            {
                "stage": e.get("stage"),
                "occurrence": e.get("occurrence"),
                "sub_design_name": e.get("sub_design_name"),
                "metrics": e.get("metrics"),
            }
            for e in selected_stage_entries
        ]
        stages_json_str = json.dumps(stages_json, indent=2, default=str)
        include_rpt = "rpt_metric" in stages_json_str
        parsing_examples = get_multi_stage_parsing_examples(
            selected_stage_entries, include_rpt_metrics=include_rpt,
        )
        stage_desc = ", ".join(
            f"{e.get('stage')}#{e.get('occurrence', 1)}" for e in selected_stage_entries
        )

    print(
        f"[DCE] {len(direct_qa_pairs)} Direct questions → code extraction "
        f"({'single' if is_single else 'multi'}-stage, "
        f"validate={validate_against_direct}, thr={validation_threshold})"
    )

    final_qa_pairs: List[Dict[str, Any]] = []
    dce_diagnostics: List[Dict[str, Any]] = []

    for idx, dqa in enumerate(direct_qa_pairs):
        question = _clean_question_text(dqa.get("question", ""))
        direct_answer = dqa.get("answer")
        if not question.strip():
            continue

        diag: Dict[str, Any] = {
            "pair_index": idx,
            "question": question,
            "direct_answer": direct_answer,
            "code_answer": None,
            "extraction_code": None,
            "code_exec_ok": False,
            "validation_score": None,
            "kept": False,
            "drop_reason": None,
        }

        # Step 1: Generate extraction code
        if is_single:
            code_prompt = _build_dce_code_prompt_single(
                question, st, stage_json_str, parsing_examples,
            )
        else:
            code_prompt = _build_dce_code_prompt_multi(
                question, stage_desc, stages_json_str, parsing_examples,
            )

        code_response = _tracked_llm_call(
            system_prompt=_DCE_CODE_SYSTEM_PROMPT,
            user_prompt=code_prompt,
            temperature=0.1,
        )
        code = _extract_code_block(code_response)
        diag["extraction_code"] = code

        # Step 2: Execute the code
        if is_single:
            exec_result = _exec_extract_code(
                code, {"stage_entry": stage_entry}, question,
            )
        else:
            exec_result = _exec_extract_code(
                code, {"stages": stages_json}, question,
            )

        if exec_result is None or exec_result.get("answer") is None:
            diag["drop_reason"] = "code_exec_failed"
            if rejection_stats:
                rejection_stats.record_code_failure()
            dce_diagnostics.append(diag)
            print(f"  Q{idx+1}: code execution failed / no answer")
            continue

        code_answer = sanitize_for_json(exec_result.get("answer"))
        diag["code_answer"] = code_answer
        diag["code_exec_ok"] = True

        if not validate_extract_answer(code_answer):
            diag["drop_reason"] = "invalid_code_answer"
            if rejection_stats:
                rejection_stats.record_invalid_answer()
            dce_diagnostics.append(diag)
            print(f"  Q{idx+1}: code answer invalid (None/n/a)")
            continue

        # Step 3: Optionally validate code answer against Direct's answer
        if validate_against_direct and direct_answer is not None:
            score = validate_direct_vs_code_answer(
                str(direct_answer), code_answer, question,
            )
            diag["validation_score"] = score
            if score < validation_threshold:
                diag["drop_reason"] = "low_validation_score"
                if rejection_stats:
                    rejection_stats.record_low_score(
                        direct_answer=str(direct_answer),
                        code_answer=code_answer,
                        score=score,
                        question=question,
                    )
                dce_diagnostics.append(diag)
                print(f"  Q{idx+1}: validation score {score:.2f} < {validation_threshold} — dropped")
                continue
            print(f"  Q{idx+1}: validation score {score:.2f} — kept")
        else:
            print(f"  Q{idx+1}: no validation — kept")

        diag["kept"] = True
        final_qa_pairs.append({
            "question": question,
            "answer": code_answer,
            "direct_answer": direct_answer,
            "extraction_code": code,
            "validation_score": diag["validation_score"],
        })
        if rejection_stats:
            rejection_stats.record_success()

        dce_diagnostics.append(diag)

    total = len([d for d in dce_diagnostics if d["question"]])
    kept = len(final_qa_pairs)
    code_ok = sum(1 for d in dce_diagnostics if d["code_exec_ok"])
    print(
        f"[DCE] Done: {kept}/{total} kept "
        f"(code_ok={code_ok}, dropped={total - kept})"
    )

    dce_stats = {
        "candidates_total": total,
        "code_exec_ok": code_ok,
        "code_exec_failed": sum(1 for d in dce_diagnostics if d["drop_reason"] == "code_exec_failed"),
        "invalid_code_answer": sum(1 for d in dce_diagnostics if d["drop_reason"] == "invalid_code_answer"),
        "low_validation_score": sum(1 for d in dce_diagnostics if d["drop_reason"] == "low_validation_score"),
        "kept_after_validation": kept,
        "retention_rate": 100.0 * kept / total if total > 0 else 0.0,
        "validate_against_direct": validate_against_direct,
        "validation_threshold": validation_threshold,
    }

    return {
        "design_name": design_name,
        "strategy": "direct_code_extraction",
        "num_stages": num_stages,
        "stages_info": _stages_to_info(stages),
        "qa_pairs": final_qa_pairs,
        "dce_stats": dce_stats,
        "dce_diagnostics": dce_diagnostics,
    }


# ---------------------------------------------------------------------------
# Self-Consistency QA generation.
# ---------------------------------------------------------------------------
#
# Pipeline (no answer regeneration, no code execution):
#   1. INPUT: the QA pairs produced by the Direct strategy in the same cell.
#   2. Use every Direct QA pair as-is (only drop exact duplicate question
#      strings). No semantic embedding dedupe — 6 Direct pairs → 6 votes.
#   3. For each pair, call the LLM M times at `temperature`
#      asking — given the metrics JSON — whether the existing answer is
#      consistent with that JSON. Each call returns YES / NO + a short
#      rationale.
#   4. Keep the pair iff the number of YES votes is >= `min_consistent_votes`
#      (default: strict majority of M, i.e. floor(M/2)+1 — for M=3 that is
#      at least 2 of 3 YES votes).
#   5. The kept pair's answer is Direct's ORIGINAL answer, unchanged. SC
#      never invents a new answer; it only filters Direct's output.


def _normalize_string(s: str) -> str:
    return " ".join(s.lower().split())


def _dedupe_questions(
    qa_candidates: List[Dict[str, Any]],
    embedding_model,
    *,
    dedup_cosine_thr: float,
) -> List[Dict[str, Any]]:
    """Greedy cosine-clustering on questions; keep the centroid representative.

    Cluster size becomes `votes` on the returned QA pair. The first-seen
    answer in each cluster is retained as the candidate answer (it is
    re-validated by the answer-vote step downstream).
    """
    questions = [qa.get("question", "") for qa in qa_candidates if qa.get("question")]
    if not questions:
        return []
    if embedding_model is None:
        # No embeddings: fall back to exact normalized-string dedupe.
        seen: Dict[str, Dict[str, Any]] = {}
        for qa in qa_candidates:
            key = _normalize_string(qa.get("question", ""))
            if not key:
                continue
            if key in seen:
                seen[key]["votes"] += 1
            else:
                seen[key] = {**qa, "votes": 1}
        return list(seen.values())

    try:
        emb = embedding_model.encode(questions)
    except Exception:  # noqa: BLE001
        emb = None
    if emb is None:
        return [{**qa, "votes": 1} for qa in qa_candidates if qa.get("question")]

    emb = np.asarray(emb, dtype=np.float64)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    emb = emb / norms

    cluster_ids: List[int] = []
    cluster_emb: List[np.ndarray] = []
    for v in emb:
        best_id = -1
        best_sim = -1.0
        for i, c in enumerate(cluster_emb):
            sim = float(np.dot(v, c))
            if sim > best_sim:
                best_sim = sim
                best_id = i
        if best_id >= 0 and best_sim >= dedup_cosine_thr:
            cluster_ids.append(best_id)
        else:
            cluster_ids.append(len(cluster_emb))
            cluster_emb.append(v)

    clusters: Dict[int, List[Dict[str, Any]]] = {}
    valid = [qa for qa in qa_candidates if qa.get("question")]
    for cid, qa in zip(cluster_ids, valid):
        clusters.setdefault(cid, []).append(qa)

    deduped: List[Dict[str, Any]] = []
    for members in clusters.values():
        rep = members[0]
        deduped.append({**rep, "votes": len(members)})
    deduped.sort(key=lambda x: -x.get("votes", 1))
    return deduped


def _dedupe_exact_duplicate_questions(
    qa_candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Drop only byte-identical questions (after normalization); keep order.

    Self-Consistency on Direct's output uses this instead of semantic
  clustering — each Direct pair is validated individually so 6 Direct
  pairs stay 6 unless two questions normalize to the same string.
    """
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for qa in qa_candidates:
        qtext = _clean_question_text(qa.get("question", ""))
        if not qtext.strip():
            continue
        key = _normalize_string(qtext)
        if key in seen:
            continue
        seen.add(key)
        out.append({**qa, "votes": 1})
    return out


def _build_sc_consistency_prompt(
    stages_with_raw: List[Dict[str, Any]],
    design_name: str,
    question: str,
    answer: Any,
) -> str:
    """Yes/no consistency-check prompt for an existing QA pair.

    SC validates against the SAME raw EDA log text that Direct used to
    generate the pair. Each stage section embeds the raw extract, falling
    back to the structured metrics JSON only when the extract file is
    missing on disk. The LLM must return
    ``{"consistent": true|false, "reason": "..."}``.
    """
    stage_label = ", ".join(
        f"{s.get('stage')} (occurrence {s.get('occurrence', 1)})"
        for s in stages_with_raw
    )
    sections: List[str] = []
    for s in stages_with_raw:
        stage = s.get("stage", "?")
        occ = s.get("occurrence", 1)
        if s.get("raw_text"):
            sections.append(
                f"## Stage: {stage} (occurrence {occ}) — raw EDA log\n"
                f"```text\n{s['raw_text']}\n```"
            )
        else:
            fallback = s.get("metrics_fallback") or {}
            metrics_str = json.dumps(fallback, indent=2)
            sections.append(
                f"## Stage: {stage} (occurrence {occ}) — metrics JSON "
                f"(fallback; raw log not found on disk)\n"
                f"```json\n{metrics_str}\n```"
            )
    body = "\n\n".join(sections) if sections else "(no stages provided)"

    answer_str = _answer_to_text(answer) if not isinstance(answer, str) else answer

    return f"""You are auditing ONE question-answer pair against OpenROAD stage logs.

# OpenROAD background:
{OpenRoadBackground}

# Design: {design_name}
# Stages: {stage_label}

# Raw stage logs (one section per stage):
{body}

# Question: {question}
# Proposed answer: {answer_str}

Decide whether the Proposed answer is CONSISTENT with the Raw stage logs for
the Question. "Consistent" means:
  - numeric values match the logs within reasonable rounding/unit tolerance,
  - units, signs, and qualitative claims match what the logs support,
  - the answer is fully derivable from the logs (no fabricated facts),
  - if the fact is not in the logs, the answer must say so to be consistent.

Return STRICT JSON (no markdown, no commentary):
{{"consistent": true_or_false, "reason": "<one short sentence>"}}
"""


def _parse_sc_consistency_response(response: str) -> Dict[str, Any]:
    """Parse {"consistent": bool, "reason": str}. Returns dict with keys:

    - ``vote`` : ``True`` / ``False`` / ``None`` (None when parsing failed
      or the field is missing/ambiguous).
    - ``reason`` : a short string, possibly empty.
    """
    out: Dict[str, Any] = {"vote": None, "reason": ""}
    if not response:
        return out
    text = response.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return out
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return out
    if not isinstance(data, dict):
        return out

    raw = data.get("consistent")
    if isinstance(raw, bool):
        out["vote"] = raw
    elif isinstance(raw, str):
        low = raw.strip().lower()
        if low in {"true", "yes", "y", "1"}:
            out["vote"] = True
        elif low in {"false", "no", "n", "0"}:
            out["vote"] = False
    elif isinstance(raw, (int, float)):
        out["vote"] = bool(raw)

    reason = data.get("reason")
    if isinstance(reason, str):
        out["reason"] = reason.strip()[:300]
    return out


def _vote_consistency(
    votes: List[Dict[str, Any]],
    *,
    min_consistent_votes: int,
    sc_m: int,
) -> Dict[str, Any]:
    """Tally YES/NO consistency votes and decide whether to keep the pair.

    Returns a dict with: ``kept`` (bool), ``yes_votes``, ``no_votes``,
    ``parse_failed``, ``threshold``, ``consistent_rate`` (yes / sc_m).
    """
    yes_votes = sum(1 for v in votes if v.get("vote") is True)
    no_votes = sum(1 for v in votes if v.get("vote") is False)
    parse_failed = sum(1 for v in votes if v.get("vote") is None)
    consistent_rate = float(yes_votes / sc_m) if sc_m > 0 else 0.0
    kept = yes_votes >= min_consistent_votes
    return {
        "kept": kept,
        "yes_votes": yes_votes,
        "no_votes": no_votes,
        "parse_failed": parse_failed,
        "threshold": min_consistent_votes,
        "consistent_rate": consistent_rate,
    }




def _enrich_sc_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    """Add retention / vote-pass rates to Self-Consistency stats.

    - ``retention_rate`` / ``success_rate``: kept / Direct input count
      (what users expect in the summary table).
    - ``vote_pass_rate``: kept / count after question dedupe.
    """
    total = int(stats.get("candidates_total") or 0)
    dedup = int(stats.get("candidates_after_dedupe") or 0)
    kept = int(stats.get("kept_after_vote") or 0)
    stats["dropped_by_dedupe"] = max(0, total - dedup)
    stats["dropped_by_vote"] = max(0, dedup - kept)
    stats["retention_rate"] = (100.0 * kept / total) if total > 0 else 0.0
    stats["vote_pass_rate"] = (100.0 * kept / dedup) if dedup > 0 else 0.0
    stats["success_rate"] = stats["retention_rate"]
    return stats


def self_consistency_qa_pair(
    aggregate_metrics_json_path: str | Path,
    *,
    num_stages: int,
    seed: Optional[int],
    temperature: float,
    embedding_model,
    direct_qa_pairs: List[Dict[str, Any]],
    sc_m: int = 3,
    sc_dedup_threshold: float = 0.85,
    sc_min_consistent_votes: Optional[int] = None,
    stages: Optional[List[Dict[str, Any]]] = None,
    max_chars_per_stage: Optional[int] = None,
) -> Dict[str, Any]:
    """Self-Consistency QA as a *filter* on Direct's QA pairs.

    For each QA pair from ``direct_qa_pairs`` the LLM is asked ``sc_m`` times
    whether the proposed answer is consistent with the supplied metrics JSON.
    A pair is kept iff the number of YES votes is >= ``sc_min_consistent_votes``
    (defaults to strict majority of ``sc_m`` = floor(M/2)+1).

    The answer of every kept pair is exactly Direct's original answer; SC
    never regenerates an answer.
    """
    payload = _normalize_aggregate(_read_json(aggregate_metrics_json_path))
    design_name = payload.get("design_name") or Path(aggregate_metrics_json_path).stem.replace(
        "_all_stages_metrics", ""
    )

    if stages is None:
        stages = _select_stages(
            aggregate_metrics_json_path, num_stages=num_stages, seed=seed,
        )

    stages_with_raw = _gather_stage_raw_texts(
        aggregate_metrics_json_path,
        stages,
        max_chars_per_stage=max_chars_per_stage,
    )
    missing = [s["stage"] for s in stages_with_raw if not s["raw_text_available"]]
    if missing:
        print(
            "[SC] WARNING: raw extract files missing for stage(s) "
            f"{missing}; falling back to metrics JSON for those stages."
        )

    threshold = (
        sc_min_consistent_votes
        if sc_min_consistent_votes is not None
        else (sc_m // 2) + 1
    )
    threshold = max(1, min(threshold, sc_m))

    candidate_pool: List[Dict[str, Any]] = list(direct_qa_pairs or [])
    print(
        f"  [SC input] direct_qa_pairs={len(candidate_pool)} "
        f"M={sc_m} threshold={threshold}/{sc_m}"
    )

    empty_stats = {
        "candidates_total": 0,
        "candidates_after_dedupe": 0,
        "kept_after_vote": 0,
        "params": {
            "M": sc_m,
            "min_consistent_votes": threshold,
            "dedup_threshold": sc_dedup_threshold,
            "source": "direct_qa_pairs",
            "mode": "consistency_vote",
        },
    }

    if not candidate_pool:
        print(
            "  [SC input] no Direct QA pairs to validate — "
            "returning empty result"
        )
        return {
            "design_name": design_name,
            "strategy": "self_consistency",
            "num_stages": num_stages,
            "stages_info": _stages_to_info(stages),
            "qa_pairs": [],
            "sc_stats": _enrich_sc_stats(empty_stats),
            "sc_diagnostics": [],
        }

    print(f"  [SC input] validating all {len(candidate_pool)} Direct pairs (no dedupe)")

    final_qa_pairs: List[Dict[str, Any]] = []
    per_question_diagnostics: List[Dict[str, Any]] = []

    for q_idx, qa in enumerate(candidate_pool, 1):
        question = qa.get("question", "")
        direct_answer = sanitize_for_json(qa.get("answer"))
        if not question.strip():
            continue

        cons_prompt = _build_sc_consistency_prompt(
            stages_with_raw, design_name, question, direct_answer
        )
        consistency_votes: List[Dict[str, Any]] = []
        for m_idx in range(sc_m):
            response = _tracked_llm_call(
                system_prompt=_DIRECT_SYSTEM_PROMPT,
                user_prompt=cons_prompt,
                temperature=temperature,
            )
            consistency_votes.append(_parse_sc_consistency_response(response))

        vote = _vote_consistency(
            consistency_votes,
            min_consistent_votes=threshold,
            sc_m=sc_m,
        )
        per_question_diagnostics.append(
            {
                "question": question,
                "direct_answer": direct_answer,
                "gen_votes": qa.get("votes", 1),
                "consistency_votes": consistency_votes,
                "vote_result": vote,
            }
        )

        if vote["kept"]:
            final_qa_pairs.append({"question": question, "answer": direct_answer})
            print(
                f"  [SC vote] Q{q_idx} kept "
                f"yes={vote['yes_votes']}/{sc_m} "
                f"no={vote['no_votes']} parse_fail={vote['parse_failed']}"
            )
        else:
            if vote["yes_votes"] + vote["no_votes"] == 0:
                reason = "all_parse_failed"
            else:
                reason = "below_threshold"
            print(
                f"  [SC vote] Q{q_idx} dropped ({reason}) "
                f"yes={vote['yes_votes']}/{sc_m} "
                f"no={vote['no_votes']} parse_fail={vote['parse_failed']}"
            )

    return {
        "design_name": design_name,
        "strategy": "self_consistency",
        "num_stages": num_stages,
        "stages_info": _stages_to_info(stages),
        "input_source": {
            "kind": "raw_extract_text",
            "stages": [
                {
                    "stage": s["stage"],
                    "occurrence": s["occurrence"],
                    "raw_text_available": s["raw_text_available"],
                    "raw_text_chars": len(s["raw_text"]) if s["raw_text"] else 0,
                }
                for s in stages_with_raw
            ],
        },
        "qa_pairs": final_qa_pairs,
        "sc_stats": _enrich_sc_stats(
            {
                "candidates_total": len(candidate_pool),
                "candidates_after_dedupe": len(candidate_pool),
                "kept_after_vote": len(final_qa_pairs),
                "params": {
                    "M": sc_m,
                    "min_consistent_votes": threshold,
                    "dedup_threshold": sc_dedup_threshold,
                    "source": "direct_qa_pairs",
                    "mode": "consistency_vote",
                },
            }
        ),
        "sc_diagnostics": per_question_diagnostics,
    }


# ---------------------------------------------------------------------------
# Experiment runner.
# ---------------------------------------------------------------------------


@dataclass
class RunMetrics:
    """Per-cell measurements for one (strategy, num_stages) run."""
    strategy: str
    num_stages: int
    elapsed_seconds: float = 0.0
    llm_calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    qa_pairs_count: int = 0
    error: Optional[str] = None
    rejection_stats: Optional[Dict[str, Any]] = None
    stages_info: List[Dict[str, Any]] = field(default_factory=list)
    # Full set of QA pairs produced by this cell. For `direct` and `qtree`
    # each entry is the raw QA dict. For `self_consistency` each entry is
    # annotated with `kept`, `drop_reason`, `vote` (yes/no/parse_failed
    # tallies + threshold + consistent_rate), and `consistency_votes` (the
    # M raw {vote, reason} dicts) so the JSON shows every Direct pair
    # together with whether SC's consistency vote kept or filtered it.
    qa_pairs: List[Dict[str, Any]] = field(default_factory=list)
    diversity: Optional[Dict[str, Any]] = None
    sc_stats: Optional[Dict[str, Any]] = None
    dce_stats: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _snapshot_gen_stats_delta(before: Dict[str, Any]) -> Dict[str, int]:
    after = get_qa_generation_stats()
    return {
        "num_calls": after["num_calls"] - before["num_calls"],
        "prompt_tokens": after["total_prompt_tokens"] - before["total_prompt_tokens"],
        "output_tokens": after["total_output_tokens"] - before["total_output_tokens"],
        "elapsed_seconds": after["total_time_seconds"] - before["total_time_seconds"],
    }


def _build_sc_annotated_qa_pairs(
    sc_diagnostics: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Turn SC per-question diagnostics into a full QA list with status flags.

    Each entry carries:
      - ``question``           – the Direct question being validated.
      - ``answer``             – the Direct (input) answer, kept verbatim
                                 (SC never regenerates this).
      - ``kept``               – ``True`` if the consistency vote accepted it.
      - ``drop_reason``        – ``None`` when kept, else ``"below_threshold"``
                                 or ``"all_parse_failed"``.
      - ``vote``               – tally summary: ``{yes_votes, no_votes,
                                 parse_failed, threshold, consistent_rate}``.
      - ``consistency_votes``  – the M raw vote dicts ``{vote, reason}``.
    """
    annotated: List[Dict[str, Any]] = []
    for diag in sc_diagnostics:
        vote = diag.get("vote_result", {}) or {}
        kept = bool(vote.get("kept"))
        if kept:
            drop_reason: Optional[str] = None
        elif (vote.get("yes_votes", 0) + vote.get("no_votes", 0)) == 0:
            drop_reason = "all_parse_failed"
        else:
            drop_reason = "below_threshold"
        annotated.append(
            {
                "question": diag.get("question"),
                "answer": diag.get("direct_answer"),
                "kept": kept,
                "drop_reason": drop_reason,
                "vote": {
                    "yes_votes": vote.get("yes_votes"),
                    "no_votes": vote.get("no_votes"),
                    "parse_failed": vote.get("parse_failed"),
                    "threshold": vote.get("threshold"),
                    "consistent_rate": vote.get("consistent_rate"),
                },
                "consistency_votes": diag.get("consistency_votes", []),
            }
        )
    return annotated


def run_one_cell(
    strategy: str,
    num_stages: int,
    metrics_path: Path,
    *,
    seed: Optional[int],
    temperature: float,
    n_questions: int,
    embedding_model=None,
    sc_m: int = 3,
    sc_dedup_threshold: float = 0.85,
    sc_min_consistent_votes: Optional[int] = None,
    preselected_stages: Optional[List[Dict[str, Any]]] = None,
    direct_qa_pairs_for_sc: Optional[List[Dict[str, Any]]] = None,
    raw_max_chars_per_stage: Optional[int] = None,
    dce_validate_against_direct: bool = True,
    dce_validation_threshold: float = 0.8,
) -> tuple[RunMetrics, List[Dict[str, Any]]]:
    """Run one (strategy, num_stages) cell.

    Returns `(metrics, qa_pairs)`. The runner uses the latter to thread
    Direct's QA pairs into Self-Consistency / DCE on the next cell.

    `preselected_stages` should be passed in by `run_experiment` so all
    strategies for a given `num_stages` see the identical sampled stages.
    """
    print("\n" + "=" * 80)
    print(f"[CELL] strategy={strategy} num_stages={num_stages}")
    print("=" * 80)

    cell = RunMetrics(strategy=strategy, num_stages=num_stages)
    stats_before = get_qa_generation_stats()
    wall_start = time.perf_counter()

    rejection_stats = (
        RejectionStats() if strategy in ("qtree", "direct_code_extraction") else None
    )
    full_qa_pairs: List[Dict[str, Any]] = []

    try:
        if strategy == "direct":
            result = direct_qa_pair(
                metrics_path,
                num_stages=num_stages,
                n_questions=n_questions,
                seed=seed,
                temperature=temperature,
                stages=preselected_stages,
                max_chars_per_stage=raw_max_chars_per_stage,
            )
        elif strategy == "qtree":
            result = qtree_qa_pair(
                metrics_path,
                num_stages=num_stages,
                seed=seed,
                temperature=temperature,
                rejection_stats=rejection_stats,
                stages=preselected_stages,
                n_questions=n_questions,
            )
        elif strategy == "self_consistency":
            if direct_qa_pairs_for_sc is None:
                raise QAGenerationError(
                    "self_consistency requires Direct's QA pairs as input; "
                    "make sure the 'direct' strategy runs in the same cell."
                )
            result = self_consistency_qa_pair(
                metrics_path,
                num_stages=num_stages,
                seed=seed,
                temperature=temperature,
                embedding_model=embedding_model,
                direct_qa_pairs=direct_qa_pairs_for_sc,
                sc_m=sc_m,
                sc_dedup_threshold=sc_dedup_threshold,
                sc_min_consistent_votes=sc_min_consistent_votes,
                stages=preselected_stages,
                max_chars_per_stage=raw_max_chars_per_stage,
            )
            cell.sc_stats = result.get("sc_stats")
        elif strategy == "direct_code_extraction":
            if direct_qa_pairs_for_sc is None:
                raise QAGenerationError(
                    "direct_code_extraction requires Direct's QA pairs as input; "
                    "make sure the 'direct' strategy runs in the same cell."
                )
            result = direct_code_extraction_qa_pair(
                metrics_path,
                num_stages=num_stages,
                direct_qa_pairs=direct_qa_pairs_for_sc,
                stages=preselected_stages,
                seed=seed,
                temperature=temperature,
                validate_against_direct=dce_validate_against_direct,
                validation_threshold=dce_validation_threshold,
                rejection_stats=rejection_stats,
                embedding_model=embedding_model,
            )
            cell.dce_stats = result.get("dce_stats")
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        qa_pairs = result.get("qa_pairs", []) or []
        full_qa_pairs = list(qa_pairs)
        cell.qa_pairs_count = len(qa_pairs)
        cell.stages_info = result.get("stages_info", []) or []

        if strategy == "self_consistency":
            cell.qa_pairs = _build_sc_annotated_qa_pairs(
                result.get("sc_diagnostics", []) or []
            )
        else:
            cell.qa_pairs = [dict(qa) for qa in qa_pairs]

        input_stages = [
            s.get("stage") for s in (preselected_stages or cell.stages_info or []) if s.get("stage")
        ]
        pairs_for_vendi = qa_pairs
        if strategy == "self_consistency" and cell.qa_pairs:
            kept = [p for p in cell.qa_pairs if p.get("kept")]
            pairs_for_vendi = kept if kept else qa_pairs
        if embedding_model is not None and pairs_for_vendi:
            cell.diversity = compute_qa_metrics(
                pairs_for_vendi, embedding_model, input_stage_types=input_stages,
            )
            if strategy == "self_consistency" and cell.qa_pairs:
                q = cell.diversity.get("quality", {})
                cell.diversity["quality_kept"] = q
                cell.diversity["quality_all_candidates"] = compute_qa_quality(
                    cell.qa_pairs, input_stage_types=input_stages,
                )
        elif pairs_for_vendi or cell.qa_pairs:
            cell.diversity = {
                "quality": compute_qa_quality(pairs_for_vendi or cell.qa_pairs, input_stage_types=input_stages),
            }
    except (QAGenerationError, Exception) as exc:  # noqa: BLE001
        cell.error = f"{type(exc).__name__}: {exc}"
        print(f"[ERROR] {cell.error}")

    cell.elapsed_seconds = time.perf_counter() - wall_start
    delta = _snapshot_gen_stats_delta(stats_before)
    cell.llm_calls = delta["num_calls"]
    cell.prompt_tokens = delta["prompt_tokens"]
    cell.output_tokens = delta["output_tokens"]

    if rejection_stats is not None:
        cell.rejection_stats = rejection_stats.get_summary()

    div_msg = ""
    if cell.diversity:
        q_vs = cell.diversity.get("questions", {}).get("vendi")
        qa_vs = cell.diversity.get("qa_combined", {}).get("vendi")
        if q_vs is not None and qa_vs is not None:
            div_msg = f" vendi(Q/Q+A)={q_vs:.2f}/{qa_vs:.2f}"
        qual = _quality_block(cell.diversity)
        if qual:
            div_msg += (
                f" stg_cnt={qual.get('mean_input_stages_mentioned', 0):.2f}"
                f"/{qual.get('num_input_stages', '?')}"
                f" multi={qual.get('pct_multi_stage', 0):.0f}%"
                f" qlen={qual.get('mean_question_chars', 0):.0f}"
            )

    print(
        f"[CELL DONE] qa_pairs={cell.qa_pairs_count} "
        f"calls={cell.llm_calls} tokens(p/o)={cell.prompt_tokens}/{cell.output_tokens} "
        f"elapsed={cell.elapsed_seconds:.2f}s{div_msg}"
    )
    return cell, full_qa_pairs


def _fmt_vendi(cell: Optional[Dict[str, Any]], key: str) -> str:
    """Render `vendi (vendi_norm)` for a diversity sub-block."""
    if not cell:
        return "-"
    sub = cell.get(key)
    if not sub:
        return "-"
    vs = sub.get("vendi")
    vs_norm = sub.get("vendi_normalized")
    if vs is None or vs_norm is None:
        err = sub.get("error", "n/a")
        return f"err:{err[:12]}"
    return f"{vs:.2f} ({vs_norm:.2f})"


def _quality_block(diversity: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not diversity:
        return None
    return diversity.get("quality_kept") or diversity.get("quality")


def _fmt_quality_pct(diversity: Optional[Dict[str, Any]], field: str) -> str:
    q = _quality_block(diversity)
    if not q or q.get(field) is None:
        return "-"
    val = float(q[field])
    if field.startswith("pct_"):
        return f"{val:.0f}%"
    if field in ("mean_input_stages_mentioned", "mean_stages_mentioned"):
        return f"{val:.2f}"
    if field == "mean_input_stage_coverage":
        return f"{val:.2f}"
    return f"{val:.2f}"


def _format_table(rows: List[RunMetrics]) -> str:
    """Render a compact ASCII comparison table."""
    headers = [
        "strategy",
        "stages",
        "qa_pairs",
        "calls",
        "prompt_tok",
        "output_tok",
        "elapsed_s",
        "success_rate",
        "vendi_Q",
        "vendi_A",
        "vendi_Q+A",
        "stg_cnt",
        "multi_%",
        "q_chars",
        "cmp_%",
        "error",
    ]
    table: List[List[str]] = [headers]
    for r in rows:
        qa_pairs_display = str(r.qa_pairs_count)
        if r.rejection_stats and r.rejection_stats.get("total_attempts", 0) > 0:
            success_rate = f"{r.rejection_stats['success_rate']:.1f}%"
            rej_total = r.rejection_stats.get("total_attempts", 0)
            rej_kept = r.rejection_stats.get("successful", 0)
            if rej_total > 0:
                qa_pairs_display = f"{rej_kept}/{rej_total}"
        elif r.strategy == "self_consistency" and r.sc_stats:
            total = r.sc_stats.get("candidates_total", 0) or 0
            kept = r.sc_stats.get("kept_after_vote", 0) or 0
            rate = r.sc_stats.get("retention_rate")
            if rate is None and total > 0:
                rate = 100.0 * kept / total
            success_rate = f"{rate:.1f}%" if rate is not None else "-"
            qa_pairs_display = f"{kept}/{total}" if total else str(r.qa_pairs_count)
        elif r.strategy == "direct_code_extraction" and r.dce_stats:
            total = r.dce_stats.get("candidates_total", 0) or 0
            kept = r.dce_stats.get("kept_after_validation", 0) or 0
            rate = r.dce_stats.get("retention_rate")
            if rate is None and total > 0:
                rate = 100.0 * kept / total
            success_rate = f"{rate:.1f}%" if rate is not None else "-"
            qa_pairs_display = f"{kept}/{total}" if total else str(r.qa_pairs_count)
        elif r.strategy == "direct" and r.qa_pairs_count > 0:
            success_rate = "n/a"
        else:
            success_rate = "-"

        table.append(
            [
                r.strategy,
                str(r.num_stages),
                qa_pairs_display,
                str(r.llm_calls),
                str(r.prompt_tokens),
                str(r.output_tokens),
                f"{r.elapsed_seconds:.2f}",
                success_rate,
                _fmt_vendi(r.diversity, "questions"),
                _fmt_vendi(r.diversity, "answers"),
                _fmt_vendi(r.diversity, "qa_combined"),
                _fmt_quality_pct(r.diversity, "mean_input_stages_mentioned"),
                _fmt_quality_pct(r.diversity, "pct_multi_stage"),
                (
                    f"{_quality_block(r.diversity)['mean_question_chars']:.0f}"
                    if _quality_block(r.diversity)
                    and _quality_block(r.diversity).get("mean_question_chars") is not None
                    else "-"
                ),
                _fmt_quality_pct(r.diversity, "pct_comparative"),
                (r.error or "")[:40],
            ]
        )

    widths = [max(len(row[c]) for row in table) for c in range(len(headers))]
    lines = []
    for i, row in enumerate(table):
        line = "  ".join(cell.ljust(widths[c]) for c, cell in enumerate(row))
        lines.append(line)
        if i == 0:
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)


STRATEGY_DISPLAY_ORDER: tuple[str, ...] = (
    "direct",
    "self_consistency",
    "direct_code_extraction",
    "qtree",
)


def _mean_optional(values: List[Optional[float]]) -> Optional[float]:
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def _row_metric_values(r: RunMetrics) -> Dict[str, Optional[float]]:
    """Numeric fields from one cell, for per-strategy averaging."""
    qual = _quality_block(r.diversity)
    div = r.diversity or {}
    out: Dict[str, Optional[float]] = {
        "qa_pairs": float(r.qa_pairs_count),
        "llm_calls": float(r.llm_calls),
        "prompt_tokens": float(r.prompt_tokens),
        "output_tokens": float(r.output_tokens),
        "elapsed_seconds": float(r.elapsed_seconds),
    }
    for div_key, out_key in (
        ("questions", "vendi_q"),
        ("answers", "vendi_a"),
        ("qa_combined", "vendi_qa"),
    ):
        sub = div.get(div_key) or {}
        # Strategy averages use vendi/n (same scale as per-cell table's norm value).
        vn = sub.get("vendi_normalized")
        if vn is None and sub.get("vendi") is not None and sub.get("count"):
            vn = float(sub["vendi"]) / float(sub["count"])
        out[out_key] = float(vn) if vn is not None else None

    if qual:
        mapping = (
            ("mean_input_stages_mentioned", "stg_cnt"),
            ("pct_multi_stage", "multi_pct"),
            ("mean_question_chars", "q_chars"),
            ("pct_comparative", "cmp_pct"),
        )
        for src, dst in mapping:
            v = qual.get(src)
            out[dst] = float(v) if v is not None else None

    if r.strategy == "self_consistency" and r.sc_stats:
        total = int(r.sc_stats.get("candidates_total") or 0)
        kept = int(r.sc_stats.get("kept_after_vote") or 0)
        if total > 0:
            out["sc_kept"] = float(kept)
            out["sc_total"] = float(total)
            rr = r.sc_stats.get("retention_rate")
            out["retention_rate"] = (
                float(rr) if rr is not None else 100.0 * kept / total
            )
    elif r.strategy == "direct_code_extraction" and r.dce_stats:
        total = int(r.dce_stats.get("candidates_total") or 0)
        kept = int(r.dce_stats.get("kept_after_validation") or 0)
        if total > 0:
            out["sc_kept"] = float(kept)
            out["sc_total"] = float(total)
            rr = r.dce_stats.get("retention_rate")
            out["retention_rate"] = (
                float(rr) if rr is not None else 100.0 * kept / total
            )
    elif r.rejection_stats and (r.rejection_stats.get("total_attempts") or 0) > 0:
        out["retention_rate"] = float(r.rejection_stats.get("success_rate", 0))
    return out


def _qa_pairs_for_run(r: RunMetrics, *, sc_kept_only: bool) -> List[Dict[str, Any]]:
    """QA pairs from one cell; SC may filter to kept only."""
    pairs = list(r.qa_pairs or [])
    if r.strategy == "self_consistency" and sc_kept_only:
        if any("kept" in qa for qa in pairs):
            pairs = [qa for qa in pairs if qa.get("kept")]
    return pairs


def _pooled_quality_metrics(
    cells: List[RunMetrics],
    *,
    sc_kept_only: bool,
) -> Dict[str, Optional[float]]:
    """Quality metrics averaged over every QA pair (each cell scored with its input stages)."""
    effective: List[int] = []
    multi_n = 0
    cmp_n = 0
    chars: List[float] = []
    n = 0

    for r in cells:
        pairs = _qa_pairs_for_run(r, sc_kept_only=sc_kept_only)
        if not pairs:
            continue
        stages = [s.get("stage") for s in r.stages_info if s.get("stage")]
        qual = compute_qa_quality(pairs, input_stage_types=stages)
        for p in qual.get("per_pair") or []:
            n += 1
            raw = int(p.get("num_input_stages_mentioned", 0))
            effective.append(_effective_input_stage_count(raw))
            if p.get("is_multi_stage"):
                multi_n += 1
            if p.get("is_comparative"):
                cmp_n += 1
            chars.append(float(p.get("question_chars", 0)))

    if n == 0:
        return {
            "total_qa_pairs": 0,
            "stg_cnt": None,
            "multi_pct": None,
            "q_chars": None,
            "cmp_pct": None,
        }
    return {
        "total_qa_pairs": float(n),
        "stg_cnt": float(sum(effective) / n),
        "multi_pct": 100.0 * multi_n / n,
        "q_chars": float(sum(chars) / n),
        "cmp_pct": 100.0 * cmp_n / n,
    }


def aggregate_strategy_averages(
    rows: List[RunMetrics],
    embedding_model=None,
    *,
    duplicate_report: Optional[Dict[str, Any]] = None,
    sc_kept_only: bool = True,
    cosine_threshold: float = DEFAULT_DUPLICATE_COSINE_THRESHOLD,
) -> List[Dict[str, Any]]:
    """Pooled metrics per strategy: all QA pairs across cells treated as one set.

    Vendi and structural quality (stg_cnt, multi_%, q_chars, cmp_%) are computed
    over the union of every QA pair, not as the mean of per-cell values.
    Duplicate counts come from ``duplicate_report`` or are computed on the fly.
    """
    from collections import defaultdict

    groups: Dict[str, List[RunMetrics]] = defaultdict(list)
    for r in rows:
        groups[r.strategy].append(r)

    ordered = [s for s in STRATEGY_DISPLAY_ORDER if s in groups]
    ordered.extend(sorted(s for s in groups if s not in STRATEGY_DISPLAY_ORDER))

    dup_by_strategy = (duplicate_report or {}).get("by_strategy") or {}

    aggregates: List[Dict[str, Any]] = []
    for strategy in ordered:
        cells = groups[strategy]
        pooled_pairs: List[Dict[str, Any]] = []
        for r in cells:
            pooled_pairs.extend(_qa_pairs_for_run(r, sc_kept_only=sc_kept_only))

        qual = _pooled_quality_metrics(cells, sc_kept_only=sc_kept_only)
        agg: Dict[str, Any] = {
            "strategy": strategy,
            "n_cells": len(cells),
            "num_stages": sorted({r.num_stages for r in cells}),
            "aggregation": "pooled_all_qa_pairs",
            "total_qa_pairs": int(qual["total_qa_pairs"] or 0),
            "stg_cnt": qual["stg_cnt"],
            "multi_pct": qual["multi_pct"],
            "q_chars": qual["q_chars"],
            "cmp_pct": qual["cmp_pct"],
            "llm_calls": float(sum(r.llm_calls for r in cells)),
            "prompt_tokens": float(sum(r.prompt_tokens for r in cells)),
            "output_tokens": float(sum(r.output_tokens for r in cells)),
            "elapsed_seconds": float(sum(r.elapsed_seconds for r in cells)),
        }

        if embedding_model is not None and pooled_pairs:
            div = compute_qa_metrics(pooled_pairs, embedding_model)
            for div_key, out_key in (
                ("questions", "vendi_q"),
                ("answers", "vendi_a"),
                ("qa_combined", "vendi_qa"),
            ):
                sub = div.get(div_key) or {}
                vn = sub.get("vendi_normalized")
                if vn is None and sub.get("vendi") is not None and sub.get("count"):
                    vn = float(sub["vendi"]) / float(sub["count"])
                agg[out_key] = float(vn) if vn is not None else None

        dup_strat = dup_by_strategy.get(strategy)
        if dup_strat is None and pooled_pairs:
            refs = collect_qa_refs_from_runs(
                [r.to_dict() for r in cells],
                strategies=[strategy],
                sc_kept_only=sc_kept_only,
            ).get(strategy, [])
            dup_strat = detect_qa_duplicates_for_strategy(
                refs,
                embedding_model,
                cosine_threshold=cosine_threshold,
            )

        if dup_strat:
            agg["unique_qa"] = int(dup_strat.get("unique_clusters") or 0)
            agg["redundant_qa"] = int(dup_strat.get("redundant_pairs") or 0)
            agg["dup_clusters"] = len(dup_strat.get("duplicate_clusters") or [])

        if strategy == "self_consistency":
            sc_kept = sum(int((r.sc_stats or {}).get("kept_after_vote") or 0) for r in cells)
            sc_total = sum(int((r.sc_stats or {}).get("candidates_total") or 0) for r in cells)
            agg["sc_kept"] = float(sc_kept)
            agg["sc_total"] = float(sc_total)
            agg["retention_rate"] = (
                100.0 * sc_kept / sc_total if sc_total > 0 else None
            )
        elif strategy == "qtree":
            rej_kept = sum(
                int((r.rejection_stats or {}).get("successful") or 0) for r in cells
            )
            rej_total = sum(
                int((r.rejection_stats or {}).get("total_attempts") or 0) for r in cells
            )
            if rej_total > 0:
                agg["sc_kept"] = float(rej_kept)
                agg["sc_total"] = float(rej_total)
                agg["retention_rate"] = 100.0 * rej_kept / rej_total
        elif strategy == "direct_code_extraction":
            dce_kept = sum(int((r.dce_stats or {}).get("kept_after_validation") or 0) for r in cells)
            dce_total = sum(int((r.dce_stats or {}).get("candidates_total") or 0) for r in cells)
            agg["sc_kept"] = float(dce_kept)
            agg["sc_total"] = float(dce_total)
            agg["retention_rate"] = (
                100.0 * dce_kept / dce_total if dce_total > 0 else None
            )
            agg["code_exec_failed"] = sum(
                int((r.dce_stats or {}).get("code_exec_failed") or 0) for r in cells
            )
            agg["invalid_code_answer"] = sum(
                int((r.dce_stats or {}).get("invalid_code_answer") or 0) for r in cells
            )
            agg["low_validation_score"] = sum(
                int((r.dce_stats or {}).get("low_validation_score") or 0) for r in cells
            )

        aggregates.append(agg)
    return aggregates


def _format_strategy_averages_table(aggregates: List[Dict[str, Any]]) -> str:
    """Compact table of per-strategy pooled metrics (``aggregate_strategy_averages``)."""
    if not aggregates:
        return "(no runs)"

    def _num(v: Optional[float], nd: int = 2) -> str:
        if v is None:
            return "-"
        return f"{v:.{nd}f}"

    headers = [
        "strategy",
        "n_cells",
        "stages",
        "total_qa",
        "unique_q",
        "redundant",
        "total_calls",
        "total_prompt_tok",
        "total_output_tok",
        "total_elapsed_s",
        "success%",
        "vendi_Q/n",
        "vendi_A/n",
        "vendi_Q+A/n",
        "stg_cnt",
        "multi_%",
        "q_chars",
        "cmp_%",
    ]
    table: List[List[str]] = [headers]
    for a in aggregates:
        stages = ",".join(str(s) for s in a.get("num_stages") or [])
        total_qa = str(int(a.get("total_qa_pairs") or 0))
        if a.get("sc_kept") is not None and a.get("sc_total") is not None:
            total_qa = f"{int(a['sc_kept'])}/{int(a['sc_total'])}"

        unique_q = str(a.get("unique_qa", "-"))
        redundant = str(a.get("redundant_qa", "-"))

        if a.get("retention_rate") is not None:
            success = f"{a['retention_rate']:.1f}%"
        elif a["strategy"] == "direct":
            success = "n/a"
        else:
            success = "-"

        multi = a.get("multi_pct")
        cmp_ = a.get("cmp_pct")
        table.append(
            [
                str(a["strategy"]),
                str(a["n_cells"]),
                stages or "-",
                total_qa,
                unique_q,
                redundant,
                _num(a.get("llm_calls"), 0),
                _num(a.get("prompt_tokens"), 0),
                _num(a.get("output_tokens"), 0),
                _num(a.get("elapsed_seconds"), 2),
                success,
                _num(a.get("vendi_q")),
                _num(a.get("vendi_a")),
                _num(a.get("vendi_qa")),
                _num(a.get("stg_cnt")),
                f"{multi:.0f}%" if multi is not None else "-",
                _num(a.get("q_chars"), 0),
                f"{cmp_:.0f}%" if cmp_ is not None else "-",
            ]
        )

    widths = [max(len(row[c]) for row in table) for c in range(len(headers))]
    lines: List[str] = []
    for i, row in enumerate(table):
        lines.append("  ".join(cell.ljust(widths[c]) for c, cell in enumerate(row)))
        if i == 0:
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)


def _union_find_parent(parent: List[int]) -> Any:
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    return find, union


def collect_qa_refs_from_runs(
    runs: List[Dict[str, Any]],
    *,
    strategies: Optional[List[str]] = None,
    sc_kept_only: bool = True,
    source_file: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Gather QA rows from all cells, grouped by strategy.

    Each ref dict has: ``strategy``, ``num_stages``, ``pair_index``, ``question``,
    ``question_norm``, optional ``source_file``, optional ``kept`` (SC only).
    """
    allowed = set(strategies) if strategies else None
    by_strategy: Dict[str, List[Dict[str, Any]]] = {}

    for run in runs:
        strategy = str(run.get("strategy") or "")
        if not strategy:
            continue
        if allowed is not None and strategy not in allowed:
            continue

        qa_pairs = run.get("qa_pairs") or []
        num_stages = int(run.get("num_stages") or 0)
        refs: List[Dict[str, Any]] = []

        for pair_index, qa in enumerate(qa_pairs):
            if strategy == "self_consistency" and sc_kept_only:
                if "kept" in qa and not qa.get("kept"):
                    continue
            question = _clean_question_text(qa.get("question", ""))
            if not question.strip():
                continue
            ref: Dict[str, Any] = {
                "strategy": strategy,
                "num_stages": num_stages,
                "pair_index": pair_index,
                "question": question,
                "question_norm": _normalize_string(question),
            }
            if source_file:
                ref["source_file"] = source_file
            if strategy == "self_consistency" and "kept" in qa:
                ref["kept"] = bool(qa.get("kept"))
            refs.append(ref)

        if refs:
            by_strategy.setdefault(strategy, []).extend(refs)

    return by_strategy


def _max_pairwise_cosine(emb: np.ndarray, indices: List[int]) -> float:
    if len(indices) < 2:
        return 1.0
    sub = emb[np.array(indices, dtype=int)]
    best = -1.0
    for i in range(len(sub)):
        for j in range(i + 1, len(sub)):
            sim = float(np.dot(sub[i], sub[j]))
            if sim > best:
                best = sim
    return best


def detect_qa_duplicates_for_strategy(
    refs: List[Dict[str, Any]],
    embedding_model=None,
    *,
    cosine_threshold: float = DEFAULT_DUPLICATE_COSINE_THRESHOLD,
) -> Dict[str, Any]:
    """Find duplicate questions within one strategy across cells (union-find).

    Duplicates are merged when:
      - normalized question strings are identical, or
      - cosine similarity of question embeddings is >= ``cosine_threshold``
        (default 0.98; requires an embedding model).
    """
    n = len(refs)
    if n == 0:
        return {
            "total_qa_pairs": 0,
            "unique_clusters": 0,
            "redundant_pairs": 0,
            "exact_duplicate_groups": [],
            "duplicate_clusters": [],
            "cosine_threshold": cosine_threshold,
            "cosine_matching_enabled": False,
        }

    parent = list(range(n))
    find, union = _union_find_parent(parent)

    exact_groups: List[List[int]] = []
    by_norm: Dict[str, List[int]] = {}
    for i, ref in enumerate(refs):
        by_norm.setdefault(ref["question_norm"], []).append(i)
    for indices in by_norm.values():
        if len(indices) > 1:
            exact_groups.append(indices)
            root = indices[0]
            for j in indices[1:]:
                union(root, j)

    cosine_enabled = False
    emb: Optional[np.ndarray] = None
    if embedding_model is not None and n > 1:
        try:
            texts = [r["question"] for r in refs]
            raw_emb = embedding_model.encode(texts)
            if raw_emb is not None:
                emb = np.asarray(raw_emb, dtype=np.float64)
                norms = np.linalg.norm(emb, axis=1, keepdims=True)
                norms = np.where(norms < 1e-12, 1.0, norms)
                emb = emb / norms
                cosine_enabled = True
                for i in range(n):
                    for j in range(i + 1, n):
                        if float(np.dot(emb[i], emb[j])) >= cosine_threshold:
                            union(i, j)
        except Exception:  # noqa: BLE001
            emb = None
            cosine_enabled = False

    clusters_map: Dict[int, List[int]] = {}
    for i in range(n):
        root = find(i)
        clusters_map.setdefault(root, []).append(i)

    def _member_dict(idx: int) -> Dict[str, Any]:
        ref = refs[idx]
        out = {
            "num_stages": ref["num_stages"],
            "pair_index": ref["pair_index"],
            "question": ref["question"],
        }
        if ref.get("source_file"):
            out["source_file"] = ref["source_file"]
        if "kept" in ref:
            out["kept"] = ref["kept"]
        return out

    duplicate_clusters: List[Dict[str, Any]] = []
    for indices in sorted(clusters_map.values(), key=len, reverse=True):
        if len(indices) < 2:
            continue
        norms_in = {refs[i]["question_norm"] for i in indices}
        match = "exact" if len(norms_in) == 1 else "cosine"
        max_cos = 1.0 if match == "exact" else (
            _max_pairwise_cosine(emb, indices) if emb is not None else None
        )
        duplicate_clusters.append(
            {
                "size": len(indices),
                "match": match,
                "max_cosine": max_cos,
                "members": [_member_dict(i) for i in sorted(indices, key=lambda k: (refs[k]["num_stages"], refs[k]["pair_index"]))],
            }
        )

    exact_duplicate_groups = []
    for indices in exact_groups:
        exact_duplicate_groups.append(
            {
                "size": len(indices),
                "match": "exact",
                "max_cosine": 1.0,
                "members": [_member_dict(i) for i in sorted(indices, key=lambda k: (refs[k]["num_stages"], refs[k]["pair_index"]))],
            }
        )

    unique_clusters = len(clusters_map)
    return {
        "total_qa_pairs": n,
        "unique_clusters": unique_clusters,
        "redundant_pairs": n - unique_clusters,
        "exact_duplicate_groups": exact_duplicate_groups,
        "duplicate_clusters": duplicate_clusters,
        "cosine_threshold": cosine_threshold,
        "cosine_matching_enabled": cosine_enabled,
    }


def detect_qa_duplicates_across_cells(
    runs_or_summary: Union[List[Dict[str, Any]], Dict[str, Any]],
    embedding_model=None,
    *,
    cosine_threshold: float = DEFAULT_DUPLICATE_COSINE_THRESHOLD,
    sc_kept_only: bool = True,
    source_file: Optional[str] = None,
    strategies: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Detect duplicate questions per strategy across all cells in one experiment.

    ``runs_or_summary`` may be a summary dict (uses ``runs``) or a list of run dicts.
    """
    if isinstance(runs_or_summary, dict):
        runs = runs_or_summary.get("runs") or []
        if source_file is None:
            source_file = runs_or_summary.get("metrics_path")
    else:
        runs = list(runs_or_summary)

    grouped = collect_qa_refs_from_runs(
        runs,
        strategies=strategies,
        sc_kept_only=sc_kept_only,
        source_file=source_file,
    )

    by_strategy: Dict[str, Any] = {}
    for strategy, refs in grouped.items():
        by_strategy[strategy] = detect_qa_duplicates_for_strategy(
            refs,
            embedding_model,
            cosine_threshold=cosine_threshold,
        )

    totals = {
        "total_qa_pairs": sum(s.get("total_qa_pairs", 0) for s in by_strategy.values()),
        "unique_clusters": sum(s.get("unique_clusters", 0) for s in by_strategy.values()),
        "redundant_pairs": sum(s.get("redundant_pairs", 0) for s in by_strategy.values()),
    }

    return {
        "params": {
            "cosine_threshold": cosine_threshold,
            "sc_kept_only": sc_kept_only,
            "source_file": source_file,
        },
        "totals": totals,
        "by_strategy": by_strategy,
    }


def detect_qa_duplicates_across_summaries(
    summaries: List[Dict[str, Any]],
    embedding_model=None,
    *,
    cosine_threshold: float = DEFAULT_DUPLICATE_COSINE_THRESHOLD,
    sc_kept_only: bool = True,
    strategies: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Like ``detect_qa_duplicates_across_cells`` but merges runs from multiple JSON files."""
    merged: Dict[str, List[Dict[str, Any]]] = {}
    source_files: List[str] = []

    for summary in summaries:
        runs = summary.get("runs") or []
        label = (
            summary.get("timestamp")
            or summary.get("metrics_path")
            or f"summary_{len(source_files)}"
        )
        source_files.append(str(label))
        grouped = collect_qa_refs_from_runs(
            runs,
            strategies=strategies,
            sc_kept_only=sc_kept_only,
            source_file=str(label),
        )
        for strategy, refs in grouped.items():
            merged.setdefault(strategy, []).extend(refs)

    by_strategy: Dict[str, Any] = {}
    for strategy, refs in merged.items():
        by_strategy[strategy] = detect_qa_duplicates_for_strategy(
            refs,
            embedding_model,
            cosine_threshold=cosine_threshold,
        )

    totals = {
        "total_qa_pairs": sum(s.get("total_qa_pairs", 0) for s in by_strategy.values()),
        "unique_clusters": sum(s.get("unique_clusters", 0) for s in by_strategy.values()),
        "redundant_pairs": sum(s.get("redundant_pairs", 0) for s in by_strategy.values()),
    }

    return {
        "params": {
            "cosine_threshold": cosine_threshold,
            "sc_kept_only": sc_kept_only,
            "source_files": source_files,
            "n_summaries": len(summaries),
        },
        "totals": totals,
        "by_strategy": by_strategy,
    }


def _truncate_question(q: str, max_len: int = 100) -> str:
    q = " ".join(q.split())
    if len(q) <= max_len:
        return q
    return q[: max_len - 3] + "..."


def format_qa_duplicate_report(report: Dict[str, Any], *, max_clusters: int = 15) -> str:
    """Human-readable summary of ``detect_qa_duplicates_*`` output."""
    lines: List[str] = []
    params = report.get("params") or {}
    thr = params.get("cosine_threshold", DEFAULT_DUPLICATE_COSINE_THRESHOLD)
    lines.append(f"Duplicate QA report (exact string OR cosine >= {thr})")
    if params.get("n_summaries", 1) > 1:
        lines.append(f"  Across {params['n_summaries']} summary file(s)")
    if params.get("sc_kept_only"):
        lines.append("  self_consistency: kept pairs only")

    totals = report.get("totals") or {}
    lines.append(
        f"  Overall: {totals.get('total_qa_pairs', 0)} pairs, "
        f"{totals.get('unique_clusters', 0)} unique clusters, "
        f"{totals.get('redundant_pairs', 0)} redundant"
    )

    by_strategy = report.get("by_strategy") or {}
    for strategy in (
        *STRATEGY_DISPLAY_ORDER,
        *sorted(s for s in by_strategy if s not in STRATEGY_DISPLAY_ORDER),
    ):
        if strategy not in by_strategy:
            continue
        s = by_strategy[strategy]
        lines.append("")
        lines.append(
            f"[{strategy}] {s.get('total_qa_pairs', 0)} total, "
            f"{s.get('unique_clusters', 0)} unique, "
            f"{s.get('redundant_pairs', 0)} redundant "
            f"({len(s.get('duplicate_clusters') or [])} duplicate cluster(s))"
        )
        if not s.get("cosine_matching_enabled"):
            lines.append("  (cosine matching off — exact string duplicates only)")

        clusters = s.get("duplicate_clusters") or []
        for i, cluster in enumerate(clusters[:max_clusters]):
            match = cluster.get("match", "?")
            max_cos = cluster.get("max_cosine")
            cos_note = f", max_cos={max_cos:.3f}" if max_cos is not None and match != "exact" else ""
            members = cluster.get("members") or []
            locs = ", ".join(
                f"{m.get('num_stages')}st#{m.get('pair_index')}"
                + (f"@{m['source_file']}" if m.get("source_file") else "")
                for m in members
            )
            preview = _truncate_question((members[0].get("question") if members else "") or "")
            lines.append(
                f"  cluster {i + 1} ({cluster.get('size')}x, {match}{cos_note}): "
                f"[{locs}]  «{preview}»"
            )
        if len(clusters) > max_clusters:
            lines.append(f"  ... {len(clusters) - max_clusters} more cluster(s)")

    return "\n".join(lines)


def run_experiment(
    metrics_path: Path,
    *,
    stage_counts: List[int],
    strategies: List[str],
    seed: Optional[int],
    temperature: float,
    n_questions: int,
    output_dir: Path,
    embedding_model_type: str = "sentence-transformer",
    compute_diversity: bool = True,
    sc_m: int = 3,
    sc_dedup_threshold: float = 0.85,
    sc_min_consistent_votes: Optional[int] = None,
    raw_max_chars_per_stage: Optional[int] = None,
    stage_selection: str = "fixed",
    allowed_stage_types: Optional[List[str]] = None,
    fixed_stage_sets: Optional[Dict[int, List[str]]] = None,
    dce_validate_against_direct: bool = True,
    dce_validation_threshold: float = 0.8,
) -> Dict[str, Any]:
    """Run all (strategy, num_stages) cells and persist a summary JSON."""
    reset_qa_generation_stats()

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = (
        output_dir / f"direct_vs_qtree_{Path(metrics_path).stem}_{timestamp}.json"
    ).resolve()
    print("\n" + "=" * 80)
    print("[OUTPUT] Run paths:")
    print(f"  output_dir : {output_dir.resolve()}")
    print(f"  summary    : {out_path}")
    print("=" * 80)

    embedding_model = None
    if compute_diversity or "self_consistency" in strategies:
        try:
            embedding_model = get_embedding_model(model_type=embedding_model_type)
            print(f"[Vendi/SC] Using embedding backend: {embedding_model_type}")
        except Exception as exc:  # noqa: BLE001
            print(
                f"[Vendi/SC] Failed to initialize embedding model "
                f"({type(exc).__name__}: {exc}); diversity will be skipped "
                f"and self-consistency will fall back to exact-string dedupe."
            )
            embedding_model = None

    effective_allowed = (
        list(allowed_stage_types)
        if allowed_stage_types is not None
        else list(DEFAULT_FOCUS_STAGE_TYPES)
    )
    effective_fixed = (
        {k: list(v) for k, v in fixed_stage_sets.items()}
        if fixed_stage_sets is not None
        else {k: list(v) for k, v in DEFAULT_FIXED_STAGE_SETS.items()}
    )
    if stage_selection == "fixed":
        print(
            f"[FIXED stages] "
            + "; ".join(
                f"num_stages={k}: {v}" for k, v in sorted(effective_fixed.items())
            )
        )
    else:
        print(
            f"[FILTER] eligible stage types = {effective_allowed} "
            f"(synthesis/floorplan/final excluded by default)"
        )

    # Enforce execution order: Direct -> Self-Consistency -> Q-Tree (the
    # only hard requirement is Direct before SC, since SC consumes Direct's
    # output). If the user asked for SC but not Direct, auto-include Direct
    # and emit a clear notice.
    _ORDER = {
        "direct": 0,
        "self_consistency": 1,
        "direct_code_extraction": 2,
        "qtree": 3,
    }
    requested = list(strategies)
    _needs_direct = {"self_consistency", "direct_code_extraction"}
    for strat in _needs_direct:
        if strat in requested and "direct" not in requested:
            print(
                f"[NOTE] '{strat}' requires 'direct' as input; "
                "auto-adding 'direct' to the run."
            )
            requested.append("direct")
            break
    ordered_strategies = sorted(set(requested), key=lambda s: _ORDER.get(s, 99))

    runs: List[RunMetrics] = []
    selected_per_num_stages: Dict[int, List[Dict[str, Any]]] = {}
    for num_stages in stage_counts:
        preselected_stages = _select_stages(
            metrics_path,
            num_stages=num_stages,
            seed=seed,
            mode=stage_selection,
            allowed_stage_types=effective_allowed,
            fixed_stage_sets=effective_fixed,
        )
        selected_per_num_stages[num_stages] = preselected_stages
        chosen_desc = ", ".join(
            f"{s.get('stage')}#{s.get('occurrence', 1)}"
            f"@{s.get('sub_design_name')}"
            f"(seq={s.get('sequence_number')}, "
            f"richness={_stage_richness(s)}B)"
            for s in preselected_stages
        )
        print(
            f"\n[SELECT mode={stage_selection}] num_stages={num_stages} "
            f"-> {chosen_desc} (shared across {len(strategies)} strategies)"
        )

        direct_qa_pairs_cache: Optional[List[Dict[str, Any]]] = None
        for strategy in ordered_strategies:
            cell, full_qa_pairs = run_one_cell(
                strategy=strategy,
                num_stages=num_stages,
                metrics_path=metrics_path,
                seed=seed,
                temperature=temperature,
                n_questions=n_questions,
                embedding_model=embedding_model,
                sc_m=sc_m,
                sc_dedup_threshold=sc_dedup_threshold,
                sc_min_consistent_votes=sc_min_consistent_votes,
                preselected_stages=preselected_stages,
                raw_max_chars_per_stage=raw_max_chars_per_stage,
                direct_qa_pairs_for_sc=(
                    direct_qa_pairs_cache
                    if strategy in ("self_consistency", "direct_code_extraction")
                    else None
                ),
                dce_validate_against_direct=dce_validate_against_direct,
                dce_validation_threshold=dce_validation_threshold,
            )
            runs.append(cell)
            if strategy == "direct":
                direct_qa_pairs_cache = full_qa_pairs

    summary = {
        "metrics_path": str(metrics_path),
        "timestamp": timestamp,
        "params": {
            "stage_counts": list(stage_counts),
            "strategies_requested": list(strategies),
            "strategies_run_ordered": list(ordered_strategies),
            "seed": seed,
            "temperature": temperature,
            "n_questions": n_questions,
            "diversity_metric": "vendi_score",
            "embedding_backend": embedding_model_type if embedding_model is not None else None,
            "direct": {
                "input": "raw_extract_text",
            },
            "self_consistency": {
                "mode": "consistency_vote",
                "input_for_questions": "direct_qa_pairs",
                "input_for_judgment": "raw_extract_text",
                "M": sc_m,
                "min_consistent_votes": sc_min_consistent_votes,
                "dedup_threshold": sc_dedup_threshold,
            },
            "direct_code_extraction": {
                "input_for_questions": "direct_qa_pairs",
                "input_for_code": "metrics_json",
                "validate_against_direct": dce_validate_against_direct,
                "validation_threshold": dce_validation_threshold,
            },
            "raw_max_chars_per_stage": raw_max_chars_per_stage,
            "stage_selection": stage_selection,
            "allowed_stage_types": effective_allowed,
            "fixed_stage_sets": {
                str(k): v for k, v in effective_fixed.items()
            },
        },
        "selected_stages_by_num_stages": {
            str(k): [
                {**info, "richness_bytes": _stage_richness(stage)}
                for info, stage in zip(_stages_to_info(v), v)
            ]
            for k, v in selected_per_num_stages.items()
        },
        "runs": [r.to_dict() for r in runs],
        "strategy_averages": aggregate_strategy_averages(
            runs,
            embedding_model if compute_diversity else None,
            sc_kept_only=True,
        ),
        "totals": get_qa_generation_stats(),
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    try:
        out_size_kb = out_path.stat().st_size / 1024.0
    except OSError:
        out_size_kb = float("nan")

    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY (per cell)")
    print("=" * 80)
    print(_format_table(runs))
    print("\n" + "=" * 80)
    print("COMBINED BY STRATEGY (pooled over all QA pairs)")
    print(f"({len(runs)} cells; Vendi/quality/dup stats use full QA sets, not per-cell means)")
    print("=" * 80)
    print(_format_strategy_averages_table(summary["strategy_averages"]))
    print("\n" + "=" * 80)
    print("[OUTPUT] Written:")
    print(f"  output_dir : {output_dir.resolve()}")
    print(f"  summary    : {out_path}  ({out_size_kb:.1f} KB)")
    print("=" * 80)

    return summary


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _default_metrics_path() -> Optional[Path]:
    candidates = sorted((REPO_ROOT / "extracted_logs").glob("*_all_stages_metrics.json"))
    return candidates[0] if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare direct vs QTree-driven QA pair generation across 1/2/3 stage settings.",
    )
    default_metrics = _default_metrics_path()
    parser.add_argument(
        "--metrics",
        type=Path,
        default=default_metrics,
        help=(
            "Path to an `*_all_stages_metrics.json` file. "
            f"Default: {default_metrics if default_metrics else 'no default available'}"
        ),
    )
    parser.add_argument(
        "--stage-counts",
        type=int,
        nargs="+",
        default=list(DEFAULT_STAGE_COUNTS),
        help="Stage counts to study (default: 1 2 3).",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=["direct", "self_consistency", "direct_code_extraction", "qtree"],
        default=["direct", "self_consistency", "qtree"],
        help=(
            "Strategies to run (default: direct self_consistency qtree). "
            "Add 'direct_code_extraction' for the DCE ablation."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="LLM temperature for both strategies (default: 0.7).",
    )
    parser.add_argument(
        "--n-questions",
        type=int,
        default=DEFAULT_N_QUESTIONS,
        help=(
            f"Target number of QA pairs for Direct (prompt) and Q-Tree "
            f"(question-generation cap). Default: {DEFAULT_N_QUESTIONS}."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "experiment_results" / "direct_vs_qtree",
        help="Directory to write summary JSON.",
    )
    parser.add_argument(
        "--embedding-backend",
        choices=["sentence-transformer", "openai", "keyword"],
        default="sentence-transformer",
        help="Embedding backend used for the Vendi diversity score.",
    )
    parser.add_argument(
        "--no-diversity",
        action="store_true",
        help="Skip Vendi diversity computation (e.g., if embeddings are unavailable).",
    )
    parser.add_argument(
        "--raw-max-chars",
        type=int,
        default=None,
        help=(
            "Optional cap on the raw EDA log size per stage given to "
            "Direct AND to Self-Consistency. Default = no cap (use the "
            "full extract). When set, the middle of each log is "
            "truncated so the head (command + early summary) and tail "
            "(final summary) are preserved."
        ),
    )
    parser.add_argument(
        "--sc-m",
        type=int,
        default=3,
        help=(
            "Self-Consistency: number of YES/NO consistency votes per "
            "Direct QA pair (default: 3). For each pair the LLM is asked "
            "M times whether the proposed answer is consistent with the "
            "metrics; the pair is kept if YES votes >= "
            "--sc-min-consistent-votes."
        ),
    )
    parser.add_argument(
        "--sc-min-consistent-votes",
        type=int,
        default=None,
        help=(
            "Minimum number of YES (consistent) votes required to keep a "
            "Direct QA pair. Default = strict majority of M, i.e. "
            "floor(M/2)+1 (so for M=3 the default is 2). Set to M for a "
            "unanimous-only filter."
        ),
    )
    parser.add_argument(
        "--sc-dedup-threshold",
        type=float,
        default=0.85,
        help=(
            "Legacy flag (ignored). Self-Consistency no longer semantically "
            "dedupes Direct pairs; all N Direct questions are validated."
        ),
    )
    parser.add_argument(
        "--stage-selection",
        choices=["fixed", "random", "richest"],
        default="fixed",
        help=(
            "How to pick the K sampled stages used by ALL strategies.\n"
            "  fixed   (default): use --fixed-1stage / --fixed-2stages / "
            "--fixed-3stages explicitly. Removes seed-driven variance.\n"
            "  richest          : top-K by serialized metrics size, "
            "constrained by --include-stages.\n"
            "  random           : seeded random via sample_stages, "
            "constrained by --include-stages."
        ),
    )
    parser.add_argument(
        "--include-stages",
        nargs="+",
        default=list(DEFAULT_FOCUS_STAGE_TYPES),
        metavar="STAGE",
        help=(
            "Stage types eligible for sampling under --stage-selection "
            "random/richest. Default: placement cts routing. Ignored when "
            "--stage-selection=fixed."
        ),
    )
    parser.add_argument(
        "--fixed-1stage",
        nargs=1,
        default=list(DEFAULT_FIXED_STAGE_SETS[1]),
        metavar="STAGE",
        help="Stage type for the 1-stage cell when --stage-selection=fixed (default: routing).",
    )
    parser.add_argument(
        "--fixed-2stages",
        nargs=2,
        default=list(DEFAULT_FIXED_STAGE_SETS[2]),
        metavar="STAGE",
        help="Two stage types for the 2-stage cell when --stage-selection=fixed (default: placement routing).",
    )
    parser.add_argument(
        "--fixed-3stages",
        nargs=3,
        default=list(DEFAULT_FIXED_STAGE_SETS[3]),
        metavar="STAGE",
        help="Three stage types for the 3-stage cell when --stage-selection=fixed (default: placement cts routing).",
    )
    parser.add_argument(
        "--dce-no-validate",
        action="store_true",
        help=(
            "Direct Code Extraction: skip LLM validation of code answers "
            "against Direct's original answers. Keeps any non-null code answer."
        ),
    )
    parser.add_argument(
        "--dce-validation-threshold",
        type=float,
        default=0.8,
        help=(
            "Direct Code Extraction: minimum LLM judge score to keep a "
            "code-extracted answer (default: 0.8)."
        ),
    )
    args = parser.parse_args()

    if args.metrics is None or not args.metrics.exists():
        raise SystemExit(
            "Could not find a metrics JSON. Pass --metrics <path-to-*_all_stages_metrics.json>."
        )

    fixed_sets = {
        1: list(args.fixed_1stage),
        2: list(args.fixed_2stages),
        3: list(args.fixed_3stages),
    }

    run_experiment(
        metrics_path=args.metrics,
        stage_counts=args.stage_counts,
        strategies=args.strategies,
        seed=args.seed,
        temperature=args.temperature,
        n_questions=args.n_questions,
        output_dir=args.output_dir,
        embedding_model_type=args.embedding_backend,
        compute_diversity=not args.no_diversity,
        sc_m=args.sc_m,
        sc_dedup_threshold=args.sc_dedup_threshold,
        sc_min_consistent_votes=args.sc_min_consistent_votes,
        raw_max_chars_per_stage=args.raw_max_chars,
        stage_selection=args.stage_selection,
        allowed_stage_types=args.include_stages,
        fixed_stage_sets=fixed_sets,
        dce_validate_against_direct=not args.dce_no_validate,
        dce_validation_threshold=args.dce_validation_threshold,
    )


if __name__ == "__main__":
    main()
