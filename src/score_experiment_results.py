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
Recompute Vendi and QA quality metrics on saved experiment summary JSON files.

Use this after a run (or on older JSON files) without re-calling the LLM.

Metrics written under each ``runs[]`` entry in ``diversity``:
  - ``questions`` / ``answers`` / ``qa_combined`` — Vendi score (+ normalized)
  - ``quality`` — stage coverage, multi-stage %, question length, comparative %

Also writes top-level ``strategy_averages`` (mean per strategy across all cells;
Vendi columns are ``vendi/n`` normalized scores) and prints a **COMBINED AVERAGE
BY STRATEGY** table after the per-cell table.

Examples
--------
Score one file and print the table (writes ``*_scored.json`` next to input):

    python src/score_experiment_results.py \\
        experiment_results/direct_vs_qtree/direct_vs_qtree_ariane133_nan45_all_stages_metrics_20260518_134034.json

Score in place (overwrite the original JSON):

    python src/score_experiment_results.py path/to/summary.json --write-in-place

Score every JSON in a directory (quality only, no embedding model):

    python src/score_experiment_results.py experiment_results/direct_vs_qtree/ \\
        --quality-only --write-in-place

Score multiple files:

    python src/score_experiment_results.py results/a.json results/b.json

Detect duplicate questions across 1/2/3-stage cells (98% cosine or exact string):

    python src/score_experiment_results.py path/to/summary.json --check-duplicates

Duplicates only (no rescoring):

    python src/score_experiment_results.py path/to/summary.json --duplicates-only

Extract borderline QA pairs near the threshold for human review:

    python src/score_experiment_results.py path/to/summary.json --borderline
    python src/score_experiment_results.py path/to/summary.json --borderline --borderline-lo 0.5 --borderline-hi 0.95
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experiment_direct_vs_qtree_qa import (  # noqa: E402
    DEFAULT_DUPLICATE_COSINE_THRESHOLD,
    _format_strategy_averages_table,
    _format_table,
    aggregate_strategy_averages,
    detect_qa_duplicates_across_cells,
    detect_qa_duplicates_across_summaries,
    format_qa_duplicate_report,
    run_dict_to_metrics,
    score_experiment_summary,
)
from embedding_utils import get_embedding_model  # noqa: E402


def _truncate(text: str, max_len: int = 120) -> str:
    text = " ".join(str(text).split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def extract_borderline_qa_pairs(
    summary: Dict[str, Any],
    *,
    score_lo: float = 0.7,
    score_hi: float = 0.99,
    include_rejected: bool = True,
    include_kept: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """Extract QA pairs with scores near the accept/reject threshold.

    Scans all strategies that carry per-pair scores:

    - **self_consistency**: ``consistent_rate`` from the ``vote`` block.
    - **direct_code_extraction**: ``validation_score`` on kept pairs, plus
      ``score`` from ``rejection_stats.low_score_details`` for rejected pairs.
    - **qtree**: ``validation_score`` on kept pairs, plus
      ``score`` from ``rejection_stats.low_score_details`` for rejected pairs.
      (Kept pairs from older runs without ``validation_score`` default to 1.0.)

    Returns ``{strategy: [entry, ...]}`` where each entry has:
      ``strategy``, ``num_stages``, ``question``, ``answer`` (or ``code_answer``),
      ``score``, ``kept``, ``drop_reason``, ``direct_answer`` (if available),
      ``vote_detail`` (SC), ``reasoning`` (SC vote reasons).
    """
    results: Dict[str, List[Dict[str, Any]]] = {}

    for run in summary.get("runs") or []:
        strategy = str(run.get("strategy") or "")
        num_stages = int(run.get("num_stages") or 0)

        if strategy == "self_consistency":
            for qa in run.get("qa_pairs") or []:
                vote = qa.get("vote") or {}
                rate = vote.get("consistent_rate")
                if rate is None:
                    continue
                if not (score_lo <= rate <= score_hi):
                    continue
                kept = bool(qa.get("kept"))
                if (kept and not include_kept) or (not kept and not include_rejected):
                    continue
                entry = {
                    "strategy": strategy,
                    "num_stages": num_stages,
                    "question": qa.get("question"),
                    "answer": qa.get("answer"),
                    "score": rate,
                    "kept": kept,
                    "drop_reason": qa.get("drop_reason"),
                    "vote_detail": {
                        "yes": vote.get("yes_votes"),
                        "no": vote.get("no_votes"),
                        "parse_failed": vote.get("parse_failed"),
                        "threshold": vote.get("threshold"),
                    },
                    "reasoning": [
                        v.get("reason", "")
                        for v in (qa.get("consistency_votes") or [])
                    ],
                }
                results.setdefault(strategy, []).append(entry)

        elif strategy == "direct_code_extraction":
            if include_kept:
                for qa in run.get("qa_pairs") or []:
                    score = qa.get("validation_score")
                    if score is None:
                        continue
                    if not (score_lo <= score <= score_hi):
                        continue
                    entry = {
                        "strategy": strategy,
                        "num_stages": num_stages,
                        "question": qa.get("question"),
                        "answer": qa.get("answer"),
                        "direct_answer": qa.get("direct_answer"),
                        "score": score,
                        "kept": True,
                        "drop_reason": None,
                    }
                    results.setdefault(strategy, []).append(entry)

            if include_rejected:
                rej = run.get("rejection_stats") or {}
                for detail in rej.get("low_score_details") or []:
                    score = detail.get("score")
                    if score is None:
                        continue
                    if not (score_lo <= score <= score_hi):
                        continue
                    entry = {
                        "strategy": strategy,
                        "num_stages": num_stages,
                        "question": detail.get("question"),
                        "answer": detail.get("code_answer"),
                        "direct_answer": detail.get("direct_answer"),
                        "score": score,
                        "kept": False,
                        "drop_reason": "low_validation_score",
                    }
                    results.setdefault(strategy, []).append(entry)

        elif strategy == "qtree":
            if include_kept:
                for qa in run.get("qa_pairs") or []:
                    score = qa.get("validation_score")
                    if score is None:
                        score = 1.0
                    if not (score_lo <= score <= score_hi):
                        continue
                    entry = {
                        "strategy": strategy,
                        "num_stages": num_stages,
                        "question": qa.get("question"),
                        "answer": qa.get("answer"),
                        "score": score,
                        "kept": True,
                        "drop_reason": None,
                    }
                    results.setdefault(strategy, []).append(entry)

            if include_rejected:
                rej = run.get("rejection_stats") or {}
                for detail in rej.get("low_score_details") or []:
                    score = detail.get("score")
                    if score is None:
                        continue
                    if not (score_lo <= score <= score_hi):
                        continue
                    entry = {
                        "strategy": strategy,
                        "num_stages": num_stages,
                        "question": detail.get("question"),
                        "answer": detail.get("code_answer"),
                        "direct_answer": detail.get("direct_answer"),
                        "score": score,
                        "kept": False,
                        "drop_reason": "low_score_rejected",
                    }
                    results.setdefault(strategy, []).append(entry)

    for pairs in results.values():
        pairs.sort(key=lambda e: (e["score"], e.get("num_stages", 0)))

    return results


def format_borderline_report(
    borderline: Dict[str, List[Dict[str, Any]]],
    *,
    score_lo: float,
    score_hi: float,
) -> str:
    """Human-readable report of borderline QA pairs for verification."""
    lines: List[str] = []
    lines.append(
        f"Borderline QA pairs (score in [{score_lo:.2f}, {score_hi:.2f}]) "
        "— candidates for human verification"
    )

    total = sum(len(v) for v in borderline.values())
    if total == 0:
        lines.append("  (none found)")
        return "\n".join(lines)

    lines.append(f"  Total: {total} pair(s) across {len(borderline)} strategy(ies)")

    order = ("self_consistency", "direct_code_extraction", "qtree")
    for strategy in (*order, *sorted(s for s in borderline if s not in order)):
        pairs = borderline.get(strategy)
        if not pairs:
            continue
        kept_count = sum(1 for p in pairs if p["kept"])
        rej_count = len(pairs) - kept_count
        lines.append("")
        lines.append(
            f"[{strategy}] {len(pairs)} borderline pair(s) "
            f"({kept_count} kept, {rej_count} rejected)"
        )
        for i, p in enumerate(pairs, 1):
            status = "KEPT" if p["kept"] else f"DROPPED ({p.get('drop_reason', '?')})"
            lines.append(
                f"  {i}. [{status}] score={p['score']:.2f}  "
                f"stages={p['num_stages']}"
            )
            lines.append(f"     Q: {p.get('question', '')}")
            lines.append(f"     A: {p.get('answer', '')}")
            if p.get("direct_answer") is not None:
                lines.append(
                    f"     Direct A: {p.get('direct_answer', '')}"
                )
            if p.get("vote_detail"):
                vd = p["vote_detail"]
                lines.append(
                    f"     Votes: yes={vd.get('yes')}, no={vd.get('no')}, "
                    f"parse_failed={vd.get('parse_failed')}, "
                    f"threshold={vd.get('threshold')}"
                )
            if p.get("reasoning"):
                for j, r in enumerate(p["reasoning"], 1):
                    if r:
                        lines.append(f"     Vote {j}: {r}")

    return "\n".join(lines)


def _load_summary(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _default_out_path(src: Path, write_in_place: bool) -> Path:
    if write_in_place:
        return src
    return src.with_name(f"{src.stem}_scored{src.suffix}")


def score_json_file(
    path: Path,
    *,
    embedding_model=None,
    recompute_vendi: bool = True,
    sc_use_kept_only: bool = True,
    write_in_place: bool = False,
    output_path: Path | None = None,
) -> Path:
    """Load, score, save one summary JSON. Returns path written."""
    summary = _load_summary(path)
    score_experiment_summary(
        summary,
        embedding_model,
        recompute_vendi=recompute_vendi,
        sc_use_kept_only=sc_use_kept_only,
    )
    out = output_path or _default_out_path(path, write_in_place)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    return out


def _collect_inputs(paths: List[Path]) -> List[Path]:
    files: List[Path] = []
    for p in paths:
        p = p.resolve()
        if p.is_dir():
            files.extend(sorted(p.glob("direct_vs_qtree_*.json")))
            files.extend(sorted(p.glob("*_all_stages_metrics_*.json")))
            # Also pick plain experiment summaries in the folder
            for f in sorted(p.glob("*.json")):
                if f not in files and "scored" not in f.stem:
                    files.append(f)
        elif p.is_file():
            files.append(p)
        else:
            print(f"[WARN] not found, skipping: {p}")
    # De-dupe while preserving order
    seen: set[Path] = set()
    unique: List[Path] = []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute Vendi + QA quality metrics on saved experiment JSON files.",
    )
    parser.add_argument(
        "json_paths",
        nargs="+",
        type=Path,
        help="One or more summary JSON files, or a directory containing them.",
    )
    parser.add_argument(
        "--write-in-place",
        action="store_true",
        help="Overwrite each input JSON instead of writing *_scored.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Write all scored outputs to this directory (filenames preserved).",
    )
    parser.add_argument(
        "--quality-only",
        action="store_true",
        help="Skip Vendi (no embedding model). Only recompute structural quality metrics.",
    )
    parser.add_argument(
        "--embedding-backend",
        choices=["sentence-transformer", "openai", "keyword"],
        default="sentence-transformer",
        help="Embedding backend for Vendi (default: sentence-transformer).",
    )
    parser.add_argument(
        "--sc-all-candidates",
        action="store_true",
        help=(
            "For self_consistency rows, compute Vendi on ALL candidates "
            "(including filtered), not only kept pairs."
        ),
    )
    parser.add_argument(
        "--check-duplicates",
        action="store_true",
        help=(
            "Detect duplicate questions per strategy across cells (and across "
            "files when multiple JSONs are given). Uses exact match or cosine "
            f">= threshold (default {DEFAULT_DUPLICATE_COSINE_THRESHOLD})."
        ),
    )
    parser.add_argument(
        "--duplicate-cosine-threshold",
        type=float,
        default=DEFAULT_DUPLICATE_COSINE_THRESHOLD,
        help=(
            "Cosine similarity threshold for near-duplicate questions "
            f"(default: {DEFAULT_DUPLICATE_COSINE_THRESHOLD})."
        ),
    )
    parser.add_argument(
        "--duplicates-only",
        action="store_true",
        help="Only run duplicate detection (skip Vendi/quality rescoring).",
    )
    parser.add_argument(
        "--borderline",
        action="store_true",
        help=(
            "Extract QA pairs with scores near the accept/reject threshold "
            "for human verification. Applies to self_consistency, "
            "direct_code_extraction, and qtree."
        ),
    )
    parser.add_argument(
        "--borderline-lo",
        type=float,
        default=0.7,
        help="Lower bound of the borderline score range (default: 0.7).",
    )
    parser.add_argument(
        "--borderline-hi",
        type=float,
        default=0.99,
        help="Upper bound of the borderline score range (default: 0.99).",
    )
    args = parser.parse_args()

    files = _collect_inputs(list(args.json_paths))
    if not files:
        raise SystemExit("No JSON files found.")

    need_embeddings = (
        not args.quality_only
        and not args.duplicates_only
    ) or (
        args.check_duplicates or args.duplicates_only
    )
    embedding_model = None
    if need_embeddings:
        try:
            embedding_model = get_embedding_model(model_type=args.embedding_backend)
            print(f"[score] embedding backend: {args.embedding_backend}")
        except Exception as exc:  # noqa: BLE001
            print(
                f"[WARN] embedding init failed ({exc}); "
                "falling back to quality-only / exact-string duplicate detection."
            )
            embedding_model = None

    all_rows: List[Any] = []
    scored_summaries: List[Dict[str, Any]] = []
    sc_kept = not args.sc_all_candidates
    for path in files:
        print(f"\n[score] {path}")
        summary = _load_summary(path)
        if not args.duplicates_only:
            score_experiment_summary(
                summary,
                embedding_model,
                recompute_vendi=embedding_model is not None and not args.quality_only,
                sc_use_kept_only=not args.sc_all_candidates,
            )
            out_path = None
            if args.output_dir is not None:
                out_path = args.output_dir.resolve() / path.name
            out = out_path or _default_out_path(path, args.write_in_place)
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
            written = out
            print(f"[score] wrote → {written}")
            summary = _load_summary(written)
        else:
            written = path
            print("[score] duplicates-only — file not rescored")

        scored_summaries.append(summary)
        for run in summary.get("runs") or []:
            all_rows.append(run_dict_to_metrics(run))

    dup_report = None
    if scored_summaries and all_rows:
        if len(scored_summaries) == 1:
            dup_report = detect_qa_duplicates_across_cells(
                scored_summaries[0],
                embedding_model,
                cosine_threshold=args.duplicate_cosine_threshold,
                sc_kept_only=sc_kept,
            )
        else:
            dup_report = detect_qa_duplicates_across_summaries(
                scored_summaries,
                embedding_model,
                cosine_threshold=args.duplicate_cosine_threshold,
                sc_kept_only=sc_kept,
            )

    if args.check_duplicates or args.duplicates_only:
        if dup_report is None:
            raise SystemExit("No runs found for duplicate detection.")
        print("\n" + "=" * 80)
        print("DUPLICATE QA DETECTION (across cells)")
        print("=" * 80)
        print(format_qa_duplicate_report(dup_report))

    if all_rows and not args.duplicates_only:
        print("\n" + "=" * 80)
        print("SCORED SUMMARY (per cell)")
        print("=" * 80)
        print(_format_table(all_rows))

        # --- Overall combined table (all stage counts) ---
        strategy_avgs = aggregate_strategy_averages(
            all_rows,
            embedding_model,
            duplicate_report=dup_report,
            sc_kept_only=sc_kept,
            cosine_threshold=args.duplicate_cosine_threshold,
        )
        print("\n" + "=" * 80)
        print("COMBINED BY STRATEGY — ALL STAGES (pooled over all QA pairs)")
        print(
            f"({len(all_rows)} cells across {len(files)} file(s); "
            "metrics pooled across every QA, not averaged per cell)"
        )
        print("=" * 80)
        print(_format_strategy_averages_table(strategy_avgs))

        # --- Split: single-stage (1) vs multi-stage (2+3) ---
        single_rows = [r for r in all_rows if r.num_stages == 1]
        multi_rows = [r for r in all_rows if r.num_stages >= 2]

        stage_split_avgs: Dict[str, Any] = {"all": strategy_avgs}

        if single_rows:
            single_avgs = aggregate_strategy_averages(
                single_rows,
                embedding_model,
                sc_kept_only=sc_kept,
                cosine_threshold=args.duplicate_cosine_threshold,
            )
            stage_split_avgs["single_stage"] = single_avgs
            print("\n" + "=" * 80)
            print("COMBINED BY STRATEGY — SINGLE-STAGE ONLY (num_stages=1)")
            print(f"({len(single_rows)} cell(s))")
            print("=" * 80)
            print(_format_strategy_averages_table(single_avgs))

        if multi_rows:
            multi_avgs = aggregate_strategy_averages(
                multi_rows,
                embedding_model,
                sc_kept_only=sc_kept,
                cosine_threshold=args.duplicate_cosine_threshold,
            )
            stage_split_avgs["multi_stage"] = multi_avgs
            stages_in = sorted({r.num_stages for r in multi_rows})
            print("\n" + "=" * 80)
            print(
                f"COMBINED BY STRATEGY — MULTI-STAGE ONLY "
                f"(num_stages={','.join(str(s) for s in stages_in)})"
            )
            print(f"({len(multi_rows)} cell(s), QA pairs pooled across stage counts)")
            print("=" * 80)
            print(_format_strategy_averages_table(multi_avgs))

        if args.write_in_place and len(scored_summaries) == 1:
            scored_summaries[0]["strategy_averages"] = stage_split_avgs
            if dup_report is not None:
                scored_summaries[0]["qa_duplicate_report"] = dup_report
            out = files[0].resolve()
            with out.open("w", encoding="utf-8") as f:
                json.dump(scored_summaries[0], f, indent=2, ensure_ascii=False, default=str)
            print(f"[score] strategy_averages (all / single / multi) → {out}")

    if args.borderline:
        all_borderline: Dict[str, List[Dict[str, Any]]] = {}
        for summary in scored_summaries:
            bl = extract_borderline_qa_pairs(
                summary,
                score_lo=args.borderline_lo,
                score_hi=args.borderline_hi,
            )
            for strat, pairs in bl.items():
                all_borderline.setdefault(strat, []).extend(pairs)
        for pairs in all_borderline.values():
            pairs.sort(key=lambda e: (e["score"], e.get("num_stages", 0)))

        print("\n" + "=" * 80)
        print("BORDERLINE QA PAIRS (for human verification)")
        print("=" * 80)
        print(format_borderline_report(
            all_borderline,
            score_lo=args.borderline_lo,
            score_hi=args.borderline_hi,
        ))

        if args.write_in_place and len(scored_summaries) == 1:
            scored_summaries[0]["borderline_qa_pairs"] = all_borderline
            out = files[0].resolve()
            with out.open("w", encoding="utf-8") as f:
                json.dump(scored_summaries[0], f, indent=2, ensure_ascii=False, default=str)
            print(f"\n[score] borderline_qa_pairs → {out}")

    if all_rows and not args.duplicates_only:
        print("\nColumns (per-cell table):")
        print("  qa_pairs (SC) — kept/direct_total (e.g. 4/6 = 4 passed consistency of 6 Direct pairs)")
        print("  success_rate (SC) — retention % = kept / Direct total (not vote-pass among deduped)")
        print("  stg_cnt  — mean effective # input stages (0 keywords → 1; else raw count; range 1…num_stages)")
        print("  multi_%  — % questions mentioning ≥2 INPUT stages (raw keywords)")
        print("  q_chars  — mean question length (characters)")
        print("  cmp_%    — % questions with comparative language (change, vs, across, …)")
        print("\nCombined strategy table:")
        print("  total_qa — all QA pairs pooled across 1/2/3-stage cells")
        print("  unique_q / redundant — duplicate clusters (exact or cosine >= threshold)")
        print("  vendi_* — Vendi/n computed on the full pooled set")
        print("  stg_cnt, multi_%, q_chars, cmp_% — means over every pooled QA pair")
        print("  success% (SC) — total kept / total Direct candidates across all cells")


if __name__ == "__main__":
    main()
