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

"""Failure-case analysis across multiple ``qa_output_<agent>_*`` directories.

This module post-processes the existing benchmark artifacts:

    qa_output_<agent>_<design>/
        pass_at_k_<design>_<model>_pass<i>of<n>_<ts>.json              (raw answers)
        pass_at_k_<design>_<model>_pass<i>of<n>_<ts>_evaluation.json   (LLM-judge eval)
        pass_at_k_<design>_<model>_pass<i>of<n>_<ts>_evaluation.sum    (text summary)

It produces a unified, agent-agnostic view of failures:

* one ``PassRecord`` per evaluated (agent, design, model, pass, question)
* a shared ``FAILURE_TAGS`` taxonomy that applies to every agent
* per-question persistence across pass@k
* aggregates by agent / design / question_type / stage / difficulty / tag
* cross-agent comparison for the same ``question_id``

Outputs are written under ``--output-dir`` as JSONL/CSV/Markdown so they can be
inspected directly or loaded by another notebook.

Run::

    python src/failure_analysis.py \
        --root . \
        --output-dir failure_analysis_output \
        --agents claude_code cursor rag rlm

The tagger is rule-based and stdlib-only by default. An optional LLM tagger
hook is exposed via ``--llm-tagger`` for callers who want to plug in their own
classifier (see :func:`tag_failure_with_llm`).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Failure tag taxonomy (shared across agents)
# ---------------------------------------------------------------------------

FAILURE_TAGS: Dict[str, str] = {
    "wrong_retrieved_value": "Extracted value does not match the golden reference.",
    "wrong_design_scope": "Picked the wrong sub-design / hierarchy block (rcon vs sbox vs top, etc.).",
    "wrong_stage_scope": "Used the wrong flow stage or substage (e.g. CTS substage vs full CTS).",
    "wrong_occurrence": "Picked the wrong repeated occurrence of the same metric/stage.",
    "wrong_comparison_delta": "Comparison / change / range / improvement is wrong (values may exist).",
    "wrong_unit": "Unit or numeric formatting wrong (ps vs ns, KB vs MB, seconds vs minutes).",
    "aggregation_wrong": "Counted, summed, or grouped incorrectly (e.g. substages vs logical stages).",
    "missing_required_field": "Did not provide a field/element required by the answer schema.",
    "extra_conflicting_info": "Final answer may be right but explanation contradicts or confuses it.",
    "hallucinated_context": "Introduced blocks, commands, stages, or metrics that are not in the source.",
    "format_mismatch": "Answer content roughly right but structure (dict/list/keys) differs from spec.",
    "unparseable_response": "Answer could not be reliably parsed from the response.",
    "judge_or_golden_issue": "Suspected evaluator or golden-answer problem rather than agent error.",
    "other": "Failure that does not fit any of the tags above.",
}


# Patterns are intentionally broad; multiple tags can fire on the same record.
_TAG_PATTERNS: List[Tuple[str, re.Pattern]] = [
    (
        "wrong_retrieved_value",
        re.compile(
            r"(?:wrong|incorrect|do(?:es)? not match|mismatch|differ|conflict|"
            r"contradict|not align|deviate|invalid|inaccurate)",
            re.I,
        ),
    ),
    (
        "wrong_design_scope",
        re.compile(
            r"(?:sub[-_ ]?design|sub[-_ ]?block|sub[-_ ]?module|hierarchy|"
            r"top[-_ ]?level|main design|wrong (?:block|design|module)|"
            r"aes[_-](?:rcon|sbox|cipher|block))",
            re.I,
        ),
    ),
    (
        "wrong_stage_scope",
        re.compile(
            r"(?:wrong stage|different stage|stage mismatch|substage|"
            r"final stage .* (?:wrong|incorrect|not)|"
            r"(?:cts|placement|routing|floorplan|synthesis) .* (?:wrong|mismatch|incorrect))",
            re.I,
        ),
    ),
    (
        "wrong_occurrence",
        re.compile(
            r"(?:occurrence|first (?:instance|occurrence)|second (?:instance|occurrence)|"
            r"wrong instance|repeated|repetition)",
            re.I,
        ),
    ),
    (
        "wrong_comparison_delta",
        re.compile(
            r"(?:compar(?:e|ison)|change|difference|delta|gap|improvement|"
            r"range|before and after|largest|smallest|biggest)",
            re.I,
        ),
    ),
    (
        "wrong_unit",
        re.compile(
            r"(?:\bunit\b|ps vs|ns vs|um vs|kb vs|mb vs|seconds vs|minutes vs|"
            r"(?:wrong|incorrect|mismatched) (?:unit|format))",
            re.I,
        ),
    ),
    (
        "aggregation_wrong",
        re.compile(
            r"(?:total (?:stages|number)|number of|count(?:ed)?|"
            r"sum(?:med)?|average|aggregate|grouping|categor(?:y|ies))",
            re.I,
        ),
    ),
    (
        "missing_required_field",
        re.compile(
            r"(?:missing|omit(?:ted|s)?|fail(?:s|ed)? to provide|"
            r"does not identify|does not include|lack(?:s|ing)?|absent)",
            re.I,
        ),
    ),
    (
        "extra_conflicting_info",
        re.compile(
            r"(?:extra(?:neous)?|additional (?:detail|information|data)|"
            r"verbose|unnecessary|out[-_ ]?of[-_ ]?scope|conflict|confusion)",
            re.I,
        ),
    ),
    (
        "hallucinated_context",
        re.compile(
            r"(?:fabricat(?:e|ed|ing)|hallucinat|introduce[sd]? .*"
            r"(?:not in|outside|absent)|made up|not (?:in|present in) the (?:log|reference))",
            re.I,
        ),
    ),
    (
        "format_mismatch",
        re.compile(
            r"(?:dict instead of list|list instead of dict|wrong (?:structure|schema|format)|"
            r"data structure|format mismatch|expected (?:dict|list))",
            re.I,
        ),
    ),
    (
        "unparseable_response",
        re.compile(
            r"(?:could not parse|no (?:answer|json) found|unparseable|"
            r"failed to extract|no extractable)",
            re.I,
        ),
    ),
]


def tag_failure(
    weaknesses: List[str],
    feedback: str,
    response: str = "",
    golden_answer: Any = None,
) -> List[str]:
    """Return a list of failure tags from :data:`FAILURE_TAGS` for one record.

    Tagging is rule-based on the judge's ``weaknesses`` and ``feedback`` text.
    Falls back to ``["other"]`` if nothing matches.
    """

    text = "\n".join(weaknesses or []) + "\n" + (feedback or "")
    tags: List[str] = []
    for tag, pattern in _TAG_PATTERNS:
        if pattern.search(text):
            tags.append(tag)

    # Heuristic: if the response is empty / clearly truncated, mark unparseable.
    if response is not None and len(response.strip()) < 10:
        tags.append("unparseable_response")

    # Deduplicate while preserving order.
    seen = set()
    deduped = [t for t in tags if not (t in seen or seen.add(t))]
    return deduped or ["other"]


def tag_failure_with_llm(
    record: "PassRecord",
    llm_call: Callable[[str], str],
) -> List[str]:
    """Optional LLM-based tagger; pass any callable that takes a prompt string.

    The callable should return the model's raw text. The function asks for a
    JSON list whose entries are restricted to the keys of :data:`FAILURE_TAGS`.
    Failures of the LLM call fall back to the rule-based tagger.
    """

    allowed = list(FAILURE_TAGS.keys())
    prompt = (
        "You are tagging an EDA-flow QA failure. Respond with a JSON list whose "
        f"entries are chosen ONLY from this set: {allowed}. Pick all that apply.\n\n"
        f"Question type: {record.question_type}\n"
        f"Stage: {record.stage}\n"
        f"Golden answer: {json.dumps(record.golden_answer)[:1500]}\n"
        f"Response (truncated): {record.response_excerpt}\n"
        f"Judge weaknesses: {record.weaknesses}\n"
        f"Judge feedback: {record.feedback}\n"
    )
    try:
        raw = llm_call(prompt)
        match = re.search(r"\[.*?\]", raw, re.S)
        if not match:
            raise ValueError("no list in response")
        candidates = json.loads(match.group(0))
        tags = [t for t in candidates if t in FAILURE_TAGS]
        return tags or ["other"]
    except Exception:
        return tag_failure(
            record.weaknesses, record.feedback, record.response_excerpt, record.golden_answer
        )


# ---------------------------------------------------------------------------
# File discovery and parsing
# ---------------------------------------------------------------------------

# qa_output_<agent>_<design>
#
# Agent names may themselves contain underscores (e.g. ``claude_code``), so we
# cannot tokenize on ``_`` alone. The parser instead tries a *longest-prefix*
# match against a known set of agent names. ``--agents`` (when given) is also
# fed into this set so the user can register new agent names ad-hoc.
_DIR_PREFIX = "qa_output_"
_KNOWN_AGENTS: Tuple[str, ...] = (
    "claude_code",
    "cursorv0",
    "cursor",
    "rag",
    "rlm",
    "token_count",
)


def _match_agent_prefix(name: str, candidates: Iterable[str]) -> Optional[str]:
    """Return the longest ``candidate`` that is a prefix of ``name`` followed by ``_``."""
    best: Optional[str] = None
    for cand in candidates:
        if name == cand or name.startswith(cand + "_"):
            if best is None or len(cand) > len(best):
                best = cand
    return best

# pass_at_k_<design>_<model>_pass<i>of<n>_<ts>(.json|_evaluation.json|.sum)
_PASS_FILE_RE = re.compile(
    r"^pass_at_k_(?P<design>.+?)_(?P<model>[A-Za-z0-9.\-]+)"
    r"_pass(?P<idx>\d+)of(?P<total>\d+)_(?P<ts>\d{8}_\d{6})"
    r"(?P<suffix>(?:_evaluation)?)\.json$"
)


@dataclass
class PassRecord:
    agent: str
    design: str
    model: str
    pass_idx: int
    total_passes: int
    timestamp: str
    question_id: str
    question_idx: Optional[int] = None
    question_type: str = ""
    stage: str = ""
    difficulty: str = ""
    answer_format: str = ""
    overall_score: float = 0.0
    correctness: float = 0.0
    clarity: float = 0.0
    relevance: float = 0.0
    outcome: str = "incorrect"  # correct | partial | incorrect | unknown
    tags: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    feedback: str = ""
    evaluation_method: str = ""
    golden_answer: Any = None
    response_excerpt: str = ""
    qa_file: str = ""
    eval_file: str = ""


@dataclass
class PassFilePair:
    qa_path: Path
    eval_path: Path
    agent: str
    design: str
    model: str
    pass_idx: int
    total_passes: int
    timestamp: str


def parse_agent_dir(
    directory: Path,
    known_agents: Iterable[str] = _KNOWN_AGENTS,
) -> Optional[Tuple[str, str]]:
    """Split ``qa_output_<agent>_<design>`` into ``(agent, design)``.

    Uses longest-prefix matching against ``known_agents`` so multi-token
    agent names like ``claude_code`` parse correctly.
    """
    name = directory.name
    if not name.startswith(_DIR_PREFIX):
        return None
    rest = name[len(_DIR_PREFIX) :]
    agent = _match_agent_prefix(rest, known_agents)
    if agent is None:
        return None
    if rest == agent:
        return agent, ""
    return agent, rest[len(agent) + 1 :]


def discover_pass_pairs(
    root: Path,
    agents: Optional[Iterable[str]] = None,
    designs: Optional[Iterable[str]] = None,
    known_agents: Optional[Iterable[str]] = None,
) -> List[PassFilePair]:
    """Discover ``(qa, evaluation)`` JSON file pairs under ``root``.

    ``agents`` filters the results AND extends the agent-name vocabulary used
    when parsing folder names, so passing ``--agents myagent`` alone is enough
    to register a new agent prefix.
    """

    agent_filter = set(agents) if agents else None
    design_filter = set(designs) if designs else None
    candidates = set(_KNOWN_AGENTS)
    if known_agents:
        candidates.update(known_agents)
    if agents:
        candidates.update(agents)

    pairs: List[PassFilePair] = []
    for agent_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        parsed = parse_agent_dir(agent_dir, candidates)
        if not parsed:
            continue
        agent, design = parsed
        if agent_filter and agent not in agent_filter:
            continue
        if design_filter and design not in design_filter:
            continue

        # Index files by their (model, idx, total, ts) key so we can pair them.
        index: Dict[Tuple[str, int, int, str], Dict[str, Path]] = defaultdict(dict)
        for f in agent_dir.glob("*.json"):
            m = _PASS_FILE_RE.match(f.name)
            if not m:
                continue
            key = (
                m.group("model"),
                int(m.group("idx")),
                int(m.group("total")),
                m.group("ts"),
            )
            kind = "eval" if m.group("suffix") == "_evaluation" else "qa"
            index[key][kind] = f

        for (model, idx, total, ts), files in sorted(index.items()):
            qa_path = files.get("qa")
            eval_path = files.get("eval")
            if qa_path is None or eval_path is None:
                continue
            pairs.append(
                PassFilePair(
                    qa_path=qa_path,
                    eval_path=eval_path,
                    agent=agent,
                    design=design,
                    model=model,
                    pass_idx=idx,
                    total_passes=total,
                    timestamp=ts,
                )
            )
    return pairs


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _classify_outcome(correctness: float) -> str:
    if correctness >= 0.999:
        return "correct"
    if correctness <= 0.001:
        return "incorrect"
    return "partial"


def _excerpt(text: str, limit: int = 600) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit // 2] + " ... " + text[-limit // 2 :]


def build_pass_records(
    pair: PassFilePair,
    tagger: Optional[Callable[["PassRecord"], List[str]]] = None,
    response_excerpt_chars: int = 600,
) -> List[PassRecord]:
    """Materialize :class:`PassRecord` items for a (qa, evaluation) pair."""

    qa_data = _load_json(pair.qa_path)
    eval_data = _load_json(pair.eval_path)

    qa_by_id: Dict[str, Dict[str, Any]] = {}
    if isinstance(qa_data, list):
        for item in qa_data:
            qa_by_id[item.get("question_id", "")] = item
    elif isinstance(qa_data, dict) and "qa_pairs" in qa_data:
        for item in qa_data.get("qa_pairs", []):
            qa_by_id[item.get("question_id", "")] = item

    eval_items = eval_data if isinstance(eval_data, list) else eval_data.get("evaluations", [])

    records: List[PassRecord] = []
    for item in eval_items:
        crit = item.get("criterion_scores", {}) or {}
        correctness = float(crit.get("correctness", item.get("overall_score", 0.0)) or 0.0)

        qa_item = qa_by_id.get(item.get("question_id", ""), {})

        record = PassRecord(
            agent=pair.agent,
            design=pair.design,
            model=pair.model,
            pass_idx=pair.pass_idx,
            total_passes=pair.total_passes,
            timestamp=pair.timestamp,
            question_id=item.get("question_id", ""),
            question_idx=qa_item.get("question_idx"),
            question_type=item.get("question_type", qa_item.get("question_type", "")),
            stage=item.get("stage", qa_item.get("stage", "")),
            difficulty=item.get("difficulty", qa_item.get("difficulty", "")),
            answer_format=item.get("answer_format", qa_item.get("answer_format", "")),
            overall_score=float(item.get("overall_score", 0.0) or 0.0),
            correctness=correctness,
            clarity=float(crit.get("clarity", 0.0) or 0.0),
            relevance=float(crit.get("relevance", 0.0) or 0.0),
            outcome=_classify_outcome(correctness),
            weaknesses=item.get("weaknesses", []) or [],
            strengths=item.get("strengths", []) or [],
            feedback=item.get("feedback", "") or "",
            evaluation_method=item.get("evaluation_method", ""),
            golden_answer=qa_item.get("golden_answer"),
            response_excerpt=_excerpt(qa_item.get("response", ""), response_excerpt_chars),
            qa_file=str(pair.qa_path),
            eval_file=str(pair.eval_path),
        )

        if record.outcome != "correct":
            if tagger is not None:
                record.tags = tagger(record)
            else:
                record.tags = tag_failure(
                    record.weaknesses,
                    record.feedback,
                    record.response_excerpt,
                    record.golden_answer,
                )
        records.append(record)

    return records


# ---------------------------------------------------------------------------
# Aggregations
# ---------------------------------------------------------------------------


def _persistence_label(passes: int, n_correct: int, n_partial: int) -> str:
    if passes == 0:
        return "no_data"
    if n_correct == passes:
        return "always_correct"
    if n_correct == 0 and n_partial == 0:
        return "persistent_failure"
    if n_correct == 0:
        return "persistent_partial"
    return "stochastic_failure"


def per_question_summary(records: List[PassRecord]) -> List[Dict[str, Any]]:
    """One row per ``(agent, design, model, question_id)`` aggregated across passes."""

    grouped: Dict[Tuple[str, str, str, str], List[PassRecord]] = defaultdict(list)
    for r in records:
        grouped[(r.agent, r.design, r.model, r.question_id)].append(r)

    rows: List[Dict[str, Any]] = []
    for (agent, design, model, qid), group in grouped.items():
        n = len(group)
        n_correct = sum(1 for r in group if r.outcome == "correct")
        n_partial = sum(1 for r in group if r.outcome == "partial")
        n_incorrect = sum(1 for r in group if r.outcome == "incorrect")
        tag_counter: Counter = Counter()
        for r in group:
            tag_counter.update(r.tags)
        dominant_tag = tag_counter.most_common(1)[0][0] if tag_counter else ""

        rep = group[0]
        rows.append(
            {
                "agent": agent,
                "design": design,
                "model": model,
                "question_id": qid,
                "question_type": rep.question_type,
                "stage": rep.stage,
                "difficulty": rep.difficulty,
                "passes_evaluated": n,
                "passes_correct": n_correct,
                "passes_partial": n_partial,
                "passes_incorrect": n_incorrect,
                "pass_at_1_rate": n_correct / n if n else 0.0,
                "avg_correctness": sum(r.correctness for r in group) / n if n else 0.0,
                "avg_overall_score": sum(r.overall_score for r in group) / n if n else 0.0,
                "persistence": _persistence_label(n, n_correct, n_partial),
                "tag_counts": dict(tag_counter),
                "dominant_tag": dominant_tag,
            }
        )
    return rows


def aggregate_by(
    records: List[PassRecord],
    keys: Iterable[str],
) -> List[Dict[str, Any]]:
    """Group records by a tuple of attribute names and report counts/scores/tags."""

    keys = list(keys)
    grouped: Dict[Tuple[Any, ...], List[PassRecord]] = defaultdict(list)
    for r in records:
        grouped[tuple(getattr(r, k) for k in keys)].append(r)

    out: List[Dict[str, Any]] = []
    for key_vals, group in grouped.items():
        n = len(group)
        n_correct = sum(1 for r in group if r.outcome == "correct")
        n_partial = sum(1 for r in group if r.outcome == "partial")
        n_incorrect = sum(1 for r in group if r.outcome == "incorrect")
        tag_counter: Counter = Counter()
        for r in group:
            if r.outcome != "correct":
                tag_counter.update(r.tags)
        row: Dict[str, Any] = dict(zip(keys, key_vals))
        row.update(
            {
                "n_records": n,
                "n_correct": n_correct,
                "n_partial": n_partial,
                "n_incorrect": n_incorrect,
                "pass_at_1_rate": n_correct / n if n else 0.0,
                "avg_correctness": sum(r.correctness for r in group) / n if n else 0.0,
                "avg_overall_score": sum(r.overall_score for r in group) / n if n else 0.0,
                "tag_counts": dict(tag_counter),
            }
        )
        out.append(row)
    out.sort(key=lambda r: (-r["n_incorrect"], -r["n_partial"]))
    return out


def cross_agent_comparison(
    per_question_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """For each ``(design, question_id)`` show pass@1 rate per agent side-by-side."""

    grouped: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in per_question_rows:
        grouped[(row["design"], row["question_id"])][row["agent"]] = row

    rows: List[Dict[str, Any]] = []
    for (design, qid), agent_map in grouped.items():
        flat: Dict[str, Any] = {
            "design": design,
            "question_id": qid,
            "question_type": next(iter(agent_map.values())).get("question_type", ""),
            "stage": next(iter(agent_map.values())).get("stage", ""),
            "difficulty": next(iter(agent_map.values())).get("difficulty", ""),
        }
        rates: List[float] = []
        for agent, row in agent_map.items():
            flat[f"{agent}__pass_at_1"] = row["pass_at_1_rate"]
            flat[f"{agent}__dominant_tag"] = row["dominant_tag"]
            rates.append(row["pass_at_1_rate"])
        flat["agents_evaluated"] = len(agent_map)
        flat["min_pass_at_1"] = min(rates) if rates else 0.0
        flat["max_pass_at_1"] = max(rates) if rates else 0.0
        flat["spread"] = (max(rates) - min(rates)) if rates else 0.0
        if rates and max(rates) >= 0.5 and min(rates) <= 0.0:
            flat["cross_agent_label"] = "agent_specific_failure"
        elif rates and min(rates) == 0.0 and max(rates) == 0.0:
            flat["cross_agent_label"] = "universal_failure"
        elif rates and min(rates) >= 0.999:
            flat["cross_agent_label"] = "universal_correct"
        else:
            flat["cross_agent_label"] = "mixed"
        rows.append(flat)
    rows.sort(key=lambda r: (-r["spread"], r["min_pass_at_1"]))
    return rows


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    field_set: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                field_set.append(k)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=field_set)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_value(row.get(k)) for k in field_set})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def render_markdown_report(
    pass_records: List[PassRecord],
    per_question_rows: List[Dict[str, Any]],
    cross_agent_rows: List[Dict[str, Any]],
) -> str:
    lines: List[str] = []
    lines.append("# Failure Analysis Report\n")
    lines.append(f"Total pass records: **{len(pass_records)}**\n")

    outcome_counts = Counter(r.outcome for r in pass_records)
    lines.append("## Outcome distribution")
    for k in ("correct", "partial", "incorrect"):
        n = outcome_counts.get(k, 0)
        pct = (n / len(pass_records) * 100) if pass_records else 0
        lines.append(f"- **{k}**: {n} ({pct:.1f}%)")

    lines.append("\n## Failure tags (across all agents)")
    tag_counts: Counter = Counter()
    for r in pass_records:
        if r.outcome != "correct":
            tag_counts.update(r.tags)
    total_failures = sum(1 for r in pass_records if r.outcome != "correct")
    for tag, n in tag_counts.most_common():
        pct = (n / total_failures * 100) if total_failures else 0
        lines.append(f"- `{tag}`: {n} ({pct:.1f}% of {total_failures} failures) - {FAILURE_TAGS.get(tag, '')}")

    lines.append("\n## Pass@1 by agent")
    by_agent = aggregate_by(pass_records, ["agent"])
    for row in sorted(by_agent, key=lambda r: -r["pass_at_1_rate"]):
        lines.append(
            f"- **{row['agent']}**: pass@1 = {row['pass_at_1_rate']:.2%} "
            f"(n={row['n_records']}, correctness={row['avg_correctness']:.2f})"
        )

    lines.append("\n## Pass@1 by question type")
    for row in aggregate_by(pass_records, ["question_type"]):
        lines.append(
            f"- **{row['question_type']}**: pass@1 = {row['pass_at_1_rate']:.2%} (n={row['n_records']})"
        )

    lines.append("\n## Top failing question types per agent")
    by_agent_qtype = aggregate_by(pass_records, ["agent", "question_type"])
    for row in by_agent_qtype[:15]:
        top_tags = sorted(row["tag_counts"].items(), key=lambda kv: -kv[1])[:3]
        tag_str = ", ".join(f"`{t}` ({c})" for t, c in top_tags)
        lines.append(
            f"- **{row['agent']} / {row['question_type']}**: incorrect={row['n_incorrect']} | top tags: {tag_str}"
        )

    persistent = [r for r in per_question_rows if r["persistence"] == "persistent_failure"]
    stochastic = [r for r in per_question_rows if r["persistence"] == "stochastic_failure"]
    lines.append(f"\n## Persistence (per question, across passes)")
    lines.append(f"- persistent_failure: {len(persistent)}")
    lines.append(f"- stochastic_failure: {len(stochastic)}")
    lines.append(
        f"- always_correct: {sum(1 for r in per_question_rows if r['persistence'] == 'always_correct')}"
    )

    lines.append("\n### Top 10 persistent failures")
    for row in persistent[:10]:
        lines.append(
            f"- `{row['question_id']}` | {row['agent']} / {row['design']} "
            f"| qtype={row['question_type']} stage={row['stage']} | dominant=`{row['dominant_tag']}`"
        )

    if cross_agent_rows:
        lines.append("\n## Cross-agent comparison (top spread)")
        for row in cross_agent_rows[:15]:
            agents = sorted(k.replace("__pass_at_1", "") for k in row.keys() if k.endswith("__pass_at_1"))
            agent_strs = [f"{a}={row[a + '__pass_at_1']:.0%}" for a in agents]
            lines.append(
                f"- `{row['question_id']}` ({row['design']}, {row['question_type']}) -> "
                f"{row['cross_agent_label']} | spread={row['spread']:.0%} | "
                + ", ".join(agent_strs)
            )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    root: Path,
    output_dir: Path,
    agents: Optional[List[str]] = None,
    designs: Optional[List[str]] = None,
    tagger: Optional[Callable[[PassRecord], List[str]]] = None,
    response_excerpt_chars: int = 600,
) -> Dict[str, Path]:
    pairs = discover_pass_pairs(root, agents=agents, designs=designs)
    if not pairs:
        raise SystemExit(f"No (qa, evaluation) file pairs found under {root}")

    print(f"Discovered {len(pairs)} pass-file pairs across "
          f"{len(set((p.agent, p.design) for p in pairs))} agent/design folders")

    all_records: List[PassRecord] = []
    for pair in pairs:
        try:
            recs = build_pass_records(
                pair, tagger=tagger, response_excerpt_chars=response_excerpt_chars
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  ! failed to process {pair.qa_path.name}: {exc}")
            continue
        all_records.extend(recs)
        print(
            f"  + {pair.agent}/{pair.design} pass {pair.pass_idx}/{pair.total_passes}: "
            f"{len(recs)} records"
        )

    print(f"Built {len(all_records)} total pass records")

    per_q = per_question_summary(all_records)
    cross_a = cross_agent_comparison(per_q)

    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "pass_records_jsonl": output_dir / "pass_records.jsonl",
        "failure_records_jsonl": output_dir / "failure_records.jsonl",
        "per_question_csv": output_dir / "per_question_summary.csv",
        "by_agent_csv": output_dir / "by_agent.csv",
        "by_agent_qtype_csv": output_dir / "by_agent_question_type.csv",
        "by_agent_stage_csv": output_dir / "by_agent_stage.csv",
        "by_tag_csv": output_dir / "by_tag.csv",
        "cross_agent_csv": output_dir / "cross_agent_comparison.csv",
        "report_md": output_dir / "summary_report.md",
    }

    write_jsonl(paths["pass_records_jsonl"], (asdict(r) for r in all_records))
    write_jsonl(
        paths["failure_records_jsonl"],
        (asdict(r) for r in all_records if r.outcome != "correct"),
    )
    write_csv(paths["per_question_csv"], per_q)
    write_csv(paths["by_agent_csv"], aggregate_by(all_records, ["agent"]))
    write_csv(paths["by_agent_qtype_csv"], aggregate_by(all_records, ["agent", "question_type"]))
    write_csv(paths["by_agent_stage_csv"], aggregate_by(all_records, ["agent", "stage"]))

    tag_rows: List[Dict[str, Any]] = []
    for tag in FAILURE_TAGS:
        per_agent: Dict[str, int] = defaultdict(int)
        total = 0
        for r in all_records:
            if r.outcome == "correct":
                continue
            if tag in r.tags:
                per_agent[r.agent] += 1
                total += 1
        row: Dict[str, Any] = {"tag": tag, "description": FAILURE_TAGS[tag], "total": total}
        row.update(per_agent)
        tag_rows.append(row)
    tag_rows.sort(key=lambda r: -r["total"])
    write_csv(paths["by_tag_csv"], tag_rows)

    write_csv(paths["cross_agent_csv"], cross_a)
    paths["report_md"].write_text(
        render_markdown_report(all_records, per_q, cross_a), encoding="utf-8"
    )

    print("\nWrote outputs to:")
    for k, v in paths.items():
        print(f"  - {k}: {v}")
    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Workspace root that contains qa_output_<agent>_<design> folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("failure_analysis_output"),
        help="Directory to write reports into (created if missing).",
    )
    parser.add_argument(
        "--agents",
        nargs="*",
        default=None,
        help="Filter to a subset of agent names (e.g. --agents claude_code cursor rag rlm).",
    )
    parser.add_argument(
        "--designs",
        nargs="*",
        default=None,
        help="Filter to specific designs (e.g. --designs aes_block_asap7 ariane133_nanagte45).",
    )
    parser.add_argument(
        "--llm-tagger",
        action="store_true",
        help="If set, attempt to use the optional LLM tagger via src.llm.llm_inference.",
    )
    parser.add_argument(
        "--response-excerpt-chars",
        type=int,
        default=600,
        help="Characters of the model response to keep in each record (default 600).",
    )
    return parser.parse_args()


def _build_llm_tagger() -> Callable[[PassRecord], List[str]]:
    try:
        from llm import llm_inference  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Cannot import src.llm.llm_inference for --llm-tagger: {exc}")

    def _call(prompt: str) -> str:
        return llm_inference(system_prompt="Return only valid JSON.", user_prompt=prompt)

    def _tag(record: PassRecord) -> List[str]:
        return tag_failure_with_llm(record, _call)

    return _tag


def main() -> None:
    args = _parse_args()
    tagger = _build_llm_tagger() if args.llm_tagger else None
    run_pipeline(
        root=args.root.resolve(),
        output_dir=args.output_dir.resolve(),
        agents=args.agents,
        designs=args.designs,
        tagger=tagger,
        response_excerpt_chars=args.response_excerpt_chars,
    )


if __name__ == "__main__":
    main()
