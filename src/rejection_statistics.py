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
Rejection statistics from QA raw JSON files.

Reads *_raw.json files, extracts rejection_statistics.low_score_details,
and categorizes each rejection by the score table used in qa_generation.py.
Entries with null or empty {} from the code are excluded from score-based
categories and labeled as "Code execution failed".
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from filter_invalid_answers import is_invalid_answer

# success count table
SUCCESS_COUNT_TABLE = {
    "aes_asap7": 80,
    "aes_nangate45": 75,
    "aes_sky130hd": 73,
    "aes_sky130hs": 72,
    "gcdsky130": 60,
    "riscv32i-mock-sram_asap7": 83,
    "uartgf180": 81,
    "aes-block_asap7": 102,
    "ariane133_nanagte45": 76,
    "ariane136_nangate45": 73,
    "bp_multi_top_nangate45": 57,
    "ibex_sky130hd": 77,
    "aes_asap7_vs_aes_sky130hd": 43,
    "aes_asap7_vs_aes_sky130hs": 47,
    "aes_asap7_vs_aes_nangate45": 30,
    "aes_nangate45_vs_aes_sky130hd": 38,
    "aes_nangate45_vs_aes_sky130hs": 35,
    "aes_sky130hs_vs_aes_sky130hd": 36
}

# Canonical order for design_tech in CSV and printed tables (single-file first, then multifile comparisons)
DESIGN_TECH_TABLE_ORDER: List[str] = [
    "aes_asap7",
    "aes-block_asap7",
    "aes_nangate45",
    "aes_sky130hd",
    "aes_sky130hs",
    "ariane133_nangate45",
    "ariane136_nangate45",
    "bp_multi_top_nangate45",
    "gcd_sky130",
    "ibex_sky130hd",
    "riscv32i-mock-sram_asap7",
    "uart_gf180",
    "aes_asap7_vs_aes_nangate45",
    "aes_asap7_vs_aes_sky130hd",
    "aes_asap7_vs_aes_sky130hs",
    "aes_nangate45_vs_aes_sky130hd",
    "aes_nangate45_vs_aes_sky130hs",
    "aes_sky130hs_vs_aes_sky130hd",
]

# Map data keys to canonical name for ordering (so variants get the same position)
DESIGN_TECH_ORDER_ALIASES: Dict[str, str] = {
    "ariane133_nanagte45": "ariane133_nangate45",
    "gcdsky130": "gcd_sky130",
    "uartgf180": "uart_gf180",
}


def _design_tech_order_index(design_tech: str) -> int:
    """Return sort index for design_tech (lower = earlier). Not in list -> end."""
    canonical = DESIGN_TECH_ORDER_ALIASES.get(design_tech, design_tech)
    if canonical in DESIGN_TECH_TABLE_ORDER:
        return DESIGN_TECH_TABLE_ORDER.index(canonical)
    return len(DESIGN_TECH_TABLE_ORDER)


def _design_tech_key(file_path: str) -> Optional[str]:
    """
    Derive design_tech key from file path for SUCCESS_COUNT_TABLE lookup.
    Uses parent directory name; normalizes aes_block_* -> aes-block_* to match table.
    """
    parent = Path(file_path).parent.name
    if not parent or parent == ".":
        return None
    # SUCCESS_COUNT_TABLE uses "aes-block_asap7", paths use "aes_block_asap7"
    if parent.startswith("aes_block_"):
        key = "aes-block_" + parent[len("aes_block_"):]
    else:
        key = parent
    if key in SUCCESS_COUNT_TABLE:
        return key
    if parent in SUCCESS_COUNT_TABLE:
        return parent
    return None


# Score table from qa_generation.py (lines 387-393) – score band -> category label
SCORE_TABLE = {
    1.0: "Same information (format differences OK)",
    0.9: "Semantically equivalent with minor format differences",
    0.8: "Match on key metrics but minor discrepancies in detail",
    0.7: "Related but one has more/less detail",
    0.5: "Same topic but different focus",
    0.3: "Partially related but miss key information",
    0.0: "Completely different topics or metrics",
}

# Table order for rejection categories summary: code_execution_failed, then 0.0 through 0.9, then 1.0
REJECTION_CATEGORIES_TABLE_ORDER: List[Tuple[Optional[str], float, str]] = [
    ("code_execution_failed", float("nan"), "Code execution failed"),
    (None, 0.0, "Completely different topics or metrics"),
    (None, 0.3, "Partially related but miss key information"),
    (None, 0.5, "Same topic but different focus"),
    (None, 0.7, "Related but one has more/less detail"),
    (None, 0.8, "Match on key metrics but minor discrepancies in detail"),
    (None, 0.9, "Semantically equivalent with minor format differences"),
    (None, 1.0, "Same information (format differences OK)"),
]

# Column keys for per-design_tech table (score / category)
REJECTION_TABLE_COLUMNS: List[str] = [
    "code_execution_failed", "0.0", "0.3", "0.5", "0.7", "0.8", "0.9", "1.0"
]

# Map category label (from by_category) to table column key
def _category_label_to_column(label: str) -> str:
    if label == "Code execution failed":
        return "code_execution_failed"
    for key, score_val, cat_label in REJECTION_CATEGORIES_TABLE_ORDER:
        if key == "code_execution_failed":
            continue
        if cat_label == label:
            return str(score_val)
    return "other"

# Ordered score thresholds for banding: (min_inclusive, max_exclusive) -> score key
# Scores in [1.0, inf) -> 1.0; [0.9, 1.0) -> 0.9; ...; [0, 0.3) -> 0.0
SCORE_BANDS: List[Tuple[float, float, float]] = [
    (1.0, float("inf"), 1.0),
    (0.9, 1.0, 0.9),
    (0.8, 0.9, 0.8),
    (0.7, 0.8, 0.7),
    (0.5, 0.7, 0.5),
    (0.3, 0.5, 0.3),
    (0.0, 0.3, 0.0),
]

# Score bands for common mismatch extraction (0.0, 0.3, 0.5, 0.7)
MISMATCH_SCORE_BANDS: Tuple[float, ...] = (0.0, 0.3, 0.5, 0.7)

# Failed categories for representative example extraction: code_execution_failed + 0.0, 0.3, 0.5, 0.7
FAILED_CATEGORY_LABELS: List[str] = [
    "Code execution failed",
    SCORE_TABLE[0.0],   # "Completely different topics or metrics"
    SCORE_TABLE[0.3],   # "Partially related but miss key information"
    SCORE_TABLE[0.5],   # "Same topic but different focus"
    SCORE_TABLE[0.7],   # "Related but one has more/less detail"
]


def _text_for_diversity(entry: Dict[str, Any]) -> str:
    """Build a single string from direct_answer and code_answer for diversity comparison."""
    da = (entry.get("direct_answer") or "").strip()
    ca = entry.get("code_answer")
    if ca is None:
        ca_s = ""
    elif isinstance(ca, (dict, list)):
        ca_s = json.dumps(ca, sort_keys=True)[:500]
    else:
        ca_s = str(ca).strip()[:500]
    return (da + " " + ca_s).strip() or "(empty)"


def _jaccard_distance_text(a: str, b: str) -> float:
    """Jaccard distance (1 - similarity) on word sets. Returns 0.0 (same) to 1.0 (disjoint)."""
    if not a and not b:
        return 0.0
    words_a = set(w.lower() for w in a.split() if w.strip())
    words_b = set(w.lower() for w in b.split() if w.strip())
    if not words_a and not words_b:
        return 0.0
    inter = len(words_a & words_b)
    union = len(words_a | words_b)
    if union == 0:
        return 0.0
    return 1.0 - (inter / union)


def _farthest_point_sampling(
    items: List[Dict[str, Any]],
    distance_fn: Any,
    k: int,
) -> List[Dict[str, Any]]:
    """
    Greedy farthest-point sampling: pick k items that are most different from each other.
    distance_fn(text_a: str, text_b: str) should return distance (higher = more different).
    Does not mutate items; uses _text_for_diversity for each item.
    """
    if not items or k <= 0:
        return []
    if len(items) <= k:
        return list(items)
    texts = [_text_for_diversity(it) for it in items]

    def dist(i: int, j: int) -> float:
        return distance_fn(texts[i], texts[j])

    selected: List[int] = [0]
    for _ in range(k - 1):
        best_idx = -1
        best_min_dist = -1.0
        for j in range(len(items)):
            if j in selected:
                continue
            min_d = min(dist(j, s) for s in selected)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_idx = j
        if best_idx < 0:
            break
        selected.append(best_idx)
    return [items[i] for i in selected]


def _extract_mismatch_reason(direct_answer: str, code_answer: Any, score: float) -> str:
    """Extract a short, normalized reason for mismatch from direct_answer/code_answer (for grouping)."""
    if not (direct_answer or str(code_answer or "").strip()):
        return "No direct answer provided"
    da = (direct_answer or "").lower()
    if "does not match" in da or "not match" in da:
        return "Code answer does not match expected answer"
    if "does not address" in da or "not address" in da:
        return "Code answer does not address the question"
    if "cannot be validated" in da or "cannot validate" in da:
        return "Answer cannot be validated against JSON context"
    if "data not available" in da or "not available" in da:
        return "Required data not available in JSON context"
    if "not explicitly" in da or "not provided" in da:
        return "Data not explicitly provided in JSON"
    if code_answer is None or str(code_answer).strip().lower() in ("", "null", "none", "{}"):
        return "Code returned null/empty"
    if "incorrect" in da or "does not correctly" in da:
        return "Code-extracted answer incorrect vs JSON context"
    if "wirelength" in da and ("not available" in da or "missing" in da):
        return "Wirelength/data not available for comparison"
    if "memory" in da and ("not available" in da or "missing" in da):
        return "Memory/runtime data not available"
    if "timing" in da and ("not available" in da or "n/a" in da):
        return "Timing data not available or N/A"
    if "different" in da and ("topic" in da or "metric" in da):
        return "Different topic or metric"
    if "partial" in da or "partially" in da:
        return "Partial match or partially related"
    if "more/less detail" in da or "more or less detail" in da:
        return "Related but more/less detail"
    if "different focus" in da:
        return "Same topic but different focus"
    # First sentence or first 120 chars
    for sep in (". ", "\n"):
        if sep in direct_answer:
            first = direct_answer.split(sep)[0].strip()
            if len(first) > 20:
                return first[:120] + ("..." if len(first) > 120 else "")
    return (direct_answer[:120] + "...") if len(direct_answer) > 120 else (direct_answer or "Low score mismatch")


def is_code_execution_failed(code_answer: Any) -> bool:
    """
    Return True if code_answer indicates code execution failed (null/empty).

    Treated as failed: None, null, empty string, "{}", "null", empty dict.
    """
    if code_answer is None:
        return True
    if is_invalid_answer(code_answer): # None, empty, or n/a values in any dict or list 
        return True
    return False


def score_to_category(score: float) -> str:
    """Map numeric score to category label from the score table."""
    for low, high, key in SCORE_BANDS:
        if low <= score < high:
            return SCORE_TABLE[key]
    return SCORE_TABLE[0.0]


def _get_rejection_blocks(data: Dict[str, Any]) -> List[Tuple[Dict[str, Any], Optional[str]]]:
    """Return list of (rejection_statistics dict, section name or None). Unified helper for both formats."""
    out: List[Tuple[Dict[str, Any], Optional[str]]] = []
    if "rejection_statistics" in data:
        out.append((data["rejection_statistics"], None))
    if "stage_comparison" in data and isinstance(data["stage_comparison"], dict):
        sc = data["stage_comparison"]
        if "rejection_statistics" in sc:
            out.append((sc["rejection_statistics"], "stage_comparison"))
    if "flow_execution" in data and isinstance(data["flow_execution"], dict):
        fe = data["flow_execution"]
        if "rejection_statistics" in fe:
            out.append((fe["rejection_statistics"], "flow_execution"))
    return out


def _get_single_file_rejection_blocks(data: Dict[str, Any]) -> List[Tuple[Dict[str, Any], Optional[str]]]:
    """Return rejection_statistics blocks for single-file raw JSON (top-level only)."""
    if "rejection_statistics" not in data:
        return []
    return [(data["rejection_statistics"], None)]


def _get_multifile_rejection_blocks(data: Dict[str, Any]) -> List[Tuple[Dict[str, Any], Optional[str]]]:
    """Return rejection_statistics blocks for multifile raw JSON (single_stage_comparison, multi_stage_comparison, flow_execution)."""
    out: List[Tuple[Dict[str, Any], Optional[str]]] = []
    for section_key in ("single_stage_comparison", "multi_stage_comparison", "flow_execution"):
        if section_key in data and isinstance(data[section_key], dict):
            section = data[section_key]
            if "rejection_statistics" in section:
                out.append((section["rejection_statistics"], section_key))
    return out


def get_multifile_low_score_details(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Load low_score_details from a multifile raw JSON dict.
    Multifile structure: rejection_statistics live under single_stage_comparison,
    multi_stage_comparison, and flow_execution.
    Returns a list of detail dicts, each with original keys plus:
      - file_path, file_name, section ('single_stage_comparison', 'multi_stage_comparison', or 'flow_execution').
    """
    file_path = data.get("_file_path", "unknown")
    file_name = data.get("_file_name", "")
    out: List[Dict[str, Any]] = []
    for section_key in ("single_stage_comparison", "multi_stage_comparison", "flow_execution"):
        if section_key in data and isinstance(data[section_key], dict):
            section = data[section_key]
            rej = section.get("rejection_statistics") or {}
            for d in rej.get("low_score_details", []):
                out.append({**d, "file_path": file_path, "file_name": file_name, "section": section_key})
    return out


def get_single_file_low_score_details(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Load low_score_details from a single-file raw JSON dict.
    Single-file structure: rejection_statistics at top level.
    Returns a list of detail dicts, each with original keys plus:
      - file_path, file_name, section (None).
    """
    file_path = data.get("_file_path", "unknown")
    file_name = data.get("_file_name", "")
    out: List[Dict[str, Any]] = []
    if "rejection_statistics" not in data:
        return out
    rej = data["rejection_statistics"]
    for d in rej.get("low_score_details", []):
        out.append({**d, "file_path": file_path, "file_name": file_name, "section": None})
    return out


def load_raw_json_files(qa_output_dir: str = "qa_output") -> List[Dict[str, Any]]:
    """
    Load all *_raw.json files from qa_output_dir (recursive).
    Each item has keys from JSON plus "_file_path" and "_file_name".
    """
    base = Path(qa_output_dir)
    if not base.is_dir():
        return []
    files = sorted(base.rglob("*_raw.json"))
    result = []
    for f in files:
        try:
            with open(f, "r") as fp:
                data = json.load(fp)
            data["_file_path"] = str(f.relative_to(base))
            data["_file_name"] = f.name
            result.append(data)
        except Exception as e:
            # skip broken files; could log
            pass
    return result


def _is_single_file_raw(data: Dict[str, Any]) -> bool:
    """True if data is single-file raw format (top-level rejection_statistics, no multifile sections)."""
    return (
        "rejection_statistics" in data
        and "single_stage_comparison" not in data
        and "multi_stage_comparison" not in data
    )


def _is_multifile_raw(data: Dict[str, Any]) -> bool:
    """True if data is multifile raw format (has single_stage_comparison and/or multi_stage_comparison with nested rejection_statistics)."""
    return (
        ("single_stage_comparison" in data and isinstance(data.get("single_stage_comparison"), dict))
        or ("multi_stage_comparison" in data and isinstance(data.get("multi_stage_comparison"), dict))
    )


def load_single_file_raw_json_files(qa_output_dir: str = "qa_output") -> List[Dict[str, Any]]:
    """
    Load only single-file *_raw.json files (rejection_statistics at top level).
    Excludes multifile raw JSONs (those with stage_comparison).
    """
    all_data = load_raw_json_files(qa_output_dir)
    return [d for d in all_data if _is_single_file_raw(d)]


def load_multifile_raw_json_files(qa_output_dir: str = "qa_output") -> List[Dict[str, Any]]:
    """
    Load only multifile *_raw.json files (rejection_statistics under stage_comparison / flow_execution).
    """
    all_data = load_raw_json_files(qa_output_dir)
    return [d for d in all_data if _is_multifile_raw(d)]


def categorize_rejection_statistics(
    qa_output_dir: str = "qa_output",
) -> Dict[str, Any]:
    """
    Read *_raw.json from qa_output_dir and categorize rejection statistics.
    Single-file and multifile raw JSONs are loaded separately; multifile uses
    nested rejection_statistics under stage_comparison and flow_execution.

    - Entries with null or empty {} code_answer are labeled "Code execution failed"
      and excluded from score-based categories.
    - All other low_score_details are categorized by the score table (bands 1.0, 0.9, 0.8, 0.7, 0.5, 0.3, 0.0).

    Returns a dict with:
      - by_category: { category_label: count }
      - code_execution_failed_count: int
      - details_by_category: { category_label: [ { question, score, file, section?, ... } ] }
      - total_low_score_details: int
      - files_loaded: int
    """
    single_files = load_single_file_raw_json_files(qa_output_dir)
    multifile_files = load_multifile_raw_json_files(qa_output_dir)
    by_category = defaultdict(int)
    by_design_tech_category: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    details_by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    code_execution_failed_count = 0
    total_low_score_details = 0

    def process_details(data: Dict[str, Any], get_blocks: Any) -> None:
        nonlocal total_low_score_details, code_execution_failed_count
        file_path = data.get("_file_path", "unknown")
        file_name = data.get("_file_name", "")
        design_tech = _design_tech_key(file_path) or file_path
        for reject_stats, section in get_blocks(data):
            if not reject_stats:
                continue
            low_score_details = reject_stats.get("low_score_details", [])
            for detail in low_score_details:
                total_low_score_details += 1
                code_answer = detail.get("code_answer")
                score = detail.get("score", 0.0)
                question = detail.get("question", "")
                if is_code_execution_failed(code_answer):
                    code_execution_failed_count += 1
                    category = "Code execution failed"
                else:
                    category = score_to_category(score)
                by_category[category] += 1
                col = _category_label_to_column(category)
                by_design_tech_category[design_tech][col] += 1
                direct_answer = (detail.get("direct_answer") or "")[:500]  # keep for mismatch extraction
                code_ans = detail.get("code_answer")
                entry = {
                    "question": question[:200] + ("..." if len(question) > 200 else ""),
                    "score": score,
                    "file": file_path,
                    "file_name": file_name,
                    "direct_answer": direct_answer,
                    "code_answer": code_ans if not isinstance(code_ans, (dict, list)) or len(str(code_ans)) < 200 else str(code_ans)[:200],
                }
                if section:
                    entry["section"] = section
                details_by_category[category].append(entry)

    for data in single_files:
        process_details(data, _get_single_file_rejection_blocks)
    for data in multifile_files:
        process_details(data, _get_multifile_rejection_blocks)

    # Convert defaults to plain dicts and sort category list for stable output
    by_category = dict(by_category)
    details_by_category = dict(details_by_category)
    by_design_tech_category = {k: dict(v) for k, v in by_design_tech_category.items()}

    return {
        "by_category": by_category,
        "by_design_tech_category": by_design_tech_category,
        "code_execution_failed_count": code_execution_failed_count,
        "details_by_category": details_by_category,
        "total_low_score_details": total_low_score_details,
        "files_loaded": len(single_files) + len(multifile_files),
        "score_table": SCORE_TABLE,
    }


def rejection_stats_per_file(
    qa_output_dir: str = "qa_output",
) -> List[Dict[str, Any]]:
    """
    Count rejected QA one-by-one per file from low_score_details (not from
    rejection_statistics aggregates, which may be inaccurate).
    Single-file and multifile raw JSONs are loaded separately; multifile uses
    nested rejection_statistics under stage_comparison and flow_execution.

    For each *_raw.json file, iterates every rejection_statistics block and
    counts each low_score_details entry: total_rejected = len(low_score_details),
    code_execution_failed = count where code_answer is null/empty. Uses
    total_attempts from the JSON for rejection rate denominator.
    - total_rejected: counted from low_score_details (one by one)
    - code_execution_failed: counted via is_code_execution_failed(code_answer)
    - low_score_rejected: total_rejected - code_execution_failed
    - rejection_rate_pct = 100 * total_rejected / total_attempts

    Returns a list of dicts, one per file.
    """
    single_files = load_single_file_raw_json_files(qa_output_dir)
    multifile_files = load_multifile_raw_json_files(qa_output_dir)
    per_file: List[Dict[str, Any]] = []

    def process_file(data: Dict[str, Any], get_blocks: Any) -> None:
        file_path = data.get("_file_path", "unknown")
        file_name = data.get("_file_name", "")
        design_name = data.get("design_name") or data.get("designs")
        if isinstance(design_name, list):
            design_name = ", ".join(design_name) if design_name else None

        total_attempts = 0
        total_rejected = 0   # counted from low_score_details
        code_execution_failed = 0  # counted from low_score_details

        for reject_stats, _section in get_blocks(data):
            if not reject_stats:
                continue
            total_attempts += reject_stats.get("total_attempts", 0)
            low_score_details = reject_stats.get("low_score_details", [])
            for detail in low_score_details:
                total_rejected += 1
                if is_code_execution_failed(detail.get("code_answer")):
                    code_execution_failed += 1

        low_score_rejected = total_rejected - code_execution_failed

        # Use SUCCESS_COUNT_TABLE per design_tech when available
        design_tech = _design_tech_key(file_path)
        if design_tech is None:
            print(f"Design tech not found for file: {file_path}")
            print(f"Error: exiting...")
            exit(1)
        if design_tech is not None and design_tech in SUCCESS_COUNT_TABLE:
            successful = SUCCESS_COUNT_TABLE[design_tech]
            total_attempts = successful + total_rejected
        else:
            successful = total_attempts - total_rejected if total_attempts >= total_rejected else 0

        rejection_rate_pct = (
            (100.0 * total_rejected / total_attempts) if total_attempts else 0.0
        )

        per_file.append({
            "file_path": file_path,
            "file_name": file_name,
            "design_name": design_name,
            "total_attempts": total_attempts,
            "successful": successful,
            "total_rejected": total_rejected,
            "rejection_rate_pct": round(rejection_rate_pct, 2),
            "code_execution_failed": code_execution_failed,
            "low_score_rejected": low_score_rejected,
            "invalid_answer_rejected": 0,  # not distinguished in low_score_details
        })

    for data in single_files:
        process_file(data, _get_single_file_rejection_blocks)
    for data in multifile_files:
        process_file(data, _get_multifile_rejection_blocks)
    return per_file


def rejection_totals(per_file: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate per-file rejection stats into overall totals.
    Returns dict with total_attempts, total_rejected, successful, code_execution_failed,
    low_score_rejected, rejection_rate_pct, file_count.
    """
    if not per_file:
        return {
            "file_count": 0,
            "total_attempts": 0,
            "total_rejected": 0,
            "successful": 0,
            "code_execution_failed": 0,
            "low_score_rejected": 0,
            "rejection_rate_pct": 0.0,
        }
    total_attempts = sum(r["total_attempts"] for r in per_file)
    total_rejected = sum(r["total_rejected"] for r in per_file)
    code_execution_failed = sum(r["code_execution_failed"] for r in per_file)
    low_score_rejected = sum(r["low_score_rejected"] for r in per_file)
    successful = total_attempts - total_rejected if total_attempts >= total_rejected else 0
    rejection_rate_pct = (
        round(100.0 * total_rejected / total_attempts, 2) if total_attempts else 0.0
    )
    return {
        "file_count": len(per_file),
        "total_attempts": total_attempts,
        "total_rejected": total_rejected,
        "successful": successful,
        "code_execution_failed": code_execution_failed,
        "low_score_rejected": low_score_rejected,
        "rejection_rate_pct": rejection_rate_pct,
    }


def print_rejection_per_file(per_file: List[Dict[str, Any]]) -> None:
    """Print a per-file table of rejection counts and rejection rate."""
    if not per_file:
        print("No files with rejection statistics.")
        return
    print("\n" + "=" * 90)
    print("REJECTION STATISTICS PER FILE (rejected QA count and rejection rate)")
    print("=" * 90)
    # Header
    fmt = "{:<45} {:>10} {:>10} {:>10} {:>10} {:>8}"
    print(fmt.format("File", "Attempts", "Rejected", "Success", "CodeFail", "Rej%"))
    print("-" * 90)
    for row in per_file:
        fp = (row["file_path"] or row["file_name"])[:44]
        print(fmt.format(
            fp,
            row["total_attempts"],
            row["total_rejected"],
            row["successful"],
            row["code_execution_failed"],
            row["rejection_rate_pct"],
        ))
    print("-" * 90)
    tot = rejection_totals(per_file)
    print(fmt.format(
        "TOTAL",
        tot["total_attempts"],
        tot["total_rejected"],
        tot["successful"],
        tot["code_execution_failed"],
        tot["rejection_rate_pct"],
    ))
    print("=" * 90)


def rejection_categories_table(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build a summary table of rejection categories: code_execution_failed, then score bands 0.0 to 0.9, 1.0.
    Returns a list of dicts with keys: score (str or float), category (str), count (int), percentage (float).
    """
    by_cat = result.get("by_category") or {}
    total = result.get("total_low_score_details") or 0
    rows: List[Dict[str, Any]] = []
    for key, score_val, label in REJECTION_CATEGORIES_TABLE_ORDER:
        count = by_cat.get(label, 0)
        pct = (100.0 * count / total) if total else 0.0
        score_display = "code_execution_failed" if key == "code_execution_failed" else score_val
        rows.append({
            "score": score_display,
            "category": label,
            "count": count,
            "percentage": round(pct, 1),
        })
    return rows


def print_rejection_categories_table(result: Dict[str, Any]) -> None:
    """Print a table summarizing rejection statistics by category (code_execution_failed, 0.0 … 0.9, 1.0)."""
    table = rejection_categories_table(result)
    total = result.get("total_low_score_details") or 0
    print("\n" + "=" * 95)
    print("REJECTION CATEGORIES SUMMARY (code_execution_failed | score 0.0 … 0.9 | 1.0)")
    print("=" * 95)
    fmt = "{:<8} {:>8} {:>8}   {:<55}"
    print(fmt.format("Score", "Count", "Pct %", "Category"))
    print("-" * 95)
    for row in table:
        score_s = str(row["score"]) if row["score"] != "code_execution_failed" else "code_fail"
        print(fmt.format(score_s, row["count"], row["percentage"], row["category"][:55]))
    print("-" * 95)
    print(fmt.format("TOTAL", total, "100.0", "(all rejected / low_score_details)"))
    print("=" * 95)


def write_rejection_categories_csv(result: Dict[str, Any], filepath: str) -> None:
    """Write the global rejection categories table to a CSV file. Includes long format and a wide row with columns code_execution_failed, 0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0."""
    table = rejection_categories_table(result)
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["score", "category", "count", "percentage"])
        for row in table:
            w.writerow([row["score"], row["category"], row["count"], row["percentage"]])
        total = result.get("total_low_score_details") or 0
        w.writerow(["TOTAL", "(all rejected / low_score_details)", total, 100.0])
        # Wide format: columns code_execution_failed, 0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0
        w.writerow([])
        w.writerow(["metric"] + REJECTION_TABLE_COLUMNS)
        by_cat = result.get("by_category") or {}
        count_by_col = {}
        for label, count in by_cat.items():
            col = _category_label_to_column(label)
            count_by_col[col] = count_by_col.get(col, 0) + count
        w.writerow(["count"] + [str(count_by_col.get(c, 0)) for c in REJECTION_TABLE_COLUMNS])


def extract_common_mismatches(
    result: Dict[str, Any],
    score_bands: Tuple[float, ...] = MISMATCH_SCORE_BANDS,
    top_k: int = 15,
) -> Dict[str, List[Tuple[str, int]]]:
    """
    Extract commonly occurring mismatch reasons for low score bands 0.0, 0.3, 0.5, 0.7.
    Returns a dict: score_band_str -> [(reason, count), ...] sorted by count descending (top_k per band).
    """
    details_by_category = result.get("details_by_category") or {}
    out: Dict[str, List[Tuple[str, int]]] = {}
    for score in score_bands:
        label = SCORE_TABLE.get(score)
        if not label or label not in details_by_category:
            out[str(score)] = []
            continue
        details = details_by_category[label]
        reason_counts: Dict[str, int] = defaultdict(int)
        for d in details:
            reason = _extract_mismatch_reason(
                d.get("direct_answer") or "",
                d.get("code_answer"),
                d.get("score", 0.0),
            )
            reason_counts[reason] += 1
        sorted_reasons = sorted(reason_counts.items(), key=lambda x: -x[1])[:top_k]
        out[str(score)] = sorted_reasons
    return out


def extract_representative_examples(
    result: Dict[str, Any],
    top_k: int = 5,
    method: str = "reason_stratified",
    failed_categories: Optional[List[str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract representative (direct_answer, code_answer) examples for each failed category,
    so that the chosen examples are as diverse as possible (top-k most different).

    Failed categories: Code execution failed, 0.0, 0.3, 0.5, 0.7 (see FAILED_CATEGORY_LABELS).

    Methods:
      - "reason_stratified": Group by mismatch reason (from _extract_mismatch_reason);
        take one or more examples per reason so that different failure modes are covered.
        Orders reasons by count and fills up to top_k. Best for covering all mismatch types.
      - "text_diversity": Use Jaccard distance on (direct_answer + code_answer) text;
        greedy farthest-point sampling to pick top_k most different answers. No extra deps.
      - "embedding_diversity": Embed (direct_answer + code_answer) and use cosine distance;
        farthest-point sampling in embedding space. Requires sentence-transformers.
      - "hybrid": Stratify by reason first; within each reason bucket pick examples by
        text_diversity (up to ceil(top_k / num_reasons) per bucket). Balances coverage and diversity.

    Returns:
      { category_label: [ { "question", "direct_answer", "code_answer", "score", "file", "file_name", "reason"?, "section"? }, ... ] }
    """
    details_by_category = result.get("details_by_category") or {}
    categories = failed_categories if failed_categories is not None else FAILED_CATEGORY_LABELS
    out: Dict[str, List[Dict[str, Any]]] = {}

    for label in categories:
        details = details_by_category.get(label, [])
        if not details:
            out[label] = []
            continue
        k = min(top_k, len(details))

        if method == "reason_stratified":
            # Group by reason; sort reasons by count desc; take from each until we have k
            by_reason: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for d in details:
                reason = _extract_mismatch_reason(
                    d.get("direct_answer") or "",
                    d.get("code_answer"),
                    d.get("score", 0.0),
                )
                by_reason[reason].append(d)
            sorted_reasons = sorted(by_reason.keys(), key=lambda r: -len(by_reason[r]))
            chosen: List[Dict[str, Any]] = []
            per_reason = max(1, k // len(sorted_reasons)) if sorted_reasons else 0
            for reason in sorted_reasons:
                if len(chosen) >= k:
                    break
                bucket = by_reason[reason]
                take = min(per_reason, k - len(chosen), len(bucket))
                for i in range(take):
                    e = dict(bucket[i])
                    e["reason"] = reason
                    chosen.append(e)
            out[label] = chosen[:k]

        elif method == "text_diversity":
            chosen = _farthest_point_sampling(details, _jaccard_distance_text, k)
            out[label] = chosen

        elif method == "embedding_diversity":
            try:
                from embedding_utils import get_embedding_model
                model = get_embedding_model(model_type="sentence-transformer")
                texts = [_text_for_diversity(d) for d in details]
                embs = model.encode(texts)
                import numpy as np
                n = len(embs)
                if n <= k:
                    out[label] = list(details)
                    continue
                # Cosine distance = 1 - cos_sim; normalize
                norms = np.linalg.norm(embs, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                embs_n = embs / norms

                def cos_dist(i: int, j: int) -> float:
                    return float(1.0 - np.dot(embs_n[i], embs_n[j]))

                selected_idxs = [0]
                for _ in range(k - 1):
                    best_idx = -1
                    best_min = -1.0
                    for j in range(n):
                        if j in selected_idxs:
                            continue
                        min_d = min(cos_dist(j, s) for s in selected_idxs)
                        if min_d > best_min:
                            best_min = min_d
                            best_idx = j
                    if best_idx < 0:
                        break
                    selected_idxs.append(best_idx)
                out[label] = [details[i] for i in selected_idxs]
            except Exception:
                # Fallback to text_diversity if embedding unavailable
                chosen = _farthest_point_sampling(details, _jaccard_distance_text, k)
                out[label] = chosen

        elif method == "hybrid":
            by_reason = defaultdict(list)
            for d in details:
                reason = _extract_mismatch_reason(
                    d.get("direct_answer") or "",
                    d.get("code_answer"),
                    d.get("score", 0.0),
                )
                by_reason[reason].append(d)
            sorted_reasons = sorted(by_reason.keys(), key=lambda r: -len(by_reason[r]))
            per_bucket = max(1, k // len(sorted_reasons)) if sorted_reasons else 0
            chosen = []
            for reason in sorted_reasons:
                if len(chosen) >= k:
                    break
                bucket = by_reason[reason]
                sub_k = min(per_bucket, k - len(chosen), len(bucket))
                sub_chosen = _farthest_point_sampling(bucket, _jaccard_distance_text, sub_k)
                for e in sub_chosen:
                    ex = dict(e)
                    ex["reason"] = reason
                    chosen.append(ex)
            out[label] = chosen[:k]

        else:
            # Default: take first k
            out[label] = [dict(d) for d in details[:k]]

    return out


def print_common_mismatches(
    result: Dict[str, Any],
    score_bands: Tuple[float, ...] = MISMATCH_SCORE_BANDS,
    top_k: int = 15,
) -> None:
    """Print common mismatch reasons for score bands 0.0, 0.3, 0.5, 0.7."""
    common = extract_common_mismatches(result, score_bands=score_bands, top_k=top_k)
    print("\n" + "=" * 90)
    print("COMMON MISMATCHES (0.0, 0.3, 0.5, 0.7)")
    print("=" * 90)
    for score in score_bands:
        key = str(score)
        rows = common.get(key, [])
        label = SCORE_TABLE.get(score, key)
        print(f"\n--- Score {score} ({label}) ---")
        if not rows:
            print("  (none)")
            continue
        for i, (reason, count) in enumerate(rows, 1):
            print(f"  {i:2}. [{count:4}] {reason[:75]}{'...' if len(reason) > 75 else ''}")
    print("=" * 90)


def write_common_mismatches_csv(
    result: Dict[str, Any],
    filepath: str,
    score_bands: Tuple[float, ...] = MISMATCH_SCORE_BANDS,
    top_k: int = 15,
) -> None:
    """Write common mismatches for 0.0, 0.3, 0.5, 0.7 to a CSV file. Columns: score_band, rank, reason, count."""
    common = extract_common_mismatches(result, score_bands=score_bands, top_k=top_k)
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["score_band", "rank", "reason", "count"])
        for score in score_bands:
            key = str(score)
            for rank, (reason, count) in enumerate(common.get(key, []), 1):
                w.writerow([score, rank, reason, count])


def print_representative_examples(
    result: Dict[str, Any],
    top_k: int = 5,
    method: str = "reason_stratified",
) -> None:
    """Print representative direct_answer / code_answer examples per failed category (code_execution_failed, 0.0, 0.3, 0.5, 0.7)."""
    examples = extract_representative_examples(result, top_k=top_k, method=method)
    print("\n" + "=" * 90)
    print(f"REPRESENTATIVE EXAMPLES (top_k={top_k}, method={method})")
    print("=" * 90)
    for label in FAILED_CATEGORY_LABELS:
        entries = examples.get(label, [])
        print(f"\n--- {label} ({len(entries)} examples) ---")
        if not entries:
            print("  (none)")
            continue
        for i, ex in enumerate(entries, 1):
            q = (ex.get("question") or "")[:80]
            da = (ex.get("direct_answer") or "")[:120]
            ca = ex.get("code_answer")
            ca_s = str(ca)[:80] if ca is not None else "(null)"
            print(f"  [{i}] Q: {q}...")
            print(f"      direct_answer: {da}...")
            print(f"      code_answer: {ca_s}...")
            if ex.get("reason"):
                print(f"      reason: {ex['reason'][:60]}...")
    print("=" * 90)


def write_representative_examples_json(
    result: Dict[str, Any],
    filepath: str,
    top_k: int = 5,
    method: str = "reason_stratified",
) -> None:
    """Write representative examples for each failed category to a JSON file."""
    examples = extract_representative_examples(result, top_k=top_k, method=method)
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(examples, f, indent=2)


def rejection_categories_table_by_design_tech(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build a table of rejection counts by design_tech and score category.
    Rows = design_tech, columns = code_execution_failed, 0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0, total.
    Returns list of dicts, one per design_tech, ordered by DESIGN_TECH_TABLE_ORDER.
    """
    by_dt = result.get("by_design_tech_category") or {}
    rows: List[Dict[str, Any]] = []
    # Sort by canonical design_tech order, then by name for any not in list
    ordered_keys = sorted(by_dt.keys(), key=lambda dt: (_design_tech_order_index(dt), dt))
    for design_tech in ordered_keys:
        counts = by_dt[design_tech]
        row: Dict[str, Any] = {"design_tech": design_tech}
        for col in REJECTION_TABLE_COLUMNS:
            row[col] = counts.get(col, 0)
        row["total"] = sum(counts.values())  # include any "other" or extra keys
        rows.append(row)
    return rows


def print_rejection_categories_table_by_design_tech(result: Dict[str, Any]) -> None:
    """Print a big table: row = design_tech, columns = Score (code_fail, 0.0, 0.3, … 0.9, 1.0), Count, Total."""
    table = rejection_categories_table_by_design_tech(result)
    if not table:
        print("\nNo per-design_tech category data.")
        return
    # Column widths: design_tech 32, then 6 per score column, 6 for total
    w_dt = 32
    w_num = 6
    n_score_cols = len(REJECTION_TABLE_COLUMNS)
    total_width = w_dt + (n_score_cols + 1) * (w_num + 1)
    # Header: design_tech then each score column (short label), then total
    score_headers = ["fail" if c == "code_execution_failed" else c for c in REJECTION_TABLE_COLUMNS]
    print("\n" + "=" * total_width)
    print("REJECTION CATEGORIES BY DESIGN_TECH (rows = design_tech, columns = Score → Count)")
    print("=" * total_width)
    header = f"{{:<{w_dt}}}".format("design_tech")
    for h in score_headers:
        header += f" {{:>{w_num}}}".format(h[:w_num])
    header += f" {{:>{w_num}}}".format("total")
    print(header)
    print("-" * total_width)
    for row in table:
        line = f"{{:<{w_dt}}}".format((row["design_tech"] or "")[:w_dt])
        for col in REJECTION_TABLE_COLUMNS:
            line += f" {{:>{w_num}}}".format(row.get(col, 0))
        line += f" {{:>{w_num}}}".format(row.get("total", 0))
        print(line)
    print("-" * total_width)
    # Totals row
    sum_row: Dict[str, int] = {col: sum(r.get(col, 0) for r in table) for col in REJECTION_TABLE_COLUMNS}
    sum_row["total"] = sum(r.get("total", 0) for r in table)
    line = f"{{:<{w_dt}}}".format("TOTAL")
    for col in REJECTION_TABLE_COLUMNS:
        line += f" {{:>{w_num}}}".format(sum_row.get(col, 0))
    line += f" {{:>{w_num}}}".format(sum_row.get("total", 0))
    print(line)
    print("=" * total_width)


def write_rejection_categories_by_design_tech_csv(result: Dict[str, Any], filepath: str) -> None:
    """Write the per-design_tech rejection categories table to a CSV file. Header: design_tech, code_execution_failed, 0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0, total."""
    table = rejection_categories_table_by_design_tech(result)
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Explicit columns: design_tech, code_execution_failed, 0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0, total
    cols = ["design_tech"] + list(REJECTION_TABLE_COLUMNS) + ["total"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for row in table:
            w.writerow([row.get(c, 0) for c in cols])
    # Append TOTAL row
    if table:
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            sum_row = ["TOTAL"] + [sum(r.get(c, 0) for r in table) for c in cols[1:]]
            w.writerow(sum_row)


def write_rejection_per_file_csv(per_file: List[Dict[str, Any]], filepath: str) -> None:
    """Write the per-file rejection stats table to a CSV file. Rows ordered by DESIGN_TECH_TABLE_ORDER."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file_path", "file_name", "design_name", "total_attempts", "successful",
        "total_rejected", "rejection_rate_pct", "code_execution_failed", "low_score_rejected",
        "invalid_answer_rejected",
    ]
    # Sort by design_tech order (from file_path), then by file_path
    ordered = sorted(
        per_file,
        key=lambda r: (_design_tech_order_index(_design_tech_key(r.get("file_path") or "") or ""), r.get("file_path", "")),
    )
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(ordered)
    if per_file:
        tot = rejection_totals(per_file)
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "TOTAL", "", "", tot["total_attempts"], tot["successful"], tot["total_rejected"],
                tot["rejection_rate_pct"], tot["code_execution_failed"], tot["low_score_rejected"],
                0,
            ])


def print_rejection_summary(result: Dict[str, Any]) -> None:
    """Print a human-readable summary of categorized rejection statistics."""
    print("=" * 70)
    print("REJECTION STATISTICS (by score table)")
    print("=" * 70)
    print(f"Files loaded: {result['files_loaded']}")
    print(f"Total low_score_details: {result['total_low_score_details']}")
    print(f"Code execution failed (null/empty {{}}): {result['code_execution_failed_count']}")
    print()
    print("By category (score table + code execution failed):")
    print("-" * 70)
    by_cat = result["by_category"]
    # Order: Code execution failed first, then by score descending
    order = ["Code execution failed"] + [SCORE_TABLE[k] for k in [1.0, 0.9, 0.8, 0.7, 0.5, 0.3, 0.0]]
    for cat in order:
        if cat not in by_cat:
            continue
        count = by_cat[cat]
        pct = (100.0 * count / result["total_low_score_details"]) if result["total_low_score_details"] else 0
        print(f"  {cat}")
        print(f"    Count: {count} ({pct:.1f}%)")
    # Any category not in predefined list (e.g. from future score bands)
    for cat in sorted(by_cat.keys()):
        if cat not in order:
            count = by_cat[cat]
            pct = (100.0 * count / result["total_low_score_details"]) if result["total_low_score_details"] else 0
            print(f"  {cat}")
            print(f"    Count: {count} ({pct:.1f}%)")
    print("=" * 70)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Categorize rejection statistics from QA raw JSON files")
    parser.add_argument("--qa-output-dir", type=str, default="qa_output", help="Directory containing *_raw.json")
    parser.add_argument("--export", type=str, default=None, help="Optional JSON output file")
    parser.add_argument("--csv-dir", type=str, default=None, metavar="DIR", help="Write CSV tables to this directory (rejection_categories.csv, rejection_categories_by_design_tech.csv, rejection_per_file.csv)")
    parser.add_argument("--no-print", action="store_true", help="Do not print summary to stdout")
    parser.add_argument("--per-file", action="store_true", help="Print rejection count and rejection rate per file")
    parser.add_argument("--representative-examples", action="store_true", help="Print representative direct/code answer examples per failed category")
    parser.add_argument("--representative-method", type=str, default="reason_stratified", choices=["reason_stratified", "text_diversity", "embedding_diversity", "hybrid"], help="Method for picking representative examples (default: reason_stratified)")
    parser.add_argument("--representative-top-k", type=int, default=5, metavar="K", help="Number of representative examples per category (default: 5)")
    args = parser.parse_args()

    result = categorize_rejection_statistics(qa_output_dir=args.qa_output_dir)
    per_file = rejection_stats_per_file(qa_output_dir=args.qa_output_dir)

    if not args.no_print:
        if args.per_file:
            print_rejection_per_file(per_file)
        print_rejection_summary(result)
        print_rejection_categories_table(result)
        print_rejection_categories_table_by_design_tech(result)
        print_common_mismatches(result)
        if args.representative_examples:
            print_representative_examples(result, top_k=args.representative_top_k, method=args.representative_method)

    if args.export:
        out_path = Path(args.export)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate details for export if very long (keep counts and short samples)
        categories_table = rejection_categories_table(result)
        categories_table_by_design_tech = rejection_categories_table_by_design_tech(result)
        export_result = {
            "by_category": result["by_category"],
            "by_design_tech_category": result.get("by_design_tech_category", {}),
            "code_execution_failed_count": result["code_execution_failed_count"],
            "total_low_score_details": result["total_low_score_details"],
            "files_loaded": result["files_loaded"],
            "score_table": result["score_table"],
            "rejection_categories_table": categories_table,
            "rejection_categories_table_by_design_tech": categories_table_by_design_tech,
            "common_mismatches": {k: v for k, v in extract_common_mismatches(result).items()},
            "representative_examples": extract_representative_examples(result, top_k=5, method="reason_stratified"),
            "details_by_category": {
                k: v[:50] for k, v in result["details_by_category"].items()
            },
            "per_file": per_file,
            "total_rejection": rejection_totals(per_file),
        }
        with open(out_path, "w") as f:
            json.dump(export_result, f, indent=2)
        print(f"Exported to {out_path}")
        # When exporting JSON, also write CSVs to same directory unless --csv-dir is set
        if not args.csv_dir:
            csv_dir = out_path.parent
            stem = out_path.stem
            write_rejection_categories_csv(result, str(csv_dir / f"{stem}_categories.csv"))
            write_rejection_categories_by_design_tech_csv(result, str(csv_dir / f"{stem}_by_design_tech.csv"))
            write_rejection_per_file_csv(per_file, str(csv_dir / f"{stem}_per_file.csv"))
            write_common_mismatches_csv(result, str(csv_dir / f"{stem}_common_mismatches.csv"))
            write_representative_examples_json(result, str(csv_dir / f"{stem}_representative_examples.json"), top_k=5, method="reason_stratified")
            print(f"CSV tables written to {csv_dir}")

    if args.csv_dir:
        csv_dir = Path(args.csv_dir)
        csv_dir.mkdir(parents=True, exist_ok=True)
        write_rejection_categories_csv(result, str(csv_dir / "rejection_categories.csv"))
        write_rejection_categories_by_design_tech_csv(result, str(csv_dir / "rejection_categories_by_design_tech.csv"))
        write_rejection_per_file_csv(per_file, str(csv_dir / "rejection_per_file.csv"))
        write_common_mismatches_csv(result, str(csv_dir / "common_mismatches.csv"))
        print(f"CSV tables written to {csv_dir}")


if __name__ == "__main__":
    main()
