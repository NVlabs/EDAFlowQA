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
Generate stage_combination_summary.csv from design_tech_qa_list.json,
excluding QA pairs listed in pass_k_ignored_qa_pairs.json.

Reads:
  - qa_stats_results/design_tech_qa_list.json (all questions per design_tech)
  - pass_k_ignored_qa_pairs.json (filtered_qa_pairs per design_tech to exclude)

Alternative: build_stage_combination_summary_from_combined_used() reads
  - results/combined_used_qids_by_category.json (same structure, no exclusion list)

Writes:
  - CSV with columns: scope, stage_combination, question_count
  - Same format as qa_stats_results/stage_combination_summary.csv
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# Canonical stage order (must match qa_stats_summary.STAGE_ORDER)
STAGE_ORDER = ["synthesis", "floorplan", "placement", "cts", "routing", "final"]


def _sort_stages_canonical(stages_list: List[str]) -> List[str]:
    """Sort stage names by canonical order."""
    order = {s: i for i, s in enumerate(STAGE_ORDER)}
    return sorted(stages_list, key=lambda s: (order.get((s or "").lower().strip(), len(STAGE_ORDER)), s))


def _stages_with_occurrence_labels(stages_list: List[str]) -> str:
    """
    Build stage-combination label with occurrence numbers when the same stage
    appears more than once (e.g. ['routing','routing','routing'] -> 'routing_1,routing_2,routing_3').
    """
    if not stages_list:
        return ""
    counts = Counter(stages_list)
    unique_ordered = _sort_stages_canonical(list(counts.keys()))
    parts = []
    for s in unique_ordered:
        for occ in range(1, counts[s] + 1):
            if counts[s] > 1:
                parts.append(f"{s}_{occ}")
            else:
                parts.append(s)
    return ",".join(parts)


def _stage_combination_from_entry(entry: Dict[str, Any], is_multifile: bool) -> str:
    """
    Derive stage_combination string from a design_tech_qa_list entry.
    Entry has: qid, stages (list), category (e.g. "Flow Execution", "routing", "2-Stages").
    """
    category = (entry.get("category") or "").strip()
    stages = entry.get("stages")
    if not isinstance(stages, list):
        stages = [stages] if stages is not None else []

    if category == "Flow Execution" and not stages:
        return "multi_file_flow_execution" if is_multifile else "flow_execution"

    if len(stages) == 0:
        return "multi_file_flow_execution" if is_multifile else "flow_execution"

    if len(stages) == 1:
        label = str(stages[0]).strip()
        return f"single_stage_comparison:{label}" if is_multifile else f"single_stage:{label}"

    ordered = _sort_stages_canonical(stages)
    labels = _stages_with_occurrence_labels(ordered)
    return f"multi_stage_comparison:{labels}" if is_multifile else f"multi_stage:{labels}"


def load_ignored_qa_pairs(ignored_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load pass_k_ignored_qa_pairs.json and return design_tech -> list of excluded QA pairs
    (each pair has 'id' and optionally 'question'). Used for listing excluded QAs.
    """
    path = Path(ignored_path)
    if not path.exists():
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    filtered = data.get("filtered_qa_pairs") or {}
    return {
        dt: [p for p in pairs if isinstance(p, dict) and p.get("id")]
        for dt, pairs in filtered.items()
        if isinstance(pairs, list)
    }


def load_ignored_ids(ignored_path: str) -> Dict[str, Set[str]]:
    """
    Load pass_k_ignored_qa_pairs.json and return design_tech -> set of question IDs to exclude.
    """
    pairs_by_dt = load_ignored_qa_pairs(ignored_path)
    return {dt: {p["id"] for p in pairs} for dt, pairs in pairs_by_dt.items()}


def load_design_tech_qa_list(qa_list_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """Load design_tech -> list of {qid, stages, category} from design_tech_qa_list.json."""
    with open(qa_list_path, "r") as f:
        return json.load(f)


def build_stage_combination_counts(
    qa_list: Dict[str, List[Dict[str, Any]]],
    ignored_by_design_tech: Dict[str, Set[str]],
) -> List[Tuple[str, str, int]]:
    """
    Iterate all (design_tech, question) from qa_list, exclude ignored, and count by (scope, stage_combination).
    Returns list of (scope, stage_combination, question_count) for CSV.
    """
    # Count (scope, stage_combination) -> count. Scope = "single_file" | "multi_file"
    counts: Dict[Tuple[str, str], int] = defaultdict(int)

    for design_tech, entries in qa_list.items():
        if not isinstance(entries, list):
            continue
        is_multifile = "_vs_" in design_tech
        scope = "multi_file" if is_multifile else "single_file"
        ignored = ignored_by_design_tech.get(design_tech, set())

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            qid = entry.get("qid")
            if qid in ignored:
                continue
            stage_comb = _stage_combination_from_entry(entry, is_multifile)
            counts[(scope, stage_comb)] += 1

    # Convert to list of rows, sorted by question_count descending
    rows = [(scope, stage_comb, n) for (scope, stage_comb), n in counts.items()]
    rows.sort(key=lambda x: -x[2])
    return rows


def write_csv(rows: List[Tuple[str, str, int]], out_path: str) -> None:
    """Write CSV with columns scope, stage_combination, question_count."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scope", "stage_combination", "question_count"])
        w.writerows(rows)


def build_stage_combination_summary_from_combined_used(
    combined_path: str = "results/combined_used_qids_by_category.json",
    output_path: str = "qa_stats_results/stage_combination_summary.csv",
) -> List[Tuple[str, str, int]]:
    """
    Read results/combined_used_qids_by_category.json and generate statistics
    in the same format as qa_stats_results/stage_combination_summary.csv
    (scope, stage_combination, question_count).

    The combined file has the same structure as design_tech_qa_list.json
    (design_tech -> list of {qid, stages, category}); no exclusion list
    is applied since the file already contains only "used" QIDs.

    Returns:
        List of (scope, stage_combination, question_count) rows.
    """
    path = Path(combined_path)
    if not path.exists():
        raise FileNotFoundError(f"Combined used QIDs file not found: {path}")
    qa_list = load_design_tech_qa_list(str(path))
    ignored: Dict[str, Set[str]] = {}  # no exclusions
    rows = build_stage_combination_counts(qa_list, ignored)
    if rows:
        write_csv(rows, output_path)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate stage_combination_summary.csv from design_tech_qa_list.json, excluding pass_k ignored QA pairs."
    )
    parser.add_argument(
        "--from-combined",
        action="store_true",
        help="Read results/combined_used_qids_by_category.json and generate CSV (no ignored list).",
    )
    parser.add_argument(
        "--combined",
        default="results/combined_used_qids_by_category.json",
        help="Path to combined_used_qids_by_category.json (used with --from-combined)",
    )
    parser.add_argument(
        "--ignored",
        default="pass_k_ignored_qa_pairs.json",
        help="Path to pass_k_ignored_qa_pairs.json (default: pass_k_ignored_qa_pairs.json)",
    )
    parser.add_argument(
        "--qa-list",
        default="qa_stats_results/design_tech_qa_list.json",
        help="Path to design_tech_qa_list.json (default: qa_stats_results/design_tech_qa_list.json)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="qa_stats_results/stage_combination_summary.csv",
        help="Output CSV path (default: qa_stats_results/stage_combination_summary.csv)",
    )
    args = parser.parse_args()

    if args.from_combined:
        try:
            rows = build_stage_combination_summary_from_combined_used(
                combined_path=args.combined,
                output_path=args.output,
            )
        except FileNotFoundError as e:
            raise SystemExit(e) from e
        total = sum(r[2] for r in rows)
        print(f"Total QAs (from combined used): {total}")
        print(f"Wrote {args.output} with {len(rows)} scope/combination rows, total question_count={total}.")
        return

    qa_list_path = Path(args.qa_list)
    if not qa_list_path.exists():
        raise SystemExit(f"QA list not found: {qa_list_path}")

    qa_list = load_design_tech_qa_list(str(qa_list_path))
    ignored_pairs_by_dt = load_ignored_qa_pairs(args.ignored)
    ignored = {dt: {p["id"] for p in pairs} for dt, pairs in ignored_pairs_by_dt.items()}

    # Question count per design+technology (total, excluded, included)
    print("Question count by design+technology:")
    print("  design_tech | total | excluded | included")
    print("  " + "-" * 50)
    for design_tech in sorted(qa_list.keys()):
        entries = qa_list[design_tech]
        if not isinstance(entries, list):
            continue
        total = len(entries)
        excluded_ids = ignored.get(design_tech, set())
        excluded = sum(1 for e in entries if isinstance(e, dict) and e.get("qid") in excluded_ids)
        included = total - excluded
        print(f"  {design_tech} | {total} | {excluded} | {included}")
    print()

    # List excluded QAs per design+technology
    if ignored_pairs_by_dt:
        print("Excluded QAs by design+technology:")
        for design_tech in sorted(ignored_pairs_by_dt.keys()):
            pairs_list = ignored_pairs_by_dt[design_tech]
            qids = [p["id"] for p in pairs_list]
            print(f"  {design_tech} ({len(qids)}): {', '.join(sorted(qids))}")
        print()

    rows = build_stage_combination_counts(qa_list, ignored)
    if not rows:
        print("No questions to output (all excluded or empty QA list).")
        return

    write_csv(rows, args.output)
    total = sum(r[2] for r in rows)
    print(f"Total QAs: {total}")
    print(f"Wrote {args.output} with {len(rows)} scope/combination rows, total question_count={total}.")


if __name__ == "__main__":
    main()
