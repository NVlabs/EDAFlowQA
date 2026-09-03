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
Collect per-question token usage and cost from RLM pass_at_k JSON files.

Reads pass_at_k_*.json files under qa_output_rlm_* and qa_responses_rlm_*
directories. Each file is an array of question results; each item has
usage_summary.by_model with total_calls, total_input_tokens, total_output_tokens
per model. This module collects those values per question, per model, per file,
and per run directory. Token cost (USD) is computed using the same MODEL_PRICES
and compute_cost as in calculate_rag_token_cost (input/output per 1M tokens).

Usage:
  from src.calculate_token_rlm import (
      collect_tokens_rlm,
      collect_tokens_rlm_qa_output,
      collect_tokens_rlm_qa_responses,
      collect_tokens_from_pass_at_k_file,
  )

  # Collect from all RLM dirs (both types)
  result = collect_tokens_rlm()

  # Collect only from qa_output_rlm_* directories
  token_dict_qa_output = collect_tokens_rlm_qa_output()

  # Collect only from qa_responses_rlm_* directories
  token_dict_qa_responses = collect_tokens_rlm_qa_responses()

  # Collect from a single pass_at_k JSON
  rows = collect_tokens_from_pass_at_k_file("path/to/pass_at_k_....json")

  # Custom root directory
  result = collect_tokens_rlm_qa_output(root_dir="/path/to/workspace")
"""

import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pdb
# Allow "src" to be imported when run as script: python src/calculate_token_rlm.py
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.calculate_rag_token_cost import (
    MODEL_ALIASES,
    MODEL_PRICES,
    compute_cost,
)

# Default USD per 1M tokens when model is unknown (same fallback as RAG script)
_DEFAULT_INPUT_PER_1M = 1.0
_DEFAULT_OUTPUT_PER_1M = 4.0


def _resolve_model_key_safe(model_name: str) -> Optional[str]:
    """Return MODEL_PRICES key for a given model name, or None if unknown (no exit).
    Matches exact/alias, or when name is substring of key, or when key is substring of name
    (e.g. openai/gpt-oss-120b matches key gpt-oss). Longest key match wins."""
    if not model_name:
        return None
    name = model_name.strip().lower()
    if name in MODEL_ALIASES:
        name = MODEL_ALIASES[name]
    # Exact or name contained in key
    for key in MODEL_PRICES:
        if key.lower() == name or name in key.lower():
            return key
    # Key contained in name (e.g. gpt-oss in openai/gpt-oss-120b); try longest key first
    for key in sorted(MODEL_PRICES.keys(), key=lambda k: -len(k)):
        if key.lower() in name:
            return key
    print(f"Unknown model name: {model_name}")
    exit(1)
    return None


def get_prices_for_model_rlm(model_name: Optional[str]) -> Tuple[float, float]:
    """Return (input_per_1m, output_per_1m) for a model; use defaults if unknown."""
    key = _resolve_model_key_safe(model_name) if model_name else None
    if key and key in MODEL_PRICES:
        p = MODEL_PRICES[key]
        return p["input_per_1m"], p["output_per_1m"]
    return _DEFAULT_INPUT_PER_1M, _DEFAULT_OUTPUT_PER_1M


def collect_tokens_from_pass_at_k_file(
    filepath: str,
) -> List[Dict[str, Any]]:
    """
    Collect per-question token usage from a single pass_at_k JSON file.

    The file must be a JSON array of question objects. Each object should have
    question_id, question_idx (or similar), and usage_summary.by_model with
    total_calls, total_input_tokens, total_output_tokens per model.

    Returns:
        List of dicts, one per (question, model): each dict has keys
        question_id, question_idx, model, total_calls, total_input_tokens,
        total_output_tokens, cost_usd (USD from MODEL_PRICES in calculate_rag_token_cost).
    """
    path = Path(filepath)
    if not path.is_file():
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []

    if not isinstance(data, list):
        return []

    rows: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        usage = item.get("usage_summary") or {}
        by_model = usage.get("by_model")
        if not by_model or not isinstance(by_model, dict):
            continue
        question_id = item.get("question_id") or item.get("question_idx")
        question_idx = item.get("question_idx")
        for model_name, model_usage in by_model.items():
            if not isinstance(model_usage, dict):
                continue
            in_tok = int(model_usage.get("total_input_tokens") or 0)
            out_tok = int(model_usage.get("total_output_tokens") or 0)
            total_calls = int(model_usage.get("total_calls") or 0)
            if int(total_calls) > 5: # deal with max iterations of 5 issue
                in_tok = int(in_tok / total_calls) * 5
                out_tok = int(out_tok / total_calls) * 5
                # print("Warning: max iterations of 5 issue. Adjusting token usage to 5 iterations.")
                # print(f"in_tok: {in_tok}  out_tok: {out_tok}  total_calls: {total_calls}")
            in_p, out_p = get_prices_for_model_rlm(model_name)
            # print(f"in_p: {in_p}  out_p: {out_p}")
            cost_usd = compute_cost(in_tok, out_tok, in_p, out_p)
            rows.append({
                "question_id": question_id,
                "question_idx": question_idx,
                "model": model_name,
                "total_calls": model_usage.get("total_calls"),
                "total_input_tokens": in_tok,
                "total_output_tokens": out_tok,
                "cost_usd": round(cost_usd, 4),
            })
    return rows


def _is_pass_at_k_json(name: str) -> bool:
    """True if name looks like a pass_at_k result JSON (exclude evaluation/sum)."""
    if not name.endswith(".json"):
        return False
    if "_evaluation.json" in name or name.endswith("_evaluation.json"):
        return False
    return name.startswith("pass_at_k_")


def _is_pass_N_of_10_json(name: str) -> bool:
    """True if filename is pass_at_k_*_pass<number>of10_*.json (exclude evaluation)."""
    if not _is_pass_at_k_json(name):
        return False
    # Require _pass<digits>of10_ in the name (e.g. pass10of10, pass1of10)
    return re.search(r"_pass\d+of10_", name) is not None


def _parse_stem_timestamp(stem: str) -> Optional[Tuple[str, str, str]]:
    """
    If stem ends with _YYYYMMDD_HHMMSS return (run_key, date, time); else None.
    run_key is the stem without the trailing _YYYYMMDD_HHMMSS.
    """
    m = re.match(r"^(.+)_(\d{8})_(\d{6})$", stem)
    if m:
        return (m.group(1), m.group(2), m.group(3))
    return None


def _pass_at_k_files_in_dir(dirpath: Path, pass10_only: bool = True) -> List[Path]:
    """
    Return pass_at_k_*.json files in dir (excluding evaluation).
    If pass10_only is True (default), keep only *_pass<number>of10_*.json and,
    for each run key (design_model_passNof10), keep only the file with the latest
    timestamp (YYYYMMDD_HHMMSS) in the filename.
    """
    candidates: List[Path] = []
    for f in dirpath.iterdir():
        if not f.is_file():
            continue
        if pass10_only and not _is_pass_N_of_10_json(f.name):
            continue
        if not pass10_only and not _is_pass_at_k_json(f.name):
            continue
        candidates.append(f)
    if not pass10_only:
        return sorted(candidates)
    # Group by run_key (stem minus _YYYYMMDD_HHMMSS), keep latest timestamp per key
    by_key: Dict[str, Path] = {}
    for f in candidates:
        parsed = _parse_stem_timestamp(f.stem)
        if not parsed:
            continue
        run_key, date, time = parsed
        existing = by_key.get(run_key)
        if existing is None or (date, time) > (_parse_stem_timestamp(existing.stem) or ("", "", ""))[1:]:
            by_key[run_key] = f
    return sorted(by_key.values())


def _discover_rlm_dirs(root_dir: str) -> List[Path]:
    """Return sorted list of qa_output_rlm_* and qa_responses_rlm_* directories."""
    root = Path(root_dir)
    if not root.is_dir():
        return []
    out: List[Path] = []
    for p in root.iterdir():
        if p.is_dir() and p.name.startswith(("qa_output_rlm_", "qa_responses_rlm_")):
            out.append(p)
    return sorted(out)


def _discover_qa_output_rlm_dirs(root_dir: str) -> List[Path]:
    """Return sorted list of qa_output_rlm_* directories only."""
    root = Path(root_dir)
    if not root.is_dir():
        return []
    out: List[Path] = []
    for p in root.iterdir():
        if p.is_dir() and p.name.startswith("qa_output_rlm_"):
            out.append(p)
    return sorted(out)


def _discover_qa_responses_rlm_dirs(root_dir: str) -> List[Path]:
    """Return sorted list of qa_responses_rlm_* directories only."""
    root = Path(root_dir)
    if not root.is_dir():
        return []
    out: List[Path] = []
    for p in root.iterdir():
        if p.is_dir() and p.name.startswith("qa_responses_rlm_"):
            out.append(p)
    return sorted(out)


def _collect_tokens_rlm_from_dirs(
    rlm_dirs: List[Path],
    model_filter: Optional[set] = None,
) -> Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]]:
    """
    Build the nested token dict from a list of RLM directories.
    [dir_name][file_name][model] = list of per-question token dicts.
    If model_filter is set, only include models matching _model_matches_filter;
    directories with no matching files are omitted.
    """
    result: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]] = {}
    for rlm_dir in rlm_dirs:
        dir_name = rlm_dir.name
        kept_files: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        for json_path in _pass_at_k_files_in_dir(rlm_dir):
            file_name = json_path.name
            if model_filter and not _filename_might_contain_model(file_name, model_filter):
                continue
            rows = collect_tokens_from_pass_at_k_file(str(json_path))
            if not rows:
                continue
            by_model: Dict[str, List[Dict[str, Any]]] = {}
            for r in rows:
                model = r.get("model") or "unknown"
                by_model.setdefault(model, []).append(r)
            if model_filter:
                by_model = {
                    m: rows
                    for m, rows in by_model.items()
                    if _model_matches_filter(m, model_filter)
                }
            if by_model:
                kept_files[file_name] = by_model
        if kept_files:
            result[dir_name] = kept_files
    return result


def _filename_might_contain_model(file_name: str, model_filter: set) -> bool:
    """
    True if file_name (e.g. pass_at_k_*_gpt-5_*.json) might contain data for
    any model in model_filter. Used to skip reading files before collect_tokens_from_pass_at_k_file.
    """
    if not model_filter or not file_name:
        return True
    fn = file_name.lower()
    for f in model_filter:
        fl = f.lower()
        if fl in fn:
            return True
        if fl == "sonnet":
            if any(p in fn for p in ("sonnet4p5", "sonnet-4.5", "sonnet-4-5")):
                return True
    return False


# When user filters by "sonnet", match only Sonnet 4.5 for correct pricing.
# API names: sonnet4p5, sonnet-4.5, claude-sonnet-4-5-20250929, etc.
SONNET_4P5_PATTERNS = (
    "sonnet4p5",
    "sonnet-4.5",
    "sonnet4.5",
    "sonnet-4-5",  # e.g. claude-sonnet-4-5-20250929
    "claude-sonnet-4-5",
)
SONNET_3P5_INDICATORS = ("3-5", "3.5")  # exclude claude-3-5-sonnet


def _model_matches_filter(model_name: str, filter_set: set) -> bool:
    """
    True if model_name matches any filter in filter_set.
    Exact/prefix/substring (case-insensitive). Special case: "sonnet" matches
    only Sonnet 4.5 (sonnet4p5), not 3.5.
    """
    if not model_name or not filter_set:
        return False
    ml = model_name.lower()
    for f in filter_set:
        fl = f.lower()
        if fl == "sonnet":
            if any(x in ml for x in SONNET_3P5_INDICATORS):
                continue
            if any(x in ml or ml == x for x in SONNET_4P5_PATTERNS):
                return True
            continue
        if ml == fl or fl in ml or ml.startswith(fl + "-"):
            return True
        # Kimi2: API name can be moonshotai/kimi-k2-instruct-0905 (no literal "kimi2")
        if fl in ("kimi2", "kimi2_instruct") and "kimi" in ml and ("k2" in ml or "kimi2" in ml):
            return True
    return False


def filter_nested_by_models(
    nested: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]],
    model_set: Optional[set],
) -> Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]]:
    """
    Return a copy of the nested dict keeping only models that match model_set.
    Matching uses _model_matches_filter (substring/prefix; "sonnet" => Sonnet 4.5 only).
    Directories with no matching files are omitted from the result.
    """
    if not model_set:
        return nested
    out: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]] = {}
    for dir_name, files in nested.items():
        kept: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        for file_name, by_model in files.items():
            filtered = {
                m: rows
                for m, rows in by_model.items()
                if _model_matches_filter(m, model_set)
            }
            if filtered:
                kept[file_name] = filtered
        if kept:
            out[dir_name] = kept
    return out


def collect_tokens_rlm(
    root_dir: Optional[str] = None,
    model_filter: Optional[set] = None,
) -> Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]]:
    """
    Collect per-question token usage for each model from all RLM pass_at_k files.

    Scans for directories matching qa_output_rlm_* and qa_responses_rlm_* under
    root_dir (default: workspace root). In each directory, finds all
    pass_at_k_*.json files (excluding *_evaluation.json). For each file, reads
    the array and extracts usage_summary.by_model per question.
    If model_filter is set, only models matching the filter are included.

    Returns:
        Nested dict:
          [dir_name][file_name][model] = list of per-question token dicts.
        Each per-question dict has: question_id, question_idx, model,
        total_calls, total_input_tokens, total_output_tokens, cost_usd.
        dir_name is the directory basename (e.g. qa_output_rlm_bp_multi_top_nangate45).
    """
    if root_dir is None:
        root_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )
    return _collect_tokens_rlm_from_dirs(_discover_rlm_dirs(root_dir), model_filter=model_filter)


def collect_tokens_rlm_qa_output(
    root_dir: Optional[str] = None,
    model_filter: Optional[set] = None,
) -> Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]]:
    """
    Collect per-question token usage from qa_output_rlm_* directories only.

    Same structure as collect_tokens_rlm, but only includes directories whose
    name starts with qa_output_rlm_ (e.g. qa_output_rlm_bp_multi_top_nangate45).
    If model_filter is set, only models matching the filter are included.

    Returns:
        Nested dict: [dir_name][file_name][model] = list of per-question token
        dicts (question_id, question_idx, model, total_calls, total_input_tokens,
        total_output_tokens).
    """
    if root_dir is None:
        root_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )
    return _collect_tokens_rlm_from_dirs(_discover_qa_output_rlm_dirs(root_dir), model_filter=model_filter)


def collect_tokens_rlm_qa_responses(
    root_dir: Optional[str] = None,
    model_filter: Optional[set] = None,
) -> Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]]:
    """
    Collect per-question token usage from qa_responses_rlm_* directories only.

    Same structure as collect_tokens_rlm, but only includes directories whose
    name starts with qa_responses_rlm_ (e.g. qa_responses_rlm_aes_asap7_vs_aes_nangate45).
    If model_filter is set, only models matching the filter are included.

    Returns:
        Nested dict: [dir_name][file_name][model] = list of per-question token
        dicts (question_id, question_idx, model, total_calls, total_input_tokens,
        total_output_tokens).
    """
    if root_dir is None:
        root_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )
    return _collect_tokens_rlm_from_dirs(_discover_qa_responses_rlm_dirs(root_dir), model_filter=model_filter)


def collect_tokens_rlm_flat(
    root_dir: Optional[str] = None,
    model_filter: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """
    Same as collect_tokens_rlm but returns a flat list of rows with dir and file.

    Each row has: dir_name, file_name, question_id, question_idx, model,
    total_calls, total_input_tokens, total_output_tokens, cost_usd.
    If model_filter is set, only matching models are included.
    """
    nested = collect_tokens_rlm(root_dir=root_dir, model_filter=model_filter)
    flat: List[Dict[str, Any]] = []
    for dir_name, files in nested.items():
        for file_name, by_model in files.items():
            for model, rows in by_model.items():
                for r in rows:
                    flat.append({
                        "dir_name": dir_name,
                        "file_name": file_name,
                        "question_id": r.get("question_id"),
                        "question_idx": r.get("question_idx"),
                        "model": r.get("model"),
                        "total_calls": r.get("total_calls"),
                        "total_input_tokens": r.get("total_input_tokens"),
                        "total_output_tokens": r.get("total_output_tokens"),
                        "cost_usd": r.get("cost_usd"),
                    })
    return flat


def aggregate_totals_by_dir(
    nested: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]],
) -> Dict[str, Dict[str, Any]]:
    """
    Sum token counts and cost per directory (and per model within each directory).

    Returns:
        [dir_name] = {
            "total_input_tokens": int,
            "total_output_tokens": int,
            "total_tokens": int,
            "total_cost_usd": float,
            "num_questions": int,  # distinct question rows (can repeat across files/models)
            "by_model": { model: { total_input_tokens, total_output_tokens, total_cost_usd } }
        }
    """
    out: Dict[str, Dict[str, Any]] = {}
    for dir_name, files in nested.items():
        json_files = sorted(files.keys())
        num_json_files = len(json_files)
        by_model: Dict[str, Dict[str, Any]] = {}
        total_in = total_out = 0
        total_cost = 0.0
        seen_questions: set = set()
        for file_name, models in files.items():
            for model, rows in models.items():
                for r in rows:
                    qi = r.get("question_idx")
                    qid = r.get("question_id")
                    key = (qid, qi, file_name, model)
                    seen_questions.add(key)
                    total_in += int(r.get("total_input_tokens") or 0)
                    total_out += int(r.get("total_output_tokens") or 0)
                    total_cost += float(r.get("cost_usd") or 0)
                    by_model.setdefault(model, {
                        "total_input_tokens": 0,
                        "total_output_tokens": 0,
                        "total_cost_usd": 0.0,
                    })
                    by_model[model]["total_input_tokens"] += int(r.get("total_input_tokens") or 0)
                    by_model[model]["total_output_tokens"] += int(r.get("total_output_tokens") or 0)
                    by_model[model]["total_cost_usd"] += float(r.get("cost_usd") or 0)
        out[dir_name] = {
            "json_files": json_files,
            "num_json_files": num_json_files,
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "total_tokens": total_in + total_out,
            "total_cost_usd": round(total_cost, 4),
            "num_questions": len(seen_questions),
            "by_model": {m: {**v, "total_cost_usd": round(v["total_cost_usd"], 4)} for m, v in by_model.items()},
        }
    return out


def _all_models_sorted(totals: Dict[str, Dict[str, Any]]) -> List[str]:
    """Return sorted list of all model names that appear in totals_by_dir."""
    models: set = set()
    for t in totals.values():
        models.update((t.get("by_model") or {}).keys())
    return sorted(models)


def write_totals_csv(
    totals: Dict[str, Dict[str, Any]],
    out_path: Optional[str] = None,
) -> None:
    """
    Write a CSV with one row per directory; columns are directory, num_json_files,
    then for each model: {model}_input_tokens, {model}_output_tokens, {model}_cost_usd.
    """
    models = _all_models_sorted(totals)
    dirs_sorted = sorted(totals.keys())
    header = ["directory", "num_json_files"]
    for m in models:
        header.append(f"{m}_input_tokens")
        header.append(f"{m}_output_tokens")
        header.append(f"{m}_cost_usd")
    f = open(out_path, "w", newline="", encoding="utf-8") if out_path else sys.stdout
    try:
        writer = csv.writer(f)
        writer.writerow(header)
        for dir_name in dirs_sorted:
            t = totals[dir_name]
            by_model = t.get("by_model") or {}
            row: List[Any] = [
                dir_name,
                t.get("num_json_files", 0),
            ]
            for m in models:
                b = by_model.get(m, {})
                row.append(b.get("total_input_tokens", 0))
                row.append(b.get("total_output_tokens", 0))
                row.append(b.get("total_cost_usd", 0))
            writer.writerow(row)
        if out_path and f is not sys.stdout:
            f.flush()
    finally:
        if out_path and f is not sys.stdout:
            f.close()
            print(f"Wrote CSV table to {os.path.abspath(out_path)}", file=sys.stderr)


def format_dir_model_summary_table(dir_name: str, totals_for_dir: Dict[str, Any]) -> str:
    """
    Format a summary table for one directory: rows = models, columns = input_tokens,
    output_tokens, total_tokens, cost_usd. Returns a string with aligned columns.
    """
    by_model = totals_for_dir.get("by_model") or {}
    if not by_model:
        return f"  [{dir_name}] no model data\n"
    models = sorted(by_model.keys())
    rows: List[List[str]] = []
    for m in models:
        b = by_model[m]
        inp = b.get("total_input_tokens", 0)
        out = b.get("total_output_tokens", 0)
        total = inp + out
        cost = b.get("total_cost_usd", 0)
        rows.append([m, str(inp), str(out), str(total), str(cost)])
    # Column widths: header vs content
    header = ["model", "input_tokens", "output_tokens", "total_tokens", "cost_usd"]
    widths = [max(len(header[i]), max((len(rows[j][i]) for j in range(len(rows))), default=0)) for i in range(5)]
    pad = "  "
    lines = [pad + "  ".join(h.ljust(widths[i]) for i, h in enumerate(header))]
    lines.append(pad + "-" * (sum(widths) + 4))
    for row in rows:
        lines.append(pad + "  ".join(row[i].rjust(widths[i]) for i in range(5)))
    return "\n".join(lines) + "\n"


def write_summary_by_dir_model_csv(
    totals: Dict[str, Dict[str, Any]],
    out_path: Optional[str] = None,
) -> None:
    """
    Write a CSV summary table: one row per (directory, model). Columns:
    directory, model, input_tokens, output_tokens, total_tokens, cost_usd.
    """
    header = ["directory", "model", "input_tokens", "output_tokens", "total_tokens", "cost_usd"]
    f = open(out_path, "w", newline="", encoding="utf-8") if out_path else sys.stdout
    try:
        writer = csv.writer(f)
        writer.writerow(header)
        for dir_name in sorted(totals.keys()):
            t = totals[dir_name]
            by_model = t.get("by_model") or {}
            for model in sorted(by_model.keys()):
                b = by_model[model]
                inp = b.get("total_input_tokens", 0)
                out = b.get("total_output_tokens", 0)
                writer.writerow([dir_name, model, inp, out, inp + out, b.get("total_cost_usd", 0)])
        if out_path and f is not sys.stdout:
            f.flush()
    finally:
        if out_path and f is not sys.stdout:
            f.close()
            print(f"Wrote summary CSV to {os.path.abspath(out_path)}", file=sys.stderr)


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(
        description="Collect per-question token usage from RLM pass_at_k JSONs in qa_output_rlm_* and qa_responses_rlm_*."
    )
    p.add_argument(
        "root_dir",
        nargs="?",
        default=None,
        help="Root directory to scan (default: repo root)",
    )
    p.add_argument(
        "--flat",
        action="store_true",
        help="Output flat list (one row per question per model per file)",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Print result as JSON to stdout",
    )
    p.add_argument(
        "--qa-output",
        action="store_true",
        help="Only collect from qa_output_rlm_* directories",
    )
    p.add_argument(
        "--qa-responses",
        action="store_true",
        help="Only collect from qa_responses_rlm_* directories",
    )
    p.add_argument(
        "--csv",
        nargs="?",
        const="",
        default=None,
        metavar="FILE",
        help="Write CSV table (one row per directory, columns: directory, num_json_files, json_files, then per-model input_tokens, output_tokens, cost_usd). Use --csv FILE to write to file, or --csv to print to stdout.",
    )
    p.add_argument(
        "--summary-csv",
        nargs="?",
        const="",
        default=None,
        metavar="FILE",
        help="Write summary CSV: one row per (directory, model), columns directory, model, input_tokens, output_tokens, total_tokens, cost_usd. Use --summary-csv FILE to write to file, or --summary-csv to print to stdout.",
    )
    p.add_argument(
        "-m",
        "--model",
        action="append",
        default=None,
        dest="models",
        metavar="NAME",
        help="Filter to only include this model (can be repeated, e.g. -m gpt-5 -m sonnet4p5). If not set, all models are included.",
    )
    args = p.parse_args()
    root = args.root_dir
    model_filter: Optional[set] = set(args.models) if args.models else None
    if args.qa_output:
        data = collect_tokens_rlm_qa_output(root_dir=root, model_filter=model_filter)
    elif args.qa_responses:
        data = collect_tokens_rlm_qa_responses(root_dir=root, model_filter=model_filter)
    else:
        data = collect_tokens_rlm(root_dir=root, model_filter=model_filter)
    if args.flat and not args.qa_output and not args.qa_responses:
        data = collect_tokens_rlm_flat(root_dir=root, model_filter=model_filter)
    elif args.flat:
        # Flatten the nested dict we just built
        flat: List[Dict[str, Any]] = []
        for dir_name, files in data.items():
            for file_name, by_model in files.items():
                for model, rows in by_model.items():
                    for r in rows:
                        flat.append({
                            "dir_name": dir_name,
                            "file_name": file_name,
                            "question_id": r.get("question_id"),
                            "question_idx": r.get("question_idx"),
                            "model": r.get("model"),
                            "total_calls": r.get("total_calls"),
                            "total_input_tokens": r.get("total_input_tokens"),
                            "total_output_tokens": r.get("total_output_tokens"),
                            "cost_usd": r.get("cost_usd"),
                        })
        data = flat
    else:
        pass  # data already set above
    if args.csv is not None:
        # CSV requires nested data (not flat)
        if args.flat:
            # Re-collect nested for totals
            if args.qa_output:
                data_for_csv = collect_tokens_rlm_qa_output(root_dir=root, model_filter=model_filter)
            elif args.qa_responses:
                data_for_csv = collect_tokens_rlm_qa_responses(root_dir=root, model_filter=model_filter)
            else:
                data_for_csv = collect_tokens_rlm(root_dir=root, model_filter=model_filter)
        else:
            data_for_csv = data
        totals = aggregate_totals_by_dir(data_for_csv)
        write_totals_csv(totals, out_path=args.csv if args.csv else None)
    if args.summary_csv is not None:
        if args.flat:
            if args.qa_output:
                data_for_summary = collect_tokens_rlm_qa_output(root_dir=root, model_filter=model_filter)
            elif args.qa_responses:
                data_for_summary = collect_tokens_rlm_qa_responses(root_dir=root, model_filter=model_filter)
            else:
                data_for_summary = collect_tokens_rlm(root_dir=root, model_filter=model_filter)
        else:
            data_for_summary = data
        totals_summary = aggregate_totals_by_dir(data_for_summary)
        write_summary_by_dir_model_csv(totals_summary, out_path=args.summary_csv if args.summary_csv else None)
    if args.json:
        if args.flat:
            print(json.dumps(data, indent=2))
        else:
            totals = aggregate_totals_by_dir(data)
            print(json.dumps({"directories": data, "totals_by_dir": totals}, indent=2))
    else:
        if args.flat:
            for row in data:
                print(row)
        else:
            totals = aggregate_totals_by_dir(data)
            for dir_name, files in data.items():
                print(dir_name)
                t = totals.get(dir_name, {})
                n_files = t.get("num_json_files", 0)
                json_files = t.get("json_files") or []
                print(f"  num_json_files={n_files}  json_files={json_files}")
                print(f"  TOTAL: input_tokens={t.get('total_input_tokens', 0)} output_tokens={t.get('total_output_tokens', 0)} total_tokens={t.get('total_tokens', 0)} cost_usd={t.get('total_cost_usd', 0)}")
                print(format_dir_model_summary_table(dir_name, t))
                for file_name, by_model in files.items():
                    print(f"  {file_name}")
                    for model, rows in by_model.items():
                        print(f"    {model}: {len(rows)} questions")
                        for r in rows[:3]:
                            print(f"      {r}")
                        if len(rows) > 3:
                            print(f"      ... and {len(rows) - 3} more")


if __name__ == "__main__":
    main()
