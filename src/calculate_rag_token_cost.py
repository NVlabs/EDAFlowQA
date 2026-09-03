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
Calculate token cost of RAG agent runs from pass_at_k token usage JSON files.

Reads the token usage JSON produced by the RAG pass@k pipeline (e.g.
pass_at_k_rag_token_usage_<design>_<model>_10passes_FINAL_<timestamp>.json),
applies per-model pricing, and reports cost. Supports single file, directory
glob, or multiple files to compare costs across models.

How cost is calculated
----------------------
- The JSON provides total_prompt_tokens (input) and total_completion_tokens (output).
- Each model has a price in USD per 1M tokens for input and output (see MODEL_PRICES).
- Cost (USD) = (prompt_tokens * input_per_1m + completion_tokens * output_per_1m) / 1e6
- Averages "per pass" are: total / num_passes (e.g. total ÷ 10 for 10 runs).

Usage:
  # Single file
  python -m src.calculate_rag_token_cost path/to/pass_at_k_rag_token_usage_....json
  python src/calculate_rag_token_cost.py path/to/pass_at_k_rag_token_usage_....json

  # All FINAL token usage JSONs under a directory (one per model)
  python -m src.calculate_rag_token_cost --dir qa_output_rag_aes_asap7

  # Compare multiple files (different models)
  python -m src.calculate_rag_token_cost file1.json file2.json file3.json

  # Custom price per 1M tokens (input, output)
  python -m src.calculate_rag_token_cost --input-price 1.0 --output-price 4.0 file.json

  # Only include specific model(s) when scanning a directory
  python -m src.calculate_rag_token_cost --dir qa_output_rag_aes_asap7 --filter-model kimi2_instruct
  python -m src.calculate_rag_token_cost --dir qa_output_rag_aes_asap7 -m kimi2 -m gpt-5
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


# Default pricing: USD per 1M tokens (input, output). Update as needed for your models.
MODEL_PRICES: Dict[str, Dict[str, float]] = {
    # OpenAI
    "gpt-4o": {"input_per_1m": 2.50, "output_per_1m": 10.00},
    "gpt-4o-mini": {"input_per_1m": 0.15, "output_per_1m": 0.60},
    "gpt-4-turbo": {"input_per_1m": 10.00, "output_per_1m": 30.00},
    "gpt-4": {"input_per_1m": 30.00, "output_per_1m": 60.00},
    "gpt-3.5-turbo": {"input_per_1m": 0.50, "output_per_1m": 1.50},
    "gpt-5": {"input_per_1m": 1.25, "output_per_1m": 10.00},
    # Anthropic
    "sonnet-4.5": {"input_per_1m": 3.00, "output_per_1m": 15.00},
    "claude-sonnet-4-20250514": {"input_per_1m": 3.00, "output_per_1m": 15.00},
    "claude-sonnet-4-5-20250929": {"input_per_1m": 3.00, "output_per_1m": 15.00},
    "claude-3-5-sonnet": {"input_per_1m": 3.00, "output_per_1m": 15.00},
    "claude-3-opus": {"input_per_1m": 15.00, "output_per_1m": 75.00},
    "claude-3-haiku": {"input_per_1m": 0.25, "output_per_1m": 1.25},
    # Kimi / Moonshot
    "kimi2_instruct": {"input_per_1m": 0.60, "output_per_1m": 2.40},
    "kimi-k2": {"input_per_1m": 0.60, "output_per_1m": 2.40},
    # Add more as needed; keys are matched case-insensitively or via alias.
    # gpt-oss from azure openai
    "gpt-oss": {"input_per_1m": 0.15, "output_per_1m": 0.60},
}

# Aliases for model names that appear in filenames vs our MODEL_PRICES keys.
# RLM pass_at_k JSONs may store full API names (e.g. openai/gpt-oss-120b).
MODEL_ALIASES: Dict[str, str] = {
    "kimi2": "kimi2_instruct",
    "gpt-5": "gpt-5",  # placeholder
    "gpt-5-20250807": "gpt-5",
    "sonnet": "sonnet-4.5",
    "sonnet4p5": "sonnet-4.5",
    "gpt-oss": "gpt-oss",
    "openai/gpt-oss-120b": "gpt-oss",
    "moonshotai/kimi-k2-instruct-0905": "kimi2_instruct",
}


def _resolve_model_key(model_name: str) -> Optional[str]:
    """Return MODEL_PRICES key for a given model name (from filename or user).
    Exits with an error if model_name is non-empty but does not map to any known price.
    """
    if not model_name:
        return None
    name = model_name.strip().lower()
    print(f"Resolving model name: {name}")
    if name in MODEL_ALIASES:
        name = MODEL_ALIASES[name]
    for key in MODEL_PRICES:
        if key.lower() == name or name in key.lower():
            return key
    print(f"Unknown model name: {model_name}")
    exit(1)


def _known_model_suffixes() -> List[str]:
    """Return known model identifiers (from MODEL_PRICES and MODEL_ALIASES) sorted by length descending, so longest match wins (e.g. kimi2_instruct over instruct)."""
    keys = list(MODEL_PRICES.keys()) + list(MODEL_ALIASES.keys())
    # Dedupe and sort by length descending so we match "kimi2_instruct" before "instruct"
    seen = set()
    suffixes = []
    for k in sorted(keys, key=len, reverse=True):
        k = k.strip()
        if k and k not in seen:
            seen.add(k)
            suffixes.append(k)
    return suffixes


def _parse_model_from_filename(path: Union[str, Path]) -> Optional[str]:
    """Extract model identifier from pass_at_k_rag_token_usage_<design>_<model>_10passes_FINAL_<ts>.json

    Design can contain underscores (e.g. aes_asap7 or aes_sky130hs_vs_aes_sky130hd). We match the
    prefix (before _10passes_FINAL_ or _passNofM_) against known model names and take the longest
    match so "kimi2_instruct" is chosen over "instruct".
    """
    path = Path(path)
    stem = path.stem
    # Prefix = everything before _10passes_FINAL_ or _passNofM_
    for pattern in (r"_10passes_FINAL_", r"_pass\d+of\d+_"):
        m = re.search(pattern, stem, re.IGNORECASE)
        if m:
            prefix = stem[: m.start()].rstrip("_")
            break
    else:
        return None
    # Find longest known model that prefix ends with
    for model in _known_model_suffixes():
        if prefix.endswith("_" + model) or prefix == model:
            return model
    # Fallback: last segment (e.g. unknown model)
    parts = prefix.split("_")
    return parts[-1] if parts else None


def load_token_usage(path: Union[str, Path]) -> Dict[str, Any]:
    """Load token usage JSON. Raises if file missing or invalid."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Token usage file not found: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    if "summary" not in data:
        raise ValueError(f"Expected 'summary' in token usage JSON: {path}")
    return data


def get_summary_tokens(data: Dict[str, Any]) -> Tuple[int, int]:
    """Return (total_prompt_tokens, total_completion_tokens) from token usage JSON."""
    s = data.get("summary", {})
    prompt = int(s.get("total_prompt_tokens", 0))
    completion = int(s.get("total_completion_tokens", 0))
    return prompt, completion


def compute_cost(
    prompt_tokens: int,
    completion_tokens: int,
    input_per_1m: float,
    output_per_1m: float,
) -> float:
    """Compute cost in USD from token counts and per-1M prices."""
    return (prompt_tokens * input_per_1m + completion_tokens * output_per_1m) / 1e6


def get_prices_for_model(
    model_name: Optional[str],
    input_per_1m: Optional[float] = None,
    output_per_1m: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Resolve (input_per_1m, output_per_1m) for a model.
    Explicit input_per_1m/output_per_1m override any model lookup.
    """
    if input_per_1m is not None and output_per_1m is not None:
        return float(input_per_1m), float(output_per_1m)
    key = _resolve_model_key(model_name) if model_name else None
    if key and key in MODEL_PRICES:
        p = MODEL_PRICES[key]
        in_p = input_per_1m if input_per_1m is not None else p["input_per_1m"]
        out_p = output_per_1m if output_per_1m is not None else p["output_per_1m"]
        return in_p, out_p
    # Fallback if model unknown
    default_in = 1.0 if input_per_1m is None else input_per_1m
    default_out = 4.0 if output_per_1m is None else output_per_1m
    return default_in, default_out


def calculate_rag_token_cost(
    token_usage_path: Union[str, Path],
    model_name: Optional[str] = None,
    input_per_1m: Optional[float] = None,
    output_per_1m: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Calculate token cost for a single RAG agent token usage JSON file.

    Args:
        token_usage_path: Path to pass_at_k_rag_token_usage_*_.json file.
        model_name: Model name for pricing. If None, inferred from filename.
        input_per_1m: Override input price (USD per 1M tokens).
        output_per_1m: Override output price (USD per 1M tokens).

    Returns:
        Dict with:
          - path, model_used, prompt_tokens, completion_tokens, total_tokens
          - input_per_1m, output_per_1m, cost_usd
          - summary, num_questions, num_passes
          - avg_*_per_pass: totals divided by num_passes (e.g. total/10 for 10 runs)
    """
    path = Path(token_usage_path)
    data = load_token_usage(path)
    prompt_tokens, completion_tokens = get_summary_tokens(data)
    total_tokens = prompt_tokens + completion_tokens

    if model_name is None:
        model_name = _parse_model_from_filename(path)
    model_display = model_name or "unknown"
    in_p, out_p = get_prices_for_model(model_name, input_per_1m, output_per_1m)
    cost_usd = compute_cost(prompt_tokens, completion_tokens, in_p, out_p)

    summary = data.get("summary", {})
    num_questions = int(summary.get("num_questions") or 0)
    num_passes = int(summary.get("num_passes") or 0)

    # Average = total / num_passes (each pass runs all questions once; 10 passes => divide by 10)
    avg_prompt_per_pass = round(prompt_tokens / num_passes, 1) if num_passes else None
    avg_completion_per_pass = round(completion_tokens / num_passes, 1) if num_passes else None
    avg_total_per_pass = round(total_tokens / num_passes, 1) if num_passes else None
    avg_cost_per_pass = round(cost_usd / num_passes, 4) if num_passes else None

    return {
        "path": str(path.resolve()),
        "model_used": model_display,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "input_per_1m": in_p,
        "output_per_1m": out_p,
        "cost_usd": round(cost_usd, 4),
        "summary": summary,
        "num_questions": num_questions,
        "num_passes": num_passes,
        "avg_prompt_per_pass": avg_prompt_per_pass,
        "avg_completion_per_pass": avg_completion_per_pass,
        "avg_total_per_pass": avg_total_per_pass,
        "avg_cost_per_pass": avg_cost_per_pass,
    }


def find_final_token_usage_jsons(directory: Union[str, Path]) -> List[Path]:
    """Find all pass_at_k_rag_token_usage_*_10passes_FINAL_*.json under directory."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    pattern = "pass_at_k_rag_token_usage_*_10passes_FINAL_*.json"
    return sorted(directory.glob(pattern))


def filter_paths_by_model(
    paths: List[Path],
    filter_models: List[str],
) -> List[Path]:
    """
    Keep only paths whose parsed model name matches any of filter_models.
    Matching is case-insensitive and substring-based (e.g. 'kimi2' matches 'kimi2_instruct').
    """
    if not filter_models:
        return list(paths)
    normalized = [m.strip().lower() for m in filter_models if m]
    if not normalized:
        return list(paths)
    kept = []
    for p in paths:
        model = _parse_model_from_filename(p)
        if model is None:
            kept.append(p)
            continue
        model_lower = model.lower()
        if any(f in model_lower or model_lower in f for f in normalized):
            kept.append(p)
    return kept


def calculate_rag_token_cost_multi(
    paths: List[Union[str, Path]],
    input_per_1m: Optional[float] = None,
    output_per_1m: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Calculate token cost for multiple token usage JSON files (e.g. different models).

    Args:
        paths: List of paths to pass_at_k_rag_token_usage_*.json files.
        input_per_1m: Optional override for input price (applied to all).
        output_per_1m: Optional override for output price (applied to all).

    Returns:
        List of cost dicts (same shape as calculate_rag_token_cost).
    """
    results = []
    for p in paths:
        try:
            r = calculate_rag_token_cost(p, model_name=None, input_per_1m=input_per_1m, output_per_1m=output_per_1m)
            results.append(r)
        except Exception as e:
            results.append({
                "path": str(Path(p).resolve()),
                "error": str(e),
                "cost_usd": None,
            })
    return results


def print_cost_report(records: List[Dict[str, Any]], verbose: bool = False) -> None:
    """Print a simple table and optional per-record details."""
    if not records:
        print("No records to report.")
        return
    errors = [r for r in records if r.get("error")]
    ok = [r for r in records if not r.get("error")]
    if errors:
        print("Errors:")
        for r in errors:
            print(f"  {r['path']}: {r['error']}")
        print()
    if not ok:
        return

    # Table header (totals)
    print(f"{'Model':<24} {'Prompt tokens':>14} {'Completion tokens':>18} {'Cost (USD)':>12}  Path")
    print("-" * 100)
    for r in ok:
        path_short = r["path"]
        if len(path_short) > 48:
            path_short = "..." + path_short[-45:]
        print(
            f"{r['model_used']:<24} {r['prompt_tokens']:>14,} {r['completion_tokens']:>18,} {r['cost_usd']:>12.4f}  {path_short}"
        )
    print("-" * 100)
    total_cost = sum(r["cost_usd"] for r in ok)
    total_prompt = sum(r["prompt_tokens"] for r in ok)
    total_completion = sum(r["completion_tokens"] for r in ok)
    if len(ok) > 1:
        print(f"{' (multiple files)':<24} {total_prompt:>14,} {total_completion:>18,} {total_cost:>12.4f}")
    print()

    # Averages table (total / num_passes; each pass = all questions run once)
    has_avgs = any(r.get("avg_prompt_per_pass") is not None for r in ok)
    if has_avgs:
        print("Averages (per pass = total ÷ num_passes, one run over all questions):")
        print(f"  {'Model':<22} {'Avg prompt':>12} {'Avg completion':>14} {'Avg total':>12} {'Avg cost (USD)':>14}  Passes")
        print("  " + "-" * 88)
        for r in ok:
            ap = r.get("avg_prompt_per_pass")
            ac = r.get("avg_completion_per_pass")
            at = r.get("avg_total_per_pass")
            acost = r.get("avg_cost_per_pass")
            p = r.get("num_passes")
            ap_s = f"{ap:,.0f}" if ap is not None else "—"
            ac_s = f"{ac:,.0f}" if ac is not None else "—"
            at_s = f"{at:,.0f}" if at is not None else "—"
            acost_s = f"{acost:.4f}" if acost is not None else "—"
            print(f"  {r['model_used']:<22} {ap_s:>12} {ac_s:>14} {at_s:>12} {acost_s:>14}  {p or ''}")
        print()

    if verbose:
        for r in ok:
            print(f"--- {r['model_used']} ---")
            print(f"  Path: {r['path']}")
            print(f"  Input price: ${r['input_per_1m']}/1M, Output price: ${r['output_per_1m']}/1M")
            if r.get("num_questions") and r.get("num_passes"):
                print(f"  Questions: {r['num_questions']}, Passes: {r['num_passes']}")
            if r.get("avg_prompt_per_pass") is not None:
                print(f"  Avg per pass (total÷{r['num_passes']}): prompt {r['avg_prompt_per_pass']:,.0f}, completion {r['avg_completion_per_pass']:,.0f}, total {r['avg_total_per_pass']:,.0f}, cost ${r['avg_cost_per_pass']:.4f}")
            print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate RAG agent token cost from pass_at_k token usage JSON(s)."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Token usage JSON file(s). If empty and --dir is set, use FINAL jsons under --dir.",
    )
    parser.add_argument(
        "--dir",
        "-d",
        type=str,
        default=None,
        help="Directory to search for pass_at_k_rag_token_usage_*_10passes_FINAL_*.json",
    )
    parser.add_argument(
        "--input-price",
        type=float,
        default=None,
        help="Override input price (USD per 1M tokens) for all files.",
    )
    parser.add_argument(
        "--output-price",
        type=float,
        default=None,
        help="Override output price (USD per 1M tokens) for all files.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print per-file details.",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Write results to this JSON file.",
    )
    parser.add_argument(
        "--filter-model",
        "-m",
        action="append",
        dest="filter_models",
        default=None,
        metavar="MODEL",
        help="Only include files for this model (by name parsed from filename). Can be repeated. Substring match, case-insensitive.",
    )
    args = parser.parse_args()

    paths: List[Path] = []
    if args.dir:
        paths = find_final_token_usage_jsons(args.dir)
        if not paths:
            print(f"No FINAL token usage JSONs found under {args.dir}")
            return
    for p in args.paths:
        pth = Path(p)
        if pth.is_file():
            paths.append(pth)
        elif pth.is_dir():
            paths.extend(find_final_token_usage_jsons(pth))
        else:
            print(f"Warning: not a file or directory, skipping: {p}")

    if args.filter_models:
        before = len(paths)
        paths = filter_paths_by_model(paths, args.filter_models)
        if paths and before > len(paths):
            print(f"Filtered to {len(paths)} file(s) matching model(s): {', '.join(args.filter_models)}\n")

    if not paths:
        if args.filter_models:
            print(f"No token usage files matched filter model(s): {', '.join(args.filter_models)}")
        else:
            parser.print_help()
            print("\nExample: python -m src.calculate_rag_token_cost --dir qa_output_rag_aes_asap7")
        return

    results = calculate_rag_token_cost_multi(
        paths,
        input_per_1m=args.input_price,
        output_per_1m=args.output_price,
    )
    print_cost_report(results, verbose=args.verbose)

    if args.json_out:
        out_data = {"records": results}
        with open(args.json_out, "w") as f:
            json.dump(out_data, f, indent=2)
        print(f"Results written to {args.json_out}")


if __name__ == "__main__":
    main()
