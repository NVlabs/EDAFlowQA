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
Utility module for counting tokens in design artifact documents.
Provides functions to calculate token counts for files in the design_artifacts directory.
"""

import os
import tiktoken
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Iterable
import json
import re
from collections import defaultdict
import math
import statistics


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Count the number of tokens in a text string.
    
    Args:
        text: The text to count tokens for
        model: The model name to use for tokenization (default: "gpt-4")
    
    Returns:
        Number of tokens in the text
    """
    try:
        # Try to get the appropriate encoder for the model
        if "gpt-4" in model.lower():
            encoder = tiktoken.encoding_for_model("gpt-4")
        elif "gpt-3.5" in model.lower():
            encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
        elif "claude" in model.lower():
            # Claude uses cl100k_base encoding (same as GPT-4)
            encoder = tiktoken.get_encoding("cl100k_base")
        else:
            # Default to cl100k_base encoding (used by gpt-4 and gpt-3.5-turbo)
            encoder = tiktoken.get_encoding("cl100k_base")
        
        return len(encoder.encode(text))
    except Exception as e:
        print(f"Warning: Could not count tokens for model {model}: {e}")
        # Fallback: rough estimate (4 characters per token)
        return len(text) // 4


def count_file_tokens(file_path: str, model: str = "gpt-4", encoding: str = "utf-8") -> Tuple[int, bool]:
    """
    Count tokens in a single file.
    
    Args:
        file_path: Path to the file
        model: Model name for tokenization (default: "gpt-4")
        encoding: File encoding (default: "utf-8")
    
    Returns:
        Tuple of (token_count, success_flag)
    """
    try:
        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
            content = f.read()
        return count_tokens(content, model), True
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return 0, False


def parse_design_tech_node(directory_name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse design name and technology node from directory name.
    
    Examples:
        "aes_asap7" -> ("aes", "asap7")
        "aes_block_asap7" -> ("aes_block", "asap7")
        "ariane133_nangate45" -> ("ariane133", "nangate45")
        "ibex_sky130hd" -> ("ibex", "sky130hd")
        "aes_asap7_vs_aes_nangate45" -> (None, None)  # comparison dir
    
    Args:
        directory_name: Name of the directory
    
    Returns:
        Tuple of (design_name, tech_node) or (None, None) if not parseable
    """
    # Skip comparison directories (contain "_vs_")
    if "_vs_" in directory_name:
        return None, None
    
    # Skip special directories
    if directory_name in ["tar_files", "logs"]:
        return None, None
    
    # Common technology nodes
    tech_nodes = [
        "asap7", "nangate45", "sky130hd", "sky130hs", "sky130", 
        "gf180", "nanagte45"  # typo in ariane133_nanagte45
    ]
    
    # Try to find technology node at the end
    for tech in tech_nodes:
        if directory_name.endswith(tech):
            # Extract design name (everything before tech node)
            design = directory_name[:-(len(tech) + 1)]  # +1 for underscore
            return design, tech
    
    # If no tech node found, return None
    return None, None


def get_design_tech_tokens(token_results: Dict[str, any]) -> Dict[str, Dict[str, int]]:
    """
    Group token counts by design and technology node combination.
    
    Args:
        token_results: Results from calculate_design_artifacts_tokens()
    
    Returns:
        Dictionary mapping design_tech_node to token count:
        {
            "aes_asap7": {"total_tokens": 12500, "file_count": 10},
            "aes_nangate45": {"total_tokens": 11200, "file_count": 9},
            ...
        }
    """
    design_tech_tokens = defaultdict(lambda: {"total_tokens": 0, "file_count": 0, "files": []})
    
    # Process directory tokens
    for dir_path, tokens in token_results.get('directory_tokens', {}).items():
        # Get the top-level directory name (design_tech combination)
        top_dir = dir_path.split('/')[0] if '/' in dir_path else dir_path
        
        design, tech = parse_design_tech_node(top_dir)
        if design and tech:
            key = f"{design}_{tech}"
            # Only count if this is the exact directory (not a subdirectory)
            if dir_path == top_dir:
                design_tech_tokens[key]["total_tokens"] = tokens
    
    # Count files for each design_tech combination
    for file_path, tokens in token_results.get('file_tokens', {}).items():
        # Get the top-level directory name
        top_dir = file_path.split('/')[0] if '/' in file_path else file_path
        
        design, tech = parse_design_tech_node(top_dir)
        if design and tech:
            key = f"{design}_{tech}"
            design_tech_tokens[key]["file_count"] += 1
            design_tech_tokens[key]["files"].append(file_path)
    
    # Convert defaultdict to regular dict
    return dict(design_tech_tokens)


def print_design_tech_summary(
    design_tech_tokens: Dict[str, Dict[str, int]], 
    sort_by: str = "tokens"
) -> None:
    """
    Print a formatted summary of token counts by design and technology node.
    
    Args:
        design_tech_tokens: Output from get_design_tech_tokens()
        sort_by: Sort by "tokens" or "name" (default: "tokens")
    """
    if not design_tech_tokens:
        print("No design/tech node combinations found.")
        return
    
    print("\n" + "=" * 80)
    print("TOKEN COUNTS BY DESIGN AND TECHNOLOGY NODE")
    print("=" * 80)
    
    # Sort the results
    if sort_by == "tokens":
        sorted_items = sorted(
            design_tech_tokens.items(), 
            key=lambda x: x[1]["total_tokens"], 
            reverse=True
        )
    else:
        sorted_items = sorted(design_tech_tokens.items())
    
    # Calculate totals
    total_tokens = sum(item[1]["total_tokens"] for item in sorted_items)
    total_files = sum(item[1]["file_count"] for item in sorted_items)
    
    # Print header
    print(f"{'Design + Tech Node':<40} {'Files':>8} {'Tokens':>15} {'% of Total':>12}")
    print("-" * 80)
    
    # Print each design/tech combination
    for design_tech, data in sorted_items:
        tokens = data["total_tokens"]
        file_count = data["file_count"]
        percentage = (tokens / total_tokens * 100) if total_tokens > 0 else 0
        
        print(f"{design_tech:<40} {file_count:>8} {tokens:>15,} {percentage:>11.1f}%")
    
    # Print totals
    print("-" * 80)
    print(f"{'TOTAL':<40} {total_files:>8} {total_tokens:>15,} {100.0:>11.1f}%")
    print("=" * 80 + "\n")


def group_by_design(design_tech_tokens: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, any]]:
    """
    Group token counts by design name (aggregating across technology nodes).
    
    Args:
        design_tech_tokens: Output from get_design_tech_tokens()
    
    Returns:
        Dictionary mapping design name to aggregated stats:
        {
            "aes": {
                "total_tokens": 50000,
                "file_count": 40,
                "tech_nodes": ["asap7", "nangate45", "sky130hd", "sky130hs"],
                "tokens_by_tech": {"asap7": 12500, "nangate45": 11200, ...}
            }
        }
    """
    design_groups = defaultdict(lambda: {
        "total_tokens": 0,
        "file_count": 0,
        "tech_nodes": [],
        "tokens_by_tech": {}
    })
    
    for design_tech, data in design_tech_tokens.items():
        # Parse design and tech from the key
        design, tech = parse_design_tech_node(design_tech)
        if design and tech:
            design_groups[design]["total_tokens"] += data["total_tokens"]
            design_groups[design]["file_count"] += data["file_count"]
            design_groups[design]["tech_nodes"].append(tech)
            design_groups[design]["tokens_by_tech"][tech] = data["total_tokens"]
    
    return dict(design_groups)


def group_by_tech_node(design_tech_tokens: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, any]]:
    """
    Group token counts by technology node (aggregating across designs).
    
    Args:
        design_tech_tokens: Output from get_design_tech_tokens()
    
    Returns:
        Dictionary mapping tech node to aggregated stats:
        {
            "asap7": {
                "total_tokens": 25000,
                "file_count": 20,
                "designs": ["aes", "aes_block"],
                "tokens_by_design": {"aes": 12500, "aes_block": 12500}
            }
        }
    """
    tech_groups = defaultdict(lambda: {
        "total_tokens": 0,
        "file_count": 0,
        "designs": [],
        "tokens_by_design": {}
    })
    
    for design_tech, data in design_tech_tokens.items():
        # Parse design and tech from the key
        design, tech = parse_design_tech_node(design_tech)
        if design and tech:
            tech_groups[tech]["total_tokens"] += data["total_tokens"]
            tech_groups[tech]["file_count"] += data["file_count"]
            tech_groups[tech]["designs"].append(design)
            tech_groups[tech]["tokens_by_design"][design] = data["total_tokens"]
    
    return dict(tech_groups)


def print_design_summary(design_groups: Dict[str, Dict[str, any]]) -> None:
    """
    Print a summary of token counts grouped by design.
    
    Args:
        design_groups: Output from group_by_design()
    """
    if not design_groups:
        print("No design groups found.")
        return
    
    print("\n" + "=" * 80)
    print("TOKEN COUNTS BY DESIGN (across all technology nodes)")
    print("=" * 80)
    
    # Sort by token count
    sorted_items = sorted(
        design_groups.items(), 
        key=lambda x: x[1]["total_tokens"], 
        reverse=True
    )
    
    # Calculate totals
    total_tokens = sum(item[1]["total_tokens"] for item in sorted_items)
    total_files = sum(item[1]["file_count"] for item in sorted_items)
    
    # Print header
    print(f"{'Design':<30} {'Tech Nodes':>12} {'Files':>8} {'Tokens':>15} {'% of Total':>12}")
    print("-" * 80)
    
    # Print each design
    for design, data in sorted_items:
        tokens = data["total_tokens"]
        file_count = data["file_count"]
        tech_count = len(data["tech_nodes"])
        percentage = (tokens / total_tokens * 100) if total_tokens > 0 else 0
        
        print(f"{design:<30} {tech_count:>12} {file_count:>8} {tokens:>15,} {percentage:>11.1f}%")
    
    # Print totals
    print("-" * 80)
    print(f"{'TOTAL':<30} {'-':>12} {total_files:>8} {total_tokens:>15,} {100.0:>11.1f}%")
    print("=" * 80 + "\n")


def print_tech_node_summary(tech_groups: Dict[str, Dict[str, any]]) -> None:
    """
    Print a summary of token counts grouped by technology node.
    
    Args:
        tech_groups: Output from group_by_tech_node()
    """
    if not tech_groups:
        print("No technology node groups found.")
        return
    
    print("\n" + "=" * 80)
    print("TOKEN COUNTS BY TECHNOLOGY NODE (across all designs)")
    print("=" * 80)
    
    # Sort by token count
    sorted_items = sorted(
        tech_groups.items(), 
        key=lambda x: x[1]["total_tokens"], 
        reverse=True
    )
    
    # Calculate totals
    total_tokens = sum(item[1]["total_tokens"] for item in sorted_items)
    total_files = sum(item[1]["file_count"] for item in sorted_items)
    
    # Print header
    print(f"{'Technology Node':<30} {'Designs':>12} {'Files':>8} {'Tokens':>15} {'% of Total':>12}")
    print("-" * 80)
    
    # Print each tech node
    for tech, data in sorted_items:
        tokens = data["total_tokens"]
        file_count = data["file_count"]
        design_count = len(data["designs"])
        percentage = (tokens / total_tokens * 100) if total_tokens > 0 else 0
        
        print(f"{tech:<30} {design_count:>12} {file_count:>8} {tokens:>15,} {percentage:>11.1f}%")
    
    # Print totals
    print("-" * 80)
    print(f"{'TOTAL':<30} {'-':>12} {total_files:>8} {total_tokens:>15,} {100.0:>11.1f}%")
    print("=" * 80 + "\n")


def calculate_design_artifacts_tokens(
    design_artifacts_dir: str = "./design_artifacts",
    model: str = "gpt-4",
    file_extensions: Optional[List[str]] = None,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Calculate token counts for all documents in the design_artifacts directory.
    
    Args:
        design_artifacts_dir: Path to the design_artifacts directory
        model: Model name for tokenization (default: "gpt-4")
        file_extensions: List of file extensions to process (e.g., ['.txt', '.rpt', '.json'])
                        If None, defaults to ['.txt', '.rpt']
        verbose: Print progress information
    
    Returns:
        Dictionary containing:
        - 'total_tokens': Total token count across all files
        - 'total_files': Total number of files processed
        - 'file_tokens': Dict mapping file paths to token counts
        - 'directory_tokens': Dict mapping directories to total token counts
        - 'failed_files': List of files that couldn't be processed
    """
    if file_extensions is None:
        # Default: only count design artifact text + reports
        file_extensions = ['.txt', '.rpt']
    
    # Convert to absolute path if relative
    if not os.path.isabs(design_artifacts_dir):
        design_artifacts_dir = os.path.abspath(design_artifacts_dir)
    
    if not os.path.exists(design_artifacts_dir):
        raise ValueError(f"Directory not found: {design_artifacts_dir}")
    
    file_tokens = {}
    directory_tokens = {}
    failed_files = []
    total_tokens = 0
    total_files = 0
    
    if verbose:
        print(f"Scanning directory: {design_artifacts_dir}")
        print(f"File extensions: {file_extensions}")
        print(f"Model: {model}")
        print("-" * 80)
    
    # Walk through all files in the directory
    for root, dirs, files in os.walk(design_artifacts_dir):
        dir_tokens = 0
        
        for file in files:
            # Check if file has a matching extension
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext not in file_extensions:
                continue
            
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, design_artifacts_dir)
            
            # Count tokens in the file
            token_count, success = count_file_tokens(file_path, model)
            
            if success:
                file_tokens[relative_path] = token_count
                dir_tokens += token_count
                total_tokens += token_count
                total_files += 1
                
                if verbose:
                    print(f"  {relative_path}: {token_count:,} tokens")
            else:
                failed_files.append(relative_path)
        
        # Store directory token count if any files were processed
        if dir_tokens > 0:
            relative_dir = os.path.relpath(root, design_artifacts_dir)
            directory_tokens[relative_dir] = dir_tokens
    
    if verbose:
        print("-" * 80)
        print(f"\nSummary:")
        print(f"  Total files processed: {total_files:,}")
        print(f"  Total tokens: {total_tokens:,}")
        print(f"  Failed files: {len(failed_files)}")
        if failed_files:
            print(f"\nFailed files:")
            for f in failed_files:
                print(f"    - {f}")
    
    return {
        'total_tokens': total_tokens,
        'total_files': total_files,
        'file_tokens': file_tokens,
        'directory_tokens': directory_tokens,
        'failed_files': failed_files,
        'model': model
    }


def get_top_token_files(
    token_results: Dict[str, any],
    top_n: int = 10
) -> List[Tuple[str, int]]:
    """
    Get the top N files by token count from token calculation results.
    
    Args:
        token_results: Results from calculate_design_artifacts_tokens()
        top_n: Number of top files to return
    
    Returns:
        List of (file_path, token_count) tuples sorted by token count (descending)
    """
    file_tokens = token_results.get('file_tokens', {})
    sorted_files = sorted(file_tokens.items(), key=lambda x: x[1], reverse=True)
    return sorted_files[:top_n]


def get_top_token_directories(
    token_results: Dict[str, any],
    top_n: int = 10
) -> List[Tuple[str, int]]:
    """
    Get the top N directories by total token count.
    
    Args:
        token_results: Results from calculate_design_artifacts_tokens()
        top_n: Number of top directories to return
    
    Returns:
        List of (directory_path, token_count) tuples sorted by token count (descending)
    """
    directory_tokens = token_results.get('directory_tokens', {})
    sorted_dirs = sorted(directory_tokens.items(), key=lambda x: x[1], reverse=True)
    return sorted_dirs[:top_n]


def save_token_report(
    token_results: Dict[str, any],
    output_file: str = "token_report.json",
    include_file_details: bool = True
) -> None:
    """
    Save token calculation results to a JSON file.
    
    Args:
        token_results: Results from calculate_design_artifacts_tokens()
        output_file: Path to output JSON file
        include_file_details: Include individual file token counts in the report
    """
    report = {
        'summary': {
            'total_tokens': token_results['total_tokens'],
            'total_files': token_results['total_files'],
            'failed_files_count': len(token_results['failed_files']),
            'model': token_results['model']
        },
        'directory_tokens': token_results['directory_tokens'],
        'failed_files': token_results['failed_files']
    }
    
    if include_file_details:
        report['file_tokens'] = token_results['file_tokens']
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nToken report saved to: {output_file}")


def print_token_summary(token_results: Dict[str, any], top_n: int = 5) -> None:
    """
    Print a formatted summary of token counts.
    
    Args:
        token_results: Results from calculate_design_artifacts_tokens()
        top_n: Number of top items to show in each category
    """
    print("\n" + "=" * 80)
    print("TOKEN SUMMARY")
    print("=" * 80)
    print(f"Total Tokens: {token_results['total_tokens']:,}")
    print(f"Total Files: {token_results['total_files']:,}")
    print(f"Model: {token_results['model']}")
    
    # Average tokens per file
    if token_results['total_files'] > 0:
        avg_tokens = token_results['total_tokens'] / token_results['total_files']
        print(f"Average Tokens per File: {avg_tokens:,.1f}")
    
    # Top files
    print(f"\n{'Top ' + str(top_n) + ' Files by Token Count':^80}")
    print("-" * 80)
    top_files = get_top_token_files(token_results, top_n)
    for i, (file_path, tokens) in enumerate(top_files, 1):
        print(f"{i:2d}. {file_path:60s} {tokens:>12,} tokens")
    
    # Top directories
    print(f"\n{'Top ' + str(top_n) + ' Directories by Token Count':^80}")
    print("-" * 80)
    top_dirs = get_top_token_directories(token_results, top_n)
    for i, (dir_path, tokens) in enumerate(top_dirs, 1):
        print(f"{i:2d}. {dir_path:60s} {tokens:>12,} tokens")
    
    # Failed files
    if token_results['failed_files']:
        print(f"\n{'Failed Files':^80}")
        print("-" * 80)
        for file_path in token_results['failed_files']:
            print(f"  - {file_path}")
    
    print("=" * 80 + "\n")


def filter_file_tokens_by_extension(
    file_tokens: Dict[str, int],
    include_extensions: Iterable[str] = (".txt", ".rpt"),
) -> Dict[str, int]:
    """
    Filter a {file_path: token_count} mapping by file extension.

    Args:
        file_tokens: Mapping of file path (relative or absolute) to token count
        include_extensions: Extensions to keep (case-insensitive)

    Returns:
        Filtered mapping including only files with allowed extensions.
    """
    allowed = {ext.lower() for ext in include_extensions}
    filtered: Dict[str, int] = {}
    for path, tokens in file_tokens.items():
        ext = os.path.splitext(path)[1].lower()
        if ext in allowed:
            filtered[path] = tokens
    return filtered


def _quantile_inclusive(sorted_vals: List[int], q: float) -> float:
    """
    Inclusive quantile with linear interpolation (like NumPy's default for many uses).

    Args:
        sorted_vals: Data sorted ascending
        q: Quantile in [0, 1]
    """
    if not sorted_vals:
        return float("nan")
    if q <= 0:
        return float(sorted_vals[0])
    if q >= 1:
        return float(sorted_vals[-1])
    n = len(sorted_vals)
    pos = (n - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    w = pos - lo
    return float(sorted_vals[lo] * (1 - w) + sorted_vals[hi] * w)


def compute_token_count_stats(token_counts: List[int]) -> Dict[str, Any]:
    """
    Compute summary statistics for a list of token counts.
    """
    if not token_counts:
        return {
            "n": 0,
            "total": 0,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "max": None,
            "mean": None,
            "stdev": None,
        }

    vals = sorted(int(x) for x in token_counts)
    n = len(vals)
    total = int(sum(vals))
    mean_v = float(total / n)
    median_v = float(statistics.median(vals))
    p25 = _quantile_inclusive(vals, 0.25)
    p75 = _quantile_inclusive(vals, 0.75)
    stdev_v = float(statistics.stdev(vals)) if n >= 2 else 0.0

    return {
        "n": n,
        "total": total,
        "min": int(vals[0]),
        "p25": p25,
        "median": median_v,
        "p75": p75,
        "max": int(vals[-1]),
        "mean": mean_v,
        "stdev": stdev_v,
    }


def token_counts_table_markdown(
    file_tokens: Dict[str, int],
    include_extensions: Iterable[str] = (".txt", ".rpt"),
    limit: Optional[int] = None,
    sort_desc: bool = True,
) -> str:
    """
    Create a markdown table for token counts, filtered by extension.
    """
    filtered = filter_file_tokens_by_extension(file_tokens, include_extensions)
    rows = sorted(filtered.items(), key=lambda kv: kv[1], reverse=sort_desc)
    if limit is not None:
        rows = rows[: max(0, int(limit))]

    lines = [
        "| file | tokens |",
        "|---|---:|",
    ]
    for path, tokens in rows:
        lines.append(f"| `{path}` | {tokens:,} |")
    return "\n".join(lines)


def token_counts_summary_markdown(
    file_tokens: Dict[str, int],
    include_extensions: Iterable[str] = (".txt", ".rpt"),
    top_n: int = 10,
) -> str:
    """
    Create a markdown summary (stats + top files) for token counts, filtered by extension.
    """
    filtered = filter_file_tokens_by_extension(file_tokens, include_extensions)
    counts = list(filtered.values())
    stats = compute_token_count_stats(counts)

    lines: List[str] = []
    lines.append("## Token-count summary")
    lines.append("")
    lines.append(f"- **extensions included**: {', '.join(sorted({e.lower() for e in include_extensions}))}")
    lines.append(f"- **files included**: {stats['n']}")
    if stats["n"] == 0:
        lines.append("")
        lines.append("_No matching files after extension filtering._")
        return "\n".join(lines)

    lines.append(f"- **total tokens**: {stats['total']:,}")
    lines.append(
        "- **min / p25 / median / p75 / max**: "
        f"{stats['min']:,} / {stats['p25']:.1f} / {stats['median']:.1f} / {stats['p75']:.1f} / {stats['max']:,}"
    )
    lines.append(f"- **mean (stdev)**: {stats['mean']:.1f} ({stats['stdev']:.1f})")
    lines.append("")

    top = sorted(filtered.items(), key=lambda kv: kv[1], reverse=True)[: max(0, int(top_n))]
    lines.append(f"## Top {len(top)} files")
    lines.append("")
    lines.append(token_counts_table_markdown(dict(top), include_extensions=include_extensions, limit=None, sort_desc=True))
    return "\n".join(lines)


def ascii_histogram(
    values: List[int],
    bins: int = 20,
    width: int = 40,
) -> str:
    """
    Build a simple ASCII histogram for quick inspection.
    """
    if not values:
        return "(no data)"
    vals = [int(v) for v in values]
    vmin, vmax = min(vals), max(vals)
    if vmin == vmax:
        return f"[{vmin:,}] " + ("#" * min(width, len(vals)))

    bins = max(1, int(bins))
    counts = [0] * bins
    for v in vals:
        idx = int((v - vmin) / (vmax - vmin) * (bins - 1))
        counts[idx] += 1

    max_c = max(counts) or 1
    lines: List[str] = []
    for i, c in enumerate(counts):
        lo = vmin + (vmax - vmin) * i / bins
        hi = vmin + (vmax - vmin) * (i + 1) / bins
        bar = "#" * int(round(c / max_c * width))
        lines.append(f"[{lo:,.0f}, {hi:,.0f}) {c:>4d} {bar}")
    return "\n".join(lines)


def save_token_count_distribution_plot(
    file_tokens: Dict[str, int],
    output_path: str,
    include_extensions: Iterable[str] = (".txt", ".rpt"),
    bins: int = 30,
    title: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Save a histogram plot of token counts.

    - If matplotlib is available, saves a PNG to output_path and returns {"plotted": True, ...}
    - Otherwise, returns {"plotted": False, "ascii_histogram": "..."} for fallback inspection.
    """
    filtered = filter_file_tokens_by_extension(file_tokens, include_extensions)
    values = list(filtered.values())
    if not values:
        return {"plotted": False, "output_path": output_path, "reason": "no values after extension filtering"}

    try:
        import matplotlib.pyplot as plt  # type: ignore

        plt.figure(figsize=(10, 5))
        plt.hist(values, bins=max(1, int(bins)))
        plt.xlabel("Tokens per file")
        plt.ylabel("File count")
        plt.title(title or "Token-count distribution")
        plt.tight_layout()
        plt.savefig(output_path, dpi=160)
        plt.close()
        return {"plotted": True, "output_path": output_path, "n": len(values)}
    except Exception as e:
        return {
            "plotted": False,
            "output_path": output_path,
            "reason": f"matplotlib unavailable or failed: {e}",
            "ascii_histogram": ascii_histogram(values),
            "n": len(values),
        }


def plot_tokens_by_design(
    design_groups: Dict[str, Dict[str, any]],
    output_path: str,
    figsize: Tuple[int, int] = (12, 6)
) -> Dict[str, Any]:
    """
    Create a bar chart showing token counts by design (aggregated across tech nodes).
    
    Args:
        design_groups: Output from group_by_design()
        output_path: Path to save the plot
        figsize: Figure size (width, height)
    
    Returns:
        Dictionary with plotting status
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        designs = sorted(design_groups.keys(), key=lambda d: design_groups[d]["total_tokens"], reverse=True)
        tokens = [design_groups[d]["total_tokens"] for d in designs]
        
        plt.figure(figsize=figsize)
        bars = plt.bar(range(len(designs)), tokens, color='steelblue', edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars
        for i, (bar, token_val) in enumerate(zip(bars, tokens)):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{token_val:,}',
                    ha='center', va='bottom', fontsize=9)
        
        plt.xlabel('Design', fontsize=12, fontweight='bold')
        plt.ylabel('Total Tokens', fontsize=12, fontweight='bold')
        plt.title('Token Counts by Design (Aggregated Across Technology Nodes)', 
                 fontsize=14, fontweight='bold', pad=15)
        plt.xticks(range(len(designs)), designs, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return {"plotted": True, "output_path": output_path, "n_designs": len(designs)}
    except Exception as e:
        return {"plotted": False, "output_path": output_path, "reason": str(e)}


def plot_tokens_by_tech_node(
    tech_groups: Dict[str, Dict[str, any]],
    output_path: str,
    figsize: Tuple[int, int] = (10, 6)
) -> Dict[str, Any]:
    """
    Create a bar chart showing token counts by technology node (aggregated across designs).
    
    Args:
        tech_groups: Output from group_by_tech_node()
        output_path: Path to save the plot
        figsize: Figure size (width, height)
    
    Returns:
        Dictionary with plotting status
    """
    try:
        import matplotlib.pyplot as plt
        
        tech_nodes = sorted(tech_groups.keys(), key=lambda t: tech_groups[t]["total_tokens"], reverse=True)
        tokens = [tech_groups[t]["total_tokens"] for t in tech_nodes]
        
        plt.figure(figsize=figsize)
        bars = plt.bar(range(len(tech_nodes)), tokens, color='coral', edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars
        for i, (bar, token_val) in enumerate(zip(bars, tokens)):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{token_val:,}',
                    ha='center', va='bottom', fontsize=9)
        
        plt.xlabel('Technology Node', fontsize=12, fontweight='bold')
        plt.ylabel('Total Tokens', fontsize=12, fontweight='bold')
        plt.title('Token Counts by Technology Node (Aggregated Across Designs)', 
                 fontsize=14, fontweight='bold', pad=15)
        plt.xticks(range(len(tech_nodes)), tech_nodes, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return {"plotted": True, "output_path": output_path, "n_tech_nodes": len(tech_nodes)}
    except Exception as e:
        return {"plotted": False, "output_path": output_path, "reason": str(e)}


def plot_design_tech_heatmap(
    design_tech_tokens: Dict[str, Dict[str, int]],
    output_path: str,
    figsize: Tuple[int, int] = (14, 8)
) -> Dict[str, Any]:
    """
    Create a heatmap showing token counts for each design-tech combination.
    
    Args:
        design_tech_tokens: Output from get_design_tech_tokens()
        output_path: Path to save the plot
        figsize: Figure size (width, height)
    
    Returns:
        Dictionary with plotting status
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Extract unique designs and tech nodes
        designs = set()
        tech_nodes = set()
        for design_tech in design_tech_tokens.keys():
            design, tech = parse_design_tech_node(design_tech)
            if design and tech:
                designs.add(design)
                tech_nodes.add(tech)
        
        designs = sorted(list(designs))
        tech_nodes = sorted(list(tech_nodes))
        
        # Create matrix
        matrix = np.zeros((len(designs), len(tech_nodes)))
        for design_tech, data in design_tech_tokens.items():
            design, tech = parse_design_tech_node(design_tech)
            if design and tech:
                i = designs.index(design)
                j = tech_nodes.index(tech)
                matrix[i, j] = data["total_tokens"]
        
        # Create heatmap
        plt.figure(figsize=figsize)
        im = plt.imshow(matrix, cmap='YlOrRd', aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('Total Tokens', fontsize=11, fontweight='bold')
        
        # Set ticks and labels
        plt.xticks(range(len(tech_nodes)), tech_nodes, rotation=45, ha='right')
        plt.yticks(range(len(designs)), designs)
        
        # Add text annotations
        for i in range(len(designs)):
            for j in range(len(tech_nodes)):
                if matrix[i, j] > 0:
                    text = plt.text(j, i, f'{int(matrix[i, j]):,}',
                                   ha="center", va="center", color="black", fontsize=8)
        
        plt.xlabel('Technology Node', fontsize=12, fontweight='bold')
        plt.ylabel('Design', fontsize=12, fontweight='bold')
        plt.title('Token Counts by Design-Technology Node Combination', 
                 fontsize=14, fontweight='bold', pad=15)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return {"plotted": True, "output_path": output_path, 
                "n_designs": len(designs), "n_tech_nodes": len(tech_nodes)}
    except Exception as e:
        return {"plotted": False, "output_path": output_path, "reason": str(e)}


def plot_log_vs_report_comparison(
    file_tokens: Dict[str, int],
    output_path: str,
    figsize: Tuple[int, int] = (10, 6)
) -> Dict[str, Any]:
    """
    Create a comparison plot showing token distribution for log files (.txt) vs report files (.rpt).
    
    Args:
        file_tokens: Dictionary mapping file paths to token counts
        output_path: Path to save the plot
        figsize: Figure size (width, height)
    
    Returns:
        Dictionary with plotting status
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        log_tokens = []
        report_tokens = []
        
        for file_path, tokens in file_tokens.items():
            ext = os.path.splitext(file_path)[1].lower()
            if ext == '.txt':
                log_tokens.append(tokens)
            elif ext == '.rpt':
                report_tokens.append(tokens)
        
        if not log_tokens and not report_tokens:
            return {"plotted": False, "output_path": output_path, "reason": "no log or report files found"}
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Histogram for log files
        if log_tokens:
            axes[0].hist(log_tokens, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
            axes[0].set_xlabel('Tokens per File', fontsize=11, fontweight='bold')
            axes[0].set_ylabel('File Count', fontsize=11, fontweight='bold')
            axes[0].set_title(f'Log Files (.txt)\n(n={len(log_tokens)}, total={sum(log_tokens):,} tokens)', 
                            fontsize=12, fontweight='bold')
            axes[0].grid(axis='y', alpha=0.3, linestyle='--')
        else:
            axes[0].text(0.5, 0.5, 'No log files', ha='center', va='center', transform=axes[0].transAxes)
            axes[0].set_title('Log Files (.txt)', fontsize=12, fontweight='bold')
        
        # Histogram for report files
        if report_tokens:
            axes[1].hist(report_tokens, bins=30, color='coral', edgecolor='black', alpha=0.7)
            axes[1].set_xlabel('Tokens per File', fontsize=11, fontweight='bold')
            axes[1].set_ylabel('File Count', fontsize=11, fontweight='bold')
            axes[1].set_title(f'Report Files (.rpt)\n(n={len(report_tokens)}, total={sum(report_tokens):,} tokens)', 
                            fontsize=12, fontweight='bold')
            axes[1].grid(axis='y', alpha=0.3, linestyle='--')
        else:
            axes[1].text(0.5, 0.5, 'No report files', ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Report Files (.rpt)', fontsize=12, fontweight='bold')
        
        plt.suptitle('Token Distribution: Log Files vs Report Files', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return {"plotted": True, "output_path": output_path, 
                "n_log_files": len(log_tokens), "n_report_files": len(report_tokens)}
    except Exception as e:
        return {"plotted": False, "output_path": output_path, "reason": str(e)}


def plot_token_distribution_by_design(
    file_tokens: Dict[str, int],
    design_tech_tokens: Dict[str, Dict[str, int]],
    output_path: str,
    figsize: Tuple[int, int] = (14, 8)
) -> Dict[str, Any]:
    """
    Create box plots showing token distribution for each design.
    
    Args:
        file_tokens: Dictionary mapping file paths to token counts
        design_tech_tokens: Output from get_design_tech_tokens()
        output_path: Path to save the plot
        figsize: Figure size (width, height)
    
    Returns:
        Dictionary with plotting status
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Group file tokens by design
        design_file_tokens = defaultdict(list)
        for file_path, tokens in file_tokens.items():
            top_dir = file_path.split('/')[0] if '/' in file_path else file_path
            design, tech = parse_design_tech_node(top_dir)
            if design:
                design_file_tokens[design].append(tokens)
        
        if not design_file_tokens:
            return {"plotted": False, "output_path": output_path, "reason": "no design data found"}
        
        designs = sorted(design_file_tokens.keys(), 
                        key=lambda d: np.median(design_file_tokens[d]), reverse=True)
        data = [design_file_tokens[d] for d in designs]
        
        plt.figure(figsize=figsize)
        bp = plt.boxplot(data, labels=designs, patch_artist=True, 
                        showmeans=True, meanline=True)
        
        # Color the boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(bp['boxes'])))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        plt.xlabel('Design', fontsize=12, fontweight='bold')
        plt.ylabel('Tokens per File', fontsize=12, fontweight='bold')
        plt.title('Token Distribution by Design (Box Plot)', 
                 fontsize=14, fontweight='bold', pad=15)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.yscale('log')  # Use log scale for better visualization
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return {"plotted": True, "output_path": output_path, "n_designs": len(designs)}
    except Exception as e:
        return {"plotted": False, "output_path": output_path, "reason": str(e)}


def plot_token_distribution_by_tech_node(
    file_tokens: Dict[str, int],
    design_tech_tokens: Dict[str, Dict[str, int]],
    output_path: str,
    figsize: Tuple[int, int] = (12, 6)
) -> Dict[str, Any]:
    """
    Create box plots showing token distribution for each technology node.
    
    Args:
        file_tokens: Dictionary mapping file paths to token counts
        design_tech_tokens: Output from get_design_tech_tokens()
        output_path: Path to save the plot
        figsize: Figure size (width, height)
    
    Returns:
        Dictionary with plotting status
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Group file tokens by tech node
        tech_file_tokens = defaultdict(list)
        for file_path, tokens in file_tokens.items():
            top_dir = file_path.split('/')[0] if '/' in file_path else file_path
            design, tech = parse_design_tech_node(top_dir)
            if tech:
                tech_file_tokens[tech].append(tokens)
        
        if not tech_file_tokens:
            return {"plotted": False, "output_path": output_path, "reason": "no tech node data found"}
        
        tech_nodes = sorted(tech_file_tokens.keys(), 
                           key=lambda t: np.median(tech_file_tokens[t]), reverse=True)
        data = [tech_file_tokens[t] for t in tech_nodes]
        
        plt.figure(figsize=figsize)
        bp = plt.boxplot(data, labels=tech_nodes, patch_artist=True, 
                        showmeans=True, meanline=True)
        
        # Color the boxes
        colors = plt.cm.Pastel1(np.linspace(0, 1, len(bp['boxes'])))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        plt.xlabel('Technology Node', fontsize=12, fontweight='bold')
        plt.ylabel('Tokens per File', fontsize=12, fontweight='bold')
        plt.title('Token Distribution by Technology Node (Box Plot)', 
                 fontsize=14, fontweight='bold', pad=15)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.yscale('log')  # Use log scale for better visualization
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return {"plotted": True, "output_path": output_path, "n_tech_nodes": len(tech_nodes)}
    except Exception as e:
        return {"plotted": False, "output_path": output_path, "reason": str(e)}


def plot_file_count_vs_tokens(
    design_tech_tokens: Dict[str, Dict[str, int]],
    output_path: str,
    figsize: Tuple[int, int] = (10, 8)
) -> Dict[str, Any]:
    """
    Create a scatter plot showing file count vs total tokens for each design-tech combination.
    
    Args:
        design_tech_tokens: Output from get_design_tech_tokens()
        output_path: Path to save the plot
        figsize: Figure size (width, height)
    
    Returns:
        Dictionary with plotting status
    """
    try:
        import matplotlib.pyplot as plt
        
        file_counts = []
        token_counts = []
        labels = []
        
        for design_tech, data in design_tech_tokens.items():
            file_counts.append(data["file_count"])
            token_counts.append(data["total_tokens"])
            labels.append(design_tech)
        
        if not file_counts:
            return {"plotted": False, "output_path": output_path, "reason": "no data points"}
        
        plt.figure(figsize=figsize)
        plt.scatter(file_counts, token_counts, s=100, alpha=0.6, edgecolors='black', linewidth=1)
        
        # Add labels for each point
        for i, label in enumerate(labels):
            plt.annotate(label, (file_counts[i], token_counts[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.xlabel('File Count', fontsize=12, fontweight='bold')
        plt.ylabel('Total Tokens', fontsize=12, fontweight='bold')
        plt.title('File Count vs Total Tokens per Design-Tech Combination', 
                 fontsize=14, fontweight='bold', pad=15)
        plt.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return {"plotted": True, "output_path": output_path, "n_points": len(file_counts)}
    except Exception as e:
        return {"plotted": False, "output_path": output_path, "reason": str(e)}


def plot_cumulative_distribution(
    file_tokens: Dict[str, int],
    output_path: str,
    include_extensions: Iterable[str] = (".txt", ".rpt"),
    figsize: Tuple[int, int] = (10, 6)
) -> Dict[str, Any]:
    """
    Create a cumulative distribution function (CDF) plot of token counts.
    
    Args:
        file_tokens: Dictionary mapping file paths to token counts
        output_path: Path to save the plot
        include_extensions: File extensions to include
        figsize: Figure size (width, height)
    
    Returns:
        Dictionary with plotting status
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        filtered = filter_file_tokens_by_extension(file_tokens, include_extensions)
        values = sorted(filtered.values())
        
        if not values:
            return {"plotted": False, "output_path": output_path, "reason": "no values after filtering"}
        
        # Calculate CDF
        n = len(values)
        x = np.array(values)
        y = np.arange(1, n + 1) / n
        
        plt.figure(figsize=figsize)
        plt.plot(x, y, linewidth=2, color='steelblue')
        plt.fill_between(x, y, alpha=0.3, color='steelblue')
        
        # Add percentile markers
        percentiles = [25, 50, 75, 90, 95]
        for p in percentiles:
            idx = int(n * p / 100)
            if idx < n:
                plt.axvline(x=x[idx], color='red', linestyle='--', alpha=0.5, linewidth=1)
                plt.text(x[idx], 0.02, f'P{p}', rotation=90, fontsize=8, 
                        verticalalignment='bottom')
        
        plt.xlabel('Tokens per File', fontsize=12, fontweight='bold')
        plt.ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
        plt.title('Cumulative Distribution Function (CDF) of Token Counts', 
                 fontsize=14, fontweight='bold', pad=15)
        plt.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return {"plotted": True, "output_path": output_path, "n_files": len(values)}
    except Exception as e:
        return {"plotted": False, "output_path": output_path, "reason": str(e)}


def generate_all_plots(
    token_results: Dict[str, any],
    output_dir: str = "./token_plots",
    design_tech_tokens: Optional[Dict[str, Dict[str, int]]] = None,
    design_groups: Optional[Dict[str, Dict[str, any]]] = None,
    tech_groups: Optional[Dict[str, Dict[str, any]]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Generate all research-quality plots for token analysis.
    
    Args:
        token_results: Results from calculate_design_artifacts_tokens()
        output_dir: Directory to save all plots
        design_tech_tokens: Optional pre-computed design-tech tokens (if None, will compute)
        design_groups: Optional pre-computed design groups (if None, will compute)
        tech_groups: Optional pre-computed tech groups (if None, will compute)
    
    Returns:
        Dictionary mapping plot names to their results
    """
    import os
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute groupings if not provided
    if design_tech_tokens is None:
        design_tech_tokens = get_design_tech_tokens(token_results)
    if design_groups is None:
        design_groups = group_by_design(design_tech_tokens)
    if tech_groups is None:
        tech_groups = group_by_tech_node(design_tech_tokens)
    
    results = {}
    
    # 1. Token distribution histogram
    results["distribution_histogram"] = save_token_count_distribution_plot(
        token_results["file_tokens"],
        os.path.join(output_dir, "01_token_distribution_histogram.png"),
        bins=40,
        title="Token Count Distribution Across All Files"
    )
    
    # 2. Tokens by design
    if design_groups:
        results["tokens_by_design"] = plot_tokens_by_design(
            design_groups,
            os.path.join(output_dir, "02_tokens_by_design.png")
        )
    
    # 3. Tokens by tech node
    if tech_groups:
        results["tokens_by_tech"] = plot_tokens_by_tech_node(
            tech_groups,
            os.path.join(output_dir, "03_tokens_by_tech_node.png")
        )
    
    # 4. Design-tech heatmap
    if design_tech_tokens:
        results["design_tech_heatmap"] = plot_design_tech_heatmap(
            design_tech_tokens,
            os.path.join(output_dir, "04_design_tech_heatmap.png")
        )
    
    # 5. Log vs Report comparison
    results["log_vs_report"] = plot_log_vs_report_comparison(
        token_results["file_tokens"],
        os.path.join(output_dir, "05_log_vs_report_comparison.png")
    )
    
    # 6. Token distribution by design (box plot)
    if design_tech_tokens:
        results["distribution_by_design"] = plot_token_distribution_by_design(
            token_results["file_tokens"],
            design_tech_tokens,
            os.path.join(output_dir, "06_token_distribution_by_design.png")
        )
    
    # 7. Token distribution by tech node (box plot)
    if design_tech_tokens:
        results["distribution_by_tech"] = plot_token_distribution_by_tech_node(
            token_results["file_tokens"],
            design_tech_tokens,
            os.path.join(output_dir, "07_token_distribution_by_tech_node.png")
        )
    
    # 8. File count vs tokens scatter
    if design_tech_tokens:
        results["file_count_vs_tokens"] = plot_file_count_vs_tokens(
            design_tech_tokens,
            os.path.join(output_dir, "08_file_count_vs_tokens.png")
        )
    
    # 9. Cumulative distribution
    results["cumulative_distribution"] = plot_cumulative_distribution(
        token_results["file_tokens"],
        os.path.join(output_dir, "09_cumulative_distribution.png")
    )
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"PLOT GENERATION SUMMARY")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir}")
    print(f"\nGenerated plots:")
    for plot_name, result in results.items():
        status = "✓" if result.get("plotted", False) else "✗"
        reason = result.get("reason", "")
        print(f"  {status} {plot_name:30s} {reason}")
    print(f"{'='*80}\n")
    
    return results


if __name__ == "__main__":
    """
    Example usage when running the script directly.
    
    Usage:
        python token_counter_util.py [design_artifacts_dir] [model] [output_file] [--plots] [--plot-dir PLOT_DIR]
    
    Examples:
        python token_counter_util.py ./design_artifacts gpt-4 report.json
        python token_counter_util.py ./design_artifacts gpt-4 report.json --plots
        python token_counter_util.py ./design_artifacts gpt-4 report.json --plots --plot-dir ./my_plots
    """
    import sys
    
    # Parse command line arguments
    design_artifacts_dir = "./design_artifacts"
    model = "gpt-4"
    output_file = None
    generate_plots = False
    plot_dir = "./token_plots"
    
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--plots":
            generate_plots = True
            i += 1
        elif args[i] == "--plot-dir" and i + 1 < len(args):
            plot_dir = args[i + 1]
            generate_plots = True
            i += 2
        else:
            if i == 0:
                design_artifacts_dir = args[i]
            elif i == 1:
                model = args[i]
            elif i == 2:
                output_file = args[i]
            i += 1
    
    # Calculate tokens
    print(f"Calculating tokens for documents in: {design_artifacts_dir}")
    results = calculate_design_artifacts_tokens(
        design_artifacts_dir=design_artifacts_dir,
        model=model,
        verbose=True
    )
    
    # Print summary
    print_token_summary(results, top_n=10)
    
    # Group by design and tech node
    print("\n" + "=" * 80)
    print("GROUPING BY DESIGN AND TECHNOLOGY NODE")
    print("=" * 80 + "\n")
    
    design_tech_tokens = get_design_tech_tokens(results)
    print_design_tech_summary(design_tech_tokens, sort_by="tokens")
    
    # Group by design
    design_groups = group_by_design(design_tech_tokens)
    print_design_summary(design_groups)
    
    # Group by tech node
    tech_groups = group_by_tech_node(design_tech_tokens)
    print_tech_node_summary(tech_groups)
    
    # Generate plots if requested
    if generate_plots:
        print("\n" + "=" * 80)
        print("GENERATING PLOTS")
        print("=" * 80 + "\n")
        plot_results = generate_all_plots(
            results,
            output_dir=plot_dir,
            design_tech_tokens=design_tech_tokens,
            design_groups=design_groups,
            tech_groups=tech_groups
        )
    
    # Save report if output file specified
    if output_file:
        report_data = {
            "summary": {
                "total_tokens": results["total_tokens"],
                "total_files": results["total_files"],
                "model": results["model"]
            },
            "design_tech_tokens": design_tech_tokens,
            "design_groups": design_groups,
            "tech_groups": tech_groups,
            "file_tokens": results["file_tokens"],
            "failed_files": results["failed_files"]
        }
        with open(output_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        print(f"\nDetailed report saved to: {output_file}")
    else:
        # Default output file
        save_token_report(results, "design_artifacts_token_report.json")
