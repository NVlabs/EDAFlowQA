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
Filter Invalid Answers from QA JSON Files

This module provides functionality to filter out QA pairs with invalid answers:
- "n/a" (case-insensitive)
- null/None
- Empty dictionaries {}
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union


def is_invalid_answer(answer: Any) -> bool:
    """
    Check if an answer is invalid.
    
    An answer is considered invalid if it is:
    - "n/a" (case-insensitive)
    - null/None
    - An empty dictionary {}
    - A dictionary or list containing "n/a" as any value (recursively)
    
    Args:
        answer: The answer value to check
        
    Returns:
        True if the answer is invalid, False otherwise
    """
    if answer is None:
        return True
    
    if isinstance(answer, str):
        # Check for "n/a" case-insensitively
        if answer.strip().lower() == "n/a":
            return True
    
    if isinstance(answer, dict):
        # Check if dictionary is empty
        if len(answer) == 0:
            return True
        # Recursively check all values in the dictionary
        for value in answer.values():
            if is_invalid_answer(value):
                return True
    
    if isinstance(answer, list):
        # Check if any element in the list is invalid
        for item in answer:
            if is_invalid_answer(item):
                return True
    
    return False


def filter_qa_pairs(data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, int], List[Dict[str, str]]]:
    """
    Filter out QA pairs with invalid answers from a QA data structure.
    
    Supports both single-file and multifile QA formats:
    - Single-file: {"questions": [...], ...} or {"qa_pairs": [...], ...}
    - Multifile: {"qa_pairs": [...], ...}
    
    Args:
        data: The QA data dictionary loaded from JSON
        
    Returns:
        A tuple containing:
        - Filtered data dictionary
        - Statistics dictionary with counts of filtered items
        - List of filtered QA pairs with id, question, and answer
    """
    filtered_data = data.copy()
    stats = {
        "total": 0,
        "filtered": 0,
        "remaining": 0
    }
    filtered_items = []
    
    # Determine which field contains the QA pairs
    qa_pairs_key = None
    if "qa_pairs" in data:
        qa_pairs_key = "qa_pairs"
    elif "questions" in data:
        qa_pairs_key = "questions"
    else:
        # If neither field exists, return original data
        return filtered_data, stats, filtered_items
    
    qa_pairs = data[qa_pairs_key]
    stats["total"] = len(qa_pairs)
    
    # Filter out invalid answers
    filtered_pairs = []
    for pair in qa_pairs:
        answer = pair.get("answer")
        if not is_invalid_answer(answer):
            filtered_pairs.append(pair)
        else:
            stats["filtered"] += 1
            # Collect filtered item info
            answer = pair.get("answer")
            filtered_items.append({
                "id": pair.get("id", "unknown"),
                "question": pair.get("question", ""),
                "answer": answer
            })
    
    filtered_data[qa_pairs_key] = filtered_pairs
    stats["remaining"] = len(filtered_pairs)
    
    # Update statistics if present
    if "statistics" in filtered_data:
        if isinstance(filtered_data["statistics"], dict):
            # Update total_qa_pairs if it exists
            if "total_qa_pairs" in filtered_data["statistics"]:
                filtered_data["statistics"]["total_qa_pairs"] = stats["remaining"]
    
    return filtered_data, stats, filtered_items


def filter_qa_file(input_path: Union[str, Path], 
                   output_path: Optional[Union[str, Path]] = None,
                   in_place: bool = False,
                   print_filtered: bool = True) -> Tuple[Dict[str, int], List[Dict[str, str]]]:
    """
    Filter invalid answers from a QA JSON file.
    
    Args:
        input_path: Path to the input QA JSON file
        output_path: Optional path to save the filtered output.
                     If None and in_place is False, creates a new file with "_filtered" suffix.
        in_place: If True, overwrites the input file with filtered data
        print_filtered: If True, print the mapping of filtered QA pairs
        
    Returns:
        Tuple containing:
        - Dictionary with filtering statistics
        - List of filtered QA pairs with id and question
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Load the JSON file
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Filter the data
    filtered_data, stats, filtered_items = filter_qa_pairs(data)
    
    # Determine output path
    if in_place:
        output_path = input_path
    elif output_path is None:
        # Create a new filename with "_filtered" suffix
        output_path = input_path.parent / f"{input_path.stem}_filtered{input_path.suffix}"
    else:
        output_path = Path(output_path)
    
    # Save the filtered data
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=2, ensure_ascii=False)
    
    print(f"Filtered {stats['filtered']} invalid answers out of {stats['total']} total QA pairs")
    print(f"Remaining: {stats['remaining']} valid QA pairs")
    print(f"Output saved to: {output_path}")
    
    # Print filtered QA mapping
    if print_filtered and filtered_items:
        print(f"\n=== Filtered QA Pairs ===")
        print(f"File: {input_path.name}")
        for item in filtered_items:
            qa_id = item['id']
            question = item['question']
            answer = item.get('answer', 'N/A')
            
            # Format answer for display
            if isinstance(answer, dict):
                answer_str = json.dumps(answer, indent=2, ensure_ascii=False)
            elif isinstance(answer, list):
                answer_str = json.dumps(answer, indent=2, ensure_ascii=False)
            elif answer is None:
                answer_str = "null"
            else:
                answer_str = str(answer)
            
            # Truncate long questions and answers for readability
            if len(question) > 100:
                question = question[:97] + "..."
            if len(answer_str) > 200:
                answer_str = answer_str[:197] + "..."
            
            print(f"  {qa_id}:")
            print(f"    Question: {question}")
            print(f"    Answer: {answer_str}")
    
    return stats, filtered_items


def get_design_tech_name(file_path: Union[str, Path]) -> str:
    """
    Get the design_technology name from a QA file path for use as a mapping key.
    Uses the parent directory name (e.g. aes_asap7, gcdsky130, ibex_sky130hd).

    Args:
        file_path: Path to the QA JSON file (e.g. qa_output/aes_asap7/aes_20260201_154729_qa_all_types.json)

    Returns:
        The design_technology string, e.g. "aes_asap7", "gcdsky130"
    """
    path = Path(file_path)
    return path.parent.name if path.parent else path.stem


def find_qa_output_dirs_for_design_tech(
    search_root: Union[str, Path],
    design_tech: str,
) -> List[str]:
    """
    Find directory names under search_root that match qa_output_<agent>_<design>_<technology>
    and contain the given design_technology string. Does not require any source QA file.

    Args:
        search_root: Root directory to scan (e.g. "." or workspace root)
        design_tech: design_technology string (e.g. "aes_asap7", "gcdsky130")

    Returns:
        List of directory names (basenames) that start with "qa_output_" and
        contain design_tech, e.g. ["qa_output_cursor_aes_asap7", "qa_output_claude_code_aes_asap7"]
    """
    root = Path(search_root)
    if not root.exists() or not root.is_dir():
        return []
    found = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if name.startswith("qa_output_") and design_tech in name:
            found.append(name)
    return sorted(found)


def find_qa_json_files(qa_output_dir: Union[str, Path] = "qa_output") -> List[Path]:
    """
    Find all QA JSON files in the qa_output directory.
    
    Args:
        qa_output_dir: Path to the qa_output directory
        
    Returns:
        List of Path objects for JSON files found
    """
    qa_output_path = Path(qa_output_dir)
    
    if not qa_output_path.exists():
        print(f"Warning: Directory not found: {qa_output_dir}", file=sys.stderr)
        return []
    
    # Find all JSON files recursively, excluding raw files to avoid duplicates
    json_files = [f for f in qa_output_path.rglob("*.json") 
                  if not f.name.endswith("_raw.json")]
    
    return sorted(json_files)


def extract_all_qa_pairs(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract all QA pairs from a QA data structure.
    
    Args:
        data: The QA data dictionary loaded from JSON
        
    Returns:
        List of all QA pairs with id, question, and answer
    """
    all_pairs = []
    
    # Determine which field contains the QA pairs
    qa_pairs_key = None
    if "qa_pairs" in data:
        qa_pairs_key = "qa_pairs"
    elif "questions" in data:
        qa_pairs_key = "questions"
    else:
        return all_pairs
    
    qa_pairs = data[qa_pairs_key]
    
    for pair in qa_pairs:
        all_pairs.append({
            "id": pair.get("id", "unknown"),
            "question": pair.get("question", ""),
            "answer": pair.get("answer")
        })
    
    return all_pairs


def print_all_qa_mapping(
    file_paths: List[Union[str, Path]],
    search_root: Optional[Union[str, Path]] = None,
) -> None:
    """
    Print ALL QA pairs for multiple files: design_technology -> QA id + QA question + answer.
    Uses design_technology (e.g. aes_asap7, gcdsky130) as the mapping key.
    FILE MAP is built by scanning for qa_output_<agent>_<design>_<technology> dirs (no source QA file).

    Args:
        file_paths: List of paths to QA JSON files
        search_root: Root to scan for qa_output_* directories (default: current directory)
    """
    all_mappings = {}  # design_tech -> list of qa pairs

    for file_path in file_paths:
        file_path = Path(file_path)
        if not file_path.exists():
            print(f"Warning: File not found: {file_path}", file=sys.stderr)
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            all_pairs = extract_all_qa_pairs(data)
            if all_pairs:
                design_tech = get_design_tech_name(file_path)
                all_mappings[design_tech] = all_pairs
        except Exception as e:
            print(f"Warning: Error processing {file_path}: {e}", file=sys.stderr)
            continue

    root = Path(search_root) if search_root is not None else Path(".")
    file_map = {dt: find_qa_output_dirs_for_design_tech(root, dt) for dt in all_mappings}

    # Print file map: design_technology -> directory names (qa_output_<agent>_<design>_<technology>)
    print("\n" + "="*80)
    print("FILE MAP: design_technology -> qa_output_<agent>_<design>_<technology> directory names")
    print("="*80)
    for design_tech, dirs in sorted(file_map.items()):
        print(f"  {design_tech} -> {dirs}")

    print("\n" + "="*80)
    print("ALL QA PAIRS: design_technology -> QA ID + QA Question + Answer")
    print("="*80)

    for design_tech, qa_pairs in sorted(all_mappings.items()):
        dirs = file_map.get(design_tech, [])
        print(f"\n{design_tech} (dirs: {dirs}):")
        for item in qa_pairs:
            qa_id = item['id']
            question = item['question']
            answer = item.get('answer', 'N/A')
            
            # Format answer for display
            if isinstance(answer, dict):
                answer_str = json.dumps(answer, indent=2, ensure_ascii=False)
            elif isinstance(answer, list):
                answer_str = json.dumps(answer, indent=2, ensure_ascii=False)
            elif answer is None:
                answer_str = "null"
            else:
                answer_str = str(answer)
            
            # Truncate long questions and answers for readability
            if len(question) > 100:
                question = question[:97] + "..."
            if len(answer_str) > 200:
                answer_str = answer_str[:197] + "..."
            
            print(f"  {qa_id}:")
            print(f"    Question: {question}")
            print(f"    Answer: {answer_str}")
    
    print("\n" + "="*80)
    print(f"Total files processed: {len(all_mappings)}")
    total_qa_pairs = sum(len(items) for items in all_mappings.values())
    print(f"Total QA pairs: {total_qa_pairs}")
    print("="*80)


def print_qa_mapping(
    file_paths: List[Union[str, Path]],
    search_root: Optional[Union[str, Path]] = None,
) -> None:
    """
    Print QA mapping for filtered (invalid) answers: design_technology -> QA id + QA question + answer.
    Uses design_technology (e.g. aes_asap7, gcdsky130) as the mapping key.
    FILE MAP is built by scanning for qa_output_<agent>_<design>_<technology> dirs (no source QA file).

    Args:
        file_paths: List of paths to QA JSON files
        search_root: Root to scan for qa_output_* directories (default: current directory)
    """
    all_mappings = {}  # design_tech -> list of filtered items

    for file_path in file_paths:
        file_path = Path(file_path)
        if not file_path.exists():
            print(f"Warning: File not found: {file_path}", file=sys.stderr)
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            _, _, filtered_items = filter_qa_pairs(data)
            if filtered_items:
                design_tech = get_design_tech_name(file_path)
                all_mappings[design_tech] = filtered_items
        except Exception as e:
            print(f"Warning: Error processing {file_path}: {e}", file=sys.stderr)
            continue

    root = Path(search_root) if search_root is not None else Path(".")
    file_map = {dt: find_qa_output_dirs_for_design_tech(root, dt) for dt in all_mappings}

    # Print file map: design_technology -> directory names (qa_output_<agent>_<design>_<technology>)
    print("\n" + "="*80)
    print("FILE MAP: design_technology -> qa_output_<agent>_<design>_<technology> directory names")
    print("="*80)
    for design_tech, dirs in sorted(file_map.items()):
        print(f"  {design_tech} -> {dirs}")

    print("\n" + "="*80)
    print("FILTERED QA PAIRS (Invalid Answers): design_technology -> QA ID + QA Question + Answer")
    print("="*80)

    for design_tech, filtered_items in sorted(all_mappings.items()):
        dirs = file_map.get(design_tech, [])
        print(f"\n{design_tech} (dirs: {dirs}):")
        for item in filtered_items:
            qa_id = item['id']
            question = item['question']
            answer = item.get('answer', 'N/A')
            
            # Format answer for display
            if isinstance(answer, dict):
                answer_str = json.dumps(answer, indent=2, ensure_ascii=False)
            elif isinstance(answer, list):
                answer_str = json.dumps(answer, indent=2, ensure_ascii=False)
            elif answer is None:
                answer_str = "null"
            else:
                answer_str = str(answer)
            
            # Truncate long questions and answers for readability
            if len(question) > 100:
                question = question[:97] + "..."
            if len(answer_str) > 200:
                answer_str = answer_str[:197] + "..."
            
            print(f"  {qa_id}:")
            print(f"    Question: {question}")
            print(f"    Answer: {answer_str}")
    
    print("\n" + "="*80)
    print(f"Total files processed: {len(all_mappings)}")
    total_filtered = sum(len(items) for items in all_mappings.values())
    print(f"Total filtered QA pairs: {total_filtered}")
    print("="*80)


def collect_filtered_qa_pairs(
    file_paths: List[Union[str, Path]],
    search_root: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Collect filtered (invalid) QA pairs from multiple files.
    Uses design_technology (e.g. aes_asap7, gcdsky130) as the mapping key.
    file_map is built by scanning for directories qa_output_<agent>_<design>_<technology>;
    no source QA file path is used for the mapping.

    Args:
        file_paths: List of paths to QA JSON files
        search_root: Root to scan for qa_output_* directories (default: current directory).
                     Used only to build file_map: design_technology -> list of matching dir names.

    Returns:
        Dictionary with:
        - "file_map": design_technology -> list of directory names that contain it
                      (e.g. "aes_asap7" -> ["qa_output_cursor_aes_asap7", "qa_output_claude_code_aes_asap7"])
        - "filtered_qa_pairs": design_technology -> list of {id, question, answer}
        - "statistics": total_files_with_filtered, total_filtered_pairs
    """
    filtered_by_design_tech = {}

    for file_path in file_paths:
        file_path = Path(file_path)
        if not file_path.exists():
            continue
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            _, _, filtered_items = filter_qa_pairs(data)
            if filtered_items:
                design_tech = get_design_tech_name(file_path)
                filtered_by_design_tech[design_tech] = filtered_items
        except Exception:
            continue

    total_filtered = sum(len(items) for items in filtered_by_design_tech.values())

    # Build file_map by checking whether design_technology is in directory name qa_output_<agent>_<design>_<technology>
    root = Path(search_root) if search_root is not None else Path(".")
    file_map = {}
    for design_tech in filtered_by_design_tech:
        dirs = find_qa_output_dirs_for_design_tech(root, design_tech)
        file_map[design_tech] = dirs

    return {
        "file_map": file_map,
        "filtered_qa_pairs": filtered_by_design_tech,
        "statistics": {
            "total_files_with_filtered": len(filtered_by_design_tech),
            "total_filtered_pairs": total_filtered,
        },
    }


def dump_filtered_qa_pairs_to_json(
    file_paths: List[Union[str, Path]],
    output_path: Union[str, Path],
    search_root: Optional[Union[str, Path]] = None,
) -> Tuple[Path, Dict[str, int]]:
    """
    Dump filtered QA pairs to a JSON file.
    file_map in the output is design_technology -> list of qa_output_<agent>_<design>_<technology> dir names
    (built by scanning, no source QA file path used for mapping).

    Args:
        file_paths: List of paths to QA JSON files
        output_path: Path for the output JSON file
        search_root: Root to scan for qa_output_* directories (default: current directory)

    Returns:
        Tuple of (path to the written file, statistics dict)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = collect_filtered_qa_pairs(file_paths, search_root=search_root)
    stats = data["statistics"]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return output_path, stats


def main():
    """Command-line interface for filtering QA files."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Filter invalid answers from QA JSON files"
    )
    parser.add_argument(
        "input_files",
        nargs="*",
        type=str,
        help="Path(s) to the input QA JSON file(s). If not provided, reads all files from qa_output directory."
    )
    parser.add_argument(
        "--qa-output-dir",
        type=str,
        default="qa_output",
        help="Directory to search for QA JSON files (default: qa_output)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path to the output file (only used for single file, default: input_file with '_filtered' suffix)"
    )
    parser.add_argument(
        "-i", "--in-place",
        action="store_true",
        help="Overwrite the input file(s) with filtered data"
    )
    parser.add_argument(
        "--mapping-only",
        action="store_true",
        help="Only print the QA mapping without filtering files (defaults to showing all answers)"
    )
    parser.add_argument(
        "--all-answers",
        action="store_true",
        help="Print ALL QA pairs with answers (default when using --mapping-only)"
    )
    parser.add_argument(
        "--filtered-only",
        action="store_true",
        help="Only print filtered (invalid) QA pairs (overrides --all-answers)"
    )
    parser.add_argument(
        "--no-print",
        action="store_true",
        help="Don't print filtered QA pairs for each file"
    )
    parser.add_argument(
        "-j", "--dump-filtered",
        type=str,
        default=None,
        metavar="JSON_FILE",
        help="Dump filtered (invalid) QA pairs to this JSON file"
    )
    parser.add_argument(
        "--search-root",
        type=str,
        default=None,
        metavar="DIR",
        help="Root directory to scan for qa_output_<agent>_<design>_<technology> when building file_map (default: current directory)"
    )

    args = parser.parse_args()
    
    # If no input files provided, find all JSON files in qa_output directory
    if not args.input_files:
        print(f"No input files specified. Searching for JSON files in {args.qa_output_dir}...")
        input_files = find_qa_json_files(args.qa_output_dir)
        if not input_files:
            print(f"No JSON files found in {args.qa_output_dir}", file=sys.stderr)
            return 1
        print(f"Found {len(input_files)} JSON file(s)")
        args.input_files = [str(f) for f in input_files]

    # Print file list for filtering
    print(f"\nFile list for filtering ({len(args.input_files)} file(s)):")
    for f in args.input_files:
        print(f"  {f}")
    print()

    # Dump filtered QA pairs to JSON if requested
    if args.dump_filtered:
        out_path, stats = dump_filtered_qa_pairs_to_json(
            args.input_files, args.dump_filtered, search_root=args.search_root
        )
        total = stats["total_filtered_pairs"]
        nfiles = stats["total_files_with_filtered"]
        print(f"Dumped {total} filtered QA pairs from {nfiles} file(s) to {out_path}")
    
    try:
        if args.mapping_only:
            # Just print the mapping
            # Default to all answers unless filtered-only is specified
            if args.filtered_only:
                print_qa_mapping(args.input_files, search_root=args.search_root)
            else:
                print_all_qa_mapping(args.input_files, search_root=args.search_root)
        elif len(args.input_files) == 1:
            # Single file mode
            stats, filtered_items = filter_qa_file(
                args.input_files[0],
                output_path=args.output,
                in_place=args.in_place,
                print_filtered=not args.no_print
            )
            print(f"\nFiltering complete!")
            print(f"Statistics: {stats}")
        else:
            # Multiple files mode
            all_stats = []
            for input_file in args.input_files:
                stats, filtered_items = filter_qa_file(
                    input_file,
                    output_path=None if not args.in_place else None,
                    in_place=args.in_place,
                    print_filtered=not args.no_print
                )
                all_stats.append((input_file, stats))
            
            # Print summary
            print(f"\n=== Summary ===")
            total_filtered = sum(s['filtered'] for _, s in all_stats)
            total_qa = sum(s['total'] for _, s in all_stats)
            print(f"Total files processed: {len(all_stats)}")
            print(f"Total QA pairs: {total_qa}")
            print(f"Total filtered: {total_filtered}")
            print(f"Total remaining: {total_qa - total_filtered}")
            
            # Print mapping
            if not args.no_print:
                if args.filtered_only:
                    print_qa_mapping(args.input_files, search_root=args.search_root)
                elif args.all_answers:
                    print_all_qa_mapping(args.input_files, search_root=args.search_root)
                else:
                    # Default: show filtered only when filtering files
                    print_qa_mapping(args.input_files, search_root=args.search_root)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
