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

import json
import csv
import re
import os
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import Counter, defaultdict
import math
import pdb

def calculate_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Calculate pass@k metric for code generation evaluation.
    
    Pass@k measures the probability that at least one of the top-k generated 
    samples passes all test cases.
    
    Formula: Pass@k = E[1 - C(n-c, k) / C(n, k)]
    
    where:
        n: total number of generated samples per problem
        c: number of correct samples (that pass tests)
        k: number of samples we consider (top-k)
        C(n, k): binomial coefficient "n choose k"
    
    Args:
        n: Total number of generated samples
        c: Number of correct samples
        k: Number of samples to consider (top-k)
    
    Returns:
        Pass@k probability (between 0 and 1)
    
    Examples:
        >>> calculate_pass_at_k(n=10, c=3, k=1)  # 3 out of 10 correct, pick 1
        0.3
        >>> calculate_pass_at_k(n=10, c=3, k=5)  # 3 out of 10 correct, pick 5
        0.7916666666666666
    """
    # Handle edge cases
    if n == 0 or k == 0:
        return 0.0
    
    if c == 0:
        return 0.0
    
    if k > n:
        # Can't pick more samples than we have
        k = n
    
    if c == n:
        # All samples are correct
        return 1.0
    
    # Calculate C(n-c, k) / C(n, k) using the formula for numerical stability
    # C(n-c, k) / C(n, k) = [(n-c)! / (k! * (n-c-k)!)] / [n! / (k! * (n-k)!)]
    #                     = [(n-c)! * (n-k)!] / [n! * (n-c-k)!]
    #                     = [(n-c) * (n-c-1) * ... * (n-c-k+1)] / [n * (n-1) * ... * (n-k+1)]
    
    # If k > n - c, then C(n-c, k) = 0, so the result is 1.0
    if k > n - c:
        return 1.0
    
    # Calculate the ratio using a product to avoid overflow
    # C(n-c, k) / C(n, k) = Product from i=0 to k-1 of [(n-c-i) / (n-i)]
    ratio = 1.0
    for i in range(k):
        ratio *= (n - c - i) / (n - i)
    
    pass_at_k = 1.0 - ratio
    
    return max(0.0, min(1.0, pass_at_k))  # Clamp to [0, 1]


def calculate_pass_at_k_batch(results_by_question: Dict[str, List[bool]], k: int) -> float:
    """
    Calculate pass@k across multiple questions (expected value).
    
    Args:
        results_by_question: Dictionary mapping question_id to list of boolean results
                           (True if correct, False if incorrect) for each sample
        k: Number of samples to consider (top-k)
    
    Returns:
        Average pass@k across all questions
    
    Example:
        >>> results = {
        ...     "q1": [True, False, False, True],  # 2 correct out of 4
        ...     "q2": [False, False, False, True],  # 1 correct out of 4
        ...     "q3": [True, True, True, False],   # 3 correct out of 4
        ... }
        >>> calculate_pass_at_k_batch(results, k=2)
        0.7...
    """
    if not results_by_question:
        return 0.0
    
    pass_at_k_scores = []
    for question_id, results in results_by_question.items():
        n = len(results)
        c = sum(results)
        pass_at_k = calculate_pass_at_k(n, c, k)
        pass_at_k_scores.append(pass_at_k)
    
    # Return the expected value (average across all questions)
    return sum(pass_at_k_scores) / len(pass_at_k_scores), pass_at_k_scores


def calculate_soft_pass_at_k(similarity_scores: List[float], k: int) -> float:
    """
    Calculate soft pass@k using continuous similarity scores.
    
    Instead of binary pass/fail, this uses the maximum similarity score
    among k randomly selected samples. Returns the expected maximum similarity.
    
    This is computed by considering all possible combinations of k samples
    and taking the expected value of the maximum similarity in each combination.
    
    For efficiency with large n, we can use the formula:
    E[max of k samples] ≈ average of top-k samples (when k is large)
    
    For exact calculation, we compute the expected max over all combinations.
    
    Args:
        similarity_scores: List of similarity scores (0 to 1) for all samples
        k: Number of samples to consider (top-k)
    
    Returns:
        Expected maximum similarity score when selecting k samples
    
    Example:
        >>> scores = [0.3, 0.5, 0.8, 0.2, 0.9]
        >>> calculate_soft_pass_at_k(scores, k=2)
        0.7...  # Expected max when picking 2 samples
    """
    if not similarity_scores or k == 0:
        return 0.0
    
    n = len(similarity_scores)
    if k >= n:
        # If we pick all samples, just return the maximum
        return max(similarity_scores)
    
    if k == 1:
        # If k=1, return the average (expected value of a random pick)
        return sum(similarity_scores) / n
    
    # For small n, compute exact expected maximum over all combinations
    # For large n (n < 2 * k), use an approximation for efficiency
    # if n < 2 * k:
    from itertools import combinations
    # Calculate expected max over all possible k-combinations
    total_max = 0.0
    num_combinations = 0
        
    for combo in combinations(range(n), k):
        max_in_combo = max(similarity_scores[i] for i in combo)
        total_max += max_in_combo
        num_combinations += 1
        
    return total_max / num_combinations
    
    """else:
        # Approximation: Use the CDF-based formula for expected maximum
        # E[max] = integral of (1 - (1-F(x))^k) dx
        # For discrete case, we approximate using order statistics
        
        # Sort scores in descending order
        sorted_scores = sorted(similarity_scores, reverse=True)
        
        # Expected rank of maximum in sample of k from n is approximately n/(k+1)
        # We use a weighted average of top scores
        weights = []
        expected_max = 0.0
        
        for i in range(n):
            # Probability that this score is the maximum in a random sample of k
            # P(X_i is max) ≈ k/n if X_i is among top scores
            rank = i + 1
            
            # Probability that X_i appears in the sample and is the max
            # This is approximately: (k/n) * (1 - (rank-1)/n)^k
            # Simplified: use inverse rank weighting for top k*2 elements
            if i < min(k * 2, n):
                weight = (k / n) * ((n - i) / n) ** (k - 1)
                weights.append(weight)
                expected_max += sorted_scores[i] * weight
        
        # Normalize
        total_weight = sum(weights) if weights else 1.0
        return expected_max / total_weight if total_weight > 0 else sorted_scores[0]
    """

def calculate_soft_pass_at_k_batch(similarity_by_question: Dict[str, List[float]], k: int) -> float:
    """
    Calculate soft pass@k across multiple questions using similarity scores.
    
    Args:
        similarity_by_question: Dictionary mapping question_id to list of similarity scores
                              (0 to 1) for each sample
        k: Number of samples to consider (top-k)
    
    Returns:
        Average soft pass@k across all questions
    
    Example:
        >>> scores = {
        ...     "q1": [0.3, 0.5, 0.8, 0.2],  # max of 2 would be expected ~0.65
        ...     "q2": [0.1, 0.2, 0.9, 0.3],  # max of 2 would be expected ~0.60
        ... }
        >>> calculate_soft_pass_at_k_batch(scores, k=2)
        0.6...
    """
    if not similarity_by_question:
        return 0.0
    
    soft_pass_scores = []
    for question_id, scores in similarity_by_question.items():
        soft_pass = calculate_soft_pass_at_k(scores, k)
        soft_pass_scores.append(soft_pass)
    
    # Return the expected value (average across all questions)
    return sum(soft_pass_scores) / len(soft_pass_scores), soft_pass_scores


def calculate_threshold_pass_at_k(similarity_scores: List[float], k: int, threshold: float = 0.8) -> float:
    """
    Calculate threshold-based pass@k using similarity scores.
    
    Similar to binary pass@k, but considers a sample as "passing" if its
    similarity score >= threshold.
    
    Args:
        similarity_scores: List of similarity scores (0 to 1) for all samples
        k: Number of samples to consider (top-k)
        threshold: Minimum similarity score to consider as "passing" (default: 0.8)
    
    Returns:
        Probability that at least one of k samples has similarity >= threshold
    
    Example:
        >>> scores = [0.3, 0.5, 0.85, 0.2, 0.9]  # 2 above 0.8
        >>> calculate_threshold_pass_at_k(scores, k=3, threshold=0.8)
        0.7...  # Probability at least 1 of 3 samples has score >= 0.8
    """
    if not similarity_scores or k == 0:
        return 0.0
    
    # Convert to binary based on threshold
    passing = [score >= threshold for score in similarity_scores]
    n = len(passing)
    c = sum(passing)
    
    # Use standard pass@k formula with binary outcomes
    return calculate_pass_at_k(n, c, k)


def load_pass_k_ignored_qa_pairs(
    filepath: str,
) -> Tuple[Dict[str, Set[Tuple[str, str]]], Set[str], Dict[str, List[Dict[str, Any]]], Dict[str, List[str]]]:
    """
    Load the pass_k_ignored_qa_pairs.json file (new format).

    The file is expected to have structure:
      { "file_map": { "<design>_<technology>": ["qa_output_<agent>_<design>_<technology>", ...], ... },
        "filtered_qa_pairs": { "<design>_<technology>": [ { "id": "...", "question": "...", "answer": ... }, ... ], ... } }
    Legacy format with file_map value as a single filename string is accepted (normalized to a list).

    Returns:
        Tuple of:
        - by_source_pairs: Dict mapping design_technology -> set of (qid, question) tuples to exclude.
        - all_ignored_qids: Set of all question IDs across all sources (union).
        - raw_data: The full "filtered_qa_pairs" dict (keyed by design_technology).
        - file_map: Dict mapping design_technology -> list of directory names (qa_output_<agent>_<design>_<technology>).
                    Legacy single-filename values are returned as a one-element list.

    Example:
        >>> by_source, all_qids, raw, file_map = load_pass_k_ignored_qa_pairs("pass_k_ignored_qa_pairs.json")
        >>> pairs_aes = by_source.get("aes_asap7", set())
        >>> dirs = file_map.get("aes_asap7", [])
    """
    path = Path(filepath)
    if not path.exists():
        return {}, set(), {}, {}

    with open(path, "r") as f:
        data = json.load(f)

    filtered = data.get("filtered_qa_pairs", {})
    if not isinstance(filtered, dict):
        return {}, set(), {}, {}

    # Load file_map (new format: design_tech -> list of dir names; legacy: design_tech -> filename string)
    raw_file_map = data.get("file_map", {})
    if not isinstance(raw_file_map, dict):
        raw_file_map = {}
    file_map: Dict[str, List[str]] = {}
    for k, v in raw_file_map.items():
        if isinstance(v, list):
            file_map[k] = [str(x) for x in v]
        elif isinstance(v, str):
            file_map[k] = [v]  # legacy: single filename
        else:
            file_map[k] = []

    by_source_pairs: Dict[str, Set[Tuple[str, str]]] = {}
    all_ignored_qids: Set[str] = set()

    for source_file, pairs in filtered.items():
        if not isinstance(pairs, list):
            continue
        pairs_this_source: Set[Tuple[str, str]] = set()
        for item in pairs:
            if not isinstance(item, dict):
                continue
            qid = item.get("id")
            question = item.get("question")
            if qid is not None and question is not None:
                pairs_this_source.add((qid, question))
                all_ignored_qids.add(qid)
        if pairs_this_source:
            by_source_pairs[source_file] = pairs_this_source
    # print(f"by_source_pairs: {by_source_pairs}")
    # pdb.set_trace()
    return by_source_pairs, all_ignored_qids, filtered, file_map


# Directory name prefixes that precede design_tech (e.g. qa_responses_cursor_aes_sky130hs_vs_aes_sky130hd)
_DESIGN_TECH_DIR_PREFIXES = (
    'qa_responses_cursor_',
    'qa_responses_rag_',
    'qa_responses_claude_code_',
    'qa_output_cursor_',
    'qa_output_rag_',
    'qa_output_claude_code_',
)


def resolve_design_tech_from_directory(
    directory: str,
    design_tech_keys: List[str],
) -> Optional[str]:
    """
    Resolve design_technology key from a directory path.
    - If directory name contains "_vs_": treat as multifile; only consider keys that contain "_vs_"
      (so we never pick a single-design key like aes_sky130hd for qa_responses_cursor_aes_sky130hs_vs_aes_sky130hd).
    - If directory name does not contain "_vs_": treat as single-file; only consider keys that do
      not contain "_vs_", then return longest match.

    design_tech is None when no key matches for that mode (multifile vs single-file).

    Args:
        directory: Directory path (e.g. "./qa_output_cursor_aes_asap7/" or "qa_responses_cursor_aes_sky130hs_vs_aes_sky130hd")
        design_tech_keys: Keys from pass_k_ignored_qa_pairs (e.g. ["aes_asap7", "aes_sky130hs_vs_aes_sky130hd"])

    Returns:
        The design_tech key that matches the directory (multifile key for _vs_ dirs, single-file key otherwise), or None.
    """
    dir_name = Path(directory).name
    is_multifile = "_vs_" in dir_name

    # Extract design part (suffix after known prefix) for exact match
    design_part: Optional[str] = None
    for prefix in _DESIGN_TECH_DIR_PREFIXES:
        if dir_name.startswith(prefix):
            design_part = dir_name[len(prefix):]
            break

    if is_multifile:
        # Multifile: only consider keys that contain "_vs_"
        if design_part and design_part in design_tech_keys:
            return design_part
        matches = [k for k in design_tech_keys if "_vs_" in k and k in dir_name]
        if matches:
            return max(matches, key=len)
        # No key in JSON: return extracted design part so design_tech is set (e.g. aes_sky130hs_vs_aes_sky130hd)
        if design_part:
            return design_part
        return None
    else:
        # Single-file: only consider keys that do NOT contain "_vs_"
        if design_part and design_part in design_tech_keys:
            return design_part
        matches = [k for k in design_tech_keys if "_vs_" not in k and k in dir_name]
        if matches:
            return max(matches, key=len)
        # No key in JSON: return extracted design part so design_tech is set (e.g. ariane133_nanagte45)
        if design_part:
            return design_part
        return None


def find_qa_json_in_directory(directory: str) -> Optional[str]:
    """
    Find the first QA JSON file in a directory (excluding *_raw.json, pass*of*, *_evaluation.json).
    Only returns a file that contains "qa_pairs" or "questions" (actual QA structure).
    Used when source_qa_file_path is not provided but the directory may contain a QA file.

    Returns:
        Path to the first QA JSON file with qa_pairs/questions, or None if not found.
    """
    dir_path = Path(directory)
    if not dir_path.exists() or not dir_path.is_dir():
        return None
    for f in sorted(dir_path.glob("*.json")):
        if not f.is_file() or f.name.endswith("_raw.json"):
            continue
        if "pass" in f.name and "of" in f.name:
            continue
        if f.name.endswith("_evaluation.json"):
            continue
        try:
            with open(f, "r") as fp:
                data = json.load(fp)
            if isinstance(data, dict) and ("qa_pairs" in data or "questions" in data):
                return str(f)
        except (json.JSONDecodeError, OSError):
            continue
    return None


def load_qa_pairs_id_question(filepath: str) -> List[Tuple[str, str]]:
    """
    Load the ordered list of (id, question) from a source QA JSON file.

    The file is expected to have a "qa_pairs" list with items containing "id" and "question".
    Used to match evaluation results (which are in the same order) to (qid, question) for
    filtering when the same qid can appear multiple times with different questions.

    Returns:
        List of (id, question) tuples in the same order as qa_pairs in the file.
    """
    path = Path(filepath)
    if not path.exists():
        return []

    with open(path, "r") as f:
        data = json.load(f)

    qa_pairs = data.get("qa_pairs", [])
    if not isinstance(qa_pairs, list):
        return []

    out: List[Tuple[str, str]] = []
    for item in qa_pairs:
        if not isinstance(item, dict):
            continue
        qid = item.get("id")
        question = item.get("question")
        if qid is not None and question is not None:
            out.append((qid, question))
        else:
            out.append((str(qid) if qid is not None else "", str(question) if question is not None else ""))
    return out


def load_pass_json_files(
    directory: str,
    model_name: str,
    ignored_qids: Optional[Set[str]] = None,
    ignored_pairs: Optional[Set[Tuple[str, str]]] = None,
    qa_pairs_id_question: Optional[List[Tuple[str, str]]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load JSON evaluation files from pass 1 to pass 10, filtered by model name.

    How by_source_pairs flows into exclusion:
    - load_pass_k_ignored_qa_pairs() returns by_source_pairs: design_tech -> set of (qid, question).
    - load_and_calculate_pass_at_k() picks one source: pairs_this_source = by_source_pairs[design_tech]
      when design_tech is resolved; then sets ignored_pairs (and optionally qa_pairs_id_question from
      source QA file) or ignored_qids only when design_tech is None or no QA file.
    - This function: *_evaluation.json does not contain question text. (qid, question) exclusion
      is only used when qa_pairs_id_question is provided (index-based match). When ignored_pairs
      is set but qa_pairs_id_question is missing, exclusion is by qid only. When ignored_pairs is
      not set, exclusion is by ignored_qids.

    Searches for files matching the pattern: *_pass{1-10}of10_*{model_name}*_evaluation.json

    Args:
        directory: Directory path containing the evaluation JSON files
        model_name: Model name pattern to match in filenames (e.g., "sonnet4p5", "gpt-5.2")
                   Only files containing this model name will be loaded
        ignored_qids: Optional set of question IDs to exclude (used when qid+question not available)
        ignored_pairs: Optional set of (qid, question) to exclude; (qid, question) match used only if qa_pairs_id_question is provided, else qid-only
        qa_pairs_id_question: Ordered list of (id, question) from the source QA file for index-based (qid, question) match (evaluation.json has no question field)

    Returns:
        Dictionary mapping pass number (1-10) to dict with 'filename' and 'data' keys
        Each 'data' is a list of evaluation results (excluding ignored)

    Example:
        >>> by_source, _, _, _ = load_pass_k_ignored_qa_pairs("pass_k_ignored_qa_pairs.json")
        >>> pairs = by_source.get("aes_asap7", set())
        >>> qa_list = load_qa_pairs_id_question("qa_output/aes_asap7/aes_20260201_154729_qa_all_types.json")
        >>> results = load_pass_json_files("./out", "sonnet4p5", ignored_pairs=pairs, qa_pairs_id_question=qa_list)
    """
    directory_path = Path(directory)
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    if not model_name:
        raise ValueError("model_name is required and cannot be empty")

    pass_files = {}
    if ignored_qids is None:
        ignored_qids = set()
    # (qid, question) exclusion only when we have ordered QA list: *_evaluation.json does not contain question text.
    use_pair_filter = bool(
        ignored_pairs
        and qa_pairs_id_question is not None
        and len(qa_pairs_id_question) > 0
    )
    ignored_qids_from_pairs = {qid for qid, _ in (ignored_pairs or set())}

    def should_exclude(result: Dict[str, Any], index: int) -> bool:
        if not isinstance(result, dict):
            return False
        qid = result.get("question_id")
        if use_pair_filter:
            # Index-based (qid, question) match using qa_pairs_id_question (evaluation.json has no question field).
            if index < len(qa_pairs_id_question):
                _id, question = qa_pairs_id_question[index]
                return (qid, question) in ignored_pairs
            return False
        if ignored_pairs:
            # No qa_pairs_id_question: exclude by qid only (all qids that appear in any ignored pair).
            return qid in ignored_qids_from_pairs
        return qid in ignored_qids

    # Pattern to match: *_pass{1-10}of10_*_evaluation.json
    pattern = re.compile(r'.*_pass(\d+)of10_.*_evaluation\.json$')

    for json_file in directory_path.glob('*_evaluation.json'):
        match = pattern.match(json_file.name)
        if match:
            pass_num = int(match.group(1))
            if 1 <= pass_num <= 10:
                # Only load files that contain the model name
                if model_name in json_file.name:
                    try:
                        with open(json_file, 'r') as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                filtered_data = []
                                for i, r in enumerate(data):
                                    if should_exclude(r, i):
                                        qid = r.get("question_id", "?")
                                        print(f"  Excluded question: qid={qid}")
                                    else:
                                        filtered_data.append(r)
                                n_excluded = len(data) - len(filtered_data)
                                if n_excluded > 0:
                                    kind = "ignored (qid, question)" if use_pair_filter else "ignored qids"
                                    print(f"  Excluded {n_excluded} result(s) from {json_file.name} ({kind})")
                                pass_files[pass_num] = {
                                    'filename': json_file.name,
                                    'data': filtered_data
                                }
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse {json_file}: {e}")
                        continue
    print(f"Loaded {len(pass_files)} files for model {model_name}")
    for pass_num, file_info in sorted(pass_files.items()):
        print(f"  Pass {pass_num}: {file_info['filename']}")
    return pass_files


def verify_evaluation_file_count(
    pass_files: Dict[int, Dict[str, Any]],
    directory: str,
    model_name: str,
    required_passes: int = 10,
) -> Tuple[bool, List[int]]:
    """
    Verify that exactly `required_passes` evaluation files (pass 1 through pass N) are present.

    Args:
        pass_files: Dictionary from load_pass_json_files (pass number -> file info)
        directory: Directory path (for error messages)
        model_name: Model name (for error messages)
        required_passes: Expected number of passes (default 10)

    Returns:
        (is_valid, missing_pass_numbers). is_valid is True only when all passes 1..required_passes exist.
    """
    expected = set(range(1, required_passes + 1))
    found = set(pass_files.keys())
    missing = sorted(expected - found)
    is_valid = len(missing) == 0

    if not is_valid:
        print()
        print("=" * 60)
        print("FLAG: Evaluation file count verification FAILED")
        print("=" * 60)
        if len(pass_files) == 0:
            print(f"  No evaluation files found in: {directory}")
            print(f"  Expected: {required_passes} files matching *_pass{{1-{required_passes}}}of10_*{model_name}*_evaluation.json")
        else:
            print(f"  Found {len(pass_files)} of {required_passes} required pass files in: {directory}")
            print(f"  Missing passes: {missing}")
            print(f"  Pattern: *_pass{{1-{required_passes}}}of10_*{model_name}*_evaluation.json")
        print("=" * 60)
        print()
    return is_valid, missing


def calculate_question_pass_at_k_metrics(
    pass_files: Dict[int, Dict[str, Any]], 
    k_values: List[int] = [1, 5, 10],
    threshold: float = 0.8
) -> List[Dict[str, Any]]:
    """
    Calculate pass@k metrics for each question across all passes.
    
    Args:
        pass_files: Dictionary from load_pass_json_files mapping pass number to file data
        k_values: List of k values to calculate pass@k for (default: [1, 5, 10])
        threshold: Threshold for binary pass/fail classification (default: 0.8)
    
    Returns:
        List of dictionaries, one per question, containing:
        - question_id: Question identifier
        - filename: Base filename pattern
        - overall_scores: List of overall_score values across all passes
        - soft_pass_at_k: Dict mapping k to soft pass@k value
        - pass_at_k: Dict mapping k to binary pass@k value (using threshold)
        - pass_at_k_threshold: Threshold used for binary classification
        - num_passes: Number of passes available for this question
    """
    # Collect overall_score for each question across all passes
    # Use unique question identifiers to handle duplicates within the same pass file
    # First, determine the maximum number of occurrences for each question_id across all pass files
    question_id_max_counts: Dict[str, int] = {}
    base_filename = None
    
    # Sort passes to ensure consistent ordering
    sorted_passes = sorted(pass_files.keys())
    
    # First pass: count maximum occurrences of each question_id across all files
    for pass_num in sorted_passes:
        file_info = pass_files[pass_num]
        if base_filename is None:
            # Extract base filename pattern (remove pass number and timestamp)
            base_filename = file_info['filename']
            # Remove pass-specific parts to get base pattern
            base_filename = re.sub(r'_pass\d+of10_\d+_evaluation\.json$', '', base_filename)
        
        data = file_info['data']
        question_id_counts: Dict[str, int] = {}
        
        for result in data:
            if isinstance(result, dict) and 'question_id' in result:
                original_question_id = result['question_id']
                question_id_counts[original_question_id] = question_id_counts.get(original_question_id, 0) + 1
                
                # Track maximum count across all pass files
                if original_question_id not in question_id_max_counts:
                    question_id_max_counts[original_question_id] = 0
                question_id_max_counts[original_question_id] = max(
                    question_id_max_counts[original_question_id],
                    question_id_counts[original_question_id]
                )
    
    # Second pass: collect scores using consistent unique identifiers
    question_scores: Dict[str, List[float]] = {}
    question_id_mapping: Dict[str, str] = {}  # Maps unique_id -> original question_id
    
    for pass_num in sorted_passes:
        file_info = pass_files[pass_num]
        data = file_info['data']
        
        # Track question_id occurrences within this pass file
        question_id_counts: Dict[str, int] = {}
        
        # Extract overall_score for each question
        for result in data:
            if isinstance(result, dict) and 'question_id' in result:
                original_question_id = result['question_id']
                overall_score = result.get('overall_score', 0.0)
                
                # Count occurrences of this question_id in this pass file
                question_id_counts[original_question_id] = question_id_counts.get(original_question_id, 0) + 1
                occurrence_index = question_id_counts[original_question_id]
                
                # Create unique identifier based on occurrence index
                # If max count is 1, use original ID; otherwise append index
                max_count = question_id_max_counts.get(original_question_id, 1)
                if max_count == 1:
                    unique_question_id = original_question_id
                else:
                    unique_question_id = f"{original_question_id}_#{occurrence_index}"
                
                # Store mapping from unique_id to original_id
                if unique_question_id not in question_id_mapping:
                    question_id_mapping[unique_question_id] = original_question_id
                
                # Initialize list if this is the first time we see this unique question
                if unique_question_id not in question_scores:
                    question_scores[unique_question_id] = []
                
                question_scores[unique_question_id].append(overall_score)
    
    # print(f"Question scores: {question_scores}")
    # pdb.set_trace()
    # Calculate metrics for each question
    results = []
    
    for question_id, scores in question_scores.items():
        n = len(scores)
        
        # Calculate soft_pass_at_k for each k
        soft_pass_at_k = {}
        for k in k_values:
            if k <= n:
                soft_pass_at_k[k] = calculate_soft_pass_at_k(scores, k)
            else:
                soft_pass_at_k[k] = calculate_soft_pass_at_k(scores, n)
        
        # Calculate binary pass_at_k for each k (using threshold)
        pass_at_k = {}
        for k in k_values:
            if k <= n:
                pass_at_k[k] = calculate_threshold_pass_at_k(scores, k, threshold)
            else:
                pass_at_k[k] = calculate_threshold_pass_at_k(scores, n, threshold)
        
        # Get original question_id from mapping
        original_question_id = question_id_mapping.get(question_id, question_id)
        
        # Create result dict
        result_dict = {
            'question_id': question_id,  # Use unique_id for tracking
            'original_question_id': original_question_id,  # Store original ID
            'filename': base_filename if base_filename else 'unknown',
            'overall_scores': scores,
            'soft_pass_at_k': soft_pass_at_k,
            'pass_at_k': pass_at_k,
            'pass_at_k_threshold': threshold,
            'num_passes': n,
            'mean_score': sum(scores) / len(scores) if scores else 0.0,
            'max_score': max(scores) if scores else 0.0,
            'min_score': min(scores) if scores else 0.0
        }
        
        results.append(result_dict)
    
    return results


def build_renamed_qid_map(
    results_by_design_tech: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, List[List[str]]]:
    """
    Build design_tech -> list of [original_qid, renamed_qid] for pairs where the
    question_id was renamed (e.g. duplicate qids become qid_#1, qid_#2).

    Returns a JSON-serializable dict: { "design_tech": [ ["ori", "renamed"], ... ], ... }.
    """
    out: Dict[str, List[List[str]]] = {}
    for design_tech, result_list in results_by_design_tech.items():
        seen: set = set()
        pairs: List[List[str]] = []
        for r in result_list:
            ori = r.get("original_question_id") or r.get("question_id")
            ren = r.get("question_id")
            if ori is None or ren is None:
                continue
            if ori != ren:
                key = (ori, ren)
                if key not in seen:
                    seen.add(key)
                    pairs.append([str(ori), str(ren)])
        out[design_tech] = sorted(pairs, key=lambda p: (p[0], p[1]))
    return out


def build_renamed_qid_map_from_flat_results(
    results: List[Dict[str, Any]],
    design_tech_for_single: Optional[str] = None,
) -> Dict[str, List[List[str]]]:
    """
    Build renamed_qid map from flat results list. If results have design_name, group by it;
    otherwise use design_tech_for_single as the single key (e.g. single-directory run).
    """
    if design_tech_for_single is not None and not any(r.get("design_name") for r in results):
        results_by_design_tech = {design_tech_for_single: results}
    else:
        results_by_design_tech = {}
        for r in results:
            dt = r.get("design_name") or "unknown"
            if dt not in results_by_design_tech:
                results_by_design_tech[dt] = []
            results_by_design_tech[dt].append(r)
    return build_renamed_qid_map(results_by_design_tech)


def calculate_summary_statistics(
    results: List[Dict[str, Any]],
    k_values: List[int],
    threshold: float
) -> Dict[str, Any]:
    """
    Calculate summary statistics including average pass@k metrics across all questions.
    
    Args:
        results: List of question result dictionaries from calculate_question_pass_at_k_metrics
        k_values: List of k values used in the calculations
        threshold: Threshold used for binary pass/fail classification
    
    Returns:
        Dictionary containing:
        - total_questions: Total number of questions
        - total_passes: Total number of passes across all questions
        - overall_score_stats: Dict with mean, max, min overall scores
        - average_pass_at_k: Dict mapping k to average pass@k (binary)
        - average_soft_pass_at_k: Dict mapping k to average soft pass@k
        - threshold: Threshold used for binary classification
    
    Example:
        >>> results = [...]  # List of question results
        >>> summary = calculate_summary_statistics(results, k_values=[1, 5, 10], threshold=0.8)
        >>> print(summary['average_pass_at_k'][1])
        0.75
    """
    if not results:
        return {
            'total_questions': 0,
            'total_passes': 0,
            'overall_score_stats': {'mean': 0.0, 'max': 0.0, 'min': 0.0},
            'average_pass_at_k': {},
            'average_soft_pass_at_k': {},
            'threshold': threshold
        }
    
    # Collect all overall scores
    all_scores = [score for r in results for score in r['overall_scores']]
    
    # Calculate overall score statistics
    overall_score_stats = {
        'mean': sum(all_scores) / len(all_scores) if all_scores else 0.0,
        'max': max(all_scores) if all_scores else 0.0,
        'min': min(all_scores) if all_scores else 0.0,
        'std': 0.0  # Will calculate if needed
    }
    
    # Calculate standard deviation if we have scores
    if len(all_scores) > 1:
        mean_score = overall_score_stats['mean']
        variance = sum((score - mean_score) ** 2 for score in all_scores) / len(all_scores)
        overall_score_stats['std'] = math.sqrt(variance)
    
    # Calculate average pass@k for each k value
    average_pass_at_k = {}
    average_soft_pass_at_k = {}
    
    for k in k_values:
        # Check if this k value exists in any result
        if results and k in results[0]['pass_at_k']:
            avg_pass_at_k = sum(r['pass_at_k'].get(k, 0.0) for r in results) / len(results)
            avg_soft_pass_at_k = sum(r['soft_pass_at_k'].get(k, 0.0) for r in results) / len(results)
            average_pass_at_k[k] = avg_pass_at_k
            average_soft_pass_at_k[k] = avg_soft_pass_at_k
    
    # Calculate per-question statistics
    num_passes_list = [r['num_passes'] for r in results]
    mean_scores_list = [r['mean_score'] for r in results]
    
    summary = {
        'total_questions': len(results),
        'total_passes': len(all_scores),
        'overall_score_stats': overall_score_stats,
        'average_pass_at_k': average_pass_at_k,
        'average_soft_pass_at_k': average_soft_pass_at_k,
        'threshold': threshold,
        'per_question_stats': {
            'mean_num_passes': sum(num_passes_list) / len(num_passes_list) if num_passes_list else 0.0,
            'mean_question_score': sum(mean_scores_list) / len(mean_scores_list) if mean_scores_list else 0.0,
            'min_num_passes': min(num_passes_list) if num_passes_list else 0,
            'max_num_passes': max(num_passes_list) if num_passes_list else 0,
        }
    }
    
    return summary


# Merged categories for pass@k by category: flow execution, single stage, multi-stage
MERGED_CATEGORY_FLOW_EXECUTION = 'Flow Execution'
MERGED_CATEGORY_SINGLE_STAGE = 'Single Stage'
MERGED_CATEGORY_MULTI_STAGE = 'Multi-Stage'

# Order for outputting merged categories in by_category
MERGED_CATEGORY_ORDER = [
    MERGED_CATEGORY_FLOW_EXECUTION,
    MERGED_CATEGORY_SINGLE_STAGE,
    MERGED_CATEGORY_MULTI_STAGE,
]
# Uncategorized (in results but not in design_tech_qa_list or no category match); ensures by_category_global sums to total n
MERGED_CATEGORY_OTHER = 'Other'

# Raw categories from design_tech_qa_list.json that map to each merged category
_SINGLE_STAGE_RAW_CATEGORIES = {'synthesis', 'floorplan', 'placement', 'cts', 'routing', 'final'}
_MULTI_STAGE_RAW_PATTERN = re.compile(r'^\d+-Stages?$', re.IGNORECASE)  # 2-Stages, 3-Stages, etc.


def _raw_category_to_merged_category(raw_category: str) -> str:
    """
    Map raw category from design_tech_qa_list.json to one of three merged categories:
    Flow Execution, Single Stage (synthesis, floorplan, placement, cts, routing, final), Multi-Stage (2-Stages, 3-Stages, ...).
    """
    if not raw_category:
        return MERGED_CATEGORY_SINGLE_STAGE  # fallback
    raw = raw_category.strip()
    if raw == MERGED_CATEGORY_FLOW_EXECUTION or raw.lower() == 'flow execution':
        return MERGED_CATEGORY_FLOW_EXECUTION
    if raw in _SINGLE_STAGE_RAW_CATEGORIES:
        return MERGED_CATEGORY_SINGLE_STAGE
    if _MULTI_STAGE_RAW_PATTERN.match(raw):
        return MERGED_CATEGORY_MULTI_STAGE
    # Fallback: treat as single stage if it's a known stage name variant
    return MERGED_CATEGORY_SINGLE_STAGE


def load_design_tech_qa_list(filepath: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load design_tech -> list of {qid, stages, category} from design_tech_qa_list.json.
    Format: {"design_tech": [{"qid", "stages", "category"}, ...], ...}.
    """
    path = Path(filepath)
    if not path.exists():
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    return data


def _normalize_multifile_design_tech(design_tech: str) -> str:
    """
    Return a canonical key for multifile design_tech so that both orderings map to the same key.
    E.g. aes_sky130hd_vs_aes_sky130hs and aes_sky130hs_vs_aes_sky130hd -> aes_sky130hd_vs_aes_sky130hs.
    Single-file keys (no _vs_) are returned unchanged.
    """
    if "_vs_" not in design_tech:
        return design_tech
    parts = design_tech.split("_vs_", 1)
    if len(parts) == 2:
        return "_vs_".join(sorted(parts))
    return design_tech


def _resolve_qa_list_key(design_tech_qa_list: Dict[str, List[Dict[str, Any]]], design_tech: str) -> Optional[str]:
    """
    Resolve the key to use for design_tech in the QA list.
    qa_stats_summary may key by short name (aes_sky130hs_vs_aes_sky130hd) or by full dir name
    (qa_responses_cursor_aes_sky130hs_vs_aes_sky130hd). Try exact match, then for multifile
    try canonical form (sorted _vs_ parts), then key ending with _<design_tech>.
    """
    if design_tech in design_tech_qa_list:
        return design_tech
    # Multifile: match by canonical form so aes_sky130hd_vs_aes_sky130hs resolves to aes_sky130hs_vs_aes_sky130hd if that's in the list
    if "_vs_" in design_tech:
        canonical = _normalize_multifile_design_tech(design_tech)
        for k in design_tech_qa_list:
            if "_vs_" in k and _normalize_multifile_design_tech(k) == canonical:
                return k
    suffix = '_' + design_tech
    for k in design_tech_qa_list:
        if k.endswith(suffix) or k == design_tech:
            return k
    return None


def _build_raw_category_to_qids(
    design_tech_qa_list: Dict[str, List[Dict[str, Any]]],
    design_tech: str,
) -> Tuple[Dict[str, Set[str]], Set[str]]:
    """
    Build raw (original) category -> set(qid) and combined set of all qids for design_tech.
    No merging; categories are exactly as in design_tech_qa_list.json (e.g. cts, 3-Stages, Flow Execution).
    """
    by_category: Dict[str, Set[str]] = {}
    all_qids: Set[str] = set()
    resolved_key = _resolve_qa_list_key(design_tech_qa_list, design_tech)
    entries = design_tech_qa_list.get(resolved_key, []) if resolved_key else []
    if not entries:
        return by_category, all_qids
    for item in entries:
        qid = item.get('qid') or item.get('id') or ''
        raw_cat = item.get('category', '')
        if not raw_cat:
            continue
        all_qids.add(str(qid))
        if raw_cat not in by_category:
            by_category[raw_cat] = set()
        by_category[raw_cat].add(str(qid))
    return by_category, all_qids


def _build_category_to_qids(
    design_tech_qa_list: Dict[str, List[Dict[str, Any]]],
    design_tech: str,
) -> Tuple[Dict[str, Set[str]], Set[str]]:
    """
    Build merged category -> set(qid) and combined set of all qids for design_tech.
    Raw categories from the QA list are merged into three: Flow Execution, Single Stage, Multi-Stage.
    """
    by_merged_category: Dict[str, Set[str]] = {}
    all_qids: Set[str] = set()
    resolved_key = _resolve_qa_list_key(design_tech_qa_list, design_tech)
    entries = design_tech_qa_list.get(resolved_key, []) if resolved_key else []
    if not entries:
        return by_merged_category, all_qids
    for item in entries:
        qid = item.get('qid') or item.get('id') or ''
        raw_cat = item.get('category', '')
        if not raw_cat:
            continue
        all_qids.add(str(qid))
        merged_cat = _raw_category_to_merged_category(raw_cat)
        if merged_cat not in by_merged_category:
            by_merged_category[merged_cat] = set()
        by_merged_category[merged_cat].add(str(qid))
    return by_merged_category, all_qids


def report_questions_not_in_any_category(
    results_by_design_tech: Dict[str, List[Dict[str, Any]]],
    design_tech_qa_list_path: str,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Find and report questions that appear in evaluation results but are not in
    design_tech_qa_list.json for their design_tech (i.e. not in any category).

    Returns a list of {"design_tech": str, "qid": str, "question_id": str} for each
    missing question (question_id is the unique id used in results, may differ from qid).
    """
    qa_list = load_design_tech_qa_list(design_tech_qa_list_path)
    missing: List[Dict[str, Any]] = []
    for design_tech, results in results_by_design_tech.items():
        _, combined_qids = _build_category_to_qids(qa_list, design_tech)
        if not combined_qids:
            # No QA list entry for this design_tech: all result qids are "missing" from categories
            combined_qids = set()
        result_qids = set()
        result_qid_to_question_id: Dict[str, str] = {}
        for r in results:
            qid = str(r.get("original_question_id") or r.get("question_id") or "")
            question_id = str(r.get("question_id") or "")
            result_qids.add(qid)
            result_qid_to_question_id[qid] = question_id
        not_in_list = result_qids - combined_qids
        for qid in sorted(not_in_list):
            missing.append({
                "design_tech": design_tech,
                "qid": qid,
                "question_id": result_qid_to_question_id.get(qid, qid),
            })
    if verbose and missing:
        print()
        print("=" * 80)
        print("QUESTIONS NOT IN ANY CATEGORY (in results but not in design_tech_qa_list)")
        print("=" * 80)
        print(f"Total such questions: {len(missing)}")
        print()
        for i, m in enumerate(missing, 1):
            print(f"  [{i}] design_tech={m['design_tech']}  qid={m['qid']}  question_id={m['question_id']}")
        print()
    elif verbose and not missing:
        print()
        print("All questions in the run are present in design_tech_qa_list (no uncategorized questions).")
        print()
    return missing

def _search_qa_list_by_qid(design_tech_qa_list: List[Dict[str, Any]], qid: str) -> Dict[str, Any]:
    for entry in design_tech_qa_list:
        if entry.get('qid') == qid:
            return entry
    return None

def calculate_pass_at_k_by_category(
    results: List[Dict[str, Any]],
    design_tech_qa_list_path: str,
    design_tech: str,
    k_values: List[int],
    threshold: float = 0.8,
    design_tech_qa_list: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    preserve_original_categories: bool = False,
    dump_counted_qa_path: Optional[str] = "./results/",
) -> Dict[str, Any]:
    """
    Calculate pass@k (and soft pass@k) for each merged category and for combined, for one design_tech.
    Uses qa_stats_results/design_tech_qa_list.json to map question IDs to categories, then merges into
    three categories: Flow Execution, Single Stage (synthesis, floorplan, placement, cts, routing, final),
    Multi-Stage (2-Stages, 3-Stages, 4-Stages, 5-Stages).
    results: list of per-question dicts from calculate_question_pass_at_k_metrics.
    design_tech_qa_list: optional pre-loaded dict from load_design_tech_qa_list(); if None, loads from path.
    preserve_original_categories: if True, also include by_category_original with raw categories (e.g. cts, 3-Stages).
    dump_counted_qa_path: if set, dump all counted questions by category (with original stage names) to this JSON file.
    Returns:
        {
          "design_tech": str,
          "by_category": { category: {...} }  # merged: Flow Execution, Single Stage, Multi-Stage
          "by_category_original": { category: {...} }  # only if preserve_original_categories
          "combined": { "num_questions", "average_pass_at_k", "average_soft_pass_at_k" }
        }
    """
    out: Dict[str, Any] = {
        'design_tech': design_tech,
        'by_category': {},
        'combined': None,
    }
    used_qids = []
    if preserve_original_categories:
        out['by_category_original'] = {}
    qa_list = design_tech_qa_list if design_tech_qa_list is not None else load_design_tech_qa_list(design_tech_qa_list_path)
    by_category_qids: Dict[str, Set[str]] = {}
    combined_qids: Set[str] = set()
    by_raw_category_qids: Dict[str, Set[str]] = {}
    if qa_list:
        by_category_qids, combined_qids = _build_category_to_qids(qa_list, design_tech)
        if preserve_original_categories:
            by_raw_category_qids, _ = _build_raw_category_to_qids(qa_list, design_tech)
    if not combined_qids:
        # No mapping for this design_tech: use all qids from results as "combined" only.
        # Use question_id only (not original_question_id), as it may be renamed for duplicate qids.
        combined_qids = {str(r.get('question_id') or '') for r in results}

    # Build question_id -> binary results and -> scores from results (use original_question_id to match qid)
    results_by_qid_binary: Dict[str, List[bool]] = {}
    results_by_qid_scores: Dict[str, List[float]] = {}
    for r in results:
        qid = str(r.get('question_id') or '')
        scores = r.get('overall_scores') or []
        results_by_qid_binary[qid] = [s >= threshold for s in scores]
        results_by_qid_scores[qid] = list(scores)
        # record used question (base qid for matching to design_tech_qa_list)
        # print(by_raw_category_qids.get(qid, {}))
        if design_tech == "aes_sky130hd_vs_aes_sky130hs":
            used_qids.append(_search_qa_list_by_qid(qa_list["aes_sky130hs_vs_aes_sky130hd"], qid))
        else:
            used_qids.append(_search_qa_list_by_qid(qa_list[design_tech], qid))
        # pdb.set_trace()

    # If QA list had entries but zero qids overlap results (e.g. different qid format), use all result qids for combined
    overlap = sum(1 for q in combined_qids if q in results_by_qid_binary)
    if overlap == 0 and results_by_qid_binary and combined_qids:
        combined_qids = set(results_by_qid_binary.keys())

    def _metrics_for_qid_set(qid_set: Set[str]) -> Dict[str, Any]:
        subset_binary = {qid: results_by_qid_binary[qid] for qid in qid_set if qid in results_by_qid_binary}
        subset_scores = {qid: results_by_qid_scores[qid] for qid in qid_set if qid in results_by_qid_scores}
        n = len(subset_binary)
        if n == 0:
            return {'num_questions': 0, 'average_pass_at_k': {}, 'average_soft_pass_at_k': {}}
        avg_pass_at_k = {}
        avg_soft_pass_at_k = {}
        for k in k_values:
            avg_p, _ = calculate_pass_at_k_batch(subset_binary, k)
            avg_pass_at_k[k] = avg_p
            soft_scores = [calculate_soft_pass_at_k(subset_scores[qid], min(k, len(subset_scores[qid]))) for qid in subset_scores]
            avg_soft_pass_at_k[k] = sum(soft_scores) / len(soft_scores) if soft_scores else 0.0
        return {
            'num_questions': n,
            'average_pass_at_k': avg_pass_at_k,
            'average_soft_pass_at_k': avg_soft_pass_at_k,
        }

    for cat in MERGED_CATEGORY_ORDER:
        if cat in by_category_qids:
            out['by_category'][cat] = _metrics_for_qid_set(by_category_qids[cat])
    if preserve_original_categories and by_raw_category_qids:
        for cat in sorted(by_raw_category_qids.keys()):
            out['by_category_original'][cat] = _metrics_for_qid_set(by_raw_category_qids[cat])
    out['combined'] = _metrics_for_qid_set(combined_qids)

    if dump_counted_qa_path:
        dump_counted_qa_path = os.path.join(dump_counted_qa_path, f'{design_tech}_counted_questions_by_category.json')
        with open(dump_counted_qa_path, 'w', encoding='utf-8') as f:
            json.dump(used_qids, f, indent=2)
    out['used_qids'] = used_qids
    return out


def calculate_pass_at_k_by_category_all(
    design_tech_qa_list_path: str,
    results_by_design_tech: Dict[str, List[Dict[str, Any]]],
    k_values: List[int] = [1, 5, 10],
    threshold: float = 0.8,
    preserve_original_categories: bool = False,
    dump_counted_qa_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Calculate pass@k by category (and combined) for each design_tech, plus an overall combined
    and Flow Execution / Single Stage / Multi-Stage aggregated across all design_techs.
    results_by_design_tech: design_tech -> list of per-question results from calculate_question_pass_at_k_metrics.
    preserve_original_categories: if True, each design_tech result also includes by_category_original.
    dump_counted_qa_path: if set, dump per-design_tech counted QAs by category and a combined file (all technologies).
    Returns:
        {
          "by_design_tech": { design_tech: { "by_category": {...}, "by_category_original": {...}?, "combined": {...} } },
          "combined": { "num_questions", "average_pass_at_k", "average_soft_pass_at_k" },  # all design_techs
          "by_category_global": { "Flow Execution": {...}, "Single Stage": {...}, "Multi-Stage": {...} }  # across all design_techs
        }
    """
    qa_list = load_design_tech_qa_list(design_tech_qa_list_path)
    by_design_tech: Dict[str, Any] = {}
    all_results: List[Dict[str, Any]] = []
    combined_used_qids = {}
    for design_tech, results in results_by_design_tech.items():
        by_design_tech[design_tech] = calculate_pass_at_k_by_category(
            results, design_tech_qa_list_path, design_tech, k_values, threshold,
            design_tech_qa_list=qa_list,
            preserve_original_categories=preserve_original_categories,
            dump_counted_qa_path=dump_counted_qa_path,
        )
        all_results.extend(results)
        combined_used_qids[design_tech] = by_design_tech[design_tech].get('used_qids', [])

    # dump combined used qids
    if dump_counted_qa_path:
        combined_used_qids_path = os.path.join(dump_counted_qa_path, 'combined_used_qids_by_category.json')
        with open(combined_used_qids_path, 'w', encoding='utf-8') as f:
            json.dump(combined_used_qids, f, indent=2)

    # Global combined: pass@k over all questions across all design_techs.
    # With normalized multifile design_tech, each result appears in exactly one design_tech, so (design_tech, qid) is unique and num_questions = total questions (e.g. 1138 after excluding ignored).
    combined_metrics = None
    by_category_global: Dict[str, Dict[str, Any]] = {}
    if results_by_design_tech:
        results_by_key_binary: Dict[str, List[bool]] = {}
        results_by_key_scores: Dict[str, List[float]] = {}
        for design_tech, results in results_by_design_tech.items():
            for r in results:
                qid = str(r.get('question_id') or '')
                key = f"{design_tech}|{qid}"
                scores = r.get('overall_scores') or []
                results_by_key_binary[key] = [s >= threshold for s in scores]
                results_by_key_scores[key] = list(scores)
        n = len(results_by_key_binary)
        avg_pass_at_k = {}
        avg_soft_pass_at_k = {}
        for k in k_values:
            avg_p, _ = calculate_pass_at_k_batch(results_by_key_binary, k)
            avg_pass_at_k[k] = avg_p
            soft_list = [
                calculate_soft_pass_at_k(results_by_key_scores[key], min(k, len(results_by_key_scores[key])))
                for key in results_by_key_scores
            ]
            avg_soft_pass_at_k[k] = sum(soft_list) / len(soft_list) if soft_list else 0.0
        combined_metrics = {
            'num_questions': n,
            'average_pass_at_k': avg_pass_at_k,
            'average_soft_pass_at_k': avg_soft_pass_at_k,
        }

        # Flow Execution, Single Stage, Multi-Stage across all design_techs
        global_by_category: Dict[str, Set[str]] = defaultdict(set)
        for design_tech, results in results_by_design_tech.items():
            by_category_qids, _ = _build_category_to_qids(qa_list, design_tech)
            qid_to_cat: Dict[str, str] = {}
            for cat, qid_set in by_category_qids.items():
                for q in qid_set:
                    qid_to_cat[str(q)] = cat
            for r in results:
                key = f"{design_tech}|{str(r.get('question_id') or '')}"
                qid = str(r.get('question_id') or '')
                cat = qid_to_cat.get(qid)
                if cat is not None:
                    global_by_category[cat].add(key)
        for cat in MERGED_CATEGORY_ORDER:
            key_set = global_by_category.get(cat) or set()
            subset_binary = {k: results_by_key_binary[k] for k in key_set if k in results_by_key_binary}
            subset_scores = {k: results_by_key_scores[k] for k in key_set if k in results_by_key_scores}
            n_cat = len(subset_binary)
            if n_cat == 0:
                by_category_global[cat] = {'num_questions': 0, 'average_pass_at_k': {}, 'average_soft_pass_at_k': {}}
            else:
                avg_pk = {}
                avg_soft = {}
                for k in k_values:
                    avg_p, _ = calculate_pass_at_k_batch(subset_binary, k)
                    avg_pk[k] = avg_p
                    soft_list = [
                        calculate_soft_pass_at_k(subset_scores[key], min(k, len(subset_scores[key])))
                        for key in subset_scores
                    ]
                    avg_soft[k] = sum(soft_list) / len(soft_list) if soft_list else 0.0
                by_category_global[cat] = {
                    'num_questions': n_cat,
                    'average_pass_at_k': avg_pk,
                    'average_soft_pass_at_k': avg_soft,
                }

        # Other: keys in this run not in any merged category (so by_category_global sums to total n)
        assigned = set()
        for cat in MERGED_CATEGORY_ORDER:
            assigned |= (global_by_category.get(cat) or set())
        other_keys = set(results_by_key_binary.keys()) - assigned
        if other_keys:
            subset_binary = {k: results_by_key_binary[k] for k in other_keys}
            subset_scores = {k: results_by_key_scores[k] for k in other_keys}
            n_other = len(subset_binary)
            avg_pk = {}
            avg_soft = {}
            for k in k_values:
                avg_p, _ = calculate_pass_at_k_batch(subset_binary, k)
                avg_pk[k] = avg_p
                soft_list = [
                    calculate_soft_pass_at_k(subset_scores[key], min(k, len(subset_scores[key])))
                    for key in subset_scores
                ]
                avg_soft[k] = sum(soft_list) / len(soft_list) if soft_list else 0.0
            by_category_global[MERGED_CATEGORY_OTHER] = {
                'num_questions': n_other,
                'average_pass_at_k': avg_pk,
                'average_soft_pass_at_k': avg_soft,
            }
        else:
            by_category_global[MERGED_CATEGORY_OTHER] = {'num_questions': 0, 'average_pass_at_k': {}, 'average_soft_pass_at_k': {}}

    return {
        'by_design_tech': by_design_tech,
        'combined': combined_metrics,
        'by_category_global': by_category_global,
    }


# CSV columns: (display_name, source, key). source in ('combined', 'by_category', 'by_category_original').
# key is the key in by_category / by_category_original (e.g. 'Flow Execution', '2-Stages').
_CSV_COLUMN_SPEC = [
    ('Combined', 'combined', None),
    ('Flow', 'by_category', 'Flow Execution'),
    ('Single', 'by_category', 'Single Stage'),
    ('Multi', 'by_category', 'Multi-Stage'),
    ('Flow Exe.', 'by_category_original', 'Flow Execution'),
    ('floorplan', 'by_category_original', 'floorplan'),
    ('placement', 'by_category_original', 'placement'),
    ('cts', 'by_category_original', 'cts'),
    ('routing', 'by_category_original', 'routing'),
    ('final', 'by_category_original', 'final'),
    ('2-stage', 'by_category_original', '2-Stages'),
    ('3-stage', 'by_category_original', '3-Stages'),
    ('4-stage', 'by_category_original', '4-Stages'),
    ('5-stage', 'by_category_original', '5-Stages'),
]


def _get_category_metrics(
    dt_data: Dict[str, Any],
    source: str,
    key: Optional[str],
    metric: str,
) -> Dict[int, float]:
    """Get pass@k or soft_pass_at_k dict for one category from per-design_tech data. Missing => 0."""
    if source == 'combined':
        return (dt_data.get('combined') or {}).get(metric) or {}
    if source == 'by_category' and key:
        return ((dt_data.get('by_category') or {}).get(key) or {}).get(metric) or {}
    if source == 'by_category_original' and key:
        return ((dt_data.get('by_category_original') or {}).get(key) or {}).get(metric) or {}
    return {}


def _get_global_category_metrics(
    combined_global: Optional[Dict[str, Any]],
    by_cat_global: Dict[str, Any],
    display_name: str,
    source: str,
    key: Optional[str],
    metric: str,
) -> Dict[int, float]:
    """Get pass@k or soft_pass_at_k dict for one category for the global row. Original categories => 0."""
    if source == 'combined' and combined_global:
        return (combined_global.get(metric) or {})
    if source == 'by_category' and key:
        return (by_cat_global.get(key) or {}).get(metric) or {}
    # by_category_original has no global aggregation
    return {}


def _write_pass_at_k_by_design_tech_csv(
    path: Path,
    pass_at_k_by_category_result: Dict[str, Any],
    k_values: List[int],
) -> None:
    """
    Write a CSV with rows = design_tech, columns = pass@k (pass1, pass3, pass5, ...) for each category.
    Categories: Combined, Flow, Single, Multi, Flow Exe., floorplan, placement, cts, routing, final, 2-stage..5-stage.
    Missing category data is written as 0.
    """
    by_dt = pass_at_k_by_category_result.get('by_design_tech') or {}
    combined_global = pass_at_k_by_category_result.get('combined')
    by_cat_global = pass_at_k_by_category_result.get('by_category_global') or {}
    header = ['design_tech']
    for display_name, _src, _key in _CSV_COLUMN_SPEC:
        for k in k_values:
            header.append(f"{display_name}_pass{k}")
    rows = []
    for design_tech in sorted(by_dt.keys()):
        dt_data = by_dt[design_tech]
        row = [design_tech]
        for display_name, source, key in _CSV_COLUMN_SPEC:
            data = _get_category_metrics(dt_data, source, key, 'average_pass_at_k')
            for k in k_values:
                v = data.get(k)
                if v is None:
                    v = 0
                row.append(round(v, 4) if isinstance(v, (int, float)) else v)
        rows.append(row)
    if len(by_dt) > 1 and combined_global and combined_global.get('num_questions', 0) > 0:
        row = ['all_design_techs']
        for display_name, source, key in _CSV_COLUMN_SPEC:
            data = _get_global_category_metrics(
                combined_global, by_cat_global, display_name, source, key, 'average_pass_at_k'
            )
            for k in k_values:
                v = data.get(k)
                if v is None:
                    v = 0
                row.append(round(v, 4) if isinstance(v, (int, float)) else v)
        rows.append(row)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _write_soft_pass_at_k_by_design_tech_csv(
    path: Path,
    pass_at_k_by_category_result: Dict[str, Any],
    k_values: List[int],
) -> None:
    """
    Write a CSV with rows = design_tech, columns = soft pass@k (soft1, soft3, soft5, ...) for each category.
    Same categories as pass@k table. Missing => 0.
    """
    by_dt = pass_at_k_by_category_result.get('by_design_tech') or {}
    combined_global = pass_at_k_by_category_result.get('combined')
    by_cat_global = pass_at_k_by_category_result.get('by_category_global') or {}
    header = ['design_tech']
    for display_name, _src, _key in _CSV_COLUMN_SPEC:
        for k in k_values:
            header.append(f"{display_name}_soft{k}")
    rows = []
    for design_tech in sorted(by_dt.keys()):
        dt_data = by_dt[design_tech]
        row = [design_tech]
        for display_name, source, key in _CSV_COLUMN_SPEC:
            data = _get_category_metrics(dt_data, source, key, 'average_soft_pass_at_k')
            for k in k_values:
                v = data.get(k)
                if v is None:
                    v = 0
                row.append(round(v, 4) if isinstance(v, (int, float)) else v)
        rows.append(row)
    if len(by_dt) > 1 and combined_global and combined_global.get('num_questions', 0) > 0:
        row = ['all_design_techs']
        for display_name, source, key in _CSV_COLUMN_SPEC:
            data = _get_global_category_metrics(
                combined_global, by_cat_global, display_name, source, key, 'average_soft_pass_at_k'
            )
            for k in k_values:
                v = data.get(k)
                if v is None:
                    v = 0
                row.append(round(v, 4) if isinstance(v, (int, float)) else v)
        rows.append(row)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def verify_pass_at_k_by_category_mapping(
    design_tech_qa_list_path: str,
    results_by_design_tech: Dict[str, List[Dict[str, Any]]],
    sample_size: int = 5,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Verify why some design_techs have no pass@k-by-category metrics (e.g. multifile or combined).
    Reports for each design_tech: n_results, n_qa_list_qids, n_overlap, and sample qids when
    overlap is 0 so you can spot key vs. qid format mismatches.
    """
    qa_list = load_design_tech_qa_list(design_tech_qa_list_path)
    qa_list_keys = set(qa_list.keys()) if qa_list else set()
    result_keys = set(results_by_design_tech.keys())
    all_design_techs = sorted(qa_list_keys | result_keys)

    report: Dict[str, Any] = {
        'design_tech_qa_list_path': design_tech_qa_list_path,
        'qa_list_design_techs': list(qa_list_keys),
        'results_design_techs': list(result_keys),
        'per_design_tech': {},
        'summary': {},
    }
    only_in_qa_list = qa_list_keys - result_keys
    only_in_results = result_keys - qa_list_keys

    for dt in all_design_techs:
        results = results_by_design_tech.get(dt, [])
        entries = qa_list.get(dt, [])
        result_qids = {str(r.get('original_question_id') or r.get('question_id') or '') for r in results}
        qa_qids = {str(item.get('qid') or item.get('id') or '') for item in entries}
        overlap = result_qids & qa_qids
        n_results = len(results)
        n_qa = len(qa_qids)
        n_overlap = len(overlap)
        in_qa_not_results = qa_qids - result_qids
        in_results_not_qa = result_qids - qa_qids

        per = {
            'n_results': n_results,
            'n_qa_list_qids': n_qa,
            'n_overlap': n_overlap,
            'in_qa_list_only': len(in_qa_not_results),
            'in_results_only': len(in_results_not_qa),
        }
        if n_overlap == 0 and (n_results > 0 or n_qa > 0):
            per['sample_qa_list_qids'] = list(in_qa_not_results)[:sample_size] if in_qa_not_results else list(qa_qids)[:sample_size]
            per['sample_result_qids'] = list(in_results_not_qa)[:sample_size] if in_results_not_qa else list(result_qids)[:sample_size]
        report['per_design_tech'][dt] = per

    report['summary'] = {
        'only_in_qa_list': list(only_in_qa_list),
        'only_in_results': list(only_in_results),
        'design_techs_with_zero_overlap': [dt for dt in all_design_techs if report['per_design_tech'][dt]['n_overlap'] == 0 and (report['per_design_tech'][dt]['n_results'] > 0 or report['per_design_tech'][dt]['n_qa_list_qids'] > 0)],
    }

    if verbose:
        _print_verify_pass_at_k_by_category(report, sample_size)

    return report


def _print_verify_pass_at_k_by_category(report: Dict[str, Any], sample_size: int) -> None:
    """Print human-readable verification report for pass@k by category mapping."""
    print("=" * 80)
    print("VERIFY PASS@K BY CATEGORY MAPPING")
    print("=" * 80)
    print(f"QA list path: {report['design_tech_qa_list_path']}")
    print(f"Design_techs in QA list: {len(report['qa_list_design_techs'])}")
    print(f"Design_techs in results: {len(report['results_design_techs'])}")
    only_qa = report['summary']['only_in_qa_list']
    only_res = report['summary']['only_in_results']
    zero_overlap = report['summary']['design_techs_with_zero_overlap']
    if only_qa:
        print(f"\n⚠️  In QA list but have 0 results (missing from run or design_name mismatch): {only_qa}")
    if only_res:
        print(f"\n⚠️  In results but not in QA list: {only_res}")
    if zero_overlap:
        print(f"\n⚠️  Design_techs with results and QA list entries but 0 overlapping qids: {zero_overlap}")
    print("\nPer design_tech:")
    for dt in sorted(report['per_design_tech'].keys()):
        per = report['per_design_tech'][dt]
        n_res = per['n_results']
        n_qa = per['n_qa_list_qids']
        n_ov = per['n_overlap']
        line = f"  {dt}: n_results={n_res}  n_qa_list_qids={n_qa}  n_overlap={n_ov}"
        if n_ov == 0 and (n_res > 0 or n_qa > 0):
            line += "  <- NO OVERLAP (qid mismatch?)"
        print(line)
        if 'sample_qa_list_qids' in per and per['sample_qa_list_qids']:
            print(f"      sample QA list qids: {per['sample_qa_list_qids']}")
        if 'sample_result_qids' in per and per['sample_result_qids']:
            print(f"      sample result qids:  {per['sample_result_qids']}")
    print("=" * 80)


def verify_pass_counts(
    results: List[Dict[str, Any]],
    expected_passes: int = 10
) -> Dict[str, Any]:
    """
    Verify that each question has the expected number of passes.
    
    Args:
        results: List of question result dictionaries from calculate_question_pass_at_k_metrics
        expected_passes: Expected number of passes per question (default: 10)
    
    Returns:
        Dictionary containing:
        - valid_questions: List of questions with correct pass count
        - invalid_questions: List of questions with incorrect pass count
        - statistics: Summary statistics about pass counts
        - pass_count_distribution: Dictionary mapping pass counts to question IDs
    
    Example:
        >>> results = [...]  # List of question results
        >>> verification = verify_pass_counts(results, expected_passes=10)
        >>> print(f"Found {len(verification['invalid_questions'])} questions with incorrect pass counts")
    """
    valid_questions = []
    invalid_questions = []
    pass_count_distribution = {}
    
    for result in results:
        question_id = result['question_id']
        num_passes = result['num_passes']
        
        # Track distribution
        if num_passes not in pass_count_distribution:
            pass_count_distribution[num_passes] = []
        pass_count_distribution[num_passes].append(question_id)
        
        # Check if valid
        if num_passes == expected_passes:
            valid_questions.append({
                'question_id': question_id,
                'num_passes': num_passes
            })
        else:
            invalid_questions.append({
                'question_id': question_id,
                'num_passes': num_passes,
                'expected_passes': expected_passes,
                'difference': num_passes - expected_passes,
                'overall_scores': result['overall_scores'],
                'source_directory': result.get('source_directory', 'unknown'),
                'design_name': result.get('design_name', 'unknown')
            })
    
    # Calculate statistics
    all_pass_counts = [r['num_passes'] for r in results]
    statistics = {
        'total_questions': len(results),
        'valid_questions': len(valid_questions),
        'invalid_questions': len(invalid_questions),
        'expected_passes': expected_passes,
        'min_passes': min(all_pass_counts) if all_pass_counts else 0,
        'max_passes': max(all_pass_counts) if all_pass_counts else 0,
        'mean_passes': sum(all_pass_counts) / len(all_pass_counts) if all_pass_counts else 0.0,
        'pass_count_distribution': {str(k): len(v) for k, v in sorted(pass_count_distribution.items())}
    }
    
    return {
        'valid_questions': valid_questions,
        'invalid_questions': invalid_questions,
        'statistics': statistics,
        'pass_count_distribution': pass_count_distribution
    }


def extract_design_name_from_filename(filename: str, model_name: str) -> Optional[str]:
    """
    Extract design name from evaluation filename.
    
    Filename pattern: pass_at_k_{design_name}_{model_name}_pass{num}of10_...
    or: pass_at_k_{design_name}_vs_{design_name2}_{model_name}_pass{num}of10_...
    
    Returns the full design part so that multifile comparisons (e.g. aes_asap7_vs_aes_nangate45)
    match design_tech keys in design_tech_qa_list.json and are included in by_category pass@k.
    
    Args:
        filename: Evaluation JSON filename
        model_name: Model name used to identify where to split
    
    Returns:
        Design name (e.g., "aes_asap7" or "aes_asap7_vs_aes_nangate45") or None if not found
    """
    # Pattern: pass_at_k_{design_name}_{model_name}_pass{num}of10_...
    # Extract the part between "pass_at_k_" and "_{model_name}_" (full string, including _vs_ for comparisons)
    pattern = re.compile(r'pass_at_k_(.+?)_' + re.escape(model_name) + r'_pass\d+of10')
    match = pattern.match(filename)
    if match:
        return match.group(1)
    return None


def load_and_calculate_pass_at_k(
    directory: str,
    model_name: str,
    k_values: List[int] = [1, 5, 10],
    threshold: float = 0.8,
    ignored_qa_pairs_path: Optional[str] = None,
    source_qa_file_path: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Convenience function to load JSON files and calculate pass@k metrics.

    Optionally excludes (qid, question) pairs listed in pass_k_ignored_qa_pairs.json.
    design_technology is auto-resolved from directory name (qa_output_<agent>_<design>_<technology>).
    When source_qa_file_path is set, filtering uses (qid, question) for precise exclusion.

    Args:
        directory: Directory path containing the evaluation JSON files
        model_name: Model name pattern to match in filenames (e.g., "sonnet4p5", "gpt-5.2")
        k_values: List of k values to calculate pass@k for (default: [1, 5, 10])
        threshold: Threshold for binary pass/fail classification (default: 0.8)
        ignored_qa_pairs_path: Optional path to pass_k_ignored_qa_pairs.json
        source_qa_file_path: Optional path to the source QA JSON file. Enables (qid, question) matching
                           when the QA file is not inside the evaluation directory.

    Returns:
        Tuple of (list of dictionaries with pass@k metrics for each question, design_name)
        design_name is extracted from the filename or None if not found

    Example:
        >>> results, design_name = load_and_calculate_pass_at_k(
        ...     "./qa_output_cursor_aes_asap7",
        ...     "sonnet4p5",
        ...     ignored_qa_pairs_path="pass_k_ignored_qa_pairs.json",
        ...     source_qa_file_path="qa_output/aes_asap7/aes_20260201_154729_qa_all_types.json",
        ... )
        >>> print(results[0]['question_id'])
    """
    ignored_qids: Set[str] = set()
    ignored_pairs: Optional[Set[Tuple[str, str]]] = None
    qa_pairs_id_question: Optional[List[Tuple[str, str]]] = None

    if ignored_qa_pairs_path:
        by_source_pairs, all_ignored_qids, _, _ = load_pass_k_ignored_qa_pairs(ignored_qa_pairs_path)
        # Build keys for resolution: include JSON keys plus current directory's design part when multifile,
        # so pass_k_ignored_qa_pairs.json need not list every multifile key for resolution to succeed.
        design_tech_keys = list(by_source_pairs.keys()) if by_source_pairs else []
        dir_name = Path(directory).name
        if "_vs_" in dir_name:
            for prefix in _DESIGN_TECH_DIR_PREFIXES:
                if dir_name.startswith(prefix):
                    multifile_part = dir_name[len(prefix):]
                    if multifile_part and multifile_part not in design_tech_keys:
                        design_tech_keys = design_tech_keys + [multifile_part]
                    break
        design_tech = resolve_design_tech_from_directory(directory, design_tech_keys) if design_tech_keys else None
        print(f"design_tech: {design_tech}")
        if design_tech is not None:
            pairs_this_source = by_source_pairs.get(design_tech, set())
            if pairs_this_source:
                ignored_pairs = pairs_this_source
                print(f"Excluding by (qid, question) for {design_tech} ({len(ignored_pairs)} pair(s))")
                # Print ignored QA pairs (qid, question preview), up to 15
                for i, (qid, qtext) in enumerate(sorted(ignored_pairs)[:15]):
                    preview = (qtext or "")[:60].replace("\n", " ")
                    print(f"  ignored_qa_pair[{i+1}]: qid={qid}  question={preview}...")
                if len(ignored_pairs) > 15:
                    print(f"  ... and {len(ignored_pairs) - 15} more ignored pair(s)")
                # Optional: load ordered (id, question) for index-based match when result has no question text
                qa_file_path = source_qa_file_path or find_qa_json_in_directory(directory)
                if qa_file_path:
                    qa_pairs_id_question = load_qa_pairs_id_question(qa_file_path)
            # else: pairs_this_source empty = no ignored QA pairs for this design_tech; leave ignored_qids empty
        else:
            # design_tech is None: use union of all qids from all sources; (qid, question) exclusion not possible.
            ignored_qids = all_ignored_qids
            if ignored_qids:
                print(f"Excluding {len(ignored_qids)} question ID(s) from pass_k_ignored_qa_pairs (all sources)")
                print(f"  ignored_qids: {sorted(ignored_qids)[:20]}{' ...' if len(ignored_qids) > 20 else ''}")
    # pdb.set_trace()
    pass_files = load_pass_json_files(
        directory,
        model_name,
        ignored_qids=ignored_qids if not ignored_pairs else set(),
        ignored_pairs=ignored_pairs,
        qa_pairs_id_question=qa_pairs_id_question,
    )

    # Require exactly 10 evaluation files (pass 1..10); raise flag if missing or incomplete
    valid_count, missing_passes = verify_evaluation_file_count(
        pass_files, directory, model_name, required_passes=10
    )
    if not pass_files:
        print(f"Warning: No matching pass files found in {directory}")
        exit(1)
        return [], None
    if not valid_count:
        # Flag already printed by verify_evaluation_file_count; continue with partial data if desired
        print(f"Warning: Missing or incomplete pass files in {directory}")
        print(f"  Missing passes: {missing_passes}")
        print(f"  Valid passes: {valid_count}")
        exit(1)
    
    # Extract design name from the first file
    design_name = None
    if pass_files:
        first_file_info = pass_files[min(pass_files.keys())]
        design_name = extract_design_name_from_filename(first_file_info['filename'], model_name)
    
    results = calculate_question_pass_at_k_metrics(pass_files, k_values, threshold)
    return results, design_name


def calculate_average_pass_at_k_multiple_directories(
    directories: List[str],
    model_name: str,
    k_values: List[int] = [1, 5, 10],
    threshold: float = 0.8,
    ignored_qa_pairs_path: Optional[str] = None,
    source_qa_file_path_by_directory: Optional[Dict[str, str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """
    Calculate average pass@k metrics across multiple directories (combined pass@k).

    For each directory, design_technology is auto-resolved from the directory name
    (qa_output_<agent>_<design>_<technology>). Optionally excludes (qid, question) pairs
    from pass_k_ignored_qa_pairs.json when source_qa_file_path_by_directory is provided.

    Args:
        directories: List of directory paths containing evaluation JSON files
        model_name: Model name pattern to match in filenames (e.g., "sonnet4p5", "gpt-5.2")
        k_values: List of k values to calculate pass@k for (default: [1, 5, 10])
        threshold: Threshold for binary pass/fail classification (default: 0.8)
        ignored_qa_pairs_path: Optional path to pass_k_ignored_qa_pairs.json
        source_qa_file_path_by_directory: Optional dict mapping directory path -> path to source QA JSON
                                         (enables (qid, question) filtering when provided).
                                         Example: {"./qa_output_cursor_aes_asap7": "qa_output/aes_asap7/aes_20260201_154729_qa_all_types.json",
                                         "./qa_output_cursor_gcd": "qa_output/gcdsky130/gcd_20260205_152030_qa_all_types.json"}

    Returns:
        Tuple of:
        - combined_results: List of all question results from all directories (after per-dir filtering)
        - combined_summary: Overall summary statistics across all directories
        - per_directory_summaries: Dictionary mapping directory to its individual summary

    Example:
        >>> dirs = ["./qa_output_cursor_aes_asap7/", "./qa_output_cursor_aes_nangate45/"]
        >>> results, combined_summary, per_dir = calculate_average_pass_at_k_multiple_directories(
        ...     dirs, "sonnet4p5", ignored_qa_pairs_path="pass_k_ignored_qa_pairs.json",
        ...     source_qa_file_path_by_directory=path_map,
        ... )
        >>> print(f"Total questions: {combined_summary['total_questions']}")
    """
    all_results = []
    per_directory_summaries = {}
    design_names = []
    dir_to_path = source_qa_file_path_by_directory or {}

    # Process each directory
    for directory in directories:
        print(f"Processing directory: {directory}")
        source_qa_file_path = dir_to_path.get(directory)
        results, design_name = load_and_calculate_pass_at_k(
            directory=directory,
            model_name=model_name,
            k_values=k_values,
            threshold=threshold,
            ignored_qa_pairs_path=ignored_qa_pairs_path,
            source_qa_file_path=source_qa_file_path,
        )

        if results:
            # Add directory info to each result
            for result in results:
                result['source_directory'] = directory
                if design_name:
                    result['design_name'] = design_name
            
            all_results.extend(results)
            
            # Calculate summary for this directory
            dir_summary = calculate_summary_statistics(results, k_values, threshold)
            dir_summary['directory'] = directory
            dir_summary['design_name'] = design_name
            dir_summary['num_questions'] = len(results)
            per_directory_summaries[directory] = dir_summary
            
            if design_name:
                design_names.append(design_name)
            
            print(f"  Loaded {len(results)} questions from {directory}")
        else:
            print(f"  Warning: No results found in {directory}")
    
    if not all_results:
        print("Warning: No results found in any directory")
        empty_summary = calculate_summary_statistics([], k_values, threshold)
        return [], empty_summary, {}
    
    # Calculate combined summary across all directories
    combined_summary = calculate_summary_statistics(all_results, k_values, threshold)
    combined_summary['num_directories'] = len([d for d in directories if d in per_directory_summaries])
    combined_summary['directories'] = list(per_directory_summaries.keys())
    combined_summary['design_names'] = list(set(design_names)) if design_names else None
    
    # Add per-directory statistics to combined summary
    if per_directory_summaries:
        combined_summary['per_directory_stats'] = {
            'mean_questions_per_dir': sum(s['num_questions'] for s in per_directory_summaries.values()) / len(per_directory_summaries),
            'min_questions_per_dir': min(s['num_questions'] for s in per_directory_summaries.values()),
            'max_questions_per_dir': max(s['num_questions'] for s in per_directory_summaries.values()),
        }
    
    print(f"\nCombined Results:")
    print(f"  Total directories: {combined_summary['num_directories']}")
    print(f"  Total questions: {combined_summary['total_questions']}")
    print(f"  Total passes: {combined_summary['total_passes']}")
    # Summary: number of questions loaded from JSON per directory
    if per_directory_summaries:
        print(f"\n  Summary (questions loaded from JSON):")
        for directory, dir_summary in per_directory_summaries.items():
            n = dir_summary.get('num_questions', 0)
            dn = dir_summary.get('design_name') or Path(directory).name
            print(f"    {dn}: {n}")
        total_qa = sum(s.get('num_questions', 0) for s in per_directory_summaries.values())
        print(f"    Total: {total_qa} QAs from {len(per_directory_summaries)} directories")
    
    return all_results, combined_summary, per_directory_summaries


def read_directories_from_file(file_path: str) -> List[str]:
    """
    Read directory paths from a text file.
    
    Each line in the file should contain a directory path. Empty lines and lines
    starting with '#' are ignored.
    
    Args:
        file_path: Path to the text file containing directory paths
    
    Returns:
        List of directory paths (with whitespace stripped)
    
    Example:
        directory_list.txt:
            ./qa_output_aes_asap7/
            ./qa_output_aes_nangate45/
            # ./qa_output_aes_sky130hd/  # commented out
        
        >>> dirs = read_directories_from_file("directory_list.txt")
        >>> print(dirs)
        ['./qa_output_aes_asap7/', './qa_output_aes_nangate45/']
    """
    directory_path = Path(file_path)
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory list file not found: {file_path}")
    
    directories = []
    with open(directory_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            # Strip whitespace
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Validate that the directory exists
            dir_path = Path(line)
            if not dir_path.exists():
                print(f"Warning: Directory on line {line_num} does not exist: {line}")
                continue
            
            if not dir_path.is_dir():
                print(f"Warning: Path on line {line_num} is not a directory: {line}")
                continue
            
            directories.append(line)
    
    if not directories:
        raise ValueError(f"No valid directories found in {file_path}")
    
    return directories


def get_default_source_qa_file_path_map() -> Dict[str, str]:
    return {
        "aes_asap7": "qa_output/aes_asap7/aes_20260201_154729_qa_all_types.json",
        "aes_sky130hs": "qa_output/aes_sky130hs/aes_20260131_235414_qa_all_types.json",
        "aes_sky130hd": "qa_output/aes_sky130hd/aes_20260131_225533_qa_all_types.json",
        "aes_nangate45": "qa_output/aes_nangate45/aes_20260131_215425_qa_all_types.json",
        "aes_block_asap7": "qa_output/aes_block_asap7/aes-block_20260131_205343_qa_all_types.json",
        "gcdsky130": "qa_output/gcdsky130/gcd_20260205_152030_qa_all_types.json",
        "ibex_sky130hd": "qa_output/ibex_sky130hd/ibex_20260201_130630_qa_all_types.json",
        "riscv32i-mock-sram_asap7": "qa_output/riscv32i-mock-sram_asap7/riscv32i-mock-sram_20260205_162436_qa_all_types.json",
        "bp_multi_top_nangate45": "qa_output/bp_multi_top_nangate45/bp_multi_top_20260205_142222_qa_all_types.json",
        "uartgf180": "qa_output/uartgf180/uart-blocks_20260205_173006_qa_all_types.json",
        "ariane136_nangate45": "qa_output/ariane136_nangate45/ariane136_20260201_021516_qa_all_types.json",
        "ariane133_nanagte45": "qa_output/ariane133_nanagte45/ariane133_20260201_010409_qa_all_types.json",
        # multifile comparisons:
        "aes_asap7_vs_aes_nangate45": "qa_output/multifile_comparison/aes_asap7_vs_aes_nangate45/aes_asap7_vs_aes_nangate45_20260206_100252_qa_all_multifile.json",
        "aes_asap7_vs_aes_sky130hs": "qa_output/multifile_comparison/aes_asap7_vs_aes_sky130hs/aes_asap7_vs_aes_sky130hs_20260206_100746_qa_all_multifile.json",
        "aes_asap7_vs_aes_sky130hd": "qa_output/multifile_comparison/aes_asap7_vs_aes_sky130hd/aes_asap7_vs_aes_sky130hd_20260206_101145_qa_all_multifile.json",
        "aes_nangate45_vs_aes_sky130hs": "qa_output/multifile_comparison/aes_nangate45_vs_aes_sky130hs/aes_nangate45_vs_aes_sky130hs_20260206_101924_qa_all_multifile.json",
        "aes_nangate45_vs_aes_sky130hd": "qa_output/multifile_comparison/aes_nangate45_vs_aes_sky130hd/aes_nangate45_vs_aes_sky130hd_20260206_102827_qa_all_multifile.json",
        "aes_sky130hs_vs_aes_sky130hd": "qa_output/multifile_comparison/aes_sky130hs_vs_aes_sky130hd/aes_sky130hs_vs_aes_sky130hd_20260206_103525_qa_all_multifile.json",
    }


def build_source_qa_path_map_for_directories(
    directories: List[str],
    design_tech_to_path: Dict[str, str],
) -> Dict[str, str]:
    """
    Build directory -> source QA path map by resolving design_tech from each directory name.

    Uses resolve_design_tech_from_directory so that e.g. ./qa_output_cursor_aes_asap7/
    maps to the path for key "aes_asap7" in design_tech_to_path.

    Args:
        directories: List of evaluation directory paths (e.g. from --directories).
        design_tech_to_path: Map from design_technology key to source QA JSON path (e.g. from get_default_source_qa_file_path_map()).

    Returns:
        Map from directory path -> source QA JSON path for directories that match a key.
    """
    design_tech_keys = list(design_tech_to_path.keys())
    result = {}
    for d in directories:
        key = resolve_design_tech_from_directory(d, design_tech_keys)
        if key is not None:
            result[d] = design_tech_to_path[key]
    return result


def main():
    """
    Main function to calculate pass@k metrics from command line.
    
    Example usage:
        # Single directory
        python pass_k_metric_calculation.py --directory ./qa_responses_aes_asap7_vs_aes_nangate45 --model sonnet4p5
        
        # Multiple directories from command line
        python pass_k_metric_calculation.py --directories ./qa_output_aes_asap7/ ./qa_output_aes_nangate45/ -m sonnet4p5
        
        # Multiple directories from file
        python pass_k_metric_calculation.py --directory-list directory_list.txt -m sonnet4p5 --k-values 1 5 10 --threshold 0.8 --output results.json
    """
    parser = argparse.ArgumentParser(
        description='Calculate pass@k metrics from evaluation JSON files (pass 1-10)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with single directory
  python pass_k_metric_calculation.py -d ./qa_responses_aes_asap7_vs_aes_nangate45 -m sonnet4p5
  
  # Multiple directories from command line
  python pass_k_metric_calculation.py --directories ./qa_output_aes_asap7/ ./qa_output_aes_nangate45/ -m sonnet4p5
  
  # Multiple directories from file
  python pass_k_metric_calculation.py --directory-list directory_list.txt -m sonnet4p5
  
  # Specify k values and threshold
  python pass_k_metric_calculation.py -d ./qa_responses_aes_asap7_vs_aes_nangate45 -m sonnet4p5 --k-values 1 5 10 --threshold 0.8
  
  # Save results to JSON file
  python pass_k_metric_calculation.py -d ./qa_responses_aes_asap7_vs_aes_nangate45 -m sonnet4p5 --output results.json

  # Combined (multiple directories) with ignored QA pairs and (qid, question) filtering
  python pass_k_metric_calculation.py --directories ./qa_output_cursor_aes_block/ ./qa_output_cursor_gcd/ -m sonnet4p5 \\
    --ignored-qa-pairs pass_k_ignored_qa_pairs.json \\
    --source-qa-file-path-map source_qa_paths.json --output combined_results.json

  # Example --source-qa-file-path-map JSON file (e.g. source_qa_paths.json):
  # Keys must match the directory paths passed to --directories (or in --directory-list) exactly.
  # {
  #   "./qa_output_cursor_aes_asap7/": "qa_output/aes_asap7/aes_20260201_154729_qa_all_types.json",
  #   "./qa_output_cursor_aes_nangate45/": "qa_output/aes_nangate45/aes_20260131_215425_qa_all_types.json",
  #   "./qa_output_cursor_gcdsky130/": "qa_output/gcdsky130/gcd_20260205_152030_qa_all_types.json"
  # }
        """
    )
    
    parser.add_argument(
        '-d', '--directory',
        type=str,
        default=None,
        help='Directory path containing the evaluation JSON files (use --directories for multiple)'
    )
    
    parser.add_argument(
        '--directories',
        type=str,
        nargs='+',
        default=None,
        help='Multiple directory paths to aggregate results from'
    )
    
    parser.add_argument(
        '--directory-list',
        type=str,
        default=None,
        help='Path to a text file containing directory paths (one per line, # for comments)'
    )
    
    parser.add_argument(
        '-m', '--model',
        type=str,
        required=True,
        help='Model name pattern to match in filenames (e.g., "sonnet4p5", "gpt-5.2")'
    )
    
    parser.add_argument(
        '--k-values',
        type=int,
        nargs='+',
        default=[1, 5],
        help='List of k values to calculate pass@k for (default: [1, 5, 10])'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.8,
        help='Threshold for binary pass/fail classification (default: 0.8)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output JSON file path (if not specified, prints to stdout)'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['json', 'table', 'summary'],
        default='json',
        help='Output format: json (full JSON), table (formatted table), or summary (default: summary)'
    )
    
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify that each question has exactly 10 passes (or expected number)'
    )
    
    parser.add_argument(
        '--expected-passes',
        type=int,
        default=10,
        help='Expected number of passes per question for verification (default: 10)'
    )

    parser.add_argument(
        '--ignored-qa-pairs',
        type=str,
        default=None,
        help='Path to pass_k_ignored_qa_pairs.json; question IDs listed there are excluded from pass@k'
    )

    parser.add_argument(
        '--source-qa-file-path',
        type=str,
        default=None,
        help='Path to the source QA JSON file (single-directory). Enables (qid, question) exclusion when the QA file '
             'is not inside the evaluation directory (e.g. QA in qa_output/aes_asap7/, evaluation in qa_output_cursor_aes_asap7/).'
    )

    parser.add_argument(
        '--source-qa-file-path-map',
        type=str,
        default=None,
        help='Path to a JSON file mapping each evaluation directory -> path to its source QA JSON. '
             'Used for multi-directory runs to enable (qid, question) exclusion when QA files live '
             'outside each evaluation directory. Example file content: '
             '{"qa_output_cursor_aes_asap7": "qa_output/aes_asap7/aes_20260201_154729_qa_all_types.json", '
             '"qa_output_cursor_aes_nangate45": "qa_output/aes_nangate45/aes_20260131_215425_qa_all_types.json"}'
    )

    parser.add_argument(
        '--by-category',
        action='store_true',
        help='Also compute pass@k per stage category (and combined) using qa_stats_results/design_tech_qa_list.json'
    )

    parser.add_argument(
        '--design-tech-qa-list',
        type=str,
        default=None,
        help='Path to design_tech_qa_list.json for --by-category (default: qa_stats_results/design_tech_qa_list.json)'
    )

    parser.add_argument(
        '--verify-by-category',
        action='store_true',
        help='When using --by-category, run verification to diagnose missing metrics (qid overlap, design_tech keys)'
    )

    parser.add_argument(
        '--preserve-original-categories',
        action='store_true',
        help='When using --by-category, also compute and output pass@k per original (raw) category in by_category_original'
    )

    args = parser.parse_args()
    
    # Validate arguments - check which directory input method is used
    directory_inputs = sum([
        args.directory is not None,
        args.directories is not None,
        args.directory_list is not None
    ])
    
    if directory_inputs == 0:
        parser.error("Must provide one of: --directory, --directories, or --directory-list")
    
    if directory_inputs > 1:
        parser.error("Cannot specify multiple directory input methods. Use only one of: --directory, --directories, or --directory-list")
    
    # Read directories from file if specified
    directories_to_process = None
    if args.directory_list:
        try:
            directories_to_process = read_directories_from_file(args.directory_list)
            print(f"Read {len(directories_to_process)} directories from {args.directory_list}")
        except (FileNotFoundError, ValueError) as e:
            parser.error(f"Error reading directory list file: {e}")
    elif args.directories:
        directories_to_process = args.directories
    
    print(f"Model name: {args.model}")
    print(f"K values: {args.k_values}")
    print(f"Threshold: {args.threshold}")
    print()
    
    # Handle multiple directories
    if directories_to_process:
        print(f"Processing {len(directories_to_process)} directories:")
        for d in directories_to_process:
            print(f"  - {d}")
        print()

        source_qa_file_path_by_directory = None
        if args.source_qa_file_path_map:
            try:
                with open(args.source_qa_file_path_map, "r") as f:
                    source_qa_file_path_by_directory = json.load(f)
                print(f"Using source QA file path map from {args.source_qa_file_path_map}")
            except (FileNotFoundError, json.JSONDecodeError) as e:
                parser.error(f"Failed to load --source-qa-file-path-map: {e}")
        elif args.ignored_qa_pairs is not None:
            source_qa_file_path_by_directory = build_source_qa_path_map_for_directories(
                directories_to_process,
                get_default_source_qa_file_path_map(),
            )
            if source_qa_file_path_by_directory:
                print("Using default source QA file path map (--ignored-qa-pairs specified, --source-qa-file-path-map not)")

        results, summary, per_directory_summaries = calculate_average_pass_at_k_multiple_directories(
            directories=directories_to_process,
            model_name=args.model,
            k_values=args.k_values,
            threshold=args.threshold,
            ignored_qa_pairs_path=args.ignored_qa_pairs,
            source_qa_file_path_by_directory=source_qa_file_path_by_directory,
        )
        
        if not results:
            print("No results found in any directory.")
            return
        
        design_name = None  # Multiple directories, no single design name
        print()
    
    # Handle single directory
    else:
        print(f"Loading evaluation files from: {args.directory}")
        print()
        
        results, design_name = load_and_calculate_pass_at_k(
            directory=args.directory,
            model_name=args.model,
            k_values=args.k_values,
            threshold=args.threshold,
            ignored_qa_pairs_path=args.ignored_qa_pairs,
            source_qa_file_path=args.source_qa_file_path,
        )
        
        if not results:
            print("No results found. Check that:")
            print(f"  1. Directory exists: {args.directory}")
            print(f"  2. Files match pattern: *_pass{{1-10}}of10_*{args.model}*_evaluation.json")
            return
        
        print(f"Found {len(results)} questions across {len(set(r['num_passes'] for r in results))} different pass counts")
        if design_name:
            print(f"Design name: {design_name}")
        print()
        
        # Calculate summary statistics
        summary = calculate_summary_statistics(results, args.k_values, args.threshold)
        per_directory_summaries = None
    
    # Verify pass counts if requested
    if args.verify:
        print("=" * 80)
        print("PASS COUNT VERIFICATION")
        print("=" * 80)
        print()
        
        verification = verify_pass_counts(results, expected_passes=args.expected_passes)
        stats = verification['statistics']
        
        print(f"Expected passes per question: {stats['expected_passes']}")
        print(f"Total questions: {stats['total_questions']}")
        print(f"Valid questions: {stats['valid_questions']}")
        print(f"Invalid questions: {stats['invalid_questions']}")
        print(f"Min passes: {stats['min_passes']}")
        print(f"Max passes: {stats['max_passes']}")
        print(f"Mean passes: {stats['mean_passes']:.2f}")
        print()
        
        # Show pass count distribution
        print("Pass Count Distribution:")
        for pass_count, count in sorted(stats['pass_count_distribution'].items(), key=lambda x: int(x[0])):
            print(f"  {pass_count} passes: {count} question(s)")
        print()
        
        # Show invalid questions
        if verification['invalid_questions']:
            print("=" * 80)
            print(f"INVALID QUESTIONS ({len(verification['invalid_questions'])}):")
            print("=" * 80)
            
            # Group by pass count
            by_pass_count = {}
            for q in verification['invalid_questions']:
                num_passes = q['num_passes']
                if num_passes not in by_pass_count:
                    by_pass_count[num_passes] = []
                by_pass_count[num_passes].append(q)
            
            for num_passes in sorted(by_pass_count.keys(), reverse=True):
                questions = by_pass_count[num_passes]
                print(f"\nQuestions with {num_passes} passes (expected {args.expected_passes}):")
                for q in questions[:20]:  # Show first 20
                    print(f"  - {q['question_id']} (diff: {q['difference']:+d}, dir: {q.get('source_directory', 'unknown')})")
                if len(questions) > 20:
                    print(f"  ... and {len(questions) - 20} more")
            
            print()
            print("=" * 80)
        else:
            print("✅ All questions have the expected number of passes!")
            print()
        
        # Save verification results if output is specified
        if args.output:
            verification_file = Path(args.output).parent / f"{Path(args.output).stem}_verification.json"
            with open(verification_file, 'w') as f:
                json.dump(verification, f, indent=2)
            print(f"Verification results saved to: {verification_file}")
            print()
    
    # Pass@k by category (design_tech -> categories from design_tech_qa_list.json)
    pass_at_k_by_category_result = None
    results_by_design_tech_for_verify = None
    if args.by_category:
        qa_list_path = args.design_tech_qa_list or "qa_stats_results/design_tech_qa_list.json"
        if not Path(qa_list_path).exists():
            print(f"Warning: --by-category requested but design-tech QA list not found: {qa_list_path}")
            print("  Run qa_stats_summary with --output-dir to generate qa_stats_results/design_tech_qa_list.json")
        else:
            # Use same directory as main output for counted QA dumps
            if args.output:
                output_path = Path(args.output)
                dump_counted_dir = str(output_path) if output_path.is_dir() else str(output_path.parent)
            else:
                dump_counted_dir = './results/'
            if directories_to_process:
                # Multiple directories: group results by design_name; normalize multifile so both orderings map to one key
                results_by_design_tech = {}
                for r in results:
                    dn = r.get('design_name') or 'unknown'
                    key = _normalize_multifile_design_tech(dn)
                    if key not in results_by_design_tech:
                        results_by_design_tech[key] = []
                    results_by_design_tech[key].append(r)
                results_by_design_tech_for_verify = results_by_design_tech
                
                pass_at_k_by_category_result = calculate_pass_at_k_by_category_all(
                    qa_list_path, results_by_design_tech, args.k_values, args.threshold,
                    preserve_original_categories=args.preserve_original_categories,
                    dump_counted_qa_path=dump_counted_dir,
                )
            else:
                # Single directory
                design_tech = design_name or Path(args.directory).name
                results_by_design_tech_for_verify = {design_tech: results}
                pass_at_k_by_category_result = {
                    'by_design_tech': {
                        design_tech: calculate_pass_at_k_by_category(
                            results, qa_list_path, design_tech, args.k_values, args.threshold,
                            preserve_original_categories=args.preserve_original_categories,
                            dump_counted_qa_path=dump_counted_dir,
                        )
                    },
                    'combined': None,
                }
                # Single design_tech: set combined to that design_tech's combined
                if design_tech in pass_at_k_by_category_result['by_design_tech']:
                    pass_at_k_by_category_result['combined'] = pass_at_k_by_category_result['by_design_tech'][design_tech].get('combined')

    if args.verify_by_category and args.by_category and results_by_design_tech_for_verify is not None:
        qa_list_path_verify = args.design_tech_qa_list or "qa_stats_results/design_tech_qa_list.json"
        if Path(qa_list_path_verify).exists():
            verify_pass_at_k_by_category_mapping(
                qa_list_path_verify, results_by_design_tech_for_verify, sample_size=5, verbose=True
            )

    # Print pass@k by category summary when computed
    if pass_at_k_by_category_result is not None:
        print("=" * 80)
        print("PASS@K BY CATEGORY (stage categories from design_tech_qa_list)")
        print("=" * 80)
        by_dt = pass_at_k_by_category_result.get('by_design_tech') or {}
        # Sum of "Combined" n across design_techs = total questions in this run (each question counted once, by its design_tech)
        sum_combined_n = sum(
            (dt_data.get('combined') or {}).get('num_questions', 0)
            for dt_data in by_dt.values()
        )
        print(f"Total questions in this run: {sum_combined_n} (each question appears under exactly one design+tech)")
        # Report questions that are in results but not in design_tech_qa_list (uncategorized)
        if results_by_design_tech_for_verify is not None:
            qa_list_path_for_check = args.design_tech_qa_list or "qa_stats_results/design_tech_qa_list.json"
            if Path(qa_list_path_for_check).exists():
                report_questions_not_in_any_category(
                    results_by_design_tech_for_verify,
                    qa_list_path_for_check,
                    verbose=True,
                )
        print()
        for dt, dt_data in by_dt.items():
            print(f"\nDesign+tech: {dt}")
            combined = dt_data.get('combined')
            if combined and combined.get('num_questions', 0) > 0:
                print(f"  Combined: n={combined['num_questions']}", end="")
                for k in args.k_values:
                    if k in combined.get('average_pass_at_k', {}):
                        print(f"  Pass@{k}={combined['average_pass_at_k'][k]:.3f}", end="")
                print()
                print("             ", end="")
                for k in args.k_values:
                    if k in combined.get('average_soft_pass_at_k', {}):
                        print(f"  Soft@{k}={combined['average_soft_pass_at_k'][k]:.3f}", end="")
                print()
            for cat, cat_data in sorted((dt_data.get('by_category') or {}).items()):
                n = cat_data.get('num_questions', 0)
                if n == 0:
                    continue
                print(f"  {cat}: n={n}", end="")
                for k in args.k_values:
                    if k in cat_data.get('average_pass_at_k', {}):
                        print(f"  Pass@{k}={cat_data['average_pass_at_k'][k]:.3f}", end="")
                print()
                print("             ", end="")
                for k in args.k_values:
                    if k in cat_data.get('average_soft_pass_at_k', {}):
                        print(f"  Soft@{k}={cat_data['average_soft_pass_at_k'][k]:.3f}", end="")
                print()
            if args.preserve_original_categories:
                orig = dt_data.get('by_category_original') or {}
                if orig:
                    print("  Original categories:")
                    for cat, cat_data in sorted(orig.items()):
                        n = cat_data.get('num_questions', 0)
                        if n == 0:
                            continue
                        print(f"    {cat}: n={n}", end="")
                        for k in args.k_values:
                            if k in cat_data.get('average_pass_at_k', {}):
                                print(f"  Pass@{k}={cat_data['average_pass_at_k'][k]:.3f}", end="")
                        print()
                        print("                 ", end="")
                        for k in args.k_values:
                            if k in cat_data.get('average_soft_pass_at_k', {}):
                                print(f"  Soft@{k}={cat_data['average_soft_pass_at_k'][k]:.3f}", end="")
                        print()
        combined_global = pass_at_k_by_category_result.get('combined')
        if combined_global and combined_global.get('num_questions', 0) > 0 and len(by_dt) != 1:
            n_run = combined_global['num_questions']
            # Optionally show benchmark total (design_tech_qa_list has all QA pairs)
            qa_list_path = args.design_tech_qa_list or "qa_stats_results/design_tech_qa_list.json"
            benchmark_total = None
            if Path(qa_list_path).exists():
                try:
                    qa_list = load_design_tech_qa_list(qa_list_path)
                    benchmark_total = sum(len(entries) for entries in qa_list.values()) if qa_list else 0
                except Exception:
                    pass
            n_note = f" (benchmark total: {benchmark_total})" if benchmark_total is not None and benchmark_total != n_run else ""
            print(f"\nAll design_techs combined (this run): n={n_run}{n_note}", end="")
            for k in args.k_values:
                if k in combined_global.get('average_pass_at_k', {}):
                    print(f"  Pass@{k}={combined_global['average_pass_at_k'][k]:.3f}", end="")
            print()
            print("             ", end="")
            for k in args.k_values:
                if k in combined_global.get('average_soft_pass_at_k', {}):
                    print(f"  Soft@{k}={combined_global['average_soft_pass_at_k'][k]:.3f}", end="")
            print()
        # Flow Execution, Single Stage, Multi-Stage, Other across all design_techs (sum = total n)
        by_cat_global = pass_at_k_by_category_result.get('by_category_global') or {}
        if by_cat_global and len(by_dt) != 1:
            print("\nBy category (all design_techs):")
            for cat in list(MERGED_CATEGORY_ORDER) + [MERGED_CATEGORY_OTHER]:
                cat_data = by_cat_global.get(cat)
                if not cat_data or cat_data.get('num_questions', 0) == 0:
                    continue
                n = cat_data['num_questions']
                print(f"  {cat}: n={n}", end="")
                for k in args.k_values:
                    if k in cat_data.get('average_pass_at_k', {}):
                        print(f"  Pass@{k}={cat_data['average_pass_at_k'][k]:.3f}", end="")
                print()
                print("             ", end="")
                for k in args.k_values:
                    if k in cat_data.get('average_soft_pass_at_k', {}):
                        print(f"  Soft@{k}={cat_data['average_soft_pass_at_k'][k]:.3f}", end="")
                print()
        print()

    # Output results
    if args.output:
        # Ensure output filename includes design name and model name
        output_path = Path(args.output)
        
        # If output_path is a directory, generate default filename
        if output_path.is_dir():
            output_dir = output_path
            if per_directory_summaries:
                # Multiple directories - use "combined" or design names
                if summary.get('design_names') and len(summary['design_names']) <= 3:
                    design_part = '_'.join(summary['design_names'])
                    output_stem = f"pass_at_k_{design_part}_{args.model}"
                else:
                    output_stem = f"pass_at_k_combined_{args.model}"
            elif design_name:
                output_stem = f"pass_at_k_{design_name}_{args.model}"
            else:
                output_stem = f"pass_at_k_{args.model}"
            output_suffix = ".json"
        else:
            # It's a file path
            output_stem = output_path.stem  # filename without extension
            output_suffix = output_path.suffix if output_path.suffix else ".json"  # extension
            output_dir = output_path.parent  # directory
            
            # Build the name components
            name_parts = []
            if per_directory_summaries:
                # For multiple directories, add "combined" if not already present
                if 'combined' not in output_stem and 'multiple' not in output_stem:
                    if summary.get('design_names') and len(summary['design_names']) <= 2:
                        name_parts.extend(summary['design_names'])
                    else:
                        name_parts.append('combined')
            elif design_name and design_name not in output_stem:
                name_parts.append(design_name)
            if args.model not in output_stem:
                name_parts.append(args.model)
            
            # If we have parts to add, append them
            if name_parts:
                output_stem = f"{output_stem}_{'_'.join(name_parts)}"
        
        # Construct final output path
        final_output_path = output_dir / f"{output_stem}{output_suffix}"
        
        # Ensure output directory exists
        final_output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build and dump renamed qid map (original -> renamed for duplicate qids)
        if results_by_design_tech_for_verify is not None:
            renamed_qid_map = build_renamed_qid_map(results_by_design_tech_for_verify)
        else:
            renamed_qid_map = build_renamed_qid_map_from_flat_results(
                results,
                design_tech_for_single=design_name or (Path(args.directory).name if args.directory else None),
            )
        # Save to JSON file with summary included
        output_data = {
            'summary': summary,
            'questions': results
        }
        if per_directory_summaries:
            output_data['per_directory_summaries'] = per_directory_summaries
        if pass_at_k_by_category_result is not None:
            output_data['pass_at_k_by_category'] = pass_at_k_by_category_result
        output_data['renamed_qid_map'] = renamed_qid_map
        
        with open(final_output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to: {final_output_path}")
        # Also dump renamed map as standalone file for easy inspection
        renamed_map_path = output_dir / f"{output_stem}_renamed_qid_map.json"
        with open(renamed_map_path, 'w') as f:
            json.dump(renamed_qid_map, f, indent=2)
        print(f"Renamed qid map (ori -> renamed) saved to: {renamed_map_path}")
        # Dump pass@k by design_tech as CSV (rows=design_tech, cols=pass@k per category and combined)
        if pass_at_k_by_category_result is not None:
            pass_at_k_csv_path = output_dir / f"{output_stem}_pass_at_k_by_design_tech.csv"
            _write_pass_at_k_by_design_tech_csv(pass_at_k_csv_path, pass_at_k_by_category_result, args.k_values)
            print(f"Pass@k by design_tech (CSV) saved to: {pass_at_k_csv_path}")
            soft_pass_at_k_csv_path = output_dir / f"{output_stem}_soft_pass_at_k_by_design_tech.csv"
            _write_soft_pass_at_k_by_design_tech_csv(soft_pass_at_k_csv_path, pass_at_k_by_category_result, args.k_values)
            print(f"Soft pass@k by design_tech (CSV) saved to: {soft_pass_at_k_csv_path}")
        print()
        
        # Print summary
        print("=" * 80)
        print("PASS@K METRICS SUMMARY")
        print("=" * 80)
        print()
        
        # Overall statistics
        print(f"Overall Statistics:")
        if per_directory_summaries:
            print(f"  Number of directories: {summary.get('num_directories', len(per_directory_summaries))}")
            if 'design_names' in summary and summary['design_names']:
                print(f"  Design names: {', '.join(summary['design_names'])}")
        print(f"  Total questions: {summary['total_questions']}")
        print(f"  Total passes: {summary['total_passes']}")
        print(f"  Mean overall score: {summary['overall_score_stats']['mean']:.3f}")
        print(f"  Max overall score: {summary['overall_score_stats']['max']:.3f}")
        print(f"  Min overall score: {summary['overall_score_stats']['min']:.3f}")
        if summary['overall_score_stats']['std'] > 0:
            print(f"  Std overall score: {summary['overall_score_stats']['std']:.3f}")
        if per_directory_summaries and 'per_directory_stats' in summary:
            print(f"  Mean questions per directory: {summary['per_directory_stats']['mean_questions_per_dir']:.1f}")
        print()
        
        # Average pass@k across all questions (binary with threshold)
        print(f"Average Pass@K (threshold={summary['threshold']}):")
        for k in args.k_values:
            if k in summary['average_pass_at_k']:
                print(f"  Pass@{k}: {summary['average_pass_at_k'][k]:.3f}")
        print()
        
        # Average soft pass@k across all questions
        print(f"Average Soft Pass@K:")
        for k in args.k_values:
            if k in summary['average_soft_pass_at_k']:
                print(f"  Soft Pass@{k}: {summary['average_soft_pass_at_k'][k]:.3f}")
        
        # Print per-directory summaries if available
        if per_directory_summaries:
            print()
            print("=" * 80)
            print("PER-DIRECTORY SUMMARIES")
            print("=" * 80)
            for dir_path, dir_summary in per_directory_summaries.items():
                print(f"\nDirectory: {dir_path}")
                if dir_summary.get('design_name'):
                    print(f"  Design: {dir_summary['design_name']}")
                print(f"  Questions: {dir_summary['num_questions']}")
                print(f"  Average Pass@K:")
                for k in args.k_values:
                    if k in dir_summary['average_pass_at_k']:
                        print(f"    Pass@{k}: {dir_summary['average_pass_at_k'][k]:.3f} (soft: {dir_summary['average_soft_pass_at_k'][k]:.3f})")
        print()
        
        # Per-question statistics
        print(f"Per-Question Statistics:")
        print(f"  Mean number of passes: {summary['per_question_stats']['mean_num_passes']:.1f}")
        print(f"  Mean question score: {summary['per_question_stats']['mean_question_score']:.3f}")
        print(f"  Min/Max passes: {summary['per_question_stats']['min_num_passes']}/{summary['per_question_stats']['max_num_passes']}")
    else:
        # Print to stdout based on format
        if args.format == 'json':
            output_data = {
                'summary': summary,
                'questions': results
            }
            if per_directory_summaries:
                output_data['per_directory_summaries'] = per_directory_summaries
            if pass_at_k_by_category_result is not None:
                output_data['pass_at_k_by_category'] = pass_at_k_by_category_result
            print(json.dumps(output_data, indent=2))
        elif args.format == 'table':
            # Print formatted table
            print(f"{'Question ID':<35} {'Passes':<8} {'Mean':<8} {'Pass@1':<8} {'Pass@5':<8} {'Pass@10':<8}")
            print('-' * 100)
            for result in results:
                qid = result['question_id']
                passes = result['num_passes']
                mean = result['mean_score']
                p1 = result['pass_at_k'].get(1, 0.0)
                p5 = result['pass_at_k'].get(5, 0.0)
                p10 = result['pass_at_k'].get(10, 0.0)
                print(f"{qid:<35} {passes:<8} {mean:<8.3f} {p1:<8.3f} {p5:<8.3f} {p10:<8.3f}")
            print()
            # Print summary at the end
            print("=" * 100)
            print("SUMMARY")
            print("=" * 100)
            print(f"Average Pass@K (threshold={args.threshold}):")
            for k in args.k_values:
                if k in summary['average_pass_at_k']:
                    print(f"  Pass@{k}: {summary['average_pass_at_k'][k]:.3f}")
            print()
            print(f"Average Soft Pass@K:")
            for k in args.k_values:
                if k in summary['average_soft_pass_at_k']:
                    print(f"  Soft Pass@{k}: {summary['average_soft_pass_at_k'][k]:.3f}")
        else:  # summary
            # Print summary statistics using the summary function
            print("=" * 80)
            print("PASS@K METRICS SUMMARY")
            print("=" * 80)
            print()
            
            # Overall statistics
            print(f"Overall Statistics:")
            print(f"  Total questions: {summary['total_questions']}")
            print(f"  Total passes: {summary['total_passes']}")
            print(f"  Mean overall score: {summary['overall_score_stats']['mean']:.3f}")
            print(f"  Max overall score: {summary['overall_score_stats']['max']:.3f}")
            print(f"  Min overall score: {summary['overall_score_stats']['min']:.3f}")
            if summary['overall_score_stats']['std'] > 0:
                print(f"  Std overall score: {summary['overall_score_stats']['std']:.3f}")
            print()
            
            # Average pass@k across all questions (binary with threshold)
            print(f"Average Pass@K (threshold={summary['threshold']}):")
            for k in args.k_values:
                if k in summary['average_pass_at_k']:
                    print(f"  Pass@{k}: {summary['average_pass_at_k'][k]:.3f}")
            print()
            
            # Average soft pass@k across all questions
            print(f"Average Soft Pass@K:")
            for k in args.k_values:
                if k in summary['average_soft_pass_at_k']:
                    print(f"  Soft Pass@{k}: {summary['average_soft_pass_at_k'][k]:.3f}")
            print()
            
            # Per-question statistics
            print(f"Per-Question Statistics:")
            print(f"  Mean number of passes: {summary['per_question_stats']['mean_num_passes']:.1f}")
            print(f"  Mean question score: {summary['per_question_stats']['mean_question_score']:.3f}")
            print(f"  Min/Max passes: {summary['per_question_stats']['min_num_passes']}/{summary['per_question_stats']['max_num_passes']}")
            print()
            
            # Per-question details
            print("Per-Question Details:")
            print("-" * 80)
            for result in results:
                print(f"\nQuestion: {result['question_id']}")
                print(f"  Passes: {result['num_passes']}")
                print(f"  Scores: {result['overall_scores']}")
                print(f"  Mean: {result['mean_score']:.3f}, Max: {result['max_score']:.3f}, Min: {result['min_score']:.3f}")
                print(f"  Pass@K (threshold={result['pass_at_k_threshold']}):")
                for k in args.k_values:
                    if k in result['pass_at_k']:
                        print(f"    Pass@{k}: {result['pass_at_k'][k]:.3f} (soft: {result['soft_pass_at_k'][k]:.3f})")


if __name__ == '__main__':
    main()
