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
QA generation utilities for comparing same stages across multiple OpenROAD metrics JSON files.

This module extends the qa_generation functionality to compare metrics from the same stage
across different log files (metric JSON files), enabling cross-design or cross-configuration analysis.

Usage:
    Compare routing stages from two different designs:
    >>> result = generate_multi_file_qa_pairs(
    ...     ["design1_routing_metrics.json", "design2_routing_metrics.json"],
    ...     n_questions=5,
    ...     seed=42
    ... )
"""

from __future__ import annotations

import json
import random
import re
import pdb
from collections import OrderedDict
from itertools import combinations
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from qa_generation import (
    OpenRoadBackground,
    QuestionMultiStageBackground,
    QAGenerationError,
    _read_json,
    _extract_code_block,
    _tracked_llm_call,
    get_qa_generation_stats,
    validate_extract_answer,
    generate_answer_directly_through_llm,
    validate_direct_vs_code_answer,
)
from stage_parsing_examples import get_multi_stage_parsing_examples
from embedding_utils import get_embedding_model, filter_json_by_similarity


class RejectionStats:
    """Track statistics for answer rejections during QA generation."""
    def __init__(self):
        self.code_execution_failed = 0
        self.low_score_rejected = 0
        self.invalid_answer_rejected = 0  # None, empty, or n/a values
        self.total_attempts = 0
        self.successful = 0
        
        # Detailed tracking for all rejection types
        self.code_failure_details = []  # List of dicts with code execution failure details
        self.low_score_details = []  # List of dicts with low score rejection details
        self.invalid_answer_details = []  # List of dicts with invalid answer rejection details
        
    def record_code_failure(self, error=None, question=None, code=None):
        """
        Record a code execution failure.
        
        Args:
            error: The error message or exception
            question: The question that caused the failure
            code: The generated code that failed (optional)
        """
        self.code_execution_failed += 1
        self.total_attempts += 1
        
        # Store detailed information about this rejection
        detail = {
            "error": str(error) if error is not None else "Unknown error",
            "question": str(question) if question is not None else None,
            "code": str(code)[:500] if code is not None else None,  # Truncate code to 500 chars
        }
        self.code_failure_details.append(detail)
        
    def record_low_score(self, direct_answer=None, code_answer=None, score=None, question=None):
        """
        Record a rejection due to low validation score.
        
        Args:
            direct_answer: The direct LLM-generated answer
            code_answer: The code-extracted answer
            score: The validation score (0.0-1.0)
            question: The question being answered
        """
        self.low_score_rejected += 1
        self.total_attempts += 1
        
        # Store detailed information about this rejection
        detail = {
            "direct_answer": str(direct_answer) if direct_answer is not None else None,
            "code_answer": str(code_answer) if code_answer is not None else None,
            "score": score,
            "question": str(question) if question is not None else None,
        }
        self.low_score_details.append(detail)
        
    def record_invalid_answer(self, answer=None, reason=None, question=None):
        """
        Record a rejection due to invalid answer (None, empty, n/a).
        
        Args:
            answer: The invalid answer value
            reason: The reason why it's invalid (e.g., "None value", "empty", "contains n/a")
            question: The question that produced the invalid answer
        """
        self.invalid_answer_rejected += 1
        self.total_attempts += 1
        
        # Store detailed information about this rejection
        detail = {
            "answer": str(answer) if answer is not None else "None",
            "reason": str(reason) if reason is not None else "Invalid or empty answer",
            "question": str(question) if question is not None else None,
        }
        self.invalid_answer_details.append(detail)
        
    def record_success(self):
        """Record a successful answer generation."""
        self.successful += 1
        self.total_attempts += 1
        
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of rejection statistics."""
        return {
            "total_attempts": self.total_attempts,
            "successful": self.successful,
            "rejected": {
                "code_execution_failed": self.code_execution_failed,
                "low_score_rejected": self.low_score_rejected,
                "invalid_answer_rejected": self.invalid_answer_rejected,
                "total_rejected": (self.code_execution_failed + 
                                  self.low_score_rejected + 
                                  self.invalid_answer_rejected)
            },
            "success_rate": (self.successful / self.total_attempts * 100 
                           if self.total_attempts > 0 else 0.0),
            "code_failure_details": self.code_failure_details,
            "low_score_details": self.low_score_details,
            "invalid_answer_details": self.invalid_answer_details
        }
        
    def print_summary(self, prefix: str = ""):
        """Print a formatted summary of rejection statistics."""
        summary = self.get_summary()
        print(f"\n{prefix}Answer Validation Statistics:")
        print(f"{prefix}  Total Attempts:        {summary['total_attempts']}")
        print(f"{prefix}  Successful:            {summary['successful']}")
        print(f"{prefix}  Rejected Total:        {summary['rejected']['total_rejected']}")
        print(f"{prefix}    - Code Exec Failed:  {summary['rejected']['code_execution_failed']}")
        print(f"{prefix}    - Low Score:         {summary['rejected']['low_score_rejected']}")
        print(f"{prefix}    - Invalid Answer:    {summary['rejected']['invalid_answer_rejected']}")
        print(f"{prefix}  Success Rate:          {summary['success_rate']:.1f}%")
        
        # Show sample of code execution failures if any
        if self.code_failure_details:
            print(f"\n{prefix}Code Execution Failure Examples (showing up to 3):")
            for i, detail in enumerate(self.code_failure_details[:3], 1):
                print(f"{prefix}  Example {i}:")
                print(f"{prefix}    Error: {detail['error'][:100]}...")
                if detail['question']:
                    q_preview = detail['question'][:80] + "..." if len(detail['question']) > 80 else detail['question']
                    print(f"{prefix}    Question: {q_preview}")
                if detail.get('code'):
                    print(f"{prefix}    Code: {detail['code'][:80]}...")
        
        # Show sample of low score rejections if any
        if self.low_score_details:
            print(f"\n{prefix}Low Score Rejection Examples (showing up to 3):")
            for i, detail in enumerate(self.low_score_details[:3], 1):
                print(f"{prefix}  Example {i}:")
                print(f"{prefix}    Score: {detail['score']}")
                if detail['question']:
                    q_preview = detail['question'][:100] + "..." if len(detail['question']) > 100 else detail['question']
                    print(f"{prefix}    Question: {q_preview}")
                if detail['direct_answer']:
                    ans_preview = detail['direct_answer'][:80] + "..." if len(detail['direct_answer']) > 80 else detail['direct_answer']
                    print(f"{prefix}    Direct Answer: {ans_preview}")
                if detail['code_answer']:
                    ans_preview = detail['code_answer'][:80] + "..." if len(detail['code_answer']) > 80 else detail['code_answer']
                    print(f"{prefix}    Code Answer: {ans_preview}")
        
        # Show sample of invalid answer rejections if any
        if self.invalid_answer_details:
            print(f"\n{prefix}Invalid Answer Rejection Examples (showing up to 3):")
            for i, detail in enumerate(self.invalid_answer_details[:3], 1):
                print(f"{prefix}  Example {i}:")
                print(f"{prefix}    Reason: {detail['reason']}")
                if detail['question']:
                    q_preview = detail['question'][:80] + "..." if len(detail['question']) > 80 else detail['question']
                    print(f"{prefix}    Question: {q_preview}")
                if detail['answer']:
                    ans_preview = detail['answer'][:80] + "..." if len(detail['answer']) > 80 else detail['answer']
                    print(f"{prefix}    Answer: {ans_preview}")


def sort_filtered_results_by_design(
    filtered_json_obj: Dict[str, Any],
    design_names: List[str]
) -> Dict[str, Any]:
    """
    Sort filtered JSON results to group all paths from the same design together.
    
    This function reorganizes embedding-filtered results so that all paths for
    design[0] appear first, then all paths for design[1], etc. This makes the
    filtered context much more readable and organized.
    
    Args:
        filtered_json_obj: Dictionary of filtered paths and values from filter_json_by_similarity
        design_names: List of design names in the order they should appear
    
    Returns:
        OrderedDict with paths grouped by design name
        
    Example:
        Before: {"path.design2.x": ..., "path.design1.y": ..., "path.design2.z": ...}
        After:  {"path.design1.y": ..., "path.design2.x": ..., "path.design2.z": ...}
    """
    if not isinstance(filtered_json_obj, dict):
        return filtered_json_obj
    
    # Create design order map
    design_order = {name: idx for idx, name in enumerate(design_names)}
    
    def get_sort_key(item):
        """Get sort key for a (path, value) tuple"""
        key, value = item
        # Extract design name from path
        for design_name in design_names:
            if design_name in key:
                return (design_order[design_name], key)
        # If no design name found, sort to end
        return (len(design_names), key)
    
    # Sort and return as OrderedDict
    sorted_items = sorted(filtered_json_obj.items(), key=get_sort_key)
    return OrderedDict(sorted_items)


def load_stage_metrics(
    metrics_json_path: str | Path,
) -> Dict[str, Any]:
    """
    Load a stage metrics JSON file and extract key information.
    
    Supports two file types:
    1. Single-stage metrics file (e.g., "ariane133_routing_metrics.json")
       - Contains: stage, substages, design_name, sub_design_name
    2. Aggregate all-stages file (e.g., "ariane133_all_stages_metrics.json") 
       - Contains: stages (list), design_name
       - Will be converted to stage format
    
    For aggregate files, this function automatically creates a composite design ID
    from design_name + technology_node + timestamp extracted from design_info.runs[0]
    or flow_execution_summary. Format: "{design}_{tech}_{timestamp}"
    (e.g., "aes_sky130hd_2026-01-27T23_53_55_640Z")
    
    Args:
        metrics_json_path: Path to a stage metrics JSON file
    
    Returns:
        Dict with:
        - stage: Stage type (e.g., 'routing', 'placement')
        - substages: Dict of substage data
        - design_name: Composite design ID for aggregate files, simple name for single-stage
        - sub_design_name: Sub-design name
        - file_path: Original file path
        - log_file: Original log file path (extracted from metrics JSON)
        - metrics: All metrics data
        - is_aggregate: Boolean indicating if source was aggregate file
    
    Raises:
        QAGenerationError: If file format is invalid
    """
    payload = _read_json(metrics_json_path)
    
    if not isinstance(payload, dict):
        raise QAGenerationError(f"Expected dict JSON payload in {metrics_json_path}")
    
    # Helper function to create composite design ID
    def create_composite_design_id(payload: dict) -> str:
        """Create composite design ID from design_name + technology_node + timestamp"""
        import re
        
        # Try to get from design_info first
        design_info = payload.get("design_info", {})
        if design_info and isinstance(design_info, dict):
            runs = design_info.get("runs", [])
            if runs and isinstance(runs, list) and len(runs) > 0:
                run = runs[0]
                design_name = run.get("design_name")
                technology_node = run.get("technology_node")
                timestamp = run.get("timestamp")
                
                if design_name and technology_node and timestamp:
                    # Clean timestamp: remove special chars, keep only alphanumeric, dash, underscore
                    clean_ts = re.sub(r'[^\w\-]', '_', timestamp)
                    return f"{design_name}_{technology_node}_{clean_ts}"
        
        # Fallback: try top-level fields or flow_execution_summary
        design_name = payload.get("design_name")
        
        # Try to get technology_node from flow_execution_summary
        flow_summary = payload.get("flow_execution_summary", {})
        technology_node = None
        timestamp = None
        if flow_summary and isinstance(flow_summary, dict):
            technology_node = flow_summary.get("technology_node")
            timestamp = payload.get("timestamp")  # Top-level timestamp
        
        # If we have all three, create composite ID
        if design_name and technology_node and timestamp:
            clean_ts = re.sub(r'[^\w\-]', '_', str(timestamp))
            return f"{design_name}_{technology_node}_{clean_ts}"
        
        # Final fallback: just return design_name
        return design_name or "unknown"
    
    # Check if this is an aggregate file (has "stages" list)
    if "stages" in payload and isinstance(payload.get("stages"), list):
        # This is an aggregate all_stages_metrics.json file
        # Create composite design ID from available fields
        composite_design_id = create_composite_design_id(payload)
        
        # Extract log_file from top-level or flow_execution_summary
        log_file = payload.get("log_file")
        if not log_file:
            flow_summary = payload.get("flow_execution_summary", {})
            if isinstance(flow_summary, dict):
                log_file = flow_summary.get("log_file")
        
        return {
            "stage": None,  # Will be set when stage is selected
            "substages": None,
            "design_name": composite_design_id,  # Now contains composite design ID
            "sub_design_name": None,
            "file_path": str(metrics_json_path),
            "log_file": log_file,  # Add original log file path
            "metrics": payload,
            "is_aggregate": True,
            "stages": payload.get("stages", []),
        }
    elif "stage" in payload:
        # This is a single-stage metrics file
        # Extract log_file from top-level payload
        log_file = payload.get("log_file")
        
        return {
            "stage": payload.get("stage"),
            "substages": payload.get("substages", {}),
            "design_name": payload.get("design_name"),
            "sub_design_name": payload.get("sub_design_name"),
            "file_path": str(metrics_json_path),
            "log_file": log_file,  # Add original log file path
            "metrics": payload,
            "is_aggregate": False,
        }
    else:
        raise QAGenerationError(
            f"Invalid file format in {metrics_json_path}. "
            f"Expected either 'stage' field (single-stage file) or 'stages' field (aggregate file)"
        )


def sample_stages_from_each_file(
    stage_data_list: List[Dict[str, Any]],
    *,
    num_stages: Optional[int] = None,
    seed: Optional[int] = None,
    stage_types: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Sample stages independently from each file for cross-file comparison.
    
    Unlike sample_stages_from_aggregates which finds common stages, this function
    samples stages independently from each file, allowing different stages per design.
    
    This enables comparing overall design flow quality even when designs have
    different available stages.
    
    Sampling strategy:
    1. For each file, sample 1-3 stages independently
    2. Only sample stages that have substages
    3. Return data organized by design
    
    Args:
        stage_data_list: List of stage data dicts from load_stage_metrics()
        num_stages: Number of stage types to sample per file (1-3). If None, randomly choose.
        seed: Optional random seed for reproducibility
        stage_types: Optional list of stage types to filter by (e.g., ['placement', 'routing'])
    
    Returns:
        List of dicts, one per design/file, each containing:
        - design_name: Design name
        - file_path: Source file path
        - sampled_stages: List of sampled stage data
          - stage_type: Stage type (e.g., 'routing')
          - substages_sampled: List of sampled substage names
          - substages_data: Substage metrics
    
    Raises:
        QAGenerationError: If insufficient stages available in any file
    """
    rng = random.Random(seed)
    
    # Determine number of stages to sample per file
    if num_stages is None:
        num_stages = rng.choice([1, 2, 3])
    elif num_stages not in [1, 2, 3]:
        raise ValueError("num_stages must be 1, 2, or 3")
    
    result = []
    
    for file_idx, data in enumerate(stage_data_list):
        design_result = {
            "design_name": data.get("design_name"),
            "file_path": data.get("file_path"),
            "sampled_stages": []
        }
        
        if data.get("is_aggregate", False):
            # Aggregate file - sample stages
            stages = data.get("stages", [])
            
            # Find stages with substages
            available_stages = []
            for stage_entry in stages:
                if isinstance(stage_entry, dict) and stage_entry.get("stage"):
                    st = stage_entry.get("stage")
                    
                    # Filter by stage_types if specified
                    if stage_types and st not in stage_types:
                        continue
                    
                    # Check if this stage has substages
                    substages = stage_entry.get("metrics", {}).get("substages", {})
                    if substages and len(substages) > 0:
                        available_stages.append({
                            "stage_type": st,
                            "stage_entry": stage_entry,
                            "substages": substages
                        })
            
            if len(available_stages) < 1:
                raise QAGenerationError(
                    f"No stages with substages found in {data.get('file_path')}. "
                    f"Available stages: {[s.get('stage') for s in stages]}"
                )
            
            # Sample stages
            num_to_sample = min(num_stages, len(available_stages))
            sampled = rng.sample(available_stages, num_to_sample)
            
            print(f"[sample_stages_from_each_file] Design {file_idx+1} ({data.get('design_name')}): "
                  f"Found {len(available_stages)} stages with substages, sampled {num_to_sample}")
            
            for stage_info in sampled:
                design_result["sampled_stages"].append({
                    "stage_type": stage_info["stage_type"],
                    "substages": stage_info["substages"],
                    "sub_design_name": stage_info["stage_entry"].get("sub_design_name"),
                })
        else:
            # Single-stage file
            st = data.get("stage")
            substages = data.get("substages", {})
            
            if stage_types and st not in stage_types:
                raise QAGenerationError(
                    f"Stage type '{st}' in {data.get('file_path')} not in requested stage_types"
                )
            
            if not substages or len(substages) == 0:
                raise QAGenerationError(
                    f"No substages found in {data.get('file_path')}"
                )
            
            design_result["sampled_stages"].append({
                "stage_type": st,
                "substages": substages,
                "sub_design_name": data.get("sub_design_name"),
            })
            
            print(f"[sample_stages_from_each_file] Design {file_idx+1} ({data.get('design_name')}): "
                  f"Single-stage file with stage '{st}'")
        
        result.append(design_result)
    
    return result


def sample_stages_from_aggregates(
    stage_data_list: List[Dict[str, Any]],
    *,
    num_stages: Optional[int] = None,
    seed: Optional[int] = None,
    stage_types: Optional[List[str]] = None,
    exclude_stage_combinations: Optional[List[Tuple[str, ...]]] = None,
) -> List[Dict[str, Any]]:
    """
    Sample stages from aggregate files for cross-file comparison.
    
    Similar to sample_stages() in qa_generation.py, but works across multiple files.
    Samples stages that exist in all files for fair comparison.
    
    Only samples stages that have substages for meaningful comparison.
    
    Sampling strategy:
    1. Find common stage types across all files
    2. Filter stages that have substages
    3. Randomly sample 1-3 stage types from common stages with substages
    4. Extract those stages from all files
    
    Args:
        stage_data_list: List of stage data dicts from load_stage_metrics()
        num_stages: Number of stage types to sample (1-3). If None, randomly choose.
        seed: Optional random seed for reproducibility
        stage_types: Optional list of stage types to filter by (e.g., ['placement', 'routing'])
        exclude_stage_combinations: Optional list of stage combinations to exclude (tuples of sorted stage types)
    
    Returns:
        List of dicts, one per sampled stage type, each containing:
        - stage_type: The stage type (e.g., 'routing')
        - designs: List of design data for this stage
          - design_name: Design name
          - sub_design_name: Sub-design name
          - file_path: Source file
          - stage_data: Stage metrics and substages
    
    Raises:
        QAGenerationError: If insufficient common stages with substages available
    """
    rng = random.Random(seed)
    
    # Determine number of stages to sample
    if num_stages is None:
        num_stages = rng.choice([1, 2, 3])
    elif num_stages not in [1, 2, 3]:
        raise ValueError("num_stages must be 1, 2, or 3")
    
    # Check if any files are aggregate type
    has_aggregates = any(data.get("is_aggregate", False) for data in stage_data_list)
    
    if not has_aggregates:
        # All are single-stage files - return as single stage comparison
        stage_type = stage_data_list[0].get("stage")
        return [{
            "stage_type": stage_type,
            "designs": [
                {
                    "design_name": data.get("design_name"),
                    "sub_design_name": data.get("sub_design_name"),
                    "file_path": data.get("file_path"),
                    "stage_data": {
                        "stage": data.get("stage"),
                        "substages": data.get("substages", {}),
                        "metrics": data.get("metrics", {}),
                    }
                }
                for data in stage_data_list
            ]
        }]
    
    # Find common stage types across all files that have substages
    stage_types_by_file = []
    stages_with_substages_by_file = []  # Track which stages have substages
    all_stages_by_file = []  # Store stage entries for later extraction
    
    for data in stage_data_list:
        if data.get("is_aggregate", False):
            stages = data.get("stages", [])
            available_stages = {}
            stages_with_substages = set()
            
            for stage_entry in stages:
                if isinstance(stage_entry, dict) and stage_entry.get("stage"):
                    st = stage_entry.get("stage")
                    # Filter by stage_types if specified
                    if stage_types and st not in stage_types:
                        continue
                    
                    # Check if this stage has substages
                    substages = stage_entry.get("metrics", {}).get("substages", {})
                    if substages and len(substages) > 0:
                        available_stages[st] = stage_entry
                        stages_with_substages.add(st)
            
            stage_types_by_file.append(stages_with_substages)
            stages_with_substages_by_file.append(stages_with_substages)
            all_stages_by_file.append(available_stages)
        else:
            # Single-stage file
            st = data.get("stage")
            substages = data.get("substages", {})
            
            if stage_types and st not in stage_types:
                stage_types_by_file.append(set())
                stages_with_substages_by_file.append(set())
                all_stages_by_file.append({})
            elif substages and len(substages) > 0:
                stage_types_by_file.append({st})
                stages_with_substages_by_file.append({st})
                all_stages_by_file.append({st: data})
            else:
                stage_types_by_file.append(set())
                stages_with_substages_by_file.append(set())
                all_stages_by_file.append({})
    
    # Find intersection of all stage types (that have substages)
    if not stage_types_by_file:
        raise QAGenerationError("No stage types found in any file")
    
    common_stages = stage_types_by_file[0]
    for stage_set in stage_types_by_file[1:]:
        common_stages = common_stages.intersection(stage_set)
    
    if not common_stages:
        raise QAGenerationError(
            f"No common stage types with substages found across all files. "
            f"Stages with substages by file: {[sorted(list(s)) for s in stages_with_substages_by_file]}"
        )
    
    # Sample stage types
    common_stages_list = sorted(list(common_stages))
    num_to_sample = min(num_stages, len(common_stages_list))
    
    # Generate all possible combinations and filter out used ones
    all_possible_combinations = list(combinations(common_stages_list, num_to_sample))
    
    print(f"[sample_stages_from_aggregates] Found {len(common_stages_list)} common stage types with substages: {common_stages_list}")
    print(f"[sample_stages_from_aggregates] Total possible combinations: {len(all_possible_combinations)}")
    
    # Filter out already-used combinations
    if exclude_stage_combinations:
        # Convert to sorted tuples for comparison
        excluded_set = set(exclude_stage_combinations)
        available_combinations = [
            combo for combo in all_possible_combinations 
            if combo not in excluded_set
        ]
        
        if not available_combinations:
            raise QAGenerationError(
                f"No unique stage combinations available. "
                f"All {len(all_possible_combinations)} possible combinations have been used. "
                f"Consider reducing n_questions or allowing duplicate combinations."
            )
        
        print(f"[sample_stages_from_aggregates] Available unique combinations: {len(available_combinations)} "
              f"(excluded {len(excluded_set)} previously used)")
    else:
        available_combinations = all_possible_combinations
    
    # Sample one combination from available options
    sampled_combination = rng.choice(available_combinations)
    
    # Sort sampled stages by execution order for consistency
    stage_order = ['synthesis', 'floorplan', 'placement', 'cts', 'routing', 'final']
    sampled_stage_types = sorted(
        sampled_combination,
        key=lambda st: stage_order.index(st) if st in stage_order else 999
    )
    
    print(f"[sample_stages_from_aggregates] Sampled {num_to_sample} stage type(s) in execution order: {sampled_stage_types}")
    
    # Extract data for sampled stages
    result = []
    for stage_type in sampled_stage_types:
        stage_comparison = {
            "stage_type": stage_type,
            "designs": []
        }
        
        for idx, data in enumerate(stage_data_list):
            if data.get("is_aggregate", False):
                stage_entry = all_stages_by_file[idx].get(stage_type)
                if not stage_entry:
                    raise QAGenerationError(
                        f"Stage '{stage_type}' not found in {data.get('file_path')}"
                    )
                
                substages = stage_entry.get("metrics", {}).get("substages", {})
                stage_comparison["designs"].append({
                    "design_name": data.get("design_name"),
                    "sub_design_name": stage_entry.get("sub_design_name"),
                    "file_path": data.get("file_path"),
                    "stage_data": {
                        "stage": stage_type,
                        "substages": substages,
                        "metrics": stage_entry.get("metrics", {}),
                    }
                })
            else:
                # Single-stage file
                stage_comparison["designs"].append({
                    "design_name": data.get("design_name"),
                    "sub_design_name": data.get("sub_design_name"),
                    "file_path": data.get("file_path"),
                    "stage_data": {
                        "stage": stage_type,
                        "substages": data.get("substages", {}),
                        "metrics": data.get("metrics", {}),
                    }
                })
        
        result.append(stage_comparison)
    
    return result


def extract_matching_stages_from_aggregates(
    stage_data_list: List[Dict[str, Any]],
    stage_type: Optional[str] = None,
    *,
    seed: Optional[int] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Extract matching stages from aggregate files.
    
    If files are aggregate type (all_stages_metrics.json), this function:
    1. Finds common stage types across all files
    2. Selects one stage type (random or specified)
    3. Extracts that stage's data from each file
    
    Args:
        stage_data_list: List of stage data dicts from load_stage_metrics()
        stage_type: Optional stage type to extract (e.g., 'routing', 'placement')
                   If None, randomly select from common stages
        seed: Optional random seed for stage type selection
    
    Returns:
        Tuple of (selected_stage_type, updated_stage_data_list)
        where updated_stage_data_list has the selected stage extracted
    
    Raises:
        QAGenerationError: If no common stages found or stage_type not available
    """
    rng = random.Random(seed)
    
    # Check if any files are aggregate type
    has_aggregates = any(data.get("is_aggregate", False) for data in stage_data_list)
    
    if not has_aggregates:
        # All are single-stage files, no extraction needed
        return None, stage_data_list
    
    # Find common stage types across all aggregate files
    stage_types_by_file = []
    for data in stage_data_list:
        if data.get("is_aggregate", False):
            stages = data.get("stages", [])
            available_stages = set()
            for stage_entry in stages:
                if isinstance(stage_entry, dict) and stage_entry.get("stage"):
                    available_stages.add(stage_entry.get("stage"))
            stage_types_by_file.append(available_stages)
        else:
            # Single-stage file
            stage_types_by_file.append({data.get("stage")})
    
    # Find intersection of all stage types
    if not stage_types_by_file:
        raise QAGenerationError("No stage types found in any file")
    
    common_stages = stage_types_by_file[0]
    for stage_set in stage_types_by_file[1:]:
        common_stages = common_stages.intersection(stage_set)
    
    if not common_stages:
        raise QAGenerationError(
            f"No common stage types found across all files. "
            f"Stage types by file: {[list(s) for s in stage_types_by_file]}"
        )
    
    # Select stage type
    if stage_type:
        if stage_type not in common_stages:
            raise QAGenerationError(
                f"Requested stage type '{stage_type}' not available in all files. "
                f"Common stages: {common_stages}"
            )
        selected_stage = stage_type
    else:
        selected_stage = rng.choice(sorted(list(common_stages)))
    
    print(f"[extract_matching_stages] Found {len(common_stages)} common stage types: {sorted(common_stages)}")
    print(f"[extract_matching_stages] Selected stage type: {selected_stage}")
    
    # Extract the selected stage from each file
    updated_stage_data_list = []
    for data in stage_data_list:
        if data.get("is_aggregate", False):
            # Find the stage entry
            stages = data.get("stages", [])
            stage_entry = None
            for s in stages:
                if isinstance(s, dict) and s.get("stage") == selected_stage:
                    stage_entry = s
                    break
            
            if not stage_entry:
                raise QAGenerationError(
                    f"Stage '{selected_stage}' not found in {data.get('file_path')}"
                )
            
            # Extract substages from the stage entry's metrics
            substages = stage_entry.get("metrics", {}).get("substages", {})
            
            updated_data = {
                "stage": selected_stage,
                "substages": substages,
                "design_name": data.get("design_name"),
                "sub_design_name": stage_entry.get("sub_design_name"),
                "file_path": data.get("file_path"),
                "metrics": stage_entry.get("metrics", {}),
                "is_aggregate": False,  # Now it's a single stage
            }
            updated_stage_data_list.append(updated_data)
        else:
            # Single-stage file, keep as is but verify it matches
            if data.get("stage") != selected_stage:
                raise QAGenerationError(
                    f"Stage mismatch: file {data.get('file_path')} has stage '{data.get('stage')}' "
                    f"but expected '{selected_stage}'"
                )
            updated_stage_data_list.append(data)
    
    return selected_stage, updated_stage_data_list


def validate_same_stage_type(
    stage_data_list: List[Dict[str, Any]]
) -> Tuple[bool, str]:
    """
    Validate that all loaded stage data are from the same stage type.
    
    Args:
        stage_data_list: List of stage data dicts from load_stage_metrics()
    
    Returns:
        Tuple of (is_valid, stage_type)
        - is_valid: True if all stages are the same type
        - stage_type: The common stage type (or first stage type if invalid)
    
    Raises:
        QAGenerationError: If stage types don't match
    """
    if not stage_data_list:
        raise QAGenerationError("Empty stage data list")
    
    stage_types = [data.get("stage") for data in stage_data_list]
    unique_stages = set(stage_types)
    
    if len(unique_stages) > 1:
        raise QAGenerationError(
            f"All metric files must be from the same stage type. "
            f"Found: {unique_stages}"
        )
    
    return True, stage_types[0]


def sample_matching_substages(
    stage_data_list: List[Dict[str, Any]],
    *,
    seed: Optional[int] = None,
    num_substages: Optional[int] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Sample substages that exist in all provided stage data files.
    
    This ensures we can compare the same substage across different designs/configs.
    
    Args:
        stage_data_list: List of stage data dicts from load_stage_metrics()
        seed: Optional random seed for reproducibility
        num_substages: Number of substages to sample. If None, sample 1-3 substages.
    
    Returns:
        Tuple of (substage_name, substage_data_list)
        - substage_name: The sampled substage identifier (e.g., "5_1_grt")
        - substage_data_list: List of substage data dicts, one from each file
    
    Raises:
        QAGenerationError: If no common substages found
    """
    rng = random.Random(seed)
    
    # Find common substages across all files
    substage_sets = []
    for stage_data in stage_data_list:
        substages = stage_data.get("substages", {})
        substage_sets.append(set(substages.keys()))
    
    # Get intersection of all substage sets
    if not substage_sets:
        raise QAGenerationError("No substages found in any file")
    
    common_substages = substage_sets[0]
    for substage_set in substage_sets[1:]:
        common_substages = common_substages.intersection(substage_set)
    
    if not common_substages:
        raise QAGenerationError(
            f"No common substages found across all files. "
            f"Substages by file: {[list(s) for s in substage_sets]}"
        )
    
    # Sample from common substages
    common_substages_list = sorted(list(common_substages))
    
    # Determine number of substages to sample
    if num_substages is None:
        num_substages = min(rng.randint(1, 3), len(common_substages_list))
    else:
        num_substages = min(num_substages, len(common_substages_list))
    
    # Sample substages
    sampled_substages = rng.sample(common_substages_list, num_substages)
    
    print(f"[sample_matching_substages] Found {len(common_substages_list)} common substages: {common_substages_list}")
    print(f"[sample_matching_substages] Sampled {num_substages} substage(s): {sampled_substages}")
    
    # For now, return data for all sampled substages
    # Collect substage data from all files for all sampled substages
    result_data = []
    for substage_name in sampled_substages:
        substage_comparison = {
            "substage_name": substage_name,
            "designs": []
        }
        
        for stage_data in stage_data_list:
            substages = stage_data.get("substages", {})
            substage_data = substages.get(substage_name, {})
            
            substage_comparison["designs"].append({
                "design_name": stage_data.get("design_name"),
                "sub_design_name": stage_data.get("sub_design_name"),
                "file_path": stage_data.get("file_path"),
                "substage_data": substage_data,
            })
        
        result_data.append(substage_comparison)
    
    return sampled_substages, result_data


def multi_file_qa_pair(
    metrics_json_paths: List[str | Path],
    *,
    seed: Optional[int] = None,
    temperature: float = 0.7,
    num_substages: Optional[int] = None,
    stage_types: Optional[List[str]] = None,
    num_stages: Optional[int] = None,
    rejection_stats: Optional[RejectionStats] = None,
    exclude_stage_combinations: Optional[List[Tuple[str, ...]]] = None,
) -> Dict[str, Any]:
    """
    Generate QA pairs by comparing stages across multiple log files.
    
    This function loads metrics from multiple files (e.g., different designs or configurations),
    samples common stages, and generates questions comparing metrics across the files.
    
    Supports both:
    1. Single-stage metrics files (e.g., "design_routing_metrics.json")
    2. Aggregate all-stages files (e.g., "design_all_stages_metrics.json")
       - Automatically samples common stages for comparison
    
    Args:
        metrics_json_paths: List of paths to stage metrics JSON files
        seed: Optional random seed for reproducibility
        temperature: LLM temperature for generation (default: 0.7)
        num_substages: Number of substages to include in comparison. If None, sample 1-3.
        stage_types: Optional list of stage types to filter by (e.g., ['routing', 'placement'])
        num_stages: Number of stage types to sample (1-3). If None, randomly choose.
        exclude_stage_combinations: Optional list of stage combinations to exclude (tuples of sorted stage types)
    
    Returns:
        A dict with the question set and answers for cross-file comparison:
        - id: Question ID (assigned by caller)
        - type: "multi_file"
        - stages_compared: List of stage types being compared
        - stage_data: List of stage comparison data
        - files_info: List of file information
        - qa_pairs: List of question-answer pairs
        - source_paths: List of source file paths
    
    Example questions generated:
    - "How does wirelength compare between design1 and design2 in the routing stage?"
    - "Which design has better timing metrics across placement and routing stages?"
    - "What are the optimization iteration differences across designs?"
    
    Raises:
        QAGenerationError: If files are incompatible or insufficient data
    """
    rng = random.Random(seed)
    
    if len(metrics_json_paths) < 2:
        raise QAGenerationError("Need at least 2 metric files for comparison")
    
    # Load all stage metrics
    print(f"\n[multi_file_qa_pair] Loading {len(metrics_json_paths)} metric files...")
    stage_data_list = []
    for path in metrics_json_paths:
        stage_data = load_stage_metrics(path)
        stage_data_list.append(stage_data)
        file_type = "aggregate" if stage_data.get("is_aggregate") else "single-stage"
        print(f"  - Loaded: {stage_data['design_name']} ({file_type} file)")

    # Step 1: Sample stages for comparison (common stages across all files)
    sampled_stage_comparisons = sample_stages_from_aggregates(
        stage_data_list,
        num_stages=num_stages,
        seed=seed,
        stage_types=stage_types,
        exclude_stage_combinations=exclude_stage_combinations,
    )
    
    sampled_stage_types = [sc["stage_type"] for sc in sampled_stage_comparisons]
    print(f"[multi_file_qa_pair] Sampled {len(sampled_stage_types)} stage(s) for comparison: {sampled_stage_types}")
    
    # Step 2: Build comparison JSON organized by design, but with the same sampled stages
    comparison_json = []
    
    for stage_comparison in sampled_stage_comparisons:
        stage_type = stage_comparison["stage_type"]
        designs_data = stage_comparison["designs"]
        
        print(f"[multi_file_qa_pair] Processing stage: {stage_type}")
        
        # For this stage, sample substages from each design
        for design_data in designs_data:
            design_name = design_data["design_name"]
            substages = design_data["stage_data"]["substages"]
            
            # Sample substages for this design's stage
            substage_names = list(substages.keys())
            if num_substages:
                num_to_sample = min(num_substages, len(substage_names))
            else:
                num_to_sample = min(rng.choice([1, 2, 3]), len(substage_names))
            
            sampled_substage_names = rng.sample(substage_names, num_to_sample)
            
            print(f"  [{design_name}] {stage_type}: sampled {num_to_sample} substage(s) from {len(substage_names)}")
            
            # Add to comparison JSON (organized by design)
            comparison_json.append({
                "design": design_name,
                "stage": stage_type,
                "substages": {name: substages[name] for name in sampled_substage_names}
            })
    
    comparison_json_str = json.dumps(comparison_json, indent=2)
    
    # Split comparison data by design for clearer prompt
    # Note: design_name from aggregate files contains composite ID (design + tech + timestamp)
    # This allows distinguishing between different runs of the same design
    designs_metrics = {}
    for entry in comparison_json:
        design_name = entry["design"]  # Contains composite design ID from aggregate files
        if design_name not in designs_metrics:
            designs_metrics[design_name] = []
        designs_metrics[design_name].append(entry)
    
    # Create separate JSON strings for each design
    design_metrics_strs = {}
    for design_name, entries in designs_metrics.items():
        design_metrics_strs[design_name] = json.dumps(entries, indent=2)
    
    # Build description for prompt
    design_names = [data.get("design_name") for data in stage_data_list]
    design_desc = ", ".join(design_names)
    stages_desc = ", ".join(sampled_stage_types)
    
    print(f"\n[QA Generation] Generating cross-file comparison questions...")
    print(f"  Designs: {design_desc}")
    print(f"  Stages to compare: {stages_desc}")
    
    # Step 1: Generate questions comparing across files
    # Build design-specific sections for prompt
    design_sections = []
    for idx, design_name in enumerate(design_names, 1):
        design_metrics_str = design_metrics_strs.get(design_name, "[]")
        design_sections.append(f"""# Design {idx} ({design_name}) metrics:
```json
{design_metrics_str}
```""")
    
    all_designs_str = "\n\n".join(design_sections)
    
    question_prompt = f"""You are generating question-answer pairs for comparing OpenROAD EDA design metrics across different designs or configurations.

# OpenROAD EDA log file background:
{OpenRoadBackground}

# Question Examples from Designers:
{QuestionMultiStageBackground}

You are comparing the SAME stages across multiple designs:

Designs: {design_desc}
Stages to compare: {stages_desc}

Each design has data for these stages:

{all_designs_str}

Generate a set of interesting, specific questions that compare the SAME stages across these designs. No more than 6 questions.
Try to explore different questions from different aspects. Ask at least 1 question from each of the [Metric Analysis Questions], [Path Analysis Questions], [Command Setting Analysis Questions].
Imagine you are a designer comparing different designs or configurations. What questions would you ask to understand how the same stages perform differently across designs?

Focus on:
- Metric comparison for the SAME stage across designs (timing, area, wirelength, resource usage, etc.)
- Optimization efficiency differences for the same stage (iterations, convergence patterns)
- Design quality differences for the same stage (violations, overflow, density)
- Resource usage patterns for the same stage (memory, runtime)
- Layer-specific metrics if available (per_layer_metrics)
- DRC iteration patterns in detailed routing
- Which design performs better in each stage

IMPORTANT RULES:
- DO NOT use technical stage identifiers like "3_1", "3_2", "5_1_grt", "5_2_route" in questions
- Use natural language stage names: "global routing", "detailed routing", "placement", etc.
- Refer to substages descriptively: "in the detailed routing substage", "during global routing"
- Ask questions that compare the SAME stage across designs: "Which design has better routing wirelength?", "How does placement density compare between design1 and design2?"
- Focus on comparing the same stage's metrics across different designs
- Focus on meaningful metric differences and patterns
"""

    output_format = """Output the question between ```question and ``` tags.
```question
<Q1> ...
Difficulty: <easy, medium, hard>
```
    
```question
<Q2> ...
Difficulty: <easy, medium, hard>
```
...
```question
<Qn> ...
Difficulty: <easy, medium, hard>
```
"""
    # print("question_prompt:", question_prompt)
    # pdb.set_trace()
    question = _tracked_llm_call(
        system_prompt="You are an expert in chip design and EDA tools, specializing in cross-design comparison and benchmarking.",
        user_prompt=question_prompt + "\n" + output_format,
        temperature=temperature,
    )
    
    # Parse questions with difficulty
    questions = re.findall(r"```question\s*(.*?)\s*```", question, re.DOTALL)
    parsed_questions = []
    for q_block in questions:
        lines = q_block.strip().split("\n")
        q_text = []
        difficulty = "medium"
        
        for line in lines:
            if line.strip().lower().startswith("difficulty:"):
                difficulty = line.split(":", 1)[1].strip().lower()
            else:
                q_text.append(line.strip())
        
        question_text = " ".join(q_text).strip()
        if question_text:
            parsed_questions.append({"text": question_text, "difficulty": difficulty})
    
    if not parsed_questions:
        raise QAGenerationError("No valid questions parsed from LLM output")
    
    print(f"[multi_file_qa_pair] Parsed {len(parsed_questions)} questions")
    
    print(f"[multi_file_qa_pair] Parsed {len(parsed_questions)} questions")
    
    # Build parsing examples - same stages for all designs, so generate one set of examples
    # Sort stages by execution order
    stage_order = ['synthesis', 'floorplan', 'placement', 'cts', 'routing', 'final']
    
    # Sort sampled_stage_comparisons by stage order
    sorted_stage_comparisons = sorted(
        sampled_stage_comparisons,
        key=lambda sc: stage_order.index(sc["stage_type"]) if sc["stage_type"] in stage_order else 999
    )
    
    pseudo_stages = []
    for stage_comp in sorted_stage_comparisons:
        stage_type = stage_comp["stage_type"]
        # Just use the first design's data as example (all have same stages)
        if stage_comp["designs"]:
            design_data = stage_comp["designs"][0]
            pseudo_stages.append({
                "stage": stage_type,
                "occurrence": 1,
                "sub_design_name": design_data["design_name"],
                "metrics": {"substages": design_data["stage_data"]["substages"]},
            })
    
    # Build shared code generation context (used for all questions)
    code_design_sections = []
    for idx, design_name in enumerate(design_names, 1):
        design_metrics_str = design_metrics_strs.get(design_name, "[]")
        code_design_sections.append(f"""# Design {idx} ({design_name}) metrics:
```json
{design_metrics_str}
```""")
    
    all_code_designs_str = "\n\n".join(code_design_sections)
    # Check if rpt_metric field exists in any of the stage data
    include_rpt_metric = "rpt_metric" in all_code_designs_str
    multi_file_parsing_examples = get_multi_stage_parsing_examples(pseudo_stages, include_rpt_metrics=include_rpt_metric)
    
    
    # Build function signature with design-specific parameters
    design_params = ", ".join([f"design_{i+1}_metrics" for i in range(len(design_names))])
    
    # Prepare design-specific metric lists for function calls
    design_metrics_lists = [designs_metrics.get(dn, []) for dn in design_names]
    
    # Get embedding model for filtering (reuse across questions)
    embedding_model = get_embedding_model(model_type="sentence-transformer")
    
    # Step 2: Process each question individually
    final_qa_pairs = []
    all_generation_codes = []  # Store all generated codes for debugging
    
    for i, pq in enumerate(parsed_questions):
        question_str = pq["text"]
        difficulty = pq["difficulty"]
        
        print(f"\n[multi_file_qa_pair] Processing Q{i+1}/{len(parsed_questions)}: {question_str[:80]}...")
        
        # Step 2a: Generate extraction code for THIS question
        code_prompt = f"""You are generating Python code to extract the answer from multi-file design comparison.

Designs: {design_desc}
Stages to compare: {stages_desc}
Question: {question_str}

The data is provided as separate variables for each design:
{all_code_designs_str}

Write Python code that extracts the answer to THIS SPECIFIC QUESTION from the design metrics.

Requirements:
- Each design_X_metrics variable is a list of dicts with: 'design', 'stage', and 'substages'
- Each dict represents one stage for that design
- Access substage metrics: entry['substages']['substage_name'] contains the substage data
- Access optimization_process: substage_data.get('optimization_process', [])
- Compare metrics for the SAME stage across designs
- Include units for numeric values
- Use descriptive keys in answer dictionaries (design names as keys)
- Return answer as a list with ONE item: [{{"question": str, "answer": value}}]

=== BEGINNING OF EXAMPLE CODE PATTERNS FOR PARSING EACH STAGE OF EACH DESIGN ===
{multi_file_parsing_examples}

=== END OF EXAMPLE CODE PATTERNS FOR PARSING EACH STAGE OF EACH DESIGN ===

Define the extract_answer function for THIS question:
```python
def extract_answer({design_params}, questions: list[str]) -> list:
    \"\"\"
    Extract answer to: {question_str}
    
    Args:
        design_1_metrics: List of dicts for design 1, each with 'design', 'stage', 'substages'
        design_2_metrics: List of dicts for design 2, each with 'design', 'stage', 'substages'
        questions: List with ONE question string
    
    Returns:
        List with ONE dict: [{{"question": str, "answer": value}}]
    \"\"\"
    qa_pairs = []
    question = questions[0]  # Only one question
    
    # TODO: Extract answer for this specific question
    # Example structure:
    # answer = {{}}
    # 
    # # Compare same stage across designs
    # for entry1 in design_1_metrics:
    #     stage_type = entry1['stage']
    #     design_1_substages = entry1['substages']
    #     
    #     # Find matching stage in design 2
    #     for entry2 in design_2_metrics:
    #         if entry2['stage'] == stage_type:
    #             design_2_substages = entry2['substages']
    #             
    #             # Extract and compare metrics
    #             for substage_name in design_1_substages:
    #                 if substage_name in design_2_substages:
    #                     metrics1 = design_1_substages[substage_name]
    #                     metrics2 = design_2_substages[substage_name]
    #                     answer[stage_type] = {{
    #                         entry1['design']: metrics1.get('some_metric'),
    #                         entry2['design']: metrics2.get('some_metric')
    #                     }}
    # 
    # qa_pairs.append({{"question": question, "answer": answer}})
    # return qa_pairs
    
    pass  # Replace with actual extraction logic
```

Output ONLY the complete Python code for the extract_answer function.
Make sure the code is properly formatted and executable.
"""
        
        try:
            # Generate code for this question
            # print("code_prompt:", code_prompt)
            # pdb.set_trace()
            code_output = _tracked_llm_call(
                system_prompt="You are an expert Python developer specializing in data extraction from EDA metrics.",
                user_prompt=code_prompt,
                temperature=0.3,
            )
            
            # Extract code
            code = _extract_code_block(code_output)
            if not code:
                print(f"  Q{i+1}: Failed to extract code from LLM response, skipping...")
                continue
            
            all_generation_codes.append({
                "question": question_str,
                "code": code
            })
            
            print(f"  Q{i+1}: Generated extraction code ({len(code)} chars)")
            
            # Step 2b: Execute the code to get answer
            local_scope = {"List": List, "Dict": Dict, "Any": Any, "json": json}
            
            try:
                exec(code, local_scope)
            except Exception as e:
                print(f"  Q{i+1}: Failed to execute generated code: {e}")
                if rejection_stats:
                    rejection_stats.record_code_failure(error=e, question=question_str, code=code)
                continue
            
            extract_answer_func = local_scope.get("extract_answer")
            if not extract_answer_func:
                print(f"  Q{i+1}: Generated code does not define extract_answer function, skipping...")
                if rejection_stats:
                    rejection_stats.record_code_failure(
                        error="extract_answer function not defined",
                        question=question_str,
                        code=code
                    )
                continue
            
            try:
                qa_pairs = extract_answer_func(*design_metrics_lists, [question_str])
                if not qa_pairs or len(qa_pairs) == 0:
                    print(f"  Q{i+1}: extract_answer returned no results, skipping...")
                    if rejection_stats:
                        rejection_stats.record_invalid_answer(
                            answer=None,
                            reason="extract_answer returned no results",
                            question=question_str
                        )
                    continue
                
                qa = qa_pairs[0]  # Get first (and only) QA pair
                answer = qa.get("answer")
                
                if answer is None or answer in ["", "None", "null", "NoneType"]:
                    print(f"  Q{i+1}: Warning - no answer or invalid answer")
                    if rejection_stats:
                        rejection_stats.record_invalid_answer(
                            answer=answer,
                            reason="None or empty answer value",
                            question=question_str
                        )
                    continue
                
                # Validate code-extracted answer has no n/a or null (recursive)
                if not validate_extract_answer(answer):
                    print(f"  Q{i+1}: Warning - code-extracted answer contains None or n/a values, skipping...")
                    if rejection_stats:
                        rejection_stats.record_invalid_answer(
                            answer=answer,
                            reason="Code-extracted answer contains None or n/a values",
                            question=question_str
                        )
                    continue
                
            except Exception as e:
                print(f"  Q{i+1}: extract_answer function failed: {e}")
                if rejection_stats:
                    rejection_stats.record_code_failure(error=e, question=question_str, code=code)
                continue
            
            # Get answer preview
            answer_preview = str(answer)[:100]
            answer_type = type(answer).__name__
            print(f"  Q{i+1}: Code-extracted answer [{answer_type}]: {answer_preview}...")
            
        except Exception as e:
            print(f"  Q{i+1}: Error during code generation/execution: {e}")
            continue
        
        # Step 2c: Generate direct LLM answer for validation
        print(f"  Q{i+1}: Generating direct LLM answer for validation...")
        
        try:
            # Create combined JSON context from all designs for multi-file comparison
            # Restructure stage_comparisons to use design names as keys instead of array indices
            stage_comparisons_by_design = {}
            for stage_comp in sampled_stage_comparisons:
                stage_type = stage_comp["stage_type"]
                if stage_type not in stage_comparisons_by_design:
                    stage_comparisons_by_design[stage_type] = {}
                
                for design_data in stage_comp["designs"]:
                    design_name = design_data["design_name"]
                    stage_comparisons_by_design[stage_type][design_name] = design_data["stage_data"]
            
            combined_context = {
                "designs": {
                    design_names[idx]: designs_metrics.get(design_names[idx], [])
                    for idx in range(len(design_names))
                },
                "stage_comparisons": stage_comparisons_by_design
            }
            
            filtered_json_obj = filter_json_by_similarity(
                question=question_str,
                full_json=combined_context,
                embedding_model=embedding_model,
                top_k=30,  # Keep more paths for multi-file comparisons
                min_similarity=0.0,
                preserve_structure=True
            )
            
            # Group filtered results by design for better readability
            filtered_json_obj = sort_filtered_results_by_design(filtered_json_obj, design_names)
            
            filtered_json_str = json.dumps(filtered_json_obj, indent=2)
            
            # Include code answer in context for guided validation
            validation_context = f"""Code-extracted answer: {answer}

Original JSON context (filtered by relevance to question):
{filtered_json_str}"""

            # Generate direct answer from LLM
            direct_llm_answer = generate_answer_directly_through_llm(
                question=question_str,
                background=OpenRoadBackground,
                context_data=validation_context,
                design_name="multi-file comparison",
                temperature=0.3
            )
            print(f"  Q{i+1}: Direct LLM answer: {str(direct_llm_answer)[:100]}...")
            
            # NOW validate the direct LLM answer for None/n/a values
            if not validate_extract_answer(direct_llm_answer):
                print(f"  Q{i+1}: Direct LLM answer contains None or n/a values, skipping...")
                if rejection_stats:
                    rejection_stats.record_invalid_answer(
                        answer=direct_llm_answer,
                        reason="Direct LLM answer contains None or n/a values",
                        question=question_str
                    )
                continue
            
            # Validate code answer against direct LLM answer
            score = validate_direct_vs_code_answer(direct_llm_answer, answer, question_str)
            
            if score >= 0.8:
                # Also validate code-extracted answer has no n/a or null
                if not validate_extract_answer(answer):
                    print(f"  Q{i+1}: Code-extracted answer contains None or n/a values, skipping...")
                    if rejection_stats:
                        rejection_stats.record_invalid_answer(
                            answer=answer,
                            reason="Code-extracted answer contains None or n/a values",
                            question=question_str
                        )
                else:
                    print(f"  Q{i+1}: Validation score: {score:.2f} - Valid answer ✓")
                    
                    final_qa_pairs.append({
                        "question": question_str,
                        "answer": direct_llm_answer,  # Use LLM answer as it passed validation
                        "difficulty": difficulty,
                    })
                    if rejection_stats:
                        rejection_stats.record_success()
            else:
                print(f"  Q{i+1}: Validation score: {score:.2f} - Low score, skipping...")
                if rejection_stats:
                    rejection_stats.record_low_score(
                        direct_answer=direct_llm_answer,
                        code_answer=answer,
                        score=score,
                        question=question_str
                    )
        
        except Exception as e:
            print(f"  Q{i+1}: Error during LLM validation: {e}")
            continue
    
    if not final_qa_pairs:
        raise QAGenerationError("No valid QA pairs generated after validation")
    
    print(f"[multi_file_qa_pair] Generated {len(final_qa_pairs)} valid QA pairs")
    
    # Build result
    return {
        "type": "multi_file",
        "stages_compared": sampled_stage_types,
        "stage_comparisons": sampled_stage_comparisons,
        "files_info": [
            {
                "design_name": data.get("design_name"),
                "file_path": data.get("file_path"),
                "log_file": data.get("log_file"),  # Include original log file path
            }
            for data in stage_data_list
        ],
        "qa_pairs": final_qa_pairs,
        "source_paths": [str(p) for p in metrics_json_paths],
        "generation_codes": all_generation_codes,  # Store all generated codes
    }


def flow_execution_multi_file_qa_pair(
    metrics_json_paths: List[str | Path],
    *,
    seed: Optional[int] = None,
    temperature: float = 0.7,
    rejection_stats: Optional[RejectionStats] = None,
) -> Dict[str, Any]:
    """
    Generate QA pairs by comparing flow execution summaries across multiple designs.
    
    This function compares high-level flow execution characteristics across different
    designs, focusing on:
    - Stage completion and execution order
    - Total number of stages executed
    - Sub-designs processed
    - Execution success/failure status
    - Overall flow structure differences
    
    Design identification uses composite_design_id (design_name + technology_node + timestamp)
    when available, allowing distinction between different runs of the same design.
    
    Args:
        metrics_json_paths: List of paths to aggregate metrics JSON files
                           (must contain flow_execution_summary)
        seed: Optional random seed for reproducibility
        temperature: LLM temperature for generation (default: 0.7)
    
    Returns:
        A dict with the question set and answers for flow execution comparison:
        - id: Question ID (assigned by caller)
        - type: "flow_execution_multi_file"
        - files_info: List of file information
        - qa_pairs: List of question-answer pairs
        - source_paths: List of source file paths
    
    Example questions generated:
    - "Which design executed more stages?"
    - "What stages are common across all designs?"
    - "How many sub-designs were processed in each design?"
    - "Which designs completed successfully?"
    
    Raises:
        QAGenerationError: If files don't contain flow_execution_summary or are incompatible
    """
    rng = random.Random(seed)
    
    if len(metrics_json_paths) < 2:
        raise QAGenerationError("Need at least 2 metric files for comparison")
    
    # Load all flow execution summaries
    print(f"\n[flow_execution_multi_file_qa_pair] Loading {len(metrics_json_paths)} metric files...")
    flow_summaries = []
    design_names = []
    
    for path in metrics_json_paths:
        payload = _read_json(path)
        
        if not isinstance(payload, dict):
            raise QAGenerationError(f"Expected dict JSON payload in {path}")
        
        if "flow_execution_summary" not in payload:
            raise QAGenerationError(
                f"File {path} does not contain 'flow_execution_summary'. "
                f"Only aggregate metrics files (all_stages_metrics.json) are supported."
            )
        
        flow_summary = payload["flow_execution_summary"]
        
        # Create composite design ID from available fields
        import re
        design_name = flow_summary.get("composite_design_id")
        
        # If no composite_design_id, create it from individual fields
        if not design_name:
            base_design_name = flow_summary.get("design_name")
            technology_node = flow_summary.get("technology_node")
            timestamp = payload.get("timestamp")  # Top-level timestamp
            
            if base_design_name and technology_node and timestamp:
                clean_ts = re.sub(r'[^\w\-]', '_', str(timestamp))
                design_name = f"{base_design_name}_{technology_node}_{clean_ts}"
            else:
                design_name = base_design_name or "unknown"
        
        design_names.append(design_name)
        flow_summaries.append({
            "design_name": design_name,
            "flow_summary": flow_summary,
            "file_path": str(path),
            "log_file": flow_summary.get("log_file"),  # Extract original log file path
        })
        
        print(f"  - Loaded: {design_name} ({flow_summary.get('total_stages', 0)} stages)")
    
    # Build comparison JSON organized by design
    comparison_json = []
    for flow_data in flow_summaries:
        design_name = flow_data["design_name"]
        flow_summary = flow_data["flow_summary"]
        
        # Extract key information from flow summary
        comparison_json.append({
            "design": design_name,
            "technology_node": flow_summary.get("technology_node") or flow_summary.get("platform"),
            "total_stages": flow_summary.get("total_stages"),
            "execute_successfully": flow_summary.get("execute_successfully"),
            "exit_code": flow_summary.get("exit_code"),
            "errors_detected": flow_summary.get("errors_detected", []),
            "stages": [
                {
                    "sequence_number": s.get("sequence_number"),
                    "stage_type": s.get("stage_type"),
                    "occurrence": s.get("occurrence"),
                    "sub_design_name": s.get("sub_design_name"),
                }
                for s in flow_summary.get("stages", [])
            ]
        })
    
    comparison_json_str = json.dumps(comparison_json, indent=2)
    
    # Build description for prompt
    design_desc = ", ".join(design_names)
    
    print(f"\n[QA Generation] Generating flow execution comparison questions...")
    print(f"  Designs: {design_desc}")
    
    # Step 1: Generate questions comparing flow execution across designs
    question_prompt = f"""You are generating question-answer pairs for comparing OpenROAD EDA design flow execution across different designs.

# OpenROAD EDA log file background:
{OpenRoadBackground}

You are comparing the overall FLOW EXECUTION across multiple designs:

Designs: {design_desc}

Flow execution data for each design:
```json
{comparison_json_str}
```

Generate a set of interesting, specific questions that compare the overall flow execution across these designs. No more than 6 questions.
Try to explore different questions from different aspects.
Imagine you are a designer comparing different design flows. What questions would you ask to understand how the flow execution differs across designs?

Focus on:
- Total number of stages executed in each design
- Stage types and execution order comparison
- Sub-designs processed in each design
- Execution success/failure status (execute_successfully, exit_code)
- Errors detected during flow execution
- Sequence and occurrence patterns

IMPORTANT RULES:
- Ask questions that compare flow-level characteristics across designs
- Focus on high-level flow structure, not detailed metrics
- Use natural language stage names: "synthesis", "placement", "routing", etc.
- Compare execution success, stage counts, and overall flow structure
- Ask about differences in stage sequences and sub-design processing
- Focus on meaningful differences in flow execution patterns
- NEVER ask questions about the number of lines extracted per stage - this is implementation detail, not meaningful design information 
"""

    output_format = """Output the question between ```question and ``` tags.
```question
<Q1> ...
Difficulty: <easy, medium, hard>
```
    
```question
<Q2> ...
Difficulty: <easy, medium, hard>
```
...
```question
<Qn> ...
Difficulty: <easy, medium, hard>
```
"""
    
    question = _tracked_llm_call(
        system_prompt="You are an expert in chip design and EDA tools, specializing in design flow analysis and cross-design comparison.",
        user_prompt=question_prompt + "\n" + output_format,
        temperature=temperature,
    )
    
    # Parse questions with difficulty
    questions = re.findall(r"```question\s*(.*?)\s*```", question, re.DOTALL)
    parsed_questions = []
    for q_block in questions:
        lines = q_block.strip().split("\n")
        q_text = []
        difficulty = "easy"  # Flow execution questions are typically easier
        
        for line in lines:
            if line.strip().lower().startswith("difficulty:"):
                difficulty = line.split(":", 1)[1].strip().lower()
            else:
                q_text.append(line.strip())
        
        question_text = " ".join(q_text).strip()
        if question_text:
            parsed_questions.append({"text": question_text, "difficulty": difficulty})
    
    if not parsed_questions:
        raise QAGenerationError("No valid questions parsed from LLM output")
    
    print(f"[flow_execution_multi_file_qa_pair] Parsed {len(parsed_questions)} questions")
    
    # Build function signature with design-specific parameters
    design_params = ", ".join([f"design_{i+1}_flow" for i in range(len(design_names))])
    
    # Prepare flow data for function calls
    flow_data_list = [comparison_json[i] for i in range(len(design_names))]
    
    # Get embedding model for filtering (reuse across questions)
    embedding_model = get_embedding_model(model_type="sentence-transformer")
    
    # Step 2: Process each question individually
    final_qa_pairs = []
    all_generation_codes = []  # Store all generated codes for debugging
    
    for i, pq in enumerate(parsed_questions):
        question_str = pq["text"]
        difficulty = pq["difficulty"]
        
        print(f"\n[flow_execution_multi_file_qa_pair] Processing Q{i+1}/{len(parsed_questions)}: {question_str[:80]}...")
        
        # Step 2a: Generate extraction code for THIS question
        code_prompt = f"""You are generating Python code to extract the answer from multi-file flow execution comparison.

Designs: {design_desc}
Question: {question_str}

The data is provided as separate variables for each design:
```json
{comparison_json_str}
```

Write Python code that extracts the answer to THIS SPECIFIC QUESTION from the flow execution data.

Requirements:
- Each design_X_flow variable is a dict with keys: 'design', 'technology_node', 'total_stages', 'execute_successfully', 'exit_code', 'errors_detected', 'stages'
- The 'stages' field is a list of dicts with: 'sequence_number', 'stage_type', 'occurrence', 'sub_design_name'
- Compare flow-level characteristics across designs
- Include appropriate units and context in answers
- Use descriptive keys in answer dictionaries (design names as keys)
- Return answer as a list with ONE item: [{{"question": str, "answer": value}}]

Define the extract_answer function for THIS question:
```python
def extract_answer({design_params}, questions: list[str]) -> list:
    \"\"\"
    Extract answer to: {question_str}
    
    Args:
        design_1_flow: Dict for design 1 with flow execution data
        design_2_flow: Dict for design 2 with flow execution data
        questions: List with ONE question string
    
    Returns:
        List with ONE dict: [{{"question": str, "answer": value}}]
    \"\"\"
    qa_pairs = []
    question = questions[0]  # Only one question
    
    # TODO: Extract answer for this specific question
    # Example structure:
    # answer = {{}}
    # 
    # # Compare flow characteristics across designs
    # answer[design_1_flow['design']] = design_1_flow.get('total_stages')
    # answer[design_2_flow['design']] = design_2_flow.get('total_stages')
    # 
    # qa_pairs.append({{"question": question, "answer": answer}})
    # return qa_pairs
    
    pass  # Replace with actual extraction logic
```

Output ONLY the complete Python code for the extract_answer function.
Make sure the code is properly formatted and executable.
"""
        
        try:
            # Generate code for this question
            code_output = _tracked_llm_call(
                system_prompt="You are an expert Python developer specializing in data extraction from EDA flow execution data.",
                user_prompt=code_prompt,
                temperature=0.3,
            )
            
            # Extract code
            code = _extract_code_block(code_output)
            if not code:
                print(f"  Q{i+1}: Failed to extract code from LLM response, skipping...")
                continue
            
            all_generation_codes.append({
                "question": question_str,
                "code": code
            })
            
            print(f"  Q{i+1}: Generated extraction code ({len(code)} chars)")
            
            # Step 2b: Execute the code to get answer
            local_scope = {"List": List, "Dict": Dict, "Any": Any, "json": json}
            
            try:
                exec(code, local_scope)
            except Exception as e:
                print(f"  Q{i+1}: Failed to execute generated code: {e}")
                if rejection_stats:
                    rejection_stats.record_code_failure(error=e, question=question_str, code=code)
                continue
            
            extract_answer_func = local_scope.get("extract_answer")
            if not extract_answer_func:
                print(f"  Q{i+1}: Generated code does not define extract_answer function, skipping...")
                if rejection_stats:
                    rejection_stats.record_code_failure(
                        error="extract_answer function not defined",
                        question=question_str,
                        code=code
                    )
                continue
            
            try:
                qa_pairs = extract_answer_func(*flow_data_list, [question_str])
                if not qa_pairs or len(qa_pairs) == 0:
                    print(f"  Q{i+1}: extract_answer returned no results, skipping...")
                    if rejection_stats:
                        rejection_stats.record_invalid_answer(
                            answer=None,
                            reason="extract_answer returned no results",
                            question=question_str
                        )
                    continue
                
                qa = qa_pairs[0]  # Get first (and only) QA pair
                answer = qa.get("answer")
                
                if answer is None or answer in ["", "None", "null", "NoneType"]:
                    print(f"  Q{i+1}: Warning - no answer or invalid answer")
                    if rejection_stats:
                        rejection_stats.record_invalid_answer(
                            answer=answer,
                            reason="None or empty answer value",
                            question=question_str
                        )
                    continue
                
                # Validate code-extracted answer has no n/a or null (recursive)
                if not validate_extract_answer(answer):
                    print(f"  Q{i+1}: Warning - code-extracted answer contains None or n/a values, skipping...")
                    if rejection_stats:
                        rejection_stats.record_invalid_answer(
                            answer=answer,
                            reason="Code-extracted answer contains None or n/a values",
                            question=question_str
                        )
                    continue
                
            except Exception as e:
                print(f"  Q{i+1}: extract_answer function failed: {e}")
                if rejection_stats:
                    rejection_stats.record_code_failure(error=e, question=question_str, code=code)
                continue
            
            # Get answer preview
            answer_preview = str(answer)[:100]
            answer_type = type(answer).__name__
            print(f"  Q{i+1}: Code-extracted answer [{answer_type}]: {answer_preview}...")
            
        except Exception as e:
            print(f"  Q{i+1}: Error during code generation/execution: {e}")
            continue
        
        # Step 2c: Generate direct LLM answer for validation
        print(f"  Q{i+1}: Generating direct LLM answer for validation...")
        
        try:
            # Create combined flow execution context
            combined_context = {
                "designs": {
                    design_names[idx]: comparison_json[idx]
                    for idx in range(len(design_names))
                }
            }
            
            filtered_json_obj = filter_json_by_similarity(
                question=question_str,
                full_json=combined_context,
                embedding_model=embedding_model,
                top_k=30,  # Keep more paths for flow comparisons
                min_similarity=0.0,
                preserve_structure=True
            )
            
            # Group filtered results by design for better readability
            filtered_json_obj = sort_filtered_results_by_design(filtered_json_obj, design_names)
            
            filtered_json_str = json.dumps(filtered_json_obj, indent=2)
            
            # Include code answer in context for guided validation
            validation_context = f"""Code-extracted answer: {answer}

Original JSON context (filtered by relevance to question):
{filtered_json_str}"""

            # Generate direct answer from LLM
            direct_llm_answer = generate_answer_directly_through_llm(
                question=question_str,
                background=OpenRoadBackground,
                context_data=validation_context,
                design_name="flow execution comparison",
                temperature=0.3
            )
            print(f"  Q{i+1}: Direct LLM answer: {str(direct_llm_answer)[:100]}...")
            
            # NOW validate the direct LLM answer for None/n/a values
            if not validate_extract_answer(direct_llm_answer):
                print(f"  Q{i+1}: Direct LLM answer contains None or n/a values, skipping...")
                if rejection_stats:
                    rejection_stats.record_invalid_answer(
                        answer=direct_llm_answer,
                        reason="Direct LLM answer contains None or n/a values",
                        question=question_str
                    )
                continue
            
            # Validate code answer against direct LLM answer
            score = validate_direct_vs_code_answer(direct_llm_answer, answer, question_str)
            
            if score >= 0.8:
                # Also validate code-extracted answer has no n/a or null
                if not validate_extract_answer(answer):
                    print(f"  Q{i+1}: Code-extracted answer contains None or n/a values, skipping...")
                    if rejection_stats:
                        rejection_stats.record_invalid_answer(
                            answer=answer,
                            reason="Code-extracted answer contains None or n/a values",
                            question=question_str
                        )
                else:
                    print(f"  Q{i+1}: Validation score: {score:.2f} - Valid answer ✓")
                    
                    final_qa_pairs.append({
                        "question": question_str,
                        "answer": direct_llm_answer,  # Use LLM answer as it passed validation
                        "difficulty": difficulty,
                    })
                    if rejection_stats:
                        rejection_stats.record_success()
            else:
                print(f"  Q{i+1}: Validation score: {score:.2f} - Low score, skipping...")
                if rejection_stats:
                    rejection_stats.record_low_score(
                        direct_answer=direct_llm_answer,
                        code_answer=answer,
                        score=score,
                        question=question_str
                    )
        
        except Exception as e:
            print(f"  Q{i+1}: Error during LLM validation: {e}")
            continue
    
    if not final_qa_pairs:
        raise QAGenerationError("No valid QA pairs generated after validation")
    
    print(f"[flow_execution_multi_file_qa_pair] Generated {len(final_qa_pairs)} valid QA pairs")
    
    # Build result
    return {
        "type": "flow_execution_multi_file",
        "files_info": [
            {
                "design_name": data["design_name"],
                "file_path": data["file_path"],
                "log_file": data.get("log_file"),  # Include original log file path
            }
            for data in flow_summaries
        ],
        "qa_pairs": final_qa_pairs,
        "source_paths": [str(p) for p in metrics_json_paths],
        "generation_codes": all_generation_codes,  # Store all generated codes
    }


def generate_multi_file_qa_pairs(
    metrics_json_paths: List[str | Path],
    *,
    n_questions: int = 5,
    seed: Optional[int] = None,
    qa_json_path: Optional[str] = "./qa_output",
    temperature: float = 0.7,
    num_substages: Optional[int] = None,
    stage_types: Optional[List[str]] = None,
    num_stages: Optional[int] = None,
    write_output: bool = True,
) -> Dict[str, Any]:
    """
    Generate QA pairs comparing stages across multiple metric files.
    
    This is the main entry point for generating cross-file comparison questions.
    
    Supports both:
    1. Single-stage metrics files (e.g., "design_routing_metrics.json")
    2. Aggregate all-stages files (e.g., "design_all_stages_metrics.json")
       - Automatically samples and compares common stages
    
    Args:
        metrics_json_paths: List of paths to stage metrics JSON files
        n_questions: Target number of QA pairs to generate (default: 5)
        seed: Optional random seed for reproducibility
        qa_json_path: Output directory path (default: "./qa_output")
        temperature: LLM temperature for generation (default: 0.7)
        num_substages: Number of substages to compare. If None, randomly choose 1-3.
        stage_types: Optional list of stage types to filter by (e.g., ['routing', 'placement'])
        num_stages: Number of stage types to sample (1-3). If None, randomly choose.
        write_output: Whether to write output files (default: True)
    
    Returns:
        Dict with:
        - stages: List of stage types being compared
        - designs: List of design names
        - questions: List of QA dicts with metadata
        - summary: Statistics about generated QA pairs
    
    Example:
        >>> result = generate_multi_file_qa_pairs(
        ...     ["ariane133_all_stages_metrics.json", "gcd_all_stages_metrics.json"],
        ...     n_questions=5,
        ...     num_stages=2,
        ...     seed=42
        ... )
        >>> print(f"Generated {result['summary']['total_qa_pairs']} QA pairs")
    """
    rng = random.Random(seed)
    
    # Create rejection statistics tracker
    rejection_stats = RejectionStats()
    
    # Track used stage combinations to avoid repetition
    used_stage_combinations = []
    
    # Generate multiple QA sets
    print(f"\n{'='*80}")
    print(f"Generating Multi-File QA Pairs")
    print(f"{'='*80}")
    print(f"  Files: {len(metrics_json_paths)}")
    for path in metrics_json_paths:
        print(f"    - {Path(path).name}")
    print(f"  Target QA pairs: {n_questions}")
    print(f"{'='*80}\n")
    
    all_qa_sets = []
    total_qa_pairs = 0
    stages_compared = []
    design_names = []
    log_files = []
    
    for i in range(n_questions):
        print(f"\n--- Generating QA Set {i+1}/{n_questions} ---")
        
        try:
            # Use different seed for each iteration if seed is provided
            iteration_seed = (seed + i) if seed is not None else None
            
            qa_set = multi_file_qa_pair(
                metrics_json_paths,
                seed=iteration_seed,
                temperature=temperature,
                num_substages=num_substages,
                stage_types=stage_types,
                num_stages=num_stages,
                rejection_stats=rejection_stats,
                exclude_stage_combinations=used_stage_combinations,
            )
            
            qa_set["id"] = f"Q_MULTIFILE_{i+1:03d}"
            
            # Extract metadata (use first set's metadata for summary)
            if i == 0:
                stages_compared = qa_set.get("stages_compared", [])
                design_names = [f["design_name"] for f in qa_set.get("files_info", [])]
                log_files = [f.get("log_file") for f in qa_set.get("files_info", [])]
            
            # Track this stage combination to avoid repetition
            current_stages = tuple(sorted(qa_set.get("stages_compared", [])))
            used_stage_combinations.append(current_stages)
            
            qa_count = len(qa_set.get("qa_pairs", []))
            total_qa_pairs += qa_count
            
            print(f"  ✓ Generated {qa_count} QA pairs comparing {', '.join(qa_set.get('stages_compared', []))} across designs")
            
            all_qa_sets.append(qa_set)
            
        except Exception as e:
            print(f"  ✗ Failed to generate QA set {i+1}: {e}")
            # Continue to next iteration instead of failing completely
            continue
    
    if not all_qa_sets:
        raise QAGenerationError("Failed to generate any QA sets")
    
    stats = {
        "total_sets": len(all_qa_sets),
        "total_qa_pairs": total_qa_pairs,
        "stages_compared": stages_compared,
        "designs": design_names,
    }
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Generation Summary")
    print(f"{'='*80}")
    print(f"  Designs:           {', '.join(stats['designs'])}")
    print(f"  Stages compared:   {', '.join(stats['stages_compared'])}")
    print(f"  Total QA Sets:     {stats['total_sets']:3d}")
    print(f"  Total QA Pairs:    {stats['total_qa_pairs']:3d}")
    gen_tokens = get_qa_generation_stats()
    print(f"\n⏱️  QA Generation Time & Tokens:")
    print(f"   Total time:        {gen_tokens['total_time_seconds']:.2f} s")
    print(f"   LLM calls:         {gen_tokens['num_calls']}")
    print(f"   Prompt tokens:     {gen_tokens['total_prompt_tokens']}")
    print(f"   Output tokens:     {gen_tokens['total_output_tokens']}")
    print(f"{'='*80}\n")
    
    # Print rejection statistics
    print(f"\n{'='*80}")
    print(f"ANSWER VALIDATION & REJECTION REPORT")
    print(f"{'='*80}")
    rejection_stats.print_summary(prefix="")
    print(f"{'='*80}\n")
    
    # Create output directory if needed
    output_dir = Path(qa_json_path).resolve()
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write results with timestamp (only if write_output is True)
    if write_output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Construct output filename
        design_prefix = "_vs_".join([d[:15] for d in design_names[:3]])
        stages_suffix = "_".join(stats['stages_compared']) if stats['stages_compared'] else "all"
        
        # 1. Write raw data (all QA sets)
        out_path_raw = output_dir / f"{design_prefix}_{stages_suffix}_{timestamp}_qa_multifile_raw.json"
        with out_path_raw.open("w", encoding="utf-8") as f:
            json.dump({
                "stages_compared": stats['stages_compared'],
                "designs": design_names,
                "source_files": [str(p) for p in metrics_json_paths],
                "log_files": log_files,  # Add original log file paths
                "timestamp": timestamp,
                "statistics": stats,
                "rejection_statistics": rejection_stats.get_summary(),
                "generation_time_and_tokens": get_qa_generation_stats(),
                "generation_params": {
                    "n_questions": n_questions,
                    "seed": seed,
                    "temperature": temperature,
                    "num_substages": num_substages,
                    "stage_types": stage_types,
                    "num_stages": num_stages,
                },
                "question_sets": all_qa_sets,  # Changed from single qa_set to list
            }, f, indent=2, ensure_ascii=False)
        print(f"[OK] Wrote {len(all_qa_sets)} QA sets (raw) → {out_path_raw}")
        
        # 2. Write QA pairs only (flattened across all sets)
        questions_and_answers = []
        for set_idx, qa_set in enumerate(all_qa_sets):
            for pair_idx, qa in enumerate(qa_set.get("qa_pairs", [])):
                qa_dict = {
                    "id": f"Q_MULTIFILE_{set_idx+1:03d}_P{pair_idx+1:02d}",
                    "type": "multi_file",
                    "stages_compared": qa_set.get("stages_compared", stats['stages_compared']),
                    "designs": design_names,
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "difficulty": qa.get("difficulty", "medium"),
                }
                questions_and_answers.append(qa_dict)
        
        out_path = output_dir / f"{design_prefix}_{stages_suffix}_{timestamp}_qa_multifile.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump({
                "stages_compared": stats['stages_compared'],
                "designs": design_names,
                "timestamp": timestamp,
                "source_files": [str(p) for p in metrics_json_paths],
                "log_files": log_files,  # Add original log file paths
                "statistics": stats,
                "rejection_statistics": rejection_stats.get_summary(),
                "generation_time_and_tokens": get_qa_generation_stats(),
                "qa_pairs": questions_and_answers,
            }, f, indent=2, ensure_ascii=False)
        print(f"[OK] Wrote {len(questions_and_answers)} QA pairs (flattened) → {out_path}")
    
    return {
        "stages_compared": stats['stages_compared'],
        "designs": design_names,
        "questions": all_qa_sets,  # Changed from [qa_set] to all_qa_sets
        "summary": stats,
        "rejection_stats": rejection_stats,
        "log_files": log_files,
        "metrics_json_paths": metrics_json_paths,
    }


def generate_flow_execution_multi_file_qa_pairs(
    metrics_json_paths: List[str | Path],
    *,
    seed: Optional[int] = None,
    qa_json_path: Optional[str] = "./qa_output",
    temperature: float = 0.7,
    write_output: bool = True,
) -> Dict[str, Any]:
    """
    Generate QA pairs comparing flow execution across multiple designs.
    
    This is the main entry point for generating cross-design flow execution comparison questions.
    Focuses on high-level flow characteristics rather than detailed stage metrics.
    
    Args:
        metrics_json_paths: List of paths to aggregate metrics JSON files
                           (must contain flow_execution_summary)
        seed: Optional random seed for reproducibility
        qa_json_path: Output directory path (default: "./qa_output")
        temperature: LLM temperature for generation (default: 0.7)
        write_output: Whether to write output files (default: True)
    
    Returns:
        Dict with:
        - designs: List of design names
        - questions: List of QA dicts with metadata
        - summary: Statistics about generated QA pairs
    
    Example:
        >>> result = generate_flow_execution_multi_file_qa_pairs(
        ...     ["ariane133_all_stages_metrics.json", "gcd_all_stages_metrics.json"],
        ...     seed=42
        ... )
        >>> print(f"Generated {result['summary']['total_qa_pairs']} QA pairs")
    """
    rng = random.Random(seed)
    
    # Create rejection statistics tracker
    rejection_stats = RejectionStats()
    
    # Generate one QA set
    print(f"\n{'='*80}")
    print(f"Generating Flow Execution Multi-File QA Pairs")
    print(f"{'='*80}")
    print(f"  Files: {len(metrics_json_paths)}")
    for path in metrics_json_paths:
        print(f"    - {Path(path).name}")
    print(f"{'='*80}\n")
    
    try:
        qa_set = flow_execution_multi_file_qa_pair(
            metrics_json_paths,
            seed=seed,
            temperature=temperature,
            rejection_stats=rejection_stats,
        )
        
        qa_set["id"] = "Q_FLOW_MULTIFILE_001"
        
        # Extract metadata
        design_names = [f["design_name"] for f in qa_set.get("files_info", [])]
        log_files = [f.get("log_file") for f in qa_set.get("files_info", [])]
        qa_count = len(qa_set.get("qa_pairs", []))
        
        print(f"  ✓ Generated {qa_count} QA pairs comparing flow execution across designs")
        
        stats = {
            "total_sets": 1,
            "total_qa_pairs": qa_count,
            "designs": design_names,
        }
        
    except Exception as e:
        print(f"  ✗ Failed to generate flow execution multi-file QA: {e}")
        raise
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Generation Summary")
    print(f"{'='*80}")
    print(f"  Designs:           {', '.join(stats['designs'])}")
    print(f"  Total QA Pairs:    {stats['total_qa_pairs']:3d}")
    gen_tokens = get_qa_generation_stats()
    print(f"\n⏱️  QA Generation Time & Tokens:")
    print(f"   Total time:        {gen_tokens['total_time_seconds']:.2f} s")
    print(f"   LLM calls:         {gen_tokens['num_calls']}")
    print(f"   Prompt tokens:     {gen_tokens['total_prompt_tokens']}")
    print(f"   Output tokens:     {gen_tokens['total_output_tokens']}")
    print(f"{'='*80}\n")
    
    # Print rejection statistics
    print(f"\n{'='*80}")
    print(f"ANSWER VALIDATION & REJECTION REPORT")
    print(f"{'='*80}")
    rejection_stats.print_summary(prefix="")
    print(f"{'='*80}\n")
    
    # Create output directory if needed
    output_dir = Path(qa_json_path).resolve()
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write results with timestamp (only if write_output is True)
    if write_output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Construct output filename
        design_prefix = "_vs_".join([d[:15] for d in design_names[:3]])
        
        # 1. Write raw data
        out_path_raw = output_dir / f"{design_prefix}_flow_execution_{timestamp}_qa_multifile_raw.json"
        with out_path_raw.open("w", encoding="utf-8") as f:
            json.dump({
                "designs": design_names,
                "source_files": [str(p) for p in metrics_json_paths],
                "log_files": log_files,  # Add original log file paths
                "timestamp": timestamp,
                "statistics": stats,
                "rejection_statistics": rejection_stats.get_summary(),
                "generation_time_and_tokens": get_qa_generation_stats(),
                "generation_params": {
                    "seed": seed,
                    "temperature": temperature,
                },
                "question_set": qa_set,
            }, f, indent=2, ensure_ascii=False)
        print(f"[OK] Wrote QA set (raw) → {out_path_raw}")
        
        # 2. Write QA pairs only (flattened)
        questions_and_answers = []
        for idx, qa in enumerate(qa_set.get("qa_pairs", [])):
            qa_dict = {
                "id": f"Q_FLOW_MULTIFILE_001_P{idx+1:02d}",
                "type": "flow_execution_multi_file",
                "designs": design_names,
                "question": qa["question"],
                "answer": qa["answer"],
                "difficulty": qa.get("difficulty", "easy"),
            }
            questions_and_answers.append(qa_dict)
        
        out_path = output_dir / f"{design_prefix}_flow_execution_{timestamp}_qa_multifile.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump({
                "designs": design_names,
                "timestamp": timestamp,
                "source_files": [str(p) for p in metrics_json_paths],
                "log_files": log_files,  # Add original log file paths
                "statistics": stats,
                "rejection_statistics": rejection_stats.get_summary(),
                "generation_time_and_tokens": get_qa_generation_stats(),
                "qa_pairs": questions_and_answers,
            }, f, indent=2, ensure_ascii=False)
        print(f"[OK] Wrote {len(questions_and_answers)} QA pairs (flattened) → {out_path}")
    
    return {
        "designs": design_names,
        "questions": [qa_set],
        "summary": stats,
        "rejection_stats": rejection_stats,
        "log_files": log_files,
        "metrics_json_paths": metrics_json_paths,
    }


def generate_all_multi_file_qa_pairs(
    metrics_json_paths: List[str | Path],
    *,
    n_questions: int = 5,
    seed: Optional[int] = None,
    qa_json_path: Optional[str] = "./qa_output",
    temperature: float = 0.7,
    num_substages: Optional[int] = None,
    stage_types: Optional[List[str]] = None,
    num_stages: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Generate single-stage, multi-stage, and flow-execution QA pairs for multiple designs.
    
    This is a unified function that calls:
    1. generate_multi_file_qa_pairs() with num_substages=1 - Single stage comparison across designs
    2. generate_multi_file_qa_pairs() - Multi-stage comparison across designs
    3. generate_flow_execution_multi_file_qa_pairs() - Flow execution comparison across designs
    
    All three sets of QA pairs are written to a single combined JSON file.
    
    Args:
        metrics_json_paths: List of paths to aggregate metrics JSON files
        n_questions: Target number of QA pairs per category (default: 5)
        seed: Optional random seed for reproducibility
        qa_json_path: Output directory path (default: "./qa_output")
        temperature: LLM temperature for generation (default: 0.7)
        num_substages: Number of substages to compare in multi-stage. If None, randomly choose 1-3.
        stage_types: Optional list of stage types to filter by (e.g., ['routing', 'placement'])
        num_stages: Number of stage types to sample for multi-stage (1-3). If None, randomly choose.
    
    Returns:
        Dict with:
        - designs: List of design names
        - single_stage_comparison: Results from single-stage comparison QA generation
        - multi_stage_comparison: Results from multi-stage comparison QA generation
        - flow_execution: Results from flow execution QA generation
        - summary: Combined statistics
        - output_files: Paths to generated JSON files
    
    Example:
        >>> result = generate_all_multi_file_qa_pairs(
        ...     ["ariane133_all_stages_metrics.json", "gcd_all_stages_metrics.json"],
        ...     n_questions=5,
        ...     num_stages=2,
        ...     seed=42
        ... )
        >>> print(f"Total QA pairs: {result['summary']['total_qa_pairs']}")
    """
    print(f"\n{'='*80}")
    print(f"UNIFIED MULTI-FILE QA GENERATION")
    print(f"{'='*80}")
    print(f"  Files: {len(metrics_json_paths)}")
    for path in metrics_json_paths:
        print(f"    - {Path(path).name}")
    print(f"  Output: {qa_json_path}")
    print(f"{'='*80}\n")
    
    # Extract design names and log files using load_stage_metrics to get composite IDs
    # This ensures we get design_name + technology_node + timestamp format
    try:
        design_names = []
        log_files = []
        for path in metrics_json_paths:
            stage_data = load_stage_metrics(path)
            # Use composite design ID (already contains tech node + timestamp for aggregate files)
            design_names.append(stage_data.get("design_name", Path(path).stem))
            log_files.append(stage_data.get("log_file"))
    except Exception as e:
        design_names = [Path(p).stem for p in metrics_json_paths]
        log_files = [None] * len(metrics_json_paths)
    
    # Generate single stage comparison QA pairs
    print(f"\n[1/3] Generating Single Stage Comparison QA Pairs...")
    print(f"{'='*80}")
    try:
        single_stage_result = generate_multi_file_qa_pairs(
            metrics_json_paths,
            n_questions=n_questions,
            seed=seed,
            qa_json_path=qa_json_path,
            temperature=temperature,
            num_substages=1, # set to 1 substage only
            stage_types=stage_types,
            num_stages=1,  # set to 1 stage only for single-stage comparison
            write_output=False,  # Don't write individual files in "all" mode
        )
        single_stage_qa_count = single_stage_result['summary']['total_qa_pairs']
        print(f"  ✓ Generated {single_stage_qa_count} single-stage comparison QA pairs")
    except Exception as e:
        print(f"  ✗ Failed to generate single-stage comparison QA: {e}")
        single_stage_result = None
        single_stage_qa_count = 0
    
    # Generate multi-stage comparison QA pairs
    print(f"\n[2/3] Generating Multi-Stage Comparison QA Pairs...")
    print(f"{'='*80}")
    try:
        stage_result = generate_multi_file_qa_pairs(
            metrics_json_paths,
            n_questions=n_questions,
            seed=seed,
            qa_json_path=qa_json_path,
            temperature=temperature,
            num_substages=num_substages,
            stage_types=stage_types,
            num_stages=num_stages,
            write_output=False,  # Don't write individual files in "all" mode
        )
        stage_qa_count = stage_result['summary']['total_qa_pairs']
        print(f"  ✓ Generated {stage_qa_count} multi-stage comparison QA pairs")
    except Exception as e:
        print(f"  ✗ Failed to generate multi-stage comparison QA: {e}")
        stage_result = None
        stage_qa_count = 0
    
    # Generate flow-execution QA pairs
    print(f"\n[3/3] Generating Flow-Execution QA Pairs...")
    print(f"{'='*80}")
    try:
        flow_result = generate_flow_execution_multi_file_qa_pairs(
            metrics_json_paths,
            seed=seed,
            qa_json_path=qa_json_path,
            temperature=temperature,
            write_output=False,  # Don't write individual files in "all" mode
        )
        flow_qa_count = flow_result['summary']['total_qa_pairs']
        print(f"  ✓ Generated {flow_qa_count} flow-execution QA pairs")
    except Exception as e:
        print(f"  ✗ Failed to generate flow-execution QA: {e}")
        flow_result = None
        flow_qa_count = 0
    
    # 3. Combine all QA pairs into a single JSON file
    print(f"\n{'='*80}")
    print(f"Combining Results...")
    print(f"{'='*80}")
    
    combined_qa_pairs = []
    
    # Add single-stage comparison QA pairs (unique IDs across all sets)
    if single_stage_result and single_stage_result.get('questions'):
        for set_idx, qa_set in enumerate(single_stage_result['questions']):
            for pair_idx, qa in enumerate(qa_set.get("qa_pairs", [])):
                qa_dict = {
                    "id": f"Q_MULTIFILE_SINGLE_STAGE_{set_idx+1:02d}_P{pair_idx+1:02d}",
                    "type": "multi_file_single_stage_comparison",
                    "stages_compared": qa_set.get("stages_compared", single_stage_result.get('stages_compared', [])),
                    "designs": design_names,
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "difficulty": qa.get("difficulty", "medium"),
                }
                combined_qa_pairs.append(qa_dict)
    
    # Add multi-stage comparison QA pairs (unique IDs across all sets)
    if stage_result and stage_result.get('questions'):
        for set_idx, qa_set in enumerate(stage_result['questions']):
            for pair_idx, qa in enumerate(qa_set.get("qa_pairs", [])):
                qa_dict = {
                    "id": f"Q_MULTIFILE_MULTI_STAGE_{set_idx+1:02d}_P{pair_idx+1:02d}",
                    "type": "multi_file_multi_stage_comparison",
                    "stages_compared": qa_set.get("stages_compared", stage_result.get('stages_compared', [])),
                    "designs": design_names,
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "difficulty": qa.get("difficulty", "medium"),
                }
                combined_qa_pairs.append(qa_dict)
    
    # Add flow-execution QA pairs (unique IDs across all sets)
    if flow_result and flow_result.get('questions'):
        for set_idx, qa_set in enumerate(flow_result['questions']):
            for pair_idx, qa in enumerate(qa_set.get("qa_pairs", [])):
                qa_dict = {
                    "id": f"Q_MULTIFILE_FLOW_{set_idx+1:02d}_P{pair_idx+1:02d}",
                    "type": "multi_file_flow_execution",
                    "designs": design_names,
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "difficulty": qa.get("difficulty", "easy"),
                }
                combined_qa_pairs.append(qa_dict)
    
    # 4. Write combined results
    output_dir = Path(qa_json_path).resolve()
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create design prefix from design names, extracting just design + tech node (without timestamp)
    # Composite IDs are like "aes_asap7_2026-01-27T23_49_52_530Z", we want "aes_asap7"
    def extract_design_tech(full_name: str) -> str:
        """Extract design_name + tech_node from composite ID"""
        parts = full_name.split('_')
        if len(parts) >= 2:
            # Assume format: design_tech_timestamp or just design_tech
            # Check if third part onwards looks like a timestamp (has numbers/dates)
            if len(parts) > 2 and (parts[2].startswith('20') or parts[2].isdigit()):
                # Has timestamp, take first two parts (design + tech)
                return f"{parts[0]}_{parts[1]}"
        # No timestamp or short name, use first 15 chars
        return full_name[:15]
    
    design_prefix = "_vs_".join([extract_design_tech(d) for d in design_names[:3]])
    
    # 4a. Write combined raw data (new - includes full question sets)
    out_path_combined_raw = output_dir / f"{design_prefix}_{timestamp}_qa_all_multifile_raw.json"
    
    combined_raw_output = {
        "designs": design_names,
        "timestamp": timestamp,
        "source_files": [str(p) for p in metrics_json_paths],
        "log_files": log_files,
        "generation_params": {
            "n_questions": n_questions,
            "seed": seed,
            "temperature": temperature,
            "num_substages": num_substages,
            "stage_types": stage_types,
            "num_stages": num_stages,
        },
        "statistics": {
            "total_qa_pairs": len(combined_qa_pairs),
            "single_stage_comparison_pairs": single_stage_qa_count,
            "multi_stage_comparison_pairs": stage_qa_count,
            "flow_execution_pairs": flow_qa_count,
            "single_stage_stages_compared": single_stage_result.get('stages_compared', []) if single_stage_result else [],
            "multi_stage_stages_compared": stage_result.get('stages_compared', []) if stage_result else [],
        },
        "single_stage_comparison": {
            "question_set": single_stage_result['questions'][0] if single_stage_result and single_stage_result.get('questions') else None,
            "question_sets": single_stage_result['questions'] if single_stage_result and single_stage_result.get('questions') else [],
            "rejection_statistics": single_stage_result['rejection_stats'].get_summary() if single_stage_result and 'rejection_stats' in single_stage_result else None,
        },
        "multi_stage_comparison": {
            "question_set": stage_result['questions'][0] if stage_result and stage_result.get('questions') else None,
            "question_sets": stage_result['questions'] if stage_result and stage_result.get('questions') else [],
            "rejection_statistics": stage_result['rejection_stats'].get_summary() if stage_result and 'rejection_stats' in stage_result else None,
        },
        "flow_execution": {
            "question_set": flow_result['questions'][0] if flow_result and flow_result.get('questions') else None,
            "question_sets": flow_result['questions'] if flow_result and flow_result.get('questions') else [],
            "rejection_statistics": flow_result['rejection_stats'].get_summary() if flow_result and 'rejection_stats' in flow_result else None,
        },
        "generation_time_and_tokens": get_qa_generation_stats(),
    }
    
    with out_path_combined_raw.open("w", encoding="utf-8") as f:
        json.dump(combined_raw_output, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ Wrote raw combined data (with full question sets)")
    print(f"    → {out_path_combined_raw}")
    
    # 4b. Write combined QA pairs (flattened, for easy consumption)
    out_path_combined = output_dir / f"{design_prefix}_{timestamp}_qa_all_multifile.json"
    
    combined_output = {
        "designs": design_names,
        "timestamp": timestamp,
        "source_files": [str(p) for p in metrics_json_paths],
        "log_files": log_files,  # Add original log file paths
        "generation_params": {
            "n_questions": n_questions,
            "seed": seed,
            "temperature": temperature,
            "num_substages": num_substages,
            "stage_types": stage_types,
            "num_stages": num_stages,
        },
        "statistics": {
            "total_qa_pairs": len(combined_qa_pairs),
            "single_stage_comparison_pairs": single_stage_qa_count,
            "multi_stage_comparison_pairs": stage_qa_count,
            "flow_execution_pairs": flow_qa_count,
            "single_stage_stages_compared": single_stage_result.get('stages_compared', []) if single_stage_result else [],
            "multi_stage_stages_compared": stage_result.get('stages_compared', []) if stage_result else [],
        },
        "generation_time_and_tokens": get_qa_generation_stats(),
        "qa_pairs": combined_qa_pairs,
    }
    
    with out_path_combined.open("w", encoding="utf-8") as f:
        json.dump(combined_output, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ Combined {len(combined_qa_pairs)} QA pairs into single file")
    print(f"    → {out_path_combined}")
    
    # Print final summary
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"  Designs:                         {', '.join(design_names)}")
    if single_stage_result and single_stage_result.get('stages_compared'):
        print(f"  Single-stage stages compared:    {', '.join(single_stage_result['stages_compared'])}")
    if stage_result and stage_result.get('stages_compared'):
        print(f"  Multi-stage stages compared:     {', '.join(stage_result['stages_compared'])}")
    print(f"  Single-Stage Comparison QA:      {single_stage_qa_count:3d}")
    print(f"  Multi-Stage Comparison QA:       {stage_qa_count:3d}")
    print(f"  Flow-Execution QA pairs:         {flow_qa_count:3d}")
    print(f"  Total QA pairs:                  {len(combined_qa_pairs):3d}")
    gen_tokens = get_qa_generation_stats()
    print(f"\n⏱️  QA Generation Time & Tokens:")
    print(f"   Total time:                     {gen_tokens['total_time_seconds']:.2f} s")
    print(f"   LLM calls:                      {gen_tokens['num_calls']}")
    print(f"   Prompt tokens:                  {gen_tokens['total_prompt_tokens']}")
    print(f"   Output tokens:                  {gen_tokens['total_output_tokens']}")
    print(f"  Combined output file:            {out_path_combined.name}")
    print(f"{'='*80}")
    
    # Print combined rejection statistics
    print(f"\n{'='*80}")
    print(f"COMBINED REJECTION STATISTICS")
    print(f"{'='*80}")
    
    # Aggregate rejection stats from all generators
    total_attempts = 0
    total_successful = 0
    total_code_failed = 0
    total_invalid = 0
    total_low_score = 0
    
    if single_stage_result and 'rejection_stats' in single_stage_result:
        single_stats = single_stage_result['rejection_stats'].get_summary()
        print(f"Single-Stage Comparison Generation:")
        print(f"  Attempts:     {single_stats['total_attempts']}")
        print(f"  Successful:   {single_stats['successful']}")
        print(f"  Success Rate: {single_stats['success_rate']:.1f}%")
        total_attempts += single_stats['total_attempts']
        total_successful += single_stats['successful']
        total_code_failed += single_stats['rejected']['code_execution_failed']
        total_invalid += single_stats['rejected']['invalid_answer_rejected']
        total_low_score += single_stats['rejected']['low_score_rejected']
    
    if stage_result and 'rejection_stats' in stage_result:
        stage_stats = stage_result['rejection_stats'].get_summary()
        print(f"\nMulti-Stage Comparison Generation:")
        print(f"  Attempts:     {stage_stats['total_attempts']}")
        print(f"  Successful:   {stage_stats['successful']}")
        print(f"  Success Rate: {stage_stats['success_rate']:.1f}%")
        total_attempts += stage_stats['total_attempts']
        total_successful += stage_stats['successful']
        total_code_failed += stage_stats['rejected']['code_execution_failed']
        total_invalid += stage_stats['rejected']['invalid_answer_rejected']
        total_low_score += stage_stats['rejected']['low_score_rejected']
    
    if flow_result and 'rejection_stats' in flow_result:
        flow_stats = flow_result['rejection_stats'].get_summary()
        print(f"\nFlow-Execution Generation:")
        print(f"  Attempts:     {flow_stats['total_attempts']}")
        print(f"  Successful:   {flow_stats['successful']}")
        print(f"  Success Rate: {flow_stats['success_rate']:.1f}%")
        total_attempts += flow_stats['total_attempts']
        total_successful += flow_stats['successful']
        total_code_failed += flow_stats['rejected']['code_execution_failed']
        total_invalid += flow_stats['rejected']['invalid_answer_rejected']
        total_low_score += flow_stats['rejected']['low_score_rejected']
    
    total_rejected = total_attempts - total_successful
    success_rate = (total_successful / total_attempts * 100) if total_attempts > 0 else 0.0
    
    print(f"\nOverall Combined Stats:")
    print(f"  Total Attempts:        {total_attempts}")
    print(f"  Successful:            {total_successful}")
    print(f"  Rejected Total:        {total_rejected}")
    print(f"    - Code Exec Failed:  {total_code_failed}")
    print(f"    - Low Score:         {total_low_score}")
    print(f"    - Invalid Answer:    {total_invalid}")
    print(f"  Success Rate:          {success_rate:.1f}%")
    print(f"{'='*80}\n")
    
    return {
        "designs": design_names,
        "single_stage_comparison": single_stage_result,
        "multi_stage_comparison": stage_result,
        "flow_execution": flow_result,
        "summary": {
            "total_qa_pairs": len(combined_qa_pairs),
            "single_stage_comparison_pairs": single_stage_qa_count,
            "multi_stage_comparison_pairs": stage_qa_count,
            "flow_execution_pairs": flow_qa_count,
            "single_stage_stages_compared": single_stage_result.get('stages_compared', []) if single_stage_result else [],
            "multi_stage_stages_compared": stage_result.get('stages_compared', []) if stage_result else [],
        },
        "output_files": {
            "combined": str(out_path_combined),
            "combined_raw": str(out_path_combined_raw),
        }
    }


def main() -> None:
    """
    Command-line interface for multi-file QA generation.
    """
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate QA pairs by comparing the same stage or flow execution across multiple metric JSON files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Stage-level comparison (default)
  python qa_generation_multifiles.py \\
      extracted_logs/ariane133_routing_metrics.json \\
      extracted_logs/uartgf180_routing_metrics.json

  # Flow execution comparison
  python qa_generation_multifiles.py \\
      --mode flow_execution \\
      extracted_logs/ariane133_all_stages_metrics.json \\
      extracted_logs/gcd_all_stages_metrics.json

  # Generate both stage and flow execution QA pairs (combined)
  python qa_generation_multifiles.py \\
      --mode all \\
      extracted_logs/ariane133_all_stages_metrics.json \\
      extracted_logs/gcd_all_stages_metrics.json

  # Specify number of questions and output path
  python qa_generation_multifiles.py \\
      --n_questions 10 \\
      --qa_json_path ./my_qa_output \\
      extracted_logs/design1_routing_metrics.json \\
      extracted_logs/design2_routing_metrics.json \\
      extracted_logs/design3_routing_metrics.json

  # With seed and temperature
  python qa_generation_multifiles.py \\
      --n_questions 8 \\
      --seed 123 \\
      --temperature 0.5 \\
      --qa_json_path ./results \\
      file1.json file2.json
        """
    )
    
    parser.add_argument(
        'metrics_files',
        nargs='+',
        help='Paths to metric JSON files (at least 2 required). For stage comparison, all files must be from the same stage type. For flow execution, all files must contain flow_execution_summary.'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['stage', 'flow_execution', 'all'],
        default='stage',
        help='Comparison mode: "stage" for stage-level metrics comparison, "flow_execution" for flow execution comparison, "all" for both stage and flow execution (default: stage)'
    )
    
    parser.add_argument(
        '--n_questions',
        type=int,
        default=5,
        help='Target number of QA pairs to generate (default: 5)'
    )
    
    parser.add_argument(
        '-o', '--qa_json_path',
        type=str,
        default='./qa_output',
        help='Output directory path for generated QA files (default: ./qa_output)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='LLM temperature for generation (default: 0.7)'
    )
    
    parser.add_argument(
        '--num_substages',
        type=int,
        default=None,
        help='Number of substages to compare (default: random 1-3)'
    )
    
    parser.add_argument(
        '--num_stages',
        type=int,
        default=None,
        help='Number of stage types to sample from aggregate files (1-3). If None, randomly choose.'
    )
    
    parser.add_argument(
        '--stage_types',
        type=str,
        nargs='+',
        default=None,
        help='Stage types to filter by (e.g., routing placement cts). If not specified, all stages are considered.'
    )
    
    args = parser.parse_args()
    
    # Validate at least 2 files
    if len(args.metrics_files) < 2:
        parser.error("At least 2 metric files are required for comparison")
    
    # Validate files exist
    for file_path in args.metrics_files:
        if not Path(file_path).exists():
            print(f"ERROR: File not found: {file_path}")
            sys.exit(1)
    
    # Print configuration
    print(f"\nConfiguration:")
    print(f"  Mode: {args.mode}")
    print(f"  Metric files: {len(args.metrics_files)}")
    for f in args.metrics_files:
        print(f"    - {f}")
    print(f"  Target QA pairs: {args.n_questions}")
    print(f"  Output path: {args.qa_json_path}")
    print(f"  Seed: {args.seed}")
    print(f"  Temperature: {args.temperature}")
    
    if args.mode == 'stage':
        print(f"  Num substages: {args.num_substages if args.num_substages else 'random (1-3)'}")
        print(f"  Num stages: {args.num_stages if args.num_stages else 'random (1-3)'}")
        print(f"  Stage types filter: {args.stage_types if args.stage_types else 'all stages'}")
    print()
    
    # Generate QA pairs based on mode
    if args.mode == 'flow_execution':
        result = generate_flow_execution_multi_file_qa_pairs(
            args.metrics_files,
            seed=args.seed,
            qa_json_path=args.qa_json_path,
            temperature=args.temperature,
        )
        
        print(f"\n✅ Successfully generated {result['summary']['total_qa_pairs']} QA pairs")
        print(f"   Designs: {', '.join(result['designs'])}")
    
    elif args.mode == 'all':
        result = generate_all_multi_file_qa_pairs(
            args.metrics_files,
            n_questions=args.n_questions,
            seed=args.seed,
            qa_json_path=args.qa_json_path,
            temperature=args.temperature,
            num_substages=args.num_substages,
            stage_types=args.stage_types,
            num_stages=args.num_stages,
        )
        
        print(f"\n✅ Successfully generated {result['summary']['total_qa_pairs']} total QA pairs")
        print(f"   Designs: {', '.join(result['designs'])}")
        print(f"   Single-Stage Comparison: {result['summary']['single_stage_comparison_pairs']} pairs")
        print(f"   Multi-Stage Comparison: {result['summary']['multi_stage_comparison_pairs']} pairs")
        print(f"   Flow-Execution: {result['summary']['flow_execution_pairs']} pairs")
        if result['summary'].get('single_stage_stages_compared'):
            print(f"   Single-stage stages compared: {', '.join(result['summary']['single_stage_stages_compared'])}")
        if result['summary'].get('multi_stage_stages_compared'):
            print(f"   Multi-stage stages compared: {', '.join(result['summary']['multi_stage_stages_compared'])}")
    
    else:  # stage mode
        result = generate_multi_file_qa_pairs(
            args.metrics_files,
            n_questions=args.n_questions,
            seed=args.seed,
            qa_json_path=args.qa_json_path,
            temperature=args.temperature,
            num_substages=args.num_substages,
            stage_types=args.stage_types,
            num_stages=args.num_stages,
        )
        
        print(f"\n✅ Successfully generated {result['summary']['total_qa_pairs']} QA pairs")
        print(f"   Designs: {', '.join(result['designs'])}")
        print(f"   Stages compared: {', '.join(result['stages_compared'])}")


if __name__ == "__main__":
    main()
