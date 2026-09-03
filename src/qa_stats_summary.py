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
QA Statistics Summary and Rejection Reason Clustering Analysis

This script analyzes QA generation output files and provides:
1. Statistics on generated Q&A pairs
2. Embedding-based clustering of rejection reasons
"""

import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# Try to import optional dependencies for clustering
CLUSTERING_AVAILABLE = True
try:
    import numpy as np
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.metrics import silhouette_score
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    CLUSTERING_AVAILABLE = False
    CLUSTERING_IMPORT_ERROR = str(e)

# Canonical stage order for consistent labels (synthesis -> floorplan -> placement -> cts -> routing -> final)
STAGE_ORDER = ['synthesis', 'floorplan', 'placement', 'cts', 'routing', 'final']

# Paper-style categorical columns: Flow, single stages, then n-Stages aggregates
CATEGORICAL_STAGE_COLUMNS = [
    'Flow Execution',
    'synthesis', 'floorplan', 'placement', 'cts', 'routing', 'final',
    '2-Stages', '3-Stages', '4-Stages', '5-Stages',
]


def load_design_tech_stage_categories_from_csv(
    csv_path: str,
) -> Tuple[Dict[str, Dict[str, int]], List[str]]:
    """
    Load design_tech -> (category -> count) from qa_stats_results/stage_combination_by_design_tech.csv.
    Use this to get the set of design_techs and categories for computing pass@k per category.
    Returns (by_design_tech, columns) where by_design_tech[design_tech][column] = count,
    and columns is the list of category column names (Flow Execution, synthesis, ..., 2-Stages, ..., total).
    """
    path = Path(csv_path)
    if not path.exists():
        return {}, []
    by_design_tech: Dict[str, Dict[str, int]] = {}
    columns: List[str] = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return {}, []
        # First column is design_tech; rest are category counts
        columns = [c for c in reader.fieldnames if c != 'design_tech']
        for row in reader:
            dt = row.get('design_tech', '').strip()
            if not dt:
                continue
            by_design_tech[dt] = {}
            for col in columns:
                try:
                    by_design_tech[dt][col] = int(row.get(col, 0) or 0)
                except (TypeError, ValueError):
                    by_design_tech[dt][col] = 0
    return by_design_tech, columns


class QAStatsSummary:
    """Analyze QA generation statistics and cluster rejection reasons."""
    
    def __init__(self, qa_output_dir: str = "qa_output"):
        self.qa_output_dir = Path(qa_output_dir)
        self.data = []
        self.rejection_reasons = []
        self.model = None
        
    def load_all_json_files(self) -> None:
        """Load all JSON files from qa_output directory."""
        print(f"Loading JSON files from {self.qa_output_dir}...")
        
        json_files = list(self.qa_output_dir.rglob("*.json"))
        # For multifile comparisons, prefer raw files (they have rejection stats)
        # For single files, skip raw files
        processed_files = set()
        files_to_load = []
        
        for json_file in json_files:
            if json_file.name.endswith("_raw.json"):
                # This is a raw file - check if we should use it
                base_name = json_file.name.replace("_raw.json", ".json")
                non_raw = json_file.parent / base_name
                if non_raw.exists():
                    # Both raw and non-raw exist. Prefer non-raw (flattened) for multifile
                    # so that qa_pairs have per-question "id" from the JSON.
                    if "multifile" in json_file.name:
                        processed_files.add(str(json_file))  # skip raw, load flattened below
                    else:
                        files_to_load.append(json_file)
                        processed_files.add(str(non_raw))
                else:
                    # Only raw exists
                    files_to_load.append(json_file)
            else:
                # Non-raw file (flattened); for multifile this has qa_pairs with id per item
                if str(json_file) not in processed_files:
                    files_to_load.append(json_file)
        
        print(f"Found {len(files_to_load)} JSON files\n")
        
        for idx, json_file in enumerate(sorted(files_to_load), 1):
            try:
                relative_path = json_file.relative_to(self.qa_output_dir)
                print(f"  [{idx:2d}] Loading: {relative_path}")
                
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    data['_file_path'] = str(relative_path)
                    data['_file_name'] = json_file.name
                    self.data.append(data)
                    
                    # Print file type info
                    if 'designs' in data and isinstance(data['designs'], list):
                        file_type = f"multifile ({len(data['designs'])} designs)"
                    else:
                        file_type = f"single-file ({data.get('design_name', 'unknown')})"
                    
                    qa_pairs = data.get('statistics', {}).get('total_qa_pairs', 0)
                    print(f"       Type: {file_type}, QA pairs: {qa_pairs}")
                    
                    # Extract rejection reasons
                    self._extract_rejection_reasons(data)
            except Exception as e:
                print(f"  [ERROR] Failed to load {json_file}: {e}")
        
        print()
    
    def _extract_rejection_reasons(self, data: Dict[str, Any]) -> None:
        """Extract rejection reasons from low_score_details."""
        # Handle single-file format (rejection_statistics at top level)
        if 'rejection_statistics' in data:
            self._process_rejection_stats(data, data['rejection_statistics'])
            return
        
        # Handle multifile raw format (rejection_statistics nested in sections)
        # Check stage_comparison section
        if 'stage_comparison' in data and data['stage_comparison']:
            stage_comp = data['stage_comparison']
            if isinstance(stage_comp, dict) and 'rejection_statistics' in stage_comp:
                self._process_rejection_stats(data, stage_comp['rejection_statistics'], section='stage_comparison')
        
        # Check flow_execution section  
        if 'flow_execution' in data and data['flow_execution']:
            flow_exec = data['flow_execution']
            if isinstance(flow_exec, dict) and 'rejection_statistics' in flow_exec:
                self._process_rejection_stats(data, flow_exec['rejection_statistics'], section='flow_execution')
    
    def _process_rejection_stats(self, data: Dict[str, Any], rejection_stats: Dict[str, Any], section: str = None) -> None:
        """Process rejection statistics and extract reasons."""
        if not rejection_stats:
            return
            
        low_score_details = rejection_stats.get('low_score_details', [])
        
        for detail in low_score_details:
            direct_answer = detail.get('direct_answer', '')
            code_answer = detail.get('code_answer', '')
            score = detail.get('score', 0.0)
            question = detail.get('question', '')
            
            # Extract the reason from direct_answer (usually first sentence or key phrase)
            reason = self._extract_reason_text(direct_answer, code_answer, score)
            
            if reason:
                file_path = data.get('_file_path', 'unknown')
                if section:
                    file_path = f"{file_path} [{section}]"
                
                self.rejection_reasons.append({
                    'reason': reason,
                    'score': score,
                    'question': question,
                    'direct_answer': direct_answer,
                    'code_answer': code_answer,
                    'file': file_path
                })
    
    def _extract_reason_text(self, direct_answer: str, code_answer: Any, score: float) -> str:
        """Extract a concise reason text from the direct answer."""
        if not direct_answer:
            return "No direct answer provided"
        
        # Common rejection patterns
        if "does not match" in direct_answer.lower():
            return "Code answer does not match expected answer"
        elif "does not address" in direct_answer.lower():
            return "Code answer does not address the question"
        elif "cannot be validated" in direct_answer.lower():
            return "Answer cannot be validated against JSON context"
        elif "data not available" in direct_answer.lower():
            return "Required data not available in JSON context"
        elif "not explicitly" in direct_answer.lower():
            return "Data not explicitly provided in JSON"
        elif code_answer is None or str(code_answer).lower() == 'null':
            return "Code execution returned null/None"
        elif score == 0.0:
            return "Complete mismatch or code execution failure"
        elif score < 0.5:
            return "Significant discrepancy between code and expected answer"
        elif score < 0.8:
            return "Partial match with missing information"
        else:
            # Extract first sentence as reason
            sentences = direct_answer.split('.')
            if sentences:
                return sentences[0].strip()[:200]  # Limit length
            return "Low score validation issue"
    
    def compute_statistics(self) -> Dict[str, Any]:
        """Compute comprehensive statistics from loaded data."""
        stats = {
            'total_files': len(self.data),
            'single_file_designs': 0,
            'multifile_comparisons': 0,
            'total_qa_pairs': 0,
            'by_question_type': defaultdict(int),
            'by_design': defaultdict(int),
            'rejection_stats': {
                'total_attempts': 0,
                'successful': 0,
                'code_execution_failed': 0,
                'low_score_rejected': 0,
                'invalid_answer_rejected': 0,
                'total_rejected': 0,
                'total_questions_rejected': 0,  # Total number of unique questions rejected
                'avg_success_rate': 0.0
            },
            'difficulty_distribution': defaultdict(int),
            'score_distribution': {
                '0.0': 0,
                '0.1-0.3': 0,
                '0.4-0.6': 0,
                '0.7-0.9': 0
            }
        }
        
        success_rates = []
        rejected_questions = set()  # Track unique rejected questions
        
        for data in self.data:
            # Determine if single file or multifile (by_design filled later from extracted counts)
            if 'designs' in data and isinstance(data['designs'], list):
                stats['multifile_comparisons'] += 1
            else:
                stats['single_file_designs'] += 1
            
            # Count QA pairs: use the value written by the generator in each JSON (statistics.total_qa_pairs).
            # This is the generator's count; it can differ from how many we extract (e.g. items without 'question', or different structure for raw vs non-raw).
            file_stats = data.get('statistics', {})
            qa_pairs = file_stats.get('total_qa_pairs', 0)
            stats['total_qa_pairs'] += qa_pairs
            # by_question_type is filled from extracted questions below (so prints/plots use extracted)
            
            # Rejection statistics - handle both single-file and multifile formats
            reject_stats_list = []
            
            # Single-file format: rejection_statistics at top level
            if 'rejection_statistics' in data:
                reject_stats_list.append(data['rejection_statistics'])
            
            # Multifile raw format: rejection_statistics in nested sections
            if 'stage_comparison' in data and isinstance(data['stage_comparison'], dict):
                if 'rejection_statistics' in data['stage_comparison']:
                    reject_stats_list.append(data['stage_comparison']['rejection_statistics'])
            
            if 'flow_execution' in data and isinstance(data['flow_execution'], dict):
                if 'rejection_statistics' in data['flow_execution']:
                    reject_stats_list.append(data['flow_execution']['rejection_statistics'])
            
            # Process all rejection stats found
            for reject_stats in reject_stats_list:
                if not reject_stats:
                    continue
                    
                stats['rejection_stats']['total_attempts'] += reject_stats.get('total_attempts', 0)
                stats['rejection_stats']['successful'] += reject_stats.get('successful', 0)
                
                rejected = reject_stats.get('rejected', {})
                stats['rejection_stats']['code_execution_failed'] += rejected.get('code_execution_failed', 0)
                stats['rejection_stats']['low_score_rejected'] += rejected.get('low_score_rejected', 0)
                stats['rejection_stats']['invalid_answer_rejected'] += rejected.get('invalid_answer_rejected', 0)
                stats['rejection_stats']['total_rejected'] += rejected.get('total_rejected', 0)
                
                if 'success_rate' in reject_stats:
                    success_rates.append(reject_stats['success_rate'])
                
                # Track unique rejected questions
                for detail in reject_stats.get('low_score_details', []):
                    question = detail.get('question', '')
                    if question:
                        rejected_questions.add(question)
                
                # Score distribution
                for detail in reject_stats.get('low_score_details', []):
                    score = detail.get('score', 0.0)
                    if score == 0.0:
                        stats['score_distribution']['0.0'] += 1
                    elif score <= 0.3:
                        stats['score_distribution']['0.1-0.3'] += 1
                    elif score <= 0.6:
                        stats['score_distribution']['0.4-0.6'] += 1
                    else:
                        stats['score_distribution']['0.7-0.9'] += 1
        
        # Calculate average success rate
        if success_rates:
            stats['rejection_stats']['avg_success_rate'] = sum(success_rates) / len(success_rates)
        
        # Set total unique questions rejected
        stats['rejection_stats']['total_questions_rejected'] = len(rejected_questions)
        
        # Actual extracted QA counts per file and total (used for totals, prints, and plots)
        qa_pairs_by_file = []
        total_extracted = 0
        for data in self.data:
            path = data.get('_file_path', 'unknown')
            is_multi = 'designs' in data and isinstance(data['designs'], list) and len(data.get('designs', [])) > 1
            pairs = self._get_questions_from_data(data, is_multifile=is_multi)
            extracted = len(pairs)
            statistics_total = data.get('statistics', {}).get('total_qa_pairs', 0)
            diff = statistics_total - extracted
            total_extracted += extracted
            qa_pairs_by_file.append({
                'path': path,
                'extracted': extracted,
                'statistics_total': statistics_total,
                'diff': diff,
            })
            # By-design counts from extracted (so prints/plots use extracted)
            design_key = ', '.join(data.get('designs', [])) if (is_multi and data.get('designs')) else data.get('design_name', 'unknown')
            if not design_key:
                design_key = 'multifile'
            stats['by_design'][design_key] += extracted
            # By-question-type from extracted questions (so prints match plot data)
            for q in pairs:
                q = q[0] if isinstance(q, tuple) else q
                t = q.get('type') or q.get('question_type') or 'unknown'
                if isinstance(t, str):
                    t = t.strip().lower()
                if not t:
                    t = 'unknown'
                stats['by_question_type'][t] += 1
        stats['total_qa_pairs_extracted'] = total_extracted
        stats['qa_pairs_by_file'] = qa_pairs_by_file
        # Convert defaultdicts to regular dicts (by_design and by_question_type now from extracted)
        stats['by_question_type'] = dict(stats['by_question_type'])
        stats['by_design'] = dict(stats['by_design'])
        
        return stats
    
    def cluster_rejection_reasons(self, n_clusters: int = 5, method: str = 'kmeans') -> Dict[str, Any]:
        """
        Cluster rejection reasons using embeddings.
        
        Args:
            n_clusters: Number of clusters for KMeans
            method: 'kmeans' or 'dbscan'
        """
        if not self.rejection_reasons:
            return {'error': 'No rejection reasons found'}
        
        print(f"\nClustering {len(self.rejection_reasons)} rejection reasons...")
        
        # Load sentence transformer model
        if self.model is None:
            print("Loading sentence transformer model...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Extract reason texts
        reason_texts = [r['reason'] for r in self.rejection_reasons]
        
        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.model.encode(reason_texts, show_progress_bar=True)
        
        # Perform clustering
        if method == 'kmeans':
            # Try different cluster numbers to find optimal
            if len(self.rejection_reasons) < n_clusters:
                n_clusters = max(2, len(self.rejection_reasons) // 5)
            
            clustering = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = clustering.fit_predict(embeddings)
            
            # Calculate silhouette score
            if len(set(labels)) > 1:
                silhouette = silhouette_score(embeddings, labels)
            else:
                silhouette = 0.0
        else:  # DBSCAN
            clustering = DBSCAN(eps=0.5, min_samples=2)
            labels = clustering.fit_predict(embeddings)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            silhouette = silhouette_score(embeddings, labels) if len(set(labels)) > 1 else 0.0
        
        # Organize results by cluster
        clusters = defaultdict(list)
        for idx, label in enumerate(labels):
            clusters[label].append({
                'reason': self.rejection_reasons[idx]['reason'],
                'score': self.rejection_reasons[idx]['score'],
                'question': self.rejection_reasons[idx]['question'][:100],  # Truncate for readability
                'file': self.rejection_reasons[idx]['file']
            })
        
        # Generate cluster summaries
        cluster_summaries = {}
        for cluster_id, items in clusters.items():
            # Find most common reason pattern in cluster
            reasons = [item['reason'] for item in items]
            avg_score = sum(item['score'] for item in items) / len(items)
            
            cluster_summaries[f"Cluster_{cluster_id}"] = {
                'size': len(items),
                'avg_score': avg_score,
                'representative_reason': reasons[0],  # First reason as representative
                'sample_reasons': reasons[:5],  # Top 5 samples
                'files_affected': list(set(item['file'] for item in items))[:5]
            }
        
        return {
            'method': method,
            'n_clusters': len(set(labels)),
            'silhouette_score': silhouette,
            'cluster_summaries': cluster_summaries,
            'total_reasons': len(self.rejection_reasons)
        }
    
    def _sort_stages_canonical(self, stages_list: List[str]) -> List[str]:
        """Sort stage names by canonical order: synthesis, floorplan, placement, cts, routing, final."""
        order = {s: i for i, s in enumerate(STAGE_ORDER)}
        return sorted(stages_list, key=lambda s: (order.get(s.lower(), len(STAGE_ORDER)), s))
    
    def _stage_index(self, stage_name: str) -> int:
        """Return index of stage in STAGE_ORDER (lowercase match), or len(STAGE_ORDER) if unknown."""
        s = (stage_name or '').lower().strip()
        for i, st in enumerate(STAGE_ORDER):
            if st == s:
                return i
        return len(STAGE_ORDER)
    
    def _stages_from_to(self, start_stage: str, end_stage: str = 'final') -> List[str]:
        """Return list of stages from start_stage through end_stage (inclusive) in canonical order."""
        i1 = self._stage_index(start_stage)
        i2 = self._stage_index(end_stage)
        if i1 >= len(STAGE_ORDER) or i2 >= len(STAGE_ORDER) or i1 > i2:
            return [start_stage] if i2 >= len(STAGE_ORDER) and start_stage else []
        return list(STAGE_ORDER[i1 : i2 + 1])
    
    def _expand_stage_range(self, ordered_stages: List[str]) -> List[str]:
        """
        General rule: any combination <stage1> -> <stage2> (exactly two stages) is expanded to
        the full list of stages from stage1 through stage2 in canonical order.
        E.g. [floorplan, final] -> [floorplan, placement, cts, routing, final];
        [placement, routing] -> [placement, cts, routing].
        """
        if not ordered_stages or len(ordered_stages) != 2:
            return ordered_stages
        i1 = self._stage_index(ordered_stages[0])
        i2 = self._stage_index(ordered_stages[1])
        if i1 >= len(STAGE_ORDER) or i2 >= len(STAGE_ORDER) or i1 > i2:
            return ordered_stages
        return list(STAGE_ORDER[i1 : i2 + 1])
    
    def _stages_with_occurrence_labels(self, stages_list: List[str]) -> str:
        """
        Build a stage-combination label, adding occurrence numbers when the same stage
        appears more than once (e.g. ['placement', 'placement', 'routing'] -> 'placement_1,placement_2,routing').
        Stages are ordered: synthesis, floorplan, placement, cts, routing, final.
        """
        if not stages_list:
            return ''
        from collections import Counter
        # Preserve canonical order: sort unique stages by STAGE_ORDER, then build labels in that order
        # while keeping occurrence indices for duplicates in original list
        counts = Counter(stages_list)
        # Build (stage, occurrence_index) in canonical stage order
        unique_ordered = self._sort_stages_canonical(list(counts.keys()))
        parts = []
        for s in unique_ordered:
            for occ in range(1, counts[s] + 1):
                if counts[s] > 1:
                    parts.append(f"{s}_{occ}")
                else:
                    parts.append(s)
        return ','.join(parts)
    
    def _stage_combination_category(self, q: Dict[str, Any], is_multifile: bool = False) -> str:
        """
        Return a category string for a QA item based on type and stage combination.

        Stage-combination categories:
        - flow_execution: type == flow_execution and QA does not contain 'stages'
        - single_stage:X (occurrence not shown). Multi_stage with only one stage is folded into single_stage:X and counts combined.
        - multi_stage:a,b,c with occurrence numbers when same stage appears multiple times (e.g. a_1,a_2,b)
        - stage_to_final:X->final expanded to multi_stage range when applicable

        Same logic for multi-file comparison (with comparison-style labels).
        """
        q_type = (q.get('type') or q.get('question_type') or '').strip().lower()
        stages = q.get('stages')  # list or None
        stage_singular = q.get('stage')
        occurrence = q.get('occurrence')
        try:
            occ_int = int(occurrence) if occurrence is not None else 1
        except (TypeError, ValueError):
            occ_int = 1
        stages_compared = q.get('stages_compared')  # multifile
        
        if is_multifile:
            if q_type == 'flow_execution' and not stages and not stages_compared:
                return 'flow_execution'
            if stages_compared:
                comp_list = stages_compared if isinstance(stages_compared, list) else [stages_compared]
                if len(comp_list) == 1:
                    return 'single_stage_comparison:' + str(comp_list[0])
                ordered = self._sort_stages_canonical(comp_list)
                ordered = self._expand_stage_range(ordered)
                return 'multi_stage_comparison:' + self._stages_with_occurrence_labels(ordered)
            if stages:
                stage_list = stages if isinstance(stages, list) else [stages]
                ordered = self._sort_stages_canonical(stage_list)
                ordered = self._expand_stage_range(ordered)
                if len(ordered) == 1:
                    return 'single_stage_comparison:' + str(ordered[0])
                return 'multi_stage_comparison:' + self._stages_with_occurrence_labels(ordered)
            if stage_singular:
                return 'single_stage_comparison:' + str(stage_singular)
            return q_type or 'other'
        
        if q_type == 'flow_execution' and not stages:
            return 'flow_execution'
        if q_type == 'single_stage' and stage_singular:
            return 'single_stage:' + str(stage_singular)
        if q_type == 'multi_stage' and stages:
            stage_list = stages if isinstance(stages, list) else [stages]
            ordered = self._sort_stages_canonical(stage_list)
            ordered = self._expand_stage_range(ordered)
            if len(ordered) == 1:
                return 'single_stage:' + str(ordered[0])
            return 'multi_stage:' + self._stages_with_occurrence_labels(ordered)
        if q_type == 'stage_to_final':
            s = (stage_singular or 'unknown').strip()
            expanded = self._stages_from_to(s, 'final')
            if expanded:
                return 'multi_stage:' + self._stages_with_occurrence_labels(expanded)
            s_display = stage_singular or 'unknown'
            if occ_int > 1:
                s_display += f':{occ_int}'
            return 'stage_to_final:' + str(s_display) + '->final'
        return q_type or 'other'
    
    def _get_all_qa_items_from_data(self, data: Dict[str, Any], is_multifile: bool) -> List[Dict[str, Any]]:
        """
        Return all QA-shaped items in the data blob (same locations we read from),
        including items without a 'question' field. Used to compute expected count
        and find which items are missing from extraction.
        """
        out: List[Dict[str, Any]] = []
        if is_multifile:
            top_pairs = data.get('qa_pairs', [])
            if top_pairs and isinstance(top_pairs, list):
                return list(top_pairs)
            sections = [
                ('single_stage_comparison', 'question_set', 'qa_pairs'),
                ('multi_stage_comparison', 'question_set', 'qa_pairs'),
                ('stage_comparison', 'question_set', 'qa_pairs'),
                ('flow_execution', 'question_set', 'qa_pairs'),
            ]
            for section_key, set_key, pairs_key in sections:
                section = data.get(section_key)
                if not isinstance(section, dict):
                    continue
                question_set = section.get(set_key)
                if isinstance(question_set, dict):
                    out.extend(question_set.get(pairs_key, []))
                if 'questions' in section:
                    out.extend(section.get('questions', []))
            if not out and isinstance(data.get('flow_execution'), dict):
                out.extend(data['flow_execution'].get('questions', []))
            return out
        # Single-file: flat qa_pairs or nested questions (sections with qa_pairs)
        flat_pairs = data.get('qa_pairs', [])
        if flat_pairs and isinstance(flat_pairs, list):
            first = flat_pairs[0] if flat_pairs else {}
            if isinstance(first, dict) and (first.get('question') is not None or first.get('id') is not None):
                return list(flat_pairs)
        questions = data.get('questions', [])
        if not questions:
            return []
        first = questions[0] if questions else {}
        if isinstance(first, dict) and 'qa_pairs' in first:
            out_flat = []
            for section in questions:
                if isinstance(section, dict):
                    out_flat.extend(section.get('qa_pairs', []))
            return out_flat
        return list(questions)
    
    def _get_stages_from_single_file_section(self, section: Dict[str, Any]) -> Optional[List[str]]:
        """
        Extract stage list from a single-file section only. Used for raw single-file format.
        Multifile uses stages_compared (and different section structure); never use this for multifile.
        Order: section.stages -> misc_info.stage_types -> stages_info (list of {stage}).
        """
        stages = section.get('stages')
        if stages and isinstance(stages, list) and len(stages) > 0:
            return stages
        if section.get('type') == 'multi_stage':
            misc = section.get('misc_info') or {}
            stage_types = misc.get('stage_types')
            if isinstance(stage_types, list) and stage_types:
                return stage_types
            stages_info = section.get('stages_info')
            if isinstance(stages_info, list) and stages_info:
                out = [s.get('stage') for s in stages_info if isinstance(s, dict) and s.get('stage')]
                if out:
                    return out
        return stages if stages is not None else None
    
    def _get_questions_from_data(self, data: Dict[str, Any], is_multifile: bool) -> List[Tuple[Dict[str, Any], str]]:
        """Extract (qa_item, source_path) from one data blob (single-file or multifile)."""
        out = []
        source = data.get('_file_path', 'unknown')
        
        if is_multifile:
            # Multifile: stages come from stages_compared (on item or question_set), not from misc_info/stages_info
            # Prefer top-level qa_pairs when present (flat format with stages_compared on each item)
            top_pairs = data.get('qa_pairs', [])
            if top_pairs and isinstance(top_pairs, list):
                for q in top_pairs:
                    if q.get('question'):
                        out.append((q, source))
                return out
            # Else use nested sections; stages_compared lives on question_set, not on each q
            sections = [
                ('single_stage_comparison', 'question_set', 'qa_pairs'),
                ('multi_stage_comparison', 'question_set', 'qa_pairs'),
                ('stage_comparison', 'question_set', 'qa_pairs'),
                ('flow_execution', 'question_set', 'qa_pairs'),
            ]
            for section_key, set_key, pairs_key in sections:
                section = data.get(section_key)
                if not isinstance(section, dict):
                    continue
                question_set = section.get(set_key)
                if isinstance(question_set, dict):
                    section_stages = question_set.get('stages_compared')
                    for q in question_set.get(pairs_key, []):
                        if q.get('question'):
                            need_copy = (section_stages and not q.get('stages_compared')) or (
                                (not q.get('type') and not q.get('question_type')) and section_key == 'flow_execution'
                            )
                            q_use = dict(q) if need_copy else q
                            if section_stages and not q_use.get('stages_compared'):
                                q_use['stages_compared'] = section_stages
                            if (not q_use.get('type') and not q_use.get('question_type')) and section_key == 'flow_execution':
                                q_use['type'] = 'flow_execution'
                            out.append((q_use, source))
                if 'questions' in section:
                    section_stages = section.get('stages_compared')
                    for q in section.get('questions', []):
                        if q.get('question'):
                            need_copy = (section_stages and not q.get('stages_compared')) or (
                                (not q.get('type') and not q.get('question_type')) and section_key == 'flow_execution'
                            )
                            q_use = dict(q) if need_copy else q
                            if section_stages and not q_use.get('stages_compared'):
                                q_use['stages_compared'] = section_stages
                            if (not q_use.get('type') and not q_use.get('question_type')) and section_key == 'flow_execution':
                                q_use['type'] = 'flow_execution'
                            out.append((q_use, source))
            if not out and isinstance(data.get('flow_execution'), dict):
                fe = data['flow_execution']
                fe_stages = fe.get('stages_compared')
                for q in fe.get('questions', []):
                    if q.get('question'):
                        q_use = dict(q) if (fe_stages or not q.get('type')) else q
                        if fe_stages and not q_use.get('stages_compared'):
                            q_use['stages_compared'] = fe_stages
                        if not q_use.get('type') and not q_use.get('question_type'):
                            q_use['type'] = 'flow_execution'
                        out.append((q_use, source))
            return out
        
        # Single-file: support both flat (top-level qa_pairs) and nested raw (questions = list of sections with qa_pairs)
        flat_pairs = data.get('qa_pairs', [])
        if flat_pairs and isinstance(flat_pairs, list):
            first = flat_pairs[0] if flat_pairs else {}
            if isinstance(first, dict) and (first.get('question') is not None or first.get('id') is not None):
                for q in flat_pairs:
                    if q.get('question'):
                        out.append((q, source))
                return out
        questions = data.get('questions', [])
        if not questions:
            return out
        first = questions[0] if questions else {}
        # Raw single-file format: questions is list of sections, each section has qa_pairs
        if isinstance(first, dict) and 'qa_pairs' in first:
            for section in questions:
                if not isinstance(section, dict):
                    continue
                qa_pairs = section.get('qa_pairs', [])
                sec_type = section.get('type')
                stage_singular = section.get('stage')
                # stage_to_final sections use stage_info.stage as the starting stage (e.g. routing, placement)
                if sec_type == 'stage_to_final':
                    stage_info = section.get('stage_info')
                    if isinstance(stage_info, dict):
                        stage_singular = stage_info.get('stage') or stage_singular
                # Single-file only: stages from section (stages -> misc_info.stage_types -> stages_info)
                stages = self._get_stages_from_single_file_section(section)
                occurrence = section.get('occurrence')
                section_id = section.get('id', '')
                for idx, q in enumerate(qa_pairs):
                    if not isinstance(q, dict) or not q.get('question'):
                        continue
                    q_use = dict(q)
                    if sec_type and not q_use.get('type') and not q_use.get('question_type'):
                        q_use['type'] = sec_type
                    if stage_singular is not None and not q_use.get('stage'):
                        q_use['stage'] = stage_singular
                    # Copy section stages when item has none or empty (so multi_stage is categorized as n-Stages, not Flow Execution)
                    item_stages = q_use.get('stages')
                    if stages is not None and (not item_stages or (isinstance(item_stages, list) and len(item_stages) == 0)):
                        q_use['stages'] = stages if isinstance(stages, list) else [stages]
                    if occurrence is not None and not q_use.get('occurrence'):
                        q_use['occurrence'] = occurrence
                    if section_id and not q_use.get('id'):
                        q_use['id'] = f"{section_id}_P{idx + 1:02d}"
                    out.append((q_use, source))
            return out
        for q in questions:
            if isinstance(q, dict) and q.get('question'):
                out.append((q, source))
        return out
    
    def _categorize_by_stage_combination(
        self,
        questions_with_source: List[Tuple[Dict[str, Any], str]],
        is_multifile: bool,
        max_representative: int = 3
    ) -> Dict[str, Dict[str, Any]]:
        """Group questions by stage-combination category; keep up to max_representative per category."""
        by_category = defaultdict(list)
        for q, src in questions_with_source:
            cat = self._stage_combination_category(q, is_multifile=is_multifile)
            by_category[cat].append((q, src))
        result = {}
        for cat, items in sorted(by_category.items(), key=lambda x: -len(x[1])):
            # Representative: first few, optionally prefer shorter questions
            questions_sorted = sorted(items, key=lambda x: len((x[0].get('question') or '')))
            representative = [
                {'question': (q.get('question') or '').strip()[:200], 'source': src}
                for q, src in questions_sorted[:max_representative]
            ]
            result[cat] = {'count': len(items), 'representative': representative}
        return result
    
    def print_stage_combination_summary(self, max_representative: int = 5) -> None:
        """
        Print stage-combination categorization and a few representative questions per category.

        - Single-file: categories include flow_execution (type==flow_execution, no 'stages'),
          single_stage:X, multi_stage:a,b,c, stage_to_final:X->final.
        - Multi-file: same categorization with comparison-style labels.
        - For each category, up to max_representative representative questions are printed.
        """
        single_questions = []
        multi_questions = []
        for data in self.data:
            is_multi = 'designs' in data and isinstance(data['designs'], list) and len(data.get('designs', [])) > 1
            pairs = self._get_questions_from_data(data, is_multifile=is_multi)
            if is_multi:
                multi_questions.extend(pairs)
            else:
                single_questions.extend(pairs)
        
        n_rep = max(1, max_representative)
        if single_questions:
            print("\n📐 SINGLE-FILE: QUESTIONS BY STAGE COMBINATION")
            print("-" * 60)
            cats = self._categorize_by_stage_combination(
                single_questions, is_multifile=False, max_representative=n_rep
            )
            for category, info in sorted(cats.items(), key=lambda x: -x[1]['count']):
                print(f"\n   {category}: {info['count']} question(s)")
                if info['representative']:
                    print(f"   Representative questions (up to {n_rep}):")
                    for i, rep in enumerate(info['representative'], 1):
                        q_preview = (rep['question'][:120] + '...') if len(rep['question']) > 120 else rep['question']
                        print(f"      [{i}] {q_preview}")
            print()
        
        if multi_questions:
            print("\n📐 MULTI-FILE: QUESTIONS BY STAGE COMBINATION")
            print("-" * 60)
            cats = self._categorize_by_stage_combination(
                multi_questions, is_multifile=True, max_representative=n_rep
            )
            for category, info in sorted(cats.items(), key=lambda x: -x[1]['count']):
                print(f"\n   {category}: {info['count']} question(s)")
                if info['representative']:
                    print(f"   Representative questions (up to {n_rep}):")
                    for i, rep in enumerate(info['representative'], 1):
                        q_preview = (rep['question'][:120] + '...') if len(rep['question']) > 120 else rep['question']
                        print(f"      [{i}] {q_preview}")
            print()
        
        # Print compact summary table (stage combination / flow_execution counts)
        self._print_stage_combination_table(single_questions, multi_questions)
    
    def get_stage_combination_summary(
        self,
    ) -> Dict[str, Dict[str, int]]:
        """Return question counts by stage-combination category for single-file and multi-file."""
        single_questions = []
        multi_questions = []
        for data in self.data:
            is_multi = 'designs' in data and isinstance(data['designs'], list) and len(data.get('designs', [])) > 1
            pairs = self._get_questions_from_data(data, is_multifile=is_multi)
            if is_multi:
                multi_questions.extend(pairs)
            else:
                single_questions.extend(pairs)
        cats_single = self._categorize_by_stage_combination(single_questions, is_multifile=False, max_representative=0)
        cats_multi = self._categorize_by_stage_combination(multi_questions, is_multifile=True, max_representative=0)
        return {
            'single_file': {c: info['count'] for c, info in cats_single.items()},
            'multi_file': {c: info['count'] for c, info in cats_multi.items()},
        }
    
    def _print_stage_combination_table(
        self,
        single_questions: List[Tuple[Dict[str, Any], str]],
        multi_questions: List[Tuple[Dict[str, Any], str]],
    ) -> None:
        """Print a compact table: stage combination / flow_execution -> count."""
        if not single_questions and not multi_questions:
            return
        print("\n📊 SUMMARY TABLE: Questions by stage combination / flow execution")
        print("-" * 60)
        if single_questions:
            cats = self._categorize_by_stage_combination(single_questions, is_multifile=False, max_representative=0)
            rows = sorted(cats.items(), key=lambda x: -x[1]['count'])
            print("\n   Single-file:")
            for category, info in rows:
                print(f"      {category}: {info['count']}")
        if multi_questions:
            cats = self._categorize_by_stage_combination(multi_questions, is_multifile=True, max_representative=0)
            rows = sorted(cats.items(), key=lambda x: -x[1]['count'])
            print("\n   Multi-file:")
            for category, info in rows:
                print(f"      {category}: {info['count']}")
        self._print_stage_category_coverage_check(single_questions, multi_questions)
        print()
    
    def _print_stage_category_coverage_check(
        self,
        single_questions: List[Tuple[Dict[str, Any], str]],
        multi_questions: List[Tuple[Dict[str, Any], str]],
    ) -> None:
        """
        Check that every extracted question is counted exactly once in the stage-combination
        categories. If sum of category counts != extracted count, warn about possible miscount.
        """
        if not single_questions and not multi_questions:
            return
        single_sum = 0
        multi_sum = 0
        if single_questions:
            cats = self._categorize_by_stage_combination(single_questions, is_multifile=False, max_representative=0)
            single_sum = sum(info['count'] for info in cats.values())
        if multi_questions:
            cats = self._categorize_by_stage_combination(multi_questions, is_multifile=True, max_representative=0)
            multi_sum = sum(info['count'] for info in cats.values())
        single_ok = single_sum == len(single_questions)
        multi_ok = multi_sum == len(multi_questions)
        extracted_total = len(single_questions) + len(multi_questions)
        stats = getattr(self, '_last_stats', None)
        if stats is None:
            stats = self.compute_statistics()
        stats_total = stats.get('total_qa_pairs', 0)
        extracted_from_stats = stats.get('total_qa_pairs_extracted', extracted_total)
        stats_ok = stats_total == extracted_from_stats
        print("\n   📋 Category coverage check:")
        if single_questions:
            status = "✓" if single_ok else "⚠ MISMATCH"
            print(f"      Single-file: sum of category counts = {single_sum}, extracted = {len(single_questions)} {status}")
        if multi_questions:
            status = "✓" if multi_ok else "⚠ MISMATCH"
            print(f"      Multi-file:  sum of category counts = {multi_sum}, extracted = {len(multi_questions)} {status}")
        print(f"      Total: extracted = {extracted_from_stats} (file statistics total = {stats_total}) {'✓' if stats_ok else '⚠ (some QA pairs not extracted)'}")
        if not stats_ok:
            print("      (total_qa_pairs = sum of statistics.total_qa_pairs from each JSON; extracted = QA items we actually read with a non-empty 'question' field)")
            print("      (Difference can be: items without 'question', or we load _raw for multi-file and count differently than extraction paths)")
        if (single_questions and not single_ok) or (multi_questions and not multi_ok):
            print("      Some question-answer pairs may be miscounted in categories. Run: python src/check_stage_category_coverage.py --output report.json")
        elif not stats_ok:
            self._print_missing_qa_pairs_on_the_fly(single_questions, multi_questions)
            print("      For full report: python src/check_stage_category_coverage.py --output report.json")
    
    def _qa_item_key(self, q: Dict[str, Any]) -> Tuple[Any, str]:
        """Stable key for matching QA items: (id if present, question text)."""
        return (q.get('id'), (q.get('question') or '').strip())
    
    def _print_missing_qa_pairs_on_the_fly(
        self,
        single_questions: List[Tuple[Dict[str, Any], str]],
        multi_questions: List[Tuple[Dict[str, Any], str]],
    ) -> None:
        """
        For each file, compare len(all QA items) vs extracted count and list which
        QA pairs are missing (not extracted), on the fly.
        """
        all_pairs = single_questions + multi_questions
        extracted_by_file: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for q, src in all_pairs:
            extracted_by_file[src].append(q)
        missing_any = False
        for data in self.data:
            path = data.get('_file_path', 'unknown')
            is_multi = 'designs' in data and isinstance(data['designs'], list) and len(data.get('designs', [])) > 1
            all_items = self._get_all_qa_items_from_data(data, is_multifile=is_multi)
            expected = len(all_items)
            extracted_list = extracted_by_file.get(path, [])
            extracted_count = len(extracted_list)
            if expected <= extracted_count:
                continue
            missing_any = True
            extracted_keys = {self._qa_item_key(q) for q in extracted_list}
            missing_with_reason = []
            for item in all_items:
                key = self._qa_item_key(item)
                if key in extracted_keys:
                    continue
                if not (item.get('question') or item.get('question_text') or '').strip():
                    missing_with_reason.append(('no_question', item.get('id'), None))
                else:
                    missing_with_reason.append(('not_extracted', item.get('id'), (item.get('question') or '')[:80]))
            if not missing_with_reason:
                continue
            print(f"      Missing in {path}: expected {expected}, extracted {extracted_count}, missing {len(missing_with_reason)}")
            no_q = sum(1 for r, _, _ in missing_with_reason if r == 'no_question')
            if no_q:
                print(f"        - {no_q} item(s) have no 'question' field")
            for reason, qid, preview in missing_with_reason[:5]:
                if reason == 'no_question':
                    print(f"        - id={qid} (no question field)")
                else:
                    print(f"        - id={qid} {(preview or '')[:60]}")
            if len(missing_with_reason) > 5:
                print(f"        - ... and {len(missing_with_reason) - 5} more")
    
    def check_stage_category_coverage(
        self,
    ) -> Dict[str, Any]:
        """
        Verify that the sum of stage-combination category counts equals the number of
        extracted questions, and report statistics total vs extracted. Also list
        questions that fall in 'other' and any QA-shaped items in the raw data that
        were not extracted (e.g. missing 'question' field).
        """
        single_questions = []
        multi_questions = []
        for data in self.data:
            is_multi = 'designs' in data and isinstance(data['designs'], list) and len(data.get('designs', [])) > 1
            pairs = self._get_questions_from_data(data, is_multifile=is_multi)
            if is_multi:
                multi_questions.extend(pairs)
            else:
                single_questions.extend(pairs)
        summary = self.get_stage_combination_summary()
        sum_single = sum(summary['single_file'].values())
        sum_multi = sum(summary['multi_file'].values())
        len_single = len(single_questions)
        len_multi = len(multi_questions)
        categories_match_single = sum_single == len_single
        categories_match_multi = sum_multi == len_multi
        stats = self.compute_statistics()
        statistics_total = stats.get('total_qa_pairs', 0)
        extracted_total = len_single + len_multi
        by_cat_single = defaultdict(list)
        for q, src in single_questions:
            by_cat_single[self._stage_combination_category(q, is_multifile=False)].append((q, src))
        by_cat_multi = defaultdict(list)
        for q, src in multi_questions:
            by_cat_multi[self._stage_combination_category(q, is_multifile=True)].append((q, src))
        other_single = [
            {'question_preview': (q.get('question') or '').strip()[:200], 'source': src, 'type': q.get('type'), 'stage': q.get('stage'), 'stages': q.get('stages')}
            for q, src in by_cat_single.get('other', [])
        ]
        other_multi = [
            {'question_preview': (q.get('question') or '').strip()[:200], 'source': src, 'type': q.get('type'), 'stages_compared': q.get('stages_compared'), 'stages': q.get('stages')}
            for q, src in by_cat_multi.get('other', [])
        ]
        missed_by_file = []
        for data in self.data:
            path = data.get('_file_path', 'unknown')
            expected = data.get('statistics', {}).get('total_qa_pairs', 0)
            is_multi = 'designs' in data and isinstance(data['designs'], list) and len(data.get('designs', [])) > 1
            pairs = self._get_questions_from_data(data, is_multifile=is_multi)
            extracted = len(pairs)
            no_question_count = 0
            for q in data.get('qa_pairs', []) or data.get('questions', []) or []:
                if isinstance(q, dict) and not (q.get('question') or q.get('question_text')):
                    no_question_count += 1
            missed_by_file.append({'file': path, 'statistics_total_qa_pairs': expected, 'extracted': extracted, 'diff': expected - extracted, 'items_without_question_field': no_question_count})
        return {
            'extracted_single': len_single,
            'extracted_multi': len_multi,
            'extracted_total': extracted_total,
            'sum_single_categories': sum_single,
            'sum_multi_categories': sum_multi,
            'categories_match_single': categories_match_single,
            'categories_match_multi': categories_match_multi,
            'statistics_total_qa_pairs': statistics_total,
            'statistics_vs_extracted_diff': statistics_total - extracted_total,
            'other_single': other_single,
            'other_multi': other_multi,
            'missed_by_file': missed_by_file,
        }
    
    def export_stage_combination_table(self, output_dir: str) -> Optional[str]:
        """
        Write stage-combination summary table to CSV in output_dir.
        Returns the path of the written file, or None if no data.
        """
        summary = self.get_stage_combination_summary()
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        import csv
        csv_path = out_path / "stage_combination_summary.csv"
        rows = []
        for scope, counts in summary.items():
            if not counts:
                continue
            for category, count in sorted(counts.items(), key=lambda x: -x[1]):
                rows.append((scope, category, count))
        if not rows:
            return None
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['scope', 'stage_combination', 'question_count'])
            w.writerows(rows)
        return str(csv_path)
    
    def _short_design_id(self, design_id: str) -> str:
        """
        Strip timestamp suffix from a design ID for a readable design_tech key.
        E.g. aes_asap7_2026-01-27T23_49_52_530Z -> aes_asap7.
        """
        s = (design_id or '').strip()
        if not s:
            return s
        # Match trailing _<date>T<time>_<tz> (ISO-like) and strip it
        for i in range(len(s) - 1, -1, -1):
            if s[i] == '_' and i > 0:
                suffix = s[i + 1:]
                if 'T' in suffix and ('Z' in suffix or len(suffix) > 18):
                    return s[:i]
                break
        return s

    def _design_tech_key(self, data: Dict[str, Any]) -> str:
        """
        Return a stable key for design+technology (or multi-file comparison).
        Supports both single-file (e.g. aes_asap7, aes_nangate45) and multifile
        (e.g. aes_asap7_vs_aes_nangate45) QA data.
        """
        path = data.get('_file_path', '')
        if path:
            p = Path(path)
            # Parent directory name: e.g. ariane133_nanagte45, aes_asap7_vs_aes_nangate45
            key = p.parent.name if p.parent and p.parent.name else (p.stem or path)
            return key
        if 'designs' in data and isinstance(data['designs'], list) and len(data['designs']) > 1:
            # Short form for pass@k keys: aes_asap7_vs_aes_nangate45 instead of full IDs
            short = [self._short_design_id(d) for d in data['designs']]
            return '_vs_'.join(sorted(short))
        return data.get('design_name', 'unknown')
    
    def _raw_category_to_categorical(self, raw: str) -> str:
        """
        Map raw stage-combination category to paper-style categorical column:
        Flow Execution, single stages (synthesis..final), or 2-Stages..5-Stages.
        """
        raw = (raw or '').strip()
        if not raw or raw == 'other':
            return 'Flow Execution'  # fallback
        if raw == 'flow_execution':
            return 'Flow Execution'
        if raw.startswith('single_stage:') or raw.startswith('single_stage_comparison:'):
            prefix = 'single_stage_comparison:' if 'single_stage_comparison:' in raw else 'single_stage:'
            x = raw.split(prefix, 1)[-1].strip()
            # Normalize to canonical single-stage name (column must exist)
            if x in STAGE_ORDER:
                return x
            # e.g. placement_1 -> placement
            base = x.split('_')[0] if '_' in x else x
            return base if base in STAGE_ORDER else 'synthesis'  # safe fallback
        if raw.startswith('stage_to_final:'):
            # e.g. stage_to_final:placement->final -> 4 stages (placement, cts, routing, final)
            rest = raw.split('stage_to_final:', 1)[-1].split('->')[0].strip()
            base = rest.split(':')[0].strip() if ':' in rest else rest
            expanded = self._stages_from_to(base, 'final')
            n = len(expanded) if expanded else 2
            n = min(max(n, 2), 5)
            return f'{n}-Stages'
        if raw.startswith('multi_stage:') or raw.startswith('multi_stage_comparison:'):
            prefix = 'multi_stage_comparison:' if 'multi_stage_comparison:' in raw else 'multi_stage:'
            rest = raw.split(prefix, 1)[-1].strip()
            # Count stages: comma-separated, e.g. "floorplan,placement,cts" -> 3
            n = len([p for p in rest.split(',') if p.strip()])
            n = min(max(n, 2), 5)
            return f'{n}-Stages'
        # type is multi_stage but stages were missing/empty (e.g. single-file from section without stages)
        if raw == 'multi_stage':
            return '2-Stages'
        return 'Flow Execution'
    
    def _get_stages_list_from_qa_item(self, q: Dict[str, Any], is_multifile: bool) -> List[str]:
        """
        Return the list of stage names for a QA item (for pass@k category mapping).
        - flow_execution: []
        - single_stage: [stage]
        - multi_stage: list of stages (canonical order)
        - stage_to_final: expanded range from stage to final
        """
        q_type = (q.get('type') or q.get('question_type') or '').strip().lower()
        stages = q.get('stages')
        stage_singular = q.get('stage')
        stages_compared = q.get('stages_compared')
        if is_multifile:
            if stages_compared:
                comp_list = stages_compared if isinstance(stages_compared, list) else [stages_compared]
                ordered = self._sort_stages_canonical(comp_list)
                return self._expand_stage_range(ordered) if len(ordered) == 2 else ordered
            if stages:
                stage_list = stages if isinstance(stages, list) else [stages]
                ordered = self._sort_stages_canonical(stage_list)
                return self._expand_stage_range(ordered) if len(ordered) == 2 else ordered
            if stage_singular:
                return [str(stage_singular)]
            return []
        if q_type == 'flow_execution' and not stages:
            return []
        if stage_singular and (q_type == 'single_stage' or q_type == 'stage_to_final'):
            if q_type == 'stage_to_final':
                expanded = self._stages_from_to(stage_singular, 'final')
                return expanded if expanded else [str(stage_singular)]
            return [str(stage_singular)]
        if stages:
            stage_list = stages if isinstance(stages, list) else [stages]
            ordered = self._sort_stages_canonical(stage_list)
            return self._expand_stage_range(ordered) if len(ordered) == 2 else ordered
        return []
    
    def get_design_tech_to_qa_list(
        self,
    ) -> Dict[str, List[Tuple[str, List[str], str]]]:
        """
        Build design_tech -> [(qid, stages_list, category), ...] from loaded QA data (merged).
        Supports both single-file (design_tech e.g. aes_asap7, aes_nangate45) and
        multifile (e.g. aes_asap7_vs_aes_nangate45) QA JSON. Uses the same
        categorization as stage_combination_by_design_tech.csv (categorical
        columns: Flow Execution, synthesis, ..., 2-Stages, 5-Stages). Intended for
        computing pass@k metrics per (design_tech, category).
        """
        separated = self.get_design_tech_to_qa_list_separated()
        merged: Dict[str, List[Tuple[str, List[str], str]]] = {}
        for dt, entries in (separated.get('single_file') or {}).items():
            merged[dt] = list(entries)
        for dt, entries in (separated.get('multifile') or {}).items():
            merged[dt] = merged.get(dt, []) + list(entries)
        return merged

    def get_design_tech_to_qa_list_separated(
        self,
    ) -> Dict[str, Dict[str, List[Tuple[str, List[str], str]]]]:
        """
        Build design_tech -> [(qid, stages_list, category), ...] split by single_file vs multifile.
        Returns {"single_file": {design_tech: entries, ...}, "multifile": {design_tech: entries, ...}}.
        """
        single: Dict[str, List[Tuple[str, List[str], str]]] = defaultdict(list)
        multifile: Dict[str, List[Tuple[str, List[str], str]]] = defaultdict(list)
        for data in self.data:
            is_multi = 'designs' in data and isinstance(data['designs'], list) and len(data.get('designs', [])) > 1
            pairs = self._get_questions_from_data(data, is_multifile=is_multi)
            if not pairs:
                continue
            design_tech = self._design_tech_key(data)
            target = multifile if is_multi else single
            for q, _ in pairs:
                qid = q.get('id') or q.get('question_id')
                if qid is None:
                    qid = ''
                stages_list = self._get_stages_list_from_qa_item(q, is_multifile=is_multi)
                raw_cat = self._stage_combination_category(q, is_multifile=is_multi)
                category = self._raw_category_to_categorical(raw_cat)
                target[design_tech].append((str(qid), stages_list, category))
        return {
            'single_file': dict(single),
            'multifile': dict(multifile),
        }

    def check_duplicate_question_ids_per_design_tech(
        self,
    ) -> Dict[str, Dict[str, int]]:
        """
        Check for duplicate question IDs within the same design_tech.
        Returns design_tech -> {qid: count} for qids that appear more than once.
        design_techs with no duplicates are omitted from the result.
        """
        by_design_tech = self.get_design_tech_to_qa_list()
        duplicates: Dict[str, Dict[str, int]] = {}
        for design_tech, entries in by_design_tech.items():
            qid_counts: Dict[str, int] = defaultdict(int)
            for qid, _stages, _cat in entries:
                qid_counts[str(qid)] += 1
            dup_only = {qid: c for qid, c in qid_counts.items() if c > 1}
            if dup_only:
                duplicates[design_tech] = dict(dup_only)
        return duplicates

    def print_duplicate_question_ids_report(
        self,
        duplicates: Optional[Dict[str, Dict[str, int]]] = None,
    ) -> None:
        """
        Print a report of duplicate question IDs per design_tech.
        If duplicates is None, runs check_duplicate_question_ids_per_design_tech() first.
        """
        if duplicates is None:
            duplicates = self.check_duplicate_question_ids_per_design_tech()
        if not duplicates:
            print("\n✅ No duplicate question IDs within any design_tech.")
            return
        print("\n⚠️  Duplicate question IDs within same design_tech:")
        for design_tech in sorted(duplicates.keys()):
            dup = duplicates[design_tech]
            total_dup_occurrences = sum(dup.values())
            print(f"  {design_tech}: {len(dup)} qid(s) repeated ({total_dup_occurrences} extra occurrences)")
            for qid, count in sorted(dup.items(), key=lambda x: -x[1]):
                print(f"    - {qid}: {count} times")

    def _dedupe_qids_for_dump(
        self, entries: List[Tuple[str, List[str], str]]
    ) -> List[Tuple[str, List[str], str]]:
        """
        For qids that appear more than once, rename to <qid>_#1, <qid>_#2, ... so all are unique.
        For qids that appear only once, leave unchanged (no _#1 suffix).
        Returns a new list with unique qids per design_tech for dumping.
        """
        counts: Dict[str, int] = defaultdict(int)
        for qid, _stages, _cat in entries:
            counts[qid] += 1
        seen: Dict[str, int] = defaultdict(int)
        result: List[Tuple[str, List[str], str]] = []
        for qid, stages, cat in entries:
            if counts[qid] > 1:
                seen[qid] += 1
                out_qid = f"{qid}_#{seen[qid]}"
            else:
                out_qid = qid
            result.append((out_qid, stages, cat))
        return result

    def dump_design_tech_to_qa_list(
        self,
        output_path: Optional[str] = None,
        include_category: bool = True,
    ) -> Dict[str, List[Tuple[str, List[str], str]]]:
        """
        Build and optionally write design_tech -> [(qid, stages, category), ...] for pass@k per category.
        Writes JSON as a flat dict: {"design_tech": [{"qid", "stages", "category"}, ...], ...}.
        Duplicate question IDs within the same design_tech (e.g. in multifile cases) are renamed
        to <qid>_#1, <qid>_#2, ...; unique qids are left unchanged.
        If include_category is False, each entry has "qid", "stages" only.
        Returns the same dict as get_design_tech_to_qa_list() (unchanged, no renaming).
        """
        raw = self.get_design_tech_to_qa_list()
        if not output_path:
            return raw
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        # De-duplicate qids per design_tech for the written file only
        deduped = {dt: self._dedupe_qids_for_dump(entries) for dt, entries in raw.items()}
        if include_category:
            serializable = {
                dt: [{"qid": qid, "stages": stages, "category": cat} for qid, stages, cat in entries]
                for dt, entries in deduped.items()
            }
        else:
            serializable = {
                dt: [{"qid": qid, "stages": stages} for qid, stages, _ in entries]
                for dt, entries in deduped.items()
            }
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        return raw
    
    def get_stage_combination_by_design_tech(
        self,
    ) -> Tuple[Dict[str, Dict[str, int]], List[str]]:
        """
        Return question counts by (design+tech) and stage-combination category.
        Returns (by_design_tech, all_categories) where by_design_tech[design_tech][category] = count,
        and all_categories is sorted list of all stage categories seen.
        """
        by_design_tech: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        all_categories: Set[str] = set()
        for data in self.data:
            is_multi = 'designs' in data and isinstance(data['designs'], list) and len(data.get('designs', [])) > 1
            pairs = self._get_questions_from_data(data, is_multifile=is_multi)
            if not pairs:
                continue
            design_tech = self._design_tech_key(data)
            cats = self._categorize_by_stage_combination(pairs, is_multifile=is_multi, max_representative=0)
            for cat, info in cats.items():
                by_design_tech[design_tech][cat] = info['count']
                all_categories.add(cat)
        # Convert defaultdict to dict and ensure total is present
        result: Dict[str, Dict[str, int]] = {}
        for dt, counts in by_design_tech.items():
            total = sum(counts.values())
            result[dt] = dict(counts)
            result[dt]['total'] = total
        all_categories.discard('total')
        categories_sorted = sorted(all_categories, key=lambda c: (-sum(result.get(dt, {}).get(c, 0) for dt in result)))
        return result, categories_sorted
    
    def get_stage_combination_by_design_tech_categorical(
        self,
    ) -> Tuple[Dict[str, Dict[str, int]], List[str]]:
        """
        Return question counts by (design+tech) and paper-style categories:
        Flow Execution, single stages (synthesis..final), 2-Stages..5-Stages.
        Returns (by_design_tech, columns) with columns = CATEGORICAL_STAGE_COLUMNS.
        """
        by_design_tech_raw, _ = self.get_stage_combination_by_design_tech()
        by_design_tech: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for design_tech, raw_counts in by_design_tech_raw.items():
            if design_tech == 'total':
                continue
            for raw_cat, count in raw_counts.items():
                if raw_cat == 'total':
                    continue
                cat = self._raw_category_to_categorical(raw_cat)
                by_design_tech[design_tech][cat] += count
        result = {}
        for dt, counts in by_design_tech.items():
            total = sum(counts.values())
            result[dt] = dict(counts)
            result[dt]['total'] = total
        return result, list(CATEGORICAL_STAGE_COLUMNS)
    
    def export_stage_combination_by_design_tech_csv(self, output_dir: str) -> Optional[str]:
        """
        Write CSV: design_tech x categorical stage columns + total (one row per design+tech).
        Columns: design_tech, Flow Execution, synthesis, floorplan, placement, cts, routing, final,
                 2-Stages, 3-Stages, 4-Stages, 5-Stages, total
        """
        by_design_tech, col_order = self.get_stage_combination_by_design_tech_categorical()
        if not by_design_tech:
            return None
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        import csv
        csv_path = out_path / "stage_combination_by_design_tech.csv"
        headers = ['design_tech'] + col_order + ['total']
        rows = []
        for design_tech in sorted(by_design_tech.keys(), key=lambda dt: (-by_design_tech[dt].get('total', 0))):
            counts = by_design_tech[design_tech]
            row = [design_tech] + [counts.get(c, 0) for c in col_order] + [counts.get('total', 0)]
            rows.append(row)
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(headers)
            w.writerows(rows)
        return str(csv_path)
    
    def write_extraction_diff_report(self, output_dir: str) -> Optional[str]:
        """
        Write a report of files where statistics total != extracted count.
        Returns the path of the written file, or None if no differences.
        """
        stats = getattr(self, '_last_stats', None)
        if not stats:
            return None
        by_file = stats.get('qa_pairs_by_file', [])
        differing = [r for r in by_file if r.get('diff', 0) != 0]
        if not differing:
            return None
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        report = {
            'total_qa_pairs_extracted': stats.get('total_qa_pairs_extracted'),
            'total_qa_pairs_from_statistics': stats.get('total_qa_pairs'),
            'files_with_difference': differing,
            'all_files': by_file,
        }
        report_path = out_path / 'qa_extraction_diff_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        return str(report_path)
    
    def _print_file_list(self, stats: Optional[Dict[str, Any]] = None) -> None:
        """Print detailed list of analyzed files. Uses extracted QA count per file when stats is provided."""
        if not self.data:
            return
        by_path = {}
        if stats and stats.get('qa_pairs_by_file'):
            for row in stats['qa_pairs_by_file']:
                by_path[row['path']] = row
        print("\n📋 ANALYZED FILES:")
        single_files = []
        multi_files = []
        for data in self.data:
            path = data.get('_file_path', 'unknown')
            row = by_path.get(path, {})
            extracted = row.get('extracted')
            statistics_total = row.get('statistics_total')
            diff = row.get('diff', 0)
            if extracted is not None:
                qa_display = f"{extracted} extracted"
                if diff != 0:
                    qa_display += f" (statistics: {statistics_total}, diff: {diff:+d})"
            else:
                qa_display = str(data.get('statistics', {}).get('total_qa_pairs', 0))
            file_info = {
                'path': path,
                'name': data.get('_file_name', 'unknown'),
                'qa_pairs': qa_display,
                'extracted': extracted,
                'diff': diff,
            }
            if 'designs' in data and isinstance(data['designs'], list):
                file_info['type'] = 'multifile'
                file_info['designs'] = data['designs']
                multi_files.append(file_info)
            else:
                file_info['type'] = 'single-file'
                file_info['design'] = data.get('design_name', 'unknown')
                single_files.append(file_info)
        if single_files:
            print(f"\n   Single-file designs ({len(single_files)}):")
            for idx, f in enumerate(sorted(single_files, key=lambda x: x['path']), 1):
                print(f"     {idx:2d}. {f['name']}")
                print(f"         Path: {f['path']}")
                print(f"         Design: {f['design']}, QA pairs: {f['qa_pairs']}")
        if multi_files:
            print(f"\n   Multi-file comparisons ({len(multi_files)}):")
            for idx, f in enumerate(sorted(multi_files, key=lambda x: x['path']), 1):
                print(f"     {idx:2d}. {f['name']}")
                print(f"         Path: {f['path']}")
                designs_str = ', '.join(f['designs'][:3])
                if len(f['designs']) > 3:
                    designs_str += f" (+{len(f['designs']) - 3} more)"
                print(f"         Designs: {designs_str}, QA pairs: {f['qa_pairs']}")
    
    def print_summary(self) -> None:
        """Print comprehensive summary of QA statistics."""
        stats = self.compute_statistics()
        self._last_stats = stats
        
        print("\n" + "="*80)
        print("QA GENERATION STATISTICS SUMMARY")
        print("="*80)
        
        print(f"\n📁 FILES ANALYZED: {stats['total_files']}")
        print(f"   - Single-file designs: {stats['single_file_designs']}")
        print(f"   - Multi-file comparisons: {stats['multifile_comparisons']}")
        
        # Check if multifile has rejection stats
        has_multifile_rejection_stats = any(
            'rejection_statistics' in d 
            for d in self.data 
            if 'designs' in d and isinstance(d['designs'], list)
        )
        
        if stats['multifile_comparisons'] > 0 and not has_multifile_rejection_stats:
            print(f"\n   ⚠️  Note: Multi-file comparisons do not include rejection statistics")
            print(f"       (rejection analysis only available for single-file designs)")
        
        # Print detailed file list (use extracted counts; show diff when present)
        self._print_file_list(stats=stats)
        
        # Stage-combination categorization and representative questions (single-file and multi-file)
        self.print_stage_combination_summary(max_representative=getattr(self, '_max_representative', 5))
        
        print(f"\n❓ TOTAL Q&A PAIRS (EXTRACTED): {stats['total_qa_pairs_extracted']}")
        if stats['total_qa_pairs'] != stats['total_qa_pairs_extracted']:
            print(f"   (File statistics total was {stats['total_qa_pairs']}; difference = {stats['total_qa_pairs'] - stats['total_qa_pairs_extracted']} not extracted)")
        
        # Add single vs multi-design breakdown
        try:
            import numpy as np
            qa_data = self.extract_qa_data_for_plotting()
            single_count = len(qa_data['by_comparison_type'].get('single_design', []))
            multi_count = len(qa_data['by_comparison_type'].get('multi_design', []))
            
            if single_count > 0 or multi_count > 0:
                print(f"\n📊 BREAKDOWN BY COMPARISON TYPE:")
                print(f"   - Single Design: {single_count} questions")
                print(f"   - Multi-Design: {multi_count} questions")
                
                if single_count > 0 and multi_count > 0:
                    total_q = single_count + multi_count
                    single_pct = (single_count / total_q) * 100
                    multi_pct = (multi_count / total_q) * 100
                    print(f"   - Single Design: {single_pct:.1f}%")
                    print(f"   - Multi-Design: {multi_pct:.1f}%")
                
                # Add token statistics breakdown
                if single_count > 0:
                    single_q_tokens = [d['q_tokens'] for d in qa_data['by_comparison_type'].get('single_design', [])]
                    single_a_tokens = [d['a_tokens'] for d in qa_data['by_comparison_type'].get('single_design', [])]
                    if single_q_tokens:
                        print(f"\n   📝 Single Design Token Statistics:")
                        print(f"      - Question tokens: mean={np.mean(single_q_tokens):.1f}, median={np.median(single_q_tokens):.1f}")
                        print(f"      - Answer tokens: mean={np.mean(single_a_tokens):.1f}, median={np.median(single_a_tokens):.1f}")
                
                if multi_count > 0:
                    multi_q_tokens = [d['q_tokens'] for d in qa_data['by_comparison_type'].get('multi_design', [])]
                    multi_a_tokens = [d['a_tokens'] for d in qa_data['by_comparison_type'].get('multi_design', [])]
                    if multi_q_tokens:
                        print(f"\n   📝 Multi-Design Token Statistics:")
                        print(f"      - Question tokens: mean={np.mean(multi_q_tokens):.1f}, median={np.median(multi_q_tokens):.1f}")
                        print(f"      - Answer tokens: mean={np.mean(multi_a_tokens):.1f}, median={np.median(multi_a_tokens):.1f}")
                        
                        if single_count > 0:
                            single_q_tokens = [d['q_tokens'] for d in qa_data['by_comparison_type'].get('single_design', [])]
                            single_a_tokens = [d['a_tokens'] for d in qa_data['by_comparison_type'].get('single_design', [])]
                            if single_q_tokens:
                                print(f"\n   📊 Comparison:")
                                print(f"      - Question tokens difference: {np.mean(multi_q_tokens) - np.mean(single_q_tokens):.1f} tokens")
                                print(f"      - Answer tokens difference: {np.mean(multi_a_tokens) - np.mean(single_a_tokens):.1f} tokens")
        except ImportError:
            # numpy not available, skip detailed statistics
            pass
        
        print("\n📊 BREAKDOWN BY QUESTION TYPE:")
        for qtype, count in sorted(stats['by_question_type'].items()):
            print(f"   - {qtype}: {count}")
        
        # Warn about unknown/other categories
        unknown_type_count = stats['by_question_type'].get('unknown', 0)
        other_type_count = stats['by_question_type'].get('other', 0)
        try:
            qa_data = self.extract_qa_data_for_plotting()
            difficulty_counts = {}
            for d in qa_data.get('difficulties', []):
                difficulty_counts[d] = difficulty_counts.get(d, 0) + 1
            unknown_diff_count = difficulty_counts.get('unknown', 0)
        except:
            unknown_diff_count = 0
        
        if unknown_type_count > 0 or other_type_count > 0 or unknown_diff_count > 0:
            print(f"\n   ⚠️  Note: Found {unknown_type_count} questions with unknown type, {other_type_count} with 'other' type, and {unknown_diff_count} with unknown difficulty")
            print(f"       This may occur when:")
            print(f"       - Questions don't have 'type' or 'question_type' field")
            print(f"       - Questions don't have 'difficulty' field and it's not in question text")
            print(f"       - Question format differs from expected structure")
            print(f"       - Pass-at-k format uses 'question_type' instead of 'type'")
            print(f"       - Question type values don't match standard patterns (flow_execution, single_stage, multi_stage, etc.)")
            if other_type_count > 0:
                print(f"       - 'other' type may indicate questions that don't fit standard categories")
        
        print("\n🏗️  BREAKDOWN BY DESIGN:")
        for design, count in sorted(stats['by_design'].items()):
            print(f"   - {design}: {count}")
        
        print("\n❌ REJECTION STATISTICS:")
        rej = stats['rejection_stats']
        print(f"   - Total attempts: {rej['total_attempts']}")
        print(f"   - Successful: {rej['successful']}")
        print(f"   - Total rejected: {rej['total_rejected']}")
        print(f"   - Total unique questions rejected: {rej['total_questions_rejected']}")
        print(f"   - Code execution failed: {rej['code_execution_failed']}")
        print(f"   - Low score rejected: {rej['low_score_rejected']}")
        print(f"   - Invalid answer rejected: {rej['invalid_answer_rejected']}")
        print(f"   - Average success rate: {rej['avg_success_rate']:.2f}%")
        
        print("\n📈 SCORE DISTRIBUTION (for rejected):")
        for score_range, count in sorted(stats['score_distribution'].items()):
            print(f"   - {score_range}: {count}")
        
        print("\n" + "="*80)
    
    def analyze_rejection_categories(self) -> Dict[str, Any]:
        """Analyze rejection reasons by simple categorization without embeddings."""
        if not self.rejection_reasons:
            return {'error': 'No rejection reasons found'}
        
        categories = defaultdict(int)
        category_scores = defaultdict(list)
        category_examples = defaultdict(list)
        
        for reason_data in self.rejection_reasons:
            reason = reason_data['reason']
            score = reason_data['score']
            
            # Categorize based on keywords
            category = 'Other'
            if 'does not match' in reason.lower() or 'not match' in reason.lower():
                category = 'Mismatch: Code vs Expected'
            elif 'does not address' in reason.lower() or 'not address' in reason.lower():
                category = 'Does Not Address Question'
            elif 'cannot be validated' in reason.lower() or 'cannot validate' in reason.lower():
                category = 'Cannot Validate Answer'
            elif 'data not available' in reason.lower() or 'not available' in reason.lower():
                category = 'Data Not Available'
            elif 'not explicitly' in reason.lower() or 'not provided' in reason.lower():
                category = 'Data Not Explicit in JSON'
            elif 'null' in reason.lower() or 'none' in reason.lower():
                category = 'Null/None Response'
            elif score == 0.0:
                category = 'Complete Failure (Score 0.0)'
            elif score < 0.5:
                category = 'Major Discrepancy (Score < 0.5)'
            elif score < 0.8:
                category = 'Partial Match (Score 0.5-0.8)'
            
            categories[category] += 1
            category_scores[category].append(score)
            if len(category_examples[category]) < 3:
                category_examples[category].append({
                    'reason': reason[:150],
                    'score': score,
                    'file': reason_data.get('file', 'unknown')
                })
        
        # Calculate statistics per category
        category_stats = {}
        for category, count in categories.items():
            scores = category_scores[category]
            category_stats[category] = {
                'count': count,
                'percentage': (count / len(self.rejection_reasons)) * 100,
                'avg_score': sum(scores) / len(scores) if scores else 0.0,
                'examples': category_examples[category]
            }
        
        return {
            'total_reasons': len(self.rejection_reasons),
            'categories': dict(sorted(category_stats.items(), 
                                    key=lambda x: x[1]['count'], 
                                    reverse=True))
        }
    
    def print_rejection_categories(self, analysis: Dict[str, Any]) -> None:
        """Print simple categorical analysis of rejection reasons."""
        if 'error' in analysis:
            print(f"\n⚠️  {analysis['error']}")
            return
        
        print("\n" + "="*80)
        print("REJECTION REASON CATEGORY ANALYSIS")
        print("="*80)
        
        # Count files with rejection data
        files_with_rejection = [d for d in self.data if 'rejection_statistics' in d]
        single_file_with_rejection = [d for d in files_with_rejection if not ('designs' in d and isinstance(d['designs'], list))]
        multifile_with_rejection = [d for d in files_with_rejection if 'designs' in d and isinstance(d['designs'], list)]
        
        print(f"\n📊 Total rejection reasons analyzed: {analysis['total_reasons']}")
        print(f"    From {len(single_file_with_rejection)} single-file designs")
        if len(multifile_with_rejection) > 0:
            print(f"    From {len(multifile_with_rejection)} multi-file comparisons")
        else:
            print(f"    ⚠️  Multi-file comparisons do not include rejection statistics")
        
        print("\n🏷️  CATEGORIES (sorted by frequency):\n")
        for category, stats in analysis['categories'].items():
            print(f"  {category}")
            print(f"    Count: {stats['count']} ({stats['percentage']:.1f}%)")
            print(f"    Avg Score: {stats['avg_score']:.2f}")
            print(f"    Examples:")
            for i, example in enumerate(stats['examples'][:2], 1):
                print(f"      {i}. {example['reason'][:100]}...")
                print(f"         Score: {example['score']}, File: {example['file']}")
            print()
        
        print("="*80)
    
    def print_clustering_results(self, clustering_results: Dict[str, Any]) -> None:
        
        print("\n" + "="*80)
        print("REJECTION REASON CLUSTERING ANALYSIS")
        print("="*80)
        
        print(f"\n🔍 CLUSTERING METHOD: {clustering_results['method'].upper()}")
        print(f"   - Number of clusters: {clustering_results['n_clusters']}")
        print(f"   - Silhouette score: {clustering_results['silhouette_score']:.3f}")
        print(f"   - Total reasons analyzed: {clustering_results['total_reasons']}")
        
        print("\n📦 CLUSTER SUMMARIES:")
        for cluster_name, summary in sorted(clustering_results['cluster_summaries'].items()):
            print(f"\n{cluster_name}:")
            print(f"   Size: {summary['size']} reasons")
            print(f"   Avg Score: {summary['avg_score']:.2f}")
            print(f"   Representative: {summary['representative_reason']}")
            print(f"   Sample reasons:")
            for i, reason in enumerate(summary['sample_reasons'][:3], 1):
                print(f"      {i}. {reason[:150]}...")
            print(f"   Affected files: {len(summary['files_affected'])} files")
        
        print("\n" + "="*80)
    
    def _convert_to_json_serializable(self, obj: Any) -> Any:
        """Convert numpy types and other non-serializable objects to JSON-compatible types."""
        try:
            import numpy as np
            
            if isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8,
                              np.int16, np.int32, np.int64, np.uint8, np.uint16,
                              np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
        except ImportError:
            # numpy not available, skip numpy-specific conversions
            pass
        
        # Handle other types recursively
        if isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, set):
            return list(obj)  # Convert sets to lists
        else:
            return obj
    
    def export_results(self, output_file: str = "qa_stats_summary_output.json", 
                      include_clustering: bool = True) -> None:
        """Export statistics and clustering results to JSON file."""
        stats = self.compute_statistics()
        
        results = {
            'statistics': stats,
            'rejection_reasons_sample': self.rejection_reasons[:20]  # Include sample
        }
        
        # Only include clustering if available and requested
        if include_clustering and CLUSTERING_AVAILABLE:
            try:
                clustering = self.cluster_rejection_reasons()
                results['clustering_analysis'] = clustering
            except Exception as e:
                results['clustering_analysis'] = {
                    'error': 'Clustering failed',
                    'message': str(e)
                }
        elif include_clustering and not CLUSTERING_AVAILABLE:
            results['clustering_analysis'] = {
                'error': 'Clustering dependencies not available',
                'message': 'Install with: pip install -r qa_stats_requirements.txt'
            }
        
        # Convert numpy types to JSON-serializable types
        results = self._convert_to_json_serializable(results)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Results exported to: {output_file}")
    
    def extract_qa_data_for_plotting(self) -> Dict[str, Any]:
        """Extract QA data for plotting: tokens, difficulty, question types."""
        import sys
        from pathlib import Path
        # Add src directory to path to import token_counter_util
        src_path = Path(__file__).parent
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
        from token_counter_util import count_tokens
        
        qa_data = {
            'question_tokens': [],
            'answer_tokens': [],
            'difficulties': [],
            'question_types': [],
            'question_starts': [],  # First word(s) of question
            'question_categories': [],  # Categorized by starting word
            'comparison_type': [],  # 'single_design' or 'multi_design'
            'by_difficulty': defaultdict(list),
            'by_question_type': defaultdict(list),
            'by_category': defaultdict(list),
            'by_comparison_type': defaultdict(list)  # Separate data by comparison type
        }
        
        for data in self.data:
            # Determine if this is a single-file or multifile comparison
            is_multifile = 'designs' in data and isinstance(data['designs'], list) and len(data['designs']) > 1
            comparison_type = 'multi_design' if is_multifile else 'single_design'
            
            # Extract from questions array (single-file format)
            questions = data.get('questions', [])
            
            # Also check for qa_pairs array (flattened format)
            if not questions:
                qa_pairs = data.get('qa_pairs', [])
                questions = qa_pairs
            
            # Handle nested structure (multifile format)
            # Multifile raw format has question_set.qa_pairs in multiple sections
            if not questions:
                all_multifile_questions = []
                
                # Check single_stage_comparison section
                if 'single_stage_comparison' in data and isinstance(data['single_stage_comparison'], dict):
                    question_set = data['single_stage_comparison'].get('question_set', {})
                    if isinstance(question_set, dict):
                        qa_pairs = question_set.get('qa_pairs', [])
                        all_multifile_questions.extend(qa_pairs)
                
                # Check multi_stage_comparison section
                if 'multi_stage_comparison' in data and isinstance(data['multi_stage_comparison'], dict):
                    question_set = data['multi_stage_comparison'].get('question_set', {})
                    if isinstance(question_set, dict):
                        qa_pairs = question_set.get('qa_pairs', [])
                        all_multifile_questions.extend(qa_pairs)
                
                # Check stage_comparison section (legacy/alternative format)
                if 'stage_comparison' in data and isinstance(data['stage_comparison'], dict):
                    question_set = data['stage_comparison'].get('question_set', {})
                    if isinstance(question_set, dict):
                        qa_pairs = question_set.get('qa_pairs', [])
                        all_multifile_questions.extend(qa_pairs)
                    # Also check for direct questions array (backward compatibility)
                    elif 'questions' in data['stage_comparison']:
                        all_multifile_questions.extend(data['stage_comparison'].get('questions', []))
                
                # Check flow_execution section
                if 'flow_execution' in data and isinstance(data['flow_execution'], dict):
                    question_set = data['flow_execution'].get('question_set', {})
                    if isinstance(question_set, dict):
                        qa_pairs = question_set.get('qa_pairs', [])
                        all_multifile_questions.extend(qa_pairs)
                    # Also check for direct questions array (backward compatibility)
                    elif 'questions' in data['flow_execution']:
                        all_multifile_questions.extend(data['flow_execution'].get('questions', []))
                
                if all_multifile_questions:
                    questions = all_multifile_questions
            
            for q in questions:
                # Extract question text
                question_text = q.get('question', '')
                if not question_text:
                    continue
                
                # Clean question text (remove difficulty suffix if present)
                if 'Difficulty:' in question_text:
                    question_text = question_text.split('Difficulty:')[0].strip()
                
                # Count tokens
                q_tokens = count_tokens(question_text)
                qa_data['question_tokens'].append(q_tokens)
                
                # Extract answer and count tokens
                answer = q.get('answer', '')
                if isinstance(answer, dict):
                    answer_text = json.dumps(answer)
                elif isinstance(answer, list):
                    answer_text = json.dumps(answer)
                else:
                    answer_text = str(answer)
                a_tokens = count_tokens(answer_text)
                qa_data['answer_tokens'].append(a_tokens)
                
                # Extract difficulty - check multiple possible field names
                difficulty = q.get('difficulty', None)
                if not difficulty or difficulty == '':
                    # Try alternative field names
                    difficulty = q.get('question_difficulty', None)
                if not difficulty or difficulty == '':
                    # Try to extract from question text if present
                    original_question = q.get('question', '')
                    if 'Difficulty:' in original_question:
                        parts = original_question.split('Difficulty:')
                        if len(parts) > 1:
                            difficulty = parts[1].strip().split()[0].lower()
                            # Clean up common variations
                            difficulty = difficulty.rstrip('.,;:')
                if not difficulty or difficulty == '':
                    difficulty = 'unknown'
                else:
                    # Normalize difficulty values
                    difficulty = difficulty.lower().strip()
                    if difficulty not in ['easy', 'medium', 'hard']:
                        # If it's not a standard value, try to infer or mark as unknown
                        if any(word in difficulty for word in ['easy', 'simple']):
                            difficulty = 'easy'
                        elif any(word in difficulty for word in ['medium', 'moderate', 'intermediate']):
                            difficulty = 'medium'
                        elif any(word in difficulty for word in ['hard', 'difficult', 'complex']):
                            difficulty = 'hard'
                        else:
                            difficulty = 'unknown'
                
                qa_data['difficulties'].append(difficulty)
                qa_data['by_difficulty'][difficulty].append({
                    'q_tokens': q_tokens,
                    'a_tokens': a_tokens,
                    'question': question_text[:100]  # Truncate for storage
                })
                
                # Extract question type - check multiple possible field names
                q_type = q.get('type', None)
                if not q_type or q_type == '':
                    # Try alternative field names (pass_at_k format uses 'question_type')
                    q_type = q.get('question_type', None)
                if not q_type or q_type == '':
                    # Try to infer from other fields
                    if 'stages_compared' in q or 'stage_comparison' in str(q):
                        q_type = 'multi_file_stage_comparison'
                    elif 'flow_execution' in str(q).lower():
                        q_type = 'flow_execution'
                    elif 'stage' in q:
                        stage_val = q.get('stage', '')
                        if stage_val and stage_val != 'unknown':
                            q_type = f'single_stage_{stage_val}'
                    else:
                        q_type = 'unknown'
                else:
                    # Normalize question type - map common variations to standard types
                    q_type = str(q_type).strip().lower()
                    if not q_type:
                        q_type = 'unknown'
                    else:
                        # Map common variations and aliases to standard types
                        type_mapping = {
                            # Flow execution variations
                            'flow_execution': 'flow_execution',
                            'flow_execution_qa': 'flow_execution',
                            'flow': 'flow_execution',
                            
                            # Single stage variations
                            'single_stage': 'single_stage',
                            'single_stage_qa': 'single_stage',
                            'stage_level': 'single_stage',
                            
                            # Multi-stage variations
                            'multi_stage': 'multi_stage',
                            'multi_stage_qa': 'multi_stage',
                            'multi_stage_comparison': 'multi_stage',
                            
                            # Stage to final variations
                            'stage_to_final': 'stage_to_final',
                            'stage_to_final_qa': 'stage_to_final',
                            
                            # Multi-file variations
                            'multi_file_single_stage_comparison': 'multi_file_single_stage_comparison',
                            'multi_file_multi_stage_comparison': 'multi_file_multi_stage_comparison',
                            'multi_file_stage_comparison': 'multi_file_stage_comparison',
                            'multi_file_flow_execution': 'multi_file_flow_execution',
                            'multi_file': 'multi_file_stage_comparison',  # Default for multi_file
                            
                            # Other common patterns
                            'other': 'other',
                            'unknown': 'unknown',
                            '': 'unknown',
                        }
                        
                        # Check for partial matches (e.g., "single_stage_placement" -> "single_stage")
                        normalized_type = type_mapping.get(q_type, None)
                        if not normalized_type:
                            # Try partial matching
                            if 'flow_execution' in q_type or 'flow' in q_type:
                                normalized_type = 'flow_execution'
                            elif 'multi_file' in q_type or 'multi_file' in q_type:
                                if 'single_stage' in q_type:
                                    normalized_type = 'multi_file_single_stage_comparison'
                                elif 'multi_stage' in q_type:
                                    normalized_type = 'multi_file_multi_stage_comparison'
                                elif 'flow' in q_type:
                                    normalized_type = 'multi_file_flow_execution'
                                else:
                                    normalized_type = 'multi_file_stage_comparison'
                            elif 'single_stage' in q_type:
                                normalized_type = 'single_stage'
                            elif 'multi_stage' in q_type:
                                normalized_type = 'multi_stage'
                            elif 'stage_to_final' in q_type:
                                normalized_type = 'stage_to_final'
                            else:
                                # Keep original if no match found (might be a specific stage type)
                                normalized_type = q_type
                        
                        q_type = normalized_type
                
                qa_data['question_types'].append(q_type)
                qa_data['by_question_type'][q_type].append({
                    'q_tokens': q_tokens,
                    'a_tokens': a_tokens,
                    'difficulty': difficulty
                })
                
                # Categorize by starting word
                question_lower = question_text.lower().strip()
                category = 'Other'
                if question_lower.startswith('how'):
                    category = 'How'
                elif question_lower.startswith('what'):
                    if 'progression' in question_lower or 'change' in question_lower:
                        category = 'What (Progression/Change)'
                    elif 'difference' in question_lower or 'compare' in question_lower:
                        category = 'What (Comparison)'
                    else:
                        category = 'What'
                elif question_lower.startswith('which'):
                    category = 'Which'
                elif question_lower.startswith('where'):
                    category = 'Where'
                elif question_lower.startswith('when'):
                    category = 'When'
                elif question_lower.startswith('why'):
                    category = 'Why'
                elif question_lower.startswith('who'):
                    category = 'Who'
                
                qa_data['question_starts'].append(category)
                qa_data['comparison_type'].append(comparison_type)
                
                qa_data['by_category'][category].append({
                    'q_tokens': q_tokens,
                    'a_tokens': a_tokens,
                    'difficulty': difficulty,
                    'question': question_text[:100]
                })
                
                # Store by comparison type
                qa_data['by_comparison_type'][comparison_type].append({
                    'q_tokens': q_tokens,
                    'a_tokens': a_tokens,
                    'difficulty': difficulty,
                    'question_type': q_type,
                    'category': category,
                    'question': question_text[:100]
                })
        
        return qa_data
    
    def plot_qa_statistics(self, output_dir: str = "qa_plots", 
                          figsize: Tuple[int, int] = (12, 8),
                          dpi: int = 300) -> Dict[str, str]:
        """
        Create comprehensive plots of QA statistics for research paper.
        
        Args:
            output_dir: Directory to save plots
            figsize: Figure size (width, height)
            dpi: Resolution for saved figures
        
        Returns:
            Dictionary mapping plot names to file paths
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            import seaborn as sns
            import numpy as np
        except ImportError:
            print("⚠️  Matplotlib/Seaborn not available. Install with: pip install matplotlib seaborn")
            return {}
        
        # Set publication-quality style
        try:
            plt.style.use('seaborn-v0_8-paper')
        except:
            try:
                plt.style.use('seaborn-paper')
            except:
                plt.style.use('seaborn-whitegrid')
        sns.set_palette("husl")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Extract data
        qa_data = self.extract_qa_data_for_plotting()
        
        if not qa_data['question_tokens']:
            print("⚠️  No QA data found for plotting")
            return {}
        
        # Extract single vs multi-design data for comparison plots
        single_design_data = qa_data['by_comparison_type'].get('single_design', [])
        multi_design_data = qa_data['by_comparison_type'].get('multi_design', [])
        
        plot_files = {}
        
        # 1. Token Distribution: Questions vs Answers
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Question tokens histogram
        axes[0].hist(qa_data['question_tokens'], bins=30, alpha=0.7, edgecolor='black', linewidth=0.5)
        axes[0].axvline(np.mean(qa_data['question_tokens']), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(qa_data["question_tokens"]):.1f}')
        axes[0].axvline(np.median(qa_data['question_tokens']), color='blue', linestyle='--', 
                       linewidth=2, label=f'Median: {np.median(qa_data["question_tokens"]):.1f}')
        axes[0].set_xlabel('Tokens per Question', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[0].set_title('Question Token Distribution', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Answer tokens histogram
        axes[1].hist(qa_data['answer_tokens'], bins=30, alpha=0.7, color='orange', 
                     edgecolor='black', linewidth=0.5)
        axes[1].axvline(np.mean(qa_data['answer_tokens']), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(qa_data["answer_tokens"]):.1f}')
        axes[1].axvline(np.median(qa_data['answer_tokens']), color='blue', linestyle='--', 
                       linewidth=2, label=f'Median: {np.median(qa_data["answer_tokens"]):.1f}')
        axes[1].set_xlabel('Tokens per Answer', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[1].set_title('Answer Token Distribution', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        token_dist_file = output_path / "token_distribution.png"
        plt.savefig(token_dist_file, dpi=dpi, bbox_inches='tight')
        plot_files['token_distribution'] = str(token_dist_file)
        plt.close()
        
        # 2. Difficulty Distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        difficulty_counts = defaultdict(int)
        for d in qa_data['difficulties']:
            difficulty_counts[d] += 1
        
        difficulties = list(difficulty_counts.keys())
        counts = [difficulty_counts[d] for d in difficulties]
        colors = ['#2ecc71', '#f39c12', '#e74c3c']  # green, orange, red
        
        bars = ax.bar(difficulties, counts, color=colors[:len(difficulties)], 
                     edgecolor='black', linewidth=1.5, alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        ax.set_xlabel('Difficulty Level', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Questions', fontsize=12, fontweight='bold')
        ax.set_title('Question Difficulty Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        difficulty_file = output_path / "difficulty_distribution.png"
        plt.savefig(difficulty_file, dpi=dpi, bbox_inches='tight')
        plot_files['difficulty_distribution'] = str(difficulty_file)
        plt.close()
        
        # 3. Question Type Distribution (by starting word/category)
        fig, ax = plt.subplots(figsize=(12, 7))
        category_counts = defaultdict(int)
        for cat in qa_data['question_starts']:
            category_counts[cat] += 1
        
        # Sort by count
        sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        categories = [c[0] for c in sorted_categories]
        counts = [c[1] for c in sorted_categories]
        
        bars = ax.barh(categories, counts, edgecolor='black', linewidth=1.5, alpha=0.8)
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f' {int(width)}',
                   ha='left', va='center', fontweight='bold', fontsize=10)
        
        ax.set_xlabel('Number of Questions', fontsize=12, fontweight='bold')
        ax.set_ylabel('Question Category', fontsize=12, fontweight='bold')
        ax.set_title('Question Distribution by Category', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        category_file = output_path / "question_category_distribution.png"
        plt.savefig(category_file, dpi=dpi, bbox_inches='tight')
        plot_files['question_category'] = str(category_file)
        plt.close()
        
        # 4. Token Distribution by Difficulty
        fig, ax = plt.subplots(figsize=(12, 6))
        
        difficulties_list = list(set(qa_data['difficulties']))
        difficulty_token_data = {d: [] for d in difficulties_list}
        
        for i, diff in enumerate(qa_data['difficulties']):
            difficulty_token_data[diff].append({
                'question': qa_data['question_tokens'][i],
                'answer': qa_data['answer_tokens'][i]
            })
        
        # Create box plot data
        q_tokens_by_diff = [qa_data['question_tokens'][i] for i, d in enumerate(qa_data['difficulties'])]
        a_tokens_by_diff = [qa_data['answer_tokens'][i] for i, d in enumerate(qa_data['difficulties'])]
        
        positions = np.arange(len(difficulties_list))
        width = 0.35
        
        q_means = [np.mean([item['question'] for item in difficulty_token_data[d]]) 
                   for d in difficulties_list]
        a_means = [np.mean([item['answer'] for item in difficulty_token_data[d]]) 
                   for d in difficulties_list]
        
        x = np.arange(len(difficulties_list))
        bars1 = ax.bar(x - width/2, q_means, width, label='Questions', 
                       alpha=0.8, edgecolor='black', linewidth=1)
        bars2 = ax.bar(x + width/2, a_means, width, label='Answers', 
                       alpha=0.8, edgecolor='black', linewidth=1)
        
        ax.set_xlabel('Difficulty Level', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Tokens', fontsize=12, fontweight='bold')
        ax.set_title('Average Token Count by Difficulty Level', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(difficulties_list)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        token_by_diff_file = output_path / "tokens_by_difficulty.png"
        plt.savefig(token_by_diff_file, dpi=dpi, bbox_inches='tight')
        plot_files['tokens_by_difficulty'] = str(token_by_diff_file)
        plt.close()
        
        # 5. Scatter Plot: Question vs Answer Tokens
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Color by difficulty
        difficulty_map = {'easy': '#2ecc71', 'medium': '#f39c12', 'hard': '#e74c3c', 'unknown': '#95a5a6'}
        colors_scatter = [difficulty_map.get(d, '#95a5a6') for d in qa_data['difficulties']]
        
        scatter = ax.scatter(qa_data['question_tokens'], qa_data['answer_tokens'], 
                           c=colors_scatter, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        # Add trend line
        z = np.polyfit(qa_data['question_tokens'], qa_data['answer_tokens'], 1)
        p = np.poly1d(z)
        ax.plot(qa_data['question_tokens'], p(qa_data['question_tokens']), 
               "r--", alpha=0.8, linewidth=2, label=f'Trend: y={z[0]:.2f}x+{z[1]:.1f}')
        
        ax.set_xlabel('Question Tokens', fontsize=12, fontweight='bold')
        ax.set_ylabel('Answer Tokens', fontsize=12, fontweight='bold')
        ax.set_title('Question vs Answer Token Relationship', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add legend for difficulty colors
        patches = [mpatches.Patch(color=difficulty_map.get(d, '#95a5a6'), label=d.capitalize()) 
                  for d in set(qa_data['difficulties'])]
        ax.legend(handles=patches, loc='upper left')
        
        plt.tight_layout()
        scatter_file = output_path / "question_answer_token_scatter.png"
        plt.savefig(scatter_file, dpi=dpi, bbox_inches='tight')
        plot_files['token_scatter'] = str(scatter_file)
        plt.close()
        
        # 6. Question Type Distribution (by technical type)
        fig, ax = plt.subplots(figsize=(12, 6))
        
        type_counts = defaultdict(int)
        for t in qa_data['question_types']:
            type_counts[t] += 1
        
        # Sort by count
        sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
        types = [t[0] for t in sorted_types]
        counts = [t[1] for t in sorted_types]
        
        bars = ax.bar(types, counts, edgecolor='black', linewidth=1.5, alpha=0.8)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax.set_xlabel('Question Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Questions', fontsize=12, fontweight='bold')
        ax.set_title('Question Distribution by Technical Type', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        type_file = output_path / "question_type_distribution.png"
        plt.savefig(type_file, dpi=dpi, bbox_inches='tight')
        plot_files['question_type'] = str(type_file)
        plt.close()
        
        # 7. Single vs Multi-Design Comparison Plots
        if single_design_data or multi_design_data:
            # 7a. Token Distribution Comparison
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Question tokens comparison
            if single_design_data:
                single_q_tokens = [d['q_tokens'] for d in single_design_data]
                axes[0, 0].hist(single_q_tokens, bins=25, alpha=0.6, label='Single Design', 
                               color='#3498db', edgecolor='black', linewidth=0.5)
                axes[0, 0].axvline(np.mean(single_q_tokens), color='blue', linestyle='--', 
                                  linewidth=2, label=f'Mean: {np.mean(single_q_tokens):.1f}')
            
            if multi_design_data:
                multi_q_tokens = [d['q_tokens'] for d in multi_design_data]
                axes[0, 0].hist(multi_q_tokens, bins=25, alpha=0.6, label='Multi Design', 
                               color='#e74c3c', edgecolor='black', linewidth=0.5)
                axes[0, 0].axvline(np.mean(multi_q_tokens), color='red', linestyle='--', 
                                  linewidth=2, label=f'Mean: {np.mean(multi_q_tokens):.1f}')
            
            axes[0, 0].set_xlabel('Tokens per Question', fontsize=11, fontweight='bold')
            axes[0, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
            axes[0, 0].set_title('Question Token Distribution: Single vs Multi-Design', 
                                fontsize=12, fontweight='bold')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Answer tokens comparison
            if single_design_data:
                single_a_tokens = [d['a_tokens'] for d in single_design_data]
                axes[0, 1].hist(single_a_tokens, bins=25, alpha=0.6, label='Single Design', 
                               color='#3498db', edgecolor='black', linewidth=0.5)
                axes[0, 1].axvline(np.mean(single_a_tokens), color='blue', linestyle='--', 
                                  linewidth=2, label=f'Mean: {np.mean(single_a_tokens):.1f}')
            
            if multi_design_data:
                multi_a_tokens = [d['a_tokens'] for d in multi_design_data]
                axes[0, 1].hist(multi_a_tokens, bins=25, alpha=0.6, label='Multi Design', 
                               color='#e74c3c', edgecolor='black', linewidth=0.5)
                axes[0, 1].axvline(np.mean(multi_a_tokens), color='red', linestyle='--', 
                                  linewidth=2, label=f'Mean: {np.mean(multi_a_tokens):.1f}')
            
            axes[0, 1].set_xlabel('Tokens per Answer', fontsize=11, fontweight='bold')
            axes[0, 1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
            axes[0, 1].set_title('Answer Token Distribution: Single vs Multi-Design', 
                                fontsize=12, fontweight='bold')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Difficulty distribution comparison
            single_diffs = [d['difficulty'] for d in single_design_data] if single_design_data else []
            multi_diffs = [d['difficulty'] for d in multi_design_data] if multi_design_data else []
            
            all_diffs = list(set(single_diffs + multi_diffs))
            if all_diffs:
                single_diff_counts = [single_diffs.count(d) for d in all_diffs]
                multi_diff_counts = [multi_diffs.count(d) for d in all_diffs]
                
                x = np.arange(len(all_diffs))
                width = 0.35
                
                bars1 = axes[1, 0].bar(x - width/2, single_diff_counts, width, 
                                      label='Single Design', color='#3498db', 
                                      edgecolor='black', linewidth=1, alpha=0.8)
                bars2 = axes[1, 0].bar(x + width/2, multi_diff_counts, width, 
                                      label='Multi Design', color='#e74c3c', 
                                      edgecolor='black', linewidth=1, alpha=0.8)
                
                # Add value labels
                for bars in [bars1, bars2]:
                    for bar in bars:
                        height = bar.get_height()
                        if height > 0:
                            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                                           f'{int(height)}', ha='center', va='bottom', 
                                           fontweight='bold', fontsize=9)
                
                axes[1, 0].set_xlabel('Difficulty Level', fontsize=11, fontweight='bold')
                axes[1, 0].set_ylabel('Number of Questions', fontsize=11, fontweight='bold')
                axes[1, 0].set_title('Difficulty Distribution: Single vs Multi-Design', 
                                     fontsize=12, fontweight='bold')
                axes[1, 0].set_xticks(x)
                axes[1, 0].set_xticklabels(all_diffs)
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3, axis='y')
            
            # Question type distribution comparison
            single_types = [d['question_type'] for d in single_design_data] if single_design_data else []
            multi_types = [d['question_type'] for d in multi_design_data] if multi_design_data else []
            
            all_types = list(set(single_types + multi_types))
            if all_types:
                single_type_counts = [single_types.count(t) for t in all_types]
                multi_type_counts = [multi_types.count(t) for t in all_types]
                
                # Sort by total count
                type_data = list(zip(all_types, single_type_counts, multi_type_counts))
                type_data.sort(key=lambda x: x[1] + x[2], reverse=True)
                all_types, single_type_counts, multi_type_counts = zip(*type_data[:10])  # Top 10
                
                x = np.arange(len(all_types))
                width = 0.35
                
                bars1 = axes[1, 1].bar(x - width/2, single_type_counts, width, 
                                      label='Single Design', color='#3498db', 
                                      edgecolor='black', linewidth=1, alpha=0.8)
                bars2 = axes[1, 1].bar(x + width/2, multi_type_counts, width, 
                                      label='Multi Design', color='#e74c3c', 
                                      edgecolor='black', linewidth=1, alpha=0.8)
                
                # Add value labels
                for bars in [bars1, bars2]:
                    for bar in bars:
                        height = bar.get_height()
                        if height > 0:
                            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                                           f'{int(height)}', ha='center', va='bottom', 
                                           fontweight='bold', fontsize=8)
                
                axes[1, 1].set_xlabel('Question Type', fontsize=11, fontweight='bold')
                axes[1, 1].set_ylabel('Number of Questions', fontsize=11, fontweight='bold')
                axes[1, 1].set_title('Question Type Distribution: Single vs Multi-Design', 
                                     fontsize=12, fontweight='bold')
                axes[1, 1].set_xticks(x)
                axes[1, 1].set_xticklabels(all_types, rotation=45, ha='right')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3, axis='y')
            
            plt.suptitle('Single Design vs Multi-Design Comparison', fontsize=14, fontweight='bold', y=0.995)
            plt.tight_layout()
            comparison_file = output_path / "single_vs_multi_design_comparison.png"
            plt.savefig(comparison_file, dpi=dpi, bbox_inches='tight')
            plot_files['single_vs_multi_comparison'] = str(comparison_file)
            plt.close()
            
            # 7b. Statistical Summary Comparison
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.axis('off')
            
            # Prepare statistics
            stats_text = "STATISTICAL COMPARISON: SINGLE DESIGN vs MULTI-DESIGN\n"
            stats_text += "=" * 80 + "\n\n"
            
            if single_design_data:
                single_q_tokens = [d['q_tokens'] for d in single_design_data]
                single_a_tokens = [d['a_tokens'] for d in single_design_data]
                stats_text += f"SINGLE DESIGN (n={len(single_design_data)}):\n"
                stats_text += f"  Question Tokens:\n"
                stats_text += f"    • Mean: {np.mean(single_q_tokens):.1f}\n"
                stats_text += f"    • Median: {np.median(single_q_tokens):.1f}\n"
                stats_text += f"    • Std: {np.std(single_q_tokens):.1f}\n"
                stats_text += f"    • Min: {np.min(single_q_tokens)}\n"
                stats_text += f"    • Max: {np.max(single_q_tokens)}\n"
                stats_text += f"  Answer Tokens:\n"
                stats_text += f"    • Mean: {np.mean(single_a_tokens):.1f}\n"
                stats_text += f"    • Median: {np.median(single_a_tokens):.1f}\n"
                stats_text += f"    • Std: {np.std(single_a_tokens):.1f}\n"
                stats_text += f"    • Min: {np.min(single_a_tokens)}\n"
                stats_text += f"    • Max: {np.max(single_a_tokens)}\n\n"
            
            if multi_design_data:
                multi_q_tokens = [d['q_tokens'] for d in multi_design_data]
                multi_a_tokens = [d['a_tokens'] for d in multi_design_data]
                stats_text += f"MULTI-DESIGN (n={len(multi_design_data)}):\n"
                stats_text += f"  Question Tokens:\n"
                stats_text += f"    • Mean: {np.mean(multi_q_tokens):.1f}\n"
                stats_text += f"    • Median: {np.median(multi_q_tokens):.1f}\n"
                stats_text += f"    • Std: {np.std(multi_q_tokens):.1f}\n"
                stats_text += f"    • Min: {np.min(multi_q_tokens)}\n"
                stats_text += f"    • Max: {np.max(multi_q_tokens)}\n"
                stats_text += f"  Answer Tokens:\n"
                stats_text += f"    • Mean: {np.mean(multi_a_tokens):.1f}\n"
                stats_text += f"    • Median: {np.median(multi_a_tokens):.1f}\n"
                stats_text += f"    • Std: {np.std(multi_a_tokens):.1f}\n"
                stats_text += f"    • Min: {np.min(multi_a_tokens)}\n"
                stats_text += f"    • Max: {np.max(multi_a_tokens)}\n\n"
            
            if single_design_data and multi_design_data:
                single_q_tokens = [d['q_tokens'] for d in single_design_data]
                multi_q_tokens = [d['q_tokens'] for d in multi_design_data]
                single_a_tokens = [d['a_tokens'] for d in single_design_data]
                multi_a_tokens = [d['a_tokens'] for d in multi_design_data]
                
                stats_text += "DIFFERENCES:\n"
                stats_text += f"  Question Tokens:\n"
                stats_text += f"    • Mean difference: {np.mean(multi_q_tokens) - np.mean(single_q_tokens):.1f}\n"
                stats_text += f"    • Median difference: {np.median(multi_q_tokens) - np.median(single_q_tokens):.1f}\n"
                stats_text += f"  Answer Tokens:\n"
                stats_text += f"    • Mean difference: {np.mean(multi_a_tokens) - np.mean(single_a_tokens):.1f}\n"
                stats_text += f"    • Median difference: {np.median(multi_a_tokens) - np.median(single_a_tokens):.1f}\n"
            
            ax.text(0.05, 0.95, stats_text, fontsize=10, verticalalignment='top',
                   family='monospace', fontweight='bold', transform=ax.transAxes)
            
            stats_comparison_file = output_path / "single_vs_multi_design_statistics.png"
            plt.savefig(stats_comparison_file, dpi=dpi, bbox_inches='tight')
            plot_files['single_vs_multi_statistics'] = str(stats_comparison_file)
            plt.close()
        
        # 8. Combined Summary Statistics Plot
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Token stats summary
        ax1 = fig.add_subplot(gs[0, 0])
        token_stats = {
            'Mean Q': np.mean(qa_data['question_tokens']),
            'Mean A': np.mean(qa_data['answer_tokens']),
            'Median Q': np.median(qa_data['question_tokens']),
            'Median A': np.median(qa_data['answer_tokens'])
        }
        ax1.bar(token_stats.keys(), token_stats.values(), color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'],
               edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('Tokens', fontweight='bold')
        ax1.set_title('Token Statistics Summary', fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.tick_params(axis='x', rotation=45)
        
        # Difficulty pie chart
        ax2 = fig.add_subplot(gs[0, 1])
        difficulty_counts_pie = defaultdict(int)
        for d in qa_data['difficulties']:
            difficulty_counts_pie[d] += 1
        ax2.pie(difficulty_counts_pie.values(), labels=difficulty_counts_pie.keys(), 
               autopct='%1.1f%%', startangle=90, textprops={'fontweight': 'bold'})
        ax2.set_title('Difficulty Distribution', fontweight='bold')
        
        # Category distribution (top 5)
        ax3 = fig.add_subplot(gs[0, 2])
        top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        ax3.barh([c[0] for c in top_categories], [c[1] for c in top_categories],
                edgecolor='black', linewidth=1)
        ax3.set_xlabel('Count', fontweight='bold')
        ax3.set_title('Top 5 Question Categories', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')
        
        # Token distribution overlay
        ax4 = fig.add_subplot(gs[1, :])
        ax4.hist(qa_data['question_tokens'], bins=30, alpha=0.5, label='Questions', 
                edgecolor='black', linewidth=0.5)
        ax4.hist(qa_data['answer_tokens'], bins=30, alpha=0.5, label='Answers',
                edgecolor='black', linewidth=0.5)
        ax4.set_xlabel('Tokens', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax4.set_title('Token Distribution: Questions vs Answers', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Question type distribution
        ax5 = fig.add_subplot(gs[2, :2])
        ax5.bar(types, counts, edgecolor='black', linewidth=1.5, alpha=0.8)
        ax5.set_xlabel('Question Type', fontweight='bold')
        ax5.set_ylabel('Count', fontweight='bold')
        ax5.set_title('Question Type Distribution', fontweight='bold')
        ax5.tick_params(axis='x', rotation=45)
        ax5.grid(True, alpha=0.3, axis='y')
        
        # Summary statistics text (use extracted total when available for consistency with prints)
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.axis('off')
        
        single_count = len(single_design_data) if single_design_data else 0
        multi_count = len(multi_design_data) if multi_design_data else 0
        total_for_summary = len(qa_data['question_tokens'])
        if getattr(self, '_last_stats', None) and 'total_qa_pairs_extracted' in self._last_stats:
            total_for_summary = self._last_stats['total_qa_pairs_extracted']
        
        stats_text = f"""
        Summary Statistics
        
        Total Questions (extracted): {total_for_summary}
        • Single Design: {single_count}
        • Multi Design: {multi_count}
        
        Question Tokens:
        • Mean: {np.mean(qa_data['question_tokens']):.1f}
        • Median: {np.median(qa_data['question_tokens']):.1f}
        • Std: {np.std(qa_data['question_tokens']):.1f}
        • Min: {np.min(qa_data['question_tokens'])}
        • Max: {np.max(qa_data['question_tokens'])}
        
        Answer Tokens:
        • Mean: {np.mean(qa_data['answer_tokens']):.1f}
        • Median: {np.median(qa_data['answer_tokens']):.1f}
        • Std: {np.std(qa_data['answer_tokens']):.1f}
        • Min: {np.min(qa_data['answer_tokens'])}
        • Max: {np.max(qa_data['answer_tokens'])}
        """
        ax6.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
                family='monospace', fontweight='bold')
        
        plt.suptitle('QA Statistics Comprehensive Overview', fontsize=16, fontweight='bold', y=0.98)
        
        summary_file = output_path / "qa_statistics_summary.png"
        plt.savefig(summary_file, dpi=dpi, bbox_inches='tight')
        plot_files['summary'] = str(summary_file)
        plt.close()
        
        # Stage combination / flow_execution summary: table (CSV) and bar plot
        stage_summary = self.get_stage_combination_summary()
        csv_path = self.export_stage_combination_table(output_dir)
        if csv_path:
            plot_files['stage_combination_table'] = csv_path
        has_single = bool(stage_summary.get('single_file'))
        has_multi = bool(stage_summary.get('multi_file'))
        if has_single or has_multi:
            n_sub = 2 if (has_single and has_multi) else 1
            fig, axes = plt.subplots(1, n_sub, figsize=(max(10, 6 * n_sub), 6))
            if n_sub == 1:
                axes = [axes]
            idx = 0
            if has_single:
                cats = list(stage_summary['single_file'].items())
                cats.sort(key=lambda x: -x[1])
                labels = [c[0] for c in cats]
                counts = [c[1] for c in cats]
                ax = axes[idx]
                bars = ax.barh(range(len(labels)), counts, edgecolor='black', linewidth=0.8, alpha=0.8)
                ax.set_yticks(range(len(labels)))
                ax.set_yticklabels(labels, fontsize=9)
                ax.set_xlabel('Question count', fontweight='bold')
                ax.set_title('Single-file: Stage combination / flow execution', fontweight='bold')
                ax.grid(True, alpha=0.3, axis='x')
                idx += 1
            if has_multi:
                cats = list(stage_summary['multi_file'].items())
                cats.sort(key=lambda x: -x[1])
                labels = [c[0] for c in cats]
                counts = [c[1] for c in cats]
                ax = axes[idx]
                ax.barh(range(len(labels)), counts, edgecolor='black', linewidth=0.8, alpha=0.8, color='#e74c3c')
                ax.set_yticks(range(len(labels)))
                ax.set_yticklabels(labels, fontsize=9)
                ax.set_xlabel('Question count', fontweight='bold')
                ax.set_title('Multi-file: Stage combination / flow execution', fontweight='bold')
                ax.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            stage_plot_file = output_path / "stage_combination_summary.png"
            plt.savefig(stage_plot_file, dpi=dpi, bbox_inches='tight')
            plot_files['stage_combination_summary'] = str(stage_plot_file)
            plt.close()
        
        # Design+tech x categorical stage columns: CSV and plots (heatmap + totals bar)
        dt_csv_path = self.export_stage_combination_by_design_tech_csv(output_dir)
        if dt_csv_path:
            plot_files['stage_combination_by_design_tech_table'] = dt_csv_path
        # Dump design_tech -> (qid, stages, category) for pass@k per category
        dump_path = output_path / "design_tech_qa_list.json"
        self.dump_design_tech_to_qa_list(output_path=str(dump_path))
        plot_files['design_tech_qa_list'] = str(dump_path)
        by_design_tech, cat_columns = self.get_stage_combination_by_design_tech_categorical()
        if by_design_tech and cat_columns:
            design_techs = sorted(by_design_tech.keys(), key=lambda dt: -by_design_tech[dt].get('total', 0))
            matrix = np.array([
                [by_design_tech[dt].get(c, 0) for c in cat_columns]
                for dt in design_techs
            ])
            if matrix.size > 0:
                fig2, axes2 = plt.subplots(2, 1, figsize=(max(12, len(cat_columns) * 0.7), max(6, len(design_techs) * 0.35)), height_ratios=[1.2, 0.8])
                # Heatmap: design_tech (rows) x Flow / single stages / n-Stages (columns)
                ax_hm = axes2[0]
                im = ax_hm.imshow(matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest', vmin=0)
                ax_hm.set_yticks(range(len(design_techs)))
                ax_hm.set_yticklabels(design_techs, fontsize=8)
                ax_hm.set_xticks(range(len(cat_columns)))
                ax_hm.set_xticklabels(cat_columns, rotation=45, ha='right', fontsize=8)
                ax_hm.set_xlabel('Stage category (Flow, single stages, n-Stages)', fontweight='bold')
                ax_hm.set_ylabel('Design + technology', fontweight='bold')
                ax_hm.set_title('QA count by design+tech: Flow Execution, single stages, and multi-stage', fontweight='bold')
                plt.colorbar(im, ax=ax_hm, label='Question count')
                # Totals bar chart
                ax_bar = axes2[1]
                totals = [by_design_tech[dt].get('total', 0) for dt in design_techs]
                colors = ['#3498db' if by_design_tech[dt].get('total', 0) > 0 else '#bdc3c7' for dt in design_techs]
                ax_bar.barh(range(len(design_techs)), totals, color=colors, edgecolor='black', linewidth=0.6)
                ax_bar.set_yticks(range(len(design_techs)))
                ax_bar.set_yticklabels(design_techs, fontsize=8)
                ax_bar.set_xlabel('Total questions', fontweight='bold')
                ax_bar.set_title('Total QA count per design+technology', fontweight='bold')
                ax_bar.grid(True, alpha=0.3, axis='x')
                plt.tight_layout()
                dt_plot_file = output_path / "stage_combination_by_design_tech.png"
                plt.savefig(dt_plot_file, dpi=dpi, bbox_inches='tight')
                plot_files['stage_combination_by_design_tech'] = str(dt_plot_file)
                plt.close()
        
        print(f"\n✅ Generated {len(plot_files)} plots in {output_dir}/")
        for name, path in plot_files.items():
            print(f"   - {name}: {path}")
        
        return plot_files


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze QA generation statistics and cluster rejection reasons"
    )
    parser.add_argument(
        '--qa-output-dir',
        type=str,
        default='qa_output',
        help='Directory containing QA output JSON files (default: qa_output)'
    )
    parser.add_argument(
        '--n-clusters',
        type=int,
        default=5,
        help='Number of clusters for KMeans (default: 5)'
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['kmeans', 'dbscan'],
        default='kmeans',
        help='Clustering method to use (default: kmeans)'
    )
    parser.add_argument(
        '--export',
        type=str,
        default='qa_stats_summary_output.json',
        help='Output file for results (default: qa_stats_summary_output.json)'
    )
    parser.add_argument(
        '--no-clustering',
        action='store_true',
        help='Skip clustering analysis (only show statistics)'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate visualization plots of QA statistics'
    )
    parser.add_argument(
        '--plot-dir',
        type=str,
        default='qa_plots',
        help='Directory to save plots (default: qa_plots); ignored if --output-dir is set'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        metavar='DIR',
        help='Write all outputs (JSON export and PNG plots) to this directory; creates DIR if needed'
    )
    parser.add_argument(
        '--max-representative',
        type=int,
        default=5,
        help='Max representative questions per stage-combination category (default: 5)'
    )
    
    args = parser.parse_args()
    
    # If --output-dir is set, put JSON and all PNGs there
    if args.output_dir:
        out_path = Path(args.output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        args.export = str(out_path / Path(args.export).name)
        args.plot_dir = str(out_path)
    
    # Initialize analyzer
    analyzer = QAStatsSummary(qa_output_dir=args.qa_output_dir)
    analyzer._max_representative = max(1, args.max_representative)
    
    # Load data
    analyzer.load_all_json_files()
    
    if not analyzer.data:
        print(f"❌ No JSON files found in {args.qa_output_dir}")
        return
    
    # Print statistics
    analyzer.print_summary()
    
    # Simple category analysis (always available, no dependencies)
    if analyzer.rejection_reasons:
        print("\n")
        category_analysis = analyzer.analyze_rejection_categories()
        analyzer.print_rejection_categories(category_analysis)
    
    # Perform clustering if not disabled
    clustering_results = None
    if not args.no_clustering:
        if not CLUSTERING_AVAILABLE:
            print(f"\n⚠️  Clustering analysis skipped: dependencies not available")
            print(f"   Install with: pip install -r qa_stats_requirements.txt")
        elif analyzer.rejection_reasons:
            clustering_results = analyzer.cluster_rejection_reasons(
                n_clusters=args.n_clusters,
                method=args.method
            )
            analyzer.print_clustering_results(clustering_results)
        else:
            print(f"\n⚠️  No rejection reasons found for clustering")
    
    # Export results
    analyzer.export_results(
        output_file=args.export,
        include_clustering=(not args.no_clustering and CLUSTERING_AVAILABLE)
    )
    
    # Write stage-combination summary table and design+tech table to output dir when designated
    if args.output_dir:
        csv_path = analyzer.export_stage_combination_table(args.output_dir)
        if csv_path:
            print(f"\n✅ Stage combination table written to: {csv_path}")
        dt_csv_path = analyzer.export_stage_combination_by_design_tech_csv(args.output_dir)
        if dt_csv_path:
            print(f"✅ Stage combination by design+tech table written to: {dt_csv_path}")
        dump_path = Path(args.output_dir) / "design_tech_qa_list.json"
        analyzer.dump_design_tech_to_qa_list(output_path=str(dump_path))
        print(f"✅ Design-tech -> (qid, stages, category) written to: {dump_path}")
    # Write extraction diff report when any file has statistics total != extracted
    report_dir = args.output_dir or str(Path(args.export).parent)
    diff_report = analyzer.write_extraction_diff_report(report_dir)
    if diff_report:
        print(f"✅ Extraction difference report written to: {diff_report}")
    
    # Check for duplicate question IDs within the same design_tech
    analyzer.print_duplicate_question_ids_report()
    
    # Generate plots if requested
    if args.plot:
        print("\n" + "="*80)
        print("GENERATING VISUALIZATION PLOTS")
        print("="*80)
        try:
            plot_files = analyzer.plot_qa_statistics(output_dir=args.plot_dir)
            if plot_files:
                print(f"\n✅ Successfully generated {len(plot_files)} plots")
        except Exception as e:
            print(f"\n❌ Error generating plots: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
