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
QA Output Analyzer

This script analyzes all JSON files in the qa_output directory and generates
comprehensive statistics including:
- Number of questions per file
- Rejection rates and reasons
- Success rates
- Score distributions
- Question type distributions
- Design comparisons
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import statistics


def load_json_file(filepath: Path) -> Dict[str, Any]:
    """Load a JSON file and return its contents."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def is_raw_file(filepath: Path) -> bool:
    """Check if a file is a raw output file (should be excluded from main analysis)."""
    return filepath.name.endswith('_raw.json')


def analyze_single_design_file(data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze a single design QA output file."""
    stats = data.get('statistics', {})
    rejection_stats = data.get('rejection_statistics', {})
    
    # Extract basic statistics
    total_qa_pairs = stats.get('total_qa_pairs', 0)
    failed = stats.get('failed', 0)
    successful = total_qa_pairs - failed
    
    # Extract rejection statistics
    total_attempts = rejection_stats.get('total_attempts', 0)
    successful_attempts = rejection_stats.get('successful', 0)
    rejected_dict = rejection_stats.get('rejected', {})
    
    # Extract rejection reasons
    code_execution_failed = rejected_dict.get('code_execution_failed', 0)
    low_score_rejected = rejected_dict.get('low_score_rejected', 0)
    invalid_answer_rejected = rejected_dict.get('invalid_answer_rejected', 0)
    total_rejected = rejected_dict.get('total_rejected', 0)
    
    success_rate = rejection_stats.get('success_rate', 0.0)
    
    # Analyze low score details
    low_score_details = rejection_stats.get('low_score_details', [])
    scores = [detail.get('score', 0) for detail in low_score_details if 'score' in detail]
    
    avg_low_score = statistics.mean(scores) if scores else 0
    min_score = min(scores) if scores else 0
    max_score = max(scores) if scores else 0
    
    # Extract question type breakdown
    type_breakdown = {
        'flow_execution': stats.get('flow_execution', 0),
        'single_stage': stats.get('single_stage', 0),
        'multi_stage': stats.get('multi_stage', 0),
        'stage_to_final': stats.get('stage_to_final', 0),
    }
    
    return {
        'design_name': data.get('design_name', 'unknown'),
        'timestamp': data.get('timestamp', 'unknown'),
        'source': data.get('source', 'unknown'),
        'log_file': data.get('log_file', 'unknown'),
        'total_qa_pairs': total_qa_pairs,
        'successful_pairs': successful,
        'failed_pairs': failed,
        'total_attempts': total_attempts,
        'successful_attempts': successful_attempts,
        'total_rejected': total_rejected,
        'rejection_breakdown': {
            'code_execution_failed': code_execution_failed,
            'low_score_rejected': low_score_rejected,
            'invalid_answer_rejected': invalid_answer_rejected,
        },
        'success_rate': success_rate,
        'low_score_stats': {
            'count': len(scores),
            'avg_score': avg_low_score,
            'min_score': min_score,
            'max_score': max_score,
        },
        'type_breakdown': type_breakdown,
    }


def analyze_multifile_comparison(data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze a multi-file comparison QA output file."""
    stats = data.get('statistics', {})
    
    designs = data.get('designs', [])
    source_files = data.get('source_files', [])
    log_files = data.get('log_files', [])
    generation_params = data.get('generation_params', {})
    
    total_qa_pairs = stats.get('total_qa_pairs', 0)
    stage_comparison_pairs = stats.get('stage_comparison_pairs', 0)
    flow_execution_pairs = stats.get('flow_execution_pairs', 0)
    stages_compared = stats.get('stages_compared', [])
    
    return {
        'comparison_type': 'multifile',
        'designs': designs,
        'timestamp': data.get('timestamp', 'unknown'),
        'source_files': source_files,
        'log_files': log_files,
        'total_qa_pairs': total_qa_pairs,
        'stage_comparison_pairs': stage_comparison_pairs,
        'flow_execution_pairs': flow_execution_pairs,
        'stages_compared': stages_compared,
        'generation_params': generation_params,
    }


def determine_file_type(data: Dict[str, Any]) -> str:
    """Determine if a file is a single design or multifile comparison."""
    if 'designs' in data and isinstance(data.get('designs'), list) and len(data.get('designs', [])) > 1:
        return 'multifile'
    return 'single'


def analyze_qa_output_directory(qa_output_dir: str) -> Dict[str, Any]:
    """Analyze all JSON files in the qa_output directory."""
    qa_output_path = Path(qa_output_dir)
    
    if not qa_output_path.exists():
        raise ValueError(f"Directory {qa_output_dir} does not exist")
    
    # Find all JSON files (excluding raw files)
    json_files = [f for f in qa_output_path.rglob('*.json') if not is_raw_file(f)]
    
    print(f"Found {len(json_files)} JSON files to analyze")
    
    single_design_analyses = []
    multifile_analyses = []
    errors = []
    
    for json_file in json_files:
        print(f"Analyzing: {json_file}")
        data = load_json_file(json_file)
        
        if data is None:
            errors.append(str(json_file))
            continue
        
        try:
            file_type = determine_file_type(data)
            
            if file_type == 'multifile':
                analysis = analyze_multifile_comparison(data)
                analysis['file_path'] = str(json_file.relative_to(qa_output_path))
                multifile_analyses.append(analysis)
            else:
                analysis = analyze_single_design_file(data)
                analysis['file_path'] = str(json_file.relative_to(qa_output_path))
                single_design_analyses.append(analysis)
        except Exception as e:
            print(f"Error analyzing {json_file}: {e}")
            errors.append(f"{json_file}: {e}")
    
    # Compute aggregate statistics for single design files
    single_design_aggregates = compute_single_design_aggregates(single_design_analyses)
    
    # Compute aggregate statistics for multifile comparisons
    multifile_aggregates = compute_multifile_aggregates(multifile_analyses)
    
    return {
        'analysis_timestamp': datetime.now().isoformat(),
        'total_files_analyzed': len(json_files),
        'single_design_files': len(single_design_analyses),
        'multifile_comparison_files': len(multifile_analyses),
        'errors': errors,
        'single_design_summary': single_design_aggregates,
        'multifile_summary': multifile_aggregates,
        'single_design_details': single_design_analyses,
        'multifile_details': multifile_analyses,
    }


def compute_single_design_aggregates(analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate statistics across all single design analyses."""
    if not analyses:
        return {}
    
    total_qa_pairs = sum(a['total_qa_pairs'] for a in analyses)
    total_successful_pairs = sum(a['successful_pairs'] for a in analyses)
    total_failed_pairs = sum(a['failed_pairs'] for a in analyses)
    
    total_attempts = sum(a['total_attempts'] for a in analyses)
    total_successful_attempts = sum(a['successful_attempts'] for a in analyses)
    total_rejected = sum(a['total_rejected'] for a in analyses)
    
    # Rejection reasons
    total_code_execution_failed = sum(
        a['rejection_breakdown']['code_execution_failed'] for a in analyses
    )
    total_low_score_rejected = sum(
        a['rejection_breakdown']['low_score_rejected'] for a in analyses
    )
    total_invalid_answer_rejected = sum(
        a['rejection_breakdown']['invalid_answer_rejected'] for a in analyses
    )
    
    # Success rate calculation
    overall_success_rate = (total_successful_attempts / total_attempts * 100) if total_attempts > 0 else 0
    
    # Rejection rate calculation
    overall_rejection_rate = (total_rejected / total_attempts * 100) if total_attempts > 0 else 0
    
    # Question type aggregates
    type_totals = defaultdict(int)
    for analysis in analyses:
        for qtype, count in analysis['type_breakdown'].items():
            type_totals[qtype] += count
    
    # Design breakdown
    design_counts = defaultdict(int)
    for analysis in analyses:
        design_counts[analysis['design_name']] += 1
    
    # Score statistics
    all_low_scores = []
    for analysis in analyses:
        if analysis['low_score_stats']['count'] > 0:
            # We don't have individual scores, just the stats
            # But we can count how many files had low scores
            all_low_scores.append(analysis['low_score_stats']['avg_score'])
    
    avg_low_score_across_files = statistics.mean(all_low_scores) if all_low_scores else 0
    
    return {
        'total_qa_pairs': total_qa_pairs,
        'total_successful_pairs': total_successful_pairs,
        'total_failed_pairs': total_failed_pairs,
        'pair_success_rate': (total_successful_pairs / total_qa_pairs * 100) if total_qa_pairs > 0 else 0,
        'total_attempts': total_attempts,
        'total_successful_attempts': total_successful_attempts,
        'total_rejected': total_rejected,
        'overall_success_rate': overall_success_rate,
        'overall_rejection_rate': overall_rejection_rate,
        'rejection_breakdown': {
            'code_execution_failed': total_code_execution_failed,
            'code_execution_failed_pct': (total_code_execution_failed / total_rejected * 100) if total_rejected > 0 else 0,
            'low_score_rejected': total_low_score_rejected,
            'low_score_rejected_pct': (total_low_score_rejected / total_rejected * 100) if total_rejected > 0 else 0,
            'invalid_answer_rejected': total_invalid_answer_rejected,
            'invalid_answer_rejected_pct': (total_invalid_answer_rejected / total_rejected * 100) if total_rejected > 0 else 0,
        },
        'question_type_totals': dict(type_totals),
        'design_counts': dict(design_counts),
        'avg_low_score_across_files': avg_low_score_across_files,
        'files_with_low_scores': len(all_low_scores),
    }


def compute_multifile_aggregates(analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate statistics across all multifile comparison analyses."""
    if not analyses:
        return {}
    
    total_qa_pairs = sum(a['total_qa_pairs'] for a in analyses)
    total_stage_comparison_pairs = sum(a['stage_comparison_pairs'] for a in analyses)
    total_flow_execution_pairs = sum(a['flow_execution_pairs'] for a in analyses)
    
    # Collect all unique stages compared
    all_stages = set()
    for analysis in analyses:
        all_stages.update(analysis['stages_compared'])
    
    # Design pair counts
    design_pair_counts = defaultdict(int)
    for analysis in analyses:
        designs = tuple(sorted(analysis['designs']))
        design_pair_counts[designs] += 1
    
    return {
        'total_qa_pairs': total_qa_pairs,
        'total_stage_comparison_pairs': total_stage_comparison_pairs,
        'total_flow_execution_pairs': total_flow_execution_pairs,
        'unique_stages_compared': sorted(list(all_stages)),
        'design_pair_counts': {str(k): v for k, v in design_pair_counts.items()},
    }


def format_summary_report(analysis_results: Dict[str, Any]) -> str:
    """Format the analysis results into a human-readable report."""
    report_lines = []
    
    report_lines.append("=" * 80)
    report_lines.append("QA OUTPUT ANALYSIS SUMMARY")
    report_lines.append("=" * 80)
    report_lines.append(f"Analysis Timestamp: {analysis_results['analysis_timestamp']}")
    report_lines.append(f"Total Files Analyzed: {analysis_results['total_files_analyzed']}")
    report_lines.append(f"  - Single Design Files: {analysis_results['single_design_files']}")
    report_lines.append(f"  - Multifile Comparison Files: {analysis_results['multifile_comparison_files']}")
    
    if analysis_results['errors']:
        report_lines.append(f"\nErrors Encountered: {len(analysis_results['errors'])}")
        for error in analysis_results['errors']:
            report_lines.append(f"  - {error}")
    
    report_lines.append("\n" + "=" * 80)
    report_lines.append("SINGLE DESIGN SUMMARY")
    report_lines.append("=" * 80)
    
    single_summary = analysis_results.get('single_design_summary', {})
    if single_summary:
        report_lines.append(f"\nTotal QA Pairs Generated: {single_summary['total_qa_pairs']}")
        report_lines.append(f"  - Successful: {single_summary['total_successful_pairs']} ({single_summary['pair_success_rate']:.2f}%)")
        report_lines.append(f"  - Failed: {single_summary['total_failed_pairs']}")
        
        report_lines.append(f"\nGeneration Attempts:")
        report_lines.append(f"  - Total Attempts: {single_summary['total_attempts']}")
        report_lines.append(f"  - Successful: {single_summary['total_successful_attempts']} ({single_summary['overall_success_rate']:.2f}%)")
        report_lines.append(f"  - Rejected: {single_summary['total_rejected']} ({single_summary['overall_rejection_rate']:.2f}%)")
        
        report_lines.append(f"\nRejection Breakdown:")
        rejection = single_summary['rejection_breakdown']
        report_lines.append(f"  - Code Execution Failed: {rejection['code_execution_failed']} ({rejection['code_execution_failed_pct']:.2f}%)")
        report_lines.append(f"  - Low Score Rejected: {rejection['low_score_rejected']} ({rejection['low_score_rejected_pct']:.2f}%)")
        report_lines.append(f"  - Invalid Answer Rejected: {rejection['invalid_answer_rejected']} ({rejection['invalid_answer_rejected_pct']:.2f}%)")
        
        report_lines.append(f"\nQuestion Type Distribution:")
        for qtype, count in single_summary['question_type_totals'].items():
            pct = (count / single_summary['total_qa_pairs'] * 100) if single_summary['total_qa_pairs'] > 0 else 0
            report_lines.append(f"  - {qtype}: {count} ({pct:.2f}%)")
        
        report_lines.append(f"\nDesign Coverage:")
        for design, count in single_summary['design_counts'].items():
            report_lines.append(f"  - {design}: {count} file(s)")
        
        report_lines.append(f"\nLow Score Statistics:")
        report_lines.append(f"  - Files with Low Scores: {single_summary['files_with_low_scores']}")
        report_lines.append(f"  - Average Low Score (across files): {single_summary['avg_low_score_across_files']:.3f}")
    
    report_lines.append("\n" + "=" * 80)
    report_lines.append("MULTIFILE COMPARISON SUMMARY")
    report_lines.append("=" * 80)
    
    multifile_summary = analysis_results.get('multifile_summary', {})
    if multifile_summary:
        report_lines.append(f"\nTotal QA Pairs: {multifile_summary['total_qa_pairs']}")
        report_lines.append(f"  - Stage Comparison Pairs: {multifile_summary['total_stage_comparison_pairs']}")
        report_lines.append(f"  - Flow Execution Pairs: {multifile_summary['total_flow_execution_pairs']}")
        
        report_lines.append(f"\nStages Compared:")
        for stage in multifile_summary['unique_stages_compared']:
            report_lines.append(f"  - {stage}")
        
        report_lines.append(f"\nDesign Pair Comparisons:")
        for pair, count in multifile_summary['design_pair_counts'].items():
            report_lines.append(f"  - {pair}: {count} file(s)")
    
    report_lines.append("\n" + "=" * 80)
    
    return "\n".join(report_lines)


def save_analysis_results(analysis_results: Dict[str, Any], output_dir: str):
    """Save the analysis results to JSON and text files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save full JSON results
    json_output_file = output_path / f"qa_analysis_{timestamp}.json"
    with open(json_output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2)
    print(f"\nFull analysis saved to: {json_output_file}")
    
    # Save summary report
    summary_report = format_summary_report(analysis_results)
    txt_output_file = output_path / f"qa_analysis_summary_{timestamp}.txt"
    with open(txt_output_file, 'w', encoding='utf-8') as f:
        f.write(summary_report)
    print(f"Summary report saved to: {txt_output_file}")
    
    return json_output_file, txt_output_file


def main():
    """Main function to run the QA output analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze QA output JSON files and generate statistics"
    )
    parser.add_argument(
        '--qa-output-dir',
        type=str,
        default='qa_output',
        help='Directory containing QA output JSON files (default: qa_output)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='src',
        help='Directory to save analysis results (default: src)'
    )
    parser.add_argument(
        '--print-summary',
        action='store_true',
        help='Print summary to console'
    )
    
    args = parser.parse_args()
    
    print(f"Analyzing QA output directory: {args.qa_output_dir}")
    
    # Run analysis
    analysis_results = analyze_qa_output_directory(args.qa_output_dir)
    
    # Save results
    json_file, txt_file = save_analysis_results(analysis_results, args.output_dir)
    
    # Print summary if requested
    if args.print_summary:
        print("\n" + format_summary_report(analysis_results))
    
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
