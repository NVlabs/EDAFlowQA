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
Analyze the comprehensive flow structure for both designs based on:
1. Number of sub-designs/stages processed
2. Number of lines extracted in each stage
"""

import re
import json

def analyze_flow_structure(log_file):
    """Analyze the flow structure of a log file."""
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    
    # Find all stage markers (flow.sh entries)
    stage_pattern = re.compile(r'flow\.sh\s+(\d+_\d+_[a-z_]+)\s+(\w+)')
    stages = []
    stage_line_numbers = []
    
    for i, line in enumerate(lines):
        match = stage_pattern.search(line)
        if match:
            stage_id = match.group(1)
            stage_name = match.group(2)
            stages.append({
                'stage_id': stage_id,
                'stage_name': stage_name,
                'line_number': i + 1
            })
            stage_line_numbers.append(i + 1)
    
    # Calculate lines per stage
    stage_line_counts = []
    for i, stage in enumerate(stages):
        if i < len(stages) - 1:
            lines_in_stage = stage_line_numbers[i + 1] - stage_line_numbers[i]
        else:
            lines_in_stage = total_lines - stage_line_numbers[i]
        
        stage_line_counts.append({
            'stage_id': stage['stage_id'],
            'stage_name': stage['stage_name'],
            'line_count': lines_in_stage,
            'start_line': stage['line_number']
        })
    
    # Organize stages by main flow categories
    flow_categories = {
        'synthesis': [],
        'floorplan': [],
        'placement': [],
        'cts': [],
        'routing': [],
        'finishing': []
    }
    
    for stage in stage_line_counts:
        stage_id = stage['stage_id']
        if stage_id.startswith('1_'):
            flow_categories['synthesis'].append(stage)
        elif stage_id.startswith('2_'):
            flow_categories['floorplan'].append(stage)
        elif stage_id.startswith('3_'):
            flow_categories['placement'].append(stage)
        elif stage_id.startswith('4_'):
            flow_categories['cts'].append(stage)
        elif stage_id.startswith('5_'):
            flow_categories['routing'].append(stage)
        elif stage_id.startswith('6_'):
            flow_categories['finishing'].append(stage)
    
    return {
        'total_lines': total_lines,
        'num_stages': len(stages),
        'stages': stages,
        'stage_line_counts': stage_line_counts,
        'flow_categories': flow_categories
    }

def main():
    # Analyze both designs
    nangate45_result = analyze_flow_structure('aes_nangate45.txt')
    sky130hd_result = analyze_flow_structure('aes_sky130hd.txt')
    
    print("=" * 80)
    print("AES NANGATE45 Flow Analysis")
    print("=" * 80)
    print(f"Total lines: {nangate45_result['total_lines']}")
    print(f"Number of stages: {nangate45_result['num_stages']}")
    print(f"\nStage breakdown by category:")
    for category, stages in nangate45_result['flow_categories'].items():
        if stages:
            total_lines = sum(s['line_count'] for s in stages)
            print(f"  {category.upper()}: {len(stages)} stages, {total_lines} lines")
            for stage in stages:
                print(f"    - {stage['stage_id']} ({stage['stage_name']}): {stage['line_count']} lines")
    
    print("\n" + "=" * 80)
    print("AES SKY130HD Flow Analysis")
    print("=" * 80)
    print(f"Total lines: {sky130hd_result['total_lines']}")
    print(f"Number of stages: {sky130hd_result['num_stages']}")
    print(f"\nStage breakdown by category:")
    for category, stages in sky130hd_result['flow_categories'].items():
        if stages:
            total_lines = sum(s['line_count'] for s in stages)
            print(f"  {category.upper()}: {len(stages)} stages, {total_lines} lines")
            for stage in stages:
                print(f"    - {stage['stage_id']} ({stage['stage_name']}): {stage['line_count']} lines")
    
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    
    # Compare stages
    print(f"\nStage Count:")
    print(f"  NANGATE45: {nangate45_result['num_stages']} stages")
    print(f"  SKY130HD:  {sky130hd_result['num_stages']} stages")
    print(f"  Difference: {abs(nangate45_result['num_stages'] - sky130hd_result['num_stages'])} stages")
    
    # Compare total lines
    print(f"\nTotal Line Count:")
    print(f"  NANGATE45: {nangate45_result['total_lines']} lines")
    print(f"  SKY130HD:  {sky130hd_result['total_lines']} lines")
    print(f"  Difference: {abs(nangate45_result['total_lines'] - sky130hd_result['total_lines'])} lines")
    
    # Compare by category
    print(f"\nCategory Comparison:")
    all_categories = set(nangate45_result['flow_categories'].keys()) | set(sky130hd_result['flow_categories'].keys())
    for category in sorted(all_categories):
        ng45_stages = nangate45_result['flow_categories'].get(category, [])
        sky_stages = sky130hd_result['flow_categories'].get(category, [])
        ng45_lines = sum(s['line_count'] for s in ng45_stages)
        sky_lines = sum(s['line_count'] for s in sky_stages)
        
        print(f"\n  {category.upper()}:")
        print(f"    NANGATE45: {len(ng45_stages)} stages, {ng45_lines} lines")
        print(f"    SKY130HD:  {len(sky_stages)} stages, {sky_lines} lines")
    
    # Determine which is more comprehensive
    print("\n" + "=" * 80)
    print("COMPREHENSIVE FLOW ASSESSMENT")
    print("=" * 80)
    
    comparison = {
        'nangate45': {
            'total_lines': nangate45_result['total_lines'],
            'num_stages': nangate45_result['num_stages'],
            'stages_by_category': {cat: len(stages) for cat, stages in nangate45_result['flow_categories'].items()},
            'lines_by_category': {cat: sum(s['line_count'] for s in stages) for cat, stages in nangate45_result['flow_categories'].items()}
        },
        'sky130hd': {
            'total_lines': sky130hd_result['total_lines'],
            'num_stages': sky130hd_result['num_stages'],
            'stages_by_category': {cat: len(stages) for cat, stages in sky130hd_result['flow_categories'].items()},
            'lines_by_category': {cat: sum(s['line_count'] for s in stages) for cat, stages in sky130hd_result['flow_categories'].items()}
        }
    }
    
    # Determine which is more comprehensive
    if nangate45_result['num_stages'] == sky130hd_result['num_stages']:
        if nangate45_result['total_lines'] > sky130hd_result['total_lines']:
            more_comprehensive = 'aes_nangate45'
            reason = 'Same number of stages but more detailed logging (more lines per stage)'
        elif sky130hd_result['total_lines'] > nangate45_result['total_lines']:
            more_comprehensive = 'aes_sky130hd'
            reason = 'Same number of stages but more detailed logging (more lines per stage)'
        else:
            more_comprehensive = 'both_equal'
            reason = 'Same number of stages and total lines'
    elif nangate45_result['num_stages'] > sky130hd_result['num_stages']:
        more_comprehensive = 'aes_nangate45'
        reason = f'More stages processed ({nangate45_result["num_stages"]} vs {sky130hd_result["num_stages"]})'
    else:
        more_comprehensive = 'aes_sky130hd'
        reason = f'More stages processed ({sky130hd_result["num_stages"]} vs {nangate45_result["num_stages"]})'
    
    comparison['more_comprehensive'] = more_comprehensive
    comparison['reason'] = reason
    
    print(f"\nMore Comprehensive Design: {more_comprehensive}")
    print(f"Reason: {reason}")
    
    # Save detailed comparison
    with open('q6_flow_comparison.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print("\nDetailed comparison saved to q6_flow_comparison.json")
    
    return comparison

if __name__ == '__main__':
    main()
