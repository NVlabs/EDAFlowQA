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
Analyze log files to count lines per stage for complexity comparison.
"""

import re
import json
import sys

def analyze_log_file(filename):
    """Count lines per stage in log file"""
    
    # Pattern to detect stage markers
    stage_pattern = r'Running .+?, stage (\S+)'
    
    stages = []
    total_lines = 0
    
    print(f"Analyzing {filename}...")
    
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        for line_num, line in enumerate(f, 1):
            total_lines += 1
            match = re.search(stage_pattern, line)
            if match:
                stage_name = match.group(1)
                stages.append({
                    'line_num': line_num,
                    'name': stage_name
                })
    
    # Calculate line counts for each stage
    stage_info = []
    for i, stage in enumerate(stages):
        start_line = stage['line_num']
        if i < len(stages) - 1:
            end_line = stages[i + 1]['line_num'] - 1
        else:
            end_line = total_lines
        
        line_count = end_line - start_line + 1
        
        # Categorize stage by prefix
        stage_name = stage['name']
        if stage_name.startswith('1_'):
            category = 'synthesis'
        elif stage_name.startswith('2_'):
            category = 'floorplan'
        elif stage_name.startswith('3_'):
            category = 'placement'
        elif stage_name.startswith('4_'):
            category = 'cts'
        elif stage_name.startswith('5_'):
            category = 'routing'
        elif stage_name.startswith('6_'):
            category = 'finish'
        else:
            category = 'other'
        
        stage_info.append({
            'name': stage_name,
            'category': category,
            'start_line': start_line,
            'end_line': end_line,
            'line_count': line_count
        })
    
    # Aggregate by category
    category_totals = {}
    for stage in stage_info:
        cat = stage['category']
        if cat not in category_totals:
            category_totals[cat] = 0
        category_totals[cat] += stage['line_count']
    
    return {
        'filename': filename,
        'total_lines': total_lines,
        'stages': stage_info,
        'category_totals': category_totals
    }

def main():
    # Analyze both files
    files = ['aes_nangate45.txt', 'aes_sky130hd.txt']
    results = {}
    
    for filename in files:
        results[filename] = analyze_log_file(filename)
    
    # Print results
    print("\n" + "="*80)
    print("LINE COUNT ANALYSIS BY STAGE")
    print("="*80 + "\n")
    
    for filename in files:
        design_name = filename.replace('.txt', '')
        data = results[filename]
        
        print(f"\n{design_name}:")
        print(f"  Total lines: {data['total_lines']}")
        print(f"  Number of stages: {len(data['stages'])}\n")
        
        print(f"  {'Stage Name':<30s} {'Category':<12s} {'Line Count':<10s}")
        print("  " + "-"*55)
        for stage in data['stages']:
            print(f"  {stage['name']:<30s} {stage['category']:<12s} {stage['line_count']:<10d}")
        
        print(f"\n  Category Totals:")
        print(f"  {'Category':<15s} {'Line Count':<12s} {'Percentage':<10s}")
        print("  " + "-"*40)
        for cat in ['synthesis', 'floorplan', 'placement', 'cts', 'routing', 'finish']:
            count = data['category_totals'].get(cat, 0)
            pct = (count / data['total_lines'] * 100) if data['total_lines'] > 0 else 0
            print(f"  {cat:<15s} {count:<12d} {pct:>6.2f}%")
    
    # Comparison
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80 + "\n")
    
    design1 = 'aes_nangate45'
    design2 = 'aes_sky130hd'
    data1 = results['aes_nangate45.txt']
    data2 = results['aes_sky130hd.txt']
    
    print(f"{'Category':<15s} {design1:<20s} {design2:<20s} {'Higher Complexity':<25s}")
    print("-"*85)
    
    comparison = {}
    for cat in ['synthesis', 'floorplan', 'placement', 'cts', 'routing', 'finish']:
        count1 = data1['category_totals'].get(cat, 0)
        count2 = data2['category_totals'].get(cat, 0)
        
        if count1 > count2:
            winner = design1
            diff = count1 - count2
            diff_pct = ((count1 - count2) / count2 * 100) if count2 > 0 else 100
        elif count2 > count1:
            winner = design2
            diff = count2 - count1
            diff_pct = ((count2 - count1) / count1 * 100) if count1 > 0 else 100
        else:
            winner = "Equal"
            diff = 0
            diff_pct = 0
        
        comparison[cat] = {
            f'{design1}_lines': count1,
            f'{design2}_lines': count2,
            'higher_complexity': winner,
            'difference': diff,
            'difference_pct': round(diff_pct, 2)
        }
        
        winner_str = f"{winner} (+{diff}, +{diff_pct:.1f}%)" if diff > 0 else winner
        print(f"{cat:<15s} {count1:<20d} {count2:<20d} {winner_str:<25s}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80 + "\n")
    
    print(f"Total Lines:")
    print(f"  {design1}: {data1['total_lines']} lines")
    print(f"  {design2}: {data2['total_lines']} lines")
    
    if data1['total_lines'] > data2['total_lines']:
        overall_winner = design1
        overall_diff = data1['total_lines'] - data2['total_lines']
        overall_pct = (overall_diff / data2['total_lines'] * 100)
    else:
        overall_winner = design2
        overall_diff = data2['total_lines'] - data1['total_lines']
        overall_pct = (overall_diff / data1['total_lines'] * 100)
    
    print(f"\n  Higher overall complexity: {overall_winner} (+{overall_diff} lines, +{overall_pct:.1f}%)")
    
    # Key findings for placement and routing
    print(f"\nKey Stages Analysis:")
    print(f"\n  Placement Stage:")
    print(f"    {design1}: {data1['category_totals'].get('placement', 0)} lines")
    print(f"    {design2}: {data2['category_totals'].get('placement', 0)} lines")
    print(f"    Higher complexity: {comparison['placement']['higher_complexity']}")
    
    print(f"\n  Routing Stage:")
    print(f"    {design1}: {data1['category_totals'].get('routing', 0)} lines")
    print(f"    {design2}: {data2['category_totals'].get('routing', 0)} lines")
    print(f"    Higher complexity: {comparison['routing']['higher_complexity']}")
    
    # Create JSON output
    output = {
        'aes_nangate45_2026-02-02T11_19_30_370555': {
            'total_lines': data1['total_lines'],
            'total_stages': len(data1['stages']),
            'category_line_counts': data1['category_totals']
        },
        'aes_sky130hd_2026-02-02T11_19_30_742696': {
            'total_lines': data2['total_lines'],
            'total_stages': len(data2['stages']),
            'category_line_counts': data2['category_totals']
        },
        'comparison': comparison,
        'analysis': {
            'overall_higher_complexity': overall_winner,
            'placement_higher_complexity': comparison['placement']['higher_complexity'],
            'routing_higher_complexity': comparison['routing']['higher_complexity']
        }
    }
    
    print("\n" + "="*80)
    print("JSON OUTPUT")
    print("="*80)
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
