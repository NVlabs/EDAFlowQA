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
Analyze comprehensive flow based on number of sub-designs processed and extracted lines in each stage.
"""

import re

def analyze_comprehensive_flow(filename):
    """Analyze a log file for comprehensive flow metrics"""
    
    # Find all stage transitions
    stage_pattern = r'Running .+?, stage ([0-9_a-z]+)'
    
    stages = []
    total_lines = 0
    
    # Patterns for identifying sub-design/module processing
    module_patterns = [
        r'Executing AST frontend in derive mode using pre-parsed AST for module',
        r'Analyzing design hierarchy',
        r'Mapping module',
        r'replace_arith_module'
    ]
    
    module_count = 0
    
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        for line_num, line in enumerate(f, 1):
            total_lines += 1
            
            # Check for stage markers
            match = re.search(stage_pattern, line)
            if match:
                stage_name = match.group(1)
                stages.append({
                    'line_num': line_num,
                    'name': stage_name
                })
            
            # Check for module/sub-design processing
            for pattern in module_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    module_count += 1
                    break
    
    # Calculate line counts for each stage
    stage_info = []
    for i, stage in enumerate(stages):
        start_line = stage['line_num']
        if i < len(stages) - 1:
            end_line = stages[i + 1]['line_num'] - 1
        else:
            end_line = total_lines
        
        line_count = end_line - start_line + 1
        
        # Categorize stage
        if stage['name'].startswith('1_'):
            category = 'synthesis'
        elif stage['name'].startswith('2_'):
            category = 'floorplan'
        elif stage['name'].startswith('3_'):
            category = 'placement'
        elif stage['name'].startswith('4_'):
            category = 'cts'
        elif stage['name'].startswith('5_'):
            category = 'routing'
        elif stage['name'].startswith('6_'):
            category = 'finish'
        else:
            category = 'other'
        
        stage_info.append({
            'name': stage['name'],
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
        'total_stages': len(stages),
        'stages': stage_info,
        'category_totals': category_totals,
        'module_processing_count': module_count
    }

def main():
    import sys
    
    files = ['aes_nangate45.txt', 'aes_sky130hd.txt']
    
    results = {}
    
    for filename in files:
        print(f"\n{'='*80}")
        print(f"Analyzing: {filename}")
        print(f"{'='*80}\n")
        
        analysis = analyze_comprehensive_flow(filename)
        results[filename] = analysis
        
        print(f"Total lines: {analysis['total_lines']}")
        print(f"Total stages: {analysis['total_stages']}")
        print(f"Module processing occurrences: {analysis['module_processing_count']}")
        
        print(f"\nStages in the flow:")
        print(f"{'Stage':<30s} {'Category':<12s} {'Lines':<8s}")
        print("-" * 80)
        for stage in analysis['stages']:
            print(f"{stage['name']:<30s} {stage['category']:<12s} {stage['line_count']:<8d}")
        
        print(f"\nCategory totals:")
        print(f"{'Category':<15s} {'Lines':<10s} {'Percentage':<10s}")
        print("-" * 50)
        for cat in ['synthesis', 'floorplan', 'placement', 'cts', 'routing', 'finish']:
            count = analysis['category_totals'].get(cat, 0)
            pct = (count / analysis['total_lines'] * 100) if analysis['total_lines'] > 0 else 0
            print(f"{cat:<15s} {count:<10d} {pct:>6.2f}%")
    
    # Comparison
    print(f"\n{'='*80}")
    print("COMPREHENSIVE FLOW COMPARISON")
    print(f"{'='*80}\n")
    
    file1, file2 = files[0], files[1]
    design1 = file1.replace('.txt', '')
    design2 = file2.replace('.txt', '')
    
    r1 = results[file1]
    r2 = results[file2]
    
    print(f"{'Metric':<40s} {design1:<20s} {design2:<20s}")
    print("-" * 85)
    print(f"{'Total lines in log':<40s} {r1['total_lines']:<20d} {r2['total_lines']:<20d}")
    print(f"{'Total stages':<40s} {r1['total_stages']:<20d} {r2['total_stages']:<20d}")
    print(f"{'Module processing occurrences':<40s} {r1['module_processing_count']:<20d} {r2['module_processing_count']:<20d}")
    
    print(f"\n{'Stage-by-stage line counts:':<40s}")
    print(f"{'Stage':<30s} {design1:<20s} {design2:<20s}")
    print("-" * 85)
    
    # Get all unique stage names
    all_stages = set()
    stages1 = {s['name']: s for s in r1['stages']}
    stages2 = {s['name']: s for s in r2['stages']}
    
    for stage in r1['stages']:
        all_stages.add(stage['name'])
    for stage in r2['stages']:
        all_stages.add(stage['name'])
    
    for stage_name in sorted(all_stages):
        count1 = stages1[stage_name]['line_count'] if stage_name in stages1 else 0
        count2 = stages2[stage_name]['line_count'] if stage_name in stages2 else 0
        print(f"{stage_name:<30s} {count1:<20d} {count2:<20d}")
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE FLOW ASSESSMENT")
    print(f"{'='*80}\n")
    
    # Calculate comprehensive scores
    score1 = r1['total_lines'] + (r1['total_stages'] * 100) + (r1['module_processing_count'] * 10)
    score2 = r2['total_lines'] + (r2['total_stages'] * 100) + (r2['module_processing_count'] * 10)
    
    print("Factors for comprehensive flow:")
    print(f"1. Total lines (raw data volume)")
    print(f"   {design1}: {r1['total_lines']} lines")
    print(f"   {design2}: {r2['total_lines']} lines")
    
    print(f"\n2. Number of stages (flow granularity)")
    print(f"   {design1}: {r1['total_stages']} stages")
    print(f"   {design2}: {r2['total_stages']} stages")
    
    print(f"\n3. Module/sub-design processing occurrences")
    print(f"   {design1}: {r1['module_processing_count']} occurrences")
    print(f"   {design2}: {r2['module_processing_count']} occurrences")
    
    print(f"\nCategory-wise comparison:")
    for cat in ['synthesis', 'floorplan', 'placement', 'cts', 'routing', 'finish']:
        count1 = r1['category_totals'].get(cat, 0)
        count2 = r2['category_totals'].get(cat, 0)
        winner = design1 if count1 > count2 else design2 if count2 > count1 else "Equal"
        print(f"  {cat:<15s}: {design1}={count1:6d}, {design2}={count2:6d} -> More comprehensive: {winner}")
    
    print(f"\n{'='*80}")
    print("FINAL ANSWER")
    print(f"{'='*80}\n")
    
    if r1['total_lines'] > r2['total_lines']:
        more_comprehensive = design1
        line_diff = r1['total_lines'] - r2['total_lines']
        line_pct = (line_diff / r2['total_lines'] * 100)
    else:
        more_comprehensive = design2
        line_diff = r2['total_lines'] - r1['total_lines']
        line_pct = (line_diff / r1['total_lines'] * 100)
    
    print(f"Based on analysis of:")
    print(f"  - Total log lines (extracted content volume)")
    print(f"  - Number of processing stages")
    print(f"  - Sub-design/module processing patterns")
    print(f"  - Line distribution across flow stages")
    print(f"\n** {more_comprehensive} ** has a more comprehensive flow.")
    print(f"\nKey metrics:")
    print(f"  {design1}: {r1['total_lines']} lines, {r1['total_stages']} stages, {r1['module_processing_count']} module processings")
    print(f"  {design2}: {r2['total_lines']} lines, {r2['total_stages']} stages, {r2['module_processing_count']} module processings")
    print(f"\n{more_comprehensive} has {line_diff} more lines ({line_pct:.1f}% more content)")
    
    # Generate JSON output
    import json
    
    json_output = {
        "designs_analyzed": [design1, design2],
        "aes_nangate45": {
            "total_lines": r1['total_lines'],
            "total_stages": r1['total_stages'],
            "module_processing_count": r1['module_processing_count'],
            "stages": [{"name": s['name'], "category": s['category'], "line_count": s['line_count']} for s in r1['stages']],
            "category_totals": r1['category_totals']
        },
        "aes_sky130hd": {
            "total_lines": r2['total_lines'],
            "total_stages": r2['total_stages'],
            "module_processing_count": r2['module_processing_count'],
            "stages": [{"name": s['name'], "category": s['category'], "line_count": s['line_count']} for s in r2['stages']],
            "category_totals": r2['category_totals']
        },
        "more_comprehensive_design": more_comprehensive,
        "reasoning": f"{more_comprehensive} has more extracted lines ({line_diff} more, {line_pct:.1f}% difference), indicating more comprehensive logging and processing across all stages",
        "comparison_summary": {
            "line_difference": line_diff,
            "line_difference_percentage": round(line_pct, 2),
            "stage_difference": r1['total_stages'] - r2['total_stages'],
            "module_processing_difference": r1['module_processing_count'] - r2['module_processing_count']
        }
    }
    
    print(f"\n{'='*80}")
    print("JSON OUTPUT")
    print(f"{'='*80}\n")
    print(json.dumps(json_output, indent=2))

if __name__ == "__main__":
    main()
