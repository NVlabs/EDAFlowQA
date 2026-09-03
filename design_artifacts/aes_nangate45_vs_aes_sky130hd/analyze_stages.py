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
Analyze stage complexity in EDA log files by counting lines per stage.
"""

import re
import sys
import json

def analyze_log_file(filename):
    """Analyze a log file and count lines per stage"""
    
    # Find all stage transitions with line numbers
    stage_pattern = r'Running .+?, stage ([0-9_a-z]+)'
    
    stages = []
    total_lines = 0
    
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        for line_num, line in enumerate(f, 1):
            total_lines += 1
            match = re.search(stage_pattern, line)
            if match:
                stage_name = match.group(1)
                stages.append({
                    'line_num': line_num,
                    'name': stage_name,
                    'full_line': line.strip()
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
        'stages': stage_info,
        'category_totals': category_totals
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: analyze_stages.py <logfile1> [logfile2] ...")
        sys.exit(1)
    
    results = {}
    
    for filename in sys.argv[1:]:
        print(f"\n{'='*80}")
        print(f"Analyzing: {filename}")
        print(f"{'='*80}\n")
        
        analysis = analyze_log_file(filename)
        results[filename] = analysis
        
        print(f"Total lines: {analysis['total_lines']}")
        print(f"Number of stages: {len(analysis['stages'])}\n")
        
        print("Individual stages:")
        print(f"{'Stage':<25s} {'Category':<12s} {'Lines':<8s} {'Start':<8s} {'End':<8s}")
        print("-" * 80)
        for stage in analysis['stages']:
            print(f"{stage['name']:<25s} {stage['category']:<12s} {stage['line_count']:<8d} "
                  f"{stage['start_line']:<8d} {stage['end_line']:<8d}")
        
        print("\nCategory totals:")
        print(f"{'Category':<15s} {'Lines':<10s} {'Percentage':<10s}")
        print("-" * 40)
        for cat in ['synthesis', 'floorplan', 'placement', 'cts', 'routing', 'finish']:
            count = analysis['category_totals'].get(cat, 0)
            pct = (count / analysis['total_lines'] * 100) if analysis['total_lines'] > 0 else 0
            print(f"{cat:<15s} {count:<10d} {pct:>6.2f}%")
    
    # Comparison
    if len(results) == 2:
        print(f"\n{'='*80}")
        print("COMPARISON - Line Counts by Category")
        print(f"{'='*80}\n")
        
        files = list(results.keys())
        file1, file2 = files[0], files[1]
        
        # Extract design names from filenames
        design1 = file1.split('/')[-1].replace('.txt', '')
        design2 = file2.split('/')[-1].replace('.txt', '')
        
        print(f"{'Category':<15s} {design1:<25s} {design2:<25s} {'Higher Complexity':<20s}")
        print("-" * 90)
        
        comparison_summary = {}
        
        for cat in ['synthesis', 'floorplan', 'placement', 'cts', 'routing', 'finish']:
            count1 = results[file1]['category_totals'].get(cat, 0)
            count2 = results[file2]['category_totals'].get(cat, 0)
            
            if count1 > count2:
                winner = design1
                diff = count1 - count2
                diff_pct = ((count1 - count2) / count2 * 100) if count2 > 0 else 0
            elif count2 > count1:
                winner = design2
                diff = count2 - count1
                diff_pct = ((count2 - count1) / count1 * 100) if count1 > 0 else 0
            else:
                winner = "Equal"
                diff = 0
                diff_pct = 0
            
            comparison_summary[cat] = {
                'design1_lines': count1,
                'design2_lines': count2,
                'higher_complexity': winner,
                'difference': diff,
                'difference_pct': diff_pct
            }
            
            print(f"{cat:<15s} {count1:<10d} ({count1:>5.1%})  {count2:<10d} ({count2:>5.1%})  "
                  f"{winner:<20s}")
        
        print(f"\n{'='*80}")
        print("DETAILED STAGE-BY-STAGE COMPARISON")
        print(f"{'='*80}\n")
        
        # Get all unique stage names from both files
        all_stage_names = set()
        for stage in results[file1]['stages']:
            all_stage_names.add(stage['name'])
        for stage in results[file2]['stages']:
            all_stage_names.add(stage['name'])
        
        # Create lookup for quick access
        stages1 = {s['name']: s for s in results[file1]['stages']}
        stages2 = {s['name']: s for s in results[file2]['stages']}
        
        print(f"{'Stage':<25s} {design1:<15s} {design2:<15s} {'Difference':<20s}")
        print("-" * 80)
        
        for stage_name in sorted(all_stage_names):
            count1 = stages1[stage_name]['line_count'] if stage_name in stages1 else 0
            count2 = stages2[stage_name]['line_count'] if stage_name in stages2 else 0
            diff = count1 - count2
            
            if count1 > count2:
                winner_mark = f"{design1} +{diff}"
            elif count2 > count1:
                winner_mark = f"{design2} +{abs(diff)}"
            else:
                winner_mark = "Equal"
            
            print(f"{stage_name:<25s} {count1:<15d} {count2:<15d} {winner_mark:<20s}")
        
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}\n")
        
        print(f"Design: {design1}")
        print(f"  Total lines: {results[file1]['total_lines']}")
        print(f"  Total stages: {len(results[file1]['stages'])}")
        
        print(f"\nDesign: {design2}")
        print(f"  Total lines: {results[file2]['total_lines']}")
        print(f"  Total stages: {len(results[file2]['stages'])}")
        
        print("\n** Analysis of Complexity by Stage **")
        placement_lines1 = comparison_summary['placement']['design1_lines']
        placement_lines2 = comparison_summary['placement']['design2_lines']
        routing_lines1 = comparison_summary['routing']['design1_lines']
        routing_lines2 = comparison_summary['routing']['design2_lines']
        
        print(f"\nPlacement Stage:")
        print(f"  {design1}: {placement_lines1} lines")
        print(f"  {design2}: {placement_lines2} lines")
        print(f"  Higher complexity: {comparison_summary['placement']['higher_complexity']}")
        
        print(f"\nRouting Stage:")
        print(f"  {design1}: {routing_lines1} lines")
        print(f"  {design2}: {routing_lines2} lines")
        print(f"  Higher complexity: {comparison_summary['routing']['higher_complexity']}")
        
        # Overall complexity determination
        total1 = results[file1]['total_lines']
        total2 = results[file2]['total_lines']
        
        print(f"\n** Overall Complexity Assessment **")
        if total1 > total2:
            print(f"{design1} shows higher overall complexity with {total1} total lines "
                  f"vs {total2} lines ({((total1-total2)/total2*100):.1f}% more)")
        else:
            print(f"{design2} shows higher overall complexity with {total2} total lines "
                  f"vs {total1} lines ({((total2-total1)/total1*100):.1f}% more)")
        
        # Save JSON output
        output = {
            'design1': {
                'name': design1,
                'total_lines': total1,
                'stages': len(results[file1]['stages']),
                'category_totals': results[file1]['category_totals']
            },
            'design2': {
                'name': design2,
                'total_lines': total2,
                'stages': len(results[file2]['stages']),
                'category_totals': results[file2]['category_totals']
            },
            'comparison': comparison_summary
        }
        
        print(f"\n{'='*80}")
        print("JSON OUTPUT")
        print(f"{'='*80}\n")
        print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
