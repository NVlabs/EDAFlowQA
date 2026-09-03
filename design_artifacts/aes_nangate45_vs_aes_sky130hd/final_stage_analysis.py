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
Accurate line count analysis based on actual stage transition markers.
"""

import re

def count_lines_by_stage(filename):
    """Count lines between stage transitions."""
    
    # Read all lines first
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    
    # Find stage transitions with exact line numbers
    stage_transitions = []
    stage_pattern = r'Running\s+\S+\.tcl,\s+stage\s+(\S+)'
    
    for line_num, line in enumerate(lines, 1):
        match = re.search(stage_pattern, line)
        if match:
            stage_name = match.group(1)
            stage_transitions.append((line_num, stage_name))
    
    # Calculate lines per stage
    stage_lines = {}
    
    # Lines before first stage
    if stage_transitions:
        stage_lines['0_init'] = stage_transitions[0][0] - 1
    else:
        stage_lines['0_init'] = total_lines
        return stage_lines, total_lines
    
    # Lines for each stage (from transition to next transition)
    for i in range(len(stage_transitions)):
        stage_line_num, stage_name = stage_transitions[i]
        
        if i < len(stage_transitions) - 1:
            next_stage_line_num = stage_transitions[i + 1][0]
            lines_count = next_stage_line_num - stage_line_num
        else:
            # Last stage goes to end of file
            lines_count = total_lines - stage_line_num + 1
        
        stage_lines[stage_name] = lines_count
    
    return stage_lines, total_lines

def categorize_stages(stage_lines):
    """Group stages into categories."""
    categories = {
        'synthesis': 0,
        'floorplan': 0,
        'placement': 0,
        'cts': 0,
        'routing': 0,
        'finish': 0,
        'init': 0
    }
    
    substage_details = {
        'synthesis': [],
        'floorplan': [],
        'placement': [],
        'cts': [],
        'routing': [],
        'finish': [],
        'init': []
    }
    
    for stage, count in sorted(stage_lines.items()):
        if stage.startswith('0_'):
            categories['init'] += count
            substage_details['init'].append((stage, count))
        elif stage.startswith('1_'):
            categories['synthesis'] += count
            substage_details['synthesis'].append((stage, count))
        elif stage.startswith('2_'):
            categories['floorplan'] += count
            substage_details['floorplan'].append((stage, count))
        elif stage.startswith('3_'):
            categories['placement'] += count
            substage_details['placement'].append((stage, count))
        elif stage.startswith('4_'):
            categories['cts'] += count
            substage_details['cts'].append((stage, count))
        elif stage.startswith('5_'):
            categories['routing'] += count
            substage_details['routing'].append((stage, count))
        elif stage.startswith('6_'):
            categories['finish'] += count
            substage_details['finish'].append((stage, count))
    
    return categories, substage_details

def main():
    print("="*100)
    print("DETAILED STAGE LINE COUNT ANALYSIS")
    print("="*100)
    
    # Analyze NANGATE45
    ng45_file = 'aes_nangate45.txt'
    print(f"\nAnalyzing {ng45_file}...")
    ng45_stages, ng45_total = count_lines_by_stage(ng45_file)
    ng45_categories, ng45_details = categorize_stages(ng45_stages)
    
    print(f"\n{'='*100}")
    print("AES_NANGATE45 - LINE COUNTS PER STAGE")
    print(f"{'='*100}")
    print(f"\nTotal lines: {ng45_total:,}")
    print(f"\n{'Stage':<30s} {'Lines':>10s} {'%':>8s}")
    print("-" * 100)
    
    for stage in sorted(ng45_stages.keys()):
        count = ng45_stages[stage]
        pct = (count / ng45_total * 100) if ng45_total > 0 else 0
        print(f"{stage:<30s} {count:>10,} {pct:>7.2f}%")
    
    print(f"\n{'='*100}")
    print("AES_NANGATE45 - CATEGORY SUMMARY")
    print(f"{'='*100}")
    print(f"\n{'Category':<20s} {'Lines':>10s} {'%':>8s}")
    print("-" * 100)
    
    for category in ['init', 'synthesis', 'floorplan', 'placement', 'cts', 'routing', 'finish']:
        count = ng45_categories[category]
        pct = (count / ng45_total * 100) if ng45_total > 0 else 0
        print(f"{category.upper():<20s} {count:>10,} {pct:>7.2f}%")
    
    # Analyze SKY130HD
    sky_file = 'aes_sky130hd.txt'
    print(f"\n\nAnalyzing {sky_file}...")
    sky_stages, sky_total = count_lines_by_stage(sky_file)
    sky_categories, sky_details = categorize_stages(sky_stages)
    
    print(f"\n{'='*100}")
    print("AES_SKY130HD - LINE COUNTS PER STAGE")
    print(f"{'='*100}")
    print(f"\nTotal lines: {sky_total:,}")
    print(f"\n{'Stage':<30s} {'Lines':>10s} {'%':>8s}")
    print("-" * 100)
    
    for stage in sorted(sky_stages.keys()):
        count = sky_stages[stage]
        pct = (count / sky_total * 100) if sky_total > 0 else 0
        print(f"{stage:<30s} {count:>10,} {pct:>7.2f}%")
    
    print(f"\n{'='*100}")
    print("AES_SKY130HD - CATEGORY SUMMARY")
    print(f"{'='*100}")
    print(f"\n{'Category':<20s} {'Lines':>10s} {'%':>8s}")
    print("-" * 100)
    
    for category in ['init', 'synthesis', 'floorplan', 'placement', 'cts', 'routing', 'finish']:
        count = sky_categories[category]
        pct = (count / sky_total * 100) if sky_total > 0 else 0
        print(f"{category.upper():<20s} {count:>10,} {pct:>7.2f}%")
    
    # COMPARISON
    print(f"\n\n{'='*100}")
    print("COMPARISON: NANGATE45 vs SKY130HD")
    print(f"{'='*100}")
    print(f"\n{'Category':<20s} {'NANGATE45':>12s} {'SKY130HD':>12s} {'Diff':>12s} {'Ratio':>10s} {'Winner':>15s}")
    print("-" * 100)
    
    for category in ['init', 'synthesis', 'floorplan', 'placement', 'cts', 'routing', 'finish']:
        ng45_count = ng45_categories[category]
        sky_count = sky_categories[category]
        diff = ng45_count - sky_count
        
        if sky_count > 0:
            ratio = ng45_count / sky_count
        else:
            ratio = 0
        
        if ng45_count > sky_count:
            winner = "NANGATE45"
        elif sky_count > ng45_count:
            winner = "SKY130HD"
        else:
            winner = "Equal"
        
        print(f"{category.upper():<20s} {ng45_count:>12,} {sky_count:>12,} {diff:>+12,} {ratio:>10.2f}x {winner:>15s}")
    
    print("-" * 100)
    diff_total = ng45_total - sky_total
    ratio_total = ng45_total / sky_total if sky_total > 0 else 0
    winner_total = "NANGATE45" if ng45_total > sky_total else "SKY130HD"
    print(f"{'TOTAL':<20s} {ng45_total:>12,} {sky_total:>12,} {diff_total:>+12,} {ratio_total:>10.2f}x {winner_total:>15s}")
    
    # KEY FINDINGS
    print(f"\n\n{'='*100}")
    print("KEY FINDINGS - COMPLEXITY ANALYSIS")
    print(f"{'='*100}")
    
    print(f"\n1. PLACEMENT STAGE COMPLEXITY:")
    ng45_place = ng45_categories['placement']
    sky_place = sky_categories['placement']
    place_diff = abs(ng45_place - sky_place)
    place_ratio = max(ng45_place, sky_place) / min(ng45_place, sky_place) if min(ng45_place, sky_place) > 0 else 0
    place_winner = "NANGATE45" if ng45_place > sky_place else "SKY130HD"
    
    print(f"   - NANGATE45: {ng45_place:,} lines ({ng45_place/ng45_total*100:.2f}% of total)")
    print(f"   - SKY130HD:  {sky_place:,} lines ({sky_place/sky_total*100:.2f}% of total)")
    print(f"   - {place_winner} has {place_diff:,} more lines ({place_ratio:.2f}x)")
    print(f"   - CONCLUSION: {place_winner} shows higher placement complexity/detail")
    
    print(f"\n2. ROUTING STAGE COMPLEXITY:")
    ng45_route = ng45_categories['routing']
    sky_route = sky_categories['routing']
    route_diff = abs(ng45_route - sky_route)
    route_ratio = max(ng45_route, sky_route) / min(ng45_route, sky_route) if min(ng45_route, sky_route) > 0 else 0
    route_winner = "NANGATE45" if ng45_route > sky_route else "SKY130HD"
    
    print(f"   - NANGATE45: {ng45_route:,} lines ({ng45_route/ng45_total*100:.2f}% of total)")
    print(f"   - SKY130HD:  {sky_route:,} lines ({sky_route/sky_total*100:.2f}% of total)")
    print(f"   - {route_winner} has {route_diff:,} more lines ({route_ratio:.2f}x)")
    print(f"   - CONCLUSION: {route_winner} shows higher routing complexity/detail")
    
    print(f"\n3. OVERALL DESIGN COMPLEXITY:")
    overall_winner = "NANGATE45" if ng45_total > sky_total else "SKY130HD"
    overall_ratio = max(ng45_total, sky_total) / min(ng45_total, sky_total)
    overall_diff = abs(ng45_total - sky_total)
    
    print(f"   - NANGATE45: {ng45_total:,} total lines")
    print(f"   - SKY130HD:  {sky_total:,} total lines")
    print(f"   - {overall_winner} has {overall_diff:,} more lines ({overall_ratio:.2f}x)")
    print(f"   - CONCLUSION: {overall_winner} shows higher overall log verbosity/complexity")
    
    # Detailed substage comparison for placement
    print(f"\n4. DETAILED PLACEMENT SUBSTAGE BREAKDOWN:")
    print(f"   {'Substage':<30s} {'NANGATE45':>12s} {'SKY130HD':>12s} {'Diff':>12s}")
    print("   " + "-" * 70)
    
    # Collect all placement substages
    all_place_stages = set()
    for stage in ng45_stages.keys():
        if stage.startswith('3_'):
            all_place_stages.add(stage)
    for stage in sky_stages.keys():
        if stage.startswith('3_'):
            all_place_stages.add(stage)
    
    for stage in sorted(all_place_stages):
        ng45_count = ng45_stages.get(stage, 0)
        sky_count = sky_stages.get(stage, 0)
        diff = ng45_count - sky_count
        print(f"   {stage:<30s} {ng45_count:>12,} {sky_count:>12,} {diff:>+12,}")
    
    # Detailed substage comparison for routing
    print(f"\n5. DETAILED ROUTING SUBSTAGE BREAKDOWN:")
    print(f"   {'Substage':<30s} {'NANGATE45':>12s} {'SKY130HD':>12s} {'Diff':>12s}")
    print("   " + "-" * 70)
    
    # Collect all routing substages
    all_route_stages = set()
    for stage in ng45_stages.keys():
        if stage.startswith('5_'):
            all_route_stages.add(stage)
    for stage in sky_stages.keys():
        if stage.startswith('5_'):
            all_route_stages.add(stage)
    
    for stage in sorted(all_route_stages):
        ng45_count = ng45_stages.get(stage, 0)
        sky_count = sky_stages.get(stage, 0)
        diff = ng45_count - sky_count
        print(f"   {stage:<30s} {ng45_count:>12,} {sky_count:>12,} {diff:>+12,}")
    
    print(f"\n{'='*100}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*100}")

if __name__ == '__main__':
    main()
