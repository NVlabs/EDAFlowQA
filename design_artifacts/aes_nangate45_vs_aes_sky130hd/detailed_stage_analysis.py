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
Detailed stage-by-stage line count analysis for EDA log files.
"""

import re
from collections import defaultdict

def analyze_log_file(filename):
    """
    Analyze a log file and count lines per stage.
    Returns stage line counts and detailed breakdown.
    """
    
    # Track lines for each substage
    substage_lines = defaultdict(int)
    current_stage = "0_init"
    
    # Patterns that indicate stage transitions
    stage_patterns = [
        # Synthesis
        (r'Running.*1_1_yosys', '1_1_synthesis'),
        (r'stage 1_1_yosys', '1_1_synthesis'),
        
        # Floorplan stages
        (r'Running.*2_1_floorplan', '2_1_floorplan'),
        (r'stage 2_1_floorplan', '2_1_floorplan'),
        (r'Running.*2_2_floorplan_macro', '2_2_floorplan_macro'),
        (r'stage 2_2_floorplan_macro', '2_2_floorplan_macro'),
        (r'Running.*2_3_floorplan_tapcell', '2_3_floorplan_tapcell'),
        (r'stage 2_3_floorplan_tapcell', '2_3_floorplan_tapcell'),
        (r'Running.*2_4_floorplan_pdn', '2_4_floorplan_pdn'),
        (r'stage 2_4_floorplan_pdn', '2_4_floorplan_pdn'),
        
        # Placement stages
        (r'Running.*3_1_place_gp_skip_io', '3_1_place_gp_skip_io'),
        (r'stage 3_1_place_gp_skip_io', '3_1_place_gp_skip_io'),
        (r'Running.*3_2_place_iop', '3_2_place_iop'),
        (r'stage 3_2_place_iop', '3_2_place_iop'),
        (r'Running.*3_3_place_gp', '3_3_place_gp'),
        (r'stage 3_3_place_gp', '3_3_place_gp'),
        (r'Running.*3_4_place_resized', '3_4_place_resized'),
        (r'stage 3_4_place_resized', '3_4_place_resized'),
        (r'Running.*3_5_place_dp', '3_5_place_dp'),
        (r'stage 3_5_place_dp', '3_5_place_dp'),
        
        # CTS stages
        (r'Running.*4_1_cts', '4_1_cts'),
        (r'stage 4_1_cts', '4_1_cts'),
        (r'Running.*4_2_cts_fillcell', '4_2_cts_fillcell'),
        (r'stage 4_2_cts_fillcell', '4_2_cts_fillcell'),
        
        # Routing stages
        (r'Running.*5_1_grt', '5_1_grt'),
        (r'stage 5_1_grt', '5_1_grt'),
        (r'Running.*5_2_route', '5_2_route'),
        (r'stage 5_2_route', '5_2_route'),
        (r'Running.*5_3_fillcell', '5_3_fillcell'),
        (r'stage 5_3_fillcell', '5_3_fillcell'),
        
        # Finish stages
        (r'Running.*6_1_merge', '6_1_merge'),
        (r'stage 6_1_merge', '6_1_merge'),
        (r'Running.*6_report', '6_report'),
        (r'stage 6_report', '6_report'),
    ]
    
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # Check for stage transitions
            for pattern, stage_name in stage_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    current_stage = stage_name
                    break
            
            # Count this line for current stage
            substage_lines[current_stage] += 1
    
    return dict(substage_lines)

def categorize_stages(substage_lines):
    """Group substages into major categories."""
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
    
    for stage, count in substage_lines.items():
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
    # Analyze both designs
    ng45_file = 'aes_nangate45.txt'
    sky_file = 'aes_sky130hd.txt'
    
    print("="*90)
    print("ANALYZING AES_NANGATE45")
    print("="*90)
    ng45_substages = analyze_log_file(ng45_file)
    ng45_categories, ng45_details = categorize_stages(ng45_substages)
    
    total_ng45 = sum(ng45_substages.values())
    print(f"\nTotal lines: {total_ng45}")
    
    print("\n--- Substage Details ---")
    for substage in sorted(ng45_substages.keys()):
        count = ng45_substages[substage]
        pct = (count / total_ng45 * 100) if total_ng45 > 0 else 0
        print(f"  {substage:30s}: {count:6d} lines ({pct:5.2f}%)")
    
    print("\n--- Category Summary ---")
    for category in ['init', 'synthesis', 'floorplan', 'placement', 'cts', 'routing', 'finish']:
        count = ng45_categories[category]
        pct = (count / total_ng45 * 100) if total_ng45 > 0 else 0
        print(f"  {category.upper():15s}: {count:6d} lines ({pct:5.2f}%)")
    
    print("\n" + "="*90)
    print("ANALYZING AES_SKY130HD")
    print("="*90)
    sky_substages = analyze_log_file(sky_file)
    sky_categories, sky_details = categorize_stages(sky_substages)
    
    total_sky = sum(sky_substages.values())
    print(f"\nTotal lines: {total_sky}")
    
    print("\n--- Substage Details ---")
    for substage in sorted(sky_substages.keys()):
        count = sky_substages[substage]
        pct = (count / total_sky * 100) if total_sky > 0 else 0
        print(f"  {substage:30s}: {count:6d} lines ({pct:5.2f}%)")
    
    print("\n--- Category Summary ---")
    for category in ['init', 'synthesis', 'floorplan', 'placement', 'cts', 'routing', 'finish']:
        count = sky_categories[category]
        pct = (count / total_sky * 100) if total_sky > 0 else 0
        print(f"  {category.upper():15s}: {count:6d} lines ({pct:5.2f}%)")
    
    # Detailed comparison
    print("\n" + "="*90)
    print("COMPARISON: NANGATE45 vs SKY130HD")
    print("="*90)
    
    print(f"\n{'Category':<15s} {'NANGATE45':>12s} {'SKY130HD':>12s} {'Difference':>12s} {'Ratio':>10s}")
    print("-" * 90)
    
    for category in ['init', 'synthesis', 'floorplan', 'placement', 'cts', 'routing', 'finish']:
        ng45_count = ng45_categories[category]
        sky_count = sky_categories[category]
        diff = ng45_count - sky_count
        ratio = ng45_count / sky_count if sky_count > 0 else 0
        
        print(f"{category.upper():<15s} {ng45_count:>12d} {sky_count:>12d} {diff:>+12d} {ratio:>10.2f}x")
    
    print("-" * 90)
    print(f"{'TOTAL':<15s} {total_ng45:>12d} {total_sky:>12d} {total_ng45-total_sky:>+12d} {total_ng45/total_sky:>10.2f}x")
    
    # Key insights
    print("\n" + "="*90)
    print("KEY INSIGHTS")
    print("="*90)
    
    # Find which design has more complexity in placement
    placement_winner = "NANGATE45" if ng45_categories['placement'] > sky_categories['placement'] else "SKY130HD"
    placement_ratio = max(ng45_categories['placement'], sky_categories['placement']) / min(ng45_categories['placement'], sky_categories['placement'])
    
    print(f"\nPLACEMENT COMPLEXITY:")
    print(f"  {placement_winner} shows {placement_ratio:.2f}x more lines in placement stages")
    print(f"  NANGATE45: {ng45_categories['placement']} lines")
    print(f"  SKY130HD:  {sky_categories['placement']} lines")
    
    # Find which design has more complexity in routing
    routing_winner = "NANGATE45" if ng45_categories['routing'] > sky_categories['routing'] else "SKY130HD"
    routing_ratio = max(ng45_categories['routing'], sky_categories['routing']) / min(ng45_categories['routing'], sky_categories['routing'])
    
    print(f"\nROUTING COMPLEXITY:")
    print(f"  {routing_winner} shows {routing_ratio:.2f}x more lines in routing stages")
    print(f"  NANGATE45: {ng45_categories['routing']} lines")
    print(f"  SKY130HD:  {sky_categories['routing']} lines")
    
    # Overall complexity
    overall_winner = "NANGATE45" if total_ng45 > total_sky else "SKY130HD"
    overall_ratio = max(total_ng45, total_sky) / min(total_ng45, total_sky)
    
    print(f"\nOVERALL COMPLEXITY:")
    print(f"  {overall_winner} shows {overall_ratio:.2f}x more total lines")
    print(f"  This suggests {overall_winner} has higher overall log detail/complexity")

if __name__ == '__main__':
    main()
