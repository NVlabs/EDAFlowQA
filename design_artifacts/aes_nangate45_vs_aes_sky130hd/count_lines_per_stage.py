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
Count the number of lines per stage in the design flow log files.
"""

import re
from collections import defaultdict

def count_lines_per_stage(log_file):
    """
    Count lines for each stage in the log file.
    Returns a dictionary with stage names and their line counts.
    """
    stage_lines = defaultdict(int)
    current_stage = "0_init"
    
    # Stage patterns to identify
    stage_patterns = [
        (r'1_1_yosys', '1_synthesis'),
        (r'2_1_floorplan', '2_1_floorplan'),
        (r'2_2_floorplan_macro', '2_2_floorplan_macro'),
        (r'2_3_floorplan_tapcell', '2_3_floorplan_tapcell'),
        (r'2_4_floorplan_pdn', '2_4_floorplan_pdn'),
        (r'3_1_place_gp_skip_io', '3_1_place_gp_skip_io'),
        (r'3_2_place_iop', '3_2_place_iop'),
        (r'3_3_place_gp', '3_3_place_gp'),
        (r'3_4_place_resized', '3_4_place_resized'),
        (r'3_5_place_dp', '3_5_place_dp'),
        (r'4_1_cts', '4_1_cts'),
        (r'4_2_cts_fillcell', '4_2_cts_fillcell'),
        (r'5_1_grt', '5_1_grt'),
        (r'5_2_route', '5_2_route'),
        (r'5_3_fillcell', '5_3_fillcell'),
        (r'6_1_merge', '6_1_merge'),
        (r'6_report', '6_report'),
    ]
    
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # Check if this line marks a new stage
            for pattern, stage_name in stage_patterns:
                if re.search(pattern, line):
                    current_stage = stage_name
                    break
            
            # Count this line for the current stage
            stage_lines[current_stage] += 1
    
    return dict(stage_lines)

def categorize_stages(stage_lines):
    """
    Categorize stages into major flow phases.
    """
    categories = {
        'synthesis': [],
        'floorplan': [],
        'placement': [],
        'cts': [],
        'routing': [],
        'finish': [],
        'other': []
    }
    
    for stage, count in stage_lines.items():
        if '1_' in stage or 'synthesis' in stage:
            categories['synthesis'].append((stage, count))
        elif '2_' in stage or 'floorplan' in stage:
            categories['floorplan'].append((stage, count))
        elif '3_' in stage or 'place' in stage:
            categories['placement'].append((stage, count))
        elif '4_' in stage or 'cts' in stage:
            categories['cts'].append((stage, count))
        elif '5_' in stage or 'route' in stage or 'grt' in stage:
            categories['routing'].append((stage, count))
        elif '6_' in stage or 'finish' in stage or 'report' in stage:
            categories['finish'].append((stage, count))
        else:
            categories['other'].append((stage, count))
    
    return categories

def main():
    # Analyze both designs
    nangate45_file = 'aes_nangate45.txt'
    sky130hd_file = 'aes_sky130hd.txt'
    
    print("Analyzing nangate45 design...")
    nangate45_lines = count_lines_per_stage(nangate45_file)
    nangate45_categories = categorize_stages(nangate45_lines)
    
    print("Analyzing sky130hd design...")
    sky130hd_lines = count_lines_per_stage(sky130hd_file)
    sky130hd_categories = categorize_stages(sky130hd_lines)
    
    # Print results
    print("\n" + "="*80)
    print("NANGATE45 DESIGN - Lines per Stage")
    print("="*80)
    for stage, count in sorted(nangate45_lines.items()):
        print(f"{stage:40s}: {count:8d} lines")
    
    print("\n" + "="*80)
    print("SKY130HD DESIGN - Lines per Stage")
    print("="*80)
    for stage, count in sorted(sky130hd_lines.items()):
        print(f"{stage:40s}: {count:8d} lines")
    
    # Print category summaries
    print("\n" + "="*80)
    print("SUMMARY BY CATEGORY - NANGATE45")
    print("="*80)
    for category, stages in nangate45_categories.items():
        total = sum(count for _, count in stages)
        print(f"\n{category.upper()}:")
        for stage, count in sorted(stages):
            print(f"  {stage:38s}: {count:8d} lines")
        print(f"  {'TOTAL':38s}: {total:8d} lines")
    
    print("\n" + "="*80)
    print("SUMMARY BY CATEGORY - SKY130HD")
    print("="*80)
    for category, stages in sky130hd_categories.items():
        total = sum(count for _, count in stages)
        print(f"\n{category.upper()}:")
        for stage, count in sorted(stages):
            print(f"  {stage:38s}: {count:8d} lines")
        print(f"  {'TOTAL':38s}: {total:8d} lines")
    
    # Compare placement and routing stages
    print("\n" + "="*80)
    print("COMPARISON: PLACEMENT AND ROUTING STAGES")
    print("="*80)
    
    ng45_placement = sum(count for _, count in nangate45_categories['placement'])
    ng45_routing = sum(count for _, count in nangate45_categories['routing'])
    sky_placement = sum(count for _, count in sky130hd_categories['placement'])
    sky_routing = sum(count for _, count in sky130hd_categories['routing'])
    
    print(f"\nPlacement Stage:")
    print(f"  NANGATE45: {ng45_placement:8d} lines")
    print(f"  SKY130HD:  {sky_placement:8d} lines")
    print(f"  Difference: {ng45_placement - sky_placement:8d} lines")
    print(f"  Ratio (NG45/SKY): {ng45_placement/sky_placement:.2f}x")
    
    print(f"\nRouting Stage:")
    print(f"  NANGATE45: {ng45_routing:8d} lines")
    print(f"  SKY130HD:  {sky_routing:8d} lines")
    print(f"  Difference: {ng45_routing - sky_routing:8d} lines")
    print(f"  Ratio (NG45/SKY): {ng45_routing/sky_routing:.2f}x")
    
    ng45_total = sum(nangate45_lines.values())
    sky_total = sum(sky130hd_lines.values())
    print(f"\nTotal Lines:")
    print(f"  NANGATE45: {ng45_total:8d} lines")
    print(f"  SKY130HD:  {sky_total:8d} lines")
    print(f"  Difference: {ng45_total - sky_total:8d} lines")
    print(f"  Ratio (NG45/SKY): {ng45_total/sky_total:.2f}x")

if __name__ == '__main__':
    main()
