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
Final comprehensive analysis to answer Q5 about design complexity based on line counts per stage.
"""

import json

# Stage markers with line numbers for aes_nangate45.txt
# Based on "Running *.tcl, stage X" markers found in the log
nangate45_stages = [
    (393, "1_synth", "synthesis"),
    (406, "2_1_floorplan", "floorplan"),
    (1925, "2_2_floorplan_macro", "floorplan"),
    (1936, "2_3_floorplan_tapcell", "floorplan"),
    (1948, "2_4_floorplan_pdn", "floorplan"),
    (1961, "3_1_place_gp_skip_io", "placement"),
    (1972, "3_2_place_iop", "placement"),
    (1978, "3_3_place_gp", "placement"),
    (2219, "3_4_place_resized", "placement"),
    (2267, "3_5_place_dp", "placement"),
    (2357, "4_1_cts", "cts"),
    (2467, "5_1_grt", "routing"),
    (2662, "5_2_route", "routing"),
    (3263, "5_3_fillcell", "routing"),
    (3277, "6_1_fill", "finish"),
    (3288, "6_report", "finish"),
]
nangate45_total = 3464

# Stage markers with line numbers for aes_sky130hd.txt
sky130hd_stages = [
    (386, "1_synth", "synthesis"),
    (399, "2_1_floorplan", "floorplan"),
    (443, "2_2_floorplan_macro", "floorplan"),
    (454, "2_3_floorplan_tapcell", "floorplan"),
    (465, "2_4_floorplan_pdn", "floorplan"),
    (478, "3_1_place_gp_skip_io", "placement"),
    (583, "3_2_place_iop", "placement"),
    (602, "3_3_place_gp", "placement"),
    (1175, "3_4_place_resized", "placement"),
    (1225, "3_5_place_dp", "placement"),
    (1320, "4_1_cts", "cts"),
    (1433, "5_1_grt", "routing"),
    (1710, "5_2_route", "routing"),
    (2815, "5_3_fillcell", "routing"),
    (2829, "6_1_fill", "finish"),
    (2840, "6_report", "finish"),
]
sky130hd_total = 3015

def calculate_stage_lines(stages, total_lines):
    """Calculate lines per stage"""
    stage_details = []
    category_totals = {}
    
    # Lines before first stage (initialization)
    init_lines = stages[0][0] - 1
    stage_details.append({
        'name': '0_init',
        'category': 'init',
        'start': 1,
        'end': stages[0][0] - 1,
        'lines': init_lines
    })
    category_totals['init'] = init_lines
    
    # Process each stage
    for i, (line_num, stage_name, category) in enumerate(stages):
        start = line_num
        if i < len(stages) - 1:
            end = stages[i + 1][0] - 1
        else:
            end = total_lines
        
        line_count = end - start + 1
        
        stage_details.append({
            'name': stage_name,
            'category': category,
            'start': start,
            'end': end,
            'lines': line_count
        })
        
        if category not in category_totals:
            category_totals[category] = 0
        category_totals[category] += line_count
    
    return stage_details, category_totals

# Calculate for both designs
ng45_details, ng45_totals = calculate_stage_lines(nangate45_stages, nangate45_total)
sky_details, sky_totals = calculate_stage_lines(sky130hd_stages, sky130hd_total)

# Print analysis
print("="*100)
print("COMPREHENSIVE STAGE-BY-STAGE LINE COUNT ANALYSIS")
print("Question: Which design shows higher complexity/detail in placement and routing?")
print("="*100)

print("\n" + "-"*100)
print("AES_NANGATE45 (aes_nangate45_2026-02-02T11_19_30_370555)")
print("-"*100)
print(f"Total lines: {nangate45_total}")
print(f"\n{'Stage':<30s} {'Category':<15s} {'Line Range':<20s} {'Lines':>10s} {'% of Total':>12s}")
print("-"*100)
for stage in ng45_details:
    pct = (stage['lines'] / nangate45_total * 100)
    print(f"{stage['name']:<30s} {stage['category']:<15s} {stage['start']:>6d}-{stage['end']:<12d} {stage['lines']:>10d} {pct:>11.2f}%")

print("\nCategory Totals (aes_nangate45):")
for cat in ['init', 'synthesis', 'floorplan', 'placement', 'cts', 'routing', 'finish']:
    count = ng45_totals.get(cat, 0)
    pct = (count / nangate45_total * 100)
    print(f"  {cat.upper():<15s}: {count:>6d} lines ({pct:>5.1f}%)")

print("\n" + "-"*100)
print("AES_SKY130HD (aes_sky130hd_2026-02-02T11_19_30_742696)")
print("-"*100)
print(f"Total lines: {sky130hd_total}")
print(f"\n{'Stage':<30s} {'Category':<15s} {'Line Range':<20s} {'Lines':>10s} {'% of Total':>12s}")
print("-"*100)
for stage in sky_details:
    pct = (stage['lines'] / sky130hd_total * 100)
    print(f"{stage['name']:<30s} {stage['category']:<15s} {stage['start']:>6d}-{stage['end']:<12d} {stage['lines']:>10d} {pct:>11.2f}%")

print("\nCategory Totals (aes_sky130hd):")
for cat in ['init', 'synthesis', 'floorplan', 'placement', 'cts', 'routing', 'finish']:
    count = sky_totals.get(cat, 0)
    pct = (count / sky130hd_total * 100)
    print(f"  {cat.upper():<15s}: {count:>6d} lines ({pct:>5.1f}%)")

# Comparison
print("\n" + "="*100)
print("COMPARATIVE ANALYSIS - Line Counts by Category")
print("="*100)
print(f"\n{'Category':<15s} {'NANGATE45':>15s} {'SKY130HD':>15s} {'Difference':>15s} {'Winner':>20s}")
print("-"*100)

comparison = {}
for cat in ['init', 'synthesis', 'floorplan', 'placement', 'cts', 'routing', 'finish']:
    ng45_count = ng45_totals.get(cat, 0)
    sky_count = sky_totals.get(cat, 0)
    diff = ng45_count - sky_count
    
    if ng45_count > sky_count:
        winner = "nangate45"
        diff_pct = ((ng45_count - sky_count) / sky_count * 100) if sky_count > 0 else 100
    elif sky_count > ng45_count:
        winner = "sky130hd"
        diff_pct = ((sky_count - ng45_count) / ng45_count * 100) if ng45_count > 0 else 100
    else:
        winner = "Equal"
        diff_pct = 0
    
    comparison[cat] = {
        'nangate45_lines': ng45_count,
        'sky130hd_lines': sky_count,
        'difference': abs(diff),
        'difference_pct': round(diff_pct, 2),
        'higher_complexity': winner
    }
    
    winner_str = f"{winner} (+{abs(diff)}, +{diff_pct:.1f}%)" if diff != 0 else winner
    
    print(f"{cat.upper():<15s} {ng45_count:>15d} {sky_count:>15d} {diff:>+15d} {winner_str:>20s}")

# DETAILED PLACEMENT ANALYSIS
print("\n" + "="*100)
print("DETAILED PLACEMENT STAGE ANALYSIS")
print("="*100)

placement_ng45 = [s for s in ng45_details if s['category'] == 'placement']
placement_sky = [s for s in sky_details if s['category'] == 'placement']

print(f"\n{'Substage':<30s} {'NANGATE45':>15s} {'SKY130HD':>15s} {'Difference':>15s}")
print("-"*100)

# Match substages
all_placement_names = sorted(set([s['name'] for s in placement_ng45] + [s['name'] for s in placement_sky]))
for name in all_placement_names:
    ng45_lines = next((s['lines'] for s in placement_ng45 if s['name'] == name), 0)
    sky_lines = next((s['lines'] for s in placement_sky if s['name'] == name), 0)
    diff = ng45_lines - sky_lines
    print(f"{name:<30s} {ng45_lines:>15d} {sky_lines:>15d} {diff:>+15d}")

print("-"*100)
print(f"{'TOTAL PLACEMENT':<30s} {ng45_totals['placement']:>15d} {sky_totals['placement']:>15d} {ng45_totals['placement'] - sky_totals['placement']:>+15d}")

# DETAILED ROUTING ANALYSIS
print("\n" + "="*100)
print("DETAILED ROUTING STAGE ANALYSIS")
print("="*100)

routing_ng45 = [s for s in ng45_details if s['category'] == 'routing']
routing_sky = [s for s in sky_details if s['category'] == 'routing']

print(f"\n{'Substage':<30s} {'NANGATE45':>15s} {'SKY130HD':>15s} {'Difference':>15s}")
print("-"*100)

# Match substages
all_routing_names = sorted(set([s['name'] for s in routing_ng45] + [s['name'] for s in routing_sky]))
for name in all_routing_names:
    ng45_lines = next((s['lines'] for s in routing_ng45 if s['name'] == name), 0)
    sky_lines = next((s['lines'] for s in routing_sky if s['name'] == name), 0)
    diff = ng45_lines - sky_lines
    print(f"{name:<30s} {ng45_lines:>15d} {sky_lines:>15d} {diff:>+15d}")

print("-"*100)
print(f"{'TOTAL ROUTING':<30s} {ng45_totals['routing']:>15d} {sky_totals['routing']:>15d} {ng45_totals['routing'] - sky_totals['routing']:>+15d}")

# FINAL ANSWER
print("\n" + "="*100)
print("ANSWER TO Q5: COMPLEXITY ANALYSIS SUMMARY")
print("="*100)

placement_winner = comparison['placement']['higher_complexity']
placement_diff = comparison['placement']['difference']
placement_pct = comparison['placement']['difference_pct']

routing_winner = comparison['routing']['higher_complexity']
routing_diff = comparison['routing']['difference']
routing_pct = comparison['routing']['difference_pct']

overall_winner = "nangate45" if nangate45_total > sky130hd_total else "sky130hd"

print(f"""
PLACEMENT STAGE COMPLEXITY:
  - aes_nangate45: {ng45_totals['placement']:,} lines ({ng45_totals['placement']/nangate45_total*100:.1f}% of total)
  - aes_sky130hd:  {sky_totals['placement']:,} lines ({sky_totals['placement']/sky130hd_total*100:.1f}% of total)
  - Winner: {placement_winner.upper()}
  - Difference: {placement_diff:,} lines ({placement_pct:.1f}% more)
  
ROUTING STAGE COMPLEXITY:
  - aes_nangate45: {ng45_totals['routing']:,} lines ({ng45_totals['routing']/nangate45_total*100:.1f}% of total)
  - aes_sky130hd:  {sky_totals['routing']:,} lines ({sky_totals['routing']/sky130hd_total*100:.1f}% of total)
  - Winner: {routing_winner.upper()}
  - Difference: {routing_diff:,} lines ({routing_pct:.1f}% more)

OVERALL COMPLEXITY:
  - aes_nangate45: {nangate45_total:,} total lines
  - aes_sky130hd:  {sky130hd_total:,} total lines
  - Winner: {overall_winner.upper()}
  - Difference: {abs(nangate45_total - sky130hd_total):,} lines

CONCLUSION:
In terms of the number of extracted lines per stage, "{placement_winner.upper()}" shows a higher 
degree of complexity or detail in the PLACEMENT stage, while "{routing_winner.upper()}" shows
higher complexity in the ROUTING stage. Overall, "{overall_winner.upper()}" has more total lines,
indicating higher overall log verbosity and complexity.
""")

# Create JSON output for the answer
answer = {
    "question": "In terms of the number of extracted lines per stage, which design shows a higher degree of complexity or detail in specific stages such as placement or routing?",
    "designs_compared": {
        "design_1": "aes_nangate45_2026-02-02T11_19_30_370555",
        "design_2": "aes_sky130hd_2026-02-02T11_19_30_742696"
    },
    "analysis_method": "Line count per stage extracted from log files based on stage transition markers",
    "total_lines": {
        "aes_nangate45": nangate45_total,
        "aes_sky130hd": sky130hd_total,
        "difference": nangate45_total - sky130hd_total,
        "higher_total": "aes_nangate45"
    },
    "placement_stage": {
        "aes_nangate45_lines": ng45_totals['placement'],
        "aes_sky130hd_lines": sky_totals['placement'],
        "difference_lines": placement_diff,
        "difference_percent": placement_pct,
        "higher_complexity": placement_winner,
        "substages": {
            "aes_nangate45": {s['name']: s['lines'] for s in placement_ng45},
            "aes_sky130hd": {s['name']: s['lines'] for s in placement_sky}
        }
    },
    "routing_stage": {
        "aes_nangate45_lines": ng45_totals['routing'],
        "aes_sky130hd_lines": sky_totals['routing'],
        "difference_lines": routing_diff,
        "difference_percent": routing_pct,
        "higher_complexity": routing_winner,
        "substages": {
            "aes_nangate45": {s['name']: s['lines'] for s in routing_ng45},
            "aes_sky130hd": {s['name']: s['lines'] for s in routing_sky}
        }
    },
    "all_categories": {
        "aes_nangate45": ng45_totals,
        "aes_sky130hd": sky_totals,
        "comparison": comparison
    },
    "summary": {
        "placement_higher_complexity": placement_winner,
        "routing_higher_complexity": routing_winner,
        "overall_higher_complexity": overall_winner,
        "key_finding": f"In placement stage, {placement_winner} shows {placement_pct:.1f}% more lines. In routing stage, {routing_winner} shows {routing_pct:.1f}% more lines."
    }
}

print("\n" + "="*100)
print("JSON OUTPUT - FINAL ANSWER")
print("="*100)
print(json.dumps(answer, indent=2))
