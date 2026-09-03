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
Analyze comprehensive flow based on stages and line counts.
"""

# Data from grep analysis of stage markers

aes_nangate45_stages = [
    {"line": 393, "stage": "1_synth"},
    {"line": 406, "stage": "2_1_floorplan"},
    {"line": 1925, "stage": "2_2_floorplan_macro"},
    {"line": 1936, "stage": "2_3_floorplan_tapcell"},
    {"line": 1948, "stage": "2_4_floorplan_pdn"},
    {"line": 1961, "stage": "3_1_place_gp_skip_io"},
    {"line": 1972, "stage": "3_2_place_iop"},
    {"line": 1978, "stage": "3_3_place_gp"},
    {"line": 2219, "stage": "3_4_place_resized"},
    {"line": 2267, "stage": "3_5_place_dp"},
    {"line": 2357, "stage": "4_1_cts"},
    {"line": 2467, "stage": "5_1_grt"},
    {"line": 2662, "stage": "5_2_route"},
    {"line": 3263, "stage": "5_3_fillcell"},
    {"line": 3277, "stage": "6_1_fill"},
    {"line": 3288, "stage": "6_report"}
]

aes_sky130hd_stages = [
    {"line": 386, "stage": "1_synth"},
    {"line": 399, "stage": "2_1_floorplan"},
    {"line": 443, "stage": "2_2_floorplan_macro"},
    {"line": 454, "stage": "2_3_floorplan_tapcell"},
    {"line": 465, "stage": "2_4_floorplan_pdn"},
    {"line": 478, "stage": "3_1_place_gp_skip_io"},
    {"line": 583, "stage": "3_2_place_iop"},
    {"line": 602, "stage": "3_3_place_gp"},
    {"line": 1175, "stage": "3_4_place_resized"},
    {"line": 1225, "stage": "3_5_place_dp"},
    {"line": 1320, "stage": "4_1_cts"},
    {"line": 1433, "stage": "5_1_grt"},
    {"line": 1710, "stage": "5_2_route"},
    {"line": 2815, "stage": "5_3_fillcell"},
    {"line": 2829, "stage": "6_1_fill"},
    {"line": 2840, "stage": "6_report"}
]

# Total lines from file ending
total_lines_nangate45 = 3465
total_lines_sky130hd = 3016

# Calculate line counts per stage
def calculate_stage_lines(stages, total_lines):
    results = []
    for i, stage in enumerate(stages):
        start = stage['line']
        if i < len(stages) - 1:
            end = stages[i + 1]['line'] - 1
        else:
            end = total_lines
        
        line_count = end - start + 1
        
        # Categorize
        if stage['stage'].startswith('1_'):
            category = 'synthesis'
        elif stage['stage'].startswith('2_'):
            category = 'floorplan'
        elif stage['stage'].startswith('3_'):
            category = 'placement'
        elif stage['stage'].startswith('4_'):
            category = 'cts'
        elif stage['stage'].startswith('5_'):
            category = 'routing'
        elif stage['stage'].startswith('6_'):
            category = 'finish'
        else:
            category = 'other'
        
        results.append({
            'stage': stage['stage'],
            'category': category,
            'start_line': start,
            'end_line': end,
            'line_count': line_count
        })
    
    return results

nangate45_analysis = calculate_stage_lines(aes_nangate45_stages, total_lines_nangate45)
sky130hd_analysis = calculate_stage_lines(aes_sky130hd_stages, total_lines_sky130hd)

# Aggregate by category
def aggregate_by_category(analysis):
    categories = {}
    for stage in analysis:
        cat = stage['category']
        if cat not in categories:
            categories[cat] = 0
        categories[cat] += stage['line_count']
    return categories

nangate45_categories = aggregate_by_category(nangate45_analysis)
sky130hd_categories = aggregate_by_category(sky130hd_analysis)

print("="*80)
print("COMPREHENSIVE FLOW ANALYSIS")
print("="*80)
print()

print("Design: aes_nangate45")
print(f"  Total lines: {total_lines_nangate45}")
print(f"  Total stages: {len(aes_nangate45_stages)}")
print()

print("Stage-by-stage breakdown:")
print(f"{'Stage':<30s} {'Category':<12s} {'Start':<8s} {'End':<8s} {'Lines':<8s}")
print("-" * 80)
for stage in nangate45_analysis:
    print(f"{stage['stage']:<30s} {stage['category']:<12s} {stage['start_line']:<8d} {stage['end_line']:<8d} {stage['line_count']:<8d}")

print()
print("Category totals:")
for cat in ['synthesis', 'floorplan', 'placement', 'cts', 'routing', 'finish']:
    count = nangate45_categories.get(cat, 0)
    pct = (count / total_lines_nangate45 * 100)
    print(f"  {cat:<15s}: {count:5d} lines ({pct:5.1f}%)")

print()
print("="*80)
print()

print("Design: aes_sky130hd")
print(f"  Total lines: {total_lines_sky130hd}")
print(f"  Total stages: {len(aes_sky130hd_stages)}")
print()

print("Stage-by-stage breakdown:")
print(f"{'Stage':<30s} {'Category':<12s} {'Start':<8s} {'End':<8s} {'Lines':<8s}")
print("-" * 80)
for stage in sky130hd_analysis:
    print(f"{stage['stage']:<30s} {stage['category']:<12s} {stage['start_line']:<8d} {stage['end_line']:<8d} {stage['line_count']:<8d}")

print()
print("Category totals:")
for cat in ['synthesis', 'floorplan', 'placement', 'cts', 'routing', 'finish']:
    count = sky130hd_categories.get(cat, 0)
    pct = (count / total_lines_sky130hd * 100)
    print(f"  {cat:<15s}: {count:5d} lines ({pct:5.1f}%)")

print()
print("="*80)
print("COMPARISON")
print("="*80)
print()

print(f"{'Metric':<40s} {'aes_nangate45':<20s} {'aes_sky130hd':<20s}")
print("-" * 85)
print(f"{'Total lines (log content)':<40s} {total_lines_nangate45:<20d} {total_lines_sky130hd:<20d}")
print(f"{'Total stages':<40s} {len(aes_nangate45_stages):<20d} {len(aes_sky130hd_stages):<20d}")

line_diff = total_lines_nangate45 - total_lines_sky130hd
line_pct = (line_diff / total_lines_sky130hd * 100)

print()
print("Stage-by-stage comparison:")
print(f"{'Stage':<30s} {'Nangate45':<15s} {'Sky130hd':<15s} {'Difference':<20s}")
print("-" * 85)

# Create lookup dictionaries
ng45_dict = {s['stage']: s for s in nangate45_analysis}
sky_dict = {s['stage']: s for s in sky130hd_analysis}

for stage in aes_nangate45_stages:
    stage_name = stage['stage']
    ng45_lines = ng45_dict[stage_name]['line_count']
    sky_lines = sky_dict[stage_name]['line_count']
    diff = ng45_lines - sky_lines
    if diff > 0:
        diff_str = f"NG45 +{diff}"
    elif diff < 0:
        diff_str = f"Sky130 +{abs(diff)}"
    else:
        diff_str = "Equal"
    print(f"{stage_name:<30s} {ng45_lines:<15d} {sky_lines:<15d} {diff_str:<20s}")

print()
print("Category-by-category comparison:")
print(f"{'Category':<15s} {'Nangate45':<12s} {'Sky130hd':<12s} {'More Lines':<20s}")
print("-" * 70)

for cat in ['synthesis', 'floorplan', 'placement', 'cts', 'routing', 'finish']:
    ng45_count = nangate45_categories.get(cat, 0)
    sky_count = sky130hd_categories.get(cat, 0)
    
    if ng45_count > sky_count:
        winner = "aes_nangate45"
        diff = ng45_count - sky_count
    elif sky_count > ng45_count:
        winner = "aes_sky130hd"
        diff = sky_count - ng45_count
    else:
        winner = "Equal"
        diff = 0
    
    print(f"{cat:<15s} {ng45_count:<12d} {sky_count:<12d} {winner:<15s} (+{diff})")

print()
print("="*80)
print("COMPREHENSIVE FLOW ASSESSMENT")
print("="*80)
print()

print("Based on the analysis of log file content:")
print()
print("1. Total extracted lines:")
print(f"   - aes_nangate45: {total_lines_nangate45} lines")
print(f"   - aes_sky130hd: {total_lines_sky130hd} lines")
print(f"   - Difference: {line_diff} lines ({line_pct:.1f}% more in nangate45)")
print()

print("2. Number of processing stages:")
print(f"   - aes_nangate45: {len(aes_nangate45_stages)} stages")
print(f"   - aes_sky130hd: {len(aes_sky130hd_stages)} stages")
print(f"   - Same number of stages (both have complete flow)")
print()

print("3. Stage-by-stage line distribution:")
ng45_placement = nangate45_categories['placement']
sky_placement = sky130hd_categories['placement']
ng45_routing = nangate45_categories['routing']
sky_routing = sky130hd_categories['routing']

print(f"   - Placement stage:")
print(f"     aes_nangate45: {ng45_placement} lines")
print(f"     aes_sky130hd: {sky_placement} lines")
print(f"     More comprehensive: {'aes_nangate45' if ng45_placement > sky_placement else 'aes_sky130hd'}")
print()
print(f"   - Routing stage:")
print(f"     aes_nangate45: {ng45_routing} lines")
print(f"     aes_sky130hd: {sky_routing} lines")
print(f"     More comprehensive: {'aes_nangate45' if ng45_routing > sky_routing else 'aes_sky130hd'}")
print()

print("4. Key observations:")
print(f"   - Both designs complete all 6 major flow stages (synthesis -> finish)")
print(f"   - Both have 16 sub-stages total")
print(f"   - aes_nangate45 has {line_diff} more lines of logged content")
print(f"   - This indicates more detailed processing/logging in nangate45")
print()

print("="*80)
print("FINAL ANSWER")
print("="*80)
print()
print("** aes_nangate45 ** appears to have a more comprehensive flow based on:")
print()
print(f"1. More extracted lines: {total_lines_nangate45} vs {total_lines_sky130hd} ({line_pct:.1f}% more)")
print(f"2. More detailed logging in placement: {ng45_placement} vs {sky_placement} lines")
print(f"3. More detailed logging in routing: {ng45_routing} vs {sky_routing} lines")
print(f"4. Greater overall log content volume indicates more processing detail")
print()
print("While both designs have the same number of stages (16 sub-stages across 6 major")
print("stages), the significantly higher line count in aes_nangate45 indicates more")
print("comprehensive processing, more iterations, and more detailed reporting throughout")
print("the flow, particularly in the critical placement and routing stages.")

# Generate JSON output
import json

json_output = {
    "question": "Which design appears to have a more comprehensive flow based on the number of sub-designs processed and the extracted lines in each stage?",
    "answer": "aes_nangate45",
    "reasoning": f"aes_nangate45 has more comprehensive flow with {total_lines_nangate45} total lines vs {total_lines_sky130hd} lines in aes_sky130hd ({line_pct:.1f}% more content). While both designs complete the same 16 sub-stages across 6 major flow stages, nangate45 shows significantly more detailed processing and logging throughout, particularly in placement ({ng45_placement} vs {sky_placement} lines) and routing ({ng45_routing} vs {sky_routing} lines) stages.",
    "detailed_comparison": {
        "aes_nangate45": {
            "total_lines": total_lines_nangate45,
            "total_stages": len(aes_nangate45_stages),
            "category_totals": nangate45_categories,
            "stages": [{"stage": s['stage'], "category": s['category'], "line_count": s['line_count']} for s in nangate45_analysis]
        },
        "aes_sky130hd": {
            "total_lines": total_lines_sky130hd,
            "total_stages": len(aes_sky130hd_stages),
            "category_totals": sky130hd_categories,
            "stages": [{"stage": s['stage'], "category": s['category'], "line_count": s['line_count']} for s in sky130hd_analysis]
        },
        "line_difference": line_diff,
        "line_difference_percentage": round(line_pct, 2)
    }
}

print()
print("="*80)
print("JSON OUTPUT")
print("="*80)
print()
print(json.dumps(json_output, indent=2))
