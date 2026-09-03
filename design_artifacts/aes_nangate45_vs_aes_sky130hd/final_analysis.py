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
Calculate line counts per stage based on extracted stage markers.
"""

import json

# Stage markers with line numbers for aes_nangate45.txt
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

# Print detailed analysis
print("="*90)
print("STAGE-BY-STAGE LINE COUNT ANALYSIS")
print("="*90)

print("\n" + "-"*90)
print("aes_nangate45 (Total: {} lines)".format(nangate45_total))
print("-"*90)
print("{:<30s} {:>12s} {:>15s} {:>12s}".format("Stage", "Category", "Line Range", "Lines"))
print("-"*90)
for stage in ng45_details:
    print("{:<30s} {:>12s} {:>6d}-{:<6d} {:>12d}".format(
        stage['name'], stage['category'], stage['start'], stage['end'], stage['lines']))

print("\nCategory Totals (aes_nangate45):")
for cat in ['synthesis', 'floorplan', 'placement', 'cts', 'routing', 'finish']:
    count = ng45_totals.get(cat, 0)
    pct = (count / nangate45_total * 100)
    print("  {:<15s}: {:>6d} lines ({:>5.1f}%)".format(cat, count, pct))

print("\n" + "-"*90)
print("aes_sky130hd (Total: {} lines)".format(sky130hd_total))
print("-"*90)
print("{:<30s} {:>12s} {:>15s} {:>12s}".format("Stage", "Category", "Line Range", "Lines"))
print("-"*90)
for stage in sky_details:
    print("{:<30s} {:>12s} {:>6d}-{:<6d} {:>12d}".format(
        stage['name'], stage['category'], stage['start'], stage['end'], stage['lines']))

print("\nCategory Totals (aes_sky130hd):")
for cat in ['synthesis', 'floorplan', 'placement', 'cts', 'routing', 'finish']:
    count = sky_totals.get(cat, 0)
    pct = (count / sky130hd_total * 100)
    print("  {:<15s}: {:>6d} lines ({:>5.1f}%)".format(cat, count, pct))

# Comparison
print("\n" + "="*90)
print("COMPARISON - Line Counts by Category")
print("="*90)
print("{:<15s} {:>15s} {:>15s} {:>20s} {:>20s}".format(
    "Category", "nangate45", "sky130hd", "Difference", "Higher Complexity"))
print("-"*90)

comparison = {}
for cat in ['synthesis', 'floorplan', 'placement', 'cts', 'routing', 'finish']:
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
    
    winner_str = "{} (+{}, +{:.1f}%)".format(winner, abs(diff), diff_pct) if diff != 0 else winner
    
    print("{:<15s} {:>15d} {:>15d} {:>20d} {:>20s}".format(
        cat, ng45_count, sky_count, diff, winner_str))

# Overall comparison
print("\n" + "="*90)
print("OVERALL COMPLEXITY ANALYSIS")
print("="*90)

if nangate45_total > sky130hd_total:
    overall_winner = "aes_nangate45"
    overall_diff = nangate45_total - sky130hd_total
    overall_pct = (overall_diff / sky130hd_total * 100)
else:
    overall_winner = "aes_sky130hd"
    overall_diff = sky130hd_total - nangate45_total
    overall_pct = (overall_diff / nangate45_total * 100)

print("\nTotal Lines:")
print("  aes_nangate45: {} lines".format(nangate45_total))
print("  aes_sky130hd:  {} lines".format(sky130hd_total))
print("\n  Overall higher complexity: {} (+{} lines, +{:.1f}%)".format(
    overall_winner, overall_diff, overall_pct))

print("\nKey Stage Analysis:")
print("\n  PLACEMENT Stage:")
print("    aes_nangate45: {} lines ({:.1f}%)".format(
    ng45_totals.get('placement', 0),
    (ng45_totals.get('placement', 0) / nangate45_total * 100)))
print("    aes_sky130hd:  {} lines ({:.1f}%)".format(
    sky_totals.get('placement', 0),
    (sky_totals.get('placement', 0) / sky130hd_total * 100)))
print("    Higher complexity: {}".format(comparison['placement']['higher_complexity']))
if comparison['placement']['difference'] > 0:
    print("    Difference: +{} lines (+{:.1f}%)".format(
        comparison['placement']['difference'],
        comparison['placement']['difference_pct']))

print("\n  ROUTING Stage:")
print("    aes_nangate45: {} lines ({:.1f}%)".format(
    ng45_totals.get('routing', 0),
    (ng45_totals.get('routing', 0) / nangate45_total * 100)))
print("    aes_sky130hd:  {} lines ({:.1f}%)".format(
    sky_totals.get('routing', 0),
    (sky_totals.get('routing', 0) / sky130hd_total * 100)))
print("    Higher complexity: {}".format(comparison['routing']['higher_complexity']))
if comparison['routing']['difference'] > 0:
    print("    Difference: +{} lines (+{:.1f}%)".format(
        comparison['routing']['difference'],
        comparison['routing']['difference_pct']))

# JSON output
output = {
    "aes_nangate45_2026-02-02T11_19_30_370555": {
        "total_lines": nangate45_total,
        "total_stages": len(ng45_details),
        "category_line_counts": ng45_totals,
        "stage_details": ng45_details
    },
    "aes_sky130hd_2026-02-02T11_19_30_742696": {
        "total_lines": sky130hd_total,
        "total_stages": len(sky_details),
        "category_line_counts": sky_totals,
        "stage_details": sky_details
    },
    "comparison_by_category": comparison,
    "summary": {
        "overall_higher_complexity": overall_winner,
        "placement_higher_complexity": comparison['placement']['higher_complexity'],
        "routing_higher_complexity": comparison['routing']['higher_complexity'],
        "key_findings": {
            "placement": {
                "nangate45_lines": ng45_totals.get('placement', 0),
                "sky130hd_lines": sky_totals.get('placement', 0),
                "higher_complexity": comparison['placement']['higher_complexity'],
                "difference": comparison['placement']['difference'],
                "difference_pct": comparison['placement']['difference_pct']
            },
            "routing": {
                "nangate45_lines": ng45_totals.get('routing', 0),
                "sky130hd_lines": sky_totals.get('routing', 0),
                "higher_complexity": comparison['routing']['higher_complexity'],
                "difference": comparison['routing']['difference'],
                "difference_pct": comparison['routing']['difference_pct']
            }
        }
    }
}

print("\n" + "="*90)
print("JSON OUTPUT")
print("="*90)
print(json.dumps(output, indent=2))
