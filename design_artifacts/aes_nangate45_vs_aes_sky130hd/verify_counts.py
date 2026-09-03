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
Quick verification and calculation of stage line counts.
"""

# NANGATE45 stage transitions (line number where stage starts)
ng45_transitions = [
    (393, "1_synth"),
    (406, "2_1_floorplan"),
    (1925, "2_2_floorplan_macro"),
    (1936, "2_3_floorplan_tapcell"),
    (1948, "2_4_floorplan_pdn"),
    (1961, "3_1_place_gp_skip_io"),
    (1972, "3_2_place_iop"),
    (1978, "3_3_place_gp"),
    (2219, "3_4_place_resized"),
    (2267, "3_5_place_dp"),
    (2357, "4_1_cts"),
    (2467, "5_1_grt"),
    (2662, "5_2_route"),
    (3263, "5_3_fillcell"),
    (3277, "6_1_fill"),
    (3288, "6_report"),
]
ng45_total = 3464

# SKY130HD stage transitions
sky_transitions = [
    (386, "1_synth"),
    (399, "2_1_floorplan"),
    (443, "2_2_floorplan_macro"),
    (454, "2_3_floorplan_tapcell"),
    (465, "2_4_floorplan_pdn"),
    (478, "3_1_place_gp_skip_io"),
    (583, "3_2_place_iop"),
    (602, "3_3_place_gp"),
    (1175, "3_4_place_resized"),
    (1225, "3_5_place_dp"),
    (1320, "4_1_cts"),
    (1433, "5_1_grt"),
    (1710, "5_2_route"),
    (2815, "5_3_fillcell"),
    (2829, "6_1_fill"),
    (2840, "6_report"),
]
sky_total = 3015

def calculate_lines(transitions, total):
    """Calculate lines for each stage."""
    lines = {}
    
    # Init lines (before first stage)
    lines['0_init'] = transitions[0][0] - 1
    
    # Each stage runs from its line to the next stage's line - 1
    for i in range(len(transitions)):
        start_line, stage_name = transitions[i]
        
        if i < len(transitions) - 1:
            end_line = transitions[i + 1][0] - 1
        else:
            end_line = total
        
        lines[stage_name] = end_line - start_line + 1
    
    return lines

# Calculate
ng45_lines = calculate_lines(ng45_transitions, ng45_total)
sky_lines = calculate_lines(sky_transitions, sky_total)

# Categorize
def categorize(lines):
    cats = {
        'init': 0,
        'synthesis': 0,
        'floorplan': 0,
        'placement': 0,
        'cts': 0,
        'routing': 0,
        'finish': 0
    }
    
    for stage, count in lines.items():
        if stage.startswith('0_'):
            cats['init'] += count
        elif stage.startswith('1_'):
            cats['synthesis'] += count
        elif stage.startswith('2_'):
            cats['floorplan'] += count
        elif stage.startswith('3_'):
            cats['placement'] += count
        elif stage.startswith('4_'):
            cats['cts'] += count
        elif stage.startswith('5_'):
            cats['routing'] += count
        elif stage.startswith('6_'):
            cats['finish'] += count
    
    return cats

ng45_cats = categorize(ng45_lines)
sky_cats = categorize(sky_lines)

print("VERIFICATION OF LINE COUNTS")
print("="*80)

print("\nNANGATE45 - Line counts per stage:")
for stage in sorted(ng45_lines.keys()):
    print(f"  {stage:<30s}: {ng45_lines[stage]:>6d} lines")

print(f"\nNANGATE45 - Category totals:")
for cat, count in ng45_cats.items():
    print(f"  {cat:<15s}: {count:>6d} lines")
print(f"  {'TOTAL':<15s}: {sum(ng45_lines.values()):>6d} lines (should be {ng45_total})")

print("\n" + "="*80)
print("\nSKY130HD - Line counts per stage:")
for stage in sorted(sky_lines.keys()):
    print(f"  {stage:<30s}: {sky_lines[stage]:>6d} lines")

print(f"\nSKY130HD - Category totals:")
for cat, count in sky_cats.items():
    print(f"  {cat:<15s}: {count:>6d} lines")
print(f"  {'TOTAL':<15s}: {sum(sky_lines.values()):>6d} lines (should be {sky_total})")

print("\n" + "="*80)
print("COMPARISON")
print("="*80)

print(f"\n{'Category':<15s} {'NANGATE45':>12s} {'SKY130HD':>12s} {'Diff':>12s} {'Winner':>15s}")
print("-"*80)

for cat in ['init', 'synthesis', 'floorplan', 'placement', 'cts', 'routing', 'finish']:
    ng = ng45_cats[cat]
    sk = sky_cats[cat]
    diff = ng - sk
    winner = "nangate45" if ng > sk else ("sky130hd" if sk > ng else "equal")
    print(f"{cat:<15s} {ng:>12d} {sk:>12d} {diff:>+12d} {winner:>15s}")

print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)

placement_winner = "nangate45" if ng45_cats['placement'] > sky_cats['placement'] else "sky130hd"
routing_winner = "nangate45" if ng45_cats['routing'] > sky_cats['routing'] else "sky130hd"

print(f"\nPLACEMENT: {placement_winner.upper()} has more lines")
print(f"  nangate45: {ng45_cats['placement']} lines")
print(f"  sky130hd:  {sky_cats['placement']} lines")
print(f"  Difference: {abs(ng45_cats['placement'] - sky_cats['placement'])} lines")

print(f"\nROUTING: {routing_winner.upper()} has more lines")
print(f"  nangate45: {ng45_cats['routing']} lines")
print(f"  sky130hd:  {sky_cats['routing']} lines")
print(f"  Difference: {abs(ng45_cats['routing'] - sky_cats['routing'])} lines")
