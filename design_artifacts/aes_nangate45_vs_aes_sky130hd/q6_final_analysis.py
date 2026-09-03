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
Comprehensive flow analysis for Q6 based on actual stage markers and line extraction.
"""

import json

# Data extracted from the log files:
# Format: stage_id: (stage_name, line_start, line_end or next_stage_start)

# AES NANGATE45 - based on grep and read results
aes_nangate45_stages = {
    'synthesis': {
        '1_1_yosys_canonicalize': {'start': 82, 'end': 119},  # Before line 119
        '1_2_yosys': {'start': 119, 'end': 400},  # Lines 119-400 (write_db 1_synth.odb at line 400)
    },
    'floorplan': {
        '2_1_floorplan': {'start': 405, 'end': 1924},  # flow.sh at 405, next stage at 1924
        '2_2_floorplan_macro': {'start': 1924, 'end': 1935},
        '2_3_floorplan_tapcell': {'start': 1935, 'end': 1947},
        '2_4_floorplan_pdn': {'start': 1947, 'end': 1960},
    },
    'placement': {
        '3_1_place_gp_skip_io': {'start': 1960, 'end': 1971},
        '3_2_place_iop': {'start': 1971, 'end': 1977},
        '3_3_place_gp': {'start': 1977, 'end': 2218},
        '3_4_place_resized': {'start': 2218, 'end': 2266},
        '3_5_place_dp': {'start': 2266, 'end': 2356},
    },
    'cts': {
        '4_1_cts': {'start': 2356, 'end': 2466},
    },
    'routing': {
        '5_1_grt': {'start': 2466, 'end': 2661},
        '5_2_route': {'start': 2661, 'end': 3262},
        '5_3_fillcell': {'start': 3262, 'end': 3276},
    },
    'finishing': {
        '6_1_fill': {'start': 3276, 'end': 3374},  # Metadata summary starts around 3374
    }
}

# AES SKY130HD - based on grep and read results
aes_sky130hd_stages = {
    'synthesis': {
        '1_1_yosys_canonicalize': {'start': 82, 'end': 119},
        '1_2_yosys': {'start': 119, 'end': 393},  # write_db 1_synth.odb at line 393
    },
    'floorplan': {
        '2_1_floorplan': {'start': 398, 'end': 442},
        '2_2_floorplan_macro': {'start': 442, 'end': 453},
        '2_3_floorplan_tapcell': {'start': 453, 'end': 464},
        '2_4_floorplan_pdn': {'start': 464, 'end': 477},
    },
    'placement': {
        '3_1_place_gp_skip_io': {'start': 477, 'end': 582},
        '3_2_place_iop': {'start': 582, 'end': 601},
        '3_3_place_gp': {'start': 601, 'end': 1174},
        '3_4_place_resized': {'start': 1174, 'end': 1224},
        '3_5_place_dp': {'start': 1224, 'end': 1319},
    },
    'cts': {
        '4_1_cts': {'start': 1319, 'end': 1432},
    },
    'routing': {
        '5_1_grt': {'start': 1432, 'end': 1709},
        '5_2_route': {'start': 1709, 'end': 2814},
        '5_3_fillcell': {'start': 2814, 'end': 2828},
    },
    'finishing': {
        '6_1_fill': {'start': 2828, 'end': 2924},  # Metadata summary starts around 2924
    }
}

def calculate_line_counts(stages_dict):
    """Calculate line counts for each stage and category."""
    results = {}
    total_lines = 0
    total_stages = 0
    
    for category, stages in stages_dict.items():
        category_lines = 0
        category_stages = []
        
        for stage_id, info in stages.items():
            lines = info['end'] - info['start']
            category_lines += lines
            total_lines += lines
            total_stages += 1
            
            category_stages.append({
                'stage_id': stage_id,
                'line_count': lines,
                'start_line': info['start'],
                'end_line': info['end']
            })
        
        results[category] = {
            'num_stages': len(stages),
            'total_lines': category_lines,
            'stages': category_stages
        }
    
    results['overall'] = {
        'total_stages': total_stages,
        'total_lines': total_lines
    }
    
    return results

def main():
    print("=" * 80)
    print("COMPREHENSIVE FLOW ANALYSIS - Q6")
    print("=" * 80)
    
    # Calculate line counts for both designs
    ng45_results = calculate_line_counts(aes_nangate45_stages)
    sky_results = calculate_line_counts(aes_sky130hd_stages)
    
    print("\n" + "=" * 80)
    print("AES NANGATE45 Flow Structure")
    print("=" * 80)
    print(f"Total Stages: {ng45_results['overall']['total_stages']}")
    print(f"Total Lines Extracted: {ng45_results['overall']['total_lines']}")
    print("\nBreakdown by Category:")
    
    for category in ['synthesis', 'floorplan', 'placement', 'cts', 'routing', 'finishing']:
        if category in ng45_results:
            data = ng45_results[category]
            print(f"\n  {category.upper()}:")
            print(f"    Sub-stages: {data['num_stages']}")
            print(f"    Total Lines: {data['total_lines']}")
            for stage in data['stages']:
                print(f"      - {stage['stage_id']}: {stage['line_count']} lines (lines {stage['start_line']}-{stage['end_line']})")
    
    print("\n" + "=" * 80)
    print("AES SKY130HD Flow Structure")
    print("=" * 80)
    print(f"Total Stages: {sky_results['overall']['total_stages']}")
    print(f"Total Lines Extracted: {sky_results['overall']['total_lines']}")
    print("\nBreakdown by Category:")
    
    for category in ['synthesis', 'floorplan', 'placement', 'cts', 'routing', 'finishing']:
        if category in sky_results:
            data = sky_results[category]
            print(f"\n  {category.upper()}:")
            print(f"    Sub-stages: {data['num_stages']}")
            print(f"    Total Lines: {data['total_lines']}")
            for stage in data['stages']:
                print(f"      - {stage['stage_id']}: {stage['line_count']} lines (lines {stage['start_line']}-{stage['end_line']})")
    
    print("\n" + "=" * 80)
    print("COMPARATIVE ANALYSIS")
    print("=" * 80)
    
    # Compare overall metrics
    print(f"\nTotal Number of Sub-stages:")
    print(f"  NANGATE45: {ng45_results['overall']['total_stages']} stages")
    print(f"  SKY130HD:  {sky_results['overall']['total_stages']} stages")
    print(f"  Difference: {abs(ng45_results['overall']['total_stages'] - sky_results['overall']['total_stages'])} stages")
    
    print(f"\nTotal Lines Extracted:")
    print(f"  NANGATE45: {ng45_results['overall']['total_lines']} lines")
    print(f"  SKY130HD:  {sky_results['overall']['total_lines']} lines")
    print(f"  Difference: {abs(ng45_results['overall']['total_lines'] - sky_results['overall']['total_lines'])} lines")
    print(f"  Ratio: {ng45_results['overall']['total_lines'] / sky_results['overall']['total_lines']:.2f}x")
    
    # Compare by category
    print(f"\nCategory-by-Category Comparison:")
    for category in ['synthesis', 'floorplan', 'placement', 'cts', 'routing', 'finishing']:
        ng45_data = ng45_results.get(category, {'num_stages': 0, 'total_lines': 0})
        sky_data = sky_results.get(category, {'num_stages': 0, 'total_lines': 0})
        
        print(f"\n  {category.upper()}:")
        print(f"    NANGATE45: {ng45_data['num_stages']} stages, {ng45_data['total_lines']} lines")
        print(f"    SKY130HD:  {sky_data['num_stages']} stages, {sky_data['total_lines']} lines")
        if sky_data['total_lines'] > 0:
            print(f"    NANGATE45 has {ng45_data['total_lines'] - sky_data['total_lines']:+d} more lines ({(ng45_data['total_lines'] / sky_data['total_lines']):.2f}x)")
    
    # Determine which design is more comprehensive
    print("\n" + "=" * 80)
    print("ANSWER TO Q6")
    print("=" * 80)
    
    # Both have the same number of stages (14 each)
    same_stages = ng45_results['overall']['total_stages'] == sky_results['overall']['total_stages']
    
    if same_stages:
        if ng45_results['overall']['total_lines'] > sky_results['overall']['total_lines']:
            more_comprehensive = 'aes_nangate45'
            reason = f"Both designs process the same number of sub-stages ({ng45_results['overall']['total_stages']} stages), but NANGATE45 has significantly more extracted lines ({ng45_results['overall']['total_lines']} vs {sky_results['overall']['total_lines']} lines, {(ng45_results['overall']['total_lines'] / sky_results['overall']['total_lines']):.2f}x ratio), indicating more detailed logging and comprehensive flow documentation."
        else:
            more_comprehensive = 'aes_sky130hd'
            reason = f"Both designs process the same number of sub-stages ({sky_results['overall']['total_stages']} stages), but SKY130HD has more extracted lines ({sky_results['overall']['total_lines']} vs {ng45_results['overall']['total_lines']} lines), indicating more detailed logging and comprehensive flow documentation."
    elif ng45_results['overall']['total_stages'] > sky_results['overall']['total_stages']:
        more_comprehensive = 'aes_nangate45'
        reason = f"NANGATE45 processes more sub-stages ({ng45_results['overall']['total_stages']} vs {sky_results['overall']['total_stages']}) with more extracted lines ({ng45_results['overall']['total_lines']} vs {sky_results['overall']['total_lines']})."
    else:
        more_comprehensive = 'aes_sky130hd'
        reason = f"SKY130HD processes more sub-stages ({sky_results['overall']['total_stages']} vs {ng45_results['overall']['total_stages']}) with more extracted lines ({sky_results['overall']['total_lines']} vs {ng45_results['overall']['total_lines']})."
    
    answer = {
        'more_comprehensive_design': more_comprehensive,
        'reason': reason,
        'nangate45_metrics': {
            'total_stages': ng45_results['overall']['total_stages'],
            'total_lines': ng45_results['overall']['total_lines'],
            'stages_by_category': {cat: ng45_results[cat]['num_stages'] for cat in ['synthesis', 'floorplan', 'placement', 'cts', 'routing', 'finishing']},
            'lines_by_category': {cat: ng45_results[cat]['total_lines'] for cat in ['synthesis', 'floorplan', 'placement', 'cts', 'routing', 'finishing']}
        },
        'sky130hd_metrics': {
            'total_stages': sky_results['overall']['total_stages'],
            'total_lines': sky_results['overall']['total_lines'],
            'stages_by_category': {cat: sky_results[cat]['num_stages'] for cat in ['synthesis', 'floorplan', 'placement', 'cts', 'routing', 'finishing']},
            'lines_by_category': {cat: sky_results[cat]['total_lines'] for cat in ['synthesis', 'floorplan', 'placement', 'cts', 'routing', 'finishing']}
        }
    }
    
    print(f"\nMore Comprehensive Design: {more_comprehensive}")
    print(f"\nReason: {reason}")
    
    # Additional insights
    print(f"\nKey Observations:")
    print(f"1. Both designs follow the same standard EDA flow with identical stage structure")
    print(f"2. NANGATE45 has {(ng45_results['overall']['total_lines'] / sky_results['overall']['total_lines']):.1%} more log content")
    print(f"3. The most significant difference is in the floorplan stage:")
    print(f"   - NANGATE45: {ng45_results['floorplan']['total_lines']} lines")
    print(f"   - SKY130HD:  {sky_results['floorplan']['total_lines']} lines")
    print(f"   - Ratio: {(ng45_results['floorplan']['total_lines'] / sky_results['floorplan']['total_lines']):.2f}x")
    
    # Save to JSON
    with open('q6_answer.json', 'w') as f:
        json.dump(answer, f, indent=2)
    
    print("\n" + "=" * 80)
    print("Answer saved to q6_answer.json")
    print("=" * 80)
    
    return answer

if __name__ == '__main__':
    main()
