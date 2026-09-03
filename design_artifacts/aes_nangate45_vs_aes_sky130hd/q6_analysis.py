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
Analyze which design has a more comprehensive flow based on:
1. Number of sub-designs (stages) processed
2. Extracted lines in each stage
"""

import re
import json

def analyze_log_file(file_path):
    """Analyze a log file to extract stage information and line counts."""
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find all stage markers
    stage_pattern = re.compile(r'Running .*, stage (\S+)')
    stages = []
    stage_line_counts = {}
    
    for idx, line in enumerate(lines):
        match = stage_pattern.search(line)
        if match:
            stage_name = match.group(1)
            stages.append({
                'name': stage_name,
                'line_number': idx + 1,
                'index': idx
            })
    
    # Calculate line counts for each stage
    for i, stage in enumerate(stages):
        stage_name = stage['name']
        start_idx = stage['index']
        
        # End index is either the next stage or end of file
        if i < len(stages) - 1:
            end_idx = stages[i + 1]['index']
        else:
            end_idx = len(lines)
        
        line_count = end_idx - start_idx
        stage_line_counts[stage_name] = line_count
    
    return {
        'stages': stages,
        'stage_line_counts': stage_line_counts,
        'total_stages': len(stages),
        'total_lines': len(lines)
    }

def main():
    # Analyze both designs
    nangate45_analysis = analyze_log_file('aes_nangate45.txt')
    sky130hd_analysis = analyze_log_file('aes_sky130hd.txt')
    
    print("=" * 80)
    print("AES NANGATE45 FLOW ANALYSIS")
    print("=" * 80)
    print(f"Total lines: {nangate45_analysis['total_lines']}")
    print(f"Total stages: {nangate45_analysis['total_stages']}")
    print("\nStages and line counts:")
    for stage_name, line_count in nangate45_analysis['stage_line_counts'].items():
        print(f"  {stage_name:30s}: {line_count:6d} lines")
    
    print("\n" + "=" * 80)
    print("AES SKY130HD FLOW ANALYSIS")
    print("=" * 80)
    print(f"Total lines: {sky130hd_analysis['total_lines']}")
    print(f"Total stages: {sky130hd_analysis['total_stages']}")
    print("\nStages and line counts:")
    for stage_name, line_count in sky130hd_analysis['stage_line_counts'].items():
        print(f"  {stage_name:30s}: {line_count:6d} lines")
    
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print(f"NANGATE45: {nangate45_analysis['total_stages']} stages, {nangate45_analysis['total_lines']} total lines")
    print(f"SKY130HD:  {sky130hd_analysis['total_stages']} stages, {sky130hd_analysis['total_lines']} total lines")
    
    # Determine which has more comprehensive flow
    if nangate45_analysis['total_stages'] > sky130hd_analysis['total_stages']:
        more_stages = "aes_nangate45"
    elif sky130hd_analysis['total_stages'] > nangate45_analysis['total_stages']:
        more_stages = "aes_sky130hd"
    else:
        more_stages = "equal"
    
    if nangate45_analysis['total_lines'] > sky130hd_analysis['total_lines']:
        more_lines = "aes_nangate45"
    elif sky130hd_analysis['total_lines'] > nangate45_analysis['total_lines']:
        more_lines = "aes_sky130hd"
    else:
        more_lines = "equal"
    
    # Calculate average lines per stage
    ng45_avg = nangate45_analysis['total_lines'] / nangate45_analysis['total_stages']
    sky130_avg = sky130hd_analysis['total_lines'] / sky130hd_analysis['total_stages']
    
    print(f"\nAverage lines per stage:")
    print(f"  NANGATE45: {ng45_avg:.1f}")
    print(f"  SKY130HD:  {sky130_avg:.1f}")
    
    # Detailed stage-by-stage comparison
    print("\n" + "=" * 80)
    print("STAGE-BY-STAGE LINE COUNT COMPARISON")
    print("=" * 80)
    print(f"{'Stage':<30s} {'NANGATE45':>12s} {'SKY130HD':>12s} {'Difference':>12s}")
    print("-" * 80)
    
    all_stages = set(nangate45_analysis['stage_line_counts'].keys()) | set(sky130hd_analysis['stage_line_counts'].keys())
    for stage in sorted(all_stages):
        ng45_count = nangate45_analysis['stage_line_counts'].get(stage, 0)
        sky130_count = sky130hd_analysis['stage_line_counts'].get(stage, 0)
        diff = ng45_count - sky130_count
        print(f"{stage:<30s} {ng45_count:>12d} {sky130_count:>12d} {diff:>12d}")
    
    # Determine overall answer
    if more_stages == "equal" and nangate45_analysis['total_lines'] > sky130hd_analysis['total_lines']:
        comprehensive_design = "aes_nangate45"
        reason = f"Both designs have the same number of stages ({nangate45_analysis['total_stages']}), but aes_nangate45 has more total extracted lines ({nangate45_analysis['total_lines']} vs {sky130hd_analysis['total_lines']}), indicating more detailed logging and processing."
    elif more_stages == "equal" and sky130hd_analysis['total_lines'] > nangate45_analysis['total_lines']:
        comprehensive_design = "aes_sky130hd"
        reason = f"Both designs have the same number of stages ({sky130hd_analysis['total_stages']}), but aes_sky130hd has more total extracted lines ({sky130hd_analysis['total_lines']} vs {nangate45_analysis['total_lines']}), indicating more detailed logging and processing."
    elif more_stages != "equal":
        comprehensive_design = more_stages
        reason = f"{comprehensive_design} has more stages ({nangate45_analysis['total_stages'] if more_stages == 'aes_nangate45' else sky130hd_analysis['total_stages']}) compared to the other."
    else:
        comprehensive_design = "equal"
        reason = "Both designs have equal number of stages and similar line counts."
    
    # Create answer
    answer = {
        "design_with_more_comprehensive_flow": comprehensive_design,
        "reasoning": reason,
        "aes_nangate45": {
            "total_stages": nangate45_analysis['total_stages'],
            "total_lines": nangate45_analysis['total_lines'],
            "average_lines_per_stage": round(ng45_avg, 1),
            "stage_line_counts": nangate45_analysis['stage_line_counts']
        },
        "aes_sky130hd": {
            "total_stages": sky130hd_analysis['total_stages'],
            "total_lines": sky130hd_analysis['total_lines'],
            "average_lines_per_stage": round(sky130_avg, 1),
            "stage_line_counts": sky130hd_analysis['stage_line_counts']
        }
    }
    
    print("\n" + "=" * 80)
    print("FINAL ANSWER")
    print("=" * 80)
    print(f"Design with more comprehensive flow: {comprehensive_design}")
    print(f"Reason: {reason}")
    
    # Save to JSON
    with open('q6_final_answer.json', 'w') as f:
        json.dump(answer, f, indent=2)
    
    print("\nAnswer saved to q6_final_answer.json")
    
    return answer

if __name__ == '__main__':
    main()
