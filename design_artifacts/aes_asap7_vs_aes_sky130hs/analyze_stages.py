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

"""Analyze stage line counts in log files."""

import re
from collections import OrderedDict

def analyze_log_file(filepath):
    """Analyze log file and count lines for each stage."""
    
    # Stage markers and their line numbers
    stage_markers = OrderedDict()
    
    # Read file and identify stage boundaries
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            # Look for stage markers
            if 'Executing SYNTH pass' in line and line_num < 200:
                if 'synthesis' not in stage_markers:
                    stage_markers['synthesis'] = line_num
            elif 'Floorplan check_setup' in line:
                if 'floorplan' not in stage_markers:
                    stage_markers['floorplan'] = line_num
            elif 'global place report_design_area' in line:
                if 'global_placement' not in stage_markers:
                    stage_markers['global_placement'] = line_num
            elif 'resizer report_design_area' in line:
                if 'resizer' not in stage_markers:
                    stage_markers['resizer'] = line_num
            elif 'detailed place report_design_area' in line:
                if 'detailed_placement' not in stage_markers:
                    stage_markers['detailed_placement'] = line_num
            elif 'cts final report_design_area' in line:
                if 'cts' not in stage_markers:
                    stage_markers['cts'] = line_num
            elif 'global route report_design_area' in line:
                if 'routing' not in stage_markers:
                    stage_markers['routing'] = line_num
            elif 'finish report_design_area' in line:
                if 'finish' not in stage_markers:
                    stage_markers['finish'] = line_num
    
    # Get total lines
    with open(filepath, 'r') as f:
        total_lines = sum(1 for _ in f)
    
    # Calculate line counts for each stage
    stage_line_counts = OrderedDict()
    stage_names = list(stage_markers.keys())
    
    for i, stage_name in enumerate(stage_names):
        start_line = stage_markers[stage_name]
        
        # End line is the start of next stage or end of file
        if i + 1 < len(stage_names):
            end_line = stage_markers[stage_names[i + 1]] - 1
        else:
            end_line = total_lines
        
        line_count = end_line - start_line + 1
        stage_line_counts[stage_name] = {
            'start': start_line,
            'end': end_line,
            'lines': line_count
        }
    
    return stage_line_counts, total_lines

def main():
    # Analyze both log files
    print("=" * 80)
    print("Stage-by-Stage Line Count Analysis")
    print("=" * 80)
    
    asap7_results, asap7_total = analyze_log_file('aes_asap7.txt')
    sky130hs_results, sky130hs_total = analyze_log_file('aes_sky130hs.txt')
    
    print(f"\n{'Stage':<25} {'aes_asap7 Lines':<20} {'aes_sky130hs Lines':<20} {'Difference':<15}")
    print("-" * 80)
    
    all_stages = set(asap7_results.keys()) | set(sky130hs_results.keys())
    
    for stage in ['synthesis', 'floorplan', 'global_placement', 'resizer', 
                  'detailed_placement', 'cts', 'routing', 'finish']:
        if stage in all_stages:
            asap7_lines = asap7_results.get(stage, {}).get('lines', 0)
            sky130hs_lines = sky130hs_results.get(stage, {}).get('lines', 0)
            diff = asap7_lines - sky130hs_lines
            
            print(f"{stage:<25} {asap7_lines:<20} {sky130hs_lines:<20} {diff:+15}")
    
    print("-" * 80)
    print(f"{'TOTAL':<25} {asap7_total:<20} {sky130hs_total:<20} {asap7_total - sky130hs_total:+15}")
    print("\n")
    
    # Print detailed stage information
    print("=" * 80)
    print("Detailed Stage Information - aes_asap7")
    print("=" * 80)
    for stage, info in asap7_results.items():
        print(f"\n{stage.upper()}:")
        print(f"  Start Line: {info['start']}")
        print(f"  End Line: {info['end']}")
        print(f"  Total Lines: {info['lines']}")
    
    print("\n" + "=" * 80)
    print("Detailed Stage Information - aes_sky130hs")
    print("=" * 80)
    for stage, info in sky130hs_results.items():
        print(f"\n{stage.upper()}:")
        print(f"  Start Line: {info['start']}")
        print(f"  End Line: {info['end']}")
        print(f"  Total Lines: {info['lines']}")
    
    # Determine which design has more lines
    print("\n" + "=" * 80)
    print("Summary and Analysis")
    print("=" * 80)
    
    asap7_more_count = sum(1 for stage in all_stages 
                           if asap7_results.get(stage, {}).get('lines', 0) > 
                           sky130hs_results.get(stage, {}).get('lines', 0))
    
    sky130hs_more_count = sum(1 for stage in all_stages 
                              if sky130hs_results.get(stage, {}).get('lines', 0) > 
                              asap7_results.get(stage, {}).get('lines', 0))
    
    print(f"\nStages where aes_asap7 has more lines: {asap7_more_count}")
    print(f"Stages where aes_sky130hs has more lines: {sky130hs_more_count}")
    
    if asap7_total > sky130hs_total:
        print(f"\nOverall, aes_asap7 has more extracted lines ({asap7_total} vs {sky130hs_total}).")
        print("This indicates higher verbosity or complexity in the asap7 design flow.")
    else:
        print(f"\nOverall, aes_sky130hs has more extracted lines ({sky130hs_total} vs {asap7_total}).")
        print("This indicates higher verbosity or complexity in the sky130hs design flow.")
    
    # Find stages with largest differences
    print("\nStages with largest line count differences:")
    differences = []
    for stage in all_stages:
        asap7_lines = asap7_results.get(stage, {}).get('lines', 0)
        sky130hs_lines = sky130hs_results.get(stage, {}).get('lines', 0)
        diff = abs(asap7_lines - sky130hs_lines)
        differences.append((stage, asap7_lines, sky130hs_lines, asap7_lines - sky130hs_lines))
    
    differences.sort(key=lambda x: abs(x[3]), reverse=True)
    
    for i, (stage, asap7_lines, sky130hs_lines, diff) in enumerate(differences[:5], 1):
        design_with_more = "aes_asap7" if diff > 0 else "aes_sky130hs"
        print(f"{i}. {stage}: {design_with_more} has {abs(diff)} more lines ({asap7_lines} vs {sky130hs_lines})")

if __name__ == '__main__':
    main()
