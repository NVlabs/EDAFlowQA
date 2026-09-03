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
Script to count lines per stage in EDA log files.
Stages are identified by:
1. Synthesis (1_*)
2. Floorplan (2_*)
3. Placement (3_*)
4. CTS (4_*)
5. Routing (5_*)
6. Finish (6_*)
"""

import re
import sys

def identify_stage(line):
    """Identify which stage a line belongs to based on log/result file references"""
    # Look for stage markers like logs/.../base/N_... or results/.../base/N_...
    patterns = {
        'synthesis': r'(logs|results)/\w+/\w+/base/1_',
        'floorplan': r'(logs|results)/\w+/\w+/base/2_',
        'placement': r'(logs|results)/\w+/\w+/base/3_',
        'cts': r'(logs|results)/\w+/\w+/base/4_',
        'routing': r'(logs|results)/\w+/\w+/base/5_',
        'finish': r'(logs|results)/\w+/\w+/base/6_'
    }
    
    for stage, pattern in patterns.items():
        if re.search(pattern, line):
            return stage
    return None

def count_lines_per_stage(filename):
    """Count lines per stage in a log file"""
    
    stage_counts = {
        'synthesis': 0,
        'floorplan': 0,
        'placement': 0,
        'cts': 0,
        'routing': 0,
        'finish': 0,
        'other': 0
    }
    
    current_stage = None
    total_lines = 0
    stage_transitions = []
    
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        for line_num, line in enumerate(f, 1):
            total_lines += 1
            
            # Check if this line indicates a stage transition
            detected_stage = identify_stage(line)
            
            if detected_stage:
                if current_stage != detected_stage:
                    stage_transitions.append((line_num, detected_stage, line.strip()[:100]))
                    current_stage = detected_stage
            
            # Count the line for the current stage
            if current_stage:
                stage_counts[current_stage] += 1
            else:
                stage_counts['other'] += 1
    
    return stage_counts, total_lines, stage_transitions

def main():
    if len(sys.argv) < 2:
        print("Usage: count_stage_lines.py <logfile1> [logfile2] ...")
        sys.exit(1)
    
    results = {}
    
    for filename in sys.argv[1:]:
        print(f"\n{'='*80}")
        print(f"Analyzing: {filename}")
        print(f"{'='*80}")
        
        try:
            stage_counts, total_lines, stage_transitions = count_lines_per_stage(filename)
            results[filename] = stage_counts
            
            print(f"\nTotal lines: {total_lines}")
            print(f"\nStage transitions detected: {len(stage_transitions)}")
            print("\nFirst few stage transitions:")
            for line_num, stage, line_text in stage_transitions[:10]:
                print(f"  Line {line_num}: {stage} - {line_text}")
            
            print(f"\nLines per stage:")
            for stage in ['synthesis', 'floorplan', 'placement', 'cts', 'routing', 'finish', 'other']:
                count = stage_counts[stage]
                pct = (count / total_lines * 100) if total_lines > 0 else 0
                print(f"  {stage:15s}: {count:6d} lines ({pct:5.1f}%)")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    # Comparison
    if len(results) == 2:
        print(f"\n{'='*80}")
        print("COMPARISON")
        print(f"{'='*80}")
        
        files = list(results.keys())
        file1, file2 = files[0], files[1]
        
        print(f"\n{'Stage':<15s} | {file1.split('/')[-1]:<20s} | {file2.split('/')[-1]:<20s} | Difference")
        print("-" * 80)
        
        for stage in ['synthesis', 'floorplan', 'placement', 'cts', 'routing', 'finish']:
            count1 = results[file1][stage]
            count2 = results[file2][stage]
            diff = count1 - count2
            
            if count1 > count2:
                winner = file1.split('/')[-1]
            elif count2 > count1:
                winner = file2.split('/')[-1]
            else:
                winner = "same"
            
            print(f"{stage:<15s} | {count1:6d} lines       | {count2:6d} lines       | {diff:+6d} ({winner})")

if __name__ == "__main__":
    main()
