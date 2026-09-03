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

import re

def count_lines_per_stage(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    stage_pattern = re.compile(r'Running\s+(\S+\.tcl),\s+stage\s+(\S+)')
    stages = []
    
    for i, line in enumerate(lines):
        match = stage_pattern.search(line)
        if match:
            tcl_file = match.group(1)
            stage_name = match.group(2)
            stages.append({
                'line_num': i + 1,
                'stage': stage_name,
                'tcl': tcl_file
            })
    
    # Calculate line counts for each stage
    stage_info = []
    for i in range(len(stages)):
        start_line = stages[i]['line_num']
        if i < len(stages) - 1:
            end_line = stages[i + 1]['line_num'] - 1
        else:
            end_line = len(lines)
        
        line_count = end_line - start_line + 1
        stage_info.append({
            'stage': stages[i]['stage'],
            'tcl': stages[i]['tcl'],
            'start': start_line,
            'end': end_line,
            'lines': line_count
        })
    
    return stage_info, len(lines)

# Analyze both designs
nangate45_stages, nangate45_total = count_lines_per_stage('aes_nangate45.txt')
sky130hs_stages, sky130hs_total = count_lines_per_stage('aes_sky130hs.txt')

print("=" * 100)
print(f"AES_NANGATE45 - Total lines: {nangate45_total}")
print("=" * 100)
for stage in nangate45_stages:
    print(f"{stage['stage']:30s} | Lines: {stage['lines']:6d} | [{stage['start']:6d} - {stage['end']:6d}]")

print("\n" + "=" * 100)
print(f"AES_SKY130HS - Total lines: {sky130hs_total}")
print("=" * 100)
for stage in sky130hs_stages:
    print(f"{stage['stage']:30s} | Lines: {stage['lines']:6d} | [{stage['start']:6d} - {stage['end']:6d}]")

# Create comparison table
print("\n" + "=" * 120)
print("STAGE COMPARISON")
print("=" * 120)
print(f"{'Stage':30s} | {'Nangate45':>12s} | {'Sky130hs':>12s} | {'Difference':>12s} | {'Ratio (Sky/Nan)':>15s}")
print("-" * 120)

stage_dict_ng45 = {s['stage']: s['lines'] for s in nangate45_stages}
stage_dict_sky = {s['stage']: s['lines'] for s in sky130hs_stages}

all_stages = sorted(set(stage_dict_ng45.keys()) | set(stage_dict_sky.keys()))

for stage in all_stages:
    ng45_count = stage_dict_ng45.get(stage, 0)
    sky_count = stage_dict_sky.get(stage, 0)
    diff = sky_count - ng45_count
    ratio = sky_count / ng45_count if ng45_count > 0 else 0
    print(f"{stage:30s} | {ng45_count:12d} | {sky_count:12d} | {diff:+12d} | {ratio:15.2f}x")

print("-" * 120)
print(f"{'TOTAL':30s} | {nangate45_total:12d} | {sky130hs_total:12d} | {sky130hs_total - nangate45_total:+12d} | {sky130hs_total/nangate45_total:15.2f}x")
print("=" * 120)
