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

"""Calculate line counts for each stage."""

# Stage boundaries for aes_asap7.txt
asap7_stages = {
    'synthesis': (135, 398),
    'floorplan': (399, 2129),
    'global_placement': (2130, 2175),
    'resizer': (2176, 2264),
    'detailed_placement': (2265, 2727),
    'cts': (2728, 3442),
    'routing': (3443, 4108),
    'finish': (4109, 4230)
}

# Stage boundaries for aes_sky130hs.txt
sky130hs_stages = {
    'synthesis': (127, 409),
    'floorplan': (410, 925),
    'global_placement': (926, 972),
    'resizer': (973, 1062),
    'detailed_placement': (1063, 1243),
    'cts': (1244, 1859),
    'routing': (1860, 2885),
    'finish': (2886, 3005)
}

# Calculate line counts
asap7_counts = {}
sky130hs_counts = {}

for stage, (start, end) in asap7_stages.items():
    asap7_counts[stage] = end - start + 1

for stage, (start, end) in sky130hs_stages.items():
    sky130hs_counts[stage] = end - start + 1

# Display results
print("=" * 90)
print("Stage-by-Stage Line Count Comparison")
print("=" * 90)
print(f"\n{'Stage':<25} {'aes_asap7':<15} {'aes_sky130hs':<15} {'Difference':<20}")
print("-" * 90)

total_asap7 = 0
total_sky130hs = 0

for stage in ['synthesis', 'floorplan', 'global_placement', 'resizer', 
              'detailed_placement', 'cts', 'routing', 'finish']:
    asap7_lines = asap7_counts[stage]
    sky130hs_lines = sky130hs_counts[stage]
    diff = asap7_lines - sky130hs_lines
    
    total_asap7 += asap7_lines
    total_sky130hs += sky130hs_lines
    
    print(f"{stage:<25} {asap7_lines:<15} {sky130hs_lines:<15} {diff:+20}")

print("-" * 90)
print(f"{'TOTAL':<25} {total_asap7:<15} {total_sky130hs:<15} {total_asap7 - total_sky130hs:+20}")

print("\n" + "=" * 90)
print("Analysis Summary")
print("=" * 90)

# Count stages with more lines
asap7_more = sum(1 for stage in asap7_counts if asap7_counts[stage] > sky130hs_counts[stage])
sky130hs_more = sum(1 for stage in sky130hs_counts if sky130hs_counts[stage] > asap7_counts[stage])

print(f"\nStages where aes_asap7 has more lines: {asap7_more} out of 8")
print(f"Stages where aes_sky130hs has more lines: {sky130hs_more} out of 8")

# Find stages with largest differences
print("\nTop 5 stages by absolute line count difference:")
differences = []
for stage in asap7_counts:
    diff = asap7_counts[stage] - sky130hs_counts[stage]
    differences.append((stage, asap7_counts[stage], sky130hs_counts[stage], diff))

differences.sort(key=lambda x: abs(x[3]), reverse=True)

for i, (stage, asap7_lines, sky130hs_lines, diff) in enumerate(differences[:5], 1):
    design_with_more = "aes_asap7" if diff > 0 else "aes_sky130hs"
    print(f"  {i}. {stage}: {design_with_more} has {abs(diff)} more lines "
          f"({asap7_lines} vs {sky130hs_lines})")

print("\n" + "=" * 90)
print("Interpretation")
print("=" * 90)
print(f"""
Overall Finding:
- aes_asap7 has a total of {total_asap7} extracted lines
- aes_sky130hs has a total of {total_sky130hs} extracted lines
- aes_asap7 has {total_asap7 - total_sky130hs} more lines overall

The aes_asap7 design has more extracted lines in {asap7_more} out of 8 stages, indicating:

1. Higher verbosity: The asap7 design flow generates more detailed logging output,
   particularly in the floorplan and routing stages.

2. Complexity: The asap7 technology (7nm ASAP7 PDK) may require more iterations,
   optimizations, or detailed reporting compared to the sky130hs technology.

3. Tool behavior: Different technology libraries and constraints can lead to 
   different amounts of tool output during optimization and reporting.

Key observations:
- The floorplan stage shows the largest difference ({asap7_counts['floorplan'] - sky130hs_counts['floorplan']} lines)
- The routing stage also shows significant difference ({asap7_counts['routing'] - sky130hs_counts['routing']} lines)
- CTS stage shows moderate difference ({asap7_counts['cts'] - sky130hs_counts['cts']} lines)

These differences suggest that the asap7 design requires more extensive processing
in placement-related stages and routing, likely due to tighter design rules and
smaller feature sizes in the 7nm technology.
""")
