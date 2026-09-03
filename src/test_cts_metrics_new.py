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
Test script to demonstrate the new CTS metrics structure with flattened metrics
and optimization processes in execution order.
"""
from cts_metrics_parser import extract_cts_metrics, print_metrics_summary
import json

print("="*80)
print("TESTING NEW CTS METRICS STRUCTURE")
print("="*80)

# Test with ariane log
print("\n\n1. Testing with ariane133_nan45 CTS log...")
print("-" * 80)

with open('test_extraction_filtered/ariane133_nan45_cts_extract.txt', 'r') as f:
    log_content = f.read()

metrics = extract_cts_metrics(log_content)

# Show structure
print(f"\nDetected substages: {list(metrics['substages'].keys())}")

# Show metrics for first substage
if metrics['substages']:
    first_substage = list(metrics['substages'].keys())[0]
    substage_data = metrics['substages'][first_substage]
    
    print(f"\nSubstage '{first_substage}' structure:")
    print(f"  Direct metrics available:")
    for key in ['wns', 'tns', 'clock_sink_wirelength_um', 'hpwl', 'area', 'utilization', 'density', 'displacement', 'violations']:
        val = substage_data.get(key)
        if val is not None:
            print(f"    - {key}: {val}")
    
    print(f"\n  Optimization processes: {len(substage_data.get('optimization_process', []))} found")
    for idx, opt in enumerate(substage_data.get('optimization_process', []), 1):
        print(f"    [{idx}] {opt['type']}")

# Print full summary
print("\n" + "=" * 80)
print_metrics_summary(metrics)

# Save to JSON
with open('cts_metrics_ariane_new.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"\nMetrics saved to cts_metrics_ariane_new.json")


# Test with gcdshy130 log
print("\n\n" + "#" * 80)
print("2. Testing with gcdshy130 CTS log...")
print("#" * 80)

with open('test_extraction_filtered/gcdshy130_cts_extract.txt', 'r') as f:
    log_content2 = f.read()

metrics2 = extract_cts_metrics(log_content2)

# Show structure
print(f"\nDetected substages: {list(metrics2['substages'].keys())}")

# Show metrics for first substage
if metrics2['substages']:
    first_substage = list(metrics2['substages'].keys())[0]
    substage_data = metrics2['substages'][first_substage]
    
    print(f"\nSubstage '{first_substage}' structure:")
    print(f"  Direct metrics available:")
    for key in ['wns', 'tns', 'clock_sink_wirelength_um', 'hpwl', 'area', 'utilization', 'density', 'displacement', 'violations']:
        val = substage_data.get(key)
        if val is not None:
            print(f"    - {key}: {val}")
    
    print(f"\n  Optimization processes: {len(substage_data.get('optimization_process', []))} found")
    for idx, opt in enumerate(substage_data.get('optimization_process', []), 1):
        print(f"    [{idx}] {opt['type']}")
        if opt['type'] == 'repair_timing':
            print(f"        - Iterations: {opt['iterations_count']}")
            if opt.get('start_metrics') and opt.get('end_metrics'):
                start = opt['start_metrics']
                end = opt['end_metrics']
                if start.get('wns') is not None and end.get('wns') is not None:
                    print(f"        - WNS: {start['wns']:.3f} → {end['wns']:.3f}")
                if start.get('tns') is not None and end.get('tns') is not None:
                    print(f"        - TNS: {start['tns']:.1f} → {end['tns']:.1f}")

# Print full summary
print("\n" + "=" * 80)
print_metrics_summary(metrics2)

# Save to JSON
with open('cts_metrics_gcdshy130_new.json', 'w') as f:
    json.dump(metrics2, f, indent=2)
print(f"\nMetrics saved to cts_metrics_gcdshy130_new.json")


print("\n\n" + "=" * 80)
print("SUMMARY OF NEW STRUCTURE")
print("=" * 80)
print("""
Key improvements:
1. Metrics are now directly accessible at substage level:
   - substage['wns'], substage['tns'], substage['hpwl'], etc.
   
2. Optimization processes are tracked in execution order:
   - substage['optimization_process'] is a list
   - Each process has 'type' ('clock_tree_synthesis', 'repair_timing', etc.)
   - All details are nested within each process
   
3. Example access patterns:
   - Get final WNS: metrics['substages']['4_1_cts']['wns']
   - Get HPWL: metrics['substages']['4_1_cts']['hpwl']
   - Get clock sink wirelength: metrics['substages']['4_1_cts']['clock_sink_wirelength_um']
   - Get optimization history: metrics['substages']['4_1_cts']['optimization_process']
   - Get repair timing iterations: metrics['substages']['4_1_cts']['optimization_process'][1]['iterations_count']
   - Get start/end metrics: opt_process['start_metrics'] and opt_process['end_metrics']
   - Get summary: metrics['summary']

4. Important distinction:
   - 'clock_sink_wirelength_um' is the CLOCK TREE sink wirelength
   - This is NOT the routed wirelength (which is 0 before routing stage)
   - Routed wirelength will be extracted from the routing stage

5. Simplified optimization tracking:
   - Only tracks iteration count, start metrics, and end metrics
   - No longer stores all intermediate iterations (more concise)
   - Focus on: How many iterations? Where did we start? Where did we end?
""")

