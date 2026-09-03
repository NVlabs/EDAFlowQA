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

"""Quick test script for CTS metrics parser"""
from cts_metrics_parser import extract_cts_metrics, print_metrics_summary
import json

# Test with ariane log
print("Testing with ariane133_nan45 CTS log...")
print("=" * 80)
with open('test_extraction_filtered/ariane133_nan45_cts_extract.txt', 'r') as f:
    log_content = f.read()

metrics = extract_cts_metrics(log_content)

# Print what substages were detected
print(f"\nDetected substages: {list(metrics['substages'].keys())}")

# Print metrics summary
print_metrics_summary(metrics)

# Save to JSON
with open('cts_metrics_ariane.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"\nMetrics saved to cts_metrics_ariane.json")

print("\n\n")
print("#" * 80)
print("Testing with gcdshy130 CTS log...")
print("#" * 80)

# Test with gcdshy130 log
with open('test_extraction_filtered/gcdshy130_cts_extract.txt', 'r') as f:
    log_content2 = f.read()

metrics2 = extract_cts_metrics(log_content2)

# Print what substages were detected
print(f"\nDetected substages: {list(metrics2['substages'].keys())}")

# Print metrics summary
print_metrics_summary(metrics2)

# Save to JSON
with open('cts_metrics_gcdshy130.json', 'w') as f:
    json.dump(metrics2, f, indent=2)
print(f"\nMetrics saved to cts_metrics_gcdshy130.json")


