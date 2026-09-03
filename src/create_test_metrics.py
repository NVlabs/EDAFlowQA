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
Helper script to create a sample metrics JSON file for testing QA generation.
This creates a realistic test file that can be used with test_qa_generation.py or qa_generation.py
"""

import json
import sys
from pathlib import Path


def create_sample_metrics_json():
    """Create a sample metrics JSON file for testing."""
    sample_data = {
        "design_name": "test_design",
        "stages": [
            {
                "stage": "synthesis",
                "occurrence": 1,
                "design_name": "test_design",
                "sub_design_name": "test_sub",
                "metrics_file": "test_metrics.json",
                "metrics": {
                    "area": {"total": 1000.5, "unit": "um^2"},
                    "timing": {
                        "wns": -0.5,
                        "tns": -10.2,
                        "whs": 0.0,
                        "ths": 0.0
                    },
                    "power": {
                        "total": 50.0,
                        "dynamic": 40.0,
                        "leakage": 10.0,
                        "unit": "mW"
                    },
                    "violations": {
                        "drv": 5,
                        "short": 0,
                        "spacing": 2
                    }
                }
            },
            {
                "stage": "floorplan",
                "occurrence": 1,
                "design_name": "test_design",
                "sub_design_name": "test_sub",
                "metrics_file": "test_metrics_fp.json",
                "metrics": {
                    "area": {"total": 1200.8, "unit": "um^2"},
                    "utilization": 0.65,
                    "core_area": 2000.0,
                    "die_area": 2500.0
                }
            },
            {
                "stage": "placement",
                "occurrence": 1,
                "design_name": "test_design",
                "sub_design_name": "test_sub",
                "metrics_file": "test_metrics_place.json",
                "metrics": {
                    "timing": {
                        "wns": -0.3,
                        "tns": -5.1
                    },
                    "density": 0.70,
                    "wirelength": 50000
                }
            },
            {
                "stage": "cts",
                "occurrence": 1,
                "design_name": "test_design",
                "sub_design_name": "test_sub",
                "metrics_file": "test_metrics_cts.json",
                "metrics": {
                    "timing": {
                        "wns": -0.8,
                        "tns": -15.3
                    },
                    "clock_skew": 0.05,
                    "total_buffer_count": 120
                }
            },
            {
                "stage": "route",
                "occurrence": 1,
                "design_name": "test_design",
                "sub_design_name": "test_sub",
                "metrics_file": "test_metrics_route.json",
                "metrics": {
                    "timing": {
                        "wns": -0.2,
                        "tns": -3.5
                    },
                    "violations": {
                        "drv": 0,
                        "short": 1,
                        "spacing": 0
                    },
                    "wirelength": 55000,
                    "via_count": 8500
                }
            }
        ]
    }
    return sample_data


def main():
    # Get output path from command line or use default
    if len(sys.argv) > 1:
        output_path = Path(sys.argv[1])
    else:
        output_path = Path("test_design_all_stages_metrics.json")
    
    # Create sample data
    sample_data = create_sample_metrics_json()
    
    # Write to file
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Created sample metrics file: {output_path.absolute()}")
    print(f"\n📊 File contains:")
    print(f"   - Design: {sample_data['design_name']}")
    print(f"   - Stages: {len(sample_data['stages'])}")
    for stage in sample_data['stages']:
        print(f"     • {stage['stage']}")
    
    print(f"\n💡 Usage examples:")
    print(f"   1. Test QA generation with this file:")
    print(f"      python test_qa_generation.py --real {output_path}")
    print(f"\n   2. Generate QA pairs directly:")
    print(f"      python qa_generation.py {output_path} -n 5")
    print(f"\n   3. Generate with custom settings:")
    print(f"      python qa_generation.py {output_path} -n 10 --temperature 0.7 --seed 42")


if __name__ == "__main__":
    main()

