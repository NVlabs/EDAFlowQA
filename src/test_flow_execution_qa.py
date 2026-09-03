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
Test the flow_execution_qa_pair function to generate questions from flow_execution_summary.
"""

import json
from pathlib import Path
from qa_generation import flow_execution_qa_pair


def test_flow_execution_qa():
    """Test generating flow execution QA pairs."""
    
    # Path to the sample metrics file
    metrics_file = Path("extracted_logs/ariane133_nan45_all_stages_metrics.json")
    
    if not metrics_file.exists():
        print(f"Error: {metrics_file} not found")
        return
    
    print("="*80)
    print("Testing flow_execution_qa_pair function")
    print("="*80)
    
    try:
        # Generate flow execution QA pairs
        qa_result = flow_execution_qa_pair(
            metrics_file,
            seed=42,
            temperature=0.7
        )
        
        # Set an ID for the result
        qa_result["id"] = "Q_FLOW_001"
        
        # Display the result
        print("\n" + "="*80)
        print("RESULT")
        print("="*80)
        print(f"\nGenerated {len(qa_result['qa_pairs'])} QA pairs for flow execution")
        print(f"Design: {qa_result['design_name']}")
        print(f"Difficulty: {qa_result['difficulty']}")
        print(f"Stage: {qa_result['stage']}")
        
        print("\nQuestions and Answers:")
        print("-"*80)
        for i, qa_pair in enumerate(qa_result['qa_pairs'], 1):
            print(f"\nQ{i}: {qa_pair['question']}")
            answer = qa_pair['answer']
            if isinstance(answer, dict):
                print(f"A{i}: (dict with {len(answer)} keys)")
                for key, value in answer.items():
                    print(f"  - {key}: {value}")
            elif isinstance(answer, list):
                print(f"A{i}: (list with {len(answer)} items)")
                for item in answer:
                    print(f"  - {item}")
            else:
                print(f"A{i}: {answer}")
        
        # Save to file
        output_file = Path("extracted_logs/ariane133_nan45_flow_execution_qa_test.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(qa_result, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*80}")
        print(f"Saved result to: {output_file}")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_flow_execution_qa()
