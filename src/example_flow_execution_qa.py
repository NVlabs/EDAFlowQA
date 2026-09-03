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
Example script demonstrating how to generate both flow execution and stage-level QA pairs.
"""

import json
from pathlib import Path
from qa_generation import flow_execution_qa_pair, single_stage_qa_pair, generate_qa_pairs


def example1_standalone_flow_execution():
    """Example 1: Generate flow execution QA pairs standalone."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Standalone Flow Execution QA Generation")
    print("="*80 + "\n")
    
    metrics_file = Path("extracted_logs/ariane133_nan45_all_stages_metrics.json")
    
    if not metrics_file.exists():
        print(f"Error: {metrics_file} not found")
        return
    
    # Generate flow execution QA pairs
    qa_result = flow_execution_qa_pair(
        metrics_file,
        seed=42,
        temperature=0.7
    )
    
    qa_result["id"] = "Q_FLOW_001"
    
    # Display results
    print(f"Generated {len(qa_result['qa_pairs'])} QA pairs for flow execution\n")
    for i, qa_pair in enumerate(qa_result['qa_pairs'], 1):
        print(f"Q{i}: {qa_pair['question']}")
        print(f"A{i}: {qa_pair['answer']}\n")


def example2_standalone_stage():
    """Example 2: Generate stage-level QA pairs standalone."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Standalone Stage-Level QA Generation")
    print("="*80 + "\n")
    
    metrics_file = Path("extracted_logs/ariane133_nan45_all_stages_metrics.json")
    
    if not metrics_file.exists():
        print(f"Error: {metrics_file} not found")
        return
    
    # Generate stage-level QA pairs
    qa_result = single_stage_qa_pair(
        metrics_file,
        stage="synthesis",  # Constrain to synthesis stage
        seed=42,
        temperature=0.7
    )
    
    qa_result["id"] = "Q_STG_001"
    
    # Display results
    print(f"Generated {len(qa_result['qa_pairs'])} QA pairs for {qa_result['stage']} stage\n")
    for i, qa_pair in enumerate(qa_result['qa_pairs'], 1):
        print(f"Q{i}: {qa_pair['question']}")
        print(f"A{i}: {qa_pair['answer']}\n")


def example3_integrated():
    """Example 3: Generate both flow execution and stage-level QA pairs together."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Integrated QA Generation (Flow + Stages)")
    print("="*80 + "\n")
    
    metrics_file = Path("extracted_logs/ariane133_nan45_all_stages_metrics.json")
    
    if not metrics_file.exists():
        print(f"Error: {metrics_file} not found")
        return
    
    # Generate both types of QA pairs
    result = generate_qa_pairs(
        metrics_file,
        n=3,  # Generate 3 stage-level QA pairs
        include_flow_execution=True,  # Also generate flow execution QA pair
        seed=42,
        temperature=0.7
    )
    
    # Display summary
    print(f"\n{'='*80}")
    print(f"Generated {len(result['questions'])} QA sets for {result['design_name']}")
    print(f"{'='*80}\n")
    
    for q in result['questions']:
        print(f"ID: {q['id']}")
        print(f"Stage: {q['stage']}")
        print(f"Difficulty: {q['difficulty']}")
        print(f"QA pairs: {len(q['qa_pairs'])}")
        print("-"*40)
        for i, qa_pair in enumerate(q['qa_pairs'], 1):
            print(f"  Q{i}: {qa_pair['question'][:60]}...")
        print()
    
    # Save to file
    output_file = Path("extracted_logs/ariane133_nan45_example_qa.json")
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Saved to: {output_file}")


def example4_custom_combination():
    """Example 4: Custom combination - manually combine different QA types."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Custom Combination")
    print("="*80 + "\n")
    
    metrics_file = Path("extracted_logs/ariane133_nan45_all_stages_metrics.json")
    
    if not metrics_file.exists():
        print(f"Error: {metrics_file} not found")
        return
    
    questions = []
    
    # 1. Generate flow execution QA
    try:
        flow_qa = flow_execution_qa_pair(metrics_file, seed=42)
        flow_qa["id"] = "Q_FLOW_001"
        questions.append(flow_qa)
        print(f"✓ Generated flow execution QA: {len(flow_qa['qa_pairs'])} pairs")
    except Exception as e:
        print(f"✗ Failed to generate flow execution QA: {e}")
    
    # 2. Generate stage-specific QA for different stages
    stage_types = ["synthesis", "placement", "cts"]
    for i, stage_type in enumerate(stage_types, 1):
        try:
            stage_qa = single_stage_qa_pair(
                metrics_file,
                stage=stage_type,
                seed=42 + i,
                temperature=0.7
            )
            stage_qa["id"] = f"Q{i:03d}"
            questions.append(stage_qa)
            print(f"✓ Generated {stage_type} QA: {len(stage_qa['qa_pairs'])} pairs")
        except Exception as e:
            print(f"✗ Failed to generate {stage_type} QA: {e}")
    
    # 3. Create benchmark structure
    benchmark = {
        "design_name": "ariane133",
        "total_questions": len(questions),
        "questions": questions
    }
    
    # Save to file
    output_file = Path("extracted_logs/ariane133_nan45_custom_qa.json")
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(benchmark, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"Generated {len(questions)} QA sets")
    print(f"Saved to: {output_file}")
    print(f"{'='*80}")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("FLOW EXECUTION QA GENERATION EXAMPLES")
    print("="*80)
    
    try:
        # Run examples
        example1_standalone_flow_execution()
        example2_standalone_stage()
        example3_integrated()
        example4_custom_combination()
        
        print("\n" + "="*80)
        print("All examples completed!")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
