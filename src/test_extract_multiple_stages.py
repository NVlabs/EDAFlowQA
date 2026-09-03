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
Test script to verify the refactored extract_all_stages with stage-based splitting.
"""

from extract_all_stages import LogExtractionPipeline
import os

def test_stage_splitting():
    """Test the stage splitting functionality."""
    
    # Find a sample log file in the current directory
    log_files = [f for f in os.listdir('.') if f.endswith('.txt') and 'extract' not in f and 'summary' not in f]
    
    if not log_files:
        print("No log files found in current directory")
        print("Please place a log file in the directory to test.")
        return
    
    test_file = log_files[0]
    print(f"Testing with: {test_file}")
    print("=" * 80)
    
    pipeline = LogExtractionPipeline(test_file, output_dir="test_extracted_logs")
    
    # Load the log
    if not pipeline.load_log_file():
        print("Failed to load log file")
        return
    
    print()
    print("Testing stage splitting...")
    print("-" * 80)
    
    # Split by stages
    stage_chunks = pipeline.split_log_by_stages()
    
    print(f"\nFound {len(stage_chunks)} stage chunk(s):\n")
    for i, chunk in enumerate(stage_chunks, 1):
        print(f"{i}. Stage {chunk['stage_number']} - {chunk['stage_type'].upper()}")
        print(f"   Occurrence: {chunk['occurrence']}")
        print(f"   Lines: {chunk['start_line']}-{chunk['end_line']} ({chunk['end_line']-chunk['start_line']} lines)")
        print(f"   Content preview: {chunk['content'][:100].replace(chr(10), ' ')[:80]}...")
        print()


def test_full_extraction():
    """Test full extraction pipeline."""
    
    # Find a sample log file
    log_files = [f for f in os.listdir('.') if f.endswith('.txt') and 'extract' not in f and 'summary' not in f]
    
    if not log_files:
        print("No log files found in current directory")
        return
    
    test_file = log_files[0]
    print(f"\n{'=' * 80}")
    print(f"FULL EXTRACTION TEST: {test_file}")
    print("=" * 80)
    
    pipeline = LogExtractionPipeline(test_file, output_dir="test_extracted_logs")
    results = pipeline.extract_all_stages()
    
    if results['success']:
        print("\n" + "=" * 80)
        print("EXTRACTION RESULTS:")
        print("=" * 80)
        
        # Show chronological order
        print("\nStages in chronological order:")
        for i, stage in enumerate(results['stages_chronological'], 1):
            print(f"  {i}. {stage['stage']:12s} #{stage['occurrence']:2d} - {stage['stage_name']:30s}")
            print(f"      Lines: {stage.get('start_line', 'N/A'):6d}-{stage.get('end_line', 'N/A'):6d}  |  Output: {os.path.basename(stage['output_file'])}")
        
        # Show by type
        print("\n" + "-" * 80)
        print("Stages grouped by type:")
        for stage_type, stages in results['stages_by_type'].items():
            print(f"\n  {stage_type.upper()}:")
            for stage in stages:
                print(f"    - Occurrence {stage['occurrence']}: {stage['line_count']:5d} lines -> {os.path.basename(stage['output_file'])}")
        
        # Generate and display summary
        print("\n" + "=" * 80)
        print(pipeline.generate_summary(results))
        
    else:
        print(f"\nX Extraction failed: {results.get('error', 'Unknown error')}")


if __name__ == "__main__":
    print("=" * 80)
    print("TESTING REFACTORED EXTRACT_ALL_STAGES")
    print("Strategy: Split by stage markers (1_*, 2_*, etc.) then extract")
    print("=" * 80)
    
    # Test 1: Stage splitting
    print("\nTEST 1: Stage Splitting")
    print("=" * 80)
    test_stage_splitting()
    
    # Test 2: Full extraction
    print("\n\nTEST 2: Full Extraction Pipeline")
    print("=" * 80)
    test_full_extraction()
    
    print("\n" + "=" * 80)
    print("Testing complete!")
    print("=" * 80)
