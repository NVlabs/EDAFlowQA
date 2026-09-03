#!/bin/bash
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


# Script to run extraction and QA generation for all designs under ./design_artifacts
# This script processes each design's log file and generates QA pairs
#
# Usage:
#   ./run_qa_gen.sh                                    # Run for all designs
#   ./run_qa_gen.sh --dry-run                          # Show commands without executing them
#   ./run_qa_gen.sh --designs design1,design2          # Run for specific designs only
#   ./run_qa_gen.sh --output-dir my_qa                 # Write QA output to my_qa/ (default: qa_output)
#   ./run_qa_gen.sh --designs design1 --dry-run        # Combine options

set -e  # Exit on error

# Parse command line arguments
DRY_RUN=false
SPECIFIC_DESIGNS=""
OUTPUT_DIR="qa_output"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run|-n)
            DRY_RUN=true
            shift
            ;;
        --designs|-d)
            SPECIFIC_DESIGNS="$2"
            shift 2
            ;;
        --output-dir|-o)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run] [--designs design1,design2,...] [--output-dir DIR]"
            exit 1
            ;;
    esac
done

if [ "$DRY_RUN" = true ]; then
    echo "========================================"
    echo "DRY RUN MODE - Commands will be shown but not executed"
    echo "========================================"
    echo ""
fi

if [ -n "$SPECIFIC_DESIGNS" ]; then
    echo "========================================"
    echo "Processing specific designs only:"
    echo "$SPECIFIC_DESIGNS"
    echo "========================================"
    echo ""
fi

# Create base output directories if they don't exist
if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] mkdir -p \"$OUTPUT_DIR\""
else
    mkdir -p "$OUTPUT_DIR"
fi

echo "========================================"
echo "Starting extraction and QA generation"
echo "========================================"
echo ""

# Function to check if a design should be processed
should_process_design() {
    local design=$1
    
    # If no specific designs specified, process all
    if [ -z "$SPECIFIC_DESIGNS" ]; then
        return 0
    fi
    
    # Check if the design is in the comma-separated list
    IFS=',' read -ra DESIGNS_ARRAY <<< "$SPECIFIC_DESIGNS"
    for specific_design in "${DESIGNS_ARRAY[@]}"; do
        # Trim whitespace
        specific_design=$(echo "$specific_design" | xargs)
        if [ "$design" == "$specific_design" ]; then
            return 0
        fi
    done
    
    return 1
}

# Find all .txt log files under design_artifacts
for logfile in ./design_artifacts/*/*.txt; do
    if [ -f "$logfile" ]; then
        # Extract design name from path (e.g., aes_asap7 from ./design_artifacts/aes_asap7/aes_asap7.txt)
        design_dir=$(basename $(dirname "$logfile"))
        design_name=$(basename "$logfile" .txt)
        
        # Check if this design should be processed
        if ! should_process_design "$design_name"; then
            echo "Skipping: $design_name (not in specified designs list)"
            continue
        fi
        
        echo "----------------------------------------"
        echo "Processing: $design_name"
        echo "Log file: $logfile"
        echo "----------------------------------------"
        
        # Step 1: Extract metrics from log file
        echo "Step 1: Extracting metrics..."
        output_dir="test_output_${design_name}"
        
        if [ "$DRY_RUN" = true ]; then
            echo "[DRY RUN] mkdir -p \"$output_dir\""
            echo "[DRY RUN] python src/extract_all_stages.py \"$logfile\" -o \"$output_dir\""
        else
            mkdir -p "$output_dir"
            python src/extract_all_stages.py "$logfile" -o "$output_dir"
        fi
        
        if [ $? -eq 0 ] || [ "$DRY_RUN" = true ]; then
            echo "✓ Extraction completed for $design_name"
        else
            echo "✗ Extraction failed for $design_name"
            continue
        fi
        
        # Step 2: Generate QA pairs from extracted metrics
        metrics_file="${output_dir}/${design_name}_all_stages_metrics.json"
        
        if [ -f "$metrics_file" ] || [ "$DRY_RUN" = true ]; then
            echo "Step 2: Generating QA pairs..."
            qa_output_dir="./${OUTPUT_DIR}/${design_name}"
            
            if [ "$DRY_RUN" = true ]; then
                echo "[DRY RUN] mkdir -p \"$qa_output_dir\""
                echo "[DRY RUN] python src/qa_generation.py -json \"$metrics_file\" -n 10 -o \"$qa_output_dir\" --mode all"
            else
                mkdir -p "$qa_output_dir"
                python src/qa_generation.py -json "$metrics_file" -n 10 -o "$qa_output_dir" --mode all
            fi
            
            if [ $? -eq 0 ] || [ "$DRY_RUN" = true ]; then
                echo "✓ QA generation completed for $design_name"
                echo "  Output: $qa_output_dir"
            else
                echo "✗ QA generation failed for $design_name"
            fi
        else
            echo "✗ Metrics file not found: $metrics_file"
        fi
        
        echo ""
    fi
done

echo "========================================"
echo "All designs processed!"
echo "========================================"

