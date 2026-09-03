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
#   ./run_qa_multifiles_gen.sh                        # Run normally
#   ./run_qa_multifiles_gen.sh --dry-run              # Show commands without executing them
#   ./run_qa_multifiles_gen.sh --output-dir my_qa     # Write QA output to my_qa/ (default: qa_output)

set -e  # Exit on error

# Parse command line arguments
DRY_RUN=false
OUTPUT_DIR="qa_output"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run|-n)
            DRY_RUN=true
            shift
            ;;
        --output-dir|-o)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run] [--output-dir DIR]"
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

# Find all .txt log files under design_artifacts
for logfile in ./design_artifacts/*/*.txt; do
    if [ -f "$logfile" ]; then
        # Extract design name from path (e.g., aes_asap7 from ./design_artifacts/aes_asap7/aes_asap7.txt)
        design_dir=$(basename $(dirname "$logfile"))
        design_name=$(basename "$logfile" .txt)
        
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
                
        echo ""
    fi
done

echo "========================================"
echo "All designs processed!"
echo "========================================"
echo ""
echo "Summary of outputs:"
echo "- Extracted metrics: test_output_*/"
echo "- QA pairs by design: ${OUTPUT_DIR}/<design_name>/"
echo "- Pairwise comparison QA: ${OUTPUT_DIR}/multifile_comparison/<design1>_vs_<design2>/"
echo ""

# Optional: Generate multifile QA if multiple designs were processed
echo "----------------------------------------"
echo "Generating multifile QA (pairwise comparisons)"
echo "----------------------------------------"

# Collect metrics files from all successfully extracted designs
declare -a valid_metrics_files
for metrics_file in ./test_output_*/*_all_stages_metrics.json; do
    if [ -f "$metrics_file" ]; then
        valid_metrics_files+=("$metrics_file")
    fi
done

echo "Found ${#valid_metrics_files[@]} design metrics files for pairwise comparison"

if [ ${#valid_metrics_files[@]} -ge 2 ]; then
    # Generate pairwise comparisons
    for ((i=0; i<${#valid_metrics_files[@]}; i++)); do
        for ((j=i+1; j<${#valid_metrics_files[@]}; j++)); do
            file1="${valid_metrics_files[$i]}"
            file2="${valid_metrics_files[$j]}"
            
            # Extract design names from file paths
            design1=$(basename "$file1" "_all_stages_metrics.json")
            design2=$(basename "$file2" "_all_stages_metrics.json")
            
            echo ""
            echo "Comparing: $design1 vs $design2"
            
            multifile_qa_dir="./${OUTPUT_DIR}/multifile_comparison/${design1}_vs_${design2}"
            
            if [ "$DRY_RUN" = true ]; then
                echo "[DRY RUN] mkdir -p \"$multifile_qa_dir\""
                echo "[DRY RUN] python src/qa_generation_multifiles.py --mode all \\"
                echo "  $file1 \\"
                echo "  $file2 \\"
                echo "  --n_questions 5 -o \"$multifile_qa_dir\""
            else
                mkdir -p "$multifile_qa_dir"
                python src/qa_generation_multifiles.py --mode all "$file1" "$file2" --n_questions 6 -o "$multifile_qa_dir"
            fi
            
            if [ $? -eq 0 ] || [ "$DRY_RUN" = true ]; then
                echo "✓ Pairwise QA generation completed: $design1 vs $design2"
                echo "  Output: $multifile_qa_dir"
            else
                echo "✗ Pairwise QA generation failed: $design1 vs $design2"
            fi
        done
    done
else
    echo "Skipping multifile QA (need at least 2 designs, found ${#valid_metrics_files[@]})"
fi

echo ""
echo "========================================"
echo "QA Generation Pipeline Complete!"
echo "========================================"
