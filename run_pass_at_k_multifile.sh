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


# Script to run pass_at_k evaluation for all multifile comparison QA files
# This script processes each QA JSON file under ./qa_release/multifile_comparison
# and runs the pass_at_k evaluation using the specified model
#
# Usage:
#   ./run_pass_at_k_multifile.sh                                    # Run all comparisons with default model and cursor agent
#   ./run_pass_at_k_multifile.sh --designs "aes_asap7_vs_aes_sky130hd aes_nangate45_vs_aes_sky130hd"  # Run specific comparisons
#   ./run_pass_at_k_multifile.sh --model gpt4                       # Run with different model
#   ./run_pass_at_k_multifile.sh --agent-type claude_code           # Run with claude code agent
#   ./run_pass_at_k_multifile.sh --designs "aes_asap7_vs_aes_sky130hd" --model gpt4
#   ./run_pass_at_k_multifile.sh --output-base ./results            # Custom base output dir
#   ./run_pass_at_k_multifile.sh --dry-run                          # Show commands without executing them
#
# Output directories:
#   Results will be saved to <output-base>/qa_responses_<agent_type>_<comparison_name>/ for each comparison

# Default values
MODEL="${MODEL:-claude-4.5-sonnet}"
AGENT_TYPE="cursor"
OUTPUT_BASE_DIR="."
DRY_RUN=false
DESIGNS=()  # Empty means all comparisons (design pairs)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --agent-type|--agent_type)
            AGENT_TYPE="$2"
            shift 2
            ;;
        --designs)
            # Read space-separated comparison names (e.g. aes_asap7_vs_aes_sky130hd)
            IFS=' ' read -ra DESIGNS <<< "$2"
            shift 2
            ;;
        --output-base)
            OUTPUT_BASE_DIR="$2"
            shift 2
            ;;
        --output_dir)
            # Backward compatibility: treat as output-base
            OUTPUT_BASE_DIR="$2"
            shift 2
            ;;
        --dry-run|-n)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --designs \"comp1 comp2\"         Specify which comparisons to process (space-separated)"
            echo "                                  e.g. \"aes_asap7_vs_aes_sky130hd aes_nangate45_vs_aes_sky130hd\""
            echo "  --model MODEL                   Model to use (default: sonnet4p5)"
            echo "  --agent-type TYPE              Agent type: cursor or claude_code (default: cursor)"
            echo "  --output-base DIR               Base directory for outputs (default: .)"
            echo "  --dry-run, -n                   Show commands without executing"
            echo "  --help, -h                      Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                              # Run all comparisons with cursor agent"
            echo "  $0 --designs \"aes_asap7_vs_aes_sky130hd\"       # Run one comparison"
            echo "  $0 --designs \"aes_asap7_vs_aes_sky130hd aes_nangate45_vs_aes_sky130hd\" --model gpt4"
            echo "  $0 --agent-type claude_code                     # Use claude_code agent"
            echo "  $0 --output-base ./results                       # Save to ./results/qa_responses_<agent_type>_<comparison>/"
            echo ""
            echo "Output:"
            echo "  Results are saved to <output-base>/qa_responses_<agent_type>_<comparison_name>/"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
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

echo "========================================"
echo "Starting pass_at_k evaluation for multifile comparisons"
echo "Model: $MODEL"
echo "Agent type: $AGENT_TYPE"
echo "Output base directory: $OUTPUT_BASE_DIR"
if [ ${#DESIGNS[@]} -gt 0 ]; then
    echo "Selected comparisons: ${DESIGNS[*]}"
else
    echo "Processing: All available comparisons"
fi
echo "========================================"
echo ""

# Counter for tracking progress
total_files=0
processed_files=0
failed_files=0

# Find all QA JSON files under multifile_comparison (exclude *_raw.json files)
mapfile -t all_qa_files < <(find ./qa_release/multifile_comparison -name "*_qa_all_multifile.json" -type f | grep -v "_raw\.json" | sort)

# If specific designs (comparisons) requested, filter to only those
qa_files=()
if [ ${#DESIGNS[@]} -gt 0 ]; then
    for qa_file in "${all_qa_files[@]}"; do
        comparison_dir=$(basename "$(dirname "$qa_file")")
        for design in "${DESIGNS[@]}"; do
            if [ "$comparison_dir" = "$design" ]; then
                qa_files+=("$qa_file")
                break
            fi
        done
    done
else
    qa_files=("${all_qa_files[@]}")
fi

total_files=${#qa_files[@]}

echo "Found $total_files QA files to process"
echo ""

if [ $total_files -eq 0 ]; then
    echo "No QA files found."
    if [ ${#DESIGNS[@]} -gt 0 ]; then
        echo "Requested comparisons: ${DESIGNS[*]}"
    else
        echo "Searched in: ./qa_release/multifile_comparison/"
    fi
    echo "Exiting."
    exit 0
fi

# Process each QA file
for qa_file in "${qa_files[@]}"; do
    if [ -f "$qa_file" ]; then
        # Extract design comparison name from path
        comparison_dir=$(basename $(dirname "$qa_file"))
        qa_filename=$(basename "$qa_file")
        # Output directory: <output-base>/qa_responses_<agent_type>_<comparison_name>
        #   e.g. ./qa_responses_cursor_aes_asap7_vs_aes_sky130hd
        #   or   ./results/qa_responses_cursor_aes_asap7_vs_aes_sky130hd
        per_comparison_output_dir="${OUTPUT_BASE_DIR}/qa_responses_${AGENT_TYPE}_${comparison_dir}"
        
        echo "----------------------------------------"
        echo "[$((processed_files + 1))/$total_files] Processing: $comparison_dir"
        echo "QA file: $qa_file"
        echo "Output directory: $per_comparison_output_dir"
        echo "----------------------------------------"
        
        # Create per-comparison output directory
        if [ "$DRY_RUN" = true ]; then
            echo "[DRY RUN] mkdir -p \"$per_comparison_output_dir\""
        else
            mkdir -p "$per_comparison_output_dir"
        fi

        # Run pass_at_k evaluation
        if [ "$DRY_RUN" = true ]; then
            echo "[DRY RUN] python src/test_llm_qa_multifile.py pass_at_k \\"
            echo "  --qa_file \"$qa_file\" \\"
            echo "  --model \"$MODEL\" \\"
            echo "  --agent_type \"$AGENT_TYPE\" \\"
            echo "  --output_dir \"$per_comparison_output_dir\""
        else
            python src/test_llm_qa_multifile.py pass_at_k \
                --qa_file "$qa_file" \
                --model "$MODEL" \
                --agent_type "$AGENT_TYPE" \
                --output_dir "$per_comparison_output_dir"
        fi
        
        if [ $? -eq 0 ] || [ "$DRY_RUN" = true ]; then
            echo "✓ Evaluation completed for $comparison_dir"
            processed_files=$((processed_files + 1))
        else
            echo "✗ Evaluation failed for $comparison_dir"
            failed_files=$((failed_files + 1))
        fi
        
        echo ""
    fi
done

echo "========================================"
echo "Pass@k Evaluation Complete!"
echo "========================================"
echo ""
echo "Summary:"
echo "- Total files found: $total_files"
echo "- Successfully processed: $processed_files"
echo "- Failed: $failed_files"
echo "- Model: $MODEL"
echo "- Agent type: $AGENT_TYPE"
echo "- Output base directory: $OUTPUT_BASE_DIR"
echo "- Output pattern: ${OUTPUT_BASE_DIR}/qa_responses_${AGENT_TYPE}_<comparison_name>/"
echo ""
