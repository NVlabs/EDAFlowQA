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


# Script to run pass_at_k evaluation for single-design QA files
# This script processes QA JSON files under ./qa_release/<design_name>
# and runs the test evaluation using the specified model
#
# Usage:
#   ./run_pass_at_k_single.sh                                    # Run all designs with default model and cursor agent
#   ./run_pass_at_k_single.sh --designs "ibex_sky130hd aes_asap7"  # Run specific designs
#   ./run_pass_at_k_single.sh --model gpt4                       # Run all designs with gpt4
#   ./run_pass_at_k_single.sh --designs "ibex_sky130hd" --model gpt4  # Run specific design with gpt4
#   ./run_pass_at_k_single.sh --agent-type claude_code           # Use claude_code agent
#   ./run_pass_at_k_single.sh --dry-run                          # Show commands without executing
#   ./run_pass_at_k_single.sh --designs "ibex_sky130hd" --output-base ./results  # Custom base output dir
#
# Output directories:
#   Results will be saved to ./qa_output_<agent_type>_<design_name>/ for each design

# Default values
MODEL="${MODEL:-claude-4.5-sonnet}"
AGENT_TYPE="cursor"  # Agent type: cursor or claude_code
OUTPUT_BASE_DIR="."  # Base directory, actual output will be ./qa_output_<agent_type>_<design_name>
DRY_RUN=false
DESIGNS=()  # Empty means all designs

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --agent-type)
            AGENT_TYPE="$2"
            shift 2
            ;;
        --designs)
            # Read space-separated design names
            IFS=' ' read -ra DESIGNS <<< "$2"
            shift 2
            ;;
        --output-base)
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
            echo "  --designs \"design1 design2\"    Specify which designs to process (space-separated)"
            echo "  --model MODEL                  Model to use (default: sonnet4p5)"
            echo "  --agent-type TYPE              Agent type: cursor or claude_code (default: cursor)"
            echo "  --output-base DIR              Base directory for outputs (default: .)"
            echo "  --dry-run, -n                  Show commands without executing"
            echo "  --help, -h                     Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                              # Run all designs with cursor agent"
            echo "  $0 --designs \"ibex_sky130hd aes_asap7\"         # Run specific designs"
            echo "  $0 --designs \"ibex_sky130hd\" --model gpt4      # Run one design with gpt4"
            echo "  $0 --agent-type claude_code                     # Use claude_code agent"
            echo "  $0 --output-base ./results                      # Save to ./results/qa_output_<agent_type>_<design>/"
            echo ""
            echo "Output:"
            echo "  Results are saved to <output-base>/qa_output_<agent_type>_<design_name>/"
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
echo "Starting pass_at_k evaluation for single designs"
echo "Model: $MODEL"
echo "Agent type: $AGENT_TYPE"
echo "Output base directory: $OUTPUT_BASE_DIR"
if [ ${#DESIGNS[@]} -gt 0 ]; then
    echo "Selected designs: ${DESIGNS[*]}"
else
    echo "Processing: All available designs"
fi
echo "========================================"
echo ""

# Counter for tracking progress
total_files=0
processed_files=0
failed_files=0

# Determine which designs to process
if [ ${#DESIGNS[@]} -gt 0 ]; then
    # Use specified designs
    design_dirs=()
    for design in "${DESIGNS[@]}"; do
        design_path="./qa_release/$design"
        if [ -d "$design_path" ]; then
            design_dirs+=("$design_path")
        else
            echo "⚠️  Warning: Design directory not found: $design_path"
        fi
    done
else
    # Find all design directories (exclude multifile_comparison)
    mapfile -t design_dirs < <(find ./qa_release -maxdepth 1 -type d | grep -v "multifile_comparison" | grep -v "^\./qa_release$" | sort)
fi

# For each design directory, find the latest QA file
declare -a qa_files
declare -a design_names
for design_dir in "${design_dirs[@]}"; do
    # Get the latest (most recent timestamp) non-raw QA file
    latest_file=$(find "$design_dir" -maxdepth 1 -name "*_qa_all_types.json" -type f | grep -v "_raw\.json" | sort -r | head -1)
    if [ -n "$latest_file" ]; then
        qa_files+=("$latest_file")
        design_names+=("$(basename "$design_dir")")
    fi
done

total_files=${#qa_files[@]}

echo "Found $total_files QA files to process"
echo ""

if [ $total_files -eq 0 ]; then
    echo "No QA files found."
    if [ ${#DESIGNS[@]} -gt 0 ]; then
        echo "Requested designs: ${DESIGNS[*]}"
    fi
    echo "Exiting."
    exit 0
fi

# Process each QA file
for i in "${!qa_files[@]}"; do
    qa_file="${qa_files[$i]}"
    design_name="${design_names[$i]}"
    
    if [ -f "$qa_file" ]; then
        qa_filename=$(basename "$qa_file")
        
        # Create design-specific output directory
        OUTPUT_DIR="${OUTPUT_BASE_DIR}/qa_output_${AGENT_TYPE}_${design_name}"
        
        echo "----------------------------------------"
        echo "[$((processed_files + 1))/$total_files] Processing: $design_name"
        echo "QA file: $qa_file"
        echo "Output directory: $OUTPUT_DIR"
        echo "----------------------------------------"
        
        # Create output directory
        if [ "$DRY_RUN" = true ]; then
            echo "[DRY RUN] mkdir -p \"$OUTPUT_DIR\""
        else
            mkdir -p "$OUTPUT_DIR"
        fi
        
        # Run test evaluation
        if [ "$DRY_RUN" = true ]; then
            echo "[DRY RUN] python src/test_llm_qa.py pass_at_k \\"
            echo "  --qa_file \"$qa_file\" \\"
            echo "  --model \"$MODEL\" \\"
            echo "  --agent_type \"$AGENT_TYPE\" \\"
            echo "  --output_dir \"$OUTPUT_DIR\""
        else
            python src/test_llm_qa.py pass_at_k \
                --qa_file "$qa_file" \
                --model "$MODEL" \
                --agent_type "$AGENT_TYPE" \
                --output_dir "$OUTPUT_DIR"
        fi
        
        if [ $? -eq 0 ] || [ "$DRY_RUN" = true ]; then
            echo "✓ Evaluation completed for $design_name"
            echo "  Results saved to: $OUTPUT_DIR"
            processed_files=$((processed_files + 1))
        else
            echo "✗ Evaluation failed for $design_name"
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
echo "- Agent type: $AGENT_TYPE"
echo "- Output base directory: $OUTPUT_BASE_DIR"
echo "- Design-specific outputs: ${OUTPUT_BASE_DIR}/qa_output_${AGENT_TYPE}_<design_name>/"
echo ""

if [ $processed_files -gt 0 ]; then
    echo "Output directories created:"
    for design_name in "${design_names[@]}"; do
        echo "  - ${OUTPUT_BASE_DIR}/qa_output_${AGENT_TYPE}_${design_name}/"
    done
    echo ""
fi
