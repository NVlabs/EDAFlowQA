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
Example: Using Unified Log Structure for QA Benchmark Generation
==================================================================
This script demonstrates practical usage of the unified log structure
for creating QA benchmarks from OpenROAD design flow logs.
"""

from create_unified_benchmark import create_unified_benchmark
from unified_log_structure import (
    DesignData,
    query_metric,
    query_substage_info,
    query_warnings_by_code
)
import json


def example1_basic_extraction():
    """Example 1: Basic extraction and metric querying."""
    print("=" * 80)
    print("Example 1: Basic Extraction and Querying")
    print("=" * 80)
    
    # Extract all stages from log file
    design_data = create_unified_benchmark(
        log_file='gcdshy130.txt',
        output_json='example_output/gcdshy130_benchmark.json',
        output_dir='example_output/stages/',
        verbose=True
    )
    
    print("\n" + "=" * 80)
    print("Querying Metrics:")
    print("=" * 80)
    
    # Query key metrics from different stages
    metrics_to_query = [
        ('floorplan', 'area', 'Floorplan Area'),
        ('floorplan', 'utilization', 'Floorplan Utilization'),
        ('placement', 'final_area', 'Placement Area'),
        ('placement', 'final_utilization', 'Placement Utilization'),
        ('cts', 'clock_buffers', 'Clock Buffers'),
        ('cts', 'clock_nets', 'Clock Nets'),
        ('routing', 'total_wirelength', 'Total Wirelength'),
        ('routing', 'final_vias', 'Final Vias'),
    ]
    
    for stage, metric, label in metrics_to_query:
        value = query_metric(design_data, stage, metric)
        if value is not None:
            print(f"{label}: {value}")
        else:
            print(f"{label}: Not found")
    
    return design_data


def example2_performance_analysis(design_data):
    """Example 2: Analyze performance (time and memory) across stages."""
    print("\n" + "=" * 80)
    print("Example 2: Performance Analysis")
    print("=" * 80)
    
    performance = design_data.get_performance_summary()
    
    print(f"\n{'Stage':<25} {'Substage':<30} {'Time':<15} {'Memory (MB)':<12}")
    print("-" * 82)
    
    for stage_name, substages in performance.items():
        for i, (substage_name, perf) in enumerate(substages.items()):
            stage_col = stage_name if i == 0 else ""
            time_str = perf['elapsed_time'] or 'N/A'
            memory_str = str(perf['memory_mb']) if perf['memory_mb'] else 'N/A'
            
            print(f"{stage_col:<25} {substage_name:<30} {time_str:<15} {memory_str:<12}")
        
        if substages:  # Add separator between stages
            print("-" * 82)


def example3_warning_analysis(design_data):
    """Example 3: Analyze warnings across all stages."""
    print("\n" + "=" * 80)
    print("Example 3: Warning Analysis")
    print("=" * 80)
    
    all_warnings = design_data.get_all_warnings()
    
    total_warnings = 0
    for stage, warnings in all_warnings.items():
        stage_total = sum(warnings.values())
        if stage_total > 0:
            print(f"\n{stage.upper()} - Total: {stage_total}")
            for warning_code, count in sorted(warnings.items(), key=lambda x: x[1], reverse=True):
                print(f"  {warning_code}: {count}")
                total_warnings += count
    
    print(f"\nTotal warnings across all stages: {total_warnings}")
    
    # Find specific warning types
    print("\n" + "-" * 80)
    print("RSZ (Resizer) Warnings:")
    rsz_warnings = query_warnings_by_code(design_data, 'RSZ-')
    for location, count in rsz_warnings.items():
        print(f"  {location}: {count}")


def example4_generate_qa_benchmark(design_data):
    """Example 4: Generate QA benchmark pairs."""
    print("\n" + "=" * 80)
    print("Example 4: Generate QA Benchmark Pairs")
    print("=" * 80)
    
    # Generate QA pairs automatically
    qa_pairs = design_data.generate_metric_qa_pairs()
    
    print(f"\nGenerated {len(qa_pairs)} QA pairs")
    
    # Group by category
    by_category = {}
    for qa in qa_pairs:
        category = qa['category']
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(qa)
    
    print("\nQA Pairs by Category:")
    for category, pairs in by_category.items():
        print(f"  {category}: {len(pairs)}")
    
    # Show sample QA pairs
    print("\n" + "-" * 80)
    print("Sample QA Pairs:")
    print("-" * 80)
    
    for qa in qa_pairs[:10]:
        print(f"\nQ: {qa['question']}")
        print(f"A: {qa['answer']}")
        print(f"Stage: {qa['stage']}, Category: {qa['category']}")
    
    # Save to file
    output_file = 'example_output/qa_benchmark.json'
    with open(output_file, 'w') as f:
        json.dump(qa_pairs, f, indent=2)
    print(f"\n✓ QA pairs saved to: {output_file}")
    
    return qa_pairs


def example5_custom_qa_generation(design_data):
    """Example 5: Custom QA generation with domain logic."""
    print("\n" + "=" * 80)
    print("Example 5: Custom QA Generation")
    print("=" * 80)
    
    custom_qa = []
    
    # 1. Compare metrics across stages
    placement_area = query_metric(design_data, 'placement', 'final_area')
    cts_area = query_metric(design_data, 'cts', 'final_area')
    
    if placement_area and cts_area:
        area_change_pct = ((cts_area - placement_area) / placement_area) * 100
        custom_qa.append({
            'question': 'What is the percentage change in design area from placement to CTS?',
            'answer': f'{area_change_pct:.2f}%',
            'category': 'analysis',
            'type': 'comparison'
        })
    
    # 2. Buffer insertion questions
    clock_buffers = query_metric(design_data, 'cts', 'clock_buffers')
    if clock_buffers:
        custom_qa.append({
            'question': 'How many clock buffers were inserted during CTS?',
            'answer': str(clock_buffers),
            'category': 'cts',
            'type': 'factoid'
        })
    
    # 3. Utilization questions
    final_util = query_metric(design_data, 'final', 'final_utilization')
    if final_util:
        custom_qa.append({
            'question': 'What is the final design utilization?',
            'answer': f'{final_util}%',
            'category': 'final',
            'type': 'factoid'
        })
    
    # 4. Routing complexity
    total_wirelength = query_metric(design_data, 'routing', 'total_wirelength')
    final_vias = query_metric(design_data, 'routing', 'final_vias')
    
    if total_wirelength and final_vias:
        custom_qa.append({
            'question': f'If the total wirelength is {total_wirelength} um and there are {final_vias} vias, what is the average wirelength per via?',
            'answer': f'{total_wirelength / final_vias:.2f} um',
            'category': 'routing',
            'type': 'calculation'
        })
    
    # 5. Stage completion questions
    for stage_name in ['synthesis', 'floorplan', 'placement', 'cts', 'routing', 'final']:
        stage = design_data.get_stage(stage_name)
        if stage and stage.substages:
            num_substages = len(stage.substages)
            custom_qa.append({
                'question': f'How many substages are in the {stage_name} stage?',
                'answer': str(num_substages),
                'category': 'structure',
                'type': 'count'
            })
    
    print(f"\nGenerated {len(custom_qa)} custom QA pairs")
    print("\nCustom QA Examples:")
    for qa in custom_qa:
        print(f"\nQ: {qa['question']}")
        print(f"A: {qa['answer']}")
        print(f"Category: {qa['category']}, Type: {qa['type']}")
    
    # Save custom QA
    output_file = 'example_output/custom_qa_benchmark.json'
    with open(output_file, 'w') as f:
        json.dump(custom_qa, f, indent=2)
    print(f"\n✓ Custom QA pairs saved to: {output_file}")


def example6_summary_report(design_data):
    """Example 6: Generate and display summary report."""
    print("\n" + "=" * 80)
    print("Example 6: Summary Report")
    print("=" * 80)
    
    summary = design_data.generate_summary_report()
    print(summary)
    
    # Save to file
    with open('example_output/design_summary.txt', 'w') as f:
        f.write(summary)
    print("\n✓ Summary saved to: example_output/design_summary.txt")


def example7_load_and_query():
    """Example 7: Load saved JSON and query."""
    print("\n" + "=" * 80)
    print("Example 7: Load Saved Data and Query")
    print("=" * 80)
    
    try:
        # Load from previously saved JSON
        design_data = DesignData.from_json('example_output/gcdshy130_benchmark.json')
        
        print(f"\nLoaded design: {design_data.design_name}")
        print(f"Extraction timestamp: {design_data.extraction_timestamp}")
        
        # Query specific stage
        placement = design_data.placement
        if placement:
            print(f"\nPlacement stage metrics:")
            for metric, value in placement.key_metrics.items():
                print(f"  {metric}: {value}")
        
        # Access substage information
        cts_substage = query_substage_info(design_data, 'cts', '4_1_cts')
        if cts_substage:
            print(f"\nCTS Substage 4_1_cts:")
            print(f"  Elapsed time: {cts_substage.elapsed_time}")
            print(f"  Peak memory: {cts_substage.peak_memory} KB")
            print(f"  Info messages: {len(cts_substage.info)}")
            print(f"  Warnings: {len(cts_substage.warnings)}")
        
    except FileNotFoundError:
        print("\n⚠ Benchmark JSON not found. Run Example 1 first.")


def example8_multi_design_comparison():
    """Example 8: Compare multiple designs (if multiple logs available)."""
    print("\n" + "=" * 80)
    print("Example 8: Multi-Design Comparison")
    print("=" * 80)
    
    import glob
    import os
    
    # Find all log files
    log_files = glob.glob('*.txt')
    
    if len(log_files) < 2:
        print("\n⚠ Need at least 2 log files for comparison")
        print(f"Found: {log_files}")
        return
    
    designs = []
    for log_file in log_files[:3]:  # Limit to 3 for demo
        print(f"\nProcessing {log_file}...")
        design_data = create_unified_benchmark(
            log_file=log_file,
            include_raw=False,  # Faster, smaller
            verbose=False
        )
        designs.append(design_data)
    
    # Comparison table
    print("\n" + "-" * 100)
    print(f"{'Design':<20} {'Final Area':<15} {'Clock Buffers':<15} {'Wirelength':<15} {'Warnings':<10}")
    print("-" * 100)
    
    for design in designs:
        final_area = query_metric(design, 'final', 'final_area') or 'N/A'
        clock_buffers = query_metric(design, 'cts', 'clock_buffers') or 'N/A'
        wirelength = query_metric(design, 'routing', 'total_wirelength') or 'N/A'
        
        all_warnings = design.get_all_warnings()
        total_warnings = sum(sum(w.values()) for w in all_warnings.values())
        
        print(f"{design.design_name:<20} {str(final_area):<15} {str(clock_buffers):<15} {str(wirelength):<15} {total_warnings:<10}")


def main():
    """Run all examples."""
    import os
    
    # Create output directory
    os.makedirs('example_output', exist_ok=True)
    os.makedirs('example_output/stages', exist_ok=True)
    
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "UNIFIED LOG STRUCTURE EXAMPLES" + " " * 28 + "║")
    print("╚" + "═" * 78 + "╝")
    
    # Check if log file exists
    if not os.path.exists('gcdshy130.txt'):
        print("\n⚠ Error: gcdshy130.txt not found in current directory")
        print("Please provide a valid OpenROAD log file")
        return
    
    # Run examples
    design_data = example1_basic_extraction()
    example2_performance_analysis(design_data)
    example3_warning_analysis(design_data)
    qa_pairs = example4_generate_qa_benchmark(design_data)
    example5_custom_qa_generation(design_data)
    example6_summary_report(design_data)
    example7_load_and_query()
    example8_multi_design_comparison()
    
    print("\n" + "=" * 80)
    print("All examples completed!")
    print("Output files created in: example_output/")
    print("=" * 80)


if __name__ == "__main__":
    main()

