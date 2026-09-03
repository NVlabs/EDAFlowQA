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
Compare Metrics Testing File
Compares metrics across all stages (synthesis, floorplan, placement, CTS, routing, final)
for different designs (gcd and ariane) using extracted logs.

Includes preprocessing step to extract logs from raw log files using extract_all_stages.py
"""

import json
import os
import sys
from typing import Dict, List, Optional

# Import all stage metric parsers
from synthesis_metrics_parser import extract_synthesis_metrics
from floorplan_metrics_parser import extract_floorplan_metrics
from placement_metrics_parser import extract_placement_metrics
from cts_metrics_parser import extract_cts_metrics
from route_metrics_parser import extract_route_metrics
from final_metrics_parser import extract_final_metrics

# Import extraction pipeline
from extract_all_stages import extract_log_file


class MultiStageMetricsComparator:
    """Compares metrics across all stages for multiple designs."""
    
    def __init__(self, extraction_dir: str = "test_extraction_filtered", 
                 raw_logs_dir: str = ".", auto_extract: bool = True):
        self.extraction_dir = extraction_dir
        self.raw_logs_dir = raw_logs_dir
        self.auto_extract = auto_extract
        self.designs = {}
        
        # Create extraction directory if it doesn't exist
        if not os.path.exists(self.extraction_dir):
            os.makedirs(self.extraction_dir)
            print(f"Created extraction directory: {self.extraction_dir}")
    
    def find_raw_log_file(self, design_name: str) -> Optional[str]:
        """Find the raw log file for a design."""
        # Common patterns for log files
        patterns = [
            f"{design_name}.txt",
            f"{design_name}.log",
            f"{design_name}_log.txt",
            f"log_{design_name}.txt"
        ]
        
        for pattern in patterns:
            filepath = os.path.join(self.raw_logs_dir, pattern)
            if os.path.exists(filepath):
                return filepath
        
        return None
    
    def check_extracted_files(self, design_name: str) -> Dict[str, bool]:
        """Check which extracted files exist for a design."""
        stages = {
            'synthesis': f"{design_name}_synthesis_extract.txt",
            'floorplan': f"{design_name}_floorplan_extract.txt",
            'placement': f"{design_name}_placement_extract.txt",
            'cts': f"{design_name}_cts_extract.txt",
            'routing': f"{design_name}_route_extract.txt",
            'final': f"{design_name}_final_extract.txt"
        }
        
        status = {}
        for stage_name, filename in stages.items():
            filepath = os.path.join(self.extraction_dir, filename)
            status[stage_name] = os.path.exists(filepath)
        
        return status
    
    def extract_logs_if_needed(self, design_name: str) -> bool:
        """Extract logs from raw log file if extracted files don't exist."""
        # Check if extracted files exist
        extraction_status = self.check_extracted_files(design_name)
        
        if all(extraction_status.values()):
            print(f"  ℹ️  All extracted files already exist for {design_name}")
            return True
        
        missing_stages = [stage for stage, exists in extraction_status.items() if not exists]
        print(f"  ⚠️  Missing extracted files for {design_name}: {', '.join(missing_stages)}")
        
        if not self.auto_extract:
            print(f"  ℹ️  Auto-extraction disabled. Please extract logs manually.")
            return False
        
        # Find raw log file
        raw_log_file = self.find_raw_log_file(design_name)
        if not raw_log_file:
            print(f"  ✗ Could not find raw log file for {design_name}")
            print(f"     Searched in: {self.raw_logs_dir}")
            print(f"     Expected patterns: {design_name}.txt, {design_name}.log, etc.")
            return False
        
        print(f"  ℹ️  Found raw log file: {raw_log_file}")
        print(f"  🔄 Extracting logs to {self.extraction_dir}...")
        
        try:
            # Run extraction
            extract_log_file(raw_log_file, self.extraction_dir)
            print(f"  ✓ Extraction complete for {design_name}")
            return True
        except Exception as e:
            print(f"  ✗ Error during extraction: {str(e)}")
            return False
        
    def load_design_metrics(self, design_name: str) -> Dict:
        """Load and extract metrics from all stages for a design."""
        print(f"\n{'=' * 80}")
        print(f"Loading metrics for: {design_name}")
        print(f"{'=' * 80}")
        
        # Preprocessing: Extract logs if needed
        if not self.extract_logs_if_needed(design_name):
            print(f"  ⚠️  Warning: Some extracted files may be missing")
        
        design_metrics = {
            'design_name': design_name,
            'stages': {}
        }
        
        # Define stage files and their parsers
        stages = {
            'synthesis': ('synthesis_extract.txt', extract_synthesis_metrics),
            'floorplan': ('floorplan_extract.txt', extract_floorplan_metrics),
            'placement': ('placement_extract.txt', extract_placement_metrics),
            'cts': ('cts_extract.txt', extract_cts_metrics),
            'routing': ('route_extract.txt', extract_route_metrics),
            'final': ('final_extract.txt', extract_final_metrics)
        }
        
        for stage_name, (filename_suffix, parser_func) in stages.items():
            filepath = os.path.join(self.extraction_dir, f"{design_name}_{filename_suffix}")
            
            if not os.path.exists(filepath):
                print(f"  ⚠️  {stage_name:12s}: File not found - {filepath}")
                continue
            
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    log_content = f.read()
                
                metrics = parser_func(log_content)
                design_metrics['stages'][stage_name] = metrics
                
                # Print summary info
                if stage_name == 'synthesis':
                    summary = metrics.get('summary', {})
                    print(f"  ✓ {stage_name:12s}: {summary.get('total_stages', 0)} stages, "
                          f"{summary.get('total_cpu_time', 0):.2f}s CPU")
                elif stage_name in ['placement', 'cts', 'routing']:
                    summary = metrics.get('summary', {})
                    wns = summary.get('wns', 'N/A')
                    tns = summary.get('tns', 'N/A')
                    if isinstance(wns, (int, float)):
                        print(f"  ✓ {stage_name:12s}: WNS={wns:.3f}, TNS={tns:.1f}")
                    else:
                        print(f"  ✓ {stage_name:12s}: Loaded")
                else:
                    print(f"  ✓ {stage_name:12s}: Loaded")
                    
            except Exception as e:
                print(f"  ✗ {stage_name:12s}: Error - {str(e)}")
                continue
        
        return design_metrics
    
    def compare_stage_metrics(self, stage_name: str, design1: Dict, design2: Dict) -> Dict:
        """Compare metrics for a specific stage between two designs."""
        metrics1 = design1['stages'].get(stage_name, {})
        metrics2 = design2['stages'].get(stage_name, {})
        
        if not metrics1 or not metrics2:
            return {'error': 'Stage not available for both designs'}
        
        comparison = {
            'stage': stage_name,
            'design1': design1['design_name'],
            'design2': design2['design_name'],
            'differences': {}
        }
        
        # Get summaries
        summary1 = metrics1.get('summary', {})
        summary2 = metrics2.get('summary', {})
        
        # Compare common metrics
        if stage_name == 'synthesis':
            comparison['differences'] = {
                'cpu_time': {
                    'design1': summary1.get('total_cpu_time', 0),
                    'design2': summary2.get('total_cpu_time', 0),
                    'diff': summary1.get('total_cpu_time', 0) - summary2.get('total_cpu_time', 0)
                },
                'memory_mb': {
                    'design1': summary1.get('peak_memory_kb', 0) / 1024,
                    'design2': summary2.get('peak_memory_kb', 0) / 1024,
                    'diff': (summary1.get('peak_memory_kb', 0) - summary2.get('peak_memory_kb', 0)) / 1024
                },
                'warnings': {
                    'design1': summary1.get('total_warnings', 0),
                    'design2': summary2.get('total_warnings', 0),
                    'diff': summary1.get('total_warnings', 0) - summary2.get('total_warnings', 0)
                },
                'stages': {
                    'design1': summary1.get('total_stages', 0),
                    'design2': summary2.get('total_stages', 0),
                    'diff': summary1.get('total_stages', 0) - summary2.get('total_stages', 0)
                },
                'modules': {
                    'design1': summary1.get('total_modules', 0),
                    'design2': summary2.get('total_modules', 0),
                    'diff': summary1.get('total_modules', 0) - summary2.get('total_modules', 0)
                }
            }
        elif stage_name in ['floorplan', 'placement', 'cts', 'routing', 'final']:
            # Common metrics for OpenROAD stages
            comparison['differences'] = {
                'area': {
                    'design1': summary1.get('area'),
                    'design2': summary2.get('area'),
                    'diff': (summary1.get('area', 0) or 0) - (summary2.get('area', 0) or 0)
                },
                'utilization': {
                    'design1': summary1.get('utilization'),
                    'design2': summary2.get('utilization'),
                    'diff': (summary1.get('utilization', 0) or 0) - (summary2.get('utilization', 0) or 0)
                }
            }
            
            # Timing metrics if available
            if summary1.get('wns') is not None or summary2.get('wns') is not None:
                comparison['differences']['wns'] = {
                    'design1': summary1.get('wns'),
                    'design2': summary2.get('wns'),
                    'diff': (summary1.get('wns', 0) or 0) - (summary2.get('wns', 0) or 0)
                }
                comparison['differences']['tns'] = {
                    'design1': summary1.get('tns'),
                    'design2': summary2.get('tns'),
                    'diff': (summary1.get('tns', 0) or 0) - (summary2.get('tns', 0) or 0)
                }
            
            # HPWL if available
            if summary1.get('hpwl') is not None or summary2.get('hpwl') is not None:
                comparison['differences']['hpwl'] = {
                    'design1': summary1.get('hpwl'),
                    'design2': summary2.get('hpwl'),
                    'diff': (summary1.get('hpwl', 0) or 0) - (summary2.get('hpwl', 0) or 0)
                }
        
        return comparison
    
    def generate_comparison_report(self, design1: Dict, design2: Dict) -> str:
        """Generate a comprehensive comparison report."""
        lines = []
        lines.append("=" * 100)
        lines.append("MULTI-STAGE METRICS COMPARISON REPORT")
        lines.append("=" * 100)
        lines.append(f"Design 1: {design1['design_name']}")
        lines.append(f"Design 2: {design2['design_name']}")
        lines.append("=" * 100)
        
        # Get all stages that exist in both designs
        stages1 = set(design1['stages'].keys())
        stages2 = set(design2['stages'].keys())
        common_stages = sorted(stages1 & stages2)
        
        if not common_stages:
            lines.append("\n⚠️  No common stages found for comparison")
            return '\n'.join(lines)
        
        lines.append(f"\nComparing {len(common_stages)} stages: {', '.join(common_stages)}")
        lines.append("")
        
        # Stage-by-stage comparison
        for stage_name in ['synthesis', 'floorplan', 'placement', 'cts', 'routing', 'final']:
            if stage_name not in common_stages:
                continue
            
            lines.append("\n" + "-" * 100)
            lines.append(f"STAGE: {stage_name.upper()}")
            lines.append("-" * 100)
            
            comparison = self.compare_stage_metrics(stage_name, design1, design2)
            
            if 'error' in comparison:
                lines.append(f"  {comparison['error']}")
                continue
            
            differences = comparison.get('differences', {})
            
            # Format output based on metric type
            lines.append(f"{'Metric':<25} | {design1['design_name']:>20} | {design2['design_name']:>20} | {'Difference':>20}")
            lines.append("-" * 100)
            
            for metric_name, values in differences.items():
                d1_val = values.get('design1')
                d2_val = values.get('design2')
                diff_val = values.get('diff')
                
                # Format values based on type
                if d1_val is None:
                    d1_str = "N/A"
                elif isinstance(d1_val, float):
                    d1_str = f"{d1_val:.2f}"
                else:
                    d1_str = str(d1_val)
                
                if d2_val is None:
                    d2_str = "N/A"
                elif isinstance(d2_val, float):
                    d2_str = f"{d2_val:.2f}"
                else:
                    d2_str = str(d2_val)
                
                if diff_val is None:
                    diff_str = "N/A"
                elif isinstance(diff_val, float):
                    diff_str = f"{diff_val:+.2f}"
                else:
                    diff_str = f"{diff_val:+d}"
                
                lines.append(f"{metric_name:<25} | {d1_str:>20} | {d2_str:>20} | {diff_str:>20}")
        
        lines.append("\n" + "=" * 100)
        lines.append("SUMMARY")
        lines.append("=" * 100)
        
        # Overall summary
        lines.append(f"\n{design1['design_name']}:")
        lines.append(f"  Stages processed: {len(design1['stages'])}")
        
        lines.append(f"\n{design2['design_name']}:")
        lines.append(f"  Stages processed: {len(design2['stages'])}")
        
        lines.append("\n" + "=" * 100)
        
        return '\n'.join(lines)
    
    def save_comparison_results(self, design1: Dict, design2: Dict, output_file: str):
        """Save detailed comparison results to JSON."""
        comparison_data = {
            'designs': {
                design1['design_name']: design1,
                design2['design_name']: design2
            },
            'stage_comparisons': {}
        }
        
        # Get common stages
        stages1 = set(design1['stages'].keys())
        stages2 = set(design2['stages'].keys())
        common_stages = sorted(stages1 & stages2)
        
        for stage_name in common_stages:
            comparison = self.compare_stage_metrics(stage_name, design1, design2)
            comparison_data['stage_comparisons'][stage_name] = comparison
        
        with open(output_file, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        print(f"\n✓ Comparison results saved to: {output_file}")


def main():
    """Main function to run the comparison tests."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Compare metrics across all stages for multiple designs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare GCD and Ariane designs (auto-extracts if needed)
  python compare_metrics_test.py gcdshy130 ariane133_nan45
  
  # Specify raw log file directory
  python compare_metrics_test.py gcdshy130 ariane133_nan45 --raw-logs-dir ./logs
  
  # Disable auto-extraction
  python compare_metrics_test.py gcdshy130 ariane133_nan45 --no-auto-extract
  
  # Save comparison to JSON
  python compare_metrics_test.py gcdshy130 ariane133_nan45 -o comparison.json
  
  # Use custom extraction directory
  python compare_metrics_test.py gcdshy130 ariane133_nan45 -d my_extraction_dir
  
  # Force re-extraction
  python compare_metrics_test.py gcdshy130 ariane133_nan45 --force-extract
        """
    )
    
    parser.add_argument('design1', help='First design name (e.g., gcdshy130)')
    parser.add_argument('design2', help='Second design name (e.g., ariane133_nan45)')
    parser.add_argument('-d', '--dir', default='test_extraction_filtered',
                       help='Directory containing/for extracted logs (default: test_extraction_filtered)')
    parser.add_argument('--raw-logs-dir', default='.',
                       help='Directory containing raw log files (default: current directory)')
    parser.add_argument('--no-auto-extract', action='store_true',
                       help='Disable automatic extraction of logs')
    parser.add_argument('--force-extract', action='store_true',
                       help='Force re-extraction even if files exist')
    parser.add_argument('-o', '--output', help='Output JSON file for detailed comparison')
    parser.add_argument('--save-metrics', action='store_true',
                       help='Save individual design metrics to separate JSON files')
    
    args = parser.parse_args()
    
    # Force extraction if requested
    if args.force_extract:
        import shutil
        print(f"\n🔄 Force extraction requested - removing existing extracted files...")
        for design in [args.design1, args.design2]:
            for stage in ['synthesis', 'floorplan', 'placement', 'cts', 'route', 'final']:
                filepath = os.path.join(args.dir, f"{design}_{stage}_extract.txt")
                if os.path.exists(filepath):
                    os.remove(filepath)
                    print(f"  Removed: {filepath}")
    
    # Create comparator
    comparator = MultiStageMetricsComparator(
        extraction_dir=args.dir,
        raw_logs_dir=args.raw_logs_dir,
        auto_extract=not args.no_auto_extract
    )
    
    # Load metrics for both designs
    print("\n" + "=" * 80)
    print("MULTI-STAGE METRICS COMPARISON TEST")
    print("=" * 80)
    
    design1_metrics = comparator.load_design_metrics(args.design1)
    design2_metrics = comparator.load_design_metrics(args.design2)
    
    for stage_name, metrics in design1_metrics['stages'].items():
        print(stage_name)
        print(metrics)
    
    # Save individual metrics if requested
    if args.save_metrics:
        design1_file = f"{args.design1}_all_stages_metrics.json"
        with open(design1_file, 'w') as f:
            json.dump(design1_metrics, f, indent=2)
        print(f"\n✓ {args.design1} metrics saved to: {design1_file}")
        
        design2_file = f"{args.design2}_all_stages_metrics.json"
        with open(design2_file, 'w') as f:
            json.dump(design2_metrics, f, indent=2)
        print(f"✓ {args.design2} metrics saved to: {design2_file}")
    
    # Generate and print comparison report
    print("\n\n")
    report = comparator.generate_comparison_report(design1_metrics, design2_metrics)
    print(report)
    
    # Save detailed comparison if output file specified
    if args.output:
        comparator.save_comparison_results(design1_metrics, design2_metrics, args.output)


if __name__ == "__main__":
    main()

