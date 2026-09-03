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
Load Design Metrics
Loads and extracts metrics from all stages for a single design.
Includes preprocessing step to extract logs from raw log files.
"""

import json
import os
from typing import Dict, Optional

# Import all stage metric parsers
from synthesis_metrics_parser import extract_synthesis_metrics
from floorplan_metrics_parser import extract_floorplan_metrics
from placement_metrics_parser import extract_placement_metrics
from cts_metrics_parser import extract_cts_metrics
from route_metrics_parser import extract_route_metrics
from final_metrics_parser import extract_final_metrics

# Import extraction pipeline
from extract_all_stages import extract_log_file


class DesignMetricsLoader:
    """Loads metrics from all stages for a single design."""
    
    def __init__(self, extraction_dir: str = "test_extraction_filtered", 
                 raw_logs_dir: str = ".", auto_extract: bool = True):
        """
        Initialize the design metrics loader.
        
        Args:
            extraction_dir: Directory containing/for extracted logs
            raw_logs_dir: Directory containing raw log files
            auto_extract: Whether to automatically extract logs if missing
        """
        self.extraction_dir = extraction_dir
        self.raw_logs_dir = raw_logs_dir
        self.auto_extract = auto_extract
        
        # Create extraction directory if it doesn't exist
        if not os.path.exists(self.extraction_dir):
            os.makedirs(self.extraction_dir)
            print(f"Created extraction directory: {self.extraction_dir}")
    
    def find_raw_log_file(self, design_name: str) -> Optional[str]:
        """Find the raw log file for a design."""
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
        extraction_status = self.check_extracted_files(design_name)
        
        if all(extraction_status.values()):
            print(f"✓ All extracted files already exist for {design_name}")
            return True
        
        missing_stages = [stage for stage, exists in extraction_status.items() if not exists]
        print(f"⚠️  Missing extracted files: {', '.join(missing_stages)}")
        
        if not self.auto_extract:
            print(f"ℹ️  Auto-extraction disabled. Please extract logs manually.")
            return False
        
        raw_log_file = self.find_raw_log_file(design_name)
        if not raw_log_file:
            print(f"✗ Could not find raw log file for {design_name}")
            print(f"   Searched in: {self.raw_logs_dir}")
            return False
        
        print(f"ℹ️  Found raw log file: {raw_log_file}")
        print(f"🔄 Extracting logs to {self.extraction_dir}...")
        
        try:
            extract_log_file(raw_log_file, self.extraction_dir)
            print(f"✓ Extraction complete for {design_name}")
            return True
        except Exception as e:
            print(f"✗ Error during extraction: {str(e)}")
            return False
    
    def load_design_metrics(self, design_name: str, verbose: bool = True) -> Dict:
        """
        Load and extract metrics from all stages for a design.
        
        Args:
            design_name: Name of the design (e.g., 'gcdshy130', 'ariane133_nan45')
            verbose: Whether to print detailed loading information
            
        Returns:
            Dictionary containing all stage metrics
        """
        if verbose:
            print(f"\n{'=' * 80}")
            print(f"Loading metrics for: {design_name}")
            print(f"{'=' * 80}\n")
        
        # Preprocessing: Extract logs if needed
        if not self.extract_logs_if_needed(design_name):
            if verbose:
                print(f"⚠️  Warning: Some extracted files may be missing\n")
        
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
                if verbose:
                    print(f"⚠️  {stage_name:12s}: File not found")
                continue
            
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    log_content = f.read()
                
                metrics = parser_func(log_content)
                design_metrics['stages'][stage_name] = metrics
                
                if verbose:
                    # Print summary info for each stage
                    if stage_name == 'synthesis':
                        summary = metrics.get('summary', {})
                        print(f"✓ {stage_name:12s}: {summary.get('total_stages', 0)} stages, "
                              f"{summary.get('total_cpu_time', 0):.2f}s CPU, "
                              f"{summary.get('peak_memory_kb', 0) / 1024:.1f} MB")
                    elif stage_name in ['placement', 'cts', 'routing']:
                        summary = metrics.get('summary', {})
                        wns = summary.get('wns', 'N/A')
                        tns = summary.get('tns', 'N/A')
                        area = summary.get('area', 'N/A')
                        if isinstance(wns, (int, float)) and isinstance(area, (int, float)):
                            print(f"✓ {stage_name:12s}: Area={area:.1f}, WNS={wns:.3f}, TNS={tns:.1f}")
                        elif isinstance(area, (int, float)):
                            print(f"✓ {stage_name:12s}: Area={area:.1f}")
                        else:
                            print(f"✓ {stage_name:12s}: Loaded")
                    else:
                        print(f"✓ {stage_name:12s}: Loaded")
                        
            except Exception as e:
                if verbose:
                    print(f"✗ {stage_name:12s}: Error - {str(e)}")
                continue
        
        if verbose:
            print(f"\n{'=' * 80}")
            print(f"Loaded {len(design_metrics['stages'])} stages")
            print(f"{'=' * 80}")
        
        return design_metrics


def print_design_summary(metrics: Dict):
    """Print a summary of the design metrics."""
    design_name = metrics.get('design_name', 'Unknown')
    stages = metrics.get('stages', {})
    
    print(f"\n{'=' * 80}")
    print(f"DESIGN METRICS SUMMARY: {design_name}")
    print(f"{'=' * 80}")
    
    if not stages:
        print("\nNo stages loaded.")
        return
    
    print(f"\nStages loaded: {len(stages)}")
    print(f"Stage names: {', '.join(stages.keys())}\n")
    
    # Print key metrics for each stage
    for stage_name, stage_metrics in stages.items():
        print(f"\n--- {stage_name.upper()} ---")
        summary = stage_metrics.get('summary', {})
        
        if stage_name == 'synthesis':
            print(f"  Yosys Version: {summary.get('yosys_version', 'N/A')}")
            print(f"  Clock Period: {summary.get('clock_period', 'N/A')} ns")
            print(f"  Total Stages: {summary.get('total_stages', 0)}")
            print(f"  Total Modules: {summary.get('total_modules', 0)}")
            print(f"  Total CPU Time: {summary.get('total_cpu_time', 0):.2f}s")
            print(f"  Peak Memory: {summary.get('peak_memory_kb', 0) / 1024:.1f} MB")
            print(f"  Warnings: {summary.get('total_warnings', 0)}")
            print(f"  Errors: {summary.get('total_errors', 0)}")
        else:
            # OpenROAD stages
            if summary.get('area'):
                print(f"  Area: {summary['area']:.2f} u^2")
            if summary.get('utilization'):
                print(f"  Utilization: {summary['utilization']}%")
            if summary.get('wns') is not None:
                print(f"  WNS: {summary['wns']:.3f}")
            if summary.get('tns') is not None:
                print(f"  TNS: {summary['tns']:.1f}")
            if summary.get('violations') is not None:
                print(f"  Violations: {summary['violations']}")
            if summary.get('hpwl'):
                print(f"  HPWL: {summary['hpwl']:.1f} u")
    
    print(f"\n{'=' * 80}")


def save_design_metrics(metrics: Dict, output_file: str, pretty: bool = True):
    """
    Save design metrics to JSON file.
    
    Args:
        metrics: Design metrics dictionary
        output_file: Output file path
        pretty: Whether to format JSON with indentation
    """
    with open(output_file, 'w') as f:
        if pretty:
            json.dump(metrics, f, indent=2)
        else:
            json.dump(metrics, f)
    
    print(f"\n✓ Metrics saved to: {output_file}")


def save_stage_metrics_to_directory(metrics: Dict, output_dir: str, pretty: bool = True):
    """
    Save each stage's metrics to individual JSON files in a directory.
    
    Args:
        metrics: Design metrics dictionary
        output_dir: Output directory path
        pretty: Whether to format JSON with indentation
    """
    design_name = metrics.get('design_name', 'unknown')
    stages = metrics.get('stages', {})
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Save each stage to a separate file
    for stage_name, stage_metrics in stages.items():
        output_file = os.path.join(output_dir, f"{design_name}_{stage_name}_metrics.json")
        
        with open(output_file, 'w') as f:
            if pretty:
                json.dump(stage_metrics, f, indent=2)
            else:
                json.dump(stage_metrics, f)
        
        print(f"✓ {stage_name:12s} metrics saved to: {output_file}")
    
    # Also save the complete metrics with all stages
    complete_file = os.path.join(output_dir, f"{design_name}_all_stages_metrics.json")
    with open(complete_file, 'w') as f:
        if pretty:
            json.dump(metrics, f, indent=2)
        else:
            json.dump(metrics, f)
    
    print(f"✓ Complete metrics saved to: {complete_file}")
    print(f"\n✓ Total: {len(stages)} stage files + 1 complete file saved to {output_dir}")


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Load and extract metrics from all stages for a single design',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load metrics for GCD design (auto-extracts if needed)
  python load_design_metrics.py gcdshy130
  
  # Load with custom directories
  python load_design_metrics.py ariane133_nan45 --raw-logs-dir ./logs -d ./extracted
  
  # Save metrics to JSON
  python load_design_metrics.py gcdshy130 -o gcdshy130_metrics.json
  
  # Save each stage to separate files in a directory
  python load_design_metrics.py gcdshy130 --output-dir ./metrics_output
  
  # Disable auto-extraction
  python load_design_metrics.py gcdshy130 --no-auto-extract
  
  # Force re-extraction
  python load_design_metrics.py gcdshy130 --force-extract
  
  # Quiet mode (minimal output)
  python load_design_metrics.py gcdshy130 -q -o metrics.json
        """
    )
    
    parser.add_argument('design_name', help='Design name (e.g., gcdshy130, ariane133_nan45)')
    parser.add_argument('-d', '--dir', default='test_extraction_filtered',
                       help='Directory containing/for extracted logs (default: test_extraction_filtered)')
    parser.add_argument('--raw-logs-dir', default='.',
                       help='Directory containing raw log files (default: current directory)')
    parser.add_argument('--no-auto-extract', action='store_true',
                       help='Disable automatic extraction of logs')
    parser.add_argument('--force-extract', action='store_true',
                       help='Force re-extraction even if files exist')
    parser.add_argument('-o', '--output', help='Output JSON file for complete metrics')
    parser.add_argument('--output-dir', help='Output directory for individual stage metrics')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Quiet mode - minimal output')
    parser.add_argument('--summary', action='store_true',
                       help='Print summary after loading')
    
    args = parser.parse_args()
    
    # Force extraction if requested
    if args.force_extract:
        print(f"\n🔄 Force extraction requested - removing existing extracted files...")
        for stage in ['synthesis', 'floorplan', 'placement', 'cts', 'route', 'final']:
            filepath = os.path.join(args.dir, f"{args.design_name}_{stage}_extract.txt")
            if os.path.exists(filepath):
                os.remove(filepath)
                if not args.quiet:
                    print(f"  Removed: {filepath}")
    
    # Create loader
    loader = DesignMetricsLoader(
        extraction_dir=args.dir,
        raw_logs_dir=args.raw_logs_dir,
        auto_extract=not args.no_auto_extract
    )
    
    # Load metrics
    metrics = loader.load_design_metrics(args.design_name, verbose=not args.quiet)
    
    # Print summary if requested
    if args.summary and not args.quiet:
        print_design_summary(metrics)
    
    # Save to directory if requested (each stage in separate file)
    if args.output_dir:
        save_stage_metrics_to_directory(metrics, args.output_dir, pretty=True)
    
    # Save to single JSON file if requested
    if args.output:
        save_design_metrics(metrics, args.output, pretty=True)
    
    # If no output file/dir and not quiet, print summary
    if not args.output and not args.output_dir and not args.quiet and not args.summary:
        print_design_summary(metrics)
    
    return metrics


if __name__ == "__main__":
    main()

