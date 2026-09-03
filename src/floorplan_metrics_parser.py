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

import re
from typing import Dict, Optional, List


def remove_null_values(obj):
    """Recursively remove null/None values from dictionaries and lists."""
    if isinstance(obj, dict):
        return {k: remove_null_values(v) for k, v in obj.items() if v is not None and v != [] and v != {}}
    elif isinstance(obj, list):
        cleaned_list = [remove_null_values(item) for item in obj if item is not None]
        return [item for item in cleaned_list if item != {} and item != []]
    else:
        return obj


def extract_floorplan_metrics(log_content: str, design_name: Optional[str] = None, sub_design_name: Optional[str] = None) -> Dict:
    """
    Extract area, utilization, timing, and density metrics from floorplan stage extracted log.
    Returns a structured dictionary with metrics organized by substages (e.g., 2_1_floorplan, 2_2_floorplan_macro).
    
    Optimization processes track: iteration count, start metrics, and end metrics (not all iterations).
    
    Args:
        log_content: The extracted floorplan log content as a string
        
    Returns:
        Dictionary containing:
        {
            'stage': 'floorplan',
            'substages': {
                '2_1_floorplan': {
                    # Direct metrics at substage level
                    'wns': float,
                    'tns': float,
                    'area': float,
                    'utilization': float,
                    'density': float,
                    'violations': int,
                    
                    # Optimization processes in execution order
                    'optimization_process': [
                        {
                            'type': 'repair_tie_fanout',
                            'results': {
                                'tie_lo_instances': int,
                                'tie_hi_instances': int
                            }
                        },
                        {
                            'type': 'repair_timing',
                            'command': str,
                            'iterations_count': int,
                            'start_metrics': {'wns': float, 'tns': float, 'violation_end_points': int},
                            'end_metrics': {'wns': float, 'tns': float, 'violation_end_points': int},
                            'summary': {
                                'removed_buffers': int,
                                'inserted_buffers': int,
                                'resized_instances': int
                            }
                        }
                    ],
                    
                    # Performance metrics
                    'elapsed_time': str,
                    'peak_memory_kb': int,
                    'elapsed_s': int,
                    'memory_mb': int,
                    'sha1sum': str
                },
                '2_2_floorplan_macro': {
                    'macro_placement': {...},
                    'performance': {...}
                },
                ...
            },
            'summary': {
                'area': float,
                'utilization': float,
                'wns': float,
                'tns': float,
                'violations': int,
                'density': float
            }
        }
    """
    lines = log_content.split('\n')
    
    result = {
        'stage': 'floorplan',
        'substages': {},
        'summary': {}
    }

    if design_name is not None:
        result['design_name'] = design_name
    if sub_design_name is not None:
        result['sub_design_name'] = sub_design_name
    
    # Track current substage
    current_substage = None
    in_repair_timing_table = False
    substages = {}
    
    # Helper function to initialize a substage
    def init_substage():
        return {
            # Direct metrics at substage level
            'wns': None,
            'tns': None,
            'area': None,
            'utilization': None,
            'density': None,
            'violations': None,
            
            # Optimization processes in execution order
            'optimization_process': [],
            
            # Report metrics section (from "Report metrics stage 2, substage..." sections)
            'report_metrics': None,
            
            # Performance metrics
            'elapsed_time': None,
            'peak_memory_kb': None,
            'elapsed_s': None,
            'memory_mb': None,
            'sha1sum': None
        }
    
    # Track current optimization process
    current_optimization = None
    
    # FIRST PASS: Parse execution details and populate substages
    for i, line in enumerate(lines):
        # Detect substages - look for patterns like "2_1_floorplan", "2_2_floorplan_macro" etc.
        stage_match = re.search(r'(2_\d+_\w+)', line)
        if stage_match and 'Running' in line:
            current_substage = stage_match.group(1)
            if current_substage not in substages:
                substages[current_substage] = init_substage()
            in_repair_timing_table = False
            current_optimization = None
        
        # Also check for stage in log summary lines
        stage_summary_match = re.search(r'^[^\]]*\]\s+(2_\d+_\w+)\s+\d+\s+\d+', line)
        if stage_summary_match:
            stage_name = stage_summary_match.group(1)
            if stage_name not in substages:
                substages[stage_name] = init_substage()
        
        # If we have a current substage, extract metrics
        if current_substage and current_substage in substages:
            # Detect tie lo/hi fanout repair
            match = re.search(r'Repair tie lo fanout', line)
            if match:
                current_optimization = {
                    'type': 'repair_tie_fanout',
                    'results': {
                        'tie_lo_instances': None,
                        'tie_hi_instances': None
                    }
                }
                substages[current_substage]['optimization_process'].append(current_optimization)
            
            # Capture tie lo instances
            match = re.search(r'Inserted\s+(\d+)\s+tie\s+LOGIC0', line)
            if match and current_optimization and current_optimization['type'] == 'repair_tie_fanout':
                current_optimization['results']['tie_lo_instances'] = int(match.group(1))
            
            # Capture tie hi instances
            match = re.search(r'Inserted\s+(\d+)\s+tie\s+LOGIC1', line)
            if match and current_optimization and current_optimization['type'] == 'repair_tie_fanout':
                current_optimization['results']['tie_hi_instances'] = int(match.group(1))
            
            # Detect repair timing operation - avoid "Took X seconds" message
            if 'repair_timing' in line and '-setup' in line and 'Took' not in line and 'seconds:' not in line:
                current_optimization = {
                    'type': 'repair_timing',
                    'command': line.split('] ')[-1].strip() if ']' in line else line.strip(),
                    'iterations_count': 0,
                    'start_metrics': {
                        'wns': None,
                        'tns': None,
                        'violation_end_points': None
                    },
                    'end_metrics': {
                        'wns': None,
                        'tns': None,
                        'violation_end_points': None
                    },
                    'summary': {
                        'removed_buffers': None,
                        'inserted_buffers': None,
                        'resized_instances': None,
                        'cloned_instances': None
                    }
                }
                substages[current_substage]['optimization_process'].append(current_optimization)
                in_repair_timing_table = False
            
            # Detect macro placer operation
            if 'rtl_macro_placer' in line and 'Took' not in line:
                current_optimization = {
                    'type': 'macro_placement',
                    'command': line.split('] ')[-1].strip() if ']' in line else line.strip(),
                    'results': {
                        'num_std_cells': None,
                        'area_std_cells': None,
                        'num_macros': None,
                        'area_macros': None,
                        'design_utilization': None,
                        'floorplan_utilization': None
                    }
                }
                substages[current_substage]['optimization_process'].append(current_optimization)
            
            # Capture macro placement metrics
            if current_optimization and current_optimization['type'] == 'macro_placement':
                match = re.search(r'Number of std cell instances:\s+(\d+)', line)
                if match:
                    current_optimization['results']['num_std_cells'] = int(match.group(1))
                
                match = re.search(r'Area of std cell instances:\s+([\d.]+)', line)
                if match:
                    current_optimization['results']['area_std_cells'] = float(match.group(1))
                
                match = re.search(r'Number of macros:\s+(\d+)', line)
                if match:
                    current_optimization['results']['num_macros'] = int(match.group(1))
                
                match = re.search(r'Area of macros:\s+([\d.]+)', line)
                if match:
                    current_optimization['results']['area_macros'] = float(match.group(1))
                
                match = re.search(r'Design Utilization:\s+([\d.]+)', line)
                if match:
                    util = float(match.group(1)) * 100  # Convert to percentage
                    current_optimization['results']['design_utilization'] = util
                    substages[current_substage]['utilization'] = util
                    substages[current_substage]['density'] = util
                
                match = re.search(r'Floorplan Utilization:\s+([\d.]+)', line)
                if match:
                    current_optimization['results']['floorplan_utilization'] = float(match.group(1)) * 100
            
            # Detect tapcell insertion
            match = re.search(r'Inserted\s+(\d+)\s+endcaps', line)
            if match:
                current_optimization = {
                    'type': 'tapcell_insertion',
                    'results': {
                        'endcaps_inserted': int(match.group(1)),
                        'tapcells_inserted': None
                    }
                }
                substages[current_substage]['optimization_process'].append(current_optimization)
            
            match = re.search(r'Inserted\s+(\d+)\s+tapcells', line)
            if match and current_optimization and current_optimization['type'] == 'tapcell_insertion':
                current_optimization['results']['tapcells_inserted'] = int(match.group(1))
            
            # Repair timing table
            if 'Iter' in line and 'WNS' in line and 'TNS' in line:
                in_repair_timing_table = True
                continue
            
            # Parse repair timing table rows
            if in_repair_timing_table and current_optimization and current_optimization['type'] == 'repair_timing':
                if '---' in line:
                    continue
                
                # Parse iteration data
                match = re.search(
                    r'(\d+\*?|final)\s+\|\s+(\d+)\s+\|\s+(\d+)\s+\|\s+(\d+)\s+\|\s+(\d+)\s+\|\s+(\d+)\s+\|\s+'
                    r'([+\-\d.]+)%\s+\|\s+([+\-\d.]+)\s+\|\s+([+\-\d.]+)\s+\|\s+(\d+)',
                    line
                )
                if match:
                    iter_num = match.group(1)
                    wns = float(match.group(8))
                    tns = float(match.group(9))
                    viol = int(match.group(10))
                    
                    # Count iterations
                    current_optimization['iterations_count'] += 1
                    
                    # Capture start metrics (first iteration, usually iter 0)
                    if current_optimization['start_metrics']['wns'] is None:
                        current_optimization['start_metrics']['wns'] = wns
                        current_optimization['start_metrics']['tns'] = tns
                        current_optimization['start_metrics']['violation_end_points'] = viol
                    
                    # Always update end metrics (last iteration will be final)
                    current_optimization['end_metrics']['wns'] = wns
                    current_optimization['end_metrics']['tns'] = tns
                    current_optimization['end_metrics']['violation_end_points'] = viol
                    
                    # Update substage-level metrics with final values
                    if iter_num == 'final':
                        substages[current_substage]['wns'] = wns
                        substages[current_substage]['tns'] = tns
                        substages[current_substage]['violations'] = viol
            
            # Repair summary
            match = re.search(r'Removed\s+(\d+)\s+buffers', line)
            if match and current_optimization and current_optimization['type'] == 'repair_timing':
                current_optimization['summary']['removed_buffers'] = int(match.group(1))
            
            match = re.search(r'Inserted\s+(\d+)\s+buffers', line)
            if match and current_optimization and current_optimization['type'] == 'repair_timing':
                current_optimization['summary']['inserted_buffers'] = int(match.group(1))
            
            match = re.search(r'Resized\s+(\d+)\s+instances', line)
            if match and current_optimization and current_optimization['type'] == 'repair_timing':
                current_optimization['summary']['resized_instances'] = int(match.group(1))
            
            match = re.search(r'Cloned\s+(\d+)\s+instances', line)
            if match and current_optimization and current_optimization['type'] == 'repair_timing':
                current_optimization['summary']['cloned_instances'] = int(match.group(1))
            
            # Design area and utilization
            match = re.search(r'Design area\s+([\d.]+)\s+u\^2\s+(\d+)%\s+utilization', line)
            if match:
                area = float(match.group(1))
                util = int(match.group(2))
                substages[current_substage]['area'] = area
                substages[current_substage]['utilization'] = util
                substages[current_substage]['density'] = util  # Density is essentially utilization
            
            # Performance metrics
            match = re.search(r'Elapsed time:\s+([\d:.]+)\[h:\]min:sec', line)
            if match:
                substages[current_substage]['elapsed_time'] = match.group(1)
            
            match = re.search(r'Peak memory:\s+(\d+)KB', line)
            if match:
                substages[current_substage]['peak_memory_kb'] = int(match.group(1))
        
        # Also parse log summary lines (at the end) - these can reference any substage
        log_summary_match = re.search(r'^[^\]]*\]\s+(2_\d+_\w+)\s+(\d+)\s+(\d+)\s+([a-f0-9]+)', line)
        if log_summary_match:
            stage_name, elapsed_s, memory_mb, sha1 = log_summary_match.groups()
            if stage_name in substages:
                substages[stage_name]['elapsed_s'] = int(elapsed_s)
                substages[stage_name]['memory_mb'] = int(memory_mb)
                substages[stage_name]['sha1sum'] = sha1
    
    # Store substages
    result['substages'] = substages
    
    # Create summary with key metrics from the last substage or aggregated from all substages
    summary = {}
    
    # Get the list of substages sorted (2_1_floorplan, 2_2_floorplan_macro, etc.)
    sorted_substages = sorted(substages.keys())
    
    # For each metric, use the last substage that has a non-None value
    for substage in reversed(sorted_substages):
        if summary.get('area') is None and substages[substage]['area'] is not None:
            summary['area'] = substages[substage]['area']
        if summary.get('utilization') is None and substages[substage]['utilization'] is not None:
            summary['utilization'] = substages[substage]['utilization']
        if summary.get('density') is None and substages[substage]['density'] is not None:
            summary['density'] = substages[substage]['density']
        if summary.get('wns') is None and substages[substage]['wns'] is not None:
            summary['wns'] = substages[substage]['wns']
        if summary.get('tns') is None and substages[substage]['tns'] is not None:
            summary['tns'] = substages[substage]['tns']
        if summary.get('violations') is None and substages[substage]['violations'] is not None:
            summary['violations'] = substages[substage]['violations']
    
    result['summary'] = summary
    
    # SECOND PASS: Add "Report metrics" title to existing substages
    for i, line in enumerate(lines):
        report_metrics_match = re.search(r'Report metrics stage 2,\s+(.+?)\.\.\.', line)
        if report_metrics_match:
            substage_name = report_metrics_match.group(1).strip()
            
            # Just store the first line (the title) instead of the full report
            report_title = line.strip()
            
            # Map report name to substage (e.g., "floorplan final" -> "2_4_floorplan_pdn")
            substage_mapping = {
                'floorplan final': '2_4_floorplan_pdn',
                'floorplan': '2_1_floorplan',
                'io place': '2_2_floorplan_io',
                'macro place': '2_3_floorplan_macro',
                'pdn': '2_4_floorplan_pdn'
            }
            
            for key, stage_id in substage_mapping.items():
                if key in substage_name.lower():
                    # Only add report_metrics field if substage already exists
                    if stage_id in substages:
                        substages[stage_id]['report_metrics'] = report_title
                    break
    
    # Remove null values from the result
    result = remove_null_values(result)
    
    return result


def print_metrics_summary(metrics: Dict):
    """Pretty print the extracted metrics."""
    print("=" * 80)
    print("FLOORPLAN STAGE METRICS SUMMARY")
    print("=" * 80)
    
    # Summary
    if metrics.get('summary'):
        print("\n--- Overall Summary ---")
        summary = metrics['summary']
        if summary.get('area') is not None:
            print(f"  Area: {summary['area']:.2f} u^2")
        if summary.get('utilization') is not None:
            print(f"  Utilization: {summary['utilization']}%")
        if summary.get('density') is not None:
            print(f"  Density: {summary['density']}%")
        if summary.get('wns') is not None:
            print(f"  WNS: {summary['wns']:.3f}")
        if summary.get('tns') is not None:
            print(f"  TNS: {summary['tns']:.3f}")
        if summary.get('violations') is not None:
            print(f"  Violations: {summary['violations']}")
    
    # Substage details (sorted by substage name)
    if metrics.get('substages'):
        substages = metrics['substages']
        sorted_substage_names = sorted(substages.keys())
        
        for substage_name in sorted_substage_names:
            substage = substages[substage_name]
            print(f"\n--- Substage: {substage_name} ---")
            
            # Direct metrics at substage level
            print(f"  Metrics:")
            if substage.get('wns') is not None:
                print(f"    WNS: {substage['wns']:.3f}")
            if substage.get('tns') is not None:
                print(f"    TNS: {substage['tns']:.3f}")
            if substage.get('violations') is not None:
                print(f"    Violations: {substage['violations']}")
            if substage.get('area') is not None:
                print(f"    Area: {substage['area']:.2f} u^2")
            if substage.get('utilization') is not None:
                print(f"    Utilization: {substage['utilization']}%")
            if substage.get('density') is not None:
                print(f"    Density: {substage['density']}%")
            
            # Optimization processes
            if substage.get('optimization_process') and len(substage['optimization_process']) > 0:
                print(f"\n  Optimization Processes ({len(substage['optimization_process'])} total):")
                for idx, opt_process in enumerate(substage['optimization_process'], 1):
                    print(f"\n    [{idx}] {opt_process['type'].upper()}")
                    
                    if opt_process['type'] == 'repair_tie_fanout':
                        results = opt_process.get('results', {})
                        if results.get('tie_lo_instances'):
                            print(f"        Tie LO instances: {results['tie_lo_instances']}")
                        if results.get('tie_hi_instances'):
                            print(f"        Tie HI instances: {results['tie_hi_instances']}")
                    
                    elif opt_process['type'] == 'repair_timing':
                        print(f"        Iterations: {opt_process['iterations_count']}")
                        
                        start = opt_process.get('start_metrics', {})
                        end = opt_process.get('end_metrics', {})
                        if start.get('wns') is not None and end.get('wns') is not None:
                            print(f"        WNS: {start['wns']:.3f} → {end['wns']:.3f}")
                        if start.get('tns') is not None and end.get('tns') is not None:
                            print(f"        TNS: {start['tns']:.1f} → {end['tns']:.1f}")
                        if start.get('violation_end_points') is not None and end.get('violation_end_points') is not None:
                            print(f"        Violation End Points: {start['violation_end_points']} → {end['violation_end_points']}")
                        
                        summary = opt_process.get('summary', {})
                        if any(summary.values()):
                            changes = []
                            if summary.get('inserted_buffers'):
                                changes.append(f"+{summary['inserted_buffers']} buffers")
                            if summary.get('removed_buffers'):
                                changes.append(f"-{summary['removed_buffers']} buffers")
                            if summary.get('resized_instances'):
                                changes.append(f"{summary['resized_instances']} resized")
                            if summary.get('cloned_instances'):
                                changes.append(f"{summary['cloned_instances']} cloned")
                            if changes:
                                print(f"        Changes: {', '.join(changes)}")
                    
                    elif opt_process['type'] == 'macro_placement':
                        results = opt_process.get('results', {})
                        if results.get('num_macros'):
                            print(f"        Macros: {results['num_macros']}")
                        if results.get('area_macros'):
                            print(f"        Macro area: {results['area_macros']:.2f} u^2")
                        if results.get('num_std_cells'):
                            print(f"        Std cells: {results['num_std_cells']}")
                        if results.get('area_std_cells'):
                            print(f"        Std cell area: {results['area_std_cells']:.2f} u^2")
                        if results.get('design_utilization'):
                            print(f"        Design utilization: {results['design_utilization']:.1f}%")
                        if results.get('floorplan_utilization'):
                            print(f"        Floorplan utilization: {results['floorplan_utilization']:.1f}%")
                    
                    elif opt_process['type'] == 'tapcell_insertion':
                        results = opt_process.get('results', {})
                        if results.get('endcaps_inserted'):
                            print(f"        Endcaps: {results['endcaps_inserted']}")
                        if results.get('tapcells_inserted'):
                            print(f"        Tapcells: {results['tapcells_inserted']}")
            
            # Performance metrics
            perf_items = []
            if substage.get('elapsed_time'):
                perf_items.append(f"elapsed_time={substage['elapsed_time']}")
            if substage.get('peak_memory_kb'):
                perf_items.append(f"peak_memory={substage['peak_memory_kb']}KB")
            if substage.get('elapsed_s'):
                perf_items.append(f"elapsed={substage['elapsed_s']}s")
            if substage.get('memory_mb'):
                perf_items.append(f"memory={substage['memory_mb']}MB")
            
            if perf_items:
                print(f"\n  Performance: {', '.join(perf_items)}")
    
    print("\n" + "=" * 80)


# Example usage
if __name__ == "__main__":
    # Read the extracted floorplan log
    with open('test_extraction_filtered/ariane133_nan45_floorplan_extract.txt', 'r') as f:
        log_content = f.read()
    
    # Extract metrics
    metrics = extract_floorplan_metrics(log_content)
    
    # Print summary
    print_metrics_summary(metrics)
    
    # Save to JSON
    import json
    with open('floorplan_metrics_ariane.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\n\nMetrics saved to floorplan_metrics_ariane.json")
    
    # also test with gcdshy130 log
    with open('test_extraction_filtered/gcdshy130_floorplan_extract.txt', 'r') as f:
        log_content2 = f.read()
    metrics2 = extract_floorplan_metrics(log_content2)
    print_metrics_summary(metrics2)
    with open('floorplan_metrics_gcdshy130.json', 'w') as f:
        json.dump(metrics2, f, indent=2)
    print("\n\nMetrics saved to floorplan_metrics_gcdshy130.json")


