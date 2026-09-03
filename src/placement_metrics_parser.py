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
from typing import Dict, Optional


def remove_null_values(obj):
    """Recursively remove null/None values from dictionaries and lists."""
    if isinstance(obj, dict):
        return {k: remove_null_values(v) for k, v in obj.items() if v is not None and v != [] and v != {}}
    elif isinstance(obj, list):
        cleaned_list = [remove_null_values(item) for item in obj if item is not None]
        return [item for item in cleaned_list if item != {} and item != []]
    else:
        return obj


def extract_placement_metrics(log_content: str, design_name: Optional[str] = None, sub_design_name: Optional[str] = None) -> Dict:
    """
    Extract area, HPWL, utilization, timing, and density metrics from placement stage extracted log.
    Returns a structured dictionary with metrics organized by substages (e.g., 3_1_place_gp_skip_io, 3_2_place_iop).
    
    Optimization processes track: iteration count, start metrics, and end metrics (not all iterations).
    
    Args:
        log_content: The extracted placement log content as a string
        
    Returns:
        Dictionary containing:
        {
            'stage': 'placement',
            'substages': {
                '3_1_place_gp_skip_io': {
                    # Direct metrics at substage level
                    'hpwl': float,
                    'area': float,
                    'utilization': float,
                    'density': float,
                    'overflow': float,
                    
                    # Optimization processes in execution order
                    'optimization_process': [
                        {
                            'type': 'global_placement',
                            'command': str,
                            'iterations_count': int,
                            'start_metrics': {'overflow': float, 'hpwl': float},
                            'end_metrics': {'overflow': float, 'hpwl': float},
                            'results': {
                                'movable_instances': int,
                                'fixed_instances': int,
                                'num_nets': int,
                                'num_pins': int,
                                'core_area_um2': float,
                                'utilization': float,
                                'target_density': float,
                                'final_overflow': float,
                                'final_hpwl_um': float
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
                '3_2_place_iop': {...},
                '3_3_place_gp': {...},
                '3_4_place_resized': {...},
                '3_5_place_dp': {...}
            },
            'summary': {
                'area': float,
                'utilization': float,
                'hpwl': float,
                'density': float
            }
        }
    """
    lines = log_content.split('\n')
    
    result = {
        'stage': 'placement',
        'substages': {},
        'summary': {},
        'design_metrics_final': {}
    }

    if design_name is not None:
        result['design_name'] = design_name
    if sub_design_name is not None:
        result['sub_design_name'] = sub_design_name
    
    # Track current substage
    current_substage = None
    in_nesterov_table = False
    in_detailed_placement_table = False
    substages = {}
    
    # Helper function to initialize a substage
    def init_substage():
        return {
            # Direct metrics at substage level
            'hpwl': None,
            'area': None,  # Design area (for placement, this is typically reported at end)
            'utilization': None,
            'density': None,
            'overflow': None,
            
            # Optimization processes in execution order
            'optimization_process': [],
            
            # Report metrics section (from "Report metrics stage X, substage..." sections)
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
    
    for i, line in enumerate(lines):
        # Detect "Report metrics stage 3, substage..." sections
        report_metrics_match = re.search(r'Report metrics stage 3,\s+(.+?)\.\.\.', line)
        if report_metrics_match:
            substage_name = report_metrics_match.group(1).strip()
            
            # Just store the first line (the title) instead of the full report
            report_title = line.strip()
            
            # Map report name to substage (e.g., "global place" -> "3_3_place_gp")
            substage_mapping = {
                'global place': '3_3_place_gp',
                'resizer': '3_4_place_resized',
                'detailed place': '3_5_place_dp',
                'place_gp_skip_io': '3_1_place_gp_skip_io',
                'io place': '3_2_place_iop'
            }
            
            for key, stage_id in substage_mapping.items():
                if key in substage_name.lower():
                    # Only add report_metrics field to existing substages
                    if stage_id in substages:
                        substages[stage_id]['report_metrics'] = report_title
                    break
        
        # Detect substages - look for patterns like "3_1_place_gp_skip_io", "3_2_place_iop" etc.
        stage_match = re.search(r'(3_\d+_\w+)', line)
        if stage_match and 'Running' in line:
            current_substage = stage_match.group(1)
            if current_substage not in substages:
                substages[current_substage] = init_substage()
            in_nesterov_table = False
            in_detailed_placement_table = False
            current_optimization = None
        
        # Also check for stage in log summary lines
        stage_summary_match = re.search(r'^[^\]]*\]\s+(3_\d+_\w+)\s+\d+\s+\d+', line)
        if stage_summary_match:
            stage_name = stage_summary_match.group(1)
            if stage_name not in substages:
                substages[stage_name] = init_substage()
        
        # If we have a current substage, extract metrics
        if current_substage and current_substage in substages:
            # Detect global placement operation - avoid "Took X seconds" message
            if 'global_placement' in line and any(kw in line for kw in ['-density', '-skip_io', '-timing_driven', '-routability_driven']) and 'Took' not in line and 'seconds:' not in line:
                current_optimization = {
                    'type': 'global_placement',
                    'command': line.split('] ')[-1].strip() if ']' in line else line.strip(),
                    'iterations_count': 0,
                    'start_metrics': {
                        'overflow': None,
                        'hpwl': None
                    },
                    'end_metrics': {
                        'overflow': None,
                        'hpwl': None
                    },
                    'results': {
                        'movable_instances': None,
                        'fixed_instances': None,
                        'num_nets': None,
                        'num_pins': None,
                        'core_area_um2': None,
                        'fixed_instances_area_um2': None,
                        'movable_instances_area_um2': None,
                        'standard_cells_area_um2': None,
                        'utilization': None,
                        'target_density': None,
                        'final_overflow': None,
                        'final_hpwl_um': None
                    }
                }
                substages[current_substage]['optimization_process'].append(current_optimization)
                in_nesterov_table = False
            
            # Detect IO placement operation - avoid "Took X seconds" message
            if 'place_pins' in line and 'Took' not in line and 'seconds:' not in line:
                current_optimization = {
                    'type': 'io_placement',
                    'command': line.split('] ')[-1].strip() if ']' in line else line.strip(),
                    'results': {
                        'io_nets_hpwl_um': None
                    }
                }
                substages[current_substage]['optimization_process'].append(current_optimization)
            
            # Capture IO placement HPWL (I/O nets only, not overall design HPWL)
            match = re.search(r'I/O nets HPWL:\s+([\d.]+)\s+um', line)
            if match and current_optimization and current_optimization['type'] == 'io_placement':
                hpwl = float(match.group(1))
                current_optimization['results']['io_nets_hpwl_um'] = hpwl
                # Note: Do NOT set substage-level hpwl here as this is I/O nets only
            
            # Detect buffer removal/insertion
            match = re.search(r'Removed\s+(\d+)\s+buffers', line)
            if match:
                current_optimization = {
                    'type': 'buffer_removal',
                    'results': {
                        'removed_buffers': int(match.group(1))
                    }
                }
                substages[current_substage]['optimization_process'].append(current_optimization)
            
            match = re.search(r'Inserted\s+(\d+)\s+(\w+)\s+input buffers', line)
            if match:
                current_optimization = {
                    'type': 'buffer_insertion',
                    'results': {
                        'input_buffers': int(match.group(1)),
                        'buffer_type': match.group(2),
                        'output_buffers': None
                    }
                }
                substages[current_substage]['optimization_process'].append(current_optimization)
            
            match = re.search(r'Inserted\s+(\d+)\s+(\w+)\s+output buffers', line)
            if match and current_optimization and current_optimization['type'] == 'buffer_insertion':
                current_optimization['results']['output_buffers'] = int(match.group(1))
            
            # Global placement metrics
            if current_optimization and current_optimization['type'] == 'global_placement':
                match = re.search(r'Movable instances:\s+(\d+)', line)
                if match:
                    current_optimization['results']['movable_instances'] = int(match.group(1))
                
                match = re.search(r'Fixed instances:\s+(\d+)', line)
                if match:
                    current_optimization['results']['fixed_instances'] = int(match.group(1))
                
                match = re.search(r'Number of nets:\s+(\d+)', line)
                if match:
                    current_optimization['results']['num_nets'] = int(match.group(1))
                
                match = re.search(r'Number of pins:\s+(\d+)', line)
                if match:
                    current_optimization['results']['num_pins'] = int(match.group(1))
                
                match = re.search(r'Core area:\s+([\d.]+)\s+um\^2', line)
                if match:
                    core_area = float(match.group(1))
                    current_optimization['results']['core_area_um2'] = core_area
                    # Core area is the total area - store at substage level as 'area'
                    substages[current_substage]['area'] = core_area
                
                match = re.search(r'Fixed instances area:\s+([\d.]+)\s+um\^2', line)
                if match:
                    fixed_area = float(match.group(1))
                    current_optimization['results']['fixed_instances_area_um2'] = fixed_area
                
                match = re.search(r'Movable instances area:\s+([\d.]+)\s+um\^2', line)
                if match:
                    movable_area = float(match.group(1))
                    current_optimization['results']['movable_instances_area_um2'] = movable_area
                
                match = re.search(r'Standard cells area:\s+([\d.]+)\s+um\^2', line)
                if match:
                    std_cell_area = float(match.group(1))
                    current_optimization['results']['standard_cells_area_um2'] = std_cell_area
                
                match = re.search(r'Utilization:\s+([\d.]+)\s+%', line)
                if match:
                    util = float(match.group(1))
                    current_optimization['results']['utilization'] = util
                    substages[current_substage]['utilization'] = util
                    substages[current_substage]['density'] = util
                
                match = re.search(r'Placement target density:\s+([\d.]+)', line)
                if match:
                    current_optimization['results']['target_density'] = float(match.group(1)) * 100
            
            # Detect Nesterov optimization table start
            if 'Iteration' in line and 'Overflow' in line and 'HPWL' in line:
                in_nesterov_table = True
                continue
            
            # Parse Nesterov optimization table rows
            if in_nesterov_table and current_optimization and current_optimization['type'] == 'global_placement':
                if '---' in line:
                    continue
                
                # Check for the actual iteration count message
                # [INFO GPL-1001] Global placement finished at iteration 511
                match_finished = re.search(r'\[INFO GPL-1001\].*at iteration\s+(\d+)', line)
                if match_finished:
                    # Use the actual iteration count from the finish message
                    current_optimization['iterations_count'] = int(match_finished.group(1)) + 1  # +1 because iterations are 0-indexed
                    in_nesterov_table = False
                    continue
                
                # End of table - check for markers
                if 'INFO GPL-' in line and '|' not in line:
                    in_nesterov_table = False
                    continue
                
                # Parse iteration data - look for the pipe-separated values
                match = re.search(
                    r'^\s*\[?[^\]]*\]?\s*(\d+)\s+\|\s+([\d.]+)\s+\|\s+([\d.e+\-]+)\s+\|',
                    line
                )
                if match:
                    iter_num = int(match.group(1))
                    overflow = float(match.group(2))
                    hpwl = float(match.group(3))
                    
                    # Don't count iterations here - we'll use the GPL-1001 finish message for accuracy
                    # (The table only shows every 10th iteration, not all iterations)
                    
                    # Capture start metrics (first iteration for this optimization)
                    if current_optimization['start_metrics']['overflow'] is None:
                        current_optimization['start_metrics']['overflow'] = overflow
                        current_optimization['start_metrics']['hpwl'] = hpwl
                    
                    # Always update end metrics (last iteration will be final)
                    current_optimization['end_metrics']['overflow'] = overflow
                    current_optimization['end_metrics']['hpwl'] = hpwl
                    
                    # Update substage-level metrics with final values
                    substages[current_substage]['overflow'] = overflow
                    substages[current_substage]['hpwl'] = hpwl
                    current_optimization['results']['final_overflow'] = overflow
                    current_optimization['results']['final_hpwl_um'] = hpwl
            
            # Detect repair_design operation
            if 'repair_design' in line and '-verbose' in line and 'Took' not in line:
                in_detailed_placement_table = False
                current_optimization = {
                    'type': 'repair_design',
                    'command': line.split('] ')[-1].strip() if ']' in line else line.strip(),
                    'iterations_count': 0,
                    'start_metrics': {},
                    'end_metrics': {},
                    'summary': {
                        'resized_instances': None,
                        'inserted_buffers': None,
                        'inserted_buffer_nets': None
                    }
                }
                substages[current_substage]['optimization_process'].append(current_optimization)
            
            # Capture repair_design summary
            match = re.search(r'Resized\s+(\d+)\s+instances', line)
            if match and current_optimization and current_optimization['type'] == 'repair_design':
                current_optimization['summary']['resized_instances'] = int(match.group(1))
            
            match = re.search(r'Inserted\s+(\d+)\s+buffers in\s+(\d+)\s+nets', line)
            if match and current_optimization and current_optimization['type'] == 'repair_design':
                current_optimization['summary']['inserted_buffers'] = int(match.group(1))
                current_optimization['summary']['inserted_buffer_nets'] = int(match.group(2))
            
            # Detect detail placement
            if 'Placement Analysis' in line:
                current_optimization = {
                    'type': 'detailed_placement',
                    'results': {
                        'total_displacement_u': None,
                        'average_displacement_u': None,
                        'max_displacement_u': None,
                        'original_hpwl_u': None,
                        'legalized_hpwl_u': None,
                        'delta_hpwl_percent': None,
                        'final_hpwl_u': None,
                        'final_delta_hpwl_percent': None
                    }
                }
                substages[current_substage]['optimization_process'].append(current_optimization)
            
            # Capture placement analysis metrics
            if current_optimization and current_optimization['type'] == 'detailed_placement':
                match = re.search(r'total displacement\s+([\d.]+)\s+u', line)
                if match:
                    current_optimization['results']['total_displacement_u'] = float(match.group(1))
                
                match = re.search(r'average displacement\s+([\d.]+)\s+u', line)
                if match:
                    current_optimization['results']['average_displacement_u'] = float(match.group(1))
                
                match = re.search(r'max displacement\s+([\d.]+)\s+u', line)
                if match:
                    current_optimization['results']['max_displacement_u'] = float(match.group(1))
                
                match = re.search(r'original HPWL\s+([\d.]+)\s+u', line)
                if match:
                    current_optimization['results']['original_hpwl_u'] = float(match.group(1))
                
                match = re.search(r'legalized HPWL\s+([\d.]+)\s+u', line)
                if match:
                    hpwl = float(match.group(1))
                    current_optimization['results']['legalized_hpwl_u'] = hpwl
                    substages[current_substage]['hpwl'] = hpwl
                
                match = re.search(r'delta HPWL\s+([\d.]+)\s+%', line)
                if match:
                    current_optimization['results']['delta_hpwl_percent'] = float(match.group(1))
                
                # Final HPWL after detailed improvement
                match = re.search(r'Final HPWL\s+([\d.]+)\s+u', line)
                if match:
                    hpwl = float(match.group(1))
                    current_optimization['results']['final_hpwl_u'] = hpwl
                    substages[current_substage]['hpwl'] = hpwl
                
                match = re.search(r'Delta HPWL\s+([+\-\d.]+)\s+%', line)
                if match:
                    current_optimization['results']['final_delta_hpwl_percent'] = float(match.group(1))
            
            # Design area and utilization
            match = re.search(r'Design area\s+([\d.]+)\s+u\^2\s+(\d+)%\s+utilization', line)
            if match:
                area = float(match.group(1))
                util = int(match.group(2))
                substages[current_substage]['area'] = area
                substages[current_substage]['utilization'] = util
                substages[current_substage]['density'] = util
            
            # Performance metrics
            match = re.search(r'Elapsed time:\s+([\d:.]+)\[h:\]min:sec', line)
            if match:
                substages[current_substage]['elapsed_time'] = match.group(1)
            
            match = re.search(r'Peak memory:\s+(\d+)KB', line)
            if match:
                substages[current_substage]['peak_memory_kb'] = int(match.group(1))
        
        # Also parse log summary lines (at the end) - these can reference any substage
        log_summary_match = re.search(r'^[^\]]*\]\s+(3_\d+_\w+)\s+(\d+)\s+(\d+)\s+([a-f0-9]+)', line)
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
    
    # Get the list of substages sorted
    sorted_substages = sorted(substages.keys())
    
    # For each metric, use the last substage that has a non-None value
    for substage in reversed(sorted_substages):
        if summary.get('area') is None and substages[substage]['area'] is not None:
            summary['area'] = substages[substage]['area']
        if summary.get('utilization') is None and substages[substage]['utilization'] is not None:
            summary['utilization'] = substages[substage]['utilization']
        if summary.get('density') is None and substages[substage]['density'] is not None:
            summary['density'] = substages[substage]['density']
        if summary.get('hpwl') is None and substages[substage]['hpwl'] is not None:
            summary['hpwl'] = substages[substage]['hpwl']
        if summary.get('overflow') is None and substages[substage]['overflow'] is not None:
            summary['overflow'] = substages[substage]['overflow']
    
    result['summary'] = summary
    
    # Parse placeopt-related test metrics from final stage (if present in log)
    # These are test metrics that start with "placeopt__"
    for line in lines:
        test_match = re.search(r'\[INFO\]\s+(placeopt__[a-z_]+)(?:__([a-z_]+))?(?:__([a-z_]+))?(?:__([a-z_]+))?\s+pass test:\s+([\d.-]+)\s+(==|<=|>=)\s+([\d.-]+)', line)
        if test_match:
            groups = test_match.groups()
            # Build metric name from non-None groups (excluding the last 3 which are value, operator, threshold)
            metric_parts = [g for g in groups[:-3] if g is not None]
            metric_name = '__'.join(metric_parts)
            actual_value = float(groups[-3])
            
            # Store in design_metrics_final
            result['design_metrics_final'][metric_name] = actual_value
    
    # Remove null values from the result
    result = remove_null_values(result)
    
    return result


def print_metrics_summary(metrics: Dict):
    """Pretty print the extracted metrics."""
    print("=" * 80)
    print("PLACEMENT STAGE METRICS SUMMARY")
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
        if summary.get('hpwl') is not None:
            print(f"  HPWL: {summary['hpwl']:.2f} um")
        if summary.get('overflow') is not None:
            print(f"  Overflow: {summary['overflow']:.4f}")
    
    # Substage details (sorted by substage name)
    if metrics.get('substages'):
        substages = metrics['substages']
        sorted_substage_names = sorted(substages.keys())
        
        for substage_name in sorted_substage_names:
            substage = substages[substage_name]
            print(f"\n--- Substage: {substage_name} ---")
            
            # Direct metrics at substage level
            print(f"  Metrics:")
            if substage.get('hpwl') is not None:
                print(f"    HPWL: {substage['hpwl']:.2f} um")
            if substage.get('overflow') is not None:
                print(f"    Overflow: {substage['overflow']:.4f}")
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
                    
                    if opt_process['type'] == 'global_placement':
                        print(f"        Iterations: {opt_process['iterations_count']}")
                        
                        start = opt_process.get('start_metrics', {})
                        end = opt_process.get('end_metrics', {})
                        if start.get('overflow') is not None and end.get('overflow') is not None:
                            print(f"        Overflow: {start['overflow']:.4f} → {end['overflow']:.4f}")
                        if start.get('hpwl') is not None and end.get('hpwl') is not None:
                            print(f"        HPWL: {start['hpwl']:.2e} → {end['hpwl']:.2e}")
                        
                        results = opt_process.get('results', {})
                        if results.get('movable_instances'):
                            print(f"        Movable instances: {results['movable_instances']}")
                        if results.get('utilization'):
                            print(f"        Utilization: {results['utilization']:.2f}%")
                        if results.get('target_density'):
                            print(f"        Target density: {results['target_density']:.2f}%")
                    
                    elif opt_process['type'] == 'io_placement':
                        results = opt_process.get('results', {})
                        if results.get('io_nets_hpwl_um'):
                            print(f"        I/O nets HPWL: {results['io_nets_hpwl_um']:.2f} um")
                    
                    elif opt_process['type'] == 'buffer_removal':
                        results = opt_process.get('results', {})
                        if results.get('removed_buffers'):
                            print(f"        Removed buffers: {results['removed_buffers']}")
                    
                    elif opt_process['type'] == 'buffer_insertion':
                        results = opt_process.get('results', {})
                        if results.get('input_buffers'):
                            print(f"        Input buffers: {results['input_buffers']}")
                        if results.get('output_buffers'):
                            print(f"        Output buffers: {results['output_buffers']}")
                        if results.get('buffer_type'):
                            print(f"        Buffer type: {results['buffer_type']}")
                    
                    elif opt_process['type'] == 'repair_design':
                        print(f"        Iterations: {opt_process['iterations_count']}")
                        
                        summary = opt_process.get('summary', {})
                        if summary.get('resized_instances'):
                            print(f"        Resized instances: {summary['resized_instances']}")
                        if summary.get('inserted_buffers'):
                            print(f"        Inserted buffers: {summary['inserted_buffers']} in {summary['inserted_buffer_nets']} nets")
                    
                    elif opt_process['type'] == 'detailed_placement':
                        results = opt_process.get('results', {})
                        if results.get('total_displacement_u'):
                            print(f"        Total displacement: {results['total_displacement_u']:.1f} u")
                        if results.get('average_displacement_u'):
                            print(f"        Average displacement: {results['average_displacement_u']:.1f} u")
                        if results.get('max_displacement_u'):
                            print(f"        Max displacement: {results['max_displacement_u']:.1f} u")
                        if results.get('original_hpwl_u'):
                            print(f"        Original HPWL: {results['original_hpwl_u']:.1f} u")
                        if results.get('final_hpwl_u'):
                            print(f"        Final HPWL: {results['final_hpwl_u']:.1f} u")
                        if results.get('final_delta_hpwl_percent'):
                            print(f"        Delta HPWL: {results['final_delta_hpwl_percent']:.1f}%")
            
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
    # Read the extracted placement log
    with open('test_extraction_filtered/ariane133_nan45_placement_extract.txt', 'r') as f:
        log_content = f.read()
    
    # Extract metrics
    metrics = extract_placement_metrics(log_content)
    
    # Print summary
    print_metrics_summary(metrics)
    
    # Save to JSON
    import json
    with open('placement_metrics_ariane.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\n\nMetrics saved to placement_metrics_ariane.json")
    
    # also test with gcdshy130 log
    with open('test_extraction_filtered/gcdshy130_placement_extract.txt', 'r') as f:
        log_content2 = f.read()
    metrics2 = extract_placement_metrics(log_content2)
    print_metrics_summary(metrics2)
    with open('placement_metrics_gcdshy130.json', 'w') as f:
        json.dump(metrics2, f, indent=2)
    print("\n\nMetrics saved to placement_metrics_gcdshy130.json")


