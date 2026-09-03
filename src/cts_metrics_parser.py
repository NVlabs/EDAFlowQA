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


def extract_cts_metrics(log_content: str, design_name: Optional[str] = None, sub_design_name: Optional[str] = None) -> Dict:
    """
    Extract area, HPWL, clock sink wirelength, timing, and density metrics from CTS stage extracted log.
    Returns a structured dictionary with metrics organized by substages (e.g., 4_1_cts, 4_2_cts).
    
    Note: 'clock_sink_wirelength_um' is the average clock tree sink wire length, NOT the routed 
    wirelength (which is 0 before routing stage). This is specific to the clock tree.
    
    Optimization processes track: iteration count, start metrics, and end metrics (not all iterations).
    
    Args:
        log_content: The extracted CTS log content as a string
        
    Returns:
        Dictionary containing:
        {
            'stage': 'cts',
            'substages': {
                '4_1_cts': {
                    # Direct metrics at substage level
                    'wns': float,
                    'tns': float,
                    'clock_sink_wirelength_um': float,  # Clock tree sink wirelength (NOT routed wirelength)
                    'hpwl': float,
                    'area': float,
                    'utilization': int,
                    'density': int,
                    'displacement': float,
                    'violations': int,
                    
                    # Optimization processes in execution order
                    'optimization_process': [
                        {
                            'type': 'clock_tree_synthesis',
                            'command': str,
                            'results': {
                                'total_sinks': int,
                                'clock_buffers_created': int,
                                'min_buffer_depth': int,
                                'max_buffer_depth': int,
                                'avg_sink_wirelength_um': float,
                                'placement_total_displacement_u': float,
                                'placement_avg_displacement_u': float,
                                'placement_max_displacement_u': float,
                                'hpwl_before': float,
                                'hpwl_after': float,
                                'clock_nets': [
                                    {
                                        'name': str,
                                        'sinks': int,
                                        'leaf_buffers': int,
                                        'avg_sink_wire_length_um': float,
                                        'path_depth_min': int,
                                        'path_depth_max': int,
                                        'leaf_load_cells': int
                                    }
                                ]
                            }
                        },
                        {
                            'type': 'repair_timing',
                            'command': str,
                            'iterations_count': int,  # Total number of iterations
                            'start_metrics': {        # First iteration
                                'wns': float,
                                'tns': float,
                                'violation_end_points': int
                            },
                            'end_metrics': {          # Final iteration
                                'wns': float,
                                'tns': float,
                                'violation_end_points': int
                            },
                            'summary': {
                                'removed_buffers': int,
                                'inserted_buffers': int,
                                'hold_buffers_inserted': int,
                                'resized_instances': int,
                                'cloned_instances': int
                            },
                            'placement_impact': {
                                'total_displacement_u': float,
                                'avg_displacement_u': float,
                                'max_displacement_u': float,
                                'hpwl_before': float,
                                'hpwl_after': float
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
                '4_2_cts': {...},
                ...
            },
            'summary': {
                'area': float,
                'utilization': int,
                'hpwl': float,
                'clock_sink_wirelength_um': float,  # Clock tree sink wirelength
                'wns': float,
                'tns': float,
                'violations': int,
                'density': int,
                'displacement': float
            }
        }
    """
    lines = log_content.split('\n')
    
    result = {
        'stage': 'cts',
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
    in_repair_timing_table = False
    in_placement_analysis = False
    substages = {}
    
    # Helper function to initialize a substage
    def init_substage():
        return {
            # Direct metrics at substage level
            'wns': None,
            'tns': None,
            'clock_sink_wirelength_um': None,  # Clock tree sink wirelength (not routed wirelength)
            'hpwl': None,
            'area': None,
            'utilization': None,
            'density': None,
            'displacement': None,
            'violations': None,
            
            # Optimization processes in execution order
            'optimization_process': [],
            
            # Report metrics section (from "Report metrics stage 4, substage..." sections)
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
    current_clock_net = None  # Track current clock net being parsed
    
    for i, line in enumerate(lines):
        # Detect "Report metrics stage 4, substage..." sections
        report_metrics_match = re.search(r'Report metrics stage 4,\s+(.+?)\.\.\.', line)
        if report_metrics_match:
            substage_name = report_metrics_match.group(1).strip()
            
            # Just store the first line (the title) instead of the full report
            report_title = line.strip()
            
            # Map report name to substage (e.g., "cts final" -> "4_1_cts")
            substage_mapping = {
                'cts final': '4_1_cts',
                'cts': '4_1_cts'
            }
            
            for key, stage_id in substage_mapping.items():
                if key in substage_name.lower():
                    # Only add report_metrics field to existing substages
                    if stage_id in substages:
                        substages[stage_id]['report_metrics'] = report_title
                    break
        
        # Detect substages - look for patterns like "4_1_cts", "4_2_cts" etc.
        stage_match = re.search(r'(4_\d+_\w+)', line)
        if stage_match and 'Running' in line:
            current_substage = stage_match.group(1)
            if current_substage not in substages:
                substages[current_substage] = init_substage()
            in_repair_timing_table = False
            current_optimization = None
        
        # Also check for stage in log summary lines
        stage_summary_match = re.search(r'^[^\]]*\]\s+(4_\d+_\w+)\s+\d+\s+\d+', line)
        if stage_summary_match:
            stage_name = stage_summary_match.group(1)
            if stage_name not in substages:
                substages[stage_name] = init_substage()
        
        # Detect placement analysis sections
        if 'Placement Analysis' in line:
            in_placement_analysis = True
            continue
        
        # If we have a current substage, extract metrics
        if current_substage and current_substage in substages:
            # Detect clock tree synthesis operation - avoid "Took X seconds" message
            if 'clock_tree_synthesis' in line and any(kw in line for kw in ['-sink_clustering_enable', '-repair_clock_nets']) and 'Took' not in line and 'seconds:' not in line:
                current_optimization = {
                    'type': 'clock_tree_synthesis',
                    'command': line.split('] ')[-1].strip() if ']' in line else line.strip(),
                    'results': {
                        'total_sinks': None,
                        'clock_buffers_created': None,
                        'min_buffer_depth': None,
                        'max_buffer_depth': None,
                        'avg_sink_wirelength_um': None,
                        'placement_total_displacement_u': None,
                        'placement_avg_displacement_u': None,
                        'placement_max_displacement_u': None,
                        'hpwl_before': None,
                        'hpwl_after': None,
                        'clock_nets': []  # Clock nets information for this command
                    }
                }
                substages[current_substage]['optimization_process'].append(current_optimization)
            
            # Clock Tree Synthesis metrics
            if current_optimization and current_optimization['type'] == 'clock_tree_synthesis':
                # Total number of sinks
                match = re.search(r'Total number of sinks:\s*(\d+)', line)
                if match:
                    current_optimization['results']['total_sinks'] = int(match.group(1))
                
                # Clock buffers created
                match = re.search(r'Created\s+(\d+)\s+clock buffers', line)
                if match:
                    current_optimization['results']['clock_buffers_created'] = int(match.group(1))
                
                # Buffer path depth
                match = re.search(r'Minimum number of buffers in the clock path:\s*(\d+)', line)
                if match:
                    current_optimization['results']['min_buffer_depth'] = int(match.group(1))
                
                match = re.search(r'Maximum number of buffers in the clock path:\s*(\d+)', line)
                if match:
                    current_optimization['results']['max_buffer_depth'] = int(match.group(1))
                
                # Wire length
                match = re.search(r'Average sink wire length\s+([\d.]+)\s*um', line)
                if match:
                    wire_len = float(match.group(1))
                    current_optimization['results']['avg_sink_wirelength_um'] = wire_len
                    # Update substage-level metric (use latest) - this is CLOCK SINK wirelength
                    substages[current_substage]['clock_sink_wirelength_um'] = wire_len
            
            # Extract clock net information (CTS-0098 to CTS-0207)
            # [INFO CTS-0098] Clock net "clk_regs"
            match = re.search(r'\[INFO CTS-0098\]\s+Clock net\s+"([^"]+)"', line)
            if match:
                current_clock_net = {
                    'name': match.group(1),
                    'sinks': None,
                    'leaf_buffers': None,
                    'avg_sink_wire_length_um': None,
                    'path_depth_min': None,
                    'path_depth_max': None,
                    'leaf_load_cells': None
                }
                # Add to the current clock_tree_synthesis optimization if it exists
                if current_optimization and current_optimization['type'] == 'clock_tree_synthesis':
                    current_optimization['results']['clock_nets'].append(current_clock_net)
            
            # [INFO CTS-0099]  Sinks 38
            if current_clock_net:
                match = re.search(r'\[INFO CTS-0099\].*Sinks\s+(\d+)', line)
                if match:
                    current_clock_net['sinks'] = int(match.group(1))
                
                # [INFO CTS-0100]  Leaf buffers 0
                match = re.search(r'\[INFO CTS-0100\].*Leaf buffers\s+(\d+)', line)
                if match:
                    current_clock_net['leaf_buffers'] = int(match.group(1))
                
                # [INFO CTS-0101]  Average sink wire length 114.94 um
                match = re.search(r'\[INFO CTS-0101\].*Average sink wire length\s+([\d.]+)\s*um', line)
                if match:
                    current_clock_net['avg_sink_wire_length_um'] = float(match.group(1))
                
                # [INFO CTS-0102]  Path depth 2 - 2
                match = re.search(r'\[INFO CTS-0102\].*Path depth\s+(\d+)\s*-\s*(\d+)', line)
                if match:
                    current_clock_net['path_depth_min'] = int(match.group(1))
                    current_clock_net['path_depth_max'] = int(match.group(2))
                
                # [INFO CTS-0207]  Leaf load cells 3
                match = re.search(r'\[INFO CTS-0207\].*Leaf load cells\s+(\d+)', line)
                if match:
                    current_clock_net['leaf_load_cells'] = int(match.group(1))
                    # Reset current_clock_net after last field
                    current_clock_net = None
            
            # Detect repair timing operation - look for the actual command, not the "Took X seconds" message
            if 'repair_timing' in line and '-verbose' in line and 'Took' not in line and 'seconds:' not in line:
                # Create a temporary optimization entry - we'll only add it if we see a timing table
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
                    'placement_impact': {
                        'total_displacement_u': None,
                        'avg_displacement_u': None,
                        'max_displacement_u': None,
                        'hpwl_before': None,
                        'hpwl_after': None
                    },
                    '_has_data': False  # Track if we actually got timing data
                }
                # Don't add to optimization_process yet - wait until we see actual data
                in_repair_timing_table = False
            
            # Placement Analysis - extract metrics and update substage
            if in_placement_analysis:
                match = re.search(r'total displacement\s+([\d.]+)\s*u', line)
                if match:
                    total_displacement = float(match.group(1))
                    if current_optimization:
                        if current_optimization['type'] == 'clock_tree_synthesis':
                            current_optimization['results']['placement_total_displacement_u'] = total_displacement
                        elif current_optimization['type'] == 'repair_timing':
                            current_optimization['placement_impact']['total_displacement_u'] = total_displacement
                    substages[current_substage]['displacement'] = total_displacement
                
                match = re.search(r'average displacement\s+([\d.]+)\s*u', line)
                if match:
                    avg_displacement = float(match.group(1))
                    if current_optimization:
                        if current_optimization['type'] == 'clock_tree_synthesis':
                            current_optimization['results']['placement_avg_displacement_u'] = avg_displacement
                        elif current_optimization['type'] == 'repair_timing':
                            current_optimization['placement_impact']['avg_displacement_u'] = avg_displacement
                
                match = re.search(r'max displacement\s+([\d.]+)\s*u', line)
                if match:
                    max_displacement = float(match.group(1))
                    if current_optimization:
                        if current_optimization['type'] == 'clock_tree_synthesis':
                            current_optimization['results']['placement_max_displacement_u'] = max_displacement
                        elif current_optimization['type'] == 'repair_timing':
                            current_optimization['placement_impact']['max_displacement_u'] = max_displacement
                
                match = re.search(r'original HPWL\s+([\d.]+)\s*u', line)
                if match:
                    hpwl_before = float(match.group(1))
                    if current_optimization:
                        if current_optimization['type'] == 'clock_tree_synthesis':
                            current_optimization['results']['hpwl_before'] = hpwl_before
                        elif current_optimization['type'] == 'repair_timing':
                            current_optimization['placement_impact']['hpwl_before'] = hpwl_before
                
                match = re.search(r'legalized HPWL\s+([\d.]+)\s*u', line)
                if match:
                    hpwl = float(match.group(1))
                    if current_optimization:
                        if current_optimization['type'] == 'clock_tree_synthesis':
                            current_optimization['results']['hpwl_after'] = hpwl
                        elif current_optimization['type'] == 'repair_timing':
                            current_optimization['placement_impact']['hpwl_after'] = hpwl
                    # Update substage-level metric
                    substages[current_substage]['hpwl'] = hpwl
                
                match = re.search(r'delta HPWL\s+([\d.]+)\s*%', line)
                if match:
                    in_placement_analysis = False
            
            # Repair timing table
            if 'Iter' in line and 'WNS' in line and 'TNS' in line and current_optimization and current_optimization['type'] == 'repair_timing':
                in_repair_timing_table = True
                # Now we know repair_timing has actual timing data, add it to the list
                if not current_optimization.get('_has_data'):
                    current_optimization['_has_data'] = True
                    substages[current_substage]['optimization_process'].append(current_optimization)
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
            
            # Repair timing summary messages - create a separate optimization_process entry
            # [INFO RSZ-0059] Removed 4 buffers.
            match = re.search(r'\[INFO RSZ-0059\]\s+Removed\s+(\d+)\s+buffers?', line)
            if match and current_substage and current_substage in substages:
                # Create a new summary entry
                summary_entry = {
                    'type': 'repair_timing_summary',
                    'removed_buffers': int(match.group(1)),
                    'inserted_buffers': None,
                    'hold_buffers_inserted': None,
                    'resized_instances': None,
                    'cloned_instances': None
                }
                substages[current_substage]['optimization_process'].append(summary_entry)
                current_optimization = summary_entry  # Update reference for subsequent lines
            
            # [INFO RSZ-0040] Inserted 42 buffers.
            match = re.search(r'\[INFO RSZ-0040\]\s+Inserted\s+(\d+)\s+buffers?', line)
            if match and current_optimization and current_optimization.get('type') == 'repair_timing_summary':
                current_optimization['inserted_buffers'] = int(match.group(1))
            
            # [INFO RSZ-0032] Inserted 21 hold buffers.
            match = re.search(r'\[INFO RSZ-0032\]\s+Inserted\s+(\d+)\s+hold\s+buffers?', line)
            if match and current_substage and current_substage in substages:
                hold_buffers = int(match.group(1))
                # Check if we have a current repair_timing_summary to update
                if current_optimization and current_optimization.get('type') == 'repair_timing_summary':
                    current_optimization['hold_buffers_inserted'] = hold_buffers
                else:
                    # Create a new summary entry just for hold buffers
                    summary_entry = {
                        'type': 'repair_timing_summary',
                        'removed_buffers': None,
                        'inserted_buffers': None,
                        'hold_buffers_inserted': hold_buffers,
                        'resized_instances': None,
                        'cloned_instances': None
                    }
                    substages[current_substage]['optimization_process'].append(summary_entry)
                    current_optimization = summary_entry
            
            # [INFO RSZ-0051] Resized 127 instances: ...
            match = re.search(r'\[INFO RSZ-0051\]\s+Resized\s+(\d+)\s+instances', line)
            if match and current_optimization and current_optimization.get('type') == 'repair_timing_summary':
                current_optimization['resized_instances'] = int(match.group(1))
            
            # [INFO RSZ-0049] Cloned 2 instances.
            match = re.search(r'\[INFO RSZ-0049\]\s+Cloned\s+(\d+)\s+instances?', line)
            if match and current_optimization and current_optimization.get('type') == 'repair_timing_summary':
                current_optimization['cloned_instances'] = int(match.group(1))
            
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
        log_summary_match = re.search(r'^[^\]]*\]\s+(4_\d+_\w+)\s+(\d+)\s+(\d+)\s+([a-f0-9]+)', line)
        if log_summary_match:
            stage_name, elapsed_s, memory_mb, sha1 = log_summary_match.groups()
            if stage_name in substages:
                substages[stage_name]['elapsed_s'] = int(elapsed_s)
                substages[stage_name]['memory_mb'] = int(memory_mb)
                substages[stage_name]['sha1sum'] = sha1
    
    # Clean up temporary tracking fields
    for substage_data in substages.values():
        for opt in substage_data.get('optimization_process', []):
            if '_has_data' in opt:
                del opt['_has_data']
    
    # Store substages
    result['substages'] = substages
    
    # Create summary with key metrics from the last substage or aggregated from all substages
    summary = {}
    
    # Get the list of substages sorted (4_1_cts, 4_2_cts, etc.)
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
        if summary.get('clock_sink_wirelength_um') is None and substages[substage]['clock_sink_wirelength_um'] is not None:
            summary['clock_sink_wirelength_um'] = substages[substage]['clock_sink_wirelength_um']
        if summary.get('wns') is None and substages[substage]['wns'] is not None:
            summary['wns'] = substages[substage]['wns']
        if summary.get('tns') is None and substages[substage]['tns'] is not None:
            summary['tns'] = substages[substage]['tns']
        if summary.get('violations') is None and substages[substage]['violations'] is not None:
            summary['violations'] = substages[substage]['violations']
        if summary.get('displacement') is None and substages[substage]['displacement'] is not None:
            summary['displacement'] = substages[substage]['displacement']
    
    result['summary'] = summary
    
    # Parse CTS-related test metrics from final stage (if present in log)
    # These are test metrics that start with "cts__"
    for line in lines:
        test_match = re.search(r'\[INFO\]\s+(cts__[a-z_]+)(?:__([a-z_]+))?(?:__([a-z_]+))?(?:__([a-z_]+))?\s+pass test:\s+([\d.-]+)\s+(==|<=|>=)\s+([\d.-]+)', line)
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
    print("CTS STAGE METRICS SUMMARY")
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
            print(f"  HPWL: {summary['hpwl']:.2f} u")
        if summary.get('clock_sink_wirelength_um') is not None:
            print(f"  Clock Sink Wirelength: {summary['clock_sink_wirelength_um']:.2f} um")
        if summary.get('wns') is not None:
            print(f"  WNS: {summary['wns']:.3f}")
        if summary.get('tns') is not None:
            print(f"  TNS: {summary['tns']:.3f}")
        if summary.get('violations') is not None:
            print(f"  Violations: {summary['violations']}")
        if summary.get('displacement') is not None:
            print(f"  Displacement: {summary['displacement']:.2f} u")
    
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
            if substage.get('hpwl') is not None:
                print(f"    HPWL: {substage['hpwl']:.2f} u")
            if substage.get('clock_sink_wirelength_um') is not None:
                print(f"    Clock Sink Wirelength: {substage['clock_sink_wirelength_um']:.2f} um")
            if substage.get('area') is not None:
                print(f"    Area: {substage['area']:.2f} u^2")
            if substage.get('utilization') is not None:
                print(f"    Utilization: {substage['utilization']}%")
            if substage.get('density') is not None:
                print(f"    Density: {substage['density']}%")
            if substage.get('displacement') is not None:
                print(f"    Displacement: {substage['displacement']:.2f} u")
            
            # Optimization processes
            if substage.get('optimization_process') and len(substage['optimization_process']) > 0:
                print(f"\n  Optimization Processes ({len(substage['optimization_process'])} total, in execution order):")
                for idx, opt_process in enumerate(substage['optimization_process'], 1):
                    print(f"\n    [{idx}] {opt_process['type'].upper()}")
                    if opt_process.get('command'):
                        print(f"        Command: {opt_process['command']}")
                    
                    if opt_process['type'] == 'clock_tree_synthesis':
                        results = opt_process.get('results', {})
                        if results.get('total_sinks'):
                            print(f"        Total sinks: {results['total_sinks']}")
                        if results.get('clock_buffers_created'):
                            print(f"        Clock buffers created: {results['clock_buffers_created']}")
                        if results.get('min_buffer_depth') and results.get('max_buffer_depth'):
                            print(f"        Buffer depth: {results['min_buffer_depth']} - {results['max_buffer_depth']}")
                        if results.get('avg_sink_wirelength_um'):
                            print(f"        Avg sink wirelength: {results['avg_sink_wirelength_um']:.2f} um")
                        if results.get('hpwl_before') and results.get('hpwl_after'):
                            print(f"        HPWL: {results['hpwl_before']:.1f} → {results['hpwl_after']:.1f} u")
                        if results.get('placement_total_displacement_u'):
                            print(f"        Displacement (total): {results['placement_total_displacement_u']:.2f} u")
                        if results.get('placement_avg_displacement_u'):
                            print(f"        Displacement (avg): {results['placement_avg_displacement_u']:.2f} u")
                        if results.get('placement_max_displacement_u'):
                            print(f"        Displacement (max): {results['placement_max_displacement_u']:.2f} u")
                        
                        # Print clock nets information
                        clock_nets = results.get('clock_nets', [])
                        if clock_nets and len(clock_nets) > 0:
                            print(f"\n        Clock Nets ({len(clock_nets)} total):")
                            for net_idx, clock_net in enumerate(clock_nets, 1):
                                print(f"\n          [{net_idx}] {clock_net.get('name', 'unknown')}")
                                if clock_net.get('sinks') is not None:
                                    print(f"              Sinks: {clock_net['sinks']}")
                                if clock_net.get('leaf_buffers') is not None:
                                    print(f"              Leaf buffers: {clock_net['leaf_buffers']}")
                                if clock_net.get('avg_sink_wire_length_um') is not None:
                                    print(f"              Average sink wire length: {clock_net['avg_sink_wire_length_um']:.2f} um")
                                if clock_net.get('path_depth_min') is not None and clock_net.get('path_depth_max') is not None:
                                    print(f"              Path depth: {clock_net['path_depth_min']} - {clock_net['path_depth_max']}")
                                if clock_net.get('leaf_load_cells') is not None:
                                    print(f"              Leaf load cells: {clock_net['leaf_load_cells']}")
                    
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
                        
                        placement = opt_process.get('placement_impact', {})
                        if placement.get('hpwl_before') and placement.get('hpwl_after'):
                            print(f"        HPWL impact: {placement['hpwl_before']:.1f} → {placement['hpwl_after']:.1f} u")
                        if placement.get('total_displacement_u'):
                            print(f"        Displacement (total): {placement['total_displacement_u']:.2f} u")
                        if placement.get('avg_displacement_u'):
                            print(f"        Displacement (avg): {placement['avg_displacement_u']:.2f} u")
                        if placement.get('max_displacement_u'):
                            print(f"        Displacement (max): {placement['max_displacement_u']:.2f} u")
            
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
    # Read the extracted CTS log
    with open('test_extraction_filtered/ariane133_nan45_cts_extract.txt', 'r') as f:
        log_content = f.read()
    
    # Extract metrics
    metrics = extract_cts_metrics(log_content)
    
    # Print summary
    print_metrics_summary(metrics)
    
    # Also test with the other log file
    print("\n\n")
    print("#" * 80)
    print("TESTING WITH gcdshy130 LOG")
    print("#" * 80)
    
    with open('test_extraction_filtered/gcdshy130_cts_extract.txt', 'r') as f:
        log_content2 = f.read()
    
    metrics2 = extract_cts_metrics(log_content2)
    print_metrics_summary(metrics2)
    
    # Optionally save to JSON
    import json
    with open('cts_metrics_ariane.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    with open('cts_metrics_gcdshy130.json', 'w') as f:
        json.dump(metrics2, f, indent=2)
    
    print("\n\nMetrics saved to cts_metrics_ariane.json and cts_metrics_gcdshy130.json")

