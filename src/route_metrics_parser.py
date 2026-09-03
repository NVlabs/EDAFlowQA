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


def extract_route_metrics(log_content: str, design_name: Optional[str] = None, sub_design_name: Optional[str] = None) -> Dict:
    """
    Extract wirelength, violations, and resource usage metrics from routing stage extracted log.
    Returns a structured dictionary with metrics organized by substages (e.g., 5_1_grt, 5_2_route, 5_3_fillcell).
    
    Args:
        log_content: The extracted routing log content as a string
        
    Returns:
        Dictionary containing routing metrics by substage
    """
    lines = log_content.split('\n')
    
    result = {
        'stage': 'routing',
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
    in_resource_table = False
    in_drc_iter_table = False
    substages = {}
    
    # Helper function to initialize a substage
    def init_substage():
        return {
            # Direct metrics at substage level
            'wirelength_um': None,
            'total_vias': None,
            'violations': None,
            'area': None,
            'utilization': None,
            
            # Optimization processes in execution order
            'optimization_process': [],
            
            # Report metrics section (from "Report metrics stage 5, substage..." sections)
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
    in_repair_timing_table = False
    
    # FIRST PASS: Parse execution details and populate substages
    for i, line in enumerate(lines):
        # Detect substages - look for patterns like "5_1_grt", "5_2_route" etc.
        stage_match = re.search(r'(5_\d+_\w+)', line)
        if stage_match and 'Running' in line:
            current_substage = stage_match.group(1)
            if current_substage not in substages:
                substages[current_substage] = init_substage()
            in_resource_table = False
            in_drc_iter_table = False
            current_optimization = None
        
        # Also check for stage in log summary lines
        stage_summary_match = re.search(r'^[^\]]*\]\s+(5_\d+_\w+)\s+(\d+)\s+(\d+)', line)
        if stage_summary_match:
            stage_name = stage_summary_match.group(1)
            if stage_name not in substages:
                substages[stage_name] = init_substage()
        
        # If we have a current substage, extract metrics
        if current_substage and current_substage in substages:
            # Detect global_route operation - avoid "Took X seconds" message and -end_incremental
            if 'global_route' in line and ('-congestion_report' in line or '-start_incremental' in line) and '-end_incremental' not in line and 'Took' not in line and 'seconds:' not in line:
                current_optimization = {
                    'type': 'global_route',
                    'command': line.split('] ')[-1].strip() if ']' in line else line.strip(),
                    'results': {
                        'total_wirelength_um': None,
                        'total_resource': None,
                        'total_demand': None,
                        'total_usage_percent': None,
                        'total_overflow': None,
                        'per_layer_metrics': []
                    },
                    'placement_impact': {
                        'total_displacement_u': None,
                        'avg_displacement_u': None,
                        'max_displacement_u': None,
                        'hpwl_before': None,
                        'hpwl_after': None,
                        'delta_hpwl_percent': None
                    }
                }
                substages[current_substage]['optimization_process'].append(current_optimization)
            
            # Detect repair_design operation - avoid "Took X seconds" message
            if 'repair_design' in line and '-verbose' in line and 'Took' not in line and 'seconds:' not in line:
                current_optimization = {
                    'type': 'repair_design',
                    'command': line.split('] ')[-1].strip() if ']' in line else line.strip(),
                    'iterations_count': 0,
                    'start_metrics': {},
                    'end_metrics': {},
                    'summary': {
                        'resized_instances': None,
                        'inserted_buffers': None,
                        'nets_repaired': None
                    }
                }
                substages[current_substage]['optimization_process'].append(current_optimization)
                in_repair_timing_table = False
            
            # Detect repair_timing operation - avoid "Took X seconds" message
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
                    'summary': {
                        'removed_buffers': None,
                        'inserted_buffers': None,
                        'resized_instances': None,
                        'cloned_instances': None
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
            
            # Detect detailed_route operation - avoid "Took X seconds" message
            if 'detailed_route' in line and '-output_drc' in line and 'Took' not in line and 'seconds:' not in line:
                current_optimization = {
                    'type': 'detailed_route',
                    'command': line.split('] ')[-1].strip() if ']' in line else line.strip(),
                    'drc_iterations': [],
                    'final_wirelength_um': None,
                    'final_vias': None,
                    'final_violations': None
                }
                substages[current_substage]['optimization_process'].append(current_optimization)
            
            # Parse resource/demand table for global_route
            if 'Layer' in line and 'Resource' in line and 'Demand' in line and 'Usage' in line:
                in_resource_table = True
                continue
            
            if in_resource_table and current_optimization and current_optimization['type'] == 'global_route':
                if '---' in line:
                    continue
                
                # Parse per-layer metrics
                match = re.search(r'(metal\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)%\s+(\d+)\s+/\s+(\d+)\s+/\s+(\d+)', line)
                if match:
                    layer_data = {
                        'layer': match.group(1),
                        'resource': int(match.group(2)),
                        'demand': int(match.group(3)),
                        'usage_percent': float(match.group(4)),
                        'overflow_h': int(match.group(5)),
                        'overflow_v': int(match.group(6)),
                        'overflow_total': int(match.group(7))
                    }
                    current_optimization['results']['per_layer_metrics'].append(layer_data)
                
                # Parse total row
                match = re.search(r'Total\s+(\d+)\s+(\d+)\s+([\d.]+)%\s+(\d+)\s+/\s+(\d+)\s+/\s+(\d+)', line)
                if match:
                    current_optimization['results']['total_resource'] = int(match.group(1))
                    current_optimization['results']['total_demand'] = int(match.group(2))
                    current_optimization['results']['total_usage_percent'] = float(match.group(3))
                    current_optimization['results']['total_overflow'] = int(match.group(6))
                    in_resource_table = False
            
            # Parse total wirelength
            match = re.search(r'Total wirelength:\s+([\d.]+)\s+um', line)
            if match:
                wl = float(match.group(1))
                if current_optimization and current_optimization['type'] == 'global_route':
                    current_optimization['results']['total_wirelength_um'] = wl
                substages[current_substage]['wirelength_um'] = wl
            
            # Repair timing table header
            if 'Iter' in line and 'WNS' in line and 'TNS' in line and current_optimization and current_optimization.get('type') == 'repair_timing':
                in_repair_timing_table = True
                # Now we know repair_timing has actual timing data, add it to the list
                if not current_optimization.get('_has_data'):
                    current_optimization['_has_data'] = True
                    substages[current_substage]['optimization_process'].append(current_optimization)
                continue
            
            # Parse repair timing table rows
            if in_repair_timing_table and current_optimization and current_optimization.get('type') == 'repair_timing':
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
                    removed_buffers = int(match.group(2))
                    resized_gates = int(match.group(3))
                    inserted_buffers = int(match.group(4))
                    cloned_gates = int(match.group(5))
                    pin_swaps = int(match.group(6))
                    area = match.group(7)
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
                        # Store summary from final row
                        current_optimization['summary']['removed_buffers'] = removed_buffers
                        current_optimization['summary']['resized_instances'] = resized_gates
                        current_optimization['summary']['inserted_buffers'] = inserted_buffers
                        current_optimization['summary']['cloned_instances'] = cloned_gates
                        in_repair_timing_table = False
                
                # Check for INFO/WARNING messages that indicate end of table
                if '[INFO RSZ' in line or '[WARNING RSZ' in line:
                    in_repair_timing_table = False
                    # Extract summary info from messages
                    match = re.search(r'Inserted (\d+) buffers?', line)
                    if match and current_optimization.get('type') == 'repair_timing':
                        current_optimization['summary']['inserted_buffers'] = int(match.group(1))
                    
                    match = re.search(r'Resized (\d+) instances?', line)
                    if match and current_optimization.get('type') == 'repair_timing':
                        current_optimization['summary']['resized_instances'] = int(match.group(1))
            
            # Parse placement analysis after global_route -start_incremental
            if 'Placement Analysis' in line and current_optimization:
                # Next lines contain placement metrics (can be from global_route or repair_timing)
                in_placement_analysis = True
                continue
            
            # Parse placement displacement metrics (can be for global_route or repair_timing)
            if current_optimization and current_optimization.get('type') in ['global_route', 'repair_timing']:
                match = re.search(r'total displacement\s+([\d.]+)\s*u', line)
                if match:
                    current_optimization['placement_impact']['total_displacement_u'] = float(match.group(1))
                
                match = re.search(r'average displacement\s+([\d.]+)\s*u', line)
                if match:
                    current_optimization['placement_impact']['avg_displacement_u'] = float(match.group(1))
                
                match = re.search(r'max displacement\s+([\d.]+)\s*u', line)
                if match:
                    current_optimization['placement_impact']['max_displacement_u'] = float(match.group(1))
                
                match = re.search(r'original HPWL\s+([\d.]+)\s*u', line)
                if match:
                    current_optimization['placement_impact']['hpwl_before'] = float(match.group(1))
                
                match = re.search(r'legalized HPWL\s+([\d.]+)\s*u', line)
                if match:
                    current_optimization['placement_impact']['hpwl_after'] = float(match.group(1))
                
                match = re.search(r'delta HPWL\s+([\d.]+)\s*%', line)
                if match:
                    current_optimization['placement_impact']['delta_hpwl_percent'] = float(match.group(1))
            
            # Parse repair_design summary
            match = re.search(r'final\s+\|\s+([+\-\d.]+)%\s+\|\s+(\d+)\s+\|\s+(\d+)\s+\|\s+(\d+)\s+\|\s+(\d+)', line)
            if match and current_optimization and current_optimization['type'] == 'repair_design':
                current_optimization['summary']['resized_instances'] = int(match.group(2))
                current_optimization['summary']['inserted_buffers'] = int(match.group(3))
                current_optimization['summary']['nets_repaired'] = int(match.group(4))
            
            # Parse detailed routing DRC iterations
            if 'Start' in line and 'optimization iteration' in line:
                match = re.search(r'Start\s+(\d+)', line)
                if match and current_optimization and current_optimization['type'] == 'detailed_route':
                    iter_num = int(match.group(1))
                    in_drc_iter_table = True
                    current_drc_iter = {
                        'iteration': iter_num,
                        'violations': None,
                        'wirelength_um': None,
                        'vias': None
                    }
                    current_optimization['drc_iterations'].append(current_drc_iter)
            
            # Parse number of violations after each DRC iteration
            match = re.search(r'Number of violations\s+=\s+(\d+)', line)
            if match and current_optimization and current_optimization['type'] == 'detailed_route':
                violations = int(match.group(1))
                if len(current_optimization['drc_iterations']) > 0:
                    current_optimization['drc_iterations'][-1]['violations'] = violations
                    # Update current substage violation count (last is final)
                    substages[current_substage]['violations'] = violations
                    current_optimization['final_violations'] = violations
            
            # Parse total wire length during detailed routing
            match = re.search(r'Total wire length\s+=\s+([\d.]+)\s+um', line)
            if match and current_optimization and current_optimization['type'] == 'detailed_route':
                wl = float(match.group(1))
                if len(current_optimization['drc_iterations']) > 0:
                    current_optimization['drc_iterations'][-1]['wirelength_um'] = wl
                    substages[current_substage]['wirelength_um'] = wl
                    current_optimization['final_wirelength_um'] = wl
            
            # Parse total number of vias during detailed routing
            match = re.search(r'Total number of vias\s+=\s+([\d.]+)', line)
            if match and current_optimization and current_optimization['type'] == 'detailed_route':
                vias = int(float(match.group(1)))
                if len(current_optimization['drc_iterations']) > 0:
                    current_optimization['drc_iterations'][-1]['vias'] = vias
                    substages[current_substage]['total_vias'] = vias
                    current_optimization['final_vias'] = vias
            
            # Design area and utilization
            match = re.search(r'Design area\s+([\d.]+)\s+u\^2\s+(\d+)%\s+utilization', line)
            if match:
                area = float(match.group(1))
                util = int(match.group(2))
                substages[current_substage]['area'] = area
                substages[current_substage]['utilization'] = util
            
            # Performance metrics
            match = re.search(r'Elapsed time:\s+([\d:.]+)\[h:\]min:sec', line)
            if match:
                substages[current_substage]['elapsed_time'] = match.group(1)
            
            match = re.search(r'Peak memory:\s+(\d+)KB', line)
            if match:
                substages[current_substage]['peak_memory_kb'] = int(match.group(1))
        
        # Also parse log summary lines (at the end) - these can reference any substage
        log_summary_match = re.search(r'^[^\]]*\]\s+(5_\d+_\w+)\s+(\d+)\s+(\d+)\s+([a-f0-9]+)', line)
        if log_summary_match:
            stage_name, elapsed_s, memory_mb, sha1 = log_summary_match.groups()
            if stage_name in substages:
                substages[stage_name]['elapsed_s'] = int(elapsed_s)
                substages[stage_name]['memory_mb'] = int(memory_mb)
                substages[stage_name]['sha1sum'] = sha1
    
    # SECOND PASS: Add "Report metrics" title to existing substages
    for i, line in enumerate(lines):
        report_metrics_match = re.search(r'Report metrics stage 5,\s+(.+?)\.\.\.', line)
        if report_metrics_match:
            substage_name = report_metrics_match.group(1).strip()
            
            # Just store the first line (the title) instead of the full report
            report_title = line.strip()
            
            # Map report name to substage (e.g., "global route" -> "5_1_grt", "detailed route" -> "5_2_route")
            substage_mapping = {
                'global route': '5_1_grt',
                'detailed route': '5_2_route',
                'fillcell': '5_3_fillcell'
            }
            
            for key, stage_id in substage_mapping.items():
                if key in substage_name.lower():
                    # Only add report_metrics field if substage already exists
                    if stage_id in substages:
                        substages[stage_id]['report_metrics'] = report_title
                    break
    
    # Store substages
    result['substages'] = substages
    
    # Clean up internal tracking fields before returning
    for substage_name, substage_data in result['substages'].items():
        for opt in substage_data.get('optimization_process', []):
            if '_has_data' in opt:
                del opt['_has_data']
    
    # Create summary with key metrics from the last substage
    summary = {}
    sorted_substages = sorted(substages.keys())
    
    for substage in reversed(sorted_substages):
        if summary.get('area') is None and substages[substage]['area'] is not None:
            summary['area'] = substages[substage]['area']
        if summary.get('utilization') is None and substages[substage]['utilization'] is not None:
            summary['utilization'] = substages[substage]['utilization']
        if summary.get('wirelength_um') is None and substages[substage]['wirelength_um'] is not None:
            summary['wirelength_um'] = substages[substage]['wirelength_um']
        if summary.get('total_vias') is None and substages[substage]['total_vias'] is not None:
            summary['total_vias'] = substages[substage]['total_vias']
        if summary.get('violations') is None and substages[substage]['violations'] is not None:
            summary['violations'] = substages[substage]['violations']
    
    result['summary'] = summary
    
    # Parse detailedroute-related test metrics from final stage (if present in log)
    # These are test metrics that start with "detailedroute__"
    for line in lines:
        test_match = re.search(r'\[INFO\]\s+(detailedroute__[a-z_]+)(?:__([a-z_]+))?(?:__([a-z_]+))?(?:__([a-z_]+))?\s+pass test:\s+([\d.-]+)\s+(==|<=|>=)\s+([\d.-]+)', line)
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


# Example usage
if __name__ == "__main__":
    # Read the extracted routing log
    with open('test_extraction_filtered/ariane133_nan45_route_extract.txt', 'r') as f:
        log_content = f.read()
    
    # Extract metrics
    metrics = extract_route_metrics(log_content)
    
    # Save to JSON
    import json
    with open('route_metrics_ariane.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\n\nMetrics saved to route_metrics_ariane.json")
    
    # also test with gcdshy130 log
    with open('test_extraction_filtered/gcdshy130_route_extract.txt', 'r') as f:
        log_content2 = f.read()
    metrics2 = extract_route_metrics(log_content2)
    print_metrics_summary(metrics2)
    with open('route_metrics_gcdshy130.json', 'w') as f:
        json.dump(metrics2, f, indent=2)
    print("\n\nMetrics saved to route_metrics_gcdshy130.json")

