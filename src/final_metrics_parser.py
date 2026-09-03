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


def extract_final_metrics(log_content: str, design_name: Optional[str] = None, sub_design_name: Optional[str] = None) -> Dict:
    """
    Extract final design metrics including IR drop, cell counts, and complete flow summary.
    Returns a structured dictionary with final design metrics for stage 6.
    
    Args:
        log_content: The extracted final log content as a string
        
    Returns:
        Dictionary containing final design metrics
    """
    lines = log_content.split('\n')
    
    result = {
        'stage': 'final',
        'ir_drop': {
            'vdd': {},
            'vss': {}
        },
        'cell_counts': {},
        'design_metrics': {
            'area': None
        },
        'flow_summary': {},
        'performance': {
            'elapsed_time': None,
            'peak_memory_kb': None
        },
        'test_metrics': {}
    }

    if design_name is not None:
        result['design_name'] = design_name
    if sub_design_name is not None:
        result['sub_design_name'] = sub_design_name
    
    current_ir_net = None
    
    for i, line in enumerate(lines):
        # Parse IR drop reports
        if 'IR report' in line:
            # Determine if this is VDD or VSS report
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                if 'Net' in next_line and 'VDD' in next_line:
                    current_ir_net = 'vdd'
                elif 'Net' in next_line and 'VSS' in next_line:
                    current_ir_net = 'vss'
            continue
        
        if current_ir_net:
            # Parse supply voltage
            match = re.search(r'Supply voltage\s+:\s+([\d.e+\-]+)\s+V', line)
            if match:
                result['ir_drop'][current_ir_net]['supply_voltage_v'] = float(match.group(1))
            
            # Parse worstcase voltage
            match = re.search(r'Worstcase voltage:\s+([\d.e+\-]+)\s+V', line)
            if match:
                result['ir_drop'][current_ir_net]['worstcase_voltage_v'] = float(match.group(1))
            
            # Parse average voltage
            match = re.search(r'Average voltage\s+:\s+([\d.e+\-]+)\s+V', line)
            if match:
                result['ir_drop'][current_ir_net]['average_voltage_v'] = float(match.group(1))
            
            # Parse average IR drop
            match = re.search(r'Average IR drop\s+:\s+([\d.e+\-]+)\s+V', line)
            if match:
                result['ir_drop'][current_ir_net]['average_ir_drop_v'] = float(match.group(1))
            
            # Parse worstcase IR drop
            match = re.search(r'Worstcase IR drop:\s+([\d.e+\-]+)\s+V', line)
            if match:
                result['ir_drop'][current_ir_net]['worstcase_ir_drop_v'] = float(match.group(1))
            
            # Parse percentage drop
            match = re.search(r'Percentage drop\s+:\s+([\d.]+)\s+%', line)
            if match:
                result['ir_drop'][current_ir_net]['percentage_drop'] = float(match.group(1))
                current_ir_net = None  # End of this IR report
        
        # Parse cell type report
        if 'Cell type report:' in line:
            # Skip header line and read cell counts
            continue
        
        # Parse cell counts
        match = re.search(r'Macro\s+(\d+)\s+([\d.]+)', line)
        if match:
            result['cell_counts']['macro_count'] = int(match.group(1))
            result['cell_counts']['macro_area'] = float(match.group(2))
        
        match = re.search(r'Fill cell\s+(\d+)\s+([\d.]+)', line)
        if match:
            result['cell_counts']['fill_cell_count'] = int(match.group(1))
            result['cell_counts']['fill_cell_area'] = float(match.group(2))
        
        match = re.search(r'Tap cell\s+(\d+)\s+([\d.]+)', line)
        if match:
            result['cell_counts']['tap_cell_count'] = int(match.group(1))
            result['cell_counts']['tap_cell_area'] = float(match.group(2))
        
        match = re.search(r'Clock buffer\s+(\d+)\s+([\d.]+)', line)
        if match:
            result['cell_counts']['clock_buffer_count'] = int(match.group(1))
            result['cell_counts']['clock_buffer_area'] = float(match.group(2))
        
        match = re.search(r'Timing Repair Buffer\s+(\d+)\s+([\d.]+)', line)
        if match:
            result['cell_counts']['timing_repair_buffer_count'] = int(match.group(1))
            result['cell_counts']['timing_repair_buffer_area'] = float(match.group(2))
        
        match = re.search(r'Inverter\s+(\d+)\s+([\d.]+)', line)
        if match:
            result['cell_counts']['inverter_count'] = int(match.group(1))
            result['cell_counts']['inverter_area'] = float(match.group(2))
        
        match = re.search(r'Clock inverter\s+(\d+)\s+([\d.]+)', line)
        if match:
            result['cell_counts']['clock_inverter_count'] = int(match.group(1))
            result['cell_counts']['clock_inverter_area'] = float(match.group(2))
        
        match = re.search(r'Sequential cell\s+(\d+)\s+([\d.]+)', line)
        if match:
            result['cell_counts']['sequential_cell_count'] = int(match.group(1))
            result['cell_counts']['sequential_cell_area'] = float(match.group(2))
        
        match = re.search(r'Multi-Input combinational cell\s+(\d+)\s+([\d.]+)', line)
        if match:
            result['cell_counts']['combinational_cell_count'] = int(match.group(1))
            result['cell_counts']['combinational_cell_area'] = float(match.group(2))
        
        match = re.search(r'Total\s+(\d+)\s+([\d.]+)', line)
        if match and 'cell' in lines[i - 1].lower():
            result['cell_counts']['total_cell_count'] = int(match.group(1))
            result['cell_counts']['total_cell_area'] = float(match.group(2))
        
        # Parse design area (keep only area, not utilization)
        match = re.search(r'Design area\s+([\d.]+)\s+u\^2\s+(\d+)%\s+utilization', line)
        if match:
            result['design_metrics']['area'] = float(match.group(1))
        
        # Parse performance metrics for 6_report stage
        if '6_report' in line and 'Elapsed time' in line:
            match = re.search(r'Elapsed time:\s+([\d:.]+)\[h:\]min:sec', line)
            if match:
                result['performance']['elapsed_time'] = match.group(1)
            
            match = re.search(r'Peak memory:\s+(\d+)KB', line)
            if match:
                result['performance']['peak_memory_kb'] = int(match.group(1))
        
        # Parse flow summary - all stages with their metrics
        log_summary_match = re.search(r'^[^\]]*\]\s+([\d_\w]+)\s+(\d+)\s+(\d+)\s+([a-f0-9]+|N/A)', line)
        if log_summary_match:
            stage_name, elapsed_s, memory_mb, sha1 = log_summary_match.groups()
            # Only include if it looks like a valid stage name (contains underscore or digit)
            if '_' in stage_name or any(c.isdigit() for c in stage_name):
                result['flow_summary'][stage_name] = {
                    'elapsed_s': int(elapsed_s),
                    'memory_mb': int(memory_mb),
                    'sha1sum': sha1 if sha1 != 'N/A' else None
                }
        
        # Parse test validation metrics
        test_match = re.search(r'\[INFO\]\s+([a-z_]+)__([a-z_]+)(?:__([a-z_]+))?(?:__([a-z_]+))?\s+pass test:\s+([\d.-]+)\s+(==|<=|>=)\s+([\d.-]+)', line)
        if test_match:
            groups = test_match.groups()
            # Build metric name from non-None groups (excluding the last 3 which are value, operator, threshold)
            metric_parts = [g for g in groups[:-3] if g is not None]
            metric_name = '__'.join(metric_parts)
            actual_value = float(groups[-3])
            operator = groups[-2]
            threshold_value = float(groups[-1])
            
            result['test_metrics'][metric_name] = {
                'actual_value': actual_value,
                'operator': operator,
                'threshold_value': threshold_value,
                'passed': True  # Only pass tests are in the log
            }
    
    # Parse test summary line
    for line in lines:
        summary_match = re.search(r'All metadata rules passed \((\d+) rules\)', line)
        if summary_match:
            result['test_metrics']['summary'] = {
                'total_rules': int(summary_match.group(1)),
                'all_passed': True
            }
    
    # Remove null values from the result
    result = remove_null_values(result)
    
    return result


# Example usage
if __name__ == "__main__":
    # Read the extracted final log
    with open('test_extraction_filtered/ariane133_nan45_final_extract.txt', 'r') as f:
        log_content = f.read()
    
    # Extract metrics
    metrics = extract_final_metrics(log_content)
    
    # Save to JSON
    import json
    with open('final_metrics_ariane.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\n\nMetrics saved to final_metrics_ariane.json")

    # also test with gcdshy130 log
    with open('test_extraction_filtered/gcdshy130_final_extract.txt', 'r') as f:
        log_content2 = f.read()
    metrics2 = extract_final_metrics(log_content2)
    print_metrics_summary(metrics2)
    with open('final_metrics_gcdshy130.json', 'w') as f:
        json.dump(metrics2, f, indent=2)
    print("\n\nMetrics saved to final_metrics_gcdshy130.json")

