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
from typing import Dict, List, Tuple, Optional

def extract_final_section(log_content: str, output_file: Optional[str] = None) -> Dict:
    """
    Extract Final section (Stage 6) from OpenROAD log file with final metrics.
    
    Args:
        log_content: The full log file content as a string
        output_file: Optional path to write the extracted results
    
    Returns a dictionary containing:
    - section_bounds: Start and end lines of the final section
    - substages: Information about each final substage (6_1_fill, 6_report)
    - key_metrics: Final metrics (area, utilization, IR drop, cell counts, etc.)
    - warnings_summary: Count of unique warning types
    """
    lines = log_content.split('\n')
    
    result = {
        'section_bounds': None,
        'substages': {},
        'key_metrics': {},
        'warnings_summary': {},
        'raw_section': ''
    }
    
    # Find final section boundaries
    start_idx = None
    end_idx = None
    
    for i, line in enumerate(lines):
        # Check for "Running" pattern (full execution logs)
        if 'Running density_fill.tcl, stage 6_1_fill' in line or 'stage 6_1_fill' in line:
            start_idx = i
        # Also check for "Report metrics stage 6" pattern (report logs)
        if 'Report metrics stage 6' in line and start_idx is None:
            start_idx = i - 2  # Include separator line
        if start_idx is not None and i > start_idx:
            # End after "All metadata rules passed" line (includes test metrics)
            if 'All metadata rules passed' in line:
                end_idx = i + 1  # Include this line
                break
    
    if start_idx is None:
        return result
    
    if end_idx is None:
        end_idx = len(lines)
    
    result['section_bounds'] = (start_idx + 1, end_idx + 1)
    section_lines = lines[start_idx:end_idx]
    
    # Extract substages
    substages = {}
    current_stage = None
    
    for i, line in enumerate(section_lines):
        # Detect substage start
        stage_match = re.search(r'(6_[0-9a-z_]+)', line)
        if stage_match and 'Running' in line:
            current_stage = stage_match.group(1)
            substages[current_stage] = {
                'name': current_stage,
                'info': [],
                'elapsed_time': None,
                'peak_memory': None,
                'sha1sum': None
            }
        
        if current_stage:
            # Capture elapsed time and memory
            if 'Elapsed time:' in line and 'Peak memory:' in line:
                time_match = re.search(r'Elapsed time: ([\d:.]+)', line)
                mem_match = re.search(r'Peak memory: (\d+KB)', line)
                if time_match:
                    substages[current_stage]['elapsed_time'] = time_match.group(1)
                if mem_match:
                    substages[current_stage]['peak_memory'] = mem_match.group(1)
    
    result['substages'] = substages
    
    # Extract key metrics
    for line in section_lines:
        # Utilization
        if 'Design area' in line and 'utilization' in line:
            match = re.search(r'Design area (\d+) u\^2 (\d+)% utilization', line)
            if match:
                result['key_metrics']['final_area'] = match.group(1) + ' u^2'
                result['key_metrics']['final_utilization'] = match.group(2) + '%'
        
        # IR drop
        if 'Worstcase IR drop:' in line:
            match = re.search(r'Worstcase IR drop: ([\d.e+-]+) V', line)
            if match:
                net_match = re.search(r'Net\s+:\s+(\w+)', '\n'.join(section_lines[max(0, section_lines.index(line)-5):section_lines.index(line)]))
                net = net_match.group(1) if net_match else 'unknown'
                result['key_metrics'][f'ir_drop_{net}'] = match.group(1) + ' V'
        
        if 'Percentage drop' in line:
            match = re.search(r'Percentage drop\s+:\s+([\d.]+)\s+%', line)
            if match:
                net_match = re.search(r'Net\s+:\s+(\w+)', '\n'.join(section_lines[max(0, section_lines.index(line)-8):section_lines.index(line)]))
                net = net_match.group(1) if net_match else 'unknown'
                result['key_metrics'][f'ir_drop_percent_{net}'] = match.group(1) + '%'
    
    # Count warnings (compact summary)
    warning_counts = {}
    for line in section_lines:
        warning_match = re.search(r'\[(WARNING|ERROR) ([A-Z]+-\d+)\]', line)
        if warning_match:
            level, code = warning_match.groups()
            key = f"{level} {code}"
            warning_counts[key] = warning_counts.get(key, 0) + 1
    
    result['warnings_summary'] = warning_counts
    
    # Store raw section, filtering unnecessary verbose output
    filtered_section = []
    
    # Keywords to skip
    skip_patterns = [
        r'read_liberty',
        r'read_db',
        r'^\[.*\] cp ',
        r'^\[.*\] mkdir',
        r'source .*\.tcl',
        r'link_design',
        r'Master .* is loaded but not used',
        r'LEF file:',
        r'/tmp/workspace/.*flow\.sh',
        r'^\[.*\] exec cp',
        r'^\[.*\] sed ',
        r'^\[.*\] env time',
        r'klayout\.sh',
        r'def2stream\.py',
        r'in_def=|in_files=|out_file=|tech_file=|layer_map=',
        r'KLayout \d+\.\d+',
        r'Deleted \d+ routing obstructions',
        r'SC_LEF_RELATIVE_PATH=',
        r'OTHER_LEFS_RELATIVE_PATHS=',
        r'-rd seal_file=',
        r'^\[.*\]\s+/tmp/workspace/',
        r'No elapsed time found',
        r'^\[.*\]\s+\./logs/',
    ]
    
    # Keep patterns - ensure these are NOT filtered even if they match skip patterns
    keep_patterns = [
        r'Running.*\.tcl, stage',  # Stage markers
        r'Elapsed time.*Memory',  # Performance metrics
        r'Design area.*u\^2.*% utilization',  # Utilization metric
        # IR drop analysis
        r'########## IR report',
        r'Net\s+:',
        r'Corner\s+:',
        r'Supply voltage\s+:',
        r'Worstcase voltage:',
        r'Average voltage\s+:',
        r'Average IR drop\s+:',
        r'Worstcase IR drop:',
        r'Percentage drop\s+:',
        r'##+$',  # Report boundary
        # Cell type report
        r'Cell type report:',
        r'Fill cell',
        r'Tap cell',
        r'Clock buffer',
        r'Timing Repair Buffer',
        r'Inverter',
        r'Sequential cell',
        r'Multi-Input combinational cell',
        r'Total\s+\d+\s+[\d.]+',  # Total row
        # Final reports
        r'finish report_design_area',
        r'Report metrics stage',
        # Summary table
        r'Log\s+Elapsed/s Peak Memory',
        r'[0-9]_[0-9a-z_]+\s+\d+\s+\d+',  # Stage summary rows
        # Test validation metrics
        r'\[INFO\].*pass test:',  # All test validation metrics
        r'All metadata rules passed',  # Test summary
        # Important INFO messages
        r'\[INFO FLW-',  # Flow info
        r'\[ERROR',  # Keep all errors
        r'\[WARNING',  # Keep all warnings in final stage for review
    ]
    
    # Skip INFO/WARNING unless they match keep_patterns
    info_warning_skip = [
        r'^\[.*\]\s*\[INFO RCX-',  # RCX info (not critical)
        r'^\[.*\]\s*\[INFO PSM-',  # PSM info (not critical)
        r'^\[.*\]\s*\[WARNING GUI-',  # GUI warnings not relevant
    ]
    
    for line in section_lines:
        # Check if line should be kept first (priority over skip patterns)
        should_keep = any(re.search(pattern, line) for pattern in keep_patterns)
        if should_keep:
            filtered_section.append(line)
            continue
        
        # Skip specific INFO/WARNING patterns (not all)
        if any(re.search(pattern, line) for pattern in info_warning_skip):
            continue
        
        # Skip unnecessary commands and file operations
        should_skip = any(re.search(pattern, line) for pattern in skip_patterns)
        if should_skip:
            continue
            
        filtered_section.append(line)
    
    # Just use the filtered section as the raw log
    result['raw_section'] = '\n'.join(filtered_section)
    
    # Write to file if output_file is specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write(result['raw_section'])
    
    return result


# Example usage
if __name__ == "__main__":
    # Read log file
    with open('gcdshy130.txt', 'r') as f:
        log_content = f.read()
    
    # Extract final section and write to file
    output_file = 'final_extract.txt'
    final_data = extract_final_section(log_content, output_file=output_file)
    
    print(f"Final section extracted and saved to: {output_file}")
    print("\nFirst 50 lines of output:")
    print("-" * 80)
    
    # Print first 50 lines of the raw section
    lines = final_data['raw_section'].split('\n')
    for line in lines[:50]:
        print(line)

