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

def extract_floorplan_section(log_content: str, output_file: Optional[str] = None) -> Dict:
    """
    Extract floorplan section from OpenROAD log file with timing metrics.
    
    Args:
        log_content: The full log file content as a string
        output_file: Optional path to write the extracted results
    
    Returns a dictionary containing:
    - section_bounds: Start and end lines of the floorplan section
    - timing_table: Compact representation of repair_timing iterations
    - key_metrics: Final metrics (WNS, TNS, violations, area, utilization)
    - warnings_summary: Count of unique warning types
    """
    lines = log_content.split('\n')
    
    result = {
        'section_bounds': None,
        'timing_table': [],
        'key_metrics': {},
        'warnings_summary': {},
        'substages': {},
        'raw_section': ''
    }
    
    # Find floorplan section boundaries
    start_idx = None
    end_idx = None
    
    for i, line in enumerate(lines):
        # Check for "Running" pattern or "Floorplan check_setup" (full execution logs)
        if 'Floorplan check_setup' in line or ('2_1_floorplan' in line and 'Running' in line):
            if start_idx is None:
                start_idx = max(0, i - 2)  # Include separator line
        # Also check for "Report metrics stage 2" pattern (report logs)
        if 'Report metrics stage 2' in line and start_idx is None:
            start_idx = max(0, i - 2)  # Include separator line
        # Look for the end of the last floorplan substage (2_4_floorplan_pdn)
        if '2_4_floorplan_pdn' in line and 'sha1sum' in line:
            end_idx = i + 1
            break
        # Alternative ending: when we see stage 3 starting
        # Prefer explicit "Running ... stage 3_" lines; otherwise accept a generic 3_ token
        # once we've started capturing.
        if start_idx is not None:
            if re.search(r'\bRunning\b.*?\.tcl,\s*stage\s+3_', line):
                end_idx = i
                break
            if re.search(r'(?:^|\s)3_', line) and ('Running' in line or 'flow.sh' in line or 'place' in line.lower()):
                end_idx = i
                break
    
    if start_idx is None:
        return result
    if end_idx is None:
        # If we didn't see a clear stage-3 boundary, treat end-of-input as end.
        end_idx = len(lines)
    
    result['section_bounds'] = (start_idx, end_idx)
    
    # Extract and compact the section
    section_lines = lines[start_idx:end_idx]
    
    # Extract floorplan substages (2_1, 2_2, 2_3, 2_4)
    substages = {}
    current_stage = None
    
    for i, line in enumerate(section_lines):
        # Detect stage starts
        stage_match = re.search(r'(2_[1-4]_floorplan[_\w]*)', line)
        if stage_match and 'Running' in line:
            current_stage = stage_match.group(1)
            substages[current_stage] = {
                'name': current_stage,
                'info': [],
                'elapsed_time': None,
                'peak_memory': None,
                'sha1sum': None
            }
        
        # Capture stage-specific info
        if current_stage:
            # Skip unnecessary commands and file operations
            skip_keywords = [
                'read_liberty', 'read_db', 'cp ', 'mkdir', 'source ',
                'link_design', 'Master ', 'LEF file:', 'read_'
            ]
            
            should_skip = any(keyword in line for keyword in skip_keywords)
            
            if not should_skip:
                # Capture important info messages
                if '[INFO' in line and current_stage in substages:
                    info_match = re.search(r'\[INFO [A-Z]+-\d+\] (.+)', line)
                    if info_match:
                        substages[current_stage]['info'].append(info_match.group(1))
                
                # Capture "No macros found" and similar messages
                if any(keyword in line for keyword in ['No macros found', 'Skipping', 'Inserted', 'Inserting grid']):
                    if current_stage in substages:
                        clean_line = line.split('] ', 1)[-1] if ']' in line else line
                        substages[current_stage]['info'].append(clean_line.strip())
            
            # Capture timing and memory metrics
            elapsed_match = re.search(r'Elapsed time: ([\d:.]+)\[h:\]min:sec', line)
            if elapsed_match and current_stage in substages:
                substages[current_stage]['elapsed_time'] = elapsed_match.group(1)
            
            memory_match = re.search(r'Peak memory: (\d+)KB', line)
            if memory_match and current_stage in substages:
                substages[current_stage]['peak_memory'] = int(memory_match.group(1))
            
            # Capture log summary line with sha1sum
            log_summary_match = re.search(r'(2_[1-4]_floorplan[_\w]*)\s+(\d+)\s+(\d+)\s+([a-f0-9]+)', line)
            if log_summary_match:
                stage_name, elapsed_s, memory_mb, sha1 = log_summary_match.groups()
                if stage_name in substages:
                    substages[stage_name]['elapsed_s'] = int(elapsed_s)
                    substages[stage_name]['memory_mb'] = int(memory_mb)
                    substages[stage_name]['sha1sum'] = sha1
    
    result['substages'] = substages
    
    # Parse timing repair table
    in_table = False
    table_data = []
    
    for line in section_lines:
        # Detect table header
        if 'Iter' in line and 'WNS' in line and 'TNS' in line:
            in_table = True
            continue
        
        # Detect table separator
        if '---' in line and in_table:
            continue
        
        # Parse table rows
        if in_table:
            # Match iteration rows with timing data
            match = re.search(r'^\s*(\d+|final)\s+\|\s+(\d+)\s+\|\s+(\d+)\s+\|\s+(\d+)\s+\|\s+(\d+)\s+\|\s+(\d+)\s+\|\s+([\d.+%]+)\s+\|\s+([-\d.]+)\s+\|\s+([-\d.]+)\s+\|\s+(\d+)', line)
            if match:
                iter_num, removed, resized, inserted, cloned, swaps, area, wns, tns, viols = match.groups()
                table_data.append({
                    'iter': iter_num,
                    'resized': int(resized),
                    'inserted': int(inserted),
                    'swaps': int(swaps),
                    'area': area,
                    'wns': float(wns),
                    'tns': float(tns),
                    'violations': int(viols)
                })
            elif '[INFO RSZ' in line or '[WARNING RSZ' in line or 'Elapsed time' in line:
                in_table = False
    
    # Keep all iterations
    result['timing_table'] = table_data
    
    # Extract key metrics
    for line in section_lines:
        # Final repair summary
        if 'Resized' in line and 'instances' in line:
            match = re.search(r'Resized (\d+) instances', line)
            if match:
                result['key_metrics']['resized_instances'] = int(match.group(1))
        
        if 'Swapped pins on' in line:
            match = re.search(r'Swapped pins on (\d+) instances', line)
            if match:
                result['key_metrics']['swapped_pins'] = int(match.group(1))
        
        # Design area and utilization
        if 'Design area' in line:
            match = re.search(r'Design area (\d+) u\^2 (\d+)% utilization', line)
            if match:
                result['key_metrics']['area'] = int(match.group(1))
                result['key_metrics']['utilization'] = int(match.group(2))
        
        # Number of instances
        if 'number instances in verilog' in line:
            match = re.search(r'number instances in verilog is (\d+)', line)
            if match:
                result['key_metrics']['num_instances'] = int(match.group(1))
        
        # Rows added
        if 'Added' in line and 'rows' in line:
            match = re.search(r'Added (\d+) rows', line)
            if match:
                result['key_metrics']['rows'] = int(match.group(1))
    
    # Count warnings (compact summary)
    warning_counts = {}
    for line in section_lines:
        warning_match = re.search(r'\[(WARNING|ERROR) ([A-Z]+-\d+)\]', line)
        if warning_match:
            level, code = warning_match.groups()
            key = f"{level} {code}"
            warning_counts[key] = warning_counts.get(key, 0) + 1
    
    result['warnings_summary'] = warning_counts
    
    # Store raw section, filtering out repetitive warning messages and unnecessary commands
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
        r'^\[.*\]\s*$',  # Empty timestamp lines
        # Filter out repetitive warnings
        r'\[WARNING RSZ-0075\]',  # Repetitive makeBufferedNet warnings
        # Filter out repetitive PDN macro grid insertions
        r'\[INFO PDN-0001\] Inserting grid: .+/macro_mem',  # Individual macro grid insertions
    ]
    
    # Keep patterns - metrics and important info only
    keep_patterns = [
        r'Running.*\.tcl, stage',  # Stage markers
        r'Elapsed time.*Memory',  # Performance metrics
        r'\[WARNING RSZ-0062\]',  # Critical timing warnings
        r'\[WARNING IFP-',  # Floorplan warnings
        r'\[WARNING EST-',  # Estimation warnings
        r'\[ERROR',  # Keep errors
        r'number instances in verilog',
        # Table content
        r'^\[.*\]\s+[-]+\s*$',  # Table separator lines
        r'^\[.*\]\s+\d+\s+\|.*\|\s+[+\-\d.e%]+\s+\|',  # Table rows with metrics
        r'^\[.*\]\s+(Iter|Removed|Inserted|Resized|Buffers|Pins|Swaps|Cloned|Gates|WNS|TNS|Viol|Endpts|Worst)',  # Table headers
        # Commands
        r'repair_timing|repair_design',
        r'Floorplan check_setup',
        r'No macros found',
    ]
    
    # Skip INFO/WARNING unless they match keep_patterns
    info_warning_skip = [
        r'^\[.*\]\s*\[INFO ',  # All INFO messages unless in keep_patterns
        r'^\[.*\]\s*\[WARNING ',  # All WARNING messages unless in keep_patterns
    ]
    
    for line in section_lines:
        # Check if line should be kept first (priority over skip patterns)
        should_keep = any(re.search(pattern, line) for pattern in keep_patterns)
        if should_keep:
            filtered_section.append(line)
            continue
        
        # Skip INFO/WARNING unless they match keep_patterns
        if any(re.search(pattern, line) for pattern in info_warning_skip):
            continue
        
        # Skip unnecessary commands and file operations
        should_skip = any(re.search(pattern, line) for pattern in skip_patterns)
        if should_skip:
            continue
            
        filtered_section.append(line)
    
    result['raw_section'] = '\n'.join(filtered_section)
    
    # Write to file if output_file is specified
    if output_file:
        with open(output_file, 'w') as f:
            # Just write the raw section (like other stages)
            f.write(result['raw_section'])
    
    return result


# Example usage
if __name__ == "__main__":
    # Read log file
    with open('gcdshy130.txt', 'r') as f:
        log_content = f.read()
    
    # Extract floorplan section and write to file
    output_file = 'floorplan_extract.txt'
    floorplan_data = extract_floorplan_section(log_content, output_file=output_file)
    
    print(f"Floorplan section extracted and saved to: {output_file}")
    print("\nFirst 50 lines of output:")
    print("-" * 80)
    
    # Print first 50 lines of the raw section
    lines = floorplan_data['raw_section'].split('\n')
    for line in lines[:50]:
        print(line)


