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

def extract_cts_section(log_content: str, output_file: Optional[str] = None) -> Dict:
    """
    Extract CTS (Clock Tree Synthesis) section (Stage 4) from OpenROAD log file with CTS metrics.
    
    Args:
        log_content: The full log file content as a string
        output_file: Optional path to write the extracted results
    
    Returns a dictionary containing:
    - section_bounds: Start and end lines of the CTS section
    - substages: Information about each CTS substage (4_1, etc.)
    - key_metrics: Final metrics (area, utilization, clock buffers, etc.)
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
    
    # Find CTS section boundaries
    start_idx = None
    end_idx = None
    
    for i, line in enumerate(lines):
        # Check for "Running" pattern (full execution logs)
        if '4_1_cts' in line and 'Running' in line:
            if start_idx is None:
                start_idx = max(0, i - 2)  # Include separator line
        # Also check for "Report metrics stage 4" pattern (report logs)
        if 'Report metrics stage 4' in line and start_idx is None:
            start_idx = max(0, i - 2)  # Include separator line
        # Look for the end of the last CTS substage
        if '4_1_cts' in line and 'sha1sum' in line:
            end_idx = i + 1
        # Alternative ending: when we see stage 5 starting
        if start_idx is not None and '5_1_grt' in line and ('Running' in line or 'global_route' in line):
            end_idx = i
            break
    
    if start_idx is None:
        return result
    if end_idx is None:
        # If we didn't see the next stage starting inside this chunk, treat end-of-input as end.
        end_idx = len(lines)
    
    result['section_bounds'] = (start_idx, end_idx)
    
    # Extract and compact the section
    section_lines = lines[start_idx:end_idx]
    
    # Extract CTS substages (4_1, etc.)
    substages = {}
    current_stage = None
    
    for i, line in enumerate(section_lines):
        # Detect stage starts
        stage_match = re.search(r'(4_[1-9]_cts[_\w]*)', line)
        if stage_match and 'Running' in line:
            current_stage = stage_match.group(1)
            substages[current_stage] = {
                'name': current_stage,
                'info': [],
                'metrics': {},
                'tables': [],
                'warnings': [],
                'errors': [],
                'elapsed_time': None,
                'peak_memory': None,
                'sha1sum': None
            }
        
        # Capture stage-specific info
        if current_stage and current_stage in substages:
            # Capture warnings and errors for this substage
            warning_match = re.search(r'\[WARNING ([A-Z]+-\d+)\] (.+)', line)
            if warning_match:
                code, msg = warning_match.groups()
                substages[current_stage]['warnings'].append(f"[{code}] {msg}")
            
            error_match = re.search(r'\[ERROR ([A-Z]+-\d+)\] (.+)', line)
            if error_match:
                code, msg = error_match.groups()
                substages[current_stage]['errors'].append(f"[{code}] {msg}")
            
            # Skip unnecessary commands and file operations
            skip_keywords = [
                'read_liberty', 'read_db', 'cp ', 'mkdir', 'source ',
                'link_design', 'Master ', 'LEF file:', 'read_', '/tmp/workspace'
            ]
            
            should_skip = any(keyword in line for keyword in skip_keywords)
            
            if not should_skip:
                # Capture CTS metrics
                cts_metrics = [
                    'Root buffer', 'Sink buffer', 'clock buffers will be used',
                    'Characterization buffer', 'Clock net', 'found for clock',
                    'TritonCTS found', 'clock nets', 'Total number of sinks',
                    'Created', 'clock buffers', 'Minimum number of buffers',
                    'Maximum number of buffers', 'clock nets', 'Fanout distribution',
                    'Max level of the clock tree', 'Average sink wire length',
                    'Path depth', 'Leaf load cells', 'Leaf buffers'
                ]
                
                for metric in cts_metrics:
                    if metric in line:
                        info_match = re.search(r'\[INFO CTS-\d+\] (.+)', line)
                        if info_match:
                            substages[current_stage]['info'].append(info_match.group(1))
                        else:
                            clean_line = line.split('] ', 1)[-1] if ']' in line else line
                            substages[current_stage]['info'].append(clean_line.strip())
                        break
                
                # Capture placement analysis results
                if 'Placement Analysis' in line:
                    substages[current_stage]['info'].append('--- Placement Analysis ---')
                elif 'total displacement' in line or 'average displacement' in line or 'max displacement' in line:
                    clean_line = line.split('] ', 1)[-1] if ']' in line else line
                    substages[current_stage]['info'].append(clean_line.strip())
                elif 'original HPWL' in line or 'legalized HPWL' in line or 'delta HPWL' in line:
                    clean_line = line.split('] ', 1)[-1] if ']' in line else line
                    substages[current_stage]['info'].append(clean_line.strip())
                
                # Capture repair timing info
                if 'repair_timing' in line and any(kw in line for kw in ['-verbose', '-setup_margin', '-hold_margin']):
                    clean_line = line.split('] ', 1)[-1] if ']' in line else line
                    substages[current_stage]['info'].append(clean_line.strip())
                
                # Capture repair summary
                if 'RSZ-' in line and any(kw in line for kw in ['Removed', 'Inserted', 'Resized', 'Swapped pins', 'Cloned']):
                    info_match = re.search(r'\[INFO RSZ-\d+\] (.+)', line)
                    if info_match:
                        substages[current_stage]['info'].append(info_match.group(1))
                
                # Capture timing violations
                if 'RSZ-' in line and any(kw in line for kw in ['endpoints with setup violations', 'endpoints with hold violations', 'Repairing']):
                    info_match = re.search(r'\[INFO RSZ-\d+\] (.+)', line)
                    if info_match:
                        substages[current_stage]['info'].append(info_match.group(1))
            
            # Capture timing and memory metrics
            elapsed_match = re.search(r'Elapsed time: ([\d:.]+)\[h:\]min:sec', line)
            if elapsed_match:
                substages[current_stage]['elapsed_time'] = elapsed_match.group(1)
            
            memory_match = re.search(r'Peak memory: (\d+)KB', line)
            if memory_match:
                substages[current_stage]['peak_memory'] = int(memory_match.group(1))
            
            # Capture log summary line with sha1sum
            log_summary_match = re.search(r'(4_[1-9]_cts[_\w]*)\s+(\d+)\s+(\d+)\s+([a-f0-9]+)', line)
            if log_summary_match:
                stage_name, elapsed_s, memory_mb, sha1 = log_summary_match.groups()
                if stage_name in substages:
                    substages[stage_name]['elapsed_s'] = int(elapsed_s)
                    substages[stage_name]['memory_mb'] = int(memory_mb)
                    substages[stage_name]['sha1sum'] = sha1
    
    result['substages'] = substages
    
    # Extract final key metrics
    for line in section_lines:
        # Final design area and utilization
        if 'Design area' in line and 'utilization' in line:
            match = re.search(r'Design area (\d+) u\^2 (\d+)% utilization', line)
            if match:
                result['key_metrics']['final_area'] = int(match.group(1))
                result['key_metrics']['final_utilization'] = int(match.group(2))
        
        # Clock tree metrics
        if 'Created' in line and 'clock buffers' in line:
            match = re.search(r'Created (\d+) clock buffers', line)
            if match:
                result['key_metrics']['clock_buffers'] = int(match.group(1))
        
        if 'Created' in line and 'clock nets' in line:
            match = re.search(r'Created (\d+) clock nets', line)
            if match:
                result['key_metrics']['clock_nets'] = int(match.group(1))
    
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
        r'^\[.*\] Iteration \|.*\|\s+\d+\s+\|.*Remaining$',
        r'Perform port buffering',
        r'Found \d+ macro blocks',
        r'Using \d+ tracks',
        r'^\[.*\]\s+Level \d+$',  # Clock tree hierarchy level
        r'^\[.*\]\s+Direction:',  # Clock tree direction
        r'^\[.*\]\s+Sinks per sub-region:',  # Sub-region sink count
        r'^\[.*\]\s+Sub-region size:',  # Sub-region dimensions
    ]
    
    # Keep patterns - ensure these are NOT filtered even if they match skip patterns
    keep_patterns = [
        r'Design area.*um\^2.*% utilization',  # Utilization metric
        r'Design area.*u\^2.*% utilization',  # Utilization metric (alternative format)
        r'Total displacement.*um',
        r'Average displacement.*um',
        r'max displacement.*um',
        r'Setup.*slack',
        r'Hold.*slack',
        r'WNS',
        r'TNS',
        r'Running.*\.tcl, stage',  # Stage markers
        r'Elapsed time.*Memory',  # Performance metrics
        # CTS specific INFO
        r'\[INFO CTS-0010\].*Clock net.*sinks',  # Clock net info
        r'\[INFO CTS-0028\].*Total number of sinks',  # Sink count
        r'\[INFO CTS-0018\].*Created.*clock buffers',  # Buffer count
        r'\[INFO CTS-0012\].*Minimum number of buffers',  # Min buffers
        r'\[INFO CTS-0013\].*Maximum number of buffers',  # Max buffers
        r'\[INFO CTS-0098\].*Clock net',  # Clock net name
        r'\[INFO CTS-0099\].*Sinks',  # Sinks for each clock net
        r'\[INFO CTS-0100\].*Leaf buffers',  # Leaf buffers for each clock net
        r'\[INFO CTS-0101\].*Average sink wire length',  # Wire length
        r'\[INFO CTS-0102\].*Path depth',  # Path depth
        r'\[INFO CTS-0207\].*Leaf load cells',  # Leaf load cells
        r'\[INFO RSZ-0094\].*Found.*endpoints with.*violations',  # Timing violations
        r'\[WARNING RSZ-0062\]',  # Critical timing warnings
        r'\[ERROR',  # Keep errors
        r'clock_tree_synthesis',  # CTS command
        r'repair_timing',  # Repair commands
        # Table content
        r'^\[.*\]\s+[-]+\s*$',  # Table separator lines
        r'^\[.*\]\s+\d+\s+\|.*\|\s+[+\-\d.e%]+\s+\|',  # Table rows with metrics
        r'^\[.*\]\s+(Iter|Removed|Inserted|Resized|Buffers|Pins|Swaps|Cloned|Gates|WNS|TNS|Viol|Endpts|Worst)',  # Table headers
        # Placement analysis
        r'Placement Analysis',
        r'total displacement',
        r'average displacement',
        r'original HPWL',
        r'legalized HPWL',
        r'delta HPWL',
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
    
    # Extract CTS section and write to file
    output_file = 'cts_extract.txt'
    cts_data = extract_cts_section(log_content, output_file=output_file)
    
    print(f"CTS section extracted and saved to: {output_file}")
    print("\nFirst 50 lines of output:")
    print("-" * 80)
    
    # Print first 50 lines of the raw section
    lines = cts_data['raw_section'].split('\n')
    for line in lines[:50]:
        print(line)

