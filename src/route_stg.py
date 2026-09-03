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

def extract_route_section(log_content: str, output_file: Optional[str] = None) -> Dict:
    """
    Extract Routing section (Stage 5) from OpenROAD log file with routing metrics.
    
    Args:
        log_content: The full log file content as a string
        output_file: Optional path to write the extracted results
    
    Returns a dictionary containing:
    - section_bounds: Start and end lines of the routing section
    - substages: Information about each routing substage (5_1, 5_2, 5_3)
    - key_metrics: Final metrics (area, utilization, wire length, vias, etc.)
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
    
    # Find routing section boundaries
    start_idx = None
    end_idx = None
    
    for i, line in enumerate(lines):
        # Check for "Running" pattern (full execution logs)
        if '5_1_grt' in line and 'Running' in line:
            if start_idx is None:
                start_idx = max(0, i - 2)  # Include separator line
        # Also check for "Report metrics stage 5" pattern (report logs)
        if 'Report metrics stage 5' in line and start_idx is None:
            start_idx = max(0, i - 2)  # Include separator line
        # Look for the end of the last routing substage (5_3_fillcell)
        if '5_3_fillcell' in line and 'sha1sum' in line:
            end_idx = i + 1
        # Alternative ending: when we see stage 6 starting
        if start_idx is not None and '6_1_fill' in line and ('Running' in line or 'density_fill' in line):
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
    
    # Extract routing substages (5_1, 5_2, 5_3)
    substages = {}
    current_stage = None
    
    for i, line in enumerate(section_lines):
        # Detect stage starts
        stage_match = re.search(r'(5_[1-3]_[a-z_]+)', line)
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
            
            # Capture timing and memory metrics
            elapsed_match = re.search(r'Elapsed time: ([\d:.]+)\[h:\]min:sec', line)
            if elapsed_match:
                substages[current_stage]['elapsed_time'] = elapsed_match.group(1)
            
            memory_match = re.search(r'Peak memory: (\d+)KB', line)
            if memory_match:
                substages[current_stage]['peak_memory'] = int(memory_match.group(1))
            
            # Capture log summary line with sha1sum
            log_summary_match = re.search(r'(5_[1-3]_[a-z_]+)\s+(\d+)\s+(\d+)\s+([a-f0-9]+)', line)
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
        
        # Wire length
        if 'Total wirelength:' in line:
            match = re.search(r'Total wirelength: (\d+) um', line)
            if match:
                result['key_metrics']['total_wirelength'] = int(match.group(1))
        
        # Vias
        if 'Final number of vias:' in line:
            match = re.search(r'Final number of vias: (\d+)', line)
            if match:
                result['key_metrics']['final_vias'] = int(match.group(1))
    
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
        # Filter out repetitive warnings
        r'\[WARNING RSZ-0073\]',  # Repetitive driver pin warnings
        r'\[WARNING RSZ-0075\]',  # Repetitive makeBufferedNet warnings
        r'\[WARNING DRT-0349\]',  # LEF58_ENCLOSURE warnings (not timing related)
        # Filter out descriptive info not related to metrics
        r'^\[.*\]\s*$',  # Empty timestamp lines
        r'Units:',
        r'Number of layers:',
        r'Number of macros:',
        r'Number of vias:',
        r'Number of viarulegen:',
        r'Die area:',
        r'Number of track patterns:',
        r'Number of DEF vias:',
        r'Number of components:',
        r'Number of terminals:',
        r'Number of snets:',
        r'Number of nets:',
        r'Layer via[0-9]?$',
        r'default via:',
        r'#scanned instances',
        r'#unique\s+instances',
        r'#stdCell',
        r'#macro',
        r'#instTerm',
        r'Design:.*gcd$',
        r'global_route -congestion',  # Command echo (not metrics)
        # Filter detailed optimization progress
        r'^\[.*\]\s+Completing \d+% with \d+ violations',  # Detailed completion percentage
        r'^\[.*\]\s+elapsed time = \d+:\d+:\d+, memory =',  # Elapsed time during optimization
        # Filter per-layer wire length details
        r'^\[.*\]\s+Total wire length on LAYER',  # Per-layer wire length breakdown
    ]
    
    # Keep patterns - ensure these are NOT filtered even if they match skip patterns
    keep_patterns = [
        r'Design area.*um\^2.*% utilization',  # Utilization metric
        r'Design area.*u\^2.*% utilization',  # Utilization metric (alternative format)
        r'Running.*\.tcl, stage',  # Stage markers
        r'Elapsed time.*Memory',  # Performance metrics
        # HPWL metrics
        r'original HPWL.*\d+\s+um',  # HPWL with values
        r'legalized HPWL.*\d+\s+um',  # HPWL with values
        r'delta HPWL.*\d+\s+%',  # HPWL change
        # Routing specific - timing, wirelength, violations
        r'\[INFO GRT-0018\].*Total wirelength',  # Wirelength
        r'\[INFO GRT-0012\].*antenna violations',  # Antenna violations
        r'\[INFO ANT-000[1-2]\].*violations',  # Antenna violation details
        r'\[INFO DRT-0199\].*violations',  # DRC Violation count
        r'\[INFO DRT-0195\].*optimization iteration',  # Iteration markers
        # Repair timing - only summary info
        r'\[INFO RSZ-0094\].*Found.*endpoints with.*violations',  # Timing violations
        r'\[WARNING RSZ-0062\]',  # Critical timing warnings
        r'\[ERROR',  # Keep errors
        # Commands - only key routing commands
        r'repair_timing|repair_design',
        r'global_route|detailed_route|detailed_placement',
        # Table content
        r'^\[.*\]\s+[-]+\s*$',  # Table separator lines (dashes)
        r'^\[.*\]\s+\d+\s+\|.*\|\s+[+\-\d.e%]+\s+\|',  # Table rows with metrics
        r'^\[.*\]\s+(Iter|Removed|Inserted|Resized|Buffers|Pins|Swaps|Cloned|Gates|WNS|TNS|Viol|Endpts|Worst)',  # Table headers
        r'Routing\s+Direction|Layer\s+Resources',  # Routing table headers
        r'Total wire length = \d+',  # Total wire length (not per-layer)
        r'Up-via summary',  # Via summary
        r'Viol/Layer',  # Violation by layer
        r'Metal Spacing|Short|Recheck',  # Violation types
        r'Placement Analysis',  # Placement analysis
        r'Took \d+ seconds:',  # Timing info
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
    with open('bp_multi_top_nangate45.txt', 'r') as f:
        log_content = f.read()
    
    # Extract routing section and write to file
    output_file = 'route_extract.txt'
    route_data = extract_route_section(log_content, output_file=output_file)
    
    print(f"Routing section extracted and saved to: {output_file}")
    print("\nFirst 50 lines of output:")
    print("-" * 80)
    
    # Print first 50 lines of the raw section
    lines = route_data['raw_section'].split('\n')
    for line in lines[:50]:
        print(line)

