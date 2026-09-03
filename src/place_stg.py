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

def extract_placement_section(log_content: str, output_file: Optional[str] = None) -> Dict:
    """
    Extract placement section (Stage 3) from OpenROAD log file with placement metrics.
    
    Args:
        log_content: The full log file content as a string
        output_file: Optional path to write the extracted results
    
    Returns a dictionary containing:
    - section_bounds: Start and end lines of the placement section
    - substages: Information about each placement substage (3_1 through 3_5)
    - key_metrics: Final metrics (area, utilization, HPWL, etc.)
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
    
    # Find placement section boundaries
    start_idx = None
    end_idx = None
    
    for i, line in enumerate(lines):
        # Check for "Running" pattern (full execution logs)
        if '3_1_place_gp_skip_io' in line and 'Running' in line:
            if start_idx is None:
                start_idx = max(0, i - 2)  # Include separator line
        # Also check for "Report metrics stage 3" pattern (report logs)
        if 'Report metrics stage 3' in line and start_idx is None:
            start_idx = max(0, i - 2)  # Include separator line
        
        # Look for the end of the last placement substage (3_5_place_dp)
        if '3_5_place_dp' in line and 'sha1sum' in line:
            end_idx = i + 1
            break
        # Alternative ending: when we see stage 4 starting
        if start_idx is not None and '4_1_cts' in line and 'Running' in line:
            end_idx = i
            break
    
    if start_idx is None:
        return result
    if end_idx is None:
        # If we didn't see a clear stage-4 boundary or the final sha1sum row,
        # treat end-of-input as end.
        end_idx = len(lines)
    
    result['section_bounds'] = (start_idx, end_idx)
    
    # Extract and compact the section
    section_lines = lines[start_idx:end_idx]
    
    # Extract placement substages (3_1, 3_2, 3_3, 3_4, 3_5)
    substages = {}
    current_stage = None
    
    for i, line in enumerate(section_lines):
        # Detect stage starts
        stage_match = re.search(r'(3_[1-5]_place[_\w]*)', line)
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
                # Capture GPL (global placement) initialization and metrics
                gpl_metrics = [
                    'Execute conjugate gradient initial placement',
                    'DBU:', 'SiteSize:', 'CoreBBox:', 'Die BBox:',
                    'Number of instances:', 'Movable instances:', 'Fixed instances:', 'Dummy instances:',
                    'Number of nets:', 'Number of pins:', 
                    'Core area:', 'Fixed instances area:', 'Movable instances area:',
                    'Utilization:', 'Standard cells area:', 'Large instances area:',
                    'Placement target density:', 'Movable insts average area:',
                    'Ideal bin area:', 'Ideal bin count:', 'Total bin area:',
                    'Bin count', 'Bin size', 'Number of bins:',
                    'Execute nesterov global placement',
                    'Final placement area:', 'Global placement finished',
                    'Minimum Feasible Density',
                    'Placed Cell Area', 'Available Free Area',
                    'Suggested Target Densities', 'usage of free space',
                    'Routability iteration:', 'Total routing overflow:', 
                    'Number of overflowed tiles:'
                ]
                
                for metric in gpl_metrics:
                    if metric in line:
                        info_match = re.search(r'\[INFO [A-Z]+-\d+\] (.+)', line)
                        if info_match:
                            substages[current_stage]['info'].append(info_match.group(1))
                        else:
                            clean_line = line.split('] ', 1)[-1] if ']' in line else line
                            substages[current_stage]['info'].append(clean_line.strip())
                        break
                
                # Capture placement analysis results (summary only)
                if 'Placement Analysis' in line:
                    substages[current_stage]['info'].append('--- Placement Analysis ---')
                elif 'total displacement' in line or 'average displacement' in line or 'max displacement' in line:
                    clean_line = line.split('] ', 1)[-1] if ']' in line else line
                    substages[current_stage]['info'].append(clean_line.strip())
                elif 'original HPWL' in line or 'legalized HPWL' in line or 'delta HPWL' in line:
                    clean_line = line.split('] ', 1)[-1] if ']' in line else line
                    substages[current_stage]['info'].append(clean_line.strip())
                
                # Capture critical timing-driven and routability info only
                if 'Timing-driven: worst slack' in line:
                    info_match = re.search(r'\[INFO [A-Z]+-\d+\] (.+)', line)
                    if info_match:
                        substages[current_stage]['info'].append(info_match.group(1))
                elif 'Timing-driven: repair_design delta area' in line:
                    info_match = re.search(r'\[INFO [A-Z]+-\d+\] (.+)', line)
                    if info_match:
                        substages[current_stage]['info'].append(info_match.group(1))
                elif 'Routability finished' in line or 'Target routing congestion achieved' in line:
                    info_match = re.search(r'\[INFO [A-Z]+-\d+\] (.+)', line)
                    if info_match:
                        substages[current_stage]['info'].append(info_match.group(1))
                
                # Capture I/O placement summary only
                if 'PPL-' in line and any(kw in line for kw in ['Number of I/O', 'I/O nets HPWL', 'assigned pins']):
                    info_match = re.search(r'\[INFO PPL-\d+\] (.+)', line)
                    if info_match:
                        substages[current_stage]['info'].append(info_match.group(1))
                
                # Capture detailed placement summary only
                if 'DPL-' in line and any(kw in line for kw in ['End of', 'improvement', 'Final HPWL', 'Mirrored']):
                    info_match = re.search(r'\[INFO DPL-\d+\] (.+)', line)
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
            log_summary_match = re.search(r'(3_[1-5]_place[_\w]*)\s+(\d+)\s+(\d+)\s+([a-f0-9]+)', line)
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
        
        # Instance counts
        if 'Instance count before' in line:
            match = re.search(r'Instance count before (\d+), after (\d+)', line)
            if match:
                result['key_metrics']['instances_before'] = int(match.group(1))
                result['key_metrics']['instances_after'] = int(match.group(2))
    
    # Count warnings (compact summary)
    warning_counts = {}
    for line in section_lines:
        warning_match = re.search(r'\[(WARNING|ERROR) ([A-Z]+-\d+)\]', line)
        if warning_match:
            level, code = warning_match.groups()
            key = f"{level} {code}"
            warning_counts[key] = warning_counts.get(key, 0) + 1
    
    result['warnings_summary'] = warning_counts
    
    # Store raw section with substage info and key metrics, filtering unnecessary verbose output
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
        r'\[InitialPlace\]\s+Iter:',
        r'Perform port buffering',
        r'Found \d+ macro blocks',
        r'Using \d+ tracks',
        r'Elapsed time:.*CPU time:.*Peak memory:',  # Skip elapsed time with CPU and memory details
    ]
    
    # Keep patterns - ensure these are NOT filtered even if they match skip patterns
    keep_patterns = [
        r'Design area.*um\^2.*% utilization',  # Utilization metric
        r'Design area.*u\^2.*% utilization',  # Utilization metric (alternative format)
        r'Total displacement.*um',
        r'Average displacement.*um',
        r'Setup.*slack',
        r'Hold.*slack',
        r'WNS',
        r'TNS',
        r'Running.*\.tcl, stage',  # Stage markers
        r'HPWL:.*um',  # HPWL metrics with values
        r'I/O nets HPWL',  # I/O placement HPWL
        r'avg displacement',  # Displacement summary
        r'Inserted \d+ buffers',  # Buffer insertion summary
        r'resized \d+ instances',  # Resizing summary
        r'Overflow',  # Overflow info
        r'Execute conjugate gradient',  # GPL initialization
        r'DBU:',  # GPL DBU
        r'SiteSize:',  # GPL site size
        r'CoreBBox:',  # Core bounding box
        r'Die BBox:',  # Die bounding box
        r'Number of instances:',  # Instance count
        r'Movable instances:',  # Movable instances
        r'Fixed instances:',  # Fixed instances
        r'Dummy instances:',  # Dummy instances
        r'Number of nets:',  # Net count
        r'Number of pins:',  # Pin count
        r'Core area:',  # Core area
        r'Fixed instances area:',  # Fixed area
        r'Movable instances area:',  # Movable area
        r'Utilization:',  # Utilization
        r'Standard cells area:',  # Standard cells area
        r'Large instances area:',  # Large instances area
        r'Placement target density:',  # Target density
        r'Movable insts average area:',  # Average area
        r'Ideal bin area:',  # Ideal bin area
        r'Ideal bin count:',  # Ideal bin count
        r'Total bin area:',  # Total bin area
        r'Bin count \(X, Y\):',  # Bin count
        r'Bin size \(W \* H\):',  # Bin size
        r'Number of bins:',  # Number of bins
        r'Execute nesterov global placement',  # GPL nesterov placement
        r'Global placement finished',  # Global placement completion
        r'Placed Cell Area',  # Placed cell area metric
        r'Available Free Area',  # Available free area metric
        r'Minimum Feasible Density',  # Minimum feasible density
        r'Suggested Target Densities',  # Suggested densities header
        r'usage of free space:',  # Free space usage metrics
        r'Original area \(um\^2\):',  # Original area before optimization
        r'routability artificial inflation',  # Routability inflation
        r'timing-driven delta area',  # Timing-driven area changes
        r'Final placement area:',  # Final placement area
        r'Routability mode iteration count',  # Routability iterations
        r'Routability iteration:',  # Individual routability iteration
        r'Total routing overflow:',  # Total routing overflow metric
        r'Number of overflowed tiles:',  # Overflowed tiles count
        r'Routability final weighted congestion',  # Routability congestion
        r'Final HPWL',  # Final HPWL
        r'\[ERROR',  # Keep errors
        r'antenna diodes',
        r'global_placement|place_pins|detailed_placement|improve_placement',  # Commands
        r'repair_design|repair_timing|estimate_parasitics',  # Repair commands
        r'Iteration.*Overflow.*HPWL',  # Table headers
        r'^\[.*\]\s+[-]+\s*$',  # Table separator lines (dashes)
        r'^\[.*\]\s+\d+\s+\|.*\|\s+[+\-\d.e%]+\s+\|',  # Table rows with pipe-separated values
        r'^\[.*\]\s+(Iter|Area|Removed|Inserted|Resized|Buffers|Pins|Nets|Remaining)',  # Other table headers
    ]
    
    # Skip INFO/WARNING unless they match keep_patterns
    info_warning_skip = [
        r'^\[.*\]\s*\[INFO ',  # All INFO messages unless in keep_patterns
        r'^\[.*\]\s*\[WARNING ',  # All WARNING messages unless in keep_patterns
    ]
    
    for line in section_lines:
        # Skip empty lines or lines with only whitespace
        if not line.strip():
            continue
        
        # Skip lines that only have timestamp with no content
        # Matches lines like "[2025-10-11T16:02:31.133Z] " with nothing after
        if re.match(r'^\[[\d\-T:.Z]+\]\s*$', line):
            continue
        
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
            # Just write the raw section which now includes summary at the top
            f.write(result['raw_section'])
    
    return result


# Example usage
if __name__ == "__main__":
    # Read log file
    with open('gcdsky130_report.log', 'r') as f:
        log_content = f.read()
    
    # Extract placement section and write to file
    output_file = 'placement_extract.txt'
    placement_data = extract_placement_section(log_content, output_file=output_file)
    
    print(f"Placement section extracted and saved to: {output_file}")
    print("\nFirst 50 lines of output:")
    print("-" * 80)
    
    # Print first 50 lines of the raw section
    lines = placement_data['raw_section'].split('\n')
    for line in lines[:50]:
        print(line)

