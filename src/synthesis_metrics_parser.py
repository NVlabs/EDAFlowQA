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


def extract_synthesis_metrics(log_content: str, design_name: Optional[str] = None, sub_design_name: Optional[str] = None, log_file: Optional[str] = None) -> Dict:
    """
    Extract synthesis metrics from synthesis log file.
    Handles multiple synthesis runs if present in the log.
    
    Args:
        log_content: Content of synthesis log/extract file as a string
        
    Returns:
        Dictionary containing:
        {
            'stage': 'synthesis',
            'runs': [
                {
                    'run_number': int,
                    'script_name': str,
                    'design_name': str,
                    'summary': {...},
                    'timing': {...},
                    'steps': [...],
                    'warnings': [...],
                    'statistics': {
                        'wires': int,
                        'wire_bits': int,
                        'public_wires': int,
                        'public_wire_bits': int,
                        'ports': int,
                        'port_bits': int,
                        'cells': {
                            'total_count': int,
                            'total_area': float,
                            'top_5_cell_by_count': [...]
                        },
                        'chip_area': {
                            'total_um2': float,
                            'sequential_elements_um2': float,
                            'sequential_elements_percent': float
                        }
                    },
                    'check': {
                        'problems_found': int,
                        'status': str
                    }
                }
            ]
        }
    """
    # Detect synthesis runs by looking for script execution patterns
    lines = log_content.split('\n')
    
    # Find synthesis run boundaries
    run_starts = []
    for i, line in enumerate(lines):
        # Look for synth.sh script executions which indicate a new synthesis run
        if '/synth.sh' in line and '.tcl' in line:
            script_match = re.search(r'synth\.sh\s+([^\s]+\.tcl)', line)
            if script_match:
                run_starts.append({
                    'line': i,
                    'script': script_match.group(1)
                })
    
    # If no explicit run markers found, treat entire log as single run
    if not run_starts:
        run_starts = [{'line': 0, 'script': 'synthesis'}]
    
    # Add end marker
    run_starts.append({'line': len(lines), 'script': None})
    
    # Process each run
    runs = []
    for run_idx in range(len(run_starts) - 1):
        start_line = run_starts[run_idx]['line']
        end_line = run_starts[run_idx + 1]['line']
        script_name = run_starts[run_idx]['script']
        
        # Extract script name (e.g., synth_canonicalize.tcl -> canonicalize)
        script_short_name = script_name
        if '/' in script_name:
            script_short_name = script_name.split('/')[-1]
        if script_short_name.endswith('.tcl'):
            script_short_name = script_short_name[:-4]
        if script_short_name.startswith('synth_'):
            script_short_name = script_short_name[6:]
        
        run_result = {
            'run_number': run_idx + 1,
            'script_name': script_short_name,
            'design_name': None,
            'summary': {
                'clock_period': None,
                'total_modules': None,
                'yosys_version': None,
                'total_cpu_time': None,
                'total_stages': None,
                'warnings': 0,
                'errors': 0
            },
            'steps': [],
            'timing': {
                'clock_period_ns': None,
                'clock_period_set_count': 0
            },
            'warnings': [],
            'statistics': {
                'wires': None,
                'wire_bits': None,
                'public_wires': None,
                'public_wire_bits': None,
                'ports': None,
                'port_bits': None,
                'cells': {
                    'total_count': None,
                    'total_area': None,
                    'top_5_cell_by_count': []
                },
                'chip_area': {
                    'total_um2': None,
                    'sequential_elements_um2': None,
                    'sequential_elements_percent': None
                }
            },
            'check': {
                'problems_found': None,
                'status': None
            }
        }
        
        # Temporary storage for all cells in this run
        all_cells = []
        
        # Process lines for this run
        run_lines = lines[start_line:end_line]
        
        for i, line in enumerate(run_lines):
            # Extract timestamp and message
            timestamp_match = re.match(r'\[([^\]]+)\]\s+(.*)', line)
            timestamp = timestamp_match.group(1) if timestamp_match else None
            message = timestamp_match.group(2) if timestamp_match else line
            
            # Extract design name (e.g., "=== gcd ===")
            match = re.search(r'===\s+(\w+)\s+===', message)
            if match:
                run_result['design_name'] = match.group(1)

            # Extract design name from KLayout invocation (common in flow logs):
            #   ... klayout.sh ... -rd design_name=uart_rx ...
            if run_result.get('design_name') is None:
                match = re.search(r'\bdesign_name=([A-Za-z0-9_.\-]+)\b', message)
                if match:
                    run_result['design_name'] = match.group(1)
            
            # If caller provided sub_design_name and we still don't have a run design name, use it.
            if run_result.get('design_name') is None and sub_design_name is not None:
                run_result['design_name'] = sub_design_name
            
            # Extract clock period (e.g., "Setting clock period to 4.8")
            match = re.search(r'Setting clock period to\s+([\d.]+)', message)
            if match:
                clock_period = float(match.group(1))
                run_result['timing']['clock_period_ns'] = clock_period
                run_result['summary']['clock_period'] = f"{clock_period} ns"
                run_result['timing']['clock_period_set_count'] += 1
            
            # Extract synthesis steps (numbered steps like "19.1. Analyzing design hierarchy..")
            match = re.search(r'^(\d+(?:\.\d+)?)\.\s+(.+)', message)
            if match:
                step_number = match.group(1)
                step_description = match.group(2).strip()
                run_result['steps'].append({
                    'number': step_number,
                    'description': step_description,
                    'timestamp': timestamp
                })
                run_result['summary']['total_stages'] = len(run_result['steps'])
            
            # Extract Yosys version
            match = re.search(r'Yosys\s+([\d.+]+)\s+\(git sha1 ([a-f0-9]+)', message, re.IGNORECASE)
            if match:
                run_result['summary']['yosys_version'] = f"{match.group(1)} (git sha1 {match.group(2)}"
            
            # Extract warnings
            if 'Warning:' in message:
                warning_text = message.split('Warning:', 1)[1].strip() if 'Warning:' in message else message
                run_result['warnings'].append({
                    'message': warning_text,
                    'timestamp': timestamp
                })
                run_result['summary']['warnings'] += 1
            
            # Extract unique warnings summary (e.g., "Warnings: 99 unique messages, 99 total")
            match = re.search(r'Warnings:\s+(\d+)\s+unique messages,\s+(\d+)\s+total', message)
            if match:
                run_result['summary']['warnings'] = int(match.group(2))
            
            # Extract errors
            if 'Error:' in message and 'no such object' not in message.lower():
                error_text = message.split('Error:', 1)[1].strip() if 'Error:' in message else message
                if 'errors' not in run_result:
                    run_result['errors'] = []
                run_result['errors'].append({
                    'message': error_text,
                    'timestamp': timestamp
                })
                run_result['summary']['errors'] += 1
            
            # Extract CPU time (e.g., "CPU time: user 75.70 sys 0.53")
            match = re.search(r'CPU time:\s+user\s+([\d.]+)s?\s+sys\s+([\d.]+)', message)
            if match:
                user_time = float(match.group(1))
                sys_time = float(match.group(2))
                total_cpu_time = user_time + sys_time
                run_result['summary']['total_cpu_time'] = total_cpu_time  # Store as numeric value
            
            # Extract number of modules (e.g., "Number of selected modules: 200")
            match = re.search(r'Number of selected modules:\s+(\d+)', message)
            if match:
                run_result['summary']['total_modules'] = int(match.group(1))
            
            # Extract wire count (local count, excluding submodules)
            match = re.search(r'^\s*(\d+)\s+-\s+wires', line)
            if match:
                run_result['statistics']['wires'] = int(match.group(1))
            
            # Extract wire bits
            match = re.search(r'^\s*(\d+)\s+-\s+wire bits', line)
            if match:
                run_result['statistics']['wire_bits'] = int(match.group(1))
            
            # Extract public wires
            match = re.search(r'^\s*(\d+)\s+-\s+public wires', line)
            if match:
                run_result['statistics']['public_wires'] = int(match.group(1))
            
            # Extract public wire bits
            match = re.search(r'^\s*(\d+)\s+-\s+public wire bits', line)
            if match:
                run_result['statistics']['public_wire_bits'] = int(match.group(1))
            
            # Extract ports
            match = re.search(r'^\s*(\d+)\s+-\s+ports\s*$', line)
            if match:
                run_result['statistics']['ports'] = int(match.group(1))
            
            # Extract port bits
            match = re.search(r'^\s*(\d+)\s+-\s+port bits', line)
            if match:
                run_result['statistics']['port_bits'] = int(match.group(1))
            
            # Extract total cell count and area (e.g., "242  2.4E+03 cells")
            match = re.search(r'^\s*(\d+)\s+([\d.E+\-]+)\s+cells', line)
            if match:
                run_result['statistics']['cells']['total_count'] = int(match.group(1))
                # Parse area (can be in scientific notation like 2.4E+03)
                area_str = match.group(2)
                run_result['statistics']['cells']['total_area'] = float(area_str)
            
            # Extract individual cell types (e.g., "1    10.01   sky130_fd_sc_hd__a2111oi_1")
            match = re.search(r'^\s*(\d+)\s+([\d.]+)\s+(sky130_\S+|[\w_]+__\w+)', line)
            if match:
                cell_count = int(match.group(1))
                cell_area = float(match.group(2))
                cell_type = match.group(3)
                all_cells.append({
                    'type': cell_type,
                    'count': cell_count,
                    'area': cell_area
                })
            
            # Extract chip area (e.g., "Chip area for module '\gcd': 2402.304000")
            match = re.search(r"Chip area for module ['\\\"]\\?(\w+)['\\\"]:\s+([\d.]+)", line)
            if match:
                run_result['statistics']['chip_area']['total_um2'] = float(match.group(2))
            
            # Extract sequential elements area and percentage
            # (e.g., "of which used for sequential elements: 1020.979200 (42.50%)")
            match = re.search(r'of which used for sequential elements:\s+([\d.]+)\s+\(([\d.]+)%\)', line)
            if match:
                run_result['statistics']['chip_area']['sequential_elements_um2'] = float(match.group(1))
                run_result['statistics']['chip_area']['sequential_elements_percent'] = float(match.group(2))
        
        # Calculate top 5 cells by count for this run
        if all_cells and run_result['statistics']['cells']['total_count'] is not None:
            total_count = run_result['statistics']['cells']['total_count']
            total_area = run_result['statistics']['cells']['total_area']
            
            # Sort by count (descending) and take top 5
            cells_by_count = sorted(all_cells, key=lambda x: x['count'], reverse=True)[:5]
            for cell in cells_by_count:
                run_result['statistics']['cells']['top_5_cell_by_count'].append({
                    'type': cell['type'],
                    'count': cell['count'],
                    'area': cell['area'],
                    'count_percent': round((cell['count'] / total_count) * 100, 2),
                    'area_percent': round((cell['area'] / total_area) * 100, 2)
                })
        
        # Parse check results for this run
        for line in run_lines:
            # Extract number of problems found (e.g., "Found and reported 0 problems.")
            match = re.search(r'Found and reported (\d+) problems?', line)
            if match:
                problems = int(match.group(1))
                run_result['check']['problems_found'] = problems
                run_result['check']['status'] = 'passed' if problems == 0 else 'failed'
        
        # Remove null values from this run
        run_result = remove_null_values(run_result)
        runs.append(run_result)
    
    # Build final result
    result = {
        'stage': 'synthesis',
        'total_runs': len(runs),
        'runs': runs
    }
    
    # Attach top-level design identifiers when provided by caller
    if design_name is not None:
        result['design_name'] = design_name
    if sub_design_name is not None:
        result['sub_design_name'] = sub_design_name
    if log_file is not None:
        result['log_file'] = log_file
    
    # If only one run and caller didn't provide a design_name, include the run design_name for backward compatibility
    if len(runs) == 1 and result.get('design_name') is None:
        result['design_name'] = runs[0].get('design_name')
    
    return result


def print_synthesis_summary(metrics: Dict):
    """Pretty print the extracted synthesis metrics."""
    print("=" * 80)
    print("SYNTHESIS STAGE METRICS SUMMARY")
    print("=" * 80)
    
    # Check if this is the new multi-run format
    if 'runs' in metrics and isinstance(metrics['runs'], list):
        print(f"\nTotal Synthesis Runs: {metrics.get('total_runs', len(metrics['runs']))}")
        
        for run_idx, run in enumerate(metrics['runs'], 1):
            print("\n" + "=" * 80)
            print(f"RUN #{run_idx}: {run.get('script_name', 'Unknown Script')}")
            print("=" * 80)
            _print_single_run_summary(run)
    else:
        # Backward compatibility: treat as single run
        _print_single_run_summary(metrics)


def _print_single_run_summary(metrics: Dict):
    """Helper function to print a single synthesis run summary."""
    # Design name
    if metrics.get('design_name'):
        print(f"\nDesign: {metrics['design_name']}")
    
    # Summary section
    if metrics.get('summary'):
        summary = metrics['summary']
        print("\n--- Summary ---")
        if summary.get('clock_period'):
            print(f"  Clock Period: {summary['clock_period']}")
        if summary.get('yosys_version'):
            print(f"  Yosys Version: {summary['yosys_version']}")
        if summary.get('total_modules') is not None:
            print(f"  Total Modules: {summary['total_modules']}")
        if summary.get('total_stages') is not None:
            print(f"  Total Synthesis Steps: {summary['total_stages']}")
        if summary.get('total_cpu_time') is not None:
            print(f"  Total CPU Time: {summary['total_cpu_time']:.2f} seconds")
        if summary.get('warnings') is not None:
            print(f"  Warnings: {summary['warnings']}")
        if summary.get('errors') is not None:
            print(f"  Errors: {summary['errors']}")
    
    # Timing section
    if metrics.get('timing'):
        timing = metrics['timing']
        if timing.get('clock_period_ns') is not None:
            print("\n--- Timing ---")
            print(f"  Clock Period: {timing['clock_period_ns']} ns")
    
    # Steps (show first 5 and last 5)
    if metrics.get('steps') and len(metrics['steps']) > 0:
        steps = metrics['steps']
        print(f"\n--- Synthesis Steps ({len(steps)} total) ---")
        display_steps = steps[:5] + (['...'] if len(steps) > 10 else []) + (steps[-5:] if len(steps) > 10 else [])
        for step in display_steps:
            if step == '...':
                print(f"  ...")
            else:
                print(f"  {step['number']}. {step['description'][:70]}...")
    
    # Warnings (show first 5)
    if metrics.get('warnings') and len(metrics['warnings']) > 0:
        warnings = metrics['warnings']
        print(f"\n--- Warnings ({len(warnings)} total) ---")
        for i, warning in enumerate(warnings[:5]):
            print(f"  {i+1}. {warning['message'][:70]}...")
        if len(warnings) > 5:
            print(f"  ... and {len(warnings) - 5} more warnings")
    
    # Statistics
    if metrics.get('statistics'):
        stats = metrics['statistics']
        print("\n--- Statistics ---")
        
        # Wires and ports
        if stats.get('wires') is not None:
            print(f"  Wires: {stats['wires']}")
        if stats.get('wire_bits') is not None:
            print(f"  Wire bits: {stats['wire_bits']}")
        if stats.get('public_wires') is not None:
            print(f"  Public wires: {stats['public_wires']}")
        if stats.get('public_wire_bits') is not None:
            print(f"  Public wire bits: {stats['public_wire_bits']}")
        if stats.get('ports') is not None:
            print(f"  Ports: {stats['ports']}")
        if stats.get('port_bits') is not None:
            print(f"  Port bits: {stats['port_bits']}")
        
        # Cells
        if stats.get('cells'):
            cells = stats['cells']
            print(f"\n  Cells:")
            if cells.get('total_count') is not None:
                print(f"    Total count: {cells['total_count']}")
            if cells.get('total_area') is not None:
                print(f"    Total area: {cells['total_area']:.2f} um^2")
            
            # Top 5 by count
            if cells.get('top_5_cell_by_count') and len(cells['top_5_cell_by_count']) > 0:
                print(f"\n    Top 5 Cell Types by Count:")
                for idx, cell in enumerate(cells['top_5_cell_by_count'], 1):
                    print(f"      {idx}. {cell['type']}")
                    print(f"         Count: {cell['count']} ({cell['count_percent']}% of total)")
                    print(f"         Area: {cell['area']:.3f} um^2 ({cell['area_percent']}% of total)")
        
        # Chip area
        if stats.get('chip_area'):
            chip_area = stats['chip_area']
            print(f"\n  Chip Area:")
            if chip_area.get('total_um2') is not None:
                print(f"    Total: {chip_area['total_um2']:.2f} um^2")
            if chip_area.get('sequential_elements_um2') is not None:
                print(f"    Sequential elements: {chip_area['sequential_elements_um2']:.2f} um^2 ({chip_area.get('sequential_elements_percent', 0):.2f}%)")
    
    # Check results
    if metrics.get('check'):
        check = metrics['check']
        print("\n--- Check Results ---")
        if check.get('status'):
            status_symbol = "[PASS]" if check['status'] == 'passed' else "[FAIL]"
            print(f"  Status: {status_symbol} {check['status'].upper()}")
        if check.get('problems_found') is not None:
            print(f"  Problems found: {check['problems_found']}")


# Example usage
if __name__ == "__main__":
    import sys
    import json
    
    # Read the synthesis log/extract file
    synth_log_file = 'ariane133_nan45/ariane133_nan45_synthesis_extract.txt'  # or synthesis log file
    
    try:
        with open(synth_log_file, 'r') as f:
            log_content = f.read()
        
        # Extract metrics
        metrics = extract_synthesis_metrics(log_content)
        
        # Print summary
        print_synthesis_summary(metrics)
        
        # Save to JSON
        output_file = 'synthesis_metrics.json'
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n\nMetrics saved to {output_file}")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
