#!/usr/bin/env python3
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

"""
Extract timing metrics and path information from OpenROAD .rpt files.

This script extracts:
- Metrics: TNS, WNS, worst slack, clock period, violation counts
- Max (setup) path: startpoint, endpoint, slack
- Min (hold) path: startpoint, endpoint, slack
"""

import re
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Any


class ReportMetricExtractor:
    """Extract timing metrics and path information from .rpt files."""
    
    def __init__(self, rpt_file: str):
        """Initialize the extractor with a report file path."""
        self.rpt_file = Path(rpt_file)
        self.content = self._read_file()
        self.stage = self._extract_stage()
        
    def _read_file(self) -> str:
        """Read the report file content."""
        try:
            with open(self.rpt_file, 'r') as f:
                return f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Report file not found: {self.rpt_file}")
        except Exception as e:
            raise Exception(f"Error reading file {self.rpt_file}: {e}")
    
    def _extract_stage(self) -> str:
        """Extract the stage name from the report content or filename."""
        # Try to extract from content (e.g., "floorplan final report_tns")
        stage_patterns = [
            r'^([a-z_ ]+)\s+report_',  # e.g., "floorplan final report_tns"
            r'^([a-z_ ]+)\s+max_',     # e.g., "detailed place max_slew_check_slack"
        ]
        
        for pattern in stage_patterns:
            match = re.search(pattern, self.content, re.MULTILINE | re.IGNORECASE)
            if match:
                stage_name = match.group(1).strip()
                # Clean up stage name
                stage_name = re.sub(r'\s+', '_', stage_name)
                return stage_name
        
        # Fallback: extract from filename
        filename = self.rpt_file.stem  # Get filename without extension
        # Remove common suffixes
        stage_name = filename.replace('_final', '').replace('_rpt', '')
        return stage_name
    
    def extract_all(self) -> Dict[str, Any]:
        """Extract all metrics and path information."""
        result = {
            'stage': self.stage,
            'metrics': self.extract_metrics(),
        }
        
        # Include paths if they exist
        max_paths = self.extract_max_paths()
        if max_paths:
            result['max_paths'] = max_paths
        
        min_paths = self.extract_min_paths()
        if min_paths:
            result['min_paths'] = min_paths
        
        return result
    
    def extract_metrics(self) -> Dict[str, Any]:
        """Extract timing metrics from the report."""
        metrics = {}
        
        # TNS (Total Negative Slack) - max (setup)
        tns_max_match = re.search(r'tns max\s+([-\d.]+)', self.content)
        if tns_max_match:
            metrics['tns_max'] = float(tns_max_match.group(1))
        
        # TNS (Total Negative Slack) - min (hold)
        tns_min_match = re.search(r'tns min\s+([-\d.]+)', self.content)
        if tns_min_match:
            metrics['tns_min'] = float(tns_min_match.group(1))
        
        # WNS (Worst Negative Slack) - max (setup)
        wns_max_match = re.search(r'wns max\s+([-\d.]+)', self.content)
        if wns_max_match:
            metrics['wns_max'] = float(wns_max_match.group(1))
        
        # WNS (Worst Negative Slack) - min (hold)
        wns_min_match = re.search(r'wns min\s+([-\d.]+)', self.content)
        if wns_min_match:
            metrics['wns_min'] = float(wns_min_match.group(1))
        
        # Worst Slack - max (setup)
        worst_slack_max_match = re.search(r'worst slack max\s+([-\d.]+)', self.content)
        if worst_slack_max_match:
            metrics['worst_slack_max'] = float(worst_slack_max_match.group(1))
        
        # Worst Slack - min (hold)
        worst_slack_min_match = re.search(r'worst slack min\s+([-\d.]+)', self.content)
        if worst_slack_min_match:
            metrics['worst_slack_min'] = float(worst_slack_min_match.group(1))
        
        # Clock minimum period and fmax
        clock_period_match = re.search(
            r'clk period_min\s*=\s*([-\d.]+)\s+fmax\s*=\s*([-\d.]+)', 
            self.content
        )
        if clock_period_match:
            metrics['clock_period_min'] = float(clock_period_match.group(1))
            metrics['fmax'] = float(clock_period_match.group(2))
        
        # Setup violation count
        setup_viol_match = re.search(r'setup violation count\s+(\d+)', self.content)
        if setup_viol_match:
            metrics['setup_violation_count'] = int(setup_viol_match.group(1))
        
        # Hold violation count
        hold_viol_match = re.search(r'hold violation count\s+(\d+)', self.content)
        if hold_viol_match:
            metrics['hold_violation_count'] = int(hold_viol_match.group(1))
        
        # Max slew violation count
        max_slew_viol_match = re.search(r'max slew violation count\s+(\d+)', self.content)
        if max_slew_viol_match:
            metrics['max_slew_violation_count'] = int(max_slew_viol_match.group(1))
        
        # Max fanout violation count
        max_fanout_viol_match = re.search(r'max fanout violation count\s+(\d+)', self.content)
        if max_fanout_viol_match:
            metrics['max_fanout_violation_count'] = int(max_fanout_viol_match.group(1))
        
        # Max capacitance violation count
        max_cap_viol_match = re.search(r'max cap violation count\s+(\d+)', self.content)
        if max_cap_viol_match:
            metrics['max_cap_violation_count'] = int(max_cap_viol_match.group(1))
        
        # DRV metrics (slew, fanout, capacitance)
        max_slew_slack_match = re.search(r'max_slew_check_slack\s*\n[-]+\n([-\d.eE+]+)', self.content)
        if max_slew_slack_match:
            metrics['max_slew_check_slack'] = float(max_slew_slack_match.group(1))
        
        max_slew_limit_match = re.search(r'max_slew_check_limit\s*\n[-]+\n([-\d.eE+]+)', self.content)
        if max_slew_limit_match:
            metrics['max_slew_check_limit'] = float(max_slew_limit_match.group(1))
        
        max_fanout_slack_match = re.search(r'max_fanout_check_slack\s*\n[-]+\n([-\d.eE+]+)', self.content)
        if max_fanout_slack_match:
            metrics['max_fanout_check_slack'] = float(max_fanout_slack_match.group(1))
        
        max_fanout_limit_match = re.search(r'max_fanout_check_limit\s*\n[-]+\n([-\d.eE+]+)', self.content)
        if max_fanout_limit_match:
            metrics['max_fanout_check_limit'] = float(max_fanout_limit_match.group(1))
        
        max_cap_slack_match = re.search(r'max_capacitance_check_slack\s*\n[-]+\n([-\d.eE+]+)', self.content)
        if max_cap_slack_match:
            metrics['max_capacitance_check_slack'] = float(max_cap_slack_match.group(1))
        
        max_cap_limit_match = re.search(r'max_capacitance_check_limit\s*\n[-]+\n([-\d.eE+]+)', self.content)
        if max_cap_limit_match:
            metrics['max_capacitance_check_limit'] = float(max_cap_limit_match.group(1))
        
        # Critical path delay
        critical_path_delay_match = re.search(r'critical path delay\s*\n[-]+\n([-\d.]+)', self.content)
        if critical_path_delay_match:
            metrics['critical_path_delay'] = float(critical_path_delay_match.group(1))
        
        # Critical path slack
        critical_path_slack_match = re.search(r'critical path slack\s*\n[-]+\n([-\d.]+)', self.content)
        if critical_path_slack_match:
            metrics['critical_path_slack'] = float(critical_path_slack_match.group(1))
        
        # Design area and utilization
        design_area_match = re.search(r'report_design_area\s*\n[-]+\s*\nDesign area\s+([-\d.]+)\s+um\^2\s+([-\d.]+)%\s+utilization', self.content)
        if design_area_match:
            metrics['design_area_um2'] = float(design_area_match.group(1))
            metrics['utilization_percent'] = float(design_area_match.group(2))
        
        return metrics
    
    def extract_max_paths(self) -> List[Dict[str, Any]]:
        """Extract all max (setup) path information: startpoint, endpoint, and slack."""
        paths = []
        
        # Look for all "report_checks -path_delay max" sections
        max_path_pattern = re.compile(
            r'report_checks -path_delay max(?:\s+[^\n]*)?\s*\n'
            r'[-]+\s*\n'
            r'Startpoint:\s*(.+?)\n'
            r'(?:.*?\n)*?'  # Skip lines until Endpoint
            r'Endpoint:\s*(.+?)\n'
            r'(?:.*?\n)*?'  # Skip until slack
            r'[-]+\s*\n'
            r'\s*([-\d.]+)\s+slack',
            re.MULTILINE | re.DOTALL
        )
        
        for match in max_path_pattern.finditer(self.content):
            startpoint = match.group(1).strip()
            endpoint = match.group(2).strip()
            slack = float(match.group(3))
            
            # Extract report check command to determine path category
            report_cmd_match = re.search(
                r'report_checks -path_delay max([^\n]*)',
                self.content[max(0, match.start() - 100):match.start()]
            )
            path_category = 'default'
            if report_cmd_match:
                cmd_suffix = report_cmd_match.group(1).strip()
                if cmd_suffix:
                    path_category = cmd_suffix
            
            # Extract data arrival and required times
            arrival_match = re.search(
                r'([-\d.]+)\s+data arrival time.*?'
                r'([-\d.]+)\s+data required time.*?'
                r'[-]+\s*\n\s*([-\d.]+)\s+slack',
                match.group(0),
                re.DOTALL
            )
            
            result = {
                'startpoint': startpoint,
                'endpoint': endpoint,
                'slack': slack,
                'path_type': 'max',
                'analysis_type': 'setup',
                'path_category': path_category
            }
            
            if arrival_match:
                result['data_arrival_time'] = float(arrival_match.group(1))
                result['data_required_time'] = float(arrival_match.group(2))
            
            # Check if violated or met
            if 'VIOLATED' in match.group(0):
                result['status'] = 'VIOLATED'
            elif 'MET' in match.group(0):
                result['status'] = 'MET'
            
            paths.append(result)
        
        return paths
    
    def extract_min_paths(self) -> List[Dict[str, Any]]:
        """Extract all min (hold) path information: startpoint, endpoint, and slack."""
        paths = []
        
        # Look for all "report_checks -path_delay min" sections
        min_path_pattern = re.compile(
            r'report_checks -path_delay min(?:\s+[^\n]*)?\s*\n'
            r'[-]+\s*\n'
            r'Startpoint:\s*(.+?)\n'
            r'(?:.*?\n)*?'  # Skip lines until Endpoint
            r'Endpoint:\s*(.+?)\n'
            r'(?:.*?\n)*?'  # Skip until slack
            r'[-]+\s*\n'
            r'\s*([-\d.]+)\s+slack',
            re.MULTILINE | re.DOTALL
        )
        
        for match in min_path_pattern.finditer(self.content):
            startpoint = match.group(1).strip()
            endpoint = match.group(2).strip()
            slack = float(match.group(3))
            
            # Extract report check command to determine path category
            report_cmd_match = re.search(
                r'report_checks -path_delay min([^\n]*)',
                self.content[max(0, match.start() - 100):match.start()]
            )
            path_category = 'default'
            if report_cmd_match:
                cmd_suffix = report_cmd_match.group(1).strip()
                if cmd_suffix:
                    path_category = cmd_suffix
            
            # Extract data arrival and required times
            arrival_match = re.search(
                r'([-\d.]+)\s+data arrival time.*?'
                r'([-\d.]+)\s+data required time.*?'
                r'[-]+\s*\n\s*([-\d.]+)\s+slack',
                match.group(0),
                re.DOTALL
            )
            
            result = {
                'startpoint': startpoint,
                'endpoint': endpoint,
                'slack': slack,
                'path_type': 'min',
                'analysis_type': 'hold',
                'path_category': path_category
            }
            
            if arrival_match:
                result['data_arrival_time'] = float(arrival_match.group(1))
                result['data_required_time'] = float(arrival_match.group(2))
            
            # Check if violated or met
            if 'VIOLATED' in match.group(0):
                result['status'] = 'VIOLATED'
            elif 'MET' in match.group(0):
                result['status'] = 'MET'
            
            paths.append(result)
        
        return paths
    
    def extract_clock_skew(self) -> Optional[Dict[str, Any]]:
        """Extract clock skew information."""
        skew_pattern = re.compile(
            r'report_clock_skew\s*\n'
            r'[-]+\s*\n'
            r'Clock\s+(\w+)\s*\n'
            r'\s*([-\d.]+)\s+source latency\s+(.+?)\s*\n'
            r'\s*([-\d.]+)\s+target latency\s+(.+?)\s*\n'
            r'\s*([-\d.]+)\s+CRPR\s*\n'
            r'[-]+\s*\n'
            r'\s*([-\d.]+)\s+setup skew',
            re.MULTILINE
        )
        
        match = skew_pattern.search(self.content)
        if match:
            return {
                'clock_name': match.group(1),
                'source_latency': float(match.group(2)),
                'source_pin': match.group(3).strip(),
                'target_latency': float(match.group(4)),
                'target_pin': match.group(5).strip(),
                'crpr': float(match.group(6)),
                'setup_skew': float(match.group(7))
            }
        
        return None
    
    def extract_power(self) -> Optional[Dict[str, Any]]:
        """Extract power consumption information."""
        power_pattern = re.compile(
            r'report_power\s*\n'
            r'[-]+\s*\n'
            r'Group\s+Internal\s+Switching\s+Leakage\s+Total\s*\n'
            r'\s+Power\s+Power\s+Power\s+Power\s+\(Watts\)\s*\n'
            r'[-]+\s*\n'
            r'Sequential\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.]+)%\s*\n'
            r'Combinational\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.]+)%\s*\n'
            r'Clock\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.]+)%\s*\n'
            r'Macro\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.]+)%\s*\n'
            r'Pad\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.]+)%\s*\n'
            r'[-]+\s*\n'
            r'Total\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+100.0%',
            re.MULTILINE
        )
        
        match = power_pattern.search(self.content)
        if match:
            return {
                'sequential': {
                    'internal': float(match.group(1)),
                    'switching': float(match.group(2)),
                    'leakage': float(match.group(3)),
                    'total': float(match.group(4)),
                    'percent': float(match.group(5))
                },
                'combinational': {
                    'internal': float(match.group(6)),
                    'switching': float(match.group(7)),
                    'leakage': float(match.group(8)),
                    'total': float(match.group(9)),
                    'percent': float(match.group(10))
                },
                'clock': {
                    'internal': float(match.group(11)),
                    'switching': float(match.group(12)),
                    'leakage': float(match.group(13)),
                    'total': float(match.group(14)),
                    'percent': float(match.group(15))
                },
                'macro': {
                    'internal': float(match.group(16)),
                    'switching': float(match.group(17)),
                    'leakage': float(match.group(18)),
                    'total': float(match.group(19)),
                    'percent': float(match.group(20))
                },
                'pad': {
                    'internal': float(match.group(21)),
                    'switching': float(match.group(22)),
                    'leakage': float(match.group(23)),
                    'total': float(match.group(24)),
                    'percent': float(match.group(25))
                },
                'total': {
                    'internal': float(match.group(26)),
                    'switching': float(match.group(27)),
                    'leakage': float(match.group(28)),
                    'total': float(match.group(29))
                }
            }
        
        return None


def extract_from_file(rpt_file: str, output_format: str = 'json') -> str:
    """
    Extract metrics from a single .rpt file.
    
    Args:
        rpt_file: Path to the .rpt file
        output_format: Output format ('json' or 'text')
    
    Returns:
        Formatted output string
    """
    extractor = ReportMetricExtractor(rpt_file)
    data = extractor.extract_all()
    
    # Add clock skew if available
    clock_skew = extractor.extract_clock_skew()
    if clock_skew:
        data['clock_skew'] = clock_skew
    
    # Add power metrics if available
    power = extractor.extract_power()
    if power:
        data['power'] = power
    
    if output_format == 'json':
        return json.dumps(data, indent=2)
    else:
        # Text format
        lines = [f"Report File: {data['file']}", "=" * 80, ""]
        
        # Metrics
        lines.append("METRICS:")
        lines.append("-" * 80)
        for key, value in data['metrics'].items():
            lines.append(f"  {key:30s}: {value}")
        lines.append("")
        
        # Max Path (Setup)
        if 'max_paths' in data and data['max_paths']:
            lines.append("MAX PATHS (SETUP):")
            lines.append("-" * 80)
            for i, max_path in enumerate(data['max_paths'], 1):
                lines.append(f"  Path {i} ({max_path.get('path_category', 'default')}):")
                for key, value in max_path.items():
                    if key != 'path_category':
                        lines.append(f"    {key:28s}: {value}")
                lines.append("")
        
        # Min Path (Hold)
        if 'min_paths' in data and data['min_paths']:
            lines.append("MIN PATHS (HOLD):")
            lines.append("-" * 80)
            for i, min_path in enumerate(data['min_paths'], 1):
                lines.append(f"  Path {i} ({min_path.get('path_category', 'default')}):")
                for key, value in min_path.items():
                    if key != 'path_category':
                        lines.append(f"    {key:28s}: {value}")
                lines.append("")
        
        # Clock Skew
        if 'clock_skew' in data:
            lines.append("CLOCK SKEW:")
            lines.append("-" * 80)
            for key, value in data['clock_skew'].items():
                lines.append(f"  {key:30s}: {value}")
            lines.append("")
        
        # Power
        if 'power' in data:
            lines.append("POWER:")
            lines.append("-" * 80)
            lines.append(f"  {'Total Power (W)':30s}: {data['power']['total']['total']:.6e}")
            lines.append(f"  {'  Internal Power (W)':30s}: {data['power']['total']['internal']:.6e}")
            lines.append(f"  {'  Switching Power (W)':30s}: {data['power']['total']['switching']:.6e}")
            lines.append(f"  {'  Leakage Power (W)':30s}: {data['power']['total']['leakage']:.6e}")
            lines.append(f"  {'Sequential Power (%)':30s}: {data['power']['sequential']['percent']:.1f}%")
            lines.append(f"  {'Combinational Power (%)':30s}: {data['power']['combinational']['percent']:.1f}%")
            lines.append(f"  {'Clock Power (%)':30s}: {data['power']['clock']['percent']:.1f}%")
            lines.append("")
        
        return "\n".join(lines)


def extract_from_directory(directory: str, pattern: str = "*.rpt", 
                          output_format: str = 'json') -> Dict[str, Any]:
    """
    Extract metrics from all .rpt files in a directory.
    
    Args:
        directory: Directory path
        pattern: File pattern to match (default: "*.rpt")
        output_format: Output format ('json' or 'text')
    
    Returns:
        Dictionary with all extracted data
    """
    dir_path = Path(directory)
    rpt_files = sorted(dir_path.glob(pattern))
    
    if not rpt_files:
        raise FileNotFoundError(f"No .rpt files found in {directory}")
    
    results = {}
    for rpt_file in rpt_files:
        try:
            extractor = ReportMetricExtractor(str(rpt_file))
            data = extractor.extract_all()
            clock_skew = extractor.extract_clock_skew()
            if clock_skew:
                data['clock_skew'] = clock_skew
            
            power = extractor.extract_power()
            if power:
                data['power'] = power
            
            results[rpt_file.name] = data
        except Exception as e:
            print(f"Warning: Error processing {rpt_file}: {e}")
            results[rpt_file.name] = {'error': str(e)}
    
    return results


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Extract timing metrics and path information from .rpt files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract from a single file
  python rpt_metric_extract.py report.rpt
  
  # Extract from a single file and save to JSON
  python rpt_metric_extract.py report.rpt -o output.json
  
  # Extract from all .rpt files in a directory
  python rpt_metric_extract.py -d ./reports/
  
  # Extract with text output format
  python rpt_metric_extract.py report.rpt -f text
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('rpt_file', nargs='?', help='Path to a single .rpt file')
    group.add_argument('-d', '--directory', help='Directory containing .rpt files')
    
    parser.add_argument(
        '-o', '--output', 
        help='Output file (default: print to stdout)'
    )
    parser.add_argument(
        '-f', '--format', 
        choices=['json', 'text'], 
        default='json',
        help='Output format (default: json)'
    )
    parser.add_argument(
        '-p', '--pattern',
        default='*.rpt',
        help='File pattern for directory mode (default: *.rpt)'
    )
    
    args = parser.parse_args()
    
    # Extract data
    if args.rpt_file:
        output = extract_from_file(args.rpt_file, args.format)
    else:
        data = extract_from_directory(args.directory, args.pattern, args.format)
        output = json.dumps(data, indent=2) if args.format == 'json' else str(data)
    
    # Write output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"Results written to {args.output}")
    else:
        print(output)


if __name__ == '__main__':
    main()
