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
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class SynthesisStage:
    """Represents a synthesis stage with key information."""
    stage_name: str
    raw_filtered_log: List[str] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    script_path: Optional[str] = None
    log_file: Optional[str] = None
    clock_period: Optional[float] = None
    
    # Yosys specific
    yosys_version: Optional[str] = None
    
    # ABC specific
    abc_script: Optional[str] = None
    liberty_file: Optional[str] = None
    
    # Warnings and Errors
    warnings_count: int = 0
    errors_count: int = 0
    
    # Performance metrics
    cpu_time_user: Optional[float] = None
    cpu_time_sys: Optional[float] = None
    peak_memory_kb: Optional[int] = None
    elapsed_time: Optional[str] = None
    time_breakdown: Optional[str] = None


class SynthesisLogParser:
    """Parser for Yosys and ABC synthesis logs with line-by-line timing preserved."""
    
    def __init__(self):
        self.timestamp_pattern = re.compile(r'\[([\d\-T:\.Z]+)\]')
        self.yosys_version_pattern = re.compile(r'Yosys ([\d\.+\w\s\(\)]+)')
        self.cpu_pattern = re.compile(r'CPU: user ([\d\.]+)s system ([\d\.]+)s')
        self.memory_pattern = re.compile(r'MEM: ([\d\.]+) ([KMG]B) peak')
        self.elapsed_pattern = re.compile(r'Elapsed time: ([\d:\.]+\[h:\]min:sec)')
        self.clock_period_pattern = re.compile(r'Setting clock period to ([\d\.]+)')
        self.peak_memory_alt_pattern = re.compile(r'Peak memory: (\d+)KB')
        self.cpu_time_alt_pattern = re.compile(r'CPU time: user ([\d\.]+) sys ([\d\.]+)')
        
        # Define patterns to filter out (but keep top-level)
        self.verbose_subpass_patterns = [
            # Skip deep sub-passes like 3.7.1, 3.7.2, but keep 3.7, 3.8, etc.
            re.compile(r'^\[.*?\]\s+\d+\.\d+\.\d+\.'),  # e.g., 3.7.1.
            re.compile(r'^\[.*?\]\s+\d+\.\d+\.\d+\.\d+\.'),  # e.g., 3.7.1.1.
        ]
        
        # Keep important patterns even if they match filters
        self.important_patterns = [
            'Executing ABC pass',
            'abc -script',
            'Executing SYNTH pass',
            'Yosys',
            'End of script',
            'Time spent',
            'Elapsed time',
            'CPU time',
            'Peak memory',
            'Error',  # Keep errors
            '/synth.sh',
            'Setting clock period',
            'Using ABC',
            'read_liberty',
            'link_design',
            'Printing statistics',
            'exec cp',
            'mkdir',
            # Important metrics
            'WNS',
            'TNS',
            'HPWL',
            'area',
            'power',
            'slack',
            'violation',
            'Design area',
            'Chip area',
            'Number of cells',
            'Number of nets',
            'tns ',
            'wns ',
        ]
        
        # Warnings to filter out (common, not critical)
        self.filtered_warnings = [
            'Ignoring module',  # Yosys module warnings during early passes
            'because it contains processes',  # Expected during synthesis
            'no estimated parasitics',  # Expected before routing
            'is assigned in a continuous assignment',  # Repetitive reg assignment warnings
        ]
        
        # Info messages to filter out (verbose, not critical)
        self.filtered_info = [
            'LEF file:',  # LEF loading info
            'library cells',  # Library loading info
            'layers',
            'vias',
        ]
        
        # Track repetitive warnings for deduplication
        self.warning_patterns_seen = {}
        self.max_repetitive_warnings = 3  # Keep first N occurrences of repetitive warnings
    
    def parse_timestamp(self, line: str) -> Optional[datetime]:
        """Extract timestamp from log line."""
        match = self.timestamp_pattern.search(line)
        if match:
            try:
                return datetime.fromisoformat(match.group(1).replace('Z', '+00:00'))
            except:
                return None
        return None
    
    def extract_warning_pattern(self, line: str) -> Optional[str]:
        """Extract a generic pattern from a warning line for deduplication.
        
        Replaces variable names, line numbers, and specific identifiers with placeholders
        to identify similar repetitive warnings.
        """
        if 'Warning' not in line and 'WARNING' not in line:
            return None
        
        # Remove timestamp
        pattern = re.sub(r'\[[\d\-T:\.Z]+\]', '', line)
        
        # Replace quoted identifiers with placeholder
        pattern = re.sub(r"'[^']*'", "'<IDENTIFIER>'", pattern)
        pattern = re.sub(r'"[^"]*"', '"<IDENTIFIER>"', pattern)
        pattern = re.sub(r'\\[a-zA-Z_]\w*', '\\<IDENTIFIER>', pattern)
        
        # Replace line numbers and file locations
        pattern = re.sub(r':\d+\.\d+-\d+\.\d+', ':<LOC>', pattern)
        pattern = re.sub(r':\d+', ':<LINE>', pattern)
        pattern = re.sub(r'\d+\.\d+', '<NUM>', pattern)
        
        # Replace file paths
        pattern = re.sub(r'/[/\w\.\-]+\.(v|sv|vh|svh)', '<FILE>', pattern)
        
        return pattern.strip()
    
    def should_keep_line(self, line: str, prev_line: str = "") -> bool:
        """Determine if a line should be kept based on filtering rules."""
        # Always keep important lines (including metrics)
        for pattern in self.important_patterns:
            if pattern in line:
                return True
        
        # Filter out specific warnings and handle repetitive warnings
        if 'Warning' in line or 'WARNING' in line:
            # Check if it's a filtered warning type
            for filtered_warning in self.filtered_warnings:
                if filtered_warning in line:
                    return False
            
            # Check for repetitive warnings
            warning_pattern = self.extract_warning_pattern(line)
            if warning_pattern:
                if warning_pattern in self.warning_patterns_seen:
                    # Increment count
                    self.warning_patterns_seen[warning_pattern] += 1
                    # Only keep first N occurrences of repetitive warnings
                    if self.warning_patterns_seen[warning_pattern] > self.max_repetitive_warnings:
                        return False
                else:
                    # First occurrence of this warning pattern
                    self.warning_patterns_seen[warning_pattern] = 1
            
            # Keep other warnings
            return True
        
        # Filter out specific info messages
        if '[INFO' in line:
            for filtered_info in self.filtered_info:
                if filtered_info in line:
                    return False
            # Keep other info messages (they might contain important metrics)
            return True
        
        # Check if it's a verbose sub-pass
        for pattern in self.verbose_subpass_patterns:
            if pattern.match(line):
                # Only filter if it's a repetitive optimization pass
                if any(x in line for x in ['OPT_EXPR', 'OPT_MERGE', 'OPT_MUXTREE', 
                                            'OPT_REDUCE', 'OPT_DFF', 'OPT_CLEAN',
                                            'Rerunning OPT', 'Finished fast OPT']):
                    return False
                return True  # Keep other sub-passes
        
        # Keep top-level passes (single digit followed by dot)
        if re.search(r'^\[.*?\]\s+\d+\.\s+Executing', line):
            return True
        
        # Keep second-level passes for important operations
        if re.search(r'^\[.*?\]\s+\d+\.\d+\.\s+Executing', line):
            if any(x in line for x in ['SYNTH', 'ABC', 'PROC', 'FSM', 'MEMORY', 
                                        'TECHMAP', 'DFFLIBMAP', 'HIERARCHY']):
                return True
            # Filter repetitive optimization sub-passes
            if 'OPT' in line:
                return False
            return True
        
        # Keep command invocations and file operations
        if any(x in line for x in ['.sh ', '.tcl', 'Executing', 'frontend:', 
                                     'backend', 'Analyzing design']):
            return True
        
        # Keep summary and result lines
        if any(x in line for x in ['Logfile hash', 'Warnings:', 'Errors:']):
            return True
        
        return False
    
    def filter_log_lines(self, lines: List[str]) -> List[str]:
        """Filter log lines to remove unnecessary verbose content while preserving structure."""
        # Reset warning tracking for this log file
        self.warning_patterns_seen = {}
        
        filtered = []
        prev_line = ""
        
        for i, line in enumerate(lines):
            # Stop at next stage separator (e.g., OpenROAD, floorplan, placement, etc.)
            if 'OpenROAD' in line and i > 10:
                # OpenROAD indicates end of Yosys synthesis stage
                break
            
            if '==========================================================================' in line and i > 10:
                # Check if this is a new stage (not in the beginning)
                # Look for stage indicators after the separator
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    # Stop if next line indicates a new stage (not synthesis)
                    if any(stage in next_line.lower() for stage in ['floorplan', 'placement', 'cts', 'routing', 'finishing']):
                        break
            
            if self.should_keep_line(line, prev_line):
                filtered.append(line)
                prev_line = line
        
        return filtered
    
    def extract_summary_metrics(self, lines: List[str]) -> Dict[str, any]:
        """Extract summary metrics from log."""
        metrics = {
            'yosys_version': None,
            'clock_period': None,
            'abc_script': None,
            'liberty': None,
            'cpu_user': None,
            'cpu_sys': None,
            'peak_memory_kb': None,
            'elapsed_time': None,
            'time_breakdown': None,
            'warnings_count': 0,
            'errors_count': 0,
            'script_path': None,
            'log_file': None
        }
        
        for line in lines:
            # Yosys version
            if not metrics['yosys_version']:
                version_match = self.yosys_version_pattern.search(line)
                if version_match:
                    metrics['yosys_version'] = version_match.group(1)
            
            # Clock period
            if not metrics['clock_period']:
                clock_match = self.clock_period_pattern.search(line)
                if clock_match:
                    metrics['clock_period'] = float(clock_match.group(1))
            
            # ABC info
            if 'abc -script' in line.lower():
                script_match = re.search(r'-script\s+(\S+)', line)
                if script_match:
                    metrics['abc_script'] = script_match.group(1)
                
                liberty_match = re.search(r'-liberty\s+(\S+)', line)
                if liberty_match:
                    metrics['liberty'] = liberty_match.group(1)
            
            # Performance
            cpu_match = self.cpu_pattern.search(line)
            if cpu_match:
                metrics['cpu_user'] = float(cpu_match.group(1))
                metrics['cpu_sys'] = float(cpu_match.group(2))
            
            cpu_alt_match = self.cpu_time_alt_pattern.search(line)
            if cpu_alt_match:
                metrics['cpu_user'] = float(cpu_alt_match.group(1))
                metrics['cpu_sys'] = float(cpu_alt_match.group(2))
            
            mem_match = self.memory_pattern.search(line)
            if mem_match:
                value = float(mem_match.group(1))
                unit = mem_match.group(2)
                if unit == 'MB':
                    metrics['peak_memory_kb'] = int(value * 1024)
                elif unit == 'GB':
                    metrics['peak_memory_kb'] = int(value * 1024 * 1024)
                else:
                    metrics['peak_memory_kb'] = int(value)
            
            mem_alt_match = self.peak_memory_alt_pattern.search(line)
            if mem_alt_match:
                metrics['peak_memory_kb'] = int(mem_alt_match.group(1))
            
            elapsed_match = self.elapsed_pattern.search(line)
            if elapsed_match:
                metrics['elapsed_time'] = elapsed_match.group(1)
            
            if 'Time spent:' in line:
                breakdown = line.split('Time spent:', 1)[1].strip()
                metrics['time_breakdown'] = breakdown
            
            # Script paths
            if 'synth.sh' in line or 'synth_' in line:
                parts = line.split('] ', 1)
                if len(parts) > 1:
                    cmd_parts = parts[1].split()
                    if len(cmd_parts) >= 2 and not metrics['script_path']:
                        metrics['script_path'] = cmd_parts[1]
                    if len(cmd_parts) >= 3 and not metrics['log_file']:
                        metrics['log_file'] = cmd_parts[2]
            
            # Warnings and errors
            if 'Warning:' in line or 'WARNING' in line:
                metrics['warnings_count'] += 1
            if 'Error:' in line or 'ERROR' in line:
                metrics['errors_count'] += 1
        
        return metrics
    
    def parse_synthesis_stage(self, log_content: str, stage_name: str = "Synthesis") -> SynthesisStage:
        """Parse a complete synthesis stage from log content."""
        lines = log_content.strip().split('\n')
        
        # Filter lines
        filtered_lines = self.filter_log_lines(lines)
        
        # Extract metrics
        metrics = self.extract_summary_metrics(lines)
        
        # Extract timestamps
        start_time = None
        end_time = None
        for line in lines:
            timestamp = self.parse_timestamp(line)
            if timestamp:
                if not start_time:
                    start_time = timestamp
                end_time = timestamp
        
        # Create stage object
        stage = SynthesisStage(
            stage_name=stage_name,
            raw_filtered_log=filtered_lines,
            start_time=start_time,
            end_time=end_time,
            script_path=metrics['script_path'],
            log_file=metrics['log_file'],
            clock_period=metrics['clock_period'],
            yosys_version=metrics['yosys_version'],
            abc_script=metrics['abc_script'],
            liberty_file=metrics['liberty'],
            warnings_count=metrics['warnings_count'],
            errors_count=metrics['errors_count'],
            cpu_time_user=metrics['cpu_user'],
            cpu_time_sys=metrics['cpu_sys'],
            peak_memory_kb=metrics['peak_memory_kb'],
            elapsed_time=metrics['elapsed_time'],
            time_breakdown=metrics['time_breakdown']
        )
        
        return stage
    
    def format_stage_output(self, stage: SynthesisStage, include_summary: bool = True) -> str:
        """Format synthesis stage with preserved timing structure."""
        lines = []
        
        if include_summary:
            lines.append("=" * 80)
            lines.append(f"SYNTHESIS STAGE: {stage.stage_name}")
            lines.append("=" * 80)
            
            if stage.start_time and stage.end_time:
                duration = (stage.end_time - stage.start_time).total_seconds()
                lines.append(f"Duration: {duration:.2f}s | Start: {stage.start_time} | End: {stage.end_time}")
            
            if stage.yosys_version:
                lines.append(f"Yosys: {stage.yosys_version}")
            
            if stage.clock_period:
                lines.append(f"Clock Period: {stage.clock_period}ns")
            
            if stage.script_path:
                lines.append(f"Script: {stage.script_path}")
            
            if stage.abc_script:
                lines.append(f"ABC Script: {stage.abc_script}")
            
            # Performance summary
            perf_parts = []
            if stage.elapsed_time:
                perf_parts.append(f"Time: {stage.elapsed_time}")
            if stage.cpu_time_user is not None:
                perf_parts.append(f"CPU: {stage.cpu_time_user + stage.cpu_time_sys:.2f}s")
            if stage.peak_memory_kb:
                perf_parts.append(f"Memory: {stage.peak_memory_kb / 1024:.1f}MB")
            if perf_parts:
                lines.append(" | ".join(perf_parts))
            
            if stage.warnings_count > 0:
                lines.append(f"⚠️  Warnings: {stage.warnings_count}")
            if stage.errors_count > 0:
                lines.append(f"❌ Errors: {stage.errors_count}")
            
            lines.append("=" * 80)
            lines.append("")
        
        # Add filtered log with preserved timing
        lines.extend(stage.raw_filtered_log)
        
        return '\n'.join(lines)


def parse_synthesis_log_file(log_file_path: str) -> List[SynthesisStage]:
    """Parse synthesis log file and extract all stages with preserved timing."""
    with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    parser = SynthesisLogParser()
    
    # Split by synthesis script invocations
    stages = []
    stage_pattern = r'(\[[\d\-T:\.Z]+\]\s+/[^\n]+/synth\.sh[^\n]+)'
    matches = list(re.finditer(stage_pattern, content))
    
    if matches:
        for i, match in enumerate(matches):
            start_pos = match.start()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            stage_content = content[start_pos:end_pos]
            
            # Extract stage name from script
            script_line = match.group(1)
            stage_name = f"Synthesis_Stage_{i + 1}"
            if 'canonicalize' in script_line:
                stage_name = f"Synthesis_{i + 1}_Canonicalize"
            elif 'synth.tcl' in script_line:
                stage_name = f"Synthesis_{i + 1}_Main"
            
            stage = parser.parse_synthesis_stage(stage_content, stage_name)
            stages.append(stage)
    else:
        # Single stage
        stage = parser.parse_synthesis_stage(content, "Synthesis")
        stages.append(stage)
    
    return stages


def write_filtered_log(stages: List[SynthesisStage], output_path: str, include_summary: bool = True):
    """Write filtered log to file with preserved timing structure."""
    parser = SynthesisLogParser()
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, stage in enumerate(stages):
            if i > 0:
                f.write("\n\n")
            f.write(parser.format_stage_output(stage, include_summary))


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python log_parser.py <log_file_path> [output_file]")
        print("  If output_file is specified, filtered log will be written there")
        sys.exit(1)
    
    log_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        print(f"Parsing {log_file}...")
        stages = parse_synthesis_log_file(log_file)
        
        print(f"Found {len(stages)} synthesis stage(s)\n")
        
        parser = SynthesisLogParser()
        
        if output_file:
            # Write to file
            write_filtered_log(stages, output_file, include_summary=True)
            print(f"Filtered log written to: {output_file}")
            
            # Also print summary to console
            for stage in stages:
                print(f"\n{stage.stage_name}:")
                print(f"  Lines: {len(stage.raw_filtered_log)}")
                print(f"  Duration: {stage.elapsed_time}")
                print(f"  Warnings: {stage.warnings_count}")
                if stage.peak_memory_kb:
                    print(f"  Memory: {stage.peak_memory_kb / 1024:.1f} MB")
        else:
            # Print to console
            for stage in stages:
                print(parser.format_stage_output(stage, include_summary=True))
                print("\n")
            
    except FileNotFoundError:
        print(f"Error: File '{log_file}' not found")
    except Exception as e:
        print(f"Error parsing log: {e}")
        import traceback
        traceback.print_exc()
