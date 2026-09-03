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
Complete Log Extraction Pipeline
Extracts all stages (synthesis, floorplan, placement, CTS, routing, final) from OpenROAD log files.
Handles multiple occurrences by splitting log by stage markers (1_*, 2_*, 3_*, etc.), 
then passing each stage chunk to the appropriate extraction function.
"""

import os
import re
import glob
from typing import Dict, Optional, List
from datetime import datetime
import json

# Import all stage extraction functions
from syn_stage import SynthesisLogParser
from flr_stg import extract_floorplan_section
from place_stg import extract_placement_section
from cts_stg import extract_cts_section
from route_stg import extract_route_section
from final_stg import extract_final_section

# Import metric parsers (produce JSON-friendly dicts)
from synthesis_metrics_parser import extract_synthesis_metrics
from floorplan_metrics_parser import extract_floorplan_metrics
from placement_metrics_parser import extract_placement_metrics
from cts_metrics_parser import extract_cts_metrics
from route_metrics_parser import extract_route_metrics
from final_metrics_parser import extract_final_metrics

# Import RPT metric extractor
from rpt_metric_extract import ReportMetricExtractor


def extract_design_info(log_content: str) -> Dict:
    """
    Extract top-level design metadata from the raw log.

    Typical OpenROAD-flow test harness logs include lines like:
      + DESIGN_NAME=uart-blocks
      + PLATFORM=gf180
      + CONFIG_MK=config.mk

    Returns:
        {
          'runs': [
             {'line': int, 'timestamp': str|None, 'design_name': str|None, 'technology_node': str|None, 'config_mk': str|None}
          ],
          'summary': {
             'design_names': [...], 'technology_nodes': [...], 'config_mks': [...]
          }
        }
    """
    lines = (log_content or "").splitlines()
    ts_re = re.compile(r'^\[([^\]]+)\]')
    design_re = re.compile(r'\bDESIGN_NAME=([^\s]+)')
    platform_re = re.compile(r'\bPLATFORM=([^\s]+)')
    config_re = re.compile(r'\bCONFIG_MK=([^\s]+)')

    runs: List[Dict] = []
    current: Dict = {'line': None, 'timestamp': None, 'design_name': None, 'technology_node': None, 'config_mk': None}
    current_start_line = None

    def _flush_current():
        nonlocal current, current_start_line
        if current_start_line is not None and any(current.get(k) for k in ('design_name', 'technology_node', 'config_mk')):
            runs.append(dict(current))
        current = {'line': None, 'timestamp': None, 'design_name': None, 'technology_node': None, 'config_mk': None}
        current_start_line = None

    for idx, line in enumerate(lines):
        # These variables are typically printed near each other (within a few lines).
        m_design = design_re.search(line)
        m_platform = platform_re.search(line)
        m_config = config_re.search(line)
        if not (m_design or m_platform or m_config):
            continue

        if current_start_line is None:
            current_start_line = idx
            current['line'] = idx + 1  # 1-based for humans
            m_ts = ts_re.search(line)
            current['timestamp'] = m_ts.group(1) if m_ts else None

        if m_design:
            current['design_name'] = m_design.group(1)
        if m_platform:
            # Logs export the variable as PLATFORM=..., but we store it as technology_node.
            current['technology_node'] = m_platform.group(1)
        if m_config:
            current['config_mk'] = m_config.group(1)

        # If we already have all fields, flush immediately (handles repeated runs).
        if current.get('design_name') and current.get('technology_node') and current.get('config_mk'):
            _flush_current()

    _flush_current()

    def _uniq(xs: List[Optional[str]]) -> List[str]:
        out = []
        for x in xs:
            if x and x not in out:
                out.append(x)
        return out

    return {
        'runs': runs,
        'summary': {
            'design_names': _uniq([r.get('design_name') for r in runs]),
            'technology_nodes': _uniq([r.get('technology_node') for r in runs]),
            'config_mks': _uniq([r.get('config_mk') for r in runs]),
        }
    }


def extract_klayout_design_names(log_content: str) -> List[str]:
    """
    Extract KLayout `-rd design_name=<name>` values from the raw log in appearance order.

    Example log line:
      (env time ... klayout.sh ... -rd design_name=uart_rx \\

    Returns:
        List of design names in order of appearance. (Not deduplicated across runs;
        if the same name appears multiple times in the log, it will appear multiple times.)
    """
    names: List[str] = []
    if not log_content:
        return names

    klayout_re = re.compile(r'-rd\s+design_name=([A-Za-z0-9_.\-]+)')
    for line in log_content.splitlines():
        m = klayout_re.search(line)
        if m:
            names.append(m.group(1))
    return names


def strip_summary_fields(obj):
    """
    Recursively remove any dict key named exactly 'summary' from a nested structure.

    Used to keep exported JSON metrics concise while preserving full metrics in-memory.
    """
    if isinstance(obj, dict):
        return {k: strip_summary_fields(v) for k, v in obj.items() if k != 'summary'}
    if isinstance(obj, list):
        return [strip_summary_fields(v) for v in obj]
    return obj


def annotate_stages_with_rpt_metrics(stages: List[Dict], rpt_dir: Optional[str] = None) -> None:
    """
    Annotate each stage in the stages list with metrics extracted from corresponding .rpt files.
    
    This function finds .rpt files matching each stage and adds the extracted metrics to the 
    stage dictionary under the 'rpt_metrics' key. Only processes stages when there is exactly
    1 occurrence in the flow.
    
    Args:
        stages: List of stage dictionaries from extract_all_stages
        rpt_dir: Directory containing .rpt files. If None, tries to infer from first stage's metrics_file
    
    Stage name mapping (from qa_generation.py lines 33-58):
        synthesis: 1_synth.rpt (if exists)
        2_1_floorplan: 2_floorplan_final.rpt
        3_3_place_gp: 3_global_place.rpt
        3_4_place_resized: 3_resizer.rpt  
        3_5_place_dp: 3_detailed_place.rpt
        4_1_cts: 4_cts_final.rpt
        5_1_grt: 5_global_route.rpt
        5_2_route: 5_route_drc.rpt (if exists)
        6_1_fill: 6_finish.rpt
    """
    if not stages:
        return
    
    # Count occurrences per stage type
    stage_counts = {}
    for stage in stages:
        stage_type = stage.get('stage')
        stage_counts[stage_type] = stage_counts.get(stage_type, 0) + 1
    
    # Try to infer rpt_dir from first stage's metrics_file path
    if rpt_dir is None:
        for stage in stages:
            metrics_file = stage.get('metrics_file')
            if metrics_file and os.path.exists(metrics_file):
                # Go up from metrics file to find reports directory
                # Typical structure: .../extracted_logs/..._metrics.json
                # We need to find: .../reports/.../base/*.rpt
                base_dir = os.path.dirname(os.path.dirname(metrics_file))
                potential_rpt_dir = os.path.join(base_dir, 'reports')
                if os.path.exists(potential_rpt_dir):
                    # Find the actual reports/platform/design/base directory
                    for root, dirs, files in os.walk(potential_rpt_dir):
                        if 'base' in dirs or any(f.endswith('.rpt') for f in files):
                            rpt_dir = root if any(f.endswith('.rpt') for f in files) else os.path.join(root, 'base')
                            break
                    if rpt_dir:
                        break
    
    if not rpt_dir or not os.path.exists(rpt_dir):
        print(f"  [INFO] RPT directory not found, skipping .rpt metrics annotation")
        return
    
    # Mapping of stage type to .rpt filename pattern
    stage_to_rpt = {
        'synthesis': '1_synth*.rpt',
        'floorplan': '2_floorplan_final.rpt',
        'placement': {
            '3_1_place_gp_skip_io': None,  # No corresponding .rpt
            '3_2_place_iop': None,  # No corresponding .rpt
            '3_3_place_gp': '3_global_place.rpt',
            '3_4_place_resized': '3_resizer.rpt',
            '3_5_place_dp': '3_detailed_place.rpt',
        },
        'cts': '4_cts_final.rpt',
        'routing': {
            '5_1_grt': '5_global_route.rpt',
            '5_2_route': '5_route_drc.rpt',
            '5_3_fillcell': None,  # No corresponding .rpt
        },
        'final': '6_finish.rpt',
    }
    
    print()
    print("Annotating stages with .rpt metrics...")
    print("-" * 80)
    
    for stage in stages:
        stage_type = stage.get('stage')
        occurrence = stage.get('occurrence', 1)
        
        # Only process if there's exactly 1 occurrence of this stage type
        if stage_counts.get(stage_type, 0) != 1:
            continue
        
        # Get metrics dict
        metrics = stage.get('metrics', {})
        if not isinstance(metrics, dict):
            continue
        
        # Check if this stage has substages or uses flow_summary (for final stage)
        substages = metrics.get('substages', {})
        flow_summary = metrics.get('flow_summary', {})
        
        # Process substages if they exist
        if isinstance(substages, dict) and substages:
            # Process each substage that has 'report_metrics'
            for substage_name, substage_data in substages.items():
                if not isinstance(substage_data, dict):
                    continue
                
                # Only process substages that have 'report_metrics' field
                if 'report_metrics' not in substage_data:
                    continue
                
                # Determine .rpt file pattern for this specific substage
                rpt_pattern = None
                if stage_type == 'placement':
                    rpt_pattern = stage_to_rpt.get('placement', {}).get(substage_name)
                elif stage_type == 'routing':
                    rpt_pattern = stage_to_rpt.get('routing', {}).get(substage_name)
                elif stage_type == 'floorplan':
                    rpt_pattern = stage_to_rpt.get('floorplan')
                elif stage_type == 'cts':
                    rpt_pattern = stage_to_rpt.get('cts')
                elif stage_type == 'synthesis':
                    rpt_pattern = stage_to_rpt.get('synthesis')
                elif stage_type == 'final':
                    rpt_pattern = stage_to_rpt.get('final')
                
                if not rpt_pattern:
                    continue
                
                # Find matching .rpt file
                rpt_files = glob.glob(os.path.join(rpt_dir, rpt_pattern))
                
                if not rpt_files:
                    continue
                
                rpt_file = rpt_files[0]  # Take first match
                
                try:
                    # Extract metrics from .rpt file
                    extractor = ReportMetricExtractor(rpt_file)
                    rpt_data = extractor.extract_all()
                    
                    # Add optional metrics
                    clock_skew = extractor.extract_clock_skew()
                    if clock_skew:
                        rpt_data['clock_skew'] = clock_skew
                    
                    power = extractor.extract_power()
                    if power:
                        rpt_data['power'] = power
                    
                    # Add rpt_metrics to this substage
                    substage_data['rpt_metrics'] = rpt_data
                    
                    print(f"  ✓ {substage_name}: Added metrics from {os.path.basename(rpt_file)}")
                    
                except Exception as e:
                    print(f"  ✗ {substage_name}: Failed to extract from {os.path.basename(rpt_file)}: {e}")
                    continue
        
        # Handle stages without substages (like final stage with flow_summary)
        elif stage_type in ['final', 'synthesis']:
            # Determine .rpt file pattern
            rpt_pattern = stage_to_rpt.get(stage_type)
            
            if not rpt_pattern:
                continue
            
            # Find matching .rpt file
            rpt_files = glob.glob(os.path.join(rpt_dir, rpt_pattern))
            
            if not rpt_files:
                continue
            
            rpt_file = rpt_files[0]  # Take first match
            
            try:
                # Extract metrics from .rpt file
                extractor = ReportMetricExtractor(rpt_file)
                rpt_data = extractor.extract_all()
                
                # Add optional metrics
                clock_skew = extractor.extract_clock_skew()
                if clock_skew:
                    rpt_data['clock_skew'] = clock_skew
                
                power = extractor.extract_power()
                if power:
                    rpt_data['power'] = power
                
                # Add rpt_metrics directly to metrics (no substages to add to)
                metrics['rpt_metrics'] = rpt_data
                
                print(f"  ✓ {stage_type}: Added metrics from {os.path.basename(rpt_file)}")
                
            except Exception as e:
                print(f"  ✗ {stage_type}: Failed to extract from {os.path.basename(rpt_file)}: {e}")
                continue
    
    print("-" * 80)



class LogExtractionPipeline:
    """
    Complete pipeline to extract all stages from an OpenROAD log file.
    
    Workflow:
    1. Load log file once into memory
    2. Split by stage markers (1_*, 2_*, 3_*, etc.)
    3. Pass each stage chunk to appropriate extraction function
    4. Return results in chronological order
    """
    
    def __init__(self, log_file_path: str, output_dir: str = "extracted_logs"):
        """
        Initialize the extraction pipeline.
        
        Args:
            log_file_path: Path to the input log file
            output_dir: Directory to save extracted stage files (default: "extracted_logs")
        """
        self.log_file_path = log_file_path
        self.output_dir = output_dir
        self.log_content = None
        self.base_name = os.path.splitext(os.path.basename(log_file_path))[0]
        self.klayout_design_names: List[str] = []
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def load_log_file(self) -> bool:
        """
        Load the log file into memory.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(self.log_file_path, 'r', encoding='utf-8') as f:
                self.log_content = f.read()
            print(f"[OK] Loaded log file: {self.log_file_path}")
            print(f"  File size: {len(self.log_content)} bytes")
            return True
        except Exception as e:
            print(f"[ERROR] Error loading log file: {e}")
            return False
    
    def split_log_by_stages(self) -> List[Dict]:
        """
        Split the log file into stage chunks by detecting stage transitions from
        lines containing the `Running` keyword.

        A new chunk starts at lines like:
          `Running <something>.tcl, stage <stage_num>_...`

        The chunk ends right before the next *different* stage (by leading stage
        number) `Running ...tcl` line.

        Returns:
            List of dictionaries with stage information, in chronological order.
        """
        if not self.log_content:
            return []

        lines = self.log_content.splitlines()
        stages: List[Dict] = []
        
        # Stage mapping
        stage_map = {
            1: 'synthesis',
            2: 'floorplan',
            3: 'placement',
            4: 'cts',
            5: 'routing',
            6: 'final'
        }
        
        # Track occurrence count for each stage type
        stage_occurrence_count = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}

        current_stage_num = None
        stage_start_line = None

        # Stage 1 (synthesis) typically does not have "Running ...tcl" markers in logs.
        # Use "Using ABC speed script." to identify the synthesis stage beginning.
        abc_synth_re = re.compile(r'\bUsing ABC speed script\b')
        running_stage_re = re.compile(r'\bRunning\b.*?\.tcl,\s*stage\s+([1-6])_')

        # Identify design name and sub-design name (via KLayout)
        design_name_re = re.compile(r'\bDESIGN_NAME=([^\s]+)')
        # Example: ... klayout.sh ... -rd design_name=uart_rx \
        klayout_design_re = re.compile(r'-rd\s+design_name=([A-Za-z0-9_.\-]+)')

        # Use the first DESIGN_NAME seen in the log (typically stable for the whole run)
        design_name = None
        for l in lines:
            m_dn = design_name_re.search(l)
            if m_dn:
                design_name = m_dn.group(1)
                break

        # Use KLayout sub-design names in appearance order (deduped), mapped by occurrence index
        klayout_names_raw = extract_klayout_design_names(self.log_content or "")
        sub_design_names: List[str] = []
        _seen_sd = set()
        for n in klayout_names_raw:
            if n not in _seen_sd:
                _seen_sd.add(n)
                sub_design_names.append(n)

        for i, line in enumerate(lines):
            stage_num = None

            # Event 1: synthesis begin
            if abc_synth_re.search(line):
                stage_num = 1

            # Event 2: stage begin for 2..6 (and sometimes 1, but we treat ABC as 1)
            m = running_stage_re.search(line)
            if m:
                stage_num = int(m.group(1))

            if stage_num is None or stage_num not in stage_map:
                continue

            # First stage encountered (if we didn't see synthesis marker)
            if current_stage_num is None:
                stage_occurrence_count[stage_num] += 1
                current_stage_num = stage_num
                stage_start_line = i
                continue

            # Same stage number => keep accumulating lines into current chunk
            if stage_num == current_stage_num:
                continue

            # Stage number changed => close previous chunk, start a new one
            if stage_start_line is not None:
                occ = stage_occurrence_count[current_stage_num]
                sub_design_name = sub_design_names[occ - 1] if 0 <= (occ - 1) < len(sub_design_names) else None
                stages.append({
                    'design_name': design_name,
                    'sub_design_name': sub_design_name,
                    'stage_number': current_stage_num,
                    'stage_type': stage_map[current_stage_num],
                    'occurrence': occ,
                    'start_line': stage_start_line,
                    'end_line': i,
                    'content': '\n'.join(lines[stage_start_line:i])
                })

            stage_occurrence_count[stage_num] += 1
            current_stage_num = stage_num
            stage_start_line = i
        
        # Don't forget the last stage
        if current_stage_num is not None and stage_start_line is not None:
            occ = stage_occurrence_count[current_stage_num]
            sub_design_name = sub_design_names[occ - 1] if 0 <= (occ - 1) < len(sub_design_names) else None
            stages.append({
                'design_name': design_name,
                'sub_design_name': sub_design_name,
                'stage_number': current_stage_num,
                'stage_type': stage_map[current_stage_num],
                'occurrence': occ,
                'start_line': stage_start_line,
                'end_line': len(lines),
                'content': '\n'.join(lines[stage_start_line:])
            })
        return stages
    
    def extract_stage_by_type(self, stage_chunk: Dict) -> Optional[Dict]:
        """
        Extract a stage based on its type by calling the appropriate parser function.
        
        Args:
            stage_chunk: Dictionary with stage information including content
            
        Returns:
            Dictionary with extraction results or None if extraction fails
        """
        stage_type = stage_chunk['stage_type']
        occurrence = stage_chunk['occurrence']
        stage_content = stage_chunk['content']
        start_line = stage_chunk['start_line']
        end_line = stage_chunk['end_line']
        design_name = stage_chunk.get('design_name')
        sub_design_name = stage_chunk.get('sub_design_name')

        def _write_stage_metrics(stage: str, occ: int, raw_section: str, top_design: Optional[str], sub_design: Optional[str]) -> Optional[Dict]:
            """
            Run the stage metric parser and write a `<base>_<stage>[_<occ>]_metrics.json` file.
            Returns dict with {'metrics_file': str, 'metrics': dict} or None on failure.
            """
            parser_map = {
                'synthesis': extract_synthesis_metrics,
                'floorplan': extract_floorplan_metrics,
                'placement': extract_placement_metrics,
                'cts': extract_cts_metrics,
                'routing': extract_route_metrics,
                'final': extract_final_metrics,
            }

            parser_func = parser_map.get(stage)
            if not parser_func or not raw_section:
                return None

            try:
                # Pass design/sub-design info through the metrics parser interface
                metrics = parser_func(raw_section, design_name=top_design, sub_design_name=sub_design)
                
                # Post-process final stage metrics to map test metrics to design_metrics with renamed fields
                if stage == 'final' and metrics and metrics.get('test_metrics'):
                    test_metrics = metrics['test_metrics']
                    dm = metrics.setdefault('design_metrics', {})
                    
                    # Map finish__timing__setup__ws -> setup_wns, etc.
                    metric_mapping = {
                        'finish__timing__setup__ws': 'setup_wns',
                        'finish__timing__setup__tns': 'setup_tns',
                        'finish__timing__hold__ws': 'hold_wns',
                        'finish__timing__hold__tns': 'hold_tns',
                        'finish__design__instance__area': 'area',
                        'finish__timing__wns_percent_delay': 'wns_percent_delay'
                    }
                    
                    for test_key, new_key in metric_mapping.items():
                        if test_key in test_metrics:
                            dm[new_key] = test_metrics[test_key].get('actual_value')

                # For exported JSON, strip 'summary' fields (requested by user).
                metrics_for_file = strip_summary_fields(metrics)

                if occ > 1:
                    metrics_file = os.path.join(self.output_dir, f"{self.base_name}_{stage}_{occ}_metrics.json")
                else:
                    metrics_file = os.path.join(self.output_dir, f"{self.base_name}_{stage}_metrics.json")

                with open(metrics_file, 'w', encoding='utf-8') as f:
                    json.dump(metrics_for_file, f, indent=2)

                return {'metrics_file': metrics_file, 'metrics': metrics}
            except Exception as e:
                print(f"[WARN] Metrics parse failed for {stage} stage {occ}: {e}")
                return None
        
        try:
            # Generate output file name
            if occurrence > 1:
                output_file = os.path.join(
                    self.output_dir, 
                    f"{self.base_name}_{stage_type}_{occurrence}_extract.txt"
                )
            else:
                output_file = os.path.join(
                    self.output_dir, 
                    f"{self.base_name}_{stage_type}_extract.txt"
                )
            
            # Call appropriate parser based on stage type
            result = None
            
            if stage_type == 'synthesis':
                parser = SynthesisLogParser()
                stage_name = f"synthesis_{occurrence}"
                result_obj = parser.parse_synthesis_stage(stage_content, stage_name)
                
                if result_obj and result_obj.raw_filtered_log:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(result_obj.raw_filtered_log))
                    
                    result = {
                        'raw_section': '\n'.join(result_obj.raw_filtered_log),
                        'result': result_obj
                    }
                    print(f"[OK] Extracted {stage_type} stage {occurrence}: {len(result_obj.raw_filtered_log)} lines -> {output_file}")
                    
            elif stage_type == 'floorplan':
                result = extract_floorplan_section(stage_content, output_file)
                if result and result.get('raw_section'):
                    print(f"[OK] Extracted {stage_type} stage {occurrence}: {len(result['raw_section'].split(chr(10)))} lines -> {output_file}")
                    
            elif stage_type == 'placement':
                result = extract_placement_section(stage_content, output_file)
                if result and result.get('raw_section'):
                    print(f"[OK] Extracted {stage_type} stage {occurrence}: {len(result['raw_section'].split(chr(10)))} lines -> {output_file}")
                    
            elif stage_type == 'cts':
                result = extract_cts_section(stage_content, output_file)
                if result and result.get('raw_section'):
                    print(f"[OK] Extracted {stage_type} stage {occurrence}: {len(result['raw_section'].split(chr(10)))} lines -> {output_file}")
                    
            elif stage_type == 'routing':
                result = extract_route_section(stage_content, output_file)
                if result and result.get('raw_section'):
                    print(f"[OK] Extracted {stage_type} stage {occurrence}: {len(result['raw_section'].split(chr(10)))} lines -> {output_file}")
                    
            elif stage_type == 'final':
                result = extract_final_section(stage_content, output_file)
                if result and result.get('raw_section'):
                    print(f"[OK] Extracted {stage_type} stage {occurrence}: {len(result['raw_section'].split(chr(10)))} lines -> {output_file}")
            
            # If extraction was successful, return standardized result
            if result:
                line_count = len(result.get('raw_section', '').split('\n')) if result.get('raw_section') else 0
                raw_section = result.get('raw_section') or ''
                metrics_result = _write_stage_metrics(stage_type, occurrence, raw_section, design_name, sub_design_name)
                return {
                    'stage': stage_type,
                    'occurrence': occurrence,
                    'stage_name': f"{stage_type}_{occurrence}",
                    'output_file': output_file,
                    'metrics_file': metrics_result.get('metrics_file') if metrics_result else None,
                    'metrics': metrics_result.get('metrics') if metrics_result else None,
                    'line_count': line_count,
                    'design_name': design_name,
                    'sub_design_name': sub_design_name,
                    'start_line': start_line,
                    'end_line': end_line,
                    'result': result
                }
            else:
                print(f"  [WARN] No content extracted for {stage_type} stage {occurrence}")
                return None
                
        except Exception as e:
            print(f"[ERROR] Error extracting {stage_type} stage {occurrence}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _map_final_metrics_to_substages(self, all_stage_results: List[Dict]):
        """
        Map final stage test metrics to the appropriate substages:
        - cts__* -> CTS stage substages (design_metrics_final)
        - placeopt__* -> Placement stage (design_metrics_final)
        - globalroute__* -> Routing 5_1_grt substage (design_metrics_final)
        - detailedroute__* -> Routing 5_2_route substage (design_metrics_final)
        """
        # Find final stage result
        final_result = None
        for result in all_stage_results:
            if result.get('stage') == 'final':
                final_result = result
                break
        
        if not final_result or not final_result.get('metrics'):
            return
        
        final_metrics = final_result['metrics']
        test_metrics = final_metrics.get('test_metrics', {})
        
        if not test_metrics:
            return
        
        print(f"  Found {len(test_metrics)} test metrics in final stage")
        
        # Group test metrics by prefix
        cts_metrics = {}
        placeopt_metrics = {}
        globalroute_metrics = {}
        detailedroute_metrics = {}
        
        for metric_name, metric_data in test_metrics.items():
            actual_value = metric_data.get('actual_value')
            
            if metric_name.startswith('cts__'):
                cts_metrics[metric_name] = actual_value
            elif metric_name.startswith('placeopt__'):
                placeopt_metrics[metric_name] = actual_value
            elif metric_name.startswith('globalroute__'):
                globalroute_metrics[metric_name] = actual_value
            elif metric_name.startswith('detailedroute__'):
                detailedroute_metrics[metric_name] = actual_value
        
        print(f"    CTS metrics: {len(cts_metrics)}")
        print(f"    Placement (placeopt) metrics: {len(placeopt_metrics)}")
        print(f"    Routing (globalroute) metrics: {len(globalroute_metrics)}")
        print(f"    Routing (detailedroute) metrics: {len(detailedroute_metrics)}")
        
        # Map to appropriate stages/substages
        for result in all_stage_results:
            stage = result.get('stage')
            metrics = result.get('metrics')
            
            if not metrics:
                continue
            
            # CTS stage: add cts__ metrics to design_metrics_final
            if stage == 'cts' and cts_metrics:
                if 'design_metrics_final' not in metrics:
                    metrics['design_metrics_final'] = {}
                metrics['design_metrics_final'].update(cts_metrics)
                print(f"  → Mapped {len(cts_metrics)} CTS metrics to CTS stage")
            
            # Placement stage: add placeopt__ metrics to design_metrics_final
            elif stage == 'placement' and placeopt_metrics:
                if 'design_metrics_final' not in metrics:
                    metrics['design_metrics_final'] = {}
                metrics['design_metrics_final'].update(placeopt_metrics)
                print(f"  → Mapped {len(placeopt_metrics)} placeopt metrics to placement stage")
            
            # Routing stage: add globalroute__ and detailedroute__ to appropriate substages
            elif stage == 'routing':
                substages = metrics.get('substages', {})
                
                # Map globalroute__ to 5_1_grt
                if '5_1_grt' in substages and globalroute_metrics:
                    if 'design_metrics_final' not in substages['5_1_grt']:
                        substages['5_1_grt']['design_metrics_final'] = {}
                    substages['5_1_grt']['design_metrics_final'].update(globalroute_metrics)
                    print(f"  → Mapped {len(globalroute_metrics)} globalroute metrics to routing substage 5_1_grt")
                
                # Map detailedroute__ to 5_2_route
                if '5_2_route' in substages and detailedroute_metrics:
                    if 'design_metrics_final' not in substages['5_2_route']:
                        substages['5_2_route']['design_metrics_final'] = {}
                    substages['5_2_route']['design_metrics_final'].update(detailedroute_metrics)
                    print(f"  → Mapped {len(detailedroute_metrics)} detailedroute metrics to routing substage 5_2_route")
        
        # Re-write metrics files with updated data
        print()
        print("  Re-writing metrics files with mapped data...")
        for result in all_stage_results:
            if not result.get('metrics') or not result.get('metrics_file'):
                continue
            
            stage = result.get('stage')
            if stage not in ['cts', 'placement', 'routing']:
                continue
            
            try:
                metrics_for_file = strip_summary_fields(result['metrics'])
                with open(result['metrics_file'], 'w', encoding='utf-8') as f:
                    json.dump(metrics_for_file, f, indent=2)
                print(f"    Updated: {os.path.basename(result['metrics_file'])}")
            except Exception as e:
                print(f"    [WARN] Failed to update {result['metrics_file']}: {e}")
    
    def extract_all_stages(self) -> Dict:
        """
        Extract all stages from the log file by:
        1. Reading the log file once
        2. Splitting by stage markers (1_*, 2_*, 3_*, etc.)
        3. Passing each stage chunk to the appropriate extraction function
        Returns results in chronological order.
        
        Returns:
            Dictionary containing results from all stages in chronological order
        """
        print("=" * 80)
        print(f"LOG EXTRACTION PIPELINE: {self.log_file_path}")
        print("=" * 80)
        print()
        
        # Step 1: Load log file once
        if not self.load_log_file():
            return {'success': False, 'error': 'Failed to load log file'}

        # Line accounting (original log)
        log_lines = self.log_content.splitlines() if self.log_content else []
        total_log_lines = len(log_lines)

        # Design info extraction (from raw log, before splitting)
        design_info = extract_design_info(self.log_content or "")
        design_info_file = None
        try:
            design_info_file = os.path.join(self.output_dir, f"{self.base_name}_design_info.json")
            with open(design_info_file, 'w', encoding='utf-8') as f:
                json.dump(design_info, f, indent=2)
        except Exception as e:
            print(f"[WARN] Failed to write design info JSON: {e}")
            design_info_file = None

        # Capture KLayout "design_name" (often indicates the block/module name, e.g., uart_rx).
        # We keep appearance order; for grouping we later map occurrence index -> design_name.
        self.klayout_design_names = extract_klayout_design_names(self.log_content or "")
        
        print()
        print("Splitting log by stage markers...")
        print("-" * 80)
        
        # Step 2: Split log by stage markers
        stage_chunks = self.split_log_by_stages()
        
        if not stage_chunks:
            print("  [WARN] No stages found in log file")
            return {
                'success': False,
                'error': 'No stages detected in log file',
                'log_file': self.log_file_path
            }
        
        # Line accounting (stage chunks)
        chunk_total_lines = sum(max(0, c['end_line'] - c['start_line']) for c in stage_chunks)
        intervals = sorted(
            [(max(0, int(c['start_line'])), max(0, int(c['end_line']))) for c in stage_chunks],
            key=lambda x: (x[0], x[1])
        )
        merged = []
        for s, e in intervals:
            if not merged or s > merged[-1][1]:
                merged.append([s, e])
            else:
                merged[-1][1] = max(merged[-1][1], e)
        covered_lines = sum(max(0, e - s) for s, e in merged)
        overlap_lines = max(0, chunk_total_lines - covered_lines)
        gap_lines = max(0, total_log_lines - covered_lines)

        # Compute gap ranges (uncovered line intervals) for debugging / audit
        gap_ranges = []
        if merged:
            if merged[0][0] > 0:
                gap_ranges.append((0, merged[0][0]))
            for (ps, pe), (ns, ne) in zip(merged, merged[1:]):
                if ns > pe:
                    gap_ranges.append((pe, ns))
            if merged[-1][1] < total_log_lines:
                gap_ranges.append((merged[-1][1], total_log_lines))
        else:
            if total_log_lines > 0:
                gap_ranges.append((0, total_log_lines))

        print(f"Line check: log_total={total_log_lines}, chunks_total={chunk_total_lines}, chunks_covered={covered_lines}, gaps={gap_lines}, overlaps={overlap_lines}")

        # Print uncovered lines (bounded) and save full uncovered lines to file
        gap_file = None
        if gap_lines > 0 and log_lines:
            try:
                gap_file = os.path.join(self.output_dir, f"{self.base_name}_gap_lines.txt")
                max_print = 50  # avoid flooding console
                printed = 0

                print("-" * 80)
                print(f"UNCOVERED LINES (showing up to {max_print}; full list in {os.path.basename(gap_file)}):")

                with open(gap_file, 'w', encoding='utf-8') as f:
                    f.write(f"Log file: {self.log_file_path}\n")
                    f.write(f"Total log lines: {total_log_lines}\n")
                    f.write(f"Uncovered line count: {gap_lines}\n")
                    f.write(f"Gap ranges (0-based, end exclusive): {gap_ranges}\n")
                    f.write("-" * 80 + "\n")

                    for gs, ge in gap_ranges:
                        for ln in range(gs, ge):
                            # 1-based line numbers in output for readability
                            line_no = ln + 1
                            line_txt = log_lines[ln] if ln < len(log_lines) else ""
                            f.write(f"L{line_no}: {line_txt}\n")

                            if printed < max_print:
                                print(f"L{line_no}: {line_txt}")
                                printed += 1

                if printed >= max_print and gap_lines > max_print:
                    print(f"... truncated ({gap_lines - max_print} more uncovered lines not shown)")
                print("-" * 80)
            except Exception as e:
                print(f"[WARN] Failed to write uncovered lines file: {e}")
                gap_file = None
        
        print(f"Found {len(stage_chunks)} stage chunk(s):")
        for chunk in stage_chunks:
            print(f"  - Stage {chunk['stage_number']} ({chunk['stage_type']}) occurrence {chunk['occurrence']}: lines {chunk['start_line']}-{chunk['end_line']}")
        
        print()
        print("Extracting stages...")
        print("-" * 80)
        
        # Step 3: Extract each stage chunk
        all_stage_results = []
        
        for chunk in stage_chunks:
            result = self.extract_stage_by_type(chunk)
            if result:
                all_stage_results.append(result)

        # Line accounting (extracted / filtered)
        extracted_total_lines = sum(int(s.get('line_count', 0) or 0) for s in all_stage_results)
        print(f"Line check: extracted_total={extracted_total_lines} (filtered lines written across all extracted stages)")

        # Post-process: Map final stage test metrics to appropriate substages
        print()
        print("Post-processing metrics...")
        print("-" * 80)
        self._map_final_metrics_to_substages(all_stage_results)
        
        # Check if any stage has multiple occurrences
        stage_counts = {}
        for stage in all_stage_results:
            stage_type = stage.get('stage')
            stage_counts[stage_type] = stage_counts.get(stage_type, 0) + 1
        
        max_occurrence = max(stage_counts.values()) if stage_counts else 0
        
        # Only annotate .rpt metrics if all stages have exactly 1 occurrence
        if max_occurrence > 1:
            print()
            print(f"[INFO] Skipping .rpt metrics annotation (max stage occurrence: {max_occurrence} > 1)")
            print("       RPT annotation is only supported for single-occurrence flows")
        else:
            # Annotate stages with .rpt metrics (only for single occurrence stages)
            # Try to infer rpt_dir from log file path
            rpt_dir = None
            log_dir = os.path.dirname(os.path.abspath(self.log_file_path))
            
            # Strategy 1: Look for reports directory in parent directories (up to 3 levels)
            search_dir = log_dir
            for _ in range(3):
                potential_reports = os.path.join(search_dir, 'reports')
                if os.path.exists(potential_reports):
                    rpt_dir = potential_reports
                    break
                search_dir = os.path.dirname(search_dir)
            
            # Strategy 2: Look for reports directory in subdirectories (common for final_report_* structure)
            if not rpt_dir:
                # Look for final_report_*/reports pattern
                report_patterns = [
                    os.path.join(log_dir, 'final_report_*', 'reports'),
                    os.path.join(log_dir, '*', 'reports'),
                ]
                for pattern in report_patterns:
                    matches = glob.glob(pattern)
                    if matches:
                        rpt_dir = matches[0]  # Take first match
                        break
            
            # Strategy 3: Find the base directory with .rpt files
            if rpt_dir and os.path.exists(rpt_dir):
                # Navigate to the actual .rpt files (usually in reports/platform/design/base/)
                for root, dirs, files in os.walk(rpt_dir):
                    if any(f.endswith('.rpt') for f in files):
                        rpt_dir = root
                        break
            
            annotate_stages_with_rpt_metrics(all_stage_results, rpt_dir)

        # Build the results dictionary
        results = {
            'success': True,
            'log_file': self.log_file_path,
            'output_dir': self.output_dir,
            'timestamp': datetime.now().isoformat(),
            'log_content': self.log_content,  # Store for flow_execution_summary
            'stages_chronological': all_stage_results,
            'stages_by_type': {},
            'design_info': design_info,
            'design_info_file': design_info_file,
            'klayout_design_names': self.klayout_design_names,
            'metrics': {
                'per_stage': [],
                'aggregate_file': None
            },
            'line_accounting': {
                'log_total_lines': total_log_lines,
                'chunks_total_lines': chunk_total_lines,
                'chunks_covered_lines': covered_lines,
                'gap_lines': gap_lines,
                'gap_ranges': gap_ranges,
                'gap_file': gap_file,
                'overlap_lines': overlap_lines,
                'extracted_total_lines': extracted_total_lines
            }
        }
        
        # Group by stage type for easy access
        for stage_result in all_stage_results:
            stage_type = stage_result['stage']
            if stage_type not in results['stages_by_type']:
                results['stages_by_type'][stage_type] = []
            results['stages_by_type'][stage_type].append(stage_result)

            if stage_result.get('metrics') is not None:
                per_stage_entry = {
                    'stage': stage_type,
                    'occurrence': stage_result.get('occurrence', 1),
                    'metrics_file': stage_result.get('metrics_file'),
                    'metrics': stage_result.get('metrics'),
                }
                # Note: rpt_metrics is now inside the metrics dict, so it's already included
                
                results['metrics']['per_stage'].append(per_stage_entry)

        # Write one aggregate metrics JSON (includes per-stage metrics dicts)
        try:
            agg_file = os.path.join(self.output_dir, f"{self.base_name}_all_stages_metrics.json")

            # Prefer human design name from DESIGN_NAME=... when available; otherwise fall back to base_name.
            di_sum = (results.get('design_info') or {}).get('summary', {}) if isinstance(results.get('design_info'), dict) else {}
            top_design_name = None
            if isinstance(di_sum, dict):
                dns = di_sum.get('design_names') or []
                if isinstance(dns, list) and dns:
                    top_design_name = dns[0]

            # Generate flow execution summary
            flow_summary = generate_flow_execution_summary(results)

            aggregate_payload = {
                'design_name': top_design_name or self.base_name,
                'sub_design_names': results.get('klayout_design_names', []),
                # Keep design-info compact as well (remove 'summary' sub-dict)
                'design_info': strip_summary_fields(results.get('design_info', {})),
                'log_file': self.log_file_path,
                'timestamp': results['timestamp'],
                # Add flow_execution_summary
                'flow_execution_summary': flow_summary,
                'stages': [
                    {
                        # Add sequence_number
                        'sequence_number': idx + 1,
                        # explicit stage identity
                        'stage': s.get('stage'),
                        'occurrence': s.get('occurrence', 1),
                        # explicit design identity
                        'design_name': (s.get('metrics') or {}).get('design_name'),
                        'sub_design_name': (s.get('metrics') or {}).get('sub_design_name'),
                        # file + full parsed metrics (rpt_metrics is now inside metrics)
                        'metrics_file': s.get('metrics_file'),
                        # Export stripped metrics (no 'summary')
                        'metrics': strip_summary_fields(s.get('metrics')),
                    }
                    for idx, s in enumerate(results['metrics']['per_stage'])
                ],
            }
            with open(agg_file, 'w', encoding='utf-8') as f:
                json.dump(aggregate_payload, f, indent=2)
            results['metrics']['aggregate_file'] = agg_file
        except Exception as e:
            print(f"[WARN] Failed to write aggregate metrics JSON: {e}")

        # Group stages under KLayout design_name keys (design occurrence index -> design_name)
        # This enables consuming results like: results['by_design_name']['uart_rx']['stages_by_type']['cts']
        by_design_name: Dict[str, Dict] = {}
        for stage_result in all_stage_results:
            occ = int(stage_result.get('occurrence', 1) or 1)
            design_name = None
            if 0 <= (occ - 1) < len(self.klayout_design_names):
                design_name = self.klayout_design_names[occ - 1]
            if not design_name:
                design_name = f"{self.base_name}_occurrence_{occ}"

            if design_name not in by_design_name:
                by_design_name[design_name] = {
                    'design_name': design_name,
                    'occurrence': occ,
                    'stages_chronological': [],
                    'stages_by_type': {},
                }

            by_design_name[design_name]['stages_chronological'].append(stage_result)
            st = stage_result.get('stage', 'unknown')
            by_design_name[design_name]['stages_by_type'].setdefault(st, []).append(stage_result)

        results['by_design_name'] = by_design_name
        
        print()
        print("-" * 80)
        print(f"[OK] Extraction complete! {len(all_stage_results)} total stage(s) extracted.")
        
        # Print summary by type
        for stage_type, stages in results['stages_by_type'].items():
            print(f"  {stage_type}: {len(stages)} occurrence(s)")
        
        print(f"  Output directory: {self.output_dir}")
        print("=" * 80)
        
        return results
    
    def generate_summary(self, results: Dict) -> str:
        """
        Generate a summary report of the extraction.
        
        Args:
            results: Results dictionary from extract_all_stages()
        
        Returns:
            Summary string
        """
        summary = []
        summary.append("=" * 80)
        summary.append("EXTRACTION SUMMARY")
        summary.append("=" * 80)
        summary.append(f"Log file: {results.get('log_file', 'N/A')}")
        summary.append(f"Output directory: {results.get('output_dir', 'N/A')}")
        summary.append(f"Timestamp: {results.get('timestamp', 'N/A')}")
        summary.append("")

        # Design info summary
        di = results.get('design_info', {}) or {}
        di_sum = di.get('summary', {}) if isinstance(di, dict) else {}
        if di_sum:
            summary.append("DESIGN INFO:")
            summary.append(f"  design_names={di_sum.get('design_names', [])}")
            summary.append(f"  technology_nodes={di_sum.get('technology_nodes', [])}")
            summary.append(f"  config_mks={di_sum.get('config_mks', [])}")
            if results.get('design_info_file'):
                summary.append(f"  design_info_file={os.path.basename(results['design_info_file'])}")
            summary.append("")

        # Line accounting
        la = results.get('line_accounting', {}) or {}
        if la:
            summary.append("LINE CHECK:")
            summary.append(
                f"  log_total_lines={la.get('log_total_lines', 'N/A')}  "
                f"chunks_total_lines={la.get('chunks_total_lines', 'N/A')}  "
                f"chunks_covered_lines={la.get('chunks_covered_lines', 'N/A')}  "
                f"gaps={la.get('gap_lines', 'N/A')}  "
                f"overlaps={la.get('overlap_lines', 'N/A')}  "
                f"extracted_total_lines={la.get('extracted_total_lines', 'N/A')}"
            )
            if la.get('gap_file'):
                summary.append(f"  gap_file={os.path.basename(la['gap_file'])}")
            summary.append("")
        
        # Summary by stage type
        stages_by_type = results.get('stages_by_type', {})
        total_occurrences = sum(len(stages) for stages in stages_by_type.values())
        
        summary.append(f"Total stage occurrences extracted: {total_occurrences}")
        summary.append("-" * 80)
        
        for stage_type, stages in sorted(stages_by_type.items()):
            summary.append(f"\n{stage_type.upper()} ({len(stages)} occurrence(s)):")
            for stage_data in stages:
                occurrence = stage_data.get('occurrence', 1)
                line_count = stage_data.get('line_count', 0)
                output_file = os.path.basename(stage_data.get('output_file', 'N/A'))
                summary.append(f"  #{occurrence}: {line_count:5d} lines -> {output_file}")
        
        # Chronological order
        stages_chrono = results.get('stages_chronological', [])
        if stages_chrono:
            summary.append("\n" + "-" * 80)
            summary.append("CHRONOLOGICAL ORDER:")
            summary.append("-" * 80)
            for i, stage_data in enumerate(stages_chrono, 1):
                stage_type = stage_data.get('stage', 'unknown')
                occurrence = stage_data.get('occurrence', 1)
                line_count = stage_data.get('line_count', 0)
                stage_name = stage_data.get('stage_name', f"{stage_type}_{occurrence}")
                summary.append(f"{i:2d}. {stage_name:30s} ({line_count:5d} lines)")

        # Metrics summary (compact)
        metrics_entries = results.get('metrics', {}).get('per_stage', [])
        if metrics_entries:
            metrics_map = {(m.get('stage'), m.get('occurrence', 1)): m.get('metrics') for m in metrics_entries}

            def _fmt(v, ndigits: int = 3) -> str:
                if v is None:
                    return "N/A"
                if isinstance(v, float):
                    return f"{v:.{ndigits}f}"
                return str(v)

            summary.append("\n" + "-" * 80)
            summary.append("METRICS SUMMARY:")
            summary.append("-" * 80)

            for i, stage_data in enumerate(stages_chrono, 1):
                stage = stage_data.get('stage', 'unknown')
                occ = stage_data.get('occurrence', 1)
                metrics = metrics_map.get((stage, occ))
                label = f"{i:2d}. {stage}_{occ}"

                if not metrics:
                    summary.append(f"{label:20s}  metrics: N/A")
                    continue

                if stage == 'synthesis':
                    run0 = (metrics.get('runs') or [{}])[0]
                    syn_sum = run0.get('summary', {})
                    clock = syn_sum.get('clock_period')
                    cpu = syn_sum.get('total_cpu_time')
                    warns = syn_sum.get('warnings')
                    errs = syn_sum.get('errors')
                    summary.append(
                        f"{label:20s}  clock={_fmt(clock)}  cpu_s={_fmt(cpu, ndigits=2)}  warnings={_fmt(warns)}  errors={_fmt(errs)}"
                    )
                    continue

                if stage in ('floorplan', 'placement', 'cts', 'routing'):
                    s = metrics.get('summary', {}) if isinstance(metrics, dict) else {}
                    area = s.get('area') or s.get('final_area')
                    util = s.get('utilization') or s.get('final_utilization')
                    wns = s.get('wns')
                    tns = s.get('tns')
                    hpwl = s.get('hpwl')
                    viol = s.get('violations')
                    extras = []
                    if stage == 'routing':
                        if s.get('total_wirelength') is not None:
                            extras.append(f"wirelen={_fmt(s.get('total_wirelength'), ndigits=1)}")
                        if s.get('final_vias') is not None:
                            extras.append(f"vias={_fmt(s.get('final_vias'))}")
                    if stage == 'cts' and s.get('clock_buffers') is not None:
                        extras.append(f"clk_buf={_fmt(s.get('clock_buffers'))}")

                    extra_str = ("  " + "  ".join(extras)) if extras else ""
                    summary.append(
                        f"{label:20s}  area={_fmt(area, ndigits=1)}  util={_fmt(util, ndigits=1)}  wns={_fmt(wns)}  tns={_fmt(tns, ndigits=1)}  viol={_fmt(viol)}  hpwl={_fmt(hpwl, ndigits=1)}{extra_str}"
                    )
                    continue

                if stage == 'final':
                    dm = metrics.get('design_metrics', {}) if isinstance(metrics, dict) else {}
                    area = dm.get('area')
                    util = dm.get('utilization')
                    ir = metrics.get('ir_drop', {}) if isinstance(metrics, dict) else {}
                    vdd_wc = (ir.get('vdd') or {}).get('worstcase_ir_drop_v')
                    vss_wc = (ir.get('vss') or {}).get('worstcase_ir_drop_v')
                    summary.append(
                        f"{label:20s}  area={_fmt(area)}  util={_fmt(util)}  ir_vdd_wc={_fmt(vdd_wc)}  ir_vss_wc={_fmt(vss_wc)}"
                    )
                    continue

                summary.append(f"{label:20s}  metrics: present")
        
        summary.append("=" * 80)
        
        return '\n'.join(summary)


def generate_flow_execution_summary(results: Dict) -> Dict:
    """
    Generate a flow execution summary in JSON format with design, technology_node,
    and each stage in chronological order (without detailed metrics).
    
    Args:
        results: Results dictionary from extract_all_stages()
    
    Returns:
        Dictionary with flow execution summary in chronological order
    """
    if not results.get('success'):
        return {}
    
    # Extract design info
    design_info = results.get('design_info', {})
    design_info_summary = design_info.get('summary', {}) if isinstance(design_info, dict) else {}
    
    # Get design name and technology node
    design_name = None
    technology_node = None
    if design_info_summary:
        dns = design_info_summary.get('design_names', [])
        if dns:
            design_name = dns[0]
        technology_nodes = design_info_summary.get('technology_nodes', [])
        if technology_nodes:
            technology_node = technology_nodes[0]
    
    # Detect execution success/failure
    log_content = results.get('log_content', '')
    execute_successfully = False
    exit_code = None
    exit_line = None
    errors_detected = []
    
    if log_content:
        lines = log_content.splitlines() if isinstance(log_content, str) else []
        # Look for exit status
        exit_re = re.compile(r'\+\s+exit\s+(\d+)')
        for i, line in enumerate(lines):
            m = exit_re.search(line)
            if m:
                exit_code = int(m.group(1))
                exit_line = line.strip()
                if exit_code == 0:
                    execute_successfully = True
                break
        
        # Look for error lines
        error_re = re.compile(r'\b[Ee]rror\b', re.IGNORECASE)
        for i, line in enumerate(lines):
            if error_re.search(line):
                errors_detected.append({
                    'line': i + 1,
                    'content': line.strip()
                })
    
    # Get stages in chronological order with sequence numbers
    stages_chrono = results.get('stages_chronological', [])
    stages_summary = []
    
    for seq_num, stage_result in enumerate(stages_chrono, start=1):
        stage_info = {
            'sequence_number': seq_num,
            'stage_type': stage_result.get('stage'),
            'occurrence': stage_result.get('occurrence', 1),
            'sub_design_name': stage_result.get('sub_design_name'),
            'line_range': {
                'start': stage_result.get('start_line'),
                'end': stage_result.get('end_line')
            },
            'extracted_lines': stage_result.get('line_count', 0),
            'output_file': os.path.basename(stage_result.get('output_file', '')) if stage_result.get('output_file') else None,
            'metrics_file': os.path.basename(stage_result.get('metrics_file', '')) if stage_result.get('metrics_file') else None
        }
        stages_summary.append(stage_info)
    
    flow_summary = {
        'design_name': design_name,
        'technology_node': technology_node,
        'log_file': results.get('log_file'),
        'timestamp': results.get('timestamp'),
        'execute_successfully': execute_successfully,
        'exit_code': exit_code,
        'exit_line': exit_line,
        'errors_detected': errors_detected[:10],  # Limit to first 10 errors
        'total_stages': len(stages_summary),
        'stages': stages_summary
    }
    
    return flow_summary


def extract_log_file(log_file_path: str, output_dir: str = "extracted_logs") -> Dict:
    """
    Convenience function to extract all stages from a log file.
    
    Args:
        log_file_path: Path to the input log file
        output_dir: Directory to save extracted stage files (default: "extracted_logs")
    
    Returns:
        Dictionary containing extraction results
    """
    pipeline = LogExtractionPipeline(log_file_path, output_dir)
    results = pipeline.extract_all_stages()
    
    # Print summary
    print()
    summary = pipeline.generate_summary(results)
    print(summary)
    
    # Save summary to file
    summary_file = os.path.join(output_dir, f"{pipeline.base_name}_extraction_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"\n[OK] Summary saved to: {summary_file}")
    
    return results


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extract all stages from OpenROAD log files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract all stages from a single log file
  python extract_all_stages.py gcdshy130.txt
  
  # Extract to a custom output directory
  python extract_all_stages.py gcdshy130.txt -o my_extracted_logs
  
  # Process multiple log files
  python extract_all_stages.py log1.txt log2.txt log3.txt
        """
    )
    
    parser.add_argument('log_files', nargs='+', help='Path to log file(s) to process')
    parser.add_argument('-o', '--output-dir', default='extracted_logs',
                       help='Output directory for extracted files (default: extracted_logs)')
    
    args = parser.parse_args()
    
    # Process each log file
    for log_file in args.log_files:
        if not os.path.exists(log_file):
            print(f"[ERROR] Log file not found: {log_file}")
            continue
        
        try:
            extract_log_file(log_file, args.output_dir)
            print()
        except Exception as e:
            print(f"[ERROR] Error processing {log_file}: {e}")
            continue


if __name__ == "__main__":
    main()
