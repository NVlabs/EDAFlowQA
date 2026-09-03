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
Stage-specific parsing examples for optimization_process and design_metrics_final.

These examples are used as prompts in qa_generation.py and are dynamically selected
based on the sampled stage.
"""

# Floorplan stage parsing examples
FLOORPLAN_EXAMPLES = """
## Floorplan Stage Parsing Examples

### Accessing ALL repair_timing progressions (WNS improvement across all optimizations):
```python
substages = stage_entry['metrics']['substages']
wns_progression = []

# Iterate through all substages to find all repair_timing operations metric progressions
for substage_name, substage_data in substages.items():
    if 'floorplan' in substage_name:
        for opt in substage_data.get('optimization_process', []):
            if opt['type'] == 'repair_timing':
                start_wns = opt['start_metrics']['wns']
                end_wns = opt['end_metrics']['wns']
                start_tns = opt['start_metrics']['tns']
                end_tns = opt['end_metrics']['tns']
                iterations = opt['iterations_count']
                
                wns_progression.append({
                    'substage': substage_name,
                    'iterations': iterations,
                    'wns_start': f"{start_wns} ns",
                    'wns_end': f"{end_wns} ns",
                    'wns_improvement': f"{end_wns - start_wns} ns",
                    'tns_start': f"{start_tns} ns",
                    'tns_end': f"{end_tns} ns",
                    'tns_improvement': f"{end_tns - start_tns} ns"
                })

# Answer: wns_progression (list of all repair_timing operations with improvements)
```

### Accessing single repair_timing (if only one expected):
```python
substages = stage_entry['metrics']['substages']
floorplan_2_1 = substages.get('2_1_floorplan', {})
for opt in floorplan_2_1.get('optimization_process', []):
    # there could be multiple repair_timing optimizations
    if opt['type'] == 'repair_timing':
        iterations = opt['iterations_count']
        start_wns = opt['start_metrics']['wns']
        end_wns = opt['end_metrics']['wns']
        improvement = end_wns - start_wns
        # Answer: f"{improvement} ns improvement after {iterations} iterations"
        break
```

### Accessing final metrics after floorplan:
```python
floorplan_2_1 = substages.get('2_1_floorplan', {})
wns = floorplan_2_1.get('wns')
tns = floorplan_2_1.get('tns')
area = floorplan_2_1.get('area')
violations = floorplan_2_1.get('violations')
# Answer: {'wns': f"{wns} ns", 'tns': f"{tns} ns", 'area': area, 'violations': violations}
```
"""

# Placement stage parsing examples
PLACEMENT_EXAMPLES = """
## Placement Stage Parsing Examples

### Accessing ALL global_placement convergence progressions:
```python
substages = stage_entry['metrics']['substages']
place_gp = substages.get('3_3_place_gp', {})
gp_progression = []

for opt in place_gp.get('optimization_process', []):
    # there could be multiple global_placement commands in the optimization process
    if opt['type'] == 'global_placement':
        iterations = opt['iterations_count']
        overflow_start = opt['start_metrics']['overflow']
        overflow_end = opt['end_metrics']['overflow']
        reduction = overflow_start - overflow_end
        
        gp_progression.append({
            'iterations': iterations,
            'overflow_start': overflow_start,
            'overflow_end': overflow_end,
            'overflow_reduction': reduction,
            'hpwl_start': opt['start_metrics'].get('hpwl'),
            'hpwl_end': opt['end_metrics'].get('hpwl')
        })

# Answer: gp_progression (list of all global_placement operations)
```

### Accessing FINAL (last) global_placement convergence:
```python
substages = stage_entry['metrics']['substages']
place_gp = substages.get('3_3_place_gp', {})
final_gp = None

for opt in place_gp.get('optimization_process', []):
    if opt['type'] == 'global_placement':
        # Keep updating to get the last one
        iterations = opt['iterations_count']
        overflow_start = opt['start_metrics']['overflow']
        overflow_end = opt['end_metrics']['overflow']
        reduction = overflow_start - overflow_end
        
        final_gp = {
            'iterations': iterations,
            'overflow_reduction': reduction,
            'final_overflow': overflow_end,
            'utilization': opt['results'].get('utilization'),
            'target_density': opt['results'].get('target_density')
        }

# Answer: final_gp (last global_placement operation only)
```

### Accessing buffer insertion/removal:
```python
total_inserted = 0
total_removed = 0
for substage_data in substages.values():
    for opt in substage_data.get('optimization_process', []):
        if opt['type'] == 'buffer_insertion':
            if 'input_buffers' in opt['results']:
                total_inserted += opt['results']['input_buffers']
            if 'output_buffers' in opt['results']:
                total_inserted += opt['results']['output_buffers']
        elif opt['type'] == 'buffer_removal':
            if 'removed_buffers' in opt['results']:
                total_removed += opt['results']['removed_buffers']
        # check if there any related buffer information in the optimization process
# Answer: {'inserted': total_inserted, 'removed': total_removed}
```

### Accessing ALL detailed_placement displacement progressions:
```python
place_dp = substages.get('3_5_place_dp', {})
displacement_progression = []

for opt in place_dp.get('optimization_process', []):
    if opt['type'] == 'detailed_placement':
        results = opt['results']
        displacement_progression.append({
            'total_displacement_u': results.get('total_displacement_u'),
            'average_displacement_u': results.get('average_displacement_u'),
            'max_displacement_u': results.get('max_displacement_u'),
            'hpwl_original': results.get('original_hpwl_u'),
            'hpwl_final': results.get('final_hpwl_u'),
            'hpwl_delta_percent': results.get('final_delta_hpwl_percent')
        })

# Answer: displacement_progression (list of all detailed_placement operations)
```

### Accessing FINAL (last) detailed_placement metrics:
```python
place_dp = substages.get('3_5_place_dp', {})
final_displacement = None

for opt in place_dp.get('optimization_process', []):
    if opt['type'] == 'detailed_placement':
        # Keep updating to get the last one
        results = opt['results']
        final_displacement = {
            'total': f"{results.get('total_displacement_u')} units",
            'average': f"{results.get('average_displacement_u')} units",
            'max': f"{results.get('max_displacement_u')} units",
            'hpwl_final': results.get('final_hpwl_u')
        }

# Answer: final_displacement (last detailed_placement operation only)
```

### Accessing io_placement:
```python
place_iop = substages.get('3_2_place_iop', {})
for opt in place_iop.get('optimization_process', []):
    if opt['type'] == 'io_placement':
        io_hpwl = opt['results'].get('io_nets_hpwl_um')
        # Answer: f"{io_hpwl} um"
```

### Accessing design_metrics_final:
```python
design_metrics_final = stage_entry['metrics'].get('design_metrics_final', {})
instance_area = design_metrics_final.get('placeopt__design__instance__area')
stdcell_count = design_metrics_final.get('placeopt__design__instance__count__stdcell')
# Answer: {'area': instance_area, 'count': stdcell_count}
```
"""

# CTS stage parsing examples
CTS_EXAMPLES = """
## CTS Stage Parsing Examples

### Accessing clock_tree_synthesis:
```python
substages = stage_entry['metrics']['substages']
cts_4_1 = substages.get('4_1_cts', {})
for opt in cts_4_1.get('optimization_process', []):
    if opt['type'] == 'clock_tree_synthesis':
        results = opt['results']
        total_sinks = results.get('total_sinks')
        clock_buffers = results.get('clock_buffers_created')
        avg_wirelength = results.get('avg_sink_wirelength_um')
        clock_nets = results.get('clock_nets', [])
        # Answer: {'sinks': total_sinks, 'buffers': clock_buffers, 
        #          'wirelength': f"{avg_wirelength} um", 'nets': clock_nets}
```

### Accessing clock nets details:
```python
for opt in cts_4_1.get('optimization_process', []):
    if opt['type'] == 'clock_tree_synthesis':
        clock_nets = opt['results'].get('clock_nets', [])
        for net in clock_nets:
            net_name = net['name']
            sinks = net['sinks']
            avg_length = net['avg_sink_wire_length_um']
            depth_min = net['path_depth_min']
            depth_max = net['path_depth_max']
            # Answer: list of net details
```

### Accessing ALL repair_timing progressions in CTS:
```python
cts_4_1 = substages.get('4_1_cts', {})
repair_timing_progression = []

for opt in cts_4_1.get('optimization_process', []):
    if opt['type'] == 'repair_timing':
        iterations = opt['iterations_count']
        wns_start = opt['start_metrics']['wns']
        wns_end = opt['end_metrics']['wns']
        tns_start = opt['start_metrics']['tns']
        tns_end = opt['end_metrics']['tns']
        violations_start = opt['start_metrics'].get('violation_end_points')
        violations_end = opt['end_metrics'].get('violation_end_points')
        
        repair_timing_progression.append({
            'iterations': iterations,
            'wns_start': f"{wns_start} ns",
            'wns_end': f"{wns_end} ns",
            'wns_improvement': f"{wns_end - wns_start} ns",
            'tns_start': f"{tns_start} ns",
            'tns_end': f"{tns_end} ns",
            'tns_improvement': f"{tns_end - tns_start} ns",
            'violations_start': violations_start,
            'violations_end': violations_end
        })

# Answer: repair_timing_progression (list of all repair_timing operations in CTS)
```

### Accessing FINAL (last) repair_timing in CTS:
```python
cts_4_1 = substages.get('4_1_cts', {})
final_repair_timing = None

for opt in cts_4_1.get('optimization_process', []):
    if opt['type'] == 'repair_timing':
        # Keep updating to get the last one
        iterations = opt['iterations_count']
        wns_start = opt['start_metrics']['wns']
        wns_end = opt['end_metrics']['wns']
        
        final_repair_timing = {
            'iterations': iterations,
            'wns_improvement': f"{wns_end - wns_start} ns",
            'final_wns': f"{wns_end} ns",
            'final_tns': f"{opt['end_metrics']['tns']} ns"
        }

# Answer: final_repair_timing (last repair_timing operation only)
```

### Accessing repair_timing_summary:
```python
for opt in cts_4_1.get('optimization_process', []):
    if opt['type'] == 'repair_timing_summary':
        removed_buffers = opt.get('removed_buffers')
        inserted_buffers = opt.get('inserted_buffers')
        resized_instances = opt.get('resized_instances')
        cloned_instances = opt.get('cloned_instances')
        # Answer: {'removed': removed_buffers, 'inserted': inserted_buffers, 
        #          'resized': resized_instances, 'cloned': cloned_instances}
```

### Accessing CTS finished timing metrics:
```python
design_metrics_final = stage_entry['metrics'].get('design_metrics_final', {})
setup_buffers = design_metrics_final.get('cts__design__instance__count__setup_buffer')
hold_buffers = design_metrics_final.get('cts__design__instance__count__hold_buffer')
setup_wns = design_metrics_final.get('cts__timing__setup__ws')
setup_tns = design_metrics_final.get('cts__timing__setup__tns')
# Answer: {'setup_buffers': setup_buffers, 'wns': f"{setup_wns} ns"}
```
"""

# Routing stage parsing examples
ROUTING_EXAMPLES = """
## Routing Stage Parsing Examples

### Accessing global_route:
```python
substages = stage_entry['metrics']['substages']
grt = substages.get('5_1_grt', {})
for opt in grt.get('optimization_process', []):
    if opt['type'] == 'global_route' and 'congestion_report_file' in opt.get('command', ''):
        results = opt['results']
        wirelength = results.get('total_wirelength_um')
        overflow = results.get('total_overflow')
        usage = results.get('total_usage_percent')
        # Answer: {'wirelength': f"{wirelength} um", 'overflow': overflow}
```

### Accessing repair_timing in routing:
```python
for opt in grt.get('optimization_process', []):
    if opt['type'] == 'repair_timing':
        iterations = opt['iterations_count']
        wns_start = opt['start_metrics']['wns']
        wns_end = opt['end_metrics']['wns']
        # Answer: {'iterations': iterations, 'improvement': f"{wns_end - wns_start} ns"}
```

### Accessing detailed_route DRC convergence:
```python
route_5_2 = substages.get('5_2_route', {})
for opt in route_5_2.get('optimization_process', []):
    if opt['type'] == 'detailed_route':
        drc_iters = opt['drc_iterations']
        initial_viols = drc_iters[0]['violations']
        final_viols = drc_iters[-1]['violations']
        iterations = len(drc_iters)
        # Answer: {'iterations': iterations, 'initial': initial_viols, 
        #          'final': final_viols, 'all_iters': drc_iters}
```

### Accessing placement_impact from global_route:
```python
for opt in grt.get('optimization_process', []):
    if opt['type'] == 'global_route' and opt.get('placement_impact'):
        impact = opt['placement_impact']
        displacement = impact.get('total_displacement_u')
        hpwl_before = impact.get('hpwl_before')
        hpwl_after = impact.get('hpwl_after')
        # Answer: {'displacement': displacement, 'hpwl_change': hpwl_after - hpwl_before}
```

### Accessing final routing metrics of the global/detailed routing stage:
```python
# Global routing metrics
grt = substages.get('5_1_grt', {})
grt_metrics = grt.get('design_metrics_final', {})
gr_setup_wns = grt_metrics.get('globalroute__timing__setup__ws')
gr_antenna = grt_metrics.get('globalroute__antenna_diodes_count')

# Detailed routing metrics
route_5_2 = substages.get('5_2_route', {})
route_metrics = route_5_2.get('design_metrics_final', {})
dr_wirelength = route_metrics.get('detailedroute__route__wirelength')
dr_drc_errors = route_metrics.get('detailedroute__route__drc_errors')
dr_setup_wns = route_metrics.get('detailedroute__timing__setup__ws')
# Answer: {'global_route': {'wns': f"{gr_setup_wns} ns"}, 
#          'detailed_route': {'wirelength': dr_wirelength, 'drc': dr_drc_errors}}
```
"""

# Synthesis stage parsing examples
SYNTHESIS_EXAMPLES = """
## Synthesis Stage Parsing Examples

### Accessing synthesis steps:
```python
metrics = stage_entry.get('metrics', {})
runs = metrics.get('runs', [])
for run in runs:
    steps = run.get('steps', [])
    for step in steps:
        number = step.get('number')
        description = step.get('description')
        # Answer: list of steps
```

### Accessing timing constraints:
```python
for run in runs:
    timing = run.get('timing', {})
    clock_period = timing.get('clock_period_ns')
    # Answer: f"{clock_period} ns"
```

### Accessing statistics:
```python
for run in runs:
    stats = run.get('statistics', {})
    cells = stats.get('cells', {})
    chip_area = stats.get('chip_area', {})
    # Answer: {'cells': cells, 'area': chip_area}
```
"""

# Final stage parsing examples
FINAL_EXAMPLES = """
## Final Stage Parsing Examples

========== CRITICAL STRUCTURAL DIFFERENCES =============
**The final stage has a COMPLETELY DIFFERENT structure than other stages:**
1. NO substages - all metrics are at the TOP LEVEL under stage_entry['metrics']
2. Uses 'design_metrics' (NOT 'design_metrics_final' like other stages)
3. rpt_metrics is directly at stage_entry['metrics']['rpt_metrics']
4. Cell counts, IR drop, and other metrics are directly under stage_entry['metrics']

**Key differences summary:**
- Other stages: stage_entry['metrics']['substages'][substage_name][...metrics...]
- Final stage:  stage_entry['metrics'][...metrics directly here...]

Always check: if stage_entry['stage'] == 'final' or stage_entry['stage_type'] == 'final'

### Accessing IR drop (FINAL STAGE ONLY):
```python
metrics = stage_entry.get('metrics', {})
ir_drop = metrics.get('ir_drop', {})
vdd = ir_drop.get('vdd', {})
vss = ir_drop.get('vss', {})
avg_drop_vdd = vdd.get('average_ir_drop_v')
worst_drop_vdd = vdd.get('worstcase_ir_drop_v')
avg_drop_vss = vss.get('average_ir_drop_v')
worst_drop_vss = vss.get('worstcase_ir_drop_v')
# Answer: {'vdd_avg': f"{avg_drop_vdd} V", 'vdd_worst': f"{worst_drop_vdd} V",
#          'vss_avg': f"{avg_drop_vss} V", 'vss_worst': f"{worst_drop_vss} V"}
```

### Accessing cell counts (FINAL STAGE ONLY):
```python
metrics = stage_entry.get('metrics', {})
cell_counts = metrics.get('cell_counts', {})
fill_cells = cell_counts.get('fill_cell_count')
clock_buffers = cell_counts.get('clock_buffer_count')
timing_repair_buffers = cell_counts.get('timing_repair_buffer_count')
sequential_cells = cell_counts.get('sequential_cell_count')
combinational_cells = cell_counts.get('combinational_cell_count')
# Answer: {'fill': fill_cells, 'clk_buf': clock_buffers, 
#          'timing_repair_buf': timing_repair_buffers,
#          'sequential': sequential_cells, 'combinational': combinational_cells}
```

### Accessing partial final design metrics (uses 'design_metrics') If failed, try to access through rpt_metrics:
```python
metrics = stage_entry.get('metrics', {})
design_metrics = metrics.get('design_metrics', {})  # NOTE: NOT design_metrics_final!
area = design_metrics.get('area')
setup_wns = design_metrics.get('setup_wns')
setup_tns = design_metrics.get('setup_tns')
hold_wns = design_metrics.get('hold_wns')
hold_tns = design_metrics.get('hold_tns')
# Answer: {'area': f"{area} um^2", 'setup_wns': f"{setup_wns} ns", 
#          'setup_tns': f"{setup_tns} ns", 'hold_wns': f"{hold_wns} ns", 
#          'hold_tns': f"{hold_tns} ns"}
```

### Accessing test metrics (pass/fail):
```python
metrics = stage_entry.get('metrics', {})
test_metrics = metrics.get('test_metrics', {})
for metric_name, metric_data in test_metrics.items():
    actual = metric_data.get('actual_value')
    threshold = metric_data.get('threshold_value')
    passed = metric_data.get('passed')
    # Answer: dict of metrics with pass/fail status
```

### Accessing timing from rpt_metrics in final stage:
```python
# IMPORTANT: For final stage, rpt_metrics is at TOP LEVEL, not in substages!
metrics = stage_entry.get('metrics', {})
rpt_metrics = metrics.get('rpt_metrics', {})
if rpt_metrics:
    rpt_data = rpt_metrics.get('metrics', {})
    wns = rpt_data.get('wns_max')
    tns = rpt_data.get('tns_max')
    fmax = rpt_data.get('fmax')
    # Answer: {'wns': f"{wns} ns", 'tns': f"{tns} ns", 'fmax': f"{fmax} MHz"}
```
"""

FINAL_RPT_METRICS_EXAMPLES="""
## Final Stage RPT Metrics Parsing Examples

The final stage contains comprehensive report metrics in the 'rpt_metrics' field with detailed timing, power, and path analysis.

### Example 1: Extracting basic timing metrics from rpt_metrics
Question: What is the worst negative slack (WNS) in the final stage?
```python
final_stage = next((s for s in stages if s['stage'] == 'final'), None)
if final_stage and 'rpt_metrics' in final_stage['metrics']:
    rpt = final_stage['metrics']['rpt_metrics']
    wns = rpt['metrics'].get('wns_max')
    # Answer: -0.01
```

### Example 2: Getting maximum frequency (fmax)
Question: What is the maximum frequency achieved in the final stage?
```python
final_stage = next((s for s in stages if s['stage'] == 'final'), None)
if final_stage and 'rpt_metrics' in final_stage['metrics']:
    rpt = final_stage['metrics']['rpt_metrics']
    fmax = rpt['metrics'].get('fmax')
    # Answer: 1210.14 MHz
```

### Example 3: Counting timing violations
Question: How many setup and hold violations are there in the final stage?
```python
final_stage = next((s for s in stages if s['stage'] == 'final'), None)
if final_stage and 'rpt_metrics' in final_stage['metrics']:
    rpt = final_stage['metrics']['rpt_metrics']
    setup_violations = rpt['metrics'].get('setup_violation_count')
    hold_violations = rpt['metrics'].get('hold_violation_count')
    # Answer: 4 setup violations, 0 hold violations (only if fields exist)
```

### Example 4: Extracting critical path information
Question: What is the critical path delay and slack in the final stage?
```python
final_stage = next((s for s in stages if s['stage'] == 'final'), None)
if final_stage and 'rpt_metrics' in final_stage['metrics']:
    rpt = final_stage['metrics']['rpt_metrics']
    delay = rpt['metrics'].get('critical_path_delay')
    slack = rpt['metrics'].get('critical_path_slack')
    # Answer: delay=0.9765ns, slack=-0.0063ns
```

### Example 5: Getting worst timing path details
Question: What are the startpoint and endpoint of the worst setup path?
```python
final_stage = next((s for s in stages if s['stage'] == 'final'), None)
if final_stage and 'rpt_metrics' in final_stage['metrics']:
    rpt = final_stage['metrics']['rpt_metrics']
    max_paths = rpt.get('max_paths', [])
    if max_paths:
        worst_path = max_paths[0]
        startpoint = worst_path.get('startpoint')
        endpoint = worst_path.get('endpoint')
        slack = worst_path.get('slack')
        # Answer: startpoint="sa00_sr[1]$_DFF_P_", endpoint="sa31_sr[5]$_DFF_P_", slack=-0.01
```

### Example 6: Extracting power consumption breakdown
Question: What is the total power consumption and breakdown by component type?
```python
final_stage = next((s for s in stages if s['stage'] == 'final'), None)
if final_stage and 'rpt_metrics' in final_stage['metrics']:
    rpt = final_stage['metrics']['rpt_metrics']
    power = rpt.get('power', {})
    total = power.get('total', {}).get('total')
    sequential = power.get('sequential', {}).get('total')
    combinational = power.get('combinational', {}).get('total')
    clock = power.get('clock', {}).get('total')
    # Answer: total=0.303W, sequential=0.0126W (4.2%), combinational=0.286W (94.6%), clock=0.00364W (1.2%)
```

### Example 7: Getting clock skew information
Question: What is the setup clock skew in the final stage?
```python
final_stage = next((s for s in stages if s['stage'] == 'final'), None)
if final_stage and 'rpt_metrics' in final_stage['metrics']:
    rpt = final_stage['metrics']['rpt_metrics']
    clock_skew = rpt.get('clock_skew', {})
    setup_skew = clock_skew.get('setup_skew')
    clock_name = clock_skew.get('clock_name')
    # Answer: clock=clk, setup_skew=0.04ns
```

### Example 8: Extracting IR drop analysis
Question: What is the worst-case IR drop for VDD and VSS?
```python
final_stage = next((s for s in stages if s['stage'] == 'final'), None)
if final_stage:
    ir_drop = final_stage['metrics'].get('ir_drop', {})
    vdd_worst = ir_drop.get('vdd', {}).get('worstcase_ir_drop_v')
    vss_worst = ir_drop.get('vss', {}).get('worstcase_ir_drop_v')
    vdd_percent = ir_drop.get('vdd', {}).get('percentage_drop')
    # Answer: VDD=0.0691V (6.29%), VSS=0.0664V (6.04%)
```

### Example 9: Getting cell count statistics
Question: How many fill cells and sequential cells are in the final design?
```python
final_stage = next((s for s in stages if s['stage'] == 'final'), None)
if final_stage:
    cell_counts = final_stage['metrics'].get('cell_counts', {})
    fill_cells = cell_counts.get('fill_cell_count')
    sequential = cell_counts.get('sequential_cell_count')
    total_cells = cell_counts.get('total_cell_count')
    # Answer: 17718 fill cells, 562 sequential cells, 33917 total cells
```

### Example 10: Extracting design metrics (WNS/TNS)
Question: What are the final setup and hold WNS/TNS values?
```python
final_stage = next((s for s in stages if s['stage'] == 'final'), None)
if final_stage:
    design_metrics = final_stage['metrics'].get('design_metrics', {})
    setup_wns = design_metrics.get('setup_wns')
    setup_tns = design_metrics.get('setup_tns')
    hold_wns = design_metrics.get('hold_wns')
    hold_tns = design_metrics.get('hold_tns')
    # Answer: setup_wns=-0.00635ns, setup_tns=-0.01828ns, hold_wns=0.00921ns, hold_tns=0.0ns
```

### Example 11: Getting test metrics pass/fail status
Question: Did the finish__timing__setup__ws metric pass its threshold?
```python
final_stage = next((s for s in stages if s['stage'] == 'final'), None)
if final_stage:
    test_metrics = final_stage['metrics'].get('test_metrics', {})
    metric = test_metrics.get('finish__timing__setup__ws', {})
    actual = metric.get('actual_value')
    threshold = metric.get('threshold_value')
    passed = metric.get('passed')
    operator = metric.get('operator')
    # Answer: actual=-0.00635, threshold=-0.0473, operator='>=', passed=True
```

### Example 12: Extracting flow summary timing
Question: How much time did the routing stage take?
```python
final_stage = next((s for s in stages if s['stage'] == 'final'), None)
if final_stage:
    flow_summary = final_stage['metrics'].get('flow_summary', {})
    route_step = flow_summary.get('5_2_route', {})
    elapsed_s = route_step.get('elapsed_s')
    memory_mb = route_step.get('memory_mb')
    # Answer: elapsed=103s, memory=2715MB
```

### Example 13: Getting hold path details
Question: What is the startpoint and slack of the best hold path?
```python
final_stage = next((s for s in stages if s['stage'] == 'final'), None)
if final_stage and 'rpt_metrics' in final_stage['metrics']:
    rpt = final_stage['metrics']['rpt_metrics']
    min_paths = rpt.get('min_paths', [])
    if min_paths:
        best_path = min_paths[0]
        startpoint = best_path.get('startpoint')
        endpoint = best_path.get('endpoint')
        slack = best_path.get('slack')
        # Answer: startpoint="text_in[127]", endpoint="text_in_r[127]$_DFFE_PP_", slack=0.01ns
```

### Example 14: Extracting power by switching/internal/leakage
Question: What is the internal vs switching power for combinational logic?
```python
final_stage = next((s for s in stages if s['stage'] == 'final'), None)
if final_stage and 'rpt_metrics' in final_stage['metrics']:
    rpt = final_stage['metrics']['rpt_metrics']
    comb = rpt.get('power', {}).get('combinational', {})
    internal = comb.get('internal')
    switching = comb.get('switching')
    leakage = comb.get('leakage')
    # Answer: internal=0.128W, switching=0.158W, leakage=0.000456W
```

### Example 15: Getting max capacitance/slew checks
Question: What is the slack on max slew and max capacitance checks?
```python
final_stage = next((s for s in stages if s['stage'] == 'final'), None)
if final_stage and 'rpt_metrics' in final_stage['metrics']:
    rpt = final_stage['metrics']['rpt_metrics']
    max_slew_slack = rpt['metrics'].get('max_slew_check_slack')
    max_cap_slack = rpt['metrics'].get('max_capacitance_check_slack')
    max_slew_limit = rpt['metrics'].get('max_slew_check_limit')
    # Answer: max_slew_slack=0.0813ns, max_cap_slack=2.017fF, max_slew_limit=0.1985ns
```

### Example 16: Comparing actual vs threshold for all test metrics
Question: Which test metrics failed their thresholds?
```python
final_stage = next((s for s in stages if s['stage'] == 'final'), None)
failed_metrics = []
if final_stage:
    test_metrics = final_stage['metrics'].get('test_metrics', {})
    for metric_name, data in test_metrics.items():
        if not data.get('passed', True):
            failed_metrics.append({
                'name': metric_name,
                'actual': data.get('actual_value'),
                'threshold': data.get('threshold_value')
            })
    # Answer: [] (all metrics passed in this example)
```

### Example 17: Getting cell area breakdown
Question: What is the total area occupied by different cell types?
```python
final_stage = next((s for s in stages if s['stage'] == 'final'), None)
if final_stage:
    cell_counts = final_stage['metrics'].get('cell_counts', {})
    fill_area = cell_counts.get('fill_cell_area')
    clock_buf_area = cell_counts.get('clock_buffer_area')
    sequential_area = cell_counts.get('sequential_cell_area')
    combinational_area = cell_counts.get('combinational_cell_area')
    total_area = cell_counts.get('total_cell_area')
    # Answer: fill=32255.43, clock_buf=157.47, seq=2541.9, comb=15016.76, total=52785.04
```

### Example 18: Extracting minimum clock period
Question: What is the minimum clock period and corresponding frequency?
```python
final_stage = next((s for s in stages if s['stage'] == 'final'), None)
if final_stage and 'rpt_metrics' in final_stage['metrics']:
    rpt = final_stage['metrics']['rpt_metrics']
    clock_period_min = rpt['metrics'].get('clock_period_min')
    fmax = rpt['metrics'].get('fmax')
    # Answer: clock_period_min=0.83ns, fmax=1210.14MHz
```
"""

# Generic examples (for all stages)
GENERIC_EXAMPLES = """
## Generic Parsing Patterns (applicable to all stages)

### Finding metrics across substages:
```python
substages = stage_entry['metrics'].get('substages', {})
for substage_name, substage_data in substages.items():
    wns = substage_data.get('wns')
    tns = substage_data.get('tns')
    area = substage_data.get('area')
    # Use first non-None value found
```

### Checking design_metrics_final:
```python
design_metrics_final = stage_entry['metrics'].get('design_metrics_final', {})
if design_metrics_final:
    # Stage has design_metrics_final (placement, CTS)
    for key, value in design_metrics_final.items():
        # Process metrics
else:
    # Check substage-level (routing)
    for substage_data in substages.values():
        substage_metrics = substage_data.get('design_metrics_final', {})
```

### Iterating through all optimizations:
```python
all_optimizations = []
for substage_name, substage_data in substages.items():
    for opt in substage_data.get('optimization_process', []):
        opt_type = opt.get('type')
        all_optimizations.append({
            'substage': substage_name,
            'type': opt_type,
            'data': opt
        })
# Answer: all_optimizations
```
"""

GENERIC_RPT_METRICS_EXAMPLES = """
## Generic Parsing Patterns for report metrics (rpt_metrics) (applicable to all substages that have rpt_metrics)
========== CRITICAL NOTES FOR FINAL STAGE =============
**IMPORTANT**: The final stage has a DIFFERENT structure than other stages:
- Final stage: NO substages, metrics are at TOP LEVEL under stage_entry['metrics']
- Final stage: Uses 'design_metrics' (NOT 'design_metrics_final')
- Final stage: rpt_metrics is at stage_entry['metrics']['rpt_metrics']
- Other stages: rpt_metrics is at stage_entry['metrics']['substages'][substage_name]['rpt_metrics'] (e.g. 3_3_place_gp, 3_4_place_resized, 3_5_place_dp for substage name)

ALWAYS check if stage_entry['stage'] == 'final' or stage_entry['stage_type'] == 'final' and handle accordingly!
========== Hints =============
If the above example code can not find the metrics, you can try to use the rpt_metrics to find the metrics.
```python
rpt_metrics = substage_data.get('rpt_metrics', {})
if rpt_metrics:
    metrics = rpt_metrics.get('metrics', {})
    for metric_name, metric_data in metrics.items():
        actual = metric_data.get('actual_value')
        threshold = metric_data.get('threshold_value')
        passed = metric_data.get('passed')
        # Answer: dict of metrics with pass/fail status
```


### Generic rpt_metrics parsing (basic timing metrics):
```python
def get_basic_timing_metrics(rpt_metrics):
    metrics = rpt_metrics.get('metrics', {})
        
    # Basic timing metrics
    tns_max = metrics.get('tns_max')
    wns_max = metrics.get('wns_max')
    worst_slack_max = metrics.get('worst_slack_max')
    clock_period_min = metrics.get('clock_period_min')
    fmax = metrics.get('fmax')
        
    # Violation counts
    setup_violations = metrics.get('setup_violation_count')
    hold_violations = metrics.get('hold_violation_count')
    max_slew_violations = metrics.get('max_slew_violation_count')
    max_fanout_violations = metrics.get('max_fanout_violation_count')
    max_cap_violations = metrics.get('max_cap_violation_count')
        
    # Critical path metrics
    critical_path_delay = metrics.get('critical_path_delay')
    critical_path_slack = metrics.get('critical_path_slack')
        
    # Answer: {'tns': tns_max, 'wns': wns_max, 'fmax': fmax, 
    #          'setup_viols': setup_violations, 'hold_viols': hold_violations}

# Main code
if stage_entry['stage'] == 'final':
    rpt_metrics = stage_entry['metrics'].get('rpt_metrics', {})
    answer = get_basic_timing_metrics(rpt_metrics)
else:
    substages = stage_entry['metrics'].get('substages', {})
    for substage_name, substage_data in substages.items():
        rpt_metrics = substage_data.get('rpt_metrics', {}) 
        if rpt_metrics:
            answer = get_basic_timing_metrics(rpt_metrics)
```

### Generic max_path (setup) parsing (handles both final and non-final stages):
```python
# For final stage: rpt_metrics is at top level
# For other stages: rpt_metrics is inside substages
if stage_entry['stage'] == 'final':
    rpt_metrics = stage_entry['metrics'].get('rpt_metrics', {})
    if rpt_metrics:
        max_path = rpt_metrics.get('max_path', {})
        if max_path:
            startpoint = max_path.get('startpoint')
            endpoint = max_path.get('endpoint')
            slack = max_path.get('slack')
            data_arrival = max_path.get('data_arrival_time')
            data_required = max_path.get('data_required_time')
            # Answer: {'endpoint': endpoint, 'slack': slack, 'arrival': data_arrival}
else:
    substages = stage_entry['metrics'].get('substages', {})
    for substage_name, substage_data in substages.items():
        rpt_metrics = substage_data.get('rpt_metrics', {})
        if rpt_metrics:
            max_path = rpt_metrics.get('max_path', {})
            if max_path:
                startpoint = max_path.get('startpoint')
                endpoint = max_path.get('endpoint')
                slack = max_path.get('slack')
                data_arrival = max_path.get('data_arrival_time')
                data_required = max_path.get('data_required_time')
                
                # Answer: {'endpoint': endpoint, 'slack': slack, 'arrival': data_arrival}
```

### Generic clock_skew parsing (handles both final and non-final stages):
```python
# For final stage: rpt_metrics is at top level
# For other stages: rpt_metrics is inside substages
if stage_entry['stage'] == 'final':
    rpt_metrics = stage_entry['metrics'].get('rpt_metrics', {})
    if rpt_metrics:
        clock_skew = rpt_metrics.get('clock_skew', {})
        if clock_skew:
            setup_skew = clock_skew.get('setup_skew')
            source_latency = clock_skew.get('source_latency')
            target_latency = clock_skew.get('target_latency')
            # Answer: {'setup_skew': setup_skew}
else:
    substages = stage_entry['metrics'].get('substages', {})
    for substage_name, substage_data in substages.items():
        rpt_metrics = substage_data.get('rpt_metrics', {})
        if rpt_metrics:
            clock_skew = rpt_metrics.get('clock_skew', {})
            if clock_skew:
                setup_skew = clock_skew.get('setup_skew')
                source_latency = clock_skew.get('source_latency')
                target_latency = clock_skew.get('target_latency')
                
                # Answer: {'setup_skew': setup_skew}
```

### Generic power breakdown parsing (handles both final and non-final stages):
```python
# For final stage: rpt_metrics is at top level
# For other stages: rpt_metrics is inside substages
if stage_entry['stage'] == 'final':
    rpt_metrics = stage_entry['metrics'].get('rpt_metrics', {})
    if rpt_metrics:
        power = rpt_metrics.get('power', {})
        if power:
            total_power = power.get('total', {}).get('total')
            clock_power = power.get('clock', {}).get('total')
            clock_percent = power.get('clock', {}).get('percent')
            sequential_power = power.get('sequential', {}).get('total')
            combinational_power = power.get('combinational', {}).get('total')
            # Answer: {'total': total_power, 'clock_percent': clock_percent}
else:
    substages = stage_entry['metrics'].get('substages', {})
    for substage_name, substage_data in substages.items():
        rpt_metrics = substage_data.get('rpt_metrics', {})
        if rpt_metrics:
            power = rpt_metrics.get('power', {})
            if power:
                total_power = power.get('total', {}).get('total')
                clock_power = power.get('clock', {}).get('total')
                clock_percent = power.get('clock', {}).get('percent')
                sequential_power = power.get('sequential', {}).get('total')
                combinational_power = power.get('combinational', {}).get('total')
                
                # Answer: {'total': total_power, 'clock_percent': clock_percent}
```

### Parsing max paths (setup) from rpt_metrics across substages for stages not final stage:
```python
substages = stage_entry['metrics'].get('substages', {})
max_paths = []

for substage_name, substage_data in substages.items():
    rpt_metrics = substage_data.get('rpt_metrics', {})
    if rpt_metrics:
        max_path = rpt_metrics.get('max_path', {})
        if max_path:
            max_paths.append({
                'substage': substage_name,
                'startpoint': max_path.get('startpoint'),
                'endpoint': max_path.get('endpoint'),
                'slack': max_path.get('slack'),
                'path_type': max_path.get('path_type'),  # 'max' for setup
                'analysis_type': max_path.get('analysis_type'),  # 'setup'
                'data_arrival_time': max_path.get('data_arrival_time'),
                'data_required_time': max_path.get('data_required_time')
            })

# Answer: max_paths (list of setup critical paths from all substages)
```

### Parsing min path (hold) from rpt_metrics across substages for stages not final stage:
```python
substages = stage_entry['metrics'].get('substages', {})
min_paths = []

for substage_name, substage_data in substages.items():
    rpt_metrics = substage_data.get('rpt_metrics', {})
    if rpt_metrics:
        min_path = rpt_metrics.get('min_path', {})
        if min_path:
            min_paths.append({
                'substage': substage_name,
                'startpoint': min_path.get('startpoint'),
                'endpoint': min_path.get('endpoint'),
                'slack': min_path.get('slack'),
                'path_type': min_path.get('path_type'),  # 'min' for hold
                'analysis_type': min_path.get('analysis_type'),  # 'hold'
                'data_arrival_time': min_path.get('data_arrival_time'),
                'data_required_time': min_path.get('data_required_time')
            })

# Answer: min_paths (list of hold critical paths from all substages)
```

### Comparing setup slack across substages for stages not final stage:
```python
substages = stage_entry['metrics'].get('substages', {})

# Extract max path info for each substage
slack_progression = []
for substage_name, substage_data in substages.items():
    rpt_metrics = substage_data.get('rpt_metrics', {})
    if rpt_metrics:
        max_path = rpt_metrics.get('max_path', {})
        if max_path:
            slack = max_path.get('slack')
            slack_progression.append({
                'substage': substage_name,
                'setup_slack': slack,
                'startpoint': max_path.get('startpoint'),
                'endpoint': max_path.get('endpoint')
            })

# Sort by substage numeric prefix order
def get_substage_order(item):
    parts = item['substage'].split('_')
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        return (int(parts[0]), int(parts[1]))
    return (999, 999)

slack_progression.sort(key=get_substage_order)
worst_slack = min(slack_progression, key=lambda x: x['setup_slack']) if slack_progression else None
# Answer: {'progression': slack_progression, 'worst': worst_slack}
```

### Comparing hold slack across substages for stages not final stage:
```python
substages = stage_entry['metrics'].get('substages', {})
hold_slack_progression = []

for substage_name, substage_data in substages.items():
    rpt_metrics = substage_data.get('rpt_metrics', {})
    if rpt_metrics:
        min_path = rpt_metrics.get('min_path', {})
        if min_path:
            slack = min_path.get('slack')
            hold_slack_progression.append({
                'substage': substage_name,
                'hold_slack': slack,
                'startpoint': min_path.get('startpoint'),
                'endpoint': min_path.get('endpoint')
            })

# Find minimum hold slack (most critical)
worst_hold = min(hold_slack_progression, key=lambda x: x['hold_slack']) if hold_slack_progression else None
# Answer: {'progression': hold_slack_progression, 'worst': worst_hold}
```

### Extracting both max and min paths for timing analysis for stages not final stage:
```python
substages = stage_entry['metrics'].get('substages', {})
timing_paths = {}

for substage_name, substage_data in substages.items():
    rpt_metrics = substage_data.get('rpt_metrics', {})
    if rpt_metrics:
        max_paths = rpt_metrics.get('max_paths', {})
        min_paths = rpt_metrics.get('min_paths', {})
        
        if max_paths:
            for max_path in max_paths:
                timing_paths[substage_name] = {
                    'setup': {
                        'slack': max_path.get('slack'),
                        'startpoint': max_path.get('startpoint'),
                        'endpoint': max_path.get('endpoint'),
                    }
                }
        if min_paths:
            for min_path in min_paths:
                timing_paths[substage_name] = {
                    'hold': {
                        'slack': min_path.get('slack'),
                        'startpoint': min_path.get('startpoint'),
                        'endpoint': min_path.get('endpoint'),
                    }
                }

# Answer: timing_paths (dict with both setup and hold paths per substage)
```

## Multi-Stage Progression Examples (Including Final Stage)
========== CRITICAL NOTES =============
When analyzing stage-to-final progressions, ALWAYS remember:
1. Final stage has NO substages - metrics are at the TOP LEVEL
2. Final stage uses 'design_metrics' not 'design_metrics_final'
3. Final stage rpt_metrics is at stage_entry['metrics']['rpt_metrics']
4. Other stages have rpt_metrics at stage_entry['metrics']['substages'][substage_name]['rpt_metrics']

### Multi-stage timing progression (including final stage):
```python
# Extract timing metrics across ALL stages including final
timing_progression = {}

for stage_entry in progression_data['progression']:
    stage_name = stage_entry['stage_type']
    
    if stage_name == 'final':
        # Final stage: metrics at top level
        design_metrics = stage_entry['metrics'].get('design_metrics', {})
        if design_metrics:
            wns = design_metrics.get('setup_wns')
            tns = design_metrics.get('setup_tns')
            if wns is not None and tns is not None:
                timing_progression[stage_name] = {
                    'wns': f"{wns} ns",
                    'tns': f"{tns} ns"
                }
    else:
        # Other stages: metrics in substages with rpt_metrics
        substages = stage_entry['metrics'].get('substages', {})
        for substage_name, substage_data in substages.items():
            rpt_metrics = substage_data.get('rpt_metrics', {})
            if rpt_metrics:
                metrics = rpt_metrics.get('metrics', {})
                wns = metrics.get('wns_max')
                tns = metrics.get('tns_max')
                if wns is not None and tns is not None:
                    timing_progression[stage_name] = {
                        'wns': f"{wns} ns",
                        'tns': f"{tns} ns"
                    }
                    break  # Take first substage with rpt_metrics

# Answer: timing_progression (dict with stage_name -> timing metrics)
```

### Multi-stage area progression (including final stage):
```python
# Extract area metrics across ALL stages including final
area_progression = {}

for stage_entry in progression_data['progression']:
    stage_name = stage_entry['stage_type']
    
    if stage_name == 'final':
        # Final stage: area in design_metrics
        design_metrics = stage_entry['metrics'].get('design_metrics', {})
        area = design_metrics.get('area')
        if area is not None:
            area_progression[stage_name] = f"{area} um^2"
    else:
        # Other stages: area in design_metrics_final or substages
        design_metrics_final = stage_entry['metrics'].get('design_metrics_final', {})
        if design_metrics_final:
            area = design_metrics_final.get('area')
            if area is not None:
                area_progression[stage_name] = f"{area} um^2"

# Answer: area_progression (dict with stage_name -> area)
```

### Multi-stage buffer insertion tracking (including final stage):
```python
# Track buffer insertions across ALL stages including final
buffer_tracking = {}

for stage_entry in progression_data['progression']:
    stage_name = stage_entry['stage_type']
    
    if stage_name == 'final':
        # Final stage: check cell_counts for buffer counts
        cell_counts = stage_entry['metrics'].get('cell_counts', {})
        if 'timing_repair_buffer_count' in cell_counts:
            buffer_tracking[stage_name] = {
                'timing_repair_buffers': cell_counts['timing_repair_buffer_count']
            }
    else:
        # Other stages: check optimization_process in substages
        substages = stage_entry['metrics'].get('substages', {})
        total_inserted = 0
        total_removed = 0
        
        for substage_data in substages.values():
            for opt in substage_data.get('optimization_process', []):
                if opt.get('type') == 'buffer_insertion':
                    results = opt.get('results', {})
                    if 'input_buffers' in results:
                        total_inserted += results['input_buffers']
                    if 'output_buffers' in results:
                        total_inserted += results['output_buffers']
                elif opt.get('type') == 'buffer_removal':
                    results = opt.get('results', {})
                    if 'removed_buffers' in results:
                        total_removed += results['removed_buffers']
        
        buffer_tracking[stage_name] = {
            'inserted': total_inserted,
            'removed': total_removed
        }

# Answer: buffer_tracking (dict with stage_name -> buffer operations)
```

### Multi-stage violation tracking (including final stage):
```python
# Track violations across ALL stages including final
violation_progression = {}

for stage_entry in progression_data['progression']:
    stage_name = stage_entry['stage_type']
    
    if stage_name == 'final':
        # Final stage: check rpt_metrics at top level
        rpt_metrics = stage_entry['metrics'].get('rpt_metrics', {})
        if rpt_metrics:
            metrics = rpt_metrics.get('metrics', {})
            violations = {}
            if 'setup_violation_count' in metrics:
                violations['setup_violations'] = metrics['setup_violation_count']
            if 'hold_violation_count' in metrics:
                violations['hold_violations'] = metrics['hold_violation_count']
            if violations:
                violation_progression[stage_name] = violations
    else:
        # Other stages: check rpt_metrics in substages
        substages = stage_entry['metrics'].get('substages', {})
        for substage_name, substage_data in substages.items():
            rpt_metrics = substage_data.get('rpt_metrics', {})
            if rpt_metrics:
                metrics = rpt_metrics.get('metrics', {})
                violations = {}
                if 'setup_violation_count' in metrics:
                    violations['setup_violations'] = metrics['setup_violation_count']
                if 'hold_violation_count' in metrics:
                    violations['hold_violations'] = metrics['hold_violation_count']
                if violations:
                    violation_progression[stage_name] = violations
                    break  # Take first substage with rpt_metrics

# Answer: violation_progression (dict with stage_name -> violations)
```

### Multi-stage power progression (including final stage):
```python
# Track power metrics across ALL stages including final
power_progression = {}

for stage_entry in progression_data['progression']:
    stage_name = stage_entry['stage_type']
    
    if stage_name == 'final':
        # Final stage: check rpt_metrics for power
        rpt_metrics = stage_entry['metrics'].get('rpt_metrics', {})
        if rpt_metrics:
            power = rpt_metrics.get('power', {})
            if power:
                power_data = {}
                if 'total' in power.get('sequential', {}):
                    power_data['sequential'] = f"{power['sequential']['total']} W"
                if 'total' in power.get('combinational', {}):
                    power_data['combinational'] = f"{power['combinational']['total']} W"
                if 'total' in power.get('clock', {}):
                    power_data['clock'] = f"{power['clock']['total']} W"
                if 'total' in power.get('total', {}):
                    power_data['total'] = f"{power['total']['total']} W"
                if power_data:
                    power_progression[stage_name] = power_data
    else:
        # Other stages: check rpt_metrics in substages
        substages = stage_entry['metrics'].get('substages', {})
        for substage_name, substage_data in substages.items():
            rpt_metrics = substage_data.get('rpt_metrics', {})
            if rpt_metrics:
                power = rpt_metrics.get('power', {})
                if power:
                    power_data = {}
                    if 'total' in power.get('sequential', {}):
                        power_data['sequential'] = f"{power['sequential']['total']} W"
                    if 'total' in power.get('combinational', {}):
                        power_data['combinational'] = f"{power['combinational']['total']} W"
                    if 'total' in power.get('clock', {}):
                        power_data['clock'] = f"{power['clock']['total']} W"
                    if 'total' in power.get('total', {}):
                        power_data['total'] = f"{power['total']['total']} W"
                    if power_data:
                        power_progression[stage_name] = power_data
                    break  # Take first substage with rpt_metrics

# Answer: power_progression (dict with stage_name -> power metrics)
```

### Checking if critical path changed for multiple stage comparison:
For multiple stage comparison, the stage_entry is a list of stage dictionaries, each with 'stage', 'occurrence', 'sub_design_name', and 'metrics'.
```python
# Step 1: Extract max path for each substage across all stages
critical_paths_by_substage = []

for stage_dict in stage_entry:
    stage_name = stage_dict['stage']
    if stage_name == 'final':
        rpt_metrics = stage_dict['metrics'].get('rpt_metrics', {})
        if rpt_metrics:
            max_paths = rpt_metrics.get('max_paths', {})
            if max_paths:
                for max_path in max_paths:
                    critical_paths_by_substage.append({
                        'stage': stage_name,
                        'substage': 'final',
                        'startpoint': max_path.get('startpoint'),
                        'endpoint': max_path.get('endpoint'),
                        'slack': max_path.get('slack')
                    })
    else:
        substages = stage_dict['metrics'].get('substages', {})
        
        for substage_name, substage_data in substages.items():
            rpt_metrics = substage_data.get('rpt_metrics', {})
            if rpt_metrics:
                max_paths = rpt_metrics.get('max_paths', {})
                if max_paths:
                    for max_path in max_paths:
                        critical_paths_by_substage.append({
                        'stage': stage_name,
                        'substage': substage_name,
                        'startpoint': max_path.get('startpoint')
                        'endpoint': max_path.get('endpoint'),
                        'slack': max_path.get('slack')
                    })
                })

# Step 2: Sort by numeric prefix order
def get_substage_order(item):
    parts = item['substage'].split('_')
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        return (int(parts[0]), int(parts[1]))
    return (999, 999)

critical_paths_by_substage.sort(key=get_substage_order)

# Step 3: Compare consecutive substages to find path changes
path_changes = []
for i in range(len(critical_paths_by_substage) - 1):
    curr = critical_paths_by_substage[i]
    next_item = critical_paths_by_substage[i + 1]
    
    # Check if either startpoint or endpoint changed
    if (curr['startpoint'] != next_item['startpoint'] or 
        curr['endpoint'] != next_item['endpoint']):
        path_changes.append({
            'from_stage': curr['stage'],
            'from_substage': curr['substage'],
            'to_stage': next_item['stage'],
            'to_substage': next_item['substage'],
            'from_startpoint': curr['startpoint'],
            'to_startpoint': next_item['startpoint'],
            'from_endpoint': curr['endpoint'],
            'to_endpoint': next_item['endpoint'],
            'from_slack': curr['slack'],
            'to_slack': next_item['slack']
        })

# Answer: path_changes (list of critical path changes between substages)
```
"""


def get_parsing_examples(stage: str, include_rpt_metrics: bool = False) -> str:
    """
    Get stage-specific parsing examples for prompts.
    
    Args:
        stage: Stage name (synthesis, floorplan, placement, cts, routing, final)
        include_rpt_metrics: If True, include generic rpt_metrics parsing examples
    
    Returns:
        String containing relevant parsing examples for the stage
    """
    stage_lower = stage.lower() if stage else ""
    
    examples = []
    
    # Add stage-specific examples
    if "synthesis" in stage_lower:
        examples.append(SYNTHESIS_EXAMPLES)
    elif "floorplan" in stage_lower:
        examples.append(FLOORPLAN_EXAMPLES)
    elif "placement" in stage_lower or "place" in stage_lower:
        examples.append(PLACEMENT_EXAMPLES)
    elif "cts" in stage_lower or "clock" in stage_lower:
        examples.append(CTS_EXAMPLES)
    elif "routing" in stage_lower or "route" in stage_lower:
        examples.append(ROUTING_EXAMPLES)
    elif "final" in stage_lower:
        examples.append(FINAL_EXAMPLES)
        # if include_rpt_metrics:
        #     examples.append(FINAL_RPT_METRICS_EXAMPLES)
    
    # Always add generic examples
    examples.append(GENERIC_EXAMPLES)
    
    # Optionally add rpt_metrics examples
    if include_rpt_metrics:
        examples.append(GENERIC_RPT_METRICS_EXAMPLES)
    
    return "\n".join(examples)


def get_multi_stage_parsing_examples(stages: list, include_rpt_metrics: bool = False) -> str:
    """
    Get parsing examples for multiple stages.
    
    Args:
        stages: List of stage dictionaries with 'stage' field
        include_rpt_metrics: If True, include generic rpt_metrics parsing examples
    
    Returns:
        String containing relevant parsing examples for all stages
    """
    stage_names = [s.get('stage', '') for s in stages]
    unique_stages = set(stage_names)
    
    examples = []
    
    # Add stage-specific examples (without generic examples)
    for stage in unique_stages:
        stage_lower = stage.lower() if stage else ""
        
        # Add stage-specific examples only (not generic)
        if "synthesis" in stage_lower:
            examples.append(SYNTHESIS_EXAMPLES)
        elif "floorplan" in stage_lower:
            examples.append(FLOORPLAN_EXAMPLES)
        elif "placement" in stage_lower or "place" in stage_lower:
            examples.append(PLACEMENT_EXAMPLES)
        elif "cts" in stage_lower or "clock" in stage_lower:
            examples.append(CTS_EXAMPLES)
        elif "routing" in stage_lower or "route" in stage_lower:
            examples.append(ROUTING_EXAMPLES)
        elif "final" in stage_lower:
            examples.append(FINAL_EXAMPLES)
            # if include_rpt_metrics:
            #    examples.append(FINAL_RPT_METRICS_EXAMPLES)
    
    examples_str = "\n".join(examples)
    
    # Add multi-stage specific patterns
    multi_stage_pattern = """
## Multi-Stage Comparison Patterns

### Comparing metrics across stages:
```python
stage_metrics = []
for stage in stages:
    stage_name = stage['stage']
    metrics = stage['metrics']
    substages = metrics.get('substages', {})
    
    # Extract relevant metric
    for substage_data in substages.values():
        metric_value = substage_data.get('wns')  # or other metric
        if metric_value is not None:
            stage_metrics.append({
                'stage': stage_name,
                'value': metric_value
            })
# Answer: stage_metrics
```

### Tracking optimization progression:
```python
optimizations_by_stage = {}
for stage in stages:
    stage_name = stage['stage']
    substages = stage['metrics'].get('substages', {})
    
    opt_count = 0
    for substage_data in substages.values():
        opt_count += len(substage_data.get('optimization_process', []))
    
    optimizations_by_stage[stage_name] = opt_count
# Answer: optimizations_by_stage
```
"""
    
    # Add generic examples once at the end
    result = examples_str + multi_stage_pattern + "\n" + GENERIC_EXAMPLES
    
    # Optionally add rpt_metrics examples
    if include_rpt_metrics:
        result += "\n" + GENERIC_RPT_METRICS_EXAMPLES
    
    return result


# Example usage
if __name__ == '__main__':
    print("=== Placement Stage Examples ===")
    print(get_parsing_examples("placement"))
    print("\n=== CTS Stage Examples ===")
    print(get_parsing_examples("cts"))
    print("\n=== Multi-Stage Examples ===")
    print(get_multi_stage_parsing_examples([
        {'stage': 'placement'},
        {'stage': 'cts'},
        {'stage': 'routing'}
    ]))
