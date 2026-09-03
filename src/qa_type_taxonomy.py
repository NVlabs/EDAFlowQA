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
HotpotQA-inspired taxonomy for EDA QA.

HotpotQA popularized categorizing questions by *reasoning type* (e.g., bridge vs comparison)
and answers by coarse *answer type* (e.g., yes/no, number, span).

This script keeps that spirit but adapts the labels to this repo’s EDA QA:
- Reasoning type (Hotpot-style): single | bridge | comparison
- EDA semantic question type: timing_metrics, optimization_command, routing_metrics, ...
- EDA semantic answer type: yes_no, number, number_with_unit, command, structured, ...

It reads a single `*_qa_all_types.json` file (or a directory) and prints distributions and
simple cross-tabs. It is intentionally rule-based (fast, reproducible, easy to tweak).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


_DIFFICULTY_RE = re.compile(r"Difficulty:\s*([a-zA-Z]+)", re.IGNORECASE)


def _norm(s: Any) -> str:
    if s is None:
        return ""
    return str(s).strip()


def extract_difficulty(question_text: str) -> str:
    m = _DIFFICULTY_RE.search(question_text or "")
    if not m:
        return "unknown"
    d = m.group(1).strip().lower().rstrip(".,;:")
    if d in {"easy", "medium", "hard"}:
        return d
    if any(k in d for k in ["easy", "simple"]):
        return "easy"
    if any(k in d for k in ["medium", "moderate", "intermediate"]):
        return "medium"
    if any(k in d for k in ["hard", "difficult", "complex"]):
        return "hard"
    return "unknown"


def answer_shape(a: Any) -> str:
    if a is None:
        return "null"
    if isinstance(a, bool):
        return "boolean"
    if isinstance(a, (int, float)) and not isinstance(a, bool):
        return "number"
    if isinstance(a, str):
        s = a.strip()
        if s == "":
            return "string_empty"
        return "string"
    if isinstance(a, list):
        return "array"
    if isinstance(a, dict):
        return "object"
    return type(a).__name__


_UNIT_RE = re.compile(r"\b(kb|mb|gb|um|ns|ps|s|min|sec|%|percent)\b", re.IGNORECASE)
_COMMAND_HINT_RE = re.compile(
    r"\b("
    r"repair_[a-z_]+|clock_tree_synthesis|global_placement|detailed_placement|"
    r"global_routing|detailed_routing|"
    r"rtl_macro_placer|tapcell|pdn|route|cts|place|synth"
    r")\b",
    re.IGNORECASE,
)
_CLI_FLAG_RE = re.compile(r"(^|\s)-{1,2}[a-zA-Z_][\w-]*")


def _textify(obj: Any) -> str:
    if obj is None:
        return ""
    if isinstance(obj, (dict, list)):
        try:
            return json.dumps(obj, ensure_ascii=False)
        except Exception:
            return str(obj)
    return str(obj)


def _has_command_like_text(text: str) -> bool:
    if not text:
        return False
    return bool(_CLI_FLAG_RE.search(text) or _COMMAND_HINT_RE.search(text))


def classify_answer_type(answer: Any) -> str:
    """
    Coarse answer types (Hotpot-style), with EDA-specific refinements.
    """
    if answer is None:
        return "null"
    if isinstance(answer, bool):
        return "yes_no"
    if isinstance(answer, (int, float)) and not isinstance(answer, bool):
        return "number"
    if isinstance(answer, dict):
        return "structured_object"
    if isinstance(answer, list):
        # Distinguish list-of-records vs list-of-scalars
        if answer and all(isinstance(x, dict) for x in answer):
            return "list_of_records"
        return "list"
    if isinstance(answer, str):
        s = answer.strip()
        s_lower = s.lower()
        if s_lower in {"yes", "no", "true", "false"}:
            return "yes_no"
        if _CLI_FLAG_RE.search(s) or _COMMAND_HINT_RE.search(s):
            # Commands often include flags; also allow command-like strings without flags
            return "command"
        if _UNIT_RE.search(s):
            return "number_with_unit"
        # Short "span" answers (tech node, stage name, etc.)
        if len(s) <= 60 and "\n" not in s:
            return "span"
        return "free_text"
    return f"other:{type(answer).__name__}"


@dataclass(frozen=True)
class Rule:
    label: str
    patterns: Tuple[re.Pattern[str], ...]

    def score(self, text: str) -> int:
        if not text:
            return 0
        return sum(1 for p in self.patterns if p.search(text))


def _re(p: str) -> re.Pattern[str]:
    return re.compile(p, re.IGNORECASE)


_STAGE_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("synthesis", _re(r"\bsynth(esis)?\b")),
    ("floorplan", _re(r"\bfloorplan\b")),
    ("placement", _re(r"\b(place(ment)?|global placement|detailed placement)\b")),
    ("cts", _re(r"\b(cts|clock tree synthesis)\b")),
    ("routing", _re(r"\b(routing|global routing|detailed routing)\b")),
    ("final", _re(r"\bfinal\b")),
)

# Topic detectors (EDA semantic type), later combined with scope (single vs multi-stage comparison).
_TOPIC_TIMING = (
    _re(r"\bwns\b"),
    _re(r"\btns\b"),
    _re(r"\bsetup\b"),
    _re(r"\bhold\b"),
    _re(r"\bslack\b"),
    _re(r"\bviolat\w*\b"),
    _re(r"\btiming\b"),
    _re(r"\bpath\s+depth\b"),
    _re(r"\bclock\s+period\b"),
    _re(r"\bfmax\b"),
)
_TOPIC_HPWL = (
    _re(r"\bhpwl\b"),
    _re(r"half[-\s]*perimeter"),
    _re(r"\bwirelength\b"),
)
_TOPIC_AREA = (
    _re(r"\barea\b"),
    _re(r"\bdie area\b"),
    _re(r"\bcore area\b"),
)
_TOPIC_UTIL = (
    _re(r"\butilization\b"),
    _re(r"\bdensity\b"),
)
_TOPIC_COMMAND_SETTINGS = (
    _re(r"\bcommand settings\b"),
    _re(r"\bcommand\b"),
    _re(r"\bsettings\b"),
    _re(r"\bflags?\b"),
    _re(r"\boptions?\b"),
)
_TOPIC_OPT_PROCEDURE = (
    _re(r"\boptimization (process|procedure)\b"),
    _re(r"\biterations?\b"),
    _re(r"\biterations count\b"),
    _re(r"\bhow many\b"),
    _re(r"\bcommitted\b"),
    _re(r"\boperation(s)?\b"),
    _re(r"\brepair_[a-z_]+\b"),
    _re(r"\b(repair|resize|buffer|unbuffer|sizeup|swap|vt_swap)\b"),
    _re(r"\bbuffer\s+insertion\b"),
    _re(r"\binsertion\s+and\s+removal\b"),
    _re(r"\bremoval\s+counts?\b"),
    _re(r"\binsertion\s+trends?\b"),
)

_TOPIC_ROUTING_OPT_HINT = (
    _re(r"\brouting\b"),
    _re(r"\b(iteration(s)?|iterations count|optimization|optimization process)\b"),
    _re(r"\bviolat\w*\b"),
    _re(r"\boverflow\b"),
    _re(r"\bcongestion\b"),
)

_TOPIC_TIMING_PROGRESS = (
    _re(r"\brepair[_\s-]?timing\b"),
    _re(r"\b(before|after|start|end|initial)\b"),
    _re(r"\bevolv(e|es|ed|ing)\b"),
    _re(r"\bprogress(ion)?\b"),
    _re(r"\bimprov(e|ement)\b"),
    _re(r"\bphase\b"),
)

_TOPIC_REQUIREMENTS = (
    _re(r"\brequirements?\b"),
    _re(r"\b(not\s+met|not\s+meet|meet|met)\b"),
    _re(r"\bgap(s)?\b"),
    _re(r"\bviolat\w*\b"),
    _re(r"\bconstraint(s)?\b"),
)

# Flow execution: completed/execute successfully, exit code, errors/warnings, stages executed, flow stability, via count
_TOPIC_FLOW_EXECUTION = (
    _re(r"\bflow\s+execution\b"),
    _re(r"\bcompleted\s+successfully\b"),
    _re(r"\bexecuted?\s+successfully\b"),
    _re(r"\bexecute\s+successfully\b"),
    _re(r"\bexecution\s+success\b"),
    _re(r"\bexecute_?successfully\b"),
    _re(r"\bexit\s+code(s)?\b"),
    _re(r"\berror\s+message(s)?\b"),
    _re(r"\berror\s+occurrences?\b"),
    _re(r"\berrors?\s+detected\b"),
    _re(r"\bwarnings?\s+reported\b"),
    _re(r"\bwhat\s+warnings?\b"),
    _re(r"\btimestamps?\b"),
    _re(r"\btotal\s+number\s+of\s+vias?\b"),
    _re(r"\bnumber\s+of\s+vias?\s+used\b"),
    _re(r"\bvias?\s+used\b"),
    _re(r"\bstages?\s+executed\b"),
    _re(r"\btechnology\s+node\b"),
    _re(r"\bline\s+numbers?\b"),
    _re(r"\bflow\s+stability\b"),
    _re(r"\bboth\s+designs?\s+(execute|executed)\b"),
    _re(r"\bexecution\s+success\s+indicators?\b"),
    _re(r"\bpasses?\s+(are\s+)?executed\b"),
    _re(r"\bdesign\s+hierarchy\b"),
    _re(r"\bmanage(s|ing)?\s+(design\s+)?hierarchy\b"),
)
# Memory and runtime (peak memory, elapsed time, runtime, how long, completes quickly)
_TOPIC_MEMORY_RUNTIME = (
    _re(r"\bpeak\s+memory\b"),
    _re(r"\bmemory\s+usage\b"),
    _re(r"\bmemory\s+requirements?\b"),
    _re(r"\bmegabytes?\b"),
    _re(r"\belapsed\s+time(s)?\b"),
    _re(r"\bruntime\b"),
    _re(r"\b(how\s+long|how\s+much\s+memory)\b"),
    _re(r"\bcompletes?\s+(this\s+stage\s+)?more\s+quickly\b"),
)
# Displacement (placement/CTS displacement reported)
_TOPIC_DISPLACEMENT = (
    _re(r"\bdisplacement\b"),
    _re(r"\btotal\s+displacement\b"),
    _re(r"\baverage\s+displacement\b"),
)
# Tapcell / endcaps / capcell
_TOPIC_TAPCELL_CAPCELL = (
    _re(r"\btapcell\b"),
    _re(r"\bendcaps?\b"),
    _re(r"\btapcell\s+insertion\b"),
    _re(r"\bcapcell\b"),
)
# IR drop / power (voltage drop)
_TOPIC_POWER_IR = (
    _re(r"\bir\s+drop\b"),
    _re(r"\bvdd\b"),
    _re(r"\bvss\b"),
)
# Power metric (consumption, total/breakdown, internal/switching/leakage, sequential/combinational/clock power)
_TOPIC_POWER_METRIC = (
    _re(r"\bpower\s+metric(s)?\b"),
    _re(r"\bpower\s+consumption\b"),
    _re(r"\btotal\s+power\b"),
    _re(r"\binternal\s+power\b"),
    _re(r"\bswitching\s+power\b"),
    _re(r"\bleakage\s+power\b"),
    _re(r"\bdynamic\s+power\b"),
    _re(r"\bstatic\s+power\b"),
    _re(r"\bsequential\s+power\b"),
    _re(r"\bcombinational\s+power\b"),
    _re(r"\bclock\s+power\b"),
    _re(r"\bpower\s+report(ed)?\b"),
    _re(r"\bpower\s+(changes?|differences?|evolves?)\b"),
    _re(r"\bpercentage\b.*\bpower\b"),
    _re(r"\bpower\b.*\b(attributed|break\s+down|components?)\b"),
)
# Vias (for multi_metrics when combined with wirelength etc.)
_TOPIC_VIAS = (_re(r"\bvias?\b"),)
# Checksum / integrity / verify
_TOPIC_CHECKSUM_VERIFY = (
    _re(r"\bsha1\b"),
    _re(r"\bchecksum(s)?\b"),
    _re(r"\bintegrity\b"),
    _re(r"\bverif(y|ication)\b"),
)
# Constraint / requirement check (requirements met or not met, unmet requirements, gaps, test_metrics, threshold)
_TOPIC_CONSTRAINT_REQUIREMENT_CHECK = (
    _re(r"\brequirements?\s+(are\s+)?(not\s+)?met\b"),
    _re(r"\b(not\s+)?met\s+requirements?\b"),
    _re(r"\bunmet\s+requirements?\b"),
    _re(r"\bconstraint(s)?\s+(are\s+)?(not\s+)?met\b"),
    _re(r"\b(not\s+)?met\s+constraint(s)?\b"),
    _re(r"\bwhich\s+requirements?\b"),
    _re(r"\btest_metrics\b"),
    _re(r"\blargest\s+gap\b"),
    _re(r"\bgap(s)?\s+(for\s+)?(any\s+)?(unmet\s+)?requirements?\b"),
    _re(r"\brequirements?\s+in\s+the\s+test_metrics\b"),
    _re(r"\bdo\s+not\s+meet\b"),
    _re(r"\bdon't\s+meet\b"),
    _re(r"\bthreshold\s+requirements?\b"),
    _re(r"\bthreshold\s+values?\b"),
    _re(r"\bactual\s+values?\s+and\s+threshold\b"),
    _re(r"\bpercentage\s+gap\s+between\s+actual\b"),
    _re(r"\btest\s+metrics?\s+fields?\b"),
    _re(r"\bmetrics?\s+that\s+(do\s+not\s+)?meet\b"),
    _re(r"\bmeet\s+their\s+(threshold\s+)?requirements?\b"),
)
# Clock tree / CTS optimization (clock buffers, buffer depth, clock tree depth, sink wire length, clock skew in CTS)
_TOPIC_CLOCK_TREE_OPT = (
    _re(r"\bclock\s+buffer(s)?\b"),
    _re(r"\bbuffer\s+depth(s)?\b"),
    _re(r"\bclock\s+tree\s+(depth|synthesis|structure)?\b"),
    _re(r"\bclock\s+tree\s+synthesis\b"),
    _re(r"\bcts\s+phase\b"),
    _re(r"\bclock\s+tree\s+synthesis\s+\(cts\)\b"),
    _re(r"\bsink\s+wire\s+length\b"),
    _re(r"\bclock\s+skew\b"),
    _re(r"\bclock\s+buffer\s+count\b"),
    _re(r"\b(min|max)\w*\s+buffer\s+depth\b"),
    _re(r"\baverage\s+sink\s+wire\b"),
    _re(r"\bclock\s+nets?\b"),
)
# Congestion / overflow / usage (routing overflow, placement overflow, density, usage percentage, resource/demand)
_TOPIC_CONGESTION_METRIC = (
    _re(r"\boverflow\b"),
    _re(r"\bcongestion\b"),
    _re(r"\busage\s+percentage\b"),
    _re(r"\b(total\s+)?usage\s+was\s+achieved\b"),
    _re(r"\boverflow\s+percentage\b"),
    _re(r"\b(initial|final)\s+overflow\b"),
    _re(r"\bdesign\s+density\b"),
    _re(r"\boverflow\s+metrics?\b"),
    _re(r"\bdensity\s+and\s+overflow\b"),
    _re(r"\bresource\s+and\s+demand\s+metrics?\b"),
    _re(r"\bhighest\s+usage\s+percentage\b"),
    _re(r"\blayers?\s+with\s+.*usage\b"),
    _re(r"\bplacement\s+.*overflow\b"),
    _re(r"\bglobal\s+placement\s+.*overflow\b"),
)
# Antenna diode / antenna effect (diode count, antenna effect management)
_TOPIC_ANTENNA_DIODE = (
    _re(r"\bantenna\s+diode(s)?\b"),
    _re(r"\bantenna\s+effect\b"),
    _re(r"\bantenna\s+diode\s+count(s)?\b"),
    _re(r"\bantenna\s+effect\s+management\b"),
)


_COMPARISON_RE = re.compile(
    r"\b(compare|difference|different|change|changed|evolve|trend|versus|vs\.?|"
    r"from\b.*\bto\b|across|throughout|how does .* (change|compare))\b",
    re.IGNORECASE,
)
_BRIDGE_RE = re.compile(
    r"\b(which|what)\b.*\b(most|highest|largest|top|maximum|minim(um|ize)|"
    r"consume(s)? the most|saw the most)\b",
    re.IGNORECASE,
)


def classify_reasoning_type(question_text: str) -> str:
    """
    Hotpot-style reasoning type:
    - comparison: asks for comparing two+ entities/stages/values
    - bridge: requires selecting/aggregating across candidates (e.g., "which stage had the most ...")
    - single: default
    """
    q = question_text or ""
    if _COMPARISON_RE.search(q):
        return "comparison"
    if _BRIDGE_RE.search(q):
        return "bridge"
    return "single"


def _detect_stages(question_text: str) -> List[str]:
    q = (question_text or "").lower()
    stages: List[str] = []
    for name, pat in _STAGE_PATTERNS:
        if pat.search(q):
            stages.append(name)
    # preserve canonical order, unique
    seen: set[str] = set()
    out: List[str] = []
    for s in stages:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out


def _is_multi_stage(gen_qtype: str, question_text: str) -> bool:
    """
    Decide if the question spans more than one *design stage*.
    Uses stage mentions in the question text only (design-stage scope).
    """
    stages = _detect_stages(question_text)
    return len(stages) >= 2


def _scope_label(gen_qtype: str, question_text: str) -> str:
    return "multi" if _is_multi_stage(gen_qtype, question_text) else "single"


def _topic_hit(patterns: Tuple[re.Pattern[str], ...], question_text: str) -> int:
    if not question_text:
        return 0
    return sum(1 for p in patterns if p.search(question_text))


def classify_eda_question_type(question_text: str, gen_qtype: str = "", answer: Any = None) -> str:
    """
    Produce a combined semantic label: <topic>_single_stage | <topic>_multi_stage_comp.
    Final category names (all topics, consistent suffix):

      - flow_execution_result (no stage suffix)
      - memory_runtime_single_stage | memory_runtime_multi_stage_comp
      - checksum_verify_single_stage | checksum_verify_multi_stage_comp
      - displacement_optimization_single_stage | displacement_optimization_multi_stage_comp
      - tapcell_capcell_single_stage | tapcell_capcell_multi_stage_comp
      - ir_drop_single_stage | ir_drop_multi_stage_comp
      - power_metric_single_stage | power_metric_multi_stage_comp
      - clock_tree_opt_single_stage | clock_tree_opt_multi_stage_comp
      - congestion_metric_single_stage | congestion_metric_multi_stage_comp
      - antenna_diode_single_stage | antenna_diode_multi_stage_comp
      - constraint_requirement_check_single_stage | constraint_requirement_check_multi_stage_comp
      - timing_metric_progression_in_optimization_single_stage | timing_metric_progression_in_optimization_multi_stage_comp
      - timing_metric_requirement_met_single_stage | timing_metric_requirement_met_multi_stage_comp
      - routing_optimization_single_stage | routing_optimization_multi_stage_comp
      - command_settings_comparison_single_stage | command_settings_comparison_multi_stage_comp
      - optimization_procedure_single_stage | optimization_procedure_multi_stage_comp
      - multi_metrics_single_stage | multi_metrics_multi_stage_comp
      - timing_metric_single_stage | timing_metric_multi_stage_comp
      - hpwl_single_stage | hpwl_multi_stage_comp
      - area_single_stage | area_multi_stage_comp
      - utilization_single_stage | utilization_multi_stage_comp
      - other_single_stage | other_multi_stage_comp (uncategorizable)
    """
    q = question_text or ""
    gen_qtype_lower = (gen_qtype or "").strip().lower()
    multi = _is_multi_stage(gen_qtype, q)
    answer_text = _textify(answer)
    answer_has_command = _has_command_like_text(answer_text)

    # Flow execution Q&A (single-file or multi-file) → flow_execution_result category
    if gen_qtype_lower in ("flow_execution", "multi_file_flow_execution"):
        return "flow_execution_result"
    if _topic_hit(_TOPIC_FLOW_EXECUTION, q) > 0:
        return "flow_execution_result"

    # Memory / runtime → memory_runtime
    if _topic_hit(_TOPIC_MEMORY_RUNTIME, q) > 0:
        return "memory_runtime_multi_stage_comp" if multi else "memory_runtime_single_stage"

    # Checksum / integrity / verify → checksum_verify
    if _topic_hit(_TOPIC_CHECKSUM_VERIFY, q) > 0:
        return "checksum_verify_multi_stage_comp" if multi else "checksum_verify_single_stage"

    # Displacement (placement/CTS) → displacement_optimization
    if _topic_hit(_TOPIC_DISPLACEMENT, q) > 0:
        return "displacement_optimization_multi_stage_comp" if multi else "displacement_optimization_single_stage"

    # Tapcell / endcaps / capcell → tapcell_capcell (multi if generator is multi_stage or question spans stages)
    if _topic_hit(_TOPIC_TAPCELL_CAPCELL, q) > 0:
        multi_tap = multi or gen_qtype_lower in ("multi_stage", "stage_to_final")
        return "tapcell_capcell_multi_stage_comp" if multi_tap else "tapcell_capcell_single_stage"

    # IR drop / power (voltage) → ir_drop
    if _topic_hit(_TOPIC_POWER_IR, q) > 0:
        return "ir_drop_multi_stage_comp" if multi else "ir_drop_single_stage"

    # Power metric (consumption, total/sequential/combinational/clock power, etc.) → power_metric
    if _topic_hit(_TOPIC_POWER_METRIC, q) > 0:
        return "power_metric_multi_stage_comp" if multi else "power_metric_single_stage"

    # Clock tree / CTS (clock buffers, buffer depth, clock tree depth, sink wire length, clock skew) → clock_tree_opt
    if _topic_hit(_TOPIC_CLOCK_TREE_OPT, q) > 0:
        return "clock_tree_opt_multi_stage_comp" if multi else "clock_tree_opt_single_stage"

    # Congestion / overflow / usage (overflow, congestion, usage percentage, density) → congestion_metric
    if _topic_hit(_TOPIC_CONGESTION_METRIC, q) > 0:
        return "congestion_metric_multi_stage_comp" if multi else "congestion_metric_single_stage"

    # Antenna diode / antenna effect (diode count, antenna effect management) → antenna_diode
    if _topic_hit(_TOPIC_ANTENNA_DIODE, q) > 0:
        return "antenna_diode_multi_stage_comp" if multi else "antenna_diode_single_stage"

    # Constraint / requirement check (requirements met, constraints met, unmet, gaps) → constraint_requirement_check
    if _topic_hit(_TOPIC_CONSTRAINT_REQUIREMENT_CHECK, q) > 0:
        return "constraint_requirement_check_multi_stage_comp" if multi else "constraint_requirement_check_single_stage"

    # Topic hits (some questions mention multiple metrics)
    hits = {
        "timing_metric": _topic_hit(_TOPIC_TIMING, q),
        "hpwl": _topic_hit(_TOPIC_HPWL, q),
        "area": _topic_hit(_TOPIC_AREA, q),
        "utilization": _topic_hit(_TOPIC_UTIL, q),
        "vias": _topic_hit(_TOPIC_VIAS, q),
        "command_settings": _topic_hit(_TOPIC_COMMAND_SETTINGS, q),
        "optimization_procedure": _topic_hit(_TOPIC_OPT_PROCEDURE, q),
        "routing_optimization": _topic_hit(_TOPIC_ROUTING_OPT_HINT, q),
        "timing_progress": _topic_hit(_TOPIC_TIMING_PROGRESS, q),
        "requirements": _topic_hit(_TOPIC_REQUIREMENTS, q),
    }

    # If the *answer* contains known command names (global_placement, global_routing, etc.)
    # treat it as command_settings even if the question text is terse.
    if answer_has_command:
        hits["command_settings"] += 3

    # Timing progression during optimization (repair_timing evolution)
    if hits["timing_metric"] > 0 and hits["timing_progress"] > 0:
        return (
            "timing_metric_progression_in_optimization_multi_stage_comp"
            if multi
            else "timing_metric_progression_in_optimization_single_stage"
        )

    # Timing requirements unmet/met (gaps, constraints, violations)
    if hits["timing_metric"] > 0 and hits["requirements"] > 0:
        return "timing_metric_requirement_met_multi_stage_comp" if multi else "timing_metric_requirement_met_single_stage"

    # Routing optimization: avoid stealing timing questions; require stronger routing signals
    if hits["routing_optimization"] >= 3 and hits["timing_metric"] == 0:
        return "routing_optimization_multi_stage_comp" if multi else "routing_optimization_single_stage"

    # Highest priority: command settings
    if hits["command_settings"] > 0 and (
        _CLI_FLAG_RE.search(q)
        or _COMMAND_HINT_RE.search(q)
        or "command" in q.lower()
        or answer_has_command
    ):
        return "command_settings_comparison_multi_stage_comp" if multi else "command_settings_comparison_single_stage"

    # Optimization procedures/knobs (repair_timing, buffer insertion/removal, iterations, etc.)
    opt_proc = hits["optimization_procedure"] > 0
    opt_keywords = (
        "repair_" in q.lower()
        or "optimization" in q.lower()
        or "iterations" in q.lower()
        or "how many" in q.lower()
        or ("buffer" in q.lower() and ("insertion" in q.lower() or "removal" in q.lower() or "removed" in q.lower() or "trends" in q.lower()))
    )
    if opt_proc and opt_keywords:
        return "optimization_procedure_multi_stage_comp" if multi else "optimization_procedure_single_stage"

    metric_hits = {k: v for k, v in hits.items() if k in {"timing_metric", "hpwl", "area", "utilization", "vias"} and v > 0}
    if len(metric_hits) >= 2:
        multi_mm = multi or gen_qtype_lower in ("multi_stage", "stage_to_final")
        return "multi_metrics_multi_stage_comp" if multi_mm else "multi_metrics_single_stage"

    # Single metric topics (hpwl: multi also when generator is multi_stage/stage_to_final)
    if hits["timing_metric"] > 0:
        return "timing_metric_multi_stage_comp" if multi else "timing_metric_single_stage"
    if hits["hpwl"] > 0:
        multi_hpwl = multi or gen_qtype_lower in ("multi_stage", "stage_to_final")
        return "hpwl_multi_stage_comp" if multi_hpwl else "hpwl_single_stage"
    if hits["area"] > 0:
        return "area_multi_stage_comp" if multi else "area_single_stage"
    if hits["utilization"] > 0:
        return "utilization_multi_stage_comp" if multi else "utilization_single_stage"

    return "other_multi_stage_comp" if multi else "other_single_stage"


def _is_multifile_raw(data: Dict[str, Any]) -> bool:
    """True if data is multifile raw format (nested single_stage_comparison / multi_stage_comparison / flow_execution with question_set.qa_pairs)."""
    for section in ("single_stage_comparison", "multi_stage_comparison", "flow_execution"):
        sec = data.get(section)
        if isinstance(sec, dict):
            qs = sec.get("question_set")
            if isinstance(qs, dict) and isinstance(qs.get("qa_pairs"), list):
                return True
    return False


def _iter_questions_from_multifile_raw(data: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """
    Parse multifile raw JSON: single_stage_comparison, multi_stage_comparison, flow_execution
    each with question_set.qa_pairs (list of {question, answer, difficulty}).
    Yields normalized items with id, type, question, answer, difficulty for taxonomy.
    """
    section_config = [
        ("single_stage_comparison", "multi_file_single_stage_comparison"),
        ("multi_stage_comparison", "multi_file_multi_stage_comparison"),
        ("flow_execution", "flow_execution"),
    ]
    for section_key, type_label in section_config:
        sec = data.get(section_key)
        if not isinstance(sec, dict):
            continue
        qs = sec.get("question_set")
        if not isinstance(qs, dict):
            continue
        pairs = qs.get("qa_pairs")
        if not isinstance(pairs, list):
            continue
        section_id = (qs.get("id") or section_key).strip()
        for idx, item in enumerate(pairs, start=1):
            if not isinstance(item, dict):
                continue
            qtext = item.get("question")
            if not qtext:
                continue
            # Synthetic id: e.g. Q_MULTIFILE_001_P01, Q_FLOW_MULTIFILE_001_P02
            synthetic_id = f"{section_id}_P{idx:02d}"
            yield {
                "id": synthetic_id,
                "type": type_label,
                "question": qtext,
                "answer": item.get("answer"),
                "difficulty": item.get("difficulty"),
            }


def iter_questions_from_json(data: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """
    Supports:
    - multifile raw: single_stage_comparison / multi_stage_comparison / flow_execution with question_set.qa_pairs
    - top-level `qa_pairs`
    - top-level `questions`
    """
    if _is_multifile_raw(data):
        yield from _iter_questions_from_multifile_raw(data)
        return
    if isinstance(data.get("qa_pairs"), list):
        for q in data["qa_pairs"]:
            if isinstance(q, dict):
                yield q
        return
    if isinstance(data.get("questions"), list):
        for q in data["questions"]:
            if isinstance(q, dict):
                yield q


def iter_input_files(path: Path) -> List[Path]:
    """Collect QA JSON files from a single path (file or directory)."""
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.glob("*_qa_all_types.json"))
    raise FileNotFoundError(str(path))


# Filename patterns for single-design and multifile QA JSONs (non-raw only; raw excluded to avoid double-counting)
_QA_ALL_TYPES_PATTERN = "*_qa_all_types.json"
_QA_ALL_MULTIFILE_PATTERN = "*_qa_all_multifile.json"


def _glob_qa_jsons_in_dir(path: Path) -> List[Path]:
    """
    Find QA JSON files in a directory and its subdirs (single-design + multifile non-raw only).
    - Single-design: *_qa_all_types.json
    - Multifile: *_qa_all_multifile.json only (raw *_qa_all_multifile_raw.json excluded so
      counts equal len(qa_pairs) from the non-raw file; no id-based tracking).
    """
    files: List[Path] = []
    # Direct in directory
    files.extend(path.glob(_QA_ALL_TYPES_PATTERN))
    files.extend(path.glob(_QA_ALL_MULTIFILE_PATTERN))
    # One level down (design_tech subdirs: aes_block_asap7, uartgf180, ...)
    for sub in path.iterdir():
        if not sub.is_dir():
            continue
        files.extend(sub.glob(_QA_ALL_TYPES_PATTERN))
        files.extend(sub.glob(_QA_ALL_MULTIFILE_PATTERN))
        # Multifile: one more level for multifile_comparison/<design_vs_design>/
        if sub.name == "multifile_comparison":
            for sub2 in sub.iterdir():
                if sub2.is_dir():
                    files.extend(sub2.glob(_QA_ALL_TYPES_PATTERN))
                    files.extend(sub2.glob(_QA_ALL_MULTIFILE_PATTERN))
    return files


def collect_input_files(paths: List[Path]) -> List[Path]:
    """
    Collect all QA JSON files from a list of paths (each can be a file or directory).
    For a directory (e.g. qa_output), finds *_qa_all_types.json in the directory and
    in each design_tech subdirectory (qa_output/aes_block_asap7/, qa_output/uartgf180/, ...).
    Duplicates (by resolved path) are removed.
    """
    seen: Set[str] = set()
    out: List[Path] = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            continue
        if path.is_file():
            key = str(path.resolve())
            if key not in seen:
                seen.add(key)
                out.append(path)
            continue
        if path.is_dir():
            for f in sorted(_glob_qa_jsons_in_dir(path)):
                key = str(f.resolve())
                if key not in seen:
                    seen.add(key)
                    out.append(f)
    return sorted(out, key=lambda x: (str(x.parent), x.name))


def load_pass_k_ignored(
    filepath: Optional[str],
) -> Tuple[Dict[str, Set[str]], Dict[str, List[str]], int]:
    """
    Load pass_k_ignored_qa_pairs.json and build design_tech -> set of question IDs to filter out.

    Expected JSON: { "file_map": { "<design_tech>": ["qa_output_*_<design_tech>", ...], ... },
                     "filtered_qa_pairs": { "<design_tech>": [ {"id": "...", "question": "...", "answer": ...}, ... ], ... } }

    Returns:
        - ignored_qids_by_design_tech: design_tech -> set of qids to exclude
        - file_map: design_tech -> list of directory names (for resolving design_tech from path)
        - total_ignored: total number of ignored qids across all design_tech
    """
    ignored: Dict[str, Set[str]] = {}
    file_map: Dict[str, List[str]] = {}
    total = 0
    if not filepath:
        return ignored, file_map, total
    path = Path(filepath)
    if not path.exists():
        return ignored, file_map, total
    try:
        data = json.loads(path.read_text())
    except Exception:
        return ignored, file_map, total
    filtered = data.get("filtered_qa_pairs") or {}
    if not isinstance(filtered, dict):
        return ignored, file_map, total
    raw_fm = data.get("file_map") or {}
    if isinstance(raw_fm, dict):
        for k, v in raw_fm.items():
            if isinstance(v, list):
                file_map[k] = [str(x) for x in v]
            elif isinstance(v, str):
                file_map[k] = [str(v)]
            else:
                file_map[k] = []
    for design_tech, pairs in filtered.items():
        if not isinstance(pairs, list):
            continue
        qids: Set[str] = set()
        for item in pairs:
            if isinstance(item, dict) and item.get("id") is not None:
                qids.add(str(item["id"]))
        if qids:
            ignored[design_tech] = qids
            total += len(qids)
    return ignored, file_map, total


def resolve_design_tech_for_file(
    json_path: Path,
    design_tech_keys: List[str],
    file_map: Dict[str, List[str]],
) -> Optional[str]:
    """
    Resolve design_tech key for a QA JSON file path for use with pass_k_ignored filtering.

    - Parent directory name (e.g. aes_block_asap7) is used.
    - If it is a key in design_tech_keys, return it.
    - Else find design_tech such that parent name is in file_map[design_tech].
    - Else return parent name (caller may have no ignored set for it).
    """
    parent_name = json_path.parent.name
    if parent_name in design_tech_keys:
        return parent_name
    for dt, dirs in file_map.items():
        if parent_name in dirs:
            return dt
    return parent_name


def _answer_preview(answer: Any, max_len: int = 220) -> str:
    if answer is None:
        return "null"
    if isinstance(answer, (dict, list)):
        s = json.dumps(answer, ensure_ascii=False)
    else:
        s = str(answer)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


def _question_preview(question_text: str, max_len: int = 220) -> str:
    s = re.sub(r"\s+", " ", question_text or "").strip()
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


def _pick_representatives(items: List[Dict[str, Any]], k: int = 3) -> List[Dict[str, Any]]:
    """
    Pick up to k representative items.
    Preference order:
    - maximize generator-type variety
    - then maximize difficulty variety
    - then shorter question for readability
    """
    if not items:
        return []

    # Stable sort for readability
    items_sorted = sorted(items, key=lambda x: (len(x.get("question", "")), x.get("id", "")))

    chosen: List[Dict[str, Any]] = []
    used_gen: set[str] = set()
    used_diff: set[str] = set()

    # Pass 1: unique generator types
    for it in items_sorted:
        if len(chosen) >= k:
            break
        g = it.get("generator_question_type", "unknown")
        if g not in used_gen:
            chosen.append(it)
            used_gen.add(g)
            used_diff.add(it.get("difficulty", "unknown"))

    # Pass 2: add new difficulties if still room
    for it in items_sorted:
        if len(chosen) >= k:
            break
        if it in chosen:
            continue
        d = it.get("difficulty", "unknown")
        if d not in used_diff:
            chosen.append(it)
            used_diff.add(d)

    # Pass 3: fill remainder
    for it in items_sorted:
        if len(chosen) >= k:
            break
        if it in chosen:
            continue
        chosen.append(it)

    return chosen[:k]


def summarize(
    files: List[Path],
    ignored_qids_by_design_tech: Optional[Dict[str, Set[str]]] = None,
    file_map: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    counts = {
        "by_generator_question_type": Counter(),  # q['type'] e.g. single_stage/multi_stage...
        "by_reasoning_type": Counter(),  # hotpot-inspired
        "by_eda_question_type": Counter(),  # timing/commands/routing/...
        "by_answer_type": Counter(),  # hotpot-inspired + EDA refinements
        "by_difficulty": Counter(),
        "by_design_tech": Counter(),  # e.g. aes_block_asap7, uartgf180
        "qtype_x_answer_type": defaultdict(Counter),
        "eda_qtype_x_answer_type": defaultdict(Counter),
        "eda_qtype_x_reasoning": defaultdict(Counter),
    }

    examples_by_answer_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    examples_by_eda_qtype: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    all_labeled_examples: List[Dict[str, Any]] = []
    design_tech_keys = list(ignored_qids_by_design_tech.keys()) if ignored_qids_by_design_tech else []
    filtered_out_by_design_tech: Dict[str, int] = defaultdict(int)
    total_filtered_out = 0

    n_questions = 0
    for f in files:
        data = json.loads(f.read_text())
        design_tech: Optional[str] = None
        ignored_qids: Set[str] = set()
        if ignored_qids_by_design_tech is not None and file_map is not None:
            design_tech = resolve_design_tech_for_file(f, design_tech_keys, file_map)
            ignored_qids = ignored_qids_by_design_tech.get(design_tech or "", set()) or set()
        if design_tech is None or design_tech == "":
            design_tech = f.parent.name  # e.g. aes_block_asap7, uartgf180
        for q in iter_questions_from_json(data):
            qid = _norm(q.get("id"))
            if qid and ignored_qids and qid in ignored_qids:
                total_filtered_out += 1
                if design_tech:
                    filtered_out_by_design_tech[design_tech] += 1
                continue
            qtext = _norm(q.get("question", ""))
            gen_qtype = _norm(q.get("type") or q.get("question_type") or "unknown").lower()
            reasoning = classify_reasoning_type(qtext)
            eda_qtype = classify_eda_question_type(qtext, gen_qtype, q.get("answer"))
            ans_type = classify_answer_type(q.get("answer"))
            difficulty = extract_difficulty(qtext)
            stages_mentioned = _detect_stages(qtext)
            scope = _scope_label(gen_qtype, qtext)

            counts["by_generator_question_type"][gen_qtype or "unknown"] += 1
            counts["by_reasoning_type"][reasoning] += 1
            counts["by_eda_question_type"][eda_qtype] += 1
            counts["by_answer_type"][ans_type] += 1
            counts["by_difficulty"][difficulty] += 1

            counts["qtype_x_answer_type"][gen_qtype][ans_type] += 1
            counts["eda_qtype_x_answer_type"][eda_qtype][ans_type] += 1
            counts["eda_qtype_x_reasoning"][eda_qtype][reasoning] += 1
            counts["by_design_tech"][design_tech or "unknown"] += 1

            # Capture representative QA record (preview for summaries; full for listing)
            rec = {
                "id": _norm(q.get("id")) or None,
                "design_tech": design_tech,
                "generator_question_type": gen_qtype or "unknown",
                "reasoning_type": reasoning,
                "eda_question_type": eda_qtype,
                "answer_type": ans_type,
                "difficulty": difficulty,
                "stage": _norm(q.get("stage")) or None,
                "stages_mentioned": stages_mentioned,
                "scope": scope,
                "question": _question_preview(qtext),
                "answer_preview": _answer_preview(q.get("answer")),
                "question_full": qtext,
                "answer_full": _textify(q.get("answer")),
                "source_file": str(f),
            }
            examples_by_answer_type[ans_type].append(rec)
            examples_by_eda_qtype[eda_qtype].append(rec)
            all_labeled_examples.append(rec)

            n_questions += 1

    result: Dict[str, Any] = {"total_questions_extracted": n_questions}
    for k, v in counts.items():
        if isinstance(v, Counter):
            result[k] = dict(v)
        else:
            # nested defaultdict(Counter)
            result[k] = {kk: dict(vv) for kk, vv in v.items()}

    # Pick top-3 representatives per category
    result["representatives"] = {
        "by_answer_type_top3": {
            k: _pick_representatives(v, k=3) for k, v in examples_by_answer_type.items()
        },
        "by_eda_question_type_top3": {
            k: _pick_representatives(v, k=3) for k, v in examples_by_eda_qtype.items()
        },
    }
    # Full labeled QA records (for filtering/debugging). Can be large across many files.
    result["labeled_qa"] = all_labeled_examples
    if ignored_qids_by_design_tech:
        result["pass_k_filtered_out_total"] = total_filtered_out
        result["pass_k_filtered_out_by_design_tech"] = dict(filtered_out_by_design_tech)
    return result


def _print_counter(title: str, c: Dict[str, int]) -> None:
    print(f"\n{title}")
    for k, v in sorted(c.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {k:24s} {v}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Hotpot-inspired question/answer type stats for EDA QA JSON.")
    ap.add_argument(
        "paths",
        type=str,
        nargs="+",
        metavar="PATH",
        help="One or more paths: a *_qa_all_types.json file, or a directory. For a dir (e.g. qa_output), finds all design_tech subdirs (aes_block_asap7, uartgf180, ...) and their *_qa_all_types.json. Summary aggregates all.",
    )
    ap.add_argument(
        "--export-json",
        type=str,
        default=None,
        help="Write full summary JSON to this path",
    )
    ap.add_argument(
        "--print-representatives",
        action="store_true",
        help="Print top-3 representative QA examples per answer type and semantic question type",
    )
    ap.add_argument(
        "--list-other",
        action="store_true",
        help="List QA pairs whose EDA semantic question type is other_* (prints id/Q/A preview)",
    )
    ap.add_argument(
        "--list-eda-type",
        type=str,
        nargs="*",
        default=None,
        metavar="TYPE",
        help="List QA pairs with these EDA semantic question type(s). Pass one or more types (exact match). E.g. --list-eda-type timing_metric_single_stage optimization_procedure_multi_stage_comp",
    )
    ap.add_argument(
        "--list-top-per-eda-type",
        type=int,
        default=None,
        metavar="N",
        help="List top N QA per EDA semantic question type (grouped by type). E.g. --list-top-per-eda-type 3",
    )
    ap.add_argument(
        "--list-reasoning-type",
        type=str,
        default=None,
        metavar="TYPE",
        help="List QA pairs with this reasoning type: single | comparison | bridge",
    )
    ap.add_argument(
        "--list-answer-type",
        type=str,
        default=None,
        metavar="TYPE",
        help="List QA pairs with this answer type (exact match, e.g. number, structured_object)",
    )
    ap.add_argument(
        "--max-list",
        type=int,
        default=200,
        help="Max QA pairs to print for --list-* options (default: 200)",
    )
    ap.add_argument(
        "--pass-k-ignored",
        type=str,
        default=None,
        metavar="JSON",
        help="Path to pass_k_ignored_qa_pairs.json; filter out listed QA pairs per design_tech",
    )
    args = ap.parse_args()

    paths = [Path(p) for p in args.paths]
    files = collect_input_files(paths)
    if not files:
        print("No QA JSON files found from the given path(s).")
        return
    ignored_qids, file_map, _ = load_pass_k_ignored(args.pass_k_ignored)
    out = summarize(
        files,
        ignored_qids_by_design_tech=ignored_qids or None,
        file_map=file_map or None,
    )
    out["input_files"] = [str(f) for f in files]
    out["input_paths"] = [str(p) for p in paths]

    print(f"Input path(s): {len(paths)}")
    print(f"QA JSON file(s): {len(files)}")
    if len(files) <= 20:
        for f in files:
            print(f"  - {f}")
    else:
        for f in files[:5]:
            print(f"  - {f}")
        print(f"  ... and {len(files) - 5} more (see input_files in export)")
    print(f"Total questions extracted: {out['total_questions_extracted']}")
    if out.get("pass_k_filtered_out_total") is not None:
        print(f"Filtered out (pass_k_ignored): {out['pass_k_filtered_out_total']}")
        for dt, n in sorted(out.get("pass_k_filtered_out_by_design_tech", {}).items()):
            print(f"  - {dt}: {n}")

    _print_counter("By design_tech:", out["by_design_tech"])
    _print_counter("Generator question type (existing q['type']):", out["by_generator_question_type"])
    _print_counter("Reasoning type (Hotpot-style):", out["by_reasoning_type"])
    _print_counter("EDA semantic question type:", out["by_eda_question_type"])
    _print_counter("Answer type:", out["by_answer_type"])
    _print_counter("Difficulty:", out["by_difficulty"])

    if args.print_representatives:
        reps = out.get("representatives", {})
        by_ans = reps.get("by_answer_type_top3", {})
        by_sem = reps.get("by_eda_question_type_top3", {})

        print("\nRepresentative QAs (top-3) by answer type:")
        for ans_type, items in sorted(by_ans.items(), key=lambda kv: kv[0]):
            print(f"\n- {ans_type} ({len(items)} shown)")
            for it in items:
                qid = it.get("id") or "n/a"
                print(f"  - {qid} | {it.get('generator_question_type')} | {it.get('difficulty')} | {it.get('eda_question_type')}")
                print(f"    Q: {it.get('question')}")
                print(f"    A: {it.get('answer_preview')}")

        print("\nRepresentative QAs (top-3) by EDA semantic question type:")
        for sem_type, items in sorted(by_sem.items(), key=lambda kv: kv[0]):
            print(f"\n- {sem_type} ({len(items)} shown)")
            for it in items:
                qid = it.get("id") or "n/a"
                print(f"  - {qid} | {it.get('generator_question_type')} | {it.get('difficulty')} | {it.get('answer_type')}")
                print(f"    Q: {it.get('question')}")
                print(f"    A: {it.get('answer_preview')}")

    if (
        args.list_other
        or (args.list_eda_type is not None and len(args.list_eda_type) > 0)
        or args.list_reasoning_type
        or args.list_answer_type
    ):
        labeled = out.get("labeled_qa", [])
        if args.list_eda_type is not None and len(args.list_eda_type) > 0:
            wanted_set = {t.strip() for t in args.list_eda_type if t and t.strip()}
            matches = [x for x in labeled if x.get("eda_question_type") in wanted_set]
            title = f"QA pairs with EDA semantic type in {sorted(wanted_set)!r} ({len(matches)} total)"
        elif args.list_reasoning_type:
            wanted = args.list_reasoning_type.strip().lower()
            matches = [x for x in labeled if (x.get("reasoning_type") or "").lower() == wanted]
            title = f"QA pairs with reasoning type == {wanted!r}"
        elif args.list_answer_type:
            wanted = args.list_answer_type.strip()
            matches = [x for x in labeled if x.get("answer_type") == wanted]
            title = f"QA pairs with answer type == {wanted!r}"
        elif args.list_other:
            matches = [
                x
                for x in labeled
                if isinstance(x.get("eda_question_type"), str)
                and x["eda_question_type"].startswith("other_")
            ]
            title = "QA pairs with EDA semantic type other_*"
        else:
            matches = []
            title = ""

        if title:
            print(f"\n{title}: {len(matches)} found")
            for it in matches[: max(1, args.max_list)]:
                qid = it.get("id") or "n/a"
                stages = ",".join(it.get("stages_mentioned") or []) or "n/a"
                print(
                    f"- {qid} | gen={it.get('generator_question_type')} | scope={it.get('scope')} | stages={stages} | diff={it.get('difficulty')}"
                )
                q_text = it.get("question_full") or it.get("question")
                a_text = it.get("answer_full") or it.get("answer_preview")
                print(f"  Q: {q_text}")
                print(f"  A: {a_text}")

            if len(matches) > args.max_list:
                print(f"\n(truncated) showing first {args.max_list} of {len(matches)}; increase with --max-list")

    if args.list_top_per_eda_type is not None and args.list_top_per_eda_type >= 1:
        labeled = out.get("labeled_qa", [])
        by_eda: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for it in labeled:
            eda = it.get("eda_question_type") or "unknown"
            by_eda[eda].append(it)
        n_per = args.list_top_per_eda_type
        print(f"\nTop {n_per} QA per EDA semantic question type:")
        for eda_type in sorted(by_eda.keys()):
            items = by_eda[eda_type][:n_per]
            print(f"\n--- {eda_type} ({len(items)} shown, {len(by_eda[eda_type])} total) ---")
            for it in items:
                qid = it.get("id") or "n/a"
                stages = ",".join(it.get("stages_mentioned") or []) or "n/a"
                print(
                    f"- {qid} | gen={it.get('generator_question_type')} | scope={it.get('scope')} | stages={stages} | diff={it.get('difficulty')}"
                )
                q_text = it.get("question_full") or it.get("question")
                a_text = it.get("answer_full") or it.get("answer_preview")
                print(f"  Q: {q_text}")
                print(f"  A: {a_text}")

    if args.export_json:
        Path(args.export_json).write_text(json.dumps(out, indent=2))
        print(f"\nWrote: {args.export_json}")


if __name__ == "__main__":
    main()

