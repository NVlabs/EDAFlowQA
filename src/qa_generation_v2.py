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
QA generation utilities for OpenROAD metrics JSON files produced by `extract_all_stages.py`.

Simple LLM-based approach:
1. Sample a stage/substage from the aggregate metrics JSON
2. Use LLM to generate an interesting question about that stage
3. Use LLM to generate Python code to extract the answer
4. Execute the code to get the answer
"""

from __future__ import annotations

import json
import random
import re
import pdb
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional
from llm import llm_inference, call_Perflab_claude37, call_dar_gateway
from stage_parsing_examples import get_parsing_examples, get_multi_stage_parsing_examples


class RejectionStats:
    """Track statistics for answer rejections during QA generation."""
    def __init__(self):
        self.code_execution_failed = 0
        self.low_score_rejected = 0
        self.invalid_answer_rejected = 0  # None, empty, or n/a values
        self.total_attempts = 0
        self.successful = 0
        
        # Detailed tracking for low score rejections
        self.low_score_details = []  # List of dicts with rejection details
        
    def record_code_failure(self):
        """Record a code execution failure."""
        self.code_execution_failed += 1
        self.total_attempts += 1
        
    def record_low_score(self, direct_answer=None, code_answer=None, score=None, question=None):
        """
        Record a rejection due to low validation score.
        
        Args:
            direct_answer: The direct LLM-generated answer
            code_answer: The code-extracted answer
            score: The validation score (0.0-1.0)
            question: The question being answered
        """
        self.low_score_rejected += 1
        self.total_attempts += 1
        
        # Store detailed information about this rejection
        detail = {
            "direct_answer": str(direct_answer) if direct_answer is not None else None,
            "code_answer": str(code_answer) if code_answer is not None else None,
            "score": score,
            "question": str(question) if question is not None else None,
        }
        self.low_score_details.append(detail)
        
    def record_invalid_answer(self):
        """Record a rejection due to invalid answer (None, empty, n/a)."""
        self.invalid_answer_rejected += 1
        self.total_attempts += 1
        
    def record_success(self):
        """Record a successful answer generation."""
        self.successful += 1
        self.total_attempts += 1
        
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of rejection statistics."""
        return {
            "total_attempts": self.total_attempts,
            "successful": self.successful,
            "rejected": {
                "code_execution_failed": self.code_execution_failed,
                "low_score_rejected": self.low_score_rejected,
                "invalid_answer_rejected": self.invalid_answer_rejected,
                "total_rejected": (self.code_execution_failed + 
                                  self.low_score_rejected + 
                                  self.invalid_answer_rejected)
            },
            "success_rate": (self.successful / self.total_attempts * 100 
                           if self.total_attempts > 0 else 0.0),
            "low_score_details": self.low_score_details
        }
        
    def print_summary(self, prefix: str = ""):
        """Print a formatted summary of rejection statistics."""
        summary = self.get_summary()
        print(f"\n{prefix}Answer Validation Statistics:")
        print(f"{prefix}  Total Attempts:        {summary['total_attempts']}")
        print(f"{prefix}  Successful:            {summary['successful']}")
        print(f"{prefix}  Rejected Total:        {summary['rejected']['total_rejected']}")
        print(f"{prefix}    - Code Exec Failed:  {summary['rejected']['code_execution_failed']}")
        print(f"{prefix}    - Low Score:         {summary['rejected']['low_score_rejected']}")
        print(f"{prefix}    - Invalid Answer:    {summary['rejected']['invalid_answer_rejected']}")
        print(f"{prefix}  Success Rate:          {summary['success_rate']:.1f}%")
        
        # Show sample of low score rejections if any
        if self.low_score_details:
            print(f"\n{prefix}Low Score Rejection Examples (showing up to 3):")
            for i, detail in enumerate(self.low_score_details[:3], 1):
                print(f"{prefix}  Example {i}:")
                print(f"{prefix}    Score: {detail['score']}")
                if detail['question']:
                    q_preview = detail['question'][:100] + "..." if len(detail['question']) > 100 else detail['question']
                    print(f"{prefix}    Question: {q_preview}")
                if detail['direct_answer']:
                    ans_preview = detail['direct_answer'][:80] + "..." if len(detail['direct_answer']) > 80 else detail['direct_answer']
                    print(f"{prefix}    Direct Answer: {ans_preview}")
                if detail['code_answer']:
                    ans_preview = detail['code_answer'][:80] + "..." if len(detail['code_answer']) > 80 else detail['code_answer']
                    print(f"{prefix}    Code Answer: {ans_preview}")


OpenRoadBackground = """In modern digital design flows, the design flow is from synthesis, floorplan, placement, cts (clo,ck tree synthesis), routing, and final stages. The EDA tools emit detailed log files throughout this process, capturing a hierarchy of stepsâfrom synthesis, floorplan, placement, cts, routing, and final. These logs serve as a chronological record.

A typical design flow log file includes three core elements:
1. ImportantMessage that include multi-line or single line messages related to commands, warnings, errors, buffer insertion, and any related to timing metrics (TNS, WNS), HPWL, total wirelength, area, overflow, density, powre, and IR drop.
2. Metric tables reporting timing (Worst Negative Slack, Total Negative Slack), design density, resource usage, overflow, HPWL, total wirelength, area, runtime, and memory for each iteration.
3. Stage markers and hierarchy, which reveal where subâflows (e.g., Synthesis -> Floorplan -> GlobalPlace â DetailedPlace â CTS -> Route) occur and help pinpoint bottlenecks or time degradations. The stages should be chronically ordered in the final schema.
4. The synthesis information from YOSYS contains the YOSYS github version, and the warning and error messages. 
5. The HPWL is estimated from floorplan and optimized in the placement stages. The final HPWL estimation is usually reported in detailed placement stage (i.e., 3_5_place_dp). In the routing stage, the actual wirelength is reported.

Below is the stage mapping:
synthesis: synthesis stage

Floorplan stages:
2_1_floorplan: floorplan stage
2_2_floorplan_macro: floorplan macro stage
2_3_floorplan_tapcell: floorplan tapcell stage
2_4_floorplan_pdn: floorplan pdn stage

Placement stages:
3_1_place_gp_skip_io: skip io stage (pre-global placement)
3_2_place_iop: io placement stage (pre-global placement)
3_3_place_gp: main global placement stage
3_4_place_resized: resized placement stage
3_5_place_dp: detailed placement stage

Clock tree synthesis (CTS) stages:
4_1_cts: clock tree synthesis stage

Routing stages:
5_1_grt: global routing stage
5_2_route: detailed routing stage
5_3_fillcell: fillcell stage

Final stages:
6_1_fill: fill stage (final design stage report final area, wirelength, and power)

# Metrics Knowledge Base:
Metrics are evaluated across multiple stages:
- WNS: Worst Negative Slack
- TNS: Total Negative Slack
- Area: Design area
- Utilization: Design utilization

Buffer insertion and removal: Usually happend in the clock tree synthesis (CTS) and placement stage.
HPWL metric: It is estimated from placement stage.
Wirelength metric: It is only available after the routing stage.
DRV violation metric: Design Rule Violation only happened in the detailed routing stage. 
"""

QuestionExecutionBackground = """[Question Examples from Designers]: Execution successful? Hang? Error? What Errors are in the flow (List the error messages and the line number)? Is there any subdesigns? How many synthesis runs in the log file (reply with the subdesigns and the number of synthesis runs)?  What technology node is the runs?
[Rules]:
Do not generate questions about the line range of the stages.
Do not generate questions about the buffer count in the optimization process since it is changing and not captured will in the metrics.
For the error messages, you need to ask listing the error messages and its line number in the flow execution summary.
"""

QuestionSingleStageBackground = """[Question Examples from Designers]: Focus question related to the resource, settings,and PPA metrics of the <substage>.
1. Execution successful? Hang? Error? What Error? Is there any subdesigns? How many synthesis runs in the log file? 
2. Numerical Metric Comparison Type Question, and the Question need to be open-ending, such as "How the timing progression in the <substage>?"
   The interesting metrics are Density, Timing Metric (WNS, TNS), Design violations, Progression of metrics.
3. Open-ending Question: What is the timing metric progression in the optimization process of <substage>? 
4. How many timing_repair operations are committed in the optimization process of <substage> and what are the improvements? 
5. How many clock nets, and their corresponding sinks, leaf buffers, average sink wire length, path depth in the initial clock tree synthesis (CTS)? 
6. In the final stage, what requirements in the test_metrics field are not met? (e.g., WNS, TNS, DRV violations, etc.) What is the top <k> largest gaps that met and not met the requirements?
7. If there is report metrics (rpt_metrics) available, what are the top paths that violate the setup/hold timing constraints?
8. If there is report metrics (rpt_metrics) available, what are the power differences between dynamic and static power?

[Rules]:
Do not generate questions about the line range of the stages.
Do not generate questions about the buffer count in the optimization process since it is changing and not captured will in the metrics.
"""

QuestionMultiStageBackground = """[Question Examples from Designers]: Focus question related to the resource, settings, and PPA metrics.
1. Numerical Metric Comparison Type Question, and the Question need to be open-ending, such as "How the timing progression in the <substage>?"
   The interesting metrics are Density, Timing Metric (WNS, TNS), Design violations, Progression of metrics.
3. Open-ending Question: What is the timing metric progression in the optimization process of <substage1> and <substage2>? 
4. How many timing_repair operations are committed in the optimization process of <substage1> and <substage2> and what are the changes? 
4. Analyze the metric change and gap across design stages
   for example, you can treat this is the first question, and its report can goes to the next round such as asking for degradation or correlation gap analysis
   What is the hpwl and wirelength gap from the <substage> to the final stage?
5. What are the differences between two <substage> for different subdesigns?
6. Show the timing progression metric from <substage> to the <substage>? List all the changes in each stage.
7. In the final stage, what requirements in the test_metrics field are not met? (e.g., WNS, TNS, DRV violations, etc.) What is the top <k> largest gaps that met and not met the requirements?
8. If there is report metrics (rpt_metrics) available, list the vilolation paths that violate the setup/hold timing constraints across stages?
9. If there is report metrics (rpt_metrics) available, what are the power differences between two <substage>?

[Rules]:
Do not generate questions about the line range of the stages.
Do not generate questions about the buffer count in the optimization process since it is changing and not captured will in the metrics.
"""

QuestionConsecutiveStageBackground = """[Question Examples from Designers]: Focus question related to the resource, settings, and PPA metrics.
1. What is the timing metric progression in <substage>? 
2. What optimization committed in <substage> and remaining DRV violations? 
4. Analyze the metric change and gap across design stages
   for example, you can treat this is the first question, and its report can goes to the next round such as asking for degradation or correlation gap analysis
   What is the hpwl and wirelength gap from the <substage> to the final stage?
3. Open-ending Question: What is the timing metric progression in the optimization process of <substage1> and <substage2>? 
4. How many timing repair operations are committed in the optimization process of <substage1> and <substage2> and what are the changes? 
5. What are the progressions of the timing metrics across the whole <substage> to final stage? Make sure the metrics can be found and compared across multiple stages.
6. In the final stage, what requirements in the test_metrics field are not met? (e.g., WNS, TNS, DRV violations, etc.) What is the top <k> largest gaps that met and not met the requirements?
8. If there is report metrics (rpt_metrics) available, list the vilolation paths that violate the setup/hold timing constraints across stages?
9. If there is report metrics (rpt_metrics) available, what are the power differences between two <substage>?

[Rules]:
Do not generate questions about the line range of the stages.
Do not generate questions about the buffer count in the optimization process since it is changing and not captured will in the metrics.
"""

class QAGenerationError(RuntimeError):
    pass


def _read_json(path: str | Path) -> Any:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_aggregate(payload: Any) -> Dict[str, Any]:
    """
    Normalize the aggregate `*_all_stages_metrics.json` payload to a dict with:
      - design_name: str
      - stages: List[Dict]
    """
    if not isinstance(payload, dict):
        raise QAGenerationError("Expected dict JSON payload.")
    if "stages" not in payload or not isinstance(payload.get("stages"), list):
        raise QAGenerationError("Expected aggregate JSON to contain a top-level 'stages' list.")
    return payload


def _extract_code_block(text: str) -> Optional[str]:
    """Extract Python code from markdown code blocks or plain code."""
    # Try to find ```python ... ``` blocks
    python_block = re.search(r'```python\s+(.*?)\s+```', text, re.DOTALL)
    if python_block:
        return python_block.group(1).strip()
    
    # Try to find ``` ... ``` blocks
    generic_block = re.search(r'```\s+(.*?)\s+```', text, re.DOTALL)
    if generic_block:
        return generic_block.group(1).strip()
    
    # If no code blocks, assume the entire text is code
    return text.strip()


def validate_extract_answer(answer: Any) -> bool:
    """
    Recursively validate that an answer does not contain None or "n/a" values.
    
    Args:
        answer: The answer value to validate (can be dict, list, or any type)
    
    Returns:
        True if the answer is valid (no None or "n/a" values), False otherwise
    """
    if answer is None:
        return False
    
    # Check for string "n/a" (case-insensitive)
    if isinstance(answer, str):
        if answer.lower() in ["n/a", "na", "none", ""]:
            return False
        return True
    
    # Recursively check dictionaries
    if isinstance(answer, dict):
        if not answer:  # Empty dict
            return False
        for key, value in answer.items():
            if not validate_extract_answer(value):
                return False
        return True
    
    # Recursively check lists
    if isinstance(answer, list):
        if not answer:  # Empty list
            return False
        for item in answer:
            if not validate_extract_answer(item):
                return False
        return True
    
    # All other types (numbers, booleans, etc.) are valid
    return True

def validate_direct_vs_code_answer(
    direct_answer: str,
    code_answer: Any,
    question: str,
    tolerance: float = 0.01
) -> Dict[str, Any]:
    """
    Validate and compare direct LLM answer with code-generated answer.
    
    Args:
        direct_answer: Answer generated directly by LLM (string)
        code_answer: Answer extracted by generated code (can be any type)
        question: The original question being answered
        tolerance: Relative tolerance for numeric comparisons (default: 1%)
    
    Returns:
        Dict with validation results:
        - matches: bool - whether answers are considered equivalent
        - direct_answer: normalized direct answer
        - code_answer: normalized code answer
        - comparison_notes: details about the comparison
        - validation_method: how the answers were compared
    """
    import re

    system_prompt = f"""You are an expert QA evaluator for chip design metrics. Evaluate if two answers are equivalent and consistent.

SCORING GUIDELINES:
- Score 1.0: Answers contain the SAME INFORMATION (format differences are OK)
- Score 0.9: Answers are semantically equivalent with minor format differences
- Score 0.8: Answers match on key metrics but have minor discrepancies in detail
- Score 0.7: Answers are related but one has more/less detail
- Score 0.5: Answers address the same topic but with different focus
- Score 0.3: Answers are partially related but miss key information
- Score 0.0: Answers address completely different topics or metrics

IMPORTANT:
- Focus on whether answers ADDRESS THE SAME QUESTION, not format
- If Direct answer discusses timing (WNS/TNS) but Code answer discusses area → Score 0.0
- If both answers provide the same numbers with different units/format → Score 1.0
- Numeric equivalence: "0.233 units" == "0.233" → Score 1.0
- Structure equivalence: {{"area": "17171.896 um^2"}} == "17171.896 um^2" → Score 1.0
"""
    user_prompt = f"""
Question: {question}

Direct LLM answer: {direct_answer}
Code-generated answer: {code_answer}

Evaluate if these answers are equivalent. Return score in JSON format:
```json
{{
    "score": <0.0-1.0>,
    "reasoning": "<brief explanation focusing on whether answers address the same metrics>"
}}
```
"""

    
    # use llm judges call dar gateway
    judge_response = call_dar_gateway(
        system_prompt=system_prompt, user_prompt=user_prompt,
        temperature=0.1
    )
    # extract the score from the judge response
    # get the json block from the judge response
    json_block = re.search(r'```json\s*(.*?)\s*```', judge_response, re.DOTALL)
    if json_block:
        score = json.loads(json_block.group(1).strip())["score"]
    else:
        raise QAGenerationError("No score found in the judge response")
    print(f"Judge response: {judge_response}")
    print(f"Score: {score}")
    return score

def sample_stages(
    aggregate_metrics_json_path: str | Path,
    *,
    num_stages: Optional[int] = None,
    seed: Optional[int] = None,
    stage_types: Optional[List[str]] = None,
    min_stage_separation: int = 5,
) -> List[Dict[str, Any]]:
    """
    Sample 2 or 3 stages from the aggregate metrics JSON with guaranteed execution order.
    
    Automatically samples between two variant types:
    - Type 1: Different sub-designs, SAME stage type
      Example: Compare 'synthesis' stage across uart_rx, uart_tx, etc.
      Use case: "How does synthesis perform across different sub-designs?"
    
    - Type 2: Same sub-design, DIFFERENT stages
      Example: Compare synthesis→placement→routing for uart_rx
      Use case: "How do metrics progress through the flow for uart_rx?"
    
    The function randomly chooses between Type 1 and Type 2 based on data availability.
    
    Sampling strategy:
    1. Randomly decide Type 1 or Type 2
    2. Type 1: Randomly select one stage type, sample from different sub-designs
    3. Type 2: Randomly select one sub-design, sample different stages from it
    4. Enforce minimum separation between stages (e.g., 2nd stage >= 6th in sequence)
    
    Args:
        aggregate_metrics_json_path: Path to `*_all_stages_metrics.json`
        num_stages: Number of stages to sample (2 or 3). If None, randomly choose 2 or 3.
        seed: Optional random seed for reproducibility
        stage_types: Optional list of stage types to filter by (e.g., ['placement', 'routing'])
        min_stage_separation: Minimum sequence_number gap between sampled stages (default: 5)
                             For example, if 1st sampled stage is at sequence 1, 
                             2nd stage must be at sequence >= 6 (1+5)
    
    Returns:
        List of sampled stage dictionaries sorted by execution order (sequence_number), each containing:
        - sequence_number: Position in the execution flow (1-based)
        - stage: Stage type (e.g., 'synthesis', 'placement')
        - occurrence: Occurrence number (1-based, for hierarchical/multi-block designs)
        - sub_design_name: Sub-design name
        - design_name: Top-level design name
        - metrics: Stage metrics
        - metrics_file: Path to metrics file
        - Other stage metadata
    
    Raises:
        QAGenerationError: If insufficient stages available for sampling with constraints
    
    Examples:
        >>> # Automatically samples Type 1 or Type 2
        >>> stages = sample_stages("design_all_stages_metrics.json", num_stages=2, seed=42)
        >>> # Could return Type 2:
        >>> print([(s['sub_design_name'], s['stage'], s['sequence_number']) for s in stages])
        [('uart_rx', 'synthesis', 1), ('uart_rx', 'routing', 7)]
        >>> # Or could return Type 1:
        >>> print([(s['sub_design_name'], s['stage'], s['sequence_number']) for s in stages])
        [('uart_rx', 'synthesis', 1), ('uart_tx', 'synthesis', 8)]
    """
    rng = random.Random(seed)
    
    # Determine number of stages to sample
    if num_stages is None:
        num_stages = rng.choice([2, 3])
    elif num_stages not in [2, 3]:
        raise ValueError("num_stages must be 2 or 3")
    
    payload = _normalize_aggregate(_read_json(aggregate_metrics_json_path))
    all_stages = payload.get("stages") or []
    
    # Filter candidates - only stages with metrics
    candidates: List[Dict[str, Any]] = []
    for s in all_stages:
        if not isinstance(s, dict):
            continue
        
        # Filter by stage_types if specified
        if stage_types and s.get("stage") not in stage_types:
            continue
        
        # Only include stages with metrics
        if s.get("metrics"):
            candidates.append(s)
    
    if len(candidates) < num_stages:
        raise QAGenerationError(
            f"Not enough stages available. Need {num_stages}, but only {len(candidates)} found."
        )
    
    from collections import defaultdict
    
    # Analyze what's possible: Type 1 (same stage, different sub-designs) or Type 2 (same sub-design, different stages)
    
    # Check Type 1 feasibility: Different sub-designs, SAME stage type
    stages_by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in candidates:
        stage_type = s.get("stage")
        if stage_type:
            stages_by_type[stage_type].append(s)
    
    valid_stage_types = {}
    for stage_type, stages in stages_by_type.items():
        # Get unique sub-designs for this stage type
        subdesigns = set(s.get("sub_design_name") for s in stages if s.get("sub_design_name"))
        if len(subdesigns) >= num_stages:
            valid_stage_types[stage_type] = stages
    
    # Check Type 2 feasibility: Same sub-design, DIFFERENT stages
    stages_by_subdesign: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in candidates:
        subdesign = s.get("sub_design_name")
        if subdesign:
            stages_by_subdesign[subdesign].append(s)
    
    valid_subdesigns = {
        name: stages 
        for name, stages in stages_by_subdesign.items() 
        if len(stages) >= num_stages
    }
    
    # Randomly decide which type to use based on availability
    can_do_type1 = bool(valid_stage_types)
    can_do_type2 = bool(valid_subdesigns)
    
    if not can_do_type1 and not can_do_type2:
        raise QAGenerationError(
            f"Cannot satisfy either Type 1 or Type 2 constraints. "
            f"Need either: (1) same stage across {num_stages} different sub-designs, "
            f"or (2) {num_stages} different stages within same sub-design."
        )
    
    # Randomly choose between available types
    if can_do_type1 and can_do_type2:
        use_type1 = rng.choice([True, False])
        # use_type1 = True
    elif can_do_type1:
        use_type1 = True
    else:
        use_type1 = False
    
    # Apply the selected type
    if use_type1:
        # Type 1: Different sub-designs, SAME stage type
        selected_stage_type = rng.choice(list(valid_stage_types.keys()))
        candidates = valid_stage_types[selected_stage_type]
        candidates.sort(key=lambda s: s.get("sequence_number", 0))
        print(f"[sample_stages] Type 1: Selected stage type: {selected_stage_type}")
        same_stage = True
    else:
        # Type 2: Same sub-design, DIFFERENT stages
        selected_subdesign = rng.choice(list(valid_subdesigns.keys()))
        candidates = valid_subdesigns[selected_subdesign]
        candidates.sort(key=lambda s: s.get("sequence_number", 0))
        print(f"[sample_stages] Type 2: Selected sub-design: {selected_subdesign}")
        same_stage = False
    
    # Sample stages with minimum separation constraint using a unified loop-based approach
    # This works for any num_stages (2, 3, or more)
    sampled_stages: List[Dict[str, Any]] = []
    
    # Track already used sub-designs (for Type 1: same_stage=True)
    used_subdesigns = set()
    
    # Iteratively sample each stage with separation constraint
    for i in range(num_stages):
        if i == 0:
            # First stage: can be any stage from candidates
            eligible = candidates
        else:
            # Subsequent stages: must be separated from the previous stage
            prev_seq = sampled_stages[-1].get("sequence_number", 0)
            
            # Try with minimum separation constraint
            eligible = [
                s for s in candidates 
                if s.get("sequence_number", 0) >= prev_seq + min_stage_separation
            ]
            
            # If constraint cannot be satisfied, relax to just require seq > prev_seq
            if not eligible:
                eligible = [
                    s for s in candidates 
                    if s.get("sequence_number", 0) > prev_seq
                ]
            
            # For Type 1 (same_stage), ensure different sub-designs
            if same_stage and eligible:
                eligible = [
                    s for s in eligible
                    if s.get("sub_design_name") not in used_subdesigns
                ]
            
            # If still no eligible stages, raise error
            if not eligible:
                constraint_desc = "same stage type" if same_stage else f"same sub-design"
                raise QAGenerationError(
                    f"Cannot find stage {i+1} after sequence {prev_seq} "
                    f"for {constraint_desc}"
                )
        
        # Randomly select from eligible stages
        selected_stage = rng.choice(eligible)
        sampled_stages.append(selected_stage)
        
        # Track used sub-design for Type 1
        if same_stage:
            used_subdesigns.add(selected_stage.get("sub_design_name"))
    
    # Sort by sequence_number to guarantee execution order (should already be sorted, but ensure)
    sampled_stages.sort(key=lambda s: s.get("sequence_number", 0))
    
    # Log the sampled stages with all key information for debugging
    if sampled_stages:
        for idx, stage in enumerate(sampled_stages, 1):
            seq = stage.get("sequence_number", "N/A")
            stage_type = stage.get("stage", "unknown")
            subdesign = stage.get("sub_design_name", "N/A")
            occ = stage.get("occurrence", "N/A")
            print(f"[sample_stages]   Stage {idx}: seq={seq}, type={stage_type}, sub_design={subdesign}, occurrence={occ}")
    
    return sampled_stages


def select_stage_and_final_with_same_subdesign(
    aggregate_metrics_json_path: str | Path,
    *,
    seed: Optional[int] = None,
    stage_type: Optional[str] = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Select a stage and the final stage that have the same sub_design_name.
    
    This is useful for comparing intermediate stage metrics with final design metrics,
    ensuring they belong to the same sub-design (hierarchical block).
    
    Args:
        aggregate_metrics_json_path: Path to `*_all_stages_metrics.json`
        seed: Optional random seed for reproducibility
        stage_type: Optional stage type to select (e.g., 'placement', 'routing').
                   If None, randomly select from available stages.
    
    Returns:
        Tuple of (selected_stage, final_stage), where:
        - selected_stage: A non-final stage dictionary
        - final_stage: The final stage dictionary with matching sub_design_name
        
        Both dictionaries contain:
        - stage: Stage type
        - occurrence: Occurrence number
        - sub_design_name: Sub-design name (guaranteed to be the same)
        - metrics: Stage metrics
        - sequence_number: Order in the flow
        - Other stage metadata
    
    Raises:
        QAGenerationError: If no matching stage-final pair found
    
    Example:
        >>> stage, final = select_stage_and_final_with_same_subdesign(
        ...     "design_all_stages_metrics.json",
        ...     stage_type="placement",
        ...     seed=42
        ... )
        >>> print(f"Comparing {stage['stage']} with {final['stage']}")
        >>> print(f"Both belong to sub-design: {stage['sub_design_name']}")
        Comparing placement with final
        Both belong to sub-design: uart_rx
    """
    rng = random.Random(seed)
    
    payload = _normalize_aggregate(_read_json(aggregate_metrics_json_path))
    all_stages = payload.get("stages") or []
    
    # Separate final stages from other stages
    final_stages: List[Dict[str, Any]] = []
    non_final_stages: List[Dict[str, Any]] = []
    
    for s in all_stages:
        if not isinstance(s, dict) or not s.get("metrics"):
            continue
        
        stage = s.get("stage")
        if stage == "final":
            final_stages.append(s)
        else:
            non_final_stages.append(s)
    
    if not final_stages:
        raise QAGenerationError("No final stage found in the metrics JSON")
    
    if not non_final_stages:
        raise QAGenerationError("No non-final stages found in the metrics JSON")
    
    # Filter non-final stages by stage_type if specified
    if stage_type:
        non_final_stages = [s for s in non_final_stages if s.get("stage") == stage_type]
        if not non_final_stages:
            raise QAGenerationError(f"No stages of type '{stage_type}' found")
    
    # Group final stages by sub_design_name
    final_by_subdesign: Dict[str, List[Dict[str, Any]]] = {}
    for fs in final_stages:
        subdesign = fs.get("sub_design_name")
        if subdesign:
            if subdesign not in final_by_subdesign:
                final_by_subdesign[subdesign] = []
            final_by_subdesign[subdesign].append(fs)
    
    # Find non-final stages that have matching final stages
    matching_candidates: List[tuple[Dict[str, Any], Dict[str, Any]]] = []
    
    for stage in non_final_stages:
        subdesign = stage.get("sub_design_name")
        if subdesign and subdesign in final_by_subdesign:
            # Pick the first (or only) final stage with matching subdesign
            final_stage = final_by_subdesign[subdesign][0]
            matching_candidates.append((stage, final_stage))
    
    if not matching_candidates:
        raise QAGenerationError(
            "No stage-final pairs found with matching sub_design_name. "
            "This may happen if sub-designs are not properly labeled or if there are no final stages."
        )
    
    # Randomly select one matching pair
    selected_stage, final_stage = rng.choice(matching_candidates)
    
    return selected_stage, final_stage

def generate_answer_directly_through_llm(
    question: str, 
    background: str,
    context_data: str,
    design_name: str = "",
    temperature: float = 0.7
) -> str:
    """
    Generate an answer to a question directly through LLM with context data.
    
    Args:
        question: The question to answer
        context_data: JSON string containing the metrics/data to answer from
        design_name: Name of the design (optional)
        temperature: LLM temperature for generation
    
    Returns:
        The direct answer from LLM as a string
    """
    prompt = f"""You are an expert in chip design and EDA tools. Answer the following question based on the provided metrics data.

# OpenROAD EDA log file background:
{OpenRoadBackground}

# Design: {design_name}

# Metrics Data:
```json
{context_data}
```json

# Question: {question}

CRITICAL INSTRUCTIONS:
1. READ THE QUESTION CAREFULLY - Answer ONLY what is asked
2. IDENTIFY KEY METRICS mentioned in the question (e.g., "area", "utilization", "WNS", "TNS", "IR drop", "HPWL", "wirelength", "power")
3. EXTRACT ONLY THE METRICS mentioned in the question from the JSON
4. If the question asks about a specific stage, answer for THAT STAGE ONLY
5. If the question asks about comparison between stages, include data for ALL mentioned stages
6. If data is missing for the requested metric, state "Data not available" instead of providing unrelated metrics
7. Be SPECIFIC with numbers and units from the JSON

FORMAT YOUR ANSWER:
- For numeric questions: Provide exact values with units (e.g., "17171.896 um^2", "-0.234 ns")
- For comparison questions: Provide a structured comparison (e.g., dict or table format)
- For boolean questions: Provide clear yes/no with brief justification

Generate the answer in json format. The json format should be like:
```json
{{
    "answer": "the answer to the question"
}}
```
""" 
    response = llm_inference(
        system_prompt="You are an expert in chip design and EDA tools, generating precise, focused answers that directly address what is asked in the question. DO NOT provide timing metrics when asked about area, or vice versa.",
        user_prompt=prompt,
        temperature=temperature
    )
    return response


def flow_execution_qa_pair(
    aggregate_metrics_json_path: str | Path,
    *,
    seed: Optional[int] = None,
    temperature: float = 0.7,
    is_direct_llm_answer: bool = True,
    rejection_stats: Optional[RejectionStats] = None,
) -> Dict[str, Any]:
    """
    Generate QA pairs from the `flow_execution_summary` key in the metrics JSON.
    
    These questions focus on high-level execution aspects like:
    - Execution success/failure
    - Stage completion
    - Sub-designs
    - Overall flow structure
    - Error detection
    
    Args:
        aggregate_metrics_json_path: path to `*_all_stages_metrics.json`
        seed: optional random seed for reproducibility
        temperature: LLM temperature for generation (default: 0.7)
    
    Returns:
        A dict with the question set and answers for flow execution.
        Key fields:
        - id: Question ID (assigned by caller)
        - difficulty: Difficulty level
        - stage: "flow_execution" (special marker)
        - occurrence: 1 (always)
        - qa_pairs: List of question-answer pairs
        - source_path: Path to the source metrics
        - design_name: Design metadata
        - misc_info: Additional info for validation
    """
    rng = random.Random(seed)
    
    payload = _normalize_aggregate(_read_json(aggregate_metrics_json_path))
    
    # Extract flow_execution_summary
    flow_summary = payload.get("flow_execution_summary", {})
    if not flow_summary:
        raise QAGenerationError("No 'flow_execution_summary' found in the metrics JSON.")
    
    design_name = flow_summary.get("design_name") or payload.get("design_name", "unknown")
    
    # Convert to compact JSON string
    flow_summary_str = json.dumps(flow_summary, indent=2)
    
    # Step 1: Use LLM to generate questions about flow execution
    print(f"\n[QA Generation] Generating flow execution questions for design={design_name}...")
    question_prompt = f"""You are generating a question-answer pair for an OpenROAD EDA design flow benchmark.

# OpenROAD EDA log file background:
{OpenRoadBackground}

# Question Background:
{QuestionExecutionBackground}

Given the following flow execution summary from a chip design flow:

Design: {design_name}
Flow Execution Summary JSON:
```json
{flow_summary_str}
```

Generate a set of interesting, specific questions about the flow execution that can be answered using the summary data. Try to explore different questions from different aspects compared to the question examples.

Focus on:
- Execution success/completion status
- Number and types of stages executed
- Sub-designs involved
- Stage sequence and coverage
- Any potential issues or anomalies


IMPORTANT RULES:
- DO NOT use technical stage identifiers like "3_1", "3_2", "3_3", "2_1_floorplan", "3_5_place_dp", etc. in questions
- Use natural language stage names: "synthesis", "floorplan", "placement", "CTS", "routing", "final" instead
- Refer to stages descriptively: "the placement stage", "during routing", "in the CTS phase"
- If referring to sub-stages, use descriptions like "global placement", "detailed placement" instead of numbers
- Do not ask questions about how the transition from one stage to another, such as "how the transition from placement to routing"? This is already designed by the user. 
Generate questions between 3~6 questions. You don't need to always generate 6 questions.
"""
    
    output_format = """Output the question between ```question and ``` tags.
```question
<Q1>...
Difficulty: <easy, medium>
```
    
```question
<Q2>...
Difficulty: <easy, medium>
```
...
```question
<Qn>...
Difficulty: <easy, medium>
```
"""
    # print(f"Question prompt: {question_prompt}")
    # pdb.set_trace()
    question = llm_inference(
        system_prompt="You are an expert in chip design and EDA tools, specializing in creating educational QA pairs about design flow execution.",
        user_prompt=question_prompt + "\n" + output_format,
        temperature=temperature
    )
    
    # print(f"Generated question: {question}")
    
    # Parse the questions; use regex to find the question between ```question and ``` tags
    questions = re.findall(r"```question\s*(.*?)\s*```", question, re.DOTALL)
    
    # Batch generation disabled
    #print(f"Parsed questions: {questions}")
    # question_str = ""
    # for q in questions:
    #     question_str += f"{q}\n"
    direct_llm_answers = []
    if is_direct_llm_answer:
        for q in questions:
            
            direct_llm_answer = generate_answer_directly_through_llm(
                question=q,
                background=OpenRoadBackground,
                context_data=flow_summary_str, 
                design_name=design_name,
                temperature=0.3  # Lower temperature for more focused, deterministic answers
            )
            print(f"Direct LLM answer: {direct_llm_answer[:200]}...")
            direct_llm_answers.append(direct_llm_answer)
    final_qa_pairs = []
    for i in range(len(questions)):
        question_str = questions[i]
        if is_direct_llm_answer and len(direct_llm_answers) > i:
            direct_llm_answer = direct_llm_answers[i]
            question_str = f"{question_str}\nDirect LLM answer for reference (Try to write code to extract the answer from the flow execution summary JSON): {direct_llm_answer}"
              
        # Step 2: Use LLM to generate Python code to extract the answer
        # print(f"[QA Generation] Generating extraction code...")
        code_prompt = """You are generating Python code to extract an answer from flow execution summary JSON.

    Design: """ + design_name + """
    Question: """ + question_str + """

    Flow Execution Summary (available as variable `flow_summary` - already a Python dict):
    ```json
    """ + flow_summary_str + """
    ```

    Write Python code that extracts the answer to the question from the `flow_summary` dict.

    Requirements:
    - The variable `flow_summary` is already loaded as a Python dict (NOT a string)
    - Keep the code simple and robust
    - Return answers directly as values (numbers, strings, booleans, lists, dicts, etc.)
    - For questions about stage types, return lists of stage names
    - For questions about counts, return integers
    - For questions about success/completion, return booleans or descriptive strings

    Example code patterns:
    ```python
    def extract_answer(flow_summary: Dict[str, Any], questions: list[str]) -> list:
        # flow_summary is already a dict - no need to parse
        
        qa_pairs = []
        for question in questions:
            # IMPORTANT: Match the answer to what the question is asking
            # If question asks "how many", return a number
            # If question asks "what are the types", return a list of strings
            # If question asks both, return a dict with both pieces of info
            
            answer = None  # Initialize
            
            # Check what the question is asking and extract accordingly
            question_lower = question.lower()
            
            # Example 1: Count stages
            if "how many stages" in question_lower or "total number of stages" in question_lower:
                answer = flow_summary.get('total_stages', 0)
            
            # Example 2: List stage types
            elif "what are the types" in question_lower or "stage types" in question_lower:
                answer = [stage.get('stage_type') for stage in flow_summary.get('stages', [])]
            
            # Example 3: Both count and types
            elif "how many stages" in question_lower and ("types" in question_lower or "what are" in question_lower):
                answer = {
                    'total_stages': flow_summary.get('total_stages', 0),
                    'stage_types': [stage.get('stage_type') for stage in flow_summary.get('stages', [])]
                }
            
            # Example 4: Get unique sub-designs
            elif "sub-design" in question_lower or "subdesign" in question_lower:
                sub_designs = list(set(stage.get('sub_design_name') for stage in flow_summary.get('stages', []) if stage.get('sub_design_name')))
                if "how many" in question_lower:
                    answer = len(sub_designs)
                else:
                    answer = sub_designs
            
            # Example 5: Check completion status
            elif "complete" in question_lower or "successful" in question_lower:
                answer = flow_summary.get('total_stages', 0) > 0
            
            # Example 6: Stage sequence (ordered list)
            elif "sequence" in question_lower or "order" in question_lower:
                answer = [s.get('stage_type') for s in sorted(flow_summary.get('stages', []), key=lambda x: x.get('sequence_number', 0))]
            
            # Add your answer extraction logic here based on the question
            # Make sure answer is not None or empty string
            
            qa_pairs.append({"question": question, "answer": answer})    
            
        return qa_pairs
    ```

    CRITICAL RULES:
    1. Read the question carefully and extract the appropriate answer type
    2. If question asks "how many", return an integer count
    3. If question asks "what are" or "list", return a list or dict with the data
    4. If question asks multiple things, return a dict with all requested information
    5. Never return None, True/False (unless explicitly asking yes/no), or empty values
    6. Match your answer format to what the question is asking for

    Output ONLY the Python code, nothing else (no markdown, no explanations)."""
        print(f"Code prompt: {code_prompt}")
        # pdb.set_trace()
        
        code_response = llm_inference(
            system_prompt="You are an expert Python programmer specializing in data extraction from JSON structures.",
            user_prompt=code_prompt,
            temperature=0.1  # Lower temperature for code generation
        )
        
        # Extract code from potential markdown blocks
        code = _extract_code_block(code_response)
        print(f"Generated code:\n{code}\n")
        
        # Step 3: Execute the code to get the answer with retry mechanism
        drop_qa_idx = []
        max_retries = 3
        retry_count = 0
        execution_successful = False
        
        try:
            # use one question at a time to extract the answer
            local_vars = {"flow_summary": flow_summary, "questions": [question_str], "json": json}
            # Execute the code to define the function
            exec(code, {"json": json}, local_vars)
                
            # Check if extract_answer function was defined
            if "extract_answer" in local_vars:
                # Call the function to get qa_pairs
                qa_pairs = local_vars["extract_answer"](flow_summary, [question_str])
                print(f"Extracted {len(qa_pairs)} QA pairs")
                    
                # Display preview of each answer
                for i, qa in enumerate(qa_pairs):
                    if "answer" in qa:
                        answer = qa["answer"]
                        if answer is None or answer in ["", "None", "null", "NoneType"]:
                            print(f"  Q{i+1}: Warning - no answer or invalid answer")
                            if rejection_stats:
                                rejection_stats.record_invalid_answer()
                            # drop the qa pair
                            # drop_qa_idx.append(i)
                            continue
                        # Get a preview of the answer
                        answer_preview = str(answer)[:100]
                        answer_type = type(answer).__name__
                        # get the score of the answer by validating the answer with the direct LLM answer
                        score = validate_direct_vs_code_answer(answer, direct_llm_answer, question_str)
                        if score > 0.75:
                            print(f"  Q{i+1}: [{answer_type}] {answer_preview}...")
                            print(f"  Q{i+1}: Score: {score} - Valid answer")
                            final_qa_pairs.append(qa)
                            if rejection_stats:
                                rejection_stats.record_success()
                        else:
                            print(f"  Q{i+1}: Score: {score} - Invalid answer")
                            if rejection_stats:
                                rejection_stats.record_low_score(
                                    direct_answer=direct_llm_answer,
                                    code_answer=answer,
                                    score=score,
                                    question=question_str
                                )
                            continue
                    else:
                        print(f"  Q{i+1}: Warning - no answer field")
                        if rejection_stats:
                            rejection_stats.record_invalid_answer()
                        # drop_qa_idx.append(i)
            else:
                raise QAGenerationError("Extraction code did not define an 'extract_answer' function")
                
        except Exception as e:
            print(f"[ERROR] Code execution failed (attempt {retry_count}/{max_retries}): {e}")
            if rejection_stats:
                rejection_stats.record_code_failure()
    
    # drop the qa pairs: no need to drop now since we are using one question at a time
    # final_qa_pairs = [qa for i, qa in enumerate(qa_pairs) if i not in drop_qa_idx]
    # Build question object
    return {
        "id": None,  # caller can assign later (e.g., Q001)
        "stage": "flow_execution",  # Special marker for flow-level questions
        "occurrence": 1,
        "qa_pairs": final_qa_pairs,
        "source_path": "flow_execution_summary (generated by LLM)",
        # Helpful metadata
        "design_name": design_name,
        "sub_design_name": None,  # N/A for flow-level questions
        "metrics_file": None,  # N/A for flow-level questions
        "log_file": payload.get("log_file"),
        # Misc info for validation and debugging
        "misc_info": {
            "extraction_code": code,
            "sampled_stage_info": {
                "stage": "flow_execution",
                "occurrence": 1,
                "design_name": design_name,
                "total_stages": flow_summary.get("total_stages", 0),
                "stage_types": [s.get("stage_type") for s in flow_summary.get("stages", [])],
            },
            "generation_params": {
                "temperature": temperature,
                "seed": seed,
            },
            "metrics_summary": {
                "metrics_json_length": len(flow_summary_str),
                "metrics_truncated": len(flow_summary_str) > 8000,
                "num_questions_generated": len(questions),
            },
        },
    }


def single_stage_qa_pair(
    aggregate_metrics_json_path: str | Path,
    *,
    seed: Optional[int] = None,
    stage: Optional[str] = None,
    occurrence: Optional[int] = None,
    temperature: float = 0.7,
    is_direct_llm_answer: bool = True,
    rejection_stats: Optional[RejectionStats] = None,
) -> Dict[str, Any]:
    """
    Sample ONE stage occurrence from an `*_all_stages_metrics.json` file and
    generate ONE or MORE QA pairs using LLM.

    Args:
        aggregate_metrics_json_path: path to `*_all_stages_metrics.json`
        seed: optional random seed for reproducibility
        stage: optionally constrain to a specific stage type ('synthesis', 'floorplan', ...)
        occurrence: optionally constrain to a specific occurrence (1-based)
        temperature: LLM temperature for generation (default: 0.7)

    Returns:
        A dict with the question set and answers for one stage.
        Key fields:
        - id: Question ID (assigned by caller)
        - difficulty: Difficulty level
        - stage: Stage name (e.g., 'synthesis', 'placement')
        - occurrence: Occurrence number (1-based)
        - qa_pairs: List of question-answer pairs
        - source_path: Path to the source metrics
        - design_name, sub_design_name, metrics_file: Design metadata
        - misc_info: Additional info for validation including:
            - extraction_code: Generated Python code for extracting answers
            - sampled_stage_info: Details about the sampled stage
            - generation_params: Parameters used for generation
            - metrics_summary: Summary of the metrics used
        
        Each qa_pair has the format:
        {
            "question": str,
            "answer": <any>  # Direct answer value (number, string, boolean, list, dict, etc.)
        }
        
        Examples:
        - {"question": "What is the WNS?", "answer": "-0.234 ns"}
        - {"question": "Did it pass DRC?", "answer": False}
        - {"question": "What is the area?", "answer": 12345.67}
    """
    rng = random.Random(seed)

    payload = _normalize_aggregate(_read_json(aggregate_metrics_json_path))
    stages = payload.get("stages") or []

    # Filter candidates
    candidates: List[Dict[str, Any]] = []
    for s in stages:
        if not isinstance(s, dict):
            continue
        st = s.get("stage")
        occ = s.get("occurrence")
        if stage and st != stage:
            continue
        if occurrence is not None:
            try:
                if int(occ) != int(occurrence):
                    continue
            except Exception:
                continue
        # Include all stages that have metrics
        if s.get("metrics"):
            candidates.append(s)

    if not candidates:
        raise QAGenerationError("No stage candidates found that match the constraints.")

    stage_entry = rng.choice(candidates)
    st = stage_entry.get("stage")
    occ = int(stage_entry.get("occurrence", 1) or 1)
    metrics = stage_entry.get("metrics", {})

    # Convert stage_entry to a compact JSON string
    stage_json_str = json.dumps(stage_entry, indent=2)

    # Step 1: Use LLM to generate a question about this stage
    print(f"\n[QA Generation] Generating question for stage={st} occurrence={occ}...")
    
    question_prompt = f"""You are generating a question-answer pair for an OpenROAD EDA design flow benchmark.
# OpenROAD EDA log file background:
{OpenRoadBackground}

Given the following stage metrics from a chip design flow:

Stage: {st}
Metrics JSON:
```json
{stage_json_str}
```

Generate a set of interesting, specific questions about this stage that can be answered using the metrics data. No more than 6 questions.
Try to explore different questions from different aspects compared to the question examples.
Imagine you are a designer and you are trying to understand the flow execution and the design flow. What questions you would like to ask in the given stage metrics information to examine the design flow and the design quality?

CRITICAL REQUIREMENTS FOR QUESTION QUALITY:
1. **BE SPECIFIC** - Ask about SPECIFIC metrics that exist in the JSON
   - Good: "What is the total cell area reported in the final stage?"
   - Bad: "How does the design's area and utilization evolve?" (too broad)

2. **ONE TOPIC PER QUESTION** - Each question should focus on ONE metric or concept
   - Good: "What is the WNS after CTS optimization?"
   - Bad: "How does area, utilization, and timing evolve?" (multiple topics)

3. **CLEARLY IDENTIFY THE STAGE** - If asking about a specific substage, be clear
   - Good: "What is the HPWL after detailed placement?"
   - Bad: "What is the HPWL during placement?" (placement has multiple substages)

4. **ANSWERABLE WITH NUMBERS** - Prefer questions that have concrete numeric answers
   - Good: "What is the IR drop percentage for VDD in the final stage?"
   - Bad: "Are there significant IR drop metrics?" (subjective)

5. **AVOID VAGUE COMPARISONS** - If comparing, specify what to compare
   - Good: "What is the difference between initial and final WNS during floorplan optimization?"
   - Bad: "How do metrics compare?" (too vague)

IMPORTANT RULES:
- DO NOT use technical stage identifiers like "3_1", "3_2", "3_3", "2_1_floorplan", "3_5_place_dp", etc. in questions
- Use natural language stage names: "synthesis", "floorplan", "placement", "CTS", "routing", "final" instead
- Refer to stages descriptively: "the placement stage", "during routing", "in the CTS phase", "at floorplan"
- If referring to sub-stages, use descriptions like "global placement", "detailed placement", "detailed routing" instead of numbers
- Consider asking about optimization_process metrics shown in the parsing examples
- Consider asking about design_metrics_final when available (shown in parsing examples)
- You need to write the code to handle the case that the metrics are not extracted successfully in the general fields for the substage, and then try to use the rpt_metrics for each substage to find the metrics.

""" + QuestionSingleStageBackground

    output_fornat = """Output the question between ```question and ``` tags.
```question
<Q1> ...
Difficulty: <easy, medium, hard>
```
    
```question
<Q2> ...
Difficulty: <easy, medium, hard>
```
...
```question
<Qn> ...
Difficulty: <easy, medium, hard>
```
"""
    
    question = llm_inference(
        system_prompt="You are an expert in chip design and EDA tools, specializing in creating educational QA pairs.",
        user_prompt=question_prompt + "\n" + output_fornat,
        temperature=temperature
    )
    
    # print(f"Generated question: {question}")

    # parsing the question; use regex to find the question between ```question and ``` tags
    questions = re.findall(r"```question\s*(.*?)\s*```", question, re.DOTALL)
    # print(f"Parsed questions: {questions}")
    
    # Check if rpt_metric field exists in the stage data
    include_rpt_metric = "rpt_metric" in stage_json_str
    stage_parsing_examples = get_parsing_examples(st, include_rpt_metrics=include_rpt_metric)
    
    # Generate direct LLM answers if requested
    direct_llm_answers = []
    if is_direct_llm_answer:
        for q in questions:
            direct_llm_answer = generate_answer_directly_through_llm(
                question=q,
                background=OpenRoadBackground,
                context_data=stage_json_str,
                design_name=stage_entry.get("design_name"),
                temperature=0.3  # Lower temperature for more focused, deterministic answers
            )
            print(f"Direct LLM answer: {direct_llm_answer[:200]}...")
            direct_llm_answers.append(direct_llm_answer)
    
    final_qa_pairs = []
    for i in range(len(questions)):
        question_str = questions[i]
        if is_direct_llm_answer and len(direct_llm_answers) > i:
            direct_llm_answer = direct_llm_answers[i]
            question_str = f"{question_str}\nDirect LLM answer for reference (Try to write code to extract the answer from the stage metrics JSON): {direct_llm_answer}"
        # Step 2: Use LLM to generate Python code to extract the answer
        # print(f"[QA Generation] Generating extraction code...")
        # Use raw string concatenation to avoid f-string issues with curly braces
        code_prompt = """You are generating Python code to extract an answer from stage metrics JSON.

Stage: """ + st + """
Question: """ + question_str + """

Metrics (available as variable `stage_entry` - already a Python dict):
```json
""" + stage_json_str + """
```

Write Python code that extracts the answer to the question from the `stage_entry` dict.

Requirements:
- The variable `stage_entry` is already loaded as a Python dict (NOT a string)
- Access substages: substages = stage_entry['metrics'].get('substages', {})
- Access optimization_process: for opt in substage_data.get('optimization_process', [])
- Access design_metrics_final: design_metrics_final = stage_entry['metrics'].get('design_metrics_final', {})
- Return answers with appropriate units (e.g., "-0.234 ns", "12345.67 um")

Define the extract_answer function with stage-specific parsing guidance:
```python
def extract_answer(stage_entry: Dict[str, Any], questions: list[str]) -> list:
    \"\"\"
    Extract answers from stage metrics.
    
    Stage-specific parsing patterns:
""" + stage_parsing_examples + """
    \"\"\"
    metrics = stage_entry.get('metrics', {})
    substages = metrics.get('substages', {})
    design_metrics_final = metrics.get('design_metrics_final', {})
    
    qa_pairs = []
    for question in questions:
        answer = None
        question_lower = question.lower()
        
        # Add extraction logic based on the question
        # Use the parsing patterns from the docstring above
        
        qa_pairs.append({"question": question, "answer": answer})
    
    return qa_pairs
```

CRITICAL RULES:
1. Read the question carefully to determine what type of answer is expected
2. For "how many" questions, return an integer count
3. For "what is the value" questions, return the value with units (e.g., "-0.234 ns")
4. For "what are" or "list" questions, return a list or dict
5. For comparison/progression questions, return a dict with all relevant values
6. Never return None, True/False (unless yes/no question), or empty values. If failed extracted the answer, return "n/a" or "None", which can be filtered out by the validation function.
7. Match your answer format to what the question is specifically asking for

Output ONLY the Python code, nothing else (no markdown, no explanations)."""

        code_response = llm_inference(
            system_prompt="You are an expert Python programmer specializing in data extraction from JSON structures.",
            user_prompt=code_prompt,
            temperature=0.1  # Lower temperature for code generation
        )

        # Extract code from potential markdown blocks
        code = _extract_code_block(code_response)
        print(f"Generated code:\n{code}\n")

        # Step 3: Execute the code to get the answer
        try:
            # use one question at a time to extract the answer
            local_vars = {"stage_entry": stage_entry, "questions": [question_str], "json": json}
            # Execute the code to define the function
            exec(code, {"json": json}, local_vars)
            
            # Check if extract_answer function was defined
            if "extract_answer" in local_vars:
                # Call the function to get qa_pairs
                qa_pairs = local_vars["extract_answer"](stage_entry, [question_str])
                print(f"Extracted {len(qa_pairs)} QA pairs")
                
                # Display preview of each answer
                for i, qa in enumerate(qa_pairs):
                    if "answer" in qa:
                        answer = qa["answer"]
                        if answer is None or answer in ["", "None", "null", "NoneType"]:
                            print(f"  Q{i+1}: Warning - no answer or invalid answer")
                            if rejection_stats:
                                rejection_stats.record_invalid_answer()
                            continue
                        # Validate answer doesn't contain None or "n/a" values
                        if not validate_extract_answer(answer):
                            print(f"  Q{i+1}: Warning - answer contains None or n/a values, dropping")
                            if rejection_stats:
                                rejection_stats.record_invalid_answer()
                            continue
                        
                        # Get a preview of the answer
                        answer_preview = str(answer)[:100]
                        answer_type = type(answer).__name__
                        
                        # If direct LLM answer is available, validate with score
                        if is_direct_llm_answer and len(direct_llm_answers) > i:
                            direct_llm_answer = direct_llm_answers[i]
                            # Get the score of the answer by validating the answer with the direct LLM answer
                            score = validate_direct_vs_code_answer(answer, direct_llm_answer, question_str)
                            if score > 0.75:
                                print(f"  Q{i+1}: [{answer_type}] {answer_preview}...")
                                print(f"  Q{i+1}: Score: {score} - Valid answer")
                                final_qa_pairs.append(qa)
                                if rejection_stats:
                                    rejection_stats.record_success()
                            else:
                                print(f"  Q{i+1}: Score: {score} - Low score, dropping")
                                if rejection_stats:
                                    rejection_stats.record_low_score(
                                        direct_answer=direct_llm_answer,
                                        code_answer=answer,
                                        score=score,
                                        question=question_str
                                    )
                        else:
                            # No direct LLM answer available, accept the answer
                            print(f"  Q{i+1}: [{answer_type}] {answer_preview}...")
                            final_qa_pairs.append(qa)
                            if rejection_stats:
                                rejection_stats.record_success()
                    else:
                        print(f"  Q{i+1}: Warning - no answer field")
                        if rejection_stats:
                            rejection_stats.record_invalid_answer()
            else:
                print(f"[ERROR] Extraction code did not define an 'extract_answer' function")
            
        except Exception as e:
            print(f"[ERROR] Code execution failed: {e}")
            import traceback
            print(traceback.format_exc())
            if rejection_stats:
                rejection_stats.record_code_failure()
            # Continue to next question
    # Build question object
    return {
        "id": None,  # caller can assign later (e.g., Q001)
        "stage": str(st),
        "occurrence": occ,
        "qa_pairs": final_qa_pairs,
        "source_path": f"stages[{occ-1}].metrics (generated by LLM)",
        # Helpful metadata
        "design_name": stage_entry.get("design_name"),
        "sub_design_name": stage_entry.get("sub_design_name"),
        "metrics_file": stage_entry.get("metrics_file"),
        "log_file": payload.get("log_file"),
        # Misc info for validation and debugging
        "misc_info": {
            "extraction_code": code,
            "sampled_stage_info": {
                "stage": str(st),
                "occurrence": occ,
                "design_name": stage_entry.get("design_name"),
                "sub_design_name": stage_entry.get("sub_design_name"),
                "metrics_file": stage_entry.get("metrics_file"),
                "total_candidates": len(candidates),
                "stage_index": stages.index(stage_entry) if stage_entry in stages else -1,
            },
            "generation_params": {
                "temperature": temperature,
                "seed": seed,
                "stage_constraint": stage,
                "occurrence_constraint": occurrence,
            },
            "metrics_summary": {
                "metrics_json_length": len(stage_json_str),
                "metrics_truncated": len(stage_json_str) > 8000,
                "num_questions_generated": len(questions),
            },
        },
    }


def multi_stage_qa_pair(
    aggregate_metrics_json_path: str | Path,
    *,
    seed: Optional[int] = None,
    num_stages: Optional[int] = None,
    stage_types: Optional[List[str]] = None,
    temperature: float = 0.7,
    is_direct_llm_answer: bool = True,
    rejection_stats: Optional[RejectionStats] = None,
) -> Dict[str, Any]:
    """
    Generate QA pairs by sampling 2-3 stages and comparing metrics across them.
    
    This function uses `sample_stages()` to select multiple stages and generates
    questions about metric progression, optimization patterns, and cross-stage comparisons.
    
    The function automatically samples between two variant types:
    - Type 1: Different sub-designs, SAME stage type
    - Type 2: Same sub-design, DIFFERENT stages
    
    Args:
        aggregate_metrics_json_path: Path to `*_all_stages_metrics.json`
        seed: Optional random seed for reproducibility
        num_stages: Number of stages to sample (2 or 3). If None, randomly choose.
        stage_types: Optional list of stage types to filter by
        temperature: LLM temperature for generation (default: 0.7)
    
    Returns:
        A dict with the question set and answers for multiple stages.
        Key fields:
        - id: Question ID (assigned by caller)
        - difficulty: Difficulty level
        - stage: "multi_stage" (special marker)
        - stages_info: List of stage information for all sampled stages
        - qa_pairs: List of question-answer pairs
        - source_path: Path to the source metrics
        - misc_info: Additional info including sampled stages
    
    Example questions generated:
    - "How does WNS/TNS progress from placement to routing to final?"
    - "What is the buffer insertion pattern across these stages?"
    - "How does design area change from floorplan to placement to routing?"
    """
    rng = random.Random(seed)
    
    # Load the aggregate JSON to get log_file
    payload = _normalize_aggregate(_read_json(aggregate_metrics_json_path))
    
    # Sample multiple stages (automatically chooses Type 1 or Type 2)
    stages = sample_stages(
        aggregate_metrics_json_path,
        num_stages=num_stages,
        seed=seed,
        stage_types=stage_types,
    )
    
    # Build combined JSON for all stages
    stages_json = []
    stages_summary = []
    for s in stages:
        stages_summary.append({
            "stage": s.get("stage"),
            "occurrence": s.get("occurrence"),
            "sub_design_name": s.get("sub_design_name"),
            "sequence_number": s.get("sequence_number"),
        })
        stages_json.append({
            "stage": s.get("stage"),
            "occurrence": s.get("occurrence"),
            "sub_design_name": s.get("sub_design_name"),
            "metrics": s.get("metrics"),
        })
    
    stages_json_str = json.dumps(stages_json, indent=2)
    
    # Build stage description for prompt
    stage_desc = ", ".join([f"{s['stage']} (occurrence {s['occurrence']})" for s in stages_summary])
    
    print(f"\n[QA Generation] Generating multi-stage questions for: {stage_desc}...")
        
    # Step 1: Generate questions about multiple stages
    question_prompt = f"""You are generating question-answer pairs for an OpenROAD EDA design flow benchmark.

# OpenROAD EDA log file background:
{OpenRoadBackground}

# Question Examples from Designers:
{QuestionMultiStageBackground}

Given the following stages from a chip design flow (in chronological order):

Stages: {stage_desc}
Metrics JSON:
```json
{stages_json_str}
```

Generate a set of interesting, specific questions that compare or analyze metrics across these stages. No more than 6 questions.
Try to explore different questions from different aspects compared to the question examples.
Imagine you are a designer and you are trying to understand the flow execution and the design flow. What questions you would like to ask in the given stage metrics information to examine the design flow and the design quality?

Focus on:
- Metric progression (timing, area, wirelength, etc.); Make sure the metrics can be found and compared across multiple stages.
- Optimization patterns between stages (refer to optimization_process examples)
- Correlation analysis
- Design quality evolution
- Buffer/cell insertion trends
- Design_metrics_final comparison across stages

CRITICAL REQUIREMENTS FOR QUESTION QUALITY:
1. **BE SPECIFIC ABOUT METRICS** - Name the exact metric you're comparing
   - Good: "How does WNS change from floorplan to placement?"
   - Bad: "How does timing evolve?" (too vague - timing includes WNS, TNS, slack, etc.)

2. **VERIFY METRICS EXIST IN ALL STAGES** - Only ask about metrics present in ALL compared stages
   - Before asking about "HPWL from floorplan to placement", verify both stages have HPWL data
   - If a metric is missing in one stage, DON'T ask about it

3. **ONE COMPARISON PER QUESTION** - Compare ONE metric type at a time
   - Good: "What is the total power difference between floorplan and placement?"
   - Bad: "How do area, power, and timing change?" (multiple comparisons)

4. **SPECIFY SUBSTAGES IF NEEDED** - Be clear about which substage
   - Good: "What is the WNS after global placement vs detailed placement?"
   - Bad: "What is the WNS during placement?" (unclear which substage)

5. **QUANTITATIVE OVER QUALITATIVE** - Ask for numbers, not interpretations
   - Good: "What is the buffer count difference between floorplan and CTS?"
   - Bad: "Are there significant changes in buffer insertion?" (subjective)

IMPORTANT RULES:
- DO NOT use technical stage identifiers like "3_1", "3_2", "3_3", "2_1_floorplan", "3_5_place_dp", etc. in questions
- Use natural language stage names: "synthesis", "floorplan", "placement", "CTS", "routing", "final" instead
- Refer to stages descriptively: "from placement to routing", "between CTS and routing", "during placement"
- If referring to sub-stages, use descriptions like "global placement", "detailed placement", "detailed routing" instead of numbers
- Ask questions about "the floorplan stage" not "stage 2_1" or "the 2_1_floorplan stage"
"""

    output_format = """Output the question between ```question and ``` tags.
```question
<Q1> ...
Difficulty: <easy, medium, hard>
```
    
```question
<Q2> ...
Difficulty: <easy, medium, hard>
```
...
```question
<Qn> ...
Difficulty: <easy, medium, hard>
```
"""
    # print(f"Question prompt: {question_prompt}")
    # pdb.set_trace()
    question = llm_inference(
        system_prompt="You are an expert in chip design and EDA tools, specializing in multi-stage analysis and optimization.",
        user_prompt=question_prompt + "\n" + output_format,
        temperature=temperature
    )
    
    # Parse questions
    questions = re.findall(r"```question\s*(.*?)\s*```", question, re.DOTALL)
    
    # Multi-stage parsing examples:
    # Check if rpt_metric field exists in any of the stage data
    include_rpt_metric = "rpt_metric" in stages_json_str
    multi_stage_parsing_examples = get_multi_stage_parsing_examples(stages, include_rpt_metrics=include_rpt_metric)
    
    # Generate direct LLM answers if requested
    direct_llm_answers = []
    if is_direct_llm_answer:
        for q in questions:
            direct_llm_answer = generate_answer_directly_through_llm(
                question=q,
                background=OpenRoadBackground,
                context_data=stages_json_str,
                design_name=payload.get("design_name", "unknown"),
                temperature=0.3  # Lower temperature for more focused, deterministic answers
            )
            print(f"Direct LLM answer: {direct_llm_answer[:200]}...")
            direct_llm_answers.append(direct_llm_answer)
    
    final_qa_pairs = []
    for i in range(len(questions)):
        question_str = questions[i]
        if is_direct_llm_answer and len(direct_llm_answers) > i:
            direct_llm_answer = direct_llm_answers[i]
            question_str = f"{question_str}\nDirect LLM answer for reference (Try to write code to extract the answer from the multiple stage metrics JSON): {direct_llm_answer}"
        # Step 2: Generate extraction code
        code_prompt = f"""You are generating Python code to extract answers from multiple stage metrics.

Stages: {stage_desc}
Question: {question_str}

Metrics (available as variable `stages` - already a Python list of dicts):
```json
{stages_json_str}
```

Write Python code that extracts answers to the questions from the `stages` list.

Requirements:
- The variable `stages` is a list of stage dicts, each with 'stage', 'occurrence', 'sub_design_name', and 'metrics'
- Access substages: substages = stage['metrics'].get('substages', {{}})
- Access optimization_process: for opt in substage_data.get('optimization_process', [])
- Compare metrics across stages as needed
- Include units for numeric values
- IMPORTANT: Use stage names (e.g., 'placement', 'cts', 'routing') as dictionary keys, NOT 'stage1', 'stage2'

Define the extract_answer function with multi-stage parsing guidance:
```python
def extract_answer(stages: List[Dict[str, Any]], questions: list[str]) -> list:
    \"\"\"
    Extract answers from multiple stage metrics.
    
    ### Multi-stage parsing example patterns:
{multi_stage_parsing_examples}
    \"\"\"
    qa_pairs = []
    
    for question in questions:
        answer = None
        question_lower = question.lower()
        
        # Add extraction logic based on the question
        # Use the parsing patterns from the docstring above
        # Always use stage names as dictionary keys in answers
        
        qa_pairs.append({{"question": question, "answer": answer}})
    
    return qa_pairs
```

CRITICAL RULES:
1. Read the question carefully to determine what type of answer is expected
2. For "how many" questions, return an integer count
3. For "what is the value" questions, return the value with units (e.g., "-0.234 ns")
4. For "what are" or "list" questions, return a list or dict
5. For comparison/progression questions, return a dict with all relevant values
6. Never return None, True/False (unless yes/no question), or empty values. If failed extracted the answer, return "n/a" or "None", which can be filtered out by the validation function.
7. Match your answer format to what the question is specifically asking for
8. You need to write the code to handle the case that the metrics are not extracted successfully in the general fields for the substage, and then try to use the rpt_metrics for each substage to find the metrics.

Output ONLY the Python code, nothing else (no markdown, no explanations)."""
        
        code_response = llm_inference(
            system_prompt="You are an expert Python programmer specializing in data extraction and comparison.",
            user_prompt=code_prompt,
            temperature=0.1
        )
        
        code = _extract_code_block(code_response)
        print(f"Generated code:\n{code}\n")
        
        # Step 3: Execute the code
        try:
            # use one question at a time to extract the answer
            local_vars = {"stages": stages_json, "questions": [question_str], "json": json}
            exec(code, {}, local_vars)
            
            if "extract_answer" not in local_vars:
                print(f"[ERROR] Generated code did not define extract_answer function")
                continue
            
            extract_answer = local_vars["extract_answer"]
            qa_pairs = extract_answer(stages_json, [question_str])
            
            # Validate answers
            for idx, qa in enumerate(qa_pairs):
                ans = qa.get("answer")
                if ans is None or ans == "" or (isinstance(ans, (list, dict)) and not ans):
                    print(f"[WARN] Question {idx+1} has invalid answer, dropping")
                    if rejection_stats:
                        rejection_stats.record_invalid_answer()
                    continue
                elif not validate_extract_answer(ans):
                    print(f"[WARN] Question {idx+1} contains None or n/a values, dropping")
                    if rejection_stats:
                        rejection_stats.record_invalid_answer()
                    continue
                
                # Get a preview of the answer
                answer_preview = str(ans)[:100]
                answer_type = type(ans).__name__
                
                # If direct LLM answer is available, validate with score
                if is_direct_llm_answer and len(direct_llm_answers) > idx:
                    direct_llm_answer = direct_llm_answers[idx]
                    # Get the score of the answer by validating the answer with the direct LLM answer
                    score = validate_direct_vs_code_answer(ans, direct_llm_answer, question_str)
                    if score > 0.75:
                        print(f"  Q{idx+1}: [{answer_type}] {answer_preview}...")
                        print(f"  Q{idx+1}: Score: {score} - Valid answer")
                        final_qa_pairs.append(qa)
                        if rejection_stats:
                            rejection_stats.record_success()
                    else:
                        print(f"  Q{idx+1}: Score: {score} - Low score, dropping")
                        if rejection_stats:
                            rejection_stats.record_low_score(
                                direct_answer=direct_llm_answer,
                                code_answer=ans,
                                score=score,
                                question=question_str
                            )
                else:
                    # No direct LLM answer available, accept the answer
                    print(f"  Q{idx+1}: [{answer_type}] {answer_preview}...")
                    final_qa_pairs.append(qa)
                    if rejection_stats:
                        rejection_stats.record_success()
            
        except Exception as e:
            print(f"[ERROR] Code execution failed: {e}")
            import traceback
            print(traceback.format_exc())
            if rejection_stats:
                rejection_stats.record_code_failure()
            # Continue to next question
    
    if not final_qa_pairs:
        raise QAGenerationError("No valid QA pairs generated")
    
    return {
        "stage": "multi_stage",
        "stages_info": stages_summary,
        "qa_pairs": final_qa_pairs,
        "source_path": str(aggregate_metrics_json_path),
        "log_file": payload.get("log_file"),
        "misc_info": {
            "extraction_code": code,
            "num_stages_sampled": len(stages),
            "stage_types": [s.get("stage") for s in stages_summary],
            "generation_params": {
                "temperature": temperature,
                "seed": seed,
                "num_stages": num_stages,
                "stage_types_filter": stage_types,
            },
            "metrics_summary": {
                "metrics_json_length": len(stages_json_str),
                "metrics_truncated": len(stages_json_str) > 12000,
                "num_questions_generated": len(questions),
            },
        },
    }


def stage_to_final_qa_pair(
    aggregate_metrics_json_path: str | Path,
    *,
    seed: Optional[int] = None,
    stage_type: Optional[str] = None,
    temperature: float = 0.7,
    is_direct_llm_answer: bool = True,
    rejection_stats: Optional[RejectionStats] = None,
) -> Dict[str, Any]:
    """
    Generate QA pairs comparing a stage with the final design, including all intermediate stages.
    
    This function selects a starting stage and final stage (same sub-design), then includes
    ALL intermediate stages between them to show the complete progression path.
    
    Args:
        aggregate_metrics_json_path: Path to `*_all_stages_metrics.json`
        seed: Optional random seed for reproducibility
        stage_type: Optional specific stage type (e.g., 'placement', 'routing')
        temperature: LLM temperature for generation (default: 0.7)
    
    Returns:
        A dict with the question set and answers comparing stage progression to final.
        Key fields:
        - id: Question ID (assigned by caller)
        - difficulty: Difficulty level
        - stage: "stage_to_final" (special marker)
        - stage_info: Information about the selected starting stage
        - final_info: Information about the final stage
        - all_stages_info: List of all stages in the progression (including intermediates)
        - qa_pairs: List of question-answer pairs
        - source_path: Path to the source metrics
        - misc_info: Additional info including all stages
    
    Example questions generated:
    - "Show the timing progression from placement through CTS through routing to final"
    - "At which stage did the WNS degrade the most?"
    - "What is the HPWL vs actual wirelength gap from placement to final?"
    - "How does buffer count change across all stages from placement to final?"
    """
    rng = random.Random(seed)
    
    # Load the aggregate JSON to get log_file
    payload = _normalize_aggregate(_read_json(aggregate_metrics_json_path))
    
    # Select starting stage and final with matching subdesign
    starting_stage, final = select_stage_and_final_with_same_subdesign(
        aggregate_metrics_json_path,
        seed=seed,
        stage_type=stage_type,
    )
    
    stage_name = starting_stage.get("stage")
    final_name = final.get("stage")
    sub_design_name = starting_stage.get("sub_design_name")
    start_seq = starting_stage.get("sequence_number", 0)
    final_seq = final.get("sequence_number", 0)
    
    # Find all intermediate stages with the same sub_design_name
    all_stages_list = payload.get("stages", [])
    intermediate_stages = []
    
    for s in all_stages_list:
        if not isinstance(s, dict) or not s.get("metrics"):
            continue
        
        # Check if it's the same sub-design
        if s.get("sub_design_name") != sub_design_name:
            continue
        
        seq = s.get("sequence_number", 0)
        # Include stages between start and final (inclusive)
        if start_seq <= seq <= final_seq:
            intermediate_stages.append(s)
    
    # Sort by sequence_number to ensure chronological order
    intermediate_stages.sort(key=lambda x: x.get("sequence_number", 0))
    
    # Build JSON for all stages in the progression
    progression_stages = []
    stages_summary = []
    
    for s in intermediate_stages:
        stage_data = {
            "stage_type": s.get("stage"),
            "occurrence": s.get("occurrence"),
            "sub_design_name": s.get("sub_design_name"),
            "sequence_number": s.get("sequence_number"),
            "metrics": s.get("metrics"),
        }
        progression_stages.append(stage_data)
        stages_summary.append({
            "stage": s.get("stage"),
            "occurrence": s.get("occurrence"),
            "sequence_number": s.get("sequence_number"),
        })
    
    comparison_json = {
        "progression": progression_stages,
        "num_stages": len(progression_stages),
        "starting_stage": stage_name,
        "final_stage": final_name,
        "sub_design_name": sub_design_name,
    }
    
    comparison_json_str = json.dumps(comparison_json, indent=2)
    
    stage_names = " → ".join([s['stage'] for s in stages_summary])
    print(f"\n[QA Generation] Generating stage-to-final questions with full progression:")
    print(f"  Path: {stage_names}")
    print(f"  Sub-design: {sub_design_name}")
    print(f"  Total stages: {len(progression_stages)}")
    
    # Step 1: Generate questions
    question_prompt = f"""You are generating question-answer pairs for an OpenROAD EDA design flow benchmark.

# OpenROAD EDA log file background:
{OpenRoadBackground}

# Question Examples from Designers:
{QuestionConsecutiveStageBackground}

Given the following complete stage progression from a chip design flow (same sub-design: {sub_design_name}):

Progression path: {stage_names}
Number of stages: {len(progression_stages)}
Starting stage: {stage_name}
Final stage: {final_name}

Metrics JSON (all stages in progression):
```json
{comparison_json_str}
```

Generate a set of interesting, specific questions that analyze the complete progression from the starting stage through all intermediate stages to the final design. No more than 6 questions.
Try to explore different questions from different aspects compared to the question examples.
Imagine you are a designer and you are trying to understand the flow execution and the design flow. What questions you would like to ask in the given stage metrics information to examine the design flow and the design quality?

Focus on:
- Timing progression across all stages (WNS/TNS evolution); Make sure the Timing metrics can be found and compared across multiple stages.
- At which stage did metrics improve or degrade the most?
- Estimation accuracy (HPWL vs actual wirelength from placement to final)
- Cumulative impact of optimizations across the entire flow
- Buffer/cell insertion patterns throughout the progression
- Design quality evolution (violations, density, area)
- Correlation between intermediate estimates and final results
- Stage-by-stage metric changes

IMPORTANT RULES:
- DO NOT use technical stage identifiers like "3_1", "3_2", "3_3", "2_1_floorplan", "3_5_place_dp", etc. in questions
- Use natural language stage names: "synthesis", "floorplan", "placement", "CTS", "routing", "final" instead
- Refer to stages descriptively: "from placement through CTS to final", "across all stages", "during the CTS stage"
- If referring to sub-stages, use descriptions like "global placement", "detailed placement", "detailed routing" instead of numbers
- Ask questions about "the placement stage" not "stage 3_3" or "the 3_5_place_dp stage"
- Since you have ALL intermediate stages, ask questions about the complete progression path
"""

    output_format = """Output the question between ```question and ``` tags.
```question
<Q1> ...
Difficulty: <easy, medium, hard>
```
    
```question
<Q2> ...
Difficulty: <easy, medium, hard>
```
...
```question
<Qn> ...
Difficulty: <easy, medium, hard>
```
"""
    
    question = llm_inference(
        system_prompt="You are an expert in chip design and EDA tools, specializing in PPA correlation analysis.",
        user_prompt=question_prompt + "\n" + output_format,
        temperature=temperature
    )
    
    # Parse questions
    questions = re.findall(r"```question\s*(.*?)\s*```", question, re.DOTALL)
    
    # Get multi-stage parsing examples for the progression
    progression_stages_for_examples = [{"stage": s.get("stage")} for s in intermediate_stages]
    # Check if rpt_metric field exists in any of the progression data
    include_rpt_metric = "rpt_metric" in comparison_json_str
    multi_stage_parsing_examples = get_multi_stage_parsing_examples(progression_stages_for_examples, include_rpt_metrics=include_rpt_metric)
    
    # Generate direct LLM answers if requested
    direct_llm_answers = []
    if is_direct_llm_answer:
        for q in questions:
            direct_llm_answer = generate_answer_directly_through_llm(
                question=q,
                background=OpenRoadBackground,
                context_data=comparison_json_str,
                design_name=payload.get("design_name", "unknown"),
                temperature=0.3  # Lower temperature for more focused, deterministic answers
            )
            print(f"Direct LLM answer: {direct_llm_answer[:200]}...")
            direct_llm_answers.append(direct_llm_answer)
    
    final_qa_pairs = []
    for i in range(len(questions)):
        question_str = questions[i]
        if is_direct_llm_answer and len(direct_llm_answers) > i:
            direct_llm_answer = direct_llm_answers[i]
            question_str = f"{question_str}\nDirect LLM answer for reference (Try to write code to extract the answer from the complete stage progression metrics JSON): {direct_llm_answer}"
        # Step 2: Generate extraction code
        code_prompt = f"""You are generating Python code to extract answers comparing metrics across a complete stage progression.

Progression path: {stage_names}
Starting stage: {stage_name}
Final stage: {final_name}
Sub-design: {sub_design_name}
Number of stages: {len(progression_stages)}
Question: {question_str}

Metrics (available as variable `progression_data` - already a Python dict with 'progression' list):
```json
{comparison_json_str}
```

Write Python code that extracts answers to the questions from the `progression_data` dict.

Requirements:
- The variable `progression_data` has a 'progression' key containing a list of stage dicts
- Each stage has 'stage_type', 'occurrence', 'sequence_number', and 'metrics'
- Access substages: substages = stage['metrics'].get('substages', {{}})
- Access optimization_process: for opt in substage_data.get('optimization_process', [])
- Calculate differences, gaps, progressions across stages
- Include units for numeric values
- IMPORTANT: Use stage names (stage['stage_type']) as dictionary keys

Define the extract_answer function with stage progression parsing guidance:
```python
def extract_answer(progression_data: Dict[str, Any], questions: list[str]) -> list:
    \"\"\"
    Extract answers from stage progression metrics.
    
    ### Stage progression parsing example patterns:
{multi_stage_parsing_examples}
    \"\"\"
    qa_pairs = []
    
    progression = progression_data.get('progression', [])
    starting_stage = progression_data.get('starting_stage')
    final_stage = progression_data.get('final_stage')
    
    for question in questions:
        answer = None
        question_lower = question.lower()
        
        # Add extraction logic using patterns from docstring above
        # Use stage names from stage['stage_type'] as dictionary keys
        
        qa_pairs.append({{"question": question, "answer": answer}})
    
    return qa_pairs
```

CRITICAL RULES:
1. Read the question carefully to determine what type of answer is expected
2. For "how many" questions, return an integer count
3. For "what is the value" questions, return the value with units (e.g., "-0.234 ns")
4. For "what are" or "list" questions, return a list or dict
5. For comparison/progression questions, return a dict with all relevant values
6. Never return None, True/False (unless yes/no question), or empty values. If failed extracted the answer, return "n/a" or "None", which can be filtered out by the validation function.
7. Use actual stage names from stage['stage_type'] as dictionary keys

Output ONLY the Python code, nothing else (no markdown, no explanations)."""
        
        code_response = llm_inference(
            system_prompt="You are an expert Python programmer specializing in data comparison and progression analysis.",
            user_prompt=code_prompt,
            temperature=0.1
        )
        
        code = _extract_code_block(code_response)
        print(f"Generated code:\n{code}\n")
        
        # Step 3: Execute the code
        try:
            # use one question at a time to extract the answer
            local_vars = {"progression_data": comparison_json, "questions": [question_str], "json": json}
            exec(code, {}, local_vars)
            
            if "extract_answer" not in local_vars:
                print(f"[ERROR] Generated code did not define extract_answer function")
                continue
            
            extract_answer = local_vars["extract_answer"]
            qa_pairs = extract_answer(comparison_json, [question_str])
            
            # Validate answers
            for idx, qa in enumerate(qa_pairs):
                ans = qa.get("answer")
                if ans is None or ans == "" or (isinstance(ans, (list, dict)) and not ans):
                    print(f"[WARN] Question {idx+1} has invalid answer, dropping")
                    if rejection_stats:
                        rejection_stats.record_invalid_answer()
                    continue
                elif not validate_extract_answer(ans):
                    print(f"[WARN] Question {idx+1} contains None or n/a values, dropping")
                    if rejection_stats:
                        rejection_stats.record_invalid_answer()
                    continue
                
                # Get a preview of the answer
                answer_preview = str(ans)[:100]
                answer_type = type(ans).__name__
                
                # If direct LLM answer is available, validate with score
                if is_direct_llm_answer and len(direct_llm_answers) > idx:
                    direct_llm_answer = direct_llm_answers[idx]
                    # Get the score of the answer by validating the answer with the direct LLM answer
                    score = validate_direct_vs_code_answer(ans, direct_llm_answer, question_str)
                    if score > 0.75:
                        print(f"  Q{idx+1}: [{answer_type}] {answer_preview}...")
                        print(f"  Q{idx+1}: Score: {score} - Valid answer")
                        final_qa_pairs.append(qa)
                        if rejection_stats:
                            rejection_stats.record_success()
                    else:
                        print(f"  Q{idx+1}: Score: {score} - Low score, dropping")
                        if rejection_stats:
                            rejection_stats.record_low_score(
                                direct_answer=direct_llm_answer,
                                code_answer=ans,
                                score=score,
                                question=question_str
                            )
                else:
                    # No direct LLM answer available, accept the answer
                    print(f"  Q{idx+1}: [{answer_type}] {answer_preview}...")
                    final_qa_pairs.append(qa)
                    if rejection_stats:
                        rejection_stats.record_success()
            
        except Exception as e:
            print(f"[ERROR] Code execution failed: {e}")
            import traceback
            print(traceback.format_exc())
            if rejection_stats:
                rejection_stats.record_code_failure()
            # Continue to next question
    
    if not final_qa_pairs:
        raise QAGenerationError("No valid QA pairs generated")
    
    return {
        "stage": "stage_to_final",
        "stage_info": {
            "stage": stage_name,
            "occurrence": starting_stage.get("occurrence"),
            "sub_design_name": sub_design_name,
            "sequence_number": starting_stage.get("sequence_number"),
        },
        "final_info": {
            "stage": final_name,
            "occurrence": final.get("occurrence"),
            "sub_design_name": sub_design_name,
            "sequence_number": final.get("sequence_number"),
        },
        "all_stages_info": stages_summary,  # List of all stages in progression
        "qa_pairs": final_qa_pairs,
        "source_path": str(aggregate_metrics_json_path),
        "log_file": payload.get("log_file"),
        "misc_info": {
            "extraction_code": code,
            "stage_type": stage_name,
            "sub_design_name": sub_design_name,
            "num_stages_in_progression": len(progression_stages),
            "progression_path": stage_names,
            "generation_params": {
                "temperature": temperature,
                "seed": seed,
                "stage_type_filter": stage_type,
            },
            "metrics_summary": {
                "metrics_json_length": len(comparison_json_str),
                "num_questions_generated": len(questions),
            },
        },
    }


def generate_qa_pairs(
    aggregate_metrics_json_path: str | Path,
    *,
    n: int = 10,
    seed: Optional[int] = None,
    qa_json_path: Optional[str] = "./",
    temperature: float = 0.7,
    include_flow_execution: bool = True,
) -> Dict[str, Any]:
    """
    Generate N QA pairs by repeatedly sampling stages and using LLM.

    Args:
        aggregate_metrics_json_path: path to `*_all_stages_metrics.json`
        n: number of stage-level QA pairs to generate
        seed: optional random seed for reproducibility
        temperature: LLM temperature for generation (default: 0.7)
        include_flow_execution: if True, generate one flow execution QA pair in addition to stage pairs

    Returns:
        Dict with:
          - design_name
          - questions: list of QA dicts
    """
    payload = _normalize_aggregate(_read_json(aggregate_metrics_json_path))
    design_name = payload.get("design_name") or Path(aggregate_metrics_json_path).stem.replace("_all_stages_metrics", "")

    rng = random.Random(seed)
    questions: List[Dict[str, Any]] = []
    
    # Create rejection statistics tracker
    rejection_stats = RejectionStats()
    
    # Track sampled stages to avoid duplicates
    sampled_stages = set()  # Store (stage_name, occurrence) tuples
    
    print(f"\n{'='*80}")
    print(f"Generating QA pairs for {design_name}")
    if include_flow_execution:
        print(f"  - 1 flow execution QA pair")
    print(f"  - {n} stage-level QA pairs")
    print(f"{'='*80}\n")
    
    current_total_qa_pairs = 0
    # Generate flow execution QA pair if requested
    if include_flow_execution:
        try:
            q = flow_execution_qa_pair(
                aggregate_metrics_json_path,
                seed=rng.randint(0, 2**31 - 1),
                temperature=temperature,
                rejection_stats=rejection_stats
            )
            q["id"] = "Q_FLOW_001"
            questions.append(q)
            qa_count = len(q.get('qa_pairs', []))
            print(f"\n[OK] Generated Q_FLOW_001: {qa_count} QA pair(s) for flow execution")
            current_total_qa_pairs += qa_count
        except Exception as e:
            print(f"\n[ERROR] Failed to generate flow execution QA: {e}")
    
    # Generate stage-level QA pairs with duplicate avoidance
    max_retries = n * 5  # Allow retries to find unique stages
    attempts = 0
    generated = 0
    
    while generated < n and attempts < max_retries:
        attempts += 1
        try:
            q = single_stage_qa_pair(
                aggregate_metrics_json_path, 
                seed=rng.randint(0, 2**31 - 1),
                temperature=temperature,
                rejection_stats=rejection_stats
            )
            
            # Check if this stage has been sampled before
            stage_key = (q.get('stage'), q.get('occurrence'))
            if stage_key in sampled_stages:
                print(f"[SKIP] Stage {stage_key[0]} (occurrence {stage_key[1]}) already sampled, retrying...")
                continue
            
            # Add to sampled set and questions list
            sampled_stages.add(stage_key)
            q["id"] = f"Q{generated+1:03d}"
            questions.append(q)
            
            # Display preview of qa_pairs
            qa_count = len(q.get('qa_pairs', []))
            print(f"\n[OK] Generated Q{generated+1:03d}: {qa_count} QA pair(s) for stage {q.get('stage', 'unknown')}")
            current_total_qa_pairs += qa_count
            generated += 1
            
        except Exception as e:
            print(f"\n[ERROR] Failed to generate stage QA (attempt {attempts}): {e}")
            continue
    
    if generated < n:
        print(f"\n[WARN] Only generated {generated}/{n} unique stage QA pairs after {attempts} attempts")

    # Calculate statistics
    stats = {
        "flow_execution": 0,
        "stage_level": 0,
        "total_sets": len(questions),
        "total_qa_pairs": current_total_qa_pairs,
        "failed": max(0, (n + (1 if include_flow_execution else 0)) - len(questions)),
    }
    
    # Count by type
    for q in questions:
        if q.get('stage') == 'flow_execution':
            stats['flow_execution'] += len(q.get('qa_pairs', []))
        else:
            stats['stage_level'] += len(q.get('qa_pairs', []))
    
    print(f"\n{'='*80}")
    print(f"Generation Summary for {design_name}")
    print(f"{'='*80}")
    print(f"  Flow Execution:    {stats['flow_execution']:3d} QA pairs")
    print(f"  Stage Level:       {stats['stage_level']:3d} QA pairs")
    print(f"  {'─'*40}")
    print(f"  Total Sets:        {stats['total_sets']:3d}")
    print(f"  Total QA Pairs:    {stats['total_qa_pairs']:3d}")
    print(f"  Failed:            {stats['failed']:3d}")
    print(f"{'='*80}")
    
    # Print rejection statistics prominently
    rejection_summary = rejection_stats.get_summary()
    print(f"\n{'='*80}")
    print(f"ANSWER VALIDATION & REJECTION REPORT")
    print(f"{'='*80}")
    rejection_stats.print_summary(prefix="")
    print(f"{'='*80}\n")
    
    # Add metadata to each question
    for q in questions:
        q['aggregate_metrics_json_path'] = str(aggregate_metrics_json_path)

    # Create output directory if needed
    output_dir = Path(qa_json_path).resolve()
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write results with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Write raw data (full question objects with metadata)
    out_path_raw = output_dir / f"{design_name}_{timestamp}_questions_and_answers_llm_raw.json"
    with out_path_raw.open("w", encoding="utf-8") as f:
        json.dump({
            "design_name": design_name,
            "log_file": payload.get("log_file"),
            "source": str(aggregate_metrics_json_path),
            "timestamp": timestamp,
            "statistics": stats,
            "rejection_statistics": rejection_stats.get_summary(),
            "generation_params": {
                "n": n,
                "seed": seed,
                "temperature": temperature,
                "include_flow_execution": include_flow_execution,
            },
            "questions": questions,
        }, f, indent=2, ensure_ascii=False)
    print(f"[OK] Wrote {len(questions)} QA sets (raw) → {out_path_raw}")
    
    # 2. Write QA pairs only (flattened)
    questions_and_answers = []
    for q in questions:
        qa_stage = q.get('stage', 'unknown')
        qa_id = q.get('id', 'unknown')
        qa_type = "flow_execution" if qa_stage == "flow_execution" else "stage_level"
        
        for idx, qa in enumerate(q.get('qa_pairs', [])):
            qa_dict = {
                "id": f"{qa_id}_P{idx+1:02d}",
                "type": qa_type,
                "question": qa['question'],
                "answer": qa['answer'],
            }
            
            # Add stage info for context
            if qa_type == "stage_level":
                qa_dict["stage"] = qa_stage
                qa_dict["occurrence"] = q.get('occurrence', 1)
            
            questions_and_answers.append(qa_dict)
    
    out_path = output_dir / f"{design_name}_{timestamp}_questions_and_answers_llm.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "design_name": design_name,
            "timestamp": timestamp,
            "log_file": payload.get("log_file"),
            "source": str(aggregate_metrics_json_path),
            "statistics": stats,
            "rejection_statistics": rejection_stats.get_summary(),
            "qa_pairs": questions_and_answers,
        }, f, indent=2, ensure_ascii=False)
    print(f"[OK] Wrote {len(questions_and_answers)} QA pairs (flattened) → {out_path}")

    # Print final comprehensive report
    rejection_summary = rejection_stats.get_summary()
    print(f"\n{'='*80}")
    print(f"FINAL REPORT: {design_name}")
    print(f"{'='*80}")
    print(f"📁 Output Files:")
    print(f"   Raw Data:      {out_path_raw}")
    print(f"   QA Pairs:      {out_path}")
    print(f"\n📊 Generation Statistics:")
    print(f"   Total QA Pairs Generated:     {stats['total_qa_pairs']}")
    print(f"   Flow Execution QA:            {stats['flow_execution']}")
    print(f"   Stage Level QA:               {stats['stage_level']}")
    print(f"\n✅ Answer Validation Results:")
    print(f"   Total Attempts:               {rejection_summary['total_attempts']}")
    print(f"   ✓ Successful:                 {rejection_summary['successful']} ({rejection_summary['success_rate']:.1f}%)")
    print(f"   ✗ Rejected:                   {rejection_summary['rejected']['total_rejected']}")
    print(f"      • Code Execution Failed:   {rejection_summary['rejected']['code_execution_failed']}")
    print(f"      • Low Validation Score:    {rejection_summary['rejected']['low_score_rejected']}")
    print(f"      • Invalid Answer Format:   {rejection_summary['rejected']['invalid_answer_rejected']}")
    
    if rejection_summary['rejected']['low_score_rejected'] > 0:
        print(f"\n⚠️  Low Score Rejections: {rejection_summary['rejected']['low_score_rejected']} cases")
        print(f"   (Details saved in rejection_statistics.low_score_details)")
    
    print(f"{'='*80}\n")

    return {
        "design_name": design_name,
        "questions": questions,
        "summary": stats,
        "rejection_statistics": rejection_stats.get_summary(),
    }


def generate_multistages_qa_pairs(
    aggregate_metrics_json_path: str | Path,
    *,
    n_multi_stage: int = 3,
    n_stage_to_final: int = 2,
    seed: Optional[int] = None,
    qa_json_path: Optional[str] = "./",
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """
    Generate QA pairs focusing on multi-stage analysis (excluding synthesis):
    - Multi-stage comparison QA (n_multi_stage sets) - compares 2-3 non-synthesis stages
    - Stage-to-final correlation QA (n_stage_to_final sets) - compares non-synthesis stage to final
    
    This function generates questions that compare metrics across multiple stages
    and analyze correlations between intermediate stages and final design.
    
    Synthesis stage is excluded because:
    - Synthesis metrics differ significantly from physical design stages
    - Cross-stage comparisons are more meaningful among physical design stages
    - Focus on PPA progression during physical implementation
    
    Args:
        aggregate_metrics_json_path: Path to `*_all_stages_metrics.json`
        n_multi_stage: Number of multi-stage QA sets to generate (default: 3)
        n_stage_to_final: Number of stage-to-final QA sets to generate (default: 2)
        seed: Optional random seed for reproducibility
        qa_json_path: Output directory path (default: "./")
        temperature: LLM temperature for generation (default: 0.7)
    
    Returns:
        Dict with:
        - design_name: Design name
        - questions: List of all QA dicts with metadata
        - summary: Statistics about generated QA pairs
    
    Stages included:
        - floorplan
        - placement
        - cts (clock tree synthesis)
        - routing
        - final
    
    Stages excluded:
        - synthesis (excluded from all multi-stage comparisons)
    
    Example:
        >>> result = generate_multistages_qa_pairs(
        ...     "design_metrics.json",
        ...     n_multi_stage=3,
        ...     n_stage_to_final=2,
        ...     seed=42
        ... )
        >>> print(f"Total QA pairs: {result['summary']['total_qa_pairs']}")
    """
    payload = _normalize_aggregate(_read_json(aggregate_metrics_json_path))
    design_name = payload.get("design_name") or Path(aggregate_metrics_json_path).stem.replace("_all_stages_metrics", "")

    rng = random.Random(seed)
    questions: List[Dict[str, Any]] = []
    
    # Create rejection statistics tracker
    rejection_stats = RejectionStats()
    
    # Track sampled combinations to avoid duplicates
    sampled_multi_stage_combos = set()  # Store frozenset of (stage, occurrence) tuples
    sampled_stage_to_final = set()  # Store (stage, occurrence) tuples
    
    # Track statistics
    stats = {
        "multi_stage": 0,
        "stage_to_final": 0,
        "total_sets": 0,
        "total_qa_pairs": 0,
        "failed": 0,
    }
    
    print(f"\n{'='*80}")
    print(f"Generating Multi-stages QA Pairs for {design_name}")
    print(f"{'='*80}")
    print(f"  - {n_multi_stage} multi-stage QA sets")
    print(f"  - {n_stage_to_final} stage-to-final QA sets")
    print(f"{'='*80}\n")
    
    # 3. Generate multi-stage QA pairs (excluding synthesis) with duplicate avoidance
    print(f"\n[1/2] Generating {n_multi_stage} multi-stage QA sets (excluding synthesis)...")
    # Define non-synthesis stages for filtering
    non_synthesis_stages = ['floorplan', 'placement', 'cts', 'routing', 'final']
    
    max_retries = n_multi_stage * 10  # Allow more retries for multi-stage
    attempts = 0
    generated_multi = 0
    
    while generated_multi < n_multi_stage and attempts < max_retries:
        attempts += 1
        try:
            # Vary the number of stages (2 or 3) randomly
            num_stages = rng.choice([2, 3])
            
            q = multi_stage_qa_pair(
                aggregate_metrics_json_path,
                num_stages=num_stages,
                stage_types=non_synthesis_stages,  # Exclude synthesis
                seed=rng.randint(0, 2**31 - 1),
                temperature=temperature,
                rejection_stats=rejection_stats
            )
            
            # Create a unique key for this combination
            stages_info = q.get('stages_info', [])
            combo_key = frozenset((s['stage'], s['occurrence']) for s in stages_info)
            
            if combo_key in sampled_multi_stage_combos:
                stages_list = [s['stage'] for s in stages_info]
                print(f"  [SKIP] Multi-stage combination {stages_list} already sampled, retrying...")
                continue
            
            # Add to sampled set
            sampled_multi_stage_combos.add(combo_key)
            
            q["id"] = f"Q_MULTI_{generated_multi+1:03d}"
            q["type"] = "multi_stage"
            questions.append(q)
            
            qa_count = len(q.get('qa_pairs', []))
            stages_list = [s['stage'] for s in stages_info]
            stats["multi_stage"] += qa_count
            stats["total_sets"] += 1
            stats["total_qa_pairs"] += qa_count
            print(f"  ✓ Generated Q_MULTI_{generated_multi+1:03d}: {qa_count} QA pair(s) for stages {stages_list}")
            generated_multi += 1
            
        except Exception as e:
            print(f"  ✗ Failed to generate multi-stage QA (attempt {attempts}): {e}")
            continue
    
    if generated_multi < n_multi_stage:
        print(f"  [WARN] Only generated {generated_multi}/{n_multi_stage} unique multi-stage QA sets after {attempts} attempts")
        stats["failed"] += (n_multi_stage - generated_multi)
    
    # 4. Generate stage-to-final QA pairs (excluding synthesis) with duplicate avoidance
    print(f"\n[2/2] Generating {n_stage_to_final} stage-to-final QA sets (excluding synthesis)...")
    
    max_retries_final = n_stage_to_final * 10
    attempts_final = 0
    generated_final = 0
    
    while generated_final < n_stage_to_final and attempts_final < max_retries_final:
        attempts_final += 1
        try:
            # Specify stage types for variety (excluding synthesis)
            stage_types = ['placement', 'routing', 'cts', 'floorplan']
            stage_type = rng.choice(stage_types)
            
            q = stage_to_final_qa_pair(
                aggregate_metrics_json_path,
                stage_type=stage_type,
                seed=rng.randint(0, 2**31 - 1),
                temperature=temperature,
                rejection_stats=rejection_stats
            )
            
            # Create a unique key for this stage-to-final pair
            stage_info = q.get('stage_info', {})
            stage_key = (stage_info.get('stage'), stage_info.get('occurrence'))
            
            if stage_key in sampled_stage_to_final:
                print(f"  [SKIP] Stage-to-final {stage_key[0]} (occurrence {stage_key[1]}) already sampled, retrying...")
                continue
            
            # Add to sampled set
            sampled_stage_to_final.add(stage_key)
            
            q["id"] = f"Q_FINAL_{generated_final+1:03d}"
            q["type"] = "stage_to_final"
            questions.append(q)
            
            qa_count = len(q.get('qa_pairs', []))
            stage_name = stage_info.get('stage', 'unknown')
            stats["stage_to_final"] += qa_count
            stats["total_sets"] += 1
            stats["total_qa_pairs"] += qa_count
            print(f"  ✓ Generated Q_FINAL_{generated_final+1:03d}: {qa_count} QA pair(s) for {stage_name} → final")
            generated_final += 1
            
        except Exception as e:
            print(f"  ✗ Failed to generate stage-to-final QA (attempt {attempts_final}): {e}")
            continue
    
    if generated_final < n_stage_to_final:
        print(f"  [WARN] Only generated {generated_final}/{n_stage_to_final} unique stage-to-final QA sets after {attempts_final} attempts")
        stats["failed"] += (n_stage_to_final - generated_final)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Generation Summary for {design_name}")
    print(f"{'='*80}")
    print(f"  Multi-Stage:       {stats['multi_stage']:3d} QA pairs")
    print(f"  Stage-to-Final:    {stats['stage_to_final']:3d} QA pairs")
    print(f"  {'─'*40}")
    print(f"  Total Sets:        {stats['total_sets']:3d}")
    print(f"  Total QA Pairs:    {stats['total_qa_pairs']:3d}")
    print(f"  Failed:            {stats['failed']:3d}")
    print(f"{'='*80}")
    
    # Print rejection statistics prominently
    rejection_summary = rejection_stats.get_summary()
    print(f"\n{'='*80}")
    print(f"ANSWER VALIDATION & REJECTION REPORT")
    print(f"{'='*80}")
    rejection_stats.print_summary(prefix="")
    print(f"{'='*80}\n")
    
    # Add metadata to each question
    for q in questions:
        q['aggregate_metrics_json_path'] = str(aggregate_metrics_json_path)
    
    # Create output directory if needed
    output_dir = Path(qa_json_path).resolve()
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write results with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Write raw data (full question objects with metadata)
    out_path_raw = output_dir / f"{design_name}_{timestamp}_qa_multistages_raw.json"
    with out_path_raw.open("w", encoding="utf-8") as f:
        json.dump({
            "design_name": design_name,
            "log_file": payload.get("log_file"),
            "source": str(aggregate_metrics_json_path),
            "timestamp": timestamp,
            "statistics": stats,
            "rejection_statistics": rejection_stats.get_summary(),
            "generation_params": {
                "n_multi_stage": n_multi_stage,
                "n_stage_to_final": n_stage_to_final,
                "seed": seed,
                "temperature": temperature,
            },
            "questions": questions,
        }, f, indent=2, ensure_ascii=False)
    print(f"[OK] Wrote {len(questions)} QA sets (raw) → {out_path_raw}")
    
    # 2. Write QA pairs only (flattened)
    questions_and_answers = []
    for q in questions:
        qa_type = q.get('type', 'unknown')
        qa_id = q.get('id', 'unknown')
        
        for idx, qa in enumerate(q.get('qa_pairs', [])):
            qa_dict = {
                "id": f"{qa_id}_P{idx+1:02d}",
                "type": qa_type,
                "question": qa['question'],
                "answer": qa['answer'],
            }
            
            # Add stage info for context
            if qa_type == "multi_stage":
                qa_dict["stages"] = [s['stage'] for s in q.get('stages_info', [])]
            elif qa_type == "stage_to_final":
                qa_dict["stage"] = q.get('stage_info', {}).get('stage')
                qa_dict["final"] = q.get('final_info', {}).get('stage')
            
            questions_and_answers.append(qa_dict)
    
    out_path = output_dir / f"{design_name}_{timestamp}_qa_multistages.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "design_name": design_name,
            "timestamp": timestamp,
            "log_file": payload.get("log_file"),
            "source": str(aggregate_metrics_json_path),
            "statistics": stats,
            "rejection_statistics": rejection_stats.get_summary(),
            "qa_pairs": questions_and_answers,
        }, f, indent=2, ensure_ascii=False)
    print(f"[OK] Wrote {len(questions_and_answers)} QA pairs (flattened) → {out_path}")
    
    return {
        "design_name": design_name,
        "questions": questions,
        "summary": stats,
        "rejection_statistics": rejection_stats.get_summary(),
    }


def generate_all_qa_pairs(
    aggregate_metrics_json_path: str | Path,
    *,
    n_single_stage: int = 10,
    n_multi_stage: int = 3,
    n_stage_to_final: int = 2,
    seed: Optional[int] = None,
    qa_json_path: Optional[str] = "./",
    temperature: float = 0.7,
    include_flow_execution: bool = True,
) -> Dict[str, Any]:
    """
    Generate comprehensive QA pairs covering all types of analysis:
    - Flow execution QA (1 set if enabled)
    - Single-stage QA (n_single_stage sets)
    - Multi-stage comparison QA (n_multi_stage sets)
    - Stage-to-final correlation QA (n_stage_to_final sets)
    
    This unified function combines the functionality of generate_qa_pairs() and
    generate_multistages_qa_pairs() to generate all types of QA pairs in one call.
    
    Args:
        aggregate_metrics_json_path: Path to `*_all_stages_metrics.json`
        n_single_stage: Number of single-stage QA sets to generate (default: 10)
        n_multi_stage: Number of multi-stage QA sets to generate (default: 3)
        n_stage_to_final: Number of stage-to-final QA sets to generate (default: 2)
        seed: Optional random seed for reproducibility
        qa_json_path: Output directory path (default: "./")
        temperature: LLM temperature for generation (default: 0.7)
        include_flow_execution: If True, generate one flow execution QA set (default: True)
    
    Returns:
        Dict with:
        - design_name: Design name
        - questions: List of all QA dicts with metadata
        - summary: Statistics about generated QA pairs
    
    QA Types Generated:
        1. Flow Execution (1 set): High-level execution status, stages, sub-designs
        2. Single Stage (n_single_stage sets): Individual stage metrics and analysis
        3. Multi-Stage (n_multi_stage sets): Compare metrics across 2-3 stages (excluding synthesis)
        4. Stage-to-Final (n_stage_to_final sets): Correlation between intermediate and final stages
    
    Example:
        >>> result = generate_all_qa_pairs(
        ...     "design_metrics.json",
        ...     n_single_stage=10,
        ...     n_multi_stage=3,
        ...     n_stage_to_final=2,
        ...     seed=42
        ... )
        >>> print(f"Total QA pairs: {result['summary']['total_qa_pairs']}")
    """
    payload = _normalize_aggregate(_read_json(aggregate_metrics_json_path))
    design_name = payload.get("design_name") or Path(aggregate_metrics_json_path).stem.replace("_all_stages_metrics", "")

    rng = random.Random(seed)
    questions: List[Dict[str, Any]] = []
    
    # Create rejection statistics tracker
    rejection_stats = RejectionStats()
    
    # Track sampled items to avoid duplicates
    sampled_single_stages = set()  # (stage_name, occurrence) tuples
    sampled_multi_stage_combos = set()  # frozenset of (stage, occurrence) tuples
    sampled_stage_to_final = set()  # (stage, occurrence) tuples
    
    # Track statistics
    stats = {
        "flow_execution": 0,
        "single_stage": 0,
        "multi_stage": 0,
        "stage_to_final": 0,
        "total_sets": 0,
        "total_qa_pairs": 0,
        "failed": 0,
    }
    
    print(f"\n{'='*80}")
    print(f"Generating All QA Pairs for {design_name}")
    print(f"{'='*80}")
    if include_flow_execution:
        print(f"  - 1 flow execution QA set")
    print(f"  - {n_single_stage} single-stage QA sets")
    print(f"  - {n_multi_stage} multi-stage QA sets")
    print(f"  - {n_stage_to_final} stage-to-final QA sets")
    print(f"{'='*80}\n")
    
    # 1. Generate flow execution QA pair if requested
    if include_flow_execution:
        print(f"[1/4] Generating flow execution QA...")
        try:
            q = flow_execution_qa_pair(
                aggregate_metrics_json_path,
                seed=rng.randint(0, 2**31 - 1),
                temperature=temperature
            )
            q["id"] = "Q_FLOW_001"
            q["type"] = "flow_execution"
            questions.append(q)
            qa_count = len(q.get('qa_pairs', []))
            stats['flow_execution'] += qa_count
            stats['total_sets'] += 1
            stats['total_qa_pairs'] += qa_count
            print(f"  ✓ Generated Q_FLOW_001: {qa_count} QA pair(s) for flow execution\n")
        except Exception as e:
            print(f"  ✗ Failed to generate flow execution QA: {e}\n")
            stats['failed'] += 1
    
    # 2. Generate single-stage QA pairs with duplicate avoidance
    print(f"[2/4] Generating {n_single_stage} single-stage QA sets...")
    max_retries = n_single_stage * 5
    attempts = 0
    generated_single = 0
    
    while generated_single < n_single_stage and attempts < max_retries:
        attempts += 1
        try:
            q = single_stage_qa_pair(
                aggregate_metrics_json_path, 
                seed=rng.randint(0, 2**31 - 1),
                temperature=temperature,
                rejection_stats=rejection_stats
            )
            
            # Check if this stage has been sampled before
            stage_key = (q.get('stage'), q.get('occurrence'))
            if stage_key in sampled_single_stages:
                print(f"  [SKIP] Stage {stage_key[0]} (occurrence {stage_key[1]}) already sampled, retrying...")
                continue
            
            # Add to sampled set and questions list
            sampled_single_stages.add(stage_key)
            q["id"] = f"Q_SINGLE_{generated_single+1:03d}"
            q["type"] = "single_stage"
            questions.append(q)
            
            qa_count = len(q.get('qa_pairs', []))
            stats['single_stage'] += qa_count
            stats['total_sets'] += 1
            stats['total_qa_pairs'] += qa_count
            print(f"  ✓ Generated Q_SINGLE_{generated_single+1:03d}: {qa_count} QA pair(s) for stage {q.get('stage', 'unknown')}")
            generated_single += 1
            
        except Exception as e:
            print(f"  ✗ Failed to generate single-stage QA (attempt {attempts}): {e}")
            continue
    
    if generated_single < n_single_stage:
        print(f"  [WARN] Only generated {generated_single}/{n_single_stage} unique single-stage QA sets after {attempts} attempts")
        stats['failed'] += (n_single_stage - generated_single)
    print()
    
    # 3. Generate multi-stage QA pairs (excluding synthesis) with duplicate avoidance
    print(f"[3/4] Generating {n_multi_stage} multi-stage QA sets (excluding synthesis)...")
    non_synthesis_stages = ['floorplan', 'placement', 'cts', 'routing', 'final']
    
    max_retries_multi = n_multi_stage * 10
    attempts_multi = 0
    generated_multi = 0
    
    while generated_multi < n_multi_stage and attempts_multi < max_retries_multi:
        attempts_multi += 1
        try:
            # Vary the number of stages (2 or 3) randomly
            num_stages = rng.choice([2, 3])
            
            q = multi_stage_qa_pair(
                aggregate_metrics_json_path,
                num_stages=num_stages,
                stage_types=non_synthesis_stages,  # Exclude synthesis
                seed=rng.randint(0, 2**31 - 1),
                temperature=temperature,
                rejection_stats=rejection_stats
            )
            
            # Create a unique key for this combination
            stages_info = q.get('stages_info', [])
            combo_key = frozenset((s['stage'], s['occurrence']) for s in stages_info)
            
            if combo_key in sampled_multi_stage_combos:
                stages_list = [s['stage'] for s in stages_info]
                print(f"  [SKIP] Multi-stage combination {stages_list} already sampled, retrying...")
                continue
            
            # Add to sampled set
            sampled_multi_stage_combos.add(combo_key)
            
            q["id"] = f"Q_MULTI_{generated_multi+1:03d}"
            q["type"] = "multi_stage"
            questions.append(q)
            
            qa_count = len(q.get('qa_pairs', []))
            stages_list = [s['stage'] for s in stages_info]
            stats["multi_stage"] += qa_count
            stats["total_sets"] += 1
            stats["total_qa_pairs"] += qa_count
            print(f"  ✓ Generated Q_MULTI_{generated_multi+1:03d}: {qa_count} QA pair(s) for stages {stages_list}")
            generated_multi += 1
            
        except Exception as e:
            print(f"  ✗ Failed to generate multi-stage QA (attempt {attempts_multi}): {e}")
            continue
    
    if generated_multi < n_multi_stage:
        print(f"  [WARN] Only generated {generated_multi}/{n_multi_stage} unique multi-stage QA sets after {attempts_multi} attempts")
        stats["failed"] += (n_multi_stage - generated_multi)
    print()
    
    # 4. Generate stage-to-final QA pairs (excluding synthesis) with duplicate avoidance
    print(f"[4/4] Generating {n_stage_to_final} stage-to-final QA sets (excluding synthesis)...")
    
    max_retries_final = n_stage_to_final * 10
    attempts_final = 0
    generated_final = 0
    
    while generated_final < n_stage_to_final and attempts_final < max_retries_final:
        attempts_final += 1
        try:
            # Specify stage types for variety (excluding synthesis)
            stage_types = ['placement', 'routing', 'cts', 'floorplan']
            stage_type = rng.choice(stage_types)
            
            q = stage_to_final_qa_pair(
                aggregate_metrics_json_path,
                stage_type=stage_type,
                seed=rng.randint(0, 2**31 - 1),
                temperature=temperature,
                rejection_stats=rejection_stats
            )
            
            # Create a unique key for this stage-to-final pair
            stage_info = q.get('stage_info', {})
            stage_key = (stage_info.get('stage'), stage_info.get('occurrence'))
            
            if stage_key in sampled_stage_to_final:
                print(f"  [SKIP] Stage-to-final {stage_key[0]} (occurrence {stage_key[1]}) already sampled, retrying...")
                continue
            
            # Add to sampled set
            sampled_stage_to_final.add(stage_key)
            
            q["id"] = f"Q_FINAL_{generated_final+1:03d}"
            q["type"] = "stage_to_final"
            questions.append(q)
            
            qa_count = len(q.get('qa_pairs', []))
            stage_name = stage_info.get('stage', 'unknown')
            stats["stage_to_final"] += qa_count
            stats["total_sets"] += 1
            stats["total_qa_pairs"] += qa_count
            print(f"  ✓ Generated Q_FINAL_{generated_final+1:03d}: {qa_count} QA pair(s) for {stage_name} → final")
            generated_final += 1
            
        except Exception as e:
            print(f"  ✗ Failed to generate stage-to-final QA (attempt {attempts_final}): {e}")
            continue
    
    if generated_final < n_stage_to_final:
        print(f"  [WARN] Only generated {generated_final}/{n_stage_to_final} unique stage-to-final QA sets after {attempts_final} attempts")
        stats["failed"] += (n_stage_to_final - generated_final)
    print()
    
    # Print summary
    print(f"{'='*80}")
    print(f"Generation Summary for {design_name}")
    print(f"{'='*80}")
    if include_flow_execution:
        print(f"  Flow Execution:    {stats['flow_execution']:3d} QA pairs")
    print(f"  Single-Stage:      {stats['single_stage']:3d} QA pairs")
    print(f"  Multi-Stage:       {stats['multi_stage']:3d} QA pairs")
    print(f"  Stage-to-Final:    {stats['stage_to_final']:3d} QA pairs")
    print(f"  {'─'*40}")
    print(f"  Total Sets:        {stats['total_sets']:3d}")
    print(f"  Total QA Pairs:    {stats['total_qa_pairs']:3d}")
    print(f"  Failed:            {stats['failed']:3d}")
    print(f"{'='*80}")
    
    # Print rejection statistics prominently
    rejection_summary = rejection_stats.get_summary()
    print(f"\n{'='*80}")
    print(f"ANSWER VALIDATION & REJECTION REPORT")
    print(f"{'='*80}")
    rejection_stats.print_summary(prefix="")
    print(f"{'='*80}\n")
    
    # Add metadata to each question
    for q in questions:
        q['aggregate_metrics_json_path'] = str(aggregate_metrics_json_path)
    
    # Create output directory if needed
    output_dir = Path(qa_json_path).resolve()
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write results with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Write raw data (full question objects with metadata)
    out_path_raw = output_dir / f"{design_name}_{timestamp}_qa_all_types_raw.json"
    with out_path_raw.open("w", encoding="utf-8") as f:
        json.dump({
            "design_name": design_name,
            "log_file": payload.get("log_file"),
            "source": str(aggregate_metrics_json_path),
            "timestamp": timestamp,
            "statistics": stats,
            "rejection_statistics": rejection_stats.get_summary(),
            "generation_params": {
                "n_single_stage": n_single_stage,
                "n_multi_stage": n_multi_stage,
                "n_stage_to_final": n_stage_to_final,
                "seed": seed,
                "temperature": temperature,
                "include_flow_execution": include_flow_execution,
            },
            "questions": questions,
        }, f, indent=2, ensure_ascii=False)
    print(f"[OK] Wrote {len(questions)} QA sets (raw) → {out_path_raw}")
    
    # 2. Write QA pairs only (flattened)
    questions_and_answers = []
    for q in questions:
        qa_type = q.get('type', 'unknown')
        qa_id = q.get('id', 'unknown')
        
        for idx, qa in enumerate(q.get('qa_pairs', [])):
            qa_dict = {
                "id": f"{qa_id}_P{idx+1:02d}",
                "type": qa_type,
                "question": qa['question'],
                "answer": qa['answer'],
            }
            
            # Add stage info for context based on type
            if qa_type == "single_stage":
                qa_dict["stage"] = q.get('stage', 'unknown')
                qa_dict["occurrence"] = q.get('occurrence', 1)
            elif qa_type == "multi_stage":
                qa_dict["stages"] = [s['stage'] for s in q.get('stages_info', [])]
            elif qa_type == "stage_to_final":
                qa_dict["stage"] = q.get('stage_info', {}).get('stage')
                qa_dict["final"] = q.get('final_info', {}).get('stage')
            
            questions_and_answers.append(qa_dict)
    
    out_path = output_dir / f"{design_name}_{timestamp}_qa_all_types.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "design_name": design_name,
            "timestamp": timestamp,
            "log_file": payload.get("log_file"),
            "source": str(aggregate_metrics_json_path),
            "statistics": stats,
            "rejection_statistics": rejection_stats.get_summary(),
            "qa_pairs": questions_and_answers,
        }, f, indent=2, ensure_ascii=False)
    print(f"[OK] Wrote {len(questions_and_answers)} QA pairs (flattened) → {out_path}")
    
    # Print final comprehensive report
    rejection_summary = rejection_stats.get_summary()
    print(f"\n{'='*80}")
    print(f"FINAL REPORT: {design_name}")
    print(f"{'='*80}")
    print(f"📁 Output Files:")
    print(f"   Raw Data:      {out_path_raw}")
    print(f"   QA Pairs:      {out_path}")
    print(f"\n📊 Generation Statistics:")
    print(f"   Total QA Pairs Generated:     {stats['total_qa_pairs']}")
    if include_flow_execution:
        print(f"   Flow Execution QA:            {stats['flow_execution']}")
    print(f"   Single-Stage QA:              {stats['single_stage']}")
    print(f"   Multi-Stage QA:               {stats['multi_stage']}")
    print(f"   Stage-to-Final QA:            {stats['stage_to_final']}")
    print(f"\n✅ Answer Validation Results:")
    print(f"   Total Attempts:               {rejection_summary['total_attempts']}")
    print(f"   ✓ Successful:                 {rejection_summary['successful']} ({rejection_summary['success_rate']:.1f}%)")
    print(f"   ✗ Rejected:                   {rejection_summary['rejected']['total_rejected']}")
    print(f"      • Code Execution Failed:   {rejection_summary['rejected']['code_execution_failed']}")
    print(f"      • Low Validation Score:    {rejection_summary['rejected']['low_score_rejected']}")
    print(f"      • Invalid Answer Format:   {rejection_summary['rejected']['invalid_answer_rejected']}")
    
    if rejection_summary['rejected']['low_score_rejected'] > 0:
        print(f"\n⚠️  Low Score Rejections: {rejection_summary['rejected']['low_score_rejected']} cases")
        print(f"   (Details saved in rejection_statistics.low_score_details)")
    
    print(f"{'='*80}\n")
    
    return {
        "design_name": design_name,
        "questions": questions,
        "summary": stats,
        "rejection_statistics": rejection_stats.get_summary(),
    }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Generate QA pairs from *_all_stages_metrics.json using LLM")
    ap.add_argument("-json", "--aggregate_json", help="Path to *_all_stages_metrics.json")
    ap.add_argument("-n", type=int, default=10, help="Number of stage-level QA pairs to generate (default: 10)")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    ap.add_argument("--temperature", type=float, default=0.7, help="LLM temperature (default: 0.7)")
    ap.add_argument("-o", "--output_path", default="./", help="Output JSON path (default: ./)")
    ap.add_argument("--mode", choices=["all", "single", "multi"], default="all",
                    help="Generation mode: 'all' for unified generation, 'single' for single-stage only, 'multi' for multi-stage only (default: all)")
    args = ap.parse_args()

    print(f"Output path: {args.output_path}")
    print(f"Aggregate JSON: {args.aggregate_json}")
    print(f"Number of QA pairs: {args.n}")
    print(f"Seed: {args.seed}")
    print(f"Temperature: {args.temperature}")
    print(f"Mode: {args.mode}")

    if args.mode == "all":
        # Use the new unified function
        out = generate_all_qa_pairs(
            args.aggregate_json,
            n_single_stage=args.n,
            n_multi_stage=args.n,
            n_stage_to_final=args.n,
            seed=args.seed,
            qa_json_path=args.output_path,
            temperature=args.temperature,
            include_flow_execution=True
        )
    elif args.mode == "single":
        # Use the original single-stage function
        out = generate_qa_pairs(
            args.aggregate_json, 
            n=args.n, 
            seed=args.seed,
            qa_json_path=args.output_path,
            temperature=args.temperature,
            include_flow_execution=True
        )
    elif args.mode == "multi":
        # Use the original multi-stage function
        out = generate_multistages_qa_pairs(
            args.aggregate_json,
            n_multi_stage=args.n,
            n_stage_to_final=args.n,
            seed=args.seed,
            qa_json_path=args.output_path,
            temperature=args.temperature
        )

if __name__ == "__main__":
    main()
