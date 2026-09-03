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
Script to load masked Q&A files and prepare prompts for LLM testing.

This script:
1. Loads masked Q&A JSON files
2. Prepares formatted prompts for LLM testing
3. Provides utilities to collect and evaluate LLM responses
4. Supports batch testing and result export
"""

import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import os
import subprocess
import pdb
from datetime import datetime
import csv
import argparse
from rag_agent import RAGAgent
# from llm import llm_inference, call_dar_gateway # token limits exceed
current_path = os.getcwd()
CURSOR_AGENT_PATH = os.environ.get(
    "CURSOR_AGENT_PATH"
)
CLAUDE_CODE_PATH = os.environ.get(
    "CLAUDE_CODE_PATH"
)

openroad_background = """In modern digital design flows, the design flow is from synthesis, floorplan, placement, cts (clo,ck tree synthesis), routing, and final stages. The EDA tools emit detailed log files throughout this process, capturing a hierarchy of stepsâfrom synthesis, floorplan, placement, cts, routing, and final. These logs serve as a chronological record.

A typical design flow log file includes three core elements:
1. ImportantMessage that include multi-line or single line messages related to commands, warnings, errors, buffer insertion, and any related to timing metrics (TNS, WNS), HPWL, total wirelength, area, overflow, density, powre, and IR drop.
2. Metric tables reporting timing (Worst Negative Slack, Total Negative Slack), design density, resource usage, overflow, HPWL, total wirelength, area, runtime, and memory for each iteration.
3. Stage markers and hierarchy, which reveal where sub-flows (e.g., Synthesis -> Floorplan -> GlobalPlace -> DetailedPlace -> CTS -> Route) occur and help pinpoint bottlenecks or time degradations. The stages should be chronically ordered in the final schema.
4. The synthesis information from YOSYS contains the YOSYS github version, and the warning and error messages. 
5. The HPWL is estimated from floorplan and optimized in the placement stages. The final HPWL estimation is usually reported in detailed placement stage (i.e., 3_5_place_dp). In the routing stage, the actual wirelength is reported.
"""
stage_mapping = """Below is the stage mapping:
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
"""

def extract_cost_from_polymath(cost_dict):
    """
    Extracts all cost values from a nested polymath cost dictionary.
    The first element in the returned list is the total sum of all costs.

    Example:
        cost_dict = {
            'graph_planning': 0.0004299,
            'graph_opt': 0,
            'graph_status_update': 0.00024074999999999997,
            'assistance': {
                'planning': 0.0208928,
                'workflow_execution': 0.00042119999999999994,
                'retrieve_workflow': 0.00045015,
                'evaluation': 0.0013931999999999998,
                'evolve': 0.0006313499999999999
            }
        }

    Returns:
        list: [total_cost, individual_cost_1, individual_cost_2, ...]
    """
    costs = []

    for key, value in cost_dict.items():
        if key == "assistance" and isinstance(value, dict):
            costs.extend(value.values())
        else:
            costs.append(value)

    total_cost = sum(costs)
    return [total_cost] + costs

@dataclass
class Question:
    """Data class representing a single question."""
    id: str
    difficulty: str
    stage: str
    question: str
    answer: Any
    answer_placeholder: str
    question_type: Optional[str] = None  # New: flow_execution or stage_level
    occurrence: Optional[int] = None  # New: for stage-level questions
    
    def to_dict(self):
        return asdict(self)


@dataclass
class LLMResponse:
    """Data class representing an LLM's response to a question."""
    question_id: str
    question_text: str
    llm_answer: Any
    expected_format: str  # "simple" or "dict"
    difficulty: str
    stage: str


class QAPromptGenerator:
    """Generator for LLM test prompts from masked Q&A files."""
    
    def __init__(self, masked_qa_path: str):
        """
        Initialize with a masked Q&A file.
        
        Args:
            masked_qa_path: Path to the masked Q&A JSON file
        """
        self.qa_path = Path(masked_qa_path)
        self.data = self._load_qa_file()
        self.metrics_file_path = self.data.get("source", None)
        self.design_name = self.data.get("design_name", "")
        self.statistics = self.data.get("statistics", None)  # New: preserve statistics if available
        # self.log_file_dir = os.path.dirname(self.log_file_path)
        self.log_file_path = self.data.get("log_file", None)
        self.log_file_dir = os.path.dirname(self.log_file_path)
        self.log_file = self._load_log_file_content()  # Load log file content for LLM inference
        self.questions = self._parse_questions()
    
    def _load_qa_file(self) -> Dict:
        """Load the masked Q&A JSON file."""
        if not self.qa_path.exists():
            raise FileNotFoundError(f"Q&A file not found: {self.qa_path}")
        
        with open(self.qa_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Handle different JSON formats
        # Format 1: List of Q&A objects (llm.json format)
        if isinstance(data, list):
            return {
                "design_name": self._extract_design_name_from_filename(),
                "questions": data
            }
        # Format 2: Dictionary with qa_pairs (multistages.json format and new format)
        elif isinstance(data, dict) and "qa_pairs" in data:
            return {
                "design_name": data.get("design_name", self._extract_design_name_from_filename()),
                "questions": data.get("qa_pairs", []),
                "statistics": data.get("statistics", None),
                "timestamp": data.get("timestamp", None),
                "log_file": data.get("log_file", None),
                "source": data.get("source", None)
            }
        # Format 3: Standard format with questions array
        else:
            return data
    
    def _load_log_file_content(self) -> Optional[str]:
        """Load the log file content for LLM inference."""
        log_file_path = self.data.get("log_file", None)
        
        if not log_file_path:
            print("⚠️  No log_file specified in the Q&A file")
            return None
        
        # Convert to Path object and resolve relative to workspace root
        log_path = Path(log_file_path)
        
        # If the path is relative, resolve it from the Q&A file's parent directory
        if not log_path.is_absolute():
            # Try relative to the workspace root (where the script is run)
            if log_path.exists():
                pass  # Use as is
            # Try relative to the Q&A file location
            elif (self.qa_path.parent.parent / log_file_path).exists():
                log_path = self.qa_path.parent.parent / log_file_path
            else:
                print(f"⚠️  Log file not found: {log_file_path}")
                print(f"    Searched: {log_path} and {self.qa_path.parent.parent / log_file_path}")
                return None
        
        if not log_path.exists():
            print(f"⚠️  Log file not found: {log_path}")
            return None
        
        try:
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            print(f"✅ Loaded log file: {log_path} ({len(content)} chars)")
            return content
        except Exception as e:
            print(f"❌ Error reading log file {log_path}: {e}")
            return None
    
    def _extract_design_name_from_filename(self) -> str:
        """Extract design name from the filename."""
        filename = self.qa_path.stem
        # Extract design name from pattern like: uart-blocks_20260121_173733_questions_and_answers_llm
        parts = filename.split('_')
        if parts:
            return parts[0]
        return "unknown"
    
    def _parse_questions(self) -> List[Question]:
        """Parse questions from the loaded data."""
        questions = []
        for idx, q in enumerate(self.data.get("questions", [])):
            # Extract question text and parse difficulty
            question_text = q.get("question", "")
            
            # Parse difficulty from question text if present
            difficulty = q.get("difficulty", "")
            if not difficulty and "Difficulty:" in question_text:
                # Extract difficulty from text like "Difficulty: easy"
                parts = question_text.split("Difficulty:")
                if len(parts) > 1:
                    difficulty = parts[1].strip().lower()
                    # Clean up question text to remove difficulty suffix
                    question_text = parts[0].strip()
            
            # Extract stage information
            stage = q.get("stage", q.get("stages", ["unknown"])[0] if "stages" in q else "unknown")
            if isinstance(stage, list):
                stage = stage[0] if stage else "unknown"
            
            # Extract question type (new field)
            question_type = q.get("type", None)
            
            # Extract occurrence (new field for stage-level questions)
            occurrence = q.get("occurrence", None)
            
            # Handle different answer formats
            answer = q.get("answer", "")
            if isinstance(answer, dict):
                answer_placeholder = "[dict format]"
            elif isinstance(answer, list):
                answer_placeholder = "[list format]"
            elif isinstance(answer, str) and ("```json" in answer or (answer.strip().startswith("{") and answer.strip().endswith("}"))):
                # String contains JSON (either in markdown code blocks or raw JSON)
                answer_placeholder = "[dict format]"
            else:
                answer_placeholder = str(answer) if answer else ""
            
            # Generate ID if not present
            question_id = q.get("id", f"Q{idx+1:03d}")
            
            question = Question(
                id=question_id,
                difficulty=difficulty,
                stage=stage,
                question=question_text,
                answer=answer,  # Add the actual answer field
                answer_placeholder=answer_placeholder,
                question_type=question_type,
                occurrence=occurrence
            )
            questions.append(question)
        return questions
    
    def get_question(self, question_id: str) -> Optional[Question]:
        """Get a specific question by ID."""
        for q in self.questions:
            if q.id == question_id:
                return q
        return None
    
    def get_questions_by_stage(self, stage: str) -> List[Question]:
        """Get all questions for a specific stage."""
        return [q for q in self.questions if q.stage == stage]
    
    def get_questions_by_difficulty(self, difficulty: str) -> List[Question]:
        """Get all questions by difficulty level."""
        return [q for q in self.questions if q.difficulty == difficulty]
    
    def get_questions_by_type(self, question_type: str) -> List[Question]:
        """Get all questions by type (flow_execution or stage_level)."""
        return [q for q in self.questions if q.question_type == question_type]
    
    def create_prompt_with_output_json_key_hints(self, question: Question, context: str = "") -> str:
        """
        Create a prompt with JSON output key hints to guide the LLM's response format.
        Only works for questions where the answer is in dictionary/JSON format.
        
        Args:
            question: Question object
            context: Optional context (not used currently)
        
        Returns:
            Formatted prompt string with JSON key hints
        """
        log_file_basename = os.path.basename(self.log_file_path)
        prompt = f"""You are analyzing the log file '{log_file_basename}' in current directory. There are some other *rpt files for reference in the current directory. You only need the log file and *rpt to answer, do not use any other files under subdirectory.

Log File Background:
{openroad_background}
   
Question: {question.question}
    
"""
        
        # Only add JSON key hints if the answer is in dictionary format
        if "dict format" in question.answer_placeholder or isinstance(question.answer, dict):
            try:
                # Parse the answer if it's a string, otherwise use as-is if it's already a dict
                if isinstance(question.answer, str):
                    # Extract JSON from markdown code blocks if present
                    answer_str = question.answer
                    if "```json" in answer_str:
                        # Extract content between ```json and ```
                        start_marker = "```json"
                        end_marker = "```"
                        start_idx = answer_str.find(start_marker)
                        if start_idx != -1:
                            start_idx += len(start_marker)
                            end_idx = answer_str.find(end_marker, start_idx)
                            if end_idx != -1:
                                answer_str = answer_str[start_idx:end_idx].strip()
                    
                    answer_json = deepcopy(json.loads(answer_str))
                else:
                    answer_json = deepcopy(question.answer)
                
                # Create a template with placeholder values; the nested structure should be preserved
                answer_template = deepcopy(answer_json)
                # traverse the answer_template recursively and replace the value with "<your answer here>"
                def traverse_and_replace(d):
                    for key, value in d.items():
                        if isinstance(value, dict):
                            traverse_and_replace(value)
                        else:
                            d[key] = "<your answer here>"
                traverse_and_replace(answer_template)
                answer_json_str = json.dumps(answer_template, indent=2)
                prompt += f"Please provide your final answer in dictionary/JSON format within ```json and ```.\n{answer_json_str}\n"
            except Exception as e:
                print(f"⚠️  Error creating JSON key hints for question {question.id}: {e}")
                prompt += "Please provide your final answer in dictionary/JSON format within ```json and ```.\n"
        else:
            # For non-dictionary answers, provide standard instruction
            prompt += "Please provide your final answer value only within ** and **.\n"
        
        return prompt
        
    def create_prompt_with_background(self, question: Question, context: str = "") -> str:
        difficulty = question.difficulty
        if difficulty == "easy":
            max_subtasks = 3
        else:
            max_subtasks = 10
        log_file_basename = os.path.basename(self.log_file_path)
        prompt = f"""You are analyzing the log file '{log_file_basename}' in current directory. There are some other *rpt files for reference in the current directory. You only need the log file and *rpt to answer, do not use any other files under subdirectory.

Log File Background:
{openroad_background}
   
Question: {question.question}
    
"""
        if "dict format" in question.answer_placeholder:
            prompt += "Please provide your final answer in dictionary/JSON format within ```json and ```.\n"
        else:
            prompt += "Please provide your final answer value onlywithin ** and **.\n"
        
        prompt += "\nAnswer: " + question.answer_placeholder
        
        return prompt
    
    def create_prompt_with_log_content(self, question: Question, include_full_log: bool = False) -> str:
        """
        Create a prompt for LLM inference with log file content.
        
        Args:
            question: Question object
            include_full_log: If True, include full log content. If False, just reference it.
        
        Returns:
            Formatted prompt string with log file content for LLM inference
        """
        difficulty = question.difficulty
        if difficulty == "easy":
            max_subtasks = 3
        else:
            max_subtasks = 10
        
        # Base prompt
        log_file_name = self.data.get("log_file", f"{self.design_name}.txt")
        prompt = f"""You are analyzing the log file '{log_file_name}'. """
        
        if self.log_file and include_full_log:
            prompt += f"""The complete log file content is provided below:

<log_file>
{self.log_file}
</log_file>

"""
        elif self.log_file:
            prompt += f"""The log file content has been loaded ({len(self.log_file)} chars). """
        
        prompt += f"""This is a {difficulty} difficulty QA task. Generate the subtasks less than {max_subtasks}. You need to reply the answer only without additional information.

Log File Background:
{openroad_background}

For clock buffer insertion or clock net related questions, if the output is a dictionary, you need to reply the answer with clock net name if there are multiple clock nets.
For example:
{{
    "clock_net_name1": ...,
    "clock_net_name2": ...,
}}

For some QAs that want to know the clock buffer/cell/area is not created in certain stages, you need to examine design final stage to see the final clock buffer/cell/area information.

Question: {question.question}

"""
        
        if "dict format" in question.answer_placeholder or "list format" in question.answer_placeholder:
            prompt += "Please provide your final answer in dictionary/JSON format within ```json and ```.\n"
        else:
            prompt += "Please provide your final answer value only within ** and **.\n"
        
        return prompt
    
    def create_simple_prompt(self, question: Question, context: str = "") -> str:
        """
        Create a simple prompt for a question.
        
        Args:
            question: Question object
            context: Optional context information (e.g., log data)
        
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are analyzing the log file '{self.data.get("log_file", "unknown")}' in current directory. It is a simple QA task. Do not write any code for answering the question. You are given a question and you need to reply the answer only without additional information.

Question: {question.question}

"""
        if context:
            prompt += f"""Context/Log Data:
{context}

"""
        
        if "dict format" in question.answer_placeholder:
            prompt += "Please provide your answer in dictionary/JSON format within ```json and ```.\n"
        else:
            prompt += "Please provide your answer within ** and **.\n"
        
        prompt += "\nAnswer: " + question.answer_placeholder
        
        return prompt
    
    def create_batch_prompts(self, context: str = "") -> List[Dict[str, Any]]:
        """
        Create prompts for all questions in batch.
        
        Args:
            context: Optional context for all questions
        
        Returns:
            List of dictionaries containing prompt data for each question
        """
        prompts = []
        for question in self.questions:
            prompt_data = {
                "question_id": question.id,
                "difficulty": question.difficulty,
                "stage": question.stage,
                "question_text": question.question,
                "answer_placeholder": question.answer_placeholder,
                "prompt": self.create_prompt_with_background(question, context)
            }
            prompts.append(prompt_data)
        return prompts
        
    def export_prompts_to_file(self, 
                               output_path: str,
                               context: str = "",
                               format: str = "json") -> None:
        """
        Export all prompts to a file for batch processing.
        
        Args:
            output_path: Path to output file
            context: Optional context for all questions
            format: Output format ("json" or "txt")
        """
        prompts = self.create_batch_prompts(context)
        output_path = Path(output_path)
        
        if format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(prompts, f, indent=2, ensure_ascii=False)
        elif format == "txt":
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, prompt_data in enumerate(prompts):
                    f.write(f"=" * 80 + "\n")
                    f.write(f"Question {i+1}/{len(prompts)}: {prompt_data['question_id']}\n")
                    f.write(f"Difficulty: {prompt_data['difficulty']} | Stage: {prompt_data['stage']}\n")
                    f.write(f"=" * 80 + "\n\n")
                    f.write(prompt_data['prompt'])
                    f.write("\n\n" + "=" * 80 + "\n\n")
        
        print(f"â Exported {len(prompts)} prompts to {output_path}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the Q&A dataset."""
        stages = {}
        difficulties = {}
        question_types = {}
        
        for q in self.questions:
            stages[q.stage] = stages.get(q.stage, 0) + 1
            difficulties[q.difficulty] = difficulties.get(q.difficulty, 0) + 1
            if q.question_type:
                question_types[q.question_type] = question_types.get(q.question_type, 0) + 1
        
        dict_questions = sum(1 for q in self.questions if "dict format" in q.answer_placeholder)
        list_questions = sum(1 for q in self.questions if "list format" in q.answer_placeholder)
        simple_questions = len(self.questions) - dict_questions - list_questions
        
        return {
            "design_name": self.design_name,
            "total_questions": len(self.questions),
            "by_stage": stages,
            "by_difficulty": difficulties,
            "by_type": question_types,
            "simple_answers": simple_questions,
            "dict_answers": dict_questions,
            "list_answers": list_questions
        }
    
    def get_file_statistics(self) -> Optional[Dict[str, Any]]:
        """Get the statistics from the loaded file (if available in new format)."""
        return self.statistics
    
    def get_file_metadata(self) -> Dict[str, Any]:
        """Get metadata about the loaded file."""
        return {
            "design_name": self.design_name,
            "timestamp": self.data.get("timestamp", None),
            "log_file": self.data.get("log_file", None),
            "log_file_loaded": self.log_file is not None,
            "log_file_size": len(self.log_file) if self.log_file else 0,
            "source": self.data.get("source", None),
            "statistics": self.statistics
        }


class LLMResponseCollector:
    """Collector for storing and managing LLM responses."""
    
    def __init__(self, design_name: str):
        """Initialize the response collector."""
        self.design_name = design_name
        self.responses: List[LLMResponse] = []
    
    def add_response(self, 
                    question_id: str,
                    question_text: str,
                    llm_answer: Any,
                    expected_format: str,
                    difficulty: str,
                    stage: str) -> None:
        """Add an LLM response."""
        response = LLMResponse(
            question_id=question_id,
            question_text=question_text,
            llm_answer=llm_answer,
            expected_format=expected_format,
            difficulty=difficulty,
            stage=stage
        )
        self.responses.append(response)
    
    def export_responses(self, output_path: str) -> None:
        """Export collected responses to JSON file."""
        output_data = {
            "design_name": self.design_name,
            "total_responses": len(self.responses),
            "responses": [asdict(r) for r in self.responses]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"â Exported {len(self.responses)} responses to {output_path}")
    
    def load_responses(self, input_path: str) -> None:
        """Load responses from a JSON file."""
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.design_name = data.get("design_name", "")
        self.responses = [
            LLMResponse(**r) for r in data.get("responses", [])
        ]
        
        print(f"â Loaded {len(self.responses)} responses from {input_path}")

def agent_inference(qid: str, prompt: str, dir_path: str, model: str = "sonnet-4.5", csv_file: str = "test_responses.csv", design_name: str = None, json_file: str = None) -> str:
    """
    Inference the prompt using the agent; Not used now.
    """
    # Run cursor Polymath
    print("[Prompt]: ", prompt)
    # agent = PolymathConfig.graph_based_logQA_evolved_code_general_agent_config(benchmark_name="logQA_" + design_name, work_plan_dir="./work/graph_reflect_evolvecode/", log_background=openroad_background, json_file=json_file)
    # agent.clear_history()
    # answer = agent.initiate_chat(prompt)
    
    # get the cost
    # cost = agent.get_token_cost() # get the cost
    # cost_arrays = extract_cost_from_polymath(cost)
    # total_cost, graph_planning_cost, graph_opt_cost, garph_status_update, code_planning, workflow_execution_cost, retrieve_workflow_cost, evaluation_cost, evolve_cost = tuple(cost_arrays)

    # write the answer to the csv file
    # with open(csv_file, 'a', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow([qid, answer, total_cost, graph_planning_cost, graph_opt_cost, garph_status_update, code_planning, workflow_execution_cost, retrieve_workflow_cost, evaluation_cost, evolve_cost])
    # print("[Agent Answer]: ", answer)
    return

def cursor_agent_inference(qid: str, prompt: str, log_file_path: str, model: str = "sonnet-4.5") -> str:
    """
    Inference the prompt using the cursor agent.
    """
    # Run cursor
    # get current working directory for cursor agent path
    current_path = os.getcwd()
    
    # change to the log file directory
    # get the log file basename: 
    log_file_basename = os.path.basename(log_file_path)
    log_file_dir = os.path.dirname(log_file_path)
    os.chdir(log_file_dir)
    # change the directory to the question directory
        
    print(f"Changed directory to: {log_file_dir}")
    # list the files in the current directory
    print(f"Files in the current directory: {os.listdir()}")
    # pdb.set_trace()
    # output = subprocess.run([CURSOR_AGENT_PATH, "-p", enhanced_problem_statement, "--model", "claude-4.5-sonnet"], capture_output=True, text=True)
    # output = subprocess.run([CURSOR_AGENT_PATH, "-p", prompt, "--model", "claude-4.5-sonnet"], capture_output=True, text=True)
    print(f"Running cursor agent with model: {model}")
    output = subprocess.run([CURSOR_AGENT_PATH, "-p", prompt, "--model", model], capture_output=True, text=True)
        
    answer = output.stdout.strip()
    
    # change back to the current directory
    os.chdir(current_path)
    print(f"Changed back to the current directory: {current_path}")
    # print current directory
    print(f"Current directory: {os.getcwd()}")
    return answer

def claude_code_agent_inference(qid: str, prompt: str, log_file_path: str, model: str = "sonnet-4.5") -> str:
    """
    Inference the prompt using the claude code agent.
    """
    # Run claude code
    # get current working directory for claude code agent path
    current_path = os.getcwd()
    
    # change to the log file directory
    # get the log file basename: 
    log_file_basename = os.path.basename(log_file_path)
    log_file_dir = os.path.dirname(log_file_path)
    os.chdir(log_file_dir)
    # change the directory to the question directory
        
    print(f"Changed directory to: {log_file_dir}")
    # list the files in the current directory
    print(f"Files in the current directory: {os.listdir()}")
    print(f"Running claude code agent with model: {model}")
    # pdb.set_trace()
    output = subprocess.run([CLAUDE_CODE_PATH, "-p", prompt], capture_output=True, text=True)
        
    answer = output.stdout.strip()
    # pdb.set_trace()
    # change back to the current directory
    os.chdir(current_path)
    print(f"Changed back to the current directory: {current_path}")
    # print current directory
    print(f"Current directory: {os.getcwd()}")
    return answer

def rag_agent_inference(qid: str, prompt: str, rag_agent: RAGAgent) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Run inference with the RAG agent and return the answer plus token usage.
    """
    print(f"Running rag agent with prompt: {prompt}...")
    result = rag_agent.answer(prompt, return_full_result=True)
    answer = result.get("answer", "")
    token_usage = result.get("token_usage")
    print(f"RAG agent answer: {answer}")
    print(f"Token usage: {token_usage}")
    # pdb.set_trace()
    if token_usage:
        print(f"Token usage: {token_usage.get('total_tokens', 0)} total (prompt: {token_usage.get('prompt_tokens', 0)}, completion: {token_usage.get('completion_tokens', 0)})")
    return answer, token_usage

def test_all_questions(qa_file: str, model: str = "gpt-4o", qidxs: List[int] = None, output_dir: str = "./", use_json_hints: bool = False, agent_type: str = "cursor") -> List[Dict[str, Any]]:
    """
    Prepare prompts for all questions in the QA file for LLM inference.
    
    Args:
        qa_file: Path to the Q&A file
        model: Model name to use for testing (for reference only)
        qidxs: List of question indices to test (None = all questions)
        output_dir: Directory to save output files
        use_json_hints: If True, use create_prompt_with_output_json_key_hints; otherwise use create_prompt_with_background
        agent_type: Type of agent to use ("cursor" or "claude_code", default: "cursor")
    
    Returns:
        List of dictionaries containing question info and prepared prompts
    """
    print(f"Loading QA file: {qa_file}")
    print(f"Model: {model}")
    print(f"Output directory: {output_dir}")
    
    # Load QA file and log content
    generator = QAPromptGenerator(qa_file)
    
    # Display loaded information
    print(f"\n📊 Design: {generator.design_name}")
    print(f"📊 Total Questions: {len(generator.questions)}")
    if generator.log_file:
        print(f"✅ Log file loaded: {len(generator.log_file)} chars")
    else:
        print(f"⚠️  No log file content available")
    
    # Get file metadata
    metadata = generator.get_file_metadata()
    print(f"📋 Log File Path: {metadata.get('log_file')}")
    print(f"📋 Source: {metadata.get('source')}")
    
    # Try to initialize the rag agent to reduce the runtime cost
    if agent_type == "rag":
        if not generator.log_file_path:
            raise ValueError("QA file has no 'log_file' path; cannot initialize RAG agent.")
        # log_file_path is the path to the log file (may be .txt or .log); use its directory for listing
        log_dir = os.path.dirname(os.path.abspath(generator.log_file_path))
        if not os.path.isdir(log_dir):
            raise FileNotFoundError(f"Log file directory not found: {log_dir}")
        log_file_paths = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.endswith(".log")]
        rpt_file_paths = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.endswith(".rpt")]
        # Include the referenced log file if it exists and is not a .log (e.g. .txt) so we index it
        ref_path = os.path.abspath(generator.log_file_path)
        if os.path.isfile(ref_path) and ref_path not in [os.path.abspath(p) for p in log_file_paths]:
            log_file_paths = [generator.log_file_path] + log_file_paths
        if not log_file_paths:
            log_file_paths = [generator.log_file_path] if os.path.isfile(generator.log_file_path) else []
        print(f"Log file paths: {log_file_paths}")
        print(f"RPT file paths: {rpt_file_paths}")
        rag_agent = RAGAgent(log_files=log_file_paths, rpt_files=rpt_file_paths)
    
    # Prepare prompts for all questions
    question_responses = []
    
    for qid in range(len(generator.questions)):
        if qidxs is not None and qid not in qidxs:
            continue
        
        question = generator.questions[qid]
        
        # Create prompt based on use_json_hints option
        if use_json_hints:
            prompt = generator.create_prompt_with_output_json_key_hints(question)
        else:
            prompt = generator.create_prompt_with_background(question)
        
        print(f"\n{'='*80}")
        print(f"Question {qid}: {question.id}")
        print(f"Type: {question.question_type} | Stage: {question.stage} | Difficulty: {question.difficulty}")
        print(f"Prompt: {prompt}...")
        print(f"Prompt length: {len(prompt)} chars")
        print(f"Agent type: {agent_type}")
        # response = llm_inference(system_prompt="You are a helpful assistant analyzing hardware design log files.", user_prompt=prompt)
        # pdb.set_trace()
        token_usage = None
        if agent_type == "claude_code":
            response = claude_code_agent_inference(question.id, prompt, generator.log_file_path, model)
        elif agent_type == "rag":
            response, token_usage = rag_agent_inference(question.id, prompt, rag_agent)
        else:  # default to cursor agent
            response = cursor_agent_inference(question.id, prompt, generator.log_file_path, model)
        print(f"Response: {response}")
        # pdb.set_trace()
        # print question keys
        # print(f"Question keys: {question.__dict__.keys()}")
        question_responses.append({
            "json_metric_file": generator.metrics_file_path,
            "log_file_path": generator.log_file_path,
            "question_id": question.id,
            "question_idx": qid,
            "question_type": question.question_type,
            "stage": question.stage,
            "difficulty": question.difficulty,
            "occurrence": question.occurrence,
            "question_text": question.question,
            "golden_answer": question.answer,
            "answer_format": question.answer_placeholder,
            "response": response,
            "token_usage": token_usage,
        })
            
    return question_responses


def run_llm_inference_on_prompts(prepared_questions: List[Dict[str, Any]], output_dir: str = "./") -> List[Dict[str, Any]]:
    """
    Run LLM inference on prepared prompts.
    
    Args:
        prepared_questions: List of prepared question data from test_all_questions
        output_dir: Directory to save results
    
    Returns:
        List of results with LLM responses
    """
    print(f"\n{'='*80}")
    print(f"Running LLM Inference on {len(prepared_questions)} prompts")
    print(f"{'='*80}")
    
    results = []
    
    for i, q_data in enumerate(prepared_questions):
        print(f"\n[{i+1}/{len(prepared_questions)}] Processing Question: {q_data['question_id']}")
        print(f"Type: {q_data['question_type']} | Stage: {q_data['stage']} | Difficulty: {q_data['difficulty']}")
        
        try:
            # Run LLM inference
            # response = llm_inference(
            #    system_prompt="You are a helpful assistant analyzing hardware design log files.",
            #    user_prompt=q_data['prompt']
            #)
            response = "Dummy inference response"
            
            result = {
                **q_data,  # Include all original data
                "llm_response": response,
                "status": "success"
            }
            
            print(f"✅ Response received ({len(str(response))} chars)")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            result = {
                **q_data,
                "llm_response": None,
                "status": "error",
                "error_message": str(e)
            }
        
        results.append(result)
    
    # Save results to JSON file
    design_name = prepared_questions[0].get('question_id', 'unknown').split('_')[0] if prepared_questions else 'unknown'
    model = prepared_questions[0].get('model', 'unknown') if prepared_questions else 'unknown'
    output_file = os.path.join(output_dir, f"{design_name}_{model}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"✅ Completed {len(results)} inferences")
    print(f"💾 Results saved to: {output_file}")
    print(f"{'='*80}")
    
    return results
    

def run_multiple_tests(qa_file: str, model: str = "gpt-4o", qidxs: List[int] = None, output_dir: str = "./", num_runs: int = 1, use_json_hints: bool = False, agent_type: str = "cursor") -> None:
    """
    Run test_all_questions multiple times with different run indices.
    
    Args:
        qa_file: Path to the Q&A file
        model: Model name to use for testing
        qidxs: List of question indices to test (None = all questions)
        output_dir: Directory to save output files
        num_runs: Number of times to run the tests
        use_json_hints: If True, use create_prompt_with_output_json_key_hints; otherwise use create_prompt_with_background
        agent_type: Type of agent to use ("cursor" or "claude_code", default: "cursor")
    """
    print(f"Starting {num_runs} test runs...")
    for run_idx in range(num_runs):
        print(f"\n{'='*80}")
        print(f"Run {run_idx + 1}/{num_runs}")
        print(f"{'='*80}")
        prepared = test_all_questions(qa_file, model, qidxs, output_dir, use_json_hints, agent_type)
        
        # Optionally run inference
        # run_llm_inference_on_prompts(prepared, output_dir)
    
    print(f"\nCompleted all {num_runs} runs!")


def list_available_qa_files(qa_output_dir: str = "./qa_output") -> List[str]:
    """
    List all available Q&A JSON files in the qa_output directory.
    
    Args:
        qa_output_dir: Path to the qa_output directory
        
    Returns:
        List of Q&A JSON file paths
    """
    qa_dir = Path(qa_output_dir)
    if not qa_dir.exists():
        print(f"Directory not found: {qa_output_dir}")
        return []
    
    qa_files = list(qa_dir.glob("*.json"))
    
    if not qa_files:
        print(f"No JSON files found in {qa_output_dir}")
        return []
    
    print(f"\nAvailable Q&A files in {qa_output_dir}:")
    print("=" * 80)
    for i, file_path in enumerate(qa_files, 1):
        file_size = file_path.stat().st_size / 1024  # Size in KB
        print(f"{i}. {file_path.name} ({file_size:.2f} KB)")
        
        # Try to load and show quick stats
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    print(f"   └─ Format: List, {len(data)} items")
                elif isinstance(data, dict):
                    if "qa_pairs" in data:
                        print(f"   └─ Format: Multi-stage, {len(data['qa_pairs'])} Q&A pairs")
                        print(f"      Design: {data.get('design_name', 'N/A')}")
                    else:
                        print(f"   └─ Format: Dictionary")
        except Exception as e:
            print(f"   └─ Error reading file: {e}")
    
    print("=" * 80)
    return [str(f) for f in qa_files]

   
def demo_usage():
    """Demonstrate how to use the QA prompt generator."""
    
    # Example 1: Load and explore a masked Q&A file
    print("=" * 80)
    print("Example 1: Load and explore Q&A file (LLM format)")
    print("=" * 80)
    
    qa_file = "./qa_output/uart-blocks_20260122_105329_questions_and_answers_llm.json"
    
    if not Path(qa_file).exists():
        print(f"File not found: {qa_file}")
        print("Please update the path to your Q&A file.")
        return
    
    generator = QAPromptGenerator(qa_file)
    
    # Get summary
    summary = generator.get_summary()
    print(f"\nDesign: {summary['design_name']}")
    print(f"Total Questions: {summary['total_questions']}")
    print(f"By Stage: {summary['by_stage']}")
    print(f"By Difficulty: {summary['by_difficulty']}")
    print(f"By Type: {summary.get('by_type', {})}")
    print(f"Simple Answers: {summary['simple_answers']}")
    print(f"Dict Answers: {summary['dict_answers']}")
    print(f"List Answers: {summary.get('list_answers', 0)}")
    
    # Example 2: Create a prompt for a single question
    print("\n" + "=" * 80)
    print("Example 2: Create prompt for a single question")
    print("=" * 80)
    
    question = generator.questions[0]  # Get first question
    prompt = generator.create_simple_prompt(question)
    print(f"\nQuestion ID: {question.id}")
    print(f"Difficulty: {question.difficulty}")
    print(f"Stage: {question.stage}")
    print(f"Prompt:\n{prompt}")
    # pdb.set_trace()
    # response = llm_inference(system_prompt="You are a helpful assistant.", user_prompt=prompt)
    # print(f"Response: {response}")
    
    # Example 3: Load and explore multi-stage Q&A file
    print("\n" + "=" * 80)
    print("Example 3: Load and explore multi-stage Q&A file")
    print("=" * 80)
    
    qa_file_multistage = "./qa_output/uart-blocks_20260122_105349_qa_multistages.json"
    
    if Path(qa_file_multistage).exists():
        generator_multi = QAPromptGenerator(qa_file_multistage)
        summary_multi = generator_multi.get_summary()
        print(f"\nDesign: {summary_multi['design_name']}")
        print(f"Total Questions: {summary_multi['total_questions']}")
        print(f"By Stage: {summary_multi['by_stage']}")
        print(f"By Difficulty: {summary_multi['by_difficulty']}")
        
        # Show a sample multi-stage question
        if generator_multi.questions:
            sample_q = generator_multi.questions[0]
            print(f"\nSample Multi-Stage Question:")
            print(f"ID: {sample_q.id}")
            print(f"Difficulty: {sample_q.difficulty}")
            print(f"Stage: {sample_q.stage}")
            print(f"Question: {sample_q.question[:100]}...")
            prompt = generator_multi.create_simple_prompt(sample_q)
            print(f"Prompt: {prompt}")
            pdb.set_trace()
            # response = llm_inference(system_prompt="You are a helpful assistant.", user_prompt=prompt)
            # print(f"Response: {response}")

def write_rag_token_usage_to_json(question_responses: List[Dict[str, Any]], output_file_path: str) -> None:
    """
    Write RAG token usage summary to a JSON file (only when any response has token_usage).
    """
    per_question = []
    total_prompt = 0
    total_completion = 0
    total_tokens = 0
    for r in question_responses:
        u = r.get("token_usage")
        if not u:
            per_question.append({
                "question_id": r.get("question_id"),
                "question_idx": r.get("question_idx"),
                "token_usage": None,
            })
            continue
        total_prompt += u.get("prompt_tokens", 0)
        total_completion += u.get("completion_tokens", 0)
        total_tokens += u.get("total_tokens", 0)
        per_question.append({
            "question_id": r.get("question_id"),
            "question_idx": r.get("question_idx"),
            "token_usage": u,
        })
    if total_tokens == 0 and not any(r.get("token_usage") for r in question_responses):
        return
    payload = {
        "summary": {
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_tokens": total_tokens,
            "num_questions": len(question_responses),
        },
        "per_question": per_question,
    }
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"RAG token usage written to: {output_file_path}")


def write_question_responses_to_json(question_responses: List[Dict[str, Any]], output_dir: str = "./"):
    """
    Write question responses to a JSON file.
    When any response has token_usage (e.g. RAG agent), also write a matching token-usage JSON file.
    """
    original_filename = os.path.basename(question_responses[0].get("log_file_path") or "unknown")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"question_responses_{original_filename}_{timestamp}.json")
    if os.path.exists(output_file):
        print(f"File already exists: {output_file}")
        output_file = os.path.join(output_dir, f"question_responses_{original_filename}_{timestamp}_v1.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(question_responses, f, indent=2, ensure_ascii=False)
    print(f"Question responses written to: {output_file}")
    # Write RAG token usage file with matching name (test mode): same stem as output_file, prefixed with rag_token_usage_
    out_basename = os.path.basename(output_file)
    token_usage_file = os.path.join(output_dir, "rag_token_usage_" + out_basename.replace("question_responses_", "", 1))
    write_rag_token_usage_to_json(question_responses, token_usage_file)
    return


def run_pass_at_k_tests(qa_file: str, model: str = "gpt-4o", qidxs: List[int] = None, 
                        output_dir: str = "./", num_passes: int = 10, use_json_hints: bool = False, agent_type: str = "cursor") -> str:
    """
    Run tests N times (pass1 to passN) for calculating pass@k metrics.
    
    This function runs the same set of questions multiple times to generate
    multiple response samples per question, which is necessary for computing
    pass@k metrics (the probability that at least one of k responses is correct).
    
    Args:
        qa_file: Path to the Q&A file
        model: Model name to use for testing
        qidxs: List of question indices to test (None = all questions)
        output_dir: Directory to save output files
        num_passes: Number of passes to run (default: 10 for pass@1 through pass@10)
        use_json_hints: If True, use create_prompt_with_output_json_key_hints
    
    Returns:
        Path to the output JSON file
        
    Output Format:
        {
            "metadata": {
                "source_qa_file": "path/to/qa_file.json",
                "model": "gpt-4o",
                "num_passes": 10,
                "num_questions": 5,
                "timestamp": "20260131_120000"
            },
            "questions": [
                {
                    "question_id": "Q001",
                    "question_idx": 0,
                    "question_text": "...",
                    "question_type": "flow_execution",
                    "stage": "synthesis",
                    "difficulty": "easy",
                    "occurrence": 1,
                    "golden_answer": "...",
                    "answer_format": "[dict format]",
                    "json_metric_file": "path/to/metrics.json",
                    "log_file_path": "path/to/log.txt",
                    "passes": {
                        "pass1": {"response": "...", "timestamp": "..."},
                        "pass2": {"response": "...", "timestamp": "..."},
                        ...
                        "pass10": {"response": "...", "timestamp": "..."}
                    }
                },
                ...
            ]
        }
    """
    print(f"{'='*80}")
    print(f"Starting Pass@K Test Run")
    print(f"{'='*80}")
    print(f"QA File: {qa_file}")
    print(f"Model: {model}")
    print(f"Number of passes: {num_passes}")
    print(f"Output directory: {output_dir}")
    print(f"Use JSON hints: {use_json_hints}")
    print(f"{'='*80}\n")
    
    # Ensure output directory exists (needed when running Python directly without the shell script)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load QA file once to get metadata
    generator = QAPromptGenerator(qa_file)
    
    # Determine which questions to test
    questions_to_test = []
    if qidxs is not None:
        questions_to_test = [generator.questions[qid] for qid in qidxs if qid < len(generator.questions)]
        test_qidxs = qidxs
    else:
        questions_to_test = generator.questions
        test_qidxs = list(range(len(generator.questions)))
    
    # Extract design name with technology node from metrics file path
    design_name_with_tech = None
    if generator.metrics_file_path:
        filename = os.path.basename(generator.metrics_file_path)
        # Remove _all_stages_metrics.json suffix to get design_name_tech_node
        design_name_with_tech = filename.replace("_all_stages_metrics.json", "")
    else:
        design_name_with_tech = generator.design_name
    
    # Initialize results structure
    results = {
        "metadata": {
            "source_qa_file": qa_file,
            "design_name": design_name_with_tech,  # Add design name with technology node
            "model": model,
            "num_passes": num_passes,
            "num_questions": len(questions_to_test),
            "timestamp": datetime.now().strftime('%Y%m%d_%H%M%S'),
            "use_json_hints": use_json_hints,
            "agent_type": agent_type
        },
        "questions": []
    }
    
    # Initialize question results with base information
    for qid, question in zip(test_qidxs, questions_to_test):
        question_result = {
            "question_id": question.id,
            "question_idx": qid,
            "question_text": question.question,
            "question_type": question.question_type,
            "stage": question.stage,
            "difficulty": question.difficulty,
            "occurrence": question.occurrence,
            "golden_answer": question.answer,
            "answer_format": question.answer_placeholder,
            "design_name": design_name_with_tech,  # Add design name with technology node
            "json_metric_file": generator.metrics_file_path,
            "log_file_path": generator.log_file_path,
            "passes": {}
        }
        results["questions"].append(question_result)
    
    # Run multiple passes
    for pass_num in range(1, num_passes + 1):
        print(f"\n{'='*80}")
        print(f"Running Pass {pass_num}/{num_passes}")
        print(f"{'='*80}")
        
        # Run test for this pass
        pass_responses = test_all_questions(qa_file, model, test_qidxs, output_dir, use_json_hints, agent_type)
        
        # Store responses for this pass (include token_usage when agent_type is rag)
        pass_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        for i, question_result in enumerate(results["questions"]):
            if i < len(pass_responses):
                pass_key = f"pass{pass_num}"
                entry = {
                    "response": pass_responses[i]["response"],
                    "timestamp": pass_timestamp,
                }
                if pass_responses[i].get("token_usage") is not None:
                    entry["token_usage"] = pass_responses[i]["token_usage"]
                results["questions"][i]["passes"][pass_key] = entry
        
        # Write intermediate results file for this pass in the same format as test_all_questions()
        design_name_for_file = results["metadata"].get("design_name", Path(qa_file).stem)
        pass_output_file = os.path.join(
            output_dir,
            f"pass_at_k_{design_name_for_file}_{model}_pass{pass_num}of{num_passes}_{pass_timestamp}.json"
        )
        with open(pass_output_file, 'w', encoding='utf-8') as f:
            json.dump(pass_responses, f, indent=2, ensure_ascii=False)
        print(f"✅ Completed Pass {pass_num}/{num_passes}")
        print(f"💾 Intermediate results saved to: {pass_output_file}")
        # Write RAG token usage for this pass (pass-at-k style filename)
        if agent_type == "rag":
            pass_token_file = os.path.join(
                output_dir,
                f"pass_at_k_rag_token_usage_{design_name_for_file}_{model}_pass{pass_num}of{num_passes}_{pass_timestamp}.json"
            )
            write_rag_token_usage_to_json(pass_responses, pass_token_file)
    
    # Write final consolidated results file
    design_name_for_file = results["metadata"].get("design_name", Path(qa_file).stem)
    final_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(
        output_dir,
        f"pass_at_k_{design_name_for_file}_{model}_{num_passes}passes_FINAL_{final_timestamp}.json"
    )
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    # Write RAG token usage summary across all passes (pass-at-k style filename)
    if agent_type == "rag":
        all_pass_token_usage = []
        total_prompt, total_completion, total_tokens = 0, 0, 0
        for q in results["questions"]:
            row = {"question_id": q["question_id"], "question_idx": q["question_idx"], "passes": {}}
            for pass_key, pass_data in q.get("passes", {}).items():
                u = pass_data.get("token_usage")
                if u:
                    row["passes"][pass_key] = u
                    total_prompt += u.get("prompt_tokens", 0)
                    total_completion += u.get("completion_tokens", 0)
                    total_tokens += u.get("total_tokens", 0)
            all_pass_token_usage.append(row)
        final_token_file = os.path.join(
            output_dir,
            f"pass_at_k_rag_token_usage_{design_name_for_file}_{model}_{num_passes}passes_FINAL_{final_timestamp}.json"
        )
        with open(final_token_file, "w", encoding="utf-8") as f:
            json.dump({
                "summary": {
                    "total_prompt_tokens": total_prompt,
                    "total_completion_tokens": total_completion,
                    "total_tokens": total_tokens,
                    "num_questions": len(results["questions"]),
                    "num_passes": num_passes,
                },
                "per_question": all_pass_token_usage,
            }, f, indent=2, ensure_ascii=False)
        print(f"💾 RAG token usage (all passes) saved to: {final_token_file}")
    print(f"\n{'='*80}")
    print(f"✅ Pass@K Test Run Complete!")
    print(f"{'='*80}")
    print(f"Total questions tested: {len(results['questions'])}")
    print(f"Number of passes per question: {num_passes}")
    print(f"Total inferences: {len(results['questions']) * num_passes}")
    print(f"💾 Results saved to: {output_file}")
    print(f"{'='*80}\n")
    return output_file


def calculate_pass_at_k(results_file: str, k: int = 1) -> Dict[str, Any]:
    """
    Calculate pass@k metrics from a pass@k results file.
    
    This function evaluates the pass@k metric: the probability that at least
    one of k generated responses is correct. This is a common metric for 
    evaluating code generation models.
    
    Args:
        results_file: Path to the pass@k results JSON file
        k: Number of samples to consider (default: 1 for pass@1)
    
    Returns:
        Dictionary containing pass@k statistics
        
    Note:
        This function currently returns a placeholder. You need to implement
        the actual validation logic based on your answer validation function.
        The validation should compare each response against the golden_answer
        for each question.
    """
    print(f"Calculating pass@{k} metrics from: {results_file}")
    
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_questions = data["metadata"]["num_questions"]
    num_passes = data["metadata"]["num_passes"]
    
    if k > num_passes:
        print(f"⚠️  Warning: k={k} is greater than num_passes={num_passes}")
        print(f"    Using k={num_passes} instead")
        k = num_passes
    
    print(f"\n{'='*80}")
    print(f"Pass@{k} Analysis")
    print(f"{'='*80}")
    print(f"Total questions: {total_questions}")
    print(f"Number of passes available: {num_passes}")
    print(f"Evaluating pass@{k}")
    print(f"{'='*80}\n")
    
    # TODO: Implement actual validation logic here
    # For each question:
    #   1. Get the golden_answer
    #   2. Get the first k responses from passes
    #   3. Validate each response against golden_answer
    #   4. Mark as correct if at least one of k responses is correct
    
    stats = {
        "total_questions": total_questions,
        "num_passes": num_passes,
        "k": k,
        "results_file": results_file,
        "note": "Validation logic not yet implemented. Integrate with your answer validation function."
    }
    
    print("⚠️  Note: Actual validation logic needs to be implemented.")
    print("   Integrate with your existing answer validation function to compare responses with golden_answer.")
    
    return stats

if __name__ == "__main__":
    import sys
    
    # add argparse to parse the arguments
    parser = argparse.ArgumentParser(description='Test LLM QA')
    parser.add_argument('command', type=str, help='Command to execute')
    parser.add_argument('args', nargs='*', help='Arguments for the command')
    # add arguments for the command: prepare, infer, test, demo, list
    parser.add_argument('--prepare', type=str, help='Prepare prompts for LLM inference')
    parser.add_argument('--infer', type=str, help='Run LLM inference on prepared prompts')
    parser.add_argument('--test', type=str, help='Test all questions')
    parser.add_argument('--demo', type=str, help='Run demo')
    parser.add_argument('--list', type=str, help='List available Q&A files')
    # qa file, output dir, model, qidxs
    parser.add_argument('--qa_file', type=str, help='QA file')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--model', type=str, help='Model')
    parser.add_argument('--qidxs', type=str, help='Question indices')
    parser.add_argument('--use_json_hints', action='store_true', help='Use create_prompt_with_output_json_key_hints instead of create_prompt_with_background')
    # Pass@k arguments
    parser.add_argument('--num_passes', type=int, default=10, help='Number of passes for pass@k testing (default: 10)')
    parser.add_argument('--k', type=int, default=1, help='k value for pass@k calculation (default: 1)')
    parser.add_argument('--results_file', type=str, help='Results file for pass@k calculation')
    parser.add_argument('--agent_type', type=str, choices=['cursor', 'claude_code', 'rag'], default='cursor', help='Type of agent to use: "cursor" or "claude_code" or "rag" (default: cursor)')
    args = parser.parse_args()
    
    # Handle commands based on parsed arguments
    if args.command == "demo":
        # Run demo
        demo_usage()
        
    elif args.command == "list":
        # List available Q&A files
        qa_dir = args.qa_file if args.qa_file else "./qa_output"
        list_available_qa_files(qa_dir)
        
    elif args.command == "pass_at_k":
        # Run pass@k tests
        # Usage: python test_llm_qa.py pass_at_k --qa_file <file> --model <model> [--num_passes <N>] [--qidxs <indices>] [--output_dir <dir>] [--use_json_hints] [--agent_type <cursor|claude_code>]
        if not args.qa_file or not args.model:
            print("Usage: python test_llm_qa.py pass_at_k --qa_file <file> --model <model> [--num_passes <N>] [--qidxs <indices>] [--output_dir <dir>] [--use_json_hints] [--agent_type <cursor|claude_code>]")
            print("Example: python test_llm_qa.py pass_at_k --qa_file ./qa_output/uart-blocks_20260122_105329_questions_and_answers_llm.json --model gpt-4o")
            print("Example: python test_llm_qa.py pass_at_k --qa_file ./qa_output/uart-blocks_20260122_105329_questions_and_answers_llm.json --model gpt-4o --num_passes 5")
            print("Example: python test_llm_qa.py pass_at_k --qa_file ./qa_output/uart-blocks_20260122_105329_questions_and_answers_llm.json --model gpt-4o --num_passes 10 --qidxs 0,1,2")
            print("Example: python test_llm_qa.py pass_at_k --qa_file ./qa_output/uart-blocks_20260122_105329_questions_and_answers_llm.json --model gpt-4o --agent_type claude_code")
            sys.exit(1)
        
        qa_file = args.qa_file
        model = args.model
        qidxs = None if not args.qidxs else [int(x) for x in args.qidxs.split(',')]
        output_dir = args.output_dir if args.output_dir else "./"
        num_passes = args.num_passes
        use_json_hints = args.use_json_hints
        agent_type = args.agent_type
        
        output_file = run_pass_at_k_tests(qa_file, model, qidxs, output_dir, num_passes, use_json_hints, agent_type)
        print(f"\n✅ Pass@K test completed. Results saved to: {output_file}")
        
    elif args.command == "calc_pass_at_k":
        # Calculate pass@k metrics from results file
        # Usage: python test_llm_qa.py calc_pass_at_k --results_file <file> [--k <value>]
        if not args.results_file:
            print("Usage: python test_llm_qa.py calc_pass_at_k --results_file <file> [--k <value>]")
            print("Example: python test_llm_qa.py calc_pass_at_k --results_file ./pass_at_k_uart-blocks_gpt-4o_10passes_20260131_120000.json")
            print("Example: python test_llm_qa.py calc_pass_at_k --results_file ./pass_at_k_uart-blocks_gpt-4o_10passes_20260131_120000.json --k 5")
            sys.exit(1)
        
        results_file = args.results_file
        k = args.k
        
        stats = calculate_pass_at_k(results_file, k)
        print(f"\nPass@{k} Statistics:")
        print(json.dumps(stats, indent=2))
        
    elif args.command == "prepare":
        # Prepare prompts for LLM inference
        # Usage: python test_llm_qa.py prepare --qa_file <file> --model <model> [--qidxs <indices>] [--output_dir <dir>] [--use_json_hints] [--agent_type <cursor|claude_code>]
        if not args.qa_file or not args.model:
            print("Usage: python test_llm_qa.py prepare --qa_file <file> --model <model> [--qidxs <indices>] [--output_dir <dir>] [--use_json_hints] [--agent_type <cursor|claude_code>]")
            print("Example: python test_llm_qa.py prepare --qa_file ./qa_output/uart-blocks_20260122_105329_questions_and_answers_llm.json --model gpt-4o")
            print("Example: python test_llm_qa.py prepare --qa_file ./qa_output/uart-blocks_20260122_105329_questions_and_answers_llm.json --model gpt-4o --qidxs 0,1,2 --output_dir ./results")
            print("Example: python test_llm_qa.py prepare --qa_file ./qa_output/uart-blocks_20260122_105329_questions_and_answers_llm.json --model gpt-4o --use_json_hints")
            print("Example: python test_llm_qa.py prepare --qa_file ./qa_output/uart-blocks_20260122_105329_questions_and_answers_llm.json --model gpt-4o --agent_type claude_code")
            sys.exit(1)
        
        qa_file = args.qa_file
        model = args.model
        qidxs = None if not args.qidxs else [int(x) for x in args.qidxs.split(',')]
        output_dir = args.output_dir if args.output_dir else "./"
        use_json_hints = args.use_json_hints
        agent_type = args.agent_type
        
        question_responses = test_all_questions(qa_file, model, qidxs, output_dir, use_json_hints, agent_type)
        write_question_responses_to_json(question_responses, output_dir)
        print(f"\n✅ Completed {len(question_responses)} questions with LLM inference")
        
    elif args.command == "infer":
        # Run LLM inference on prepared prompts file (deprecated - now done in prepare)
        print("⚠️  The 'infer' command is deprecated. Use 'prepare' command instead, which now runs inference automatically.")
        print("Usage: python test_llm_qa.py prepare --qa_file <file> --model <model>")
        
    elif args.command == "test":
        # Run test with provided arguments
        # Usage: python test_llm_qa.py test --qa_file <file> --model <model> [--qidxs <indices>] [--output_dir <dir>] [--use_json_hints] [--agent_type <cursor|claude_code>]
        if not args.qa_file or not args.model:
            print("Usage: python test_llm_qa.py test --qa_file <file> --model <model> [--qidxs <indices>] [--output_dir <dir>] [--use_json_hints] [--agent_type <cursor|claude_code>]")
            print("Example: python test_llm_qa.py test --qa_file ./qa_output/uart-blocks_20260122_105329_questions_and_answers_llm.json --model gpt-4o")
            print("Example: python test_llm_qa.py test --qa_file ./qa_output/uart-blocks_20260122_105329_questions_and_answers_llm.json --model gpt-4o --use_json_hints")
            print("Example: python test_llm_qa.py test --qa_file ./qa_output/uart-blocks_20260122_105329_questions_and_answers_llm.json --model gpt-4o --agent_type claude_code")
            print("\nNote: Log file is now automatically loaded from the QA file metadata.")
            sys.exit(1)
        
        qa_file = args.qa_file
        model = args.model
        qidxs = None if not args.qidxs else [int(x) for x in args.qidxs.split(',')]
        output_dir = args.output_dir if args.output_dir else "./"
        use_json_hints = args.use_json_hints
        agent_type = args.agent_type
        
        question_responses = test_all_questions(qa_file, model, qidxs, output_dir, use_json_hints, agent_type)
        write_question_responses_to_json(question_responses, output_dir)
        
    else:
        print("Usage:")
        print("  python test_llm_qa.py demo                                        # Run demo")
        print("  python test_llm_qa.py list [--qa_file <dir>]                     # List available Q&A files")
        print("  python test_llm_qa.py prepare --qa_file <file> --model <model>   # Prepare and run LLM inference")
        print("  python test_llm_qa.py test --qa_file <file> --model <model>      # Run tests (same as prepare)")
        print("  python test_llm_qa.py pass_at_k --qa_file <file> --model <model> # Run pass@k tests")
        print("  python test_llm_qa.py calc_pass_at_k --results_file <file>       # Calculate pass@k metrics")
        print("\nFor prepare/test mode:")
        print("  Required arguments:")
        print("    --qa_file <path>      Path to Q&A JSON file")
        print("    --model <name>        Model name (e.g., gpt-4o)")
        print("  Optional arguments:")
        print("    --qidxs <indices>     Comma-separated question indices (e.g., 0,1,2)")
        print("    --output_dir <path>   Output directory (default: ./)")
        print("    --use_json_hints      Use JSON key hints in prompts (creates prompts with output structure hints)")
        print("    --agent_type <type>   Agent type: 'cursor' or 'claude_code' (default: cursor)")
        print("\nFor pass@k mode:")
        print("  Required arguments:")
        print("    --qa_file <path>      Path to Q&A JSON file")
        print("    --model <name>        Model name (e.g., gpt-4o)")
        print("  Optional arguments:")
        print("    --num_passes <N>      Number of passes to run (default: 10)")
        print("    --qidxs <indices>     Comma-separated question indices (e.g., 0,1,2)")
        print("    --output_dir <path>   Output directory (default: ./)")
        print("    --use_json_hints      Use JSON key hints in prompts")
        print("    --agent_type <type>   Agent type: 'cursor' or 'claude_code' (default: cursor)")
        print("\nFor calc_pass_at_k mode:")
        print("  Required arguments:")
        print("    --results_file <path> Path to pass@k results JSON file")
        print("  Optional arguments:")
        print("    --k <value>           k value for pass@k calculation (default: 1)")
        print("\nExamples:")
        print("  python test_llm_qa.py demo")
        print("  python test_llm_qa.py list --qa_file ./qa_output")
        print("  python test_llm_qa.py prepare --qa_file ./qa_output/uart-blocks_20260122_105329_questions_and_answers_llm.json --model gpt-4o")
        print("  python test_llm_qa.py prepare --qa_file ./qa_output/uart-blocks_20260122_105329_questions_and_answers_llm.json --model gpt-4o --qidxs 0,1,2 --output_dir ./results")
        print("  python test_llm_qa.py prepare --qa_file ./qa_output/uart-blocks_20260122_105329_questions_and_answers_llm.json --model gpt-4o --use_json_hints")
        print("  python test_llm_qa.py prepare --qa_file ./qa_output/uart-blocks_20260122_105329_questions_and_answers_llm.json --model gpt-4o --agent_type claude_code")
        print("  python test_llm_qa.py test --qa_file ./qa_output/uart-blocks_20260122_105329_questions_and_answers_llm.json --model gpt-4o")
        print("  python test_llm_qa.py test --qa_file ./qa_output/uart-blocks_20260122_105329_questions_and_answers_llm.json --model gpt-4o --agent_type claude_code")
        print("  python test_llm_qa.py pass_at_k --qa_file ./qa_output/uart-blocks_20260122_105329_questions_and_answers_llm.json --model gpt-4o")
        print("  python test_llm_qa.py pass_at_k --qa_file ./qa_output/uart-blocks_20260122_105329_questions_and_answers_llm.json --model gpt-4o --num_passes 5")
        print("  python test_llm_qa.py pass_at_k --qa_file ./qa_output/uart-blocks_20260122_105329_questions_and_answers_llm.json --model gpt-4o --agent_type claude_code")
        print("  python test_llm_qa.py calc_pass_at_k --results_file ./pass_at_k_uart-blocks_gpt-4o_10passes_20260131_120000.json --k 5")



