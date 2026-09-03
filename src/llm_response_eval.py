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
LLM Judge Evaluation for EDA Log QA Responses
Evaluates the quality and correctness of LLM responses to EDA log questions.
"""

import json
import os
import re
import pdb
from typing import Dict, List, Any, Optional, Tuple
from llm import llm_inference, call_dar_gateway
from config_FLAGS import FLAGS


def extract_json_from_response(response: str) -> Optional[Any]:
    """
    Extract JSON content from response text.
    Looks for JSON between ```json and ``` markers, or tries to parse the whole response.
    
    Args:
        response: Raw response text
        
    Returns:
        Parsed JSON object or None if extraction fails
    """
    # Try to find JSON in markdown code blocks
    json_pattern = r'```json\s*(.*?)\s*```'
    matches = re.findall(json_pattern, response, re.DOTALL | re.IGNORECASE)
    
    if matches:
        # Try each match
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue
    
    # Try to find any JSON in triple backticks
    code_block_pattern = r'```\s*(.*?)\s*```'
    matches = re.findall(code_block_pattern, response, re.DOTALL)
    
    if matches:
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue
    
    # Try to parse the entire response as JSON
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass
    
    return None


def extract_answer_from_response(response: str) -> Optional[str]:
    """
    Extract the answer value from response text.
    Looks for content between ** and ** markers for list/text format answers.
    
    Args:
        response: Raw response text
        
    Returns:
        Extracted answer string or None if extraction fails
    """
    # Try to find answer between ** markers
    answer_pattern = r'\*\*(.*?)\*\*'
    matches = re.findall(answer_pattern, response, re.DOTALL)
    
    if matches:
        # Return the last match (usually the main answer)
        return matches[-1].strip()
    
    return None


def compare_json_answers(predicted: Any, golden: Any, flexible_keys: bool = True) -> Tuple[float, Dict[str, Any]]:
    """
    Compare predicted JSON answer with golden reference.
    Returns a score between 0 and 1.
    
    Args:
        predicted: Predicted answer (parsed JSON)
        golden: Golden reference answer (parsed JSON)
        flexible_keys: If True, use flexible key matching (case-insensitive, ignore separators)
        
    Returns:
        Tuple of (score, details_dict)
    """
    details = {
        "exact_match": False,
        # "type_match": False,
        "partial_matches": {},
        "differences": []
    }
    
    # Type check
    # if type(predicted) != type(golden):
    #     details["differences"].append(f"Type mismatch: {type(predicted).__name__} vs {type(golden).__name__}")
    #     return 0.0, details
    
    # details["type_match"] = True
    
    # Exact match
    if predicted == golden:
        details["exact_match"] = True
        return 1.0, details
    
    # For dictionaries, calculate partial match
    if isinstance(predicted, dict) and isinstance(golden, dict):
        return compare_dict_answers(predicted, golden, details, flexible_keys=flexible_keys)
    
    # For lists, calculate partial match
    elif isinstance(predicted, list) and isinstance(golden, list):
        return compare_list_answers(predicted, golden, details)
    
    # For primitives, binary match
    elif predicted == golden:
        details["exact_match"] = True
        return 1.0, details
    else:
        details["differences"].append(f"Value mismatch: {predicted} vs {golden}")
        return 0.0, details


def normalize_key(key: str) -> str:
    """
    Normalize dictionary key for flexible matching.
    Converts to lowercase and removes separators.
    
    Examples:
        "design_name" -> "designname"
        "designName" -> "designname"
        "design-name" -> "designname"
        "Design Name" -> "designname"
    """
    return key.lower().replace('_', '').replace('-', '').replace(' ', '')


def find_matching_key(target_key: str, available_keys: List[str], normalize: bool = True) -> Optional[str]:
    """
    Find a matching key from available keys, with flexible matching.
    
    Args:
        target_key: The key to find
        available_keys: List of available keys to search
        normalize: If True, use normalized key matching
        
    Returns:
        Matching key from available_keys, or None if not found
    """
    # Exact match first
    if target_key in available_keys:
        return target_key
    
    if not normalize:
        return None
    
    # Normalized match
    normalized_target = normalize_key(target_key)
    for key in available_keys:
        if normalize_key(key) == normalized_target:
            return key
    
    return None


def compare_dict_answers(predicted: Dict, golden: Dict, details: Dict, flexible_keys: bool = True) -> Tuple[float, Dict]:
    """
    Compare two dictionary answers with flexible key matching.
    
    Args:
        predicted: Predicted answer dictionary
        golden: Golden reference dictionary
        details: Details dictionary to populate
        flexible_keys: If True, use flexible key matching (case-insensitive, ignore separators)
    """
    golden_keys = list(golden.keys())
    predicted_keys = list(predicted.keys())
    
    # Track matched keys
    matched_golden_keys = set()
    key_mapping = {}  # golden_key -> predicted_key
    
    # Build key mapping with flexible matching
    for golden_key in golden_keys:
        matched_pred_key = find_matching_key(golden_key, predicted_keys, normalize=flexible_keys)
        if matched_pred_key:
            key_mapping[golden_key] = matched_pred_key
            matched_golden_keys.add(golden_key)
    
    # Key coverage
    missing_keys = set(golden_keys) - matched_golden_keys
    matched_pred_keys = set(key_mapping.values())
    extra_keys = set(predicted_keys) - matched_pred_keys
    
    if missing_keys:
        details["differences"].append(f"Missing keys: {list(missing_keys)}")
    if extra_keys:
        details["differences"].append(f"Extra keys (ignored): {list(extra_keys)}")
    
    # Track key name differences
    key_name_diffs = []
    for golden_key, pred_key in key_mapping.items():
        if golden_key != pred_key:
            key_name_diffs.append(f"'{golden_key}' matched with '{pred_key}'")
    
    if key_name_diffs:
        details["key_name_differences"] = key_name_diffs
    
    # If no common keys, score is 0
    if not key_mapping:
        return 0.0, details
    
    # Calculate value matches for matched keys
    matched_values = 0
    total_values = len(golden_keys)
    
    for golden_key, pred_key in key_mapping.items():
        pred_val = predicted[pred_key]
        gold_val = golden[golden_key]
        
        if isinstance(pred_val, (dict, list)) and isinstance(gold_val, (dict, list)):
            # Recursive comparison for nested structures
            sub_score, _ = compare_json_answers(pred_val, gold_val)
            details["partial_matches"][golden_key] = sub_score
            matched_values += sub_score
        elif pred_val == gold_val:
            details["partial_matches"][golden_key] = 1.0
            matched_values += 1
        # Handle numeric comparisons with tolerance
        elif isinstance(pred_val, (int, float)) and isinstance(gold_val, (int, float)):
            # Allow small numeric differences (e.g., 1.0 vs 1)
            if abs(pred_val - gold_val) < 1e-6:
                details["partial_matches"][golden_key] = 1.0
                matched_values += 1
            else:
                details["partial_matches"][golden_key] = 0.0
                details["differences"].append(f"Key '{golden_key}': {pred_val} vs {gold_val}")
        # Handle boolean vs int (True vs 1, False vs 0)
        elif isinstance(pred_val, bool) and isinstance(gold_val, int):
            if (pred_val and gold_val == 1) or (not pred_val and gold_val == 0):
                details["partial_matches"][golden_key] = 1.0
                matched_values += 1
            else:
                details["partial_matches"][golden_key] = 0.0
                details["differences"].append(f"Key '{golden_key}': {pred_val} vs {gold_val}")
        elif isinstance(pred_val, int) and isinstance(gold_val, bool):
            if (pred_val == 1 and gold_val) or (pred_val == 0 and not gold_val):
                details["partial_matches"][golden_key] = 1.0
                matched_values += 1
            else:
                details["partial_matches"][golden_key] = 0.0
                details["differences"].append(f"Key '{golden_key}': {pred_val} vs {gold_val}")
        else:
            details["partial_matches"][golden_key] = 0.0
            details["differences"].append(f"Key '{golden_key}': {pred_val} vs {gold_val}")
    
    # Score is ratio of matched values to total expected values
    score = matched_values / total_values
    return score, details


def compare_list_answers(predicted: List, golden: List, details: Dict) -> Tuple[float, Dict]:
    """Compare two list answers."""
    if len(predicted) != len(golden):
        details["differences"].append(f"Length mismatch: {len(predicted)} vs {len(golden)}")
    
    # For empty golden list, check if predicted is also empty
    if not golden:
        return 1.0 if not predicted else 0.0, details
    
    # Calculate element-wise similarity
    matched_elements = 0
    total_elements = len(golden)
    
    for i, gold_item in enumerate(golden):
        if i < len(predicted):
            pred_item = predicted[i]
            
            if isinstance(pred_item, (dict, list)) and isinstance(gold_item, (dict, list)):
                item_score, _ = compare_json_answers(pred_item, gold_item)
                matched_elements += item_score
            elif pred_item == gold_item:
                matched_elements += 1
            else:
                details["differences"].append(f"Index {i}: {pred_item} vs {gold_item}")
        else:
            details["differences"].append(f"Missing element at index {i}")
    
    score = matched_elements / total_elements if total_elements > 0 else 0.0
    return score, details


def create_judge_prompt(question_data: Dict[str, Any], golden_answer: Optional[Any] = None) -> Dict[str, str]:
    """
    Create judge evaluation prompt based on question data.
    
    Args:
        question_data: Dictionary containing question, response, and metadata
        golden_answer: Optional golden reference answer for comparison
        
    Returns:
        Dictionary with 'system_prompt' and 'user_prompt' keys
    """
    system_prompt = """You are an expert judge evaluating LLM responses to questions about EDA (Electronic Design Automation) log files. You need to be very strict and critical with the answer correctness and the design stage of the answer.

Your task is to assess the quality and correctness of the LLM's response to a given question about a design flow log file.

Evaluation Criteria:
1. **Correctness**: Is the answer factually correct based on the log file information? If there are any information missing or in the wrong design stage, the answer is incorrect (0.0). If the time stamp is missing, it doesn't impact the correctness.
2. **Clarity**: Is the answer clear, well-structured, and easy to understand? Make sure the answer is not too long or too short.
3. **Relevance**: Does the answer stay focused on the question without unnecessary information? 

Scoring Guide (0-1 scale):
- 1.0: Perfect - Fully correct, clear, and relevant
- 0.8-0.9: Excellent - Minor issues but substantially correct
- 0.6-0.7: Good - Partially correct with some gaps
- 0.4-0.5: Fair - Significant issues or missing information
- 0.2-0.3: Poor - Major errors or mostly incorrect
- 0.0-0.1: Very Poor - Completely wrong or no answer

Please provide:
1. An overall score (0.0-1.0): If the answer is incorrect (0.0), the overall score is 0.0.
2. Scores for each criterion (0.0-1.0)
3. Specific feedback on strengths and weaknesses

You need be harsh on the score if there is any information missing or in the wrong design stage. If the time stamp is missing, it doesn't impact the correctness.
"""

    golden_section = ""
    if golden_answer is not None:
        golden_section = f"""
{json.dumps(golden_answer, indent=2)}
"""

    # Extract and format response based on answer format
    raw_response = question_data.get('response', 'N/A')
    answer_format = question_data.get('answer_format', '').lower()
    formatted_response = raw_response
    
    # For dict/JSON type answers, try to extract JSON for cleaner presentation
    if 'dict' in answer_format or 'json' in answer_format:
        extracted_json = extract_json_from_response(raw_response)
        if extracted_json is not None:
            # Successfully extracted JSON, format it nicely
            formatted_response = json.dumps(extracted_json, indent=2)
        # Otherwise, keep the original response as-is

    user_prompt = f"""Please evaluate the following LLM response:

**Question Metadata:**
- Question ID: {question_data.get('question_id', 'N/A')}
- Question Type: {question_data.get('question_type', 'N/A')}
- Stage: {question_data.get('stage', 'N/A')}
- Difficulty: {question_data.get('difficulty', 'N/A')}
- Expected Format: {question_data.get('answer_format', 'N/A')}

### Question Content ###
{question_data.get('question_text', 'N/A')}
### End of Question Content ###

### LLM Response ###
{formatted_response}
### End of LLM Response ###

### Golden Reference Answer ###
{golden_section}
### End of Golden Reference Answer ###

Please provide your evaluation in the following JSON format between golden reference answer and LLM response:

```json
{{
  "overall_score": <0.0-1.0>,
  "criterion_scores": {{
    "correctness": <0.0-1.0>,
    "clarity": <0.0-1.0>,
    "relevance": <0.0-1.0>
  }},
  "strengths": [
    "List specific strengths of the response"
  ],
  "weaknesses": [
    "List specific weaknesses or errors"
  ],
  "feedback": "Detailed feedback explaining the evaluation"
}}
```"""

    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt
    }


def evaluate_single_response(question_data: Dict[str, Any], golden_answer: Optional[Any] = None,
                            temperature: float = 0.1, n_judge_runs: int = 1) -> Dict[str, Any]:
    """
    Evaluate a single question-response pair using LLM judge (default behavior).
    
    This is the main evaluation function that uses LLM judge for qualitative assessment.
    For automatic JSON comparison with golden references, use evaluate_single_response_with_json().
    
    Args:
        question_data: Dictionary containing question, response, and metadata
        golden_answer: Optional golden reference answer to include in judge prompt (for context)
        temperature: Temperature for LLM inference (default: 0.1 for consistency)
        n_judge_runs: Number of times to run LLM judge and average scores (default: 1)
        
    Returns:
        Dictionary containing evaluation results
    """
    print(f"\n{'='*80}")
    print(f"Evaluating Question: {question_data.get('question_id', 'N/A')}")
    if n_judge_runs > 1:
        print(f"LLM Judge Runs: {n_judge_runs} (averaging scores)")
    print(f"{'='*80}")
    
    evaluation = {
        'question_id': question_data.get('question_id', 'N/A'),
        'question_type': question_data.get('question_type', 'N/A'),
        'stage': question_data.get('stage', 'N/A'),
        'difficulty': question_data.get('difficulty', 'N/A'),
        'answer_format': question_data.get('answer_format', 'N/A')
    }
    
    # Use LLM judge evaluation
    print(f"🤖 Using LLM judge evaluation...")
    if n_judge_runs > 1:
        print(f"📊 Running {n_judge_runs} independent evaluations...")
    
    prompts = create_judge_prompt(question_data, golden_answer)
    try:
        # Run LLM judge n times and collect results
        all_judge_evals = []
        
        for run_idx in range(n_judge_runs):
            if n_judge_runs > 1:
                print(f"\n  Run {run_idx + 1}/{n_judge_runs}...")
            
            judge_response = call_dar_gateway(
                system_prompt=prompts['system_prompt'],
                user_prompt=prompts['user_prompt'],
                temperature=temperature
            )
            # judge_response = llm_inference(
            #    system_prompt=prompts['system_prompt'],
            #     user_prompt=prompts['user_prompt'],
            #    temperature=temperature
            #)
            # print(f"Judge response: {judge_response}")
            # Parse JSON response
            judge_eval = extract_json_from_response(judge_response)
            # print(f"Judge eval: {judge_eval}")
            # pdb.set_trace()
            if not judge_eval:
                # Try parsing the whole response
                try:
                    judge_eval = json.loads(judge_response)
                except json.JSONDecodeError as e:
                    print(f"❌ Error parsing judge response (run {run_idx + 1}): {e}")
                    print(f"Raw response: {judge_response[:500]}")
                    continue
            
            all_judge_evals.append(judge_eval)
            
            if n_judge_runs > 1 and 'overall_score' in judge_eval:
                print(f"    Score: {judge_eval['overall_score']:.3f}")
        
        # Check if we got any valid evaluations
        if not all_judge_evals:
            evaluation['error'] = "All judge runs failed to parse"
            return evaluation
        
        # If only one run or only one valid result, use it directly
        if len(all_judge_evals) == 1:
            evaluation.update(all_judge_evals[0])
            evaluation['evaluation_method'] = 'llm_judge'
        else:
            # Average the scores across all runs
            print(f"\n📊 Averaging scores across {len(all_judge_evals)} runs...")
            averaged_eval = average_judge_evaluations(all_judge_evals)
            evaluation.update(averaged_eval)
            evaluation['evaluation_method'] = 'llm_judge_averaged'
            evaluation['n_judge_runs'] = len(all_judge_evals)
            evaluation['individual_runs'] = all_judge_evals  # Keep individual results for analysis
        
        print(f"\n✅ Evaluation completed - Overall Score: {evaluation.get('overall_score', 'N/A'):.3f}")
        
        return evaluation
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        evaluation['error'] = str(e)
        return evaluation


def evaluate_single_response_with_json(question_data: Dict[str, Any], golden_data: Optional[Dict[str, Any]] = None, 
                                      temperature: float = 0.1, use_json_comparison: bool = True,
                                      flexible_keys: bool = True, n_judge_runs: int = 1) -> Dict[str, Any]:
    """
    Evaluate a single question-response pair with automatic JSON comparison and LLM judge fallback.
    
    This function intelligently chooses between JSON comparison and LLM judge:
    - If golden reference available and JSON extractable: uses fast JSON comparison
    - Otherwise: falls back to LLM judge evaluation
    
    For always using LLM judge (qualitative evaluation), use evaluate_single_response() instead.
    
    Args:
        question_data: Dictionary containing question, response, and metadata
        golden_data: Optional dictionary containing golden reference answers
        temperature: Temperature for LLM inference (default: 0.1 for consistency)
        use_json_comparison: If True, use JSON comparison with golden answers when available
        flexible_keys: If True, use flexible key matching (case-insensitive, ignore separators)
        n_judge_runs: Number of times to run LLM judge and average scores (default: 1)
        
    Returns:
        Dictionary containing evaluation results
    """
    print(f"\n{'='*80}")
    print(f"Evaluating Question: {question_data.get('question_id', 'N/A')} [AUTO MODE]")
    if n_judge_runs > 1:
        print(f"LLM Judge Runs: {n_judge_runs} (averaging scores)")
    print(f"{'='*80}")
    
    evaluation = {
        'question_id': question_data.get('question_id', 'N/A'),
        'question_type': question_data.get('question_type', 'N/A'),
        'stage': question_data.get('stage', 'N/A'),
        'difficulty': question_data.get('difficulty', 'N/A'),
        'answer_format': question_data.get('answer_format', 'N/A')
    }
    
    # Extract the answer from response
    response_text = question_data.get('response', '')
    answer_format = question_data.get('answer_format', '').lower()
    
    # Try to extract JSON or text answer
    extracted_json = None
    extracted_text = None
    
    if 'dict' in answer_format or 'json' in answer_format:
        extracted_json = extract_json_from_response(response_text)
        print(f"📄 Extracted JSON answer: {json.dumps(extracted_json, indent=2)[:200] if extracted_json else 'None'}")
    elif 'list' in answer_format:
        # Try both JSON and text extraction for lists
        extracted_json = extract_json_from_response(response_text)
        if not extracted_json:
            extracted_text = extract_answer_from_response(response_text)
        print(f"📄 Extracted answer: {extracted_json if extracted_json else extracted_text}")
    else:
        # Text format
        extracted_text = extract_answer_from_response(response_text)
        print(f"📄 Extracted text answer: {extracted_text[:200] if extracted_text else 'None'}")
    
    evaluation['extracted_answer'] = extracted_json if extracted_json is not None else extracted_text
    
    # If golden data provided and JSON comparison enabled, compare answers
    golden_answer = None
    if golden_data and use_json_comparison:
        question_id = question_data.get('question_id', 'N/A')
        # Look for golden answer by question_id
        if isinstance(golden_data, dict) and question_id in golden_data:
            golden_answer = golden_data[question_id]
        elif isinstance(golden_data, list):
            # Search in list of golden answers
            for item in golden_data:
                if item.get('question_id') == question_id:
                    golden_answer = item.get('golden_answer')
                    break
        
        if golden_answer and extracted_json:
            print(f"🎯 Comparing with golden reference (flexible_keys={flexible_keys})...")
            score, comparison_details = compare_json_answers(extracted_json, golden_answer, flexible_keys=flexible_keys)
            evaluation['json_comparison'] = {
                'score': score,
                'details': comparison_details,
                'golden_answer': golden_answer
            }
            print(f"📊 JSON Comparison Score: {score:.3f}")
            
            # If we have a direct comparison, use it as the overall score
            evaluation['overall_score'] = score
            evaluation['criterion_scores'] = {
                'correctness': score,
                'clarity': 0.8,  # Default for JSON format
                'relevance': 0.9   # Default for JSON format
            }
            evaluation['evaluation_method'] = 'json_comparison'
            
            if score == 1.0:
                evaluation['strengths'] = ['Exact match with golden reference']
                evaluation['weaknesses'] = []
            else:
                evaluation['strengths'] = []
                evaluation['weaknesses'] = comparison_details.get('differences', [])
            
            print(f"\n✅ Evaluation completed - Overall Score: {evaluation.get('overall_score', 'N/A'):.3f}")
            return evaluation
    
    # Fall back to LLM judge evaluation
    print(f"🤖 Using LLM judge evaluation...")
    if n_judge_runs > 1:
        print(f"📊 Running {n_judge_runs} independent evaluations...")
    
    prompts = create_judge_prompt(question_data, golden_answer)
    
    try:
        # Run LLM judge n times and collect results
        all_judge_evals = []
        
        for run_idx in range(n_judge_runs):
            if n_judge_runs > 1:
                print(f"\n  Run {run_idx + 1}/{n_judge_runs}...")
            # print(f"System prompt: {prompts['system_prompt']}")
            # print(f"User prompt: {prompts['user_prompt']}")
            # pdb.set_trace()
            judge_response = call_dar_gateway(
                system_prompt=prompts['system_prompt'],
                user_prompt=prompts['user_prompt'],
                temperature=temperature
            )
            # judge_response = llm_inference(
            #     system_prompt=prompts['system_prompt'],
            #     user_prompt=prompts['user_prompt'],
            #     temperature=temperature
            #)
            # pdb.set_trace()
            # Parse JSON response
            judge_eval = extract_json_from_response(judge_response)
            
            if not judge_eval:
                # Try parsing the whole response
                try:
                    judge_eval = json.loads(judge_response)
                except json.JSONDecodeError as e:
                    print(f"❌ Error parsing judge response (run {run_idx + 1}): {e}")
                    print(f"Raw response: {judge_response[:500]}")
                    continue
            
            all_judge_evals.append(judge_eval)
            
            if n_judge_runs > 1 and 'overall_score' in judge_eval:
                print(f"    Score: {judge_eval['overall_score']:.3f}")
        
        # Check if we got any valid evaluations
        if not all_judge_evals:
            evaluation['error'] = "All judge runs failed to parse"
            return evaluation
        
        # If only one run or only one valid result, use it directly
        if len(all_judge_evals) == 1:
            evaluation.update(all_judge_evals[0])
            evaluation['evaluation_method'] = 'llm_judge'
        else:
            # Average the scores across all runs
            print(f"\n📊 Averaging scores across {len(all_judge_evals)} runs...")
            averaged_eval = average_judge_evaluations(all_judge_evals)
            evaluation.update(averaged_eval)
            evaluation['evaluation_method'] = 'llm_judge_averaged'
            evaluation['n_judge_runs'] = len(all_judge_evals)
            evaluation['individual_runs'] = all_judge_evals  # Keep individual results for analysis
        
        print(f"\n✅ Evaluation completed - Overall Score: {evaluation.get('overall_score', 'N/A'):.3f}")
        
        return evaluation
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        evaluation['error'] = str(e)
        return evaluation


def average_judge_evaluations(evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Average multiple LLM judge evaluations.
    
    Args:
        evaluations: List of evaluation dictionaries from multiple judge runs
        
    Returns:
        Dictionary with averaged scores and aggregated results
    """
    if not evaluations:
        return {}
    
    # Average overall score
    overall_scores = [e.get('overall_score', 0) for e in evaluations if 'overall_score' in e]
    avg_overall_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
    
    # Average criterion scores
    criteria = ['correctness', 'clarity', 'relevance']
    avg_criterion_scores = {}
    
    for criterion in criteria:
        scores = [e.get('criterion_scores', {}).get(criterion, 0) for e in evaluations 
                  if 'criterion_scores' in e and criterion in e['criterion_scores']]
        if scores:
            avg_criterion_scores[criterion] = sum(scores) / len(scores)
    
    # Aggregate strengths and weaknesses (collect all unique items)
    all_strengths = set()
    all_weaknesses = set()
    
    for e in evaluations:
        if 'strengths' in e:
            all_strengths.update(e['strengths'])
        if 'weaknesses' in e:
            all_weaknesses.update(e['weaknesses'])
    
    # Combine feedback from all runs
    all_feedback = []
    for i, e in enumerate(evaluations, 1):
        if 'feedback' in e:
            all_feedback.append(f"Run {i}: {e['feedback']}")
    
    combined_feedback = "\n\n".join(all_feedback) if all_feedback else "No feedback available"
    
    # Calculate score statistics
    if overall_scores:
        score_std = (sum((s - avg_overall_score) ** 2 for s in overall_scores) / len(overall_scores)) ** 0.5
        score_min = min(overall_scores)
        score_max = max(overall_scores)
    else:
        score_std = 0.0
        score_min = 0.0
        score_max = 0.0
    
    return {
        'overall_score': avg_overall_score,
        'score_statistics': {
            'mean': avg_overall_score,
            'std': score_std,
            'min': score_min,
            'max': score_max,
            'individual_scores': overall_scores
        },
        'criterion_scores': avg_criterion_scores,
        'strengths': list(all_strengths),
        'weaknesses': list(all_weaknesses),
        'feedback': combined_feedback,
        'averaged_from_runs': len(evaluations)
    }


def evaluate_response_file(input_file: str, golden_file: Optional[str] = None, output_file: Optional[str] = None, 
                          temperature: float = 0.1, n_judge_runs: int = 1,
                          use_auto_mode: bool = False, flexible_keys: bool = True) -> List[Dict[str, Any]]:
    """
    Evaluate all responses in a question-response JSON file.
    
    By default, uses LLM judge for all evaluations (qualitative assessment).
    Set use_auto_mode=True to enable automatic JSON comparison when golden reference available.
    
    Args:
        input_file: Path to input JSON file with question-response pairs
        golden_file: Optional path to golden reference answers JSON file (deprecated - golden_answer field in input file is used)
        output_file: Optional path to save evaluation results (auto-generated if None)
        temperature: Temperature for LLM inference
        n_judge_runs: Number of times to run LLM judge and average scores (default: 1)
        use_auto_mode: If True, use JSON comparison when possible, LLM judge otherwise (default: False)
        flexible_keys: If True, use flexible key matching (only for auto mode with JSON comparison)
        
    Returns:
        List of evaluation results
    """
    print(f"\n{'='*80}")
    print(f"Starting Evaluation")
    print(f"Input File: {input_file}")
    if golden_file:
        print(f"⚠️  Warning: golden_file parameter is deprecated. Using 'golden_answer' field from input file instead.")
    print(f"Mode: {'AUTO (JSON + LLM Judge)' if use_auto_mode else 'LLM Judge Only'}")
    if use_auto_mode:
        print(f"Flexible Key Matching: {flexible_keys}")
    if n_judge_runs > 1:
        print(f"LLM Judge Runs per Question: {n_judge_runs} (averaging)")
    print(f"{'='*80}\n")
    
    # Load question-response data
    with open(input_file, 'r') as f:
        questions_data = json.load(f)
    
    print(f"📊 Loaded {len(questions_data)} question-response pairs")
    
    # Build golden_data from the input file's golden_answer fields
    golden_data = {}
    has_golden_answers = False
    for question_data in questions_data:
        question_id = question_data.get('question_id')
        golden_answer = question_data.get('golden_answer')
        if question_id and golden_answer is not None:
            golden_data[question_id] = golden_answer
            has_golden_answers = True
    
    if has_golden_answers:
        print(f"🎯 Found golden reference answers in {len(golden_data)} questions")
    else:
        print(f"ℹ️  No golden_answer fields found in input file")
        golden_data = None
    
    # Load external golden answers if provided (for backward compatibility, but deprecated)
    if golden_file and os.path.exists(golden_file):
        with open(golden_file, 'r') as f:
            external_golden_data = json.load(f)
        print(f"⚠️  Warning: Loaded external golden file (deprecated). Prioritizing golden_answer fields from input file.")
        # Merge external golden data, but prioritize the input file's golden_answer fields
        if isinstance(external_golden_data, dict):
            for key, value in external_golden_data.items():
                if key not in golden_data:
                    golden_data[key] = value
        elif isinstance(external_golden_data, list):
            for item in external_golden_data:
                q_id = item.get('question_id')
                g_ans = item.get('golden_answer')
                if q_id and g_ans is not None and q_id not in golden_data:
                    golden_data[q_id] = g_ans
    
    # Evaluate each response
    evaluations = []
    for idx, question_data in enumerate(questions_data, 1):
        print(f"\n{'='*80}")
        print(f"Progress: {idx}/{len(questions_data)}")
        print(f"{'='*80}")
        
        if use_auto_mode:
            # Use auto mode with JSON comparison
            evaluation = evaluate_single_response_with_json(
                question_data, 
                golden_data=golden_data,
                temperature=temperature,
                use_json_comparison=True,
                flexible_keys=flexible_keys,
                n_judge_runs=n_judge_runs
            )
        else:
            # Use LLM judge only (default)
            # Extract golden answer for this question if available
            golden_answer = None
            if golden_data:
                question_id = question_data.get('question_id', 'N/A')
                if isinstance(golden_data, dict) and question_id in golden_data:
                    golden_answer = golden_data[question_id]
                elif isinstance(golden_data, list):
                    for item in golden_data:
                        if item.get('question_id') == question_id:
                            golden_answer = item.get('golden_answer')
                            break
            
            evaluation = evaluate_single_response(
                question_data,
                golden_answer=golden_answer,
                temperature=temperature,
                n_judge_runs=n_judge_runs
            )
        
        evaluations.append(evaluation)
    
    # Generate output filename if not provided
    if output_file is None:
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_dir = os.path.dirname(input_file) or "qa_responses"
        output_file = os.path.join(output_dir, f"{base_name}_evaluation.json")
    
    # Save results
    output_dir = os.path.dirname(output_file)
    print(f"Output directory: {output_dir}")
    if output_dir:  # Only create directory if path contains a directory component
        os.makedirs(output_dir, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(evaluations, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✅ Evaluation Complete!")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}\n")
    
    # Write summary statistics to file and print to console
    summary_file = write_evaluation_summary(evaluations, output_file)
    
    print(f"\n{'='*80}")
    print(f"📄 Summary saved to: {summary_file}")
    print(f"{'='*80}\n")
    
    return evaluations


def write_evaluation_summary(evaluations: List[Dict[str, Any]], output_file: str):
    """
    Write summary statistics from evaluations to a file and print to console.
    
    Args:
        evaluations: List of evaluation dictionaries
        output_file: Path to output file (will create .sum file)
        
    Returns:
        Path to the created summary file
    """
    # Generate summary filename
    base_name = os.path.splitext(output_file)[0]
    summary_file = f"{base_name}.sum"
    
    with open(summary_file, 'w') as f:
        # Helper function to write to both file and console
        def write_line(line=""):
            print(line)
            f.write(line + "\n")
        
        write_line("\n" + "="*80)
        write_line("EVALUATION SUMMARY")
        write_line("="*80)
        
        # Filter out error cases
        valid_evals = [e for e in evaluations if 'overall_score' in e]
        error_count = len(evaluations) - len(valid_evals)
        
        if not valid_evals:
            write_line("❌ No valid evaluations found")
            return summary_file
        
        # Overall statistics
        overall_scores = [e['overall_score'] for e in valid_evals]
        avg_overall = sum(overall_scores) / len(overall_scores)
        
        write_line(f"\n📊 Overall Statistics:")
        write_line(f"  Total Evaluated: {len(evaluations)}")
        write_line(f"  Valid Evaluations: {len(valid_evals)}")
        write_line(f"  Errors: {error_count}")
        write_line(f"  Average Overall Score: {avg_overall:.3f}/1.0")
        
        # Evaluation method breakdown
        json_comp_count = sum(1 for e in valid_evals if e.get('evaluation_method') == 'json_comparison')
        llm_judge_count = sum(1 for e in valid_evals if e.get('evaluation_method') == 'llm_judge')
        llm_judge_avg_count = sum(1 for e in valid_evals if e.get('evaluation_method') == 'llm_judge_averaged')
        
        write_line(f"\n🔍 Evaluation Methods:")
        write_line(f"  JSON Comparison: {json_comp_count}")
        write_line(f"  LLM Judge (single): {llm_judge_count}")
        if llm_judge_avg_count > 0:
            write_line(f"  LLM Judge (averaged): {llm_judge_avg_count}")
            # Show average number of runs
            avg_runs = [e.get('n_judge_runs', 1) for e in valid_evals 
                       if e.get('evaluation_method') == 'llm_judge_averaged']
            if avg_runs:
                write_line(f"    Average runs per question: {sum(avg_runs) / len(avg_runs):.1f}")
        
        # Score variance for averaged runs
        averaged_evals = [e for e in valid_evals if 'score_statistics' in e]
        if averaged_evals:
            write_line(f"\n📉 Score Variance (Averaged Runs):")
            avg_std = sum(e['score_statistics']['std'] for e in averaged_evals) / len(averaged_evals)
            avg_range = sum(e['score_statistics']['max'] - e['score_statistics']['min'] 
                           for e in averaged_evals) / len(averaged_evals)
            write_line(f"  Average Std Dev: {avg_std:.3f}")
            write_line(f"  Average Range: {avg_range:.3f}")
            write_line(f"  (Lower values indicate more consistent judge responses)")
        
        # Score distribution (in 0.1 increments for 0-1 scale)
        write_line(f"\n📈 Score Distribution:")
        for i in range(11):
            score_threshold = i / 10.0
            if i < 10:
                count = sum(1 for s in overall_scores if score_threshold <= s < score_threshold + 0.1)
                percentage = (count / len(overall_scores)) * 100
                bar = "█" * int(percentage / 2)
                write_line(f"  {score_threshold:.1f}-{score_threshold+0.1:.1f}: {count:3d} ({percentage:5.1f}%) {bar}")
            else:
                count = sum(1 for s in overall_scores if s == 1.0)
                percentage = (count / len(overall_scores)) * 100
                bar = "█" * int(percentage / 2)
                write_line(f"  {score_threshold:.1f}     : {count:3d} ({percentage:5.1f}%) {bar}")
        
        # Criterion averages
        write_line(f"\n📋 Average Scores by Criterion:")
        criteria = ['correctness', 'clarity', 'relevance']
        for criterion in criteria:
            scores = [e['criterion_scores'].get(criterion, 0) for e in valid_evals if 'criterion_scores' in e]
            if scores:
                avg = sum(scores) / len(scores)
                write_line(f"  {criterion.replace('_', ' ').title():20s}: {avg:.3f}/1.0")
        
        # By question type
        write_line(f"\n🏷️  Scores by Question Type:")
        by_type = {}
        for e in valid_evals:
            q_type = e.get('question_type', 'unknown')
            if q_type not in by_type:
                by_type[q_type] = []
            by_type[q_type].append(e['overall_score'])
        
        for q_type, scores in sorted(by_type.items()):
            avg = sum(scores) / len(scores)
            write_line(f"  {q_type:20s}: {avg:.3f}/1.0 (n={len(scores)})")
        
        # By difficulty
        write_line(f"\n⚡ Scores by Difficulty:")
        by_difficulty = {}
        for e in valid_evals:
            difficulty = e.get('difficulty', 'unknown')
            if difficulty not in by_difficulty:
                by_difficulty[difficulty] = []
            by_difficulty[difficulty].append(e['overall_score'])
        
        difficulty_order = ['easy', 'medium', 'hard']
        for difficulty in difficulty_order:
            if difficulty in by_difficulty:
                scores = by_difficulty[difficulty]
                avg = sum(scores) / len(scores)
                write_line(f"  {difficulty.title():20s}: {avg:.3f}/1.0 (n={len(scores)})")
        
        # Perfect scores
        perfect_count = sum(1 for s in overall_scores if s == 1.0)
        write_line(f"\n🎯 Perfect Scores (1.0): {perfect_count}/{len(valid_evals)} ({perfect_count/len(valid_evals)*100:.1f}%)")
        
        # Passing threshold (e.g., 0.7)
        passing_threshold = 0.7
        passing_count = sum(1 for s in overall_scores if s >= passing_threshold)
        write_line(f"✅ Passing Scores (≥{passing_threshold}): {passing_count}/{len(valid_evals)} ({passing_count/len(valid_evals)*100:.1f}%)")
        
        write_line("\n" + "="*80)
    
    return summary_file


def print_evaluation_summary(evaluations: List[Dict[str, Any]]):
    """
    Print summary statistics from evaluations.
    
    Args:
        evaluations: List of evaluation dictionaries
    """
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    # Filter out error cases
    valid_evals = [e for e in evaluations if 'overall_score' in e]
    error_count = len(evaluations) - len(valid_evals)
    
    if not valid_evals:
        print("❌ No valid evaluations found")
        return
    
    # Overall statistics
    overall_scores = [e['overall_score'] for e in valid_evals]
    avg_overall = sum(overall_scores) / len(overall_scores)
    
    print(f"\n📊 Overall Statistics:")
    print(f"  Total Evaluated: {len(evaluations)}")
    print(f"  Valid Evaluations: {len(valid_evals)}")
    print(f"  Errors: {error_count}")
    print(f"  Average Overall Score: {avg_overall:.3f}/1.0")
    
    # Evaluation method breakdown
    json_comp_count = sum(1 for e in valid_evals if e.get('evaluation_method') == 'json_comparison')
    llm_judge_count = sum(1 for e in valid_evals if e.get('evaluation_method') == 'llm_judge')
    llm_judge_avg_count = sum(1 for e in valid_evals if e.get('evaluation_method') == 'llm_judge_averaged')
    
    print(f"\n🔍 Evaluation Methods:")
    print(f"  JSON Comparison: {json_comp_count}")
    print(f"  LLM Judge (single): {llm_judge_count}")
    if llm_judge_avg_count > 0:
        print(f"  LLM Judge (averaged): {llm_judge_avg_count}")
        # Show average number of runs
        avg_runs = [e.get('n_judge_runs', 1) for e in valid_evals 
                   if e.get('evaluation_method') == 'llm_judge_averaged']
        if avg_runs:
            print(f"    Average runs per question: {sum(avg_runs) / len(avg_runs):.1f}")
    
    # Score variance for averaged runs
    averaged_evals = [e for e in valid_evals if 'score_statistics' in e]
    if averaged_evals:
        print(f"\n📉 Score Variance (Averaged Runs):")
        avg_std = sum(e['score_statistics']['std'] for e in averaged_evals) / len(averaged_evals)
        avg_range = sum(e['score_statistics']['max'] - e['score_statistics']['min'] 
                       for e in averaged_evals) / len(averaged_evals)
        print(f"  Average Std Dev: {avg_std:.3f}")
        print(f"  Average Range: {avg_range:.3f}")
        print(f"  (Lower values indicate more consistent judge responses)")
    
    # Score distribution (in 0.1 increments for 0-1 scale)
    print(f"\n📈 Score Distribution:")
    for i in range(11):
        score_threshold = i / 10.0
        if i < 10:
            count = sum(1 for s in overall_scores if score_threshold <= s < score_threshold + 0.1)
            percentage = (count / len(overall_scores)) * 100
            bar = "█" * int(percentage / 2)
            print(f"  {score_threshold:.1f}-{score_threshold+0.1:.1f}: {count:3d} ({percentage:5.1f}%) {bar}")
        else:
            count = sum(1 for s in overall_scores if s == 1.0)
            percentage = (count / len(overall_scores)) * 100
            bar = "█" * int(percentage / 2)
            print(f"  {score_threshold:.1f}     : {count:3d} ({percentage:5.1f}%) {bar}")
    
    # Criterion averages
    print(f"\n📋 Average Scores by Criterion:")
    criteria = ['correctness', 'clarity', 'relevance']
    for criterion in criteria:
        scores = [e['criterion_scores'].get(criterion, 0) for e in valid_evals if 'criterion_scores' in e]
        if scores:
            avg = sum(scores) / len(scores)
            print(f"  {criterion.replace('_', ' ').title():20s}: {avg:.3f}/1.0")
    
    # By question type
    print(f"\n🏷️  Scores by Question Type:")
    by_type = {}
    for e in valid_evals:
        q_type = e.get('question_type', 'unknown')
        if q_type not in by_type:
            by_type[q_type] = []
        by_type[q_type].append(e['overall_score'])
    
    for q_type, scores in sorted(by_type.items()):
        avg = sum(scores) / len(scores)
        print(f"  {q_type:20s}: {avg:.3f}/1.0 (n={len(scores)})")
    
    # By difficulty
    print(f"\n⚡ Scores by Difficulty:")
    by_difficulty = {}
    for e in valid_evals:
        difficulty = e.get('difficulty', 'unknown')
        if difficulty not in by_difficulty:
            by_difficulty[difficulty] = []
        by_difficulty[difficulty].append(e['overall_score'])
    
    difficulty_order = ['easy', 'medium', 'hard']
    for difficulty in difficulty_order:
        if difficulty in by_difficulty:
            scores = by_difficulty[difficulty]
            avg = sum(scores) / len(scores)
            print(f"  {difficulty.title():20s}: {avg:.3f}/1.0 (n={len(scores)})")
    
    # Perfect scores
    perfect_count = sum(1 for s in overall_scores if s == 1.0)
    print(f"\n🎯 Perfect Scores (1.0): {perfect_count}/{len(valid_evals)} ({perfect_count/len(valid_evals)*100:.1f}%)")
    
    # Passing threshold (e.g., 0.7)
    passing_threshold = 0.7
    passing_count = sum(1 for s in overall_scores if s >= passing_threshold)
    print(f"✅ Passing Scores (≥{passing_threshold}): {passing_count}/{len(valid_evals)} ({passing_count/len(valid_evals)*100:.1f}%)")
    
    print("\n" + "="*80)


def compare_evaluations(eval_file1: str, eval_file2: str):
    """
    Compare two evaluation files (e.g., different models).
    
    Args:
        eval_file1: Path to first evaluation file
        eval_file2: Path to second evaluation file
    """
    with open(eval_file1, 'r') as f:
        evals1 = json.load(f)
    
    with open(eval_file2, 'r') as f:
        evals2 = json.load(f)
    
    print(f"\n{'='*80}")
    print(f"COMPARISON: {os.path.basename(eval_file1)} vs {os.path.basename(eval_file2)}")
    print(f"{'='*80}")
    
    # Filter valid evaluations
    valid1 = [e for e in evals1 if 'overall_score' in e]
    valid2 = [e for e in evals2 if 'overall_score' in e]
    
    # Overall comparison
    avg1 = sum(e['overall_score'] for e in valid1) / len(valid1) if valid1 else 0
    avg2 = sum(e['overall_score'] for e in valid2) / len(valid2) if valid2 else 0
    
    print(f"\nOverall Average Scores:")
    print(f"  File 1: {avg1:.2f}/5.0 (n={len(valid1)})")
    print(f"  File 2: {avg2:.2f}/5.0 (n={len(valid2)})")
    print(f"  Difference: {avg2 - avg1:+.2f}")
    
    print(f"\n{'='*80}")


if __name__ == "__main__":
    import sys
    
    # Example usage
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        golden_file = sys.argv[2] if len(sys.argv) > 2 else None
        output_file = sys.argv[3] if len(sys.argv) > 3 else None
        
        print(f"🚀 Starting evaluation of {input_file}")
        evaluations = evaluate_response_file(input_file, golden_file, output_file)


