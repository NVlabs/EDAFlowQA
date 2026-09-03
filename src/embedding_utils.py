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
Embedding-based JSON filtering utilities for QA generation.

This module provides functions to:
1. Flatten JSON to paths with semantic meaning
2. Compute similarity between questions and JSON paths
3. Extract only relevant portions of JSON for LLM validation

This dramatically reduces token usage and improves validation accuracy.
"""

import json
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import pdb


class EmbeddingModel:
    """Wrapper for different embedding model backends."""
    
    def __init__(self, model_type: str = "sentence-transformer"):
        """
        Initialize embedding model.
        
        Args:
            model_type: One of "sentence-transformer", "openai", or "llm"
        """
        self.model_type = model_type
        self.model = None
        
        if model_type == "sentence-transformer":
            try:
                from sentence_transformers import SentenceTransformer
                # Use a small, fast model (80MB, good quality)
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                print("[Embedding] Loaded sentence-transformer model: all-MiniLM-L6-v2")
            except ImportError:
                print("[Embedding] sentence-transformers not installed. Install with: pip install sentence-transformers")
                print("[Embedding] Falling back to keyword matching")
                self.model_type = "keyword"
        
        elif model_type == "openai":
            # Will use OpenAI API for embeddings
            print("[Embedding] Using OpenAI embeddings (requires API key)")
        
        elif model_type == "keyword":
            # Simple keyword matching fallback
            print("[Embedding] Using simple keyword matching (no embeddings)")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            numpy array of embeddings, shape (len(texts), embedding_dim)
        """
        if self.model_type == "sentence-transformer":
            return self.model.encode(texts, convert_to_numpy=True)
        
        elif self.model_type == "openai":
            import openai
            embeddings = []
            for text in texts:
                response = openai.Embedding.create(
                    input=text,
                    model="text-embedding-3-small"
                )
                embeddings.append(response['data'][0]['embedding'])
            return np.array(embeddings)
        
        elif self.model_type == "keyword":
            # Fallback: simple keyword matching (not real embeddings)
            return None
        
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")


def flatten_json_to_paths(
    json_obj: Any,
    prefix: str = "",
    max_depth: int = 10,
    current_depth: int = 0,
    include_values: bool = True
) -> List[Tuple[str, Any, str]]:
    """
    Flatten JSON into semantic chunks with hierarchical context.
    
    CHUNKING STRATEGY:
    - "Leaf objects" (dicts with only primitives) → kept as complete chunks
    - Primitive groups (related fields) → kept together as chunks
    - Large nested structures → recursively chunked
    
    Each chunk is self-contained and includes its hierarchical path for context.
    
    Args:
        json_obj: JSON object to flatten (dict, list, or primitive)
        prefix: Current path prefix (hierarchical context)
        max_depth: Maximum depth to traverse
        current_depth: Current recursion depth
        include_values: Whether to include primitive values in paths
        
    Returns:
        List of (path, chunk, semantic_text) tuples where:
        - path: hierarchical path like "flow_execution_summary.execution_status"
        - chunk: meaningful semantic unit (dict/value)
        - semantic_text: rich description for embedding
        
    Chunking Example:
        Input: {
            "flow_execution_summary": {
                "execute_successfully": true,
                "exit_code": 0,
                "exit_line": "...",
                "stages": [...]
            }
        }
        
        Output chunks:
        1. ("flow_execution_summary.execution_status", 
            {"execute_successfully": true, "exit_code": 0, "exit_line": "..."},
            "flow execution summary execute successfully True exit code 0 exit line ...")
        2. ("flow_execution_summary.stages", [...], "...")
    """
    if current_depth >= max_depth:
        return []
    
    paths = []
    
    if isinstance(json_obj, dict):
        # Check if this is a "leaf object" - only contains primitive values
        is_leaf_object = _is_leaf_object(json_obj)
        
        if is_leaf_object:
            # Keep this entire dict as a single semantic chunk
            semantic_text = _create_semantic_text_for_object(prefix, json_obj)
            paths.append((prefix, json_obj, semantic_text))
        else:
            # Mixed dict - check if we should keep related small items together
            primitives = {}
            nested = {}
            leaf_objects = {}  # Track nested leaf objects
            
            for key, value in json_obj.items():
                if isinstance(value, dict):
                    if _is_leaf_object(value):
                        leaf_objects[key] = value
                    else:
                        nested[key] = value
                elif isinstance(value, list):
                    nested[key] = value
                else:
                    primitives[key] = value
            
            # Strategy: If dict is small with only primitives + few leaf objects, keep together
            total_items = len(primitives) + len(leaf_objects)
            if total_items <= 5 and len(nested) == 0:
                # Small dict with only primitives and leaf objects - keep all together
                combined_data = {**primitives}
                for key, leaf_obj in leaf_objects.items():
                    combined_data[key] = leaf_obj
                semantic_text = _create_semantic_text_for_object(prefix, combined_data)
                paths.append((prefix, combined_data, semantic_text))
            else:
                # Larger or more complex - use normal chunking
                
                # Group primitives by semantic category
                if primitives:
                    primitive_chunks = _chunk_primitives_by_semantics(primitives, prefix)
                    paths.extend(primitive_chunks)
                
                # Add leaf objects as separate chunks
                for key, leaf_obj in leaf_objects.items():
                    new_prefix = f"{prefix}.{key}" if prefix else key
                    semantic_text = _create_semantic_text_for_object(new_prefix, leaf_obj)
                    paths.append((new_prefix, leaf_obj, semantic_text))
                
                # Process nested structures
                for key, value in nested.items():
                    new_prefix = f"{prefix}.{key}" if prefix else key
                    
                    if isinstance(value, dict):
                        # Recurse into nested dict
                        paths.extend(flatten_json_to_paths(
                            value, new_prefix, max_depth, current_depth + 1, include_values
                        ))
                    elif isinstance(value, list):
                        # Recurse into list
                        paths.extend(flatten_json_to_paths(
                            value, new_prefix, max_depth, current_depth + 1, include_values
                        ))
    
    elif isinstance(json_obj, list):
        for idx, item in enumerate(json_obj):
            # Try to create a descriptive identifier for this array item
            item_id = _get_item_identifier(item, idx)
            new_prefix = f"{prefix}[{item_id}]"
            
            paths.extend(flatten_json_to_paths(
                item, new_prefix, max_depth, current_depth + 1, include_values
            ))
    
    else:
        # Primitive value at root
        semantic_text = f"{prefix} value {json_obj}" if include_values else prefix
        paths.append((prefix, json_obj, semantic_text))
    
    return paths


def _is_leaf_object(obj: Dict) -> bool:
    """
    Check if a dict is a "leaf object" that should be kept together.
    
    A leaf object contains only primitive values (no nested dicts/lists),
    and represents a cohesive semantic unit (e.g., a timing path, a metric set).
    
    Args:
        obj: Dictionary to check
        
    Returns:
        True if this is a leaf object that should be kept together
        
    Examples:
        {"wns": 0.012, "tns": 0.0} → True (keep together)
        {"startpoint": "FF1", "endpoint": "FF2", "slack": -0.01} → True (timing path)
        {"substages": {...}, "metrics": {...}} → False (has nested structures)
    """
    if not isinstance(obj, dict) or len(obj) == 0:
        return False
    
    # Check if all values are primitives (not dict/list)
    for value in obj.values():
        if isinstance(value, (dict, list)):
            return False
    
    # All values are primitives - this is a leaf object
    return True


def _get_item_identifier(item: Any, idx: int) -> str:
    """
    Extract a meaningful identifier from an array item for use in paths.
    
    Tries to find identifying fields like stage_type, name, id, etc. to create
    descriptive paths instead of just numeric indices.
    
    Args:
        item: Array item (dict, primitive, etc.)
        idx: Numeric index as fallback
        
    Returns:
        Descriptive identifier string
        
    Examples:
        {"stage_type": "synthesis", "occurrence": 1} → "synthesis_1"
        {"stage_type": "cts"} → "cts"
        {"name": "buffer_123"} → "buffer_123"
        {"id": 42} → "42"
        "some_string" → "0" (use index for non-dict)
    """
    if not isinstance(item, dict):
        # Not a dict, use numeric index
        return str(idx)
    
    # Priority order for identifier fields
    identifier_fields = [
        # Stage-related
        ("stage_type", "occurrence"),  # e.g., synthesis_1, placement_1
        ("stage_type", "sequence_number"),  # e.g., synthesis_1
        "stage_type",
        "stage_name",
        
        # General identifiers
        "name",
        "id",
        "identifier",
        "key",
        "type",
        
        # Substage patterns
        "substage_name",
        "substage_id",
    ]
    
    # Try combined fields first (e.g., stage_type + occurrence)
    for field_combo in identifier_fields:
        if isinstance(field_combo, tuple):
            # Multiple fields to combine
            if all(f in item for f in field_combo):
                parts = [str(item[f]) for f in field_combo]
                return "_".join(parts)
        elif isinstance(field_combo, str):
            # Single field
            if field_combo in item:
                value = item[field_combo]
                # Clean up the value
                if isinstance(value, str):
                    # Remove special chars, truncate if too long
                    clean = value.replace(" ", "_").replace("/", "_")
                    if len(clean) > 30:
                        clean = clean[:30]
                    return clean
                else:
                    return str(value)
    
    # Fallback: use numeric index
    return str(idx)


def _chunk_primitives_by_semantics(
    primitives: Dict[str, Any],
    parent_prefix: str
) -> List[Tuple[str, Dict, str]]:
    """
    Chunk primitive fields into semantic groups for meaningful context.
    
    SMART GROUPING STRATEGY:
    1. If dict is small (<= 6 primitives), keep all together as one chunk
    2. Otherwise, group by semantic categories
    3. Always respect dict boundaries (don't merge across different parents)
    
    Args:
        primitives: Dict of primitive field names and values
        parent_prefix: Hierarchical path prefix
        
    Returns:
        List of (path, chunk_dict, semantic_text) tuples
        
    Example Small Dict (keep together):
        Input: {"design_name": "aes", "technology_node": "nangate45", "log_file": "..."}
        Output: [("parent.all_fields", {...}, "...")]
        
    Example Large Dict (group by semantics):
        Input: {8+ fields with execution, design, file info mixed}
        Output: [
            ("parent.execution_status", {...}),
            ("parent.design_info", {...}),
            ("parent.file_info", {...})
        ]
    """
    if len(primitives) == 0:
        return []
    
    # Strategy 1: Small dict - keep all primitives together
    if len(primitives) <= 6:
        chunk_path = parent_prefix if parent_prefix else "root_fields"
        chunk_semantic = _create_semantic_text_for_object(chunk_path, primitives)
        return [(chunk_path, primitives, chunk_semantic)]
    
    # Strategy 2: Large dict - group by semantic categories
    semantic_categories = {
        "execution_status": ["execute", "exit", "success", "failure", "status", "code", "line"],
        "timing_info": ["time", "timestamp", "duration", "date", "start", "end"],
        "design_info": ["design", "name", "technology", "node", "platform", "config"],
        "file_info": ["file", "path", "directory", "location", "output"],
        "error_info": ["error", "warning", "issue", "problem", "detected"],
        "metric_info": ["metric", "value", "count", "total", "number", "wns", "tns", "slack"],
        "stage_info": ["stage", "sequence", "occurrence", "type"],
        "line_range_info": ["line", "range", "start", "end", "extracted"],
    }
    
    # Categorize each primitive field
    categorized = {}
    uncategorized = {}
    
    for field_name, field_value in primitives.items():
        assigned = False
        field_lower = field_name.lower()
        
        for category, keywords in semantic_categories.items():
            if any(keyword in field_lower for keyword in keywords):
                if category not in categorized:
                    categorized[category] = {}
                categorized[category][field_name] = field_value
                assigned = True
                break
        
        if not assigned:
            uncategorized[field_name] = field_value
    
    # Create chunks
    chunks = []
    
    # Add categorized chunks
    for category, fields in categorized.items():
        # Only create separate chunks for categories with multiple fields
        if len(fields) >= 2:
            chunk_path = f"{parent_prefix}.{category}" if parent_prefix else category
            chunk_semantic = _create_semantic_text_for_object(chunk_path, fields)
            chunks.append((chunk_path, fields, chunk_semantic))
        else:
            # Single field - add to uncategorized
            uncategorized.update(fields)
    
    # If we have uncategorized fields, decide how to handle them
    if uncategorized:
        if len(uncategorized) <= 3 and len(chunks) > 0:
            # Few uncategorized fields and we have other chunks - merge with closest semantic chunk
            # For simplicity, merge into first chunk or create "other" chunk
            if chunks:
                # Merge into first chunk
                first_path, first_data, _ = chunks[0]
                first_data.update(uncategorized)
                # Recreate semantic text
                chunks[0] = (first_path, first_data, _create_semantic_text_for_object(first_path, first_data))
            else:
                # No other chunks, create one for uncategorized
                chunk_path = parent_prefix if parent_prefix else "fields"
                chunk_semantic = _create_semantic_text_for_object(chunk_path, uncategorized)
                chunks.append((chunk_path, uncategorized, chunk_semantic))
        else:
            # Many uncategorized fields - create separate chunk
            chunk_path = f"{parent_prefix}.other_fields" if parent_prefix else "other_fields"
            chunk_semantic = _create_semantic_text_for_object(chunk_path, uncategorized)
            chunks.append((chunk_path, uncategorized, chunk_semantic))
    
    # If no chunks created (all single-field categories), keep all together
    if not chunks:
        chunk_path = parent_prefix if parent_prefix else "fields"
        chunk_semantic = _create_semantic_text_for_object(chunk_path, primitives)
        return [(chunk_path, primitives, chunk_semantic)]
    
    return chunks


def _group_related_primitives(
    primitives: List[Tuple[str, Any]], 
    parent_prefix: str
) -> List[Tuple[str, Dict, str]]:
    """
    DEPRECATED - Use _chunk_primitives_by_semantics instead.
    """
    return []


def _create_enhanced_semantic_text(
    path: str,
    key: str, 
    value: Any,
    sibling_primitives: Dict[str, Any],
    include_values: bool = True
) -> str:
    """
    Create enhanced semantic text for a primitive field by including sibling context.
    
    This helps related fields (like execute_successfully, exit_code, exit_line) to be
    found together when they're semantically related to a query, even though they're
    stored as separate paths in the original JSON.
    
    Args:
        path: Full dot-separated path to this field
        key: Field name
        value: Field value
        sibling_primitives: Dict of all primitive siblings at same level
        include_values: Whether to include values in semantic text
        
    Returns:
        Enhanced semantic text including sibling context
        
    Example:
        key="execute_successfully", value=True
        siblings={"exit_code": 0, "exit_line": "..."}
        
        Result: "execute successfully True exit code 0 exit line [...]"
        (includes sibling context for better matching!)
    """
    # Start with the path
    base_text = path.replace("_", " ").replace(".", " ")
    
    # Add this field's value
    if include_values:
        value_str = str(value)[:50]  # Truncate long values
        base_text = f"{base_text} {value_str}"
    
    # Define semantic groups - which fields are related
    semantic_groups = {
        "execution": ["execute", "exit", "success", "failure", "status", "code"],
        "timing": ["time", "timestamp", "duration", "date"],
        "identification": ["name", "id", "identifier", "label"],
        "location": ["file", "path", "directory", "location"],
        "error": ["error", "warning", "issue", "problem"],
    }
    
    # Find which group this field belongs to
    key_lower = key.lower()
    field_group = None
    for group_name, keywords in semantic_groups.items():
        if any(kw in key_lower for kw in keywords):
            field_group = group_name
            break
    
    # If this field is in a semantic group, include related sibling values
    if field_group and len(sibling_primitives) > 1:
        related_context = []
        for sibling_key, sibling_value in sibling_primitives.items():
            if sibling_key == key:
                continue  # Skip self
            
            sibling_lower = sibling_key.lower()
            # Check if sibling is in same semantic group
            for keyword in semantic_groups[field_group]:
                if keyword in sibling_lower:
                    # Include this sibling in context
                    sibling_text = f"{sibling_key.replace('_', ' ')} {str(sibling_value)[:50]}"
                    related_context.append(sibling_text)
                    break
        
        # Add up to 3 related siblings to context
        if related_context:
            context_str = " ".join(related_context[:3])
            base_text = f"{base_text} {context_str}"
    
    return base_text


def _create_semantic_text_for_object(prefix: str, obj: Dict) -> str:
    """
    Create semantic text for a leaf object that includes key-value pairs.
    
    Args:
        prefix: Path prefix
        obj: Dictionary object
        
    Returns:
        Semantic text describing the object
        
    Example:
        prefix = "timing_paths[0]"
        obj = {"startpoint": "FF1", "endpoint": "FF2", "slack": -0.01}
        Result: "timing paths 0 startpoint FF1 endpoint FF2 slack -0.01"
    """
    # Start with the path
    semantic_parts = [prefix.replace("_", " ").replace(".", " ").replace("[", " ").replace("]", " ")]
    
    # Add key-value pairs (limit to avoid extremely long text)
    max_fields = 8  # Include up to 8 fields in semantic text
    for i, (key, value) in enumerate(obj.items()):
        if i >= max_fields:
            break
        
        key_clean = key.replace("_", " ")
        
        # Format value based on type
        if isinstance(value, (int, float)):
            value_str = str(value)
        elif isinstance(value, str):
            # Truncate long strings
            value_str = value[:50] if len(value) > 50 else value
        else:
            value_str = str(value)[:50]
        
        semantic_parts.append(f"{key_clean} {value_str}")
    
    return " ".join(semantic_parts)


def compute_path_similarities(
    question: str,
    json_paths: List[Tuple[str, Any, str]],
    embedding_model: EmbeddingModel
) -> List[Tuple[str, Any, float]]:
    """
    Compute similarity scores between question and each JSON path.
    
    Args:
        question: The question text
        json_paths: List of (path, value, semantic_text) tuples from flatten_json_to_paths
        embedding_model: Initialized EmbeddingModel instance
        
    Returns:
        List of (path, value, similarity_score) sorted by similarity (highest first)
    """
    if embedding_model.model_type == "keyword":
        # Fallback: simple keyword matching
        return _keyword_similarity(question, json_paths)
    
    # Extract semantic texts for embedding
    semantic_texts = [semantic_text for _, _, semantic_text in json_paths]
    # print("Semantic texts:", semantic_texts)
    # pdb.set_trace()
    
    # Encode question and all paths
    question_embedding = embedding_model.encode([question])[0]
    path_embeddings = embedding_model.encode(semantic_texts)
    
    # Compute cosine similarities
    similarities = cosine_similarity_batch(question_embedding, path_embeddings)
    
    # Combine paths with their similarity scores
    results = [
        (path, value, float(sim))
        for (path, value, _), sim in zip(json_paths, similarities)
    ]
    
    # Sort by similarity (highest first)
    results.sort(key=lambda x: x[2], reverse=True)
    
    return results


def _keyword_similarity(
    question: str,
    json_paths: List[Tuple[str, Any, str]]
) -> List[Tuple[str, Any, float]]:
    """
    Fallback keyword-based similarity when embeddings not available.
    
    Simple but effective: count keyword overlaps.
    """
    question_lower = question.lower()
    question_words = set(question_lower.split())
    
    results = []
    for path, value, semantic_text in json_paths:
        semantic_lower = semantic_text.lower()
        semantic_words = set(semantic_lower.split())
        
        # Count keyword overlaps
        overlap = len(question_words & semantic_words)
        
        # Boost score if exact substring match
        boost = 0
        for word in question_words:
            if len(word) > 3 and word in semantic_lower:
                boost += 0.5
        
        similarity = overlap + boost
        results.append((path, value, similarity))
    
    # Sort by similarity
    results.sort(key=lambda x: x[2], reverse=True)
    
    return results


def cosine_similarity_batch(
    query_embedding: np.ndarray,
    embeddings: np.ndarray
) -> np.ndarray:
    """
    Compute cosine similarity between a query and batch of embeddings.
    
    Args:
        query_embedding: 1D array of shape (dim,)
        embeddings: 2D array of shape (n, dim)
        
    Returns:
        1D array of similarities, shape (n,)
    """
    # Normalize embeddings
    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
    embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Dot product gives cosine similarity
    similarities = np.dot(embeddings_norm, query_norm)
    
    return similarities


def extract_relevant_json(
    full_json: Dict[str, Any],
    relevant_paths: List[str],
    preserve_structure: bool = True
) -> Dict[str, Any]:
    """
    Extract relevant chunks from JSON with hierarchical context.
    
    NEW APPROACH: Instead of reconstructing exact structure, return relevant chunks
    with their hierarchical paths as context. This is simpler and more readable.
    
    Args:
        full_json: The full JSON object
        relevant_paths: List of paths to relevant chunks
        preserve_structure: If False, returns flat dict; if True, includes hierarchical context
        
    Returns:
        Dict of relevant chunks with hierarchical context
        
    Example:
        full_json = {"flow_execution_summary": {"execution_status": {...}, "stages": [...]}}
        relevant_paths = ["flow_execution_summary.execution_status"]
        
        Result: {
            "flow_execution_summary.execution_status": {
                "execute_successfully": True,
                "exit_code": 0,
                "exit_line": "..."
            }
        }
    """
    result = {}
    
    for path in relevant_paths:
        value = get_value_by_path(full_json, path)
        if value is not None:
            if preserve_structure:
                # Include hierarchical path as key for context
                result[path] = value
            else:
                # Extract just the final key name
                final_key = path.split(".")[-1]
                result[final_key] = value
    
    return result


def get_value_by_path(json_obj: Any, path: str) -> Any:
    """
    Get value from JSON by dot-separated path.
    
    Handles both real paths and synthetic chunk paths created during flattening.
    
    Args:
        json_obj: JSON object  
        path: Dot-separated path like "metrics.substages.4_1_cts.wns" 
              or synthetic chunk path like "flow_execution_summary.execution_status"
        
    Returns:
        Value at that path, or None if not found
        
    Note: For synthetic chunk paths (like "execution_status" which doesn't exist
    in original JSON), we need to rebuild them from the flattened representation.
    This is handled by the caller through proper flatten -> filter -> extract flow.
    """
    parts = path.split(".")
    current = json_obj
    
    for part in parts:
        # Handle array indices like "stages[0]"
        if "[" in part and "]" in part:
            key = part[:part.index("[")]
            idx = int(part[part.index("[")+1:part.index("]")])
            if isinstance(current, dict) and key in current:
                current = current[key]
                if isinstance(current, list) and 0 <= idx < len(current):
                    current = current[idx]
                else:
                    return None
            else:
                return None
        else:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
    
    return current


def set_value_by_path(json_obj: Dict, path: str, value: Any) -> None:
    """
    Set value in JSON by dot-separated path, creating nested dicts as needed.
    
    Args:
        json_obj: JSON object to modify (in-place)
        path: Dot-separated path
        value: Value to set
    """
    if value is None:
        return
    
    parts = path.split(".")
    current = json_obj
    
    for i, part in enumerate(parts[:-1]):
        if part not in current:
            current[part] = {}
        current = current[part]
    
    current[parts[-1]] = value


def filter_json_by_similarity(
    question: str,
    full_json: Dict[str, Any],
    embedding_model: EmbeddingModel,
    top_k: int = 15,
    min_similarity: float = 0.0,
    preserve_structure: bool = True
) -> Dict[str, Any]:
    """
    Filter JSON to only include chunks relevant to the question.
    
    NEW APPROACH: Returns chunks with hierarchical paths as keys (not reconstructed structure).
    This is simpler, more readable, and preserves semantic grouping.
    
    Args:
        question: The question being asked
        full_json: Full JSON context
        embedding_model: Initialized EmbeddingModel
        top_k: Number of top chunks to keep
        min_similarity: Minimum similarity threshold (0-1)
        preserve_structure: Whether to include hierarchical paths (True) or just final keys (False)
        
    Returns:
        Dict of relevant chunks with hierarchical context
        
    Example:
        question = "Was the flow execution successful?"
        full_json = {...huge JSON...}
        filtered = filter_json_by_similarity(question, full_json, model, top_k=10)
        
        Result: {
            "flow_execution_summary.execution_status": {
                "execute_successfully": True,
                "exit_code": 0,
                "exit_line": "..."
            },
            "flow_execution_summary.design_info": {
                "design_name": "aes",
                "technology_node": "nangate45"
            }
        }
    """
    # Step 1: Flatten JSON to semantic chunks
    json_chunks = flatten_json_to_paths(full_json, max_depth=10, include_values=True)
    
    if not json_chunks:
        return full_json
    
    # Step 2: Compute similarities
    scored_chunks = compute_path_similarities(question, json_chunks, embedding_model)
    
    # Step 3: Filter by top-k and min_similarity
    relevant_chunks = [
        (path, value, score) for path, value, score in scored_chunks[:top_k]
        if score >= min_similarity
    ]
    
    # Debug: print top chunks
    print(f"[Embedding Filter] Top {min(5, len(scored_chunks))} relevant chunks for question:")
    print(f"  Q: {question[:80]}...")
    for path, value, score in scored_chunks[:5]:
        print(f"    {score:.3f}: {path}")
    
    # Step 4: Build output directly from chunks
    filtered_result = {}
    for path, chunk_data, score in relevant_chunks:
        if preserve_structure:
            # Use full hierarchical path as key
            filtered_result[path] = chunk_data
        else:
            # Use just the final segment as key
            final_key = path.split(".")[-1]
            filtered_result[final_key] = chunk_data
    
    # Calculate size reduction
    original_size = len(json.dumps(full_json))
    filtered_size = len(json.dumps(filtered_result))
    reduction = (1 - filtered_size / original_size) * 100 if original_size > 0 else 0
    
    print(f"[Embedding Filter] Reduced JSON from {original_size} to {filtered_size} chars ({reduction:.1f}% reduction)")
    
    return filtered_result


# Global embedding model instance (lazy initialization)
_global_embedding_model = None


def get_embedding_model(model_type: str = "sentence-transformer") -> EmbeddingModel:
    """
    Get or create global embedding model instance.
    
    Args:
        model_type: Type of embedding model to use
        
    Returns:
        EmbeddingModel instance
    """
    global _global_embedding_model
    
    if _global_embedding_model is None:
        _global_embedding_model = EmbeddingModel(model_type)
    
    return _global_embedding_model
