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
Baseline RAG (Retrieval-Augmented Generation) method for log and RPT file analysis.

This module implements a baseline agentic RAG system that:
1. Creates a vector database from multiple log and RPT files
2. Uses semantic search to retrieve relevant context
3. Links retrieved context with LLM calls to answer questions

Features:
- Boundary-aware chunking: Respects paragraph, line, and sentence boundaries
- JSON-aware chunking: Preserves JSON structure when chunking structured data
- Special character handling: Properly handles Unicode and control characters
- Overlapping chunks: Maintains context with smart overlap at boundaries

Usage:
    from baseline_rag import BaselineRAG
    
    rag = BaselineRAG(log_files=["file1.log", "file2.log"], 
                      rpt_files=["file1.rpt", "file2.rpt"])
    rag.build_vector_db()
    answer = rag.query("What is the WNS after placement?")
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime
# import pdb

# Import existing utilities
try:
    from embedding_utils import EmbeddingModel
    from llm import llm_inference, check_token_limits, count_tokens, call_Perflab_claude37, call_NIM_openmodels
    from rpt_metric_extract import ReportMetricExtractor, extract_from_directory
except ImportError:
    # Try importing from src package
    from src.embedding_utils import EmbeddingModel
    from src.llm import llm_inference, check_token_limits, count_tokens, call_Perflab_claude37, call_NIM_openmodels
    from src.rpt_metric_extract import ReportMetricExtractor, extract_from_directory

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("[Warning] ChromaDB not available. Install with: pip install chromadb")


class BaselineRAG:
    """
    Baseline RAG system for log and RPT file analysis.
    
    Creates a vector database from log and RPT files and uses it for
    agentic question answering.
    """
    
    def __init__(
        self,
        log_files: List[str],
        rpt_files: Optional[List[str]] = None,
        rpt_directories: Optional[List[str]] = None,
        embedding_model_type: str = "sentence-transformer",
        lines_per_chunk: int = 250,
        lines_overlap: int = 25,
        max_lines_per_chunk: int = 500,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        vector_db_path: Optional[str] = None,
        collection_name: str = "log_rpt_collection"
    ):
        """
        Initialize the BaselineRAG system.
        
        Args:
            log_files: List of paths to log files (.log)
            rpt_files: Optional list of paths to RPT files (.rpt)
            rpt_directories: Optional list of directories containing RPT files
            embedding_model_type: Type of embedding model ("sentence-transformer", "openai", "keyword")
            lines_per_chunk: Target lines per chunk for raw log/RPT text (default 250)
            lines_overlap: Overlap in lines between consecutive raw-text chunks (default 25)
            max_lines_per_chunk: Hard cap on lines per chunk; no chunk exceeds this (default 500)
            chunk_size: Character chunk size for structured JSON (RPT metrics) only (default 1000)
            chunk_overlap: Character overlap for structured JSON chunks only (default 200)
            vector_db_path: Path to store vector database (None = in-memory)
            collection_name: Name of the ChromaDB collection
        """
        self.log_files = [Path(f) for f in log_files]
        self.rpt_files = [Path(f) for f in (rpt_files or [])]
        self.rpt_directories = [Path(d) for d in (rpt_directories or [])]
        
        # Validate files exist
        for log_file in self.log_files:
            if not log_file.exists():
                raise FileNotFoundError(f"Log file not found: {log_file}")
        
        for rpt_file in self.rpt_files:
            if not rpt_file.exists():
                raise FileNotFoundError(f"RPT file not found: {rpt_file}")
        
        # Initialize embedding model
        self.embedding_model = EmbeddingModel(model_type=embedding_model_type)
        
        # Chunking: line-based for raw log/RPT (default)
        self.lines_per_chunk = lines_per_chunk
        self.lines_overlap = min(lines_overlap, max(0, lines_per_chunk - 1))
        self.max_lines_per_chunk = max_lines_per_chunk
        # Character-based only for structured JSON (RPT metrics)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Vector database setup
        self.vector_db_path = vector_db_path
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self._chunks = []  # Store chunks with metadata
        
        # Initialize ChromaDB
        if CHROMADB_AVAILABLE:
            if vector_db_path:
                self.client = chromadb.PersistentClient(path=vector_db_path)
            else:
                self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        else:
            print("[Warning] ChromaDB not available. Using in-memory storage.")
            self._in_memory_store = {}
    
    def _find_boundary(self, text: str, start_pos: int, preferred_end: int, look_back: int = 100) -> int:
        """
        Find a good boundary point near the preferred end position.
        
        Looks for natural boundaries in this order:
        1. Double newlines (paragraph breaks)
        2. Single newlines (line breaks)
        3. Sentence endings (. ! ?)
        4. Common delimiters (; : ,)
        5. Whitespace
        
        Args:
            text: Full text
            start_pos: Start position of current chunk
            preferred_end: Preferred end position (chunk_size from start)
            look_back: How far back to look for boundary (default: 100 chars)
            
        Returns:
            Best boundary position, or preferred_end if no good boundary found
        """
        # Don't look back beyond start
        search_start = max(start_pos, preferred_end - look_back)
        search_end = min(len(text), preferred_end + 50)  # Also look a bit forward
        
        # Priority order: paragraph > line > sentence > delimiter > whitespace
        boundary_patterns = [
            (r'\n\n+', 0),           # Double newlines (paragraph break) - prefer end
            (r'\n', 0),               # Single newline (line break) - prefer end
            (r'[.!?]+\s+', 0),         # Sentence endings - prefer end
            (r'[;:]\s+', 0),          # Semicolon/colon - prefer end
            (r',\s+', 0),              # Comma - prefer end
            (r'\s+', 0),               # Any whitespace - prefer end
        ]
        
        # Search backwards from preferred_end
        for pattern, offset in boundary_patterns:
            matches = list(re.finditer(pattern, text[search_start:search_end]))
            if matches:
                # Find the match closest to preferred_end (but not beyond it)
                best_match = None
                best_distance = float('inf')
                
                for match in matches:
                    match_pos = search_start + match.end() + offset
                    # Prefer boundaries before preferred_end, but allow slight overflow
                    if match_pos <= preferred_end + 20:
                        distance = abs(match_pos - preferred_end)
                        if distance < best_distance:
                            best_distance = distance
                            best_match = match_pos
                
                if best_match and best_match > start_pos:
                    return best_match
        
        # Fallback: if no boundary found, try to find whitespace near preferred_end
        if preferred_end < len(text):
            # Look for whitespace within 50 chars
            for i in range(preferred_end, max(start_pos, preferred_end - 50), -1):
                if i < len(text) and text[i].isspace():
                    return i + 1
        
        # Last resort: use preferred_end
        return min(preferred_end, len(text))
    
    def _is_empty_or_whitespace(self, line: str) -> bool:
        """True if line is empty or only whitespace."""
        return not line.strip()

    def _is_timestamp_only(self, line: str) -> bool:
        """True if line is only a timestamp (e.g. [12:00:00], 2024-01-01 12:00:00, or similar)."""
        s = line.strip()
        if not s or len(s) > 50:
            return False
        # Only digits, colons, dashes, dots, spaces, brackets
        if not re.match(r"^[\d\[\]\-\:\.\s\,]+$", s):
            return False
        # Must look like a timestamp: has digits and at least one colon or dash
        return bool(re.search(r"\d", s)) and (":" in s or re.search(r"\d{4}\-\d{2}", s))

    def _is_separator_line(self, line: str) -> bool:
        """True if line is only dashes, equals, or whitespace (table/section separator)."""
        s = line.strip()
        return len(s) > 0 and all(c in "-=\t " for c in s)

    def _find_safe_chunk_end(
        self, lines: List[str], start: int, preferred_end: int
    ) -> int:
        """
        Find a line index (inclusive) to end the chunk at a safe boundary.
        Prefer empty lines or timestamp-only lines. Do not split inside a table
        (block from a separator line until the next empty line).
        """
        n = len(lines)
        if preferred_end >= n:
            return n - 1
        # Check if preferred_end is inside a "table" (after a separator, before next empty)
        in_table_start = None
        i = start
        while i <= min(preferred_end, n - 1):
            if self._is_separator_line(lines[i]):
                in_table_start = i
            elif self._is_empty_or_whitespace(lines[i]) and in_table_start is not None:
                in_table_start = None  # table ended
            i += 1
        # If we're inside a table, extend to the next empty line so we don't split the table
        if in_table_start is not None:
            j = preferred_end
            while j < n:
                if self._is_empty_or_whitespace(lines[j]):
                    return j
                j += 1
            return n - 1
        # Look backward from preferred_end for a safe boundary (empty or timestamp-only)
        for j in range(preferred_end, start - 1, -1):
            if j < n and (
                self._is_empty_or_whitespace(lines[j])
                or self._is_timestamp_only(lines[j])
            ):
                return j
        # Look forward for a safe boundary
        for j in range(preferred_end, n):
            if self._is_empty_or_whitespace(lines[j]) or self._is_timestamp_only(
                lines[j]
            ):
                return j
        return min(preferred_end, n - 1)

    def _chunk_text_by_lines(self, text: str, source: str, file_type: str) -> List[Dict[str, Any]]:
        """
        Split text into chunks at safe boundaries: empty lines or timestamp-only lines.
        Does not split in the middle of a table (separator-to-empty block).
        """
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        lines = text.split('\n')
        n = self.lines_per_chunk
        overlap = self.lines_overlap
        step = max(1, n - overlap)
        chunks = []
        start_line = 0
        while start_line < len(lines):
            preferred_end = min(start_line + n, len(lines)) - 1
            if preferred_end < start_line:
                preferred_end = start_line
            end_line = self._find_safe_chunk_end(lines, start_line, preferred_end)
            end_line = end_line + 1  # make exclusive for slicing
            end_line = min(end_line, len(lines), start_line + self.max_lines_per_chunk)
            chunk_lines = lines[start_line:end_line]
            chunk_text = '\n'.join(chunk_lines)
            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "source": str(source),
                    "file_type": file_type,
                    "start_pos": start_line,
                    "end_pos": end_line,
                    "chunk_length": len(chunk_text),
                    "start_line": start_line,
                    "end_line": end_line,
                    "line_count": len(chunk_lines),
                })
            # Next chunk: overlap by `overlap` lines, but do not go backwards
            chunk_start = start_line
            start_line = end_line - overlap
            if start_line <= chunk_start or overlap <= 0:
                start_line = end_line
            if start_line >= len(lines):
                break
        return chunks
    
    def _chunk_text(self, text: str, source: str, file_type: str) -> List[Dict[str, Any]]:
        """
        Split raw text into chunks by lines (lines_per_chunk lines per chunk, with lines_overlap).
        
        Args:
            text: Text to chunk
            source: Source file path
            file_type: Type of file ("log_raw" or "rpt_raw")
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        return self._chunk_text_by_lines(text, source, file_type)
    
    def _chunk_json_text(self, json_text: str, source: str, file_type: str) -> List[Dict[str, Any]]:
        """
        Chunk JSON text while respecting JSON structure boundaries.
        
        This method is optimized for JSON/text data and tries to:
        1. Keep complete JSON objects together
        2. Respect indentation levels
        3. Avoid splitting in the middle of keys/values
        
        Args:
            json_text: JSON-formatted text to chunk
            source: Source file path
            file_type: Type of file
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        start = 0
        min_chunk_size = max(100, self.chunk_size // 4)
        
        while start < len(json_text):
            preferred_end = start + self.chunk_size
            
            if preferred_end >= len(json_text):
                # Last chunk
                chunk_text = json_text[start:]
                if chunk_text.strip():
                    chunks.append({
                        "text": chunk_text,
                        "source": str(source),
                        "file_type": file_type,
                        "start_pos": start,
                        "end_pos": len(json_text),
                        "chunk_length": len(chunk_text)
                    })
                break
            
            # For JSON, look for better boundaries:
            # 1. End of complete JSON objects (closing brace with proper indentation)
            # 2. End of array elements
            # 3. End of key-value pairs
            # 4. Newlines with consistent indentation
            
            boundary = preferred_end
            
            # Look backwards for JSON structure boundaries
            search_start = max(start, preferred_end - 200)
            search_end = min(len(json_text), preferred_end + 100)
            search_text = json_text[search_start:search_end]
            
            # Pattern 1: End of JSON object (closing brace followed by comma or closing bracket)
            obj_end_match = re.search(r'\}\s*[,}\]]', search_text)
            if obj_end_match:
                candidate = search_start + obj_end_match.end()
                if start < candidate <= preferred_end + 50:
                    boundary = candidate
            
            # Pattern 2: End of array element (closing bracket followed by comma)
            array_end_match = re.search(r'\]\s*,', search_text)
            if array_end_match:
                candidate = search_start + array_end_match.end()
                if start < candidate <= preferred_end + 50 and abs(candidate - preferred_end) < abs(boundary - preferred_end):
                    boundary = candidate
            
            # Pattern 3: Complete key-value pair (ending with comma or closing brace)
            kv_end_match = re.search(r'"[^"]+"\s*:\s*[^,}\]]+[,}\]]', search_text)
            if kv_end_match:
                candidate = search_start + kv_end_match.end()
                if start < candidate <= preferred_end + 50 and abs(candidate - preferred_end) < abs(boundary - preferred_end):
                    boundary = candidate
            
            # Pattern 4: Newline with consistent indentation (for formatted JSON)
            # Find newline followed by same or less indentation
            if boundary == preferred_end:
                # Look for newline boundaries
                nl_match = re.search(r'\n\s{0,4}[}\]]', search_text)
                if nl_match:
                    candidate = search_start + nl_match.end()
                    if start < candidate <= preferred_end + 50:
                        boundary = candidate
            
            # Fallback to regular boundary finding
            if boundary == preferred_end:
                boundary = self._find_boundary(json_text, start, preferred_end)
            
            # Ensure minimum chunk size
            if boundary - start < min_chunk_size and boundary < len(json_text):
                extended_end = min(start + min_chunk_size, len(json_text))
                boundary = self._find_boundary(json_text, start, extended_end)
            
            chunk_text = json_text[start:boundary]
            
            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "source": str(source),
                    "file_type": file_type,
                    "start_pos": start,
                    "end_pos": boundary,
                    "chunk_length": len(chunk_text),
                    "is_json": True
                })
            
            # Calculate next start with overlap
            overlap_start = max(start + 1, boundary - self.chunk_overlap)
            if overlap_start < len(json_text):
                # Try to start overlap at a JSON boundary
                overlap_boundary = self._find_boundary(
                    json_text,
                    start,
                    overlap_start,
                    look_back=min(50, self.chunk_overlap)
                )
                if overlap_boundary > start and overlap_boundary < boundary:
                    start = overlap_boundary
                else:
                    start = overlap_start
            else:
                break
        
        return chunks
    
    def _process_log_file(self, log_file: Path) -> List[Dict[str, Any]]:
        """
        Process a log file as raw text only (no structured extraction).
        
        Args:
            log_file: Path to log file
            
        Returns:
            List of chunks from the raw log content
        """
        print(f"[RAG] Processing log file: {log_file}")
        print(f"[RAG] Raw chunk config: lines_per_chunk={self.lines_per_chunk}, lines_overlap={self.lines_overlap}, max_lines_per_chunk={self.max_lines_per_chunk}")

        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            raw_content = f.read()

        raw_chunks = self._chunk_text(raw_content, log_file, "log_raw")
        if raw_chunks:
            line_counts = [c.get("line_count", len(c["text"].splitlines())) for c in raw_chunks]
            char_lengths = [c.get("chunk_length", len(c["text"])) for c in raw_chunks]
            print(f"[RAG] Raw chunks: {len(raw_chunks)} chunks, lines per chunk min/avg/max = {min(line_counts)}/{sum(line_counts)//len(line_counts)}/{max(line_counts)}, chars min/avg/max = {min(char_lengths)}/{sum(char_lengths)//len(char_lengths)}/{max(char_lengths)}")
        # pdb.set_trace()
        return raw_chunks
    
    def _process_rpt_file(self, rpt_file: Path) -> List[Dict[str, Any]]:
        """
        Process an RPT file and extract metrics.
        
        Args:
            rpt_file: Path to RPT file
            
        Returns:
            List of chunks with extracted information
        """
        print(f"[RAG] Processing RPT file: {rpt_file}")
        
        chunks = []
        
        # Read raw RPT content
        with open(rpt_file, 'r', encoding='utf-8', errors='ignore') as f:
            raw_content = f.read()
        
        # Add raw RPT chunks
        raw_chunks = self._chunk_text(raw_content, rpt_file, "rpt_raw")
        chunks.extend(raw_chunks)
        
        # Extract structured metrics
        try:
            extractor = ReportMetricExtractor(str(rpt_file))
            metrics = extractor.extract_all()
            
            # Convert metrics to text
            metrics_text = json.dumps(metrics, indent=2)
            metrics_chunks = self._chunk_json_text(metrics_text, rpt_file, "rpt_metrics")
            
            # Add metadata
            for chunk in metrics_chunks:
                chunk["extraction_type"] = "metrics"
                chunk["stage"] = metrics.get("stage", "unknown")
            
            chunks.extend(metrics_chunks)
            
        except Exception as e:
            print(f"[RAG] Warning: Could not extract metrics from {rpt_file}: {e}")
        
        return chunks
    
    def _process_rpt_directory(self, rpt_dir: Path) -> List[Dict[str, Any]]:
        """
        Process all RPT files in a directory.
        
        Args:
            rpt_dir: Path to directory containing RPT files
            
        Returns:
            List of chunks from all RPT files
        """
        print(f"[RAG] Processing RPT directory: {rpt_dir}")
        
        chunks = []
        rpt_files = list(rpt_dir.glob("*.rpt"))
        
        for rpt_file in rpt_files:
            file_chunks = self._process_rpt_file(rpt_file)
            chunks.extend(file_chunks)
        
        return chunks
    
    def build_vector_db(self):
        """
        Build the vector database from all log and RPT files.
        """
        print("[RAG] Building vector database...")
        
        all_chunks = []
        
        # Process log files
        for log_file in self.log_files:
            chunks = self._process_log_file(log_file)
            all_chunks.extend(chunks)
        
        # Process individual RPT files
        for rpt_file in self.rpt_files:
            chunks = self._process_rpt_file(rpt_file)
            all_chunks.extend(chunks)
        
        # Process RPT directories
        for rpt_dir in self.rpt_directories:
            if rpt_dir.exists():
                chunks = self._process_rpt_directory(rpt_dir)
                all_chunks.extend(chunks)
        
        print(f"[RAG] Created {len(all_chunks)} chunks from {len(self.log_files)} log files and {len(self.rpt_files) + sum(len(list(d.glob('*.rpt'))) for d in self.rpt_directories if d.exists())} RPT files")
        
        # Store chunks
        self._chunks = all_chunks
        
        # Create embeddings and store in vector database
        if CHROMADB_AVAILABLE and self.client:
            self._build_chromadb(all_chunks)
        else:
            self._build_in_memory_db(all_chunks)
        
        print("[RAG] Vector database built successfully!")
    
    def _build_chromadb(self, chunks: List[Dict[str, Any]]):
        """Build ChromaDB vector database."""
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            print(f"[RAG] Using existing collection: {self.collection_name}")
        except:
            self.collection = self.client.create_collection(name=self.collection_name)
            print(f"[RAG] Created new collection: {self.collection_name}")
        
        # Prepare texts and metadata
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [
            {
                "source": chunk["source"],
                "file_type": chunk["file_type"],
                "start_pos": chunk["start_pos"],
                "end_pos": chunk["end_pos"],
                **{k: str(v) for k, v in chunk.items() if k not in ["text", "source", "file_type", "start_pos", "end_pos"]}
            }
            for chunk in chunks
        ]
        
        # Generate IDs
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        
        # Create embeddings
        print("[RAG] Generating embeddings...")
        embeddings = self.embedding_model.encode(texts)
        
        # Add to collection
        print("[RAG] Adding chunks to vector database...")
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"[RAG] Added {len(chunks)} chunks to vector database")
    
    def _build_in_memory_db(self, chunks: List[Dict[str, Any]]):
        """Build in-memory vector database (fallback)."""
        print("[RAG] Building in-memory vector database...")
        
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_model.encode(texts)
        
        self._in_memory_store = {
            "chunks": chunks,
            "embeddings": embeddings,
            "texts": texts
        }
        
        print(f"[RAG] Stored {len(chunks)} chunks in memory")
    
    def _retrieve_relevant_chunks(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks from vector database.
        
        Args:
            query: Query text
            top_k: Number of top chunks to retrieve
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of relevant chunks with similarity scores
        """
        # Encode query
        query_embedding = self.embedding_model.encode([query])[0]
        
        if CHROMADB_AVAILABLE and self.collection:
            # Use ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )
            
            # Format results
            retrieved_chunks = []
            if results["documents"] and len(results["documents"][0]) > 0:
                for i in range(len(results["documents"][0])):
                    chunk = {
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i] if "distances" in results else None
                    }
                    # Convert distance to similarity (ChromaDB uses distance, lower is better)
                    if chunk["distance"] is not None:
                        chunk["similarity"] = 1.0 / (1.0 + chunk["distance"])
                    else:
                        chunk["similarity"] = 1.0
                    
                    if chunk["similarity"] >= min_similarity:
                        retrieved_chunks.append(chunk)
            
            return retrieved_chunks
        
        else:
            # Use in-memory database
            embeddings = self._in_memory_store["embeddings"]
            
            # Compute cosine similarities
            query_norm = np.linalg.norm(query_embedding)
            similarities = []
            
            for i, emb in enumerate(embeddings):
                emb_norm = np.linalg.norm(emb)
                if emb_norm > 0 and query_norm > 0:
                    similarity = np.dot(query_embedding, emb) / (query_norm * emb_norm)
                else:
                    similarity = 0.0
                similarities.append((i, similarity))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Retrieve top-k
            retrieved_chunks = []
            for idx, similarity in similarities[:top_k]:
                if similarity >= min_similarity:
                    chunk = self._in_memory_store["chunks"][idx].copy()
                    chunk["similarity"] = float(similarity)
                    retrieved_chunks.append({
                        "text": chunk["text"],
                        "metadata": {k: v for k, v in chunk.items() if k != "text"},
                        "similarity": float(similarity)
                    })
            
            return retrieved_chunks
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        min_similarity: float = 0.0,
        temperature: float = 0.1,
        model_source: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Answer a question using RAG.
        
        Args:
            question: Question to answer
            top_k: Number of relevant chunks to retrieve
            min_similarity: Minimum similarity threshold for chunks
            temperature: LLM temperature
            model_source: LLM source ("Mark" or "Perflab"), None uses default from FLAGS
            
        Returns:
            Dictionary with answer and retrieved context
        """
        print(f"[RAG] Querying: {question}")
        
        # Retrieve relevant chunks
        retrieved_chunks = self._retrieve_relevant_chunks(
            question,
            top_k=top_k,
            min_similarity=min_similarity
        )
        # print(f"Retrieved chunks: {retrieved_chunks}")
        if not retrieved_chunks:
            return {
                "answer": "No relevant information found in the database.",
                "retrieved_chunks": [],
                "num_chunks": 0,
                "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "system_tokens": 0, "user_tokens": 0},
            }
        
        print(f"[RAG] Retrieved {len(retrieved_chunks)} relevant chunks")
        
        # Build context from retrieved chunks
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            source = chunk["metadata"].get("source", "unknown")
            file_type = chunk["metadata"].get("file_type", "unknown")
            similarity = chunk.get("similarity", 0.0)
            
            context_parts.append(
                f"[Context {i}] (Source: {source}, Type: {file_type}, Similarity: {similarity:.3f})\n"
                f"{chunk['text']}\n"
            )
        
        context = "\n".join(context_parts)
        
        # Build prompts
        system_prompt = """You are an expert assistant for analyzing EDA (Electronic Design Automation) log files and RPT (report) files.

Your task is to answer questions about design metrics, flow execution, and timing information based on the provided context from log and RPT files.

Guidelines:
- Answer based ONLY on the provided context
- If the information is not in the context, say so clearly
- Be precise with numbers and metrics
- Reference the source file when possible
- If multiple values exist, mention all relevant ones"""
        
        user_prompt = f"""Question: {question}

Context from log and RPT files:
{context}

Please answer the question based on the provided context."""
        
        # Token usage: input side (before LLM call)
        try:
            from config_FLAGS import FLAGS
        except ImportError:
            from src.config_FLAGS import FLAGS
        model = getattr(FLAGS, "model", "gpt-4")
        token_info = check_token_limits(system_prompt, user_prompt, model)
        prompt_tokens = token_info["total_tokens"]
        system_tokens = token_info["system_tokens"]
        user_tokens = token_info["user_tokens"]

        # Call LLM
        try:
            model_source = model_source or FLAGS.model_source
            # answer = call_Perflab_claude37(
            answer = call_NIM_openmodels(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature
            )
        except Exception as e:
            print(f"[RAG] Error calling LLM: {e}")
            answer = f"Error generating answer: {str(e)}"

        # Token usage: completion (estimated via count_tokens)
        completion_tokens = count_tokens(answer, model)
        total_tokens = prompt_tokens + completion_tokens
        token_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "system_tokens": system_tokens,
            "user_tokens": user_tokens,
        }
        
        return {
            "answer": answer,
            "question": question,
            "retrieved_chunks": [
                {
                    "text": chunk["text"][:500] + "..." if len(chunk["text"]) > 500 else chunk["text"],
                    "source": chunk["metadata"].get("source", "unknown"),
                    "file_type": chunk["metadata"].get("file_type", "unknown"),
                    "similarity": chunk.get("similarity", 0.0)
                }
                for chunk in retrieved_chunks
            ],
            "num_chunks": len(retrieved_chunks),
            "token_usage": token_usage,
        }
    
    def clear_db(self):
        """Clear the vector database."""
        if CHROMADB_AVAILABLE and self.client:
            try:
                self.client.delete_collection(name=self.collection_name)
                print(f"[RAG] Deleted collection: {self.collection_name}")
            except:
                pass
        
        self._chunks = []
        self._in_memory_store = {}
        print("[RAG] Database cleared")


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Baseline RAG for log and RPT files")
    parser.add_argument("--log-files", nargs="+", required=True, help="Log files to process")
    parser.add_argument("--rpt-files", nargs="+", help="RPT files to process")
    parser.add_argument("--rpt-dirs", nargs="+", help="Directories containing RPT files")
    parser.add_argument("--query", help="Question to answer")
    parser.add_argument("--vector-db-path", help="Path to store vector database")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--lines-per-chunk", type=int, default=250, help="Target lines per chunk (default 250)")
    parser.add_argument("--lines-overlap", type=int, default=25, help="Line overlap between chunks (default 25)")
    parser.add_argument("--max-lines-per-chunk", type=int, default=500, help="Max lines per chunk cap (default 500)")
    
    args = parser.parse_args()
    
    # Initialize RAG (line-based chunking by default)
    rag = BaselineRAG(
        log_files=args.log_files,
        rpt_files=args.rpt_files or [],
        rpt_directories=args.rpt_dirs or [],
        vector_db_path=args.vector_db_path,
        lines_per_chunk=args.lines_per_chunk,
        lines_overlap=args.lines_overlap,
        max_lines_per_chunk=args.max_lines_per_chunk,
    )
    
    # Build vector database
    rag.build_vector_db()
    
    # Query if provided
    if args.query:
        result = rag.query(args.query, top_k=args.top_k)
        print("\n" + "="*80)
        print("QUESTION:", result["question"])
        print("="*80)
        print("\nANSWER:")
        print(result["answer"])
        print("\n" + "="*80)
        print(f"Retrieved {result['num_chunks']} chunks:")
        for i, chunk in enumerate(result["retrieved_chunks"], 1):
            print(f"\n[{i}] Source: {chunk['source']}")
            print(f"    Type: {chunk['file_type']}")
            print(f"    Similarity: {chunk['similarity']:.3f}")
            print(f"    Preview: {chunk['text'][:200]}...")


if __name__ == "__main__":
    main()
