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
RAG Agent: an agent that uses Baseline RAG as a tool to answer questions.

Given an input prompt (question), the agent collects information from the RAG
vector database and returns an answer grounded in the retrieved context.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from llm import call_Perflab_claude37, check_token_limits, count_tokens, call_NIM_openmodels

try:
    from baseline_rag import BaselineRAG
except ImportError:
    from src.baseline_rag import BaselineRAG

try:
    from config_FLAGS import FLAGS
except ImportError:
    from src.config_FLAGS import FLAGS


class RAGAgent:
    """
    Agent that answers questions using a Baseline RAG database as a tool.

    The agent holds a RAG instance and exposes:
    - answer(prompt): answer a question by querying the RAG and returning the result
    - rag_query(question, ...): the RAG tool call (retrieve + generate)
    - retrieve(question, top_k): retrieve-only tool (no LLM answer)
    """

    def __init__(
        self,
        rag: Optional[BaselineRAG] = None,
        log_files: Optional[List[Union[str, Path]]] = None,
        rpt_files: Optional[List[Union[str, Path]]] = None,
        rpt_directories: Optional[List[Union[str, Path]]] = None,
        vector_db_path: Optional[str] = None,
        build_db: bool = True,
        **rag_kwargs: Any,
    ):
        """
        Initialize the RAG agent.

        Either pass an existing BaselineRAG instance, or provide file paths to
        create one. If paths are given and build_db is True, the vector DB
        is built on init.

        Args:
            rag: Existing BaselineRAG instance. If provided, log_files/rpt_*
                   are ignored.
            log_files: Paths to log files (used only if rag is None).
            rpt_files: Paths to RPT files (used only if rag is None).
            rpt_directories: Directories containing RPT files (used only if rag is None).
            vector_db_path: Path for persistent vector DB (used only if rag is None).
            build_db: If True and rag is created from paths, call build_vector_db().
            **rag_kwargs: Extra arguments for BaselineRAG (chunk_size, embedding_model_type, etc.).
        """
        if rag is not None:
            self._rag = rag
        else:
            if not log_files:
                raise ValueError("Either provide rag or log_files (and optionally rpt_files/rpt_directories).")
            self._rag = BaselineRAG(
                log_files=[str(p) for p in log_files],
                rpt_files=[str(p) for p in (rpt_files or [])],
                rpt_directories=[str(d) for d in (rpt_directories or [])],
                vector_db_path=vector_db_path,
                **rag_kwargs,
            )
            if build_db:
                self._rag.build_vector_db()
        self._last_token_usage: Optional[Dict[str, Any]] = None
        self._last_keyword_token_usage: Optional[Dict[str, Any]] = None

    def _generate_keywords(
        self,
        question: str,
        temperature: float = 0.1,
        model_source: Optional[str] = None,
    ) -> tuple:
        """
        Call the LLM to generate search keywords from the question for better retrieval.
        Returns (keywords_string, token_usage).
        """
        system_prompt = """You are a helper for EDA (Electronic Design Automation) log and RPT report search.
Given a user question, output a short list of search keywords that would help find relevant passages.
Include: metric names (e.g. WNS, TNS, area, slack), stage names (e.g. placement, routing, synthesis), and technical terms.
Output only the keywords, comma or space separated, no explanation."""
        user_prompt = f"Question: {question}\n\nKeywords:"
        model = "claude-3-7-sonnet"
        token_info = check_token_limits(system_prompt, user_prompt, model)
        prompt_tokens = token_info["total_tokens"]
        system_tokens = token_info["system_tokens"]
        user_tokens = token_info["user_tokens"]
        try:
            # response = call_Perflab_claude37(
            response = call_NIM_openmodels(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
            )
        except Exception as e:
            response = ""
        keywords = (response or "").strip()
        completion_tokens = count_tokens(keywords, model)
        total_tokens = prompt_tokens + completion_tokens
        token_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "system_tokens": system_tokens,
            "user_tokens": user_tokens,
        }
        return keywords, token_usage

    @property
    def rag(self) -> BaselineRAG:
        """Access the underlying Baseline RAG instance."""
        return self._rag

    def rag_query(
        self,
        question: str,
        top_k: int = 3,
        min_similarity: float = 0.0,
        temperature: float = 0.1,
        model_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        RAG tool: retrieve relevant chunks and generate an answer.

        Use this as the agent's main tool for answering questions from the
        RAG database.

        Args:
            question: Question to answer.
            top_k: Number of chunks to retrieve.
            min_similarity: Minimum similarity for retrieved chunks.
            temperature: LLM temperature for generation.
            model_source: LLM source (e.g. "Mark", "Perflab"); None uses default.

        Returns:
            Dict with keys: answer, question, retrieved_chunks, num_chunks, token_usage,
            keyword_query (expanded query used for retrieval), keyword_token_usage.
            token_usage is the combined total (keyword generation + RAG answer); keyword_token_usage is from the keyword LLM call only.
        """
        # Call the LLM to generate keywords for the question; record its token usage
        keywords, keyword_token_usage = self._generate_keywords(
            question=question,
            temperature=temperature,
            model_source=model_source,
        )
        self._last_keyword_token_usage = keyword_token_usage
        # Expand query for retrieval: original question + keywords
        retrieval_query = f"{question} {keywords}".strip() if keywords else question

        result = self._rag.query(
            question=retrieval_query,
            top_k=top_k,
            min_similarity=min_similarity,
            temperature=temperature,
            model_source=model_source,
        )
        # print(f"RAG agent query result: {result}")
        # import pdb; pdb.set_trace()
        # Keep the original question in the result for display
        result["question"] = question
        result["keyword_query"] = retrieval_query
        result["keyword_token_usage"] = keyword_token_usage
        # Combine token usage: keyword generation + RAG answer
        rag_usage = result.get("token_usage") or {}
        combined = {
            "prompt_tokens": keyword_token_usage.get("prompt_tokens", 0) + rag_usage.get("prompt_tokens", 0),
            "completion_tokens": keyword_token_usage.get("completion_tokens", 0) + rag_usage.get("completion_tokens", 0),
            "total_tokens": keyword_token_usage.get("total_tokens", 0) + rag_usage.get("total_tokens", 0),
            "system_tokens": keyword_token_usage.get("system_tokens", 0) + rag_usage.get("system_tokens", 0),
            "user_tokens": keyword_token_usage.get("user_tokens", 0) + rag_usage.get("user_tokens", 0),
        }
        result["token_usage"] = combined
        self._last_token_usage = combined
        return result

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        min_similarity: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve-only tool: get relevant chunks without generating an answer.

        Useful when the agent only needs to gather context.

        Args:
            question: Query text.
            top_k: Number of chunks to retrieve.
            min_similarity: Minimum similarity threshold.

        Returns:
            List of chunk dicts with text, metadata, and similarity.
        """
        return self._rag._retrieve_relevant_chunks(
            question,
            top_k=top_k,
            min_similarity=min_similarity,
        )

    def answer(
        self,
        prompt: str,
        top_k: int = 5,
        min_similarity: float = 0.0,
        temperature: float = 0.1,
        model_source: Optional[str] = None,
        return_full_result: bool = False,
    ) -> Union[str, Dict[str, Any]]:
        """
        Answer a question using the RAG database (main entry point).

        Given an input prompt (question), collects information from the RAG
        database and returns the generated answer.

        Args:
            prompt: The question or prompt to answer.
            top_k: Number of relevant chunks to retrieve.
            min_similarity: Minimum similarity for chunks.
            temperature: LLM temperature.
            model_source: LLM source; None uses default.
            return_full_result: If True, return the full RAG result dict;
                               if False, return only the answer string.

        Returns:
            Answer string, or full result dict if return_full_result is True.
            The full result includes token_usage from the LLM call. You can also
            read agent.last_token_usage after any answer() or rag_query() call.
        """
        result = self.rag_query(
            question=prompt,
            top_k=top_k,
            min_similarity=min_similarity,
            temperature=temperature,
            model_source=model_source,
        )
        if return_full_result:
            return result
        return result.get("answer", "")

    @property
    def last_token_usage(self) -> Optional[Dict[str, Any]]:
        """
        Combined token usage from the most recent rag_query (keyword generation + RAG answer).

        None if no call has been made yet or the call did not return token_usage.
        Otherwise a dict with: prompt_tokens, completion_tokens, total_tokens,
        system_tokens, user_tokens.
        """
        return self._last_token_usage

    @property
    def last_keyword_token_usage(self) -> Optional[Dict[str, Any]]:
        """
        Token usage from the most recent keyword-generation LLM call only.

        None if no rag_query has been made yet. Same shape as last_token_usage.
        """
        return self._last_keyword_token_usage

    def get_tools(self) -> Dict[str, Any]:
        """
        Return a description of the agent's tools for use by orchestrators.

        Returns:
            Dict mapping tool names to short descriptions and callables.
        """
        return {
            "rag_query": {
                "description": "Query the RAG database with a question; returns an answer and retrieved context.",
                "callable": self.rag_query,
            },
            "retrieve": {
                "description": "Retrieve relevant chunks from the RAG database for a query (no LLM answer).",
                "callable": self.retrieve,
            },
        }


def main():
    """Example: create agent from paths and answer a question."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="RAG Agent: answer questions using Baseline RAG")
    parser.add_argument("--log-files", nargs="+", required=True, help="Log files for RAG")
    parser.add_argument("--rpt-files", nargs="*", default=[], help="RPT files")
    parser.add_argument("--rpt-dirs", nargs="*", default=[], help="Directories with RPT files")
    parser.add_argument("--query", required=True, help="Question to answer")
    parser.add_argument("--vector-db-path", help="Path for persistent vector DB")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--full", action="store_true", help="Print full result (answer + chunks)")

    args = parser.parse_args()

    agent = RAGAgent(
        log_files=args.log_files,
        rpt_files=args.rpt_files or None,
        rpt_directories=args.rpt_dirs or None,
        vector_db_path=args.vector_db_path,
        build_db=True,
    )

    if args.full:
        result = agent.answer(args.query, top_k=args.top_k, return_full_result=True)
        print("Question:", result.get("question"))
        print("Answer:", result.get("answer"))
        print("Retrieved chunks:", result.get("num_chunks"))
        if result.get("token_usage"):
            u = result["token_usage"]
            print("Token usage:", u.get("total_tokens"), "total (prompt:", u.get("prompt_tokens"), ", completion:", u.get("completion_tokens"), ")")
        for i, c in enumerate(result.get("retrieved_chunks", []), 1):
            print(f"  [{i}] {c.get('source')} (sim={c.get('similarity', 0):.3f})")
    else:
        answer = agent.answer(args.query, top_k=args.top_k)
        print(answer)
        if agent.last_token_usage:
            u = agent.last_token_usage
            print("Tokens:", u.get("total_tokens"), "(prompt:", u.get("prompt_tokens"), ", completion:", u.get("completion_tokens"), ")", file=sys.stderr)


if __name__ == "__main__":
    main()
