# EDAFlowQA Benchmark

A benchmark for evaluating LLMs on Electronic Design Automation (EDA) log file understanding. The benchmark provides a complete pipeline from raw OpenROAD flow logs to structured QA generation using Q-Tree sampling, and LLM evaluation with Pass@K metrics.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Repository Structure](#repository-structure)
- [Pipeline Overview](#pipeline-overview)
- [Step 1: Log Extraction](#step-1-log-extraction)
- [Step 2: QA Generation (Q-Tree Sampling)](#step-2-qa-generation-q-tree-sampling)
- [Step 3: LLM Evaluation](#step-3-llm-evaluation)
- [Step 4: Pass@K Metrics Calculation](#step-4-passk-metrics-calculation)
- [Benchmark Data](#benchmark-data)
- [Environment Variables](#environment-variables)
- [License](#license)

## Overview

EDAFlowQA evaluates LLMs' ability to understand and reason about EDA tool logs. The pipeline consists of four stages:

1. **Log Extraction** — Parse raw OpenROAD flow logs into structured per-stage metrics JSON
2. **QA Generation** — Use Q-Tree sampling to generate diverse question-answer pairs from the structured metrics
3. **LLM Evaluation** — Test LLMs (via API, Cursor agent, or Claude Code CLI) against the generated QA pairs
4. **Scoring** — Calculate Pass@K metrics to measure LLM accuracy

## Prerequisites

- Python 3.10+
- OpenAI-compatible API endpoint (for QA generation and evaluation)
- (Optional) [Cursor CLI](https://www.cursor.com/) or [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) for agent-based evaluation

### Python Dependencies

Install dependencies:

```bash
pip install -r requirements.txt
```

Key packages:

| Package | Version | Purpose |
|---|---|---|
| openai | >= 2.30.0 | LLM API calls (OpenAI-compatible) |
| tiktoken | >= 0.7.0 | Token counting for prompt length estimation |
| numpy | >= 2.0.0 | Numerical operations, metrics computation |
| scikit-learn | >= 1.9.0 | Embedding similarity for QA filtering |
| tenacity | >= 9.0.0 | Retry logic for robust API calls |
| httpx | >= 0.28.0 | HTTP client for API communication |
| tqdm | >= 4.60.0 | Progress bars |

## Repository Structure

```
.
├── README.md
├── LICENSE                          # Apache 2.0
├── SECURITY.md
├── requirements.txt
├── OAI_CONFIG_LIST_EXAMPLE          # Example OpenAI API config
├── run_qa_gen.sh                    # Script: single-design QA generation
├── run_qa_multifiles_gen.sh         # Script: cross-design QA generation
├── run_pass_at_k_single.sh          # Script: single-design Pass@K evaluation
├── run_pass_at_k_multifile.sh       # Script: multifile Pass@K evaluation
│
├── design_artifacts/                # Raw OpenROAD flow log and report files
│   ├── aes_asap7/
│   │   ├── aes_asap7.txt            # Flow log
│   │   ├── 2_floorplan_final.rpt    # Stage report files
│   │   └── ...
│   ├── aes_nangate45/
│   ├── ibex_sky130hd/
│   └── ...                          # 12 single-design + 6 cross-design comparison dirs
│
├── extracted_design_stage_metrics/   # Extracted metrics (per design)
│   ├── test_output_aes_asap7/
│   │   ├── aes_asap7_all_stages_metrics.json   # Aggregate metrics (input for QA generation)
│   │   ├── aes_asap7_<stage>_extract.txt       # Per-stage extracted log text
│   │   ├── aes_asap7_<stage>_metrics.json      # Per-stage parsed metrics
│   │   └── _extraction_summary.txt
│   └── ...                          # 12 design directories
│
├── qa_release/                      # Benchmark QA pairs (for evaluation) in Paper
│   ├── aes_asap7/                   # Single-design QAs
│   │   └── aes_*_qa_all_types.json
│   ├── ...                          # 12 design directories
│   └── multifile_comparison/        # Cross-design comparison QAs
│       ├── aes_asap7_vs_aes_nangate45/
│       │   └── *_qa_all_multifile.json
│       └── ...                      # 6 design-pair directories
│
└── src/                             # Source code
    ├── extract_all_stages.py        # Step 1: Log extraction pipeline
    ├── qa_generation.py             # Step 2: Q-Tree QA generation (single-design)
    ├── qa_generation_multifiles.py  # Step 2: Q-Tree QA generation (cross-design)
    ├── test_llm_qa.py               # Step 3: Single-design LLM evaluation
    ├── test_llm_qa_multifile.py     # Step 3: Multifile LLM evaluation
    ├── pass_k_metric_calculation.py # Step 4: Pass@K metric computation
    │
    ├── llm.py                       # LLM API backbone (OpenAI-compatible)
    ├── config_FLAGS.py              # Global configuration flags
    ├── baseline_rag.py              # Baseline RAG system
    ├── rag_agent.py                 # RAG agent for evaluation
    ├── embedding_utils.py           # Embedding similarity filtering
    ├── filter_invalid_answers.py    # QA answer validation/filtering
    ├── qa_stats_summary.py          # QA statistics and analysis
    │
    ├── experiment_direct_vs_qtree_qa.py   # Ablation: direct vs Q-Tree QA generation
    ├── score_experiment_results.py        # Scoring for ablation experiments
    │
    ├── syn_stage.py                 # Synthesis log parser
    ├── flr_stg.py                   # Floorplan log parser
    ├── place_stg.py                 # Placement log parser
    ├── cts_stg.py                   # CTS log parser
    ├── route_stg.py                 # Routing log parser
    ├── final_stg.py                 # Final stage log parser
    ├── synthesis_metrics_parser.py   # Synthesis metrics extractor
    ├── floorplan_metrics_parser.py   # Floorplan metrics extractor
    ├── placement_metrics_parser.py   # Placement metrics extractor
    ├── cts_metrics_parser.py         # CTS metrics extractor
    ├── route_metrics_parser.py       # Routing metrics extractor
    ├── final_metrics_parser.py       # Final stage metrics extractor
    ├── rpt_metric_extract.py         # Report (.rpt) metrics extractor
    │
    ├── load_design_metrics.py       # Utility: load metrics for a single design
    └── compare_metrics_test.py      # Utility: compare metrics between two designs
```

## Pipeline Overview

```
Raw EDA Log (.txt)
       │
       ▼
 ┌─────────────────────┐
 │  extract_all_stages  │  Step 1: Parse log → structured metrics JSON
 └─────────┬───────────┘
           │
           ▼
  *_all_stages_metrics.json
           │
           ▼
 ┌─────────────────────┐
 │   qa_generation.py   │  Step 2: Q-Tree sampling → QA pairs
 └─────────┬───────────┘
           │
           ▼
  *_qa_all_types.json  (question + golden answer)
           │
           ▼
 ┌─────────────────────┐
 │   test_llm_qa.py     │  Step 3: LLM evaluation (Cursor / Claude Code / RAG / API)
 └─────────┬───────────┘
           │
           ▼
  *_evaluation.json  (LLM responses)
           │
           ▼
 ┌──────────────────────────┐
 │ pass_k_metric_calculation │  Step 4: Pass@K scoring
 └──────────────────────────┘
```

## Step 1: Log Extraction

Parse raw OpenROAD flow logs into structured per-stage metrics.

```bash
cd src

# Extract all stages from a single log file
python extract_all_stages.py ../design_artifacts/aes_asap7/aes_asap7.txt -o ../test_output_aes_asap7

# Process multiple log files
python extract_all_stages.py ../design_artifacts/aes_asap7/aes_asap7.txt \
                              ../design_artifacts/ibex_sky130hd/ibex_sky130hd.txt \
                              -o ../extracted_logs
```

**Output**: `<design>_all_stages_metrics.json` containing structured metrics for all EDA stages (synthesis, floorplan, placement, CTS, routing, final).

## Step 2: QA Generation (Q-Tree Sampling)

Generate question-answer pairs from the extracted metrics using LLM-based Q-Tree sampling.

### Single-Design QA

```bash
cd src

# Generate all QA types (flow execution + single-stage + multi-stage + stage-to-final)
python qa_generation.py -json ../test_output_aes_asap7/aes_asap7_all_stages_metrics.json \
                        -n 10 --seed 42 -o ../qa_output/aes_asap7/

# Single-stage only
python qa_generation.py -json ../test_output_aes_asap7/aes_asap7_all_stages_metrics.json \
                        -n 10 --mode single -o ../qa_output/aes_asap7/

# Multi-stage only
python qa_generation.py -json ../test_output_aes_asap7/aes_asap7_all_stages_metrics.json \
                        -n 3 --mode multi -o ../qa_output/aes_asap7/
```

### Cross-Design (Multifile) QA

```bash
cd src

python qa_generation_multifiles.py \
    -json1 ../test_output_aes_asap7/aes_asap7_all_stages_metrics.json \
    -json2 ../test_output_aes_nangate45/aes_nangate45_all_stages_metrics.json \
    -o ../qa_output/multifile_comparison/aes_asap7_vs_aes_nangate45/
```

### QA Types Generated

| Type | Description | Example Question |
|---|---|---|
| `flow_execution` | High-level flow status, stages, errors | "What is the execution status of the design flow?" |
| `single_stage` | Individual stage metrics | "What is the WNS after CTS optimization?" |
| `multi_stage` | Compare metrics across 2-3 stages | "How does HPWL change from placement to routing?" |
| `stage_to_final` | Correlation between intermediate and final stages | "How does WNS evolve from floorplan to final?" |

## Step 3: LLM Evaluation

Evaluate LLMs against the benchmark QA pairs. Three agent types are supported:

### Using Cursor Agent

```bash
cd src

# Run evaluation
python test_llm_qa.py prepare \
    --qa_file ../qa_release/aes_asap7/aes_20260201_154729_qa_all_types.json \
    --model claude-4.5-sonnet \
    --agent_type cursor

# Run pass@k (multiple passes)
python test_llm_qa.py pass_at_k \
    --qa_file ../qa_release/aes_asap7/aes_20260201_154729_qa_all_types.json \
    --model claude-4.5-sonnet \
    --num_passes 10 \
    --agent_type cursor
```

### Using Claude Code CLI

```bash
cd src

python test_llm_qa.py prepare \
    --qa_file ../qa_release/aes_asap7/aes_20260201_154729_qa_all_types.json \
    --model claude-4.5-sonnet \
    --agent_type claude_code
```

### Using RAG Agent

```bash
cd src

python test_llm_qa.py prepare \
    --qa_file ../qa_release/aes_asap7/aes_20260201_154729_qa_all_types.json \
    --model gpt-4o \
    --agent_type rag
```

### Multifile Evaluation

```bash
cd src

python test_llm_qa_multifile.py prepare \
    --qa_file ../qa_release/multifile_comparison/aes_asap7_vs_aes_nangate45/aes_asap7_vs_aes_nangate45_*_qa_all_multifile.json \
    --model claude-4.5-sonnet \
    --agent_type cursor
```

### Evaluation CLI Commands

| Command | Description |
|---|---|
| `prepare` | Prepare prompts and run LLM inference |
| `test` | Same as `prepare` |
| `pass_at_k` | Run multiple passes for Pass@K evaluation |
| `calc_pass_at_k` | Calculate Pass@K metrics from saved results |
| `list` | List available QA files |
| `demo` | Run demo with sample data |

### Optional Arguments

| Argument | Description | Default |
|---|---|---|
| `--qa_file` | Path to QA JSON file | (required) |
| `--model` | Model name (e.g., `gpt-4o`, `claude-4.5-sonnet`) | (required) |
| `--qidxs` | Comma-separated question indices (e.g., `0,1,2`) | All |
| `--output_dir` | Output directory | `./` |
| `--num_passes` | Number of passes for Pass@K | 10 |
| `--use_json_hints` | Include output structure hints in prompts | False |
| `--agent_type` | Agent type: `cursor`, `claude_code`, or `rag` | `cursor` |

## Step 4: Pass@K Metrics Calculation

Calculate Pass@K metrics from evaluation results.

```bash
cd src

# Calculate pass@1 from results
python test_llm_qa.py calc_pass_at_k \
    --results_file ../pass_at_k_aes_asap7_claude-4.5-sonnet_10passes_*.json

# Calculate pass@5
python test_llm_qa.py calc_pass_at_k \
    --results_file ../pass_at_k_aes_asap7_claude-4.5-sonnet_10passes_*.json \
    --k 5
```

## Benchmark Data

### Summary

| Category | Designs | Unique QA Pairs |
|---|---|---|
| Single-design | 12 | 909 |
| Multifile comparison | 6 pairs | 227 |
| **Total** | | **1,136** |

### Single-Design QA Counts (qa_release/)

| Design | Technology | QA Pairs |
|---|---|---|
| aes_asap7 | ASAP7 | 80 |
| aes_block_asap7 | ASAP7 | 109 |
| aes_nangate45 | NanGate45 | 74 |
| aes_sky130hd | SKY130 HD | 75 |
| aes_sky130hs | SKY130 HS | 69 |
| ariane133_nangate45 | NanGate45 | 72 |
| ariane136_nangate45 | NanGate45 | 72 |
| bp_multi_top_nangate45 | NanGate45 | 60 |
| gcdsky130 | SKY130 | 61 |
| ibex_sky130hd | SKY130 HD | 73 |
| riscv32i-mock-sram_asap7 | ASAP7 | 83 |
| uartgf180 | GF180 | 81 |

### Multifile Comparison QA Counts (qa_release/)

| Design Pair | QA Pairs |
|---|---|
| aes_asap7 vs aes_nangate45 | 30 |
| aes_asap7 vs aes_sky130hd | 43 |
| aes_asap7 vs aes_sky130hs | 47 |
| aes_nangate45 vs aes_sky130hd | 38 |
| aes_nangate45 vs aes_sky130hs | 33 |
| aes_sky130hs vs aes_sky130hd | 36 |

### QA JSON Format

Each QA file in `qa_release/` uses a flat structure:

```json
{
  "design_name": "aes",
  "timestamp": "20260201_154729",
  "log_file": "./design_artifacts/aes_asap7/aes_asap7.txt",
  "source": "test_output_aes_asap7/aes_asap7_all_stages_metrics.json",
  "statistics": {
    "flow_execution": 1,
    "single_stage": 10,
    "multi_stage": 3,
    "stage_to_final": 2,
    "total_qa_pairs": 80
  },
  "qa_pairs": [
    {
      "id": "Q_FLOW_001_P01",
      "type": "flow_execution",
      "question": "<Q1> What is the execution status ...?\nDifficulty: easy",
      "answer": "Execution completed successfully."
    },
    {
      "id": "Q_SINGLE_002_P01",
      "type": "single_stage",
      "question": "<Q1> What is the WNS after CTS?\nDifficulty: medium",
      "answer": "-0.043 ns"
    }
  ]
}
```

## Environment Variables

| Variable | Description | Required For |
|---|---|---|
| `LLM_CONFIG_FILE` | Path to model config JSON (AutoGen-style, see below) | QA generation, API-based evaluation |
| `OPENAI_API_KEY` | OpenAI API key (alternative to config file) | QA generation, API-based evaluation |
| `OPENAI_BASE_URL` | Custom OpenAI-compatible API base URL | Custom LLM endpoints |
| `CURSOR_AGENT_PATH` | Path to Cursor agent CLI binary | Cursor agent evaluation |
| `CLAUDE_CODE_PATH` | Path to Claude Code CLI binary | Claude Code evaluation |

### LLM Configuration File

The `LLM_CONFIG_FILE` environment variable points to a JSON file specifying the LLM endpoint. The format follows AutoGen's `OAI_CONFIG_LIST` convention:

```json
[
  {
    "model": "gpt-4o",
    "api_key": "<your api key>",
    "base_url": "<url>"
  }
]
```

This works with any OpenAI-compatible API (OpenAI, NVIDIA NIM, vLLM, etc.). If `LLM_CONFIG_FILE` is not set, it defaults to looking for a file named `OAI_CONFIG_LIST_Perflab` in the working directory or `src/` directory.

## License

Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Licensed under the [Apache License, Version 2.0](LICENSE).
