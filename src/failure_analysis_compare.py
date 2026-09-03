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

"""Cross-agent failure-analysis charts and auto-generated insights.

This script consumes ``pass_records.jsonl`` produced by ``failure_analysis.py``
and writes a set of comparison charts plus a markdown report of the form
"SoTA agents struggle on X, while open-source agents fail at Y".

Typical usage::

    # 1) Run the per-record analysis once over all agents (no filters).
    python src/failure_analysis.py \
        --root /path/to/edalogqabenchmark \
        --output-dir failure_analysis_output

    # 2) Build cross-agent charts and insights.
    python src/failure_analysis_compare.py \
        --records failure_analysis_output/pass_records.jsonl \
        --output-dir failure_analysis_output

You can also pass ``--auto-run`` to do step 1 implicitly when the records file
is missing.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

# Reuse the canonical tag taxonomy and pipeline from the sibling module.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from failure_analysis import FAILURE_TAGS, run_pipeline  # noqa: E402


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_records(jsonl_path: Path) -> pd.DataFrame:
    """Read ``pass_records.jsonl`` into a tidy DataFrame and add helper cols."""
    df = pd.read_json(jsonl_path, lines=True)
    if df.empty:
        raise SystemExit(f"No records found in {jsonl_path}")

    df["agent_model"] = df["agent"].astype(str) + " / " + df["model"].astype(str)
    df["is_correct"] = (df["outcome"] == "correct").astype(int)
    df["is_failure"] = (df["outcome"] != "correct").astype(int)
    return df


def explode_tags(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (record, tag), keeping only failures."""
    failures = df[df["is_failure"] == 1].copy()
    if failures.empty:
        return failures
    failures = failures.assign(tag=failures["tags"]).explode("tag")
    failures = failures[failures["tag"].notna() & (failures["tag"] != "")]
    return failures


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_pass_at_1_by_agent(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    g = (
        df.groupby("agent_model")
        .agg(pass_at_1=("is_correct", "mean"), n=("is_correct", "size"))
        .reset_index()
        .sort_values("pass_at_1", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * len(g) + 2)))
    sns.barplot(g, y="agent_model", x="pass_at_1", ax=ax, color="steelblue")
    for i, row in g.reset_index(drop=True).iterrows():
        ax.text(
            row["pass_at_1"] + 0.01,
            i,
            f"{row['pass_at_1']:.1%}  (n={row['n']})",
            va="center",
            fontsize=9,
        )
    ax.set_xlim(0, 1)
    ax.set_xlabel("pass@1 (correctness == 1.0)")
    ax.set_ylabel("")
    ax.set_title("Pass@1 by agent / model")
    _save(fig, out)
    return g


def chart_heatmap(
    df: pd.DataFrame,
    row: str,
    col: str,
    value: str,
    out: Path,
    title: str,
    cmap: str = "RdYlGn",
    fmt: str = ".0%",
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> pd.DataFrame:
    pivot = (
        df.groupby([row, col])[value]
        .mean()
        .unstack(col)
        .sort_index()
        .reindex(sorted(df[col].dropna().unique()), axis=1)
    )
    fig, ax = plt.subplots(figsize=(max(6, 0.7 * pivot.shape[1] + 3), max(3, 0.45 * pivot.shape[0] + 2)))
    sns.heatmap(pivot, annot=True, fmt=fmt, cmap=cmap, vmin=vmin, vmax=vmax, ax=ax, linewidths=0.4)
    ax.set_title(title)
    ax.set_xlabel(col)
    ax.set_ylabel(row)
    _save(fig, out)
    return pivot


def chart_pass_at_1_by_agent_question_type(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    return chart_heatmap(
        df, "agent_model", "question_type", "is_correct", out,
        title="Pass@1 by agent and question_type",
    )


def chart_pass_at_1_by_agent_stage(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    return chart_heatmap(
        df, "agent_model", "stage", "is_correct", out,
        title="Pass@1 by agent and stage",
    )


def chart_pass_at_1_by_agent_difficulty(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    order = ["easy", "medium", "hard"]
    sub = df[df["difficulty"].isin(order)]
    return chart_heatmap(
        sub, "agent_model", "difficulty", "is_correct", out,
        title="Pass@1 by agent and difficulty",
    )


def chart_pass_at_1_by_agent_design(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    return chart_heatmap(
        df, "agent_model", "design", "is_correct", out,
        title="Pass@1 by agent and design",
    )


def chart_tag_share_stacked(failures: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Normalized stacked bar: composition of failures per agent."""
    if failures.empty:
        return pd.DataFrame()
    pivot = (
        failures.groupby(["agent_model", "tag"]).size().unstack(fill_value=0)
    )
    pivot = pivot.reindex(columns=list(FAILURE_TAGS.keys()), fill_value=0)
    norm = pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    fig, ax = plt.subplots(figsize=(11, max(3.5, 0.5 * len(norm) + 2)))
    bottom = np.zeros(len(norm))
    palette = sns.color_palette("tab20", n_colors=len(norm.columns))
    ys = np.arange(len(norm))
    for i, tag in enumerate(norm.columns):
        widths = norm[tag].values
        ax.barh(ys, widths, left=bottom, color=palette[i], label=tag, edgecolor="white", linewidth=0.3)
        bottom += widths
    ax.set_yticks(ys)
    ax.set_yticklabels(norm.index)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of agent's failures")
    ax.set_title("Failure-tag composition by agent (each row sums to 100%)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, fontsize=8)
    _save(fig, out)
    return norm


def chart_tag_overindex_heatmap(failures: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Heatmap of (agent_share - mean_share) per tag — highlights over-indexing."""
    if failures.empty:
        return pd.DataFrame()
    pivot = failures.groupby(["agent_model", "tag"]).size().unstack(fill_value=0)
    pivot = pivot.reindex(columns=list(FAILURE_TAGS.keys()), fill_value=0)
    share = pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    delta = share.sub(share.mean(axis=0), axis=1)

    fig, ax = plt.subplots(figsize=(max(8, 0.55 * delta.shape[1] + 4), max(3, 0.45 * delta.shape[0] + 2)))
    bound = max(0.05, float(delta.abs().to_numpy().max()))
    sns.heatmap(
        delta, annot=True, fmt=".0%", cmap="RdBu_r", center=0,
        vmin=-bound, vmax=bound, ax=ax, linewidths=0.4,
    )
    ax.set_title("Over/under-indexing by tag (agent share - cross-agent mean)")
    ax.set_xlabel("failure tag")
    ax.set_ylabel("")
    _save(fig, out)
    return delta


def chart_persistence_by_agent(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Stacked bar of per-question persistence labels per agent."""
    grouped = (
        df.groupby(["agent_model", "design", "question_id"])
        .agg(n=("is_correct", "size"), n_correct=("is_correct", "sum"))
        .reset_index()
    )

    def label(row):
        n, c = row["n"], row["n_correct"]
        if n == 0:
            return "no_data"
        if c == n:
            return "always_correct"
        if c == 0:
            return "persistent_failure"
        return "stochastic"

    grouped["persistence"] = grouped.apply(label, axis=1)
    pivot = (
        grouped.groupby(["agent_model", "persistence"]).size().unstack(fill_value=0)
    )
    order = ["always_correct", "stochastic", "persistent_failure"]
    pivot = pivot.reindex(columns=order, fill_value=0)
    norm = pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    fig, ax = plt.subplots(figsize=(9, max(3, 0.45 * len(norm) + 2)))
    bottom = np.zeros(len(norm))
    colors = {"always_correct": "#2ca02c", "stochastic": "#ff7f0e", "persistent_failure": "#d62728"}
    ys = np.arange(len(norm))
    for col in order:
        widths = norm[col].values
        ax.barh(ys, widths, left=bottom, color=colors[col], label=col, edgecolor="white")
        for j, w in enumerate(widths):
            if w > 0.04:
                ax.text(bottom[j] + w / 2, ys[j], f"{w:.0%}", ha="center", va="center", fontsize=8, color="white")
        bottom += widths
    ax.set_yticks(ys)
    ax.set_yticklabels(norm.index)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of unique questions")
    ax.set_title("Per-question persistence across pass@k by agent")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    _save(fig, out)
    return pivot


# ---------------------------------------------------------------------------
# Insight extraction
# ---------------------------------------------------------------------------


def _classify_agent_tier(agent_models: Iterable[str]) -> Dict[str, str]:
    """Heuristic split into 'sota' vs 'opensource' tiers from agent name + model."""
    tiers: Dict[str, str] = {}
    sota_keywords = ("claude", "gpt", "sonnet", "opus", "o1", "o3", "gemini", "cursor")
    opensource_keywords = ("llama", "mixtral", "qwen", "deepseek", "rlm", "rag")
    for am in agent_models:
        s = am.lower()
        if any(k in s for k in sota_keywords) and not any(k in s for k in opensource_keywords):
            tiers[am] = "sota"
        elif any(k in s for k in opensource_keywords):
            tiers[am] = "opensource"
        else:
            tiers[am] = "other"
    return tiers


def _top_overindex(delta: pd.DataFrame, k: int = 3) -> Dict[str, List[Tuple[str, float]]]:
    """For each agent, return its top-k tags by positive over-index."""
    out: Dict[str, List[Tuple[str, float]]] = {}
    for agent in delta.index:
        row = delta.loc[agent]
        sorted_tags = row.sort_values(ascending=False)
        out[agent] = [(t, float(v)) for t, v in sorted_tags.head(k).items() if v > 0]
    return out


def _worst_buckets(
    df: pd.DataFrame, by: str, k: int = 3, min_n: int = 20
) -> Dict[str, List[Tuple[str, float, int]]]:
    """Return per-agent worst k buckets (lowest pass@1) for a given grouping column."""
    g = (
        df.groupby(["agent_model", by])
        .agg(pass_at_1=("is_correct", "mean"), n=("is_correct", "size"))
        .reset_index()
    )
    g = g[g["n"] >= min_n]
    out: Dict[str, List[Tuple[str, float, int]]] = {}
    for agent, sub in g.groupby("agent_model"):
        sub = sub.sort_values("pass_at_1", ascending=True).head(k)
        out[agent] = [(row[by], float(row["pass_at_1"]), int(row["n"])) for _, row in sub.iterrows()]
    return out


def _largest_gap_questions(
    df: pd.DataFrame, top_k: int = 10
) -> pd.DataFrame:
    """Per (design, question_id), pass@1 spread across agent_model."""
    g = (
        df.groupby(["design", "question_id", "question_type", "stage", "difficulty", "agent_model"])
        .agg(pass_at_1=("is_correct", "mean"), n=("is_correct", "size"))
        .reset_index()
    )
    pivot = g.pivot_table(
        index=["design", "question_id", "question_type", "stage", "difficulty"],
        columns="agent_model",
        values="pass_at_1",
    )
    pivot = pivot.dropna(thresh=2)
    pivot["min"] = pivot.min(axis=1)
    pivot["max"] = pivot.max(axis=1)
    pivot["spread"] = pivot["max"] - pivot["min"]
    pivot = pivot.sort_values("spread", ascending=False).head(top_k)
    return pivot.reset_index()


def _universal_failures(
    df: pd.DataFrame, threshold: float = 0.0, top_k: int = 10
) -> pd.DataFrame:
    g = (
        df.groupby(["design", "question_id", "question_type", "stage", "difficulty"])
        .agg(pass_at_1=("is_correct", "mean"), n=("is_correct", "size"), n_agents=("agent_model", "nunique"))
        .reset_index()
    )
    g = g[(g["pass_at_1"] <= threshold) & (g["n_agents"] >= 2)]
    return g.sort_values(["n_agents", "n"], ascending=False).head(top_k)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _format_pct(x: float) -> str:
    return f"{x:.0%}"


def render_insights(
    df: pd.DataFrame,
    pass_at_1_summary: pd.DataFrame,
    delta: pd.DataFrame,
    pivot_qtype: pd.DataFrame,
    chart_paths: Dict[str, Path],
    output_dir: Path,
) -> str:
    def rel(p: Path) -> str:
        try:
            return str(p.relative_to(output_dir))
        except ValueError:
            return p.name

    lines: List[str] = []
    tiers = _classify_agent_tier(df["agent_model"].unique())

    lines.append("# Cross-agent failure analysis\n")
    lines.append(
        f"Total pass records: **{len(df):,}** across "
        f"**{df['agent_model'].nunique()}** agent/model combos and "
        f"**{df['design'].nunique()}** designs.\n"
    )

    lines.append("## Headline pass@1\n")
    for _, row in pass_at_1_summary.iterrows():
        tier = tiers.get(row["agent_model"], "other")
        lines.append(
            f"- **{row['agent_model']}** [{tier}]: pass@1 = "
            f"{_format_pct(row['pass_at_1'])} (n={row['n']})"
        )
    lines.append(f"\n![pass_at_1_by_agent]({rel(chart_paths['pass_at_1_by_agent'])})\n")

    lines.append("## Where each agent struggles most\n")
    worst_qtype = _worst_buckets(df, "question_type")
    worst_stage = _worst_buckets(df, "stage")
    worst_diff = _worst_buckets(df, "difficulty", min_n=10)
    for agent in pass_at_1_summary["agent_model"]:
        lines.append(f"### {agent} ({tiers.get(agent, 'other')})")
        if agent in worst_qtype:
            parts = ", ".join(
                f"`{b}` {_format_pct(p)} (n={n})" for b, p, n in worst_qtype[agent]
            )
            lines.append(f"- worst question types: {parts}")
        if agent in worst_stage:
            parts = ", ".join(
                f"`{b}` {_format_pct(p)} (n={n})" for b, p, n in worst_stage[agent]
            )
            lines.append(f"- worst stages: {parts}")
        if agent in worst_diff:
            parts = ", ".join(
                f"`{b}` {_format_pct(p)} (n={n})" for b, p, n in worst_diff[agent]
            )
            lines.append(f"- by difficulty: {parts}")
        lines.append("")
    lines.append(f"![pass_at_1_by_agent_question_type]({rel(chart_paths['pass_at_1_by_agent_question_type'])})\n")
    lines.append(f"![pass_at_1_by_agent_stage]({rel(chart_paths['pass_at_1_by_agent_stage'])})\n")
    lines.append(f"![pass_at_1_by_agent_difficulty]({rel(chart_paths['pass_at_1_by_agent_difficulty'])})\n")

    lines.append("## Failure-mode signature per agent\n")
    if not delta.empty:
        top_over = _top_overindex(delta, k=3)
        for agent in pass_at_1_summary["agent_model"]:
            tier = tiers.get(agent, "other")
            entries = top_over.get(agent, [])
            if not entries:
                lines.append(f"- **{agent}** [{tier}]: no clearly over-indexed failure mode.")
                continue
            parts = ", ".join(
                f"`{tag}` (+{_format_pct(v)} vs cross-agent mean)" for tag, v in entries
            )
            lines.append(f"- **{agent}** [{tier}]: over-indexes on {parts}")
    lines.append(f"\n![tag_share_stacked]({rel(chart_paths['tag_share_stacked'])})\n")
    lines.append(f"![tag_overindex_heatmap]({rel(chart_paths['tag_overindex_heatmap'])})\n")

    lines.append("## SoTA vs open-source narrative\n")
    sota_agents = [a for a, t in tiers.items() if t == "sota"]
    oss_agents = [a for a, t in tiers.items() if t == "opensource"]
    if sota_agents and oss_agents and not delta.empty:
        sota_share = delta.loc[delta.index.isin(sota_agents)].mean(axis=0)
        oss_share = delta.loc[delta.index.isin(oss_agents)].mean(axis=0)
        diff = (sota_share - oss_share).sort_values(ascending=False)
        sota_top = [(t, float(v)) for t, v in diff.head(3).items() if v > 0]
        oss_top = [(t, float(v)) for t, v in diff.tail(3)[::-1].items() if v < 0]
        if sota_top:
            parts = ", ".join(f"`{t}` (+{_format_pct(v)})" for t, v in sota_top)
            lines.append(f"- SoTA agents disproportionately fail with: {parts}")
        if oss_top:
            parts = ", ".join(f"`{t}` (+{_format_pct(-v)})" for t, v in oss_top)
            lines.append(f"- Open-source agents disproportionately fail with: {parts}")

        sota_qtype = (
            df[df["agent_model"].isin(sota_agents)].groupby("question_type")["is_correct"].mean()
        )
        oss_qtype = (
            df[df["agent_model"].isin(oss_agents)].groupby("question_type")["is_correct"].mean()
        )
        gap = (sota_qtype - oss_qtype).dropna().sort_values(ascending=False)
        if not gap.empty:
            biggest_for_sota = gap.head(1)
            biggest_for_oss = gap.tail(1)
            lines.append(
                f"- Biggest SoTA advantage: question type "
                f"`{biggest_for_sota.index[0]}` "
                f"(SoTA {_format_pct(sota_qtype[biggest_for_sota.index[0]])} vs "
                f"OSS {_format_pct(oss_qtype[biggest_for_sota.index[0]])})."
            )
            lines.append(
                f"- Where SoTA hurts most relative to OSS: question type "
                f"`{biggest_for_oss.index[0]}` "
                f"(SoTA {_format_pct(sota_qtype[biggest_for_oss.index[0]])} vs "
                f"OSS {_format_pct(oss_qtype[biggest_for_oss.index[0]])})."
            )
    else:
        lines.append("- Not enough agents on each tier to produce a SoTA-vs-OSS comparison.")

    lines.append("\n## Hardest questions across agents\n")
    universal = _universal_failures(df)
    if not universal.empty:
        lines.append("Questions every agent failed (potential benchmark / golden-data candidates):\n")
        for _, row in universal.iterrows():
            lines.append(
                f"- `{row['question_id']}` ({row['design']}, "
                f"{row['question_type']}, stage={row['stage']}, "
                f"difficulty={row['difficulty']}) — pass@1=0 across {int(row['n_agents'])} agents"
            )
    gaps = _largest_gap_questions(df)
    if not gaps.empty:
        lines.append("\nQuestions with the largest agent disagreement (likely solvable, agent-specific failures):\n")
        for _, row in gaps.iterrows():
            lines.append(
                f"- `{row['question_id']}` ({row['design']}, {row['question_type']}, "
                f"stage={row['stage']}, difficulty={row['difficulty']}) — spread="
                f"{_format_pct(row['spread'])} (min={_format_pct(row['min'])}, max={_format_pct(row['max'])})"
            )

    lines.append("\n## Per-question-type pass@1 matrix\n")
    if not pivot_qtype.empty:
        lines.append("```")
        lines.append(pivot_qtype.map(lambda v: f"{v:.1%}" if pd.notna(v) else "").to_string())
        lines.append("```")

    lines.append("\n## Persistence (across pass@k)\n")
    lines.append(f"![persistence_by_agent]({rel(chart_paths['persistence_by_agent'])})\n")

    lines.append("\n## Per-design pass@1\n")
    lines.append(f"![pass_at_1_by_agent_design]({rel(chart_paths['pass_at_1_by_agent_design'])})\n")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_compare(
    records_path: Path,
    output_dir: Path,
    designs: Optional[List[str]] = None,
    agents: Optional[List[str]] = None,
) -> None:
    df = load_records(records_path)
    if agents:
        df = df[df["agent"].isin(agents)]
    if designs:
        df = df[df["design"].isin(designs)]
    if df.empty:
        raise SystemExit("No records left after filtering --agents/--designs.")

    failures = explode_tags(df)

    chart_dir = output_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)

    chart_paths: Dict[str, Path] = {
        "pass_at_1_by_agent": chart_dir / "pass_at_1_by_agent.png",
        "pass_at_1_by_agent_question_type": chart_dir / "pass_at_1_by_agent_question_type.png",
        "pass_at_1_by_agent_stage": chart_dir / "pass_at_1_by_agent_stage.png",
        "pass_at_1_by_agent_difficulty": chart_dir / "pass_at_1_by_agent_difficulty.png",
        "pass_at_1_by_agent_design": chart_dir / "pass_at_1_by_agent_design.png",
        "tag_share_stacked": chart_dir / "tag_share_stacked.png",
        "tag_overindex_heatmap": chart_dir / "tag_overindex_heatmap.png",
        "persistence_by_agent": chart_dir / "persistence_by_agent.png",
    }

    summary = chart_pass_at_1_by_agent(df, chart_paths["pass_at_1_by_agent"])
    pivot_qtype = chart_pass_at_1_by_agent_question_type(df, chart_paths["pass_at_1_by_agent_question_type"])
    chart_pass_at_1_by_agent_stage(df, chart_paths["pass_at_1_by_agent_stage"])
    chart_pass_at_1_by_agent_difficulty(df, chart_paths["pass_at_1_by_agent_difficulty"])
    chart_pass_at_1_by_agent_design(df, chart_paths["pass_at_1_by_agent_design"])
    chart_tag_share_stacked(failures, chart_paths["tag_share_stacked"])
    delta = chart_tag_overindex_heatmap(failures, chart_paths["tag_overindex_heatmap"])
    chart_persistence_by_agent(df, chart_paths["persistence_by_agent"])

    # Persist underlying tables next to charts for downstream notebooks.
    summary.to_csv(output_dir / "pass_at_1_by_agent.csv", index=False)
    pivot_qtype.to_csv(output_dir / "pass_at_1_by_agent_question_type.csv")
    if not delta.empty:
        delta.to_csv(output_dir / "tag_overindex_delta.csv")

    insights_md = render_insights(df, summary, delta, pivot_qtype, chart_paths, output_dir)
    (output_dir / "insights.md").write_text(insights_md, encoding="utf-8")

    print("Charts written to:", chart_dir)
    for k, p in chart_paths.items():
        print(f"  - {k}: {p}")
    print("Insights:", output_dir / "insights.md")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--records",
        type=Path,
        default=Path("failure_analysis_output/pass_records.jsonl"),
        help="Path to pass_records.jsonl produced by failure_analysis.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("failure_analysis_output"),
        help="Directory to write charts/ and insights.md into.",
    )
    parser.add_argument(
        "--auto-run",
        action="store_true",
        help="If --records is missing, invoke failure_analysis.run_pipeline first.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Workspace root for --auto-run (passed to failure_analysis.run_pipeline).",
    )
    parser.add_argument(
        "--agents",
        nargs="*",
        default=None,
        help="Optional filter applied to the records (and to --auto-run discovery).",
    )
    parser.add_argument(
        "--designs",
        nargs="*",
        default=None,
        help="Optional filter applied to the records (and to --auto-run discovery).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    records = args.records
    if not records.exists():
        if args.auto_run:
            print(f"{records} missing; running failure_analysis.run_pipeline first.")
            run_pipeline(
                root=args.root.resolve(),
                output_dir=args.output_dir.resolve(),
                agents=args.agents,
                designs=args.designs,
            )
        else:
            raise SystemExit(
                f"{records} not found. Run failure_analysis.py first, or pass --auto-run."
            )
    run_compare(records, args.output_dir.resolve(), designs=args.designs, agents=args.agents)


if __name__ == "__main__":
    main()
