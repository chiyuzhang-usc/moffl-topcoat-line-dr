#!/usr/bin/env python3
"""
Step-05 — Oven Convoy Serialization (2→1 Merge with Global Adjacency)

This script implements STEP-05 of the MOFFS pipeline:

- Inputs: SDSU per-booth JOB sequences from STEP-04.
- Output: a single global oven convoy (2→1 merge), preserving in-booth order and enforcing
  global adjacency forbids (C2) on parts, based solely on the dataset adjacency_rules and
  adjacency policy from global_config.toml.
- No paint or purge decisions are made here (C1/C3 are upstream, in SDSU).
- Tokens are not used to relax adjacency; adjacency is always parts-only.

The implementation is deterministic and emits a manifest and a report per the step-gating protocol.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
import tomllib


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_parent_dir(path: Path) -> None:
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def load_toml(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def pick_first_existing(colnames: Iterable[str], candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in colnames}
    for cand in candidates:
        c = cols.get(cand.lower())
        if c is not None:
            return c
    return None


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Step05Paths:
    output_dir: Path
    dataset_dir: Path
    booth1_input: Path
    booth2_input: Path
    global_line_output: Path
    manifest_path: Path
    report_path: Path
    adjacency_rules_path: Path
    global_config_path: Path
    steps_config_path: Path


@dataclass
class Step05Config:
    allow_postpaint_empty: bool
    tie_break_policy: str
    adjacency_enabled: bool
    scope_escalation: str
    empty_units_postpaint_ratio: float


@dataclass
class SchemaConfig:
    part_id_col: str
    booth_id_col: str


@dataclass
class AdjacencyModel:
    forbid_pairs: Set[Tuple[str, str]]  # (from_part_id, to_part_id)


@dataclass
class MergeMetrics:
    n_jobs_total: int
    n_jobs_booth1: int
    n_jobs_booth2: int
    n_conflict_events: int
    n_deadlocks: int
    n_postpaint_empty: int
    max_run_booth1: int
    max_run_booth2: int


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------


def resolve_paths(global_cfg: dict, steps_cfg: dict) -> Step05Paths:
    paths_cfg = global_cfg["paths"]
    data_files = global_cfg["data_files"]
    step05_cfg = steps_cfg["step05"]

    dataset_dir = Path(paths_cfg["dataset_dir"])
    output_dir = Path(paths_cfg["output_dir"])

    return Step05Paths(
        output_dir=output_dir,
        dataset_dir=dataset_dir,
        booth1_input=output_dir / step05_cfg["booth1_input_relpath"],
        booth2_input=output_dir / step05_cfg["booth2_input_relpath"],
        global_line_output=output_dir / step05_cfg["global_line_relpath"],
        manifest_path=output_dir / step05_cfg["manifest_relpath"],
        report_path=output_dir / step05_cfg["report_relpath"],
        adjacency_rules_path=dataset_dir / data_files["adjacency_rules"],
        global_config_path=Path("global_config.toml"),
        steps_config_path=Path("steps_config.toml"),
    )


def build_step05_config(global_cfg: dict, steps_cfg: dict) -> Step05Config:
    adj_cfg = global_cfg.get("adjacency", {})
    tokens_cfg = global_cfg.get("tokens", {})

    step05_cfg = steps_cfg.get("step05", {})
    allow_postpaint_empty = bool(step05_cfg.get("allow_postpaint_empty", False))
    tie_break_policy = str(step05_cfg.get("tie_break_policy", "longer_remaining")).lower()

    adjacency_enabled = bool(adj_cfg.get("enabled", True))
    scope_escalation = str(adj_cfg.get("scope_escalation", "global_if_missing"))
    empty_units_postpaint_ratio = float(tokens_cfg.get("empty_units_postpaint_ratio", 1.0))

    if tie_break_policy not in ("longer_remaining", "round_robin"):
        raise ValueError(f"Unsupported tie_break_policy={tie_break_policy!r} in step05 config")

    return Step05Config(
        allow_postpaint_empty=allow_postpaint_empty,
        tie_break_policy=tie_break_policy,
        adjacency_enabled=adjacency_enabled,
        scope_escalation=scope_escalation,
        empty_units_postpaint_ratio=empty_units_postpaint_ratio,
    )


def resolve_schema(global_cfg: dict, df: pd.DataFrame) -> SchemaConfig:
    ds = global_cfg.get("data_schema", {})
    demands_schema = ds.get("demands", {})
    parts_schema = ds.get("parts", {})

    part_syn = demands_schema.get("part_id_synonyms", [])
    if not part_syn:
        part_syn = parts_schema.get("part_id_synonyms", [])
    booth_syn = ["booth_id"]  # booth_id is introduced upstream

    part_col = pick_first_existing(df.columns, part_syn + ["part_id"])
    booth_col = pick_first_existing(df.columns, booth_syn)

    if part_col is None:
        raise KeyError(f"Could not resolve part_id column in STEP-05 input; candidates={part_syn}")
    if booth_col is None:
        # Upstream should have booth_id; fail fast if missing
        raise KeyError("Could not find 'booth_id' column in STEP-05 input")

    return SchemaConfig(part_id_col=part_col, booth_id_col=booth_col)


# ---------------------------------------------------------------------------
# Adjacency model (global line)
# ---------------------------------------------------------------------------


def load_adjacency_model_global(
    paths: Step05Paths,
    global_cfg: dict,
    step05_cfg: Step05Config,
) -> AdjacencyModel:
    if not step05_cfg.adjacency_enabled:
        return AdjacencyModel(forbid_pairs=set())

    df = pd.read_csv(paths.adjacency_rules_path)
    schema = global_cfg.get("data_schema", {}).get("adjacency_rules", {})
    from_syn = schema.get("from_part_id_synonyms", ["from_part_id"])
    to_syn = schema.get("to_part_id_synonyms", ["to_part_id"])
    scope_syn = schema.get("scope_synonyms", ["scope"])
    rel_syn = schema.get("relation_synonyms", ["relation"])

    from_col = pick_first_existing(df.columns, from_syn)
    to_col = pick_first_existing(df.columns, to_syn)
    scope_col = pick_first_existing(df.columns, scope_syn)
    rel_col = pick_first_existing(df.columns, rel_syn)

    if from_col is None or to_col is None or rel_col is None:
        # If we cannot resolve adjacency columns, conservatively disable adjacency for this step.
        return AdjacencyModel(forbid_pairs=set())

    forbid_pairs: Set[Tuple[str, str]] = set()
    for _, r in df.iterrows():
        rel = str(r[rel_col]).lower()
        if "forbid" not in rel:
            continue

        scope_val = ""
        if scope_col is not None:
            scope_val = str(r[scope_col]).strip().lower()

        # Determine if this rule applies to global line
        applies = False
        if scope_val in ("global", "across_skids"):
            applies = True
        elif not scope_val and step05_cfg.scope_escalation == "global_if_missing":
            applies = True

        if not applies:
            continue

        from_part = str(r[from_col])
        to_part = str(r[to_col])
        forbid_pairs.add((from_part, to_part))

    return AdjacencyModel(forbid_pairs=forbid_pairs)


# ---------------------------------------------------------------------------
# Merge algorithm
# ---------------------------------------------------------------------------

def merge_booth_streams(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    schema: SchemaConfig,
    adj_model: AdjacencyModel,
    cfg: Step05Config,
) -> Tuple[pd.DataFrame, MergeMetrics]:
    """
    Merge SDSU per-booth JOB sequences (PB1, PB2) into a single oven stream.

    Semantics:
    - Stream-level adjacency: adjacency is checked on the full symbol stream.
    - Symbols are:
        "B1"    → job from booth 1
        "B2"    → job from booth 2
        "EMPTY" → explicit empty carrier in the oven
    - adjacency_rules.csv contains only real part→part forbids; "EMPTY" never
      appears in the dataset, so any pair involving "EMPTY" is automatically allowed.
    - If no head is feasible and cfg.allow_postpaint_empty is True, we insert
      an "EMPTY" symbol on the stream to break the adjacency deadlock.
    """

    n1 = len(df1)
    n2 = len(df2)
    h1 = 0
    h2 = 0

    forbid_pairs = adj_model.forbid_pairs
    adjacency_enabled = cfg.adjacency_enabled and bool(forbid_pairs)
    tie_break_policy = cfg.tie_break_policy

    # Outgoing stream: kind ∈ {"B1", "B2", "EMPTY"}, index is local job index (or -1 for EMPTY)
    out_stream: List[Tuple[str, int]] = []
    # last_symbol is the last symbol on the stream (part_id or "EMPTY")
    last_symbol: Optional[str] = None
    # last_booth_kind remembers the last booth used for round_robin tie-breaking
    last_booth_kind: Optional[str] = None

    n_conflict_events = 0
    n_deadlocks = 0
    n_postpaint_empty = 0

    while h1 < n1 or h2 < n2:
        candidates: List[Tuple[str, int, str, bool]] = []

        # Booth 1 candidate
        if h1 < n1:
            part1 = str(df1.iloc[h1][schema.part_id_col])
            feasible1 = True
            if adjacency_enabled and last_symbol is not None:
                if (last_symbol, part1) in forbid_pairs:
                    feasible1 = False
                    n_conflict_events += 1
            candidates.append(("B1", h1, part1, feasible1))

        # Booth 2 candidate
        if h2 < n2:
            part2 = str(df2.iloc[h2][schema.part_id_col])
            feasible2 = True
            if adjacency_enabled and last_symbol is not None:
                if (last_symbol, part2) in forbid_pairs:
                    feasible2 = False
                    n_conflict_events += 1
            candidates.append(("B2", h2, part2, feasible2))

        feasible = [c for c in candidates if c[3]]

        if not feasible:
            # No admissible next job without violating adjacency.
            n_deadlocks += 1

            if cfg.allow_postpaint_empty:
                # Stream-level EMPTY insertion: "EMPTY" is a real symbol on the oven stream.
                out_stream.append(("EMPTY", -1))
                last_symbol = "EMPTY"
                # We deliberately reset the booth tiebreaker here; EMPTY is not a booth.
                last_booth_kind = None
                n_postpaint_empty += 1
                print(
                    f"STEP-05 merge: inserted EMPTY at stream position {len(out_stream)} "
                    f"to resolve adjacency deadlock (token: {last_symbol!r})"
                )
                continue
            else:
                raise RuntimeError(
                    f"STEP-05: global adjacency deadlock with last_symbol={last_symbol!r}; "
                    "no booth head can be placed without violating adjacency and "
                    "allow_postpaint_empty is False."
                )

        # Choose candidate according to tie_break_policy
        if len(feasible) == 1:
            chosen = feasible[0]
        else:
            if tie_break_policy == "longer_remaining":
                rem1 = n1 - h1 if any(k == "B1" for (k, _, _, _) in feasible) else -1
                rem2 = n2 - h2 if any(k == "B2" for (k, _, _, _) in feasible) else -1
                if rem1 > rem2:
                    prefer = "B1"
                elif rem2 > rem1:
                    prefer = "B2"
                else:
                    prefer = "B1"  # deterministic tie-break
                chosen = next(c for c in feasible if c[0] == prefer)
            elif tie_break_policy == "round_robin":
                # Alternate booths when both are feasible
                if last_booth_kind is None or last_booth_kind == "B2":
                    preferred = "B1"
                else:
                    preferred = "B2"
                if any(c[0] == preferred for c in feasible):
                    chosen = next(c for c in feasible if c[0] == preferred)
                else:
                    other = "B1" if preferred == "B2" else "B2"
                    chosen = next(c for c in feasible if c[0] == other)
            else:
                # Should not happen due to validation in build_step05_config
                chosen = feasible[0]

        kind, local_idx, part_id, _ = chosen
        out_stream.append((kind, local_idx))
        last_symbol = part_id
        if kind in ("B1", "B2"):
            last_booth_kind = kind

        if kind == "B1":
            h1 += 1
        elif kind == "B2":
            h2 += 1
        else:
            # EMPTY should never be chosen here
            raise RuntimeError(f"STEP-05 internal error: unexpected kind={kind!r} in chosen candidate")

    # ----------------------------------------------------------------------
    # Build output DataFrame, preserving EMPTY as a distinct symbol
    # ----------------------------------------------------------------------
    records = []

    # Choose a base column set from one of the inputs
    if len(df1) > 0:
        base_cols = list(df1.columns)
    else:
        base_cols = list(df2.columns)

    for kind, local_idx in out_stream:
        if kind == "B1":
            row = df1.iloc[local_idx].copy()
        elif kind == "B2":
            row = df2.iloc[local_idx].copy()
        elif kind == "EMPTY":
            # Synthetic EMPTY row with same schema as the booth inputs
            row = pd.Series({col: None for col in base_cols})
            row[schema.part_id_col] = "EMPTY"
            # Either leave booth_id as None or mark as "EMPTY"
            row[schema.booth_id_col] = "EMPTY"
        else:
            raise RuntimeError(f"STEP-05 internal error: unexpected kind in out_stream: {kind!r}")
        records.append(row)

    global_df = pd.DataFrame(records).reset_index(drop=True)
    global_df["global_seq_index"] = list(range(1, len(global_df) + 1))

    # ----------------------------------------------------------------------
    # Compute max run length per booth, ignoring EMPTY
    # ----------------------------------------------------------------------
    max_run_b1 = 0
    max_run_b2 = 0
    cur_run_b1 = 0
    cur_run_b2 = 0

    for kind, _ in out_stream:
        if kind == "B1":
            cur_run_b1 += 1
            cur_run_b2 = 0
        elif kind == "B2":
            cur_run_b2 += 1
            cur_run_b1 = 0
        else:  # "EMPTY"
            cur_run_b1 = 0
            cur_run_b2 = 0
        max_run_b1 = max(max_run_b1, cur_run_b1)
        max_run_b2 = max(max_run_b2, cur_run_b2)

    metrics = MergeMetrics(
        n_jobs_total=len(global_df),      # includes EMPTY; raw stream length
        n_jobs_booth1=n1,
        n_jobs_booth2=n2,
        n_conflict_events=n_conflict_events,
        n_deadlocks=n_deadlocks,
        n_postpaint_empty=n_postpaint_empty,
        max_run_booth1=max_run_b1,
        max_run_booth2=max_run_b2,
    )

    return global_df, metrics

# ---------------------------------------------------------------------------
# Final adjacency verification (stream-level)
# ---------------------------------------------------------------------------


def verify_global_adjacency(
    global_df: pd.DataFrame,
    schema: SchemaConfig,
    adj_model: AdjacencyModel,
) -> None:
    """
    Final safety check: ensure that no consecutive symbols violate global adjacency.

    Semantics:
    - We check adjacency on the full oven stream, exactly as merge_booth_streams does.
    - `adjacency_rules.csv` defines forbidden ordered pairs for real parts only.
      "EMPTY" is just another symbol in the stream and never appears in forbid_pairs,
      so (A, "EMPTY") and ("EMPTY", B) are always allowed.
    """
    forbid_pairs = adj_model.forbid_pairs
    if not forbid_pairs:
        return

    part_col = schema.part_id_col
    stream = [str(p) for p in global_df[part_col].tolist()]

    for i in range(len(stream) - 1):
        a = stream[i]
        b = stream[i + 1]
        if (a, b) in forbid_pairs:
            raise RuntimeError(
                f"STEP-05 verification: global adjacency violation detected at "
                f"positions {i}->{i+1}: ({a}, {b})"
            )


# ---------------------------------------------------------------------------
# Manifest and report
# ---------------------------------------------------------------------------


def write_manifest(
    paths: Step05Paths,
    cfg: Step05Config,
    metrics: MergeMetrics,
    input_files: List[Path],
    output_files: List[Path],
    start_time: float,
    end_time: float,
) -> None:
    ensure_parent_dir(paths.manifest_path)
    manifest = {
        "step": "step05_merge_global_line",
        "version": "1.0",
        "config_files": {
            str(paths.global_config_path): sha256_of_file(paths.global_config_path),
            str(paths.steps_config_path): sha256_of_file(paths.steps_config_path),
        },
        "input_files": {str(p): sha256_of_file(p) for p in input_files if p.exists()},
        "output_files": {str(p): sha256_of_file(p) for p in output_files if p.exists()},
        "settings_effective": dataclasses.asdict(cfg),
        "metrics": dataclasses.asdict(metrics),
        "runtime_seconds": end_time - start_time,
    }
    with paths.manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def write_report(paths: Step05Paths, cfg: Step05Config, metrics: MergeMetrics) -> None:
    ensure_parent_dir(paths.report_path)
    lines: List[str] = []

    lines.append("# Step-05 — Oven Convoy Serialization (2→1 Merge)")
    lines.append("")
    lines.append("## Purpose / What / Why")
    lines.append("")
    lines.append("- **Purpose.** Merge SDSU-optimized booth sequences (PB1, PB2) into a single oven convoy while")
    lines.append("  enforcing global adjacency forbids on parts and preserving in-booth order.")
    lines.append("- **What.** Deterministic greedy 2→1 interleave, using adjacency_rules.csv for C2(global).")
    lines.append("- **Why.** The merged line is the canonical schedule for DES and for computing f2 (window mix) and")
    lines.append("  f4 (blocking/idle). This step is the unique owner of C2 on the merged 2→1 line.")
    lines.append("")
    lines.append("## Effective Settings")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(dataclasses.asdict(cfg), indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Merge Metrics")
    lines.append("")
    lines.append(f"- Total jobs on global line: {metrics.n_jobs_total}")
    lines.append(f"- Jobs from PB1: {metrics.n_jobs_booth1}")
    lines.append(f"- Jobs from PB2: {metrics.n_jobs_booth2}")
    lines.append(f"- Conflict events (candidates rejected due to adjacency): {metrics.n_conflict_events}")
    lines.append(f"- Deadlocks encountered: {metrics.n_deadlocks}")
    lines.append(f"- Post-paint EMPTY used: {metrics.n_postpaint_empty}")
    lines.append(f"- Max consecutive PB1 jobs: {metrics.max_run_booth1}")
    lines.append(f"- Max consecutive PB2 jobs: {metrics.max_run_booth2}")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Adjacency is evaluated on parts-only; no tokens are used to relax C2.")
    lines.append("- Paint feasibility and purge (C1/C3) remain as enforced in per-booth SDSU (Step-04).")
    lines.append("")

    with paths.report_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def run_step05() -> None:
    start_time = time.time()

    global_cfg_path = Path("global_config.toml")
    steps_cfg_path = Path("steps_config.toml")
    if not global_cfg_path.exists():
        raise FileNotFoundError("global_config.toml not found")
    if not steps_cfg_path.exists():
        raise FileNotFoundError("steps_config.toml not found")

    global_cfg = load_toml(global_cfg_path)
    steps_cfg = load_toml(steps_cfg_path)

    paths = resolve_paths(global_cfg, steps_cfg)
    cfg = build_step05_config(global_cfg, steps_cfg)

    # Load inputs
    if not paths.booth1_input.exists():
        raise FileNotFoundError(f"STEP-05: booth1 input not found: {paths.booth1_input}")
    if not paths.booth2_input.exists():
        raise FileNotFoundError(f"STEP-05: booth2 input not found: {paths.booth2_input}")

    df1 = pd.read_csv(paths.booth1_input)
    df2 = pd.read_csv(paths.booth2_input)

    # Resolve schema from PB1 (assume PB2 matches)
    schema = resolve_schema(global_cfg, df1)

    # Load adjacency model for global line
    adj_model = load_adjacency_model_global(paths, global_cfg, cfg)

    # Merge
    global_df, metrics = merge_booth_streams(df1, df2, schema, adj_model, cfg)

    # Verify global adjacency on the full stream (jobs + EMPTY)
    verify_global_adjacency(global_df, schema, adj_model)
    print("STEP-05 verification: no global adjacency violations detected.")

    # Write output
    ensure_parent_dir(paths.global_line_output)
    global_df.to_csv(paths.global_line_output, index=False)

    end_time = time.time()

    # Manifest & report
    input_files = [
        paths.global_config_path,
        paths.steps_config_path,
        paths.booth1_input,
        paths.booth2_input,
        paths.adjacency_rules_path,
    ]
    output_files = [
        paths.global_line_output,
        paths.manifest_path,
        paths.report_path,
    ]

    write_manifest(paths, cfg, metrics, input_files, output_files, start_time, end_time)
    write_report(paths, cfg, metrics)


if __name__ == "__main__":
    run_step05()
