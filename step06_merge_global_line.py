#!/usr/bin/env python3
"""
Step-05/06 — Oven Convoy Serialization (2→1 Merge with Global Adjacency)

This script can merge either:
  • Step-04 SDSU per-booth outputs (legacy Step-05 mode), or
  • Step-06 MILP per-booth outputs (new Step-06 mode)

Key settings:
  - c2_mode:
      * "symbol_stream": adjacency evaluated on the full symbol stream.
                         "EMPTY" is a real symbol and can break deadlocks.
      * "parts_only":    adjacency evaluated on parts-only; tokens never waive C2.
  - allow_postpaint_empty: if True and c2_mode=="symbol_stream", we may insert
                          EMPTY to resolve deadlocks. (Ignored for parts_only.)

Paths are read from steps_config.toml:
  - Under [step06] when --source-step=step06
  - Fallback to [step05] keys if missing.

This implementation is deterministic and emits a manifest/report per the step-gating protocol.

Run with: `py .\\step06_merge_global_line.py --source-step step06`
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
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


def ensure_booth_id(df: pd.DataFrame, booth_id_value: int, booth_col_name: str = "booth_id") -> pd.DataFrame:
    """
    Ensure a 'booth_id' column exists; synthesize it if missing.
    """
    if pick_first_existing(df.columns, [booth_col_name]) is None:
        df = df.copy()
        df[booth_col_name] = booth_id_value
    return df


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class StepPaths:
    step_id: str
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
class MergeConfig:
    allow_postpaint_empty: bool
    tie_break_policy: str
    adjacency_enabled: bool
    scope_escalation: str
    empty_units_postpaint_ratio: float
    c2_mode: str  # "symbol_stream" or "parts_only"


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

def _get_step_cfg(steps_cfg: dict, step_id: str) -> dict:
    # Fallback to step05 section if step_id missing
    return steps_cfg.get(step_id, steps_cfg.get("step05", {}))


def resolve_paths(global_cfg: dict, steps_cfg: dict, step_id: str) -> StepPaths:
    paths_cfg = global_cfg["paths"]
    data_files = global_cfg["data_files"]
    step_cfg = _get_step_cfg(steps_cfg, step_id)
    step05_fallback = _get_step_cfg(steps_cfg, "step05")

    def _rp(key: str) -> str:
        if key in step_cfg:
            return step_cfg[key]
        if key in step05_fallback:
            return step05_fallback[key]
        raise KeyError(f"Missing relpath '{key}' in steps_config for [{step_id}] and [step05] fallback.")

    dataset_dir = Path(paths_cfg["dataset_dir"])
    output_dir = Path(paths_cfg["output_dir"])

    return StepPaths(
        step_id=step_id,
        output_dir=output_dir,
        dataset_dir=dataset_dir,
        booth1_input=output_dir / _rp("booth1_input_relpath"),
        booth2_input=output_dir / _rp("booth2_input_relpath"),
        global_line_output=output_dir / _rp("global_line_relpath"),
        manifest_path=output_dir / _rp("manifest_relpath"),
        report_path=output_dir / _rp("report_relpath"),
        adjacency_rules_path=dataset_dir / data_files["adjacency_rules"],
        global_config_path=Path("global_config.toml"),
        steps_config_path=Path("steps_config.toml"),
    )


def build_merge_config(global_cfg: dict, steps_cfg: dict, step_id: str, cli_c2_mode: Optional[str]) -> MergeConfig:
    adj_cfg = global_cfg.get("adjacency", {})
    tokens_cfg = global_cfg.get("tokens", {})

    step_cfg = _get_step_cfg(steps_cfg, step_id)

    allow_postpaint_empty = bool(step_cfg.get("allow_postpaint_empty", step_id == "step06"))
    tie_break_policy = str(step_cfg.get("tie_break_policy", "longer_remaining")).lower()

    adjacency_enabled = bool(adj_cfg.get("enabled", True))
    scope_escalation = str(adj_cfg.get("scope_escalation", "global_if_missing"))
    empty_units_postpaint_ratio = float(tokens_cfg.get("empty_units_postpaint_ratio", 1.0))

    # Default C2 mode by step: step06→symbol_stream, step05→parts_only (can be overridden)
    default_c2 = "symbol_stream" if step_id == "step06" else "parts_only"
    c2_mode = (cli_c2_mode or step_cfg.get("c2_mode") or default_c2).lower()
    if c2_mode not in ("symbol_stream", "parts_only"):
        raise ValueError(f"Unsupported c2_mode={c2_mode!r}; use 'symbol_stream' or 'parts_only'.")

    if tie_break_policy not in ("longer_remaining", "round_robin"):
        raise ValueError(f"Unsupported tie_break_policy={tie_break_policy!r}")

    return MergeConfig(
        allow_postpaint_empty=allow_postpaint_empty,
        tie_break_policy=tie_break_policy,
        adjacency_enabled=adjacency_enabled,
        scope_escalation=scope_escalation,
        empty_units_postpaint_ratio=empty_units_postpaint_ratio,
        c2_mode=c2_mode,
    )


def resolve_schema(global_cfg: dict, df: pd.DataFrame) -> SchemaConfig:
    ds = global_cfg.get("data_schema", {})
    demands_schema = ds.get("demands", {})
    parts_schema = ds.get("parts", {})

    part_syn = demands_schema.get("part_id_synonyms", []) or parts_schema.get("part_id_synonyms", [])
    booth_syn = ["booth_id"]

    part_col = pick_first_existing(df.columns, part_syn + ["part_id"])
    booth_col = pick_first_existing(df.columns, booth_syn)

    if part_col is None:
        raise KeyError(f"Could not resolve part_id column in merge inputs; candidates={part_syn + ['part_id']}")
    if booth_col is None:
        # resolve_schema is called after ensure_booth_id; this should not happen
        raise KeyError("Could not find or synthesize 'booth_id' column in merge inputs.")

    return SchemaConfig(part_id_col=part_col, booth_id_col=booth_col)


# ---------------------------------------------------------------------------
# Adjacency model (global line)
# ---------------------------------------------------------------------------

def load_adjacency_model_global(
    paths: StepPaths,
    global_cfg: dict,
    cfg: MergeConfig,
) -> AdjacencyModel:
    if not cfg.adjacency_enabled:
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

        applies = False
        if scope_val in ("global", "across_skids"):
            applies = True
        elif not scope_val and cfg.scope_escalation == "global_if_missing":
            applies = True

        if not applies:
            continue

        forbid_pairs.add((str(r[from_col]), str(r[to_col])))

    return AdjacencyModel(forbid_pairs=forbid_pairs)


# ---------------------------------------------------------------------------
# Merge algorithm
# ---------------------------------------------------------------------------

def merge_booth_streams(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    schema: SchemaConfig,
    adj_model: AdjacencyModel,
    cfg: MergeConfig,
) -> Tuple[pd.DataFrame, MergeMetrics]:
    """
    Merge per-booth JOB sequences (PB1, PB2) into a single oven stream.

    c2_mode == "symbol_stream":
        - Adjacency is checked on the full symbol stream.
        - Symbols: "B1", "B2", "EMPTY".
        - "EMPTY" is a real symbol and may be inserted (if allowed) to resolve deadlocks.

    c2_mode == "parts_only":
        - Adjacency is checked on the parts-only projection.
        - Tokens never waive adjacency; "EMPTY" (if present) does not reset predecessor.
        - In this mode, when both heads violate C2, we raise (no repair).
    """

    n1, n2 = len(df1), len(df2)
    h1 = h2 = 0

    forbid_pairs = adj_model.forbid_pairs
    adjacency_enabled = cfg.adjacency_enabled and bool(forbid_pairs)
    tie_break_policy = cfg.tie_break_policy
    parts_only = (cfg.c2_mode == "parts_only")

    out_stream: List[Tuple[str, int]] = []  # ("B1"/"B2"/"EMPTY", local_idx)
    last_symbol: Optional[str] = None        # last symbol's part_id or "EMPTY"
    last_part: Optional[str] = None          # last real part_id (unchanged by EMPTY)
    last_booth_kind: Optional[str] = None    # for round_robin

    n_conflict_events = 0
    n_deadlocks = 0
    n_postpaint_empty = 0

    def _violates(pred: Optional[str], cand_part: str) -> bool:
        if not adjacency_enabled or pred is None:
            return False
        return (pred, cand_part) in forbid_pairs

    while h1 < n1 or h2 < n2:
        candidates: List[Tuple[str, int, str, bool]] = []

        # PB1 candidate
        if h1 < n1:
            p1 = str(df1.iloc[h1][schema.part_id_col])
            pred = last_part if parts_only else last_symbol
            feas1 = not _violates(pred, p1)
            if not feas1:
                n_conflict_events += 1
            candidates.append(("B1", h1, p1, feas1))

        # PB2 candidate
        if h2 < n2:
            p2 = str(df2.iloc[h2][schema.part_id_col])
            pred = last_part if parts_only else last_symbol
            feas2 = not _violates(pred, p2)
            if not feas2:
                n_conflict_events += 1
            candidates.append(("B2", h2, p2, feas2))

        feasible = [c for c in candidates if c[3]]

        if not feasible:
            n_deadlocks += 1
            if not parts_only and cfg.allow_postpaint_empty:
                # Insert EMPTY as a real symbol that resets the stream predecessor
                out_stream.append(("EMPTY", -1))
                last_symbol = "EMPTY"
                # NOTE: last_part remains unchanged in symbol_stream mode; it is unused for checks.
                last_booth_kind = None
                n_postpaint_empty += 1
                continue
            else:
                raise RuntimeError(
                    f"Merge deadlock under C2 ({cfg.c2_mode}); "
                    f"no admissible head from PB1/PB2 and EMPTY repair "
                    f"{'disabled' if parts_only or not cfg.allow_postpaint_empty else 'not allowed in parts_only'}."
                )

        # Choose candidate according to tie_break_policy
        if len(feasible) == 1:
            chosen = feasible[0]
        else:
            if tie_break_policy == "longer_remaining":
                rem1 = n1 - h1 if any(k == "B1" for (k, _, _, _) in feasible) else -1
                rem2 = n2 - h2 if any(k == "B2" for (k, _, _, _) in feasible) else -1
                prefer = "B1" if rem1 > rem2 else ("B2" if rem2 > rem1 else "B1")
                chosen = next(c for c in feasible if c[0] == prefer)
            else:  # round_robin
                preferred = "B1" if (last_booth_kind is None or last_booth_kind == "B2") else "B2"
                if any(c[0] == preferred for c in feasible):
                    chosen = next(c for c in feasible if c[0] == preferred)
                else:
                    other = "B1" if preferred == "B2" else "B2"
                    chosen = next(c for c in feasible if c[0] == other)

        kind, local_idx, part_id, _ = chosen
        out_stream.append((kind, local_idx))
        last_symbol = part_id
        if kind in ("B1", "B2"):
            last_booth_kind = kind
            last_part = part_id    # only advance last_part on real jobs
            if kind == "B1":
                h1 += 1
            else:
                h2 += 1
        else:
            # EMPTY cannot be chosen here
            raise RuntimeError(f"Internal error: unexpected kind={kind!r} in chosen candidate")

    # ----------------------------------------------------------------------
    # Build output DataFrame; preserve EMPTY rows when inserted
    # ----------------------------------------------------------------------
    records: List[pd.Series] = []
    base_cols = list(df1.columns) if len(df1) > 0 else list(df2.columns)

    for kind, local_idx in out_stream:
        if kind == "B1":
            row = df1.iloc[local_idx].copy()
        elif kind == "B2":
            row = df2.iloc[local_idx].copy()
        elif kind == "EMPTY":
            row = pd.Series({col: None for col in base_cols})
            row[schema.part_id_col] = "EMPTY"
            row[schema.booth_id_col] = "EMPTY"
        else:
            raise RuntimeError(f"Internal error: unexpected kind {kind!r}")
        records.append(row)

    global_df = pd.DataFrame(records).reset_index(drop=True)
    global_df["global_seq_index"] = list(range(1, len(global_df) + 1))

    # Compute max run length per booth, ignoring EMPTY
    max_run_b1 = max_run_b2 = 0
    cur_run_b1 = cur_run_b2 = 0
    for kind, _ in out_stream:
        if kind == "B1":
            cur_run_b1 += 1; cur_run_b2 = 0
        elif kind == "B2":
            cur_run_b2 += 1; cur_run_b1 = 0
        else:
            cur_run_b1 = cur_run_b2 = 0
        max_run_b1 = max(max_run_b1, cur_run_b1)
        max_run_b2 = max(max_run_b2, cur_run_b2)

    metrics = MergeMetrics(
        n_jobs_total=len(global_df),
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
# Final adjacency verification
# ---------------------------------------------------------------------------

def verify_global_adjacency(
    global_df: pd.DataFrame,
    schema: SchemaConfig,
    adj_model: AdjacencyModel,
    cfg: MergeConfig,
) -> None:
    """
    Safety check according to c2_mode.

    - symbol_stream: verify consecutive symbols (A->B) are not forbidden; "EMPTY" never appears
                     in forbid_pairs, so (EMPTY, x) is allowed.
    - parts_only:    verify consecutive *parts* (skipping EMPTY rows) are not forbidden.
    """
    forbid_pairs = adj_model.forbid_pairs
    if not forbid_pairs:
        return

    part_col = schema.part_id_col
    stream = [str(p) for p in global_df[part_col].tolist()]

    if cfg.c2_mode == "symbol_stream":
        for i in range(len(stream) - 1):
            a, b = stream[i], stream[i + 1]
            if (a, b) in forbid_pairs:
                raise RuntimeError(
                    f"Verification (symbol_stream): adjacency violation at positions {i}->{i+1}: ({a}, {b})"
                )
    else:  # parts_only
        prev = None
        for i, s in enumerate(stream):
            if s == "EMPTY" or s is None or s == "":
                continue
            if prev is not None and (prev, s) in forbid_pairs:
                raise RuntimeError(
                    f"Verification (parts_only): adjacency violation at parts indices "
                    f"(..., {prev}) -> ({s}) around row {i}"
                )
            prev = s


# ---------------------------------------------------------------------------
# Manifest and report
# ---------------------------------------------------------------------------

def write_manifest(
    paths: StepPaths,
    cfg: MergeConfig,
    metrics: MergeMetrics,
    input_files: List[Path],
    output_files: List[Path],
    start_time: float,
    end_time: float,
) -> None:
    ensure_parent_dir(paths.manifest_path)
    manifest = {
        "step": f"{paths.step_id}_merge_global_line",
        "version": "1.1",
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


def write_report(paths: StepPaths, cfg: MergeConfig, metrics: MergeMetrics) -> None:
    ensure_parent_dir(paths.report_path)
    lines: List[str] = []

    lines.append(f"# {paths.step_id.upper()} — Oven Convoy Serialization (2→1 Merge)")
    lines.append("")
    lines.append("## Purpose / What / Why")
    lines.append("")
    lines.append("- **Purpose.** Merge per-booth sequences (PB1, PB2) into a single oven convoy "
                 "while enforcing global adjacency forbids.")
    lines.append("- **What.** Deterministic greedy 2→1 interleave, using adjacency_rules.csv for C2(global).")
    lines.append(f"- **C2 Mode.** {cfg.c2_mode} "
                 "(symbol_stream: EMPTY may break deadlocks; parts_only: tokens never waive C2).")
    lines.append("")
    lines.append("## Effective Settings")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(dataclasses.asdict(cfg), indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Merge Metrics")
    lines.append("")
    lines.append(f"- Total stream rows (incl. EMPTY if inserted): {metrics.n_jobs_total}")
    lines.append(f"- Jobs from PB1: {metrics.n_jobs_booth1}")
    lines.append(f"- Jobs from PB2: {metrics.n_jobs_booth2}")
    lines.append(f"- Conflict rejections: {metrics.n_conflict_events}")
    lines.append(f"- Deadlocks encountered: {metrics.n_deadlocks}")
    lines.append(f"- Post-paint EMPTY inserted: {metrics.n_postpaint_empty}")
    lines.append(f"- Max consecutive PB1 jobs: {metrics.max_run_booth1}")
    lines.append(f"- Max consecutive PB2 jobs: {metrics.max_run_booth2}")
    lines.append("")

    with paths.report_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="STEP-05/06 Merge Global Line")
    parser.add_argument("--source-step", choices=("step05", "step06"), default="step06",
                        help="Which step's per-booth outputs to merge (default: step06).")
    parser.add_argument("--c2-mode", choices=("symbol_stream", "parts_only"), default=None,
                        help="Override C2 mode. Default depends on --source-step.")
    args = parser.parse_args(argv)

    global_cfg_path = Path("global_config.toml")
    steps_cfg_path = Path("steps_config.toml")
    if not global_cfg_path.exists():
        raise FileNotFoundError("global_config.toml not found")
    if not steps_cfg_path.exists():
        raise FileNotFoundError("steps_config.toml not found")

    global_cfg = load_toml(global_cfg_path)
    steps_cfg = load_toml(steps_cfg_path)

    paths = resolve_paths(global_cfg, steps_cfg, args.source_step)
    cfg = build_merge_config(global_cfg, steps_cfg, args.source_step, args.c2_mode)

    print(f"STEP: {paths.step_id} — Merging global line from PB1/PB2")

    # Load inputs and normalize schema
    if not paths.booth1_input.exists():
        raise FileNotFoundError(f"Merge input not found (PB1): {paths.booth1_input}")
    if not paths.booth2_input.exists():
        raise FileNotFoundError(f"Merge input not found (PB2): {paths.booth2_input}")

    df1 = pd.read_csv(paths.booth1_input)
    df2 = pd.read_csv(paths.booth2_input)

    # Synthesize booth_id if MILP outputs don't carry it
    df1 = ensure_booth_id(df1, booth_id_value=1, booth_col_name="booth_id")
    df2 = ensure_booth_id(df2, booth_id_value=2, booth_col_name="booth_id")

    schema = resolve_schema(global_cfg, df1)
    adj_model = load_adjacency_model_global(paths, global_cfg, cfg)

    start_time = time.time()
    global_df, metrics = merge_booth_streams(df1, df2, schema, adj_model, cfg)

    # Verify C2 according to mode
    verify_global_adjacency(global_df, schema, adj_model, cfg)
    print(f"Verification ({cfg.c2_mode}): no global adjacency violations detected.")

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
    main()
