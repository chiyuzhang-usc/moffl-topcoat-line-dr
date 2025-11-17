#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP-08 — DES evaluator with EMPTY-as-adjacency-reset

This version evaluates fixed JOB/TOKEN streams produced upstream (Step-05/06/07),
treating EMPTY tokens as **adjacency resets with cost ratio**. Concretely:

- Global C2 (parts-only) is enforced across *jobs*; EMPTY resets the adjacency state
  (A → EMPTY → B is allowed even if A→B is forbidden).
- EMPTY consumes oven time = oven_time_per_job * empty_time_ratio,
  counted as **oven idle** (not busy).
- No resequencing; convoy order is taken as-is.
- FLUSH (paint purge) remains a booth-local option via SDST kernel (for f₁/f₄ time realism).

Inputs resolved from configs:
- Step-05: global merged line (may include EMPTY tokens).
- Step-06: MILP global convoy (preferred).
- Step-07: one or more GLOBAL files; supports multi-solution CSVs with `solution_id`.

Outputs:
- step08_des_metrics.csv
- manifests/step08_manifest.json
- reports/step08_report.md
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

try:
    import tomllib  # py311+
except Exception:
    import tomli as tomllib  # type: ignore


# -----------------------------------------------------------------------------
# Logging & IO utils
# -----------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[STEP08] {msg}")
    sys.stdout.flush()


def ensure_parent_dir(path: Path) -> None:
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def sha256_of_file(path: Path) -> Optional[str]:
    import hashlib
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_toml(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def detect_column(df: pd.DataFrame, candidates: Sequence[str], required: bool = True) -> Optional[str]:
    """
    Return first present column (case-insensitive) among candidates. Raise if required and none.
    """
    lut = {c.lower(): c for c in df.columns}
    for cand in candidates:
        col = lut.get(str(cand).lower())
        if col:
            return col
    if required:
        raise KeyError(f"None of {candidates} found in {list(df.columns)}")
    return None


# -----------------------------------------------------------------------------
# Tokens (EMPTY etc.)
# -----------------------------------------------------------------------------

TOKEN_STRINGS = {
    "EMPTY", "TOKEN_EMPTY", "EMPTY_TOKEN", "TOKEN", "FLUSH", "NONE"
}

def is_token_jobid(val) -> bool:
    if val is None:
        return True
    s = str(val).strip()
    if s == "":
        return True
    u = s.upper()
    return u == "NAN" or u in TOKEN_STRINGS


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------

@dataclass
class Step08Paths:
    dataset_dir: Path
    output_dir: Path

    # dataset
    adjacency_rules_path: Path
    s_matrix_path: Path
    topcoat_codebook_path: Path
    topcoat_rules_path: Path

    # step-03 (catalog)
    assignment_path: Path
    booth1_step03_path: Path
    booth2_step03_path: Path

    # step-05 (baseline)
    step05_global_path: Path

    # step-06 (MILP)
    step06_milp_global_path: Path  # e.g., output_dir / (milp_prefix + 'global.csv')

    # step-07 (NSGA exports or consolidated file)
    step07_exports_dir: Optional[Path]
    step07_consolidated_path: Optional[Path]  # e.g., output_dir / 'step07_global_line.csv'

    # outputs
    metrics_path: Path
    manifest_path: Path
    report_path: Path

    # config files
    global_config_path: Path
    steps_config_path: Path


@dataclass
class Step08Settings:
    # Paint & oven timing
    base_paint_time: float
    time_per_setup_unit: float
    oven_time_per_job: float

    # EMPTY handling
    empty_time_ratio: float  # EMPTY time = oven_time_per_job * ratio

    # Paint purge (booth-local FLUSH)
    allow_purge: bool
    flush_cost_units: float

    # Step-05 parity (used only if we need to re-merge; here we consume given convoy)
    tie_break_policy: str

    # Scenario toggles
    run_step05: bool
    run_step06: bool
    run_step07: bool

    # Optional scalarization for f4
    f4_weights: Optional[Dict[str, float]]  # {"pb_block_frac_max": α, "oven_idle_frac": β, "makespan": γ}


@dataclass
class JobRecord:
    job_id: str
    feeder_index: int
    part_id: str
    topcoat: str
    base_booth: int  # 1 or 2


@dataclass
class BoothSchedule:
    finish_time_by_job: Dict[int, float]
    busy_time: float


@dataclass
class DesMetricsRow:
    method: str
    scenario: str
    solution_id: Optional[str]
    seed: Optional[int]
    throughput_jobs: int
    tokens_empty: int
    makespan: float
    pb1_busy_time: float
    pb1_blocked_time: float
    pb1_idle_time: float
    pb2_busy_time: float
    pb2_blocked_time: float
    pb2_idle_time: float
    oven_busy_time: float
    oven_idle_time: float
    pb_block_frac_max: float
    oven_idle_frac: float
    f4: Optional[float]


# -----------------------------------------------------------------------------
# Dataset loaders
# -----------------------------------------------------------------------------

def load_s_matrix_transition(s_matrix_path: Path, global_cfg: dict) -> Dict[Tuple[str, str], float]:
    schema = global_cfg.get("data_schema", {}).get("s_matrix_transition", {})
    df = pd.read_csv(s_matrix_path)
    from_col = detect_column(df, schema.get("from_code_synonyms", ["from"]))
    costs: Dict[Tuple[str, str], float] = {}
    for _, row in df.iterrows():
        from_code = str(row[from_col])
        for col in df.columns:
            if col == from_col:
                continue
            raw = row[col]
            val = math.inf
            if isinstance(raw, str):
                if raw.strip().lower() != "inf":
                    try:
                        val = float(raw)
                    except ValueError:
                        val = math.inf
            else:
                try:
                    val = float(raw)
                except (TypeError, ValueError):
                    val = math.inf
            costs[(from_code, str(col))] = val
    return costs


def load_topcoat_codebook(codebook_path: Path, global_cfg: dict) -> Dict[int, str]:
    schema = global_cfg.get("data_schema", {}).get("topcoat_codebook", {})
    df = pd.read_csv(codebook_path)
    no_col = detect_column(df, schema.get("topcoat_no_synonyms", ["topcoat_no"]))
    code_col = detect_column(df, schema.get("topcoat_code_synonyms", ["topcoat_code", "color_code"]))
    m: Dict[int, str] = {}
    for _, row in df.iterrows():
        try:
            m[int(row[no_col])] = str(row[code_col])
        except (TypeError, ValueError):
            continue
    return m


def load_topcoat_forbids(rules_path: Path, code_by_no: Dict[int, str], global_cfg: dict) -> set:
    if not rules_path.exists():
        return set()
    schema = global_cfg.get("data_schema", {}).get("topcoat_rules", {})
    df = pd.read_csv(rules_path)
    rel_col = detect_column(df, schema.get("relation_synonyms", ["relation"]), required=False)
    from_code_col = detect_column(df, schema.get("from_topcoat_code_synonyms", ["from_topcoat_code", "from_code"]), required=False)
    to_code_col = detect_column(df, schema.get("to_topcoat_code_synonyms", ["to_topcoat_code", "to_code"]), required=False)
    from_no_col = detect_column(df, schema.get("from_topcoat_no_synonyms", ["from_topcoat_no"]), required=(from_code_col is None))
    to_no_col = detect_column(df, schema.get("to_topcoat_no_synonyms", ["to_topcoat_no"]), required=(to_code_col is None))
    forbids: set = set()
    for _, row in df.iterrows():
        relation = str(row[rel_col]).lower() if rel_col else ""
        if "forbid" not in relation:
            continue
        try:
            from_code = str(row[from_code_col]) if from_code_col else code_by_no.get(int(row[from_no_col]), "")
            to_code = str(row[to_code_col]) if to_code_col else code_by_no.get(int(row[to_no_col]), "")
        except (TypeError, ValueError):
            continue
        if from_code and to_code:
            forbids.add((from_code, to_code))
    return forbids


def build_adjacency_sets(adjacency_path: Path, global_cfg: dict) -> Tuple[set, set]:
    schema = global_cfg.get("data_schema", {}).get("adjacency_rules", {})
    df = pd.read_csv(adjacency_path)
    from_col = detect_column(df, schema.get("from_part_id_synonyms", ["from_part_id"]))
    to_col = detect_column(df, schema.get("to_part_id_synonyms", ["to_part_id"]))
    scope_col = detect_column(df, schema.get("scope_synonyms", ["scope"]), required=False)
    rel_col = detect_column(df, schema.get("relation_synonyms", ["relation"]), required=False)
    forb_in_booth, forb_global = set(), set()
    escalation = global_cfg.get("adjacency", {}).get("scope_escalation", "global_if_missing").lower()
    for _, row in df.iterrows():
        relation = str(row[rel_col]).lower() if rel_col else ""
        if "forbid" not in relation:
            continue
        f = str(row[from_col])
        t = str(row[to_col])
        scope = str(row[scope_col]).lower() if scope_col else ""
        if scope in ("in_booth", "inbooth", "booth"):
            forb_in_booth.add((f, t))
        elif scope in ("global", "merge", "line", "across_skids"):
            forb_global.add((f, t))
        else:
            if escalation == "global_if_missing":
                forb_global.add((f, t))
            elif escalation == "in_booth_if_missing":
                forb_in_booth.add((f, t))
            else:
                forb_global.add((f, t))
    return forb_in_booth, forb_global


# -----------------------------------------------------------------------------
# Paint kernel (SDST + purge)
# -----------------------------------------------------------------------------

class PaintFeasibilityError(RuntimeError):
    pass


@dataclass
class PaintCostKernel:
    s_matrix: Dict[Tuple[str, str], float]
    forbidden_pairs: set
    allow_purge: bool
    flush_cost_units: float

    def transition_units(self, prev_color: Optional[str], next_color: str) -> float:
        if prev_color is None:
            return 0.0
        if (prev_color, next_color) in self.forbidden_pairs:
            raise PaintFeasibilityError(f"Paint forbid {prev_color}->{next_color}")
        direct = self.s_matrix.get((prev_color, next_color), math.inf)
        purge = self.flush_cost_units if self.allow_purge else math.inf
        if math.isinf(direct) and math.isinf(purge):
            raise PaintFeasibilityError(f"Infeasible transition {prev_color}->{next_color}")
        if math.isinf(direct):
            return purge
        if math.isinf(purge):
            return direct
        return min(direct, purge)


# -----------------------------------------------------------------------------
# Job catalog (Step-03)
# -----------------------------------------------------------------------------

def build_job_catalog(global_cfg: dict, paths: Step08Paths) -> Tuple[List[JobRecord], Dict[str, int]]:
    assignment = pd.read_csv(paths.assignment_path).sort_values("feeder_job_index")
    b1 = pd.read_csv(paths.booth1_step03_path)
    b2 = pd.read_csv(paths.booth2_step03_path)

    schema_dem = global_cfg.get("data_schema", {}).get("demands", {})
    part_col1 = detect_column(b1, schema_dem.get("part_id_synonyms", ["part_id"]))
    topcoat_col1 = detect_column(b1, schema_dem.get("topcoat_code_synonyms", ["topcoat_code", "color_code"]))
    job_col1 = detect_column(b1, schema_dem.get("job_id_synonyms", ["job_id", "job", "jobid"]))

    part_col2 = detect_column(b2, schema_dem.get("part_id_synonyms", ["part_id"]))
    topcoat_col2 = detect_column(b2, schema_dem.get("topcoat_code_synonyms", ["topcoat_code", "color_code"]))
    job_col2 = detect_column(b2, schema_dem.get("job_id_synonyms", ["job_id", "job", "jobid"]))

    job_id_col = detect_column(assignment, ["job_id", "job", "jobid"])
    feeder_col = detect_column(assignment, ["feeder_job_index", "feeder_index"])
    # accept booth_id as base-booth
    base_booth_col = detect_column(assignment, ["base_booth", "booth", "pb", "booth_id"], required=False)

    b1_ids = set(b1[job_col1].astype(str))
    b2_ids = set(b2[job_col2].astype(str))

    meta: Dict[str, Tuple[str, str]] = {}
    for _, row in b1.iterrows():
        jid = str(row[job_col1]); meta[jid] = (str(row[part_col1]), str(row[topcoat_col1]))
    for _, row in b2.iterrows():
        jid = str(row[job_col2])
        if jid not in meta:
            meta[jid] = (str(row[part_col2]), str(row[topcoat_col2]))

    job_records: List[JobRecord] = []
    job_index_by_id: Dict[str, int] = {}

    for _, row in assignment.iterrows():
        jid = str(row[job_id_col])
        feeder_idx = int(row[feeder_col])
        if base_booth_col is not None:
            try:
                base_booth = int(row[base_booth_col])
            except Exception:
                base_booth = 1 if jid in b1_ids else (2 if jid in b2_ids else 1)
        else:
            if jid in b1_ids:
                base_booth = 1
            elif jid in b2_ids:
                base_booth = 2
            else:
                raise KeyError(f"Cannot infer base_booth for job_id={jid}")

        part_id, topcoat = meta[jid]
        job_index_by_id[jid] = len(job_records)
        job_records.append(JobRecord(
            job_id=jid,
            feeder_index=feeder_idx,
            part_id=part_id,
            topcoat=topcoat,
            base_booth=base_booth,
        ))
    return job_records, job_index_by_id


# -----------------------------------------------------------------------------
# Convoy loaders (with EMPTY)
# -----------------------------------------------------------------------------

ConvoyItem = Union[Tuple[str, int], str]  # ('PB1', jidx) / ('PB2', jidx) / 'TOKEN'

def parse_convoy_df_rows(
    df: pd.DataFrame,
    job_index_by_id: Dict[str, int],
    job_records: List[JobRecord],
) -> List[ConvoyItem]:
    job_col = detect_column(df, ["job_id", "job", "jobid"])
    booth_col = detect_column(df, ["booth_label", "booth", "booth_id", "pb"], required=False)

    convoy: List[ConvoyItem] = []
    for _, row in df.iterrows():
        raw_jid = row.get(job_col, None)
        if is_token_jobid(raw_jid):
            convoy.append("TOKEN")
            continue
        jid = str(raw_jid)
        if jid not in job_index_by_id:
            raise KeyError(f"Unknown job_id in convoy: {jid}")
        jidx = job_index_by_id[jid]
        if booth_col:
            booth_raw = str(row[booth_col]).strip().upper()
            if booth_raw in ("PB1", "1", "B1"):
                booth_tag = "PB1"
            elif booth_raw in ("PB2", "2", "B2"):
                booth_tag = "PB2"
            else:
                booth_tag = "PB1" if job_records[jidx].base_booth == 1 else "PB2"
        else:
            booth_tag = "PB1" if job_records[jidx].base_booth == 1 else "PB2"
        convoy.append((booth_tag, jidx))
    return convoy


def load_global_convoy_csv(path: Path,
                           job_index_by_id: Dict[str, int],
                           job_records: List[JobRecord]) -> List[ConvoyItem]:
    df = pd.read_csv(path)
    return parse_convoy_df_rows(df, job_index_by_id, job_records)


def iter_step07_solutions_from_csv(
    csv_path: Path,
    job_index_by_id: Dict[str, int],
    job_records: List[JobRecord],
) -> List[Tuple[str, Optional[int], str, List[ConvoyItem]]]:
    """
    Parse a step07_global_line.csv containing multiple solutions.
    Returns list of (scenario_name, seed, solution_id, convoy).
    """
    df = pd.read_csv(csv_path)

    def col(df, names, required=True):
        return detect_column(df, names if isinstance(names, (list, tuple)) else [names], required=required)

    sid_col = col(df, ["solution_id"])
    pos_col = col(df, ["global_position", "position", "pos"])
    seed_col = col(df, ["seed"], required=False)

    out: List[Tuple[str, Optional[int], str, List[ConvoyItem]]] = []
    for sid, g in df.groupby(sid_col, sort=True):
        g = g.sort_values(pos_col, kind="mergesort")
        convoy = parse_convoy_df_rows(g, job_index_by_id, job_records)

        seed_val: Optional[int] = None
        if seed_col and g[seed_col].nunique(dropna=False) == 1:
            try:
                seed_val = int(g[seed_col].iloc[0])
            except Exception:
                seed_val = None

        scenario_name = f"{csv_path.stem}::sid{sid}"
        out.append((scenario_name, seed_val, str(sid), convoy))
    return out


# -----------------------------------------------------------------------------
# Global C2 verification WITH EMPTY-as-reset
# -----------------------------------------------------------------------------

def first_c2_violation_with_tokens(convoy: List[ConvoyItem],
                                   job_records: List[JobRecord],
                                   forb_global: set) -> Optional[Dict[str, object]]:
    last_part: Optional[str] = None
    last_job_id: Optional[str] = None
    for k, item in enumerate(convoy):
        if item == "TOKEN":
            last_part = None  # reset adjacency
            last_job_id = None
            continue
        booth_tag, jidx = item  # type: ignore
        p = job_records[jidx].part_id
        if last_part is not None and (last_part, p) in forb_global:
            return {
                "pos": k,
                "prev_part": last_part,
                "curr_part": p,
                "prev_job_id": last_job_id,
                "curr_job_id": job_records[jidx].job_id,
            }
        last_part = p
        last_job_id = job_records[jidx].job_id
    return None


# -----------------------------------------------------------------------------
# Booth timing & DES (no buffer; EMPTY consumes idle time)
# -----------------------------------------------------------------------------

def compute_booth_schedule(
    seq: List[int],
    job_records: List[JobRecord],
    kernel: PaintCostKernel,
    base_paint_time: float,
    time_per_setup_unit: float,
) -> BoothSchedule:
    t = 0.0
    busy = 0.0
    prev_color: Optional[str] = None
    finish: Dict[int, float] = {}
    for jidx in seq:
        color = job_records[jidx].topcoat
        units = kernel.transition_units(prev_color, color)
        dur = base_paint_time + units * time_per_setup_unit
        t += dur
        busy += dur
        finish[jidx] = t
        prev_color = color
    return BoothSchedule(finish_time_by_job=finish, busy_time=busy)


def simulate_convoy_with_tokens(
    convoy: List[ConvoyItem],
    booth1_sched: BoothSchedule,
    booth2_sched: BoothSchedule,
    oven_time_per_job: float,
    empty_time_ratio: float,
) -> Tuple[int, float, float, float, float, float]:
    """
    Deterministic timeline:
      - Jobs: start at max(booth_finish, oven_ready); add oven_time_per_job.
      - EMPTY: start at oven_ready; add oven_time_per_job * empty_time_ratio; counts as oven idle.
    Returns: (count_jobs, makespan, pb1_blocked, pb2_blocked, oven_busy, oven_idle)
    """
    oven_t = 0.0
    pb1_blocked = 0.0
    pb2_blocked = 0.0
    jobs = 0
    oven_busy = 0.0
    oven_idle = 0.0

    for item in convoy:
        if item == "TOKEN":
            empty_dur = oven_time_per_job * empty_time_ratio
            start = oven_t
            oven_t = start + empty_dur
            oven_idle += empty_dur  # EMPTY consumes idle time
            continue

        booth_tag, jidx = item  # type: ignore
        ready = booth1_sched.finish_time_by_job[jidx] if booth_tag == "PB1" else booth2_sched.finish_time_by_job[jidx]
        start = max(ready, oven_t)
        # blocked time at booth
        if booth_tag == "PB1":
            pb1_blocked += max(0.0, start - ready)
        else:
            pb2_blocked += max(0.0, start - ready)
        oven_t = start + oven_time_per_job
        oven_busy += oven_time_per_job
        jobs += 1

    makespan = oven_t
    return jobs, makespan, pb1_blocked, pb2_blocked, oven_busy, oven_idle


# -----------------------------------------------------------------------------
# Scenario runners
# -----------------------------------------------------------------------------

def build_paint_kernel(paths: Step08Paths, global_cfg: dict, settings: Step08Settings) -> PaintCostKernel:
    s_matrix = load_s_matrix_transition(paths.s_matrix_path, global_cfg)
    codebook = load_topcoat_codebook(paths.topcoat_codebook_path, global_cfg)
    forbids = load_topcoat_forbids(paths.topcoat_rules_path, codebook, global_cfg)
    return PaintCostKernel(
        s_matrix=s_matrix,
        forbidden_pairs=forbids,
        allow_purge=settings.allow_purge,
        flush_cost_units=settings.flush_cost_units,
    )


def run_scenario_from_convoy(
    method: str,
    scenario: str,
    solution_id: Optional[str],
    seed: Optional[int],
    convoy: List[ConvoyItem],
    job_records: List[JobRecord],
    kernel: PaintCostKernel,
    settings: Step08Settings,
) -> DesMetricsRow:
    # Build booth schedules from convoy content
    b1_seq = [j for it in convoy if it != "TOKEN" for (b, j) in [it] if b == "PB1"]  # type: ignore
    b2_seq = [j for it in convoy if it != "TOKEN" for (b, j) in [it] if b == "PB2"]  # type: ignore

    b1_sched = compute_booth_schedule(b1_seq, job_records, kernel, settings.base_paint_time, settings.time_per_setup_unit)
    b2_sched = compute_booth_schedule(b2_seq, job_records, kernel, settings.base_paint_time, settings.time_per_setup_unit)

    jobs, makespan, pb1_blocked, pb2_blocked, oven_busy, oven_idle = simulate_convoy_with_tokens(
        convoy=convoy,
        booth1_sched=b1_sched,
        booth2_sched=b2_sched,
        oven_time_per_job=settings.oven_time_per_job,
        empty_time_ratio=settings.empty_time_ratio,
    )
    T = makespan
    pb1_idle = max(0.0, T - b1_sched.busy_time - pb1_blocked)
    pb2_idle = max(0.0, T - b2_sched.busy_time - pb2_blocked)

    pb_block_frac_max = max(pb1_blocked / T if T > 0 else 0.0, pb2_blocked / T if T > 0 else 0.0)
    oven_idle_frac = (oven_idle / T) if T > 0 else 0.0

    f4_value: Optional[float] = None
    if settings.f4_weights:
        w = settings.f4_weights
        f4_value = (
            w.get("pb_block_frac_max", 0.0) * pb_block_frac_max
            + w.get("oven_idle_frac", 0.0) * oven_idle_frac
            + w.get("makespan", 0.0) * T
        )

    return DesMetricsRow(
        method=method,
        scenario=scenario,
        solution_id=solution_id,
        seed=seed,
        throughput_jobs=jobs,
        tokens_empty=sum(1 for it in convoy if it == "TOKEN"),
        makespan=makespan,
        pb1_busy_time=b1_sched.busy_time,
        pb1_blocked_time=pb1_blocked,
        pb1_idle_time=pb1_idle,
        pb2_busy_time=b2_sched.busy_time,
        pb2_blocked_time=pb2_blocked,
        pb2_idle_time=pb2_idle,
        oven_busy_time=oven_busy,
        oven_idle_time=oven_idle,
        pb_block_frac_max=pb_block_frac_max,
        oven_idle_frac=oven_idle_frac,
        f4=f4_value,
    )


# -----------------------------------------------------------------------------
# Paths & settings
# -----------------------------------------------------------------------------

def build_paths(global_cfg: dict, steps_cfg: dict, global_cfg_path: Path, steps_cfg_path: Path) -> Step08Paths:
    paths_cfg = global_cfg["paths"]
    data_files = global_cfg["data_files"]
    step03 = steps_cfg["step03"]
    step05 = steps_cfg.get("step05", {})
    step06 = steps_cfg.get("step06", {})
    step07 = steps_cfg.get("step07", {})
    step08 = steps_cfg["step08"]

    dataset_dir = Path(paths_cfg["dataset_dir"])
    output_dir = Path(paths_cfg["output_dir"])

    milp_prefix = step06.get("milp_output_prefix", "step06_milp_")
    step07_exports = step07.get("exports_dir_relpath", "step07_exports")
    consolidated_rel = step07.get("global_line_relpath", "step07_global_line.csv")

    return Step08Paths(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        adjacency_rules_path=dataset_dir / data_files["adjacency_rules"],
        s_matrix_path=dataset_dir / data_files["s_matrix_transition"],
        topcoat_codebook_path=dataset_dir / data_files["topcoat_codebook"],
        topcoat_rules_path=dataset_dir / data_files["topcoat_rules"],
        assignment_path=output_dir / step03["assignment_relpath"],
        booth1_step03_path=output_dir / step03["booth1_relpath"],
        booth2_step03_path=output_dir / step03["booth2_relpath"],
        step05_global_path=output_dir / step05.get("global_line_relpath", "step05_global_line.csv"),
        step06_milp_global_path=output_dir / f"{milp_prefix}global.csv",
        step07_exports_dir=(output_dir / step07_exports) if step07_exports else None,
        step07_consolidated_path=(output_dir / consolidated_rel) if consolidated_rel else None,
        metrics_path=output_dir / step08["metrics_relpath"],
        manifest_path=output_dir / step08["manifest_relpath"],
        report_path=output_dir / step08["report_relpath"],
        global_config_path=global_cfg_path,
        steps_config_path=steps_cfg_path,
    )


def build_settings(global_cfg: dict, steps_cfg: dict) -> Step08Settings:
    des_cfg = global_cfg.get("des", {})
    tokens_cfg = global_cfg.get("tokens", {})
    sdsu_cfg = global_cfg.get("sdsu", {})
    step05_cfg = steps_cfg.get("step05", {})
    step08_cfg = steps_cfg.get("step08", {})

    base_paint_time = float(des_cfg.get("base_paint_time", 1.0))
    time_per_setup_unit = float(des_cfg.get("time_per_setup_unit", 1.0))
    oven_time_per_job = float(des_cfg.get("theta", 1.0))

    empty_time_ratio = float(tokens_cfg.get("empty_time_ratio", 1.0))  # cost ratio for EMPTY at oven

    allow_purge = bool(sdsu_cfg.get("allow_purge", True))
    purge_weight = float(sdsu_cfg.get("purge_cost_weight", 1.0))
    flush_units_ratio = float(tokens_cfg.get("flush_units_ratio", 1.0))
    flush_cost_units = flush_units_ratio * purge_weight

    tie_break_policy = str(step05_cfg.get("tie_break_policy", "longer_remaining"))

    run_step05 = bool(step08_cfg.get("run_step05", True))
    run_step06 = bool(step08_cfg.get("run_step06", True))
    run_step07 = bool(step08_cfg.get("run_step07", True))

    f4_weights = step08_cfg.get("f4_weights", None)
    if f4_weights is not None and not isinstance(f4_weights, dict):
        f4_weights = None

    return Step08Settings(
        base_paint_time=base_paint_time,
        time_per_setup_unit=time_per_setup_unit,
        oven_time_per_job=oven_time_per_job,
        empty_time_ratio=empty_time_ratio,
        allow_purge=allow_purge,
        flush_cost_units=flush_cost_units,
        tie_break_policy=tie_break_policy,
        run_step05=run_step05,
        run_step06=run_step06,
        run_step07=run_step07,
        f4_weights=f4_weights,
    )


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------

def write_manifest_and_report(
    paths: Step08Paths,
    settings: Step08Settings,
    inputs: List[Path],
    outputs: List[Path],
    metrics_df: pd.DataFrame,
    runtime_seconds: float,
) -> None:
    ensure_parent_dir(paths.manifest_path)
    ensure_parent_dir(paths.report_path)

    manifest = {
        "step": "step08_des",
        "timestamp": dt.datetime.now().isoformat(),
        "config_files": {
            os.path.relpath(str(paths.global_config_path), str(paths.output_dir)): sha256_of_file(paths.global_config_path),
            os.path.relpath(str(paths.steps_config_path), str(paths.output_dir)): sha256_of_file(paths.steps_config_path),
        },
        "inputs": {
            os.path.relpath(str(p), str(paths.output_dir)): sha256_of_file(p) for p in inputs if p.exists()
        },
        "outputs": {
            os.path.relpath(str(p), str(paths.output_dir)): sha256_of_file(p) for p in outputs if p.exists()
        },
        "settings_effective": dataclasses.asdict(settings),
        "summary": {
            "scenarios": int(metrics_df.shape[0]),
            "runtime_seconds": runtime_seconds,
        },
    }
    with paths.manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    lines: List[str] = []
    lines.append("# STEP-08 — DES (EMPTY resets adjacency)")
    lines.append("")
    lines.append(f"Scenarios evaluated: {int(metrics_df.shape[0])}")
    lines.append(f"Runtime: {runtime_seconds:.2f} seconds")
    lines.append("")
    lines.append("## Settings")
    lines.append("```json")
    lines.append(json.dumps(dataclasses.asdict(settings), indent=2))
    lines.append("```")
    lines.append("")
    if not metrics_df.empty:
        lines.append("## Metrics (head)")
        lines.append("")
        lines.append(metrics_df.head(min(12, len(metrics_df))).to_csv(index=False))
    with paths.report_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="STEP-08 — DES evaluator (EMPTY resets adjacency)")
    parser.add_argument("global_config", type=Path, nargs="?", default=Path("global_config.toml"))
    parser.add_argument("steps_config", type=Path, nargs="?", default=Path("steps_config.toml"))
    args = parser.parse_args(argv)

    t0 = dt.datetime.now().timestamp()
    global_cfg = load_toml(args.global_config)
    steps_cfg = load_toml(args.steps_config)

    paths = build_paths(global_cfg, steps_cfg, args.global_config, args.steps_config)
    settings = build_settings(global_cfg, steps_cfg)

    forb_in_booth, forb_global = build_adjacency_sets(paths.adjacency_rules_path, global_cfg)
    job_records, job_index_by_id = build_job_catalog(global_cfg, paths)
    kernel = build_paint_kernel(paths, global_cfg, settings)

    inputs_for_manifest: List[Path] = [
        paths.global_config_path,
        paths.steps_config_path,
        paths.adjacency_rules_path,
        paths.s_matrix_path,
        paths.topcoat_codebook_path,
        paths.topcoat_rules_path,
        paths.assignment_path,
        paths.booth1_step03_path,
        paths.booth2_step03_path,
    ]
    outputs_for_manifest: List[Path] = [paths.metrics_path, paths.manifest_path, paths.report_path]

    rows: List[DesMetricsRow] = []

    # --- Step-05 baseline
    if settings.run_step05 and paths.step05_global_path.exists():
        log(f"Evaluating Step-05 global convoy: {paths.step05_global_path.name}")
        convoy = load_global_convoy_csv(paths.step05_global_path, job_index_by_id, job_records)
        cert = first_c2_violation_with_tokens(convoy, job_records, forb_global)
        if cert:
            log(f"NOTE: parts-only C2 would be violated at pos {cert['pos']} ({cert['prev_part']}->{cert['curr_part']}); "
                f"accepted because EMPTY resets adjacency.")
        r = run_scenario_from_convoy(
            method="step05",
            scenario="step05_global",
            solution_id=None,
            seed=None,
            convoy=convoy,
            job_records=job_records,
            kernel=kernel,
            settings=settings,
        )
        rows.append(r)
        inputs_for_manifest.append(paths.step05_global_path)

    # --- Step-06 MILP (global)
    if settings.run_step06 and paths.step06_milp_global_path.exists():
        log(f"Evaluating Step-06 MILP global convoy: {paths.step06_milp_global_path.name}")
        convoy = load_global_convoy_csv(paths.step06_milp_global_path, job_index_by_id, job_records)
        cert = first_c2_violation_with_tokens(convoy, job_records, forb_global)
        if cert:
            log(f"NOTE: parts-only C2 would be violated at pos {cert['pos']} ({cert['prev_part']}->{cert['curr_part']}); "
                f"accepted because EMPTY resets adjacency.")
        r = run_scenario_from_convoy(
            method="step06_milp",
            scenario="step06_global",
            solution_id=None,
            seed=None,
            convoy=convoy,
            job_records=job_records,
            kernel=kernel,
            settings=settings,
        )
        rows.append(r)
        inputs_for_manifest.append(paths.step06_milp_global_path)

    # --- Step-07 NSGA-II
    if settings.run_step07:
        handled_any = False

        # 7a) consolidated file (common in your repo)
        if paths.step07_consolidated_path and paths.step07_consolidated_path.exists():
            p = paths.step07_consolidated_path
            log(f"Evaluating Step-07 consolidated: {p.name}")
            for scenario_name, seed_val, sid, convoy in iter_step07_solutions_from_csv(p, job_index_by_id, job_records):
                cert = first_c2_violation_with_tokens(convoy, job_records, forb_global)
                if cert:
                    log(f"NOTE [{scenario_name}]: parts-only C2 would be violated at pos {cert['pos']} "
                        f"({cert['prev_part']}->{cert['curr_part']}); accepted via EMPTY reset.")
                row = run_scenario_from_convoy(
                    method="step07_nsga2",
                    scenario=scenario_name,
                    solution_id=sid,
                    seed=seed_val,
                    convoy=convoy,
                    job_records=job_records,
                    kernel=kernel,
                    settings=settings,
                )
                rows.append(row)
            inputs_for_manifest.append(p)
            handled_any = True

        # 7b) exports dir with many per-solution files
        if paths.step07_exports_dir and paths.step07_exports_dir.exists():
            nsga_files = sorted([p for p in paths.step07_exports_dir.glob("*.csv") if "global" in p.stem.lower()])
            for p in nsga_files:
                # if file already equals consolidated, skip duplicate
                if paths.step07_consolidated_path and p.resolve() == paths.step07_consolidated_path.resolve():
                    continue
                log(f"Evaluating Step-07 file: {p.name}")
                df = pd.read_csv(p)
                # If multi-solution inside file
                has_sid = "solution_id" in {c.lower() for c in df.columns}
                if has_sid:
                    for scenario_name, seed_val, sid, convoy in iter_step07_solutions_from_csv(p, job_index_by_id, job_records):
                        cert = first_c2_violation_with_tokens(convoy, job_records, forb_global)
                        if cert:
                            log(f"NOTE [{scenario_name}]: parts-only C2 would be violated at pos {cert['pos']} "
                                f"({cert['prev_part']}->{cert['curr_part']}); accepted via EMPTY reset.")
                        row = run_scenario_from_convoy(
                            method="step07_nsga2",
                            scenario=scenario_name,
                            solution_id=sid,
                            seed=seed_val,
                            convoy=convoy,
                            job_records=job_records,
                            kernel=kernel,
                            settings=settings,
                        )
                        rows.append(row)
                else:
                    # single-solution file
                    # parse seed/solution_id from filename if present
                    seed_val: Optional[int] = None
                    sol_id: Optional[str] = None
                    m = re.search(r"seed(\d+)", p.stem, re.IGNORECASE)
                    if m:
                        try:
                            seed_val = int(m.group(1))
                        except Exception:
                            seed_val = None
                    m = re.search(r"sol(?:ution)?[_\-]?(\d+)", p.stem, re.IGNORECASE)
                    if m:
                        sol_id = m.group(1)
                    convoy = parse_convoy_df_rows(df, job_index_by_id, job_records)
                    cert = first_c2_violation_with_tokens(convoy, job_records, forb_global)
                    if cert:
                        log(f"NOTE [{p.stem}]: parts-only C2 would be violated at pos {cert['pos']} "
                            f"({cert['prev_part']}->{cert['curr_part']}); accepted via EMPTY reset.")
                    row = run_scenario_from_convoy(
                        method="step07_nsga2",
                        scenario=p.stem,
                        solution_id=sol_id,
                        seed=seed_val,
                        convoy=convoy,
                        job_records=job_records,
                        kernel=kernel,
                        settings=settings,
                    )
                    rows.append(row)
                inputs_for_manifest.append(p)
                handled_any = True

        if not handled_any:
            log("No Step-07 inputs found (exports dir and consolidated file missing).")
        
        # repeat for _seeded variants, the file name is _seeded.[ext]
        # we do this by check if there's _seeded file
        seeded_variant_exists = False
        if paths.step07_consolidated_path:
            seeded_path = paths.step07_consolidated_path.with_name(paths.step07_consolidated_path.stem + "_seeded.csv")
            if seeded_path.exists():
                seeded_variant_exists = True
        if seeded_variant_exists or (paths.step07_exports_dir and paths.step07_exports_dir.exists() and
                                    any(p for p in paths.step07_exports_dir.glob("*_seeded.csv"))):
            log("Evaluating Step-07 seeded NSGA-II solutions:")
            handled_any = False

            # 7a-s) consolidated file (common in your repo)
            if paths.step07_consolidated_path and paths.step07_consolidated_path.exists():
                p = paths.step07_consolidated_path.with_name(paths.step07_consolidated_path.stem + "_seeded.csv")
                if p.exists():
                    log(f"Evaluating Step-07 seeded consolidated: {p.name}")
                    for scenario_name, seed_val, sid, convoy in iter_step07_solutions_from_csv(p, job_index_by_id, job_records):
                        cert = first_c2_violation_with_tokens(convoy, job_records, forb_global)
                        if cert:
                            log(f"NOTE [{scenario_name}]: parts-only C2 would be violated at pos {cert['pos']} "
                                f"({cert['prev_part']}->{cert['curr_part']}); accepted via EMPTY reset.")
                        row = run_scenario_from_convoy(
                            method="step07_nsga2_seeded",
                            scenario=scenario_name,
                            solution_id=sid,
                            seed=seed_val,
                            convoy=convoy,
                            job_records=job_records,
                            kernel=kernel,
                            settings=settings,
                        )
                        rows.append(row)
                    inputs_for_manifest.append(p)
                    handled_any = True

            # 7b-s) exports dir with many per-solution files
            if paths.step07_exports_dir and paths.step07_exports_dir.exists():
                nsga_files = sorted([p for p in paths.step07_exports_dir.glob("*_seeded.csv") if "global" in p.stem.lower()])
                for p in nsga_files:
                    log(f"Evaluating Step-07 seeded file: {p.name}")
                    df = pd.read_csv(p)
                    # If multi-solution inside file
                    has_sid = "solution_id" in {c.lower() for c in df.columns}
                    if has_sid:
                        for scenario_name, seed_val, sid, convoy in iter_step07_solutions_from_csv(p, job_index_by_id, job_records):
                            cert = first_c2_violation_with_tokens(convoy, job_records, forb_global)
                            if cert:
                                log(f"NOTE [{scenario_name}]: parts-only C2 would be violated at pos {cert['pos']} "
                                    f"({cert['prev_part']}->{cert['curr_part']}); accepted via EMPTY reset.")
                            row = run_scenario_from_convoy(
                                method="step07_nsga2_seeded",
                                scenario=scenario_name,
                                solution_id=sid,
                                seed=seed_val,
                                convoy=convoy,
                                job_records=job_records,
                                kernel=kernel,
                                settings=settings,
                            )
                            rows.append(row)
                    else:
                        # single-solution file
                        # parse seed/solution_id from filename if present
                        seed_val: Optional[int] = None
                        sol_id: Optional[str] = None
                        m = re.search(r"seed(\d+)", p.stem, re.IGNORECASE)
                        if m:
                            try:
                                seed_val = int(m.group(1))
                            except Exception:
                                seed_val = None
                        m = re.search(r"sol(?:ution)?[_\-]?(\d+)", p.stem, re.IGNORECASE)
                        if m:
                            sol_id = m.group(1)
                        convoy = parse_convoy_df_rows(df, job_index_by_id, job_records)
                        cert = first_c2_violation_with_tokens(convoy, job_records, forb_global)
                        if cert:
                            log(f"NOTE [{p.stem}]: parts-only C2 would be violated at pos {cert['pos']} "
                                f"({cert['prev_part']}->{cert['curr_part']}); accepted via EMPTY reset.")
                        row = run_scenario_from_convoy(
                            method="step07_nsga2_seeded",
                            scenario=p.stem,
                            solution_id=sol_id,
                            seed=seed_val,
                            convoy=convoy,
                            job_records=job_records,
                            kernel=kernel,
                            settings=settings,
                        )
                        rows.append(row)
                    inputs_for_manifest.append(p)
                    handled_any = True
            if not handled_any:
                log("No Step-07 seeded inputs found (exports dir and consolidated file missing).")                


    # Write metrics
    ensure_parent_dir(paths.metrics_path)
    df = pd.DataFrame([dataclasses.asdict(r) for r in rows]) if rows else pd.DataFrame(
        columns=[
            "method", "scenario", "solution_id", "seed",
            "throughput_jobs", "tokens_empty", "makespan",
            "pb1_busy_time", "pb1_blocked_time", "pb1_idle_time",
            "pb2_busy_time", "pb2_blocked_time", "pb2_idle_time",
            "oven_busy_time", "oven_idle_time",
            "pb_block_frac_max", "oven_idle_frac", "f4",
        ]
    )
    df.to_csv(paths.metrics_path, index=False)

    t1 = dt.datetime.now().timestamp()
    write_manifest_and_report(
        paths=paths,
        settings=settings,
        inputs=inputs_for_manifest,
        outputs=[paths.metrics_path, paths.manifest_path, paths.report_path],
        metrics_df=df,
        runtime_seconds=t1 - t0,
    )


    try:
        from step08_des_plotter import run_step08_des_plotter
        run_step08_des_plotter(paths.metrics_path)
    except ImportError:
        log("step08_des_plotter module not found; skipping plotting.")

    log(f"Completed DES evaluation; scenarios={df.shape[0]} -> {paths.metrics_path}")


if __name__ == "__main__":
    main()
