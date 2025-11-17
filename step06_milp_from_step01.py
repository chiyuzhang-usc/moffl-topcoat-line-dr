#!/usr/bin/env python3
"""
STEP-06 (from STEP-01) — Global MILP baseline (assignment + sequencing)

This script formulates a global MILP directly from the STEP-01 job expansion:

- Job universe: all jobs in `step01_jobs_expanded.csv` (STEP-01 output).
- Two booths (PB1, PB2) with identical capabilities.
- Decisions:
    - Assign each job to either PB1 or PB2.
    - Sequence the jobs within each booth as an open path.
- Objective: minimise total f1 paint setup cost (setup + purge units)
  across both booths under C1 (paint) and C2 (adjacency) rules.

Gating / "toy instance" behaviour:

- The flag `steps_config.toml[step06].enable_reduced_instances` is used as a gate:
    - If `enable_reduced_instances = false`:
        - The global STEP-01 MILP is **disabled**. The script only writes a
          manifest and report with status `DISABLED_BY_CONFIG` and exits.
    - If `enable_reduced_instances = true`:
        - The script considers the full STEP-01 job set as a candidate "toy instance".
        - It still checks `global_config.toml[milp].max_jobs_per_instance`:
            - If `n_jobs > max_jobs_per_instance`:
                - The MILP is **not** solved. Manifest/report record
                  `TOO_LARGE_GLOBAL` and exit.
            - If `n_jobs <= max_jobs_per_instance`:
                - A single global MILP instance is built and solved.

No automatic reduced-instance/windowing is performed here. If you want to
study reduced subsets of STEP-01, you should build those subsets outside this
script (e.g., by creating smaller STEP-01 snapshots) and then run this script
with `enable_reduced_instances = true`.

Usage:

    python step06_milp_from_step01.py \
        --global-config global_config.toml \
        --steps-config steps_config.toml
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import tomllib

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception as exc:  # pragma: no cover - import-time guard
    raise SystemExit(
        "ERROR: gurobipy is required to run STEP-06-from-STEP01 MILP.\n"
        "Please install Gurobi Python bindings and ensure a license is available."
    ) from exc


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def load_toml(path: Path) -> Mapping:
    with path.open("rb") as f:
        return tomllib.load(f)


def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_column(df: pd.DataFrame, candidates: Sequence[str], logical_name: str) -> str:
    """
    Resolve a column name in `df` from a list of candidate names.
    Raises a clear error if none match.
    """
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"Required column for '{logical_name}' not found in dataframe. "
        f"Tried candidates: {candidates}. Available: {list(df.columns)}"
    )


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class GlobalConfig:
    dataset_dir: Path
    output_dir: Path
    manifests_dir: Path
    reports_dir: Path

    tokens_flush_units_ratio: float
    tokens_empty_units_prepaint: float
    tokens_empty_units_postpaint_ratio: float

    adjacency_enabled: bool
    adjacency_scope_escalation: str

    sdsu_allow_purge: bool
    sdsu_purge_cost_weight: float

    milp_solver: str
    milp_time_limit_sec: float
    milp_mip_gap: float
    milp_max_jobs_per_instance: int
    milp_threads: int
    milp_log_to_console: bool

    data_files: Mapping[str, str]
    data_schema: Mapping[str, Mapping]


@dataclasses.dataclass
class Step06Config:
    milp_output_prefix: str
    manifest_relpath: str
    report_relpath: str
    enable_reduced_instances: bool


@dataclasses.dataclass
class InstanceMeta:
    instance_tag: str
    n_jobs: int


@dataclasses.dataclass
class InstanceResult:
    meta: InstanceMeta
    status: str
    f1_milp: Optional[float]
    f1_baseline: Optional[float]
    solver_mip_gap: Optional[float]
    solver_time_sec: Optional[float]
    output_relpath: Optional[str]


# ---------------------------------------------------------------------------
# Dataset loaders (paint, topcoat rules, adjacency)
# ---------------------------------------------------------------------------


def load_s_matrix_transition(path: Path, data_schema: Mapping[str, Mapping]) -> Dict[Tuple[str, str], float]:
    """
    Load s_matrix_transition.csv as a dense asymmetric cost matrix.

    Format assumption:
        - First column: 'from' (or synonym)
        - Remaining columns: topcoat codes (e.g. 'C_RUBY_RED')
        - Values: numeric (>=0) for allowed transitions; 'inf' / NaN => forbidden
    """
    df = pd.read_csv(path)
    schema = data_schema.get("s_matrix_transition", {})
    from_col = resolve_column(df, schema.get("from_code_synonyms", ["from"]), "s_matrix.from")

    cost: Dict[Tuple[str, str], float] = {}
    for _, row in df.iterrows():
        from_code = str(row[from_col])
        for col in df.columns:
            if col == from_col:
                continue
            raw_val = row[col]
            if isinstance(raw_val, str) and raw_val.lower() == "inf":
                val = math.inf
            else:
                try:
                    val = float(raw_val)
                except Exception:
                    val = math.inf
            cost[(from_code, col)] = val
    return cost


def load_topcoat_codebook(path: Path) -> Dict[int, str]:
    """
    Load topcoat_codebook.csv and return mapping from topcoat_no -> topcoat_code.
    """
    df = pd.read_csv(path)
    if "topcoat_no" not in df.columns or "topcoat_code" not in df.columns:
        return {}
    mapping: Dict[int, str] = {}
    for _, row in df.iterrows():
        try:
            no = int(row["topcoat_no"])
        except Exception:
            continue
        mapping[no] = str(row["topcoat_code"])
    return mapping


def load_topcoat_forbids(
    path: Path,
    topcoat_no_to_code: Mapping[int, str],
    data_schema: Mapping[str, Mapping],
) -> List[Tuple[str, str]]:
    """
    Load topcoat_rules.csv and return a list of forbidden (from_code, to_code) pairs.
    """
    df = pd.read_csv(path)
    schema = data_schema.get("topcoat_rules", {})

    from_no_col = resolve_column(
        df, schema.get("from_topcoat_code_synonyms", ["from_topcoat_no"]), "topcoat_rules.from"
    )
    to_no_col = resolve_column(
        df, schema.get("to_topcoat_code_synonyms", ["to_topcoat_no"]), "topcoat_rules.to"
    )
    relation_col = resolve_column(df, schema.get("relation_synonyms", ["relation"]), "topcoat_rules.relation")

    forbids: List[Tuple[str, str]] = []
    for _, row in df.iterrows():
        rel = str(row[relation_col]).strip().lower()
        if not rel.startswith("forbid"):
            continue

        try:
            f_no = int(row[from_no_col])
            t_no = int(row[to_no_col])
        except Exception:
            continue

        f_code = topcoat_no_to_code.get(f_no)
        t_code = topcoat_no_to_code.get(t_no)
        if f_code is None or t_code is None:
            continue
        forbids.append((f_code, t_code))
    return forbids


def load_adjacency_forbids(path: Path, data_schema: Mapping[str, Mapping], enabled: bool) -> List[Tuple[str, str]]:
    """
    Load adjacency_rules.csv and return forbidden part pairs (from_part_id, to_part_id).
    """
    if not enabled:
        return []

    df = pd.read_csv(path)
    schema = data_schema.get("adjacency_rules", {})

    from_col = resolve_column(
        df, schema.get("from_part_id_synonyms", ["from_part_id"]), "adjacency.from_part_id"
    )
    to_col = resolve_column(df, schema.get("to_part_id_synonyms", ["to_part_id"]), "adjacency.to_part_id")
    relation_col = resolve_column(df, schema.get("relation_synonyms", ["relation"]), "adjacency.relation")

    forbids: List[Tuple[str, str]] = []
    for _, row in df.iterrows():
        rel = str(row[relation_col]).strip().lower()
        if not rel.startswith("forbid"):
            continue
        forbids.append((str(row[from_col]), str(row[to_col])))
    return forbids


# ---------------------------------------------------------------------------
# Cost kernel
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class PaintCostKernel:
    s_matrix: Dict[Tuple[str, str], float]
    explicit_forbids: List[Tuple[str, str]]
    flush_cost_units: float
    allow_purge: bool

    def is_paint_forbidden(self, from_code: str, to_code: str) -> bool:
        if (from_code, to_code) in self.explicit_forbids:
            return True
        val = self.s_matrix.get((from_code, to_code), math.inf)
        return math.isinf(val)

    def setup_time(self, from_code: str, to_code: str) -> float:
        return self.s_matrix.get((from_code, to_code), math.inf)

    def arc_cost(self, from_code: str, to_code: str) -> Optional[float]:
        """
        Effective cost of paint transition from_code -> to_code.

        - If transition is forbidden: returns None.
        - Otherwise:
            - If purges are disabled: cost = direct setup time.
            - If purges are allowed: cost = min(direct setup, flush_cost_units).
        """
        if self.is_paint_forbidden(from_code, to_code):
            return None

        direct = self.setup_time(from_code, to_code)
        if math.isinf(direct):
            return None

        if not self.allow_purge or self.flush_cost_units <= 0.0:
            return float(direct)

        return float(min(direct, self.flush_cost_units))

    def split_arc_cost(self, from_code: str, to_code: str) -> Tuple[float, float]:
        """
        Split cost into (setup_units, purge_units) for reporting.
        """
        if self.is_paint_forbidden(from_code, to_code):
            raise ValueError("split_arc_cost called on forbidden transition")

        direct = self.setup_time(from_code, to_code)
        if math.isinf(direct):
            raise ValueError("split_arc_cost called on INF transition")

        if not self.allow_purge or self.flush_cost_units <= 0.0:
            return float(direct), 0.0

        if direct <= self.flush_cost_units:
            return float(direct), 0.0
        return 0.0, float(self.flush_cost_units)


# ---------------------------------------------------------------------------
# STEP-01 jobs → MILP instance
# ---------------------------------------------------------------------------


def load_step01_jobs(output_dir: Path, jobs_relpath: str, data_schema: Mapping[str, Mapping]) -> pd.DataFrame:
    """
    Load STEP-01 jobs_expanded CSV and normalise key columns.
    """
    path = output_dir / jobs_relpath
    if not path.exists():
        raise FileNotFoundError(f"STEP-01 jobs_expanded not found: {path}")
    df = pd.read_csv(path)

    if "job_id" not in df.columns:
        raise KeyError("Expected 'job_id' column in STEP-01 jobs CSV; not found.")

    dem_schema = data_schema.get("demands", {})
    part_col = resolve_column(df, dem_schema.get("part_id_synonyms", ["part_id"]), "job.part_id")
    coat_col = resolve_column(df, dem_schema.get("topcoat_code_synonyms", ["topcoat_code"]), "job.topcoat_code")

    df = df.copy()
    if part_col != "part_id":
        df.rename(columns={part_col: "part_id"}, inplace=True)
    if coat_col != "topcoat_code":
        df.rename(columns={coat_col: "topcoat_code"}, inplace=True)

    return df[["job_id", "part_id", "topcoat_code"]].copy()


# ---------------------------------------------------------------------------
# Baseline cost (STEP-01 order)
# ---------------------------------------------------------------------------


def evaluate_baseline_cost(
    jobs_df: pd.DataFrame,
    paint_kernel: PaintCostKernel,
    adjacency_forbids: List[Tuple[str, str]],
) -> Optional[float]:
    """
    Compute f1 for the STEP-01 order (row order) under C1/C2.

    Returns None if any consecutive pair is infeasible.
    """
    if jobs_df.empty:
        return 0.0

    parts = list(jobs_df["part_id"])
    coats = list(jobs_df["topcoat_code"])
    forbidden_adj_set = set(adjacency_forbids)

    total = 0.0
    for i in range(len(jobs_df) - 1):
        part_i = str(parts[i])
        part_j = str(parts[i + 1])
        if (part_i, part_j) in forbidden_adj_set:
            return None

        coat_i = str(coats[i])
        coat_j = str(coats[i + 1])
        c_val = paint_kernel.arc_cost(coat_i, coat_j)
        if c_val is None or math.isinf(c_val):
            return None
        total += float(c_val)

    return total


# ---------------------------------------------------------------------------
# Global MILP: assignment + sequencing on two booths
# ---------------------------------------------------------------------------


def build_and_solve_global_milp(
    meta: InstanceMeta,
    jobs_df: pd.DataFrame,
    paint_kernel: PaintCostKernel,
    adjacency_forbids: List[Tuple[str, str]],
    milp_cfg: GlobalConfig,
) -> Tuple[Optional[Dict[int, List[int]]], InstanceResult]:
    """
    Build and solve global MILP with two booths (PB1, PB2).

    Returns
    -------
    booth_sequences : Optional[Dict[int, List[int]]]
        Mapping booth_id -> list of job indices (0-based in jobs_df) in sequence order.
        None if infeasible or solve failure.
    result : InstanceResult
        Solver status and metrics (output_relpath to be filled later).
    """
    n = meta.n_jobs
    assert len(jobs_df) == n

    job_ids = list(jobs_df["job_id"])
    parts = list(jobs_df["part_id"])
    coats = list(jobs_df["topcoat_code"])

    booths = [1, 2]

    # Node indexing per booth: 0 = START_b, 1..n = jobs, n+1 = END_b
    start_node = 0
    end_node = n + 1
    job_nodes = list(range(1, n + 1))

    # Build arcs for each booth
    # Each arc: (booth_id, tail_node, head_node, cost, is_job_job)
    arcs: List[Tuple[int, int, int, float, bool]] = []
    arc_index: Dict[Tuple[int, int, int], int] = {}

    forbidden_adj_set = set(adjacency_forbids)

    for b in booths:
        # Start_b -> job (0 cost)
        for j_node in job_nodes:
            idx = len(arcs)
            arcs.append((b, start_node, j_node, 0.0, False))
            arc_index[(b, start_node, j_node)] = idx

        # job -> job arcs
        for i_idx, i_node in enumerate(job_nodes):
            part_i = str(parts[i_idx])
            coat_i = str(coats[i_idx])

            for j_idx, j_node in enumerate(job_nodes):
                if i_node == j_node:
                    continue

                part_j = str(parts[j_idx])
                coat_j = str(coats[j_idx])

                if (part_i, part_j) in forbidden_adj_set:
                    continue

                c = paint_kernel.arc_cost(coat_i, coat_j)
                if c is None or math.isinf(c):
                    continue

                idx = len(arcs)
                arcs.append((b, i_node, j_node, float(c), True))
                arc_index[(b, i_node, j_node)] = idx

        # job -> END_b (0 cost)
        for i_node in job_nodes:
            idx = len(arcs)
            arcs.append((b, i_node, end_node, 0.0, False))
            arc_index[(b, i_node, end_node)] = idx

    # Build arc adjacency maps
    arcs_by_tail: Dict[Tuple[int, int], List[int]] = {}
    arcs_by_head: Dict[Tuple[int, int], List[int]] = {}
    for k, (b, tail, head, _, _) in enumerate(arcs):
        arcs_by_tail.setdefault((b, tail), []).append(k)
        arcs_by_head.setdefault((b, head), []).append(k)

    # Sanity: each job must have at least one feasible incoming and outgoing arc
    for i_idx, i_node in enumerate(job_nodes):
        incoming_any = False
        outgoing_any = False
        for b in booths:
            if arcs_by_head.get((b, i_node)):
                incoming_any = True
            if arcs_by_tail.get((b, i_node)):
                outgoing_any = True
        if not incoming_any:
            result = InstanceResult(
                meta=meta,
                status="INFEASIBLE_ARC_IN",
                f1_milp=None,
                f1_baseline=None,
                solver_mip_gap=None,
                solver_time_sec=None,
                output_relpath=None,
            )
            return None, result
        if not outgoing_any:
            result = InstanceResult(
                meta=meta,
                status="INFEASIBLE_ARC_OUT",
                f1_milp=None,
                f1_baseline=None,
                solver_mip_gap=None,
                solver_time_sec=None,
                output_relpath=None,
            )
            return None, result

    # Build MILP model
    model = gp.Model("step06_from_step01_global")
    if not milp_cfg.milp_log_to_console:
        model.Params.LogToConsole = 0

    model.Params.TimeLimit = float(milp_cfg.milp_time_limit_sec)
    model.Params.MIPGap = float(milp_cfg.milp_mip_gap)
    if milp_cfg.milp_threads > 0:
        model.Params.Threads = int(milp_cfg.milp_threads)

    # Arc usage
    x = model.addVars(len(arcs), vtype=GRB.BINARY, name="x")

    # Assignment variables: z[i, b] = 1 if job i is assigned to booth b
    z = model.addVars(
        [(i_idx, b) for i_idx in range(n) for b in booths],
        vtype=GRB.BINARY,
        name="z",
    )

    # MTZ order variables per booth
    u = model.addVars(
        [(i_idx, b) for i_idx in range(n) for b in booths],
        lb=1.0,
        ub=float(n),
        vtype=GRB.CONTINUOUS,
        name="u",
    )

    # Objective: sum of arc costs
    model.setObjective(
        gp.quicksum(arcs[k][3] * x[k] for k in range(len(arcs))),
        GRB.MINIMIZE,
    )

    # Assignment: each job to exactly one booth
    for i_idx in range(n):
        model.addConstr(
            gp.quicksum(z[i_idx, b] for b in booths) == 1.0,
            name=f"assign_job_{i_idx}",
        )

    # Flow constraints per booth, linked to assignment
    for b in booths:
        # Start_b: exactly one outgoing arc
        start_out_indices = arcs_by_tail.get((b, start_node), [])
        if not start_out_indices:
            raise RuntimeError(f"No outgoing arcs from START for booth {b}")
        model.addConstr(
            gp.quicksum(x[k] for k in start_out_indices) == 1.0,
            name=f"start_out_b{b}",
        )

        # End_b: exactly one incoming arc
        end_in_indices = arcs_by_head.get((b, end_node), [])
        if not end_in_indices:
            raise RuntimeError(f"No incoming arcs to END for booth {b}")
        model.addConstr(
            gp.quicksum(x[k] for k in end_in_indices) == 1.0,
            name=f"end_in_b{b}",
        )

        # Jobs: in/out degree equal to assignment z[i,b]
        for i_idx, i_node in enumerate(job_nodes):
            in_indices = arcs_by_head.get((b, i_node), [])
            out_indices = arcs_by_tail.get((b, i_node), [])

            model.addConstr(
                gp.quicksum(x[k] for k in in_indices) == z[i_idx, b],
                name=f"job_in_i{i_idx}_b{b}",
            )
            model.addConstr(
                gp.quicksum(x[k] for k in out_indices) == z[i_idx, b],
                name=f"job_out_i{i_idx}_b{b}",
            )

    # MTZ subtour elimination per booth on job-job arcs
    big_M = float(n)
    for b in booths:
        for i_idx, i_node in enumerate(job_nodes):
            for j_idx, j_node in enumerate(job_nodes):
                if i_node == j_node:
                    continue
                key = (b, i_node, j_node)
                if key not in arc_index:
                    continue
                k = arc_index[key]
                model.addConstr(
                    u[i_idx, b] - u[j_idx, b] + 1.0 <= big_M * (1.0 - x[k]),
                    name=f"mtz_b{b}_i{i_idx}_j{j_idx}",
                )

    # Solve
    model.optimize()

    status_code = model.Status
    if status_code in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
        result = InstanceResult(
            meta=meta,
            status="INFEASIBLE",
            f1_milp=None,
            f1_baseline=None,
            solver_mip_gap=None,
            solver_time_sec=float(model.Runtime),
            output_relpath=None,
        )
        return None, result

    if status_code == GRB.TIME_LIMIT:
        status_str = "TIME_LIMIT"
    elif status_code == GRB.OPTIMAL:
        status_str = "OPTIMAL"
    else:
        status_str = f"STATUS_{status_code}"

    # Decode sequences per booth
    succ: Dict[Tuple[int, int], int] = {}  # (b, tail) -> head
    for k, (b, tail, head, _, _) in enumerate(arcs):
        if x[k].X > 0.5:
            succ[(b, tail)] = head

    booth_sequences: Dict[int, List[int]] = {}
    for b in booths:
        seq_nodes: List[int] = []
        cur = start_node
        visited = set()
        while True:
            if (b, cur) in visited:
                # cycle detected
                break
            visited.add((b, cur))
            nxt = succ.get((b, cur))
            if nxt is None:
                break
            if nxt == end_node:
                break
            seq_nodes.append(nxt)
            cur = nxt

        # Map node indices back to job indices (0-based)
        booth_sequences[b] = [node - 1 for node in seq_nodes]

    # Sanity: total jobs across booths must equal n
    total_assigned = sum(len(v) for v in booth_sequences.values())
    if total_assigned != n:
        result = InstanceResult(
            meta=meta,
            status="DECODE_ERROR",
            f1_milp=None,
            f1_baseline=None,
            solver_mip_gap=float(model.MIPGap),
            solver_time_sec=float(model.Runtime),
            output_relpath=None,
        )
        return None, result

    # MILP objective is already total f1 (job->job arcs only have non-zero cost)
    total_f1 = float(model.ObjVal)

    result = InstanceResult(
        meta=meta,
        status=status_str,
        f1_milp=total_f1,
        f1_baseline=None,  # filled by caller
        solver_mip_gap=float(model.MIPGap),
        solver_time_sec=float(model.Runtime),
        output_relpath=None,
    )
    return booth_sequences, result


# ---------------------------------------------------------------------------
# Manifest and report
# ---------------------------------------------------------------------------


def write_manifest(
    path: Path,
    global_cfg: GlobalConfig,
    step06_cfg: Step06Config,
    inputs: List[Path],
    outputs: List[Path],
    results: List[InstanceResult],
) -> None:
    ensure_dir(path)
    data = {
        "step_id": "STEP06_MILP_FROM_STEP01",
        "timestamp": dt.datetime.now().isoformat(),
        "config_snapshot": {
            "paths": {
                "dataset_dir": str(global_cfg.dataset_dir),
                "output_dir": str(global_cfg.output_dir),
                "manifests_dir": str(global_cfg.manifests_dir),
                "reports_dir": str(global_cfg.reports_dir),
            },
            "tokens": {
                "flush_units_ratio": global_cfg.tokens_flush_units_ratio,
                "empty_units_prepaint": global_cfg.tokens_empty_units_prepaint,
                "empty_units_postpaint_ratio": global_cfg.tokens_empty_units_postpaint_ratio,
            },
            "adjacency": {
                "enabled": global_cfg.adjacency_enabled,
                "scope_escalation": global_cfg.adjacency_scope_escalation,
            },
            "sdsu": {
                "allow_purge": global_cfg.sdsu_allow_purge,
                "purge_cost_weight": global_cfg.sdsu_purge_cost_weight,
            },
            "milp": {
                "solver": global_cfg.milp_solver,
                "time_limit_sec": global_cfg.milp_time_limit_sec,
                "mip_gap": global_cfg.milp_mip_gap,
                "max_jobs_per_instance": global_cfg.milp_max_jobs_per_instance,
                "threads": global_cfg.milp_threads,
                "log_to_console": global_cfg.milp_log_to_console,
            },
            "step06": dataclasses.asdict(step06_cfg),
        },
        "inputs": [
            {"path": str(p), "sha256": sha256_of_file(p)}
            for p in inputs
            if p.exists()
        ],
        "outputs": [
            {"path": str(p), "sha256": sha256_of_file(p)}
            for p in outputs
            if p.exists()
        ],
        "instances": [
            {
                "instance_tag": r.meta.instance_tag,
                "n_jobs": r.meta.n_jobs,
                "status": r.status,
                "f1_milp": r.f1_milp,
                "f1_baseline": r.f1_baseline,
                "solver_mip_gap": r.solver_mip_gap,
                "solver_time_sec": r.solver_time_sec,
                "output_relpath": r.output_relpath,
            }
            for r in results
        ],
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_report(
    path: Path,
    global_cfg: GlobalConfig,
    step06_cfg: Step06Config,
    results: List[InstanceResult],
) -> None:
    ensure_dir(path)
    lines: List[str] = []
    lines.append("# STEP-06 (from STEP-01) — Global MILP baseline\n")
    lines.append("## Purpose / What / Why\n")
    lines.append(
        "- **Purpose:** Provide a global f1 baseline by jointly optimising booth assignment\n"
        "  and sequencing over the entire STEP-01 job universe.\n"
    )
    lines.append(
        "- **What:** Two-booth open-path ATSP with C1/C2 semantics, solved only when\n"
        "  `enable_reduced_instances = true` and the job set is within `max_jobs_per_instance`.\n"
    )
    lines.append(
        "- **Why:** Acts as a theoretical lower bound on f1 for comparison against the\n"
        "  Toyota GS → split → SDSU pipeline and NSGA-II.\n"
    )

    lines.append("\n## Config (snapshot)\n")
    lines.append("```json\n")
    cfg_snapshot = {
        "tokens": {
            "flush_units_ratio": global_cfg.tokens_flush_units_ratio,
            "empty_units_prepaint": global_cfg.tokens_empty_units_prepaint,
            "empty_units_postpaint_ratio": global_cfg.tokens_empty_units_postpaint_ratio,
        },
        "adjacency": {
            "enabled": global_cfg.adjacency_enabled,
            "scope_escalation": global_cfg.adjacency_scope_escalation,
        },
        "sdsu": {
            "allow_purge": global_cfg.sdsu_allow_purge,
            "purge_cost_weight": global_cfg.sdsu_purge_cost_weight,
        },
        "milp": {
            "solver": global_cfg.milp_solver,
            "time_limit_sec": global_cfg.milp_time_limit_sec,
            "mip_gap": global_cfg.milp_mip_gap,
            "max_jobs_per_instance": global_cfg.milp_max_jobs_per_instance,
            "threads": global_cfg.milp_threads,
            "log_to_console": global_cfg.milp_log_to_console,
        },
        "step06": dataclasses.asdict(step06_cfg),
    }
    lines.append(json.dumps(cfg_snapshot, indent=2))
    lines.append("\n```\n")

    lines.append("\n## Summary / Diagnostics\n\n")
    lines.append("| Instance | n_jobs | status | f1_milp | f1_baseline | solver_gap | time_sec |\n")
    lines.append("|---------:|-------:|--------|--------:|------------:|-----------:|---------:|\n")
    for r in results:
        lines.append(
            f"| {r.meta.instance_tag} | {r.meta.n_jobs} | {r.status} | "
            f"{'' if r.f1_milp is None else f'{r.f1_milp:.3f}'} | "
            f"{'' if r.f1_baseline is None else f'{r.f1_baseline:.3f}'} | "
            f"{'' if r.solver_mip_gap is None else f'{r.solver_mip_gap:.4f}'} | "
            f"{'' if r.solver_time_sec is None else f'{r.solver_time_sec:.2f}'} |\n"
        )

    with path.open("w", encoding="utf-8") as f:
        f.write("".join(lines))


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def load_configs(global_config_path: Path, steps_config_path: Path) -> Tuple[GlobalConfig, Step06Config]:
    gc_raw = load_toml(global_config_path)
    sc_raw = load_toml(steps_config_path)

    paths = gc_raw["paths"]
    tokens = gc_raw["tokens"]
    adjacency = gc_raw["adjacency"]
    sdsu = gc_raw["sdsu"]
    milp = gc_raw["milp"]

    gc = GlobalConfig(
        dataset_dir=Path(paths["dataset_dir"]),
        output_dir=Path(paths["output_dir"]),
        manifests_dir=Path(paths.get("manifests_dir", os.path.join(paths["output_dir"], "manifests"))),
        reports_dir=Path(paths.get("reports_dir", os.path.join(paths["output_dir"], "reports"))),
        tokens_flush_units_ratio=float(tokens["flush_units_ratio"]),
        tokens_empty_units_prepaint=float(tokens["empty_units_prepaint"]),
        tokens_empty_units_postpaint_ratio=float(tokens["empty_units_postpaint_ratio"]),
        adjacency_enabled=bool(adjacency.get("enabled", True)),
        adjacency_scope_escalation=str(adjacency.get("scope_escalation", "")),
        sdsu_allow_purge=bool(sdsu.get("allow_purge", True)),
        sdsu_purge_cost_weight=float(sdsu.get("purge_cost_weight", 1.0)),
        milp_solver=str(milp.get("solver", "gurobi")),
        milp_time_limit_sec=float(milp.get("time_limit_sec", 600.0)),
        milp_mip_gap=float(milp.get("mip_gap", 0.01)),
        milp_max_jobs_per_instance=int(milp.get("max_jobs_per_instance", 120)),
        milp_threads=int(milp.get("threads", 0)),
        milp_log_to_console=bool(milp.get("log_to_console", True)),
        data_files=gc_raw["data_files"],
        data_schema=gc_raw.get("data_schema", {}),
    )

    if gc.milp_solver.lower() != "gurobi":
        raise SystemExit(
            f"Only 'gurobi' solver is supported in STEP-06-from-STEP01 implementation; got '{gc.milp_solver}'."
        )

    sc_step06 = sc_raw["step06"]
    step06 = Step06Config(
        milp_output_prefix=str(sc_step06["milp_output_prefix"]).replace("step06", "step06_from_step01"),
        manifest_relpath=str(sc_step06["manifest_relpath"]).replace("step06", "step06_from_step01"),
        report_relpath=str(sc_step06["report_relpath"]).replace("step06", "step06_from_step01"),
        enable_reduced_instances=bool(sc_step06.get("enable_reduced_instances", True)),
    )

    return gc, step06


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="STEP-06 (from STEP-01) — Global MILP baseline (assignment + sequencing)"
    )
    parser.add_argument(
        "--global-config",
        type=Path,
        default=Path("global_config.toml"),
        help="Path to global_config.toml",
    )
    parser.add_argument(
        "--steps-config",
        type=Path,
        default=Path("steps_config.toml"),
        help="Path to steps_config.toml",
    )
    args = parser.parse_args(argv)

    global_cfg, step06_cfg = load_configs(args.global_config, args.steps_config)

    dataset_dir = global_cfg.dataset_dir
    output_dir = global_cfg.output_dir

    # STEP-01 job universe
    steps_cfg_raw = load_toml(args.steps_config)
    step01_cfg = steps_cfg_raw["step01"]
    jobs_relpath = step01_cfg["jobs_expanded_relpath"]
    jobs_df = load_step01_jobs(output_dir, jobs_relpath, global_cfg.data_schema)
    n_jobs = len(jobs_df)

    # Dataset components
    s_matrix_path = dataset_dir / global_cfg.data_files["s_matrix_transition"]
    topcoat_rules_path = dataset_dir / global_cfg.data_files["topcoat_rules"]
    topcoat_codebook_path = dataset_dir / global_cfg.data_files["topcoat_codebook"]
    adjacency_rules_path = dataset_dir / global_cfg.data_files["adjacency_rules"]

    s_matrix = load_s_matrix_transition(s_matrix_path, global_cfg.data_schema)
    topcoat_no_to_code = load_topcoat_codebook(topcoat_codebook_path)
    explicit_forbids = load_topcoat_forbids(
        topcoat_rules_path,
        topcoat_no_to_code,
        global_cfg.data_schema,
    )
    adjacency_forbids = load_adjacency_forbids(
        adjacency_rules_path,
        global_cfg.data_schema,
        enabled=global_cfg.adjacency_enabled,
    )

    flush_cost_units = global_cfg.tokens_flush_units_ratio * global_cfg.sdsu_purge_cost_weight
    paint_kernel = PaintCostKernel(
        s_matrix=s_matrix,
        explicit_forbids=explicit_forbids,
        flush_cost_units=float(flush_cost_units),
        allow_purge=global_cfg.sdsu_allow_purge,
    )

    # Gate via enable_reduced_instances
    results: List[InstanceResult] = []
    output_files: List[Path] = []
    input_files: List[Path] = [
        args.global_config,
        args.steps_config,
        s_matrix_path,
        topcoat_rules_path,
        topcoat_codebook_path,
        adjacency_rules_path,
        output_dir / jobs_relpath,
    ]

    meta = InstanceMeta(instance_tag="from_step01_full", n_jobs=n_jobs)

    if not step06_cfg.enable_reduced_instances:
        # Global MILP disabled by config
        result = InstanceResult(
            meta=meta,
            status="DISABLED_BY_CONFIG",
            f1_milp=None,
            f1_baseline=None,
            solver_mip_gap=None,
            solver_time_sec=None,
            output_relpath=None,
        )
        results.append(result)
    elif n_jobs > global_cfg.milp_max_jobs_per_instance:
        # Too large for global MILP; user should pre-reduce STEP-01 jobs
        result = InstanceResult(
            meta=meta,
            status="TOO_LARGE_GLOBAL",
            f1_milp=None,
            f1_baseline=None,
            solver_mip_gap=None,
            solver_time_sec=None,
            output_relpath=None,
        )
        results.append(result)
    else:
        # Compute baseline f1 on STEP-01 order
        f1_baseline = evaluate_baseline_cost(
            jobs_df=jobs_df,
            paint_kernel=paint_kernel,
            adjacency_forbids=adjacency_forbids,
        )

        booth_sequences, result = build_and_solve_global_milp(
            meta=meta,
            jobs_df=jobs_df,
            paint_kernel=paint_kernel,
            adjacency_forbids=adjacency_forbids,
            milp_cfg=global_cfg,
        )
        result.f1_baseline = f1_baseline

        if booth_sequences is not None and result.f1_milp is not None:
            # Write combined per-booth sequence CSV
            out_rel = f"{step06_cfg.milp_output_prefix}{meta.instance_tag}.csv"
            out_path = output_dir / out_rel
            ensure_dir(out_path)

            rows: List[Dict[str, object]] = []
            # booth_sequences: {1: [job_idx,...], 2: [job_idx,...]}
            for b, seq in booth_sequences.items():
                cum_cost = 0.0
                for pos, job_idx in enumerate(seq):
                    row = jobs_df.iloc[job_idx]
                    job_id = row["job_id"]
                    part_id = row["part_id"]
                    coat = row["topcoat_code"]

                    if pos == 0:
                        setup_units = 0.0
                        purge_units = 0.0
                        arc_cost_units = 0.0
                    else:
                        prev_job_idx = seq[pos - 1]
                        prev_row = jobs_df.iloc[prev_job_idx]
                        prev_coat = prev_row["topcoat_code"]
                        setup_units, purge_units = paint_kernel.split_arc_cost(str(prev_coat), str(coat))
                        arc_cost_units = setup_units + purge_units
                        cum_cost += arc_cost_units

                    rows.append(
                        {
                            "job_id": job_id,
                            "part_id": part_id,
                            "topcoat_code": coat,
                            "booth_id": b,
                            "milp_seq_index": pos + 1,
                            "arc_setup_units": setup_units,
                            "arc_purge_units": purge_units,
                            "arc_cost_units": arc_cost_units,
                            "cum_cost_units": cum_cost,
                            "instance_tag": meta.instance_tag,
                        }
                    )

            out_df = pd.DataFrame(rows)
            out_df.to_csv(out_path, index=False)
            result.output_relpath = out_rel
            output_files.append(out_path)

        results.append(result)

    # Manifest and report
    manifest_path = global_cfg.output_dir / step06_cfg.manifest_relpath
    report_path = global_cfg.output_dir / step06_cfg.report_relpath

    write_manifest(
        path=manifest_path,
        global_cfg=global_cfg,
        step06_cfg=step06_cfg,
        inputs=input_files,
        outputs=output_files + [manifest_path, report_path],
        results=results,
    )
    write_report(
        path=report_path,
        global_cfg=global_cfg,
        step06_cfg=step06_cfg,
        results=results,
    )


if __name__ == "__main__":
    main()
