#!/usr/bin/env python3
"""
STEP-06 — MILP (Exact / Reduced) Spot Checks

This script implements the MILP step for the MOFFS paintshop study.

- Reads configuration from `global_config.toml` and `steps_config.toml`
- Loads paint/adjacency data from the dataset directory
- Constructs per-booth MILP instances on SDSU booth outputs (STEP-04)
- Solves an open-path ATSP-style model to minimise paint setup cost f1
- Writes per-instance optimal sequences and a manifest / report

The implementation is self-contained and does not depend on other project modules.
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
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import tomllib

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception as exc:  # pragma: no cover - import-time guard
    raise SystemExit(
        "ERROR: gurobipy is required to run STEP-06 MILP.\n"
        "Please install Gurobi Python bindings and ensure a license is available."
    ) from exc


# ---------------------------------------------------------------------------
# Small utility helpers
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
    raise KeyError(f"Required column for '{logical_name}' not found in dataframe. "
                   f"Tried candidates: {candidates}. Available: {list(df.columns)}")


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
class MILPInstanceMeta:
    instance_tag: str
    booth_id: int
    n_jobs: int
    is_full_booth: bool
    is_reduced: bool
    source_relpath: str


@dataclasses.dataclass
class MILPInstanceResult:
    meta: MILPInstanceMeta
    status: str
    f1_milp: Optional[float]
    f1_sdsu: Optional[float]
    solver_mip_gap: Optional[float]
    solver_time_sec: Optional[float]
    output_relpath: Optional[str]


# ---------------------------------------------------------------------------
# Dataset loaders (paint cost matrix, topcoat rules, adjacency)
# ---------------------------------------------------------------------------


def load_s_matrix_transition(path: Path, data_schema: Mapping[str, Mapping]) -> Dict[Tuple[str, str], float]:
    """
    Load s_matrix_transition.csv as a dense asymmetric cost matrix.

    Format (for this dataset):
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

    This is used to map numeric IDs from topcoat_rules into color codes that match
    the s_matrix and job tables.
    """
    df = pd.read_csv(path)
    if "topcoat_no" not in df.columns or "topcoat_code" not in df.columns:
        # Dataset misconfigured; skip numeric mapping (we'll rely purely on s_matrix)
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

    For this dataset, topcoat_rules encodes only explicit C1 forbids; FLUSH cannot
    override these forbids.
    """
    df = pd.read_csv(path)
    schema = data_schema.get("topcoat_rules", {})

    from_no_col = resolve_column(df, schema.get("from_topcoat_code_synonyms", ["from_topcoat_no"]), "topcoat_rules.from")
    to_no_col = resolve_column(df, schema.get("to_topcoat_code_synonyms", ["to_topcoat_no"]), "topcoat_rules.to")
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
            # If numeric mapping fails, skip this row (we keep forbids from s_matrix anyway)
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

    Scope is not filtered here; STEP-04 semantics already determined that adjacency
    applies at the booth level as a hard arc filter.
    """
    if not enabled:
        return []

    df = pd.read_csv(path)
    schema = data_schema.get("adjacency_rules", {})

    from_col = resolve_column(df, schema.get("from_part_id_synonyms", ["from_part_id"]), "adjacency.from_part_id")
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
# Cost kernel (paint + purge)
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

        - If transition is forbidden (matrix INF or explicit forbid): returns None.
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

        Uses the same logic as `arc_cost`, but classifies whether the cheaper
        option is a direct transition or a purge.
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
# SDSU booth outputs → MILP instances
# ---------------------------------------------------------------------------


def load_sdsu_booth_csv(output_dir: Path, booth_output_relpath: str, data_schema: Mapping[str, Mapping]) -> pd.DataFrame:
    """
    Load a SDSU per-booth CSV (JOB-only sequence).
    """
    path = output_dir / booth_output_relpath
    if not path.exists():
        raise FileNotFoundError(f"SDSU booth output not found: {path}")
    df = pd.read_csv(path)

    # Canonicalise key columns based on demand schema
    dem_schema = data_schema.get("demands", {})
    part_col = resolve_column(df, dem_schema.get("part_id_synonyms", ["part_id"]), "job.part_id")
    coat_col = resolve_column(df, dem_schema.get("topcoat_code_synonyms", ["topcoat_code"]), "job.topcoat_code")

    if "job_id" not in df.columns:
        raise KeyError("Expected 'job_id' column in SDSU booth CSV; not found.")

    # Normalise: ensure we have exactly the names we will use later
    df = df.copy()
    if part_col != "part_id":
        df.rename(columns={part_col: "part_id"}, inplace=True)
    if coat_col != "topcoat_code":
        df.rename(columns={coat_col: "topcoat_code"}, inplace=True)

    return df[["job_id", "part_id", "topcoat_code"]].copy()


def build_instances_for_booth(
    booth_id: int,
    sdsu_df: pd.DataFrame,
    step06_cfg: Step06Config,
    milp_max_jobs_per_instance: int,
) -> List[MILPInstanceMeta]:
    """
    Build MILP instances for a given booth: full-booth model if small enough,
    otherwise contiguous windows in SDSU order when reduced instances are enabled.
    """
    n_jobs = len(sdsu_df)
    instances: List[MILPInstanceMeta] = []

    source_relpath = f"step04_booth{booth_id}_sdsu.csv"

    if n_jobs == 0:
        return instances

    if n_jobs <= milp_max_jobs_per_instance:
        instances.append(
            MILPInstanceMeta(
                instance_tag=f"pb{booth_id}_full",
                booth_id=booth_id,
                n_jobs=n_jobs,
                is_full_booth=True,
                is_reduced=False,
                source_relpath=source_relpath,
            )
        )
        return instances

    # Reduced windows (non-overlapping contiguous segments in SDSU order)
    window_length = milp_max_jobs_per_instance
    num_windows = math.ceil(n_jobs / window_length)

    for w in range(num_windows):
        start_idx = w * window_length
        end_idx = min((w + 1) * window_length, n_jobs)
        n_w = end_idx - start_idx
        tag = f"pb{booth_id}_win{w+1:04d}"
        instances.append(
            MILPInstanceMeta(
                instance_tag=tag,
                booth_id=booth_id,
                n_jobs=n_w,
                is_full_booth=False,
                is_reduced=True,
                source_relpath=source_relpath,
            )
        )

    return instances


# ---------------------------------------------------------------------------
# MILP model construction and solving
# ---------------------------------------------------------------------------


def build_and_solve_milp(
    meta: MILPInstanceMeta,
    jobs_df: pd.DataFrame,
    paint_kernel: PaintCostKernel,
    adjacency_forbids: List[Tuple[str, str]],
    milp_cfg: GlobalConfig,
) -> Tuple[Optional[List[int]], MILPInstanceResult]:
    """
    Build and solve the MILP for a single instance.

    Parameters
    ----------
    meta : MILPInstanceMeta
        Instance metadata (booth id, tag, size).
    jobs_df : DataFrame
        Subset of SDSU booth dataframe for this instance, in SDSU order.
    paint_kernel : PaintCostKernel
        Paint cost and forbid semantics.
    adjacency_forbids : list of (from_part_id, to_part_id)
        Forbidden part adjacency pairs (applied to successive jobs).
    milp_cfg : GlobalConfig
        Global config with MILP parameters.

    Returns
    -------
    order_indices : Optional[List[int]]
        Optimal job indices in jobs_df order (0-based indices). None if infeasible.
    result : MILPInstanceResult
        Solver status and metrics (without output_relpath filled yet).
    """
    n = meta.n_jobs
    assert len(jobs_df) == n

    # Precompute arrays for convenience
    job_ids = list(jobs_df["job_id"])
    parts = list(jobs_df["part_id"])
    coats = list(jobs_df["topcoat_code"])

    # Nodes: 0 = START, 1..n = real jobs, n+1 = END
    start_node = 0
    end_node = n + 1
    job_nodes = list(range(1, n + 1))

    # Build arc list with costs
    # Each arc is (tail, head, cost, is_job_job)
    arcs: List[Tuple[int, int, float, bool]] = []
    arc_index: Dict[Tuple[int, int], int] = {}

    # Start -> jobs (no cost; first-job setup omitted as constant)
    for j_node in job_nodes:
        idx = len(arcs)
        arcs.append((start_node, j_node, 0.0, False))
        arc_index[(start_node, j_node)] = idx

    # Job -> job arcs
    forbidden_adj_set = set(adjacency_forbids)

    for i_pos, i_node in enumerate(job_nodes):
        part_i = str(parts[i_pos])
        coat_i = str(coats[i_pos])

        for j_pos, j_node in enumerate(job_nodes):
            if i_node == j_node:
                continue

            part_j = str(parts[j_pos])
            coat_j = str(coats[j_pos])

            # Adjacency forbid?
            if (part_i, part_j) in forbidden_adj_set:
                continue

            # Paint feasibility and cost
            c = paint_kernel.arc_cost(coat_i, coat_j)
            if c is None or math.isinf(c):
                continue

            idx = len(arcs)
            arcs.append((i_node, j_node, float(c), True))
            arc_index[(i_node, j_node)] = idx

    # Job -> END arcs (no cost)
    for i_node in job_nodes:
        idx = len(arcs)
        arcs.append((i_node, end_node, 0.0, False))
        arc_index[(i_node, end_node)] = idx

    # Sanity: every job must have at least one incoming and outgoing arc
    arcs_by_tail: Dict[int, List[int]] = {}
    arcs_by_head: Dict[int, List[int]] = {}
    for k, (i, j, _, _) in enumerate(arcs):
        arcs_by_tail.setdefault(i, []).append(k)
        arcs_by_head.setdefault(j, []).append(k)

    for j_node in job_nodes:
        if j_node not in arcs_by_head or not arcs_by_head[j_node]:
            # Infeasible instance under current semantics
            result = MILPInstanceResult(
                meta=meta,
                status="INFEASIBLE_ARC_IN",
                f1_milp=None,
                f1_sdsu=None,
                solver_mip_gap=None,
                solver_time_sec=None,
                output_relpath=None,
            )
            return None, result
        if j_node not in arcs_by_tail or not arcs_by_tail[j_node]:
            result = MILPInstanceResult(
                meta=meta,
                status="INFEASIBLE_ARC_OUT",
                f1_milp=None,
                f1_sdsu=None,
                solver_mip_gap=None,
                solver_time_sec=None,
                output_relpath=None,
            )
            return None, result

    # Build MILP model
    model = gp.Model(f"step06_{meta.instance_tag}")
    if not milp_cfg.milp_log_to_console:
        model.Params.LogToConsole = 0

    model.Params.TimeLimit = float(milp_cfg.milp_time_limit_sec)
    model.Params.MIPGap = float(milp_cfg.milp_mip_gap)
    if milp_cfg.milp_threads > 0:
        model.Params.Threads = int(milp_cfg.milp_threads)

    # Binary vars for arcs
    x = model.addVars(len(arcs), vtype=GRB.BINARY, name="x")

    # MTZ order variables for jobs only
    u = model.addVars(job_nodes, lb=1.0, ub=float(n), vtype=GRB.CONTINUOUS, name="u")

    # Objective: sum of arc costs
    model.setObjective(
        gp.quicksum(arcs[k][2] * x[k] for k in range(len(arcs))),
        GRB.MINIMIZE,
    )

    # Flow constraints
    # Start node: exactly one outgoing arc
    model.addConstr(gp.quicksum(x[k] for k in arcs_by_tail[start_node]) == 1.0, name="start_out")

    # End node: exactly one incoming arc
    model.addConstr(gp.quicksum(x[k] for k in arcs_by_head[end_node]) == 1.0, name="end_in")

    # Each job: exactly one incoming (from START or another job) and one outgoing (to another job or END)
    for j_node in job_nodes:
        model.addConstr(gp.quicksum(x[k] for k in arcs_by_head[j_node]) == 1.0, name=f"job_in_{j_node}")
        model.addConstr(gp.quicksum(x[k] for k in arcs_by_tail[j_node]) == 1.0, name=f"job_out_{j_node}")

    # MTZ subtour-elimination constraints on job-job arcs only
    big_M = float(n)
    for i_node in job_nodes:
        for j_node in job_nodes:
            if i_node == j_node:
                continue
            key = (i_node, j_node)
            if key not in arc_index:
                continue
            k = arc_index[key]
            model.addConstr(
                u[i_node] - u[j_node] + 1.0 <= big_M * (1.0 - x[k]),
                name=f"mtz_{i_node}_{j_node}",
            )

    # Solve
    model.optimize()

    status_code = model.Status
    if status_code in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
        result = MILPInstanceResult(
            meta=meta,
            status="INFEASIBLE",
            f1_milp=None,
            f1_sdsu=None,
            solver_mip_gap=None,
            solver_time_sec=float(model.Runtime),
            output_relpath=None,
        )
        return None, result
    
    # Case 1: provably no solution or none found
    if status_code in (GRB.INFEASIBLE, GRB.INF_OR_UNBD, GRB.UNBOUNDED) or model.SolCount == 0:
    # construct a MILPInstanceResult that with "no solution"
        result = MILPInstanceResult(
            meta=meta,
            status=f"NO_SOLUTION_{status_code}",
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

    # Extract solution
    # Build successor map from chosen arcs
    succ: Dict[int, int] = {}
    for k, (i, j, _, _) in enumerate(arcs):
        if x[k].X > 0.5:
            succ[i] = j

    # Recover path from START to END
    order_nodes: List[int] = []
    cur = start_node
    visited = set()
    while True:
        if cur in visited:
            # Should not happen if MTZ worked; treat as error
            break
        visited.add(cur)
        nxt = succ.get(cur)
        if nxt is None:
            break
        if nxt == end_node:
            break
        order_nodes.append(nxt)
        cur = nxt

    if len(order_nodes) != n:
        # Something went wrong; treat as failure
        result = MILPInstanceResult(
            meta=meta,
            status="DECODE_ERROR",
            f1_milp=None,
            f1_sdsu=None,
            solver_mip_gap=float(model.MIPGap),
            solver_time_sec=float(model.Runtime),
            output_relpath=None,
        )
        return None, result

    # Map node indices back to 0-based jobs_df indices
    order_indices = [node - 1 for node in order_nodes]

    # Compute MILP f1 using the same arc_cost kernel
    total_f1 = 0.0
    for pos in range(len(order_indices) - 1):
        i_idx = order_indices[pos]
        j_idx = order_indices[pos + 1]
        c_i = str(coats[i_idx])
        c_j = str(coats[j_idx])
        c_val = paint_kernel.arc_cost(c_i, c_j)
        if c_val is None or math.isinf(c_val):
            # This should not happen; treat as diagnostic
            result = MILPInstanceResult(
                meta=meta,
                status="EVAL_INFEASIBLE",
                f1_milp=None,
                f1_sdsu=None,
                solver_mip_gap=float(model.MIPGap),
                solver_time_sec=float(model.Runtime),
                output_relpath=None,
            )
            return None, result
        total_f1 += float(c_val)

    result = MILPInstanceResult(
        meta=meta,
        status=status_str,
        f1_milp=total_f1,
        f1_sdsu=None,  # filled outside
        solver_mip_gap=float(model.MIPGap),
        solver_time_sec=float(model.Runtime),
        output_relpath=None,
    )
    return order_indices, result


# ---------------------------------------------------------------------------
# SDSU cost evaluation on same kernel
# ---------------------------------------------------------------------------


def evaluate_sdsu_cost(
    jobs_df: pd.DataFrame,
    paint_kernel: PaintCostKernel,
    adjacency_forbids: List[Tuple[str, str]],
) -> Optional[float]:
    """
    Compute f1 for the given SDSU order using the same cost kernel and adjacency rules.

    Returns None if any consecutive pair is infeasible.
    """
    if jobs_df.empty:
        return 0.0

    job_ids = list(jobs_df["job_id"])
    parts = list(jobs_df["part_id"])
    coats = list(jobs_df["topcoat_code"])
    forbidden_adj_set = set(adjacency_forbids)

    total = 0.0
    for i in range(len(job_ids) - 1):
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
# Manifest and report
# ---------------------------------------------------------------------------


def write_manifest(
    path: Path,
    global_cfg: GlobalConfig,
    step06_cfg: Step06Config,
    inputs: List[Path],
    outputs: List[Path],
    results: List[MILPInstanceResult],
) -> None:
    ensure_dir(path)
    data = {
        "step_id": "STEP06_MILP",
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
                "booth_id": r.meta.booth_id,
                "n_jobs": r.meta.n_jobs,
                "is_full_booth": r.meta.is_full_booth,
                "is_reduced": r.meta.is_reduced,
                "source_relpath": r.meta.source_relpath,
                "status": r.status,
                "f1_milp": r.f1_milp,
                "f1_sdsu": r.f1_sdsu,
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
    results: List[MILPInstanceResult],
) -> None:
    ensure_dir(path)
    lines: List[str] = []
    lines.append("# STEP-06 — MILP (Exact / Reduced) Spot Checks\n")
    lines.append("## Purpose / What / Why\n")
    lines.append(
        "- **Purpose:** Compute exact or near-exact f1 paint setup costs on selected per-booth instances.\n"
    )
    lines.append(
        "- **What:** MILP open-path ATSP models on SDSU booth job sets with the same C1/C2 semantics used in STEP-04.\n"
    )
    lines.append(
        "- **Why:** Provide benchmarks to evaluate SDSU quality and anchor NSGA-II / DES results against exact baselines.\n"
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
    lines.append("| Instance | Booth | n_jobs | status | f1_milp | f1_sdsu | solver_gap | time_sec |\n")
    lines.append("|---------:|------:|-------:|--------|--------:|--------:|-----------:|---------:|\n")
    for r in results:
        lines.append(
            f"| {r.meta.instance_tag} | {r.meta.booth_id} | {r.meta.n_jobs} | {r.status} | "
            f"{'' if r.f1_milp is None else f'{r.f1_milp:.3f}'} | "
            f"{'' if r.f1_sdsu is None else f'{r.f1_sdsu:.3f}'} | "
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
        raise SystemExit(f"Only 'gurobi' solver is supported in STEP-06 implementation; got '{gc.milp_solver}'.")

    sc_step06 = sc_raw["step06"]
    step06 = Step06Config(
        milp_output_prefix=str(sc_step06["milp_output_prefix"]),
        manifest_relpath=str(sc_step06["manifest_relpath"]),
        report_relpath=str(sc_step06["report_relpath"]),
        enable_reduced_instances=bool(sc_step06.get("enable_reduced_instances", True)),
    )

    return gc, step06


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="STEP-06 — MILP (Exact / Reduced) Spot Checks")
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

    # Prepare important paths
    dataset_dir = global_cfg.dataset_dir
    output_dir = global_cfg.output_dir

    # Load dataset components
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

    # Load SDSU per-booth outputs
    steps_cfg_raw = load_toml(args.steps_config)
    step04_cfg = steps_cfg_raw["step04"]

    booth1_sdsu_rel = step04_cfg["booth1_output_relpath"]
    booth2_sdsu_rel = step04_cfg["booth2_output_relpath"]

    booth1_df = load_sdsu_booth_csv(output_dir, booth1_sdsu_rel, global_cfg.data_schema)
    booth2_df = load_sdsu_booth_csv(output_dir, booth2_sdsu_rel, global_cfg.data_schema)

    # Build instances
    instances: List[MILPInstanceMeta] = []
    instances += build_instances_for_booth(
        booth_id=1,
        sdsu_df=booth1_df,
        step06_cfg=step06_cfg,
        milp_max_jobs_per_instance=global_cfg.milp_max_jobs_per_instance,
    )
    instances += build_instances_for_booth(
        booth_id=2,
        sdsu_df=booth2_df,
        step06_cfg=step06_cfg,
        milp_max_jobs_per_instance=global_cfg.milp_max_jobs_per_instance,
    )

    results: List[MILPInstanceResult] = []
    output_files: List[Path] = []
    input_files: List[Path] = [
        args.global_config,
        args.steps_config,
        s_matrix_path,
        topcoat_rules_path,
        topcoat_codebook_path,
        adjacency_rules_path,
        output_dir / booth1_sdsu_rel,
        output_dir / booth2_sdsu_rel,
    ]

    for meta in instances:
        if meta.booth_id == 1:
            src_df = booth1_df
        else:
            src_df = booth2_df

        # Determine slice for this instance
        if meta.is_full_booth:
            inst_df = src_df.copy()
        else:
            # Window instances: decode window index from tag
            # tag format: pb{b}_winNNNN
            try:
                win_str = meta.instance_tag.split("win", 1)[1]
                w = int(win_str)
            except Exception as exc:
                raise RuntimeError(f"Cannot parse window index from instance_tag '{meta.instance_tag}'") from exc

            window_len = global_cfg.milp_max_jobs_per_instance
            start_idx = (w - 1) * window_len
            end_idx = min(start_idx + window_len, len(src_df))
            inst_df = src_df.iloc[start_idx:end_idx].reset_index(drop=True)

        order_indices, result = build_and_solve_milp(
            meta=meta,
            jobs_df=inst_df,
            paint_kernel=paint_kernel,
            adjacency_forbids=adjacency_forbids,
            milp_cfg=global_cfg,
        )

        # Evaluate SDSU cost on the same subset (SDSU order is simply inst_df order)
        f1_sdsu = evaluate_sdsu_cost(
            jobs_df=inst_df,
            paint_kernel=paint_kernel,
            adjacency_forbids=adjacency_forbids,
        )
        result.f1_sdsu = f1_sdsu

        # If we have a MILP solution, write CSV
        if order_indices is not None and result.f1_milp is not None:
            out_rel = f"{step06_cfg.milp_output_prefix}{meta.instance_tag}.csv"
            out_path = output_dir / out_rel
            ensure_dir(out_path)

            ordered_jobs = inst_df.iloc[order_indices].reset_index(drop=True)

            rows: List[Dict[str, object]] = []
            cum_cost = 0.0

            for pos in range(len(ordered_jobs)):
                row = ordered_jobs.iloc[pos]
                job_id = row["job_id"]
                part_id = row["part_id"]
                coat = row["topcoat_code"]

                if pos == 0:
                    setup_units = 0.0
                    purge_units = 0.0
                    arc_cost_units = 0.0
                else:
                    prev_row = ordered_jobs.iloc[pos - 1]
                    prev_coat = prev_row["topcoat_code"]
                    setup_units, purge_units = paint_kernel.split_arc_cost(str(prev_coat), str(coat))
                    arc_cost_units = setup_units + purge_units
                    cum_cost += arc_cost_units

                rows.append(
                    {
                        "job_id": job_id,
                        "part_id": part_id,
                        "topcoat_code": coat,
                        "booth_id": meta.booth_id,
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
    manifest_path = output_dir / step06_cfg.manifest_relpath
    report_path = output_dir / step06_cfg.report_relpath

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
