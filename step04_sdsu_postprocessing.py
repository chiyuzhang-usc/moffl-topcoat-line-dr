#!/usr/bin/env python3
"""
Step-04 — SDSU Post-Processing (Optional PB1↔PB2 Exchange Improvement)

This script implements the optional SDSU post-processing step described in
STEP04_SDSU_POSTPROCESSING.md. It operates between per-booth SDSU (Step-04 core)
and the 2→1 merge (Step-05).

High-level behaviour:

- Inputs: per-booth SDSU sequences from Step-04 core.
- Neighbourhood: PB1↔PB2 swaps of contiguous blocks of jobs (SDSU order).
- For each candidate exchange:
    - Form new job sets for PB1 and PB2.
    - Re-run per-booth SDSU (same data-driven C1/C2/C3 model as Step-04 core).
    - If total SDSU cost f1 (PB1+PB2) strictly decreases and both booths remain feasible,
      accept the exchange.
- Outputs: improved per-booth SDSU sequences and a moves log.
- Constraints: no relaxation of C1/C2/C3; any infeasible candidate is rejected.

The implementation is deterministic: scan order and SDSU solver have no randomness.
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
class Step04PostPaths:
    output_dir: Path
    dataset_dir: Path
    booth1_input: Path
    booth2_input: Path
    booth1_output: Path
    booth2_output: Path
    moves_log_path: Path
    manifest_path: Path
    report_path: Path
    s_matrix_path: Path
    topcoat_codebook_path: Path
    topcoat_rules_path: Path
    adjacency_rules_path: Path
    global_config_path: Path
    steps_config_path: Path


@dataclass
class Step04PostConfig:
    L_max: int
    max_passes: int
    search_policy: str  # "first_improvement" or "best_improvement"


@dataclass
class SDSUConfig:
    dp_threshold_jobs: int
    max_2opt_passes: int
    allow_purge: bool
    purge_cost_weight: float
    flush_units_ratio: float


@dataclass
class SchemaConfig:
    part_id_col: str
    topcoat_code_col: str
    job_id_col: str
    booth_id_col: str


@dataclass
class PaintModel:
    setup_cost: Dict[str, Dict[str, float]]  # cost[from_color][to_color]
    forbid_pairs_no: Set[Tuple[int, int]]    # forbids at topcoat_no level
    topcoat_code_to_no: Dict[str, int]


@dataclass
class AdjacencyModel:
    forbid_pairs: Set[Tuple[str, str]]       # in-booth/global forbids: (from_part_id, to_part_id)


@dataclass
class BoothInstance:
    booth_id: int
    df: pd.DataFrame
    schema: SchemaConfig
    colors: List[str]        # per-job topcoat codes
    parts: List[str]         # per-job part ids


@dataclass
class CostMatrices:
    use_cost: List[List[float]]             # cost matrix (finite or inf)
    flush_used: List[List[bool]]            # whether arc uses purge


@dataclass
class SDSUSolution:
    order: List[int]  # permutation of indices 0..n-1
    cost: float
    branch: str       # "dp" or "heuristic" or "empty"


@dataclass
class BoothResult:
    booth_id: int
    n_jobs: int
    branch: str
    total_setup_cost: float
    num_purges: int
    order_indices: List[int]
    flush_positions: List[int]


@dataclass
class MoveRecord:
    move_id: int
    block_length: int
    pb1_start: int
    pb2_start: int
    f1_before: float
    f1_after: float
    delta_f1: float
    branch_pb1: str
    branch_pb2: str


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------


def resolve_paths(global_cfg: dict, steps_cfg: dict) -> Step04PostPaths:
    paths_cfg = global_cfg["paths"]
    data_files = global_cfg["data_files"]

    output_dir = Path(paths_cfg["output_dir"])
    dataset_dir = Path(paths_cfg["dataset_dir"])

    step04_post_cfg = steps_cfg.get("step04_post", {})

    b1_in_rel = step04_post_cfg.get("booth1_input_relpath", "step04_booth1_sdsu.csv")
    b2_in_rel = step04_post_cfg.get("booth2_input_relpath", "step04_booth2_sdsu.csv")
    b1_out_rel = step04_post_cfg.get("booth1_output_relpath", "step04_post_booth1_sdsu.csv")
    b2_out_rel = step04_post_cfg.get("booth2_output_relpath", "step04_post_booth2_sdsu.csv")
    moves_rel = step04_post_cfg.get("moves_log_relpath", "step04_post_moves.json")
    manifest_rel = step04_post_cfg.get("manifest_relpath", "manifests/step04_post_manifest.json")
    report_rel = step04_post_cfg.get("report_relpath", "reports/step04_post_report.md")

    return Step04PostPaths(
        output_dir=output_dir,
        dataset_dir=dataset_dir,
        booth1_input=output_dir / b1_in_rel,
        booth2_input=output_dir / b2_in_rel,
        booth1_output=output_dir / b1_out_rel,
        booth2_output=output_dir / b2_out_rel,
        moves_log_path=output_dir / moves_rel,
        manifest_path=output_dir / manifest_rel,
        report_path=output_dir / report_rel,
        s_matrix_path=dataset_dir / data_files["s_matrix_transition"],
        topcoat_codebook_path=dataset_dir / data_files["topcoat_codebook"],
        topcoat_rules_path=dataset_dir / data_files["topcoat_rules"],
        adjacency_rules_path=dataset_dir / data_files["adjacency_rules"],
        global_config_path=Path("global_config.toml"),
        steps_config_path=Path("steps_config.toml"),
    )


def build_step04_post_config(global_cfg: dict, steps_cfg: dict) -> Step04PostConfig:
    step04_post_cfg = steps_cfg.get("step04_post", {})
    L_max = int(step04_post_cfg.get("L_max", 10))
    max_passes = int(step04_post_cfg.get("max_passes", 1))
    search_policy = str(step04_post_cfg.get("search_policy", "first_improvement")).lower()
    if search_policy not in ("first_improvement", "best_improvement"):
        raise ValueError(f"Unsupported search_policy={search_policy!r} in [step04_post]")
    return Step04PostConfig(L_max=L_max, max_passes=max_passes, search_policy=search_policy)


def build_sdsu_config(global_cfg: dict, steps_cfg: dict) -> SDSUConfig:
    sdsu_cfg = global_cfg.get("sdsu", {})
    step04_cfg = steps_cfg.get("step04", {})
    dp_threshold = int(step04_cfg.get("dp_threshold_jobs", sdsu_cfg.get("dp_threshold_jobs", 40)))
    max_passes = int(sdsu_cfg.get("max_2opt_passes", 5))
    allow_purge = bool(sdsu_cfg.get("allow_purge", True))
    purge_cost_weight = float(sdsu_cfg.get("purge_cost_weight", 1.0))
    flush_units_ratio = float(global_cfg.get("tokens", {}).get("flush_units_ratio", 1.0))
    return SDSUConfig(
        dp_threshold_jobs=dp_threshold,
        max_2opt_passes=max_passes,
        allow_purge=allow_purge,
        purge_cost_weight=purge_cost_weight,
        flush_units_ratio=flush_units_ratio,
    )


def resolve_schema(global_cfg: dict, df: pd.DataFrame) -> SchemaConfig:
    ds = global_cfg.get("data_schema", {})
    demands_schema = ds.get("demands", {})
    parts_schema = ds.get("parts", {})
    jobs_schema = ds.get("jobs", {})

    topcoat_syn = demands_schema.get("topcoat_code_synonyms", [])
    if not topcoat_syn:
        tcb_schema = ds.get("topcoat_codebook", {})
        topcoat_syn = tcb_schema.get("topcoat_code_synonyms", [])
    part_syn = demands_schema.get("part_id_synonyms", [])
    if not part_syn:
        part_syn = parts_schema.get("part_id_synonyms", [])
    job_syn = jobs_schema.get("job_id_synonyms", ["job_id"])

    part_col = pick_first_existing(df.columns, part_syn + ["part_id"])
    topcoat_col = pick_first_existing(df.columns, topcoat_syn + ["topcoat_code", "color_code"])
    job_col = pick_first_existing(df.columns, job_syn + ["job_id"])
    booth_col = pick_first_existing(df.columns, ["booth_id"])

    if part_col is None:
        raise KeyError(f"Could not resolve part_id column in SDSU post-processing input; candidates={part_syn}")
    if topcoat_col is None:
        raise KeyError(f"Could not resolve topcoat_code column in SDSU post-processing input; candidates={topcoat_syn}")
    if job_col is None:
        raise KeyError("Could not resolve job_id column in SDSU post-processing input")
    if booth_col is None:
        raise KeyError("Could not resolve booth_id column in SDSU post-processing input")

    return SchemaConfig(
        part_id_col=part_col,
        topcoat_code_col=topcoat_col,
        job_id_col=job_col,
        booth_id_col=booth_col,
    )


# ---------------------------------------------------------------------------
# Dataset-derived paint and adjacency models
# ---------------------------------------------------------------------------


def load_paint_model(paths: Step04PostPaths, global_cfg: dict) -> PaintModel:
    # topcoat codebook: code -> topcoat_no
    tcb = pd.read_csv(paths.topcoat_codebook_path)
    if "topcoat_code" not in tcb.columns or "topcoat_no" not in tcb.columns:
        raise KeyError("topcoat_codebook.csv must contain 'topcoat_code' and 'topcoat_no' columns")
    code_to_no = {str(row["topcoat_code"]): int(row["topcoat_no"]) for _, row in tcb.iterrows()}

    # transition matrix
    s_mat = pd.read_csv(paths.s_matrix_path)
    if "from" not in s_mat.columns:
        raise KeyError("s_matrix_transition.csv must contain 'from' column")
    color_cols = [c for c in s_mat.columns if c != "from"]

    setup_cost: Dict[str, Dict[str, float]] = {}
    for _, row in s_mat.iterrows():
        from_code = str(row["from"])
        row_costs: Dict[str, float] = {}
        for col in color_cols:
            val = row[col]
            if isinstance(val, str) and val.lower() == "inf":
                c = math.inf
            else:
                c = float(val)
            row_costs[col] = c
        setup_cost[from_code] = row_costs

    # topcoat_rules forbids in topcoat_no space
    trules = pd.read_csv(paths.topcoat_rules_path)
    forbid_pairs_no: Set[Tuple[int, int]] = set()
    if {"relation", "from_topcoat_no", "to_topcoat_no"}.issubset(trules.columns):
        for _, r in trules.iterrows():
            rel = str(r["relation"]).lower()
            if "forbid" not in rel:
                continue
            try:
                i_no = int(r["from_topcoat_no"])
                j_no = int(r["to_topcoat_no"])
                forbid_pairs_no.add((i_no, j_no))
            except Exception:
                continue

    return PaintModel(setup_cost=setup_cost, forbid_pairs_no=forbid_pairs_no, topcoat_code_to_no=code_to_no)


def load_adjacency_model_inbooth(paths: Step04PostPaths, global_cfg: dict) -> AdjacencyModel:
    adj_cfg = global_cfg.get("adjacency", {})
    enabled = bool(adj_cfg.get("enabled", True))
    if not enabled:
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

    if from_col is None or to_col is None or scope_col is None or rel_col is None:
        return AdjacencyModel(forbid_pairs=set())

    forbids: Set[Tuple[str, str]] = set()
    for _, r in df.iterrows():
        rel = str(r[rel_col]).lower()
        if "forbid" not in rel:
            continue
        scope_val = str(r[scope_col]).strip().lower()
        if scope_val not in ("in_booth", "global"):
            continue
        from_part = str(r[from_col])
        to_part = str(r[to_col])
        forbids.add((from_part, to_part))

    return AdjacencyModel(forbid_pairs=forbids)


# ---------------------------------------------------------------------------
# SDSU cost matrix and solver
# ---------------------------------------------------------------------------


def build_booth_instance(booth_id: int, df: pd.DataFrame, schema: SchemaConfig) -> BoothInstance:
    colors = [str(c) for c in df[schema.topcoat_code_col].tolist()]
    parts = [str(p) for p in df[schema.part_id_col].tolist()]
    return BoothInstance(booth_id=booth_id, df=df.copy(), schema=schema, colors=colors, parts=parts)


def build_cost_matrices_for_booth(
    inst: BoothInstance,
    paint: PaintModel,
    adjacency: AdjacencyModel,
    sdsu_cfg: SDSUConfig,
) -> CostMatrices:
    n = len(inst.colors)
    use_cost = [[math.inf for _ in range(n)] for _ in range(n)]
    flush_used = [[False for _ in range(n)] for _ in range(n)]

    flush_cost_units = sdsu_cfg.flush_units_ratio * sdsu_cfg.purge_cost_weight

    s0 = {color: 0.0 for color in set(inst.colors)}  # uniform start cost
    setup_cost = paint.setup_cost
    code_to_no = paint.topcoat_code_to_no
    forbid_pairs_no = paint.forbid_pairs_no
    adj_forbids = adjacency.forbid_pairs

    for i in range(n):
        ci = inst.colors[i]
        pi = inst.parts[i]
        row_i = setup_cost.get(ci)
        for j in range(n):
            if i == j:
                continue
            cj = inst.colors[j]
            pj = inst.parts[j]

            # in-booth/global adjacency forbids
            if (pi, pj) in adj_forbids:
                continue

            direct = math.inf
            if row_i is not None and cj in row_i:
                direct = row_i[cj]

            try:
                ni = code_to_no[ci]
                nj = code_to_no[cj]
                if (ni, nj) in forbid_pairs_no:
                    direct = math.inf
            except KeyError:
                pass

            direct_feasible = direct < math.inf

            purge_cost = math.inf
            if sdsu_cfg.allow_purge and cj in s0:
                purge_cost = flush_cost_units + s0[cj]

            if direct_feasible:
                if purge_cost < direct:
                    use_cost[i][j] = purge_cost
                    flush_used[i][j] = True
                else:
                    use_cost[i][j] = direct
                    flush_used[i][j] = False
            else:
                if purge_cost < math.inf:
                    use_cost[i][j] = purge_cost
                    flush_used[i][j] = True
                else:
                    continue

    return CostMatrices(use_cost=use_cost, flush_used=flush_used)


def sdsu_open_path_dp(cost: List[List[float]]) -> SDSUSolution:
    n = len(cost)
    if n == 0:
        return SDSUSolution(order=[], cost=0.0, branch="dp")
    if n == 1:
        return SDSUSolution(order=[0], cost=0.0, branch="dp")

    INF = math.inf
    size = 1 << n
    dp = [[INF] * n for _ in range(size)]
    prev = [[-1] * n for _ in range(size)]

    # base: singletons
    for j in range(n):
        mask = 1 << j
        dp[mask][j] = 0.0
        prev[mask][j] = -1

    for mask in range(1, size):
        if (mask & (mask - 1)) == 0:
            continue  # singleton
        for j in range(n):
            if not (mask & (1 << j)):
                continue
            prev_mask = mask ^ (1 << j)
            best_cost = dp[mask][j]
            best_i = prev[mask][j]
            for i in range(n):
                if not (prev_mask & (1 << i)):
                    continue
                cij = cost[i][j]
                if cij >= INF:
                    continue
                cand = dp[prev_mask][i] + cij
                if cand < best_cost:
                    best_cost = cand
                    best_i = i
            dp[mask][j] = best_cost
            prev[mask][j] = best_i

    full_mask = (1 << n) - 1
    best_end = -1
    best_cost = INF
    for j in range(n):
        if dp[full_mask][j] < best_cost:
            best_cost = dp[full_mask][j]
            best_end = j

    if best_end == -1 or best_cost >= INF:
        raise RuntimeError("SDSU DP: no feasible open path (all arcs infinite)")

    order = []
    mask = full_mask
    j = best_end
    while j != -1:
        order.append(j)
        pj = prev[mask][j]
        if pj == -1:
            break
        mask ^= 1 << j
        j = pj
    order.reverse()

    if len(order) != n:
        raise RuntimeError(f"SDSU DP: reconstructed path length {len(order)} != {n}")

    return SDSUSolution(order=order, cost=best_cost, branch="dp")


def path_cost(cost: List[List[float]], order: List[int]) -> float:
    total = 0.0
    INF = math.inf
    if not order:
        return 0.0
    for i in range(len(order) - 1):
        u = order[i]
        v = order[i + 1]
        c = cost[u][v]
        if c >= INF:
            return INF
        total += c
    return total


def sdsu_open_path_greedy_2opt(cost: List[List[float]], max_passes: int) -> SDSUSolution:
    n = len(cost)
    if n == 0:
        return SDSUSolution(order=[], cost=0.0, branch="heuristic")
    if n == 1:
        return SDSUSolution(order=[0], cost=0.0, branch="heuristic")

    INF = math.inf
    start = 0
    order = [start]
    used = [False] * n
    used[start] = True

    # greedy construction
    for _ in range(1, n):
        last = order[-1]
        best_j = -1
        best_c = INF
        for j in range(n):
            if used[j]:
                continue
            c = cost[last][j]
            if c < best_c:
                best_c = c
                best_j = j
        if best_j == -1 or best_c >= INF:
            raise RuntimeError("SDSU greedy: no feasible successor; instance infeasible")
        order.append(best_j)
        used[best_j] = True

    best_cost = path_cost(cost, order)
    if best_cost >= INF:
        raise RuntimeError("SDSU greedy: constructed path has infinite cost; instance infeasible")

    # 2-opt improvement
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for a in range(0, n - 2):
            for b in range(a + 1, n - 1):
                new_order = order[: a + 1] + list(reversed(order[a + 1 : b + 1])) + order[b + 1 :]
                new_cost = path_cost(cost, new_order)
                if new_cost < best_cost - 1e-9:
                    order = new_order
                    best_cost = new_cost
                    improved = True
                    break
            if improved:
                break

    return SDSUSolution(order=order, cost=best_cost, branch="heuristic")


def solve_sdsu_for_booth(inst: BoothInstance, matrices: CostMatrices, sdsu_cfg: SDSUConfig) -> BoothResult:
    n = len(inst.colors)
    if n == 0:
        return BoothResult(
            booth_id=inst.booth_id,
            n_jobs=0,
            branch="empty",
            total_setup_cost=0.0,
            num_purges=0,
            order_indices=[],
            flush_positions=[],
        )

    use_cost = matrices.use_cost
    flush_used = matrices.flush_used

    if n <= sdsu_cfg.dp_threshold_jobs:
        sol = sdsu_open_path_dp(use_cost)
    else:
        sol = sdsu_open_path_greedy_2opt(use_cost, max_passes=sdsu_cfg.max_2opt_passes)

    order = sol.order
    total_cost = sol.cost

    flush_positions: List[int] = []
    num_purges = 0
    for k in range(len(order) - 1):
        i = order[k]
        j = order[k + 1]
        if flush_used[i][j]:
            flush_positions.append(k)
            num_purges += 1

    return BoothResult(
        booth_id=inst.booth_id,
        n_jobs=n,
        branch=sol.branch,
        total_setup_cost=total_cost,
        num_purges=num_purges,
        order_indices=order,
        flush_positions=flush_positions,
    )


# ---------------------------------------------------------------------------
# Post-processing search
# ---------------------------------------------------------------------------


def run_sdsu_for_booths(
    df1_jobs: pd.DataFrame,
    df2_jobs: pd.DataFrame,
    schema: SchemaConfig,
    paint_model: PaintModel,
    adj_model: AdjacencyModel,
    sdsu_cfg: SDSUConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, float, BoothResult, BoothResult]:
    """Run SDSU for PB1 and PB2 and return ordered DataFrames and cost."""
    inst1 = build_booth_instance(1, df1_jobs, schema)
    inst2 = build_booth_instance(2, df2_jobs, schema)
    mats1 = build_cost_matrices_for_booth(inst1, paint_model, adj_model, sdsu_cfg)
    mats2 = build_cost_matrices_for_booth(inst2, paint_model, adj_model, sdsu_cfg)
    res1 = solve_sdsu_for_booth(inst1, mats1, sdsu_cfg)
    res2 = solve_sdsu_for_booth(inst2, mats2, sdsu_cfg)

    if res1.n_jobs > 0:
        ordered1 = inst1.df.iloc[res1.order_indices].copy()
    else:
        ordered1 = inst1.df.iloc[0:0].copy()
    if res2.n_jobs > 0:
        ordered2 = inst2.df.iloc[res2.order_indices].copy()
    else:
        ordered2 = inst2.df.iloc[0:0].copy()

    # update booth_seq_index and booth_id
    ordered1["booth_id"] = 1
    ordered2["booth_id"] = 2
    ordered1["booth_seq_index"] = list(range(1, len(ordered1) + 1))
    ordered2["booth_seq_index"] = list(range(1, len(ordered2) + 1))

    total_cost = res1.total_setup_cost + res2.total_setup_cost
    return ordered1, ordered2, total_cost, res1, res2


def postprocess_exchange(
    df1_init: pd.DataFrame,
    df2_init: pd.DataFrame,
    schema: SchemaConfig,
    paint_model: PaintModel,
    adj_model: AdjacencyModel,
    sdsu_cfg: SDSUConfig,
    cfg: Step04PostConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, float, List[MoveRecord]]:
    """
    Perform PB1↔PB2 block exchange local search. Returns final per-booth sequences,
    final f1 cost, and list of accepted moves.
    """
    # Start by running SDSU on the initial job sets; we do not trust upstream order.
    df1_sdsu, df2_sdsu, f1_total, res1, res2 = run_sdsu_for_booths(
        df1_init, df2_init, schema, paint_model, adj_model, sdsu_cfg
    )

    moves: List[MoveRecord] = []
    move_id = 0

    for _pass in range(cfg.max_passes):
        improved = False

        n1 = len(df1_sdsu)
        n2 = len(df2_sdsu)
        if n1 == 0 or n2 == 0:
            break  # nothing to exchange

        # Deterministic neighbourhood scan
        best_delta = 0.0
        best_candidate = None

        for L in range(1, cfg.L_max + 1):
            if L > n1 or L > n2:
                continue
            for i in range(0, n1 - L + 1):
                for j in range(0, n2 - L + 1):
                    # form blocks
                    B1 = df1_sdsu.iloc[i : i + L]
                    B2 = df2_sdsu.iloc[j : j + L]

                    # job sets after swap
                    # remove B1 from PB1, add B2
                    J1_prime = pd.concat(
                        [
                            df1_sdsu.iloc[:i],
                            df1_sdsu.iloc[i + L :],
                            B2,
                        ],
                        ignore_index=True,
                    )
                    # remove B2 from PB2, add B1
                    J2_prime = pd.concat(
                        [
                            df2_sdsu.iloc[:j],
                            df2_sdsu.iloc[j + L :],
                            B1,
                        ],
                        ignore_index=True,
                    )

                    # ensure booth_id assignments
                    J1_prime = J1_prime.copy()
                    J2_prime = J2_prime.copy()
                    J1_prime[schema.booth_id_col] = 1
                    J2_prime[schema.booth_id_col] = 2

                    # run SDSU for candidate
                    try:
                        cand1, cand2, f1_new, r1_new, r2_new = run_sdsu_for_booths(
                            J1_prime, J2_prime, schema, paint_model, adj_model, sdsu_cfg
                        )
                    except RuntimeError:
                        # infeasible candidate under C1/C2; skip
                        continue

                    delta = f1_total - f1_new  # positive if improvement
                    if delta <= 1e-9:
                        continue

                    if cfg.search_policy == "first_improvement":
                        # accept immediately
                        move_id += 1
                        mr = MoveRecord(
                            move_id=move_id,
                            block_length=L,
                            pb1_start=i,
                            pb2_start=j,
                            f1_before=f1_total,
                            f1_after=f1_new,
                            delta_f1=delta,
                            branch_pb1=r1_new.branch,
                            branch_pb2=r2_new.branch,
                        )
                        moves.append(mr)
                        df1_sdsu = cand1
                        df2_sdsu = cand2
                        f1_total = f1_new
                        improved = True
                        break  # break j
                    else:
                        # best_improvement: track best candidate
                        if delta > best_delta:
                            best_delta = delta
                            best_candidate = (L, i, j, cand1, cand2, f1_new, r1_new, r2_new)
                if cfg.search_policy == "first_improvement" and improved:
                    break  # break i
            if cfg.search_policy == "first_improvement" and improved:
                break  # break L

        if cfg.search_policy == "best_improvement" and best_candidate is not None:
            L, i, j, cand1, cand2, f1_new, r1_new, r2_new = best_candidate
            move_id += 1
            mr = MoveRecord(
                move_id=move_id,
                block_length=L,
                pb1_start=i,
                pb2_start=j,
                f1_before=f1_total,
                f1_after=f1_new,
                delta_f1=f1_total - f1_new,
                branch_pb1=r1_new.branch,
                branch_pb2=r2_new.branch,
            )
            moves.append(mr)
            df1_sdsu = cand1
            df2_sdsu = cand2
            f1_total = f1_new
            improved = True

        if not improved:
            break

    return df1_sdsu, df2_sdsu, f1_total, moves


# ---------------------------------------------------------------------------
# Manifest and report
# ---------------------------------------------------------------------------


def write_manifest(
    paths: Step04PostPaths,
    post_cfg: Step04PostConfig,
    sdsu_cfg: SDSUConfig,
    f1_initial: float,
    f1_final: float,
    moves: List[MoveRecord],
    input_files: List[Path],
    output_files: List[Path],
    start_time: float,
    end_time: float,
) -> None:
    ensure_parent_dir(paths.manifest_path)
    manifest = {
        "step": "step04_sdsu_postprocessing",
        "version": "1.0",
        "config_files": {
            str(paths.global_config_path): sha256_of_file(paths.global_config_path),
            str(paths.steps_config_path): sha256_of_file(paths.steps_config_path),
        },
        "input_files": {str(p): sha256_of_file(p) for p in input_files if p.exists()},
        "output_files": {str(p): sha256_of_file(p) for p in output_files if p.exists()},
        "settings_effective": {
            "postprocessing": dataclasses.asdict(post_cfg),
            "sdsu": dataclasses.asdict(sdsu_cfg),
        },
        "metrics": {
            "f1_initial": f1_initial,
            "f1_final": f1_final,
            "delta_f1": f1_initial - f1_final,
            "n_moves_accepted": len(moves),
        },
        "runtime_seconds": end_time - start_time,
    }
    with paths.manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def write_report(
    paths: Step04PostPaths,
    post_cfg: Step04PostConfig,
    sdsu_cfg: SDSUConfig,
    f1_initial: float,
    f1_final: float,
    moves: List[MoveRecord],
) -> None:
    ensure_parent_dir(paths.report_path)
    lines: List[str] = []

    lines.append("# Step-04 — SDSU Post-Processing (PB1↔PB2 Exchange)")
    lines.append("")
    lines.append("## Purpose / What / Why")
    lines.append("")
    lines.append("- **Purpose.** Apply local PB1↔PB2 block exchanges to the per-booth SDSU schedules")
    lines.append("  to reduce total paint-domain setup+purge cost f1 under the same C1/C2/C3 semantics.")
    lines.append("- **What.** Deterministic neighbourhood search over contiguous blocks in SDSU order;")
    lines.append("  for each candidate swap, re-run SDSU per booth and accept only strictly improving moves.")
    lines.append("- **Why.** Strengthen the Toyota+SDSU heuristic baseline by adding an f1-oriented")
    lines.append("  local improvement layer, without changing feasibility rules.")
    lines.append("")
    lines.append("## Effective Settings")
    lines.append("")
    settings = {
        "postprocessing": dataclasses.asdict(post_cfg),
        "sdsu": dataclasses.asdict(sdsu_cfg),
    }
    lines.append("```json")
    lines.append(json.dumps(settings, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append(f"- f1_initial (PB1+PB2): {f1_initial:.3f}")
    lines.append(f"- f1_final (PB1+PB2): {f1_final:.3f}")
    lines.append(f"- delta_f1 = f1_initial - f1_final: {f1_initial - f1_final:.3f}")
    lines.append(f"- Moves accepted: {len(moves)}")
    lines.append("")
    lines.append("## Accepted Moves (summary)")
    lines.append("")
    if not moves:
        lines.append("- No improving exchanges found; schedules left unchanged.")
    else:
        for mr in moves:
            lines.append(
                f"- Move {mr.move_id}: L={mr.block_length}, "
                f"PB1_start={mr.pb1_start}, PB2_start={mr.pb2_start}, "
                f"delta_f1={mr.delta_f1:.3f}, branches=({mr.branch_pb1},{mr.branch_pb2})"
            )
    lines.append("")

    with paths.report_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def run_step04_post() -> None:
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
    post_cfg = build_step04_post_config(global_cfg, steps_cfg)
    sdsu_cfg = build_sdsu_config(global_cfg, steps_cfg)

    # Load dataset-level models
    paint_model = load_paint_model(paths, global_cfg)
    adj_model = load_adjacency_model_inbooth(paths, global_cfg)

    # Load inputs
    if not paths.booth1_input.exists():
        raise FileNotFoundError(f"Step-04 post-processing: booth1 input not found: {paths.booth1_input}")
    if not paths.booth2_input.exists():
        raise FileNotFoundError(f"Step-04 post-processing: booth2 input not found: {paths.booth2_input}")

    df1_init = pd.read_csv(paths.booth1_input)
    df2_init = pd.read_csv(paths.booth2_input)

    # Resolve schema
    schema = resolve_schema(global_cfg, df1_init)

    # Ensure booth_id is correct in initial job sets
    if schema.booth_id_col not in df1_init.columns or schema.booth_id_col not in df2_init.columns:
        raise KeyError("booth_id column must be present in both booth inputs for Step-04 post-processing")

    df1_init = df1_init.copy()
    df2_init = df2_init.copy()
    df1_init[schema.booth_id_col] = 1
    df2_init[schema.booth_id_col] = 2

    # Initial SDSU cost and sequences
    df1_sdsu, df2_sdsu, f1_initial, _, _ = run_sdsu_for_booths(
        df1_init, df2_init, schema, paint_model, adj_model, sdsu_cfg
    )

    # Local search
    df1_final, df2_final, f1_final, moves = postprocess_exchange(
        df1_sdsu, df2_sdsu, schema, paint_model, adj_model, sdsu_cfg, post_cfg
    )

    # Write outputs
    ensure_parent_dir(paths.booth1_output)
    ensure_parent_dir(paths.booth2_output)
    df1_final.to_csv(paths.booth1_output, index=False)
    df2_final.to_csv(paths.booth2_output, index=False)

    # Moves log
    ensure_parent_dir(paths.moves_log_path)
    moves_payload = [dataclasses.asdict(mr) for mr in moves]
    with paths.moves_log_path.open("w", encoding="utf-8") as f:
        json.dump(moves_payload, f, indent=2)

    end_time = time.time()

    input_files = [
        paths.global_config_path,
        paths.steps_config_path,
        paths.booth1_input,
        paths.booth2_input,
        paths.s_matrix_path,
        paths.topcoat_codebook_path,
        paths.topcoat_rules_path,
        paths.adjacency_rules_path,
    ]
    output_files = [
        paths.booth1_output,
        paths.booth2_output,
        paths.moves_log_path,
        paths.manifest_path,
        paths.report_path,
    ]

    write_manifest(paths, post_cfg, sdsu_cfg, f1_initial, f1_final, moves, input_files, output_files, start_time, end_time)
    write_report(paths, post_cfg, sdsu_cfg, f1_initial, f1_final, moves)


if __name__ == "__main__":
    run_step04_post()
