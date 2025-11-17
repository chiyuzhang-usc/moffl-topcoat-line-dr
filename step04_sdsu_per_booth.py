#!/usr/bin/env python3
"""
Step-04 — SDSU per Booth (Non-Cyclic Open-Path SDST in Paint Domain)

This script implements the per-booth SDSU step as specified in:
- NOTE_MASTER.md
- MOFFS_Protocol.md (STEP-04)
- STEP04_NOTES.md
- SDSU_Method.md

It:
- reads Step-03 booth JOB files,
- builds per-booth SDST cost matrices from the paint dataset,
- enforces paint and in-booth adjacency forbids,
- solves an open-path SDSU problem per booth (exact DP for small n, heuristic otherwise),
- emits refined per-booth JOB CSVs in SDSU order,
- records inferred FLUSH positions (purge arcs) in a JSON file,
- writes a manifest (with SHA-256 hashes) and a Markdown report.

The implementation is deterministic and strictly data-driven for feasibility:
all C1/C2/C3 constraints come only from the dataset and global_config.toml.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable

import pandas as pd
import tomllib


# ---------------------------------------------------------------------------
# Small utilities
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
# Data mapping helpers (from global_config + dataset)
# ---------------------------------------------------------------------------


@dataclass
class SchemaConfig:
    part_id_col: str
    topcoat_code_col: str


@dataclass
class SDSUConfig:
    dp_threshold_jobs: int
    max_2opt_passes: int
    allow_purge: bool
    purge_cost_weight: float
    flush_units_ratio: float


@dataclass
class Step04Paths:
    output_dir: Path
    dataset_dir: Path
    booth1_input: Path
    booth2_input: Path
    booth1_output: Path
    booth2_output: Path
    manifest_path: Path
    report_path: Path
    flush_positions_path: Path
    s_matrix_path: Path
    topcoat_codebook_path: Path
    topcoat_rules_path: Path
    adjacency_rules_path: Path
    global_config_path: Path
    steps_config_path: Path


@dataclass
class BoothResult:
    booth_id: int
    n_jobs: int
    branch: str  # "dp" or "heuristic" or "empty"
    total_setup_cost: float
    num_purges: int
    order_indices: List[int]
    flush_positions: List[int]  # positions between consecutive jobs where FLUSH is used


def resolve_schema(global_cfg: dict, df: pd.DataFrame) -> SchemaConfig:
    ds = global_cfg.get("data_schema", {})
    demands_schema = ds.get("demands", {})
    topcoat_syn = demands_schema.get("topcoat_code_synonyms", [])
    if not topcoat_syn:
        # fall back to topcoat_codebook if needed
        tcb_schema = ds.get("topcoat_codebook", {})
        topcoat_syn = tcb_schema.get("topcoat_code_synonyms", [])
    part_syn = demands_schema.get("part_id_synonyms", [])
    if not part_syn:
        parts_schema = ds.get("parts", {})
        part_syn = parts_schema.get("part_id_synonyms", [])

    topcoat_col = pick_first_existing(df.columns, topcoat_syn + ["topcoat_code", "color_code"])
    part_id_col = pick_first_existing(df.columns, part_syn + ["part_id"])

    if topcoat_col is None:
        raise KeyError(f"Could not resolve topcoat code column in SDSU input; candidates={topcoat_syn}")
    if part_id_col is None:
        raise KeyError(f"Could not resolve part_id column in SDSU input; candidates={part_syn}")

    return SchemaConfig(part_id_col=part_id_col, topcoat_code_col=topcoat_col)


def resolve_paths(global_cfg: dict, steps_cfg: dict) -> Step04Paths:
    paths_cfg = global_cfg["paths"]
    data_files = global_cfg["data_files"]
    step04_cfg = steps_cfg["step04"]

    dataset_dir = Path(paths_cfg["dataset_dir"])
    output_dir = Path(paths_cfg["output_dir"])

    return Step04Paths(
        output_dir=output_dir,
        dataset_dir=dataset_dir,
        booth1_input=output_dir / step04_cfg["booth1_input_relpath"],
        booth2_input=output_dir / step04_cfg["booth2_input_relpath"],
        booth1_output=output_dir / step04_cfg["booth1_output_relpath"],
        booth2_output=output_dir / step04_cfg["booth2_output_relpath"],
        manifest_path=output_dir / step04_cfg["manifest_relpath"],
        report_path=output_dir / step04_cfg["report_relpath"],
        flush_positions_path=output_dir / step04_cfg["flush_positions_relpath"],
        s_matrix_path=dataset_dir / data_files["s_matrix_transition"],
        topcoat_codebook_path=dataset_dir / data_files["topcoat_codebook"],
        topcoat_rules_path=dataset_dir / data_files["topcoat_rules"],
        adjacency_rules_path=dataset_dir / data_files["adjacency_rules"],
        global_config_path=Path("global_config.toml"),
        steps_config_path=Path("steps_config.toml"),
    )


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


# ---------------------------------------------------------------------------
# Dataset-derived paint and adjacency structures
# ---------------------------------------------------------------------------


@dataclass
class PaintModel:
    # cost[from_color_code][to_color_code] = numeric cost or math.inf
    setup_cost: Dict[str, Dict[str, float]]
    # direct forbids at topcoat_no level from topcoat_rules
    forbid_pairs_no: set
    # code -> topcoat_no
    topcoat_code_to_no: Dict[str, int]


@dataclass
class AdjacencyModel:
    # forbidden part-pairs for in-booth adjacency: (from_part_id, to_part_id)
    forbid_pairs: set


def load_paint_model(paths: Step04Paths, global_cfg: dict) -> PaintModel:
    # topcoat codebook: code -> no
    tcb = pd.read_csv(paths.topcoat_codebook_path)
    if "topcoat_code" not in tcb.columns or "topcoat_no" not in tcb.columns:
        raise KeyError("topcoat_codebook.csv must contain 'topcoat_code' and 'topcoat_no' columns")
    code_to_no = {str(row["topcoat_code"]): int(row["topcoat_no"]) for _, row in tcb.iterrows()}

    # transition matrix: wide format with 'from' and one column per topcoat_code
    s_mat = pd.read_csv(paths.s_matrix_path)
    if "from" not in s_mat.columns:
        raise KeyError("s_matrix_transition.csv must contain 'from' column")

    # Normalize: treat 'inf' or 'INF' as math.inf, everything else as float
    setup_cost: Dict[str, Dict[str, float]] = {}
    color_cols = [c for c in s_mat.columns if c != "from"]
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

    # topcoat_rules: encode forbids on numeric codes
    trules = pd.read_csv(paths.topcoat_rules_path)
    forbid_pairs_no = set()
    if "relation" in trules.columns and "from_topcoat_no" in trules.columns and "to_topcoat_no" in trules.columns:
        for _, r in trules.iterrows():
            rel = str(r["relation"]).lower()
            if "forbid" in rel:
                try:
                    i_no = int(r["from_topcoat_no"])
                    j_no = int(r["to_topcoat_no"])
                    forbid_pairs_no.add((i_no, j_no))
                except Exception:
                    continue

    return PaintModel(setup_cost=setup_cost, forbid_pairs_no=forbid_pairs_no, topcoat_code_to_no=code_to_no)


def load_adjacency_model(paths: Step04Paths, global_cfg: dict) -> AdjacencyModel:
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
        # If we can't resolve, fall back to no adjacency forbids for SDSU.
        return AdjacencyModel(forbid_pairs=set())

    forbid_pairs = set()
    for _, r in df.iterrows():
        scope = str(r[scope_col]).lower()
        rel = str(r[rel_col]).lower()
        if "forbid" not in rel:
            continue
        # In-booth C2 uses forbids with scope 'in_booth' or 'global' (not 'across_skids' only)
        if scope not in ("in_booth", "global"):
            continue
        from_part = str(r[from_col])
        to_part = str(r[to_col])
        forbid_pairs.add((from_part, to_part))

    return AdjacencyModel(forbid_pairs=forbid_pairs)


# ---------------------------------------------------------------------------
# SDSU cost matrix construction per booth
# ---------------------------------------------------------------------------


@dataclass
class BoothInstance:
    booth_id: int
    df: pd.DataFrame
    schema: SchemaConfig
    # job index -> color code, part id
    colors: List[str]
    parts: List[str]


@dataclass
class CostMatrices:
    # use_cost[i][j] = finite cost or math.inf
    use_cost: List[List[float]]
    # flush_used[i][j] = True if the minimal-cost arc uses a FLUSH + start path
    flush_used: List[List[bool]]


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
    # Pre-allocate matrices
    use_cost = [[math.inf for _ in range(n)] for _ in range(n)]
    flush_used = [[False for _ in range(n)] for _ in range(n)]

    flush_cost_units = sdsu_cfg.flush_units_ratio * sdsu_cfg.purge_cost_weight

    # s_0(c) = 0 for all colors (path start cost uniform)
    # This is consistent with non-cyclic SDSU where only relative inter-color costs matter.
    s0 = {color: 0.0 for color in set(inst.colors)}

    # convenience
    setup_cost = paint.setup_cost
    code_to_no = paint.topcoat_code_to_no
    forbid_pairs_no = paint.forbid_pairs_no
    adj_forbids = adjacency.forbid_pairs
    print(f"Booth {inst.booth_id}: Building cost matrices for n={n} jobs")

    for i in range(n):
        ci = inst.colors[i]
        pi = inst.parts[i]
        row_i = setup_cost.get(ci)
        for j in range(n):
            if i == j:
                continue
            cj = inst.colors[j]
            pj = inst.parts[j]

            # adjacency forbids (in-booth/global)
            if (pi, pj) in adj_forbids:
                continue  # infeasible arc

            # paint direct cost
            direct = math.inf
            if row_i is not None:
                if cj in row_i:
                    direct = row_i[cj]

            # apply topcoat_rules forbids
            try:
                ni = code_to_no[ci]
                nj = code_to_no[cj]
                if (ni, nj) in forbid_pairs_no:
                    direct = math.inf
            except KeyError:
                # if code not known, we rely solely on s_matrix_transition
                pass

            direct_feasible = direct < math.inf

            # purge / FLUSH path
            purge_cost = math.inf
            if sdsu_cfg.allow_purge:
                # If we allow purge, we treat FLUSH + start-to-cj as a feasible alternative
                # whenever we have a defined start cost s0(cj). Here s0(cj) is uniform 0.
                if cj in s0:
                    purge_cost = flush_cost_units + s0[cj]

            print(f"Building arc {i} -> {j}: direct={direct}, purge={purge_cost}; Progress: {i*n + j + 1}/{n*n} [{(i*n + j + 1)/(n*n)*100:.2f}%]", end="\r")
            if direct_feasible:
                if purge_cost < direct:
                    use_cost[i][j] = purge_cost
                    flush_used[i][j] = True
                else:
                    use_cost[i][j] = direct
                    flush_used[i][j] = False
            else:
                # no direct, but purge may be possible
                if purge_cost < math.inf:
                    use_cost[i][j] = purge_cost
                    flush_used[i][j] = True
                else:
                    # arc remains infeasible
                    continue
    print(f"Booth {inst.booth_id}: Cost matrices built with {n} jobs")
    return CostMatrices(use_cost=use_cost, flush_used=flush_used)


# ---------------------------------------------------------------------------
# SDSU solver: DP (exact) and greedy + 2-opt (heuristic)
# ---------------------------------------------------------------------------


@dataclass
class SDSUSolution:
    order: List[int]  # permutation of indices 0..n-1
    cost: float
    branch: str  # "dp" or "heuristic" or "empty"


def sdsu_open_path_dp(cost: List[List[float]]) -> SDSUSolution:
    """Exact open-path SDSU using subset DP (Held–Karp style).

    DP[S][j] = minimal cost of a path that starts at some job in S, ends at j, and visits exactly S.
    We take base DP[{j}][j] = s0(j) = 0 and cost increments from arcs only.
    """
    n = len(cost)
    if n == 0:
        return SDSUSolution(order=[], cost=0.0, branch="dp")
    if n == 1:
        return SDSUSolution(order=[0], cost=0.0, branch="dp")

    # DP and predecessor tables
    size = 1 << n
    INF = math.inf
    dp = [[INF] * n for _ in range(size)]
    prev = [[-1] * n for _ in range(size)]

    # base cases: singletons
    for j in range(n):
        mask = 1 << j
        dp[mask][j] = 0.0  # s0(j) = 0
        prev[mask][j] = -1

    # transitions
    for mask in range(1, size):
        # skip singleton masks; already initialised
        if (mask & (mask - 1)) == 0:
            continue
        for j in range(n):
            if not (mask & (1 << j)):
                continue
            prev_mask = mask ^ (1 << j)
            # iterate over possible predecessors i
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
    # pick best end
    best_end = -1
    best_cost = INF
    for j in range(n):
        if dp[full_mask][j] < best_cost:
            best_cost = dp[full_mask][j]
            best_end = j

    if best_cost >= INF or best_end == -1:
        raise RuntimeError("SDSU DP: no feasible open path (all arcs infinite)")

    # reconstruct path
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
        raise RuntimeError(f"SDSU DP: reconstructed path has length {len(order)} != {n}")

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

    # Greedy construction: start at node 0, then repeatedly choose cheapest feasible successor
    start = 0
    order = [start]
    used = [False] * n
    used[start] = True

    for _ in range(1, n):
        last = order[-1]
        best_j = -1
        best_c = INF
        for j in range(n):
            print(f"Greedy: Considering arc {last} -> {j}; Progress: {_+1}/{n-1}[{(_)/(n-1)*100:.2f}%]", end="\r")
            if used[j]:
                continue
            c = cost[last][j]
            if c < best_c:
                best_c = c
                best_j = j
        if best_j == -1 or best_c >= INF:
            raise RuntimeError("SDSU greedy: no feasible successor found; instance infeasible")
        order.append(best_j)
        used[best_j] = True

    best_cost = path_cost(cost, order)
    if best_cost >= INF:
        raise RuntimeError("SDSU greedy: constructed path has infinite cost; instance infeasible")

    # 2-opt improvement
    improved = True
    passes = 0
    print(f"SDSU heuristic: Starting 2-opt improvement passes (max {max_passes})")
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for a in range(0, n - 2):
            for b in range(a + 1, n - 1):
                # candidate path: reverse segment [a+1..b]
                new_order = order[: a + 1] + list(reversed(order[a + 1 : b + 1])) + order[b + 1 :]
                new_cost = path_cost(cost, new_order)
                print(f"2-opt pass {passes}: Considering swap ({a},{b}) -> new cost {new_cost}", end="\r")
                if new_cost < best_cost - 1e-9:  # strict improvement
                    order = new_order
                    best_cost = new_cost
                    improved = True
                    break
            if improved:
                break
    print(f"SDSU heuristic: 2-opt completed after {passes} passes; best cost = {best_cost}")
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

    # choose branch based on n and dp_threshold
    if n <= sdsu_cfg.dp_threshold_jobs:
        print(f"Booth {inst.booth_id}: Solving SDSU with exact DP (n={n} <= threshold={sdsu_cfg.dp_threshold_jobs})")
        sol = sdsu_open_path_dp(use_cost)
    else:
        print(f"Booth {inst.booth_id}: Solving SDSU with greedy + 2-opt heuristic (n={n} > threshold={sdsu_cfg.dp_threshold_jobs})")
        sol = sdsu_open_path_greedy_2opt(use_cost, max_passes=sdsu_cfg.max_2opt_passes)

    order = sol.order
    total_cost = sol.cost

    # derive purge positions from flush_used
    flush_positions: List[int] = []
    num_purges = 0
    for k in range(len(order) - 1):
        i = order[k]
        j = order[k + 1]
        if flush_used[i][j]:
            flush_positions.append(k)  # FLUSH between order[k] and order[k+1]
            num_purges += 1
            print(f"Booth {inst.booth_id}: FLUSH between jobs at positions {k} and {k+1} (jobs {i} -> {j})")
    
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
# Manifest and report
# ---------------------------------------------------------------------------


def write_manifest(
    paths: Step04Paths,
    sdsu_cfg: SDSUConfig,
    booth_results: List[BoothResult],
    input_files: List[Path],
    output_files: List[Path],
    start_time: float,
    end_time: float,
) -> None:
    ensure_parent_dir(paths.manifest_path)
    manifest = {
        "step": "step04_sdsu_per_booth",
        "version": "1.0",
        "config_files": {
            str(paths.global_config_path): sha256_of_file(paths.global_config_path),
            str(paths.steps_config_path): sha256_of_file(paths.steps_config_path),
        },
        "input_files": {str(p): sha256_of_file(p) for p in input_files if p.exists()},
        "output_files": {str(p): sha256_of_file(p) for p in output_files if p.exists()},
        "settings_effective": dataclasses.asdict(sdsu_cfg),
        "metrics": {
            "booths": [
                dataclasses.asdict(br) for br in booth_results
            ],
            "total_setup_cost": sum(br.total_setup_cost for br in booth_results),
            "total_purges": sum(br.num_purges for br in booth_results),
        },
        "runtime_seconds": end_time - start_time,
    }
    with paths.manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def write_report(paths: Step04Paths, sdsu_cfg: SDSUConfig, booth_results: List[BoothResult]) -> None:
    ensure_parent_dir(paths.report_path)
    lines: List[str] = []

    lines.append("# Step-04 — SDSU per Booth")
    lines.append("")
    lines.append("## Purpose / What / Why")
    lines.append("")
    lines.append("- **Purpose.** For each paint booth, compute a paint-feasible, adjacency-respecting open path order that")
    lines.append("  minimizes sequence-dependent setup cost plus purge cost, under the SDSU model.")
    lines.append("- **What.** Build per-booth SDST cost matrices from `s_matrix_transition.csv` + `topcoat_rules.csv`,")
    lines.append("  enforce in-booth adjacency forbids from `adjacency_rules.csv`, and solve SDSU with exact subset DP for")
    lines.append("  small booths and a greedy + 2-opt heuristic for larger booths.")
    lines.append("- **Why.** This step provides per-booth paint-optimal orders that feed the 2→1 merge (Step-05) and")
    lines.append("  anchor the f1 (setup + purge) dimension for comparison against MILP and NSGA-II schedules.")
    lines.append("")
    lines.append("## Effective Settings")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(dataclasses.asdict(sdsu_cfg), indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Booth Metrics")
    lines.append("")
    for br in booth_results:
        lines.append(f"### Booth {br.booth_id}")
        lines.append("")
        lines.append(f"- Jobs: {br.n_jobs}")
        lines.append(f"- Branch: `{br.branch}`")
        lines.append(f"- Total setup cost (paint units): {br.total_setup_cost:.3f}")
        lines.append(f"- Number of purge events (FLUSH): {br.num_purges}")
        lines.append(f"- Flush positions (between jobs, 0-based): {br.flush_positions}")
        lines.append("")

    with paths.report_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Main step driver
# ---------------------------------------------------------------------------


def run_step04() -> None:
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
    sdsu_cfg = build_sdsu_config(global_cfg, steps_cfg)

    # Load dataset-level models
    paint_model = load_paint_model(paths, global_cfg)
    adjacency_model = load_adjacency_model(paths, global_cfg)

    # Read booth inputs
    if not paths.booth1_input.exists():
        raise FileNotFoundError(f"Booth-1 input not found: {paths.booth1_input}")
    if not paths.booth2_input.exists():
        raise FileNotFoundError(f"Booth-2 input not found: {paths.booth2_input}")

    booth1_df = pd.read_csv(paths.booth1_input)
    booth2_df = pd.read_csv(paths.booth2_input)

    # Resolve schema on booth1 (assume same schema for booth2)
    schema = resolve_schema(global_cfg, booth1_df)

    booths: List[BoothInstance] = [
        build_booth_instance(1, booth1_df, schema),
        build_booth_instance(2, booth2_df, schema),
    ]

    booth_results: List[BoothResult] = []

    # per-booth SDSU
    for booth in booths:
        matrices = build_cost_matrices_for_booth(booth, paint_model, adjacency_model, sdsu_cfg)
        result = solve_sdsu_for_booth(booth, matrices, sdsu_cfg)
        booth_results.append(result)

        # apply order to DataFrame and write output
        if result.n_jobs == 0:
            ordered_df = booth.df.iloc[0:0].copy()
        else:
            ordered_df = booth.df.iloc[result.order_indices].copy()
        # update booth_seq_index in SDSU order (1-based)
        ordered_df["booth_seq_index"] = list(range(1, len(ordered_df) + 1))
        if booth.booth_id == 1:
            ensure_parent_dir(paths.booth1_output)
            ordered_df.to_csv(paths.booth1_output, index=False)
        else:
            ensure_parent_dir(paths.booth2_output)
            ordered_df.to_csv(paths.booth2_output, index=False)

    # flush positions JSON
    flush_payload = {
        f"booth{br.booth_id}": {
            "flush_positions": br.flush_positions,
            "num_purges": br.num_purges,
            "n_jobs": br.n_jobs,
        }
        for br in booth_results
    }
    ensure_parent_dir(paths.flush_positions_path)
    with paths.flush_positions_path.open("w", encoding="utf-8") as f:
        json.dump(flush_payload, f, indent=2)

    end_time = time.time()

    # manifest and report
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
        paths.flush_positions_path,
        paths.manifest_path,
        paths.report_path,
    ]

    write_manifest(paths, sdsu_cfg, booth_results, input_files, output_files, start_time, end_time)
    write_report(paths, sdsu_cfg, booth_results)


if __name__ == "__main__":
    run_step04()
