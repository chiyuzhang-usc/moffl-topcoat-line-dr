#!/usr/bin/env python3
"""
STEP-07 — NSGA-II search for MOFFS (Multi-Objective Flexible Flow Shop).

This step:
    - Uses the same feasibility semantics as Steps 02–05:
        * C1/C3 (paint) from topcoat_rules + s_matrix_transition
        * C2 in-booth and global from adjacency_rules
        * Merge preserves in-booth order; no tokens in outputs
    - Optimizes three static objectives:
        f1: total in-booth SDST + FLUSH cost (no EMPTY cost)
        f2: windowed color-mix deviation on merged JOB-only stream
        f3: split-based booth balance surrogate on σ (assignment string)
    - Runs NSGA-II for each configured seed, with optional Toyota/SDSU seeding,
      then consolidates a global Pareto front and writes:
        * output/step07_pareto.csv
        * manifests/step07_manifest.json
        * reports/step07_report.md
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from utilities.config_utils import load_global_config, load_steps_config, resolve_paths
from utilities.io_utils import ensure_parent_dir, sha256_of_file, write_json, write_text


LOG_PREFIX = "[STEP07]"
STEP_ID = "step07_nsga2"


# ---------------------------------------------------------------------------
# Logging / small utilities
# ---------------------------------------------------------------------------


def log_progress(message: str) -> None:
    """
    One-line log with a step prefix.
    """
    print(f"{LOG_PREFIX} {message}")


def update_progress_line(message: str, final: bool = False) -> None:
    """
    Inline progress bar style logging.
    """
    end_char = "\n" if final else "\r"
    sys.stdout.write(f"{LOG_PREFIX} {message}{end_char}")
    sys.stdout.flush()


def detect_column(
    df: pd.DataFrame,
    candidates: Sequence[str],
    required: bool = True,
) -> Optional[str]:
    """
    Find the first matching column in df for any of the 'candidates'
    (case-insensitive). Raise if required and not found.
    """
    lookup = {c.lower(): c for c in df.columns}
    for cand in candidates:
        col = lookup.get(str(cand).lower())
        if col:
            return col
    if required:
        raise KeyError(f"None of {candidates} found in {list(df.columns)}")
    return None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class JobRecord:
    job_id: str
    feeder_index: int
    part_id: str
    topcoat: str
    base_booth: int  # booth from Step-03 assignment (Toyota)


@dataclass
class Individual:
    sigma: List[int]          # assignment string (in feeder order)
    pi1: List[int]            # permutation of indices for PB1
    pi2: List[int]            # permutation of indices for PB2
    f1: float = math.inf
    f2: float = math.inf
    f3: float = math.inf
    feasible: bool = False
    rank: int = 0
    crowding_distance: float = 0.0
    seed_id: int = -1
    origin: str = "random"    # "random" | "toyota" | "sdsu"


@dataclass
class DecoderResult:
    feasible: bool
    booth1_seq: List[int]
    booth2_seq: List[int]
    global_seq: List[Tuple[str, int]]  # ("PB1"/"PB2", job_index)
    adjacency_repairs: int
    paint_repairs: int


@dataclass
class Step07Data:
    job_records: List[JobRecord]
    job_index_by_id: Dict[str, int]
    baseline_sigma: List[int]
    target_mix: Dict[str, float]
    forb_in_booth: set
    forb_global: set
    window_mode: str
    window_jobs: int
    window_ratio: float
    tie_break_policy: str
    repair_L_max: int
    rho_scale: float
    sdsu_orders: Dict[str, List[int]]  # "PB1"/"PB2" -> list of job indices


@dataclass
class Step07Paths:
    dataset_dir: Path
    output_dir: Path
    pareto_path: Path
    manifest_path: Path
    report_path: Path
    assignment_path: Path
    booth1_path: Path
    booth2_path: Path
    sdsu_booth1_path: Path
    sdsu_booth2_path: Path
    adjacency_rules_path: Path
    s_matrix_path: Path
    topcoat_codebook_path: Path
    topcoat_rules_path: Path
    cycle_quota_path: Path
    global_config_path: Path
    steps_config_path: Path


@dataclass
class Step07Settings:
    tie_break_policy: str
    repair_L_max: int
    rho_scale: float
    window_mode: str
    window_jobs: int
    window_ratio: float
    allow_purge: bool
    flush_cost_units: float
    population_size: int
    generations: int
    crossover_rate: float
    mutation_rate: float
    use_seeding: bool
    use_sdsu_kernel: bool
    seeds: List[int]
    max_eval_per_seed: int


# ---------------------------------------------------------------------------
# Paint cost kernel
# ---------------------------------------------------------------------------


class PaintFeasibilityError(RuntimeError):
    """
    Raised when no paint transition is available (even with purge).
    """


@dataclass
class PaintCostKernel:
    s_matrix: Dict[Tuple[str, str], float]
    forbidden_pairs: set
    allow_purge: bool
    flush_cost_units: float

    def transition_cost(self, prev_color: Optional[str], next_color: str) -> float:
        """
        Compute cost of going prev_color -> next_color.

        Rules:
            - If prev_color is None, cost is 0 (start-up).
            - If (prev,next) in forbidden_pairs => infeasible.
            - SDST from s_matrix_transition is authoritative.
            - If direct arc is INF and purge allowed, cost = flush_cost_units.
            - If both direct and purge are INF => infeasible.
        """
        if prev_color is None:
            return 0.0

        if (prev_color, next_color) in self.forbidden_pairs:
            raise PaintFeasibilityError(f"Paint forbid {prev_color}->{next_color}")

        direct = self.s_matrix.get((prev_color, next_color), math.inf)
        purge_cost = self.flush_cost_units if self.allow_purge else math.inf

        if math.isinf(direct):
            if math.isinf(purge_cost):
                raise PaintFeasibilityError(
                    f"Transition {prev_color}->{next_color} infeasible"
                )
            return purge_cost

        if math.isinf(purge_cost):
            return direct

        return min(direct, purge_cost)


# ---------------------------------------------------------------------------
# Path & settings construction
# ---------------------------------------------------------------------------


def build_paths(
    global_cfg: dict,
    steps_cfg: dict,
    global_cfg_path: Path,
    steps_cfg_path: Path,
) -> Step07Paths:
    resolved_paths = resolve_paths(global_cfg)
    dataset_dir: Path = resolved_paths["dataset_dir"]
    output_dir: Path = resolved_paths["output_dir"]

    step03 = steps_cfg["step03"]
    step04 = steps_cfg["step04"]
    step07 = steps_cfg["step07"]
    data_files = global_cfg["data_files"]

    return Step07Paths(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        pareto_path=output_dir / step07["pareto_relpath"],
        manifest_path=output_dir / step07["manifest_relpath"],
        report_path=output_dir / step07["report_relpath"],
        assignment_path=output_dir / step03["assignment_relpath"],
        booth1_path=output_dir / step03["booth1_relpath"],
        booth2_path=output_dir / step03["booth2_relpath"],
        sdsu_booth1_path=output_dir / step04["booth1_output_relpath"],
        sdsu_booth2_path=output_dir / step04["booth2_output_relpath"],
        adjacency_rules_path=dataset_dir / data_files["adjacency_rules"],
        s_matrix_path=dataset_dir / data_files["s_matrix_transition"],
        topcoat_codebook_path=dataset_dir / data_files["topcoat_codebook"],
        topcoat_rules_path=dataset_dir / data_files["topcoat_rules"],
        cycle_quota_path=dataset_dir / data_files["cycle_quota"],
        global_config_path=global_cfg_path,
        steps_config_path=steps_cfg_path,
    )


def build_settings(
    global_cfg: dict,
    steps_cfg: dict,
    max_eval_override: Optional[int],
) -> Step07Settings:
    window_cfg = global_cfg.get("window", {})
    tokens_cfg = global_cfg.get("tokens", {})
    sdsu_cfg = global_cfg.get("sdsu", {})
    nsga_cfg = global_cfg.get("nsga2", {})
    adjacency_cfg = global_cfg.get("adjacency", {})
    rho_cfg = global_cfg.get("rho", {})
    reproducibility_cfg = global_cfg.get("reproducibility", {})
    step05_cfg = steps_cfg.get("step05", {})
    step07_cfg = steps_cfg.get("step07", {})

    tie_break_policy = str(step05_cfg.get("tie_break_policy", "longer_remaining"))
    repair_L_max = int(nsga_cfg.get("repair_L_max", adjacency_cfg.get("fifo_lookahead", 3)))
    rho_scale = float(rho_cfg.get("scale", 1.0))
    seeds = list(reproducibility_cfg.get("seeds", [19]))

    max_eval_per_seed = int(
        max_eval_override or step07_cfg.get("max_eval_per_seed", 20000)
    )

    flush_units_ratio = float(tokens_cfg.get("flush_units_ratio", 1.0))
    purge_weight = float(sdsu_cfg.get("purge_cost_weight", 1.0))
    allow_purge = bool(sdsu_cfg.get("allow_purge", True))

    return Step07Settings(
        tie_break_policy=tie_break_policy,
        repair_L_max=repair_L_max,
        rho_scale=rho_scale,
        window_mode=str(window_cfg.get("mode", "jobs")),
        window_jobs=int(window_cfg.get("jobs", 0)),
        window_ratio=float(window_cfg.get("ratio", 0.15)),
        allow_purge=allow_purge,
        flush_cost_units=flush_units_ratio * purge_weight,
        population_size=int(nsga_cfg.get("population_size", 60)),
        generations=int(nsga_cfg.get("generations", 80)),
        crossover_rate=float(nsga_cfg.get("crossover_rate", 0.9)),
        mutation_rate=float(nsga_cfg.get("mutation_rate", 0.1)),
        use_seeding=bool(nsga_cfg.get("use_seeding", True)),
        use_sdsu_kernel=bool(nsga_cfg.get("sdsu_kernel", True)),
        seeds=seeds,
        max_eval_per_seed=max_eval_per_seed,
    )


# ---------------------------------------------------------------------------
# Data loading: s-matrix, topcoat, adjacency, target mix
# ---------------------------------------------------------------------------


def load_s_matrix_transition(
    global_cfg: dict,
    s_matrix_path: Path,
) -> Dict[Tuple[str, str], float]:
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


def load_topcoat_codebook(
    global_cfg: dict,
    codebook_path: Path,
) -> Dict[int, str]:
    """
    Map topcoat_no -> topcoat_code for use in topcoat_rules.
    """
    schema = global_cfg.get("data_schema", {}).get("topcoat_codebook", {})
    df = pd.read_csv(codebook_path)

    no_col = detect_column(df, schema.get("topcoat_no_synonyms", ["topcoat_no"]))
    code_col = detect_column(
        df,
        schema.get("topcoat_code_synonyms", ["topcoat_code", "color_code"]),
    )

    mapping: Dict[int, str] = {}
    for _, row in df.iterrows():
        try:
            mapping[int(row[no_col])] = str(row[code_col])
        except (TypeError, ValueError):
            continue
    return mapping


def load_topcoat_forbids(
    global_cfg: dict,
    rules_path: Path,
    code_by_no: Dict[int, str],
) -> set:
    """
    Build set of (from_topcoat_code, to_topcoat_code) that are forbidden
    by paint rules (C1/C3).
    """
    if not rules_path.exists():
        return set()

    schema = global_cfg.get("data_schema", {}).get("topcoat_rules", {})
    df = pd.read_csv(rules_path)

    rel_col = detect_column(
        df,
        schema.get("relation_synonyms", ["relation"]),
        required=False,
    )
    from_code_col = detect_column(
        df,
        schema.get("from_topcoat_code_synonyms", ["from_topcoat_code", "from_code"]),
        required=False,
    )
    to_code_col = detect_column(
        df,
        schema.get("to_topcoat_code_synonyms", ["to_topcoat_code", "to_code"]),
        required=False,
    )
    from_no_col = detect_column(
        df,
        schema.get("from_topcoat_no_synonyms", ["from_topcoat_no"]),
        required=from_code_col is None,
    )
    to_no_col = detect_column(
        df,
        schema.get("to_topcoat_no_synonyms", ["to_topcoat_no"]),
        required=to_code_col is None,
    )

    forbids: set = set()
    for _, row in df.iterrows():
        relation = str(row[rel_col]).lower() if rel_col else ""
        if "forbid" not in relation:
            continue

        try:
            from_code = (
                str(row[from_code_col])
                if from_code_col
                else code_by_no.get(int(row[from_no_col]), "")
            )
            to_code = (
                str(row[to_code_col])
                if to_code_col
                else code_by_no.get(int(row[to_no_col]), "")
            )
        except (TypeError, ValueError):
            continue

        if not from_code or not to_code:
            continue

        forbids.add((from_code, to_code))

    return forbids


def build_adjacency_sets(
    global_cfg: dict,
    adjacency_path: Path,
) -> Tuple[set, set]:
    """
    Build in-booth and global adjacency forbids from adjacency_rules.csv.

    C2 semantics are fully data-driven; scope escalation is controlled by
    global_cfg["adjacency"]["scope_escalation"].
    """
    schema = global_cfg.get("data_schema", {}).get("adjacency_rules", {})
    df = pd.read_csv(adjacency_path)

    from_col = detect_column(df, schema.get("from_part_id_synonyms", ["from_part_id"]))
    to_col = detect_column(df, schema.get("to_part_id_synonyms", ["to_part_id"]))
    scope_col = detect_column(
        df,
        schema.get("scope_synonyms", ["scope"]),
        required=False,
    )
    rel_col = detect_column(
        df,
        schema.get("relation_synonyms", ["relation"]),
        required=False,
    )

    forb_in_booth: set = set()
    forb_global: set = set()

    escalation = (
        global_cfg.get("adjacency", {}).get("scope_escalation", "global_if_missing")
    ).lower()

    for _, row in df.iterrows():
        relation = str(row[rel_col]).lower() if rel_col else ""
        if "forbid" not in relation:
            continue

        scope_value = str(row[scope_col]).lower() if scope_col else ""
        scopes: set = set()

        if scope_value in {"in_booth", "inbooth", "booth"}:
            scopes.add("in_booth")
        elif scope_value in {"global", "across", "across_skids"}:
            scopes.add("global")
        elif escalation == "global_if_missing":
            scopes.update({"in_booth", "global"})
        else:
            scopes.add("global")

        pair = (str(row[from_col]), str(row[to_col]))

        if "in_booth" in scopes:
            forb_in_booth.add(pair)
        if "global" in scopes:
            forb_global.add(pair)

    return forb_in_booth, forb_global


def load_cycle_quota_mix(
    global_cfg: dict,
    cycle_quota_path: Path,
) -> Dict[str, float]:
    """
    Normalized target share for each color from cycle_quota.csv.
    """
    if not cycle_quota_path.exists():
        return {}

    schema = global_cfg.get("data_schema", {}).get("cycle_quota", {})
    df = pd.read_csv(cycle_quota_path)

    code_col = detect_column(
        df,
        schema.get("topcoat_code_synonyms", ["topcoat_code", "color_code"]),
        required=False,
    )
    part_col = detect_column(
        df,
        schema.get("part_id_synonyms", ["part_id"]),
        required=False,
    )
    quota_col = detect_column(
        df,
        schema.get("quota_synonyms", ["quota"]),
    )

    key_col = code_col or part_col
    if not key_col:
        return {}

    totals: Dict[str, float] = {}
    for _, row in df.iterrows():
        try:
            qty = float(row[quota_col])
        except (TypeError, ValueError):
            continue
        key = str(row[key_col])
        totals[key] = totals.get(key, 0.0) + qty

    denom = sum(totals.values())
    if denom <= 0:
        return {}

    return {k: v / denom for k, v in totals.items()}


def derive_target_mix(
    job_records: List[JobRecord],
    quota_mix: Dict[str, float],
) -> Dict[str, float]:
    """
    If quota_mix is provided, use it. Otherwise, derive empirical mix from jobs.
    """
    if quota_mix:
        return quota_mix

    counts = Counter(job.topcoat for job in job_records)
    total = sum(counts.values())
    if total <= 0:
        return {}

    return {code: cnt / total for code, cnt in counts.items()}


def build_paint_kernel(
    global_cfg: dict,
    paths: Step07Paths,
    settings: Step07Settings,
) -> PaintCostKernel:
    s_matrix = load_s_matrix_transition(global_cfg, paths.s_matrix_path)
    code_map = load_topcoat_codebook(global_cfg, paths.topcoat_codebook_path)
    forbids = load_topcoat_forbids(global_cfg, paths.topcoat_rules_path, code_map)
    return PaintCostKernel(
        s_matrix=s_matrix,
        forbidden_pairs=forbids,
        allow_purge=settings.allow_purge,
        flush_cost_units=settings.flush_cost_units,
    )


# ---------------------------------------------------------------------------
# Job catalog and SDSU data
# ---------------------------------------------------------------------------


def detect_job_columns(
    global_cfg: dict,
    df: pd.DataFrame,
) -> Tuple[str, str, str]:
    schema = global_cfg.get("data_schema", {}).get("demands", {})
    part_col = detect_column(
        df,
        schema.get("part_id_synonyms", ["part_id"]),
    )
    topcoat_col = detect_column(
        df,
        schema.get("topcoat_code_synonyms", ["topcoat_code", "color_code"]),
    )
    job_col = detect_column(
        df,
        schema.get("job_id_synonyms", ["job_id", "job", "jobid"]),
    )
    return part_col, topcoat_col, job_col


def merge_job_metadata(
    global_cfg: dict,
    paths: Step07Paths,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load Step-03 booth metadata and return as (booth1_df, booth2_df).
    """
    b1 = pd.read_csv(paths.booth1_path)
    b2 = pd.read_csv(paths.booth2_path)
    return b1, b2


def build_job_catalog(
    global_cfg: dict,
    paths: Step07Paths,
) -> Tuple[List[JobRecord], Dict[str, int], List[int]]:
    """
    Build JobRecord list, job_index_by_id mapping, and baseline_sigma
    from Step-03 assignment and booth metadata.
    """
    assignment = pd.read_csv(paths.assignment_path).sort_values("feeder_job_index")
    jobs1, jobs2 = merge_job_metadata(global_cfg, paths)

    part_col1, topcoat_col1, job_col1 = detect_job_columns(global_cfg, jobs1)
    part_col2, topcoat_col2, job_col2 = detect_job_columns(global_cfg, jobs2)

    meta = pd.concat([jobs1, jobs2], ignore_index=True)
    part_col = part_col1 if part_col1 in meta.columns else part_col2
    topcoat_col = topcoat_col1 if topcoat_col1 in meta.columns else topcoat_col2
    job_col = job_col1 if job_col1 in meta.columns else job_col2

    meta_map = {str(row[job_col]): row for _, row in meta.iterrows()}

    records: List[JobRecord] = []
    baseline_sigma: List[int] = []
    job_index_by_id: Dict[str, int] = {}

    for idx, row in enumerate(assignment.itertuples(index=False)):
        job_id = str(row.job_id)
        meta_row = meta_map[job_id]
        records.append(
            JobRecord(
                job_id=job_id,
                feeder_index=int(row.feeder_job_index),
                part_id=str(meta_row[part_col]),
                topcoat=str(meta_row[topcoat_col]),
                base_booth=int(row.booth_id),
            )
        )
        job_index_by_id[job_id] = idx
        baseline_sigma.append(int(row.booth_id))

    return records, job_index_by_id, baseline_sigma


def load_sdsu_orders(
    paths: Step07Paths,
    job_index_by_id: Dict[str, int],
) -> Dict[str, List[int]]:
    """
    Load SDSU booth sequences (if present) and map job IDs to indices.
    """
    orders: Dict[str, List[int]] = {"PB1": [], "PB2": []}

    if paths.sdsu_booth1_path.exists():
        df1 = pd.read_csv(paths.sdsu_booth1_path)
        job_col = detect_column(df1, ["job_id", "job", "jobid"], required=False)
        if job_col:
            orders["PB1"] = [
                job_index_by_id[str(jid)]
                for jid in df1[job_col]
                if str(jid) in job_index_by_id
            ]

    if paths.sdsu_booth2_path.exists():
        df2 = pd.read_csv(paths.sdsu_booth2_path)
        job_col = detect_column(df2, ["job_id", "job", "jobid"], required=False)
        if job_col:
            orders["PB2"] = [
                job_index_by_id[str(jid)]
                for jid in df2[job_col]
                if str(jid) in job_index_by_id
            ]

    return orders


def load_step07_data(
    global_cfg: dict,
    paths: Step07Paths,
    settings: Step07Settings,
) -> Step07Data:
    job_records, job_index_by_id, baseline_sigma = build_job_catalog(global_cfg, paths)
    quota_mix = load_cycle_quota_mix(global_cfg, paths.cycle_quota_path)
    target_mix = derive_target_mix(job_records, quota_mix)
    forb_in_booth, forb_global = build_adjacency_sets(
        global_cfg, paths.adjacency_rules_path
    )
    sdsu_orders = load_sdsu_orders(paths, job_index_by_id)

    return Step07Data(
        job_records=job_records,
        job_index_by_id=job_index_by_id,
        baseline_sigma=baseline_sigma,
        target_mix=target_mix,
        forb_in_booth=forb_in_booth,
        forb_global=forb_global,
        window_mode=settings.window_mode,
        window_jobs=settings.window_jobs,
        window_ratio=settings.window_ratio,
        tie_break_policy=settings.tie_break_policy,
        repair_L_max=settings.repair_L_max,
        rho_scale=settings.rho_scale,
        sdsu_orders=sdsu_orders,
    )


# ---------------------------------------------------------------------------
# Representation utilities
# ---------------------------------------------------------------------------


def enforce_permutation_consistency(ind: Individual) -> None:
    """
    Ensure pi1 and pi2 contain exactly the indices assigned to each booth by sigma.
    """
    membership1 = {idx for idx, booth in enumerate(ind.sigma) if booth == 1}
    membership2 = {idx for idx, booth in enumerate(ind.sigma) if booth == 2}

    ind.pi1 = [idx for idx in ind.pi1 if idx in membership1]
    ind.pi2 = [idx for idx in ind.pi2 if idx in membership2]

    missing1 = membership1.difference(ind.pi1)
    missing2 = membership2.difference(ind.pi2)

    ind.pi1.extend(missing1)
    ind.pi2.extend(missing2)


def assemble_permutation(
    sigma: Sequence[int],
    booth: int,
    parent_a: Sequence[int],
    parent_b: Sequence[int],
    rng: random.Random,
) -> List[int]:
    """
    Booth-level permutation assembly inspired by order crossover:
    take jobs that belong to 'booth' in parent order, preserving relative order
    and then fill remaining in random order.
    """
    allowed = {idx for idx, value in enumerate(sigma) if value == booth}
    result: List[int] = []
    seen: set = set()

    for source in (parent_a, parent_b):
        for job_idx in source:
            if job_idx in allowed and job_idx not in seen:
                result.append(job_idx)
                seen.add(job_idx)

    remaining = [idx for idx in allowed if idx not in seen]
    rng.shuffle(remaining)
    result.extend(remaining)
    return result


def mutate_assignment(ind: Individual, rng: random.Random) -> None:
    """
    Flip a single position in sigma between 1 and 2.
    """
    if not ind.sigma:
        return
    pos = rng.randrange(len(ind.sigma))
    ind.sigma[pos] = 1 if ind.sigma[pos] == 2 else 2


def mutate_permutation(pi: List[int], rng: random.Random) -> None:
    """
    Simple swap mutation inside a booth permutation.
    """
    if len(pi) < 2:
        return
    i = rng.randrange(len(pi))
    j = rng.randrange(len(pi))
    if i != j:
        pi[i], pi[j] = pi[j], pi[i]


def apply_sdsu_kernel(
    seq: List[int],
    booth_label: str,
    data: Step07Data,
) -> List[int]:
    """
    SDSU-informed kernel: realign seq towards SDSU reference order if available.

    This is an alignment kernel, not a full embedded SDSU search.
    """
    ref = data.sdsu_orders.get(booth_label)
    if not ref:
        return seq

    ref_pos = {job_idx: pos for pos, job_idx in enumerate(ref)}
    fallback_order = {job_idx: pos for pos, job_idx in enumerate(seq)}

    def sort_key(job_idx: int) -> Tuple[int, int]:
        if job_idx in ref_pos:
            return (0, ref_pos[job_idx])
        return (1, fallback_order[job_idx])

    return sorted(seq, key=sort_key)


# ---------------------------------------------------------------------------
# Decoder: adjacency repair, merge, feasibility check
# ---------------------------------------------------------------------------


def repair_adjacency(
    seq: List[int],
    data: Step07Data,
) -> Tuple[List[int], int]:
    """
    Repair in-booth adjacency (C2 in-booth) with bounded local reshuffle.

    Returns:
        (repaired_sequence, number_of_repairs)
    """
    repairs = 0
    repaired = list(seq)
    i = 0
    n = len(repaired)
    L_max = data.repair_L_max

    while i < n - 1:
        part_a = data.job_records[repaired[i]].part_id
        part_b = data.job_records[repaired[i + 1]].part_id
        if (part_a, part_b) not in data.forb_in_booth:
            i += 1
            continue

        fixed = False
        for j in range(i + 2, min(n, i + 1 + L_max)):
            part_c = data.job_records[repaired[j]].part_id
            if (part_a, part_c) in data.forb_in_booth:
                continue
            if j + 1 < len(repaired):
                next_part = data.job_records[repaired[j + 1]].part_id
                if (part_c, next_part) in data.forb_in_booth:
                    continue
            repaired[i + 1], repaired[j] = repaired[j], repaired[i + 1]
            repairs += 1
            fixed = True
            break

        if not fixed:
            return [], repairs

        i += 1

    return repaired, repairs


def merge_sequences(
    seq1: List[int],
    seq2: List[int],
    data: Step07Data,
) -> Tuple[List[Tuple[str, int]], bool]:
    """
    Merge PB1 and PB2 sequences into a single JOB-only stream,
    enforcing global C2 from adjacency_rules.

    Returns:
        (merged_sequence, deadlock_flag)
    """
    h1 = 0
    h2 = 0
    last_part: Optional[str] = None
    merged: List[Tuple[str, int]] = []

    while h1 < len(seq1) or h2 < len(seq2):
        candidates: List[Tuple[str, int, str]] = []

        if h1 < len(seq1):
            part = data.job_records[seq1[h1]].part_id
            if last_part is None or (last_part, part) not in data.forb_global:
                candidates.append(("PB1", seq1[h1], part))

        if h2 < len(seq2):
            part = data.job_records[seq2[h2]].part_id
            if last_part is None or (last_part, part) not in data.forb_global:
                candidates.append(("PB2", seq2[h2], part))

        if not candidates:
            # Deadlock under strict no-EMPTY policy.
            return [], True

        if len(candidates) == 1:
            booth, job_idx, part = candidates[0]
        else:
            remaining1 = len(seq1) - h1
            remaining2 = len(seq2) - h2
            if data.tie_break_policy == "round_robin":
                booth = "PB1" if (len(merged) % 2 == 0 and h1 < len(seq1)) else "PB2"
            elif remaining1 > remaining2:
                booth = "PB1"
            elif remaining2 > remaining1:
                booth = "PB2"
            else:
                booth = "PB1" if h1 <= h2 else "PB2"

            job_idx = seq1[h1] if booth == "PB1" else seq2[h2]
            part = data.job_records[job_idx].part_id

        merged.append((booth, job_idx))
        last_part = part

        if booth == "PB1":
            h1 += 1
        else:
            h2 += 1

    return merged, False


def decode_individual(
    ind: Individual,
    data: Step07Data,
    kernel: PaintCostKernel,
    use_sdsu_kernel: bool,
) -> DecoderResult:
    """
    Deterministic decoder for an individual:
        σ -> booth job sets -> booth permutations (π1, π2) ->
        SDSU alignment (optional) -> adjacency repair (booth) ->
        merge with global C2.
    """
    enforce_permutation_consistency(ind)

    booth1_seq = list(ind.pi1)
    booth2_seq = list(ind.pi2)

    if use_sdsu_kernel:
        booth1_seq = apply_sdsu_kernel(booth1_seq, "PB1", data)
        booth2_seq = apply_sdsu_kernel(booth2_seq, "PB2", data)

    booth1_seq, repairs1 = repair_adjacency(booth1_seq, data)
    if not booth1_seq:
        return DecoderResult(False, [], [], [], repairs1, 0)

    booth2_seq, repairs2 = repair_adjacency(booth2_seq, data)
    if not booth2_seq:
        return DecoderResult(False, [], [], [], repairs1 + repairs2, 0)

    global_seq, deadlock = merge_sequences(booth1_seq, booth2_seq, data)
    if deadlock:
        return DecoderResult(False, [], [], [], repairs1 + repairs2, 0)

    # Paint feasibility will be validated in evaluate_individual via kernel.
    return DecoderResult(True, booth1_seq, booth2_seq, global_seq, repairs1 + repairs2, 0)


# ---------------------------------------------------------------------------
# Objectives and evaluation (f1, f2, f3)
# ---------------------------------------------------------------------------


def compute_paint_cost(
    booth_seq: List[int],
    data: Step07Data,
    kernel: PaintCostKernel,
) -> float:
    """
    Compute f1 contribution for a single booth:
        sum over transitions of SDST or FLUSH, as per PaintCostKernel.
    """
    cost = 0.0
    prev_color: Optional[str] = None

    for idx in booth_seq:
        color = data.job_records[idx].topcoat
        cost += kernel.transition_cost(prev_color, color)
        prev_color = color

    return cost


def compute_window_mix(
    global_seq: List[Tuple[str, int]],
    data: Step07Data,
) -> float:
    """
    Compute windowed color-mix deviation (f2) on the merged JOB-only convoy.

    Windows count only parts; colors from job_records.topcoat.
    """
    if not global_seq:
        return 0.0

    topcoats = [data.job_records[j_idx].topcoat for _, j_idx in global_seq]
    n = len(topcoats)

    if data.window_mode == "jobs":
        W = data.window_jobs
    else:
        ratio = data.window_ratio
        W = max(1, int(math.floor(ratio * n)))

    W = max(1, min(W, n))
    if W <= 0 or n < W:
        return 0.0

    target = data.target_mix
    window_counter: Counter = Counter(topcoats[:W])

    def window_deviation(counter: Counter) -> float:
        window_size = float(sum(counter.values()))
        if window_size <= 0:
            return 0.0
        dev = 0.0
        union_colors = set(counter.keys()) | set(target.keys())
        for coat in union_colors:
            share_window = counter.get(coat, 0) / window_size
            share_target = target.get(coat, 0.0)
            diff = share_window - share_target
            dev += diff * diff
        return dev

    total_dev = window_deviation(window_counter)
    n_windows = n - W + 1

    for start in range(1, n_windows):
        outgoing = topcoats[start - 1]
        window_counter[outgoing] -= 1
        if window_counter[outgoing] <= 0:
            window_counter.pop(outgoing, None)
        incoming = topcoats[start + W - 1]
        window_counter[incoming] += 1
        total_dev += window_deviation(window_counter)

    return total_dev / float(n_windows)


def compute_split_objective(
    labels: Sequence[int],
    rho_scale: float,
) -> float:
    """
    Split-based surrogate f3:
        f3 = |N1 - N2| + rho_scale * cross(σ),
    where cross counts PB1↔PB2 changes over feeder order.
    """
    n = len(labels)
    if n == 0:
        return 0.0
    n1 = sum(1 for b in labels if b == 1)
    n2 = n - n1

    cross = 0
    for i in range(1, n):
        if labels[i] != labels[i - 1]:
            cross += 1

    return abs(n1 - n2) + rho_scale * float(cross)


def evaluate_individual(
    ind: Individual,
    data: Step07Data,
    kernel: PaintCostKernel,
    use_sdsu_kernel: bool,
) -> None:
    """
    Decode and evaluate (f1, f2, f3) for a single individual.
    Sets ind.feasible and objective values.
    """
    result = decode_individual(ind, data, kernel, use_sdsu_kernel)
    if not result.feasible:
        ind.feasible = False
        ind.f1 = math.inf
        ind.f2 = math.inf
        ind.f3 = math.inf
        return

    try:
        f1 = compute_paint_cost(result.booth1_seq, data, kernel) + compute_paint_cost(
            result.booth2_seq, data, kernel
        )
    except PaintFeasibilityError:
        ind.feasible = False
        ind.f1 = math.inf
        ind.f2 = math.inf
        ind.f3 = math.inf
        return

    f2 = compute_window_mix(result.global_seq, data)
    f3 = compute_split_objective(ind.sigma, data.rho_scale)

    ind.f1 = f1
    ind.f2 = f2
    ind.f3 = f3
    ind.feasible = True


def evaluate_population(
    pop: List[Individual],
    data: Step07Data,
    kernel: PaintCostKernel,
    use_sdsu_kernel: bool,
) -> None:
    for ind in pop:
        evaluate_individual(ind, data, kernel, use_sdsu_kernel)


# ---------------------------------------------------------------------------
# NSGA-II core
# ---------------------------------------------------------------------------


def dominates(a: Individual, b: Individual) -> bool:
    """
    Standard Pareto dominance on (f1, f2, f3).
    """
    if not a.feasible and b.feasible:
        return False
    if a.feasible and not b.feasible:
        return True
    if not a.feasible and not b.feasible:
        return False

    better_or_equal = (
        a.f1 <= b.f1 and a.f2 <= b.f2 and a.f3 <= b.f3
    )
    strictly_better = (
        a.f1 < b.f1 or a.f2 < b.f2 or a.f3 < b.f3
    )

    return better_or_equal and strictly_better


def fast_non_dominated_sort(population: List[Individual]) -> List[List[Individual]]:
    """
    Deb's fast non-dominated sorting (NSGA-II).
    """
    S: Dict[int, List[int]] = {}
    n_dom: Dict[int, int] = {}
    fronts: List[List[Individual]] = []

    for i, p in enumerate(population):
        S[i] = []
        n_dom[i] = 0
        for j, q in enumerate(population):
            if i == j:
                continue
            if dominates(p, q):
                S[i].append(j)
            elif dominates(q, p):
                n_dom[i] += 1

        if n_dom[i] == 0:
            p.rank = 0

    current_front = [i for i, cnt in n_dom.items() if cnt == 0]
    for idx in current_front:
        population[idx].rank = 0
    fronts.append([population[i] for i in current_front])

    rank = 0
    while current_front:
        next_front: List[int] = []
        for i in current_front:
            for j in S[i]:
                n_dom[j] -= 1
                if n_dom[j] == 0:
                    population[j].rank = rank + 1
                    next_front.append(j)
        rank += 1
        current_front = next_front
        if current_front:
            fronts.append([population[i] for i in current_front])

    return fronts


def assign_crowding_distance(front: List[Individual]) -> None:
    """
    Standard NSGA-II crowding distance on (f1, f2, f3).
    """
    if not front:
        return

    for ind in front:
        ind.crowding_distance = 0.0

    objectives = ["f1", "f2", "f3"]
    for obj in objectives:
        front.sort(key=lambda ind: getattr(ind, obj))
        front[0].crowding_distance = math.inf
        front[-1].crowding_distance = math.inf

        min_val = getattr(front[0], obj)
        max_val = getattr(front[-1], obj)
        if max_val == min_val:
            continue

        for i in range(1, len(front) - 1):
            prev_val = getattr(front[i - 1], obj)
            next_val = getattr(front[i + 1], obj)
            front[i].crowding_distance += (next_val - prev_val) / (max_val - min_val)


def tournament_select(
    pop: List[Individual],
    rng: random.Random,
    k: int,
) -> Individual:
    candidates = rng.sample(pop, min(k, len(pop)))
    candidates.sort(key=lambda ind: (ind.rank, -ind.crowding_distance))
    return candidates[0]


def uniform_crossover_sigma(
    s1: Sequence[int],
    s2: Sequence[int],
    rng: random.Random,
) -> Tuple[List[int], List[int]]:
    child1: List[int] = []
    child2: List[int] = []

    for v1, v2 in zip(s1, s2):
        if rng.random() < 0.5:
            child1.append(v1)
            child2.append(v2)
        else:
            child1.append(v2)
            child2.append(v1)

    return child1, child2


def crossover(
    parent1: Individual,
    parent2: Individual,
    rng: random.Random,
) -> Tuple[Individual, Individual]:
    """
    Crossover on σ (uniform) and booth permutations (assemble_permutation).
    """
    s1, s2 = uniform_crossover_sigma(parent1.sigma, parent2.sigma, rng)

    pi1_child1 = assemble_permutation(s1, 1, parent1.pi1, parent2.pi1, rng)
    pi1_child2 = assemble_permutation(s2, 1, parent1.pi1, parent2.pi1, rng)
    pi2_child1 = assemble_permutation(s1, 2, parent1.pi2, parent2.pi2, rng)
    pi2_child2 = assemble_permutation(s2, 2, parent1.pi2, parent2.pi2, rng)

    child1 = Individual(sigma=s1, pi1=pi1_child1, pi2=pi2_child1)
    child2 = Individual(sigma=s2, pi1=pi1_child2, pi2=pi2_child2)

    return child1, child2


def mutate(
    ind: Individual,
    rng: random.Random,
    mutation_rate: float,
) -> None:
    """
    Apply sigma and permutation mutations with given rate.
    """
    if rng.random() < mutation_rate:
        mutate_assignment(ind, rng)
    if rng.random() < mutation_rate:
        mutate_permutation(ind.pi1, rng)
    if rng.random() < mutation_rate:
        mutate_permutation(ind.pi2, rng)


def random_individual(
    n_jobs: int,
    rng: random.Random,
) -> Individual:
    sigma = [rng.choice((1, 2)) for _ in range(n_jobs)]
    pi1 = [idx for idx, booth in enumerate(sigma) if booth == 1]
    pi2 = [idx for idx, booth in enumerate(sigma) if booth == 2]
    rng.shuffle(pi1)
    rng.shuffle(pi2)
    return Individual(sigma=sigma, pi1=pi1, pi2=pi2)


def build_seed_population(
    data: Step07Data,
    rng: random.Random,
    use_seeding: bool,
) -> List[Individual]:
    """
    Build initial population:
        - optional Toyota seed (baseline sigma),
        - optional SDSU seeds (if Step-04 outputs are present),
        - otherwise random individuals.
    """
    n_jobs = len(data.job_records)
    population: List[Individual] = []

    if use_seeding:
        # Toyota seed
        sigma_t = list(data.baseline_sigma)
        pi1_t = [idx for idx, booth in enumerate(sigma_t) if booth == 1]
        pi2_t = [idx for idx, booth in enumerate(sigma_t) if booth == 2]
        population.append(
            Individual(
                sigma=sigma_t,
                pi1=pi1_t,
                pi2=pi2_t,
                origin="toyota",
            )
        )

        # SDSU seed, if available
        sdsu_b1 = data.sdsu_orders.get("PB1") or []
        sdsu_b2 = data.sdsu_orders.get("PB2") or []
        if sdsu_b1 or sdsu_b2:
            sigma_sdsu = list(data.baseline_sigma)
            population.append(
                Individual(
                    sigma=sigma_sdsu,
                    pi1=list(sdsu_b1),
                    pi2=list(sdsu_b2),
                    origin="sdsu",
                )
            )

    # Population will be filled to required size later.
    return population


def make_offspring(
    pop: List[Individual],
    rng: random.Random,
    pop_size: int,
    crossover_rate: float,
    mutation_rate: float,
) -> List[Individual]:
    offspring: List[Individual] = []
    while len(offspring) < pop_size:
        p1 = tournament_select(pop, rng, 2)
        p2 = tournament_select(pop, rng, 2)
        if rng.random() < crossover_rate:
            c1, c2 = crossover(p1, p2, rng)
        else:
            c1 = Individual(sigma=list(p1.sigma), pi1=list(p1.pi1), pi2=list(p1.pi2))
            c2 = Individual(sigma=list(p2.sigma), pi1=list(p2.pi1), pi2=list(p2.pi2))
        mutate(c1, rng, mutation_rate)
        mutate(c2, rng, mutation_rate)
        offspring.append(c1)
        if len(offspring) < pop_size:
            offspring.append(c2)
    return offspring


def nsga2(
    seed: int,
    data: Step07Data,
    settings: Step07Settings,
    kernel: PaintCostKernel,
) -> List[Individual]:
    rng = random.Random(seed)

    pop_size = settings.population_size
    generations = settings.generations
    crossover_rate = settings.crossover_rate
    mutation_rate = settings.mutation_rate
    use_sdsu_kernel = settings.use_sdsu_kernel
    max_eval = settings.max_eval_per_seed

    population = build_seed_population(data, rng, settings.use_seeding)
    log_progress(
        "seed %s: initializing NSGA-II (pop=%d target_pop=%d generations=%d max_eval=%d)"
        % (seed, len(population), pop_size, generations, max_eval)
    )

    while len(population) < pop_size:
        population.append(random_individual(len(data.job_records), rng))

    evaluate_population(population, data, kernel, use_sdsu_kernel)
    eval_count = len(population)
    log_progress(
        "seed %s: initial evaluation complete (%d individuals)" % (seed, eval_count)
    )

    gen = 0
    log_interval = max(1, generations // 10) if generations else 1

    update_progress_line(
        "seed %s: gen %d/%d evals %d/%d"
        % (seed, gen, generations, eval_count, max_eval)
    )

    while gen < generations and eval_count < max_eval:
        fronts = fast_non_dominated_sort(population)
        for front in fronts:
            assign_crowding_distance(front)

        offspring = make_offspring(
            population,
            rng,
            pop_size,
            crossover_rate,
            mutation_rate,
        )
        evaluate_population(offspring, data, kernel, use_sdsu_kernel)
        eval_count += len(offspring)

        combined = population + offspring
        fronts = fast_non_dominated_sort(combined)
        new_population: List[Individual] = []

        for front in fronts:
            assign_crowding_distance(front)
            if len(new_population) + len(front) <= pop_size:
                new_population.extend(front)
            else:
                front.sort(
                    key=lambda ind: -ind.crowding_distance,
                )
                needed = pop_size - len(new_population)
                new_population.extend(front[:needed])
                break

        population = new_population
        gen += 1

        if gen % log_interval == 0 or gen == generations:
            update_progress_line(
                "seed %s: gen %d/%d evals %d/%d"
                % (seed, gen, generations, eval_count, max_eval),
                final=(gen == generations or eval_count >= max_eval),
            )

    update_progress_line(
        "seed %s: finished NSGA-II (gen=%d evals=%d)" % (seed, gen, eval_count),
        final=True,
    )

    fronts = fast_non_dominated_sort(population)
    return fronts[0] if fronts else []


def global_front(all_inds: List[Individual]) -> List[Individual]:
    """
    Compute a global non-dominated front from all seed-level individuals.
    """
    if not all_inds:
        return []
    fronts = fast_non_dominated_sort(all_inds)
    first_front = fronts[0]
    assign_crowding_distance(first_front)
    return first_front


# ---------------------------------------------------------------------------
# Manifest / report
# ---------------------------------------------------------------------------

def export_pareto_sequences(
    paths: Step07Paths,
    settings: Step07Settings,
    data: Step07Data,
    kernel: PaintCostKernel,
    front: List[Individual],
) -> Tuple[Path, Path]:
    """
    Export per-booth and global sequences for each Pareto solution.

    Output:
        - one booth file (PB1 + PB2) with suffix "booth_sequences"
        - one global file with suffix "global_line"

    Both files are keyed by solution_id.
    """
    booth_records: List[Dict[str, object]] = []
    global_records: List[Dict[str, object]] = []

    for solution_id, ind in enumerate(front):
        if not ind.feasible:
            continue

        # Deterministic decoder with same semantics as evaluation
        decoded = decode_individual(ind, data, kernel, settings.use_sdsu_kernel)
        if not decoded.feasible:
            continue

        # PB1 & PB2 per-booth sequences
        for booth_label, seq in (("PB1", decoded.booth1_seq), ("PB2", decoded.booth2_seq)):
            booth_id = 1 if booth_label == "PB1" else 2
            for pos, job_idx in enumerate(seq):
                job = data.job_records[job_idx]
                booth_records.append(
                    {
                        "solution_id": solution_id,
                        "booth_label": booth_label,
                        "booth_id": booth_id,
                        "position_in_booth": pos,
                        "job_index": job_idx,
                        "job_id": job.job_id,
                        "feeder_index": job.feeder_index,
                        "part_id": job.part_id,
                        "topcoat": job.topcoat,
                    }
                )

        # Global merged convoy G
        for global_pos, (booth_label, job_idx) in enumerate(decoded.global_seq):
            job = data.job_records[job_idx]
            booth_id = 1 if booth_label == "PB1" else 2
            global_records.append(
                {
                    "solution_id": solution_id,
                    "global_position": global_pos,
                    "booth_label": booth_label,
                    "booth_id": booth_id,
                    "job_index": job_idx,
                    "job_id": job.job_id,
                    "feeder_index": job.feeder_index,
                    "part_id": job.part_id,
                    "topcoat": job.topcoat,
                }
            )

    # File names from pareto path
    pareto_name = paths.pareto_path.name
    if "pareto" in pareto_name:
        booth_name = pareto_name.replace("pareto", "booth_sequences")
        global_name = pareto_name.replace("pareto", "global_line")
    else:
        base = paths.pareto_path.stem
        booth_name = f"{base}_booth_sequences.csv"
        global_name = f"{base}_global_line.csv"

    booth_path = paths.pareto_path.with_name(booth_name)
    global_path = paths.pareto_path.with_name(global_name)

    if booth_records:
        ensure_parent_dir(booth_path)
        pd.DataFrame.from_records(booth_records).to_csv(booth_path, index=False)
        log_progress(
            f"exported booth sequences for {len(front)} solutions to {booth_path}"
        )

    if global_records:
        ensure_parent_dir(global_path)
        pd.DataFrame.from_records(global_records).to_csv(global_path, index=False)
        log_progress(
            f"exported global lines for {len(front)} solutions to {global_path}"
        )

    return booth_path, global_path


def write_manifest_and_report(
    paths: Step07Paths,
    settings: Step07Settings,
    data: Step07Data,
    pareto_path: Path,
    inputs: List[Path],
    outputs: List[Path],
    front: List[Individual],
    runtime: float,
) -> None:
    ensure_parent_dir(paths.manifest_path)
    ensure_parent_dir(paths.report_path)

    feasible_count = sum(1 for ind in front if ind.feasible)

    manifest = {
        "step": STEP_ID,
        "config_files": {
            str(paths.global_config_path): sha256_of_file(paths.global_config_path)
            if paths.global_config_path.exists()
            else None,
            str(paths.steps_config_path): sha256_of_file(paths.steps_config_path)
            if paths.steps_config_path.exists()
            else None,
        },
        "input_files": {
            str(p): sha256_of_file(p) for p in inputs if Path(p).exists()
        },
        "output_files": {
            str(p): sha256_of_file(p) for p in outputs if Path(p).exists()
        },
        "settings_effective": asdict(settings),
        "summary": {
            "num_jobs": len(data.job_records),
            "pareto_count": len(front),
            "feasible_solutions": feasible_count,
            "runtime_seconds": runtime,
        },
    }

    write_json(paths.manifest_path, manifest)

    lines = [
        "# STEP-07 — NSGA-II Pareto Search",
        "",
        f"Jobs evaluated: {len(data.job_records)}",
        f"Pareto solutions recorded: {len(front)} (feasible: {feasible_count})",
        f"Runtime: {runtime:.2f} seconds",
        "",
        "## Seeds",
        ", ".join(str(seed) for seed in settings.seeds) or "<none>",
        "",
        "## Settings",
        json.dumps(asdict(settings), indent=2),
    ]

    write_text(paths.report_path, "\n".join(lines))


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="STEP-07 NSGA-II")
    parser.add_argument(
        "global_config",
        type=Path,
        nargs="?",
        default=Path("global_config.toml"),
        help="Path to global_config.toml (defaults to ./global_config.toml)",
    )
    parser.add_argument(
        "steps_config",
        type=Path,
        nargs="?",
        default=Path("steps_config.toml"),
        help="Path to steps_config.toml (defaults to ./steps_config.toml)",
    )
    parser.add_argument(
        "--max-eval",
        type=int,
        default=None,
        help="Override max_eval_per_seed (optional).",
    )

    args = parser.parse_args(argv)

    global_cfg_path: Path = args.global_config
    steps_cfg_path: Path = args.steps_config

    global_cfg = load_global_config(global_cfg_path)
    steps_cfg = load_steps_config(steps_cfg_path)

    paths = build_paths(global_cfg, steps_cfg, global_cfg_path, steps_cfg_path)
    settings = build_settings(global_cfg, steps_cfg, args.max_eval)

    log_progress("loading data and kernels")
    data = load_step07_data(global_cfg, paths, settings)
    kernel = build_paint_kernel(global_cfg, paths, settings)

    all_inds: List[Individual] = []
    start = time.perf_counter()

    for seed in settings.seeds:
        log_progress(f"seed {seed}: starting run")
        seed_start = time.perf_counter()
        front = nsga2(seed, data, settings, kernel)
        seed_runtime = time.perf_counter() - seed_start
        log_progress(
            f"seed {seed}: completed run in {seed_runtime:.2f}s with {len(front)} solutions"
        )
        for ind in front:
            ind.seed_id = seed
        all_inds.extend(front)

    final_front = global_front(all_inds)
    log_progress(
        f"global: pareto frontier consolidated with {len(final_front)} solutions"
    )

    # 1) Pareto metrics
    ensure_parent_dir(paths.pareto_path)
    pd.DataFrame(
        [
            {
                "solution_id": idx,
                "seed": ind.seed_id,
                "origin": ind.origin,
                "feasible": ind.feasible,
                "f1": ind.f1,
                "f2": ind.f2,
                "f3": ind.f3,
            }
            for idx, ind in enumerate(final_front)
        ]
    ).to_csv(paths.pareto_path, index=False)

    # 2) Booth + global oven lines per Pareto solution (JOB-only, adjacency-respecting)
    booth_path, sequences_path = export_pareto_sequences(
        paths,
        settings,
        data,
        kernel,
        final_front,
    )

    # 3) Manifest + report
    runtime = time.perf_counter() - start
    inputs = [
        paths.global_config_path,
        paths.steps_config_path,
        paths.assignment_path,
        paths.booth1_path,
        paths.booth2_path,
        paths.adjacency_rules_path,
        paths.cycle_quota_path,
        paths.s_matrix_path,
        paths.topcoat_codebook_path,
        paths.topcoat_rules_path,
    ]
    outputs = [
        paths.pareto_path,
        booth_path,
        sequences_path,
        paths.manifest_path,
        paths.report_path,
    ]
    write_manifest_and_report(
        paths,
        settings,
        data,
        paths.pareto_path,
        inputs,
        outputs,
        final_front,
        runtime,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"{LOG_PREFIX} fatal: {exc}", file=sys.stderr)
        raise

