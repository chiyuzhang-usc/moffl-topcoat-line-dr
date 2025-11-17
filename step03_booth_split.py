#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
import json
import time
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# TOML reader (py3.11+: tomllib; else: tomli)
try:
    import tomllib as _toml
except Exception:  # pragma: no cover
    import tomli as _toml  # type: ignore[no-redef]


# ------------------------------------------------------------------
# Generic utilities
# ------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    """Compute SHA-256 of a file, or empty string if it does not exist."""
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_toml(path: Path) -> dict:
    with path.open("rb") as f:
        return _toml.load(f)


def ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _rel(base: Path, maybe: str | Path | None) -> Optional[Path]:
    if maybe is None:
        return None
    p = Path(maybe)
    if not p.is_absolute():
        p = (base / p).resolve()
    return p


# ------------------------------------------------------------------
# Path resolution and configuration
# ------------------------------------------------------------------

@dataclass
class PathsCfg:
    # Inputs
    seed_csv: Path
    global_config: Path
    steps_config: Path

    # Outputs
    booth1_csv: Path
    booth2_csv: Path
    assignment_csv: Path
    manifest_json: Path
    report_md: Path

    # Roots
    output_dir: Path


def resolve_paths(global_cfg_path: Path, steps_cfg_path: Path) -> PathsCfg:
    gcfg = read_toml(global_cfg_path)
    scfg = read_toml(steps_cfg_path)

    gbase = global_cfg_path.parent
    paths_cfg = gcfg.get("paths", {})
    output_dir = _rel(gbase, paths_cfg.get("output_dir", "output")) or (gbase / "output")

    step03 = scfg.get("step03", {})

    seed_rel = step03.get("input_seed_relpath", "step02_paint_seed.csv")
    booth1_rel = step03.get("booth1_relpath", "step03_booth1.csv")
    booth2_rel = step03.get("booth2_relpath", "step03_booth2.csv")
    assign_rel = step03.get("assignment_relpath", "step03_assignment.csv")
    manifest_rel = step03.get("manifest_relpath", "manifests/step03_manifest.json")
    report_rel = step03.get("report_relpath", "reports/step03_report.md")

    return PathsCfg(
        seed_csv=_rel(output_dir, seed_rel) or (output_dir / seed_rel),
        global_config=global_cfg_path,
        steps_config=steps_cfg_path,
        booth1_csv=_rel(output_dir, booth1_rel) or (output_dir / booth1_rel),
        booth2_csv=_rel(output_dir, booth2_rel) or (output_dir / booth2_rel),
        assignment_csv=_rel(output_dir, assign_rel) or (output_dir / assign_rel),
        manifest_json=_rel(output_dir, manifest_rel) or (output_dir / manifest_rel),
        report_md=_rel(output_dir, report_rel) or (output_dir / report_rel),
        output_dir=output_dir,
    )


# ------------------------------------------------------------------
# Schema helpers
# ------------------------------------------------------------------

def _resolve_column(df: pd.DataFrame, candidates: List[str]) -> str:
    """Resolve a column name from a list of synonyms (case- and punctuation-insensitive)."""
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        c = cand.lower()
        if c in cols_lower:
            return cols_lower[c]
    norm = {"".join(ch for ch in c.lower() if ch.isalnum()): c for c in df.columns}
    for cand in candidates:
        key = "".join(ch for ch in cand.lower() if ch.isalnum())
        if key in norm:
            return norm[key]
    raise KeyError(f"None of {candidates} found in columns {list(df.columns)}")


def _schema_from_global(global_cfg: dict) -> dict:
    ds = global_cfg.get("data_schema", {})
    demands = ds.get("demands", {})
    d_job = demands.get("job_id_synonyms", ["job_id", "job", "id", "jobid"])
    token_type_syn = ["token_type"]
    token_index_syn = ["token_index"]

    return {
        "job_id": d_job,
        "token_type": token_type_syn,
        "token_index": token_index_syn,
    }


# ------------------------------------------------------------------
# Core data structures
# ------------------------------------------------------------------

@dataclass
class JobRecord:
    feeder_job_index: int
    feeder_token_index: int
    job_id: str
    row: Dict[str, Any]  # original row data


@dataclass
class SplitMetrics:
    num_jobs: int
    num_jobs_booth1: int
    num_jobs_booth2: int
    cross: int
    f3_split: float
    rho: float


# ------------------------------------------------------------------
# Splitting logic (pure f3; no C1/C2 gating)
# ------------------------------------------------------------------

def _build_jobs_from_seed(seed_df: pd.DataFrame, schema: dict) -> Tuple[List[JobRecord], str, str]:
    """Extract JOB tokens from seed_df and return list of JobRecord plus column names.

    The returned columns are (job_id_col, token_index_col).
    """
    token_type_col = _resolve_column(seed_df, schema["token_type"])
    token_index_col = _resolve_column(seed_df, schema["token_index"])
    job_col = _resolve_column(seed_df, schema["job_id"])

    jobs_df = seed_df[seed_df[token_type_col].astype(str).str.upper() == "JOB"].copy()
    jobs_df = jobs_df.sort_values(by=[token_index_col]).reset_index(drop=True)

    jobs: List[JobRecord] = []
    for i, (_, r) in enumerate(jobs_df.iterrows(), start=1):
        tok_idx = int(r[token_index_col])
        jid = str(r[job_col])
        jobs.append(
            JobRecord(
                feeder_job_index=i,
                feeder_token_index=tok_idx,
                job_id=jid,
                row=r.to_dict(),
            )
        )

    return jobs, job_col, token_index_col


def _compute_f3(labels: List[int], rho: float) -> Tuple[int, int, int, float]:
    """Compute (N1, N2, cross, f3) for a full label sequence."""
    n1 = sum(1 for b in labels if b == 1)
    n2 = sum(1 for b in labels if b == 2)
    cross = 0
    last_b: Optional[int] = None
    for b in labels:
        if last_b is not None and b != last_b:
            cross += 1
        last_b = b
    f3 = abs(n1 - n2) + rho * cross
    return n1, n2, cross, f3


def _split_jobs_greedy(jobs: List[JobRecord], rho: float) -> Tuple[List[int], SplitMetrics]:
    """Greedy f3-based booth assignment.

    At each feeder position k, assign job k to booth 1 or 2 so as to
    minimize the incremental f3 = |N1 - N2| + rho * cross, where

        N1, N2  = job counts in booths 1 and 2
        cross   = number of PB1↔PB2 switches along feeder order.

    No C1/C2 feasibility checks are performed here; SDSU and downstream
    steps own paint and adjacency feasibility.
    """
    n = len(jobs)
    labels: List[int] = [0] * n

    n1 = 0
    n2 = 0
    cross = 0

    for k in range(n):
        # For the first job, there is no cross term; tie-break to booth 1.
        best_b = None
        best_cost = float("inf")

        for b in (1, 2):
            if b == 1:
                n1p = n1 + 1
                n2p = n2
            else:
                n1p = n1
                n2p = n2 + 1

            cross_p = cross
            if k > 0 and labels[k - 1] != 0 and labels[k - 1] != b:
                cross_p = cross + 1

            cost = abs(n1p - n2p) + rho * cross_p

            if cost < best_cost or (cost == best_cost and (best_b is None or b < best_b)):
                best_cost = cost
                best_b = b

        assert best_b in (1, 2)
        labels[k] = best_b

        # Commit updates
        if best_b == 1:
            n1 += 1
        else:
            n2 += 1
        if k > 0 and labels[k - 1] != best_b:
            cross += 1

    N1, N2, cross_final, f3 = _compute_f3(labels, rho)
    metrics = SplitMetrics(
        num_jobs=n,
        num_jobs_booth1=N1,
        num_jobs_booth2=N2,
        cross=cross_final,
        f3_split=f3,
        rho=rho,
    )
    return labels, metrics


# ------------------------------------------------------------------
# Step-03 entry point
# ------------------------------------------------------------------

def run_step03(global_cfg_path: Path, steps_cfg_path: Path) -> None:
    t0 = time.time()

    global_cfg = read_toml(global_cfg_path)
    steps_cfg = read_toml(steps_cfg_path)
    schema = _schema_from_global(global_cfg)
    paths = resolve_paths(global_cfg_path, steps_cfg_path)

    ensure_parent(paths.booth1_csv)
    ensure_parent(paths.booth2_csv)
    ensure_parent(paths.assignment_csv)
    ensure_parent(paths.manifest_json)
    ensure_parent(paths.report_md)

    # Parameters
    rho_cfg = global_cfg.get("rho", {})
    rho = float(rho_cfg.get("scale", 1.0))

    step03_cfg = steps_cfg.get("step03", {})
    reshuffle_L_max = int(step03_cfg.get("reshuffle_L_max", 0))
    enforce_no_tokens = bool(step03_cfg.get("enforce_no_tokens", True))

    # Load seed and extract JOBs
    seed_df = pd.read_csv(paths.seed_csv)
    jobs, job_col, token_index_col = _build_jobs_from_seed(seed_df, schema)

    # Greedy f3 split
    labels, metrics = _split_jobs_greedy(jobs, rho)

    # Build booth DataFrames and assignment table
    rows_b1: List[Dict[str, Any]] = []
    rows_b2: List[Dict[str, Any]] = []
    assignments_rows: List[Dict[str, Any]] = []

    seq_idx = {1: 0, 2: 0}

    for job, b in zip(jobs, labels):
        seq_idx[b] += 1
        base_row = dict(job.row)
        base_row["booth_id"] = b
        base_row["booth_seq_index"] = seq_idx[b]
        base_row["feeder_job_index"] = job.feeder_job_index
        base_row["feeder_token_index"] = job.feeder_token_index

        if b == 1:
            rows_b1.append(base_row)
        else:
            rows_b2.append(base_row)

        assignments_rows.append(
            {
                "job_id": job.job_id,
                "feeder_job_index": job.feeder_job_index,
                "feeder_token_index": job.feeder_token_index,
                "booth_id": b,
                "booth_seq_index": seq_idx[b],
            }
        )

    booth1_df = pd.DataFrame(rows_b1)
    booth2_df = pd.DataFrame(rows_b2)
    assign_df = pd.DataFrame(assignments_rows)

    # Safety: this step must not emit tokens; if present, they should be identical
    # to STEP-02 rows and are retained only for traceability.
    if enforce_no_tokens:
        for df in (booth1_df, booth2_df):
            if "token_type" in df.columns:
                # We require that token_type is either missing or equal to "JOB" for all rows.
                bad_mask = df["token_type"].astype(str).str.upper() != "JOB"
                if bad_mask.any():
                    raise RuntimeError(
                        "STEP-03: enforce_no_tokens=true but token_type has non-JOB values "
                        "in booth outputs."
                    )

    booth1_df.to_csv(paths.booth1_csv, index=False)
    booth2_df.to_csv(paths.booth2_csv, index=False)
    assign_df.to_csv(paths.assignment_csv, index=False)

    # I/O hashes
    inputs_hash = {
        "global_config.toml": sha256_file(paths.global_config),
        "steps_config.toml": sha256_file(paths.steps_config),
        "seed_stream": sha256_file(paths.seed_csv),
    }
    outputs_hash = {
        "step03_booth1.csv": sha256_file(paths.booth1_csv),
        "step03_booth2.csv": sha256_file(paths.booth2_csv),
        "step03_assignment.csv": sha256_file(paths.assignment_csv),
    }

    settings_effective = {
        "output_dir": str(paths.output_dir),
        "seed_csv": str(paths.seed_csv),
        "booth1_csv": str(paths.booth1_csv),
        "booth2_csv": str(paths.booth2_csv),
        "assignment_csv": str(paths.assignment_csv),
        "rho": rho,
        "reshuffle_L_max": reshuffle_L_max,
        "schema": {
            "job_id_col": job_col,
            "token_index_col": token_index_col,
        },
        "enforce_no_tokens": enforce_no_tokens,
        "algorithm": "greedy_f3_split_no_C1C2",
    }

    diagnostics = {
        "num_jobs": metrics.num_jobs,
        "num_jobs_booth1": metrics.num_jobs_booth1,
        "num_jobs_booth2": metrics.num_jobs_booth2,
        "cross": metrics.cross,
        "f3_split": metrics.f3_split,
    }

    # Manifest
    manifest = {
        "step": "step03",
        "purpose": "Split global feeder JOB sequence into two booth job streams for SDSU/MILP/NSGA-II.",
        "what": "Pure f3-based greedy booth assignment (job counts + rho * cross) with no C1/C2 gating.",
        "why": "Provide balanced per-booth job sets and traceable mapping from feeder to booths.",
        "config_files": {
            "global_config.toml": str(paths.global_config),
            "steps_config.toml": str(paths.steps_config),
        },
        "io_hashes": {
            "inputs": inputs_hash,
            "outputs": outputs_hash,
        },
        "settings_effective": settings_effective,
        "metrics": {
            "num_jobs": metrics.num_jobs,
            "num_jobs_booth1": metrics.num_jobs_booth1,
            "num_jobs_booth2": metrics.num_jobs_booth2,
            "cross": metrics.cross,
            "f3_split": metrics.f3_split,
        },
        "diagnostics": diagnostics,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "version": "step03_split_v2.0_greedy_f3",
        "runtime_sec": time.time() - t0,
    }

    ensure_parent(paths.manifest_json)
    with paths.manifest_json.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Markdown report
    lines: List[str] = []
    lines.append("# STEP-03 Report — Booth Assignment (Greedy f3 Split)")
    lines.append("")
    lines.append("## Purpose / What / Why")
    lines.append("")
    lines.append("- **Purpose**: split the global feeder JOB sequence into two booth job streams for downstream SDSU/MILP/NSGA-II.")
    lines.append("- **What**: greedy assignment of each JOB to booth 1 or 2, minimizing |N1 − N2| + rho * cross along the feeder order.")
    lines.append("- **Why**: create balanced per-booth workloads and a reproducible mapping from feeder tokens to booth jobs.")
    lines.append("")
    lines.append("## Effective Settings")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(settings_effective, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    for k, v in diagnostics.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Inputs (SHA-256)")
    lines.append("")
    for k, v in inputs_hash.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Outputs (SHA-256)")
    lines.append("")
    for k, v in outputs_hash.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append(f"_Runtime: {manifest['runtime_sec']:.3f} s_")
    lines.append("")

    with paths.report_md.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(
        f"STEP-03 completed: N1={metrics.num_jobs_booth1}, "
        f"N2={metrics.num_jobs_booth2}, cross={metrics.cross}, "
        f"f3={metrics.f3_split:.3f}"
    )


def main() -> None:
    if len(sys.argv) >= 3:
        g = Path(sys.argv[1])
        s = Path(sys.argv[2])
    else:
        g = Path("global_config.toml")
        s = Path("steps_config.toml")
    run_step03(g.resolve(), s.resolve())


if __name__ == "__main__":
    main()
