#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP-09 — Aggregation & Final Report (analysis only)

This step does **no new scheduling**. It aggregates:
  - NSGA-II objective values (f1, f2, f3) from Step-07 Pareto CSVs (seeded & unseeded),
  - DES operational metrics and f4 from Step-08,
  - Baseline/diagnostic metrics from manifests of Steps 02–06,

into:
  1) step09_metrics.csv              — row-per-solution metrics and diagnostics
  2) step09_experiment_matrix.csv    — optional filtered (Pareto-only) subset for analysis
  3) manifests/step09_manifest.json  — step-gated manifest with SHA-256 hashes
  4) reports/step09_final_report.md  — human-readable summary

The only coupling is via:
  - global_config.toml
  - steps_config.toml
  - CSVs and manifests produced by Steps 02–08.

No sequences are altered; no adjacency or token logic is re-applied here.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from utilities.config_utils import load_global_config, load_steps_config, resolve_paths
from utilities.io_utils import (
    ensure_parent_dir,
    sha256_of_file,
    write_json,
    write_text,
)


# ---------------------------------------------------------------------------
# Dataclasses for paths & settings
# ---------------------------------------------------------------------------


@dataclass
class Step09Paths:
    # roots
    output_dir: Path
    manifests_dir: Path
    reports_dir: Path

    # primary inputs
    step08_metrics: Path
    step07_pareto_unseeded: Path
    step07_pareto_seeded: Optional[Path]

    # manifests used as diagnostic inputs
    step02_manifest: Optional[Path]
    step03_manifest: Optional[Path]
    step04_manifest: Optional[Path]
    step05_manifest: Optional[Path]
    step06_manifest: Optional[Path]
    step07_manifest: Optional[Path]
    step07_seeded_manifest: Optional[Path]
    step08_manifest: Optional[Path]

    # outputs
    metrics_csv: Path
    experiment_matrix_csv: Path
    manifest_json: Path
    report_md: Path

    # config files
    global_config_path: Path
    steps_config_path: Path


@dataclass
class Step09Settings:
    window_W_frac: float
    report_pareto_only: bool


# ---------------------------------------------------------------------------
# Path & settings construction
# ---------------------------------------------------------------------------


def _with_suffix(path: Path, suffix: str) -> Path:
    """
    Mirror STEP-07 seeded runner naming:
      foo.csv + suffix="seeded" → foo_seeded.csv
    """
    name = path.name
    if "." in name:
        stem, ext = name.rsplit(".", 1)
        new_name = f"{stem}_{suffix}.{ext}"
    else:
        new_name = f"{name}_{suffix}"
    return path.with_name(new_name)


def build_paths(
    global_cfg: Dict[str, Any],
    steps_cfg: Dict[str, Any],
    global_cfg_path: Path,
    steps_cfg_path: Path,
) -> Step09Paths:
    roots = resolve_paths(global_cfg)
    output_dir: Path = roots["output_dir"]
    manifests_dir: Path = roots["manifests_dir"]
    reports_dir: Path = roots["reports_dir"]

    step02 = steps_cfg.get("step02", {})
    step03 = steps_cfg.get("step03", {})
    step04 = steps_cfg.get("step04", {})
    step05 = steps_cfg.get("step05", {})
    step06 = steps_cfg.get("step06", {})
    step07 = steps_cfg.get("step07", {})
    step08 = steps_cfg.get("step08", {})
    step09 = steps_cfg.get("step09", {})

    def _out(rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else (output_dir / p)

    # Step-08 DES metrics (primary source for makespan, blocking, f4)
    step08_metrics = _out(step08.get("metrics_relpath", "step08_des_metrics.csv"))

    # Step-07 Pareto CSVs (unseeded & seeded)
    step07_pareto_unseeded = _out(step07.get("pareto_relpath", "step07_pareto.csv"))
    # seeded: mirror step07_nsga2_seeded.py suffix convention
    step07_pareto_seeded = _with_suffix(step07_pareto_unseeded, "seeded")
    if not step07_pareto_seeded.exists():
        step07_pareto_seeded = None

    # Manifests for diagnostics / extra metrics
    def _maybe_manifest(step_cfg: Dict[str, Any], default: str) -> Optional[Path]:
        rel = step_cfg.get("manifest_relpath")
        if not rel:
            return None
        return _out(rel)

    step02_manifest = _maybe_manifest(step02, "manifests/step02_manifest.json")
    step03_manifest = _maybe_manifest(step03, "manifests/step03_manifest.json")
    step04_manifest = _maybe_manifest(step04, "manifests/step04_manifest.json")
    step05_manifest = _maybe_manifest(step05, "manifests/step05_manifest.json")
    step06_manifest = _maybe_manifest(step06, "manifests/step06_manifest.json")
    step07_manifest = _maybe_manifest(step07, "manifests/step07_manifest.json")
    step08_manifest = _maybe_manifest(step08, "manifests/step08_manifest.json")

    step07_seeded_manifest: Optional[Path] = None
    if step07_manifest is not None:
        cand = _with_suffix(step07_manifest, "seeded")
        if cand.exists():
            step07_seeded_manifest = cand

    # Outputs
    metrics_csv = _out(step09.get("metrics_relpath", "step09_metrics.csv"))
    experiment_matrix_csv = _out(step09.get("experiment_matrix_relpath", "step09_experiment_matrix.csv"))

    manifest_rel = step09.get("manifest_relpath", "manifests/step09_manifest.json")
    manifest_json = _out(manifest_rel)

    report_rel = step09.get("final_report_relpath", "reports/step09_final_report.md")
    report_md = _out(report_rel)

    return Step09Paths(
        output_dir=output_dir,
        manifests_dir=manifests_dir,
        reports_dir=reports_dir,
        step08_metrics=step08_metrics,
        step07_pareto_unseeded=step07_pareto_unseeded,
        step07_pareto_seeded=step07_pareto_seeded,
        step02_manifest=step02_manifest,
        step03_manifest=step03_manifest,
        step04_manifest=step04_manifest,
        step05_manifest=step05_manifest,
        step06_manifest=step06_manifest,
        step07_manifest=step07_manifest,
        step07_seeded_manifest=step07_seeded_manifest,
        step08_manifest=step08_manifest,
        metrics_csv=metrics_csv,
        experiment_matrix_csv=experiment_matrix_csv,
        manifest_json=manifest_json,
        report_md=report_md,
        global_config_path=global_cfg_path,
        steps_config_path=steps_cfg_path,
    )


def build_settings(global_cfg: Dict[str, Any]) -> Step09Settings:
    obj_cfg = global_cfg.get("objective", {})
    window_W_frac = float(obj_cfg.get("window_W_frac", 0.15))
    report_pareto_only = bool(obj_cfg.get("report_pareto_only", True))
    return Step09Settings(
        window_W_frac=window_W_frac,
        report_pareto_only=report_pareto_only,
    )


# ---------------------------------------------------------------------------
# Helpers: manifest loading & flattening
# ---------------------------------------------------------------------------


def load_manifest_metrics(path: Optional[Path]) -> Dict[str, Any]:
    """
    Load `metrics` dict from a manifest JSON, if present.

    Returns an empty dict if the path is None or does not exist.
    """
    if path is None or not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception:
        return {}
    metrics = manifest.get("metrics") or manifest.get("metrics_summary") or {}
    if not isinstance(metrics, dict):
        return {}
    return metrics


def prefix_keys(d: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    return {f"{prefix}{k}": v for k, v in d.items()}


# ---------------------------------------------------------------------------
# Aggregation logic
# ---------------------------------------------------------------------------


def attach_nsga_objectives(
    metrics_df: pd.DataFrame,
    pareto_path: Path,
    method_name: str,
) -> None:
    """
    Attach f1,f2,f3 from a Pareto CSV to rows in metrics_df where method == method_name.

    This assumes Pareto CSV has at least: solution_id, seed, f1, f2, f3.
    """
    if not pareto_path.exists():
        return

    pareto_df = pd.read_csv(pareto_path)
    required_cols = {"solution_id", "seed", "f1", "f2", "f3"}
    if not required_cols.issubset(set(pareto_df.columns)):
        return

    sub_p = pareto_df[["solution_id", "seed", "f1", "f2", "f3"]].copy()
    mask = metrics_df["method"] == method_name
    if not mask.any():
        return

    left = metrics_df.loc[mask].copy()
    merged = left.merge(
        sub_p,
        how="left",
        on=["solution_id", "seed"],
        suffixes=("", "_nsga"),
    )

    # Overwrite / fill f1,f2,f3 where available
    for col in ("f1", "f2", "f3"):
        if col in merged.columns:
            metrics_df.loc[mask, col] = merged[col].values


def compute_pareto_mask(df: pd.DataFrame, objective_cols: List[str]) -> pd.Series:
    """
    Simple O(n^2) Pareto filter for minimization on objective_cols.

    Rows with NaNs in any objective are treated as dominated (False).
    """
    n = df.shape[0]
    mask = [True] * n
    values = df[objective_cols].to_numpy()

    for i in range(n):
        if not mask[i]:
            continue
        vi = values[i]
        if any(pd.isna(vi)):
            mask[i] = False
            continue
        for j in range(n):
            if i == j:
                continue
            vj = values[j]
            if any(pd.isna(vj)):
                continue
            # vj dominates vi if vj <= vi and at least one <
            if (vj <= vi).all() and (vj < vi).any():
                mask[i] = False
                break

    return pd.Series(mask, index=df.index)


def aggregate_step09(
    global_cfg: Dict[str, Any],
    steps_cfg: Dict[str, Any],
    paths: Step09Paths,
    settings: Step09Settings,
) -> Dict[str, Any]:
    """
    Core aggregation:

    - Load DES metrics (Step-08).
    - Attach NSGA-II objectives from Pareto CSVs (Step-07 unseeded & seeded).
    - Attach baseline diagnostics from manifests (Steps 02–08).
    - Optionally compute a Pareto subset for experiment_matrix.
    """
    t0 = time.time()

    # ------------------------------------------------------------------
    # 1) Load DES metrics (primary table)
    # ------------------------------------------------------------------
    if paths.step08_metrics.exists():
        metrics_df = pd.read_csv(paths.step08_metrics)
    else:
        # Empty frame with standard columns if DES was not run
        metrics_df = pd.DataFrame(
            columns=[
                "method",
                "scenario",
                "solution_id",
                "seed",
                "throughput_jobs",
                "tokens_empty",
                "makespan",
                "pb1_busy_time",
                "pb1_blocked_time",
                "pb1_idle_time",
                "pb2_busy_time",
                "pb2_blocked_time",
                "pb2_idle_time",
                "oven_busy_time",
                "oven_idle_time",
                "pb_block_frac_max",
                "oven_idle_frac",
                "f4",
            ]
        )

    # Standardize basic types (robust to missing columns)
    for col in ("method", "scenario"):
        if col in metrics_df.columns:
            metrics_df[col] = metrics_df[col].astype(str)
    for col in ("solution_id", "seed"):
        if col in metrics_df.columns:
            # Keep NaNs for non-NSGA cases
            metrics_df[col] = pd.to_numeric(metrics_df[col], errors="coerce")

    # Ensure f1–f3 exist as float columns
    for col in ("f1", "f2", "f3"):
        if col not in metrics_df.columns:
            metrics_df[col] = float("nan")

    # ------------------------------------------------------------------
    # 2) Attach NSGA-II f1,f2,f3 for unseeded and seeded runs
    # ------------------------------------------------------------------
    if paths.step07_pareto_unseeded.exists():
        attach_nsga_objectives(
            metrics_df=metrics_df,
            pareto_path=paths.step07_pareto_unseeded,
            method_name="step07_nsga2",
        )

    if paths.step07_pareto_seeded is not None:
        attach_nsga_objectives(
            metrics_df=metrics_df,
            pareto_path=paths.step07_pareto_seeded,
            method_name="step07_nsga2_seeded",
        )

    # ------------------------------------------------------------------
    # 3) Attach baseline / diagnostic metrics from step manifests
    #     These are flattened with prefixes; no new scheduling logic.
    # ------------------------------------------------------------------
    # Toyota GS & feeder metrics
    m02 = prefix_keys(load_manifest_metrics(paths.step02_manifest), "s02_")
    # Booth split / f3
    m03 = prefix_keys(load_manifest_metrics(paths.step03_manifest), "s03_")
    # Per-booth SDSU metrics
    m04 = prefix_keys(load_manifest_metrics(paths.step04_manifest), "s04_")
    # Global merge (SDSU global line)
    m05 = prefix_keys(load_manifest_metrics(paths.step05_manifest), "s05_")
    # MILP metrics (typically includes MILP f1,f2,f3 if defined there)
    m06 = prefix_keys(load_manifest_metrics(paths.step06_manifest), "s06_")
    # NSGA-II run-level diagnostics
    m07 = prefix_keys(load_manifest_metrics(paths.step07_manifest), "s07_")
    m07s = prefix_keys(load_manifest_metrics(paths.step07_seeded_manifest), "s07s_")
    # DES run-level diagnostics
    m08 = prefix_keys(load_manifest_metrics(paths.step08_manifest), "s08_")

    # For rows corresponding to the SDSU baseline (step05), attach 02–05 metrics
    if not metrics_df.empty and "method" in metrics_df.columns:
        mask_sdsu = metrics_df["method"] == "step05"
        if mask_sdsu.any():
            for d in (m02, m03, m04, m05):
                for k, v in d.items():
                    if k not in metrics_df.columns:
                        metrics_df[k] = float("nan")
                    metrics_df.loc[mask_sdsu, k] = v

        # For MILP rows, attach MILP-level metrics (and propagate f1–f3 if present)
        mask_milp = metrics_df["method"] == "step06_milp"
        if mask_milp.any():
            for k, v in m06.items():
                if k not in metrics_df.columns:
                    metrics_df[k] = float("nan")
                metrics_df.loc[mask_milp, k] = v
            # If MILP metrics expose canonical f1–f3, copy them into main columns
            for f in ("f1", "f2", "f3"):
                key = f"s06_{f}"
                if key in m06 and f in metrics_df.columns:
                    metrics_df.loc[mask_milp, f] = m06[key]

        # For NSGA run-level diagnostics (unseeded)
        mask_nsga = metrics_df["method"] == "step07_nsga2"
        if mask_nsga.any():
            for k, v in m07.items():
                if k not in metrics_df.columns:
                    metrics_df[k] = float("nan")
                metrics_df.loc[mask_nsga, k] = v

        # For NSGA seeded run-level diagnostics
        mask_nsga_s = metrics_df["method"] == "step07_nsga2_seeded"
        if mask_nsga_s.any():
            for k, v in m07s.items():
                if k not in metrics_df.columns:
                    metrics_df[k] = float("nan")
                metrics_df.loc[mask_nsga_s, k] = v

        # DES run-level diagnostics (applied to all rows)
        if m08:
            for k, v in m08.items():
                if k not in metrics_df.columns:
                    metrics_df[k] = float("nan")
                metrics_df[k] = v

    # ------------------------------------------------------------------
    # 4) Compute experiment matrix (optional Pareto filtering on f1–f4)
    # ------------------------------------------------------------------
    experiment_df = metrics_df.copy()

    # Only apply Pareto filter if explicitly requested and all f1–f4 present
    if settings.report_pareto_only:
        required = ["f1", "f2", "f3", "f4"]
        if all(c in experiment_df.columns for c in required):
            pareto_mask = compute_pareto_mask(experiment_df, required)
            experiment_df = experiment_df.loc[pareto_mask].reset_index(drop=True)

    # ------------------------------------------------------------------
    # 5) Write outputs
    # ------------------------------------------------------------------
    ensure_parent_dir(paths.metrics_csv)
    ensure_parent_dir(paths.experiment_matrix_csv)

    metrics_df.to_csv(paths.metrics_csv, index=False)
    experiment_df.to_csv(paths.experiment_matrix_csv, index=False)

    runtime = time.time() - t0

    # Build simple diagnostics summary for manifest/report
    by_method = {}
    if "method" in metrics_df.columns:
        for m, grp in metrics_df.groupby("method"):
            by_method[str(m)] = int(grp.shape[0])

    diag = {
        "num_rows_metrics": int(metrics_df.shape[0]),
        "num_rows_experiment_matrix": int(experiment_df.shape[0]),
        "rows_per_method": by_method,
    }

    return {
        "metrics_df": metrics_df,
        "experiment_df": experiment_df,
        "diagnostics": diag,
        "runtime_seconds": runtime,
    }


# ---------------------------------------------------------------------------
# Manifest & report
# ---------------------------------------------------------------------------


def write_manifest_and_report(
    global_cfg: Dict[str, Any],
    steps_cfg: Dict[str, Any],
    paths: Step09Paths,
    settings: Step09Settings,
    diag: Dict[str, Any],
) -> None:
    # Inputs considered for hashing
    inputs: List[Path] = [
        paths.global_config_path,
        paths.steps_config_path,
        paths.step08_metrics,
    ]
    for p in (
        paths.step07_pareto_unseeded,
        paths.step07_pareto_seeded,
        paths.step02_manifest,
        paths.step03_manifest,
        paths.step04_manifest,
        paths.step05_manifest,
        paths.step06_manifest,
        paths.step07_manifest,
        paths.step07_seeded_manifest,
        paths.step08_manifest,
    ):
        if p is not None:
            inputs.append(p)

    outputs: List[Path] = [
        paths.metrics_csv,
        paths.experiment_matrix_csv,
        paths.manifest_json,
        paths.report_md,
    ]

    # Manifest JSON
    ensure_parent_dir(paths.manifest_json)
    manifest: Dict[str, Any] = {
        "step": "step09_aggregate",
        "timestamp": dt.datetime.now().isoformat(),
        "purpose": "Aggregate multi-objective and DES metrics across methods (Toyota/SDSU, MILP, NSGA-II, DES).",
        "what": (
            "Read Step-07 Pareto fronts and Step-08 DES metrics, join them with per-step manifests "
            "from Steps 02–06, and emit a unified metrics table plus an optional Pareto-filtered "
            "experiment matrix for analysis."
        ),
        "why": (
            "Provide a single, reproducible artifact for comparing heuristics (Toyota/SDSU), exact MILP, "
            "and NSGA-II search under a common objective/operational view."
        ),
        "config_files": {
            os.path.relpath(str(paths.global_config_path), str(paths.output_dir)): sha256_of_file(
                paths.global_config_path
            ),
            os.path.relpath(str(paths.steps_config_path), str(paths.output_dir)): sha256_of_file(
                paths.steps_config_path
            ),
        },
        "inputs": {
            os.path.relpath(str(p), str(paths.output_dir)): sha256_of_file(p)
            for p in inputs
            if p.exists()
        },
        "outputs": {
            os.path.relpath(str(p), str(paths.output_dir)): sha256_of_file(p)
            for p in outputs
            if p.exists()
        },
        "settings": {
            "window_W_frac": settings.window_W_frac,
            "report_pareto_only": settings.report_pareto_only,
        },
        "diagnostics": diag,
    }

    write_json(paths.manifest_json, manifest)

    # Markdown report
    lines: List[str] = []
    lines.append("# STEP-09 Report — Aggregation & Final Metrics")
    lines.append("")
    lines.append("## Purpose / What / Why")
    lines.append("")
    lines.append("- **Purpose**: aggregate upstream objective values and DES metrics into a single analyzable table.")
    lines.append(
        "- **What**: join Step-07 Pareto CSVs (f1–f3) with Step-08 DES results (makespan, blocking, f4) "
        "and annotate with diagnostics from Steps 02–06 manifests."
    )
    lines.append(
        "- **Why**: enable cross-method comparison (Toyota/SDSU baseline vs MILP vs NSGA-II) "
        "for the MO-FFS case study without re-running schedulers or DES."
    )
    lines.append("")
    lines.append("## Settings")
    lines.append("")
    lines.append(f"- `objective.window_W_frac` = {settings.window_W_frac}")
    lines.append(f"- `objective.report_pareto_only` = {settings.report_pareto_only}")
    lines.append("")
    lines.append("## Diagnostics")
    lines.append("")
    lines.append(f"- Rows in `step09_metrics.csv`: {diag.get('num_rows_metrics', 0)}")
    lines.append(
        f"- Rows in `step09_experiment_matrix.csv`: {diag.get('num_rows_experiment_matrix', 0)}"
    )
    rows_per_method = diag.get("rows_per_method", {})
    if rows_per_method:
        lines.append("")
        lines.append("### Rows per method (from DES metrics)")
        for m, c in sorted(rows_per_method.items()):
            lines.append(f"- {m}: {c}")
    lines.append("")
    lines.append("## I/O Locations")
    lines.append("")
    lines.append(f"- Metrics CSV: `{paths.metrics_csv}`")
    lines.append(f"- Experiment matrix CSV: `{paths.experiment_matrix_csv}`")
    lines.append(f"- Manifest: `{paths.manifest_json}`")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- No new scheduling or DES simulation is performed in this step; all sequences and "
        "operational behavior are taken from Steps 05–08."
    )
    lines.append(
        "- NSGA-II objective values (f1–f3) are taken from the stored Pareto fronts; "
        "MILP and SDSU baseline metrics are attached from their manifests where available."
    )

    write_text(paths.report_md, "\n".join(lines))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="STEP-09 — Aggregation & Final Report")
    parser.add_argument(
        "global_config",
        type=Path,
        nargs="?",
        default=Path("global_config.toml"),
        help="Path to global_config.toml (default: ./global_config.toml)",
    )
    parser.add_argument(
        "steps_config",
        type=Path,
        nargs="?",
        default=Path("steps_config.toml"),
        help="Path to steps_config.toml (default: ./steps_config.toml)",
    )
    args = parser.parse_args(argv)

    global_cfg_path: Path = args.global_config
    steps_cfg_path: Path = args.steps_config

    global_cfg = load_global_config(global_cfg_path)
    steps_cfg = load_steps_config(steps_cfg_path)

    paths = build_paths(global_cfg, steps_cfg, global_cfg_path, steps_cfg_path)
    settings = build_settings(global_cfg)

    agg = aggregate_step09(global_cfg, steps_cfg, paths, settings)
    write_manifest_and_report(
        global_cfg=global_cfg,
        steps_cfg=steps_cfg,
        paths=paths,
        settings=settings,
        diag=agg["diagnostics"],
    )


if __name__ == "__main__":
    main()
