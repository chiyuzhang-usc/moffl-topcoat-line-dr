#!/usr/bin/env python3
"""
Step-09 — Integrated Research Analysis & Observational Studies

This script implements the Step-09 gate as described in STEP09_NOTES.md
and the MOFFS master/protocol notes.

Key properties:

- It **does not alter any schedule**. It only reads outputs from prior steps.
- Inputs are strictly the documented artifacts from Steps 02–08:
    ./output/step02_paint_seed.csv
    ./output/step03_booth1.csv
    ./output/step03_booth2.csv
    ./output/step04_booth1_sdsu.csv
    ./output/step04_booth2_sdsu.csv
    ./output/step05_global_line.csv
    ./output/step06_*opt*.csv          (optional; MILP optima)
    ./output/step07_pareto.csv         (NSGA-II fronts)
    ./output/step08_des_metrics.csv    (optional; f4 & DES observables)
- It uses ./dataset/adjacency_rules.csv as the C2 source.
- It performs:
    * C2 adjacency validation on the sequences (Steps 02–05).
    * Aggregation of metrics across methods (GS/SDSU, MILP, NSGA-II, DES).
    * Baseline-relative comparisons (Δf1, Δf2, Δf3).
    * Optional NSGA-II front summaries.
    * Basic plots in objective space.
    * A step manifest and a final markdown report.

There is intentionally **no dependency on results/metrics.csv** or
any synthetic "step09_input" file; everything is derived from the
documented prior-step outputs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import tomllib  # Python 3.11+
except Exception:  # pragma: no cover
    tomllib = None  # type: ignore

try:  # Optional but enables Wilcoxon tests if present
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover
    scipy_stats = None

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None  # plotting is optional


TOKEN_THRESHOLDS = {
    "empty_per_100": 10.0,
    "flush_per_100": 5.0,
}

DEFAULT_EXPERIMENT_COLUMNS = [
    "exp_id",
    "label",
    "notes",
    "window.mode",
    "window.jobs",
    "window.ratio",
    "spacing.alpha_run_cap",
    "spacing.max_run_abs",
    "spacing.picker",
    "adjacency.scope_escalation",
    "adjacency.fifo_lookahead",
    "load.rho",
    "tokens.flush_units_ratio",
    "tokens.empty_units_prepaint",
    "tokens.empty_units_postpaint_ratio",
    "step05.postpaint_empty.enabled",
    "step04.dp_threshold",
    "step04.max_passes",
    "nsga2.population",
    "nsga2.generations",
    "nsga2.pc",
    "nsga2.pm",
    "nsga2.tournament_k",
    "nsga2.sdsu_kernel",
    "des.buffer_beta",
    "des.timing.mode",
    "des.f4_alpha",
]

CONTEXT_JSON_NAME = "step09_context_metrics.json"
FRONT_SUMMARY_FILENAME = "step09_front_summary.csv"
EPSILON = 1e-9
HV_SAMPLES = 20000


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


# ---------------------------------------------------------------------------
# Dataclasses and small utilities
# ---------------------------------------------------------------------------


@dataclass
class AdjacencyDiagnostics:
    has_violation: bool
    first_violation_index: Optional[int]
    first_violation_from_part: Optional[str]
    first_violation_to_part: Optional[str]
    total_violations: int


@dataclass
class DatasetContext:
    dataset_dir: Path
    adjacency_rules: pd.DataFrame
    adjacency_rules_path: Path


@dataclass
class ContextBundle:
    token_stats: Dict[str, float]
    run_stats: Dict[str, float]
    changeover_stats: Dict[str, float]
    anomalies: List[str]
    adjacency_summary: pd.DataFrame


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config_if_present(config_path: Path) -> Dict:
    """
    Load model_config.toml if present; otherwise return {}.

    Only the [paths] section is used here, and even that is optional.
    """
    if not config_path.is_file():
        return {}
    if tomllib is None:
        raise RuntimeError(
            f"{config_path} exists but tomllib is unavailable; "
            "use Python 3.11+ or install a TOML parser and adapt this loader."
        )
    with config_path.open("rb") as f:
        return tomllib.load(f)


def load_steps_config_if_present(steps_path: Path) -> Dict:
    """Load steps_config.toml if present; otherwise {}."""

    if not steps_path.is_file():
        return {}
    if tomllib is None:
        raise RuntimeError(
            f"{steps_path} exists but tomllib is unavailable; install Python 3.11+ or a TOML parser."
        )
    with steps_path.open("rb") as f:
        return tomllib.load(f)


def resolve_base_paths(
    cfg: Dict,
    dataset_dir_arg: Optional[str],
    output_dir_arg: Optional[str],
) -> Tuple[Path, Path]:
    """
    Decide dataset_dir and output_dir using:
        1) CLI overrides if given,
        2) [paths] section in model_config.toml if present,
        3) Defaults: ./dataset and ./output.
    """
    paths_cfg = cfg.get("paths", {}) if isinstance(cfg, dict) else {}
    if dataset_dir_arg is not None:
        dataset_dir = Path(dataset_dir_arg)
    else:
        dataset_dir = Path(paths_cfg.get("dataset_dir", "./dataset"))

    if output_dir_arg is not None:
        output_dir = Path(output_dir_arg)
    else:
        output_dir = Path(paths_cfg.get("output_dir", "./output"))

    return dataset_dir.resolve(), output_dir.resolve()


# ---------------------------------------------------------------------------
# Dataset loading and C2 helpers
# ---------------------------------------------------------------------------


def load_dataset(dataset_dir: Path) -> DatasetContext:
    """
    Load adjacency_rules.csv from dataset_dir.

    This is the only dataset file Step-09 needs directly; all other
    semantics (timings, SDST, etc.) are already enforced upstream.
    """
    adj_path = dataset_dir / "adjacency_rules.csv"
    if not adj_path.is_file():
        raise FileNotFoundError(f"Missing adjacency_rules.csv at {adj_path}")

    adjacency_rules = pd.read_csv(adj_path)

    expected_cols = {
        "from_part_id",
        "to_part_id",
        "scope",
        "relation",
    }
    missing = expected_cols - set(adjacency_rules.columns)
    if missing:
        raise ValueError(
            f"adjacency_rules.csv missing expected columns: {sorted(missing)}"
        )

    return DatasetContext(
        dataset_dir=dataset_dir,
        adjacency_rules=adjacency_rules,
        adjacency_rules_path=adj_path,
    )


def _build_forbidden_pairs(adjacency_rules: pd.DataFrame) -> Dict[Tuple[str, str], str]:
    """
    Build (from_part_id, to_part_id) -> scope for any rule whose
    relation contains 'forbid' (case-insensitive).
    """
    forbid: Dict[Tuple[str, str], str] = {}
    for _, row in adjacency_rules.iterrows():
        relation = str(row["relation"]).lower()
        if "forbid" not in relation:
            continue
        from_pid = str(row["from_part_id"])
        to_pid = str(row["to_part_id"])
        scope = str(row["scope"])
        forbid[(from_pid, to_pid)] = scope
    return forbid


def check_global_adjacency(
    seq_df: pd.DataFrame,
    adjacency_rules: pd.DataFrame,
    scope_filter: Optional[Iterable[str]] = None,
    order_column_candidates: Iterable[str] = ("position", "pos", "seq_index"),
) -> AdjacencyDiagnostics:
    """
    Check C2 adjacency on a JOB-only sequence.

    seq_df is expected to have at least:
        part_id
    and optionally any of:
        position, pos, seq_index  (used to sort the stream)

    scope_filter allows restricting to certain scopes, e.g.
        {'global', 'across_skids'}.
    """
    if "part_id" not in seq_df.columns:
        return AdjacencyDiagnostics(
            has_violation=False,
            first_violation_index=None,
            first_violation_from_part=None,
            first_violation_to_part=None,
            total_violations=0,
        )

    order_col: Optional[str] = None
    for c in order_column_candidates:
        if c in seq_df.columns:
            order_col = c
            break

    if order_col is not None:
        seq = seq_df.sort_values(order_col).reset_index(drop=True)
    else:
        seq = seq_df.reset_index(drop=True)

    rules = adjacency_rules
    if scope_filter is not None:
        scope_set = {str(s) for s in scope_filter}
        rules = rules[rules["scope"].astype(str).isin(scope_set)]

    forbid_pairs = _build_forbidden_pairs(rules)

    part_ids = seq["part_id"].astype(str).tolist()
    total = 0
    first_idx: Optional[int] = None
    first_from: Optional[str] = None
    first_to: Optional[str] = None

    for idx in range(len(part_ids) - 1):
        key = (part_ids[idx], part_ids[idx + 1])
        if key in forbid_pairs:
            total += 1
            if first_idx is None:
                first_idx = idx
                first_from, first_to = key

    return AdjacencyDiagnostics(
        has_violation=(total > 0),
        first_violation_index=first_idx,
        first_violation_from_part=first_from,
        first_violation_to_part=first_to,
        total_violations=total,
    )


def compute_run_length_stats(
    seq_df: pd.DataFrame,
    key_column: str = "topcoat_code",
    order_column_candidates: Iterable[str] = ("position", "pos", "seq_index"),
) -> Dict[str, float]:
    """
    Compute basic run-length stats for a categorical key along the sequence.

    Returns longest_run, mean_run, std_run; NaNs if key_column missing.
    """
    import numpy as np

    if key_column not in seq_df.columns:
        return {"longest_run": math.nan, "mean_run": math.nan, "std_run": math.nan}

    order_col: Optional[str] = None
    for c in order_column_candidates:
        if c in seq_df.columns:
            order_col = c
            break

    if order_col is not None:
        seq = seq_df.sort_values(order_col).reset_index(drop=True)
    else:
        seq = seq_df.reset_index(drop=True)

    values = seq[key_column].astype(str).tolist()
    if not values:
        return {"longest_run": math.nan, "mean_run": math.nan, "std_run": math.nan}

    run_lengths: List[int] = []
    current_val = values[0]
    current_len = 1

    for v in values[1:]:
        if v == current_val:
            current_len += 1
        else:
            run_lengths.append(current_len)
            current_val = v
            current_len = 1
    run_lengths.append(current_len)

    longest_run = max(run_lengths)
    arr = np.asarray(run_lengths, dtype=float)
    mean_run = float(arr.mean())
    std_run = float(arr.std(ddof=0))

    return {"longest_run": float(longest_run), "mean_run": mean_run, "std_run": std_run}


# ---------------------------------------------------------------------------
# Sequence loading and adjacency validation for Steps 02–05
# ---------------------------------------------------------------------------


def load_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if not path.is_file():
        return None
    return pd.read_csv(path)


def load_step_sequences(output_dir: Path) -> Dict[str, Optional[pd.DataFrame]]:
    """Preload all step02–05 streams for reuse downstream."""

    return {
        "step02": load_csv_if_exists(output_dir / "step02_paint_seed.csv"),
        "step03_b1": load_csv_if_exists(output_dir / "step03_booth1.csv"),
        "step03_b2": load_csv_if_exists(output_dir / "step03_booth2.csv"),
        "step04_b1": load_csv_if_exists(output_dir / "step04_booth1_sdsu.csv"),
        "step04_b2": load_csv_if_exists(output_dir / "step04_booth2_sdsu.csv"),
        "step05_global": load_csv_if_exists(output_dir / "step05_global_line.csv"),
    }

def collect_adjacency_for_steps_02_to_05(
    dataset: DatasetContext,
    output_dir: Path,
    seq_cache: Optional[Dict[str, Optional[pd.DataFrame]]] = None,
) -> pd.DataFrame:
    """
    Collect C2 adjacency diagnostics and basic run-length stats for
    Step-02..05 sequences. This is *analysis only* – no exceptions raised.

    Returns a small table with one row per (step_id, stream_id).
    """
    rows: List[Dict[str, object]] = []
    scope_filter = {"global", "across_skids"}

    def _add_row(step_id: str, stream_id: str, df: Optional[pd.DataFrame]) -> None:
        import math

        if df is None or df.empty:
            rows.append(
                {
                    "step_id": step_id,
                    "stream_id": stream_id,
                    "has_data": False,
                    "c2_has_violation": False,
                    "c2_total_violations": 0,
                    "c2_first_violation_index": math.nan,
                    "c2_first_violation_from_part": None,
                    "c2_first_violation_to_part": None,
                    "longest_run": math.nan,
                    "mean_run": math.nan,
                    "std_run": math.nan,
                }
            )
            return

        if "part_id" not in df.columns:
            # Cannot check adjacency; record as missing.
            rows.append(
                {
                    "step_id": step_id,
                    "stream_id": stream_id,
                    "has_data": True,
                    "c2_has_violation": False,
                    "c2_total_violations": -1,
                    "c2_first_violation_index": math.nan,
                    "c2_first_violation_from_part": None,
                    "c2_first_violation_to_part": None,
                    "longest_run": math.nan,
                    "mean_run": math.nan,
                    "std_run": math.nan,
                }
            )
            return

        # Step-02: tokenized stream (START/JOB/EMPTY/FLUSH). Do NOT strip tokens.
        # Step-03/04/05: usually JOB-only already. Using same logic is fine.
        diag = check_global_adjacency(
            df,
            dataset.adjacency_rules,
            scope_filter=scope_filter,
        )
        run_stats = compute_run_length_stats(df, key_column="topcoat_code")

        rows.append(
            {
                "step_id": step_id,
                "stream_id": stream_id,
                "has_data": True,
                "c2_has_violation": diag.has_violation,
                "c2_total_violations": diag.total_violations,
                "c2_first_violation_index": diag.first_violation_index,
                "c2_first_violation_from_part": diag.first_violation_from_part,
                "c2_first_violation_to_part": diag.first_violation_to_part,
                "longest_run": run_stats["longest_run"],
                "mean_run": run_stats["mean_run"],
                "std_run": run_stats["std_run"],
            }
        )

    cache = seq_cache or load_step_sequences(output_dir)
    s02 = cache.get("step02")
    s03_b1 = cache.get("step03_b1")
    s03_b2 = cache.get("step03_b2")
    s04_b1 = cache.get("step04_b1")
    s04_b2 = cache.get("step04_b2")
    s05_g = cache.get("step05_global")

    _add_row("STEP02", "feeder", s02)
    _add_row("STEP03", "booth1", s03_b1)
    _add_row("STEP03", "booth2", s03_b2)
    _add_row("STEP04", "booth1_sdsu", s04_b1)
    _add_row("STEP04", "booth2_sdsu", s04_b2)
    _add_row("STEP05", "global_line", s05_g)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Context metrics helpers (tokens, runs, anomalies)
# ---------------------------------------------------------------------------


def compute_token_stats(seq_df: Optional[pd.DataFrame]) -> Dict[str, float]:
    stats = {
        "jobs": 0.0,
        "token_total": 0.0,
        "empty_tokens": 0.0,
        "flush_tokens": 0.0,
        "token_per_100_jobs": math.nan,
        "empty_per_100": math.nan,
        "flush_per_100": math.nan,
    }
    if seq_df is None or seq_df.empty or "token_type" not in seq_df.columns:
        return stats

    tokens = seq_df["token_type"].astype(str).str.upper()
    stats["jobs"] = float((tokens == "JOB").sum())
    stats["token_total"] = float(len(tokens))
    stats["empty_tokens"] = float((tokens == "EMPTY").sum())
    stats["flush_tokens"] = float((tokens == "FLUSH").sum())

    denom = max(stats["jobs"], 1.0)
    stats["token_per_100_jobs"] = (stats["token_total"] / denom) * 100.0
    stats["empty_per_100"] = (stats["empty_tokens"] / denom) * 100.0
    stats["flush_per_100"] = (stats["flush_tokens"] / denom) * 100.0
    return stats


def compute_changeover_stats(seq_df: Optional[pd.DataFrame]) -> Dict[str, float]:
    stats = {
        "transitions": 0,
        "purge_events": 0,
        "direct_events": 0,
        "purge_ratio": math.nan,
        "avg_setup_cost": math.nan,
        "max_setup_cost": math.nan,
    }
    if seq_df is None or seq_df.empty or "setup_cost" not in seq_df.columns:
        return stats

    setup = seq_df["setup_cost"].fillna(0.0).astype(float)
    stats["transitions"] = max(int(len(setup) - 1), 0)
    stats["purge_events"] = int((setup > 0).sum())
    stats["direct_events"] = max(stats["transitions"] - stats["purge_events"], 0)
    stats["purge_ratio"] = (
        stats["purge_events"] / stats["transitions"]
        if stats["transitions"] > 0
        else math.nan
    )
    stats["avg_setup_cost"] = float(setup.mean())
    stats["max_setup_cost"] = float(setup.max())
    return stats


def detect_anomalies(
    token_stats: Dict[str, float],
    run_stats: Dict[str, float],
    adjacency_summary: pd.DataFrame,
    spacing_cfg: Dict[str, Any],
    tokens_cfg: Dict[str, Any],
) -> List[str]:
    reasons: List[str] = []

    if (
        not math.isnan(token_stats.get("empty_per_100", math.nan))
        and token_stats["empty_per_100"] > TOKEN_THRESHOLDS["empty_per_100"]
    ):
        reasons.append(
            f"EMPTY tokens exceed {TOKEN_THRESHOLDS['empty_per_100']} per 100 jobs"
        )

    if (
        not math.isnan(token_stats.get("flush_per_100", math.nan))
        and token_stats["flush_per_100"] > TOKEN_THRESHOLDS["flush_per_100"]
    ):
        reasons.append(
            f"FLUSH tokens exceed {TOKEN_THRESHOLDS['flush_per_100']} per 100 jobs"
        )

    max_run = spacing_cfg.get("max_run_abs")
    if max_run is not None and not math.isnan(run_stats.get("longest_run", math.nan)):
        if run_stats["longest_run"] > float(max_run):
            reasons.append(
                f"Color run cap breached: longest run {run_stats['longest_run']:.1f} > {max_run}"
            )

    flush_ratio = tokens_cfg.get("flush_units_ratio")
    if flush_ratio is not None and float(flush_ratio) < 0.1:
        reasons.append("Flush cost ratio < 0.1 (non-comparable purge scenario)")

    if not adjacency_summary.empty and adjacency_summary["c2_has_violation"].any():
        first = adjacency_summary[adjacency_summary["c2_has_violation"]].iloc[0]
        reasons.append(
            "Adjacency violation detected "
            f"(step {first['step_id']} stream {first['stream_id']})"
        )

    return reasons


def build_context_bundle(
    dataset: DatasetContext,
    output_dir: Path,
    seq_cache: Dict[str, Optional[pd.DataFrame]],
    cfg: Dict[str, Any],
) -> ContextBundle:
    spacing_cfg = cfg.get("spacing", {}) if isinstance(cfg, dict) else {}
    tokens_cfg = cfg.get("tokens", {}) if isinstance(cfg, dict) else {}

    token_stats = compute_token_stats(seq_cache.get("step02"))
    run_stats = compute_run_length_stats(
        seq_cache.get("step05_global") or pd.DataFrame(), key_column="topcoat_name"
    )
    changeover_stats = compute_changeover_stats(seq_cache.get("step05_global"))
    adjacency_summary = collect_adjacency_for_steps_02_to_05(
        dataset, output_dir, seq_cache=seq_cache
    )
    anomalies = detect_anomalies(
        token_stats=token_stats,
        run_stats=run_stats,
        adjacency_summary=adjacency_summary,
        spacing_cfg=spacing_cfg,
        tokens_cfg=tokens_cfg,
    )

    return ContextBundle(
        token_stats=token_stats,
        run_stats=run_stats,
        changeover_stats=changeover_stats,
        anomalies=anomalies,
        adjacency_summary=adjacency_summary,
    )


def write_context_payload(
    output_dir: Path,
    context: ContextBundle,
    spacing_cfg: Dict[str, Any],
    tokens_cfg: Dict[str, Any],
) -> Path:
    payload = {
        "generated_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "token_stats": context.token_stats,
        "run_stats": context.run_stats,
        "changeover_stats": context.changeover_stats,
        "anomalies": context.anomalies,
        "thresholds": {
            "token": TOKEN_THRESHOLDS,
            "spacing.max_run_abs": spacing_cfg.get("max_run_abs"),
            "tokens.flush_units_ratio": tokens_cfg.get("flush_units_ratio"),
        },
    }

    context_path = output_dir / CONTEXT_JSON_NAME
    context_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return context_path


# ---------------------------------------------------------------------------
# Aggregated metrics table construction (Step-09 core)
# ---------------------------------------------------------------------------


def _resolve_metric_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def build_candidates_from_step05(
    output_dir: Path,
) -> List[Dict[str, object]]:
    """
    Build a single baseline candidate row representing Step-05
    GS→SDSU→Merge schedule.

    Metrics are taken from ./output/manifests/step05_manifest.json
    if present, using common field names (f1_total, f2_norm, f3_norm)
    either at top-level or under a 'metrics' object.

    If the manifest or metrics are missing, f1/f2/f3 are left as NaN.
    """
    import math

    manifest_path = output_dir / "manifests" / "step05_manifest.json"
    row: Dict[str, object] = {
        "exp_id": "E0",
        "method_id": "S05_GS_SDSU",
        "source_step": 5,
        "solution_type": "singleton",
        "solution_id": 0,
        "seed_mode": "none",
        "label": "Step-05 GS→SDSU→Merge",
        "has_des": False,
        "f1": math.nan,
        "f2": math.nan,
        "f3": math.nan,
        "f4": math.nan,
    }

    if not manifest_path.is_file():
        return [row]

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    def _get_metric(mdict: Dict, keys: List[str]) -> Optional[float]:
        for k in keys:
            if k in mdict and mdict[k] is not None:
                try:
                    return float(mdict[k])
                except Exception:
                    continue
        return None

    # Search top-level, then "metrics" sub-dict if present.
    metrics_sources: List[Dict] = [manifest]
    if isinstance(manifest.get("metrics"), dict):
        metrics_sources.append(manifest["metrics"])

    f1_val = None
    f2_val = None
    f3_val = None

    for src in metrics_sources:
        if f1_val is None:
            f1_val = _get_metric(src, ["f1", "f1_total"])
        if f2_val is None:
            f2_val = _get_metric(src, ["f2", "f2_norm"])
        if f3_val is None:
            f3_val = _get_metric(src, ["f3", "f3_norm"])
        if f1_val is not None and f2_val is not None and f3_val is not None:
            break

    if f1_val is not None:
        row["f1"] = f1_val
    if f2_val is not None:
        row["f2"] = f2_val
    if f3_val is not None:
        row["f3"] = f3_val

    return [row]


def build_candidates_from_step06(output_dir: Path) -> List[Dict[str, object]]:
    """
    Build candidate rows from any ./output/step06_*opt*.csv files.

    The expected pattern is that these are small tables with
    per-instance MILP optima and columns including:
        instance_id, f1_total, f2_norm, f3_norm
    plus any tagging fields.

    If a file instead contains per-job sequences (job_id/part_id),
    this function will skip it, since Step-09 does not recompute
    f1/f2/f3 from sequences.
    """
    import math
    rows: List[Dict[str, object]] = []

    for path in sorted(output_dir.glob("step06_*opt*.csv")):
        df = pd.read_csv(path)
        if df.empty:
            continue

        # Heuristic: if it looks like sequence (job_id present), skip
        if "job_id" in df.columns or "part_id" in df.columns:
            continue

        f1_col = _resolve_metric_column(df, ["f1", "f1_total"])
        f2_col = _resolve_metric_column(df, ["f2", "f2_norm"])
        f3_col = _resolve_metric_column(df, ["f3", "f3_norm"])

        if f1_col is None or f2_col is None or f3_col is None:
            continue

        inst_col = None
        for c in ["instance_id", "exp_id", "scenario_id"]:
            if c in df.columns:
                inst_col = c
                break

        for _, r in df.iterrows():
            exp_id = str(r[inst_col]) if inst_col is not None else "MILP"
            rows.append(
                {
                    "exp_id": exp_id,
                    "method_id": "S06_MILP",
                    "source_step": 6,
                    "solution_type": "singleton",
                    "solution_id": int(r.get("solution_id", 0)),
                    "seed_mode": "none",
                    "label": f"Step-06 MILP ({path.name})",
                    "has_des": False,
                    "f1": float(r[f1_col]),
                    "f2": float(r[f2_col]),
                    "f3": float(r[f3_col]),
                    "f4": math.nan,
                }
            )

    return rows


def build_candidates_from_step07(output_dir: Path) -> List[Dict[str, object]]:
    """
    Build candidate rows from ./output/step07_pareto.csv.

    STEP07_NOTES specifies one row per non-dominated solution with:
        f1, f2, f3, tags, and seed provenance.

    We only rely on:
        exp_id (if present; else 'E0'),
        solution_id (if present; else row index),
        f1/f2/f3 (or their canonical variants),
        seed_mode (if present; else 'none').
    """
    import math

    path = output_dir / "step07_pareto.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Required NSGA-II archive not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"step07_pareto.csv is empty at {path}")

    f1_col = _resolve_metric_column(df, ["f1", "f1_total"])
    f2_col = _resolve_metric_column(df, ["f2", "f2_norm"])
    f3_col = _resolve_metric_column(df, ["f3", "f3_norm"])
    if f1_col is None or f2_col is None or f3_col is None:
        raise KeyError(
            "step07_pareto.csv must contain f1/f2/f3 (or f1_total/f2_norm/f3_norm)."
        )

    exp_col = None
    for c in ["exp_id", "instance_id", "scenario_id"]:
        if c in df.columns:
            exp_col = c
            break

    seed_col = None
    for c in ["seed_mode", "seed_origin", "init_mode"]:
        if c in df.columns:
            seed_col = c
            break

    method_col = None
    for c in ["method_id", "method"]:
        if c in df.columns:
            method_col = c
            break

    sol_col = None
    for c in ["solution_id", "id", "front_index"]:
        if c in df.columns:
            sol_col = c
            break

    rows: List[Dict[str, object]] = []

    for idx, r in df.iterrows():
        exp_id = str(r[exp_col]) if exp_col is not None else "E0"
        seed_mode = str(r[seed_col]) if seed_col is not None else "none"
        method_id = str(r[method_col]) if method_col is not None else "S07_NSGA"
        solution_id = int(r[sol_col]) if sol_col is not None else int(idx)

        rows.append(
            {
                "exp_id": exp_id,
                "method_id": method_id,
                "source_step": 7,
                "solution_type": "front_point",
                "solution_id": solution_id,
                "seed_mode": seed_mode,
                "label": f"{method_id} (seed={seed_mode}, id={solution_id})",
                "has_des": False,
                "f1": float(r[f1_col]),
                "f2": float(r[f2_col]),
                "f3": float(r[f3_col]),
                "f4": math.nan,
            }
        )

    return rows


def build_candidates_from_step08(output_dir: Path) -> List[Dict[str, object]]:
    """
    Build candidate rows from ./output/step08_des_metrics.csv, if it exists.

    These are DES runs that provide f4 (blocking/idle) and possibly
    replicated f1/f2/f3 under specific (β, θ) settings.

    Because join keys to schedules are implementation-dependent,
    we treat each row as its own 'DES' candidate keyed by whatever
    exp_id / scenario_id column is present.
    """
    import math

    path = output_dir / "step08_des_metrics.csv"
    if not path.is_file():
        return []

    df = pd.read_csv(path)
    if df.empty:
        return []

    exp_col = None
    for c in ["exp_id", "instance_id", "scenario_id"]:
        if c in df.columns:
            exp_col = c
            break

    beta_col = None
    for c in ["beta", "buffer_beta"]:
        if c in df.columns:
            beta_col = c
            break

    theta_col = None
    for c in ["theta", "ratio_theta"]:
        if c in df.columns:
            theta_col = c
            break

    f1_col = _resolve_metric_column(df, ["f1", "f1_total"])
    f2_col = _resolve_metric_column(df, ["f2", "f2_norm"])
    f3_col = _resolve_metric_column(df, ["f3", "f3_norm"])
    f4_col = _resolve_metric_column(df, ["f4", "f4_norm", "f4_block_idle"])

    rows: List[Dict[str, object]] = []

    for idx, r in df.iterrows():
        exp_id = str(r[exp_col]) if exp_col is not None else "DES"
        label_bits: List[str] = []
        if beta_col is not None:
            label_bits.append(f"β={r[beta_col]}")
        if theta_col is not None:
            label_bits.append(f"θ={r[theta_col]}")
        label = "Step-08 DES"
        if label_bits:
            label += " (" + ", ".join(label_bits) + ")"

        def _val(col: Optional[str]) -> float:
            if col is None or pd.isna(r[col]):
                return math.nan
            return float(r[col])

        rows.append(
            {
                "exp_id": exp_id,
                "method_id": "S08_DES",
                "source_step": 8,
                "solution_type": "singleton",
                "solution_id": int(idx),
                "seed_mode": "none",
                "label": label,
                "has_des": True,
                "f1": _val(f1_col),
                "f2": _val(f2_col),
                "f3": _val(f3_col),
                "f4": _val(f4_col),
                "beta": r[beta_col] if beta_col is not None else math.nan,
                "theta": r[theta_col] if theta_col is not None else math.nan,
            }
        )

    return rows


def build_aggregated_metrics(output_dir: Path) -> pd.DataFrame:
    """
    Build the canonical aggregated metrics table for Step-09 by
    concatenating candidates from Steps 05, 06, 07, and 08.

    Required:
        - Step-07 NSGA-II archive (step07_pareto.csv).

    Optional:
        - Step-05 manifest (step05_manifest.json).
        - Step-06 MILP optima (step06_*opt*.csv).
        - Step-08 DES metrics (step08_des_metrics.csv).
    """
    rows: List[Dict[str, object]] = []

    # Step-05 baseline (GS→SDSU→Merge)
    rows.extend(build_candidates_from_step05(output_dir))

    # Step-06 MILP optima, if any
    rows.extend(build_candidates_from_step06(output_dir))

    # Step-07 NSGA-II fronts (required)
    rows.extend(build_candidates_from_step07(output_dir))

    # Step-08 DES runs, if present
    rows.extend(build_candidates_from_step08(output_dir))

    if not rows:
        raise RuntimeError("No candidates could be constructed for Step-09.")

    df = pd.DataFrame(rows)

    # Normalize dtypes
    df["exp_id"] = df["exp_id"].astype(str)
    df["method_id"] = df["method_id"].astype(str)
    df["source_step"] = df["source_step"].astype(int)
    df["solution_type"] = df["solution_type"].astype(str)
    df["solution_id"] = df["solution_id"].astype(int)
    df["seed_mode"] = df["seed_mode"].astype(str)
    df["label"] = df["label"].astype(str)

    return df


def attach_context_columns(df: pd.DataFrame, context: ContextBundle) -> pd.DataFrame:
    df = df.copy()
    for key, val in context.token_stats.items():
        df[f"context_token_{key}"] = val
    for key, val in context.run_stats.items():
        df[f"context_run_{key}"] = val
    for key, val in context.changeover_stats.items():
        df[f"context_changeover_{key}"] = val

    df["context_anomaly_flag"] = bool(context.anomalies)
    df["context_anomaly_notes"] = "; ".join(context.anomalies)
    return df


def ensure_experiment_matrix(
    matrix_path: Path,
    cfg: Dict[str, Any],
    steps_cfg: Dict[str, Any],
) -> pd.DataFrame:
    """Ensure the experiment matrix exists and matches the documented schema."""

    if matrix_path.is_file():
        existing = pd.read_csv(matrix_path)
        missing = [col for col in DEFAULT_EXPERIMENT_COLUMNS if col not in existing.columns]
        if missing:
            raise ValueError(
                f"Existing experiment matrix at {matrix_path} missing columns: {missing}"
            )
        return existing

    window_cfg = cfg.get("window", {})
    spacing_cfg = cfg.get("spacing", {})
    adjacency_cfg = cfg.get("adjacency", {})
    rho_cfg = cfg.get("rho", {})
    tokens_cfg = cfg.get("tokens", {})
    sdsu_cfg = cfg.get("sdsu", {})
    nsga_cfg = cfg.get("nsga2", {})
    des_cfg = cfg.get("des", {})
    step05_cfg = steps_cfg.get("step05", {}) if isinstance(steps_cfg, dict) else {}

    baseline = {
        "exp_id": "baseline_a",
        "label": "Baseline A",
        "notes": "Default config run (GS→Split→SDSU→Merge)",
        "window.mode": window_cfg.get("mode", "ratio"),
        "window.jobs": window_cfg.get("jobs", 0),
        "window.ratio": window_cfg.get("ratio", 0.15),
        "spacing.alpha_run_cap": spacing_cfg.get("alpha_run_cap", 0.6),
        "spacing.max_run_abs": spacing_cfg.get("max_run_abs", 30),
        "spacing.picker": spacing_cfg.get("picker", "weighted"),
        "adjacency.scope_escalation": adjacency_cfg.get("scope_escalation", "global_if_missing"),
        "adjacency.fifo_lookahead": adjacency_cfg.get("fifo_lookahead", 3),
        "load.rho": rho_cfg.get("scale", 1.0),
        "tokens.flush_units_ratio": tokens_cfg.get("flush_units_ratio", 3.0),
        "tokens.empty_units_prepaint": tokens_cfg.get("empty_units_prepaint", 1.0),
        "tokens.empty_units_postpaint_ratio": tokens_cfg.get("empty_units_postpaint_ratio", 1.0),
        "step05.postpaint_empty.enabled": bool(step05_cfg.get("allow_postpaint_empty", False)),
        "step04.dp_threshold": sdsu_cfg.get("dp_threshold_jobs", 40),
        "step04.max_passes": sdsu_cfg.get("max_2opt_passes", 5),
        "nsga2.population": nsga_cfg.get("population_size", 100),
        "nsga2.generations": nsga_cfg.get("generations", 200),
        "nsga2.pc": nsga_cfg.get("crossover_rate", 0.9),
        "nsga2.pm": nsga_cfg.get("mutation_rate", 0.1),
        "nsga2.tournament_k": nsga_cfg.get("tournament_size", 4),
        "nsga2.sdsu_kernel": bool(nsga_cfg.get("sdsu_kernel", False)),
        "des.buffer_beta": des_cfg.get("beta", 0),
        "des.timing.mode": "constant",
        "des.f4_alpha": des_cfg.get("f4_alpha", 1.0),
    }

    rows: List[Dict[str, Any]] = []

    def add_row(exp_id: str, label: str, overrides: Dict[str, Any], notes: str = "") -> None:
        row = dict(baseline)
        row.update(overrides)
        row["exp_id"] = exp_id
        row["label"] = label
        if notes:
            row["notes"] = notes
        rows.append(row)

    add_row("baseline_a", "Baseline A", {}, baseline["notes"])

    for ratio in [0.10, 0.15, 0.25]:
        add_row(
            exp_id=f"window_ratio_{int(ratio*100):02d}",
            label=f"W sweep (ratio={ratio:.2f})",
            overrides={"window.ratio": ratio},
            notes="Window size sweep",
        )

    for rho_val in [0.0, 0.5, 1.0, 2.0]:
        add_row(
            exp_id=f"rho_{str(rho_val).replace('.', '_')}",
            label=f"ρ sweep ({rho_val:.2f})",
            overrides={"load.rho": rho_val},
            notes="Split-load sensitivity",
        )

    for flush_ratio in [0.5, 1.0, 2.0, 5.0]:
        add_row(
            exp_id=f"purge_cost_{int(flush_ratio*10):02d}",
            label=f"Purge sensitivity (ratio={flush_ratio:.1f})",
            overrides={"tokens.flush_units_ratio": flush_ratio},
            notes="Tokens flush cost sweep",
        )

    for scope in ["none", "global_if_missing"]:
        for fifo in [4, 10]:
            add_row(
                exp_id=f"adj_{scope}_{fifo}",
                label=f"Adjacency strictness ({scope}, k={fifo})",
                overrides={
                    "adjacency.scope_escalation": scope,
                    "adjacency.fifo_lookahead": fifo,
                },
                notes="Adjacency escalation sweep",
            )

    for enabled in [False, True]:
        add_row(
            exp_id=f"merge_postempty_{int(enabled)}",
            label=f"Merge policy (post-empty={'on' if enabled else 'off'})",
            overrides={"step05.postpaint_empty.enabled": enabled},
            notes="Audit merge EMPTY policy",
        )

    add_row(
        exp_id="nsga_seeded",
        label="NSGA-II seeded",
        overrides={"nsga2.sdsu_kernel": True},
        notes="Seeded NSGA-II run",
    )
    add_row(
        exp_id="nsga_unseeded",
        label="NSGA-II unseeded",
        overrides={"nsga2.sdsu_kernel": False},
        notes="Unseeded NSGA-II run",
    )

    for beta in [0, 5, 10]:
        for timing_mode in ["constant", "lookup"]:
            add_row(
                exp_id=f"des_beta_{beta}_mode_{timing_mode}",
                label=f"DES buffers (β={beta}, mode={timing_mode})",
                overrides={
                    "des.buffer_beta": beta,
                    "des.timing.mode": timing_mode,
                },
                notes="DES buffer/timing sweep",
            )

    df_matrix = pd.DataFrame(rows, columns=DEFAULT_EXPERIMENT_COLUMNS)
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    df_matrix.to_csv(matrix_path, index=False)
    return df_matrix


def summarize_des_runs(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["method_id"].str.upper() == "S08_DES"
    df_des = df[mask]
    if df_des.empty:
        return pd.DataFrame()

    return (
        df_des.groupby(["beta", "theta"], dropna=False)
        .agg(
            count=("f4", "count"),
            mean_f4=("f4", "mean"),
            std_f4=("f4", "std"),
        )
        .reset_index()
    )


# ---------------------------------------------------------------------------
# Baseline-relative comparison metrics and NSGA front summary
# ---------------------------------------------------------------------------


def select_baseline_index(df_exp: pd.DataFrame) -> int:
    """
    For a given experiment subset, choose a baseline schedule.

    Preference:
        1) Step-05 GS→SDSU candidate if present (source_step==5),
        2) Otherwise, lexicographically best in (f1, f2, f3).
    """
    import numpy as np

    s05 = df_exp[df_exp["source_step"] == 5]
    if not s05.empty:
        return int(s05.index[0])

    # Fallback: lexicographically best on (f1,f2,f3)
    # Missing metrics are treated as +inf.
    def _metric(row: pd.Series, col: str) -> float:
        val = row.get(col)
        if pd.isna(val):
            return float(np.inf)
        return float(val)

    best_idx = None
    best_key = None
    for idx, row in df_exp.iterrows():
        key = (_metric(row, "f1"), _metric(row, "f2"), _metric(row, "f3"))
        if best_key is None or key < best_key:
            best_key = key
            best_idx = idx

    assert best_idx is not None
    return int(best_idx)


def attach_comparison_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach:
        baseline_method_id, baseline_solution_id,
        delta_f1, delta_f2, delta_f3,
        dominates_baseline (minimization sense).
    """
    df = df.copy()
    df["baseline_method_id"] = ""
    df["baseline_solution_id"] = -1
    df["delta_f1"] = pd.NA
    df["delta_f2"] = pd.NA
    df["delta_f3"] = pd.NA
    df["dominates_baseline"] = False

    for exp_id, df_exp in df.groupby("exp_id"):
        base_idx = select_baseline_index(df_exp)
        b_row = df.loc[base_idx]

        b_f1 = float(b_row["f1"]) if not pd.isna(b_row["f1"]) else math.inf
        b_f2 = float(b_row["f2"]) if not pd.isna(b_row["f2"]) else math.inf
        b_f3 = float(b_row["f3"]) if not pd.isna(b_row["f3"]) else math.inf
        b_method = str(b_row["method_id"])
        b_sol = int(b_row["solution_id"])

        df.loc[df_exp.index, "baseline_method_id"] = b_method
        df.loc[df_exp.index, "baseline_solution_id"] = b_sol

        for idx in df_exp.index:
            f1 = float(df.at[idx, "f1"]) if not pd.isna(df.at[idx, "f1"]) else math.inf
            f2 = float(df.at[idx, "f2"]) if not pd.isna(df.at[idx, "f2"]) else math.inf
            f3 = float(df.at[idx, "f3"]) if not pd.isna(df.at[idx, "f3"]) else math.inf

            df.at[idx, "delta_f1"] = f1 - b_f1
            df.at[idx, "delta_f2"] = f2 - b_f2
            df.at[idx, "delta_f3"] = f3 - b_f3

            dominates = (
                (f1 <= b_f1)
                and (f2 <= b_f2)
                and (f3 <= b_f3)
                and ((f1 < b_f1) or (f2 < b_f2) or (f3 < b_f3))
            )
            df.at[idx, "dominates_baseline"] = dominates

    return df


def approximate_hypervolume(
    points: np.ndarray,
    reference: np.ndarray,
    samples: int = HV_SAMPLES,
    seed: int = 19,
) -> float:
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return math.nan
    mask = ~np.isnan(pts).any(axis=1)
    pts = pts[mask]
    if pts.size == 0:
        return math.nan

    ref = np.asarray(reference, dtype=float)
    lower = np.min(pts, axis=0)
    upper = np.maximum(ref, lower + EPSILON)
    span = upper - lower
    if np.any(span <= 0):
        span = np.maximum(span, EPSILON)

    rng = np.random.default_rng(seed)
    draws = lower + rng.random((samples, pts.shape[1])) * span
    comparisons = (pts[:, None, :] <= draws[None, :, :]).all(axis=2)
    dominated = comparisons.any(axis=0)
    hv = dominated.mean() * float(np.prod(span))
    return hv


def coverage_metric(points_a: np.ndarray, points_b: np.ndarray) -> float:
    if len(points_b) == 0:
        return math.nan
    if len(points_a) == 0:
        return 0.0
    dominated = 0
    for b in points_b:
        if np.isnan(b).any():
            continue
        if any((a <= b).all() and (a < b).any() for a in points_a if not np.isnan(a).any()):
            dominated += 1
    return dominated / len(points_b)


def identify_knee_point(df_group: pd.DataFrame) -> int:
    metrics = df_group[["f1", "f2", "f3"]].to_numpy(dtype=float)
    mask = ~np.isnan(metrics).any(axis=1)
    if not mask.any():
        return int(df_group.index[0])
    metrics = metrics[mask]
    mins = metrics.min(axis=0)
    maxs = metrics.max(axis=0)
    span = np.maximum(maxs - mins, EPSILON)
    normalized = (metrics - mins) / span
    scores = normalized.sum(axis=1)
    idxs = df_group.index[mask]
    return int(idxs[int(np.argmin(scores))])


def compute_seed_vs_unseed_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    mask = (df["source_step"] == 7) & (df["solution_type"].str.lower() == "front_point")
    front = df[mask]
    if front.empty or "seed_mode" not in front.columns:
        return {}

    seeded_mask = front["seed_mode"].str.lower() != "none"
    seeded = front[seeded_mask]
    unseeded = front[~seeded_mask]
    stats_summary: Dict[str, Dict[str, float]] = {}
    if seeded.empty or unseeded.empty:
        return stats_summary

    shared_exp = sorted(set(seeded["exp_id"]) & set(unseeded["exp_id"]))

    for metric in ["f1", "f2", "f3"]:
        pairs: List[Tuple[float, float]] = []
        for exp_id in shared_exp:
            s_val = seeded.loc[seeded["exp_id"] == exp_id, metric].min()
            u_val = unseeded.loc[unseeded["exp_id"] == exp_id, metric].min()
            if not (pd.isna(s_val) or pd.isna(u_val)):
                pairs.append((float(s_val), float(u_val)))

        if not pairs:
            continue

        deltas = [a - b for a, b in pairs]
        mean_delta = float(np.mean(deltas))
        median_delta = float(np.median(deltas))
        p_value = math.nan
        effect = math.nan
        if scipy_stats is not None and not all(abs(d) < EPSILON for d in deltas):
            try:
                stat = scipy_stats.wilcoxon([a for a, _ in pairs], [b for _, b in pairs])
                p_value = float(stat.pvalue)
                n = len(pairs)
                effect = 1 - (2 * float(stat.statistic) / (n * (n + 1)))
            except Exception:  # pragma: no cover - scipy optional
                p_value = math.nan
                effect = math.nan

        stats_summary[metric] = {
            "n": len(pairs),
            "mean_delta": mean_delta,
            "median_delta": median_delta,
            "wilcoxon_p": p_value,
            "rank_biserial": effect,
        }

    return stats_summary


def compute_correlations(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {"pearson": {}, "spearman": {}}
    metrics = ["f1", "f2", "f3"]
    subset = df[metrics].dropna()
    if len(subset) < 2:
        return results

    for method in ["pearson", "spearman"]:
        corr = subset.corr(method=method)
        for i, j in itertools.combinations(metrics, 2):
            key = f"{i}_{j}"
            results[method][key] = float(corr.loc[i, j])

    return results


def summarize_nsga_fronts(df: pd.DataFrame, hv_seed: int = 19) -> pd.DataFrame:
    """Summarize NSGA-II fronts per (exp_id, seed_mode) with additional diagnostics."""

    mask = (df["source_step"] == 7) & (df["solution_type"].str.lower() == "front_point")
    df_front = df[mask]
    if df_front.empty:
        return pd.DataFrame()

    ref_vals = df[["f1", "f2", "f3"]].dropna()
    if ref_vals.empty:
        reference = np.array([1.0, 1.0, 1.0])
    else:
        maxima = ref_vals.max().to_numpy(dtype=float)
        reference = maxima + 0.1 * np.maximum(np.abs(maxima), 1.0)

    rows: List[Dict[str, object]] = []

    for (exp_id, seed_mode), df_group in df_front.groupby(["exp_id", "seed_mode"]):
        front_size = int(len(df_group))
        dom_frac = float(df_group["dominates_baseline"].mean()) if front_size > 0 else math.nan

        df_dom = df_group[df_group["dominates_baseline"]]
        if not df_dom.empty:
            best_delta_f1 = float(df_dom["delta_f1"].min())
            best_delta_f2 = float(df_dom["delta_f2"].min())
            best_delta_f3 = float(df_dom["delta_f3"].min())
        else:
            best_delta_f1 = math.nan
            best_delta_f2 = math.nan
            best_delta_f3 = math.nan

        baseline_rows = df[
            (df["exp_id"] == exp_id)
            & (df["baseline_method_id"] == df_group["baseline_method_id"].iloc[0])
        ]
        if baseline_rows.empty:
            rep_idx = int(df_group.index[0])
        else:
            b_row = baseline_rows.iloc[0]
            b_f1 = float(b_row["f1"])
            b_f2 = float(b_row["f2"])
            b_f3 = float(b_row["f3"])

            def _dist(row: pd.Series) -> float:
                f1 = float(row["f1"])
                f2 = float(row["f2"])
                f3 = float(row["f3"])
                d1 = (f1 - b_f1) / max(abs(b_f1), EPSILON)
                d2 = (f2 - b_f2) / max(abs(b_f2), EPSILON)
                d3 = (f3 - b_f3) / max(abs(b_f3), EPSILON)
                return d1 * d1 + d2 * d2 + d3 * d3

            rep_idx = min(df_group.index, key=lambda i: _dist(df_group.loc[i]))

        rep_row = df_group.loc[rep_idx]

        points = df_group[["f1", "f2", "f3"]].to_numpy(dtype=float)
        hv_seed_adj = hv_seed + (hash((exp_id, seed_mode)) & 0xFFFF)
        approx_hv = approximate_hypervolume(points, reference, seed=hv_seed_adj)

        partner = df_front[(df_front["exp_id"] == exp_id) & (df_front["seed_mode"] != seed_mode)]
        partner_points = partner[["f1", "f2", "f3"]].to_numpy(dtype=float)
        coverage_vs_partner = coverage_metric(points, partner_points)

        knee_idx = identify_knee_point(df_group)
        knee_row = df_group.loc[knee_idx]

        rows.append(
            {
                "exp_id": exp_id,
                "seed_mode": seed_mode,
                "front_size": front_size,
                "dominates_baseline_frac": dom_frac,
                "best_delta_f1": best_delta_f1,
                "best_delta_f2": best_delta_f2,
                "best_delta_f3": best_delta_f3,
                "representative_solution_id": int(rep_row["solution_id"]),
                "representative_method_id": str(rep_row["method_id"]),
                "approx_hypervolume": approx_hv,
                "coverage_vs_partner": coverage_vs_partner,
                "knee_solution_id": int(knee_row["solution_id"]),
                "knee_method_id": str(knee_row["method_id"]),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------


def plot_experiment_scatter(
    df: pd.DataFrame,
    output_dir: Path,
    exp_id: str,
    x: str = "f1",
    y: str = "f2",
) -> None:
    if plt is None:
        return

    df_exp = df[df["exp_id"] == exp_id]
    if df_exp.empty:
        return

    fig, ax = plt.subplots()

    mask_singleton = df_exp["solution_type"].str.lower() == "singleton"
    mask_front = df_exp["solution_type"].str.lower() == "front_point"

    if mask_singleton.any():
        ax.scatter(
            df_exp.loc[mask_singleton, x],
            df_exp.loc[mask_singleton, y],
            s=60,
            marker="s",
            label="Singletons (Steps 05/06/08)",
        )

    if mask_front.any():
        for seed_mode, df_seed in df_exp[mask_front].groupby("seed_mode"):
            ax.scatter(
                df_seed[x],
                df_seed[y],
                s=20,
                label=f"NSGA-II (seed={seed_mode})",
            )

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"Experiment {exp_id}: {x} vs {y}")
    ax.legend()
    fig.tight_layout()

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / f"step09_{x}_vs_{y}_exp_{exp_id}.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def generate_all_plots(df: pd.DataFrame, output_dir: Path) -> None:
    if plt is None:
        return
    for exp_id in sorted(df["exp_id"].unique()):
        plot_experiment_scatter(df, output_dir, exp_id, x="f1", y="f2")
        plot_experiment_scatter(df, output_dir, exp_id, x="f1", y="f3")


# ---------------------------------------------------------------------------
# Manifest and report
# ---------------------------------------------------------------------------


def write_manifest(
    manifest_path: Path,
    dataset: DatasetContext,
    df_metrics: pd.DataFrame,
    input_paths: Dict[str, Path],
    metrics_path: Path,
    experiment_matrix_path: Path,
    front_summary_path: Path,
    adjacency_summary_path: Path,
    context_path: Path,
    config_path: Path,
    steps_config_path: Path,
    stats_payload: Dict[str, Any],
    anomalies: List[str],
) -> None:
    manifest = {
        "step_id": "step09_analysis",
        "dataset_dir": str(dataset.dataset_dir),
        "adjacency_rules_path": str(dataset.adjacency_rules_path),
        "adjacency_rules_sha256": sha256_file(dataset.adjacency_rules_path),
        "inputs": {
            name: {
                "path": str(path),
                "exists": path.is_file(),
                "sha256": sha256_file(path) if path.is_file() else None,
            }
            for name, path in input_paths.items()
        },
        "step09_metrics_path": str(metrics_path),
        "step09_metrics_sha256": sha256_file(metrics_path),
        "experiment_matrix_path": str(experiment_matrix_path),
        "experiment_matrix_sha256": sha256_file(experiment_matrix_path)
        if experiment_matrix_path.is_file()
        else None,
        "step09_front_summary_path": str(front_summary_path),
        "step09_front_summary_sha256": sha256_file(front_summary_path)
        if front_summary_path.is_file()
        else None,
        "adjacency_summary_path": str(adjacency_summary_path),
        "adjacency_summary_sha256": sha256_file(adjacency_summary_path)
        if adjacency_summary_path.is_file()
        else None,
        "context_metrics_path": str(context_path),
        "context_metrics_sha256": sha256_file(context_path)
        if context_path.is_file()
        else None,
        "config_snapshot": {
            "path": str(config_path),
            "exists": config_path.is_file(),
            "sha256": sha256_file(config_path) if config_path.is_file() else None,
        },
        "steps_config_snapshot": {
            "path": str(steps_config_path),
            "exists": steps_config_path.is_file(),
            "sha256": sha256_file(steps_config_path) if steps_config_path.is_file() else None,
        },
        "num_rows": int(len(df_metrics)),
        "exp_ids": sorted(df_metrics["exp_id"].astype(str).unique().tolist()),
        "statistics": stats_payload,
        "anomalies": anomalies,
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def write_final_report(
    report_path: Path,
    df_metrics: pd.DataFrame,
    df_front_summary: pd.DataFrame,
    experiment_matrix: pd.DataFrame,
    context: ContextBundle,
    stats_payload: Dict[str, Any],
    input_hash_rows: List[Dict[str, str]],
    des_summary: pd.DataFrame,
    cfg: Dict[str, Any],
) -> None:
    lines: List[str] = []

    lines.append("# Step-09 — Integrated Research Analysis & Observational Studies")
    lines.append("")
    lines.append(
        "This document materializes the Step-09 protocol: it aggregates prior-step artifacts, "
        "computes comparative metrics, flags anomalies, and records reproducibility evidence "
        "without mutating any schedules."
    )
    lines.append("")

    # Section 1: Abstract & dataset hashes
    lines.append("## Abstract & Dataset Hashes")
    lines.append("")
    lines.append("| artifact | path | SHA-256 |")
    lines.append("|----------|------|---------|")
    for row in input_hash_rows:
        sha = row.get("sha") or "—"
        lines.append(f"| {row['name']} | `{row['path']}` | `{sha}` |")
    lines.append("")

    # Section 2: Methods snapshot
    lines.append("## Methods Snapshot")
    lines.append("")
    lines.append(
        "- Replayed C2 adjacency over feeder, booth, SDSU, and merged convoys (no rescheduling)."
    )
    lines.append(
        "- Aggregated f1/f2/f3/f4 from GS→SDSU, MILP optima, NSGA-II fronts, and DES replications."
    )
    lines.append(
        "- Derived diagnostics (token reliance, color-run stats, purge counts) from recorded streams."
    )
    lines.append(
        "- Backfilled the experiment matrix exactly as defined in STEP09_NOTES to capture sweeps."
    )
    lines.append("")

    # Section 3: Results
    lines.append("## Results")
    lines.append("")

    # Method comparison table
    lines.append("### Method Comparison (f1/f2/f3)")
    lines.append("| method_id | count | mean f1 | mean f2 | mean f3 | dominates baseline (%) |")
    lines.append("|-----------|-------|---------|---------|---------|-------------------------|")
    for method_id, df_m in df_metrics.groupby("method_id"):
        count = len(df_m)
        mean_f1 = float(df_m["f1"].mean()) if not df_m["f1"].isna().all() else math.nan
        mean_f2 = float(df_m["f2"].mean()) if not df_m["f2"].isna().all() else math.nan
        mean_f3 = float(df_m["f3"].mean()) if not df_m["f3"].isna().all() else math.nan
        dom_frac = float(df_m["dominates_baseline"].mean()) * 100.0
        lines.append(
            f"| {method_id} | {count:d} | {mean_f1:.3f} | {mean_f2:.3f} | {mean_f3:.3f} | {dom_frac:.1f} |"
        )
    lines.append("")

    # Experiment matrix snapshot
    lines.append("### Experiment Matrix Snapshot")
    lines.append(
        "Key rows staged for future end-to-end reruns; full matrix stored in ``step09_experiment_matrix.csv``."
    )
    cols = [
        "exp_id",
        "label",
        "window.ratio",
        "load.rho",
        "tokens.flush_units_ratio",
        "step05.postpaint_empty.enabled",
        "des.buffer_beta",
        "des.timing.mode",
    ]
    lines.append("| exp_id | label | W ratio | ρ | flush ratio | post-empty | β | timing |")
    lines.append("|--------|-------|---------|---|-------------|------------|---|--------|")
    for _, row in experiment_matrix[cols].head(12).iterrows():
        lines.append(
            "| {exp} | {label} | {wr:.2f} | {rho:.2f} | {flush:.2f} | {pe} | {beta} | {mode} |".format(
                exp=row["exp_id"],
                label=row["label"],
                wr=float(row["window.ratio"]),
                rho=float(row["load.rho"]),
                flush=float(row["tokens.flush_units_ratio"]),
                pe=str(row["step05.postpaint_empty.enabled"]).lower(),
                beta=row["des.buffer_beta"],
                mode=row["des.timing.mode"],
            )
        )
    lines.append("")

    # NSGA-II section
    lines.append("### NSGA-II Front Diagnostics")
    if df_front_summary.empty:
        lines.append("No NSGA-II fronts detected in inputs.")
    else:
        lines.append(
            "| exp_id | seed_mode | size | Δf1 best | Δf2 best | Δf3 best | hv | coverage | knee (method,id) |"
        )
        lines.append(
            "|--------|-----------|------|----------|----------|----------|----|----------|------------------|"
        )
        for _, r in df_front_summary.iterrows():
            lines.append(
                "| {exp} | {seed} | {size:d} | {d1:.3f} | {d2:.3f} | {d3:.3f} | {hv:.3f} | {cov:.2f} | {knee} |".format(
                    exp=r["exp_id"],
                    seed=r["seed_mode"],
                    size=int(r["front_size"]),
                    d1=r["best_delta_f1"],
                    d2=r["best_delta_f2"],
                    d3=r["best_delta_f3"],
                    hv=r.get("approx_hypervolume", math.nan),
                    cov=r.get("coverage_vs_partner", math.nan),
                    knee=f"{r['knee_method_id']}#{r['knee_solution_id']}",
                )
            )
    lines.append("")

    # DES curves
    lines.append("### DES Idle/Blocking (f4)")
    if des_summary.empty:
        lines.append("No DES metrics ingested.")
    else:
        lines.append("| β | θ | runs | mean f4 | std f4 |")
        lines.append("|---|---|------|---------|--------|")
        for _, row in des_summary.iterrows():
            lines.append(
                f"| {row['beta']} | {row['theta']} | {int(row['count'])} | {row['mean_f4']:.3f} | "
                f"{(row['std_f4'] if not math.isnan(row['std_f4']) else 0.0):.3f} |"
            )
    lines.append("")

    # Section 4: Observations & Interpretation
    lines.append("## Observations & Interpretation")
    lines.append("")
    corr_stats = stats_payload.get("correlations", {})
    pearson = corr_stats.get("pearson", {})
    spearman = corr_stats.get("spearman", {})
    seed_stats = stats_payload.get("seed_vs_unseed", {})
    lines.append(
        f"- Token reliance: {context.token_stats['token_per_100_jobs']:.1f} tokens/100 jobs, "
        f"EMPTY {context.token_stats['empty_per_100']:.1f}/100, FLUSH {context.token_stats['flush_per_100']:.1f}/100."
    )
    lines.append(
        f"- Color runs: longest {context.run_stats['longest_run']:.1f}, mean {context.run_stats['mean_run']:.1f} "
        f"(std {context.run_stats['std_run']:.1f}); purge ratio {context.changeover_stats['purge_ratio']:.3f}."
    )
    if pearson:
        lines.append(
            "- Pearson correlations: "
            + ", ".join(f"{k}={v:.2f}" for k, v in pearson.items())
        )
    if spearman:
        lines.append(
            "- Spearman correlations: "
            + ", ".join(f"{k}={v:.2f}" for k, v in spearman.items())
        )
    if seed_stats:
        for metric, s in seed_stats.items():
            lines.append(
                f"- Seeded vs unseeded ({metric}): mean Δ={s['mean_delta']:.3f}, median Δ="
                f"{s['median_delta']:.3f}, p={s['wilcoxon_p'] if not math.isnan(s['wilcoxon_p']) else '—'}, "
                f"rank-biserial={s['rank_biserial'] if not math.isnan(s['rank_biserial']) else '—'}."
            )
    if not seed_stats:
        lines.append("- Seeded vs unseeded NSGA-II comparison pending (insufficient paired fronts).")
    lines.append("")

    # Section 5: Anomalies & Validity
    lines.append("## Anomalies & Validity")
    lines.append("")
    if context.anomalies:
        for reason in context.anomalies:
            lines.append(f"- {reason}")
    else:
        lines.append("- No anomalies flagged under current thresholds.")
    lines.append("")

    # Section 6: Limitations & Next Steps
    lines.append("## Limitations & Next Steps")
    lines.append("")
    lines.append("- Experiment matrix rows are staged but the multi-run executor is not triggered in Step-09.")
    lines.append("- Hypervolume is Monte-Carlo based; rerun with higher samples for publication-grade precision.")
    lines.append("- DES statistics reflect available traces only; rerun Step-08 to populate missing β,θ pairs.")
    lines.append("")

    # Section 7: Reproducibility
    lines.append("## Reproducibility Notes")
    lines.append("")
    seeds = cfg.get("reproducibility", {}).get("seeds", []) if isinstance(cfg, dict) else []
    lines.append(f"- Config seeds: {seeds if seeds else 'n/a'}")
    lines.append(f"- Context metrics stored at `{CONTEXT_JSON_NAME}`; manifest contains path + SHA-256.")
    lines.append(f"- Verification checklist: adjacency summary saved, experiment matrix hashed, NSGA-II stats logged.")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Step-09 — Integrated analysis over Steps 02–08 outputs."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="model_config.toml",
        help="Optional model_config.toml; if absent, ./dataset and ./output are used.",
    )
    parser.add_argument(
        "--steps-config",
        type=str,
        default="steps_config.toml",
        help="Optional steps_config.toml for path + policy defaults.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Override dataset directory (where adjacency_rules.csv lives).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory (where step0X_*.csv live).",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if args.config == "model_config.toml" and not config_path.is_file():
        alt = Path("global_config.toml")
        if alt.is_file():
            config_path = alt

    cfg = load_config_if_present(config_path)
    steps_config_path = Path(args.steps_config)
    steps_cfg = load_steps_config_if_present(steps_config_path)
    dataset_dir, output_dir = resolve_base_paths(cfg, args.dataset_dir, args.output_dir)

    dataset = load_dataset(dataset_dir)
    seq_cache = load_step_sequences(output_dir)

    context_bundle = build_context_bundle(dataset, output_dir, seq_cache, cfg)
    adj_summary_path = output_dir / "step09_adjacency_summary.csv"
    context_bundle.adjacency_summary.to_csv(adj_summary_path, index=False)

    df_metrics = build_aggregated_metrics(output_dir)
    df_metrics = attach_comparison_metrics(df_metrics)
    df_metrics = attach_context_columns(df_metrics, context_bundle)

    experiment_matrix_path = output_dir / "step09_experiment_matrix.csv"
    experiment_matrix_df = ensure_experiment_matrix(experiment_matrix_path, cfg, steps_cfg)
    df_metrics = df_metrics.merge(experiment_matrix_df, on="exp_id", how="left", suffixes=("", "_cfg"))

    df_front_summary = summarize_nsga_fronts(df_metrics)
    des_summary = summarize_des_runs(df_metrics)

    stats_payload = {
        "seed_vs_unseed": compute_seed_vs_unseed_stats(df_metrics),
        "correlations": compute_correlations(df_metrics),
    }
    stats_payload_serializable = sanitize_for_json(stats_payload)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "step09_metrics.csv"
    df_metrics.to_csv(metrics_path, index=False)

    front_summary_path = output_dir / FRONT_SUMMARY_FILENAME
    df_front_summary.to_csv(front_summary_path, index=False)

    context_path = write_context_payload(
        output_dir,
        context_bundle,
        cfg.get("spacing", {}) if isinstance(cfg, dict) else {},
        cfg.get("tokens", {}) if isinstance(cfg, dict) else {},
    )

    experiment_matrix_df.to_csv(experiment_matrix_path, index=False)

    generate_all_plots(df_metrics, output_dir)

    manifest_path = output_dir / "manifests" / "step09_manifest.json"
    report_path = output_dir / "reports" / "step09_final_report.md"

    input_paths = {
        "step02_paint_seed": output_dir / "step02_paint_seed.csv",
        "step03_booth1": output_dir / "step03_booth1.csv",
        "step03_booth2": output_dir / "step03_booth2.csv",
        "step04_booth1_sdsu": output_dir / "step04_booth1_sdsu.csv",
        "step04_booth2_sdsu": output_dir / "step04_booth2_sdsu.csv",
        "step05_global_line": output_dir / "step05_global_line.csv",
        "step06_opt_glob": output_dir / "step06_opt_global.csv",
        "step07_pareto": output_dir / "step07_pareto.csv",
        "step08_des_metrics": output_dir / "step08_des_metrics.csv",
    }

    input_hash_rows: List[Dict[str, str]] = []
    input_hash_rows.append(
        {
            "name": "dataset.adjacency_rules",
            "path": str(dataset.adjacency_rules_path),
            "sha": sha256_file(dataset.adjacency_rules_path),
        }
    )
    for name, path in input_paths.items():
        input_hash_rows.append(
            {
                "name": name,
                "path": str(path),
                "sha": sha256_file(path) if path.is_file() else None,
            }
        )
    if config_path.is_file():
        input_hash_rows.append(
            {
                "name": "config",
                "path": str(config_path),
                "sha": sha256_file(config_path),
            }
        )

    write_manifest(
        manifest_path=manifest_path,
        dataset=dataset,
        df_metrics=df_metrics,
        input_paths=input_paths,
        metrics_path=metrics_path,
        experiment_matrix_path=experiment_matrix_path,
        front_summary_path=front_summary_path,
        adjacency_summary_path=adj_summary_path,
        context_path=context_path,
        config_path=config_path,
        steps_config_path=steps_config_path,
        stats_payload=stats_payload_serializable,
        anomalies=context_bundle.anomalies,
    )

    write_final_report(
        report_path,
        df_metrics,
        df_front_summary,
        experiment_matrix_df,
        context_bundle,
        stats_payload,
        input_hash_rows,
        des_summary,
        cfg,
    )


if __name__ == "__main__":
    main()
