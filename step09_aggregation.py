#!/usr/bin/env python3
"""
STEP-09 aggregate as specified in notes/STEP09_NOTES.md.

- Read configs: global_config.toml + steps_config.toml
- Inputs (under output_dir from [paths]):
    step05_global_line.csv
    step06_milp_opt_global.csv
    step07_pareto.csv (+ step07_pareto_seeded.csv optional)
    step07_global_line.csv (+ step07_global_line_seeded.csv optional)
    step08_des_metrics.csv
    manifests: step03_manifest.json, step04_manifest.json, step06_from_step03_manifest.json
    dataset/topcoat_codebook.csv (for color->family map)
- Outputs (paths from [step09] in steps_config, defaults given below):
    step09_metrics.csv
    step09_experiment_matrix.csv
    manifests/step09_manifest.json
    reports/step09_final_report.md
    step09_plots/* (optional)
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None  # plotting optional


# ---------------------------------------------------------------------------

@dataclass
class ConfigPaths:
    dataset_dir: Path
    output_dir: Path
    metrics_out: Path
    experiment_matrix_out: Path
    manifest_out: Path
    report_out: Path
    plots_dir: Path


def load_toml(path: Path) -> Dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing config file: {path}")
    with path.open("rb") as f:
        return tomllib.load(f)


def resolve_paths(global_cfg: Dict, steps_cfg: Dict, output_dir_arg: Optional[str], dataset_dir_arg: Optional[str]) -> ConfigPaths:
    paths_cfg = global_cfg.get("paths", {})
    output_dir = Path(output_dir_arg or paths_cfg.get("output_dir", "output")).resolve()
    dataset_dir = Path(dataset_dir_arg or paths_cfg.get("dataset_dir", "dataset")).resolve()

    step09_cfg = steps_cfg.get("step09", {})
    metrics_rel = step09_cfg.get("metrics_relpath", "step09_metrics.csv")
    matrix_rel = step09_cfg.get("experiment_matrix_relpath", "step09_experiment_matrix.csv")
    manifest_rel = step09_cfg.get("manifest_relpath", "manifests/step09_manifest.json")
    report_rel = step09_cfg.get("final_report_relpath", "reports/step09_final_report.md")
    plots_dir = output_dir / step09_cfg.get("plots_dir", "step09_plots")

    return ConfigPaths(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        metrics_out=output_dir / metrics_rel,
        experiment_matrix_out=output_dir / matrix_rel,
        manifest_out=output_dir / manifest_rel,
        report_out=output_dir / report_rel,
        plots_dir=plots_dir,
    )


def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_required(path: Path, **kwargs) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required input missing: {path}")
    return pd.read_csv(path, **kwargs)


# ---------------------------------------------------------------------------
# f2 recomputation helpers
# ---------------------------------------------------------------------------

def compute_window_size(n_jobs: int, window_frac: float) -> int:
    return max(1, int(round(window_frac * n_jobs)))


def infer_family_series(df: pd.DataFrame, codebook: Dict[str, str]) -> pd.Series:
    """
    Returns paint families, using fallbacks across available columns.
    Tries topcoat family, then color_code/topcoat_code/topcoat mappings.
    """
    candidates = []
    if "topcoat_topcoat_family" in df.columns:
        candidates.append(df["topcoat_topcoat_family"])
    if "color_code" in df.columns:
        candidates.append(df["color_code"].map(codebook))
    if "topcoat_code" in df.columns:
        candidates.append(df["topcoat_code"].map(codebook))
    if "topcoat" in df.columns:
        candidates.append(df["topcoat"].map(codebook))

    if not candidates:
        raise ValueError("Cannot infer color family: no known color columns present")

    fam = candidates[0].copy()
    for cand in candidates[1:]:
        fam = fam.fillna(cand)

    missing = fam.isna()
    if missing.any():
        # Show a small sample of unresolved rows to aid debugging
        cols = [c for c in ["topcoat_topcoat_family", "color_code", "topcoat_code", "topcoat"] if c in df.columns]
        sample = df.loc[missing, cols].head().to_dict(orient="records")
        raise ValueError(f"Color family lookup failed for {missing.sum()} rows; sample={sample}")

    return fam.reset_index(drop=True)


def mix_imbalance(families: pd.Series, window_frac: float) -> float:
    n = len(families)
    if n == 0:
        return math.nan
    W = compute_window_size(n, window_frac)
    target = families.value_counts(normalize=True)
    if n < W:
        W = n
    deltas: List[float] = []
    for start in range(0, n - W + 1):
        window = families.iloc[start:start + W].value_counts(normalize=True)
        delta = (window.reindex(target.index, fill_value=0) - target).abs().sum()
        deltas.append(delta)
    return float(sum(deltas) / len(deltas))


def strip_empty_tokens(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove EMPTY/blank tokens before computing mix metrics.
    Handles cases where token_type is missing/blank or booth_id is the string 'EMPTY'.
    """
    mask = pd.Series(True, index=df.index)
    if "token_type" in df.columns:
        mask &= df["token_type"].fillna("").astype(str).str.upper() != "EMPTY"
    if "booth_id" in df.columns:
        mask &= df["booth_id"].fillna("").astype(str).str.upper() != "EMPTY"
    if "part_id" in df.columns:
        mask &= df["part_id"].fillna("") != ""
    if "color_code" in df.columns:
        mask &= df["color_code"].fillna("") != ""
    return df.loc[mask].reset_index(drop=True)



# ---------------------------------------------------------------------------
# Load static metrics (f1, f3) per method
# ---------------------------------------------------------------------------

def sdsu_f_metrics(output_dir: Path) -> Tuple[float, float]:
    step04_manifest = read_json(output_dir / "manifests" / "step04_manifest.json")
    booths = step04_manifest.get("metrics", {}).get("booths", [])
    f1 = float(sum(b.get("total_setup_cost", 0.0) for b in booths))

    step03_manifest = read_json(output_dir / "manifests" / "step03_manifest.json")
    f3 = float(step03_manifest.get("settings_effective", {}).get("f3_split")
               or step03_manifest.get("metrics", {}).get("f3_split", 0.0))
    return f1, f3


def milp_f_metrics(output_dir: Path) -> Tuple[float, float]:
    manifest = read_json(output_dir / "manifests" / "step06_from_step03_manifest.json")
    instances = manifest.get("instances", [])
    f1 = float(sum(inst.get("f1_milp", 0.0) for inst in instances))
    # MILP keeps same split as SDSU
    step03_manifest = read_json(output_dir / "manifests" / "step03_manifest.json")
    f3 = float(step03_manifest.get("settings_effective", {}).get("f3_split")
               or step03_manifest.get("metrics", {}).get("f3_split", 0.0))
    return f1, f3


def nsga_static_metrics(pareto_path: Path, method: str) -> pd.DataFrame:
    df = read_csv_required(pareto_path)
    df = df.rename(columns={"solution_id": "solution_id", "seed": "seed"})
    df["method"] = method
    return df[["method", "solution_id", "seed", "f1", "f2", "f3"]]


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def recompute_f2_for_convoys(convoy_df: pd.DataFrame, colorbook: Dict[str, str], window_frac: float, group_col: Optional[str] = None) -> pd.DataFrame:
    convoy_df = strip_empty_tokens(convoy_df)
    results: List[Dict] = []
    if group_col is None:
        families = infer_family_series(convoy_df, colorbook)
        f2 = mix_imbalance(families, window_frac)
        results.append({"solution_id": None, "f2": f2})
    else:
        for sid, df_g in convoy_df.groupby(group_col):
            families = infer_family_series(df_g, colorbook)
            f2 = mix_imbalance(families, window_frac)
            results.append({"solution_id": sid, "f2": f2})
    return pd.DataFrame(results)


def build_metrics_table(cfg_paths: ConfigPaths, global_cfg: Dict) -> pd.DataFrame:
    objective_cfg = global_cfg.get("objective", {})
    window_frac = float(objective_cfg.get("window_W_frac", 0.1))
    report_pareto_only = bool(objective_cfg.get("report_pareto_only", False))

    # color map
    codebook_df = read_csv_required(cfg_paths.dataset_dir / "topcoat_codebook.csv")
    codebook = dict(zip(codebook_df["topcoat_code"], codebook_df["topcoat_family"]))

    des_df = read_csv_required(cfg_paths.output_dir / "step08_des_metrics.csv")
    des_df["solution_id"] = des_df["solution_id"].fillna("").astype(str)
    des_df["scenario"] = des_df["scenario"].fillna("").astype(str)

    rows: List[pd.DataFrame] = []

    # SDSU
    f1_sdsu, f3_sdsu = sdsu_f_metrics(cfg_paths.output_dir)
    sdsu_convoy = read_csv_required(cfg_paths.output_dir / "step05_global_line.csv")
    f2_sdsu = recompute_f2_for_convoys(sdsu_convoy, codebook, window_frac).iloc[0]["f2"]
    rows.append(pd.DataFrame([{
        "method": "step05",
        "scenario": "step05_global",
        "solution_id": "",
        "seed": "",
        "f1": f1_sdsu,
        "f2": f2_sdsu,
        "f3": f3_sdsu,
    }]))

    # MILP
    f1_milp, f3_milp = milp_f_metrics(cfg_paths.output_dir)
    milp_convoy = read_csv_required(cfg_paths.output_dir / "step06_milp_opt_global.csv")
    f2_milp = recompute_f2_for_convoys(milp_convoy, codebook, window_frac).iloc[0]["f2"]
    rows.append(pd.DataFrame([{
        "method": "step06_milp",
        "scenario": "step06_global",
        "solution_id": "",
        "seed": "",
        "f1": f1_milp,
        "f2": f2_milp,
        "f3": f3_milp,
    }]))

    # NSGA unseeded
    pareto_unseeded = cfg_paths.output_dir / "step07_pareto.csv"
    if pareto_unseeded.is_file():
        nsga_un = nsga_static_metrics(pareto_unseeded, "step07_nsga2")
        gl_un = read_csv_required(cfg_paths.output_dir / "step07_global_line.csv")
        f2_un = recompute_f2_for_convoys(gl_un, codebook, window_frac, group_col="solution_id")
        nsga_un = nsga_un.merge(f2_un, on="solution_id", suffixes=("", "_recomputed"))
        nsga_un["f2"] = nsga_un["f2_recomputed"]
        nsga_un = nsga_un[["method", "solution_id", "seed", "f1", "f2", "f3"]]
        nsga_un["scenario"] = nsga_un["solution_id"].apply(lambda sid: f"step07_global_line::sid{sid}")
        rows.append(nsga_un)

    # NSGA seeded (optional)
    pareto_seeded = cfg_paths.output_dir / "step07_pareto_seeded.csv"
    global_seeded = cfg_paths.output_dir / "step07_global_line_seeded.csv"
    if pareto_seeded.is_file() and global_seeded.is_file():
        nsga_sd = nsga_static_metrics(pareto_seeded, "step07_nsga2_seeded")
        gl_sd = read_csv_required(global_seeded)
        f2_sd = recompute_f2_for_convoys(gl_sd, codebook, window_frac, group_col="solution_id")
        nsga_sd = nsga_sd.merge(f2_sd, on="solution_id", suffixes=("", "_recomputed"))
        nsga_sd["f2"] = nsga_sd["f2_recomputed"]
        nsga_sd = nsga_sd[["method", "solution_id", "seed", "f1", "f2", "f3"]]
        nsga_sd["scenario"] = nsga_sd["solution_id"].apply(lambda sid: f"step07_global_line_seeded::sid{sid}")
        rows.append(nsga_sd)

    static_df = pd.concat(rows, ignore_index=True)

    # Join with DES metrics on method+scenario+solution_id
    static_df["solution_id"] = static_df["solution_id"].fillna("").astype(str)
    merged = static_df.merge(
        des_df,
        on=["method", "scenario", "solution_id"],
        how="inner",
        suffixes=("", "_des"),
    )

    required_cols = ["method", "scenario", "solution_id", "seed", "f1", "f2", "f3", "f4"]
    missing_f4 = merged["f4"].isna().sum()
    if missing_f4:
        raise ValueError(f"{missing_f4} scenarios missing f4 from DES; aborting")

    # Pareto filter if enabled
    if report_pareto_only:
        merged = pareto_front(merged, ["f1", "f2", "f3", "f4"])

    merged = merged.reset_index(drop=True)
    return merged, report_pareto_only


def pareto_front(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    mask = []
    values = df[cols].values
    for i, row in enumerate(values):
        dominated = False
        for j, other in enumerate(values):
            if i == j:
                continue
            if all(other <= row) and any(other < row):
                dominated = True
                break
        mask.append(not dominated)
    return df.loc[mask]


def build_experiment_matrix(df_metrics: pd.DataFrame, global_cfg: Dict, steps_cfg: Dict) -> pd.DataFrame:
    objective_cfg = global_cfg.get("objective", {})
    rho_cfg = global_cfg.get("rho", {})
    sdsu_cfg = global_cfg.get("sdsu", {})
    tokens_cfg = global_cfg.get("tokens", {})
    milp_cfg = global_cfg.get("milp", {})
    nsga_cfg = global_cfg.get("nsga2", {})
    des_cfg = global_cfg.get("des", {})

    exp = df_metrics.copy()
    exp["window_W_frac"] = objective_cfg.get("window_W_frac")
    exp["report_pareto_only"] = objective_cfg.get("report_pareto_only")
    exp["rho_scale"] = rho_cfg.get("scale")
    exp["dp_threshold_jobs"] = sdsu_cfg.get("dp_threshold_jobs")
    exp["max_2opt_passes"] = sdsu_cfg.get("max_2opt_passes")
    exp["allow_purge"] = sdsu_cfg.get("allow_purge")
    exp["purge_cost_weight"] = sdsu_cfg.get("purge_cost_weight")
    exp["flush_units_ratio"] = tokens_cfg.get("flush_units_ratio")
    exp["empty_time_ratio"] = tokens_cfg.get("empty_time_ratio")
    exp["milp_time_limit_sec"] = milp_cfg.get("time_limit_sec")
    exp["milp_mip_gap"] = milp_cfg.get("mip_gap")
    exp["milp_max_jobs_per_instance"] = milp_cfg.get("max_jobs_per_instance")
    exp["milp_threads"] = milp_cfg.get("threads")
    exp["nsga_population_size"] = nsga_cfg.get("population_size")
    exp["nsga_generations"] = nsga_cfg.get("generations")
    exp["nsga_crossover_rate"] = nsga_cfg.get("crossover_rate")
    exp["nsga_mutation_rate"] = nsga_cfg.get("mutation_rate")
    exp["nsga_tournament_size"] = nsga_cfg.get("tournament_size")
    exp["nsga_use_seeding"] = nsga_cfg.get("use_seeding")
    exp["nsga_sdsu_kernel"] = nsga_cfg.get("sdsu_kernel")
    exp["des_beta"] = des_cfg.get("beta")
    exp["des_theta"] = des_cfg.get("theta")
    exp["des_trace_enabled"] = des_cfg.get("trace_enabled")
    exp["des_max_time_horizon"] = des_cfg.get("max_time_horizon")

    # aggregated counters
    exp["n_pareto_nsga_unseeded"] = len(exp[exp["method"] == "step07_nsga2"])
    exp["n_pareto_nsga_seeded"] = len(exp[exp["method"] == "step07_nsga2_seeded"])
    exp["n_scenarios_des"] = len(exp)
    return exp


def write_manifest(manifest_path: Path, global_cfg: Dict, steps_cfg: Dict, df_metrics: pd.DataFrame, cfg_paths: ConfigPaths, inputs_used: Iterable[Path]) -> None:
    manifest = {
        "step": "step09_aggregate",
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "config_files": {
            "global_config.toml": str(cfg_paths.output_dir.parent / "global_config.toml"),
            "steps_config.toml": str(cfg_paths.output_dir.parent / "steps_config.toml"),
        },
        "inputs": [str(p) for p in inputs_used],
        "outputs": [str(cfg_paths.metrics_out), str(cfg_paths.experiment_matrix_out), str(cfg_paths.report_out)],
        "settings_effective": {
            "objective": global_cfg.get("objective", {}),
            "rho": global_cfg.get("rho", {}),
            "sdsu": global_cfg.get("sdsu", {}),
            "tokens": global_cfg.get("tokens", {}),
            "milp": global_cfg.get("milp", {}),
            "nsga2": global_cfg.get("nsga2", {}),
            "des": global_cfg.get("des", {}),
            "step09": steps_cfg.get("step09", {}),
        },
        "metrics_summary": {
            "n_rows_metrics": int(len(df_metrics)),
            "n_rows_experiment_matrix": int(len(df_metrics)),
            "pareto_filter_enabled": bool(global_cfg.get("objective", {}).get("report_pareto_only", False)),
            "n_rows_after_pareto": int(len(df_metrics)),
            "per_method_counts": df_metrics["method"].value_counts().to_dict(),
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def write_report(report_path: Path, df_metrics: pd.DataFrame, cfg_paths: ConfigPaths) -> None:
    lines = [
        "# STEP-09 Integrated Analysis",
        "",
        "## Methods Compared",
        "- SDSU (step05)",
        "- MILP (step06_milp)",
        "- NSGA-II (step07_nsga2)",
        "- NSGA-II seeded (step07_nsga2_seeded, if present)",
        "",
        "## Sample of metrics",
    ]
    head = df_metrics[["method", "scenario", "solution_id", "f1", "f2", "f3", "f4"]].head(20)
    lines.append(head.to_markdown(index=False))
    lines.append("")
    lines.append("## Notes")
    lines.append(f"- Metrics written to: {cfg_paths.metrics_out}")
    lines.append(f"- Experiment matrix written to: {cfg_paths.experiment_matrix_out}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def generate_plots(df: pd.DataFrame, cfg_paths: ConfigPaths) -> None:
    if plt is None:
        return
    cfg_paths.plots_dir.mkdir(parents=True, exist_ok=True)
    pairs = [("f1", "f4"), ("f2", "f4"), ("f1", "f2")]
    for x, y in pairs:
        fig, ax = plt.subplots()
        for method, df_m in df.groupby("method"):
            ax.scatter(df_m[x], df_m[y], label=method, alpha=0.7)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.legend()
        fig.tight_layout()
        fig.savefig(cfg_paths.plots_dir / f"{x}_vs_{y}.png")
        plt.close(fig)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="STEP-09 aggregation")
    parser.add_argument("--global-config", type=str, default="global_config.toml")
    parser.add_argument("--steps-config", type=str, default="steps_config.toml")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output_dir from config")
    parser.add_argument("--dataset-dir", type=str, default=None, help="Override dataset_dir from config")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    args = parser.parse_args(argv)

    global_cfg = load_toml(Path(args.global_config))
    steps_cfg = load_toml(Path(args.steps_config))
    cfg_paths = resolve_paths(global_cfg, steps_cfg, args.output_dir, args.dataset_dir)

    df_metrics, pareto_applied = build_metrics_table(cfg_paths, global_cfg)
    df_experiment = build_experiment_matrix(df_metrics, global_cfg, steps_cfg)

    cfg_paths.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    df_metrics.to_csv(cfg_paths.metrics_out, index=False)
    df_experiment.to_csv(cfg_paths.experiment_matrix_out, index=False)

    if not args.no_plots:
        generate_plots(df_metrics, cfg_paths)

    inputs_used = [
        cfg_paths.output_dir / "step05_global_line.csv",
        cfg_paths.output_dir / "step06_milp_opt_global.csv",
        cfg_paths.output_dir / "step07_pareto.csv",
        cfg_paths.output_dir / "step07_global_line.csv",
        cfg_paths.output_dir / "step08_des_metrics.csv",
    ]
    write_manifest(cfg_paths.manifest_out, global_cfg, steps_cfg, df_metrics, cfg_paths, inputs_used)
    write_report(cfg_paths.report_out, df_metrics, cfg_paths)


if __name__ == "__main__":
    main()
