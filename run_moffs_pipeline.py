#!/usr/bin/env python
"""
run_moffs_pipeline.py

Parametric pipeline driver for the MOFFS experiment.

- Modifies global_config.toml:
    [rho].scale                  ← --rho
    [window].ratio               ← --window-ratio
    [tokens].flush_units_ratio   ← --flush-units-ratio
    [paths].output_dir           ← --output-dir

- Then runs (from repo root, in this order):

    python step02_toyota_gs.py global_config.toml steps_config.toml step02_empirical.toml
    python step03_booth_split.py
    python step04_sdsu_per_booth.py
    python step05_merge_global_line.py
    python step06_milp_from_step03.py
    python step06_merge_global_line.py
    python step07_nsga2.py
    python step07_nsga2_seeded.py
    python step08_des.py
    python step09_aggregate.py

All steps read the patched global_config.toml and steps_config.toml.
"""

import argparse
import os
import subprocess
import sys

import toml  # requires `pip install toml`


def patch_global_config(
    cfg_path: str,
    rho_scale: float,
    window_ratio: float,
    flush_units_ratio: float,
    output_dir: str,
) -> None:
    """
    In-place patch of global_config.toml.

    - [rho].scale
    - [window].ratio
    - [tokens].flush_units_ratio
    - [paths].output_dir
    """
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"global_config.toml not found at: {cfg_path}")

    cfg = toml.load(cfg_path)

    # Ensure sections exist
    cfg.setdefault("rho", {})
    cfg.setdefault("window", {})
    cfg.setdefault("tokens", {})
    cfg.setdefault("paths", {})

    cfg["rho"]["scale"] = float(rho_scale)
    cfg["window"]["ratio"] = float(window_ratio)
    cfg["tokens"]["flush_units_ratio"] = float(flush_units_ratio)
    cfg["paths"]["output_dir"] = output_dir

    # Optional: if you rely on ratio-based W, enforce the mode
    # without silently changing semantics if it's set differently.
    window_mode = cfg["window"].get("mode", "ratio")
    if window_mode not in ("ratio", "jobs"):
        raise ValueError(f"Unexpected [window].mode={window_mode!r} in {cfg_path}")
    # If you *want* to force ratio mode, uncomment:
    # cfg["window"]["mode"] = "ratio"

    with open(cfg_path, "w", encoding="utf-8") as f:
        toml.dump(cfg, f)

    # Create output directory upfront so steps don't race on it
    os.makedirs(output_dir, exist_ok=True)


def run_step(cmd: list[str]) -> None:
    """Run one step and fail fast on any non-zero exit code."""
    print(f"\n[PIPELINE] Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, check=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")


def run_pipeline(
    cfg_path: str,
    steps_cfg_path: str,
    step02_empirical_path: str,
    rho_scale: float,
    window_ratio: float,
    flush_units_ratio: float,
    output_dir: str,
    skip_seeded_nsga: bool = False,
) -> None:
    """
    Patch config, then run Step-02 through Step-09 with the new settings.
    """
    print("[PIPELINE] Patching global_config.toml with experiment parameters...", flush=True)
    patch_global_config(
        cfg_path=cfg_path,
        rho_scale=rho_scale,
        window_ratio=window_ratio,
        flush_units_ratio=flush_units_ratio,
        output_dir=output_dir,
    )

    print(
        f"[PIPELINE] Effective parameters:\n"
        f"  [rho].scale                = {rho_scale}\n"
        f"  [window].ratio             = {window_ratio}\n"
        f"  [tokens].flush_units_ratio = {flush_units_ratio}\n"
        f"  [paths].output_dir         = {output_dir}",
        flush=True,
    )

    # Step-02 explicitly documents CLI: global_config, steps_config, step02_empirical
    run_step([sys.executable, "step02_toyota_gs.py", cfg_path, steps_cfg_path, step02_empirical_path])

    # Downstream steps read global_config.toml / steps_config.toml implicitly
    run_step([sys.executable, "step03_booth_split.py"])
    run_step([sys.executable, "step04_sdsu_per_booth.py"])
    run_step([sys.executable, "step05_merge_global_line.py"])

    # MILP per-booth + MILP-based global convoy
    run_step([sys.executable, "step06_milp_from_step03.py"])
    run_step([sys.executable, "step06_merge_global_line.py"])

    # NSGA-II unseeded & seeded
    run_step([sys.executable, "step07_nsga2.py"])
    if not skip_seeded_nsga:
        run_step([sys.executable, "step07_nsga2_seeded.py"])

    # DES + aggregation / analysis
    run_step([sys.executable, "step08_des.py"])
    run_step([sys.executable, "step09_aggregate.py"])

    print("\n[PIPELINE] Completed Steps 02–09.", flush=True)
    print(f"[PIPELINE] Outputs for this run are under: {output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Parametric MOFFS pipeline driver (Steps 02–09).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--config", dest="config", default="global_config.toml",
                   help="Path to global_config.toml")
    p.add_argument("--steps-config", dest="steps_config", default="steps_config.toml",
                   help="Path to steps_config.toml")
    p.add_argument("--step02-empirical", dest="step02_empirical", default="step02_empirical.toml",
                   help="Path to step02_empirical.toml")

    p.add_argument("--rho", dest="rho", type=float, required=True,
                   help="[rho].scale value for this run")
    p.add_argument("--window-ratio", dest="window_ratio", type=float, required=True,
                   help="[window].ratio value (W/N_jobs) for this run")
    p.add_argument("--flush-units-ratio", dest="flush_units_ratio", type=float, required=True,
                   help="[tokens].flush_units_ratio (purge unit cost) for this run")
    p.add_argument("--output-dir", dest="output_dir", type=str, required=True,
                   help="[paths].output_dir for this run (e.g., output_rho05)")

    p.add_argument("--skip-seeded-nsga", dest="skip_seeded_nsga", action="store_true",
                   help="If set, do not run step07_nsga2_seeded.py")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    run_pipeline(
        cfg_path=args.config,
        steps_cfg_path=args.steps_config,
        step02_empirical_path=args.step02_empirical,
        rho_scale=args.rho,
        window_ratio=args.window_ratio,
        flush_units_ratio=args.flush_units_ratio,
        output_dir=args.output_dir,
        skip_seeded_nsga=args.skip_seeded_nsga,
    )


if __name__ == "__main__":
    main()
