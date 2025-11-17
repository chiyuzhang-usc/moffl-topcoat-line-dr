#!/usr/bin/env python
"""
sweep_moffs_experiments.py

Sweeps MO-FFS parameters by repeatedly calling run_moffs_pipeline.run_pipeline:

    1) W sweep:
        window.ratio ∈ {0.10, 0.15, 0.25}, holding
            rho.scale = 0.5
            tokens.flush_units_ratio = 0.5

    2) ρ sweep:
        rho.scale ∈ {0.0, 0.5, 1.0, 2.0}, holding
            window.ratio = 0.10
            tokens.flush_units_ratio = 0.5

    3) Purge sensitivity:
        tokens.flush_units_ratio ∈ {0.5, 1.0, 2.0, 5.0}, holding
            rho.scale = 0.5
            window.ratio = 0.10

Each unique (rho, W, flush_units_ratio) triple is run once, with a distinct
output_dir of the form:

    sweep_rho<rho>_W<W>_flush<flush>

Values are encoded with 'p' instead of '.' (e.g. 0.5 → "0p5").
"""

import os
from typing import List, Tuple

from run_moffs_pipeline import run_pipeline


# ---- Baseline and sweep sets -------------------------------------------------

BASE_RHO: float = 0.5
BASE_W: float = 0.10
BASE_FLUSH: float = 0.5

W_VALUES: List[float] = [0.10, 0.15, 0.25]
RHO_VALUES: List[float] = [0.0, 0.5, 1.0, 2.0]
FLUSH_VALUES: List[float] = [0.5, 1.0, 2.0, 5.0]

# This is the already-run baseline (with output_dir="output_rho05")
BASE_TRIPLE: Tuple[float, float, float] = (BASE_RHO, BASE_W, BASE_FLUSH)


# ---- Helpers -----------------------------------------------------------------


def format_val(x: float) -> str:
    """
    Encode a float into a compact, filesystem-safe token.

    Examples:
        0.5   -> "0p5"
        0.10  -> "0p1"
        2.0   -> "2"
    """
    s = f"{x:.2f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def make_output_dir_name(rho: float, w: float, flush: float) -> str:
    """
    Construct a unique directory name for a given triple.
    """
    rho_str = format_val(rho)
    w_str = format_val(w)
    flush_str = format_val(flush)
    return f"sweep_rho{rho_str}_W{w_str}_flush{flush_str}"


def build_experiment_list(
    include_baseline: bool = False,
) -> List[Tuple[float, float, float]]:
    """
    Build an ordered list of unique (rho, W, flush) triples according to the
    specified sweeps. Optionally include the baseline triple.
    """
    combos: List[Tuple[float, float, float]] = []

    # 1) W sweep
    for w in W_VALUES:
        triple = (BASE_RHO, w, BASE_FLUSH)
        if not include_baseline and triple == BASE_TRIPLE:
            continue
        combos.append(triple)

    # 2) rho sweep
    for rho in RHO_VALUES:
        triple = (rho, BASE_W, BASE_FLUSH)
        if not include_baseline and triple == BASE_TRIPLE:
            continue
        combos.append(triple)

    # 3) purge sensitivity sweep
    for flush in FLUSH_VALUES:
        triple = (BASE_RHO, BASE_W, flush)
        if not include_baseline and triple == BASE_TRIPLE:
            continue
        combos.append(triple)

    # Deduplicate while preserving order
    seen = set()
    unique_combos: List[Tuple[float, float, float]] = []
    for triple in combos:
        if triple in seen:
            continue
        seen.add(triple)
        unique_combos.append(triple)

    return unique_combos


# ---- Main sweep driver -------------------------------------------------------


def main() -> None:
    # Ensure working directory is repo root (where configs and steps live)
    repo_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(repo_root)

    # Build experiments (baseline omitted; you already ran output_rho05)
    experiments = build_experiment_list(include_baseline=False)

    print("[SWEEP] Total experiments to run:", len(experiments), flush=True)

    for idx, (rho, w, flush) in enumerate(experiments, start=1):
        out_dir = make_output_dir_name(rho, w, flush)

        print(
            f"\n[SWEEP] Experiment {idx}/{len(experiments)}\n"
            f"  rho.scale                = {rho}\n"
            f"  window.ratio             = {w}\n"
            f"  tokens.flush_units_ratio = {flush}\n"
            f"  output_dir               = {out_dir}",
            flush=True,
        )

        # One full pipeline run for this triple
        run_pipeline(
            cfg_path="global_config.toml",
            steps_cfg_path="steps_config.toml",
            step02_empirical_path="step02_empirical.toml",
            rho_scale=rho,
            window_ratio=w,
            flush_units_ratio=flush,
            output_dir=out_dir,
            skip_seeded_nsga=False,  # keep seeded+unseeded NSGA-II
        )

    print("\n[SWEEP] All experiments completed.", flush=True)


if __name__ == "__main__":
    main()
