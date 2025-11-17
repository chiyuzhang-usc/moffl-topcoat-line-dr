#!/usr/bin/env python3
"""
STEP-07 — Seeded runner (Toyota + SDSU) for NSGA-II

Forces seeding ON and writes suffixed outputs so seeded vs unseeded
can be run side-by-side without editing TOML.

Outputs (suffix default "seeded"):
  - output/step07_pareto_<suffix>.csv
  - manifests/step07_manifest_<suffix>.json
  - reports/step07_report_<suffix>.md
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import pandas as pd
import step07_nsga2 as s7


def _with_suffix(path: Path, suffix: str) -> Path:
    name = path.name
    if "." in name:
        stem, ext = name.rsplit(".", 1)
        new_name = f"{stem}_{suffix}.{ext}"
    else:
        new_name = f"{name}_{suffix}"
    return path.with_name(new_name)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="STEP-07 Seeded Runner")
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
    parser.add_argument(
        "--suffix",
        type=str,
        default="seeded",
        help="Suffix for output/report/manifest filenames (default: seeded)",
    )
    parser.add_argument(
        "--max-eval",
        type=int,
        default=None,
        help="Override max_eval_per_seed (optional)",
    )
    args = parser.parse_args(argv)

    # Load configs
    global_cfg_path: Path = args.global_config
    steps_cfg_path: Path = args.steps_config
    global_cfg = s7.load_global_config(global_cfg_path)
    steps_cfg = s7.load_steps_config(steps_cfg_path)

    # Build base paths then suffix the three Step-07 outputs
    paths = s7.build_paths(global_cfg, steps_cfg, global_cfg_path, steps_cfg_path)
    suffix = args.suffix.strip()
    paths = s7.Step07Paths(
        dataset_dir=paths.dataset_dir,
        output_dir=paths.output_dir,
        pareto_path=_with_suffix(paths.pareto_path, suffix),
        manifest_path=_with_suffix(paths.manifest_path, suffix),
        report_path=_with_suffix(paths.report_path, suffix),
        assignment_path=paths.assignment_path,
        booth1_path=paths.booth1_path,
        booth2_path=paths.booth2_path,
        sdsu_booth1_path=paths.sdsu_booth1_path,
        sdsu_booth2_path=paths.sdsu_booth2_path,
        adjacency_rules_path=paths.adjacency_rules_path,
        s_matrix_path=paths.s_matrix_path,
        topcoat_codebook_path=paths.topcoat_codebook_path,
        topcoat_rules_path=paths.topcoat_rules_path,
        cycle_quota_path=paths.cycle_quota_path,
        global_config_path=paths.global_config_path,
        steps_config_path=paths.steps_config_path,
    )

    # Settings: force seeding ON
    settings = s7.build_settings(global_cfg, steps_cfg, args.max_eval)
    settings = s7.Step07Settings(
        tie_break_policy=settings.tie_break_policy,
        repair_L_max=settings.repair_L_max,
        rho_scale=settings.rho_scale,
        window_mode=settings.window_mode,
        window_jobs=settings.window_jobs,
        window_ratio=settings.window_ratio,
        allow_purge=settings.allow_purge,
        flush_cost_units=settings.flush_cost_units,
        population_size=settings.population_size,
        generations=settings.generations,
        crossover_rate=settings.crossover_rate,
        mutation_rate=settings.mutation_rate,
        use_seeding=True,                 # forced
        use_sdsu_kernel=settings.use_sdsu_kernel,
        seeds=settings.seeds,
        max_eval_per_seed=settings.max_eval_per_seed,
    )

    # Load data + paint kernel
    s7.log_progress("loading data and kernels (seeded run)")
    data = s7.load_step07_data(global_cfg, paths, settings)
    kernel = s7.build_paint_kernel(global_cfg, paths, settings)

    # Run per seed and consolidate
    all_inds: List[s7.Individual] = []
    start = time.perf_counter()

    for seed in settings.seeds:
        s7.log_progress(f"seed {seed}: starting seeded NSGA-II run")
        seed_start = time.perf_counter()
        front = s7.nsga2(seed, data, settings, kernel)
        seed_runtime = time.perf_counter() - seed_start
        s7.log_progress(f"seed {seed}: completed in {seed_runtime:.2f}s with {len(front)} solutions")
        for ind in front:
            ind.seed_id = seed
        all_inds.extend(front)

    final_front = s7.global_front(all_inds)
    s7.log_progress(f"global (seeded): pareto frontier consolidated with {len(final_front)} solutions")

    # 1) Pareto CSV
    s7.ensure_parent_dir(paths.pareto_path)
    import pandas as pd
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

    # 2) Booth + merged JOB-only sequences (for DES)
    booth_path, sequences_path = s7.export_pareto_sequences(
        paths, settings, data, kernel, final_front
    )

    # 3) Manifest + report
    runtime = time.perf_counter() - start
    inputs = [
        paths.global_config_path,
        paths.steps_config_path,
        paths.assignment_path,
        paths.booth1_path,
        paths.booth2_path,
        paths.sdsu_booth1_path,
        paths.sdsu_booth2_path,
        paths.adjacency_rules_path,
        paths.s_matrix_path,
        paths.topcoat_codebook_path,
        paths.topcoat_rules_path,
        paths.cycle_quota_path,
    ]
    outputs = [paths.pareto_path, booth_path, sequences_path, paths.manifest_path, paths.report_path]

    s7.write_manifest_and_report(
        paths=paths,
        settings=settings,
        data=data,
        pareto_path=paths.pareto_path,
        inputs=inputs,
        outputs=outputs,
        front=final_front,   # corrected parameter name
        runtime=runtime,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[STEP07-SEEDED] fatal: {exc}")
        raise
