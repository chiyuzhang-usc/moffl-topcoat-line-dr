# MOFFS: Multi-Objective Flexible Flow Shop (Automotive Topcoat Line)

This repository implements the full experimental pipeline for a **Multi-Objective Flexible Flow Shop (MO-FFS)** arising from an automotive **topcoat line** with:

- a single feeder / loading stage,
- two parallel paint booths with **sequence-dependent setups (SDST)**,
- a single merged oven conveyor with **global part adjacency (C2)**.

The code integrates:

- a Toyota-style **Grouping & Spacing (GS)** feeder heuristic,
- a per-booth **SDSU** (Sequence-Dependent Set-Ups) algorithm (DP + heuristic),
- an **MILP** baseline for small/medium instances,
- a multi-objective **NSGA-II** search (unseeded and SDSU-seeded),
- a **Discrete-Event Simulation (DES)** for blocking/idle evaluation.

All constraints and semantics are **data-driven** from CSVs and TOML config files; no hard-coded adjacency or paint rules.   

* The paper describes the details of data, formulation, and set up.

---

## 1. Problem & Research Context

The study targets a **2-booth → 1-oven** MO-FFS with:

- **C1 (paint feasibility)** from `s_matrix_transition.csv` + `topcoat_rules.csv`,  
- **C2 (part adjacency)** from `adjacency_rules.csv` with scope semantics,  
- **C3 (purge / DW entry)** encoded in the S-matrix and paint rules.   

Dataset is adapted from:

> *Optimizing Painting Sequence Scheduling Based on Adaptive Partheno-Genetic Algorithm*, Processes, MDPI, 2021. DOI: 10.3390/pr9101714.   

Topcoat “colors” are treated as **topcoat codes**, and constraints are mapped into setup matrices, explicit forbids, and part-adjacency rules. Tokens (`START`, `FLUSH`, `EMPTY`) are used to resolve paint and adjacency conflicts but never appear in the adjacency tables themselves.   


---

## 2. Pipeline Overview (Steps 01–09)

The repository is organized as a **stepwise pipeline**, each step producing CSV outputs, a JSON manifest, and a Markdown report.   

1. **Step-01 — Demand expansion (`step01_expand_jobs.py`)**  
   Expand `demands.csv` into one row per JOB with `job_id`, `part_id`, `topcoat_code`, and provenance metadata.   

2. **Step-02 — Toyota GS feeder (`step02_toyota_gs.py`)**  
   Construct a single **global feeder stream** with tokens (`START`, `FLUSH`, `EMPTY`), enforcing C1/C2 and Toyota-style grouping/spacing with window W and run caps. Output: `step02_paint_seed.csv`.   

3. **Step-03 — Booth split (`step03_booth_split.py`)**  
   Assign each feeder JOB to booth 1 or 2 via a greedy f₃-split:
   \[
     f_3 = |N_1 - N_2| + \rho \cdot \text{cross},
   \]
   where `N1,N2` are booth loads and `cross` counts booth switches along feeder order. Outputs: `step03_booth1.csv`, `step03_booth2.csv`, `step03_assignment.csv`.   

4. **Step-04 — Per-booth SDSU (`step04_sdsu_per_booth.py`)**  
   For each booth, solve a **sequence-dependent setup minimization** problem under C1/C2/C3 using an ATSP-path SDSU kernel (exact DP for small n, greedy + 2-opt heuristic otherwise). Outputs: `step04_booth1_sdsu.csv`, `step04_booth2_sdsu.csv`, plus `step04_flush_positions.json`.   

5. **Step-05 — SDSU global merge (`step05_merge_global_line.py`)**  
   2→1 merge of the JOB-only SDSU booth sequences into a single oven convoy, enforcing **global C2** and optionally inserting post-paint `EMPTY` tokens to break deadlocks, while preserving in-booth order. Output: `step05_global_line.csv`.   

6. **Step-06 — MILP baseline (`step06_milp[_from_step03].py`, `step06_merge_global_line.py`)**  
   Per-booth MILP (open-path ATSP) on Step-03 jobs with SDSU-consistent paint and adjacency semantics, followed by a MILP-based global merge (same C2 model as Step-05, optional post-paint EMPTY). Outputs: `step06_milp_opt_*.csv`, optionally `step06_milp_opt_global.csv`.
  

8. **Step-07 — NSGA-II search (`step07_nsga2.py`, `step07_nsga2_seeded.py`)**  
   Multi-objective NSGA-II with a feasibility-aware decoder using the same paint/adjacency kernels as SDSU/MILP.  
   - **Unseeded:** random initial population (fixed RNG seeds).  
   - **Seeded:** initial solutions from Toyota+SDSU baselines.  
   Outputs Pareto fronts and global convoys (e.g., `step07_pareto.csv`, `step07_global_line*.csv`).   

9. **Step-08 — DES evaluator (`step08_des.py`, `step08_des_plotter.py`)**  
   Discrete-event simulation of the 2-booth + oven line on fixed global convoys (SDSU, MILP, NSGA-II). Computes makespan, blocking, idle fractions, throughput, and composite f₄.   

10. **Step-09 — Aggregation & plots (`step09_aggregate.py`)**  
   Consolidate f₁–f₄ across methods/scenarios into `step09_metrics.csv` and `step09_experiment_matrix.csv`, and generate cross-method plots (f₁ vs. f₄, f₂ vs. f₄, etc.) plus `step09_final_report.md`.   

* a pipeline script that directly runs all steps throughout exists under the same path;
* However, the pipeline would only success if there are feasible solution in all steps, otherwise the script will fail with partial output.

---

## 3. Repository Layout

Canonical layout (relative to the repo root):   

```text
global_config.toml
steps_config.toml
step02_empirical.toml

dataset/
    demands.csv
    topcoat_codebook.csv
    s_matrix_transition.csv
    s_lookup_transition.csv
    topcoat_rules.csv
    adjacency_rules.csv
    cycle_quota.csv
    parts.csv
    fixtures.csv
    line_cadence.csv

step01_expand_jobs.py
step02_toyota_gs.py
step03_booth_split.py
step04_sdsu_per_booth.py
step05_merge_global_line.py
step06_milp_from_step03.py
step06_merge_global_line.py
step07_nsga2.py
step07_nsga2_seeded.py
step08_des.py
step08_des_plotter.py
step09_aggregate.py      # analysis / aggregation (see STEP09 notes)

notes/
    NOTE_MASTER.md
    MOFFS_Protocol.md
    SDSU_Method.md
    STEP01_NOTES.md ... STEP09_NOTES.md

output/
    step01_jobs_expanded.csv
    step02_paint_seed.csv
    step03_booth1.csv
    step03_booth2.csv
    step04_booth1_sdsu.csv
    step04_booth2_sdsu.csv
    step05_global_line.csv
    step06_milp_opt_*.csv
    step06_milp_opt_global.csv (optional)
    step07_pareto.csv
    step07_global_line*.csv
    step08_des_metrics.csv
    step08_trace.csv
    step09_metrics.csv
    step09_experiment_matrix.csv
    manifests/stepXX_manifest.json
    reports/stepXX_report.md
