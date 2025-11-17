
# Step‑09 — Integrated Research Analysis & Observational Studies (Final)

> Aggregates artifacts from Steps 01–08 to produce research‑grade analyses: comparative metrics, parameter‑sweep observations, anomaly detection, and narrative findings. It defines repeatable experiments, statistical tests, and report structures. Execution is deferred; this note provides the methodology and exact inputs/outputs.

This step is purely analytical.

---

## 1. Purpose — What / Why

**integration and comparison layer**:

- It **collects** all relevant metrics and sequences from Toyota GS (Step-02), Split (Step-03), SDSU (Step-04), Merge (Step-05), MILP (Step-06), NSGA-II (Step-07), and DES (Step-08).  
- It **maps** method-specific quantities into a common 4-objective vector \( (f_1,f_2,f_3,f_4) \).  
- It **joins** static schedule metrics (setup, mix, split) with dynamic DES metrics (blocking/idle) per method/solution.  
- It **emits** consolidated CSVs and plots so that manual parameter sweeps (multiple runs with different configs) can later be compared by simple concatenation.

All **experiment design** (changing W, ρ, token costs, DES buffer β, NSGA-II settings, etc.) is done **outside** Step-09 by editing TOML and re-running Steps 01–08. STEP-09 just summarizes **whatever configuration produced the current outputs**.

---


## 2. Connections to Neighbor Steps

### 2.1 Upstream (read-only dependencies)

From **configuration**:

- `global_config.toml` — single source of truth for system layout, dataset bindings, tokens, SDSU/MILP/NSGA-II/DES knobs, and `[objective]`.  
- `steps_config.toml` — I/O relpaths including `[step09]`:
  - `metrics_relpath` → `step09_metrics.csv`  
  - `experiment_matrix_relpath` → `step09_experiment_matrix.csv`  
  - `manifest_relpath` → `manifests/step09_manifest.json`  
  - `final_report_relpath` → `reports/step09_final_report.md`  

From **intermediate steps** (all under `output/`):

- STEP-02: `step02_paint_seed.csv`, `manifests/step02_manifest.json`  
- STEP-03: `step03_booth1.csv`, `step03_booth2.csv`, `step03_assignment.csv`, `manifests/step03_manifest.json`  
- STEP-04: `step04_booth1_sdsu.csv`, `step04_booth2_sdsu.csv`, `manifests/step04_manifest.json`  
- STEP-05: `step05_global_line.csv`, `manifests/step05_manifest.json`  
- STEP-06: `step06_milp_opt_*.csv` (per-booth) and `step06_milp_opt_global.csv` (global convoy), `manifests/step06_manifest.json`  
- STEP-07:
  - `step07_pareto.csv` (unseeded)  
  - `step07_global_line.csv` (unseeded sequences)  
  - Optionally `step07_pareto_seeded.csv`, `step07_global_line_seeded.csv` from `step07_nsga2_seeded.py`  
  - `manifests/step07_manifest*.json`  
- STEP-08:
  - `step08_des_metrics.csv` (per-scenario dynamic metrics)  
  - Optional `step08_trace.csv`  
  - `manifests/step08_manifest.json`  

### 2.2 Result

STEP-09 closes the pipeline, the outputs and plots are **directly used** in the paper/thesis:
- Tables comparing **Toyota/SDSU, MILP, NSGA-II (seeded/unseeded)** on \( (f_1,f_2,f_3,f_4) \).  
- Plots visualizing **trade-offs and dominance** across methods.  
- A Markdown report that stitches together numerical results and qualitative observations.

---

## 3. Inputs / Outputs and File Contracts

### 3.1 Inputs

All paths are relative to `paths.output_dir` in `global_config.toml`.

Required:

- `step05_global_line.csv` — canonical SDSU-based global oven convoy (jobs + possibly `EMPTY`).  
- `step06_milp_opt_global.csv` — MILP-based global convoy (jobs + possibly `EMPTY`).  
- `step07_pareto.csv` — NSGA-II Pareto set with columns:
  - `solution_id`, `seed`, `origin`, `feasible`, `f1`, `f2`, `f3`.  
- `step07_global_line.csv` — NSGA-II global sequences with columns:
  - `solution_id`, `global_position`, `booth_label`, `booth_id`, `job_index`, `job_id`, `feeder_index`, `part_id`, `topcoat`.  
- `step08_des_metrics.csv` — DES metrics with columns:  
  `method`, `scenario`, `solution_id`, `seed`,  
  `throughput_jobs`, `tokens_empty`, `makespan`,  
  `pb1_busy_time`, `pb1_blocked_time`, `pb1_idle_time`,  
  `pb2_busy_time`, `pb2_blocked_time`, `pb2_idle_time`,  
  `oven_busy_time`, `oven_idle_time`,  
  `pb_block_frac_max`, `oven_idle_frac`, `f4`.

Optional but used when present:

- `step07_pareto_seeded.csv`, `step07_global_line_seeded.csv` — seeded NSGA-II.  
- All manifests and reports from Steps 01–08 (for cross-checking and for embedding summaries into the final report).

### 3.2 Outputs

Under `output/`:

- `step09_metrics.csv` — **scenario-level objective table** joining static \(f_1,f_2,f_3\) and dynamic \(f_4\) for each method/solution.  
- `step09_experiment_matrix.csv` — **scenario-level design matrix**: same keys as `step09_metrics.csv` plus the resolved configuration knobs (W, ρ, token ratios, β, θ, NSGA-II settings, etc.).  
- `manifests/step09_manifest.json` — full reproducibility record for this aggregation.  
- `reports/step09_final_report.md` — human-readable analysis and figure references.  
- Optional: a small set of plots under `output/step09_plots/` (see §7).

---


## 4. Canonical Objectives and How STEP-09 Computes Them

STEP-09 expresses every method in a **common objective space**:

- \(f_1\): paint setup units (including purge/FLUSH cost) across both booths.  
- \(f_2\): windowed mix quality on the **final global line**.  
- \(f_3\): booth balance / cross structure inherited from STEP-03 and method-specific adjustments.  
- \(f_4\): DES-based blocking/idle composite, coming from STEP-08.

All methods are treated as **minimization** problems for these four objectives.

### 4.1 Objective settings from `global_config.toml`

From `[objective]`:

- `window_W_frac` — fraction of total jobs used to derive the evaluation window size \(W\):  
  \[
    W = \max\{1,\; \lfloor \text{window\_W\_frac} \cdot N \rceil \}
  \]
  where \(N\) is the number of **jobs** (not counting `EMPTY`).  
- `report_pareto_only` — if true, `step09_metrics.csv` is reduced to the **global nondominated set** across all methods on \( (f_1,f_2,f_3,f_4) \).  

### 4.2 \(f_1\) — Paint Setup Units

Per method:

- **Toyota GS + SDSU baseline (`method = "step05"`):**  
  \( f_1^{\text{SDSU}} \) is taken as the sum of per-booth SDSU setup costs in the STEP-04 manifest, measured in paint units and already combining color changes and purge units under `[sdsu]` and `[tokens]`.  
- **MILP (`method = "step06_milp"`):**  
  \( f_1^{\text{MILP}} \) is taken from the STEP-06 manifest as the sum of `f1_milp` over the two full-booth instances used to build `step06_milp_opt_global.csv`.  
- **NSGA-II unseeded (`method = "step07_nsga2"`) and seeded (`"step07_nsga2_seeded"`):**  
  \( f_1^{\text{NSGA}} \) is taken from `step07_pareto*.csv` (`f1` column) per `solution_id`.  

In all cases, Step-09 **does not re-solve** any optimization model; it only **normalizes** these per-method costs into a single column `f1`.

### 4.3 \(f_2\) — Windowed Mix Quality on the Merged Line

For any global convoy \(G = (g_1,\dots,g_N)\) (jobs only; `EMPTY` removed when computing mix):

1. Compute the evaluation window size \(W\) from `[objective].window_W_frac`.  
2. For each contiguous window \(G[i:i+W-1]\), compute a **mix imbalance** measure, e.g.  
   \[
   \Delta(i) = \sum_{c} \left| p_{c}(i) - p_{c}^{\star} \right|
   \]
   where \(p_{c}(i)\) is the fraction of jobs in family/color \(c\) in window \(i\), and \(p_c^{\star}\) is the global fraction or a family target inferred from data.  
3. Aggregate over windows, e.g.  
   \[
   f_2 = \frac{1}{N-W+1} \sum_{i} \Delta(i).
   \]

STEP-09 **recomputes \(f_2\)** with the same \(W\) and imbalance formula for:

- `step05_global_line.csv` — SDSU baseline.  
- `step06_milp_opt_global.csv` — MILP-based oven schedule.  
- Each NSGA-II solution in `step07_global_line*.csv`.

This guarantees that all methods are compared using the **same** mix metric, even if their internal evaluation used a different window setting.

### 4.4 \(f_3\) — Booth Balance / Split Structure

Baseline:

- STEP-03 yields a split metric (e.g. `f3_split`) based on job counts and cross structure along the feeder.  

STEP-09 uses:

- For SDSU and MILP, \( f_3 \) is inherited from the STEP-03 metric, because both keep the **same job sets per booth** (they reorder but do not reassign jobs to other booths).  
- For NSGA-II, \( f_3 \) is read from `step07_pareto*.csv` (`f3` column), which already encodes the effect of any booth-reassignment logic used in NSGA-II.

Thus, `f3` in `step09_metrics.csv` is populated from:

- STEP-03 manifest (for `step05` and `step06_milp`).  
- `step07_pareto*.csv` (for NSGA-II).

### 4.5 \(f_4\) — DES-Based Blocking/Idle Composite

STEP-08 writes `f4` and its components to `step08_des_metrics.csv`. For each scenario row:

- `f4` — composite cost constructed in DES, combining max booth blocking fraction and oven idle fraction under the `[des].f4_weights` policy (if used).  
- `makespan`, `pb_block_frac_max`, `oven_idle_frac`, etc. — raw dynamic metrics.

STEP-09 simply **copies** DES’s `f4` into the canonical `f4` column, and also retains the individual components in extra columns for richer analysis.

No DES is re-run in STEP-09.


## 5. `step09_metrics.csv` and `step09_experiment_matrix.csv` Schemas

### 5.1 `step09_metrics.csv` — Scenario-Level Objective Table

Each row corresponds to **one evaluated scenario** (method + solution_id). Columns:

- Keys and identifiers:
  - `method` ∈ {`step05`, `step06_milp`, `step07_nsga2`, `step07_nsga2_seeded`}  
  - `scenario` — from `step08_des_metrics.csv` (e.g. `step07_global_line::sid3`)  
  - `solution_id` — `solution_id` from NSGA-II or a fixed tag for baselines  
  - `seed` — RNG seed when available (NSGA-II)  

- Canonical objectives:
  - `f1` — paint setup units (normalized as in §4.2)  
  - `f2` — windowed mix metric recomputed on the final convoy (§4.3)  
  - `f3` — split / balance metric (§4.4)  
  - `f4` — DES composite cost from STEP-08 (§4.5)  

- DES details (copied from `step08_des_metrics.csv`):
  - `makespan`, `throughput_jobs`, `tokens_empty`  
  - `pb_block_frac_max`, `oven_idle_frac`  
  - `pb1_blocked_time`, `pb2_blocked_time`, `oven_idle_time`  
  - Any additional time or fraction fields present in DES metrics.  

If `[objective].report_pareto_only = true`, STEP-09 computes the **global nondominated subset** on \((f_1,f_2,f_3,f_4)\) and keeps only those rows.

### 5.2 `step09_experiment_matrix.csv` — Scenario-Level Design Matrix

Same row keys (`method`, `scenario`, `solution_id`, `seed`) as in `step09_metrics.csv`, plus selected **configuration knobs** resolved from TOML:

- From `[objective]`: `window_W_frac`, `report_pareto_only`.  
- From `[rho]`: `scale` (ρ for STEP-03).  
- From `[sdsu]`: `dp_threshold_jobs`, `max_2opt_passes`, `allow_purge`, `purge_cost_weight`.  
- From `[tokens]`: `flush_units_ratio`, `empty_time_ratio` and related knobs actually used.  
- From `[milp]`: `time_limit_sec`, `mip_gap`, `max_jobs_per_instance`, `threads`.  
- From `[nsga2]`: `population_size`, `generations`, `crossover_rate`, `mutation_rate`, `tournament_size`, `use_seeding`, `sdsu_kernel`.  
- From `[des]`: `beta`, `theta`, `trace_enabled`, `max_time_horizon`.  

Additional aggregated counters per run (one per row):

- `n_pareto_nsga_unseeded` — number of unseeded NSGA-II Pareto points.  
- `n_pareto_nsga_seeded` — number of seeded NSGA-II Pareto points (if present).  
- `n_scenarios_des` — number of DES scenarios evaluated.  

For multi-experiment studies, the researcher **concatenates** `step09_metrics.csv` and `step09_experiment_matrix.csv` from multiple runs (each run with its own TOML settings) to form a large design-of-experiments table.


---

## 6. High-Level Algorithm of STEP-09

Conceptual pipeline for `step09_aggregate.py`:

1. **Load configs.** Read `global_config.toml` and `steps_config.toml`. Resolve paths for all inputs and outputs (using `[paths]` and `[step09]`).  
2. **Load DES metrics.** Read `step08_des_metrics.csv` into a DataFrame `D_des`. This defines the set of scenarios (method + solution_id) that have valid \(f_4\).  
3. **Load static metrics per method.**
   - For SDSU baseline: read STEP-04 manifest and sum per-booth setup costs to obtain \(f_1\); read STEP-03 manifest for \(f_3\).  
   - For MILP: read STEP-06 manifest for per-instance `f1_milp` and aggregate to a global \(f_1\); reuse STEP-03 split metrics for \(f_3\).  
   - For NSGA-II: read `step07_pareto*.csv` for \(f_1,f_2,f_3\) per `solution_id`.  
4. **Recompute \(f_2\) on final convoys.**  
   - Derive window size \(W\) from `[objective].window_W_frac`.  
   - For each relevant convoy file (`step05_global_line.csv`, `step06_milp_opt_global.csv`, `step07_global_line*.csv`), recompute \(f_2\) using the same mix imbalance function and the same \(W\).  
5. **Build canonical rows.** For every scenario present in `D_des`, create a row with:
   - `method`, `scenario`, `solution_id`, `seed`  
   - `f1` (from the appropriate source)  
   - `f2` (recomputed)  
   - `f3` (split metric)  
   - `f4` + all DES metrics (from `D_des`).  
6. **Apply Pareto filter (optional).** If `report_pareto_only = true`, retain only rows that are not dominated in \((f_1,f_2,f_3,f_4)\).  
7. **Write `step09_metrics.csv`.** Save the consolidated table sorted by `method`, then `scenario`, then `solution_id`.  
8. **Build and write `step09_experiment_matrix.csv`.**  
   - For each row, attach the resolved config knobs (§5.2) and aggregated counters.  
   - Save with the same ordering and a superset of keys.  
9. **Emit manifest and report.**  
   - Manifest: I/O SHA-256, config snapshot, row counts, Pareto filtering status.  
   - Report: narrative summary, key tables, and pointers to plots.


---

## 7. Plots and Tables Produced by STEP-09

STEP-09 complements `step08_des_plotter.py` with **cross-method plots** that combine static and dynamic metrics. Recommended minimal set:

- **Scatter: \(f_1\) vs \(f_4\)**, colored by `method` (SDSU, MILP, NSGA-II unseeded, NSGA-II seeded). This shows paint setup vs blocking/idle trade-off.  
- **Scatter: \(f_2\) vs \(f_4\)**, colored by method. This highlights mix quality vs congestion.  
- **Scatter: \(f_1\) vs \(f_2\)**, marking DES-feasible points with different markers (e.g., those present in `step08_des_metrics.csv`).  
- **Parallel-coordinate or radar-style plot** over normalized \(f_1,f_2,f_3,f_4\) for a small selected subset (e.g., global nondominated set), giving a compact visual of trade-offs.  

Tabular summaries in `step09_final_report.md`:

- A table of **best-known points per method** (min \(f_1\), min \(f_4\), and representative Pareto points).  
- Selected **paired comparisons**:
  - SDSU baseline vs MILP on \(f_1\) and \(f_2\) at similar \(f_4\).  
  - NSGA-II seeded vs unseeded on coverage of the nondominated set and best vs worst DES performance.

The plotting code lives inside `step09_aggregate.py` (or helper modules), reusing `matplotlib` as in `step08_des_plotter.py` and using the same color convention for `method`.

---

## 8. Manifest and Report Structure

### 8.1 `step09_manifest.json`

Fields (superset of earlier steps):

- `step`: `"step09_aggregate"`  
- `timestamp`  
- `config_files`: relative paths + SHA-256 for `global_config.toml`, `steps_config.toml`  
- `inputs`: relative paths + SHA-256 for every file actually read (all step CSVs + manifests used)  
- `outputs`: relative paths + SHA-256 for `step09_metrics.csv`, `step09_experiment_matrix.csv`, report, and any plots  
- `settings_effective`: snapshot of `[objective]` and any additional knobs used in this analysis  
- `metrics_summary`:  
  - `n_rows_metrics`, `n_rows_experiment_matrix`  
  - `pareto_filter_enabled` (bool) and `n_rows_after_pareto`  
  - per-method counts of scenarios and nondominated scenarios  

### 8.2 `step09_final_report.md`

Suggested structure, consistent with MOFFS protocol:

1. **Purpose / What / Why** — restate the aggregation role and link to research questions.  
2. **Inputs and Configuration Snapshot** — list key config values that define this run (W, ρ, token ratios, NSGA-II/DES knobs).  
3. **Methods Compared** — SDSU baseline, MILP, NSGA-II unseeded, NSGA-II seeded.  
4. **Objective Tables** — embed selected slices of `step09_metrics.csv` (e.g., first few global nondominated points).  
5. **Plots** — include the generated figures with captions explaining trade-offs.  
6. **Observations and Interpretation** — text discussing:
   - Where MILP improves over SDSU.  
   - How NSGA-II explores the trade-off surface.  
   - How DES metrics reshape the ranking of schedules.  
7. **Diagnostics and Reproducibility** — number of scenarios, any missing data, manifest hashes, and explicit instructions to reproduce this exact aggregation run.



## 9. Acceptance Criteria and Failure Modes

### 9.1 Acceptance Criteria

STEP-09 is considered successful when:

- All declared inputs in §3 are found and readable.  
- `step09_metrics.csv` contains at least one row for each method that has DES metrics.  
- If `report_pareto_only = true`, the Pareto filter is applied and `n_rows_after_pareto > 0`.  
- Manifest and report are written and contain consistent hashes and summaries.

### 9.2 Failure Modes and Handling

Typical issues to guard against:

- **Missing upstream artifacts** — e.g., DES not run, NSGA-II outputs absent. STEP-09 must fail with a clear message naming the missing file, without writing partial metrics.  
- **Schema mismatches** — unexpected columns in `step07_pareto*.csv` or `step08_des_metrics.csv`. The script must validate column sets and abort with an explicit explanation if they do not match expectations.  
- **Inconsistent scenario keys** — NSGA-II solution_id present in Pareto file but absent in DES metrics (or vice versa). Those rows must be reported and either dropped or resolved deterministically, with the decision documented in the report.  
- **Degenerate objective data** — NaN or infinite values in \(f_1\)–\(f_4\). Such scenarios should be flagged, excluded from Pareto filtering, and listed in a diagnostics section.


