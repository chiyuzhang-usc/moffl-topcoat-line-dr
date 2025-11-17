# MOFFS Research Procedure Documentation

**Multi-Objective Flexible Flow Shop with 2 Paint Booths and 1 Oven**

**Version:** 03
**Purpose:** This document describes the procedure for the Multi-Objective Flexible Flow Shop (MOFFS) study using the prepared dataset.
It fixes **method**, **parameters**, **I/O**, and **reproducibility** so runs can be resumed in a new session without context loss.

**Dataset Source:** Primary source dataset is adapted from: [Optimizing Painting Sequence Scheduling Based on Adaptive Partheno-Genetic Algorithm](https://www.mdpi.com/2227-9717/9/10/1714) published in Processes, MDPI, 2021. [DOI: 10.3390/pr9101714](https://doi.org/10.3390/pr9101714).
To stay faithful to the original data, "color" are named "topcoat" without loss of meaning.

Constraints are adapted from the original dataset, with relaxed token permitted.

The paintshop setup is assumed to have multiple paint-booth, tailed by oven for curing, different from the machine setup of the origin of the dataset.

Semantics, parameters, IO contracts, step gating are defined and executed strictly to ensure the experiments can be repeated from scratch.

---

## Code and Method Principles

1. **Protocol supremacy**  
   All hard constraints (C1/C2/C3), units, and file semantics come from:
   - `global_config.toml` and `steps_config.toml`
   - `s_matrix_transition.csv`, `topcoat_rules.csv`, `adjacency_rules.csv`, `cycle_quota.csv`, `line_cadence.csv`  

   The rules for both painting and adjacency can be "reset" by inserting tokens, which has cost.

2. **C1/C2 separation**  
   - **C1 (paint):** topcoat-to-topcoat feasibility and setup weights at paint, derived from `s_matrix_transition.csv` plus `topcoat_rules.csv`.  
   - **C2 (parts):** part-to-part adjacency forbids, derived solely from `adjacency_rules.csv` with scope semantics.  

   Painting and part adjacency are logically distinct.

3. **C2 handling**  
   - C2 is **part-driven and global** wherever a single-file JOB stream exists (Toyota feeder from Step‑02, merged oven convoy from Step‑05/06).  
   - `adjacency_rules.csv` is always interpreted through `data_schema.adjacency_rules` in `global_config.toml`; there is no implicit rule.  
   - Scope escalation is controlled only by `[part_rules]` and `[adjacency]` in `global_config.toml`, not by code intuition.

4. **Tokens and modes**  
   - Tokens: `START`, `FLUSH`, `EMPTY`. They are symbols with costs but never appear in the adjacency rule tables.  
   - The C2 relation is defined on **(from_part_id, to_part_id)** only.  
   - Each step chooses a **C2 evaluation mode**:
     - `"parts_only"`: adjacency checks see parts as if tokens were invisible; tokens do not separate forbidden pairs.  
     - `"symbol_stream"`: adjacency checks operate on the realized symbol stream; tokens can separate parts (e.g., `A → EMPTY → B` avoids an `A → B` forbid because `A → B` is no longer an immediate pair).  
   - The mode is a policy choice, configured via CLI or `steps_config.toml` and defaulted in `step06_merge_global_line.py` (Step‑05 → `parts_only`, Step‑06 → `symbol_stream`).

5. **Step gating**  
   Every step emits:
   - a **manifest JSON** with SHA‑256 of inputs/outputs, resolved settings, metrics, diagnostics, runtime  
   - a **Markdown report** with the structure: Purpose / What / Why / Config snapshot / I/O hashes / Settings / Summary / Diagnostics / Runtime / Determinism  

   Steps only hand off to the next step if their acceptance criteria are satisfied (0 C1/C2 violations, no degenerate token behavior, etc.).

6. **Truth in data**  
   If data and config disagree, the dataset wins: functions must fail with a certificate instead of silently fabricating rules or default behaviors that change semantics.

---

## 1. System & Modeling Boundary

- **Topology:** 2 identical paint booths → 1 merged oven conveyor (2→1).  
- **Stages modeled:**  
  - Painting booths (C1 + SDST, purge via FLUSH)  
  - Global oven line (C2, sequence fixed; timing via DES)  
- **Processing times:**  
  - Paint is in “paint units”; the S‑matrix entries and `flush_units_ratio` are in these units.  
  - DES uses either constant cycle times or part/topcoat‑dependent times via `line_cadence.csv` and `[des]` in `global_config.toml`.

Core conceptual parameters:

Parameter | Meaning
--------- | -------
ρ (rho) | load weight for cross‑booth switches in Step‑03 split, scaled by `[rho].scale`
W | window size (jobs) for mix/spacing on the feeder and global line, resolved from `[window]` and optionally `step02_empirical.toml`
τ_flush | purge cost in paint units, from `[paint_rules]`
β | buffer capacity between paint booths and oven (jobs), from `[des].buffer_beta`
θ | service‑time ratio oven/paint, from `[des]` or `line_cadence.csv`

A non‑degenerate baseline grid for experiments (not hard‑coded):

- ρ_scale ∈ {0.5, 1, 2}  
- W / N_jobs ∈ {0.10, 0.15}  
- τ_flush ∈ {1, 3}  
- β ∈ {0, 2}  
- θ ∈ {1, 1.5}

---

## 2. Data Contract

All I/O data files are using CSV format, 
refer to `global_config.toml[data_schema.*]` for configurations and file names.

### 2.1 Datasets

File | Purpose | Key columns | Notes
---- | ------- | ----------- | -----
`demands.csv` | Job quantities by part and topcoat | `part_id, topcoat_code, quantity` | Step‑01 expands each row into individual jobs.
`topcoat_codebook.csv` | Topcoat metadata | `topcoat_code, ...` | Used for reporting and mix aggregation.
`topcoat_rules.csv` | C1 explicit forbids | `from_topcoat_code, to_topcoat_code, relation` | Rows with `relation` containing “forbid” define hard paint forbids.
`s_matrix_transition.csv` | SDST base matrix | `from` row + one col per `to_topcoat_code` | Finite entries = setup units; `INF`/blank = forbidden transition.
`s_lookup_transition.csv` | Optional SDST table | `from_topcoat_no, to_topcoat_no, setup_time` | Used only if configured.
`adjacency_rules.csv` | C2 part adjacency forbids | `from_part_id, to_part_id, scope, relation` | `relation` containing “forbid” defines forbidden successive part pairs with scope semantics.
`cycle_quota.csv` | Mix targets | `part_id, quota` | Optional f2 targets (part‑level).
`parts.csv` | Part metadata | `part_id, ...` | Reporting/joins only.
`fixtures.csv` | Fixture compatibility | `part_id, bracket_number, ...` | Metadata only in this protocol.
`line_cadence.csv` | Timing / DES input | `cycle_time_min, skids_total, brackets_per_skid, lead_time_hours` | Optional; DES can fall back to `[des]`.

### 2.2 Adjacency scopes (C2)

`adjacency_rules.scope` values:

- `global`: applies both in‑booth and on merged line when C2 is enabled  
- `in_booth`: applies only inside each booth sequence  
- `across_skids`: applies only on the merged global line  

Escalation is controlled by:

```toml
[part_rules]
enforce_in_booth   = true
enforce_on_merged  = true
scope_escalation   = "global_if_missing"   # or "none"

[adjacency]
enabled           = true
scope_escalation  = "global_if_missing"
```

If `scope_escalation = "global_if_missing"` and a row has empty scope, treat it as `global` on both booths and merged line.

---

## 3. Constraints C1–C3

### 3.1 C1 — Paint feasibility (topcoat domain, hard)

For each booth `b` and its topcoat sequence `(τ₁, τ₂, …, τ_nb)`:

- Each adjacent pair `(τ_k, τ_{k+1})` must be allowed by `s_matrix_transition` and not forbidden by `topcoat_rules`.  
- `INF`/missing in S‑matrix or an explicit “forbid” in `topcoat_rules` is a hard C1 violation.

Purge via `FLUSH`:

- Allowed targets for `FLUSH → t` are constrained by `[paint_rules].flush_allowed_targets` (`"DW_only"`, `"any"`, or explicit list).  
- Cost of FLUSH is given by `flush_cost_time` plus `flush_enter_cost` for arcs entering `FLUSH`.  
- `flush_requires_change = true` forbids `u → FLUSH → u`.

C1 is enforced in **SDSU**, **MILP**, and **NSGA‑II decoder**. Downstream steps do not re‑check paint feasibility.

### 3.2 C2 — Part adjacency (hard, scope‑dependent)

Let `p_i` denote part_ids.

- In a booth sequence `(p₁, …, p_nb)`, any forbidden pair `(p_i, p_{i+1})` with scope ∈ {`in_booth`, `global`} is disallowed.  
- On the merged global line µ = `(µ₁, …, µ_N)` (jobs only), any forbidden pair `(µ_i, µ_{i+1})` with scope ∈ {`across_skids`, `global`} is disallowed.

Evaluation modes:

- **`parts_only`**: adjacency checks are applied to consecutive **parts** after collapsing tokens; tokens do not separate forbidden part pairs.  
- **`symbol_stream`**: checks are applied to adjacent **symbols**; tokens can separate parts, so `A → EMPTY → B` does not trigger a forbid even if `(A,B)` is forbidden, because `A` and `B` are no longer immediate successors.

C2 is enforced:

- At Step‑02 on the global feeder (symbol stream with tokens, but C2 logic is effectively parts‑only per Step‑02 notes).  
- At Step‑04 within booths (SDSU).  
- At Step‑05/06 on the merged line with the configured `c2_mode`.

### 3.3 C3 — Purge / DW entry

The DW/pretreatment entry rule is encoded in the S‑matrix:

- Direct `u → DW` is allowed only if `S[u, DW]` is finite and not forbidden by `topcoat_rules`.  
- Otherwise, the allowed route is `u → FLUSH → DW` (or other allowed target under `flush_allowed_targets`).

C3 is enforced by SDSU (Step‑04), MILP (Step‑06), and NSGA‑II decoder (Step‑07).

---

## 4. Objectives

All objectives are defined on JOB‑only streams; tokens contribute to f1 but do not count as parts for f2/f3 windows or DES WIP.

Let `G = (g₁, …, g_N)` be the global JOB sequence from Step‑05/06.

### 4.1 f1 — Setup and token units

- Base setup cost:  
  \[
    	ext{Setup} = \sum_{(i	to j)\ 	\text{adjacent jobs in booth sequences}} s[	\tau(i), 	\tau(j)].
  \]

- Token units:
  - Each `FLUSH` contributes `flush_units_ratio`.  
  - Each `EMPTY` contributes `empty_units_prepaint` or `empty_units_postpaint_ratio` depending on stage.

We report:

- `f1_paint` (setup units from S‑matrix)  
- `f1_tokens` (token units converted to equivalent setup units)  
- `f1_total = f1_paint + f1_tokens`

SDSU/MILP focus on minimizing f1; NSGA‑II uses f1 as one coordinate of a vector objective.

### 4.2 f2 — Windowed mix deviation on global line

Let W be window length (jobs) from `[window]` / Step‑02.

For each class `t` (part or topcoat family) with target fraction π_t:

- Expected count in a window: `E_t = W · π_t`.  
- For each sliding window k, let `n_t(k)` be actual count of class `t` (jobs only).

A generic deviation measure:

\[
  f_2 = \sum_k \sum_t \max\{0,\ |n_t(k) - E_t| - \varepsilon_t\}
\]

where `ε_t` are tolerances.

Step‑02 enforces a window discipline on the feeder; Step‑05/06 and NSGA‑II re‑evaluate f2 on the merged convoy.

### 4.3 f3 — Booth workload balance

At Step‑03:

- `N1, N2` = job counts for booths 1 and 2  
- `cross` = number of PB1↔PB2 switches along the feeder order

Objective:

\[
  f_3^{	\text{split}} = |N_1 - N_2| + \rho \cdot 	\text{cross},
\]

with `ρ` scaled from `[rho].scale`.

NSGA‑II recomputes f3 for each decoded solution.

### 4.4 f4 — DES blocking and idle

Step‑08 DES simulates:

- Booth blocking times  
- Oven idle time  
- Makespan, average WIP and buffer occupancy  

under buffer `β` and timing ratio `θ`.

A composite form:

\[
  f_4 = w_{	\text{idle}} \cdot \frac{	\text{idle}_O}{\sum_j 	\tau_O(j)} +
        w_{	\text{block}} \cdot \sum_b \frac{	\text{block}_b}{\sum_{j\in S_b} 	\tau_P(j)},
\]

with raw components and a normalized `f4_norm` also reported.

---

## 5. Configuration Files

### 5.1 `global_config.toml`

Primary config; key sections:

- `[paths]` — `dataset_dir`, `output_dir`, etc.  
- `[data_files]` — filenames for all CSVs.  
- `[data_schema.*]` — column synonyms by dataset; scripts must use these, never hard‑code names.  
- `[paint_rules]` — purge semantics (`use_start_token`, `use_flush_token`, `flush_allowed_targets`, `flush_cost_time`, `flush_enter_cost`, `flush_requires_change`).  
- `[part_rules]` — in‑booth / merged enforcement and `scope_escalation`.  
- `[merged_rules]` — whether to re‑enforce paint rules at merge (normally `false`).  
- `[window]`, `[spacing]` — Toyota GS parameters.  
- `[tokens]` — `flush_units_ratio`, `empty_units_prepaint`, `empty_units_postpaint_ratio`.  
- `[adjacency]` — `enabled`, `fifo_lookahead`, `scope_escalation`.  
- `[rho]` — scale factor for f3.  
- `[sdsu]`, `[milp]`, `[nsga2]`, `[des]` — method‑specific sections.

### 5.2 `step02_empirical.toml`

Empirical shaping for Step‑02 (window ratio, run caps, heuristic weights, target mix approximations), without changing hard constraints.
This is used for testing model under different data parrameters

### 5.3 `steps_config.toml`

Per‑step relative paths, manifests, reports, and step‑local non‑semantic switches:

- Input/output files per step  
- Whether a step runs in a given experiment  
- Additional options (e.g., `postpaint_empty_enabled` for MILP merge, DES scenario list, NSGA‑II evaluation limits)

---

## 6. Workflow (Step‑by‑Step)

### STEP‑01 — Job Expansion (`step01_expand_jobs.py`)

**Purpose:** Normalize and expand `demands.csv` into `step01_jobs.csv` (one row per physical job) with joins to `parts.csv` and `topcoat_codebook.csv`.

- No tokens, no C1/C2 checks.  
- Deterministic `job_id`, `prov_id`, `prov_seq`.  
- Gate: all joins succeed; counts match demanded quantities; manifest/report written.

### STEP‑02 — Toyota Grouping & Spacing (`step02_toyota_gs.py`)

**Purpose:** Build a single‑file feeder that is feasible under C1/C2 and shaped by window W and spacing. Uses tokens `FLUSH` (purge) and `EMPTY` (local spacing) as needed, but tokens do not propagate to Step‑03.

- Inputs: `step01_jobs.csv`, S‑matrix, `topcoat_rules.csv`, `adjacency_rules.csv`, `global_config.toml`, `step02_empirical.toml`.  
- Output: `step02_paint_seed.csv` (typed symbol stream), manifest, report.  
- Enforces: C1 and C2 on feeder; zero violations.  
- Gate: 0 C1/C2 violations; deterministic.

### STEP‑03 — Booth Assignment (`step03_booth_split.py`)

**Purpose:** Split the feeder JOB sequence into two booth JOB streams via greedy f3 split.

- Inputs: `step02_paint_seed.csv` (JOB projection), `global_config.toml`, `steps_config.toml`.  
- Outputs: `step03_booth1.csv`, `step03_booth2.csv`, manifest, report.  
- No tokens, no C1/C2 checks.  
- Gate: all JOBs assigned; split metrics recorded.

### STEP‑04 — SDSU per Booth (`step04_sdsu_per_booth.py`)

**Purpose:** Optimize each booth sequence for f1 under C1/C2/C3 using the SDSU method.

- Inputs: Step‑03 outputs; S‑matrix; `topcoat_rules.csv`; `adjacency_rules.csv`; `global_config.toml`.  
- Outputs: `step04_booth1_sdsu.csv`, `step04_booth2_sdsu.csv`, manifest, report.  
- Enforces: per‑booth C1 & C2 with scope policy; purge semantics.  
- Gate: zero paint and adjacency violations; f1 improvement vs pre‑SDSU logged.

### STEP‑05 — Heuristic 2→1 Merge (SDSU baseline) — `step05_merge_global_line.py`

**Purpose:** Serialize the two SDSU booth sequences into a single oven convoy while enforcing global C2 on parts. Paint is not re‑checked.

- Inputs: `step04_booth1_sdsu.csv`, `step04_booth2_sdsu.csv`, `adjacency_rules.csv`, `global_config.toml`, `steps_config.toml`.  
- Policy: selection rule (e.g., feeder‑order, round‑robin, balance) plus optional post‑paint EMPTY insertion for deadlocks.  
- Mode: default C2 evaluation is `parts_only`; result is JOB‑only convoy.  
- Output: `step05_global_line.csv`, manifest, report.  
- Gate: zero global C2 violations or failure with certificate; deadlocks either resolved (if EMPTY allowed) or declared.

### STEP‑06 — MILP from Step‑03 (`step06_milp_from_step03.py`, `step06_merge_global_line.py`)

**Purpose:** Provide exact/reduced baselines with same semantics, starting from Step‑03 splits.

- MILP: per‑booth open‑path ATSP with S‑matrix + paint forbids + in‑booth C2; purge via FLUSH as allowed.  
- Merge: `step06_merge_global_line.py` implements generalized merge with:
  - `step_id="step05"` → Step‑05 semantics (default `c2_mode="parts_only"`),  
  - `step_id="step06"` → MILP semantics with `c2_mode="symbol_stream"` and post‑paint EMPTY.  

Config from `[step06]` in `steps_config.toml`.

- Outputs: per‑booth MILP sequences and `step06_milp_opt_global.csv`, manifests, reports.  
- Gate: MILP feasibility; zero C1/C2 violations; consistent merge.

### STEP‑07 — NSGA‑II Search (`step07_nsga2.py`, `step07_nsga2_seeded.py`)

**Purpose:** Explore f1–f2–f3 trade‑off using feasibility‑preserving decoder.

- Genome: split string σ (booth assignment) + per‑booth permutations.  
- Decoder:
  - Builds booth sequences and enforces C1/C2/C3 (SDSU‑like kernel).  
  - Merges with same C2 mode as Step‑05/06.  
- Unseeded: `step07_nsga2.py` starts from random population.  
- Seeded: `step07_nsga2_seeded.py` uses Toyota+SDSU solutions as seeds.  
- Inputs: dataset CSVs, `global_config.toml`, `steps_config.toml`.  
- Outputs: `step07_pareto.csv` + exports, manifests, reports.  
- Gate: all Pareto candidates feasible; seeded/unseeded runs share identical settings.

### STEP‑08 — DES (`step08_des.py`, `step08_des_plotter.py`)

**Purpose:** Quantify f4 (blocking/idle) for given per‑booth × global convoy.

- Inputs: booth sequences (from SDSU/MILP/NSGA‑II), global convoy (Step‑05/06/07), `line_cadence.csv` or `[des]` timing, `global_config.toml`, `steps_config.toml`.  
- Model: two paint servers, one oven, buffer β, deterministic processing times; `EMPTY` jobs consume oven time `θ * empty_units_postpaint_ratio` and count as oven idle.  
- Outputs: `step08_des_metrics.csv`, optional trace, manifests, reports, and plots.  
- Gate: DES does not change sequences; order invariants hold.

### STEP‑09 — Aggregation & Final Report (Analysis Only step)

**Purpose:** Aggregate metrics/fronts and generate experiment matrices and final tables/plots. No new scheduling logic.

The report are generated for analysis purpose and as a helper script for the paper.

- Inputs (all read-only):
  - Manifests and reports from all steps (01–08)  
  - `step05_global_line.csv`, `step06_milp_opt_global.csv`, NSGA-II exports  
  - `step08_des_metrics.csv` (and optionally DES traces)  
- Operations:
  - Join metrics across methods on experiment identifiers (e.g., W, ρ, token costs, adjacency mode).  
  - Build comparative tables for (f1, f2, f3, f4) across:
    - Toyota+SDSU baseline (Step-05)  
    - MILP baseline (Step-06)  
    - NSGA-II unseeded vs seeded (Step-07)  
    - DES scenarios (Step-08)  
  - Produce publication-ready plots (fronts, trade-offs, Pareto dominance statistics).  
- Outputs:
  - `step09_metrics.csv` (wide table of all objective values and diagnostics)  
  - `step09_experiment_matrix.csv` (one row per run/setting)  
  - `step09_final_report.md` (narrative, figures, tables)  
- Gate: all upstream manifests exist and show 0 C1/C2 violations; DES order invariants verified.

## 7. Manifests, Reports, and Reproducibility

Across all steps:

### 7.1 Manifests

Each manifest (JSON) records:

- `step_id`  
- `config_snapshot` (subset of `global_config.toml`, `steps_config.toml`, and step-local `.toml` actually used)  
- `inputs[{rel_path, sha256}]` and `outputs[{rel_path, sha256}]`  
- `settings{…}` (resolved parameters such as W, ρ, dp_threshold_jobs, c2_mode, token ratios)  
- `metrics{…}` (step-specific metrics)  
- `diagnostics{…}` (first violation indices, token counts, rejected moves, MILP gap, etc.)  
- `runtime_sec` and basic environment info (Python version, solver version)

### 7.2 Reports

Each report (Markdown) follows the structure:

1. **Purpose**  
2. **What** (algorithm summary, I/O contract)  
3. **Why** (role in MOFFS pipeline, link to objectives)  
4. **Config snapshot** (relevant config keys and values)  
5. **I/O hashes** (paths and SHA-256)  
6. **Settings** (resolved values of important parameters)  
7. **Metrics** (tables, plots)  
8. **Diagnostics** (violations, deadlocks, token usage, solver status)  
9. **Runtime & Determinism** (runtime_sec, RNG seeds if any, tie-breaking policy)

### 7.3 Determinism

- Step-01/02/03/04/05/06/08/09 must be deterministic functions of inputs + config (no RNG).  
- NSGA-II (Step-07) uses RNG but:
  - Seeds are fixed and recorded in manifests.  
  - Operators and parameters (population size, crossover/mutation rates, generations) are logged.  
  - Re-running with same seeds and configs must reproduce the same fronts.

---

## 8. Directory Layout (Canonical)

A canonical layout consistent with the current scripts:

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

output/
    step01_jobs.csv
    step02_paint_seed.csv
    step03_booth1.csv
    step03_booth2.csv
    step04_booth1_sdsu.csv
    step04_booth2_sdsu.csv
    step05_global_line.csv
    step06_milp_opt_*.csv
    step07_pareto.csv
    step08_des_metrics.csv
    step08_trace.csv
    step09_metrics.csv
    step09_experiment_matrix.csv
    manifests/stepXX_manifest.json
    reports/stepXX_report.md
```

---

## 9. Terminology

- **JOB:** individual paintable item derived from `demands.csv`.  
- **Token:** non-part symbol (`START`, `FLUSH`, `EMPTY`) that carries cost/time but no C2 semantics in the dataset; may separate parts in symbol-stream mode.  
    - `FLUSH`/`PURGE` are used interchangably.
- **Parts-only adjacency:** C2 evaluated after collapsing tokens; tokens do not change which consecutive parts are checked.  
- **Symbol-stream adjacency:** C2 evaluated on the realized symbol sequence; tokens can separate forbidden part pairs by changing immediacy.  
- **Window `W`:** sliding window length over JOBs; tokens never count as parts for window statistics.  
- **Feeder:** single global paint sequence from Step-02.  
- **Convoy:** single oven order after 2→1 merge (Step-05/06).  

