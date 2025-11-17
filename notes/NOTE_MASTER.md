# Study Master Note


_This is the study's design notes on research framework, semantics, and execution doctrine._

> **Scope.** Two paint booths feeding a single oven convoy (2→1). Multi-Objective Flexible Flow Shop (MOFFS) with hard feasibility taken **only** from dataset files: paint transition matrix ∪ explicit paint forbids; part adjacency forbids with scope. All sequencing logic is deterministic and token-audited. This master note precedes step notes and governs the entire pipeline.

---

## 1. Main Experiment Notes

What this project does and how it will be evaluated:
We study a paintshop configured as **2 paint booths → 1 oven**. The research problem is to generate and evaluate schedules that balance four concerns:

- **f1 (Paint setup units):** minimize color changeovers and purge units (FLUSH), using only paint-domain tokens.  
- **f2 (Windowed mix quality):** maintain diversified output on the final merged line using a parts-only window \(W\).  
- **f3 (Booth balance):** balance paint workloads between the two booths with a weighted metric `|N₁ − N₂| + ρ × (#cross)` where `N₁,N₂` are booth job counts and `cross` counts PB1↔PB2 switches along feeder order.  
- **f4 (Blocking/idle):** quantify operational congestion at the 2→1 merge using discrete-event simulation (DES) with buffer β and service ratio θ.

Feasibility is **data-driven**. Tokens `START`/`EMPTY`/`FLUSH` are unit-cost devices with strict semantics:

- Paint feasibility (C1) comes only from the paint SDST matrix and explicit paint forbids. FLUSH is the only mechanism allowed to restore paint feasibility.  
- Adjacency feasibility (C2) comes only from `adjacency_rules.csv` and scope policy. Tokens can separate parts in the realized stream but **never change which part→part pairs are forbidden in the data**.

Determinism, manifests (SHA-256), and reproducibility are first-class requirements.

---

## 2. System Model and Assumptions

- **Routing:** All parts go through Paint Booth B1 or B2, then into a single oven convoy. No re-entrant stages.  
- **Booths:** Identical paint capability w.r.t. the dataset; “setup” is realized via color transitions and associated purge units if needed.  
- **Oven:** Single serial line (global stream). On any single-file stream (feeder or merged oven), **global adjacency** forbids on parts are enforced according to `adjacency_rules.csv` and the scope policy in `global_config.toml`.  
- **Buffers:** Optional: booth output buffers and pre-oven buffer of size β; if β=0, system is blocking.  
- **Timing:** Base modeling uses **paint units** and generic time units. Mapping to physical minutes/hours (via `line_cadence.csv` and `[des]` settings) is an overlay used mainly in DES; the combinatorial structure of schedules is independent of wall-clock units.

---

## 3. Data Sources and Ground-Truth Semantics

All inputs live under `paths.dataset_dir` and are referenced via `[data_files]` in `global_config.toml`. No step hard-codes filenames; they all go through `global_config.toml` (+ `steps_config.toml` for outputs).

Conceptually we distinguish:

- **Paint transition matrix (`s_matrix_transition`):** finite entries are allowed/weighted transitions; missing/`INF`/empty ⇒ forbidden.  
- **Explicit paint forbids (`topcoat_rules`):** unioned with matrix forbids.  
- **Part adjacency forbids (`adjacency_rules`):** immediate successor forbids on parts with a `scope` field used as:
  - in-booth adjacency (C2ᵇ): `scope ∈ {"in_booth","global"}`  
  - global/oven adjacency (C2ᵍ): `scope ∈ {"global","across_skids"}` with escalation governed by `[adjacency].scope_escalation`.  
- **Demands (`demands`) and dictionaries (`topcoat_codebook`, `parts`):** define jobs and metadata (topcoat code, family, part type, etc.).  
- **Optional:** `cycle_quota` (mix targets), `fixtures`, `line_cadence` (timing extensions).

**Truth policy:** The dataset is authoritative. No step may introduce its own feasibility rules. If a paint transition or adjacency is absent/forbidden in data, it is **forbidden** in all methods (Toyota GS, SDSU, MILP, NSGA-II, DES).

---

## 4. Token Semantics (strict and minimal)

Tokens are used **only** to separate jobs in the realized streams or to model paint purges. They never create new allowed part→part pairs; they only change which parts become immediate neighbours.

- **`START`**  
  - Appears once at the beginning of the feeder stream (Step-02).  
  - Cost-neutral; used only to define first-job setup in the paint kernel.

- **`EMPTY` (spacer / carrier)**  
  - **Pre-paint / feeder (Step-02):**  
    - May be inserted to break immediate part adjacency conflicts on the **single-file feeder stream** when all FIFO-eligible jobs at the head violate C2 under `adjacency_rules`.  
    - Cost is `tokens.empty_units_prepaint` in `global_config.toml`.  
  - **Post-paint / oven (Step-05/06, and in DES interpretation Step-08):**  
    - The merged oven stream is a symbol stream over `{B1, B2, EMPTY}` where `B1/B2` stand for “job from booth 1/2”.  
    - `adjacency_rules.csv` only contains **part→part** forbids; it never includes `EMPTY`.  
    - When both booth heads would violate C2ᵍ w.r.t. the last emitted part symbol, and `allow_postpaint_empty` is enabled, we append an `EMPTY` symbol to the global stream and reset the adjacency state.  
    - In DES, EMPTY consumes oven time `oven_time_per_job × empty_units_postpaint_ratio` and is counted as **oven idle**, not busy time.  
  - Net effect: `EMPTY` **resolves adjacency by separation**: `A → B` may be forbidden, but `A → EMPTY → B` is allowed because `A` and `B` are no longer immediate successors. The forbid relation itself is never altered.

- **`FLUSH` (paint purge)**  
  - Exists purely in the **paint domain** for SDSU/MILP/NSGA-II; it never appears on the oven stream.  
  - May be used only when a paint transition is infeasible or high-cost under matrix ∪ explicit forbids, depending on `[paint_rules]` and `[sdsu].allow_purge`.  
  - Cost is controlled by `tokens.flush_units_ratio` and purge weights in `[sdsu]` (`purge_cost_weight`), always counted inside f1 (paint units).

**Key invariant:** Tokens never change the set of forbidden part→part pairs defined in `adjacency_rules.csv`. They only alter the realized stream so that forbidden pairs cease to be immediate neighbours.

---

## 5. Objectives and Metrics (canonical definitions)

- **f1 - Paint setup units.**  
  Sum of successive topcoat transition weights on per-booth sequences, plus unitized **FLUSH** where used. Tokens other than `FLUSH` do not contribute to f1. Implemented consistently in SDSU, MILP, and NSGA-II via the same SDST kernel.

- **f2 - Windowed mix quality (merged line).**  
  Evaluated on the final single-file global **JOB** line. The window \(W\) counts **parts only** (tokens excluded) and measures color/part distribution vs. targets given by `cycle_quota` or derived mix fractions. Key diagnostics include:
  - `longest_color_block_jobs`
  - window deficit score or KL-type divergence from target mix

- **f3 - Booth balance.**  
  Defined on the **Step-03 assignment** as  
  \[
  f3 = |N_1 - N_2| + ρ · \text{cross}
  \]
  where `N₁,N₂` are per-booth job counts and `cross` counts PB1↔PB2 switches when scanning the feeder order.

- **f4 - Blocking/idle (DES).**  
  Given DES settings (β, θ, timing, EMPTY cost ratio), simulate the 2→1 system and report:
  - makespan,
  - booth busy/idle/blocking fractions,
  - oven busy/idle fractions,
  - scalarized `f4_norm` for comparison if enabled.

**Comparability protocol:** All methods (Toyota GS, SDSU, MILP, NSGA-II) must emit a compatible metric set. Reports must expose at least the four canonical objectives and any additional derived indicators per step.

---

## 6. Configuration Doctrine (`global_config.toml` + `steps_config.toml`)

There are two configuration files at repository root:

- **`global_config.toml`** - **sole source of truth for semantics**, data bindings, tokens, adjacency, and solver knobs.  
- **`steps_config.toml`** - per-step I/O filenames, experiment toggles, and other **non-semantic** wiring (e.g., which seed file a step consumes, names of reports).

All paths are relative and Windows-friendly; steps must not hard-code any dataset or output paths outside these two files.

### 6.1 Structure (global_config.toml)

Key sections:

- `[system]` - topology (2paint-1merged), base time unit, capacity.  
- `[paths]` - `dataset_dir`, `output_dir`, `docs_dir`, `manifests_dir`, `reports_dir`, `logs_dir`.  
- `[logging]`, `[reproducibility]` - logging level, RNG seed set, environment tag.  
- `[data_files]` - logical dataset roles (`demands`, `topcoat_codebook`, `adjacency_rules`, `s_matrix_transition`, `topcoat_rules`, `fixtures`, `cycle_quota`, `line_cadence`, `parts`) mapped to actual filenames.  
- `[data_schema.*]` - robust column-detection hints for all major CSVs; steps must always resolve columns via these hints, never by hard-coded names.  
- `[paint_rules]` - high-level paint constraints (scope of topcoat rules, flush policy).  
- `[window]`, `[spacing]` - Toyota GS parameters (window size, run caps, greedy picker weights).  
- `[tokens]` - token unit semantics:
  - `flush_units_ratio`
  - `empty_units_prepaint`
  - `empty_units_postpaint_ratio` (post-paint EMPTY cost/time ratio used at the oven)  
- `[adjacency]` - C2 policy: master `enabled` switch, `fifo_lookahead`, and `scope_escalation ∈ {none, "global_if_missing"}`.  
- `[rho]` - base scale for booth-balance parameter ρ.  
- `[sdsu]` - SDSU controls: `dp_threshold_jobs`, `max_2opt_passes`, `allow_purge`, `purge_cost_weight`.  
- `[milp]` - MILP solver parameters: solver name, time limit, gap, etc.  
- `[nsga2]` - NSGA-II run parameters: population size, generations, crossover/mutation rates, seed usage.  
- `[des]` - DES timing and scenario knobs: `base_paint_time`, `time_per_setup_unit`, `theta` (= `oven_time_per_job`), f4 weights, buffer sizes, scenario toggles.

### 6.2 Structure (steps_config.toml)

Per-step sections record:

- Input and output filenames (relative to `paths.output_dir`).  
- Manifest and report locations.  
- Step-local toggles that **do not change semantics**, e.g.:
  - `fail_on_missing_join` for Step-01,  
  - `max_eval_per_seed` for Step-07,  
  - `postpaint_empty_enabled` for Step-06 MILP global merge, etc.

Steps must **not** encode semantic constants here (e.g., no new adjacency rules, paint costs, or token definitions).

### 6.3 Parameter Tuning - Interpretation, Effects, Degeneracy

**Window & Spacing (Toyota GS on feeder)**

- `[window].mode ∈ {"jobs","ratio"}`: basis for \(W\) computation.  
- `[window].jobs`, `[window].ratio`: set \(W\); extremes:
  - `W = 1` degenerates to pure greedy-by-setup (no mix smoothing).  
  - `W ≥ N_jobs` enforces global mixing and often forces more FLUSH/EMPTY.  
- `[spacing].alpha_run_cap` and `[spacing].max_run_abs`: per-color run limits; poorly chosen pairs can make spacing infeasible in practice unless tokens are cheap.

**Tokens**

- `tokens.flush_units_ratio`: ↑ discourages FLUSH, ↓ encourages purge; too low trivializes color changes, too high makes some sequences impossible without unbounded adjacency violations.  
- `tokens.empty_units_prepaint`: ↑ discourages pre-paint spacers on feeder; ↓ encourages them.  
- `tokens.empty_units_postpaint_ratio`: ↑ discourages oven EMPTYs; ↓ enables more aggressive adjacency resolution at the merge.

**Adjacency**

- `[adjacency].fifo_lookahead`: how many jobs ahead Toyota GS may look to avoid adjacency conflicts without inserting EMPTY; too small may stall; too large increases runtime.  
- `[adjacency].scope_escalation`: how `scope` gaps in `adjacency_rules` are treated for global C2; `"global_if_missing"` is conservative but safer.

**Split Load (Step-03)**

- `[rho].scale`: ρ multiplier in `f3 = |N₁ − N₂| + ρ·cross`; large ρ prioritizes reducing cross-switches over equalizing job counts.

**SDSU**

- `[sdsu].dp_threshold_jobs`: upper bound on booth size for exact DP/ATSP path solution; larger values increase exactness but also runtime.  
- `[sdsu].max_2opt_passes`: local improvement depth.

**DES**

- `[des].theta` (= `oven_time_per_job`) and `base_paint_time` / `time_per_setup_unit` set the relative time scale between paint and oven.  
- Buffer β and f4 weights control emphasis on blocking vs idle in `f4_norm`.

---

## 7. Pipeline Overview (from feeder to DES)

The canonical end-to-end pipeline, consistent with the finalized scripts, is:

- **Step-01 - Expand Jobs (`step01_expand_jobs.py`).**  
  Deterministic ingestion of demands into a 1-JOB/row table using `global_config.toml` schema hints; no scheduling, no tokens.

- **Step-02 - Toyota GS Feeder (`step02_toyota_gs.py`).**  
  Token-aware single-file feeder stream:
  - Paint feasibility via SDST kernel (matrix ∪ topcoat rules).  
  - C2 enforced with FIFO lookahead and `EMPTY` insertion when all heads violate adjacency.  
  - Window/run-cap spacing using `[window]` and `[spacing]`.  
  Output: a JOB/TOKEN sequence with `START`, `EMPTY`, and implicit paint information.

- **Step-03 - Split to Booths (`step03_booth_split.py`).**  
  JOB-only deterministic greedy splitter for the booth-balance metric f3. It consumes the feeder JOB sequence (ignoring tokens) and assigns each job in order to PB1 or PB2 to minimize  
  `|N₁ − N₂| + ρ·cross`. No C1/C2 checks and **no tokens**; C1/C2 are enforced later by per-booth SDSU and the merged-line interleavers.

- **Step-04 - SDSU per Booth (`step04_sdsu_per_booth.py`).**  
  Per-booth non-cyclic open-path minimization of setup cost with paint forbids:
  - Paint feasibility and cost via SDST kernel + explicit topcoat forbids.  
  - In-booth adjacency C2ᵇ using `scope ∈ {"in_booth","global"}` from `adjacency_rules`.  
  - FLUSH allowed per `[sdsu].allow_purge`.  
  Output: per-booth JOB sequences (no EMPTY on oven yet) plus purge diagnostics.

- **Step-05 - SDSU baseline merge to global oven (`step05_merge_global_line.py`).**  
  2→1 merge of the two SDSU booth sequences into a single oven stream:
  - Input symbols are `B1`, `B2` (jobs) and optional `EMPTY`.  
  - Global adjacency C2ᵍ uses `scope ∈ {"global","across_skids"}` and scope escalation from `[adjacency]`.  
  - If both heads violate C2ᵍ and `allow_postpaint_empty` is true, an `EMPTY` symbol is inserted to break the deadlock; adjacency checks see only part→part forbids from data, never tokens.  
  Output: `step05_global_line.csv` containing JOB rows and any inserted EMPTY rows, plus manifest/report.

- **Step-06 - MILP exact/reduced from Step-03 (`step06_milp_from_step03.py`).**  
  MILP builds per-booth sequences directly on the Step-03 splits, with the **same paint and adjacency semantics as SDSU**, and then merges them:
  - Per-booth open-path ATSP with SDST costs and paint forbids.  
  - Global merge using the same C2ᵍ model as Step-05 and optional post-paint EMPTY (controlled by `postpaint_empty_enabled` in `[step06]`).  
  Output: `step06_milp_opt_*.csv` for per-booth solutions and a global convoy `step06_milp_opt_global.csv` when full 2-booth instances exist.

- **Step-07 - NSGA-II search (`step07_nsga2.py`, `step07_nsga2_seeded.py`).**  
  Feasibility-aware decoder re-uses the same paint and adjacency kernels:
  - **Unseeded** runs: `step07_nsga2.py` explores the solution space from random initial populations under fixed seeds.  
  - **Seeded** runs: `step07_nsga2_seeded.py` starts from Toyota+SDSU solutions and refines them.  
  Outputs: Pareto fronts and (optionally) global sequences, exported under `step07_exports` and/or consolidated files, plus manifest/report.

- **Step-08 - DES evaluator (`step08_des.py` + `step08_des_plotter.py`).**  
  DES evaluates blocking/idle on fixed JOB/TOKEN streams:
  - Consumes baseline SDSU global convoy (Step-05) and any MILP/NSGA-II global convoys when present.  
  - Treats `EMPTY` as an adjacency reset (A→EMPTY→B allowed even if A→B is forbidden) with oven time `oven_time_per_job × empty_units_postpaint_ratio`.  
  - Computes makespan, busy/idle/blocking fractions, and optional scalar f4.  
  - Plotter script `step08_des_plotter.py` generates comparative figures from DES metrics.

Each step emits:

- a **Manifest JSON** (I/O SHA-256, config snapshot, settings, metrics, diagnostics, runtime), and  
- a **Report MD** (Purpose / Config / I/O / Settings / Metrics / Diagnostics / Runtime / Note).

---

## 8. Correctness, Non-Degeneracy, and Failure Policy

- **Feasibility:**  
  - Paint transitions always respect the SDST matrix ∪ explicit paint forbids.  
  - Adjacency always respects `adjacency_rules.csv` with scope policy from `[adjacency]`.  
  - Tokens never modify these rules; they only prevent forbidden part→part pairs from becoming immediate neighbours by inserting separation in the stream.

- **Non-degeneracy:**  
  Guard against free tokens, pathological `(W, cap)` choices, and extreme ρ values by monitoring:
  - token counts (pre-paint and post-paint EMPTY, FLUSH),  
  - longest color blocks,  
  - split balance metrics (`N₁,N₂,cross`),  
  - DES blocking/idle regimes.

- **Failure handling:**  
  If a step cannot place a next JOB without violating feasibility and permitted tokens are exhausted or disallowed, the step fail fast with exception (certified fail):
  - position information,
  - candidate heads and their forbidden predecessors;
  - relevant caps and flags (e.g., `allow_postpaint_empty=false`);
    
- **Constraint Relax**  
    Algorithms may use hard constraints that leads to infesasible solutions, merger steps will attempt to reset constraint scenario by inserting tokens like `EMPTY` and `FLUSH`.

---

## 9. Determinism, Manifests, and Reports

- **Determinism:**  
  Toyota GS, split, and SDSU are fully deterministic; NSGA-II uses fixed seeds from `[reproducibility]`. Tie-breaks are deterministic (see Appendix A).  

- **Manifests:**  
  List all inputs/outputs with **relative paths** and SHA-256, plus effective settings (resolved W, ρ, dp_threshold_jobs, etc.), metrics summaries, diagnostics, and `runtime_sec`.
  When we run next step, check SHA in code to see if the file is correctly used, in case of multiple runs with different parameters.

- **Reports:**  
  Markdown documents with sections:
  - Purpose / What / Why  
  - Config snapshot (global + step-specific)  
  - I/O (files used/created)  
  - Settings table  
  - Metrics table
  - Diagnostics (including adjacency-violation scans)  
  - Runtime and determinism note

---

## 10. Execution Doctrine (how to repeat end-to-end later)

> Run the code in order. Step 5, 6, 7 can run in parallel if no seed needed.

```bash
python step01_expand_jobs.py
python step02_toyota_gs.py
python step03_booth_split.py
python step04_sdsu_per_booth.py
python step05_merge_global_line.py          # merge SDSU per-booth to global convoy
python step06_milp_from_step03.py          # MILP + MILP-based global convoy(variants exist)
python step07_nsga2.py                     # NSGA-II unseeded
python step07_nsga2_seeded.py              # NSGA-II seeded from SDSU (variant from non-seeded)
python step08_des.py                       # DES scenarios / metrics (optional per config)
```
Each script reads `global_config.toml` and `steps_config.toml` from the same directory. Dataset paths, C1/C2 semantics, token policies, and solver knobs come from `global_config.toml`. All input/output relative paths and step-local toggles come from steps_config.toml.

### 11. Validation Plan
- Exactness (small instances):
    Use MILP (Step-06) (on potentially reduced instances) to obtain ground-truth `f1`, compare against other methods.
- Scalability and search quality:
    Run NSGA-II with feasibility-aware decoder; compare unseeded (step07_nsga2.py) vs. SDSU-seeded (step07_nsga2_seeded.py) fronts along (f1,f2,f3) and, when DES is applied, including f4.
- Operations impact:
    Use DES (Step-08) to quantify blocking/idle under different `β`, `θ`, and `EMPTY` ratios; report f4 and, if configured, a normalized f4_norm for use in multi-objective comparisons.

## 12. Glossary and Notation
- Adjacency (C2): immediate successor forbids on parts, sourced from `adjacency_rules.csv,` with scopes for in-booth vs global/oven streams.
- Paint forbids: union of explicit forbids from `topcoat_rules` and `INF`/missing entries in the paint transition matrix.
- Tokens:
    - `START` - neutral start marker on feeder, used only for initial setup cost, this is for code-purpose only;
    - `EMPTY` - spacer carrier; pre-paint on feeder and post-paint on oven; breaks adjacency by separation and consumes token-specific unit cost/time for adjacency deadlock breaking;
    - `FLUSH` - paint purge token inside SDSU/MILP/NSGA-II for paint deadlock breaking.
- W (window): parts-only count used by feeder spacing; tokens excluded.
- ρ (rho): split load parameter in Step-03.

## 13. Appendices
### A. Tie-Break Order
We define a uniformed merger for tie breaking and/or unifying shapes of data so they can be compared directly and consistently, especially where local greedy or candidate selection is required (e.g., Toyota GS, Step-05 merge):
- Lower composite score (as configured, e.g., setup + spacing weights)
- Lower setup term
- Higher spacing deficit (when blend/weighted modes apply)
- Index/Lexicographic (topcoat_code, part_id) or (booth_id, part_id) as appropriate

### B. Manifest JSON (minimum fields)
- `step_id`
- `config_snapshot`
- `inputs[{rel_path, sha256}]`
- `outputs[{rel_path, sha256}]`
- `settings{…}` (resolved parameters used by the step)
- `metrics{…}`
- `diagnostics{…}` (including any first-violation certificate if the step aborts)
- `runtime_sec`

### C. Report MD
Reading friendly version of manifest. 
Report in markdown is generated dynamically for all steps so that each step can validated and identified, potentially reproduced, with the following data:
- Purpose (\[What\] and \[Why\]): brief description for documentation purpose
- Config: Configuration used
- I/O (relative paths + SHA): file input/output and used SHA for validation 
- Settings: additional settings
- Metrics
- Diagnostics
- Runtime
- Note