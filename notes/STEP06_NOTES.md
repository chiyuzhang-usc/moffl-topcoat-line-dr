# STEP-06 NOTES — MILP

> Script: `step06_milp_from_step03.py`  
> Config sections: `[global]` (MILP, tokens, adjacency, SDSU) and `[step06]` in `steps_config.toml`  
> Primary outputs: per-booth MILP schedules `step06_milp_opt_<instance_tag>.csv` and manifest/report.  
> Optional output: global MILP-based oven convoy `step06_milp_opt_global.csv` (jobs-only).

*This step uses GUROBI*

---

## 1. Role in the MOFFS Pipeline

STEP-06 is an **exact (or near-exact) paint setup optimizer** used for **spot-checking and benchmarking**. It solves MILP models on the per-booth job sets created in STEP-03 and, when possible, builds a single **global oven convoy** from the full-booth MILP solutions.

- **Upstream dependencies**
  - STEP-01: expands `demands.csv` into unit jobs.
  - STEP-02: feeder / Toyota GS (group & spacing) defines the canonical paint seed.
  - STEP-03: splits the GS seed into per-booth job lists (`step03_booth1.csv`, `step03_booth2.csv`).

- **Downstream use**
  - Provides **exact f₁ (paint setup + purge)** benchmarks per booth for comparison against:
    - STEP-04 SDSU per-booth schedules.
    - STEP-07 NSGA-II heuristic schedules (seeded and unseeded).
  - Optional global convoy (`step06_milp_opt_global.csv`) can be used as an alternative oven sequence for DES (STEP-08) if the config is pointed at it.

Unlike SDSU and NSGA-II, STEP-06 is **not** an online scheduling heuristic: it is an **offline exact/spot-check model** that trades time for optimality on reduced instances.

Limitation: MILP is extremely *slow* and does not always yield solution.
We treat the constraints as hard-requirement while formulating it,
if infeasible is the result, we relax it by having the per-booth sequence merged 
and add TOKEN to it, particularly for EMPTY.

*SEED*: MILP should yield exact result, seeding would only effect the speed.

---

## 2. Inputs and Outputs

### 2.1 Required inputs

From `global_config.toml`:

- `dataset_dir`: location of all authoritative CSVs.
- `output_dir`: base directory for step outputs.
- **Data files** (via `data_files` and `data_schema` mappings):
  - `s_matrix_transition.csv` — SDST matrix at paint.
  - `topcoat_rules.csv` — paint and purge rules.
  - `topcoat_codebook.csv` — mapping from numeric codes to canonical topcoat codes.
  - `adjacency_rules.csv` — C2 adjacency forbids (currently used as a flat set of forbidden transitions).

From `steps_config.toml`:

- `[step03]`:
  - `booth1_relpath` — typically `step03_booth1.csv`.
  - `booth2_relpath` — typically `step03_booth2.csv`.
- `[step06]` (see §3).

From upstream steps:

- `output/step03_booth1.csv` — jobs assigned to PB1.
- `output/step03_booth2.csv` — jobs assigned to PB2.

STEP-03 outputs must expose at least the columns:

- `job_id`
- `part_id` (resolved via `data_schema.demands.part_id_synonyms`)
- `topcoat_code` (resolved via `data_schema.demands.topcoat_code_synonyms`)

The loader normalizes to exactly these three columns.

### 2.2 Outputs

Per-instance MILP schedules (primary artifacts):

- File pattern:  
  `output_dir / (step06.milp_output_prefix + <instance_tag> + ".csv")`  
  With defaults: `step06_milp_opt_<instance_tag>.csv`.

- Columns:
  - `job_id`
  - `part_id`
  - `topcoat_code`
  - `booth_id`
  - `milp_seq_index` — 1-based sequence index inside the instance.
  - `arc_setup_units` — setup portion of cost for the predecessor→current transition.
  - `arc_purge_units` — purge portion (if purge chosen over direct transition).
  - `arc_cost_units` — `arc_setup_units + arc_purge_units`.
  - `cum_cost_units` — cumulative f₁ along the sequence.
  - `instance_tag` — identifier such as `pb1_full`, `pb2_win2`, etc.

Step-level metadata:

- Manifest (JSON): path from `[step06].manifest_relpath` (default `manifests/step06_manifest.json`).
- Report (Markdown): path from `[step06].report_relpath` (default `reports/step06_report.md`).

Each instance entry in the manifest includes:

- `instance_tag`, `booth_id`, `n_jobs`.
- Flags: `is_full_booth`, `is_reduced`.
- Source file: `source_relpath` (the STEP-03 CSV used).
- Solver outputs: `status`, `f1_milp`, `f1_baseline`, `solver_mip_gap`, `solver_time_sec`, `output_relpath`.

Optional global convoy (jobs-only stream):

- `step06_milp_opt_global.csv` (fixed name inside `output_dir`).
  - Created only if full-booth MILP solutions are available for **both** booths.
  - One row per job in the oven sequence, with at least `job_id`, `part_id`, `topcoat_code`, `booth_id`, `global_seq_index`.
  - Global C2 is enforced during construction; optional **post-paint EMPTY** (conceptual) can be used to break deadlocks (see §7).

---

## 3. Configuration Knobs

### 3.1 Global MILP / SDSU / tokens

From `global_config.toml`:

- `milp_solver` — MILP backend (Gurobi assumed).
- `milp_time_limit_sec` — per-instance time limit.
- `milp_mip_gap` — target relative MIP gap.
- `milp_threads` — maximum threads (0 = solver default).
- `milp_log_to_console` — controls solver log verbosity.

- `sdsu_allow_purge` — whether purge (`FLUSH`) is allowed in the paint kernel.
- `sdsu_purge_cost_weight` — weight of purge relative to setup units.
- `tokens_flush_units_ratio` — converts purge weight into time/cost units.

The script uses these to define:

- `flush_cost_units = tokens_flush_units_ratio * sdsu_purge_cost_weight`.

This `flush_cost_units` is passed to the **paint cost kernel**:

- If a direct transition’s cost is ≤ `flush_cost_units`, the kernel uses the **direct** SDST cost.
- If the direct transition is more expensive than flushing, it uses a **pure purge** arc (`setup=0`, `purge=flush_cost_units`).
- Forbidden transitions in `s_matrix_transition` or `topcoat_rules` are treated as infeasible (no arc in the MILP).

Adjacency (C2):

- `adjacency_enabled` — if `false`, C2 is completely disabled in STEP-06 (MILP ignores adjacency forbids).
- `adjacency_scope_escalation` — recorded in the manifest but **not currently used** in this variant. STEP-06 treats adjacency rules as a **flat set of forbidden transitions**, regardless of scope. This is acceptable when the dataset only uses a single C2 scope, but should be noted as a limitation when multiple scopes are present.

### 3.2 Step-06 specific

From `[step06]` in `steps_config.toml`:

- `milp_output_prefix` — filename prefix for per-instance outputs (default `step06_milp_opt_`).
- `manifest_relpath` — manifest path (default `manifests/step06_manifest.json`).
- `report_relpath` — report path (default `reports/step06_report.md`).
- `enable_reduced_instances` (bool):
  - **Intent (docstring):** if `true`, create reduced contiguous windows when a booth exceeds `milp_max_jobs_per_instance`; if `false`, oversized booths should be skipped and only reported.
  - **Actual behavior:** the current implementation **always** creates contiguous windows whenever `n_jobs > milp_max_jobs_per_instance`, even when this flag is `false` (the flag only changes a log message). This should be documented as a divergence between design intent and code.
- `postpaint_empty_enabled` (bool):
  - Controls whether the optional **post-paint EMPTY** mechanism can be used when constructing the global convoy; see §7.
  - No effect on per-booth MILP models themselves.

---

## 4. MILP Instance Construction (per Booth)

### 4.1 Loading STEP-03 booth streams

For each booth `b ∈ {1,2}`:

1. Load the CSV specified by `[step03].booth{b}_relpath` from `output_dir`.
2. Normalize columns to:
   - `job_id`
   - `part_id`
   - `topcoat_code`
3. Build an ordered list of jobs for that booth:
   - The order is exactly the STEP-03 assignment order and is used both:
     - As the **baseline** sequence for f₁.
     - As the reference order for cutting **reduced windows**.

If no rows exist for a booth, no instances are created for that booth.

### 4.2 Full vs reduced instances

Let `N_b` be the number of jobs for booth `b`, and `L = milp_max_jobs_per_instance`.

- If `N_b <= L`:
  - A **single full-booth instance** is created:
    - `instance_tag = "pb{b}_full"`.
    - `is_full_booth = True`, `is_reduced = False`.
    - `source_relpath = "step03_booth{b}.csv"` (logical name used in the manifest).

- If `N_b > L`:
  - The code **always** partitions the STEP-03 order into **non-overlapping contiguous windows** of length `L`:
    - Number of windows: `W = ceil(N_b / L)`.
    - Window `w` covers indices `[ (w-1)·L , min(w·L, N_b) )` in STEP-03 order.
    - `instance_tag = "pb{b}_win{w}"`.
    - `is_full_booth = False`, `is_reduced = True`.
  - The `enable_reduced_instances` flag only affects the log message, not the creation of windows.

Each instance is thus a **self-contained TSP-like job set** whose order is free to change in the MILP but whose elements come from a contiguous block of STEP-03.

---

## 5. MILP Formulation per Instance

For each instance with `n` jobs and job index set `J = {1,…,n}`:

### 5.1 Graph construction

Define nodes:

- `START = 0`
- `END = n + 1`
- Job nodes `1,…,n` correspond to the rows of the instance’s `jobs_df`.

Define arcs:

1. **START → job** arcs:
   - For each job node `j`:
     - Add `(START, j)` with cost `0` (only used to begin the path).

2. **job → job** arcs:
   - For each ordered pair of distinct jobs `(i,j)`:
     - Let `p_i, p_j` be their part identifiers and `c_i, c_j` their topcoat codes.
     - If C2 is enabled and `(p_i, p_j)` (or equivalently `(c_i, c_j)` as encoded) appears in the **adjacency forbids set**, **no arc** is added.
     - Otherwise, compute `c = paint_kernel.arc_cost(c_i, c_j)`:
       - If `c` is `None` or `∞`, **no arc** is added (C1 or purge infeasible).
       - Else, add `(i, j)` with cost `c` and mark it as a **candidate transition** for f₁.

3. **job → END** arcs:
   - For each job node `i`:
     - Add `(i, END)` with cost `0` (allows any job to be the last in the path).

Sanity checks:

- Every job node must have at least **one incoming** and **one outgoing** arc; otherwise, the instance is flagged as:
  - `INFEASIBLE_ARC_IN` or `INFEASIBLE_ARC_OUT` and skipped for optimization.

### 5.2 Decision variables

- Binary arc usage:
  - `x_k ∈ {0,1}` for each arc `k` in the arc set.
- Continuous MTZ order variables:
  - `u_i ∈ [1, n]` for each job node `i ∈ J`.

### 5.3 Objective

Minimize total paint setup + purge cost over the chosen path:

\[
\min \sum_{k} c_k x_k,
\]

where `c_k` is the arc cost from the paint kernel (already incorporating the choice between direct transition vs purge when relevant).

This objective is exactly **f₁** for that instance under the same SDST/purge model as SDSU.

### 5.4 Constraints

1. **Start and end degree constraints**
   - Exactly one outgoing arc from `START`:
     - \(\sum_{k \in \text{out}(START)} x_k = 1\).
   - Exactly one incoming arc to `END`:
     - \(\sum_{k \in \text{in}(END)} x_k = 1\).

2. **Job flow constraints**
   - For each job node `j`:
     - Exactly one incoming arc:
       - \(\sum_{k \in \text{in}(j)} x_k = 1\).
     - Exactly one outgoing arc:
       - \(\sum_{k \in \text{out}(j)} x_k = 1\).

   Together, these constraints enforce that the solution is a **single path** from `START` to `END` visiting each job exactly once.

3. **Subtour elimination (MTZ)**
   - For each pair of distinct job nodes `(i,j)` where an arc `(i,j)` exists:
     - \(u_i - u_j + 1 \le M (1 - x_{ij})\), with `M = n`.
   - Bounds: \(1 \le u_i \le n\) for all job nodes.
   - This eliminates subtours among job nodes while preserving the path structure.

### 5.5 Optimization and status

- Solver parameters:
  - `TimeLimit = milp_time_limit_sec`.
  - `MIPGap = milp_mip_gap`.
  - `Threads = milp_threads` (if > 0).

- Status handling:
  - `INFEASIBLE` / `INF_OR_UNBD`: mark instance as `INFEASIBLE`.
  - `TIME_LIMIT`: mark as `TIME_LIMIT` with incumbent solution and gap recorded.
  - `OPTIMAL`: mark as `OPTIMAL`.
  - Any other Gurobi status is recorded as `STATUS_<code>`.

### 5.6 Solution decoding and f₁ evaluation

1. Extract succesor mapping from arcs with `x_k = 1`.
2. Starting at `START`, follow successors until `END`, collecting job nodes.
3. If the decoded path does not include exactly `n` job nodes, the instance is marked `DECODE_ERROR`.

Given a valid order, recompute f₁:

- Traverse the sequence of jobs; for each consecutive pair `(i,j)`:
  - Re-evaluate `paint_kernel.arc_cost(c_i, c_j)` and accumulate.
  - If any transition is now infeasible (should not happen if arc construction was correct), mark instance as `EVAL_INFEASIBLE`.

The final f₁ is stored as `f1_milp` for the instance.

--- 

### MILP Formulation
#### Sets, indices, parameters

- Jobs in the instance:  
  \[
    J = \{1,2,\dots,n\}, \quad n \ge 1.
  \]
  Job \( j \) corresponds to row \( j \) of the instance’s `jobs_df`.

- Special nodes:
  - Start node \( s \) (index 0),
  - End node \( t \) (index \( n+1 \)).

- Node set:
  \[
    V = \{s\} \cup J \cup \{t\}.
  \]

- For each job \( j \in J \):
  - `part_id(j)` = part of job \( j \),
  - `coat(j)` = topcoat code of job \( j \).

- Adjacency forbids set (flattened C2), read from `adjacency_rules.csv` as:

  \[
    \mathcal{F} \subseteq \mathcal{P} \times \mathcal{P},
  \]
  where \((p,q) \in \mathcal{F}\) iff a **forbid** relation exists for part pair \((p,q)\).
  
  In this variant, we treat C2 as:
  \[
    \text{transition } i \to j \text{ forbidden by C2 } \iff (\text{part\_id}(i), \text{part\_id}(j)) \in \mathcal{F}.
  \]

  If `adjacency.enabled = false`, we set \(\mathcal{F} = \varnothing\) for STEP-06.

- Paint kernel cost (C1 + purge):

  For each job pair \(i,j \in J\) with \(i \ne j\):

  - Let \( c_i = \text{coat}(i)\), \(c_j = \text{coat}(j)\).
  - Using `PaintCostKernel.arc_cost(c_i,c_j)`:

    \[
      \tilde{c}_{ij} =
      \begin{cases}
        +\infty, & \text{if paint transition } (c_i,c_j) \text{ is forbidden}, \\
        \min\{\text{setup}_{ij}, \text{flush\_cost}\}, & \text{if purge allowed}, \\
        \text{setup}_{ij}, & \text{if purge disabled},
      \end{cases}
    \]

    where:
    - \(\text{setup}_{ij}\) is the SDST setup time from `s_matrix_transition.csv`,
    - \(\text{flush\_cost}\) is the scalar `flush_cost_units`.

- Arc cost matrix \(c_{ij}\):

  We define feasible directed arcs \(A \subseteq V \times V\) and associated finite costs \(c_{ij}\) as:

  - Start arcs:

    \[
      A_s = \{(s,j): j \in J\}, \quad c_{s j} = 0.
    \]

  - Job–job arcs:

    \[
      A_{JJ} = \bigl\{(i,j) : i,j \in J,\ i \ne j,\ (\text{part\_id}(i), \text{part\_id}(j)) \notin \mathcal{F},\ \tilde{c}_{ij} < +\infty \bigr\},
    \]
    with \( c_{ij} = \tilde{c}_{ij} \) for all \((i,j) \in A_{JJ}\).

  - Job–end arcs:

    \[
      A_t = \{(i,t): i \in J\}, \quad c_{i t} = 0.
    \]

  - Total arc set:

    \[
      A = A_s \cup A_{JJ} \cup A_t.
    \]

  If for some job \( j \in J \) there is no incoming or outgoing arc in \(A\), the instance is declared **arc-infeasible** and skipped (`INFEASIBLE_ARC_IN` / `INFEASIBLE_ARC_OUT` in code).

#### Decision variables

- Arc usage:

  \[
    x_{ij} \in \{0,1\}, \quad \forall (i,j) \in A,
  \]
  where \(x_{ij} = 1\) means arc \(i \to j\) is used in the solution.

- MTZ ordering (subtour elimination):

  \[
    u_i \in [1,n], \quad \forall i \in J.
  \]

  These encode a topological order for the jobs to cut subtours.

#### Objective

Minimize the total paint setup + purge cost:

\[
\min \ f_1 = \sum_{(i,j) \in A} c_{ij}\, x_{ij}.
\]

This is the same f₁ objective used elsewhere in the project (setup units plus purge units in a unified scale).

#### Constraints

1. **Start node out-degree**  
   Exactly one job is chosen as the first job:

   \[
     \sum_{(s,j) \in A_s} x_{s j} = 1.
   \]

2. **End node in-degree**  
   Exactly one job is chosen as the last job:

   \[
     \sum_{(i,t) \in A_t} x_{i t} = 1.
   \]

3. **Job flow conservation**  
   For each job \( j \in J \), we require exactly one predecessor and exactly one successor (within \(\{s\} \cup J\) and \(J \cup \{t\}\), respectively):

   - One incoming arc:

     \[
       \sum_{\substack{i \in V:\\ (i,j) \in A}} x_{ij} = 1, \quad \forall j \in J.
     \]

   - One outgoing arc:

     \[
       \sum_{\substack{k \in V:\\ (j,k) \in A}} x_{jk} = 1, \quad \forall j \in J.
     \]

   Together with 1–2, this enforces a **single Hamiltonian path** from \(s\) to \(t\) that visits each job exactly once.

4. **Subtour elimination (MTZ)**  

   Using a standard MTZ linearization with \(M = n\):

   \[
     u_i - u_j + 1 \;\le\; M \bigl(1 - x_{ij}\bigr), \quad
       \forall i,j \in J,\, i \ne j,\ (i,j) \in A_{JJ}.
   \]

   and bounds:

   \[
     1 \le u_i \le n, \quad \forall i \in J.
   \]

   Intuition: if arc \(i \to j\) is used (i.e. \(x_{ij} = 1\)), then \(u_i + 1 \le u_j\), forcing strict forward progression in the order; no directed subtour among job nodes can satisfy these inequalities.

5. **Variable domains**

   \[
     x_{ij} \in \{0,1\}, \quad \forall (i,j) \in A,\qquad
     1 \le u_i \le n,\quad \forall i \in J.
   \]

---


## 6. Baseline Cost on STEP-03 Order

For each instance, STEP-06 also evaluates the **baseline f₁** on the corresponding section of the STEP-03 sequence:

- Procedure:
  - Iterate through jobs in the original STEP-03 order (for the full booth or the window).
  - For each pair of consecutive jobs `(i,j)`:
    - If C2 is enabled and `(i,j)` is forbidden by adjacency rules, baseline is **infeasible** (`f1_baseline = None`).
    - Else, compute `c = paint_kernel.arc_cost(c_i, c_j)`:
      - If `c` is `None` or `∞`, baseline is **infeasible**.
      - Else, accumulate cost.

The baseline f₁ is stored as `f1_baseline`. This allows instance-wise comparison:

- MILP optimal (or incumbent) vs original STEP-03 order.
- For reduced windows, gaps are local to each window.

---

## 7. Global MILP Convoy (Optional, Jobs-Only)

After solving all instances, the script attempts to build a **global oven convoy** when both full-booth instances exist and are solved:

- Required inputs:
  - MILP outputs for `pb1_full` and `pb2_full` (jobs and per-job f₁ decompositions).
  - Global adjacency forbids set (same as used for per-booth MILP).

### 7.1 Merge semantics

The merge is a **deterministic 2→1 greedy interleave**:

- Maintain pointers to the current head of the PB1 and PB2 MILP sequences.
- Track the `last_part` placed on the global line.
- At each step:
  - Evaluate C2 feasibility for placing PB1’s head and PB2’s head after `last_part`.
  - If exactly one is C2-feasible, place that job.
  - If both are feasible, use a deterministic tie-break rule (implementation detail; not a new objective).
  - If neither is feasible:
    - If `postpaint_empty_enabled` is **true**:
      - Conceptually insert an `EMPTY` **after paint**:
        - Increment a diagnostic `postpaint_empty_count`.
        - Reset the adjacency predecessor (`last_part = None`).
        - Do **not** emit a job row (the output convoy remains jobs-only).
      - Retry feasibility with the updated `last_part`.
    - If `postpaint_empty_enabled` is **false**:
      - If both heads are globally blocked purely by C2, raise a **pure adjacency deadlock** error.
      - Otherwise, raise an error describing the mixed conflict.

Thus **global C2 remains data-driven** (from `adjacency_rules.csv`); `EMPTY` is a modeling device that resets adjacency state at the cost of one idle carrier slot, but is not written as an explicit row in `step06_milp_opt_global.csv`.

### 7.2 Output and diagnostics

- Output: `step06_milp_opt_global.csv` in `output_dir`.
- Diagnostics (printed to stdout):
  - `head_conflict_count` — number of times a booth head candidate was rejected due to adjacency.
  - `postpaint_empty_count` — number of conceptual `EMPTY` insertions used to escape deadlocks.

The global convoy currently has **no dedicated manifest/report**; its provenance is indirectly captured via the STEP-06 manifest and console logs.

---

## 8. Metrics and Research Usage

STEP-06 is used primarily to **quantify optimality gaps**:

- Per-booth f₁:
  - Compare SDSU (STEP-04) vs MILP (STEP-06) vs NSGA-II (STEP-07) on identical job sets.
- Reduced windows:
  - Provide exact f₁ on subproblems when full booths are too large for exact solution within the time limit.
- Global convoy:
  - Provides a MILP-derived oven sequence satisfying global C2 for comparison against:
    - STEP-05 global SDSU-based convoy.
    - STEP-07 NSGA-II derived convoys merged and then evaluated in STEP-08 DES.

In the final aggregation (STEP-09), STEP-06 contributes:

- Instance-level `(f1_milp, f1_baseline, status, gap)` per booth/window.
- Optional global-convoy-based DES metrics when configured.

---

## 9. Failure Modes and Diagnostics

Common instance statuses:

- `INFEASIBLE_ARC_IN` / `INFEASIBLE_ARC_OUT` — some job has no feasible predecessor/successor under C1/C2; instance is skipped.
- `INFEASIBLE` — MILP model has no feasible solution.
- `TIME_LIMIT` — solver hit the time limit; incumbent solution and gap are recorded.
- `DECODE_ERROR` — solution could not be decoded into a single Hamiltonian path from START to END.
- `EVAL_INFEASIBLE` — decoded path contains a transition that becomes infeasible when recomputing f₁ (should not occur; indicates inconsistency between construction and evaluation).

At the step level:

- The manifest aggregates per-instance outcomes and allows quick detection of:
  - Booths with no full instance solved.
  - Windows that are systematically infeasible.
- The report summarizes:
  - Solver configuration (time limits, gaps, threads).
  - Step-level statistics (counts of OPTIMAL/TIME_LIMIT/INFEASIBLE instances).
  - Effective settings from `[global]` and `[step06]`.

---

## 10. Alignment Checklist (Code ↔ Notes)

For verification runs, STEP-06 is considered **aligned** with these notes if:

1. **Inputs**
   - Only STEP-03 booth streams and the authoritative dataset (`s_matrix_transition.csv`, `topcoat_rules.csv`, `topcoat_codebook.csv`, `adjacency_rules.csv`) are used.
   - Column normalization produces exactly `job_id`, `part_id`, `topcoat_code`.

2. **Instance generation**
   - Every booth produces either:
     - One `pb{b}_full` instance if `N_b ≤ milp_max_jobs_per_instance`, or
     - `ceil(N_b / L)` contiguous windows `pb{b}_win{w}` with no overlaps.
   - The `enable_reduced_instances` flag does **not** suppress windows in the current code and is treated as a diagnostic flag only.

3. **MILP model**
   - Arcs are present if and only if:
     - C1 (paint + purge) is feasible under the SDST + topcoat rules, and
     - C2 is either disabled or the part/topcoat pair is not forbidden.
   - Objective is exactly the sum of arc costs from the paint kernel.
   - MTZ constraints are present for each existing job→job arc.

4. **Baseline and metrics**
   - Baseline costs use the same paint kernel and C2 set as the MILP.
   - Instance results record both `f1_milp` and `f1_baseline` (or `None` if infeasible).

5. **Global convoy (if built)**
   - Uses **full-booth** MILP sequences only.
   - Enforces global C2 based on the same adjacency forbids set.
   - Uses conceptual `EMPTY` only when `postpaint_empty_enabled = true`, and never writes tokens into the CSV.

These notes should be treated as the **canonical description** of STEP-06 for the current codebase and are designed to be consistent with the MOFFS protocol, the SDSU paint kernel, and the finalized `step06_milp_from_step03.py` implementation.
