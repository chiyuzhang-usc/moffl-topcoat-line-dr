# STEP-06 — MILP Family (Per-Booth & Global Baselines)

**Scripts:**

- `step06_milp.py` — per-booth MILP on SDSU booth streams (default STEP-06).  
- `step06_milp_from_step03.py` — per-booth MILP on STEP-03 booth streams (non-SDSU baseline).  
- `step06_milp_from_step01.py` — global two-booth MILP directly from STEP-01 job universe (assignment + sequencing baseline).

All three scripts share the **same data-driven C₁/C₂ semantics** and **arc cost kernel**. They differ only in:

1. **What job set** is given to the MILP,
2. **What level** of the paint stage is optimised:
   - per-booth sequencing (fixed booth membership),
   - or global booth assignment + sequencing.

None of these MILPs alters the canonical GS/SDSU/merge convoy used for DES; they are **side-channel optimisation layers** for benchmarking and Pareto-front construction.

---

## 1. Common Data, Semantics, and Configuration

### 1.1 Upstream artefacts

The MILP family reads:

- **Config**
  - `global_config.toml`
  - `steps_config.toml` (sections `step01`, `step03`, `step04`, `step06` as needed)

- **Dataset (semantic owners)**
  - `topcoat_rules.csv` — paint family rules (C₁).  
  - `s_matrix_transition.csv` (and/or `s_lookup_transition.csv`) — SDST time units.  
  - `topcoat_codebook.csv` — maps `topcoat_no → topcoat_code` to align rules with job data.  
  - `adjacency_rules.csv` — forbidden part pairs (C₂).

- **Job streams**
  - STEP-01: `step01_jobs_expanded.csv` — full job universe.  
  - STEP-03: `step03_booth1.csv`, `step03_booth2.csv` — greedy f₃ booth split (no C₁/C₂ guarantee).  
  - STEP-04: `step04_booth1_sdsu.csv`, `step04_booth2_sdsu.csv` — SDSU per-booth C₁/C₂-feasible sequences.

Exact file names/paths come from `global_config.toml[data_files]` and `steps_config.toml[stepXX]`.

### 1.2 Column resolution and schemas

Column names are **not hard-coded**. Each script uses `global_config.toml[data_schema.*]` synonym lists to resolve:

- `job_id`  
- `part_id`  
- `topcoat_code`  
- any SDST and rule columns (e.g. `from_topcoat_no`, `to_topcoat_no`, `relation`).

All internal logic is expressed in terms of these canonical names after resolution.

### 1.3 Common C₁/C₂ semantics

**Paint feasibility (C₁).**

- Let `τ(i)` be the topcoat code of job *i*.  
- `s_matrix_transition(τ(i), τ(j))` gives the direct setup time (paint units).  
  - `INF` / missing ⇒ direct transition forbidden unless purge is used.  
- `topcoat_rules.csv` may list additional explicit forbids or relations; in this implementation:
  - any `relation` starting with `"forbid"` yields a **hard forbid** `(τ(i), τ(j))` that even purge cannot override.

**Adjacency feasibility (C₂).**

- Let `p(i)` be the part id of job *i*.  
- `adjacency_rules.csv` gives forbidden ordered pairs `(p(i), p(j))`.  
- For any such pair, **all arcs** `i → j` are removed from the MILP graph.  
- Scope escalation is already resolved at protocol level; STEP-06 simply treats these as hard arc forbids for any in-booth or global sequence it builds.

### 1.4 Cost kernel (shared across all MILPs)

Define a **paint + purge** cost kernel:

- Parameters from `global_config.toml`:
  - `tokens.flush_units_ratio`  
  - `sdsu.purge_cost_weight`  
  - `sdsu.allow_purge`

- Derived **flush cost**:

\[
C_{\text{flush}} = \text{flush\_units\_ratio} \cdot \text{purge\_cost\_weight}.
\]

- For a color transition `τ(i) → τ(j)`:

  1. If `(τ(i), τ(j))` is forbidden by either `topcoat_rules` or `s_matrix_transition` ⇒ arc is disallowed.
  2. Let `S(τ(i), τ(j))` be the direct setup time from the SDST matrix.
  3. If purge is disallowed or `C_flush ≤ 0`:
     \[
     c_{i,j} = S(τ(i), τ(j)).
     \]
  4. If purge allowed:
     \[
     c_{i,j} = \min\{S(τ(i), τ(j)), C_{\text{flush}}\}.
     \]

- For the **MILP objective**, we use `c_{i,j}` on job→job arcs only.  
- Arcs from `START` to a first job and from a job to `END` are assigned **zero cost**; this subtracts a constant from all permutations and does not change optimal orders.

All MILP variants use **exactly this kernel**, so their f₁ values are directly comparable, and SDSU heuristic costs can be evaluated on the same kernel.

---

## 2. Per-Booth MILP (STEP-06A/B) — Fixed Job Set, Open-Path Sequencing

This model underlies:

- `step06_milp.py` (SDSU-based): per-booth sequences on SDSU job sets.  
- `step06_milp_from_step03.py` (STEP-03-based): per-booth sequences on greedy split job sets.

Both solve the same **mathematical problem**; they differ only in how the job set \(J\) is chosen and how reduced instances are constructed.

### 2.1 Job set and arcs

For any single instance (booth \(b\)):

- Job set \(J = \{1, \dots, n\}\) with attributes:
  - `job_id(i)`, `part_id(i) = p(i)`, `topcoat_code(i) = τ(i)`.
- Node set \(V = \{s\} \cup J \cup \{t\}\).

Arc set \(A \subseteq V \times V\):

- **Start arcs**: \((s, j)\) for all \(j \in J\) with cost 0.  
- **Job→Job arcs**: \((i, j)\) for all distinct \(i, j \in J\) such that:
  - `(p(i), p(j))` is *not* forbidden in `adjacency_rules.csv`, and  
  - `c_{i,j}` is defined and finite by the cost kernel above.  
- **Job→End arcs**: \((i, t)\) for all \(i \in J\) with cost 0.

All other arcs are omitted from \(A\), and corresponding decision variables do not exist.

### 2.2 Decision variables

- For every arc \((i, j) \in A\):
  - \(x_{i,j} \in \{0,1\}\):  
    1 if arc \((i, j)\) is used in the booth path, 0 otherwise.

- For every job \(i \in J\):
  - \(u_i \in [1, n]\): continuous MTZ order variable used only for subtour elimination.

### 2.3 Constraints

**Flow constraints (open path).**

- Exactly one arc **leaves** the start node:
\[
\sum_{j \in J : (s,j)\in A} x_{s,j} = 1.
\]

- Exactly one arc **enters** the terminal node:
\[
\sum_{i \in J : (i,t)\in A} x_{i,t} = 1.
\]

- Each job has **exactly one predecessor** (either another job or start):
\[
\sum_{i \in \{s\} \cup J : (i,j)\in A} x_{i,j} = 1, \quad \forall j \in J.
\]

- Each job has **exactly one successor** (either another job or terminal):
\[
\sum_{k \in J \cup \{t\} : (i,k)\in A} x_{i,k} = 1, \quad \forall i \in J.
\]

These constraints enforce that all jobs are visited exactly once in a single path from `s` to `t`.

**MTZ subtour elimination.**

For every pair \(i,j \in J\), \(i \neq j\), where \((i,j)\in A\):

\[
u_i - u_j + 1 \le (n - 1)\,(1 - x_{i,j}).
\]

Together with bounds \(1 \le u_i \le n\), this prevents disconnected cycles: if arc \((i,j)\) is active, then \(u_j \ge u_i + 1\), forcing a consistent topological order across all jobs.

### 2.4 Objective function (per-booth f₁)

Minimise the total paint setup cost along the job→job arcs:

\[
\min \sum_{(i,j) \in A} c_{i,j} \, x_{i,j}.
\]

Because only job→job arcs have non-zero \(c_{i,j}\), this is precisely the per-booth f₁ cost (up to a constant start offset).

### 2.5 SDSU-based vs STEP-03-based instances

#### 2.5.1 SDSU-based (script `step06_milp.py`)

Job sets and reduced windows are defined as:

- Full SDSU job sets per booth:
  - `step04_booth1_sdsu.csv`, `step04_booth2_sdsu.csv`
  - Each is a JOB-only SDSU sequence; the job *multiset* matches STEP-03.

- For each booth \(b\):

  - Let \(J_b\) be all jobs in `step04_boothb_sdsu.csv`, \(n_b = |J_b|\).  
  - Let \(L = \text{[milp].max_jobs_per_instance}\).

  - If \(n_b \le L\):

    - Create a **single full-booth instance**:
      - instance tag: `pb{b}_full`
      - job set: \(J_b\) (order irrelevant to MILP).

  - If \(n_b > L\):

    - If `step06.enable_reduced_instances = false`:
      - No MILP instances are built for booth \(b\); manifest records `skipped_due_to_size`.

    - If `enable_reduced_instances = true`:
      - Use the SDSU order to define **non-overlapping contiguous windows** of size \(L\) (last window shorter if needed).
      - For window index \(w = 1,2,\dots\):
        - job set \(J_{b,w}\): jobs in SDSU positions \([(w-1)L+1, \dots, \min(wL, n_b)]\).
        - instance tag: `pb{b}_win{w:04d}`.
      - Each \(J_{b,w}\) is given to the MILP as a separate instance.

For each instance, the script:

- Builds the per-booth MILP as in §2.1–2.4.  
- Computes the optimal f₁: \(f_1^{\text{MILP}}(J_{b,w})\).  
- Evaluates SDSU’s f₁ on the same job set \(J_{b,w}\) by traversing the SDSU order restricted to that window.

This yields local optimality benchmarks for SDSU segments, while avoiding intractable full-booth models when \(n_b\) is large.

#### 2.5.2 STEP-03-based (script `step06_milp_from_step03.py`)

This variant is structurally identical, but job sets/windows are defined from **STEP-03 split streams**, not SDSU:

- Job sets:
  - `step03_booth1.csv`, `step03_booth2.csv`
  - These contain jobs as assigned by the greedy f₃ split, with no C₁/C₂ enforcement.

- For each booth \(b\):

  - Let \(J_b\) be all jobs in `step03_boothb.csv`, \(n_b = |J_b|\).  
  - Let \(L = \text{[milp].max_jobs_per_instance}\).

  - If \(n_b \le L\):

    - Single instance `pb{b}_full` with job set \(J_b\).

  - If \(n_b > L\):

    - If `enable_reduced_instances = false`: no instances for booth \(b\).  
    - If `true`: define windows on the **STEP-03 order** exactly as above.

For each instance:

- The MILP solves the same per-booth sequencing problem on the chosen \(J_{b,w}\).  
- A **baseline cost** \(f_1^{\text{baseline}}(J_{b,w})\) is computed by walking the STEP-03 order on the same subset (subject to C₁/C₂; infeasible pairs return `None`).

This yields an “optimisation-from-scratch” benchmark against the raw greedy split, without SDSU pre-structuring the sequence.

### 2.6 Outputs, manifest, and report

For each per-booth instance, the scripts write:

- **Solution CSV**  
  Path template:
  \[
  \texttt{{output\_dir}} / (\texttt{step06.milp\_output\_prefix} \; || \; \texttt{instance\_tag}) + \texttt{".csv"}
  \]
  Columns:
  - `job_id`, `part_id`, `topcoat_code`, `booth_id`  
  - `milp_seq_index` (1..|J|)  
  - `arc_setup_units`, `arc_purge_units`, `arc_cost_units`  
  - `cum_cost_units` (prefix sum of arc_cost_units)  
  - `instance_tag`

- **Manifest JSON**  
  - Input/output SHA-256 hashes  
  - Effective config snapshot  
  - Per-instance metrics: `status`, `f1_milp`, `f1_sdsu` or `f1_baseline`, `solver_mip_gap`, `solver_time_sec`, `is_full_booth`, `is_reduced`.

- **Markdown report**  
  - Purpose / What / Why  
  - Config snapshot (embedded JSON)  
  - Summary table of all instances and their f₁ values and gaps.

---

## 3. Global Two-Booth MILP from STEP-01 (STEP-06C)

Script: `step06_milp_from_step01.py`.

This variant builds a **single global MILP** directly on the STEP-01 job universe, jointly deciding:

1. Which booth each job uses (assignment), and  
2. The sequence of jobs within each booth (two open paths).

It is intended as a **tight f₁ lower bound** for small “toy” instances, not for production-scale data.

### 3.1 Job universe and booths

- Let \(J = \{1, \dots, n\}\) be all jobs from `step01_jobs_expanded.csv`, with:
  - `job_id(i)`, `part_id(i) = p(i)`, `topcoat_code(i) = τ(i)`.

- Booth set \(B = \{1, 2\}\).

For each booth \(b \in B\):

- Node set \(V_b = \{s_b\} \cup J \cup \{t_b\}\).

Arc set \(A_b \subseteq V_b \times V_b\):

- **Start arcs:** \((s_b, j)\) for all \(j \in J\), cost 0.  
- **Job→Job arcs:** \((i, j)\) for all distinct \(i, j \in J\) satisfying:
  - `(p(i), p(j))` not forbidden; and  
  - `c_{i,j}` finite by the common kernel.  
- **Job→End arcs:** \((i, t_b)\) for all \(i \in J\), cost 0.

No arcs connect different booths; each booth has its own start/end.

### 3.2 Decision variables

**Assignment.**

- For each job \(i \in J\), booth \(b \in B\):
  - \(z_{i,b} \in \{0,1\}\): 1 if job \(i\) runs in booth \(b\).

**Routing per booth.**

- For each booth \(b \in B\) and each arc \((u,v) \in A_b\):
  - \(x_{b,u,v} \in \{0,1\}\): 1 if arc \((u,v)\) is used in booth \(b\).

- MTZ order variables per booth:
  - For each job \(i \in J\), booth \(b \in B\):
    - \(u_{i,b} \in [1, n]\).

### 3.3 Constraints

**Assignment completeness.**

Every job must be assigned to exactly one booth:

\[
\sum_{b \in B} z_{i,b} = 1,
\quad \forall i \in J.
\]

**Flow conservation per booth.**

For each booth \(b \in B\):

- **Start node**: exactly one outgoing arc:
\[
\sum_{(s_b, j)\in A_b} x_{b,s_b,j} = 1.
\]

- **End node**: exactly one incoming arc:
\[
\sum_{(i, t_b)\in A_b} x_{b,i,t_b} = 1.
\]

- **Jobs**: in-degree and out-degree equal to the assignment:

For each job \(i \in J\):
\[
\sum_{(u,i)\in A_b} x_{b,u,i} = z_{i,b}, \\
\sum_{(i,v)\in A_b} x_{b,i,v} = z_{i,b}.
\]

Intuitively: if \(z_{i,b} = 1\), job \(i\) appears exactly once in booth \(b\)’s path; if \(z_{i,b} = 0\), no arcs incident to \(i\) are used in that booth.

**MTZ subtour elimination per booth.**

For each booth \(b \in B\) and each pair \(i,j \in J,\, i \neq j\) with arc \((i,j) \in A_b\):

\[
u_{i,b} - u_{j,b} + 1 \le (n - 1)\, (1 - x_{b,i,j}).
\]

This prevents sub-tours on the subset of jobs actually assigned to booth \(b\) (because the flows and MTZ constraints are all conditioned on the same arcs).

### 3.4 Objective function (global f₁)

Minimise total paint setup cost over both booths:

\[
\min \sum_{b \in B} \sum_{(i,j) \in A_b} c_{i,j} \, x_{b,i,j}.
\]

Again, only job→job arcs in each booth’s path contribute to the cost.

### 3.5 Gating via `enable_reduced_instances` and `max_jobs_per_instance`

The global MILP is **only attempted** under strict gating:

1. Let `n_jobs = |J|` from STEP-01.  
2. Let `L = [milp].max_jobs_per_instance` from `global_config.toml`.  
3. Let `flag = [step06].enable_reduced_instances` from `steps_config.toml`.

Then:

- If `flag = false`:

  - No global MILP is solved.  
  - Manifest records a single instance `from_step01_full` with status `DISABLED_BY_CONFIG`.  

- If `flag = true` **and** `n_jobs > L`:

  - No global MILP is solved.  
  - Manifest status: `TOO_LARGE_GLOBAL`.  
  - User is expected to pre-reduce STEP-01 to a smaller toy instance if they want a solvable baseline.

- Only if `flag = true` **and** `n_jobs ≤ L`:

  - The full global MILP is built and solved for instance `from_step01_full`.

This design ensures that **global MILP is an explicit toy/baseline tool**, opt-in via configuration, and never silently applied to production-scale data.

### 3.6 Baseline and outputs

Before solving the global MILP, the script computes:

- \(f_1^{\text{baseline}}\): f₁ cost of the **STEP-01 row order** (treating jobs in file order as a single sequence), subject to C₁/C₂.  
  - If any consecutive pair violates C₁/C₂ under the kernel, this baseline is set to `None`.

If the MILP solves successfully:

- The script decodes two sequences, one per booth:
  - `booth_id ∈ {1,2}`  
  - For each sequence, it computes arc-level setup/purge units and cumulative cost using `split_arc_cost(τ(i), τ(j))`.

The combined per-booth sequence is written as:

- `output/{step06.milp_output_prefix_with_step01}from_step01_full.csv`  
  with the same columns as the per-booth MILP solution CSV, plus `booth_id`.

As with other variants:

- **Manifest JSON** includes SHA-256 hashes of all inputs/outputs and the global MILP metrics.  
- **Markdown report** provides:
  - Purpose / What / Why  
  - Config snapshot  
  - A single-row summary for `from_step01_full` with `n_jobs`, `f1_milp`, `f1_baseline`, solver gap and runtime, or a disabled/too-large status.

---

## 4. Role in the Overall MOFFS Study

Putting the three MILP flavours together:

1. **Per-booth SDSU MILP (step06\_milp.py)**  
   - Benchmarks SDSU on actual per-booth job sets and windows.  
   - Provides **instance-level optimal f₁** points for comparison with SDSU and NSGA-II.

2. **Per-booth STEP-03 MILP (step06\_milp\_from\_step03.py)**  
   - Benchmarks the f₃-based booth split *without* SDSU ordering.  
   - Captures how far the raw greedy split is from local f₁ optima.

3. **Global STEP-01 MILP (step06\_milp\_from\_step01.py)**  
   - Provides a **tight lower bound** on f₁ for small toy instances, jointly over booth assignment and sequencing.  
   - Serves as a theoretical reference for the entire Toyota GS → split → SDSU → merge pipeline and for NSGA-II solutions.

All three share:

- The same data-driven C₁/C₂ semantics,  
- The same arc cost kernel,  
- The same SHA-256 manifest/report protocol.

This makes them directly comparable in the later **STEP-09 multi-objective analysis**, where MILP points appear alongside heuristic and metaheuristic solutions on the empirical Pareto front.

