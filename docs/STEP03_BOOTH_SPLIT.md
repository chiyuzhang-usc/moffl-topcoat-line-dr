# STEP-03 — Booth Assignment (Greedy f₃ Split on Jobs)

## 1. Role in the MOFFS Protocol

STEP-03 takes the single-file GS feeder from STEP-02 and produces **two JOB-only streams**, one per paint booth (PB1 and PB2). It does **not** emit tokens and it does **not** own paint or adjacency feasibility. Its sole optimisation role is to shape the **workload balance and booth switching pattern** before SDSU, MILP and NSGA-II operate per booth.

In terms of the overall layout (2 paint booths → 1 merged oven line), STEP-03 sits between:

- **STEP-02 — Toyota GS feeder on jobs** (global input sequence for the system), and
- **STEP-04 — per-booth SDSU ordering**, which will reorder jobs inside each booth and enforce paint feasibility (C₁) using the S-matrix and purge rules.

All methods (Toyota+SDSU, MILP, NSGA-II) see the same physical topology; STEP-03 defines the **job-to-booth mapping** that Toyota+SDSU uses and that can also be used as seeds for MILP/NSGA-II variants.

**Key principle:** STEP-03 is a **pure splitter on JOBs**. It does not insert EMPTY or FLUSH, and it does not re-check C₁/C₂. Those responsibilities live in STEP-02 (global C₂) and STEP-04+/merge (per-booth C₁ and global C₂ on the merged line).


## 2. Inputs and Outputs

### 2.1 Inputs

From `global_config.toml`:

- `[paths].output_dir` — root directory where all step outputs live.

From `steps_config.toml` (`[step03]`):

- `input_seed_relpath` — path (relative to `output_dir`) of the STEP-02 feeder seed file (typically `step02_paint_seed.csv`). This file contains the GS feeder with tokens `{JOB, EMPTY, FLUSH}`.
- `booth1_relpath` — relative path for the PB1 JOB stream CSV (default `step03_booth1.csv`).
- `booth2_relpath` — relative path for the PB2 JOB stream CSV (default `step03_booth2.csv`).
- `assignment_relpath` — relative path for the job→booth assignment table (default `step03_assignment.csv`).
- `manifest_relpath` — relative path for the STEP-03 manifest JSON (default `manifests/step03_manifest.json`).
- `report_relpath` — relative path for the Markdown report (default `reports/step03_report.md`).
- `reshuffle_L_max` — reserved parameter for future local rebalancing; the current implementation is purely greedy and does not perform reshuffle.
- `enforce_no_tokens` — if `true`, STEP-03 will fail fast if any booth output row has a `token_type` other than `JOB`.

From `global_config.toml` (for parameters):

- `[rho].scale` — scalar weight ρ used in the f₃ split objective.

Data-schema hints (optional) from `[data_schema.demands]`:

- `job_id_synonyms` — list of column names that can be used for `job_id` in the seed CSV. If omitted, STEP-03 falls back to `["job_id", "job", "id", "jobid"]`.


### 2.2 Outputs

STEP-03 emits three tabular outputs and the usual manifest/report pair.

1. **PB1 JOB stream** — `${output_dir}/step03_booth1.csv`

   - One row per job assigned to PB1.
   - Columns:
     - All original columns from the corresponding STEP-02 `JOB` row (including any traceability fields such as `token_type`, `topcoat_code`, `part_id`, etc.).
     - `booth_id` — fixed `1` for PB1.
     - `booth_seq_index` — 1-based index of the job inside PB1 (in feeder order).
     - `feeder_job_index` — 1-based index of the job inside the global feeder JOB sequence.
     - `feeder_token_index` — original `token_index` from STEP-02.

2. **PB2 JOB stream** — `${output_dir}/step03_booth2.csv`

   - Same structure as PB1, but with `booth_id = 2` and its own `booth_seq_index`.

3. **Assignment table** — `${output_dir}/step03_assignment.csv`

   - One row per job, summarising the mapping from feeder to booth:
     - `job_id`
     - `feeder_job_index`
     - `feeder_token_index`
     - `booth_id` ∈ {1, 2}
     - `booth_seq_index` — final index inside the assigned booth.

4. **Manifest** — `${output_dir}/manifests/step03_manifest.json`

   - SHA-256 hashes of inputs/outputs.
   - Effective settings (paths, schema, ρ, algorithm name).
   - Summary metrics and diagnostics (job counts, cross, f₃ split).

5. **Report** — `${output_dir}/reports/step03_report.md`

   - Human-readable summary following the standard step structure:
     - Purpose / What / Why
     - Effective settings (JSON)
     - Metrics
     - Input/output hashes
     - Runtime


## 3. Objective and Metrics

STEP-03 optimises a single scalar objective f₃ on the job-to-booth split, defined on the global feeder order (after STEP-02):

- Let the feeder JOB sequence be indexed by k = 1,…,N.
- Let `b_k ∈ {1,2}` be the booth label of job k.
- Define:
  - N₁ = |{k : b_k = 1}| — number of jobs in PB1,
  - N₂ = |{k : b_k = 2}| — number of jobs in PB2,
  - `cross` = number of indices k where `b_k ≠ b_{k-1}` (PB1↔PB2 switches along feeder order, ignoring k = 1).

The STEP-03 split objective is:

\[
  f_3^{\text{split}}(b_1,\dots,b_N)
  = \bigl|N_1 - N_2\bigr| + \rho \cdot \text{cross},
\]

where ρ is `[rho].scale` from `global_config.toml`.

- The **first term** enforces load balance between booths (job counts).
- The **second term** penalises frequent switching between PB1 and PB2 in feeder order, which tends to reduce fine-grained alternation and provides cleaner segments for downstream SDSU and merge logic.

STEP-03 reports:

- `num_jobs` — N (total jobs),
- `num_jobs_booth1` — N₁,
- `num_jobs_booth2` — N₂,
- `cross` — number of PB1↔PB2 transitions,
- `f3_split` — realised f₃ for the final split.

These are stored both in the manifest JSON and in the Markdown report.


## 4. Algorithm (Greedy f₃ Split)

### 4.1 JOB extraction

1. Read `input_seed_relpath` from `output_dir` into a DataFrame `seed_df`.
2. Resolve columns using the schema:

   - `token_type_col` — from `["token_type"]`.
   - `token_index_col` — from `["token_index"]`.
   - `job_id_col` — from `[data_schema.demands].job_id_synonyms` or `["job_id","job","id","jobid"]`.

3. Filter JOB rows:

   - Keep rows where `token_type_col` (uppercased string) is `"JOB"`.
   - Sort by `token_index_col` ascending.
   - Reindex JOBs as `feeder_job_index = 1,…,N` in that order.

4. For each JOB row i, build an in-memory `JobRecord`:

   - `feeder_job_index`
   - `feeder_token_index` (copied from `token_index_col`)
   - `job_id` (from `job_id_col`)
   - `row` — full original row as a dictionary.


### 4.2 Greedy booth assignment

The splitter then processes jobs in feeder order, without any C₁/C₂ gating:

1. Initialise:

   - N₁ = 0, N₂ = 0 (job counts in PB1 and PB2),
   - `cross = 0`,
   - `labels[1..N]` uninitialised (to store `b_k`).

2. For each feeder index `k = 1..N` (0-based index `k-1` in code):

   - For each booth `b ∈ {1,2}`, compute the hypothetical updated counts:

     - If `b == 1`: N₁' = N₁ + 1, N₂' = N₂  
     - If `b == 2`: N₁' = N₁, N₂' = N₂ + 1

   - Compute the hypothetical `cross'`:

     - If `k == 1`: `cross' = 0` (no previous booth).
     - Else if `labels[k-1] ≠ b`: `cross' = cross + 1`.  
     - Else: `cross' = cross`.

   - Evaluate the incremental objective:

     \[
       f_3'(b) = \bigl|N_1' - N_2'\bigr| + \rho \cdot \text{cross}'.
     \]

   - Choose the booth `b*` with the smallest `f_3'(b)`.
     - Ties are broken deterministically in favour of the smaller `b` (PB1 first).
   - Set `labels[k] = b*` and commit:

     - Update N₁, N₂ according to `b*`.
     - If `k > 1` and `labels[k-1] ≠ b*`, increment `cross`.

This is a **single-pass greedy algorithm** that balances job counts while controlling the number of booth switches along the feeder order.

3. After all jobs are labelled, compute final metrics `(N₁, N₂, cross, f₃)` for diagnostics and the manifest.


### 4.3 Building outputs

Using the final labels `b_k`:

1. Initialise per-booth sequence counters: `seq_idx[1] = 0`, `seq_idx[2] = 0`.

2. For each feeder job (in order) with label `b`:

   - Increment `seq_idx[b]`.
   - Clone the original `row` dict and augment with:
     - `booth_id = b`,
     - `booth_seq_index = seq_idx[b]`,
     - `feeder_job_index`,
     - `feeder_token_index`.
   - Append the augmented row to PB1 or PB2 list depending on `b`.
   - Append a row to the assignment table:

     - `job_id`,
     - `feeder_job_index`,
     - `feeder_token_index`,
     - `booth_id`,
     - `booth_seq_index`.

3. Convert PB1 and PB2 lists to DataFrames and write them to CSV paths from `[step03]`.

4. Apply the `enforce_no_tokens` guard:

   - If `enforce_no_tokens = true` and any booth output has a `token_type` column, assert that all values are `"JOB"`. If a non-`"JOB"` value is detected, STEP-03 raises a `RuntimeError`.

5. Write the assignment table CSV.

6. Build the manifest JSON and Markdown report as described above, including SHA-256 hashes of inputs/outputs and the effective settings.


## 5. Feasibility and Responsibilities

STEP-03 is intentionally **feasibility-neutral** with respect to paint and adjacency:

- It does **not** call S-matrix or topcoat rules.
- It does **not** read `adjacency_rules.csv`.
- It does **not** insert EMPTY or FLUSH.
- It does **not** guarantee that a given booth’s job set admits a paint-feasible order by itself.

Those responsibilities are located elsewhere:

- **Global C₂ (adjacency) and window constraints** are enforced on the **single-file feeder** in STEP-02 and on the **merged line** in later steps.
- **Per-booth C₁ paint feasibility** is enforced in STEP-04 (SDSU) and any MILP/NSGA-II models that operate per booth, using the same S-matrix and purge rules as the rest of the study.

The only hard invariants STEP-03 guarantees are:

1. Every feeder JOB is assigned to exactly one booth.
2. The mapping from feeder indices to booth/job positions is **deterministic and reproducible** given the configs.
3. The split objective f₃ is well-defined and reported.


## 6. Diagnostics and Extension Hooks

The manifest’s `settings_effective` section records:

- Paths for seed and outputs,
- Resolved schema columns (`job_id_col`, `token_index_col`),
- ρ (`[rho].scale`),
- `reshuffle_L_max` (currently unused),
- `enforce_no_tokens` flag,
- `algorithm = "greedy_f3_split_no_C1C2"`.

The diagnostics section records:

- `num_jobs`, `num_jobs_booth1`, `num_jobs_booth2`,
- `cross`,
- `f3_split`.

These are the primary quantities used to compare **different feeder sequences** (e.g., Toyota GS vs MILP-derived vs NSGA-II-derived) under a common booth-splitting rule. Future variants can:

- Implement local reshuffle within a window of size `reshuffle_L_max` to further reduce f₃,
- Introduce secondary objectives (e.g., colour-aware penalties) as *soft* costs, while keeping the hard responsibilities unchanged.

For the current study, the greedy f₃ split is sufficient and keeps STEP-03’s semantics simple and transparent for downstream methods and analysis.
