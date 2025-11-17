# Step-03 — Booth Assignment (Greedy f₃ Split on JOBs)

> Follows `NOTE_MASTER.md` and `MOFFS_Protocol.md`. Consumes the JOB sequence implied by the Step-02 feeder and produces deterministic, JOB-only per-booth streams and a feeder→booth mapping. Step-03 owns **only** the booth-balance objective \( f_3 \). Paint (C₁), purge (C₃), and adjacency (C₂) feasibility are enforced in their canonical steps (GS on the feeder, SDSU/MILP/NSGA-II per booth, merge/DES on the 2→1 line), not inside this splitter.

---

## 1. Purpose

### Formal


Given the JOBs in **feeder order** from Step-02 (tokens removed), construct a booth label vector
\[
  b_k \in \{1,2\}, \quad k = 1,\dots,N
\]
and the induced per-booth JOB sequences
\[
  S_1 = \{k : b_k = 1\}, \quad S_2 = \{k : b_k = 2\}
\]
such that:

1. **Job conservation.** Every feeder JOB appears in exactly one booth sequence, preserving feeder order within each booth.
2. **Greedy f₃ balance.** The assignment minimises the scalar load metric
   \[
     f_3^{\text{split}} = |N_1 - N_2| + \rho \cdot \text{cross},
   \]
   where \(N_1,N_2\) are job counts per booth, and `cross` counts PB1↔PB2 switches when scanning the feeder order.
3. **Determinism.** The mapping is a pure function of `(global_config.toml, steps_config.toml, step02_paint_seed.csv)` with no randomness.
4. **Traceability.** The step emits a manifest (SHA-256 of inputs/outputs, settings, metrics) and a Markdown report compatible with the global step-gating protocol.

Step-03 performs **no C₁/C₂/C₃ checks** and introduces **no tokens**. Paint and adjacency semantics remain exactly as defined in the dataset and are enforced only at the steps that own them (GS, SDSU/MILP/NSGA-II, merge, DES).

### Intuitive

A dispatcher stands between the **single feeder** and the **two paint booths**. Jobs arrive in the fixed global sequence determined by Toyota GS (Step-02). For each arriving job, the dispatcher must decide:

- Send it to **Booth 1** or **Booth 2**?

The dispatcher wants to:

- Keep job counts per booth reasonably balanced, and  
- Avoid “ping-pong” behaviour (PB1→PB2→PB1→PB2…) that creates many cross-booth transitions.

Crucially, the dispatcher **does not look at color feasibility**, **purge rules**, or **part adjacency** inside a booth; those are enforced later by SDSU (Step-04), MILP (Step-06), NSGA-II (Step-07), and global C₂ on the merged convoy (Step-05/06).

### Research perspective

Step-03 isolates the **job-to-booth allocation** problem and defines the f₃ dimension:

- \( f_1 \): setup + purge units (per-booth paint domain, later).  
- \( f_2 \): windowed mix on the merged line (later).  
- \( f_3 \): booth balance and cross-booth alternation (this step).  
- \( f_4 \): blocking/idle from DES (later and optional).

By keeping Step-03 **C₁/C₂ neutral** and **JOB-only**, we can re-use the same splitting logic for different upstream global sequences:

- Toyota GS feeder (Step-02).  
- MILP-derived global orders.  
- NSGA-II-decoded global orders.

This supports fair comparisons of downstream methods under a common, deterministic split.


---

## 2. Inputs and Outputs

### 2.1 Inputs

From `global_config.toml`:

- `[paths].output_dir` — root directory for step outputs.
- `[rho].scale` — scalar ρ used in the f₃ metric.

From `global_config.toml.data_schema` (optional):

- `[data_schema.demands].job_id_synonyms` — synonyms for the job ID column used to align with Step-01/02 JOBs (fallback: `["job_id","job","id","jobid"]`).
- `[data_schema.step02_seed].token_type_synonyms` and `token_index_synonyms` (if present) — used to robustly resolve `token_type` and `token_index` in the Step-02 seed file. If absent, the script assumes canonical column names `token_type` and `token_index`.

From `steps_config.toml` (`[step03]`):

- `input_seed_relpath` — Step-02 seed CSV path relative to `output_dir` (default: `step02_paint_seed.csv`).
- `booth1_relpath` — output CSV for PB1 JOBs (default: `step03_booth1.csv`).
- `booth2_relpath` — output CSV for PB2 JOBs (default: `step03_booth2.csv`).
- `assignment_relpath` — output CSV for the assignment table (default: `step03_assignment.csv`).
- `manifest_relpath` — manifest JSON (default: `manifests/step03_manifest.json`).
- `report_relpath` — Markdown report (default: `reports/step03_report.md`).
- `enforce_no_tokens` — if `true`, Step-03 fails fast if any booth output would contain a non-`JOB` token (safety check; in the current pipeline this should always pass because Step-03 never emits tokens).
- `reshuffle_L_max` — reserved / deprecated. Present only for compatibility and currently ignored by the implementation.

Step-03 reads **only** these configs and the Step-02 seed. It does **not** read `s_matrix_transition.csv`, `topcoat_rules.csv`, or `adjacency_rules.csv`.

### 2.2 Outputs

1. **PB1 JOB stream** — `${output_dir}/step03_booth1.csv`

   One row per job assigned to PB1, with:

   - All original columns from the Step-02 JOB row (e.g., `job_id`, `part_id`, `topcoat_code`, provenance).
   - `booth_id = 1`.
   - `booth_seq_index` — 1-based index inside PB1, in feeder order.
   - `feeder_job_index` — 1-based index inside the global feeder JOB sequence (JOB projection of `step02_paint_seed.csv`).
   - `feeder_token_index` — original `token_index` from Step-02.

2. **PB2 JOB stream** — `${output_dir}/step03_booth2.csv`

   Same schema as PB1, with `booth_id = 2` and its own `booth_seq_index`.

3. **Assignment table** — `${output_dir}/step03_assignment.csv`

   One row per feeder JOB, with:

   - `job_id`
   - `feeder_job_index`
   - `feeder_token_index`
   - `booth_id` (1 or 2)
   - `booth_seq_index` (position within assigned booth)

   This table is the canonical mapping used by SDSU, MILP, and NSGA-II to reconstruct per-booth instances from any global feeder sequence.

4. **Manifest** — `${output_dir}/manifests/step03_manifest.json`

   Contains at minimum:

   - SHA-256 of `global_config.toml`, `steps_config.toml`, the seed CSV, and all outputs.  
   - Effective settings (paths, schema, ρ, algorithm tag).  
   - Metrics: `num_jobs`, `num_jobs_booth1`, `num_jobs_booth2`, `cross`, `f3_split`.  
   - Runtime and environment tag.

5. **Report** — `${output_dir}/reports/step03_report.md`

   Human-readable report with:

   - Purpose / What / Why.  
   - Config snapshot (subset of config keys actually used).  
   - I/O hashes.  
   - Settings table.  
   - Metrics and diagnostics.  
   - Runtime and determinism note.

---

## 3. Objective and Metrics (f₃ Split Load)

Let `N` be the number of feeder JOBs (JOB rows in `step02_paint_seed.csv` after filtering by `token_type == "JOB"` and sorting by `token_index`).

Step-03 assigns each JOB `k` a booth label `b_k ∈ {1,2}`.

Define:

- \(N_1 = |\{k : b_k = 1\}|\) — number of jobs assigned to PB1.  
- \(N_2 = |\{k : b_k = 2\}|\) — number of jobs assigned to PB2.  
- `cross` — the number of indices \( k \in \{2,\dots,N\} \) with \( b_k \neq b_{k-1} \), i.e., the count of PB1↔PB2 switches along feeder order.

The scalar split metric is:
\[
  f_3^{\text{split}} = |N_1 - N_2| + \rho \cdot \text{cross},
\]
where ρ is read from `[rho].scale` in `global_config.toml`.

Interpretation:

- \(|N_1 - N_2|\) measures **workload imbalance** in job counts.  
- `cross` measures the **fineness of alternation** between booths.  
- ρ is the trade-off parameter:
  - ρ = 0 → pure count balancing.  
  - large ρ → strong penalty on frequent PB1↔PB2 switches.

Reported metrics (manifest + report):

- `num_jobs = N`  
- `num_jobs_booth1 = N1`  
- `num_jobs_booth2 = N2`  
- `cross`  
- `f3_split`

These are later used in Step-09 to compare split behaviour across different global sequences and parameter sets.



## 4. Algorithm — Greedy f₃ Split

### 4.1 JOB extraction

1. Read the STEP-02 seed CSV `input_seed_relpath` from `output_dir` into `seed_df`.
2. Resolve columns:
   - `token_type_col` from `["token_type"]`,
   - `token_index_col` from `["token_index"]`,
   - `job_id_col` from `[data_schema.demands].job_id_synonyms` or `["job_id","job","id","jobid"]`.
3. Filter JOB rows: keep rows with `token_type_col.upper() == "JOB"`.
4. Sort JOB rows by `token_index_col` ascending.
5. Re-index these JOBs as `feeder_job_index = 1…N` and build in-memory records containing:
   - `feeder_job_index`,
   - `feeder_token_index`,
   - `job_id`,
   - the full original row as a dict.

### 4.2 Greedy assignment

### 4.1 Feeder JOB extraction

1. Read the Step-02 seed CSV given by `[step03].input_seed_relpath` from `output_dir` into `seed_df`.
2. Resolve columns using schema hints:
   - `token_type_col` from `[data_schema.step02_seed].token_type_synonyms` (or `"token_type"` fallback).
   - `token_index_col` from `[data_schema.step02_seed].token_index_synonyms` (or `"token_index"` fallback).
   - `job_id_col` from `[data_schema.demands].job_id_synonyms` (or fallback list).
3. Filter to JOB rows: keep rows with `token_type_col.upper() == "JOB"`.
4. Sort these JOB rows by `token_index_col` ascending to recover feeder order.
5. Assign:
   - `feeder_job_index = 1..N` in this JOB order.
   - `feeder_token_index = token_index_col`.
   - Retain all other columns for later propagation.

The list of JOB records in feeder order is the only input to the splitting logic.

### 4.2 Greedy assignment rule

The splitter performs a **single forward pass** over jobs `k = 1..N`:

1. Initialise:
   - `N1 = 0`, `N2 = 0`.  
   - `cross = 0`.  
   - `labels[1..N]` uninitialised.

2. For each job `k` in feeder order:

   For each booth `b ∈ {1,2}`:

   - Hypothetical counts:
     - If `b == 1`: `N1' = N1 + 1`, `N2' = N2`.  
     - If `b == 2`: `N1' = N1`, `N2' = N2 + 1`.
   - Hypothetical cross count:
     - If `k == 1`: `cross' = 0`.  
     - Else if `labels[k-1] ≠ b`: `cross' = cross + 1`.  
     - Else: `cross' = cross`.
   - Evaluate:
     \[
       f_3'(b) = |N_1' - N_2'| + \rho \cdot \text{cross}'.
     \]

   Choose the booth \( b^\* \) with minimum \( f_3'(b) \). Ties are broken **deterministically** in favour of `b = 1`.

   Commit:

   - `labels[k] = b*`.  
   - Update `N1, N2` based on `b*`.  
   - If `k > 1` and `labels[k-1] ≠ b*`, increment `cross`.

3. After processing all jobs, recompute `(N1, N2, cross, f3_split)` from `labels` to populate manifest metrics.

Algorithm properties:

- Time complexity: \( O(N) \).  
- No calls to paint or adjacency kernels.  
- No tokens are introduced or propagated.  
- Deterministic under fixed inputs and config.


---

## 5. Feasibility Responsibilities


Step-03 is explicitly **feasibility-neutral**:

- It does **not** call the S-matrix or `topcoat_rules.csv` (no C₁ or C₃).  
- It does **not** call `adjacency_rules.csv` (no C₂).  
- It does **not** insert `EMPTY`, `FLUSH`, or any other token.  
- It does **not** guarantee that each booth’s job set admits a paint-feasible or adjacency-feasible sequence; that is the responsibility of downstream methods.

Responsibility split in the pipeline:

- **Global C₂ and window constraints on the feeder (single stream):**  
  Owned by Step-02 (Toyota GS), using the dataset and C₂ policy in `global_config.toml`.

- **Per-booth C₁/C₂/C₃ (paint domain + in-booth adjacency):**  
  Owned by SDSU (Step-04) and MILP/NSGA-II kernels (Step-06/07). These steps read `s_matrix_transition.csv`, `topcoat_rules.csv`, and `adjacency_rules.csv` with scope semantics.

- **Global C₂ on merged 2→1 line:**  
  Owned by merge steps (Step-05/06) and respected by DES (Step-08). Tokens (`EMPTY`) at the merge are used only where configured and are audited separately.

- **Job-to-booth allocation and f₃:**  
  Owned entirely by Step-03.

Thus, Step-03 never blocks the pipeline on feasibility grounds. If later steps (SDSU/MILP/NSGA-II/merge/DES) fail due to infeasibility, this reflects properties of the data and C₁/C₂/C₃ semantics, not the splitter logic.

---

## 6. Diagnostics, Extensions, and Seeds


The manifest’s `settings_effective` section records:

- Paths for seed and outputs (relative to `output_dir`).  
- Resolved schema (`job_id_col`, `token_index_col`, `token_type_col`).  
- ρ and any step-local flags (e.g., `enforce_no_tokens`).  
- Algorithm tag, e.g. `"greedy_f3_split_no_C1C2C3"`.

Diagnostics and metrics:

- `num_jobs`, `num_jobs_booth1`, `num_jobs_booth2`.  
- `cross`, `f3_split`.  
- Optional sanity checks:
  - `num_jobs_booth1 + num_jobs_booth2 = num_jobs`.  
  - `feeder_job_index` covers `1..num_jobs` exactly once.  
  - No non-JOB token types in booth outputs when `enforce_no_tokens = true`.

These outputs serve as **seeds** for downstream methods:

- SDSU (Step-04) takes per-booth JOB streams as instances for per-booth paint optimisation.  
- MILP (Step-06) can re-optimise per-booth sequences using the same splits.  
- NSGA-II (Step-07) can reuse f₃ definitions but explore alternative splits.

Future variants could:

- Activate `reshuffle_L_max` to allow bounded local rebalancing of the last `L_max` assignments (without touching C₁/C₂/C₃ or tokens).  
- Introduce soft preferences for colour or part families in the assignment cost, while remaining strictly feasibility-neutral.

---

## 7. Verification Checklist

Before handing control to SDSU and other methods:

- [ ] `step03_booth1.csv`, `step03_booth2.csv`, and `step03_assignment.csv` exist under `output_dir`.  
- [ ] The manifest JSON exists and records SHA-256 for `global_config.toml`, `steps_config.toml`, the Step-02 seed, and all Step-03 outputs.  
- [ ] The report Markdown exists and lists:
      - Purpose / What / Why.  
      - Effective settings (including ρ and algorithm tag).  
      - Metrics (`num_jobs`, `num_jobs_booth1`, `num_jobs_booth2`, `cross`, `f3_split`).  
- [ ] If `enforce_no_tokens = true`, all `token_type` values in booth outputs are `"JOB"`.  
- [ ] `num_jobs_booth1 + num_jobs_booth2 = num_jobs`, and `feeder_job_index` covers `1..num_jobs` exactly once in `step03_assignment.csv`.

Once these checks pass, Step-03 is considered valid for the experiment, and per-booth SDSU (Step-04) can be run using the resulting JOB-only booth streams.
