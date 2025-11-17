# STEP-05 — Oven Convoy Serialization (2→1 Merge with Global Adjacency)

## 1. Role in the MOFFS Pipeline

STEP-05 takes the **per-booth SDSU sequences** from STEP-04 and produces a **single oven convoy** (global line) by interleaving PB1 and PB2 while enforcing **global adjacency constraints (C₂)** on parts. It does **not** change the order of jobs inside each booth; it only decides **which booth fires next** into the oven.

Formally, let

- \(S_1 = (j^1_1,\dots,j^1_{n_1})\) be the SDSU order for PB1,  
- \(S_2 = (j^2_1,\dots,j^2_{n_2})\) be the SDSU order for PB2.

STEP-05 constructs a global sequence \(G = (g_1,\dots,g_{n_1+n_2})\) such that:

1. **2→1 interleave:** For each booth b, the subsequence of G restricted to jobs from b is exactly \(S_b\) (i.e., **no in-booth reordering**).  
2. **Global adjacency feasibility:** For every consecutive pair of **parts** in G, the pair is not forbidden by `adjacency_rules.csv` in scopes `across_skids` or `global`. Tokens are **ignored** when checking adjacency.  
3. **Determinism:** Given the same SDSU outputs, configs, and dataset, the merge order is deterministic.

STEP-05 is the **only step** that enforces **C₂ on the merged 2→1 line**. C₁ and in-booth C₂ were already enforced per booth in STEP-04; C₂ at the feeder was enforced in STEP-02. STEP-05 does not re-open C₁ or purge decisions; it operates purely at the part-adjacency level at the oven boundary.

The output of STEP-05 (`step05_global_line.csv`) is the canonical **system schedule** fed into DES (STEP-08) and used to compute window/mix metrics for \(f_2\) and blocking/idle for \(f_4\).


## 2. Inputs and Outputs

### 2.1 Inputs

From upstream steps:

- `step04_booth1_sdsu.csv`  
- `step04_booth2_sdsu.csv`  

Both are under `output_dir` (from `[paths].output_dir` in `global_config.toml`) and contain per-booth SDSU orders with at least:

- `job_id`  
- `booth_id` (1 or 2)  
- `booth_seq_index` (SDSU order)  
- part-level identifiers (e.g. `part_id`, `demand_id`)  
- `topcoat_code` and any auxiliary columns carried from STEP-03.

From dataset:

- `adjacency_rules.csv`  
  - Contains adjacency forbids; STEP-05 uses rows whose scope (after scope escalation) applies to the **global line**, i.e., scopes `across_skids` or `global`.
- (Indirectly) `topcoat_codebook.csv`, `fixtures.csv`  
  - Only if needed for reporting or part mapping; not used for feasibility in STEP-05.

From configuration:

- `global_config.toml`
  - `[paths]` — `output_dir`, `dataset_dir`.  
  - `[adjacency]` — adjacency enable flag, lookahead, and scope escalation policy (for mapping missing scopes to `global`).  
  - `[tokens]` — `empty_units_postpaint_ratio` for the unit cost of **post-paint EMPTY** at merge if enabled.  
  - `[objective]` — any weights relevant to how post-paint EMPTY contributes to \(f_2\) and \(f_4\) (for reporting only).

- `steps_config.toml` (`[step05]`)
  - `booth1_input_relpath` — relative path for PB1 input (default `step04_booth1_sdsu.csv`).  
  - `booth2_input_relpath` — relative path for PB2 input (default `step04_booth2_sdsu.csv`).  
  - `global_line_relpath` — relative path for merged line output (default `step05_global_line.csv`).  
  - `manifest_relpath` — manifest JSON (default `manifests/step05_manifest.json`).  
  - `report_relpath` — report Markdown (default `reports/step05_report.md`).  
  - `allow_postpaint_empty :: bool` — whether the merge is allowed to insert post-paint EMPTY tokens (default `false` in baseline runs).  
  - `tie_break_policy ∈ {longer_remaining, round_robin}` — deterministic tie-break rule when both booth heads are admissible.

STEP-05 does **not** read S-matrix or paint rules; purge and paint feasibility are upstream responsibilities (STEP-04).


### 2.2 Outputs

- `step05_global_line.csv` — global oven convoy, under `output_dir`

  One row per **job**, with at least:

  - all columns from the booth SDSU inputs (e.g. `job_id`, `booth_id`, `booth_seq_index`, `part_id`, `topcoat_code`, etc.),  
  - `global_seq_index` — 1-based index on the merged line,  
  - optionally `empty_after` flag or count if post-paint EMPTY was inserted after that job (if `allow_postpaint_empty = true`).

  If `allow_postpaint_empty = false` (baseline), there are **no EMPTY rows** in this CSV; EMPTY is only conceptual for DES and cost accounting.

- `manifests/step05_manifest.json`

  - SHA-256 hashes of all inputs and outputs.  
  - Effective settings: paths, adjacency policy, `tie_break_policy`, `allow_postpaint_empty`, `empty_units_postpaint_ratio`.  
  - Metrics: global job counts, per-booth share on the merged line, number of adjacency conflicts avoided, number of post-paint EMPTY used (if enabled), and runtime.

- `reports/step05_report.md`

  - Purpose / What / Why section.  
  - Effective settings snapshot (JSON).  
  - Global metrics and diagnostics:  
    - `n_jobs_total`, `n_jobs_booth1`, `n_jobs_booth2`,  
    - `n_conflict_events` (events where a booth head was rejected due to adjacency),  
    - `n_deadlocks` if any,  
    - `n_postpaint_empty` and cost in units (if enabled).  
  - Verification that global C₂ is satisfied (0 violations).


## 3. Feasibility Semantics (C₂ on Global Line)

STEP-05 is the **global C₂ owner** at the oven boundary:

### 3.1 Adjacency forbids

- Let \(G = (g_1,\dots,g_T)\) be the final merged sequence of **jobs**.  
- For any consecutive **parts** \(p 	o q\) represented by jobs \(g_t, g_{t+1}\):

  - Look up adjacency forbids in `adjacency_rules.csv`:
    - Effective scope set = rows with scope `across_skids` or `global` (plus any scope escalated to `global` by the policy).  
  - If `(p, q)` is forbidden under these scopes, the merge **must not** produce \(p 	o q\). There is no “soft” relaxation.

STEP-05 never uses tokens to waive adjacency:

- **EMPTY** (post-paint spacer), if used, occupies time but is **ignored** in adjacency checks:
  - Adjacency is always evaluated on **parts-only** sequences: `A → EMPTY → B` counts as `A → B` for C₂.  
- There are **no FLUSH tokens** at Step-05: purges are a paint-domain concept handled in SDSU.  
- Therefore, if both booth heads would produce forbidden adjacency with the last part on the global line, the instance is **infeasible** under the current data; STEP-05 must report failure rather than inserting tokens to “fix” adjacency.

### 3.2 Relationship to other constraints

- **C₁ (paint):** already satisfied per booth by SDSU (STEP-04); Step-05 does not revisit paint feasibility.  
- **C₂ (in-booth):** already satisfied per booth by SDSU; Step-05 preserves the per-booth order and thus cannot create new in-booth adjacency violations.  
- **C₂ (global):** owned and enforced by STEP-05, as above.  
- **C₃ (purge):** owned by SDSU; Step-05 does not introduce additional FLUSH events.


## 4. Merge Algorithm (2→1 Interleave)

We describe a deterministic greedy interleave consistent with the protocol and config parameters.

### 4.1 State

- Two queues (implicitly represented by indices):
  - PB1: `S1 = [j1[0], j1[1], ..., j1[n1-1]]`  
  - PB2: `S2 = [j2[0], j2[1], ..., j2[n2-1]]`
- Head indices:
  - `h1` for PB1, `h2` for PB2.
- Global output list `G` (initially empty).  
- Last part ID placed on the global line: `last_part` (initially `None`).

### 4.2 Candidate feasibility

At each step:

1. Build candidate set `C` from non-exhausted booths:
   - If `h1 < n1`, candidate `c1 = S1[h1]`.  
   - If `h2 < n2`, candidate `c2 = S2[h2]`.

2. For each candidate `c` with part ID `p`:
   - If `last_part` is `None` (first job), `c` is always admissible.  
   - Else, look up adjacency forbids for `(last_part, p)` with scope in `{across_skids, global}` after scope escalation.  
   - If forbidden, mark `c` as **inadmissible** for this step.

3. The **feasible candidates** `F ⊆ C` are those not forbidden.

### 4.3 Selection and tie-breaking

- If `|F| == 2`:
  - Apply `tie_break_policy`:
    - `longer_remaining`: choose the booth with more jobs remaining (`(n_b - h_b)` largest).  
    - `round_robin`: alternate between PB1 and PB2 subject to feasibility (keeping a deterministic pointer).  
  - Append that job to `G`, increment its booth’s head index, and update `last_part` accordingly.

- If `|F| == 1`:
  - Take the single feasible candidate, append to `G`, advance its head, update `last_part`.

- If `|F| == 0` and at least one booth still has jobs:
  - There is no way to select a next job without violating global adjacency.  
  - STEP-05 must treat this as an **infeasible merge** under the current data, emit a diagnostic, and fail the step gate.

- If both `h1 == n1` and `h2 == n2`:
  - The merge is complete; write `G` to `step05_global_line.csv` with `global_seq_index` = 1…T.

### 4.4 Optional post-paint EMPTY

When `allow_postpaint_empty = true`, an extension is allowed:

- The algorithm may, under a **separate policy** (e.g., cadence alignment), insert conceptual EMPTY slots between jobs.  
- EMPTY slots represent **idle oven pitches** and contribute to **post-paint spacing cost** via `[tokens].empty_units_postpaint_ratio`.  
- However, EMPTY never resolves an adjacency conflict:
  - Adjacency is evaluated on parts-only; `A → EMPTY → B` is treated as `A → B` for C₂.

Baseline runs set `allow_postpaint_empty = false`, so Step-05 emits a pure JOB sequence.


## 5. Objectives and Metrics

STEP-05 directly affects **f₂** and indirectly f₃/f₄:

- **f₁ — Paint setups & purges.**  
  - Determined in SDSU (STEP-04); Step-05 does not change f₁.

- **f₂ — Windowed mix smoothing on merged line.**  
  - Computed on `step05_global_line.csv` by sliding windows of size W (from `[window]` / `[spacing]` in `global_config.toml`) over the global sequence and comparing empirical topcoat mix to targets.  
  - STEP-05’s interleave pattern directly shapes f₂.

- **f₃ — Booth load balance.**  
  - Mostly shaped by STEP-03 (split to booths). Step-05 preserves per-booth counts but may introduce uneven bursts of one booth if the tie-break policy is not balanced.

- **f₄ — Blocking/idle (from DES, STEP-08).**  
  - DES consumes `step05_global_line.csv` as the arrival sequence to the oven; blocking and idle events depend on this order together with β, θ and line cadence.

Metrics to record in the Step-05 manifest/report:

- `n_jobs_total`, `n_jobs_booth1`, `n_jobs_booth2`.  
- `n_conflict_events` — how many times a booth head was rejected due to adjacency; optionally per-booth breakdown.  
- `n_deadlocks` — count of steps where no candidate was feasible (should be 0 in successful runs).  
- `n_postpaint_empty` and `empty_units_total` (if `allow_postpaint_empty = true`).  
- Simple fairness diagnostics: e.g., max run length from a single booth on the global line.


## 6. Parameters and Diagnostics

From `steps_config.toml` and `global_config.toml`:

- `step05.tie_break_policy`  
  - `longer_remaining`: bias toward the booth with more jobs left.  
  - `round_robin`: alternate booths while respecting feasibility.  
  - Deterministic; no randomness in tie-breaking.

- `step05.allow_postpaint_empty`  
  - `false` in baseline runs: no EMPTY inserted at merge.  
  - `true` (optional experiments): allow EMPTY for cadence/spacing cost but not for adjacency fixes.

- `[adjacency].fifo_lookahead`  
  - Affects how often potential conflicts are encountered; the merge itself uses only the immediate successor but can log extra diagnostics based on lookahead if desired.

- `[tokens].empty_units_postpaint_ratio`  
  - Multiplier for the unit cost of a post-paint EMPTY at merge, used in cost accounting when `allow_postpaint_empty = true`.

Diagnostics to include in the manifest/report:

- Hashes of inputs/outputs and config files.  
- Effective values of `tie_break_policy`, `allow_postpaint_empty`, and token cost ratios.  
- The metrics listed in §5.  
- Determinism note: no RNG, no time-based tie-breaking.


## 7. Verification Checklist (Step Gate)

Before proceeding to DES (STEP-08) and downstream analysis, verify:

- [ ] `step05_global_line.csv` exists in `output_dir` and contains exactly `n1 + n2` rows, where `n1,n2` are the booth job counts from STEP-04.  
- [ ] For each booth b, the subsequence of `step05_global_line.csv` restricted to `booth_id = b` is exactly the SDSU order from `step04_boothb_sdsu.csv` (no in-booth reordering, no lost or duplicate jobs).  
- [ ] Global adjacency check passes: for all consecutive **parts** in `step05_global_line.csv`, there is no forbidden pair under `adjacency_rules.csv` with scope `across_skids` or `global`.  
- [ ] Manifest `step05_manifest.json` exists with SHA-256 hashes for all inputs and outputs and a complete settings snapshot.  
- [ ] Report `step05_report.md` exists and documents Purpose/What/Why, settings, metrics, and runtime.  
- [ ] If `allow_postpaint_empty = false`, there are no EMPTY rows and no empty cost recorded.  
- [ ] If `allow_postpaint_empty = true`, the number and cost of EMPTY events are reported and consistent with `[tokens].empty_units_postpaint_ratio`.

*End of STEP-05 Oven Convoy Serialization documentation.*
