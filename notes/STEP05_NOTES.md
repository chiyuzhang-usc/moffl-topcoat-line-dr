
# Step‑05 — Oven Convoy Serialization (2→1; Global Adjacency Enforcement)

> **Coherence.** STEP-05 is the **first global C2 owner** on the merged oven line.  
> Inputs are **JOB-only** SDSU per-booth sequences from STEP-04.  
> Output is a **single oven convoy** that:
> - preserves **in-booth order**,  
> - enforces **global adjacency C2** using `adjacency_rules.csv`,  
> - and may insert **post-paint EMPTY** tokens (explicit rows) if allowed, to break deadlocks.

No new paint or purge decisions (C₁/C₃) are made here; those remain in SDSU (STEP-04). This step is purely a **deterministic 2→1 interleaver under global C₂**.

* In later steps, we used similar method to merge per-booth sequence if algorithm cannot produce valid sequence under hard-constraints, the implementation are the same.

---

## 1. Purpose — Formal and Intuitive (with Research Perspective)

### Formal
Given two per-booth JOB sequences \(S_1, S_2\) produced by STEP-04, construct a global oven sequence \(G\) such that:

1. \(G\) is a **2→1 merge** of \(S_1\) and \(S_2\) (each booth’s internal order is preserved).
2. Global adjacency rules from `adjacency_rules.csv` (scope `global` or `across_skids`, plus escalation policy) are satisfied on the oven stream.
3. If no head job can be placed without violating global adjacency:
   - and `allow_postpaint_empty = true`, insert an **`EMPTY` symbol** (explicit empty carrier) that breaks immediacy, or  
   - otherwise, **fail fast** with a deadlock error.
4. The merge is **deterministic** for fixed inputs and configuration, and emits a manifest and a report.

Formally, if `adjacency_rules` induces a set of forbidden pairs  
\(\mathcal{F} \subseteq \mathcal{P} \times \mathcal{P}\) over parts \(\mathcal{P}\), then the stream of part IDs in \(G\) (including `"EMPTY"` for spacing) must never contain a subsequence \((p_i, p_{i+1}) \in \mathcal{F}\). `"EMPTY"` never appears in \(\mathcal{F}\).


### Intuitive
Two painted booth streams (PB1, PB2) feed a **single oven conveyor**. At each oven slot:

- You can take the next job from PB1 or PB2, as long as it **does not form a forbidden part→part pair** with the last oven part.
- If both candidates are safe, you choose according to a **tie-break policy** (e.g., favoring the booth with more remaining work or alternating booths).
- If **neither** head is safe and `allow_postpaint_empty = true`, you **skip one carrier** (insert `EMPTY`) so that the next real part no longer directly follows the last incompatibility.
- If **neither** head is safe and EMPTY is disallowed, the merge **fails**, signalling that the current adjacency scope/policy is incompatible with the given per-booth sequences.

This matches plant reality where occasionally a hook is deliberately left empty to avoid an unsafe adjacency, but such spacing is **expensive and explicitly counted**.

### Research perspective

The step isolates **system-level adjacency (global C₂)** on the shared oven from **paint-stage optimisation** (local SDSU per booth):

- It quantifies how often **head conflicts** occur given SDSU-optimized booths and a given adjacency scope.
- It exposes the **need for post-paint spacing**: how many EMPTYs are required, under which adjacency scopes and tie-break policies.
- It generates the **canonical SDSU baseline convoy** consumed by DES (STEP-08) for blocking/idle evaluation and by higher-level comparisons against MILP and NSGA-II merges.

In the overall MOFFS story, STEP-05 is the baseline “what happens if we just merge SDSU booths under global C₂, with minimal device (EMPTY) to avoid deadlock” scenario.

---

## 2. Connections to Neighbor Steps

- **Upstream (STEP-04 → STEP-05).**  
  Inputs are **feasible per-booth JOB sequences** from SDSU.  
  STEP-05:
  - assumes in-booth C₁/C₂/C₃ are already satisfied,
  - **preserves per-booth order** exactly,
  - does not introduce FLUSH or any new paint tokens.

- **Downstream (STEP-05 → STEP-06/07/08/09).**
  - **STEP-06 (MILP)**: MILP per-booth solutions are later merged via `step06_merge_global_line.py` using the same adjacency model and a generalized C₂ mode; STEP-05 provides the SDSU baseline pattern.
  - **STEP-07 (NSGA-II)**: decoders must implement a merge semantics compatible with global C₂; STEP-05 is the reference “SDSU-only” schedule.
  - **STEP-08 (DES)**: consumes `step05_global_line.csv`; DES treats `EMPTY` as a real time-advancing token that **resets adjacency** in the DES kernel but does not change the underlying forbidden set.
  - **STEP-09 (aggregation)**: reads STEP-05 metrics (especially `n_postpaint_empty`, `max_run_booth1/2`) to compare SDSU baseline against MILP/NSGA-II fronts.


---

## 3. Inputs / Outputs (Relative Paths)

### 3.1 Config inputs

From `global_config.toml`:

- `[paths].output_dir` — root directory for all step outputs.  
- `[paths].dataset_dir` — root directory for datasets.  
- `[data_files].adjacency_rules` — filename of `adjacency_rules.csv` relative to `dataset_dir`.  
- `[adjacency].enabled` — if `false`, adjacency is effectively disabled in this step (for debugging only).  
- `[adjacency].scope_escalation` — how to treat adjacency rows without explicit scope; `"global_if_missing"` promotes such rows to global C₂.  
- `[tokens].empty_units_postpaint_ratio` — **unit cost weight** applied to post-paint EMPTY spacing in later analysis (Step-09); STEP-05 only counts how many EMPTYs it inserted.

From `steps_config.toml` (`[step05]`):  

- `booth1_input_relpath` — PB1 SDSU output (default `step04_booth1_sdsu.csv`).  
- `booth2_input_relpath` — PB2 SDSU output (default `step04_booth2_sdsu.csv`).  
- `global_line_relpath` — global oven convoy (default `step05_global_line.csv`).  
- `manifest_relpath` — STEP-05 manifest (default `manifests/step05_manifest.json`).  
- `report_relpath` — STEP-05 report (default `reports/step05_report.md`).  
- `allow_postpaint_empty` — `true`/`false`, whether EMPTY can be inserted to resolve C₂ deadlocks.  
- `tie_break_policy ∈ {"longer_remaining", "round_robin"}` — deterministic choice rule when both heads are feasible.


### 3.2 File inputs

All paths below are relative to `[paths.output_dir]` or `[paths.dataset_dir]`:

- **PB1 SDSU output**: `${output_dir}/step04_booth1_sdsu.csv`  
- **PB2 SDSU output**: `${output_dir}/step04_booth2_sdsu.csv`  
- **Adjacency rules**: `${dataset_dir}/${adjacency_rules}`

Both booth CSVs must contain, at minimum:

- a job identifier (`job_id` or synonym),  
- a part identifier (`part_id` or synonym),  
- `booth_id`,  
- per-booth sequence index (e.g. `booth_seq_index`) — order is assumed as in the CSV.

Column names are resolved via `data_schema` in `global_config.toml` if present.


### 3.3 Outputs

- **Global oven convoy** — `${output_dir}/step05_global_line.csv`  

  One row per **stream position**, including **EMPTY rows when inserted**. Schema:

  - All columns from the booth inputs (at least job and part info), plus  
  - `global_seq_index` — 1-based position on the oven stream.  
  - For EMPTY rows:
    - `part_id` = `"EMPTY"`,  
    - `booth_id` = `"EMPTY"`,  
    - other fields `NULL` / `NaN`.

- **Manifest** — `${output_dir}/manifests/step05_manifest.json`  

  Contains:
  - SHA-256 of `global_config.toml`, `steps_config.toml`, both booth inputs, adjacency rules, and the global convoy.  
  - Effective STEP-05 configuration (`allow_postpaint_empty`, `tie_break_policy`, `adjacency_enabled`, `scope_escalation`, `empty_units_postpaint_ratio`).  
  - Merge metrics (see §8).  
  - Runtime and version tag.

- **Report** — `${output_dir}/reports/step05_report.md`  

  Research-style narrative:

  - Purpose / What / Why.  
  - Effective configuration snapshot (JSON).  
  - Metrics (counts, runs, EMPTY usage).  
  - Notes on adjacency semantics and upstream C₁/C₃.  


## 4. Feasibility Semantics

### 4.1 Global C₂ (adjacency)

- The **only** source of adjacency forbids is `adjacency_rules.csv`.  
- The step reads that file and builds:

  \[
  \mathcal{F} = \{ (p, q) : \text{relation row says “forbid” and scope applies to global line} \},
  \]

  where “scope applies” means:
  - `scope ∈ {"global", "across_skids"}`, or  
  - `scope` is empty and `[adjacency].scope_escalation = "global_if_missing"`.
    
    However, the scope should always be set to global to obey physical rules, the rest options are for testing purpose.
- If columns cannot be resolved or `[adjacency].enabled = false`, the algorithm fallback to “no adjacency” (empty forbid set); (DEBUG-ONLY)


### 4.2 Stream-level adjacency and EMPTY

Semantics in the current implementation:

- The merge operates on a **symbol stream**:

  - `"B1"` — job taken from PB1,  
  - `"B2"` — job taken from PB2,  
  - `"EMPTY"` — explicit empty oven hook (post-paint).

- Adjacency checks use `last_symbol` (the last symbol’s part ID, including `"EMPTY"`) and candidate part IDs:

  - For a candidate job with part \(p\):
    - If `last_symbol` is a real part \(q\), we require \((q, p) \notin \mathcal{F}\).
    - If `last_symbol = "EMPTY"`, then **no adjacency rule can be triggered** because `"EMPTY"` never appears in `adjacency_rules.csv`.

- Thus, `"EMPTY"` **never appears in the forbidden set** and simply creates a gap:  
  forbidden pair \((A,B)\) can never appear as **immediate** neighbours in the stream; we may see `A → EMPTY → B`, which is allowed.

**Crucially:** tokens do **not modify** \(\mathcal{F}\); they only prevent forbidden part→part pairs from becoming immediate neighbours by inserting separation.


### 4.3 C₁/C₃ and in-booth constraints

- **Paint feasibility (C₁)** and **purge (C₃)** are **not re-checked** here.  
- STEP-05 assumes the per-booth SDSU outputs already satisfy all in-booth C₁/C₂/C₃ conditions. It **never reshuffles within booths** and introduces **no FLUSH**.

### 4.4 Order preservation

- Within each booth stream \(S_b\), local order is preserved:

  \[
  S_b = (j_{b,1}, j_{b,2}, \dots, j_{b,n_b}) \quad\Rightarrow\quad
  G|_{b} \text{ visits these jobs in the same order.}
  \]

- STEP-05 only chooses **from which booth** to take the next job; it never reorders jobs within a booth.


---

## 5. Parameters (from `model_config.toml`) — Meaning, Effects, Extremes


## 5. Configuration Parameters and Their Effects

### 5.1 STEP-05-specific keys (`steps_config.toml`)

| Key | Meaning | ↑ Effect | ↓ Effect / Specials |
|-----|---------|---------|---------------------|
| `allow_postpaint_empty` | Allow insertion of post-paint `EMPTY` tokens to break unavoidable global C₂ deadlocks | Enables progress when both heads violate C₂; may increase stream length and cost | If `false`, a global deadlock raises an error; no spacers are inserted |
| `tie_break_policy ∈ {"longer_remaining","round_robin"}` | Rule for choosing between PB1/PB2 when both heads are C₂-feasible | `"longer_remaining"` tends to **drain the longer queue**, reducing extreme imbalance at oven | `"round_robin"` alternates booths when possible, smoothing interleave but sometimes creating more conflicts |

**Embedded correction.** Earlier confusion treated the oven as another paint stage or tried to repair conflicts by **reordering** inside booths. Here we keep booth order fixed and only use **global selection** and an optional **post‑paint EMPTY** section if policy permits.

### 5.2 Global keys (`global_config.toml`)

| Key | Meaning | Comments |
|-----|---------|----------|
| `[adjacency].enabled` | Global switch for C₂ enforcement | If `false`, STEP-05 behaves as a pure 2→1 merge ignoring adjacency (for debugging only). |
| `[adjacency].scope_escalation` | Policy for treating rules without explicit scope | `"global_if_missing"` treats scope-less forbids as global, making C₂ stricter; alternative values may restrict them. |
| `[tokens].empty_units_postpaint_ratio` | Cost multiplier per post-paint EMPTY | STEP-05 only reports `n_postpaint_empty`; cost is applied downstream (e.g., Step-09: `f1_tokens_only`). |



## 6. Merge Algorithm (Deterministic 2→1 Interleave)

Let:

- `df1`, `df2` be PB1/PB2 SDSU outputs,  
- `forbid_pairs` = \(\mathcal{F}\) from `adjacency_rules.csv`,  
- `last_symbol` = last part ID on the stream, or `"EMPTY"`, or `None` for the first job.

The algorithm in `step05_merge_global_line.py` is:

1. Initialize:

   - `h1 = h2 = 0` (heads into PB1/PB2).  
   - `out_stream = []` (sequence of `(kind, local_idx)` where `kind ∈ {"B1","B2","EMPTY"}`).  
   - `last_symbol = None`, `last_booth_kind = None`.  
   - Counters: `n_conflict_events = 0`, `n_deadlocks = 0`, `n_postpaint_empty = 0`.

2. While `h1 < len(df1)` or `h2 < len(df2)`:

   1. Build candidate set:

      - If `h1 < len(df1)`, let `p1 = part_id(PB1[h1])`.  
        - It is **feasible** if either adjacency disabled or `(last_symbol, p1) ∉ forbid_pairs`.  
        - If infeasible, increment `n_conflict_events`.  
        - Append candidate (`"B1"`, `h1`, `p1`, `feasible1`).

      - Similarly for PB2 (`"B2"`, `h2`, `p2`, `feasible2`).

   2. Let `feasible = {candidates with feas = True}`.

   3. **Deadlock handling**:

      - If `feasible` is empty:
        - Increment `n_deadlocks`.
        - If `allow_postpaint_empty = true`:
          - Append `("EMPTY", -1)` to `out_stream`.  
          - Set `last_symbol = "EMPTY"`, `last_booth_kind = None`.  
          - Increment `n_postpaint_empty`.  
          - `continue` (no booth head is advanced).
        - Else:
          - Raise `RuntimeError`: “global adjacency deadlock … allow_postpaint_empty is False.”

   4. **Tie-breaking** when at least one head is feasible:

      - If `len(feasible) = 1`, choose that candidate.  
      - Else, apply `tie_break_policy`:

        - `"longer_remaining"`:  
          - Compute remaining jobs `rem1 = len(df1) − h1`, `rem2 = len(df2) − h2`.  
          - Choose from the booth with **more remaining jobs**; if tied, prefer PB1.

        - `"round_robin"`:  
          - If `last_booth_kind` is `None` or `"B2"`, preferred booth is `"B1"`, else `"B2"`.  
          - If preferred booth is in `feasible`, choose it; otherwise take the other feasible booth.

   5. Let chosen = `(kind, local_idx, part_id, _)`:

      - Append `(kind, local_idx)` to `out_stream`.  
      - Set `last_symbol = part_id`.  
      - If `kind ∈ {"B1","B2"}`:
        - Set `last_booth_kind = kind`.  
        - Increment `h1` or `h2` accordingly.  
      - `kind = "EMPTY"` is never chosen here (only inserted in step 3).

3. After the loop terminates:

   - Build `global_df` by materializing each `(kind, local_idx)`:
     - `"B1"` → copy row from `df1[local_idx]`.  
     - `"B2"` → copy row from `df2[local_idx]`.  
     - `"EMPTY"` → synthetic row with all columns `None` except:
       - `part_id_col = "EMPTY"`,  
       - `booth_id_col = "EMPTY"`.  
   - Add `global_seq_index = 1..len(global_df)`.

4. Compute metrics (see §8) and run **final C₂ verification**:

   - Scan the `part_id` stream, including `"EMPTY"`.  
   - For each consecutive pair `(a,b)`, assert `(a,b) ∉ forbid_pairs`.  
   - Because `"EMPTY"` never appears in `forbid_pairs`, adjacency is guaranteed to be satisfied on the **full stream**.



## 7. Interplay with EMPTY and DES

- This step treats `"EMPTY"` as an explicit **stream symbol** representing a skipped oven hook.  
- DES (STEP-08) interprets this token as:
  - consuming **one time unit** (one hook),  
  - **resetting adjacency** in the DES kernel,  
  - Forbidden part-part pairs becomes feasible because inserted EMPTY.

---

## 8. Metrics and Diagnostics

The STEP-05 implementation exposes the following metrics (also recorded in the manifest and report):

- `n_jobs_total` — length of the global stream (including EMPTY).  
- `n_jobs_booth1`, `n_jobs_booth2` — counts of real jobs from PB1 and PB2; must match input booth job counts.  
- `n_conflict_events` — number of times a **candidate head** was rejected due to global adjacency violation.  
- `n_deadlocks` — number of times **both heads** were infeasible w.r.t. adjacency.  
- `n_postpaint_empty` — number of post-paint EMPTY tokens inserted; downstream cost is `n_postpaint_empty * empty_units_postpaint_ratio`.  
- `max_run_booth1`, `max_run_booth2` — maximum lengths of consecutive jobs from each booth (ignoring EMPTY), useful for characterising convoy texture and potential booth blocking patterns.

These metrics provide a **quantitative link** between per-booth SDSU choices and global oven behaviour.

## 9. Acceptance Criteria 

Before treating STEP-05 as valid for experiments:

1. `step05_global_line.csv` exists and is non-empty.  
2. Final C₂ verification passes: **no global adjacency violations** on the full stream.  
3. `n_jobs_booth1` and `n_jobs_booth2` match counts from STEP-04 inputs (job conservation per booth).  
4. If `allow_postpaint_empty = false`, then `n_deadlocks = 0` and `n_postpaint_empty = 0`.  
5. If `allow_postpaint_empty = true`, then:
   - any deadlock must be resolved by EMPTY (no unhandled deadlocks),  
   - `n_postpaint_empty` is recorded and considered in downstream objectives.  
6. Manifest and report exist, with SHA-256 hashes and effective settings.



---


## 10. Reproducibility and Reporting

STEP-05 adheres to the project’s manifest/report protocol:

- **Manifest JSON**:
  - `step = "step05_merge_global_line"`, `version = "1.0"`.  
  - Hashes of config and I/O files.  
  - `settings_effective` = serialized `Step05Config`.  
  - `metrics` = serialized merge metrics.  
  - `runtime_seconds`.

- **Report MD**:
  - Title “Step-05 — Oven Convoy Serialization (2→1 Merge)”.  
  - Purpose / What / Why.  
  - JSON block of effective settings.  
  - Merge metrics, including counts and max run lengths.  
  - Notes emphasising:
    - global adjacency is data-driven from `adjacency_rules.csv`,  
    - paint C₁/C₃ remain upstream,  
    - adjacency is enforced on the full stream with EMPTY as explicit tokens,  
    - determinism and reproducibility.


---

## 12. Execution Instructions (How to Repeat)

```bash
python step05_merge_global_line.py
```
* reads `global_config.toml` and `steps_config.toml` from the current working directory,
* loads STEP-04 per-booth outputs and adjacency rules,
* writes the global convoy, manifest, and report under `[paths.output_dir]`.


---

## 13. Failure Modes & Minimal Recovery

- **Frequent deadlocks (both heads infeasible):** enable post‑paint `EMPTY` as a last resort, or relax `scope_escalation` if data semantics justify it.  
- **Too many post‑paint empties:** increase `empty_units_postpaint_ratio`; reconsider selection policy (`feeder_order` often reduces conflicts).  
- **Unexpected convoy bias:** adjust selection policy (e.g., from `round_robin` to `feeder_order`) to better track Step‑02’s global intent.  
- **Non‑reproducible interleave:** ensure tie policy and feeder indices are stable and recorded in manifest.

---

## 14. Interface to Step‑06/07/08

* **Step-06 (MILP merge)**:
  `step06_merge_global_line.py` generalises the merge to MILP per-booth outputs and supports both `symbol_stream` and `parts_only` C₂ modes. Its configuration is chosen so that **STEP-05 is the canonical SDSU baseline** against which MILP merges are compared.

* **Step-07 (NSGA-II)**:
  NSGA decoders that build global convoys must honour the **same global C₂ semantics** (with or without EMPTY) so that f₁/f₂ comparisons against STEP-05 are meaningful.

* **Step-08 (DES)**:
  DES consumes `step05_global_line.csv` and uses adjacency-aware kernels where **EMPTY resets adjacency** in the simulation, consistent with the stream semantics described above.

---

## 15. Verification Checklist

* [ ] `step05_global_line.csv` present; sizes of PB1/PB2 job sets preserved.
* [ ] Manifest and report exist; SHA-256 hashes recorded.
* [ ] Final adjacency verification passed (no RuntimeError).
* [ ] `n_deadlocks = 0` if `allow_postpaint_empty = false`.
* [ ] `n_postpaint_empty` recorded and consistent with configs if `allow_postpaint_empty = true`.
* [ ] `max_run_booth1`/`max_run_booth2` reasonable given upstream split behaviour.
* [ ] Determinism assumptions documented (tie-break policy, adjacency scope, EMPTY policy).
