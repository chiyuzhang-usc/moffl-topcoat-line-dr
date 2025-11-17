# STEP-04 — Per-Booth SDSU Ordering (Non-Cyclic)

**Script:** `step04_sdsu.py`  
**Inputs (from previous steps):**

- `output/step03_booth1.csv` — PB1 JOB-only sequence from Step-03.
- `output/step03_booth2.csv` — PB2 JOB-only sequence from Step-03.

**Outputs (this step):**

- `output/step04_booth1_sdsu.csv` — PB1 JOB-only sequence after SDSU.
- `output/step04_booth2_sdsu.csv` — PB2 JOB-only sequence after SDSU.
- `output/step04_flush_positions.json` — conceptual FLUSH placements per booth.
- `output/manifests/step04_manifest.json` — manifest (I/O SHA-256, settings, metrics).
- `output/reports/step04_report.md` — step report (research-style narrative).

All relative paths above come from:

- `[paths].output_dir` in `global_config.toml`.
- `[step04]` section in `steps_config.toml`. 

---

## 1. Purpose & Position in Pipeline

Step-04 takes the **per-booth JOB-only sequences** produced by Step-03 (booth assignment) and **locally optimizes each booth’s order** for paint setups under the SDSU method, while preserving:

- **C1 (paint feasibility)**: topcoat transitions must be allowed by `topcoat_rules.csv` and have finite setup time in `s_matrix_transition.csv`.   
- **C2 (in-booth adjacency)**: forbidden successive part pairs (from `adjacency_rules.csv`) are never introduced inside a booth.  
- **C3 (purge / DW entry)**: transitions that require purge are realized conceptually as `…→FLUSH→…` with appropriate cost and allowed-target checks.

The outputs are **SDSU-refined booth sequences** and a **FLUSH map** (where purges would occur) suitable for:

- Step-05 (2→1 merge to oven convoy).
- MILP spot checks (Step-06) and NSGA-II (Step-07) for f₁ comparison.
- DES (Step-08) when computing time-based metrics and blocking.

No **EMPTY** tokens are introduced in Step-04; spacing remains the responsibility of the feeder (Step-02) and the global merge (Step-05). Tokens remain **conceptual** here (FLUSH only), used to compute cost and diagnostics, not to alter adjacency (C2 is always evaluated on parts-only). 

---

## 2. Inputs, Config, and Data Dependencies

### 2.1 Upstream Step Outputs

From Step-03 (Booth Assignment):

- `step03_booth1.csv`
- `step03_booth2.csv`

Schema (assumed, consistent with Step-01/02/03):

- `job_id` — unique job identifier.
- `part_id` — part identifier (links to `demands.csv` / `parts.csv`).
- `topcoat_code` — topcoat applied to this job.
- Additional metadata columns are preserved and passed through unchanged.

### 2.2 Dataset Files (from `global_config.toml`)

From `[paths].dataset_dir` + `[data_files].*`: 

- `topcoat_rules.csv` — C1/C3 rules:
  - `from_topcoat_code, to_topcoat_code, relation, purge_flag, allowed_targets`.
- `s_matrix_transition.csv` — SDST matrix:
  - Square matrix on `topcoat_code`, `INF` (or sentinel) means forbidden arc.
- `adjacency_rules.csv` — C2 adjacency forbids:
  - `from_part_id, to_part_id, scope, relation`.

Column names are detected via the `data_schema.*` synonym lists in `global_config.toml`. No renaming of dataset files is performed.

### 2.3 Configuration Keys

From `global_config.toml` (relevant groups): 

- `[paths]`
  - `dataset_dir` — base dir for raw CSVs.
  - `output_dir` — base dir for step outputs.
- `[paint_rules]`
  - `use_flush_token` — whether FLUSH may be used at all.
  - `flush_cost_time` — cost/time of a single FLUSH.
  - `flush_allowed_targets` — `"DW_only"`, `"any"`, or explicit list of topcoats.
  - `flush_requires_change` — if true, FLUSH only allowed when topcoat changes.
- `[adjacency]`
  - `enabled` — whether C2 checks are active.
  - `scope_escalation` — how to treat `scope` labels (`global_if_missing` etc.).
- `[tokens]`
  - `flush_units_ratio` — used in f₁ scaling (paint-unit normalization).

From `steps_config.toml` `[step04]`: 

- `booth1_input_relpath`, `booth2_input_relpath`.
- `booth1_output_relpath`, `booth2_output_relpath`.
- `flush_positions_relpath`.
- `manifest_relpath`, `report_relpath`.
- `dp_threshold_jobs` — size cutoff controlling how aggressive the local search is (used as a guard for exhaustive 2-opt neighborhood exploration).

---

## 3. Outputs and Invariants

### 3.1 SDSU Booth CSVs

Files:

- `output/step04_booth1_sdsu.csv`
- `output/step04_booth2_sdsu.csv`

Schema:

- Exactly the **same columns** as the corresponding Step-03 booth file.
- Rows reordered to minimize per-booth f₁ (setup + FLUSH cost) under C1/C2/C3.

Invariants per booth:

1. **Job conservation.** Multiset of `job_id` is unchanged from Step-03; no new jobs; no duplicates.
2. **Paint feasibility (C1/C3).** For every successive **parts** pair `i→j` in the SDSU sequence:
   - Either there is a direct allowed transition `(τ(i), τ(j))` with finite `s[τ(i), τ(j)]`, or
   - The pair requires purge and is realized conceptually as `i→FLUSH→j`, respecting `purge_flag` + `allowed_targets` and global `flush_allowed_targets`. 
3. **Adjacency feasibility (C2 in-booth).** For every successive **parts** pair `u→v`:
   - `(u, v)` must **not** appear as forbidden under scope `in_booth` or `global` (plus any escalation policy configured).
   - Tokens are **not** used to bypass adjacency: C2 runs on parts-only.
4. **Non-cyclic open path.** Sequences are open paths (no wrap-around edge from last job back to first). SDSU operates on this open path; first-job setup is accounted for via the cost model (either explicit “START” row in the SDST matrix or zero initial cost, depending on data).

### 3.2 FLUSH Positions JSON

File:

- `output/step04_flush_positions.json`

Structure:

```json
{
  "booth1": [
    {
      "position": 10,
      "job_id": "J0011",
      "from_topcoat": "PW",
      "to_topcoat": "DW"
    }
  ],
  "booth2": [
    {
      "position": 5,
      "job_id": "J0206",
      "from_topcoat": "X1",
      "to_topcoat": "X3"
    }
  ]
}
