# STEP-02 — Toyota Grouping & Spacing Feeder (GS on Jobs)

## 1. Role in the MOFFS Protocol

STEP-02 constructs a **single, token-aware feeder stream** for paint:

- Input: the **job universe** from STEP-01 (`step01_jobs_expanded.csv`).
- Output: a linear sequence of tokens over `{START, JOB, EMPTY, FLUSH}` that:
  - Contains **each job exactly once** (`JOB`).
  - Respects **paint feasibility** (C1) using `s_lookup_transition.csv` ∪/or `s_matrix_transition.csv` plus any explicit forbids.
  - Respects **part adjacency forbids** (C2) using `adjacency_rules.csv`, applied on the single-file feeder with tokens semantics:
    - `EMPTY` resets adjacency; `FLUSH` does **not**.
  - Shapes color blocks via **window W** and **run caps** from config/empirical overrides.

This step does **not** assign booths or optimize MO objectives; it supplies a deterministic, data-driven **seed** for split → SDSU → merge → MILP/NSGA-II → DES.

---

## 2. Inputs & Config Bindings

All runtime paths are resolved from `global_config.toml` and `steps_config.toml`.  
**Every artifact written by this step is placed under** `global_config.toml[paths].output_dir`.

### 2.1 Data files (from `global_config.toml`)

- `[paths].dataset_dir` — root for datasets.
- `datasets.*` (or `data_files.*`) entries:
  - `s_lookup_transition` (preferred) or `s_matrix_transition`
  - `adjacency_rules`
- Optional: other files are **not** consumed by this step.

### 2.2 Step-01 product

- `step01_jobs_expanded.csv` located at:  
  `output_dir` + `steps_config.toml[step02].input_jobs_relpath`.

### 2.3 Step-02 configuration

From `global_config.toml`:

- `[window]`: `mode ∈ {"jobs","ratio"}`, `jobs`, `ratio`.
- `[spacing]`: `alpha_run_cap ∈ (0,1]`, `max_run_abs ≥ 1`.
- `[tokens]`: `flush_unit`, `empty_unit` (units → heuristic penalties in this step).
- `[adjacency]`: `fifo_lookahead` (scan depth within a color queue to find an adjacency-safe job).

From `step02_empirical.toml` (heuristic shaping only; semantics stay data-driven):

- `[defaults]`: `window_ratio`, `alpha_run_cap`, `max_run_abs`, `flush_weight`, `empty_weight`.
- `[topcoat.X]`: `priority_weight`, `max_run_jobs` (tightens cap for color X).

From `steps_config.toml[step02]` (all **relative to** `output_dir`):

- `input_jobs_relpath` — Step-01 CSV relative location.
- `paint_seed_relpath` — output CSV relative location.
- `manifest_relpath` — manifest JSON relative location.
- `report_relpath` — Markdown report relative location.
- `use_cycle_quota`, `export_jobs_only` (diagnostic/compat switches; no cycle quota enforced here).

---

## 3. Outputs & Invariants

### 3.1 Feeder CSV

Path: `output_dir` + `[step02].paint_seed_relpath`

Schema:

- `token_index` — 1..L over the entire stream.
- `token_type` — `{"START","JOB","EMPTY","FLUSH"}`.
- For `JOB` tokens: **all columns** from Step-01 (e.g., `job_id`, `part_id`, `topcoat_code`, provenance, etc.), plus:
  - `setup_cost` (S-units for the realized paint transition into this job’s color)
  - `cumulative_setup_cost`
- For other tokens: job columns blank.

Invariants:
1. **Job conservation:** exactly one `JOB` per Step-01 job; no duplicates/missing.
2. **Paint (C1):** every realized transition obeys S forbids (matrix/lookup and explicit forbids). `FLUSH` resets paint to neutral but **does not** legalize forbidden data transitions.
3. **Adjacency (C2):** consecutive **parts** (ignoring `EMPTY` gaps) never form a forbidden pair from `adjacency_rules.csv`. `EMPTY` resets adjacency; `FLUSH` does not.
4. **Window/run caps:** per-topcoat contiguous run on JOB-only projection is discouraged beyond
   \[ r_{\max}(c) = \min\{\lfloor \alpha W \rfloor,\ \text{max\_run\_abs},\ \text{topcoat.c.max\_run\_jobs (if any)}\} \]
   If all colors would exceed their caps simultaneously, the cap is relaxed; the event is counted in diagnostics.

### 3.2 Manifest JSON

Path: `output_dir` + `[step02].manifest_relpath`

Contains: step metadata; config file paths; effective settings; SHA-256 hashes of inputs/outputs; metrics; diagnostics; `output_dir` value; version tag.

### 3.3 Markdown Report

Path: `output_dir` + `[step02].report_relpath`

Sections: Purpose / What / Why / Config (effective JSON) / I/O hashes / Metrics / Diagnostics / Runtime.

---

## 4. Algorithm (Deterministic Greedy with FIFO Lookahead)

### State

- `last_paint` (topcoat at paint; neutral after `FLUSH`/`START`).
- `last_part_id` (for adjacency; reset by `EMPTY` only).
- `current_run_color`, `current_run_len`.
- `window_W = W` (used for cap computation; windowed mix penalty optional).

### Candidate enumeration

At each placement while jobs remain:

1. Enumerate colors with non-empty queues \(C\).  
2. For each \(c \in C\):
   - **Run projection**: `projected_len = current_run_len+1` if `c == current_run_color` else `1`; compute `run_ok` vs cap \(r_{\max}(c)\).
   - **Adjacency**: scan up to `fifo_lookahead` in queue \(Q_c\) to find first job whose `part_id` is not forbidden w.r.t. `last_part_id`. If none found, `adjacency_ok=False`.
   - **Paint feasibility**: test S forbids for direct `(last_paint → c)` and from neutral `(None → c)`.
   - Form up to four options (direct, `EMPTY+direct`, `FLUSH+place`, `EMPTY+FLUSH+place`) where feasible.
   - **Score** each option:
     \[ \text{score} = \Delta S + \mathbf{1}_{EMPTY}\cdot w_E\cdot \tau_E + \mathbf{1}_{FLUSH}\cdot w_F\cdot \tau_F + \max(0, \text{projected\_len} - r_{\max}(c)) \]
     where \( \tau_E,\tau_F \) come from `[tokens]`, and \( w_E,w_F \) from `step02_empirical.toml`.
3. Choose minimal score (ties: lower setup, then shallower depth).

### Conflict resolution

- If **only adjacency** blocks progress for at least one color: insert `EMPTY` (resets adjacency), count `adjacency_empty_events`, retry.
- Else **relax run cap** once (count `run_cap_relaxations`) to avoid deadlock if paint+C2 are otherwise feasible.
- If still impossible, the step fails with a deadlock certificate (hard data infeasibility).

---

## 5. CLI

```bash
python step02_toyota_gs.py \
    global_config.toml \
    steps_config.toml \
    step02_empirical.toml
```

- Outputs are written under `global_config.toml[paths].output_dir` using the relative paths in `steps_config.toml[step02]`.
- Manifest and report are emitted alongside the CSV under that output root.

---

## 6. Metrics & Gates

Recorded metrics:
- `num_jobs`, `window_W`, `setup_units`, `num_flush`, `num_empty`.
- Diagnostics: `adjacency_empty_events`, `run_cap_relaxations`, token distribution, `max_run_len_by_topcoat`.

Proceed to STEP-03 only if:
- CSV, manifest, report exist.
- No C1/C2 violations are indicated.
- Job conservation holds.
- Any cap relaxations are reported and acceptable.
