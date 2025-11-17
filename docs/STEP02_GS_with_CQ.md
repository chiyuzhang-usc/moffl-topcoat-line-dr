# STEP‑02 — Toyota Grouping & Spacing Feeder (GS with Optional Cycle‑Quota)

## 1. Role in the MOFFS Protocol

STEP‑02 builds the **single, token‑aware feeder stream** for the paint stage.

- **Input:** the job universe from STEP‑01 (`step01_jobs_expanded.csv`), i.e., one row per physical job with attached part, color, and metadata.
- **Output:** a linear stream of tokens over `{START, JOB, EMPTY, FLUSH}` that:
  - Contains **each job exactly once** (`JOB` tokens, job conservation).
  - Respects **paint feasibility (C1)** using the S‑transition data (`s_lookup_transition.csv` and/or `s_matrix_transition.csv` plus paint forbids).
  - Respects **part‑level adjacency forbids (C2)** on the **single‑file feeder**, using `adjacency_rules.csv`.
  - Shapes the **color mix** using a Toyota‑style **Grouping & Spacing (GS)** heuristic:
    - **Grouping:** minimizes unfavorable color changes (via S‑costs) and allows short color clusters.
    - **Spacing:** limits continuous runs via **run caps** and a **sliding‑window mix penalty**.
  - Optionally enforces **cycle‑quota** (if enabled) to match a per‑cycle part composition from `cycle_quota.csv`.

This is the project’s formalization of the Toyota **Group & Space** method at the feeder: we greedily pick the next job to keep similar colors together (grouping) while avoiding long solid blocks and respecting adjacency and (optional) cycle quotas (spacing).

The feeder sequence from STEP‑02 is the **only paint‑aware, adjacency‑aware seed** consumed by downstream steps: split (feeder → booth streams), SDSU, MILP, NSGA‑II, and DES. No other step is allowed to “invent” or relax paint/adjacency semantics.


---

## 2. Inputs, Paths, and Config Bindings

### 2.1 Global paths and datasets

All runtime paths are resolved from `global_config.toml` and `steps_config.toml`.

From `global_config.toml`:

- `[paths].output_dir`  
  Root directory for **all step outputs**. STEP‑02 will write everything under this directory.
- `[paths].dataset_dir`  
  Root directory for all static data files (`demands.csv`, `s_*`, `adjacency_rules.csv`, `cycle_quota.csv`, etc.).
- `[datasets]` or `[data_files]` (one of them is used):
  - `s_lookup_transition` → `s_lookup_transition.csv` (preferred S‑transition form).
  - `s_matrix_transition` → `s_matrix_transition.csv` (matrix form; used when lookup is absent).
  - `adjacency_rules` → `adjacency_rules.csv`.
  - `cycle_quota` → `cycle_quota.csv` (optional; only used if STEP‑02 is configured to use cycle quota).

Resolution rule: each dataset is interpreted **relative to `[paths].dataset_dir`**, unless an absolute path is given.

### 2.2 Step‑01 product (jobs expanded)

From `steps_config.toml[step02]`:

- `input_jobs_relpath`  
  Relative path (from `output_dir`) to the Step‑01 output, conventionally:
  - `step01_jobs_expanded.csv`

Actual location at runtime:

```text
jobs_expanded_csv = [paths].output_dir / [step02].input_jobs_relpath
```

### 2.3 Column schema (data‑driven, no hard‑coding)

Column names are resolved via `global_config.toml[data_schema.demands]` plus the actual CSV header. STEP‑02 uses three logical fields:

- **Topcoat / color id** (for S‑transitions and GS):
  - Synonyms list (from `data_schema.demands.topcoat_code_synonyms`, then fallbacks):  
    `["color_code", "topcoat_code", "topcoat", "paint_color", "coat_code", "color"]`
- **Part id** (for adjacency and cycle quota):
  - Synonyms list (from `data_schema.demands.part_id_synonyms`, then fallbacks):  
    `["part_id", "part", "part_code"]`
- **Job id** (for job conservation / debugging):
  - Synonyms list (from a small default list):  
    `["job_id", "job", "id", "jobid"]`

The code **auto‑selects** the first synonym that matches the Step‑01 header. For this dataset, it will effectively resolve to:

- `topcoat_col = "color_code"`
- `part_col    = "part_id"`
- `jobid_col   = "job_id"` (assuming Step‑01 wrote it, per `Finished_Step01.md`).

All semantics are attached to **logical fields**, not literal column names.

### 2.4 S‑transition data (C1)

STEP‑02 supports two equivalent forms for paint transitions:

1. **Lookup form** (`s_lookup_transition.csv`):  
   Expected logical columns (name resolved via schema and heuristics):
   - `from` (previous color)
   - `to` (next color)
   - `cost` (S‑units)
   - `allow` / `forbid` flags (optional; if absent, all listed transitions are allowed).

2. **Matrix form** (`s_matrix_transition.csv`):  
   - Row/column labels are colors.
   - Each cell is the S‑cost for that color change.

Precedence: lookup form overrides matrix if both exist. Any **explicit forbids** in the lookup file are treated as **hard C1 constraints**.

Neutral start (`START` / `FLUSH` reset) is treated as a special pseudo‑color `"__NEUTRAL__"`; transitions from neutral may have an explicit cost if present, otherwise default to `0` S‑units.

### 2.5 Adjacency data (C2)

From `adjacency_rules.csv` (path resolved via `[datasets].adjacency_rules`):

- Logical columns (via `data_schema.adjacency_rules` and synonyms):
  - previous part: e.g. `from_part_id` / `prev_part` / `from` / `prev`
  - next part: e.g. `to_part_id` / `next_part` / `to` / `next`
  - forbid / allow flag

Semantics:

- A pair `(prev_part_id, next_part_id)` is **forbidden** if:
  - an explicit `forbid` flag is truthy **or**
  - an `allowed` flag is present and is **false**/**0**/**no**.
- If no explicit allow/forbid flags exist, any row present is treated as a **forbid**.

C2 is always evaluated on the **single‑file feeder** between successive `JOB` tokens, with **token semantics**:

- `EMPTY` **breaks adjacency** (resets the previous part to `None`).
- `FLUSH` does **not** reset adjacency; it only resets paint (C1).

### 2.6 Optional cycle‑quota data

From `cycle_quota.csv` (if `[step02].use_cycle_quota = true` in `steps_config.toml`):

- Logical columns, from `data_schema.cycle_quota` and synonyms:
  - `cycle` index,
  - `part_id` (must match the same logical field as in jobs),
  - `quota` (allowed quantity per `cycle` and `part_id`).

The **cycle size** is inferred as:

- `cycle_size = sum(quota for all rows with cycle = first_cycle)`

If additional cycles are not defined in the file, the quota pattern of the first cycle is **reused** for later cycles.


---

## 3. Outputs and Invariants

### 3.1 Feeder CSV (`paint_seed`)

Path:

```text
[paths].output_dir / [step02].paint_seed_relpath
```

Schema:

- `token_index` — 1..L over the entire stream.
- `token_type` — one of `{"START", "JOB", "EMPTY", "FLUSH"}`.

For `JOB` tokens:

- **All columns from Step‑01** (`step01_jobs_expanded.csv`), unmodified:
  - includes `job_id`, `part_id`, `color_code`, and any other metadata.
- Additional fields:
  - `setup_cost` — realized S‑cost for the transition into this job’s color.
  - `cumulative_setup_cost` — running sum of S‑costs along the stream.

For `START`, `EMPTY`, and `FLUSH` tokens:

- Only `token_index` and `token_type` are populated; job columns are blank.

**Invariants:**

1. **Job conservation.** Every Step‑01 job appears **exactly once** as a `JOB` token; no duplicates, no missing jobs.
2. **Paint feasibility (C1).** Every realized color transition between successive `JOB` tokens is **allowed** by S‑data:
   - If a transition is explicitly forbidden (`s_lookup_transition` or inferred from the matrix), it never appears.
   - Transitions from neutral (`START` or after `FLUSH`) use the `("__NEUTRAL__", color)` entries if provided; otherwise zero cost.
3. **Adjacency feasibility (C2).**
   - Consider the subsequence of `JOB` tokens, ignoring `START`, `FLUSH`, and `EMPTY`.
   - For any consecutive pair `(prev_JOB, next_JOB)` in that subsequence:
     - The pair `(prev_JOB.part_id, next_JOB.part_id)` is **never** listed as a forbidden adjacency in `adjacency_rules.csv`.
   - Semantics of tokens:
     - `EMPTY` **breaks** adjacency (resets previous part to `None`).
     - `FLUSH` does **not** break adjacency; it only resets paint.
4. **GS behavior (grouping & spacing).**
   - **Grouping:** short runs of the same color are encouraged because:
     - S‑costs for staying in the same color are low or zero.
     - The heuristic penalizes transitions with higher S‑cost.
   - **Spacing:** long solid blocks are discouraged because:
     - Per‑color **run caps** bound the maximum contiguous run length.
     - A **sliding‑window mix penalty** (over color codes) penalizes imbalanced local mixes.
5. **Cycle‑quota feasibility (if enabled).**
   - The stream is partitioned into cycles of inferred length `cycle_size` (jobs only).
   - For each cycle `k` and each `part_id`:
     - The number of `JOB`s with that `part_id` in cycle `k` does not exceed the quota from `cycle_quota.csv`.
     - If quotas are tight and the heuristic must violate them to avoid deadlock, violations are counted and reported explicitly (diagnostics).

### 3.2 Manifest JSON

Path:

```text
[paths].output_dir / [step02].manifest_relpath
```

Content (high‑level):

- `step`: `"step02"`
- `purpose`, `what`, `why`: textual description of the step role.
- `config_files`: absolute paths to `global_config.toml`, `steps_config.toml`, `step02_empirical.toml`.
- `io_hashes`:
  - `inputs`: SHA‑256 hashes of all consumed files (configs, Step‑01 output, S‑matrix, adjacency, cycle_quota).
  - `outputs`: SHA‑256 hash of the feeder CSV.
- `settings_effective`:
  - `window_W`, `alpha_run_cap`, `max_run_abs`,
  - token costs and weights,
  - `run_caps` per color,
  - `colors` list,
  - `use_cycle_quota` flag and `cycle_size` (if used),
  - resolved schema (`topcoat`, `part`, `job_id`),
  - `output_dir`.
- `metrics`:
  - `num_jobs`, `window_W`, `setup_units`, `num_flush`, `num_empty`, `quota_violations`.
- `diagnostics`:
  - `token_distribution` (counts of each token type),
  - `max_run_len_by_topcoat`,
  - `quota_violations` (same as in metrics).
- `generated_at` (timestamp), `version`.

### 3.3 Report Markdown

Path:

```text
[paths].output_dir / [step02].report_relpath
```

Sections:

- Purpose / What / Why.
- Effective config (JSON snapshot).
- Key metrics and diagnostics.
- Runtime timestamp.

The report is intended as a **human‑readable gate document**, not a code dump.


---

## 4. Algorithm — Why This Is GS

### 4.1 State variables

The heuristic maintains the following state while building the stream:

- `last_paint` — topcoat at paint (reset to neutral after `FLUSH`/`START`).
- `last_part` — part id of the last `JOB` (reset to `None` by `EMPTY` only).
- `cur_run_color`, `cur_run_len` — current contiguous run in the `JOB` projection.
- `window_colors` — deque of the last `W` `JOB` colors (sliding window for mix penalty).
- `max_run_by_color[color]` — maximum observed contiguous run length for each color.
- If cycle‑quota is enabled:
  - `cycle_size` — inferred length of a quota cycle (jobs per cycle).
  - `cycle_counts[(cycle_index, part_id)]` — how many jobs of `part_id` have been placed in cycle `cycle_index`.

Queues:

- For each color `c`, we maintain a **FIFO queue** `Q_c` of jobs with that color, preserving Step‑01 order.

### 4.2 Candidate enumeration (Toyota‑style grouping & spacing)

At each step (while jobs remain), we enumerate candidate choices over **colors** rather than individual jobs, reflecting a Toyota GS picker acting on color groups:

1. Identify all **available colors** `C = {c : Q_c not empty}`.
2. For each `c ∈ C`:
   1. **Adjacency head scan:**  
      - Within `Q_c`, scan from the head up to depth `fifo_lookahead` to find the first job index `idx_ok` whose part is **not** forbidden with respect to `last_part`.  
      - If none found, then adjacency is effectively **blocked** for color `c` at the head (within allowed lookahead).
   2. **Paint feasibility:**  
      - Compute S‑cost and feasibility for:
        - Direct transition `last_paint → c`.
        - Neutral transition `NEUTRAL → c` (after a `FLUSH`).
   3. **Run cap penalty:**  
      - If choosing `c` would make the contiguous run exceed `run_caps[c]`, a large penalty is added to the candidate score (conceptually discouraging, but the algorithm can still select it if all alternatives are worse).
   4. **Cycle‑quota pre‑check (if enabled):**  
      - Determine current cycle index based on how many `JOB`s have been placed so far and `cycle_size`.
      - Probe the head job (or index 0, depending on the option) and see if placing that part in this cycle would exceed its quota.  
      - If yes, the candidate is discarded (soft block) at this stage.

For each color, the picker considers up to **four options**:

- **A. Direct placement (`JOB` only)**  
  Preconditions: `idx_ok` exists (adjacency ok) and paint direct is feasible.  
  Score components:
  - S‑cost of `last_paint → c`,
  - run‑cap penalty,
  - sliding‑window mix penalty.

- **B. `FLUSH + JOB`**  
  Preconditions: `idx_ok` exists and paint from neutral is feasible.  
  Score components:
  - S‑cost of `NEUTRAL → c`,
  - token cost for `FLUSH` scaled by `flush_weight`,
  - run‑cap penalty,
  - sliding‑window mix penalty.

- **C. `EMPTY + JOB` (adjacency spacer only)**  
  Preconditions: `idx_ok` does **not** exist (adjacency blocked) and paint direct is feasible.  
  Behavior:
  - Insert `EMPTY` to **break adjacency** (reset `last_part`),
  - Place the head job (`idx = 0`).  
  Score components:
  - S‑cost of `last_paint → c`,
  - token cost for `EMPTY` scaled by `empty_weight`,
  - run‑cap penalty,
  - sliding‑window mix penalty.

- **D. `EMPTY + FLUSH + JOB`**  
  Preconditions: `idx_ok` does **not** exist (adjacency blocked) and paint from neutral is feasible.  
  Behavior:
  - Insert `EMPTY` (break adjacency),
  - Insert `FLUSH` (reset paint),
  - Place the head job.  
  Score components:
  - S‑cost of `NEUTRAL → c`,
  - token cost for `EMPTY` and `FLUSH`,
  - run‑cap penalty,
  - sliding‑window mix penalty.

**Crucial GS rule:**

> **EMPTY is available only as an adjacency spacer.**  
> The algorithm **never inserts EMPTY solely to switch colors** or to tweak mixing. Color changes are managed by choosing among color queues and, if needed, by using `FLUSH`, not by inserting fictitious empty jobs.

### 4.3 Scoring and selection (grouping vs spacing trade‑off)

For each feasible option, we compute a scalar score:

\[
\text{score} =
    S(\text{transition}) +
    \underbrace{\text{run\_cap\_penalty}(c)}_{\text{spacing/token stability}} +
    \underbrace{\text{token\_costs}}_{\text{EMPTY/FLUSH, if used}} +
    \underbrace{\phi_{\text{window}}(c)}_{\text{mix/flatness penalty}}
\]

Where:

- \(S(\text{transition})\) is the S‑cost of the chosen color change (from direct or neutral).
- `run_cap_penalty(c)` is large when the projected run `cur_run_len+1` exceeds `run_caps[c]`.
- `token_costs` encodes the relative “pain” of inserting `EMPTY` and/or `FLUSH` based on `[tokens]` and empirical weights.
- \(\phi_{\text{window}}(c)\) is the **sliding‑window mix penalty**:
  - If `target_mix` is provided (via `step02_empirical.toml`), we penalize the squared deviation between observed frequency in the augmented window and the target vector.
  - Otherwise, we penalize deviations from a **local uniform** distribution (flattening effect).

The algorithm selects the candidate with **minimum score**. Ties are broken deterministically (e.g., by smaller job depth and lexicographic color order).

This is exactly the **GS trade‑off**:
- grouping is favored by lower S‑costs,
- spacing is favored by run caps and mix penalties,
- “weird” spacers are discouraged by token costs and by forbidding EMPTY unless adjacency forces it.

### 4.4 Cycle‑quota integration (optional)

If `[step02].use_cycle_quota = true` in `steps_config.toml`, the algorithm additionally enforces **per‑cycle part quotas**:

1. **Cycle index:** the `JOB` sequence is divided into blocks of length `cycle_size` (inferred from `cycle_quota.csv`). The k‑th block corresponds to cycle `k`.
2. **Quota pre‑check:** during candidate enumeration, any candidate that would fill a part beyond its allowed quota in the **current** cycle is discarded when possible.
3. **Quota accounting:** after each `JOB`, the per‑cycle, per‑part counter is incremented and checked. If a strict conflict is unavoidable (e.g., data infeasible under C1/C2/quotas), the algorithm can still proceed but increments `quota_violations` as a diagnostic.

Cycle‑quota acts as a **higher‑level spacing constraint** in the “time” dimension (cycles), while GS acts at the local mixed‑color window level.


---

## 5. Config Knobs and Interpretation

Relevant configuration locations:

- `global_config.toml[window]`
  - `mode ∈ {"jobs","ratio"}` — choose window by absolute job count or ratio of total jobs.
  - `jobs` / `ratio` — window size `W` or window fraction.
- `global_config.toml[spacing]`
  - `alpha_run_cap` — fractional cap relative to `W` (base for `run_caps` per color).
  - `max_run_abs` — absolute max contiguous run per color.
- `global_config.toml[tokens]`
  - `flush_unit`, `empty_unit` — base units used to weight `FLUSH` vs `EMPTY`.
- `global_config.toml[adjacency]`
  - `fifo_lookahead` — maximum depth per color queue for adjacency‑safe job search.
- `global_config.toml[data_schema.*]`
  - `demands` and `cycle_quota` sections define synonyms for column names.
- `step02_empirical.toml`
  - `[defaults]`: `window_ratio`, `alpha_run_cap`, `max_run_abs`, `empty_weight`, `flush_weight`.
  - `[topcoat.X]`: `max_run_jobs`, `priority_weight` (tightens caps for specific colors).
  - `target_mix`: optional color‑level mix target.
- `steps_config.toml[step02]`
  - `input_jobs_relpath`, `paint_seed_relpath`, `manifest_relpath`, `report_relpath` (all relative to `output_dir`).
  - `use_cycle_quota`: `true`/`false` (controls whether `cycle_quota.csv` is used).


---

## 6. Gates for Downstream Steps

Before proceeding to STEP‑03 (split) and SDSU/MILP/NSGA‑II, we require:

1. Feeder CSV, manifest, and report files exist under `output_dir` with the expected relative paths.
2. Job conservation holds (exact match between Step‑01 job count and number of `JOB` tokens).
3. No explicit C1 or C2 violation is detected:
   - All color transitions are in the allowed S‑transition set.
   - No forbidden adjacency pair appears in the JOB subsequence.
4. Run‑caps, window‑mix behavior, and (if enabled) cycle quotas are acceptable:
   - Gaps and color groups match the configured GS behavior (no degenerate all‑block behavior unless forced by data).
   - Any `quota_violations` are inspected and justified in the context of data feasibility.

Under these gates, this STEP‑02 implementation is a **Toyota GS feeder** for the MOFFS protocol: it produces a **deterministic**, **data‑driven**, **C1/C2‑feasible** grouped‑and‑spaced color stream that is suitable as the sole seed for all downstream optimization and simulation work.
