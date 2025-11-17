
# Step‑02 — Toyota Grouping & Spacing (Feeder; Paint‑Only with Adjacency Enforcement)

> This step follows the **Master Note** (`NOTE_MASTER.md`) and consumes the Step‑01 JOB table. It builds a **single‑file feeder** token stream that respects **paint** feasibility and **adjacency** at the feeder, while achieving **intended mix** via a parts‑only window \(W\) and run caps. All semantics, tokens, and parameters derive from `model_config.toml` (single source of truth).

---

## 1. Purpose — Formal and Intuitive

### Formal
Construct a deterministic tokenized feeder sequence `START, JOB, (EMPTY), (FLUSH)` such that:
1. **Paint feasibility (C₁).** For every consecutive pair of paint-relevant symbols in (S) (either `START`→JOB, JOB→JOB, or `FLUSH`→JOB), the induced topcoat transition is allowed by the SDST matrix and explicit paint forbids (`s_matrix_transition`, `topcoat_rules`, `paint_rules`).
2. **Adjacency feasibility on the feeder (C₂ᵍ).** For every consecutive JOB→JOB pair *without an intervening EMPTY*, the induced part pair is not forbidden by `adjacency_rules.csv` (global C₂). Tokens never change which part→part pairs are forbidden in the data; `EMPTY` only prevents certain pairs from becoming immediate neighbours.
3. **Mix shaping (f₂ pre-calibration).** In a sliding window of the last (W) JOBs, the realized mix over colors/parts stays close to targets implied by `cycle_quota.csv` and `[window]/[spacing]` settings; per-color run caps limit long homogeneous blocks.
4. **Determinism & traceability.** For fixed `global_config.toml`, `steps_config.toml`, `step02_empirical.toml`, and Step-01 input, the output stream is unique; the script emits a manifest and report with SHA-256 hashes and effective settings.

### Intuitive
A planner stands at the **single feeder** that precedes both paint booths:

* They want **few color changes** (to reduce purges/setup) but must maintain a **healthy mix** for downstream stability.
* If the next physical part would violate a **handling or contamination rule**, they leave one hook **empty** (`EMPTY`) to physically separate the parts.
* If the next topcoat is **paint-incompatible**, they run a **purge** (`FLUSH`) and only then spray the next color.
* They look only a **short distance into the queue** (FIFO lookahead), apply **per-color run caps**, and nudge the window mix toward targets.
* They never override the dataset: if the data says a transition or adjacency is forbidden, it is forbidden here and everywhere else.

The output is a **production-credible feeder sequence** that is then split (Step-03), reordered per booth (SDSU / MILP / NSGA-II), merged, and evaluated by DES.


---

## 2. Inputs / Outputs (Relative Paths)

All paths below are resolved via `[paths]` and `[data_files]` in `global_config.toml` and the per-step section in `steps_config.toml`. By default, datasets live under `./dataset/` and outputs under `./output/`.

### Inputs

1. **Configuration**

   * `global_config.toml` — system layout; data bindings; `[window]`, `[spacing]`, `[tokens]`, `[adjacency]`, `[paint_rules]`, `[part_rules]` for this step.
   * `steps_config.toml` (section `[step02]`) — relative filenames for input jobs, paint seed stream, manifest, and report; flag `use_cycle_quota`.
   * `step02_empirical.toml` — empirical overrides for spacing (`alpha_run_cap`, `max_run_abs`) and token weights (`empty_weight`, `flush_weight`), plus optional per-topcoat run caps and target mix.

2. **Data tables**

   * `step01_jobs_expanded.csv` — Step-01 output; one row per job with at least `job_id`, `part_id`, `topcoat_code` (names resolved via `data_schema.demands`).
   * `s_matrix_transition.csv` + `s_lookup_transition.csv` — paint SDST matrix and lookup, used to build `s_costs` and `s_forbids`.
   * `topcoat_rules.csv` — explicit paint forbids and purge requirements (unioned with the matrix forbids).
   * `adjacency_rules.csv` — part-level immediate successor forbids; Step-02 currently uses the **union** of rows marked forbidden or not-allowed, ignoring any `scope` column (interpreted here as global feeder C₂).
   * Optional `cycle_quota.csv` — cycle-based target quotas per part, if `step02.use_cycle_quota=true`.


### Outputs

1. `step02_paint_seed.csv` — tokenized feeder stream:

   * `token_index` — integer position (1-based, contiguous over the stream).
   * `token_type ∈ {START, JOB, EMPTY, FLUSH}`.
   * All columns from `step01_jobs_expanded.csv` for JOB rows (e.g., `job_id`, `part_id`, `topcoat_code`, provenance fields); these are null/empty for non-JOB tokens.
2. `manifests/step02_manifest.json` — manifest with:

   * `config_files` paths,
   * `io_hashes` (SHA-256) for all inputs/outputs,
   * `settings_effective` (W, run caps, token weights, schema),
   * `metrics` and `diagnostics` (see §10),
   * timestamp and version tag `"step02_gs_v2.0"`.
3. `reports/step02_report.md` — human-readable report with purpose, effective config (as JSON), metrics, diagnostics, and runtime stamp.


## 3. Feasibility Semantics (This Step)


* Allowed/forbidden color transitions are defined by:

  * finite entries in `s_matrix_transition` plus
  * any `relation=forbidden` entries in `topcoat_rules`,
    combined into a set `s_forbids`.
* The GS kernel maintains a `last_paint` state:

  * `last_paint=None` at `START` and immediately after any `FLUSH`.
  * For a candidate color `c`, it computes:

    * `cdir = cost(last_paint → c)` if allowed, else infeasible;
    * `cneu = cost(NEUTRAL → c)` via `"__NEUTRAL__"` row for `START`/`FLUSH`.
* A candidate that would require a **forbidden** paint transition is either:

  * realized via `FLUSH` (if a neutral path is configured and allowed), or
  * **discarded**; if no color is feasible even with purge, the step eventually fails with a `RuntimeError("Deadlock: no feasible candidate under C1/C2/quotas.")`.


### Adjacency feasibility on the feeder (C₂ᵍ)

* The kernel maintains `last_part`, representing the last **physical part** seen on the feeder. `last_part`:

  * starts as `None`,
  * is set on every JOB emission (to that job’s `part_id`),
  * is reset to `None` whenever `EMPTY` is emitted,
  * is **not** reset by `FLUSH` (purge has no physical gap).
* At candidate enumeration, a JOB with part `p_next` is adjacency-feasible iff:
  [
  (last_part, p_\text{next}) \notin \text{adjacency_forbids}.
  ]
  If this fails for all jobs of a given color within FIFO lookahead, that color is considered **adjacency-blocked** at the head.
* `EMPTY` is the **only** token that breaks adjacency: after emitting `EMPTY`, `last_part=None`, so the following JOB sees no previous part and cannot violate C₂ in the immediate step. This matches the physical interpretation of leaving a hook empty. `FLUSH` does not affect `last_part`.

### Mix shaping and windows

* A deque `window_colors` of length (W) tracks the colors of the last (W) JOBs (tokens do not enter the window).
* For a candidate with color `c`, the kernel computes a **window penalty** based on:

  * either a color-level target mix `target_mix_color` (if supplied via empirical config or derived from `cycle_quota`), using a simple quadratic deviation, or
  * a flatness penalty encouraging diversity when no explicit target mix is provided.

### Tokens and propagation

* `START` appears exactly once at the beginning; it is costed via the neutral row in the SDST kernel.
* `EMPTY` and `FLUSH` consume time and setup units via `token_costs` and weights (see §4.2), but they:

  * never change which part→part adjacencies are forbidden,
  * only influence *which* parts become immediate neighbours and which paint transitions are realized.
* Tokens in `step02_paint_seed.csv` are **not propagated** to per-booth scheduling:

  * Step-03 drops non-JOB tokens when constructing per-booth JOB streams.


## 4. Parameters (from `model_config.toml`) — Real‑World Meaning, Effects, Extremes

### 4.1 Window & Spacing

From `[window]` in `global_config.toml`:

* `mode ∈ {"jobs","ratio"}` and `jobs` / `ratio`:

  * If `mode="jobs"` and `jobs>0`, then (W = \text{jobs}).
  * Otherwise (W = \lceil \text{ratio} \cdot N_\text{jobs}\rceil), with a minimum of 1.
  * Small (W) ⇒ long homogeneous runs (setup-friendly, weak mixing).
  * Large (W) ⇒ strong pressure for mix, more purges/spacers.

From `[spacing]` in `global_config.toml` and `[defaults]` in `step02_empirical.toml`:

* `alpha_run_cap` (fraction of W) and `max_run_abs`:

  * For each color (c), the default run cap is (\min(\lfloor \alpha W\rfloor, \text{max_run_abs})).
  * Per-topcoat overrides `[topcoat.<code>].max_run_jobs` in `step02_empirical.toml` can further tighten caps.
  * Very small caps (`max_run_abs=1`) enforce alternation and may require many tokens.
* These caps feed into an internal **run cap penalty**; the kernel prefers candidates that keep runs within caps.


### 4.2 Tokens (time units & weights)

From `[tokens]` in `global_config.toml`:

* `flush_units_ratio` → `token_costs.flush_unit`.
* `empty_units_prepaint` → `token_costs.empty_unit`.

From `[defaults]` in `step02_empirical.toml`:

* `empty_weight` and `flush_weight` rescale these units in the candidate score:

  * Higher `empty_weight` discourages spacers; lower encourages more EMPTY.
  * Higher `flush_weight` discourages purges; lower encourages more FLUSH.

The effective contribution of a token to the candidate score is:

* `empty_weight * empty_unit` for EMPTY,
* `flush_weight * flush_unit` for FLUSH,
  plus the underlying setup units from `s_matrix_transition`.

### 4.3 Adjacency (Feeder)

From `[adjacency]` in `global_config.toml`:

* `enabled` (implicit: if false, this step can be run with adjacency ignored; the current script assumes adjacency is enabled when rules exist).
* `fifo_lookahead`:

  * Number of jobs inspected per color queue to find an adjacency-safe candidate.
  * `fifo_lookahead=1` ⇒ only queue head is tested; higher values can avoid unnecessary EMPTYs at extra compute cost.
* `scope_escalation` is not currently applied inside this step; all rows in `adjacency_rules.csv` marked forbidden (or not allowed) are treated as global forbids for the feeder.

### 4.4 Safety / Failure Policy

This step does **not** have explicit safety caps in config (e.g., `max_total_spacers`). Safety is instead enforced structurally:

* If there are still jobs in any color queue but no candidate satisfies C₁/C₂/quotas under the current parameters, the algorithm emits an `EMPTY` and retries.
* If *all* color queues are empty while `remaining>0` (should be impossible if data is consistent), it raises a `RuntimeError("Deadlock: no feasible candidate under C1/C2/quotas.")`.
* If a required paint transition is missing from the SDST/paint rules, it raises a `RuntimeError("Forbidden/unknown S transition: ...")`.

These runtime failures are treated as **hard step-gating failures**.

---

## 5. Quotas and Window Deficit

Let (W) be the effective window size and let (q_k) denote the target share for key (k) (either part, color, or family), with (\sum_k q_k = 1). From the implementation:

* If `use_cycle_quota=false`, (W) is used only in the sliding-window color penalty; no hard quota per window is enforced.
* If `use_cycle_quota=true` and `cycle_quota.csv` is present:

  * A **cycle size** is inferred as the total quota for the smallest cycle index.
  * For each part (p) and cycle (c), the allowed count is read from the file; a candidate that would exceed this usage is treated as blocked by `_cycle_quota_block`.

Conceptually, the window penalty encourages the realized window frequencies (\hat{p}_k) to stay close to (q_k), e.g. via
[
\text{penalty} \approx \sum_k (\hat{p}_k - q_k)^2.
]
Tokens do **not** contribute to window counts; only JOBs do.

---

## 6. Candidate Selection & Scoring

For each color with remaining jobs, the kernel computes candidates of four types (when feasible):

1. **A) Direct JOB** — place a JOB of this color:

   * Pick the earliest job in that color’s queue within `fifo_lookahead` that does **not** violate adjacency with `last_part`.
   * Feasible only if the paint transition from `last_paint` to this color is allowed.
   * Score:
     [
     s_A = \text{setup_cost} + \text{run_cap_penalty} + \text{window_penalty}.
     ]
2. **B) FLUSH + JOB** — purge then place JOB:

   * Uses the same job index as (A).
   * Feasible if the neutral transition `NEUTRAL→color` is allowed.
   * Score:
     [
     s_B = \text{neutral_setup_cost} + \text{flush_weight}\cdot\text{flush_unit}
     + \text{run_cap_penalty} + \text{window_penalty}.
     ]
3. **C) EMPTY + JOB** — spacer then JOB (only if adjacency is blocked at head):

   * If **no** job of this color within `fifo_lookahead` is adjacency-safe relative to the current `last_part`, the color is adjacency-blocked.
   * The candidate assumes an `EMPTY` is emitted, resetting `last_part`, and the JOB is placed from index 0 of that color’s queue.
   * Score:
     [
     s_C = \text{setup_cost} + \text{empty_weight}\cdot\text{empty_unit}
     + \text{run_cap_penalty} + \text{window_penalty}.
     ]
4. **D) EMPTY + FLUSH + JOB** — spacer + purge + JOB (adjacency blocked, paint change needed):

   * Combination of C) and B); uses index 0 after an EMPTY and a FLUSH.
   * Score:
     [
     s_D = \text{neutral_setup_cost}
     + \text{empty_weight}\cdot\text{empty_unit}
     + \text{flush_weight}\cdot\text{flush_unit}
     + \text{run_cap_penalty} + \text{window_penalty}.
     ]

Additional filters:

* Candidates that would violate cycle quotas are discarded.
* Among all remaining candidates across colors and patterns, the algorithm selects the one with **minimal score**, breaking ties by smaller `job_idx_in_queue` (earlier FIFO index).

---

## 7. Algorithm (Pseudocode)

```text
Input:
  jobs_df from STEP-01
  s_costs, s_forbids from SDST + paint rules
  adjacency_forbids from adjacency_rules.csv
  W, run_caps, token_costs, empty_weight, flush_weight, fifo_lookahead
  optional cycle_quota, cycle_size, target_mix_color

Init:
  queues_by_color = stable FIFO queues per topcoat
  last_paint = None
  last_part  = None
  window_colors = empty deque(maxlen = W)
  metrics = {num_jobs, window_W, num_empty, num_flush, setup_units, quota_violations}
  token_index = 1
  emit START (token_type="START")

Loop while any jobs remain:
  cand = _best_candidate()
  if cand is None:
    if any queues_by_color[c] non-empty:
      emit EMPTY (update metrics, last_part=None)
      continue
    else:
      raise RuntimeError("Deadlock: no feasible candidate under C1/C2/quotas.")

  if cand.use_empty:
    emit EMPTY; last_part = None

  if cand.use_flush:
    emit FLUSH; last_paint = None; metrics.num_flush++

  job_rec = pop job at cand.job_idx_in_queue from its color queue
  update cycle_quota counters; possibly metrics.quota_violations
  emit JOB row with all fields from job_rec
  update:
    last_paint = job_rec[topcoat]
    last_part  = job_rec[part]
    cur_run_color, cur_run_len, max_run_by_color
    window_colors.append(job_rec[topcoat])
    metrics.setup_units += setup_cost(last_paint_prev, last_paint)
    metrics.num_jobs (initially) unchanged; remaining--.

Output:
  Concatenate emitted rows into DataFrame; write to step02_paint_seed.csv
  Write manifest + report with config snapshot, metrics, diagnostics.
```


## 8. Complexity

Let (N) be the total number of jobs and (|T|) the number of distinct topcoats.

* Each job is enqueued and dequeued exactly once.
* At each step, `_best_candidate`:

  * inspects at most `fifo_lookahead` jobs per non-empty color queue, and
  * computes a constant-time score per candidate.

Therefore the time complexity is
[
O\big(N \cdot |T| \cdot \text{fifo_lookahead}\big)
]
in the worst case, typically much lower when queues thin out. Memory usage is (O(N)) for queues and (O(|T|)) for run caps and per-color stats.

---

## 9. Output Contract & Schema

`step02_paint_seed.csv` is the **single feeder canonical stream**:

* Exactly one row with `token_type="START"` at `token_index=1`.
* For each emitted token, `token_index` increments by 1 (no gaps, no duplicates).
* `token_type` ∈ {`START`, `JOB`, `EMPTY`, `FLUSH`} (uppercase).
* JOB rows contain all columns from `step01_jobs_expanded.csv` (job id, part id, color, provenance, etc.); non-JOB rows leave these null/empty.
* The realized stream is *intended* to satisfy:

  * no paint-forbidden transitions across `START`/`FLUSH`/JOB,
  * no adjacency-forbidden JOB→JOB without an intervening EMPTY.

Step-03 reads this CSV, filters to JOB rows, and uses `token_index` to preserve feeder order and to map back into the global seed when building per-booth streams.


---

## 10. Metrics & Diagnostics (for the Report)

The manifest and report record:

* **Metrics (`gs.metrics`):**

  * `num_jobs` — total number of JOB tokens (should equal the number of Step-01 rows).
  * `window_W` — effective W used.
  * `num_empty` — count of EMPTY tokens.
  * `num_flush` — count of FLUSH tokens.
  * `setup_units` — sum of setup units over all realized paint transitions, including from NEUTRAL after START/FLUSH.
  * `quota_violations` — number of times cycle quotas were exceeded (if cycle quotas are enabled).

* **Diagnostics:**

  * `token_distribution` — counts of each `token_type`.
  * `max_run_len_by_topcoat` — max contiguous JOB run length per color.
  * (Optional) `cycle_quota` metadata and `target_mix_color` if present.

These fields support f₁ (setup + token) and f₂ (mix) calibration and sanity checks.


---

## 11. Acceptance Criteria (Hard Gates)

For Step-02 to be considered successful in an experiment:

1. Script runs to completion **without** raising `RuntimeError` (deadlock or unknown S transition).
2. `step02_paint_seed.csv`, `step02_manifest.json`, and `step02_report.md` all exist, with non-empty SHA-256 hashes in the manifest.
3. `num_jobs` in metrics equals the JOB count in `step01_jobs_expanded.csv`.
4. Optional but recommended: an offline checker re-applies C₁ and C₂ to the produced stream and reports **zero violations**; any violation indicates either a data issue or a bug in GS.
5. Window/mix behaviour is *diagnostic* rather than hard-constrained here; large deviations from targets should be noted in the report but do not by themselves fail the step.

---

## 12. Reproducibility (Manifest & Report)

Reproducibility for Step-02 follows the project-wide manifest/report protocol:

* Manifest records:

  * content hashes for all inputs and outputs,
  * absolute paths of config files,
  * effective settings (W, run caps, token weights, schema),
  * metrics and diagnostics,
  * wall-clock runtime and version string.
* Report is a self-contained Markdown narrative:

  * Purpose / What / Why,
  * Config dump (as JSON),
  * Metrics & diagnostics,
  * Acceptance verdict (pass/fail) and any advisory comments (e.g., “high EMPTY usage”).

Given identical inputs and environment, re-running Step-02 should reproduce the same stream and manifest hashes.

---

## 13. Execution Instructions (How to Repeat)

```bash
python step02_toyota_gs.py global_config.toml steps_config.toml step02_empirical.toml
```

If arguments are omitted, the script defaults to these filenames in the current directory. It resolves all dataset and output paths via `global_config.toml` and `steps_config.toml`.

Downstream steps (Step-03+), MILP, NSGA-II, and DES always treat Step-02 as a **pure feeder**: they never re-interpret its tokens; they only consume JOB order and the implicit global seed.

---

## 14. Failure Modes & Minimal Recovery

Typical failure modes and suggested responses:

1. **`RuntimeError("Forbidden/unknown S transition: ...")`**

   * Indicates a missing or inconsistent entry between `s_matrix_transition` and `topcoat_rules`.
   * Recovery: fix the dataset (or setup mapping) so that every observed topcoat transition is either explicitly allowed (finite cost) or forbidden. Re-run Step-02.

2. **`RuntimeError("Deadlock: no feasible candidate under C1/C2/quotas.")`**

   * Indicates a structural deadlock even after inserting EMPTY whenever any color queue was non-empty.
   * Recovery options:

     * Inspect adjacency rules for overly strict patterns that make the instance infeasible.
     * Relax cycle quotas or disable `use_cycle_quota`.
     * Adjust `[window]` or `[spacing]` to avoid pathological W/run-cap combinations.

3. **Unexpected token flooding (large `num_empty`/`num_flush`) without explicit failure**

   * Not a hard error but a **degeneracy signal**.
   * Recovery: increase `empty_weight` or `flush_weight`, adjust `W` and run caps, or revisit adjacency rules.

All such observations should be noted in `step02_report.md` as part of the diagnostics.

---

## 15. Interface to Step‑03 (Contract)

Step-03 (`step03_booth_split.py`) treats Step-02 as providing the **canonical global feeder order** over JOBs:

* It reads `step02_paint_seed.csv`, resolves `token_type` and `token_index`, and filters to rows with `token_type="JOB"`.
* Feeder order is defined by **increasing `token_index` over JOB rows**; tokens are ignored when assigning jobs to booths.
* Step-03:

  * assigns each feeder JOB to either PB1 or PB2 to minimize the booth-balance objective (f_3^{\text{split}}),
  * produces purely JOB-only per-booth streams (`step03_booth1.csv`, `step03_booth2.csv`) and an assignment mapping.
* STEP-03 performs **no** paint or adjacency checks and **never introduces tokens**; C₁ and C₂ inside booths are enforced later by SDSU/MILP/NSGA-II, and C₂ on the merged oven line is enforced at merge (Step-05/06).

Thus, the contract is:

> Step-02 owns **global feeder C₂ and pre-paint C₁** on the single stream; Step-03 owns only the mapping “feeder JOB → booth”, preserving the feeder order.

---

## 16. Worked Micro‑Example (Didactic)

Consider a toy instance:

* Window (W=6).
* Two colors A and B with equal desired share.
* Adjacency forbids the immediate successor pair `(P2, P1)` (i.e., part `P1` cannot follow `P2` without a spacer).

Suppose the queue (by color) at some point is:

* Color A queue head: job with part `P2`.
* Color B queue head: job with part `P1`.
* `last_paint = A`, `last_part = P2`.

The kernel considers options:

1. **Direct A or B**

   * A: produces pair `(P2, P2)` — allowed if not forbidden; might exceed A’s run cap.
   * B: would produce `(P2, P1)` — **forbidden** by adjacency, so B is adjacency-blocked at head.

2. **FLUSH + B**

   * Paint domain: `FLUSH` resets `last_paint` to neutral; `NEUTRAL→B` may be allowed.
   * Adjacency domain: still `(P2, P1)` (FLUSH does not break physical adjacency), so C₂ still fails. FLUSH alone is insufficient.

3. **EMPTY + B**

   * Emit `EMPTY`: `last_part := None`.
   * Then place B: adjacency sees `(None, P1)`, which is allowed.
   * Score includes `empty_weight * empty_unit` plus B’s setup and window penalty.

The algorithm weighs these against alternative colors (if any) and may choose **EMPTY + B** if it provides the best trade-off between setup cost, spacing, and quotas. This demonstrates:

* `EMPTY` is the mechanism for resolving **part adjacency**,
* `FLUSH` is the mechanism for resolving **paint transitions**,
* (W) and run caps regulate mix and run length, not feasibility.

## 17. Verification Checklist (Before Proceeding)

Before running Step-03 or any optimization method seeded by this feeder:

* [ ] `step02_paint_seed.csv` exists and has:

  * [ ] exactly one `START` at `token_index=1`,
  * [ ] `token_index` contiguous from 1..T,
  * [ ] JOB counts matching `step01_jobs_expanded.csv`.
* [ ] `step02_manifest.json` and `step02_report.md` exist and reference the expected config paths.
* [ ] Metrics show reasonable `num_empty` and `num_flush` (no obvious flooding).
* [ ] `max_run_len_by_topcoat` aligns with configured run caps.
* [ ] If cycle quotas are enabled, `quota_violations` is 0 or acceptably small and explained.
* [ ] Optional offline checks confirm:

  * [ ] no paint forbids in the realized stream,
  * [ ] no adjacency forbids on JOB→JOB without an intervening EMPTY.

Once these conditions hold, Step-02 can be treated as the **fixed, deterministic feeder** for SDSU, MILP, NSGA-II, and DES in the rest of the pipeline.
