
# Step‑04 — SDSU per Booth (Non‑Cyclic Open‑Path Optimization; Paint‑Domain with Adjacency Preservation)

> Coherent with `NOTE_MASTER.md` and `MOFFS_Protocol.md`. Implements the per-booth SDSU kernel as coded in `step04_sdsu_per_booth.py` using `global_config.toml`, `steps_config.toml`, and the SDSU method specification in `SDSU_Method.md`. This step optimises each booth independently for ( f_1 ) (setup + purge) under in-booth paint C₁ and adjacency C₂, with optional purge C₃, without changing the job set per booth.

---

## 1. Purpose 

### Formal

For each paint booth ( b \in {1,2} ) with a JOB set ( S_b ) coming from Step-03 and ( |S_b| = n_b ), we seek a permutation ( \pi ) of ( {1,\dots,n_b} ) that minimises the open-path SDSU objective
[
\text{Cost}*b(\pi)
= \sum*{k=1}^{n_b-1} c\bigl(\pi_k,\pi_{k+1}\bigr),
]
with arc cost
[
c(i,j) = \min{, S(c_i,c_j),; \text{FLUSH}_{\text{units}} + s_0(c_j) ,},
]
subject to

1. **Paint feasibility (C₁ᵇ).** An arc ( i \to j ) is allowed only if either

   * the direct SDST transition ( S(c_i,c_j) ) from `s_matrix_transition.csv` is finite and not forbidden by `topcoat_rules.csv`, or
   * `allow_purge = true` and a “purge + restart” path into colour ( c_j ) is allowed, encoded by a finite purge cost (\text{FLUSH}_{\text{units}} + s_0(c_j)).
2. **In-booth adjacency feasibility (C₂ᵇ).** For successive jobs (i,j) with parts ((p_i,p_j)), the pair must **not** be listed in `adjacency_rules.csv` as forbidden with `scope ∈ {in_booth, global}`. Purges never waive adjacency; if ((p_i,p_j)) is forbidden, there is no feasible arc.
3. **Open-path structure.** Every job in (S_b) appears exactly once; there is no return arc from the last job back to the first.

The **per-booth SDSU objective** is thus an open-path ATSP-Path problem over the job set for that booth. The step’s contribution to ( f_1 ) is
[
f_1^{(b)} = \text{Cost}_b(\pi^\star_b),
]
and the total Step-04 ( f_1 ) is the sum over both booths.

### Intuitive
Within each booth we now have a fixed set of JOBs, already assigned by Step-03 and consistent with the Toyota GS feeder. Step-04 is the **local paint scheduler** for each booth:

* It **re-orders jobs** inside each booth to reduce colour changes and costly purges.
* It respects **in-booth handling/contamination rules** (C₂ᵇ) derived from part-level adjacency data.
* It may introduce **purge (FLUSH) events** between consecutive jobs when that path is cheaper or necessary to realise a colour change.
* It never inserts `EMPTY` hooks, never adds or removes jobs, and does not look at the downstream oven.

Outputs are per-booth JOB-only streams in SDSU order plus a JSON file that records where purges would occur.



## 2. Connections to Neighbor Steps

* **Step-03 → Step-04.** Inputs are per-booth JOB lists in feeder-consistent order: `step03_booth1.csv` and `step03_booth2.csv`. Step-04 keeps the **set** of jobs per booth fixed and changes only their order (and implicit purge positions).
* **Step-04 → Step-05.** The SDSU-ordered per-booth CSVs `step04_booth1_sdsu.csv` and `step04_booth2_sdsu.csv` are the inputs to the 2→1 merge that enforces global adjacency and spacing on the oven line. `step04_flush_positions.json` is **not** consumed by the merge; it is used later in analysis (Step-09) to decompose ( f_1 ) into setup vs purge contributions.
* **Relation to MILP (Step-06) and NSGA-II (Step-07).** MILP and NSGA-II solve larger multi-objective problems over the merged system. Step-04 _may_ provide paint-domain baselines and seeds (via per-booth SDSU sequences) for those global methods.

## 3. Inputs / Outputs (Relative Paths)

### 3.1 Configuration

From `global_config.toml`:

* `[paths]`

  * `dataset_dir` — root for static data files.
  * `output_dir` — root for step outputs.

* `[data_files]` (dataset bindings)

  * `s_matrix_transition` — SDST colour→colour transition matrix.
  * `topcoat_codebook` — mapping from `topcoat_code` to numeric `topcoat_no`.
  * `topcoat_rules` — additional topcoat transition rules (forbids).
  * `adjacency_rules` — part-level adjacency forbids with scopes.

* `[data_schema.demands]`

  * `topcoat_code_synonyms` — candidates for the colour column (e.g., `["topcoat_code","color_code"]`).
  * `part_id_synonyms` — candidates for the part identifier column (fallback to `[data_schema.parts].part_id_synonyms`).

* `[data_schema.adjacency_rules]`

  * `from_part_id_synonyms`, `to_part_id_synonyms`.
  * `scope_synonyms`, `relation_synonyms` (to identify scoped forbids).

* `[sdsu]`

  * `dp_threshold_jobs` — booth size threshold for exact DP vs heuristic.
  * `max_2opt_passes` — maximum number of 2-opt passes in the heuristic branch.
  * `allow_purge` — if `false`, purge arcs are disabled and only direct SDST arcs are used.
  * `purge_cost_weight` — weight of purge cost relative to SDST units.

* `[tokens]`

  * `flush_units_ratio` — converts the abstract FLUSH into paint-unit cost; used as `flush_units_ratio * purge_cost_weight`.

* `[adjacency]`

  * `enabled` — if `false`, Step-04 ignores adjacency forbids and behaves as a pure paint SDSU.

From `steps_config.toml` (`[step04]`):

* `booth1_input_relpath` — default `step03_booth1.csv`.
* `booth2_input_relpath` — default `step03_booth2.csv`.
* `booth1_output_relpath` — default `step04_booth1_sdsu.csv`.
* `booth2_output_relpath` — default `step04_booth2_sdsu.csv`.
* `flush_positions_relpath` — default `step04_flush_positions.json`.
* `manifest_relpath` — default `manifests/step04_manifest.json`.
* `report_relpath` — default `reports/step04_report.md`.
* `dp_threshold_jobs` — optional per-step override of `[sdsu].dp_threshold_jobs`.

All relative paths are resolved against `dataset_dir` or `output_dir` in `global_config.toml`.

### 3.2 Data Tables

**Per-booth JOB inputs (from Step-03)**

* `${output_dir}/step03_booth1.csv`
* `${output_dir}/step03_booth2.csv`

Each row is a single JOB with at least:

* `job_id`
* `part_id` (or synonym)
* `topcoat_code` (or synonym)
* `booth_id ∈ {1,2}`
* `booth_seq_index` (1-based index within the booth in feeder order)
* `feeder_job_index`, `feeder_token_index` (traces back to the global sequence)

Additional columns from Step-01/02 (demand attributes, provenance) are preserved and propagated through Step-04.

**Paint and adjacency datasets (shared across steps)**

* `${dataset_dir}/s_matrix_transition.csv`
* `${dataset_dir}/topcoat_codebook.csv`
* `${dataset_dir}/topcoat_rules.csv`
* `${dataset_dir}/adjacency_rules.csv`

### 3.3 Outputs

**Per-booth SDSU CSVs**

* `${output_dir}/step04_booth1_sdsu.csv`
* `${output_dir}/step04_booth2_sdsu.csv`

Same schema as the corresponding Step-03 inputs, with jobs **reordered** per booth and:

* `booth_seq_index` rewritten as (1,\dots,n_b) in SDSU order.
* All other columns carried through unchanged.
* No tokens are inserted; tables remain JOB-only.

**Purge positions JSON**

* `${output_dir}/step04_flush_positions.json`

Structure:

```json
{
  "booth1": { "flush_positions": [k₁,…], "num_purges": P₁, "n_jobs": n₁ },
  "booth2": { "flush_positions": [k'₁,…], "num_purges": P₂, "n_jobs": n₂ }
}
```

Each `flush_positions` entry `k` indicates “insert a FLUSH between jobs at positions `k` and `k+1` in the SDSU order for that booth” (0-based between-edge index).

**Manifest and report**

* `${output_dir}/manifests/step04_manifest.json`
* `${output_dir}/reports/step04_report.md`

The manifest and report follow the project-wide step-gating protocol (config hashes, settings, metrics, diagnostics).


## 4. Feasibility Semantics (C₁, C₂ᵇ, C₃)

### 4.1 Paint feasibility C₁ᵇ

`load_paint_model()` builds a **paint model** from the SDST matrix and topcoat rules:

* `setup_cost[ci][cj]` is the cost from `s_matrix_transition.csv` for painting colour `cj` immediately after `ci`. `"inf"` or `"INF"` in the CSV is treated as (+\infty).
* `topcoat_rules.csv` with columns `relation`, `from_topcoat_no`, `to_topcoat_no` contributes a set of numeric forbid pairs ((i_{\text{no}}, j_{\text{no}})) whenever `relation` contains `"forbid"`.

Given a job pair (i,j) with colours (c_i, c_j):

1. **Direct arc.**

   * Look up `direct = setup_cost[ci][cj]` if defined, otherwise `direct = +∞`.
   * If the numeric pair `(no(ci), no(cj))` is in the forbid set, force `direct = +∞`.

2. **Purge arc (FLUSH + restart).**

   * If `allow_purge = true`, Step-04 allows a purge-then-paint arc from (i) to (j) whenever a start cost `s₀(c_j)` is defined.
   * The implementation uses `s₀(c) = 0` for all colours and sets

     [
     \text{purge_cost} = \text{flush_units_ratio} \times \text{purge_cost_weight} + s_0(c_j).
     ]

3. **Paint feasibility.**

   * An arc is paint-feasible iff `direct < +∞` **or** `purge_cost < +∞`.
   * The arc cost `c(i,j)` is the minimum of the two, and a boolean `flush_used[i][j]` records whether the purge path is selected.

Thus, topcoat forbids encode transitions that must go through a purge when `allow_purge=true`, or are impossible if `allow_purge=false`.


### 4.2 In-booth adjacency feasibility C₂ᵇ

`load_adjacency_model()` parses `adjacency_rules.csv` with schema hints from `[data_schema.adjacency_rules]`:

* It resolves `from_part_id`, `to_part_id`, `scope`, and `relation` columns.
* For each row where `relation` contains `"forbid"` and `scope ∈ {"in_booth","global"}`, it adds the pair `(from_part_id, to_part_id)` to `AdjacencyModel.forbid_pairs`.
* Rows with other scopes such as `"across_skids"` are **ignored** here; they belong to the merged-line C₂ handled downstream.

When building arcs (i \to j) in the SDSU cost matrix, Step-04 enforces:

* If `(p_i, p_j) ∈ forbid_pairs`, the arc is **dropped** regardless of paint feasibility or purge options.

Purges do **not** waive adjacency: all adjacency checks are performed on the **parts-only** sequence. This implements “in-booth C₂” on the JOB stream for each booth.

### 4.3 Tokens and purge C₃

Step-04 does **not** introduce any `EMPTY` or `FLUSH` rows in its CSV outputs.

* Purges are represented **implicitly** as costed alternatives in the arc matrix (`flush_used[i][j] = True`) and
* **explicitly** in `step04_flush_positions.json` as a list of between-job indices.

Adjacency remains a parts-only hard constraint; FLUSH helps only with paint-domain feasibility and cost, never with adjacency.

---

## 5. Parameters (from `model_config.toml`) — Meaning, Effects, Extremes

| Key | Meaning (real‑world) | ↑ Effect | ↓ Effect | Extremes / Degeneracy |
|---|---|---|---|---|
| `[step04].dp_threshold` | Max booth size for exact subset DP | More exactness, longer runtime | More heuristic usage | Too large → exponential runtime; too small → lost exactness |
| `[step04].max_passes` | Number of 2‑opt improvement passes | Better solutions, added time | Faster, potentially worse | 0 disables improvement |
| `[step04].start_policy ∈ {require_matrix, nominal}` | How to treat missing start→color setup | Stricter realism (may forbid first job) | More permissive with nominal cost | Over‑permissive can mask data errors |
| `[step04].tie_break ∈ {setup_then_lex, lex}` | Deterministic ordering among equal-cost moves | Predictable, cost‑aware order | Pure lex stability | Must be fixed for reproducibility |
| `[tokens].flush_units_ratio` | Relative unit cost of a purge | Discourage purges | Encourage purges | ~0 ⇒ “free purge” degeneracy |
| `[adjacency].scope_escalation` | Treat across-skids as global if missing | Safer feasibility | More permissive | Over‑strict on dense forbids may force many purges or block arcs |

**Embedded correction.** Earlier failures came from (i) treating purges as a way to enable **forbidden** transitions outright; here, **FLUSH only resets to START** and cannot override an **unavailable start→color**. Also, (ii) adjacency was previously ignored during SDSU; now adjacency is a hard arc filter between jobs.

---

## 6. Methodology

SDSU is a **local optimizer** that isolates paint‑domain effort from global serialization. We study:  
- The **marginal improvement** SDSU provides over the Step‑03 split as a function of booth size and color diversity.  
- The **trade‑off** between exact DP and constructive+2‑opt: where does the heuristic match DP’s optimum, and where does it lag?  
- The **sensitivity** to `flush_units_ratio`: when purges are cheap, SDSU may prefer “reset‑and‑go” patterns; when expensive, it clusters colors more tightly—subject to adjacency.

Outputs are compared on f1 (setup units) and changeover counts; we do not alter f2/f3 here but we track their **side‑effects** (e.g., longer color blocks may influence f2 downstream).


## 7. SDSU Instance Construction per Booth

For each booth ( b \in {1,2} ):

1. **Read per-booth CSV.**
   `step04_sdsu_per_booth.py` reads `paths.booth1_input` and `paths.booth2_input` from `[step04]` (default Step-03 outputs).

2. **Resolve schema.**
   `resolve_schema(global_cfg, df)` uses `[data_schema.demands]` (falling back to `topcoat_codebook` and `parts` schemas if needed) to pick:

   * `topcoat_code_col` from `topcoat_code_synonyms` + fallbacks `["topcoat_code","color_code"]`.
   * `part_id_col` from `part_id_synonyms` + fallbacks `["part_id"]`.

3. **Build BoothInstance.**
   For each DataFrame row index (i):

   * `colors[i] = str(df[topcoat_code_col].iloc[i])`
   * `parts[i]  = str(df[part_id_col].iloc[i])`

   The original DataFrame `df` is stored alongside these vectors.

4. **Build PaintModel and AdjacencyModel.**
   These are shared between booths:

   * `PaintModel` from `s_matrix_transition.csv` and `topcoat_rules.csv` (Section 4.1).
   * `AdjacencyModel` from `adjacency_rules.csv` (Section 4.2).

The pair ((\text{PaintModel}, \text{AdjacencyModel})) and the `SDSUConfig` derived from `[sdsu]` fully define the SDSU instance per booth.


## 8. SDSU Solver — Exact DP vs Greedy + 2-opt

Let ( n = |S_b| ) for a given booth and let `dp_threshold = dp_threshold_jobs` (from `[step04]` or `[sdsu]`).

### 8.1 Trivial cases

* If ( n = 0 ), Step-04 marks the booth as empty (`branch="empty"`) and emits an empty CSV (no rows).
* If ( n = 1 ), the unique job remains as is; cost is 0; there are no purges.

### 8.2 Exact open-path DP (Held–Karp style)

If ( n \le ) `dp_threshold`, `solve_sdsu_for_booth()` calls

```python
sdsu_open_path_dp(use_cost)
```

which implements a subset DP (cf. Held–Karp) for ATSP-Path:

* State: `DP[S][j]` = minimal cost of a path that visits exactly the job subset (S) and ends at job (j).
* Base: singleton subsets with cost 0.
* Transition: extend a path ending at (i) with a feasible arc (i \to j) where `use_cost[i][j] < +∞`.
* Reconstruction: recover the minimum-cost Hamiltonian **open** path using parent pointers.

Complexity is (O(n^2 2^n)) time and (O(n 2^n)) memory, so this branch is used only for small booths where an exact solution is tractable.

### 8.3 Greedy + 2-opt heuristic

If ( n > ) `dp_threshold`, the script calls

```python
sdsu_open_path_greedy_2opt(use_cost, max_passes=max_2opt_passes)
```

Heuristic outline:

1. **Greedy construction.**

   * Start from node 0.
   * Repeatedly append the cheapest feasible successor ( j ) among unused nodes with finite `use_cost[i][j]`.
   * If no such successor exists, remaining unused nodes are appended in index order.

2. **2-opt improvement.**

   * Perform up to `max_2opt_passes` passes of classical 2-opt: for candidate pairs of edges, reverse the intermediate segment and accept a move only if it strictly reduces total path cost.
   * Track the best path cost and terminate early when no further improvements are found.

The solver returns an `SDSUSolution`:

* `order` — a permutation of indices (0,\dots,n-1).
* `cost` — total path cost under `use_cost`.
* `branch` — `"dp"` or `"heuristic"`.

From this solution, `solve_sdsu_for_booth()` computes:

* `flush_positions` — for each adjacent pair `(order[k], order[k+1])`, if `flush_used[i][j]` is `True`, record index `k`.
* `num_purges` — length of `flush_positions`.

The final per-booth summary is packed in a `BoothResult` dataclass (booth id, job count, branch, total setup cost, purge count, order indices, flush positions).


## 7. Algorithms

### 7.1 Exact Open‑Path Subset DP (when \( n \le \) `dp_threshold`)

**State.** \( DP[mask][j] \) = minimal cost of a path that **ends** at job \( j \) having visited the set `mask` (bitmask over jobs); `mask` includes `j`.  
**Init.**
\[
DP[1<<j][j] = s_0(c_j),
\]
if start→\(c_j\) is feasible per `start_policy`.  
**Transition.** For `mask` containing `j` and any `i ≠ j` with `mask` containing `i`:
- If adjacency forbids \((p_i \to p_j)\), **skip**.  
- Let `cost_direct = s(c_i,c_j)` if finite; `cost_purge = FLUSH + s_0(c_j)` if both terms exist.  
- `arc_cost = min(cost_direct, cost_purge)`; if none exist, **skip**.  
- Update:
\[
DP[mask][j] = \min\{\, DP[mask\setminus\{j\}][i] + arc\_cost \,\}.
\]
**Answer.** `min_j DP[(1<<n)-1][j]` with backpointers for reconstruction.

**Complexity.** \( O(n^2 2^n) \) transitions; feasible for `dp_threshold ≲ 22` with careful bit DP.

### 7.2 Deterministic Constructive + 2‑opt (when \( n > \) `dp_threshold`)

**Constructive seed.**
1) Start from the first job in Step‑03 order, or the job with minimal \( s_0(c) \) (per `start_policy`).  
2) Repeatedly append the **admissible** job \(j\) (adjacency + paint feasible) with minimal `arc_cost = min(s(c_i,c_j), FLUSH + s_0(c_j))`; tie by `tie_break` (setup_then_lex).  
3) If no admissible successor exists, abort with a *certificate* (this indicates data forbids make the current residual set impossible; extremely rare when Step‑03 was feasible).

**2‑opt improvement (max_passes).**  
- For all \( a<b \), consider swapping the subpath \([a..b]\) if it **reduces** total cost **and** keeps all adjacency and paint arcs admissible (including recomputed purge choices on boundaries).  
- Accept the first improving swap per pass; multiple passes up to `max_passes` or until no improvement.

**Determinism.** Candidate ordering and tie policy fixed; no randomness.


## 8. Outputs, Metrics, and Step Gating

### 8.1 Per-booth CSVs

For each booth ( b ):

* The original DataFrame `booth.df` is re-ordered using `BoothResult.order_indices`.
* Column `booth_seq_index` is overwritten with (1,\dots,n_b) in SDSU order.
* All other columns, including `job_id`, `part_id`, `topcoat_code`, and feeder indices, are preserved as-is.
* No token rows are added or removed; outputs remain JOB-only.

These CSVs are the **only** inputs from Step-04 consumed by the merge in Step-05.

### 8.2 Flush positions JSON

`step04_flush_positions.json` aggregates per-booth purge information:

* `flush_positions` — 0-based indices of edges where a purge is used between consecutive jobs.
* `num_purges` — the total count of purge events in that booth.
* `n_jobs` — the booth’s job count.

This file is used by analysis tooling (e.g., Step-09) to decompose ( f_1 ) into setup vs purge components and to inspect where purges concentrate. It is not required for the main pipeline to proceed.

### 8.3 Manifest and report

`write_manifest()` produces `manifests/step04_manifest.json` containing:

* `step = "step04_sdsu_per_booth"`, `version`.
* `config_files` — SHA-256 hashes of `global_config.toml` and `steps_config.toml`.
* `input_files` — hashes for Step-03 per-booth inputs and all paint/adjacency datasets.
* `output_files` — hashes for per-booth SDSU CSVs, flush JSON, manifest, and report.
* `settings_effective` — the realised `SDSUConfig` (dp threshold, 2-opt passes, purge flags, flush cost parameters).
* `metrics` —

  * per-booth `BoothResult` summaries (including order indices and flush positions),
  * `total_setup_cost` (sum over booths),
  * `total_purges`.

`write_report()` emits `reports/step04_report.md` with:

* A **Purpose / What / Why** section aligned with this note.
* A settings snapshot (dp threshold, max 2-opt passes, purge enable, flush cost).
* For each booth: job count, branch label, total setup cost, number of purge events, and flush positions reported explicitly.

### 8.4 Acceptance / Gate Criteria

Step-04 is considered valid for the experiment when:

1. `step04_booth1_sdsu.csv`, `step04_booth2_sdsu.csv`, `step04_flush_positions.json`, the manifest, and the report all exist under `output_dir`.
2. The manifest shows finite `total_setup_cost` and a reasonable job count per booth (matching Step-03).
3. No adjacency violations are **possible** by construction, since arcs violating in-booth/global adjacency were removed when building `use_cost`.
4. For booths where `branch="dp"`, the solution is exact under the current cost matrix; for `branch="heuristic"`, the solution is the best found by the greedy + 2-opt heuristic and serves as a baseline for MILP/NSGA-II comparison in later analysis (Step-09).



## 9. Reproducibility (Manifest & Report)

**Manifest JSON:** inputs/outputs (rel paths + SHA‑256), settings snapshot (`dp_threshold`, `max_passes`, `start_policy`, `tie_break`, `flush_units_ratio`, adjacency scope), metrics (f1, changes), diagnostics, runtime.  
**Report MD:** Purpose → What → Why (w/ SDSU intuition) → Config → Inputs/Outputs (rel+SHA) → Method (DP/Heuristic) → Metrics → Diagnostics → Runtime → Determinism note.

---

## 10. Execution Instructions

```bat
python step04_sdsu.py
```
The script will:

1. Load `global_config.toml` and `steps_config.toml`.
2. Resolve dataset and output paths via `[paths]` and `[data_files]`.
3. Read the Step-03 per-booth JOB CSVs.
4. Build paint and adjacency models.
5. Run SDSU per booth (DP or heuristic, based on `dp_threshold_jobs`).
6. Write the per-booth SDSU CSVs, flush positions JSON, manifest, and report.


## 16. Verification Checklist (Before Proceeding)


Before running the merge step, verify:

* [ ] `step04_booth1_sdsu.csv` and `step04_booth2_sdsu.csv` exist and have plausible row counts (matching Step-03).
* [ ] `step04_flush_positions.json`, `manifests/step04_manifest.json`, and `reports/step04_report.md` exist.
* [ ] In each SDSU CSV, `booth_seq_index` runs from 1 to `n_jobs` without gaps or duplicates.
* [ ] Manifest `metrics.total_setup_cost` is finite and equals the sum of per-booth `total_setup_cost`.
* [ ] No token rows (`EMPTY`, `FLUSH`) appear in per-booth CSVs; tables remain JOB-only.
