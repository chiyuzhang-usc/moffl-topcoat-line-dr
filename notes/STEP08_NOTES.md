
# Step‑08 — Discrete‑Event Simulation (DES) for 2→1 Line: Blocking & Idle Metrics

> Downstream of the **global oven convoys** produced by SDSU-merge (Step-05), MILP (Step-06), and NSGA-II (Step-07). This step **never changes** any sequence. It treats JOB/TOKEN streams as fixed, reconstructs per-booth paint timelines using the SDSU kernel, and evaluates **blocking at booths** and **idle at the oven** under an **EMPTY-as-adjacency-reset** policy.

---

## 1. Purpose

### Formal
Simulate a tandem system with two upstream servers (Paint B1/B2) and a single downstream server (Oven), connected via a finite‑capacity merge buffer (size β, can be configured to be unlimited or 0). Service times are deterministic or data‑driven. The oven must start jobs **exactly** in the order of `step05_global_line.csv`. Compute:
- **f4 (blocking/idle)**: oven idle time, booth blocking time, and normalized indicators under \( (β, θ) \).  
- Throughput, makespan, average WIP, and buffer occupancy histograms.

Given:
- a catalog of jobs with booth assignment and topcoat,
- a global convoy \( G \) consisting of JOB and TOKEN (EMPTY/FLUSH) symbols, and
- an SDST paint kernel (from \(S\)-matrix + topcoat rules),

The step builds a deterministic schedule and computes:

### Intuitive
Given the booth schedules and the enforced oven order, DES shows **where the flow stalls**: booths may finish jobs that cannot move forward because the buffer is full (**blocking**), while the oven may wait for the required next job (**idle**) if it hasn’t completed upstream (**starvation of the required sequence**).

### Research Role
This isolates **operational feasibility** from **sequencing feasibility**: we keep the sequences fixed and study how buffer and speed ratios affect performance, thus closing the loop between scheduling (Steps 02–05/06–07) and operations.

- Attach **operational metrics** (blocking/idle/makespan) to each upstream method:
  - Toyota+SDSU+merge baseline (Step-05),
  - MILP global line (Step-06),
  - NSGA-II global lines, seeded and unseeded (Step-07).
- Provide a **common DES lens** over heterogeneous schedules so Step-09 can compare fronts on \((f_1, f_2, f_3, f_4)\).

---

## 2. Connections to Neighbor Steps

- **From Previous steps → Step‑08.** Inputs are per‑booth orders and the merged convoy. DES reads booth service times from data or config and **respects** the Step‑05 serialization.  
- **To reporting.** Results augment earlier metrics (f1–f3) with **f4** to compare schedules operationally.

- **Step-03(Optional):**  
  `step03_assignment.csv`, `step03_booth1.csv`, `step03_booth2.csv`  
  Used to build a **job catalog**:
  \[
  \text{JobRecord} = (\text{job\_id}, \text{feeder\_index}, \text{part\_id}, \text{topcoat}, \text{base\_booth}\in\{1,2\})
  \]

- **Step-05 (SDSU→global merge)**
  `step05_global_line.csv` (JOB + optional EMPTY tokens, C2-clean at parts-only level).

- **Step-06 (MILP)** 
  `<milp_output_prefix>global.csv` (e.g. `step06_milp_opt_global.csv`), a global JOB/TOKEN convoy.

- **Step-07 (NSGA-II)**
  - Consolidated global line: `step07_global_line.csv` (possibly multi-solution CSV with `solution_id`,`seed`).  
  - Additional per-solution exports in `step07_exports/*.csv` (both base and `_seeded` variants).

- **Dataset (C1/C2/C3 & Setup):**
  - `adjacency_rules.csv` — global C2 forbids (parts-only)  
  - `s_matrix_transition.csv` — Setup transition units between colors  
  - `topcoat_codebook.csv`, `topcoat_rules.csv` — map codes to names, C1/C3 forbids



## 3. Files, I/O, and Metrics Table

### 3.1 Inputs (resolved via `global_config.toml` + `steps_config.toml`)

- Dataset:
  - `data_files.adjacency_rules         → adjacency_rules.csv`
  - `data_files.s_matrix_transition     → s_matrix_transition.csv`
  - `data_files.topcoat_codebook        → topcoat_codebook.csv`
  - `data_files.topcoat_rules           → topcoat_rules.csv`
- Step-03:
  - `[step03].assignment_relpath        → step03_assignment.csv`
  - `[step03].booth1_relpath            → step03_booth1.csv`
  - `[step03].booth2_relpath            → step03_booth2.csv`
- Step-05 convoy (if enabled and exists):
  - `[step05].global_line_relpath       → step05_global_line.csv`
- Step-06 convoy (if enabled and exists):
  - `[step06].milp_output_prefix + "global.csv"  → e.g. step06_milp_opt_global.csv`
- Step-07 convoys (if enabled and exist):
  - `[step07].global_line_relpath       → step07_global_line.csv` (consolidated, possibly multi-solution)
  - `[step07].exports_dir_relpath       → directory of additional global lines (base and `_seeded`)`

### 3.2 Outputs

- `[step08].metrics_relpath             → step08_des_metrics.csv`
- `[step08].manifest_relpath            → manifests/step08_manifest.json`
- `[step08].report_relpath              → reports/step08_report.md`

**Metrics CSV schema (per row = one scenario):**

- Identification:  
  `method` ∈ {`step05`,`step06_milp`,`step07_nsga2`,`step07_nsga2_seeded`}  
  `scenario` (file/sid tag), `solution_id` (NSGA-II id or `None`), `seed` (for NSGA-II, or `None`)

- Core performance:  
  `throughput_jobs`, `tokens_empty`, `makespan`

- Booth times:  
  `pb1_busy_time`, `pb1_blocked_time`, `pb1_idle_time`  
  `pb2_busy_time`, `pb2_blocked_time`, `pb2_idle_time`

- Oven times:  
  `oven_busy_time`, `oven_idle_time`

- Ratios and optional scalarization:  
  `pb_block_frac_max`, `oven_idle_frac`, `f4`

### Parameters
| Key | Meaning | ↑ Effect | ↓ Effect | Extremes / Degeneracy |
|---|---|---|---|---|
| `[des].buffer_beta` | Buffer capacity β (jobs) | Less blocking, more WIP | More blocking, less WIP | β=0 ⇒ pure blocking |
| `[des].tau_paint` | Default booth cycle time | Slower booths ⇒ more oven idle | Faster booths ⇒ possible buffer buildup | Nonpositive invalid |
| `[des].tau_oven` | Default oven cycle time | Slower oven ⇒ more blocking | Faster oven ⇒ possible oven starvation | Nonpositive invalid |
| `[des].infer_timing_from` | `"line_cadence"` or `"constant"` | Data realism | Simplicity | Missing data ⇒ fallback |
| `[des].trace_enabled` | Emit event trace CSV | Transparency | Smaller files | Off hides details |
| `[des].replications` | # of runs (for stochastic modes; default 1 here) | Statistical CI | None | n/a (deterministic) |

*warup_time* is in the parameter due to an attempted stochastic method, designed to set warmup horizon.
this method is **not** implemented as of now.


## 4. Token & Adjacency Semantics in DES

### 4.1 Tokens on the global convoy

The convoy is represented as a list of **ConvoyItem**:
- JOB elements: `("PB1", j_idx)` or `("PB2", j_idx)` referencing the job catalog.
- TOKEN elements: `"TOKEN"` when the job_id column contains one of  
  `{EMPTY, TOKEN_EMPTY, EMPTY_TOKEN, TOKEN, FLUSH, NONE}` or is empty/NaN.

**Interpretation in Step-08:**

- **JOB:** real part; processed at paint booth and oven.
- **TOKEN:** consumes oven time as designed idle capacity; no paint work, no part, no C2 contribution.

### 4.2 Global C2 with EMPTY-as-reset

- For **diagnostics only**, Step-08 reconstructs a parts-only stream from the convoy **with EMPTY as a hard reset**:
  - when seeing a TOKEN, adjacency state is cleared;
  - for JOB→JOB transitions, C2 uses `adjacency_rules.csv` (global scope, parts-only).
- `first_c2_violation_with_tokens` returns the first forbidden pair \((p_i,p_{i+1})\) that would violate C2 *if* EMPTY were not present.  
- Behaviour:
  - If a violation is detected, Step-08 logs:
    - the convoy position,  
    - the `(prev_part, curr_part)` pair,  
    - the corresponding job_ids.  
  - **No run is aborted.** Violations are treated as **NOTES**, because upstream steps are the owners of C2; Step-08 only reports whether the given convoy would be C2-clean under the EMPTY-as-reset semantics used for DES.

Adjacency semantics remain **data-driven** via `adjacency_rules.csv`; Step-08 does not introduce any additional rules.


## 5. Event Model and Timing

Step-08 uses **two layers**:

1. **Booth timing** using the SDSU paint kernel.  
2. **Oven timing** over the convoy with tokens.

### 5.1 Booth timing from SDST kernel

For each booth \(b \in \{1,2\}\):

1. Extract the booth job sequence from the convoy:
   \[
   S_b = [j_{b,1}, j_{b,2}, \dots]
   \]
   in convoy order, skipping tokens.

2. For job \(j_{b,k}\) with color \(c_{b,k}\):
   - SDST units:
     \[
     u_{b,k} = \text{transition\_units}(c_{b,k-1}, c_{b,k})
     \]
     from `s_matrix_transition.csv` plus topcoat rules (including purge/FLUSH where allowed).
   - Duration:
     \[
     d_{b,k} = \text{base\_paint\_time} + u_{b,k}\cdot \text{time\_per\_setup\_unit}
     \]
3. Paint timeline:
   \[
   t_{b,0}=0,\quad t_{b,k}=t_{b,k-1}+d_{b,k}
   \]
   - `finish_time_by_job[j_idx] = t_{b,k}`
   - `busy_time_b = \sum_k d_{b,k}`

This guarantees **consistency** with SDSU/NSGA-II cost units and purge penalties.

### 5.2 Oven and blocking dynamics (no buffer, EMPTY consumes idle)

Let:
- \( \theta = \text{oven\_time\_per\_job} = [des].theta \),
- \( \rho_E = \text{empty\_time\_ratio} = [tokens].empty\_time\_ratio \),
- convoy \( C = (c_1,\dots,c_L) \).

**Initialization**

- Oven time: \( t_O = 0 \).  
- Booth blocked times: \(B_1 = 0\), \(B_2 = 0\).  
- Oven busy and EMPTY-idle accumulators: \(T_{\text{busy}} = 0\), \(T_{\text{idle}} = 0\).  
- Job counter: `jobs = 0`.

**Loop over convoy items**

For each \(c_\ell \in C\):

1. **TOKEN case.**  
   - Duration:
     \[
     d_E = \theta \cdot \rho_E
     \]
   - Start at current oven time: \(s_E = t_O\); finish \(f_E = s_E + d_E\).  
   - Update:
     \[
     t_O \leftarrow f_E,\quad T_{\text{idle}} \leftarrow T_{\text{idle}} + d_E
     \]
   - Booth metrics unchanged; no job count increment.

2. **JOB case.** Suppose \(c_\ell = (\text{PB}b, j)\).  
   - Ready time from booth schedule:
     \[
     r_j =
       \begin{cases}
       \text{finish\_time\_by\_job}^{(1)}[j], & b=1\\
       \text{finish\_time\_by\_job}^{(2)}[j], & b=2
       \end{cases}
     \]
   - Oven start:
     \[
     s_j = \max(r_j,\ t_O)
     \]
   - Booth \(b\) **blocking time increment**:
     \[
     \Delta B_b = \max(0,\ s_j - r_j)
     \]
   - Oven finish:
     \[
     f_j = s_j + \theta
     \]
   - Update:
     \[
     t_O \leftarrow f_j,\quad B_b \leftarrow B_b + \Delta B_b,\quad 
     T_{\text{busy}} \leftarrow T_{\text{busy}} + \theta,\quad \text{jobs} \leftarrow \text{jobs}+1
     \]

**Makespan**

- \[
  C_{\max} = t_O
  \]

**Booth idle times**

For \(b \in \{1,2\}\):

\[
\text{idle}_b = \max\bigl(0,\ C_{\max} - \text{busy\_time}_b - B_b \bigr)
\]

---

## 6. Metrics and \(f_4\)

Given the DES outputs:

- \(C_{\max}\) — `makespan`,
- \(B_1,B_2\) — booth blocked times,
- \(\text{busy\_time}_1,\text{busy\_time}_2\),
- \(T_{\text{busy}},T_{\text{idle}}\) — oven busy/idle due to EMPTY tokens,

Step-08 computes:

1. **Throughput and tokens**
   - `throughput_jobs = jobs`  
   - `tokens_empty =` number of convoy elements equal to `"TOKEN"`

2. **Booth time components**
   - `pb1_busy_time = busy_time_1`  
   - `pb1_blocked_time = B_1`  
   - `pb1_idle_time = idle_1`  
   - `pb2_busy_time = busy_time_2`  
   - `pb2_blocked_time = B_2`  
   - `pb2_idle_time = idle_2`

3. **Oven time components**
   - `oven_busy_time  = T_busy`  
   - `oven_idle_time = T_idle`

4. **Normalized indicators**
   - Max booth blocking fraction:
     \[
     \text{pb\_block\_frac\_max}
       = \max\left(
         \frac{B_1}{C_{\max}},\,
         \frac{B_2}{C_{\max}}
       \right)
     \quad \text{(0 if }C_{\max}=0\text{)}
     \]
   - Oven idle fraction:
     \[
     \text{oven\_idle\_frac} = \frac{T_{\text{idle}}}{C_{\max}}
     \quad \text{(0 if }C_{\max}=0\text{)}
     \]

5. **Optional scalar \(f_4\)** (if `[step08].f4_weights` is a dict)

Let \(w = \{\alpha, \beta, \gamma\}\) be:

- \(\alpha = w["pb_block_frac_max"]\),
- \(\beta = w["oven_idle_frac"]\),
- \(\gamma = w["makespan"]\).

Then:

\[
f_4 = \alpha\cdot \text{pb\_block\_frac\_max}
    + \beta\cdot \text{oven\_idle\_frac}
    + \gamma\cdot C_{\max}
\]

If `f4_weights` is absent or malformed, `f4` is left `NaN` and the components are consumed directly in Step-09.

## 7. Configuration: Effective Settings and Knobs

Step-08 builds a `Step08Settings` instance:

- **Paint & oven timing** (from `[des]` in `global_config.toml`):
  - `base_paint_time`  
  - `time_per_setup_unit`  
  - `oven_time_per_job = theta`

- **EMPTY behaviour** (from `[tokens]`):
  - `empty_time_ratio`  
    - EMPTY duration = `oven_time_per_job * empty_time_ratio`.  
    - `>1` ⇒ EMPTY more expensive than a real job; `<1` ⇒ lighter idle slot.

- **Paint purge / FLUSH** (from `[sdsu]` + `[tokens]`):
  - `allow_purge` (bool)  
  - `flush_units_ratio`  
  - `purge_cost_weight`  
  - `flush_cost_units = flush_units_ratio * purge_cost_weight`  
  Used by the SDSU kernel to translate FLUSH into extra setup units (and hence time).

- **Scenario toggles** (from `[step08]` in `steps_config.toml`):
  - `run_step05` — whether to evaluate `step05_global_line.csv`  
  - `run_step06` — whether to evaluate the MILP global convoy  
  - `run_step07` — whether to scan Step-07 global lines (base and seeded)

- **Optional scalarization** (from `[step08].f4_weights`):
  - When present and a dict, enables \(f_4\) computation as above.

**Notably:**

- Legacy DES parameters such as buffer size β, stochastic replications, `line_cadence.csv` lookups, etc., are **not** used by this implementation. All timing is driven by the SDST kernel plus the simple oven/EMPTY rules above.
- The only “ratio” lever in Step-08 is `theta` vs. paint times (via SDST units) and `empty_time_ratio`.

## 8. Acceptance Criteria (Hard Gates)

1. The oven **starts** jobs only in the order of `step05_global_line.csv`.  
2. No sequence changes or token insertions.  
3. Manifest and report include β, θ, timing source, and metrics; paths are **relative**.

---

## 9. Reproducibility (Manifest & Report)

**Manifest JSON:** inputs/outputs (rel paths + SHA‑256), timing source and resolved parameters (β, τ values, θ), metrics, diagnostics, runtime.  
**Report MD:** Purpose → What/Why → Timing source → Settings → Metrics → Diagnostics (first idle/block episodes) → Runtime → Determinism note.

---

## 10. Execution Instructions (How to Repeat)

> **Do not execute now.** When authorized, from repository root:

```bat
python step08_des.py
```

Reads `model_config.toml`; writes metrics CSV, optional trace, manifest, and report to `./output` with **relative** paths.

---

## 11. Failure Modes & Minimal Recovery

- **Perpetual oven idle:** oven faster than booths (θ≪1) and β too small; increase β or rebalance service times.  
- **Severe booth blocking:** oven slower (θ≫1) with small β; either increase β or slow booths (if realistic).  
- **Timing inference gaps:** if `line_cadence.csv` lacks entries for some parts/colors, fall back to defaults and log the fallback count.

---

## 12. Interface Closure

DES closes the pipeline by reporting **operational** consequences of the schedules. It does not alter feasibility or order; any adjacency or paint issues must be handled in Steps 02–05.

---

*End of Step‑08 Note.*

---

## 4.1 Timing Source: `line_cadence.csv` Schema & Precedence (Exact)

**Supported columns (CSV; case-insensitive headers normalized):**
- `station` ∈ {`paint`, `oven`} — required.
- `key_type` ∈ {`part`, `topcoat`, `global`} — required.
- `key` — the value for `part_id` or `topcoat_code`; ignored for `global`.
- `cycle_time` (float) — time per job in **arbitrary but consistent** time units (see `[des].time_unit`).
- Optional: `effective_from`, `effective_to` (ISO‑8601), `priority` (int, lower is stronger).

**Lookup precedence per station (deterministic):**
1) Exact **part** match: `(station, key_type=part, key=part_id)` with the **lowest** `priority` among active (by date) rows.  
2) Exact **topcoat** match: `(station, key_type=topcoat, key=topcoat_code)` with lowest priority among active rows.  
3) **Global** default: `(station, key_type=global)` with lowest priority among active rows.  
4) If no active row is found at any level, **fallback** to config default: `[des].tau_paint` or `[des].tau_oven` respectively.

**Data validation gates:**
- `cycle_time > 0` required; nonpositive ⇒ reject row and log.  
- Overlapping effective windows at the same `(station, key_type, key)` are allowed; `priority` breaks ties. If `priority` ties, pick **deterministically** by earliest `effective_from`, then lexicographic row order.

**Units.** DES treats times as **dimensionless**; report them with the label `[des].time_unit` (e.g., `sec`, `min`, `ticks`). No conversion is performed inside DES.

---

## 4.2 θ (Service Ratio) — Definitions & Resolution

We record two θ values in the manifest:

1) **θ_static** (from config defaults):  
\[
θ_{\text{static}} = \frac{[des].\tau_{\text{oven}}}{[des].\tau_{\text{paint}}}.
\]

2) **θ_realized** (from resolved lookups over the actual instance):  
Let \(G\) be the global convoy and \(S = S_1 \cup S_2\) all booth jobs. With resolved times \( \tau_O(j) \) and \( \tau_P(j) \), define  
\[
θ_{\text{realized}} = \frac{\sum_{j\in G} \tau_O(j)}{\sum_{j\in S} \tau_P(j)}.
\]
This ratio reflects the **effective** speed relation under the dataset. Both values are reported; DES logic uses the **resolved** per‑job times and does not depend on θ directly.

---

## 6.1 Event Priorities & Tie Handling (Deterministic)

At any simulation time \(t\), events are advanced in this fixed priority to avoid ambiguity:

1) **Booth completions** that finish exactly at \(t\): push to buffer (or block if full).  
2) **Buffer unblocks** triggered by new space at \(t\): release the **earliest finished** blocked booth completion.  
3) **Oven start** at \(t\) if the **next due job** \(g_k\) is present in buffer.  
4) **Time advance** to the minimum next event (next booth completion, oven finish, or buffer space change).

**Deterministic selections:**
- If both booth completions occur at \(t\), process **B1 then B2**.  
- If multiple jobs are present in buffer and the next due \(g_k\) is present, choose \(g_k\) regardless of arrival position (order constraint dominates). The remaining jobs retain their relative arrival order (FIFO) for subsequent selections.

---

## 7.1 Expanded DES Metrics (Definitions)

**Per-resource utilizations**
- **Oven utilization:** \( U_O = \frac{\sum_{j\in G}\tau_O(j)}{C_{\max}} \).  
- **Booth utilization:** \( U_{P,b} = \frac{\sum_{j\in S_b}\tau_P(j)}{C_{\max}} \), \( b\in\{1,2\} \).

**Time components**
- **Oven idle time** (starvation): total time when oven is free but \(g_k\) not present.  
- **Booth blocking time \(B_b\)**: total time booth \(b\) is done with a job but cannot push to buffer due to capacity \(β\).

**Job-level times**
- **Paint start/finish:** \( (s^P_j, f^P_j) \).  
- **Buffer wait:** \( w_j = s^O_j - f^P_j \) (nonnegative).  
- **Oven start/finish:** \( (s^O_j, f^O_j) \).  
- **Flow time:** \( F_j = f^O_j - s^P_j \).

**Aggregates**
- **Makespan:** \( C_{\max} = \max_j f^O_j \).  
- **Avg. WIP:** time-average \( \overline{Q} = \frac{1}{C_{\max}}\int_0^{C_{\max}} |Q(t)|\,dt \).  
- **Buffer occupancy distribution:** histogram of \(|Q(t)|\) over event intervals.  
- **Percentiles:** report \(P_{50}, P_{90}, P_{95}\) for \(\{w_j\}\) and \(\{F_j\}\).

**Composite f4** (normalized):  
\[
f4 = \alpha \cdot \frac{\text{oven idle time}}{C_{\max}} + (1-\alpha)\cdot \frac{B_1+B_2}{C_{\max}},
\]
with \(\alpha \in [0,1]\) from `[des].f4_alpha`. Report the components separately.



## 8. Manifest, Report, and Acceptance

### 8.1 Manifest (`manifests/step08_manifest.json`)

Written by `write_manifest_and_report` and contains:

- `"step": "step08_des"`
- `"timestamp"` (ISO-8601)
- `"config_files"`: relative paths of `global_config.toml`, `steps_config.toml` with SHA-256
- `"inputs"`: relative paths and SHA-256 hashes for all dataset and convoy files actually used
- `"outputs"`: hashes for `step08_des_metrics.csv`, manifest, report
- `"settings_effective"`: JSON serialization of `Step08Settings`
- `"summary"`:  
  - `"scenarios"` = number of rows in metrics CSV  
  - `"runtime_seconds"` = wall-clock runtime

This satisfies the **step-gating** requirement: every run captures config + I/O fingerprints and the effective settings.

### 8.2 Report (`reports/step08_report.md`)

* This report is very important as it concludes all executable steps

Structure:

- Title line: `# STEP-08 — DES (EMPTY resets adjacency)`  
- Scenario count and runtime  
- `## Settings` block: JSON of `Step08Settings`  
- `## Metrics (head)`: CSV preview of the first up-to-12 rows of `step08_des_metrics.csv`

The detailed analysis, comparisons across methods, and plotting are intentionally deferred to **Step-09** and `step08_des_plotter.py`.

### 8.3 Acceptance

A Step-08 run is considered **valid** for the MO-FFS study when:

- `step08_des_metrics.csv` exists and is non-empty.  
- Each row has a recognized `method` and consistent `throughput_jobs` with the corresponding global convoy length (JOB count).  
- Any reported C2 notes (via EMPTY-as-reset diagnostics) are treated as **analytical flags**, not as failures; C2 enforcement remains in Steps-05/06/07.

## 9.1 Manifest Additions

* This is not implemented

- `theta_static`, `theta_realized`  
- `timing_mode`, `lookup_precedence`, `time_unit`  
- Percentiles configured and computed; buffer histogram bins


## Appendix - Worked Lookup Example

Given rows:

```
station,key_type,key,cycle_time,priority
paint,part,P123,37,1
paint,topcoat,RED,40,2
paint,global,,45,9
oven,topcoat,RED,55,2
oven,global,,60,9
```

and job `P123/RED`:
- Paint time = 37 (part match wins).  
- Oven time = 55 (topcoat match; no part row for oven).  
Thus \( θ_{\text{realized}} \) for this single job equals \( 55/37 \).

---

## Appendix - Event Trace Columns (if `[des].trace_enabled=true`)

`time,event,resource,job_id,part_id,topcoat_code,action,queue_size,blocked_flag,details`  
- `event` ∈ {`paint_start`,`paint_finish`,`buffer_push`,`buffer_block_release`,`oven_start`,`oven_finish`}  
- `resource` ∈ {`B1`,`B2`,`OVEN`}

