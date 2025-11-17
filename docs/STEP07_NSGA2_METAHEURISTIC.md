# STEP-07 — NSGA-II Metaheuristic (Feasibility-Aware Decoder)

**Script (target):** `step07_nsga2.py`  
**Variants (via config, not separate scripts):**

- **Unseeded NSGA-II** — random initial population over the search space.  
- **Seeded NSGA-II** — inject individuals derived from Toyota GS / SDSU / MILP into the initial population.

**High-level role.**  
STEP-07 builds **Pareto-approximate schedules** for the multi-objective paintshop problem:

\[
\min (f_1, f_2, f_3) \quad [\text{and optionally } f_4]
\]

over:

- **Booth assignment** (jobs → PB1/PB2),  
- **Per-booth sequencing**, and  
- **Merged global convoy** (2→1 oven),  

subject to the **same C₁/C₂/C₃ semantics** as Steps 02–06 (no new constraints, no relaxed rules).

The step outputs **JOB-only schedules** consistent with the master protocol and downstream DES.

---

## 1. Role in the MOFFS Pipeline

**Upstream:**

- STEP-01 — Job universe.  
- STEP-02 — Toyota GS feeder (`step02_paint_seed.csv`), C₁/C₂-feasible single stream.  
- STEP-03 — Greedy split to booths (`step03_booth1.csv`, `step03_booth2.csv`).  
- STEP-04 — Per-booth SDSU sequences (`step04_booth1_sdsu.csv`, `step04_booth2_sdsu.csv`).  
- STEP-05 — Canonical merged convoy (`step05_global_line.csv`).  
- STEP-06 — MILP baselines (per-booth & global).

**This step:**

- Defines a **search space** that *contains* the earlier heuristics/SDSU/MILP solutions as special points.  
- Uses NSGA-II to **explore alternative splits and sequences**, possibly improving on f₁–f₃ while preserving feasibility.

**Downstream:**

- STEP-08 — DES uses NSGA-II schedules (or a subset of them) as input to compute f₄ (blocking/idle metrics).  
- STEP-09 — Analysis step uses NSGA-II fronts alongside GS, SDSU, and MILP for comparative plots and statistics.

---

## 2. Inputs and Outputs

### 2.1 Inputs

**Config**

- `global_config.toml`
- `steps_config.toml` (sections: `[step02]`, `[step03]`, `[step04]`, `[step05]`, `[step06]`, `[step07]`, `[nsga2]`)

**Dataset (from `[paths].dataset_dir` + `[data_files]`)**

- `demands.csv` — base jobs (indirectly via STEP-01).  
- `topcoat_rules.csv` — paint forbids and purge metadata.  
- `s_matrix_transition.csv` (and/or `s_lookup_transition.csv`) — SDST times.  
- `adjacency_rules.csv` — adjacency forbids.  
- `cycle_quota.csv` — mix quotas for f₂.  
- `line_cadence.csv` — optional pitch for time-normalized reporting.  
- `fixtures.csv` — compatibility data (used only if decoder is configured to enforce fixture constraints).

**Upstream artefacts**

- `output/step02_paint_seed.csv` — GS feeder (token-aware).  
- `output/step03_booth1.csv`, `output/step03_booth2.csv` — greedy split.  
- `output/step04_booth1_sdsu.csv`, `output/step04_booth2_sdsu.csv` — SDSU per-booth sequences.  
- `output/step05_global_line.csv` — canonical oven convoy.  
- Optionally, selected MILP schedules from STEP-06 (for seeding).

Column names are resolved via `data_schema.*` from `global_config.toml` (no hard-coded names).

### 2.2 Outputs

**Primary population archive**

- `output/step07_nsga2_population.csv`  
  Each row = one candidate schedule (one genome), with columns:

  - `solution_id` — unique ID
  - `seed_source` — `"random" | "GS" | "SDSU" | "MILP" | "mixed"`  
  - `sigma_encoding` — serialized split pattern (see §3)  
  - `pi1_encoding`, `pi2_encoding` — serialized per-booth permutations  
  - `f1`, `f2`, `f3` (and `f4` if computed via DES hook)  
  - Additional diagnostics (repair counts, rejection reason counters, etc.).

**Pareto front export**

- `output/step07_nsga2_pareto.csv`  
  Subset of `population.csv`, containing only non-dominated individuals w.r.t. (f₁, f₂, f₃[, f₄]).

**Optional schedule realisations**

- `output/step07_nsga2_solutions/sol_{solution_id}.csv`  
  JOB-only per-solution realisation with the same schema as `step05_global_line.csv`:

  - `job_id, booth_id, booth_seq_index, global_seq_index, part_id, topcoat_code, ...`  
  - plus any diagnostic columns (e.g. `arc_cost_units`, `cum_cost_units`).

**Manifest & report**

- `output/manifests/step07_manifest.json`  
- `output/reports/step07_report.md`

Step-07 respects the same manifest/report contract: Purpose / What / Why / Config / I/O hashes / Settings / Summary / Diagnostics.

---

## 3. Solution Representation (Genotype)

Let \( N \) be the number of jobs (Step‑01 universe).  
Fix an index order for jobs based on STEP-02 GS feeder:

\[
J = \{1,\dots,N\} \quad \text{in feeder order}.
\]

A genome \(g\) is a triple:

\[
g = (\sigma, \pi_1, \pi_2)
\]

where:

1. **Split pattern** \( \sigma \in \{1,2\}^N \)

   - \( \sigma_k \in \{1,2\} \) assigns job \(J_k\) to booth PB1 or PB2.  
   - This directly generalizes STEP-03, which corresponds to a particular deterministic \(\sigma\).

2. **Per-booth permutations** \( \pi_1, \pi_2 \)

   - For booth \(b\), define its job index set:

     \[
     J_b(\sigma) = \{ k \in J : \sigma_k = b \}.
     \]

   - \( \pi_b \) is a permutation acting on \(J_b(\sigma)\).  
   - In practice, we store `pi1_encoding`, `pi2_encoding` as compact integer arrays over **local indices** for efficiency.

This representation is expressive enough to reproduce:

- GS+greedy split — by fixing \(\sigma = \sigma^{\text{split}}\) and \(\pi_b =\) identity.  
- SDSU — by choosing \(\sigma = \sigma^{\text{split}}\) (same split) and \(\pi_b = \pi_b^{\text{SDSU}}\).  
- MILP per-booth — by using the MILP booth assignment and orders.

---

## 4. Decoder — From Genotype to Schedule

The decoder is **deterministic** given:

- A genome \(g = (\sigma, \pi_1, \pi_2)\),  
- Dataset and configs,  
- C₁/C₂/C₃ policy from the protocol.

Its job is to produce:

- Feasible booth sequences \(S_1, S_2\) (JOB-only),  
- A merged JOB-only convoy \(G\) under the same semantics as STEP‑05,  
- Or else reject the genome as infeasible.

### 4.1 Stage A — Build Booth Job Lists

1. Start from GS feeder `step02_paint_seed.csv`; extract the JOB list in feeder order:
   \[
   (J_1, J_2, \dots, J_N).
   \]
2. Partition jobs by \(\sigma\):  
   - PB1 gets all \(J_k\) with \(\sigma_k = 1\),  
   - PB2 gets all \(J_k\) with \(\sigma_k = 2\).

This ensures that **every job appears exactly once** in exactly one booth, mirroring the split logic in STEP‑03.

### 4.2 Stage B — Apply Per-Booth Permutations

For each booth \(b\):

1. Take the list \(J_b(\sigma)\) in feeder order.  
2. Apply \( \pi_b \) as a reordering operator to get a provisional booth order:
   \[
   S_b^\star = (j^b_1, \dots, j^b_{n_b}).
   \]

This is the raw genome-induced order; it is not yet guaranteed to satisfy C₁/C₂.

### 4.3 Stage C — Feasibility Repairs (C₂ then C₁)

We apply repairs in a **bounded and deterministic** way; failure to repair leads to genome rejection.

1. **Adjacency repair (C₂).**

   - Scan \(S_b^\star\) from left to right; for each pair \((j^b_i, j^b_{i+1})\), check adjacency forbids via `adjacency_rules.csv` on parts.  
   - If a forbidden pair is detected, attempt a **local reshuffle** within a window of size `L_max` (config `[nsga2].repair_L_max`), e.g.:

     - Try swapping with a job later in the window that does not violate adjacency.  
     - Or swapping local intervals.

   - If no feasible replacement is found within the window, mark the genome as **rejected due to booth adjacency**.

2. **Paint repair (C₁ + C₃).**

   - After adjacency is fixed, scan again; for each consecutive pair \((j^b_i, j^b_{i+1})\), consider the color transition `τ(j^b_i) → τ(j^b_{i+1})`.  
   - Use the common paint cost kernel (as in STEP-06):

     - If transition is forbidden even with purge ⇒ **reject** genome (booth paint infeasible).  
     - If purge is allowed and cheaper ⇒ record purge usage and its cost.  
     - Else charge the direct setup cost.

At the end of Stage C we have per-booth sequences:

\[
S_b = (j^b_1, \dots, j^b_{n_b})
\]

that obey C₂ and (C₁/C₃) at the booth level.

### 4.4 Stage D — Optional SDSU Kernel

If `[nsga2].sdsu_kernel = true`:

- For each booth \(b\), run **a single pass** of SDSU’s constructive + local-improvement heuristic on \(S_b\), under the **same** S-matrix and adjacency rules.  
- The kernel is used as a **local search operator**; it cannot generate infeasible sequences (if it does under a bug, the decoder rejects the genome).

This connects the NSGA-II search to the SDSU heuristic as a domain-informed neighbourhood move.

### 4.5 Stage E — Merge to Global Convoy (Step-05 Semantics)

Given \(S_1, S_2\), the decoder:

1. Applies the same **2→1 selection policy** as STEP-05 (e.g., tie-breaking rules driven by C₂, quotas, or deterministic booth preference) to produce:
   \[
   G = (g_1, \dots, g_{N})
   \]
   where each job appears exactly once.

2. Enforces **global adjacency** C₂:
   - Forbid pairs labeled `across_skids` or `global` in `adjacency_rules.csv`.  
   - If the merge policy cannot find any feasible next job (both heads blocked), this genome is rejected as **merge-adjacency infeasible**.

Tokens (`EMPTY`, `FLUSH`) are **not materialized** in the final CSVs; they are only conceptual and costed via the paint kernel.

If the decoder succeeds, we obtain a fully specified NSGA-II candidate schedule (booth sequences + merged convoy) whose feasibility is consistent with Steps 02–06.

---

## 5. Objectives and Evaluation

Let \(G\) be the merged JOB-only convoy; let \(S_1, S_2\) be the booth sequences; let \(W\) be the quota window size (parts-only).

### 5.1 f₁ — Paint Setup Cost

- Use the **exact same** cost kernel as STEP-06 (S-matrix + purge rule).  
- Compute per-booth arc costs on \(S_1\) and \(S_2\); sum over both booths:

\[
f_1(g) = \sum_{b \in \{1,2\}} \sum_{(i,j) \in \text{arcs}(S_b)} c_{i,j}.
\]

This is directly comparable to SDSU and MILP f₁.

### 5.2 f₂ — Windowed Mix Deviation (Quotas)

- On the merged convoy \(G\), consider a sliding window of size \(W\) jobs (parts-only).  
- For each window, compute the empirical color distribution and compare against `cycle_quota.csv` (target shares).  
- Define f₂ as, e.g.:

\[
f_2(g) = \max_{\text{windows } w} \sum_{c} \left| \hat{q}_c(w) - q_c^{\star} \right|,
\]

where \(\hat{q}_c(w)\) is the observed share of topcoat \(c\) within window \(w\), \(q_c^{\star}\) the target share.

Implementation uses the same quota window logic as STEP-02/05.

### 5.3 f₃ — Line Balance / Crossing

A scalar balance metric; for example:

\[
f_3(g) = \left| \left( \#\text{jobs}_{B1} + \rho \cdot \#\text{cross}_{B1} \right)
       - \left( \#\text{jobs}_{B2} + \rho \cdot \#\text{cross}_{B2} \right) \right|,
\]

where:

- \(\#\text{jobs}_{B b}\) = number of jobs assigned to booth \(b\);  
- \(\#\text{cross}_{B b}\) = count of “crossing” events or other balancing proxy;  
- \(\rho\) is a weighting parameter defined in the protocol/config.

Exact definition of crosses follows the same semantics as STEP-03/05, ensuring consistency.

### 5.4 Optional f₄ — DES-Based Blocking/Idle

If configured, STEP-07 can pass a subset of solutions to STEP-08 DES as a **post-processing hook** and import f₄ values back into the evaluation:

- f₄(g): combination of oven idle time, booth blocking time, and normalized indicators, as defined in STEP-08.

In the base NSGA-II run, only (f₁,f₂,f₃) are used for ranking; f₄ is treated as an additional reported metric, or as a fourth objective in an extended experiment.

---

## 6. NSGA-II Algorithm

We use a standard NSGA-II loop (Deb et al., 2002) with a **feasibility-aware decoder** and **two initialization modes**.

### 6.1 Population and Generations

Config keys (example):

- `[nsga2].pop_size` — population size, \(P\).  
- `[nsga2].n_generations` — number of generations, \(G\).  
- `[nsga2].pc` — crossover probability.  
- `[nsga2].pm` — mutation probability.  
- `[nsga2].tournament_k` — tournament size for selection.

### 6.2 Initialization

Two modes:

1. **Unseeded**

   - For each individual:
     - Sample split pattern \(\sigma\) uniformly over \(\{1,2\}^N\) or with a mild balancing bias.  
     - For each booth \(b\), sample a random permutation \(\pi_b\) of \(J_b(\sigma)\).  
   - Run the decoder; reject infeasible genomes and resample until P feasible individuals are obtained (or a retry cap is hit).

2. **Seeded**

   - Construct seed genomes directly from:

     - GS + split: derive \(\sigma\) from STEP-03, \(\pi_b\) as identity.  
     - SDSU: \(\sigma\) from STEP-03, \(\pi_b\) from mapping SDSU sequences back to `J_b(\sigma)\).  
     - MILP (optional): derive \(\sigma\) and \(\pi_b\) from MILP best schedules.

   - Insert these seed genomes into the initial population (with or without small perturbations).  
   - Fill the remaining slots by random sampling as in the unseeded mode.

A config key `[step07].seeding_mode` switches between `"none" | "gs" | "sdsu" | "milp" | "mixed"`.

### 6.3 Variation Operators

On the genotype \( (\sigma, \pi_1, \pi_2) \):

1. **Crossover**

   - **Split crossover** on \(\sigma\): one-point or uniform crossover on the bitstring.  
   - **Permutation crossover** on each booth:
     - E.g. order-based or partially matched crossover (PMX) restricted to \(J_b(\sigma)\).

2. **Mutation**

   - **Split mutation:** flip booth assignment for a small subset of positions \(k\).  
   - **Permutation mutation:** apply swap or insertion moves within \(\pi_b\).

All offspring go through the decoder; infeasible offspring are discarded or repaired (depending on configuration) and replaced by resampled offspring if needed.

### 6.4 Selection and Replacement

At each generation:

1. Combine parent and offspring populations (size ≤ 2P).  
2. Apply **non-dominated sorting** on (f₁, f₂, f₃[, f₄]) to obtain fronts.  
3. Compute **crowding distance** within each front.  
4. Select the next generation by filling fronts in order, breaking ties by crowding distance (max distance preferred).  

This is the standard NSGA-II environmental selection.

---

## 7. Configuration, Gating, and Reproducibility

### 7.1 Step gating

STEP-07 has an explicit gate:

- If upstream manifests indicate **failed gates** for Steps 02–05, STEP-07 either refuses to run or runs only on explicitly marked “toy” datasets.  
- If `[step07].enabled = false`, the script writes a manifest with `status = "DISABLED_BY_CONFIG"` and an empty Pareto set.

### 7.2 Reproducibility

- Random seeds come from `[reproducibility].seeds` in `global_config.toml` or `[step07]` block.  
- All seeds, solver versions, and OS/environment metadata are written into `step07_manifest.json`.  
- Given the same configs, dataset, and seeds, the NSGA-II run is **bitwise deterministic** (subject to Python RNG and possible Gurobi calls if used for repairs).

### 7.3 Key Hyperparameters (Summary)

Config keys (illustrative, actual names must match the final `steps_config.toml`):

- `[nsga2].pop_size`, `[nsga2].n_generations`, `[nsga2].pc`, `[nsga2].pm`.  
- `[nsga2].tournament_k` — selection pressure.  
- `[nsga2].repair_L_max` — local repair radius for adjacency.  
- `[nsga2].sdsu_kernel` — toggle for SDSU-based local search.  
- `[step07].seeding_mode` — controls seeded vs unseeded initialization.  
- `[step07].max_decoder_retries` — cap on rejected genomes per generation.

---

## 8. Metrics and Diagnostics (for the Report)

The Step‑07 report (`step07_report.md`) summarizes:

- **Pareto front statistics**: size, objective ranges, relative improvement vs GS/SDSU/MILP.  
- **Seeding impact**: dominance coverage of seeded vs unseeded individuals, convergence curves.  
- **Feasibility diagnostics**: counts of decoder rejections by cause (booth adjacency, booth paint, merge adjacency).  
- **Diversity indicators**: entropy over split patterns and distributions over simple shape descriptors (e.g., block lengths, number of color switches).

These diagnostics are used in STEP-09 to position NSGA-II relative to all other methods in terms of both **solution quality** and **search behaviour**.

