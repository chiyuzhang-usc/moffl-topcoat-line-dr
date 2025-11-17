
# Step‑07 — NSGA‑II (Feasibility‑Aware Decoder; Unseeded & Seeded Protocols)

> This evolutionary stage adheres to the Master Note’s feasibility, token semantics, and metrics. The **decoder** maps genomes to fully feasible schedules using the same contracts as Steps 02–05, so comparisons (f1, f2, f3) are valid. We run **unseeded** and **seeded** experiments (GS and GS→SDSU) under identical evaluation to quantify seeding value.

The step implements NSGA-II (Non-dominated Sorting Genetic Algorithm II), a well-used genetic algorithm

---

## 1. Purpose — Formal, Intuitive, Research Perspective

### Formal
Compute Pareto‑optimal schedules for \((f1, f2, f3)\) (and optionally \(f4\)) using NSGA‑II with a **feasibility‑preserving** decoder. Given a genome, the decoder constructs per‑booth sequences and a 2→1 merged convoy that:
1) Satisfy **paint** forbids (matrix ∪ rules) using **purge** only where allowed and costed.  
2) Satisfy **adjacency** forbids at the feeder/booth and global levels.  
3) Use the same **selection/merge** policy as Step‑05.  
4) Evaluate objectives with identical accounting (tokens never waive adjacency; \(W\) counts parts only).

### Intuitive
We explore many alternative splits and booth orders, but **every candidate** is interpreted through the *same plant rules* used in earlier steps. Feasible candidates survive; infeasible ones are **repaired deterministically** or **rejected**. Seeding with GS and GS→SDSU provides strong starting points; unseeded runs probe novel regions of the space.

### Research perspective
We quantify:  
- The **value of seeding** (dominance coverage of seeded fronts over unseeded).  
- Sensitivity to `flush_units_ratio`, `W`, and `ρ`.  
- The **decoder’s rejection/repair rate**, indicating how much of the search space is actually acceptable under plant rules.

---

## 2. Connections to Neighbor Steps

- **From Step‑06 (MILP):** Use exact optima (small instances) to anchor/validate NSGA‑II results.  
- **To Step‑08 (DES):** Optionally simulate top Pareto points to evaluate \(f4\) and check operational consequences.

---

## 3. Inputs / Outputs (Relative Paths)

**Inputs**
- Dataset: `./dataset/s_matrix_transition.csv`, `./dataset/topcoat_rules.csv`, `./dataset/adjacency_rules.csv`, optional `cycle_quota.csv`  
- Config: `./model_config.toml` (`[nsga2]`, `[reproducibility]`, plus global keys used by the decoder: `[window]`, `[spacing]`, `[tokens]`, `[adjacency]`, `[load]`, `[step05.selection]`, `[step05.postpaint_empty]`)

**Outputs**
- Pareto archive CSV: `./output/step07_pareto.csv` (one row per nondominated solution with f1, f2, f3, tags, and seed provenance)  
- `./output/manifests/step07_manifest.json` (settings, seeds, metrics, diagnostics, runtime)  
- `./output/reports/step07_report.md`

---

## 4. Genome & Decoder (Feasibility‑Preserving)

### 4.1 Genome Encoding
We use a **two‑level** encoding:
1) **Split string** \( \sigma \in \{1,2\}^N \): assigns each job (in feeder appearance order from Step‑02) to **B1** or **B2**.  
2) **Per‑booth permutations** \( \pi_1, \pi_2 \): order the jobs within each booth.

This representation spans the Step‑03 and Step‑04 decisions.

### 4.2 Decoder (Deterministic; Contracts from Steps 03–05)
Given \( (\sigma, \pi_1, \pi_2) \):

**A) Build booth sequences (Step‑03 semantics, no tokens).**
- Respect the **feeder order** when extracting job lists per booth; then apply permutation \( \pi_b \).  
- **Repair 1 (adjacency):** If an immediate pair violates adjacency in booth \(b\), apply **bounded local swaps** (length ≤ `decoder.reshuffle_L_max`) to resolve; if impossible, **reject** genome.  
- **Repair 2 (paint):** For forbidden paint transitions, we consider **purge vs direct** semantics used in SDSU: if both are impossible (even via purge+start), **reject** genome; otherwise, the cost uses the cheaper of the two during evaluation. (No tokens are inserted in outputs; purges are internal costs.)

**B) SDSU kernel (optional, policy‑controlled).**
- If `[nsga2].sdsu_kernel = true`, run **one pass** of SDSU’s constructive+2‑opt on each booth, preserving adjacency and paint semantics. This aligns the seeded and unseeded evaluations and reduces bias from permutations that trivially violate paint costs. Deterministic tie rules apply.

**C) Merge to oven (Step‑05 semantics).**
- Serialize the two booth sequences using the **same** selection policy and global adjacency enforcement as Step‑05.  
- If `[step05.postpaint_empty].enabled = false` and both heads conflict, **reject** genome; if enabled, insert a diagnostic post‑paint EMPTY and continue (costed via ratio).

**Result.** A **JOB‑only** global convoy \(G\) plus diagnostic token counts; compute objectives \(f1, f2, f3\) identically to Steps 04–05.

---

## 5. Objectives (Evaluation)

- **f1 (setup units):** Per-booth total paint setup units with purge accounting (min of direct vs purge+start when both feasible). Sum across booths.  
- **f2 (windowed mix on \(G\)):** Evaluate on the merged JOB sequence using window \(W\) (parts‑only) and quota model from Master. Report `longest_color_block_jobs` and deficit statistics.  
- **f3 (balance):** \( |(jobs + \rho·cross)_{B1} − (·)_{B2}| \).  
- **(Optional) f4:** if configured, compute blocking/idle via DES on \(G\) (post‑evaluation hook).

All calculations mimic Steps 02–05 definitions.

---

## 6. Algorithm (NSGA‑II Loop; Deterministic Given Seed)

1) **Initialization** (two modes):
   - **Unseeded:** Randomly sample \( \sigma \), \( \pi_1 \), \( \pi_2 \) uniformly under fixed RNG seed(s). Decoder rejects infeasible genomes.  
   - **Seeded:** Inject elites from **GS** (Step‑02 → Step‑03 implied split heuristic) and **GS→SDSU** (Steps 03–04) by mapping their artifacts to \( (\sigma,\pi_1,\pi_2) \). Fill remainder randomly.

2) **Variation operators** (configured in `[nsga2]`):
   - **Split flip**: flip booth assignment on a random contiguous segment of the split string.  
   - **Booth PMX**: partially mapped crossover within each booth’s permutation.  
   - **Adjacent swap / insertion**: local mutation operators per booth.  
   - **Rates**: `pc` (crossover), `pm` (mutation).

3) **Decoder evaluation**: apply §4.2 deterministically; infeasible children are **rejected** (not silently repaired beyond bounded reshuffle for adjacency).

4) **Selection**: non-dominated sorting (NSGA‑II) with crowding distance; binary tournament (`tournament_k`).  
5) **Replacement**: elitist generational model (parents+offspring → next gen).  
6) **Termination**: fixed generations or early-stopping plateau (optional).

**Reproducibility**: record RNG seed(s), operator rates, and population/generation counts in the manifest.

---

## 7. Parameters

| Key | Meaning | ↑ Effect | ↓ Effect | Extremes / Degeneracy |
|---|---|---|---|---|
| `[nsga2].population` | Population size | Better coverage, slower | Faster, risk premature convergence | Too small ⇒ poor front |
| `[nsga2].generations` | Generations | More search time | Faster finish | Too few ⇒ immature front |
| `[nsga2].pc` | Crossover rate | Strong recombination | Rely on mutation | 0 ⇒ no crossover |
| `[nsga2].pm` | Mutation rate | Exploration | Exploitation | 0 ⇒ stagnation |
| `[nsga2].tournament_k` | Selection pressure | Faster convergence | Maintain diversity | Too high ⇒ loss of diversity |
| `[nsga2].sdsu_kernel` | Apply one-pass SDSU in decoder | Improves f1, consistency | None | Off may favor pathological permutations |
| `[nsga2].repair_L_max` | Bounded reshuffle for adjacency | Fewer rejections | More rejections | 0 ⇒ head-only, fragile |
| `[reproducibility].seeds` | RNG seeds array | Repeatability | — | Fixed per run for determinism |

All other keys consumed by the decoder (window, spacing, tokens, adjacency, load, merge policy) are **shared** with earlier steps to keep evaluation semantics identical.

---

## 8. Metrics & Diagnostics (for the Report)

- **Pareto statistics**: front size, hypervolume (if configured), dominance coverage of seeded vs unseeded.  
- **Feasibility diagnostics**: decoder rejection counts (by reason: booth adjacency, booth paint, merge adjacency).  
- **Diversity**: entropy on split patterns and booth permutations.  
- **Runtime** and RNG seed(s).

---

## 9. Acceptance Criteria (Hard Gates)

1. Decoder uses **exactly** the same feasibility rules and merge policy as Steps 03–05  to have comparable basis.
2. No candidate on the reported front violates feasibility.  
3. Seeded vs unseeded comparisons are run under **identical** settings and seeds recorded.

---

## 10. Reproducibility (Manifest & Report)

**Manifest JSON**: inputs, config snapshot (all keys used), seeds, operator settings, rejection histogram, front size, runtime.  
**Report MD**: Purpose → What/Why → Genome/Decoder → Operators → Metrics (front, coverage) → Diagnostics (rejections) → Runtime → Determinism note.

---

## 11. Execution Instructions (How to Repeat)

> **Do not execute now.** When authorized, from repository root:

```bash
python step07_nsga2.py
```
and 

```bash
python step07_nsga2_seeded.py
```

---

## 12. Failure Modes & Minimal Recovery

- **High rejection rate**: increase `repair_L_max` or relax `[adjacency].scope_escalation` if justified; confirm purge semantics allow feasible arcs.  
- **Front collapse**: increase population or generations; check operator rates.  
- **Seed bias dominance**: reduce seeded fraction or enable `sdsu_kernel` so unseeded can compete fairly.  
- **Non‑reproducibility**: ensure seeds array is fixed and recorded; avoid multi-threaded nondeterminism without controlled seeds.


---

## 13. Verification Checklist (Before Proceeding)

- [ ] Manifest and report present; seeds and operator settings recorded.  
- [ ] No feasibility violations on the front; decoder rejection histogram included.  
- [ ] Seeded‑vs‑unseeded fronts compared under identical configurations.  
- [ ] Pareto CSV includes f1, f2, f3 and provenance tags (`seeded_mode ∈ {none, gs, gs_sdsu}`).

