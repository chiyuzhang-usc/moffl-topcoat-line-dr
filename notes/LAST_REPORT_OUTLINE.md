
# Multi‑Objective Flexible Flow Paint Shop: Comparision of methods in Data‑Driven Scheduling with Toyota GS, SDSU, MILP, NSGA‑II, and DES

## Abstract
**Context:** Two paint booths feeding a single oven; costly sequence‑dependent setups; data‑encoded forbids.  
**Problem:** Construct schedules that minimize setup effort and balance work while meeting data‑driven feasibility and downstream constraints.  
**Methods:** Toyota Grouping & Spacing (GS), booth split, per‑booth SDSU (non‑cyclic), 2→1 serialization with global adjacency, exact MILP baselines, NSGA‑II (unseeded/seeded), DES for blocking/idle.  
**Results:** Summary of f1–f4 and dominant findings.  
**Conclusion:** Key takeaways and implications.

**Keywords:** flexible flow shop, paint scheduling, sequence‑dependent setups, grouping & spacing, SDSU, MILP, NSGA‑II, discrete‑event simulation

---

## 1. Introduction
### 1.1 Industrial Context
- Paint‑shop topology (2 booths → 1 oven); physical rationale for adjacency rules and purges.
- Operational pressures: color contamination, mixing targets, throughput.

### 1.2 Research Problem & Objectives
- Problem frame (MOFFS) with hard feasibility from data.  
- Objectives \(f1–f4\) and their real‑world meaning.  
- Research questions and hypotheses (e.g., value of seeding; sensitivity to purge cost; effect of window size and ρ).

### 1.3 Contributions
- A data‑driven, reproducible pipeline; strict separation of concerns across steps.  
- Exact baselines for certification; evolutionary search with feasibility‑aware decoding.  
- Operational evaluation via DES.

---

## 2. Data & Configuration (Authoritative Sources)
### 2.1 Dataset Files & Semantics
- `demands.csv`, `parts.csv`, `topcoat_codebook.csv`, `s_matrix_transition.csv`, `s_lookup_transition.csv` (optional), `topcoat_rules.csv`, `adjacency_rules.csv`, `cycle_quota.csv`, `line_cadence.csv`, `fixtures.csv` (optional).  
- Brief note on each file’s role; forbids encoded as `INF` or explicit rules.

### 2.2 Global Configuration (`model_config.toml`)
- Paths, logging, determinism, token unit ratios; window and spacing; adjacency scope policy; split load ρ; merge policy; MILP/NSGA‑II/DES sections.  
- **Snapshot table:** key parameters used for the reported runs (echo values + SHA‑256 of config).

### 2.3 Data Integrity & Feasibility Checks
- Joins and coverage (Step‑01 report cross‑check).  
- Sanity checks for shares (sum≈1), matrix squareness, rule conflicts.

---

## 3. Methods
*(Explain without code; reference step notes for details.)*

### 3.1 Toyota Grouping & Spacing (Feeder)
- Window \(W\), run caps, quota tracking, candidate scoring.  
- Adjacency enforcement on parts‑only; `EMPTY` and `FLUSH` semantics; determinism & tie‑breaks.

### 3.2 Deterministic Split to Booths
- Load metric `jobs + ρ×cross`; adjacency & paint feasibility in‑booth; bounded reshuffle (no tokens).

### 3.3 SDSU per Booth (Non‑Cyclic)
- Cost model (direct vs purge+start), adjacency preservation, constructive+2‑opt or DP for small n; determinism.

### 3.4 2→1 Oven Serialization (Global Adjacency)
- Selection policies; order preservation within booths; optional post‑paint `EMPTY` (if enabled).

### 3.5 MILP (Exact Baselines, Reduced Instances)
- Open‑path SDSU per booth; purge as reset‑and‑start; adjacency forbids enforced by arc removal; solver settings.

### 3.6 NSGA‑II (Unseeded & Seeded)
- Genome (split + per‑booth permutations); feasibility‑aware decoder mirroring Steps 03–05; operators and seeds.

### 3.7 DES (Blocking & Idle)
- Timing source (lookup vs constants), buffer β, ratio θ; event priorities; invariants.

### 3.8 Determinism & Reproducibility
- Manifests, SHA‑256, seeds; environment logging; step gates and acceptance criteria.

---

## 4. Experimental Design
### 4.1 Baseline & Variants
- **Baseline:** GS→Split→SDSU→Merge with defaults.  
- **Sweeps:** \(W\) ratio; ρ; purge unit ratio; adjacency scope; merge policy; NSGA‑II population/generations; DES β and timing mode.

### 4.2 Instance Selection
- Full dataset runs for heuristics; reduced instances for MILP; alignment across methods.

### 4.3 Evaluation Protocol
- Identical definitions for \(f1–f4\); window counts on parts‑only; identical merge policy; token counting rules.

---

## 5. Results
*(Insert real numbers once runs complete; provide tables/figures listed here.)*

### 5.1 GS vs GS→SDSU (Per‑Booth)
- **Table R1:** f1 (setup units), changeover counts, longest runs per booth.  
- **Figure F1:** f1 improvement distribution; violin/box across booths.

### 5.2 Post‑Merge Effects
- **Table R2:** f2 window deviations and longest color block on global convoy.  
- **Figure F2:** run‑length histograms (global).

### 5.3 MILP Certification (Reduced Instances)
- **Table R3:** Optimal f1 vs SDSU f1; optimality gaps; solve times/gaps.  
- **Figure F3:** gap vs instance size.

### 5.4 NSGA‑II Fronts (Unseeded vs Seeded)
- **Table R4:** front size, dominance coverage, hypervolume (if used).  
- **Figure F4:** f1–f2–f3 scatter (color by seeding mode); knee identification.

### 5.5 DES (β, θ)
- **Table R5:** oven idle, booth blocking, makespan; θ_static & θ_realized.  
- **Figure F5:** f4 vs β curves under constant vs lookup timing.

### 5.6 Sensitivities & Interactions
- Purge unit ratio vs f1; W vs f2; ρ vs f3; merge policy vs conflicts/empties.  
- **Figure F6:** tornado/radar plots or partial‑dependence style summaries.

---

## 6. Anomalies & Validity
- Token flooding checks (`EMPTY/FLUSH` per 100 jobs); any violation certificates.  
- Order/adjacency invariants: counts must be zero; if nonzero, provide exact indices and dataset rows.  
- Solver anomalies (timeouts/gaps) and how they affect interpretation.  
- Data issues encountered and fixes (e.g., missing topcoat in codebook).

---

## 7. Discussion
- Interpretation of trade‑offs among \(f1–f3\) and operational implications from DES.  
- When seeding helps (or not); when purge cost regimes change dominance.  
- Practical guidance for plant tuning (window size, ρ, purge cost).

---

## 8. Threats to Validity & Limitations
- Data representativeness; policy generality; single‑oven scope; deterministic timing assumption in DES.

---

## 9. Conclusions
- Crisp answers to the research questions; recommended configurations; future work (multi‑oven extension, richer adjacency scopes).

---

## 10. Reproducibility & Artifacts
- **Manifest table:** input hashes; config hash; environment.  
- **Code & scripts:** step names (`stepXX_*.py`), paths relative to repo root.  
- **How to regenerate figures/tables:** one‑line commands per artifact (e.g., `python step09_analyze.py --plot F1`).

---

## References
*(Add as appropriate to your style guide.)*

---

## Appendices
**A. Configuration Snapshot** — exact `model_config.toml` used.  
**B. Additional Tables** — extended result matrices.  
**C. SDSU Method Reminder** — concise restatement of the non‑cyclic formulation used (no external jargon).  
**D. DES Timing Schema** — `line_cadence.csv` schema and lookup precedence (part/topcoat/global).

