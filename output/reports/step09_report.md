# Step-09 — Integrated Research Analysis

- Experiments: **1**
- Total schedules: **340**
  - Baselines (GS→SDSU): **2**
  - MILP schedules: **0**
  - NSGA-II schedules: **338**

## Method-level metrics (f1, f2, f3)

| method_id | count | mean f1 | mean f2 | mean f3 | dominates baseline (%) |
|-----------|-------|---------|---------|---------|-------------------------|
| GS_SDSU_baseline | 2 | nan | nan | nan | 0.0 |
| NSGA_seed_19 | 114 | 171.912 | 0.235 | 43.702 | 100.0 |
| NSGA_seed_23 | 34 | 181.471 | 0.241 | 37.529 | 100.0 |
| NSGA_seed_37 | 190 | 148.337 | 0.238 | 44.905 | 100.0 |

## NSGA-II seeded vs unseeded fronts

| exp_id | seeded_mode | front size | dominates baseline (%) | best Δf1 | best Δf2 | best Δf3 |
|--------|-------------|------------|-------------------------|---------|---------|---------|
| E0 | 19 | 114 | 100.0 | -inf | -inf | -inf |
| E0 | 23 | 34 | 100.0 | -inf | -inf | -inf |
| E0 | 37 | 190 | 100.0 | -inf | -inf | -inf |
