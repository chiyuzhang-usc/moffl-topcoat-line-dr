# STEP-07 — NSGA-II Pareto Search

Jobs evaluated: 183
Pareto solutions recorded: 169 (feasible: 169)
Runtime: 160.55 seconds

## Seeds
19, 23, 37

## Settings
{
  "tie_break_policy": "longer_remaining",
  "repair_L_max": 3,
  "rho_scale": 1.0,
  "window_mode": "ratio",
  "window_jobs": 0,
  "window_ratio": 0.15,
  "allow_purge": true,
  "flush_cost_units": 3.0,
  "population_size": 100,
  "generations": 200,
  "crossover_rate": 0.9,
  "mutation_rate": 0.1,
  "use_seeding": false,
  "use_sdsu_kernel": false,
  "seeds": [
    19,
    23,
    37
  ],
  "max_eval_per_seed": 20000
}