# Step-04 — SDSU per Booth

## Purpose / What / Why

- **Purpose.** For each paint booth, compute a paint-feasible, adjacency-respecting open path order that
  minimizes sequence-dependent setup cost plus purge cost, under the SDSU model.
- **What.** Build per-booth SDST cost matrices from `s_matrix_transition.csv` + `topcoat_rules.csv`,
  enforce in-booth adjacency forbids from `adjacency_rules.csv`, and solve SDSU with exact subset DP for
  small booths and a greedy + 2-opt heuristic for larger booths.
- **Why.** This step provides per-booth paint-optimal orders that feed the 2→1 merge (Step-05) and
  anchor the f1 (setup + purge) dimension for comparison against MILP and NSGA-II schedules.

## Effective Settings

```json
{
  "dp_threshold_jobs": 40,
  "max_2opt_passes": 5,
  "allow_purge": true,
  "purge_cost_weight": 1.0,
  "flush_units_ratio": 3.0
}
```

## Booth Metrics

### Booth 1

- Jobs: 91
- Branch: `heuristic`
- Total setup cost (paint units): 9.000
- Number of purge events (FLUSH): 0
- Flush positions (between jobs, 0-based): []

### Booth 2

- Jobs: 92
- Branch: `heuristic`
- Total setup cost (paint units): 8.000
- Number of purge events (FLUSH): 0
- Flush positions (between jobs, 0-based): []
