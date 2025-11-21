# STEP-06 (variant) — MILP using STEP-03 booth streams
## Purpose / What / Why
- **Purpose:** Compute exact or near-exact f1 paint setup costs on per-booth job sets directly from STEP-03, without using SDSU ordering as input.
- **What:** MILP open-path ATSP models with the same C1/C2 semantics as SDSU, but defined over the STEP-03 split streams.
- **Why:** Provide a non-SDSU baseline MILP benchmark that operates purely on the greedy f3 booth assignment from STEP-03.

## Config (snapshot)
```json
{
  "tokens": {
    "flush_units_ratio": 3.0,
    "empty_units_prepaint": 1.0,
    "empty_units_postpaint_ratio": 1.0
  },
  "adjacency": {
    "enabled": true,
    "scope_escalation": "global_if_missing"
  },
  "sdsu": {
    "allow_purge": true,
    "purge_cost_weight": 1.0
  },
  "milp": {
    "solver": "gurobi",
    "time_limit_sec": 600.0,
    "mip_gap": 0.01,
    "max_jobs_per_instance": 120,
    "threads": 0,
    "log_to_console": true
  },
  "step06": {
    "milp_output_prefix": "step06_milp_opt_",
    "manifest_relpath": "manifests/step06_from_step03_manifest.json",
    "report_relpath": "reports/step06_from_step03_report.md",
    "enable_reduced_instances": false,
    "postpaint_empty_enabled": true
  }
}
```

## Summary / Diagnostics

| Instance | Booth | n_jobs | status | f1_milp | f1_baseline | solver_gap | time_sec |
|---------:|------:|-------:|--------|--------:|------------:|-----------:|---------:|
| pb1_full | 1 | 91 | TIME_LIMIT | 9.000 |  | 0.8889 | 600.02 |
| pb2_full | 2 | 92 | TIME_LIMIT | 8.000 |  | 0.8750 | 600.05 |
