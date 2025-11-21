# Step-05 — Oven Convoy Serialization (2→1 Merge)

## Purpose / What / Why

- **Purpose.** Merge SDSU-optimized booth sequences (PB1, PB2) into a single oven convoy while
  enforcing global adjacency forbids on parts and preserving in-booth order.
- **What.** Deterministic greedy 2→1 interleave, using adjacency_rules.csv for C2(global).
- **Why.** The merged line is the canonical schedule for DES and for computing f2 (window mix) and
  f4 (blocking/idle). This step is the unique owner of C2 on the merged 2→1 line.

## Effective Settings

```json
{
  "allow_postpaint_empty": true,
  "tie_break_policy": "longer_remaining",
  "adjacency_enabled": true,
  "scope_escalation": "global_if_missing",
  "empty_units_postpaint_ratio": 1.0
}
```

## Merge Metrics

- Total jobs on global line: 188
- Jobs from PB1: 91
- Jobs from PB2: 92
- Conflict events (candidates rejected due to adjacency): 16
- Deadlocks encountered: 5
- Post-paint EMPTY used: 5
- Max consecutive PB1 jobs: 3
- Max consecutive PB2 jobs: 2

## Notes

- Adjacency is evaluated on parts-only; no tokens are used to relax C2.
- Paint feasibility and purge (C1/C3) remain as enforced in per-booth SDSU (Step-04).
