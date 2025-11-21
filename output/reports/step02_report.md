# STEP-02 Report — Toyota Grouping & Spacing (GS)

## Purpose
Construct a C1/C2-feasible single-file feeder stream using GS; insert EMPTY only for adjacency; optional cycle quotas.

## Config (Effective)
```json
{
  "window_W": 28,
  "alpha_run_cap": 0.6,
  "max_run_abs": 30,
  "fifo_lookahead": 3,
  "token_costs": {
    "flush_unit": 1.0,
    "empty_unit": 1.0
  },
  "empty_weight": 1.0,
  "flush_weight": 1.0,
  "run_caps": {
    "C_DENIM_BLUE": 16,
    "C_DIAMOND_WHITE": 16,
    "C_IRIDIUM_SILVER": 16,
    "C_MILAN_SILVER": 16,
    "C_OBSIDIAN_BLACK": 16,
    "C_POLAR_WHITE": 16,
    "C_RUBY_RED": 16,
    "C_SAPPHIRE_BLUE": 16,
    "C_SHINING_BLUE": 16,
    "C_UNIVERSE_BLACK": 16
  },
  "colors": [
    "C_DENIM_BLUE",
    "C_DIAMOND_WHITE",
    "C_IRIDIUM_SILVER",
    "C_MILAN_SILVER",
    "C_OBSIDIAN_BLACK",
    "C_POLAR_WHITE",
    "C_RUBY_RED",
    "C_SAPPHIRE_BLUE",
    "C_SHINING_BLUE",
    "C_UNIVERSE_BLACK"
  ],
  "use_cycle_quota": false,
  "cycle_size": null,
  "schema": {
    "topcoat": "color_code",
    "part": "part_id",
    "job_id": "job_id"
  },
  "output_dir": "C:\\Users\\cz\\OneDrive - University of Southern California\\DirectedResearch_Gupta\\Procedure\\output"
}
```

## Metrics
- num_jobs: 183
- window_W: 28
- setup_units: 12.0
- num_flush: 3
- num_empty: 5
- quota_violations: 0

## Diagnostics
{
  "token_distribution": {
    "JOB": 183,
    "EMPTY": 5,
    "FLUSH": 3,
    "START": 1
  },
  "max_run_len_by_topcoat": {
    "C_IRIDIUM_SILVER": 12,
    "C_SHINING_BLUE": 16,
    "C_OBSIDIAN_BLACK": 16,
    "C_POLAR_WHITE": 16,
    "C_MILAN_SILVER": 11,
    "C_SAPPHIRE_BLUE": 11,
    "C_DIAMOND_WHITE": 16,
    "C_RUBY_RED": 13,
    "C_UNIVERSE_BLACK": 1,
    "C_DENIM_BLUE": 3
  },
  "quota_violations": 0
}

## Runtime
2025-11-15 23:16:20