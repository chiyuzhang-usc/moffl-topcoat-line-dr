# STEP-01 Report – Job Expansion

## Purpose
Expand aggregate demands into a canonical job-level table and attach part/topcoat metadata.

## What
Reads demands, parts, and topcoat codebook tables from the dataset; expands each demand row into one row per physical job; assigns stable job_ids and provenance; joins part/topcoat attributes; writes the canonical job table.

## Why
All downstream scheduling (Toyota grouping/spacing, SDSU, MILP, NSGA-II, DES) must operate on a single, consistent job universe. Step-01 isolates this concern and removes any dependence on aggregate demand representation.

## Config
- Global config: `global_config.toml`
- Steps config: `steps_config.toml`
- Dataset dir: `C:\Users\cz\OneDrive - University of Southern California\DirectedResearch_Gupta\Procedure\dataset`
- Output dir: `C:\Users\cz\OneDrive - University of Southern California\DirectedResearch_Gupta\Procedure\output`
- fail_on_missing_join: `hard`

## I/O hashes
- demands: `1a58fdebb31fc9aa73b3f6caa57b04e2b503dadb71cf7fc7405d11afb2f28bea`
- parts: `dc65368a59c113795686c25c466f70572b85eec39838e0c3bdf1765bb0392ba2`
- topcoat_codebook: `6fe43ca94c75c10caa87119e3880ba29022d8aaa33e9d5beb1ff6a912e7d40d8`
- jobs_expanded: `5343941e66d3f7b5f0ca7914023464f41d6039fe02a5750098519a1b9c00d9a5`

## Settings
- time_unit (from global_config): paint_unit

## Summary
- Demand rows processed: 83
- Jobs generated: 183
- Zero-quantity rows skipped: 0
- Negative-quantity rows skipped: 0

## Diagnostics
- Missing parts in join: 0 (policy: hard)
- Missing topcoats in join: 0 (policy: hard)

Topcoat histogram (quantity per topcoat_code):
- C_DENIM_BLUE: 3
- C_DIAMOND_WHITE: 23
- C_IRIDIUM_SILVER: 12
- C_MILAN_SILVER: 11
- C_OBSIDIAN_BLACK: 47
- C_POLAR_WHITE: 44
- C_RUBY_RED: 13
- C_SAPPHIRE_BLUE: 11
- C_SHINING_BLUE: 18
- C_UNIVERSE_BLACK: 1
