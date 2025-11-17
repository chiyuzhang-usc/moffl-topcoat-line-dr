# STEP-01 – Job Expansion and Enrichment (Implementation Doc)

## 1. Role in the MOFFS Protocol

STEP-01 is the **job-universe materialization** stage of the MOFFS paintshop study.
It takes aggregate demand records and converts them into a canonical job-level table
that every downstream method uses:

- Toyota grouping & spacing (STEP-02),
- ratio-based booth split (STEP-03),
- per-booth SDSU sequencing (STEP-04),
- MILP spot checks (STEP-06),
- NSGA-II search (STEP-07),
- DES (STEP-08),
- integrated analysis (STEP-09).

No scheduling decisions, tokens, or constraint semantics (C1/C2) are applied here.
STEP-01 is purely structural and data-driven.

## 2. Inputs and Bindings

All bindings come from `global_config.toml`:

- Dataset root: `[paths].dataset_dir`
- Files:
  - `[data_files].demands`
  - `[data_files].parts`
  - `[data_files].topcoat_codebook`
- Column detection is done via `data_schema.*` synonym lists:
  - `data_schema.demands.part_id_synonyms`
  - `data_schema.demands.topcoat_code_synonyms`
  - `data_schema.demands.quantity_synonyms`
  - `data_schema.parts.part_id_synonyms`
  - `data_schema.topcoat_codebook.topcoat_code_synonyms`

STEP-01 does **not** hard-code any dataset-specific column names; if the schema
changes, only `global_config.toml` must be updated.

## 3. Outputs

Paths come from `steps_config.toml` under `[step01]`, interpreted relative to
`[paths].output_dir` and `[paths].reports_dir`:

- `step01_jobs_expanded.csv` (JOB-universe table)
- `manifests/step01_manifest.json` (I/O hashes, diagnostics)
- `reports/step01_report.md` (run report with Purpose / What / Why / Config /
  I/O hashes / Settings / Summary / Diagnostics)

The job table contains:

- all original demand fields;
- `job_id` (`J000001`, `J000002`, ...);
- `prov_row_index` (1-based index of the demand row);
- `prov_seq_in_row` (1..quantity within that demand row);
- `part_*` fields copied from `parts.csv` (excluding the key column);
- `topcoat_*` fields copied from `topcoat_codebook.csv` (excluding the key).

## 4. Method

1. **Config loading**

   - `global_config.toml` is parsed to obtain system paths and dataset filenames.
   - `steps_config.toml` is parsed to obtain the relative output locations and
     the `fail_on_missing_join` policy for joins.

2. **Schema detection**

   For each input file, STEP-01 reads only the header row and uses the synonym
   lists under `data_schema.*` to locate required columns. If no synonym matches,
   STEP-01 raises a hard error with an explicit message.

   This makes STEP-01 resilient to harmless renamings (e.g., `topcoat_code`
   vs. `color_code`) without embedding dataset-specific knowledge in the code.

3. **Demand expansion**

   The demands file is read in stable file order:

   - For each row, the quantity field is parsed as an integer.
   - Rows with quantity < 0 are counted and skipped.
   - Rows with quantity = 0 are counted and skipped.
   - For each row with quantity `q > 0`, the row is expanded into exactly `q`
     jobs with:
     - global `job_id` numbering (1..N across the entire file);
     - `prov_row_index` = input row index (1-based);
     - `prov_seq_in_row` = 1..q within that row.

   A histogram of topcoat_code is accumulated as a basic diagnostic of mix.

4. **Metadata joins**

   The parts and topcoat codebook tables are each read once into dictionaries:

   - parts: `key → row`, where `key` is detected from `data_schema.parts`;
   - topcoats: `key → row`, where `key` is detected from
     `data_schema.topcoat_codebook`.

   For each job:

   - fields from the parts row (except the key) are added as `part_*`;
   - fields from the topcoat row (except the key) are added as `topcoat_*`.

   Missing joins are counted:

   - `missing_parts` – jobs whose part_id has no entry in `parts`;
   - `missing_topcoats` – jobs whose topcoat_code has no entry in the codebook.

   If `fail_on_missing_join = "hard"`, any missing joins cause STEP-01 to abort.
   If `fail_on_missing_join = "warn"`, the job universe is still written and
   the issue is recorded in diagnostics.

5. **Manifest and report**

   The manifest contains:

   - SHA-256 hashes for each input file (demands, parts, topcoat_codebook);
   - SHA-256 of the expanded jobs file;
   - a snapshot of relevant configuration;
   - expansion and join diagnostics.

   The Markdown report provides the required sections:

   - **Purpose** – one-paragraph description of STEP-01.
   - **What** – precise list of operations performed on the data.
   - **Why** – how STEP-01 fits the overall MOFFS experiment.
   - **Config** – paths and key settings used for the run.
   - **I/O hashes** – the SHA-256 digests for traceability.
   - **Settings** – important global settings (e.g., time unit).
   - **Summary** – headline counts (rows, jobs, skips).
   - **Diagnostics** – missing joins and topcoat histogram.

## 5. Semantics and Non-Semantics

STEP-01 is deliberately limited:

- No C1 paint constraints (no use of `s_matrix_transition.csv`).
- No C2 adjacency constraints (no use of `adjacency_rules.csv`).
- No tokens (`EMPTY`, `FLUSH`, `START`).
- No sequencing or scheduling decisions.

All of these elements are introduced **after** STEP-01, so that the job universe
is invariant under changes to scheduling policy. This is critical for a clean
comparison between Toyota, SDSU, MILP, NSGA-II, and DES in later steps.
