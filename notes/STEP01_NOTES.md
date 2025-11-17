
# Step‑01 - Expand Jobs (Join Data Tables)

## 1. Purpose & Scope

Transform `demands.csv` into a canonical, auditable **JOB table** with **one physical job per row**, stable identifiers, and attached metadata (topcoat & part descriptors). 
This step performs data preparation work only; it does not introduce tokens or enforce paint/adjacency feasibility (those begin in Step‑02).

---

## 2. Formulation

For each part type - topcoat(color) - demand set, a unique job is created with ID `J000001`, ...

---

## 3. Data & Semantics

For data processing purpose, `topcoat_no`, and other ID types are normalized

---

## 4. Algorithm (Deterministic; Pseudocode)

**Inputs:** `demands.csv`, `topcoat_codebook.csv`, `parts.csv`; schema hints from `model_config.toml`.

```
1  Read config (model_config.toml); resolve [paths] and [data_files].
2  Load CSVs with dtype=str; preserve original ordering for deterministic provenance.
3  Detect columns using [data_schema] synonyms:
      part_col  ← first match in demands_part_synonyms
      coat_col  ← first match in demands_topcoat_synonyms
      qty_col   ← first match in demands_quantity_synonyms
      book_code ← first match in codebook_topcoat_synonyms
      parts_id  ← first match in parts_id_synonyms
4  Normalize:
      part_id  := trim(part_col)
      topcoat  := upper(trim(coat_col))
      quantity := int(trim(qty_col))  with guard: quantity ≥ 0
5  Attach monotone provenance:
      prov_id  := row_index(demands) + 1
6  Expand rows:
      For each row r: repeat r quantity times; prov_seq := 1..quantity
7  Assign stable job_id:
      job_id := "J" + zfill(global_row_index, 6)
8  Left-join metadata:
      Join by topcoat on codebook; by part_id on parts
      If missing join(s) and fail policy is 'hard' → abort with diagnostics
9  Emit:
      output/step01_jobs_expanded.csv
      plus manifest (SHA-256) and report (metrics & runtime)
```

**Tie‑breaks.** Ordering is the **CSV appearance order** after normalization; if sorting is enabled (rare), the key is `(prov_id, prov_seq)`. 

---

## 5. Outputs and Column Contract

**Primary columns (required):**
- `job_id` (string; `J000001…`)
- `part_id` (string; normalized from demand)
- `topcoat_code` (string; UPPERCASE)
- `provenance_id` (int; 1..|demands|) — original demand row
- `provenance_seq` (int; 1..quantity)

**Attached metadata (examples; dataset‑dependent):**
- From codebook: color family, solvent group, booth constraints, etc.
- From parts: bracket/fixture ids, surface area class, etc.

**File locations (relative):**
- Table: `./output/step01_jobs_expanded.csv`
- Manifest: `./output/manifests/step01_manifest.json`
- Report: `./output/reports/step01_report.md`

---

## 6. Parameters & Tuning (Real‑World Meaning, Effects, Extremes)

| Key (in `model_config.toml`) | Meaning / Real‑World Interpretation | ↑ Increases … | ↓ Decreases … | Extremes & Degeneracy |
|---|---|---|---|---|
| `[data_schema].*` synonyms | Robust header detection across varying CSV exports | Robustness to messy exports | Strictness; risk of false negatives | Empty synonym lists ⇒ fragile ingestion |
| `[step01].fail_on_unknown_keys` | Hard‑fail when schema detection is incomplete | Data quality; early error discovery | Tolerance to minor schema drift | `false` with missing joins ⇒ silently incomplete metadata (avoid) |
| `[logging].level` | Diagnostic verbosity | Insight into anomalies | Noise | Too low may hide issues |
| Encoding policy (UTF‑8/Latin‑1 fallback) | Readability across vendors | Read success rate | Strictness | Disallow fallback ⇒ hard stop on encoding blips |

**Quantities.** Negative or non-integer `quantity` are invalid; zero expands to zero rows and must be reported.

**String normalization.** Trim whitespace; `topcoat_code`→UPPERCASE; redundant spaces collapsed. Over-aggressive normalization can conflate distinct codes—avoid heuristic mapping.

---

## 7. Complexity & Performance

- **Time:** \( O(|\mathcal{D}| + \sum q_d) \) to expand, plus \( O(N) \) for joins.  
- **Space:** \( O(N) \) rows in memory/streaming; consider chunked I/O for very large \( N \).  
- **Dominant factor:** the expansion size \( N \).

---

## 8. Metrics & Diagnostics (for the report)

- **Counts:** total JOB rows \(N\); distinct `topcoat_code`; distinct `part_id`.  
- **Integrity:** missing codebook joins; missing part joins; duplicate `(job_id)` (=0); non‑positive `quantity` rows (=0).  
- **Distribution:** histogram of `topcoat_code` and (optionally) of parts.  
- **Provenance:** max `provenance_id`; max `provenance_seq` per `provenance_id`.  
- **Reproducibility:** SHA‑256 of all inputs and the output table; runtime (seconds).

---

## 9. Acceptance Criteria (Hard Gates)

1. No missing joins unless explicitly allowed; if allowed, they are **counted and listed**.  
2. No negative or non‑integer quantities; zero‑quantity rows are reported and excluded from expansion.  
3. All paths are **relative** (no absolute paths); manifest present and complete.  
4. Deterministic tie‑break policy documented in the report.

---

## 10. Reproducibility: Manifest & Report

**Manifest JSON (minimum fields):**
- `step_id="step01_expand_jobs"`
- `config_snapshot` (subset of keys actually read)
- `inputs` `{rel_path, sha256}` for `demands.csv`, `topcoat_codebook.csv`, `parts.csv`
- `outputs` `{rel_path, sha256}` for `step01_jobs_expanded.csv`
- `metrics` & `diagnostics` (as in §8)
- `runtime_sec`

**Report MD (sections):**
- Purpose / What / Why / Config snapshot / Inputs-Outputs (rel paths + SHA) / Settings / Metrics / Diagnostics / Runtime / Determinism Note.

---

## 11. Execution Instructions (How to Repeat the Step)

```bat
python step01_expand_jobs.py
```

Artifacts are written under `./output` using **relative** paths controlled by config files.

---

## 12. Failure Modes & Recovery

- **Schema not found:** Fix `[data_schema]` synonyms or correct headers in source CSVs.  
- **Missing joins:** Repair codes in `demands.csv`, or complete the dictionaries (`topcoat_codebook.csv`, `parts.csv`).  
- **Invalid quantities:** Ensure integers ≥ 0; correct upstream ERP export.  
- **Encoding errors:** Convert to UTF‑8 with BOM removed; enable controlled fallback in ingestion.  
- **Duplicate job_ids:** Indicates non-deterministic ordering or re-run with partial overwrite; clear outputs and re-run.

**Minimal change policy:** Prefer fixing the **data** or **schema hints** over altering code logic.

---

## 13. Interface to Step‑02 (Contract)

Step‑02 consumes **only**:
- `job_id`, `part_id`, `topcoat_code`, `provenance_id`, `provenance_seq`, plus any topcoat/part metadata it uses for grouping/spacing.  
- No tokens appear; Step‑02 constructs tokens from scratch.

**Stability Requirement:** `job_id` ordering is stable across identical inputs/config; any change in expansion must be accompanied by a new manifest hash.

---

## 14. Verification Checklist (Quick Audit Before Proceeding)

- [ ] `step01_jobs_expanded.csv` exists; SHA recorded.  
- [ ] All joins successful (or documented exceptions).  
- [ ] \(N\) equals the sum of `quantity` in `demands.csv` (excluding zeros).  
- [ ] Distinct `topcoat_code` and `part_id` counts look plausible for the dataset.  
- [ ] Report includes runtime and determinism note.

---

## 15. Appendix

### A. Schema Detection Algorithm (Deterministic)
- Iterate synonyms in order listed in `model_config.toml`; first match wins; failure emits a structured error when `fail_on_unknown_keys=true`.

### B. Identifier Scheme
- `job_id = "J" + zfill(i, 6)` with \( i = 1..N \); `provenance_id` = original demand row index+1; `provenance_seq` = 1..quantity.
