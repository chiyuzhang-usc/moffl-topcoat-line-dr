from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple

from utilities.config_utils import load_global_config, load_steps_config, resolve_paths
from utilities.io_utils import sha256_of_file, write_csv, write_json, write_text


@dataclass
class Step01Config:
    dataset_dir: Path
    output_dir: Path
    manifests_dir: Path
    reports_dir: Path

    demands_filename: str
    parts_filename: str
    topcoat_codebook_filename: str

    jobs_expanded_relpath: str
    manifest_relpath: str
    report_relpath: str

    fail_on_missing_join: str

    # resolved column names
    demands_part_col: str
    demands_topcoat_col: str
    demands_quantity_col: str
    parts_key_col: str
    topcoat_key_col: str


def _find_column(fieldnames: List[str], candidates: List[str], context: str) -> str:
    """Return the first column name in `fieldnames` that appears in `candidates`.

    Raises KeyError if none are found.
    """
    for c in candidates:
        if c in fieldnames:
            return c
    raise KeyError(f"No matching column found for {context}; tried {candidates!r} in {fieldnames!r}")


def _load_step01_config(global_cfg: Dict[str, Any], steps_cfg: Dict[str, Any]) -> Step01Config:
    paths_resolved = resolve_paths(global_cfg)
    dataset_dir = paths_resolved["dataset_dir"]
    output_dir = paths_resolved["output_dir"]
    manifests_dir = paths_resolved["manifests_dir"]
    reports_dir = paths_resolved["reports_dir"]

    data_files = global_cfg.get("data_files", {})
    df_demands = str(data_files.get("demands", "demands.csv"))
    df_parts = str(data_files.get("parts", "parts.csv"))
    df_topcoat = str(data_files.get("topcoat_codebook", "topcoat_codebook.csv"))

    step01_cfg = steps_cfg.get("step01", {})
    jobs_expanded_relpath = str(step01_cfg.get("jobs_expanded_relpath", "step01_jobs_expanded.csv"))
    manifest_relpath = str(step01_cfg.get("manifest_relpath", "manifests/step01_manifest.json"))
    report_relpath = str(step01_cfg.get("report_relpath", "reports/step01_report.md"))
    fail_policy = str(step01_cfg.get("fail_on_missing_join", "hard")).lower()
    if fail_policy not in {"hard", "warn"}:
        raise ValueError(f"Invalid fail_on_missing_join policy: {fail_policy!r}")

    # We delay schema detection to when we open files; here we just read synonym lists.
    schema = global_cfg.get("data_schema", {})
    dem_schema = schema.get("demands", {})
    parts_schema = schema.get("parts", {})
    topcoat_schema = schema.get("topcoat_codebook", {})

    return Step01Config(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        manifests_dir=manifests_dir,
        reports_dir=reports_dir,
        demands_filename=df_demands,
        parts_filename=df_parts,
        topcoat_codebook_filename=df_topcoat,
        jobs_expanded_relpath=jobs_expanded_relpath,
        manifest_relpath=manifest_relpath,
        report_relpath=report_relpath,
        fail_on_missing_join=fail_policy,
        demands_part_col=dem_schema.get("part_id_synonyms", ["part_id"])[0],
        demands_topcoat_col=dem_schema.get("topcoat_code_synonyms", ["topcoat_code"])[0],
        demands_quantity_col=dem_schema.get("quantity_synonyms", ["quantity"])[0],
        parts_key_col=parts_schema.get("part_id_synonyms", ["part_id"])[0],
        topcoat_key_col=topcoat_schema.get("topcoat_code_synonyms", ["topcoat_code"])[0],
    )

def _detect_demands_columns(path: Path, global_cfg: Dict[str, Any]) -> Tuple[str, str, str]:
    """Infer actual column names for part, topcoat, quantity using synonyms from global config."""
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
    schema = global_cfg.get("data_schema", {}).get("demands", {})
    part_syn = schema.get("part_id_synonyms", ["part_id"])
    topcoat_syn = schema.get("topcoat_code_synonyms", ["topcoat_code"])
    qty_syn = schema.get("quantity_synonyms", ["quantity"])
    part_col = _find_column(fieldnames, part_syn, "demands.part_id")
    topcoat_col = _find_column(fieldnames, topcoat_syn, "demands.topcoat_code")
    qty_col = _find_column(fieldnames, qty_syn, "demands.quantity")
    return part_col, topcoat_col, qty_col


def _detect_parts_key(path: Path, global_cfg: Dict[str, Any]) -> str:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
    schema = global_cfg.get("data_schema", {}).get("parts", {})
    syn = schema.get("part_id_synonyms", ["part_id"])
    return _find_column(fieldnames, syn, "parts.part_id")


def _detect_topcoat_key(path: Path, global_cfg: Dict[str, Any]) -> str:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
    schema = global_cfg.get("data_schema", {}).get("topcoat_codebook", {})
    syn = schema.get("topcoat_code_synonyms", ["topcoat_code"])
    return _find_column(fieldnames, syn, "topcoat_codebook.topcoat_code")


def _read_lookup_table(path: Path, key_col: str) -> Dict[str, Dict[str, Any]]:
    """Read a CSV file into a mapping key -> row-dict (all fields)."""
    table: Dict[str, Dict[str, Any]] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if key_col not in (reader.fieldnames or []):
            raise KeyError(f"Key column {key_col!r} not found in {path}")
        for row in reader:
            key = str(row[key_col])
            table[key] = dict(row)
    return table


def _expand_demands(
    demands_path: Path,
    part_col: str,
    topcoat_col: str,
    qty_col: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Expand aggregate demands into one row per physical job."""
    jobs: List[Dict[str, Any]] = []
    diagnostics: Dict[str, Any] = {
        "num_demand_rows": 0,
        "num_jobs": 0,
        "zero_quantity_rows": 0,
        "negative_quantity_rows": 0,
        "topcoat_histogram": {},
    }

    with demands_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if part_col not in fieldnames:
            raise KeyError(f"Demand part column {part_col!r} not found in {demands_path}")
        if topcoat_col not in fieldnames:
            raise KeyError(f"Demand topcoat column {topcoat_col!r} not found in {demands_path}")
        if qty_col not in fieldnames:
            raise KeyError(f"Demand quantity column {qty_col!r} not found in {demands_path}")

        job_counter = 0
        demand_row_index = 0
        for row in reader:
            demand_row_index += 1
            diagnostics["num_demand_rows"] += 1

            qty_raw = str(row.get(qty_col, "")).strip()
            try:
                qty = int(qty_raw)
            except ValueError:
                raise ValueError(f"Non-integer quantity {qty_raw!r} at demands row {demand_row_index}")

            if qty < 0:
                diagnostics["negative_quantity_rows"] += 1
                continue
            if qty == 0:
                diagnostics["zero_quantity_rows"] += 1
                continue

            topcoat_val = str(row[topcoat_col])

            hist = diagnostics["topcoat_histogram"]
            hist[topcoat_val] = hist.get(topcoat_val, 0) + qty

            for k in range(1, qty + 1):
                job_counter += 1
                job_id = f"J{job_counter:06d}"
                job_row: Dict[str, Any] = {}
                for fn in fieldnames:
                    job_row[fn] = row[fn]
                job_row["job_id"] = job_id
                job_row["prov_row_index"] = demand_row_index
                job_row["prov_seq_in_row"] = k
                jobs.append(job_row)

        diagnostics["num_jobs"] = len(jobs)

    return jobs, diagnostics


def _attach_metadata(
    jobs: List[Dict[str, Any]],
    parts_lookup: Dict[str, Dict[str, Any]],
    topcoat_lookup: Dict[str, Dict[str, Any]],
    parts_key_col: str,
    topcoat_key_col: str,
    fail_policy: str,
) -> Dict[str, Any]:
    """Attach part/topcoat metadata to each job in-place."""
    missing_parts = 0
    missing_topcoats = 0

    for job in jobs:
        part_key = str(job.get("part_id", ""))
        topcoat_key = str(job.get("topcoat_code", job.get("color_code", "")))

        part_row = parts_lookup.get(part_key)
        if part_row is None:
            missing_parts += 1
        else:
            for k, v in part_row.items():
                if k == parts_key_col:
                    continue
                job[f"part_{k}"] = v

        topcoat_row = topcoat_lookup.get(topcoat_key)
        if topcoat_row is None:
            missing_topcoats += 1
        else:
            for k, v in topcoat_row.items():
                if k == topcoat_key_col:
                    continue
                job[f"topcoat_{k}"] = v

    diagnostics = {
        "missing_parts": missing_parts,
        "missing_topcoats": missing_topcoats,
    }

    if fail_policy == "hard" and (missing_parts or missing_topcoats):
        raise RuntimeError(
            f"Missing joins in Step-01 with hard policy: parts={missing_parts}, topcoats={missing_topcoats}"
        )

    return diagnostics


def run_step01(global_cfg_path: Path, steps_cfg_path: Path) -> None:
    """Run Step-01: expand aggregate demands into canonical job rows.

    This function:
    - loads configuration,
    - expands demands.csv into one row per physical job,
    - joins part and topcoat metadata,
    - writes step01_jobs_expanded.csv,
    - writes a manifest (JSON) and a Markdown report.
    """
    global_cfg = load_global_config(global_cfg_path)
    steps_cfg = load_steps_config(steps_cfg_path)
    cfg = _load_step01_config(global_cfg, steps_cfg)

    dataset_dir = cfg.dataset_dir
    output_dir = cfg.output_dir
    manifests_dir = cfg.manifests_dir
    reports_dir = cfg.reports_dir

    demands_path = dataset_dir / cfg.demands_filename
    parts_path = dataset_dir / cfg.parts_filename
    topcoat_path = dataset_dir / cfg.topcoat_codebook_filename

    jobs_path = output_dir / cfg.jobs_expanded_relpath
    manifest_path = output_dir / cfg.manifest_relpath
    report_path = output_dir / cfg.report_relpath

    # Detect actual column names using synonyms from global config
    part_col, topcoat_col, qty_col = _detect_demands_columns(demands_path, global_cfg)
    parts_key = _detect_parts_key(parts_path, global_cfg)
    topcoat_key = _detect_topcoat_key(topcoat_path, global_cfg)

    jobs, diag_expand = _expand_demands(demands_path, part_col, topcoat_col, qty_col)

    parts_lookup = _read_lookup_table(parts_path, parts_key)
    topcoat_lookup = _read_lookup_table(topcoat_path, topcoat_key)

    diag_join = _attach_metadata(jobs, parts_lookup, topcoat_lookup, parts_key, topcoat_key, cfg.fail_on_missing_join)

    # Determine CSV columns
    fieldnames = list(jobs[0].keys()) if jobs else []

    write_csv(jobs_path, fieldnames, jobs)

    # Build manifest with IO hashes and diagnostics
    manifest: Dict[str, Any] = {
        "step": "step01",
        "purpose": "Expand aggregate demands into canonical job rows with attached metadata.",
        "what": "Step-01 materializes the job universe from aggregate demands and enriches with part/topcoat attributes.",
        "why": "Provide a stable job-level basis for all downstream scheduling and optimization steps.",
        "config": {
            "global_config_path": str(global_cfg_path),
            "steps_config_path": str(steps_cfg_path),
            "fail_on_missing_join": cfg.fail_on_missing_join,
        },
        "io_hashes": {
            "inputs": {
                "demands": {
                    "path": str(demands_path),
                    "sha256": sha256_of_file(demands_path) if demands_path.exists() else None,
                },
                "parts": {
                    "path": str(parts_path),
                    "sha256": sha256_of_file(parts_path) if parts_path.exists() else None,
                },
                "topcoat_codebook": {
                    "path": str(topcoat_path),
                    "sha256": sha256_of_file(topcoat_path) if topcoat_path.exists() else None,
                },
            },
            "outputs": {
                "jobs_expanded": {
                    "path": str(jobs_path),
                    "sha256": sha256_of_file(jobs_path) if jobs_path.exists() else None,
                },
            },
        },
        "settings": {
            "dataset_dir": str(dataset_dir),
            "output_dir": str(output_dir),
            "manifests_dir": str(manifests_dir),
            "reports_dir": str(reports_dir),
        },
        "diagnostics": {
            "expand": diag_expand,
            "joins": diag_join,
        },
    }

    write_json(manifest_path, manifest)

    # Build Markdown report with required sections
    lines: List[str] = []
    lines.append("# STEP-01 Report – Job Expansion")
    lines.append("")
    lines.append("## Purpose")
    lines.append("Expand aggregate demands into a canonical job-level table and attach part/topcoat metadata.")
    lines.append("")
    lines.append("## What")
    lines.append(
        "Reads demands, parts, and topcoat codebook tables from the dataset; expands each "
        "demand row into one row per physical job; assigns stable job_ids and provenance; "
        "joins part/topcoat attributes; writes the canonical job table."
    )
    lines.append("")
    lines.append("## Why")
    lines.append(
        "All downstream scheduling (Toyota grouping/spacing, SDSU, MILP, NSGA-II, DES) must "
        "operate on a single, consistent job universe. Step-01 isolates this concern and "
        "removes any dependence on aggregate demand representation."
    )
    lines.append("")
    lines.append("## Config")
    lines.append(f"- Global config: `{global_cfg_path}`")
    lines.append(f"- Steps config: `{steps_cfg_path}`")
    lines.append(f"- Dataset dir: `{dataset_dir}`")
    lines.append(f"- Output dir: `{output_dir}`")
    lines.append(f"- fail_on_missing_join: `{cfg.fail_on_missing_join}`")
    lines.append("")
    lines.append("## I/O hashes")
    lines.append(f"- demands: `{sha256_of_file(demands_path) if demands_path.exists() else 'MISSING'}`")
    lines.append(f"- parts: `{sha256_of_file(parts_path) if parts_path.exists() else 'MISSING'}`")
    lines.append(f"- topcoat_codebook: `{sha256_of_file(topcoat_path) if topcoat_path.exists() else 'MISSING'}`")
    lines.append(f"- jobs_expanded: `{sha256_of_file(jobs_path) if jobs_path.exists() else 'MISSING'}`")
    lines.append("")
    lines.append("## Settings")
    lines.append(
        f"- time_unit (from global_config): {global_cfg.get('system', {}).get('time_unit', 'paint_unit')}"
    )
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Demand rows processed: {diag_expand['num_demand_rows']}")
    lines.append(f"- Jobs generated: {diag_expand['num_jobs']}")
    lines.append(f"- Zero-quantity rows skipped: {diag_expand['zero_quantity_rows']}")
    lines.append(f"- Negative-quantity rows skipped: {diag_expand['negative_quantity_rows']}")
    lines.append("")
    lines.append("## Diagnostics")
    lines.append(f"- Missing parts in join: {diag_join['missing_parts']} (policy: {cfg.fail_on_missing_join})")
    lines.append(f"- Missing topcoats in join: {diag_join['missing_topcoats']} (policy: {cfg.fail_on_missing_join})")
    lines.append("")
    lines.append("Topcoat histogram (quantity per topcoat_code):")
    for code, qty in sorted(diag_expand["topcoat_histogram"].items()):
        lines.append(f"- {code}: {qty}")
    lines.append("")
    report_text = "\n".join(lines)

    write_text(report_path, report_text)


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Run STEP-01 job expansion.")
    parser.add_argument(
        "--global-config",
        type=Path,
        default=Path("global_config.toml"),
        help="Path to global_config.toml",
    )
    parser.add_argument(
        "--steps-config",
        type=Path,
        default=Path("steps_config.toml"),
        help="Path to steps_config.toml",
    )
    args = parser.parse_args()
    run_step01(args.global_config, args.steps_config)
