from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple, Iterable, Set


def read_csv_rows(path: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Read a CSV file into a list of row dicts and return (rows, fieldnames)."""
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = [dict(r) for r in reader]
    return rows, fieldnames


def build_topcoat_code_maps(codebook_path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Build mappings between topcoat_no and topcoat_code.

    Returns:
        no_to_code, code_to_no
    """
    with codebook_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        no_to_code: Dict[str, str] = {}
        code_to_no: Dict[str, str] = {}
        for row in reader:
            no = str(row.get("topcoat_no", "")).strip()
            code = str(row.get("topcoat_code", "")).strip()
            if not no or not code:
                continue
            no_to_code[no] = code
            code_to_no[code] = no
    return no_to_code, code_to_no


def load_s_matrix(path: Path) -> Tuple[Dict[Tuple[str, str], float], float]:
    """Load s_matrix_transition.csv into a (from,to)->cost mapping and return a typical cost.

    'inf' (case-insensitive) or empty entries are treated as forbidden (math.inf).
    The typical cost is the mean of all finite, strictly positive costs, or 1.0 if none exist.
    """
    cost: Dict[Tuple[str, str], float] = {}
    finite_vals: List[float] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        if "from" not in cols:
            raise KeyError(f"Expected 'from' column in {path}")
        to_colors = [c for c in cols if c != "from"]
        for row in reader:
            from_c = str(row["from"]).strip()
            if not from_c:
                continue
            for to_c in to_colors:
                raw = str(row.get(to_c, "")).strip()
                if not raw:
                    val = math.inf
                else:
                    lowered = raw.lower()
                    if lowered in {"inf", "infinity", "nan"}:
                        val = math.inf
                    else:
                        try:
                            val = float(raw)
                        except ValueError:
                            val = math.inf
                cost[(from_c, to_c)] = val
                if math.isfinite(val) and val > 0.0:
                    finite_vals.append(val)
    if finite_vals:
        typical = sum(finite_vals) / float(len(finite_vals))
    else:
        typical = 1.0
    return cost, typical


def load_topcoat_forbids(
    rules_path: Path,
    no_to_code: Dict[str, str],
) -> Set[Tuple[str, str]]:
    """Load explicit paint forbids from topcoat_rules.csv, mapped to (from_code,to_code)."""
    forbids: Set[Tuple[str, str]] = set()
    with rules_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel = str(row.get("relation", "")).strip().lower()
            if "forbid" not in rel:
                continue
            from_no = str(row.get("from_topcoat_no", "")).strip()
            to_no = str(row.get("to_topcoat_no", "")).strip()
            if not from_no or not to_no:
                continue
            from_code = no_to_code.get(from_no)
            to_code = no_to_code.get(to_no)
            if from_code and to_code:
                forbids.add((from_code, to_code))
    return forbids


def load_adjacency_forbids(adj_path: Path) -> Set[Tuple[str, str]]:
    """Load part-level adjacency forbids from adjacency_rules.csv as (from_part_id,to_part_id)."""
    forbids: Set[Tuple[str, str]] = set()
    with adj_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel = str(row.get("relation", "")).strip().lower()
            if "forbid" not in rel:
                continue
            from_pid = str(row.get("from_part_id", "")).strip()
            to_pid = str(row.get("to_part_id", "")).strip()
            if from_pid and to_pid:
                forbids.add((from_pid, to_pid))
    return forbids
