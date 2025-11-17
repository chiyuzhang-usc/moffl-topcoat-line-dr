#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import math
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

import pandas as pd

from utilities.config_utils import load_global_config, load_steps_config, resolve_paths
from utilities.io_utils import ensure_parent_dir, sha256_of_file


# -------------------------------
# Config & data loading
# -------------------------------

@dataclass
class TokenCosts:
    flush_unit: float = 1.0
    empty_unit: float = 1.0


@dataclass
class PathsCfg:
    jobs_expanded_csv: Path
    s_matrix_csv: Optional[Path]
    s_lookup_csv: Optional[Path]
    adjacency_csv: Optional[Path]
    # Outputs under output_dir
    output_stream_csv: Path
    manifest_json: Path
    report_md: Path
    # Refs
    global_config: Path
    steps_config: Path
    empirical_config: Path
    output_dir: Path


def resolve_paths_step02(
    global_cfg: dict,
    steps_cfg: dict,
    global_cfg_path: Path,
    steps_cfg_path: Path,
    empirical_path: Path,
) -> PathsCfg:
    roots = resolve_paths(global_cfg)
    output_dir = roots["output_dir"]
    dataset_dir = roots["dataset_dir"]

    data_files = global_cfg.get("datasets", {}) or global_cfg.get("data_files", {}) or {}

    def _dataset_path(name: str, default: Optional[str]) -> Optional[Path]:
        rel = data_files.get(name)
        candidate = rel or default
        if not candidate:
            return None
        path = Path(candidate)
        if not path.is_absolute():
            path = dataset_dir / path
        if rel:
            return path
        return path if path.exists() else None

    s_matrix_csv = _dataset_path("s_matrix_transition", "s_matrix_transition.csv")
    s_lookup_csv = _dataset_path("s_lookup_transition", "s_lookup_transition.csv")
    adjacency_csv = _dataset_path("adjacency_rules", "adjacency_rules.csv")

    step02 = steps_cfg.get("step02", {})

    def _output_path(key: str, default: str) -> Path:
        rel = Path(step02.get(key, default))
        return rel if rel.is_absolute() else output_dir / rel

    return PathsCfg(
        jobs_expanded_csv=_output_path("input_jobs_relpath", "step01_jobs_expanded.csv"),
        s_matrix_csv=s_matrix_csv,
        s_lookup_csv=s_lookup_csv,
        adjacency_csv=adjacency_csv,
        output_stream_csv=_output_path("paint_seed_relpath", "step02_paint_seed.csv"),
        manifest_json=_output_path("manifest_relpath", "manifests/step02_manifest.json"),
        report_md=_output_path("report_relpath", "reports/step02_report.md"),
        global_config=global_cfg_path,
        steps_config=steps_cfg_path,
        empirical_config=empirical_path,
        output_dir=output_dir,
    )


def load_s_costs(s_matrix_csv: Optional[Path], s_lookup_csv: Optional[Path]) -> Tuple[Dict[Tuple[str, str], float], Set[Tuple[str, str]]]:
    costs: Dict[Tuple[str, str], float] = {}
    forbids: Set[Tuple[str, str]] = set()

    if s_lookup_csv and s_lookup_csv.exists():
        df = pd.read_csv(s_lookup_csv)
        cols = {c.lower(): c for c in df.columns}
        frm = cols.get('from') or cols.get('prev') or cols.get('source')
        to  = cols.get('to')   or cols.get('next') or cols.get('target')
        costc = cols.get('cost') or cols.get('setup') or cols.get('units')
        allow = cols.get('allowed') or cols.get('allow')
        forbid = cols.get('forbid') or cols.get('forbidden')
        if frm and to and costc:
            for _, r in df.iterrows():
                f = str(r[frm]).strip()
                t = str(r[to]).strip()
                try: c = float(r[costc])
                except Exception: c = 1.0
                costs[(f, t)] = c
                if forbid is not None:
                    if str(r[forbid]).strip() not in ("", "0", "False", "false", "no"):
                        forbids.add((f, t))
                if allow is not None and str(r[allow]).strip() in ("0","False","false","no"):
                    forbids.add((f, t))
            return costs, forbids

    if s_matrix_csv and s_matrix_csv.exists():
        df = pd.read_csv(s_matrix_csv, index_col=0)
        for f in df.index.astype(str):
            row = df.loc[f]
            for t in df.columns.astype(str):
                try: costs[(f, t)] = float(row[t])
                except Exception: pass
        return costs, forbids

    return costs, forbids


def load_adjacency_forbids(adjacency_csv: Optional[Path]) -> Set[Tuple[str, str]]:
    forbids: Set[Tuple[str, str]] = set()
    if adjacency_csv and adjacency_csv.exists():
        df = pd.read_csv(adjacency_csv)
        cols = {c.lower(): c for c in df.columns}
        prevc = cols.get('from_part_id') or cols.get('prev_part') or cols.get('from') or cols.get('prev')
        nextc = cols.get('to_part_id')   or cols.get('next_part') or cols.get('to')   or cols.get('next')
        forbid = cols.get('forbid') or cols.get('forbidden')
        allow  = cols.get('allowed') or cols.get('allow')
        if prevc and nextc:
            for _, r in df.iterrows():
                p = str(r[prevc]).strip(); n = str(r[nextc]).strip()
                is_forbid = False
                if forbid is not None:
                    is_forbid = str(r[forbid]).strip() not in ("", "0", "False", "false", "no")
                elif allow is not None:
                    is_forbid = str(r[allow]).strip() in ("", "0", "False", "false", "no")
                else:
                    is_forbid = True
                if p and n and is_forbid:
                    forbids.add((p, n))
    return forbids


def resolve_schema(global_cfg: dict, df: pd.DataFrame) -> Tuple[str, str, Optional[str]]:
    """Return (topcoat_col, part_col, jobid_col). Uses global_config.data_schema.demands synonyms."""
    dem = global_cfg.get("data_schema", {}).get("demands", {})
    top_syn = dem.get("topcoat_code_synonyms", ["color_code","topcoat_code"])
    part_syn = dem.get("part_id_synonyms", ["part_id"])
    job_syn  = ["job_id"]

    def pick(cands: List[str], label: str) -> str:
        cols = {c.lower(): c for c in df.columns}
        for s in cands:
            if s.lower() in cols:
                return cols[s.lower()]
        # normalized
        norm = {"".join(ch for ch in c.lower() if ch.isalnum()): c for c in df.columns}
        for s in cands:
            k = "".join(ch for ch in s.lower() if ch.isalnum())
            if k in norm:
                return norm[k]
        raise KeyError(label)

    top = pick(top_syn + ["color_code","topcoat_code","topcoat","paint_color","coat_code","color"], "topcoat_code")
    part = pick(part_syn + ["part_id","part","part_code"], "part_id")
    jid = None
    try:
        jid = pick(job_syn + ["job","id","jobid"], "job_id")
    except KeyError:
        jid = None
    return top, part, jid


def load_cycle_quota(global_cfg: dict) -> Tuple[Optional[pd.DataFrame], Optional[int], Optional[Dict[str, float]]]:
    """Return (cq_df, cycle_size, target_mix_by_part) if available."""
    datasets = global_cfg.get("datasets", {}) or global_cfg.get("data_files", {})
    roots = resolve_paths(global_cfg)
    ds_dir = roots["dataset_dir"]
    cq_name = datasets.get("cycle_quota", "cycle_quota.csv")
    path = Path(cq_name)
    if not path.is_absolute():
        path = ds_dir / path
    if not path.exists():
        return None, None, None
    cq = pd.read_csv(path)
    schema = global_cfg.get("data_schema", {}).get("cycle_quota", {})
    cyc_col = next((c for c in schema.get("cycle_synonyms",["cycle"]) if c in cq.columns), None) or "cycle"
    pid_col = next((c for c in schema.get("part_id_synonyms",["part_id"]) if c in cq.columns), None) or "part_id"
    q_col   = next((c for c in schema.get("quota_synonyms",["quota"]) if c in cq.columns), None) or "quota"
    if not (cyc_col in cq.columns and pid_col in cq.columns and q_col in cq.columns):
        return None, None, None
    first = cq[cyc_col].min()
    cycle_size = int(cq[cq[cyc_col]==first][q_col].sum())
    share = cq.groupby(pid_col)[q_col].mean()
    ssum = float(share.sum()) or 1.0
    target_mix = {str(k): float(v/ssum) for k,v in share.items()}
    # stash column names for later use
    cq.attrs["cyc_col"] = cyc_col; cq.attrs["pid_col"] = pid_col; cq.attrs["q_col"] = q_col
    return cq, cycle_size, target_mix


# -------------------------------
# Toyota GS heuristic
# -------------------------------

@dataclass
class Candidate:
    color: str
    job_idx_in_queue: int
    use_empty: bool
    use_flush: bool
    score: float


class ToyotaGS:
    def __init__(
        self,
        jobs_df: pd.DataFrame,
        s_costs: Dict[Tuple[str, str], float],
        s_forbids: Set[Tuple[str, str]],
        adjacency_forbids: Set[Tuple[str, str]],
        window_w: int,
        run_caps: Dict[str, int],
        token_costs: TokenCosts,
        empty_weight: float,
        flush_weight: float,
        fifo_lookahead: int,
        # schema
        topcoat_col: str,
        part_col: str,
        jobid_col: Optional[str],
        # cycle quota (optional)
        cycle_quota: Optional[pd.DataFrame] = None,
        cycle_size: Optional[int] = None,
        target_mix_color: Optional[Dict[str, float]] = None,
    ) -> None:
        self.df = jobs_df
        self.s_costs = s_costs
        self.s_forbids = s_forbids
        self.adj_forbids = adjacency_forbids
        self.W = max(1, int(window_w))
        self.run_caps = run_caps
        self.token_costs = token_costs
        self.empty_weight = float(empty_weight)
        self.flush_weight = float(flush_weight)
        self.fifo_lookahead = int(max(1, fifo_lookahead))

        self.topcoat_col = topcoat_col
        self.part_col = part_col
        self.jobid_col = jobid_col

        # queues by color (stable)
        self.pools: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for _, r in self.df.iterrows():
            self.pools[str(r[self.topcoat_col])].append(r.to_dict())
        self.colors = sorted(self.pools.keys())
        self.remaining = sum(len(v) for v in self.pools.values())

        # state
        self.seq_rows: List[Dict[str, Any]] = []
        self.last_paint: Optional[str] = None  # neutral after FLUSH/START
        self.last_part: Optional[str] = None   # reset by EMPTY only
        self.cur_run_color: Optional[str] = None
        self.cur_run_len: int = 0
        self.max_run_by_color: Dict[str, int] = defaultdict(int)

        self.window_colors: Deque[str] = deque(maxlen=self.W)
        # target mix by COLOR; if provided by cycle_quota (part-based), we leave None unless a part->color map exists
        self.target_mix_color = target_mix_color or None

        # cycle quota
        self.cycle_quota = cycle_quota
        self.cycle_size = int(cycle_size) if cycle_size else None
        self.cycle_counts: Dict[Tuple[int,str], int] = defaultdict(int)
        self.quota_cols = None
        if self.cycle_quota is not None:
            self.quota_cols = (self.cycle_quota.attrs.get("cyc_col","cycle"),
                               self.cycle_quota.attrs.get("pid_col","part_id"),
                               self.cycle_quota.attrs.get("q_col","quota"))

        # metrics
        self.metrics = {
            "num_jobs": self.remaining,
            "window_W": self.W,
            "num_empty": 0,
            "num_flush": 0,
            "setup_units": 0.0,
            "quota_violations": 0,
        }

    # ---- helpers ----
    def _transition_cost(self, prev_color: Optional[str], next_color: str) -> Tuple[float, bool]:
        if prev_color is None:
            c = self.s_costs.get(("__NEUTRAL__", next_color), 0.0)
            return c, True
        if (prev_color, next_color) in self.s_forbids:
            return math.inf, False
        c = self.s_costs.get((prev_color, next_color))
        if c is None:
            return math.inf, False
        return c, True

    def _adjacent_forbidden(self, prev_part: Optional[str], next_part: str, has_spacer: bool) -> bool:
        if prev_part is None or has_spacer:
            return False
        return (prev_part, next_part) in self.adj_forbids

    def _window_mix_penalty(self, next_color: str) -> float:
        counts = defaultdict(int)
        for c in self.window_colors:
            counts[c] += 1
        counts[next_color] += 1
        total = len(self.window_colors) + 1
        if total <= 1:
            return 0.0
        if self.target_mix_color:
            keys = set(counts.keys()) | set(self.target_mix_color.keys())
            pen = 0.0
            for k in keys:
                p = counts.get(k, 0) / total
                q = self.target_mix_color.get(k, 0.0)
                pen += (p - q) ** 2
            return pen
        # flatness penalty
        k = len(counts)
        if k <= 1:
            return 0.0
        ideal = 1.0 / k
        return sum((counts[c]/total - ideal)**2 for c in counts)

    def _run_cap_penalty(self, color: str) -> float:
        projected = self.cur_run_len + 1 if self.cur_run_color == color else 1
        cap = self.run_caps.get(color, 10**9)
        return 1e6 if projected > cap else 0.0

    def _cycle_quota_block(self, color: str, idx_in_queue: int) -> bool:
        if self.cycle_size is None:
            return False
        # jobs placed so far
        jobs_placed = sum(1 for r in self.seq_rows if r.get("token_type") == "JOB")
        cyc = jobs_placed // self.cycle_size
        probe = self.pools[color][idx_in_queue]
        pid = str(probe[self.part_col])
        # find allowed quota for this cycle
        if self.cycle_quota is None:
            return False
        cyc_col, pid_col, q_col = self.quota_cols
        # pick rows for this cycle; fallback to first pattern if not found
        if (self.cycle_quota[cyc_col] == cyc).any():
            sub = self.cycle_quota[self.cycle_quota[cyc_col] == cyc]
        else:
            base_cyc = self.cycle_quota[cyc_col].min()
            sub = self.cycle_quota[self.cycle_quota[cyc_col] == base_cyc]
        allowed = int(sub[sub[pid_col] == pid][q_col].sum()) if (sub[pid_col] == pid).any() else None
        if allowed is None:
            return False
        used = self.cycle_counts.get((cyc, pid), 0)
        return used >= allowed

    # ---- candidate enumeration ----
    def _best_candidate(self) -> Optional[Candidate]:
        avail = [c for c,q in self.pools.items() if q]
        if not avail:
            return None

        candidates: List[Candidate] = []
        for color in avail:
            # find an adjacency-safe index within lookahead
            idx_ok = None
            q = self.pools[color]
            L = min(len(q), self.fifo_lookahead)
            for d in range(L):
                nxt_part = str(q[d][self.part_col])
                if not self._adjacent_forbidden(self.last_part, nxt_part, has_spacer=False):
                    idx_ok = d; break

            # paint feasibility
            cdir, feas_dir = self._transition_cost(self.last_paint, color)
            cneu, feas_neu = self._transition_cost(None, color)

            # run cap
            cap_pen = self._run_cap_penalty(color)

            # A) direct (only if adjacency ok and paint ok)
            if (idx_ok is not None) and feas_dir:
                if not self._cycle_quota_block(color, idx_ok):
                    score = cdir + cap_pen + self._window_mix_penalty(color)
                    candidates.append(Candidate(color, idx_ok, False, False, score))

            # B) FLUSH + place (no spacer; adjacency ok still required)
            if (idx_ok is not None) and feas_neu:
                if not self._cycle_quota_block(color, idx_ok):
                    score = cneu + self.flush_weight * self.token_costs.flush_unit + cap_pen + self._window_mix_penalty(color)
                    candidates.append(Candidate(color, idx_ok, False, True, score))

            # C) EMPTY + direct (only if adjacency blocked)
            if (idx_ok is None) and feas_dir:
                # after EMPTY, adjacency resets, so index must be 0
                idx0 = 0
                if not self._cycle_quota_block(color, idx0):
                    score = cdir + self.empty_weight * self.token_costs.empty_unit + cap_pen + self._window_mix_penalty(color)
                    candidates.append(Candidate(color, idx0, True, False, score))

            # D) EMPTY + FLUSH + place  (only if adjacency blocked)
            if (idx_ok is None) and feas_neu:
                idx0 = 0
                if not self._cycle_quota_block(color, idx0):
                    score = cneu + self.empty_weight * self.token_costs.empty_unit + self.flush_weight * self.token_costs.flush_unit + cap_pen + self._window_mix_penalty(color)
                    candidates.append(Candidate(color, idx0, True, True, score))

        if not candidates:
            return None
        candidates.sort(key=lambda c: (c.score, c.job_idx_in_queue))
        return candidates[0]

    # ---- main ----
    def run(self) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        token_idx = 1
        rows.append({"token_index": token_idx, "token_type": "START"}); token_idx += 1

        while self.remaining > 0:
            cand = self._best_candidate()
            if cand is None:
                # If only adjacency blocks everything, insert EMPTY; else hard deadlock
                any_color = any(self.pools[c] for c in self.colors)
                if any_color:
                    rows.append({"token_index": token_idx, "token_type": "EMPTY"}); token_idx += 1
                    self.metrics["num_empty"] += 1
                    self.last_part = None
                    continue
                raise RuntimeError("Deadlock: no feasible candidate under C1/C2/quotas.")

            # Optional EMPTY
            if cand.use_empty:
                rows.append({"token_index": token_idx, "token_type": "EMPTY"}); token_idx += 1
                self.metrics["num_empty"] += 1
                self.last_part = None

            # Optional FLUSH
            if cand.use_flush:
                rows.append({"token_index": token_idx, "token_type": "FLUSH"}); token_idx += 1
                self.metrics["num_flush"] += 1
                self.last_paint = None

            # Realize job
            job_rec = self.pools[cand.color].pop(cand.job_idx_in_queue)
            self.remaining -= 1

            # S-cost
            prev = self.last_paint if not cand.use_flush else None
            cc, feas = self._transition_cost(prev, cand.color)
            if not feas or math.isinf(cc):
                raise RuntimeError(f"Forbidden/unknown S transition: {prev} -> {cand.color}")
            self.metrics["setup_units"] += cc

            # Cycle quota accounting
            if self.cycle_size:
                jobs_so_far = sum(1 for r in rows if r.get("token_type") == "JOB")
                cyc = jobs_so_far // self.cycle_size
                pid = str(job_rec[self.part_col])
                key = (cyc, pid)
                # check allowed
                if self.cycle_quota is not None:
                    # compute allowed for this cycle
                    cyc_col, pid_col, q_col = self.quota_cols
                    sub = self.cycle_quota
                    if (sub[cyc_col] == cyc).any():
                        sub = sub[sub[cyc_col] == cyc]
                    else:
                        sub = sub[sub[cyc_col] == sub[cyc_col].min()]
                    allowed = int(sub[sub[pid_col] == pid][q_col].sum()) if (sub[pid_col] == pid).any() else None
                    if allowed is not None and self.cycle_counts.get(key, 0) >= allowed:
                        self.metrics["quota_violations"] += 1
                self.cycle_counts[key] += 1

            # Emit JOB
            row = {"token_index": token_idx, "token_type": "JOB"}
            for k, v in job_rec.items():
                row[k] = v
            row["setup_cost"] = float(cc)
            row["cumulative_setup_cost"] = float(self.metrics["setup_units"])
            rows.append(row); token_idx += 1

            # Update state
            self.last_paint = cand.color
            self.last_part = str(job_rec[self.part_col])
            if self.cur_run_color == cand.color:
                self.cur_run_len += 1
            else:
                self.cur_run_color = cand.color
                self.cur_run_len = 1
            self.max_run_by_color[cand.color] = max(self.max_run_by_color[cand.color], self.cur_run_len)
            self.window_colors.append(cand.color)

        self.seq_rows = rows
        return pd.DataFrame(rows)


# -------------------------------
# Wiring
# -------------------------------

def run_step02(global_cfg_path: Path, steps_cfg_path: Path, empirical_cfg_path: Path) -> None:
    t0 = time.time()

    gcfg = load_global_config(global_cfg_path)
    scfg = load_steps_config(steps_cfg_path)
    ecfg = load_steps_config(empirical_cfg_path)

    paths = resolve_paths_step02(gcfg, scfg, global_cfg_path, steps_cfg_path, empirical_cfg_path)

    # Data
    jobs_df = pd.read_csv(paths.jobs_expanded_csv)
    s_costs, s_forbids = load_s_costs(paths.s_matrix_csv, paths.s_lookup_csv)
    adj_forbids = load_adjacency_forbids(paths.adjacency_csv)

    # Schema
    topcoat_col, part_col, jobid_col = resolve_schema(gcfg, jobs_df)

    # Parameters
    Wcfg = gcfg.get("window", {})
    mode = str(Wcfg.get("mode", "ratio")).lower()
    W = int(Wcfg.get("jobs", 0)) if (mode == "jobs" and int(Wcfg.get("jobs", 0)) > 0) else \
        max(1, int(math.ceil(float(Wcfg.get("ratio", 0.2)) * len(jobs_df))))

    Sp = gcfg.get("spacing", {})
    alpha = float(ecfg.get("defaults", {}).get("alpha_run_cap", Sp.get("alpha_run_cap", 0.5)))
    max_abs = int(ecfg.get("defaults", {}).get("max_run_abs", Sp.get("max_run_abs", 999999)))

    colors = sorted(jobs_df[topcoat_col].astype(str).unique().tolist())
    run_caps = {}
    for c in colors:
        per = max(1, int(math.floor(alpha * W)))
        per = min(per, max_abs if max_abs>0 else per)
        sect = ecfg.get(f"topcoat.{c}", {})
        mr = sect.get("max_run_jobs", None)
        if mr is not None:
            per = min(per, int(mr))
        run_caps[c] = max(1, per)

    Ttokens = gcfg.get("tokens", {})
    token_costs = TokenCosts(
        flush_unit=float(Ttokens.get("flush_units_ratio", 1.0)),
        empty_unit=float(Ttokens.get("empty_units_prepaint", 1.0)),
    )

    Emp = ecfg.get("defaults", {})
    empty_weight = float(Emp.get("empty_weight", token_costs.empty_unit))
    flush_weight = float(Emp.get("flush_weight", token_costs.flush_unit))

    adj_cfg = gcfg.get("adjacency", {})
    fifo_lookahead = int(adj_cfg.get("fifo_lookahead", 8))

    # Optional cycle quota if enabled at steps_config
    use_cycle_quota = bool(scfg.get("step02", {}).get("use_cycle_quota", False))
    cq_df, cycle_size, target_mix_part = (None, None, None)
    if use_cycle_quota:
        cq_df, cycle_size, target_mix_part = load_cycle_quota(gcfg)

    # If user provided a color-level target mix empirically, use it; else leave None
    target_mix_color = ecfg.get("target_mix", None)
    if target_mix_color:
        ssum = sum(max(0.0, float(v)) for v in target_mix_color.values()) or 1.0
        target_mix_color = {k: float(v)/ssum for k,v in target_mix_color.items()}

    # Build heuristic
    gs = ToyotaGS(
        jobs_df=jobs_df,
        s_costs=s_costs,
        s_forbids=s_forbids,
        adjacency_forbids=adj_forbids,
        window_w=W,
        run_caps=run_caps,
        token_costs=token_costs,
        empty_weight=empty_weight,
        flush_weight=flush_weight,
        fifo_lookahead=fifo_lookahead,
        topcoat_col=topcoat_col,
        part_col=part_col,
        jobid_col=jobid_col,
        cycle_quota=cq_df if use_cycle_quota else None,
        cycle_size=cycle_size if use_cycle_quota else None,
        target_mix_color=target_mix_color,
    )

    # Run
    stream_df = gs.run()

    # Outputs
    ensure_parent_dir(paths.output_stream_csv)
    stream_df.to_csv(paths.output_stream_csv, index=False)

    inputs = {
        "global_config.toml": sha256_of_file(paths.global_config) if paths.global_config.exists() else "",
        "steps_config.toml": sha256_of_file(paths.steps_config) if paths.steps_config.exists() else "",
        "step02_empirical.toml": sha256_of_file(paths.empirical_config) if paths.empirical_config.exists() else "",
        "step01_jobs_expanded.csv": sha256_of_file(paths.jobs_expanded_csv) if paths.jobs_expanded_csv.exists() else "",
        "s_matrix_transition.csv": sha256_of_file(paths.s_matrix_csv) if paths.s_matrix_csv and paths.s_matrix_csv.exists() else "",
        "s_lookup_transition.csv": sha256_of_file(paths.s_lookup_csv) if paths.s_lookup_csv and paths.s_lookup_csv.exists() else "",
        "adjacency_rules.csv": sha256_of_file(paths.adjacency_csv) if paths.adjacency_csv and paths.adjacency_csv.exists() else "",
    }
    outputs = {
        "step02_paint_seed.csv": sha256_of_file(paths.output_stream_csv) if paths.output_stream_csv.exists() else ""
    }

    settings = {
        "window_W": int(W),
        "alpha_run_cap": float(alpha),
        "max_run_abs": int(max_abs),
        "fifo_lookahead": fifo_lookahead,
        "token_costs": {"flush_unit": token_costs.flush_unit, "empty_unit": token_costs.empty_unit},
        "empty_weight": empty_weight,
        "flush_weight": flush_weight,
        "run_caps": run_caps,
        "colors": colors,
        "use_cycle_quota": use_cycle_quota,
        "cycle_size": int(cycle_size) if cycle_size else None,
        "schema": {"topcoat": topcoat_col, "part": part_col, "job_id": jobid_col},
        "output_dir": str(paths.output_dir),
    }

    token_counts = stream_df['token_type'].value_counts().to_dict()
    diagnostics = {
        "token_distribution": token_counts,
        "max_run_len_by_topcoat": gs.max_run_by_color,
        "quota_violations": gs.metrics.get("quota_violations", 0),
    }

    # Manifest
    ensure_parent_dir(paths.manifest_json)
    manifest = {
        "step": "step02",
        "purpose": "Toyota Grouping & Spacing on feeder",
        "what": "Construct C1/C2-feasible token stream with spacing/mix and optional cycle quotas.",
        "why": "Provide deterministic, data-driven seed for downstream steps.",
        "config_files": {
            "global_config.toml": str(paths.global_config),
            "steps_config.toml": str(paths.steps_config),
            "step02_empirical.toml": str(paths.empirical_config),
        },
        "io_hashes": {"inputs": inputs, "outputs": outputs},
        "settings_effective": settings,
        "metrics": gs.metrics,
        "diagnostics": diagnostics,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "version": "step02_gs_v2.0"
    }
    with paths.manifest_json.open('w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    # Report
    ensure_parent_dir(paths.report_md)
    lines = []
    lines.append("# STEP-02 Report — Toyota Grouping & Spacing (GS)")
    lines.append("")
    lines.append("## Purpose")
    lines.append("Construct a C1/C2-feasible single-file feeder stream using GS; insert EMPTY only for adjacency; optional cycle quotas.")
    lines.append("")
    lines.append("## Config (Effective)")
    lines.append("```json")
    lines.append(json.dumps(settings, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Metrics")
    for k in ["num_jobs","window_W","setup_units","num_flush","num_empty","quota_violations"]:
        lines.append(f"- {k}: {gs.metrics.get(k)}")
    lines.append("")
    lines.append("## Diagnostics")
    lines.append(json.dumps(diagnostics, indent=2))
    lines.append("")
    lines.append("## Runtime")
    lines.append(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    with paths.report_md.open('w', encoding='utf-8') as f:
        f.write("\n".join(lines))

    print(f"Step-02: W={W}, jobs={gs.metrics['num_jobs']}, out={paths.output_stream_csv}")

def main():
    g = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("global_config.toml")
    s = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("steps_config.toml")
    e = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("step02_empirical.toml")
    run_step02(g.resolve(), s.resolve(), e.resolve())

if __name__ == "__main__":
    main()
