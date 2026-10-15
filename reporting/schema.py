"""Trade-log schema: loader and validator (contract with TASK A)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

TRADE_COLS = [
    "run_id", "trade_id", "ticker", "strategy", "level", "profile", "config_tag", "entry_ts", "exit_ts",
    "expiry", "dte_entry", "exit_reason", "width", "collateral_per_contract", "credit_per_contract",
    "pnl_per_contract", "fees_per_contract", "adj_count", "adj_cost_per_contract", "hedge_cost_per_contract",
    "cf_pnl_to_expiry_per_contract", "entry_spot", "exit_spot", "entry_vix", "vix_bucket", "vix_term_ratio",
    "short_put_delta", "short_call_delta", "exdiv_flag", "guards_on", "fill_model", "spot_source",
]
MARK_COLS = ["trade_id", "date", "mark_pnl_per_contract"]
BLOCKED_COLS = ["date", "ticker", "reason"]
EXIT_REASONS = {"tp", "stop", "stop_dollar", "breach", "trigger_exit", "time", "expiry", "other"}
VIX_BUCKETS = {"lt15", "15to20", "20to30", "gt30"}
BLOCK_REASONS = {"gate", "upside", "event", "cb", "term", "spike", "ivr", "nodata", "noquote", "mincredit", "cadence"}
NUMERIC = ["level", "dte_entry", "width", "collateral_per_contract", "credit_per_contract", "pnl_per_contract",
           "fees_per_contract", "adj_count", "adj_cost_per_contract", "hedge_cost_per_contract",
           "cf_pnl_to_expiry_per_contract", "entry_spot", "exit_spot", "entry_vix", "vix_term_ratio",
           "short_put_delta", "short_call_delta", "exdiv_flag"]


class SchemaError(Exception):
    pass


@dataclass
class Logs:
    trades: pd.DataFrame
    marks: pd.DataFrame
    blocked: pd.DataFrame
    configs: dict = field(default_factory=dict)   # run_id -> config.json dict
    synthetic: bool = False
    problems: list = field(default_factory=list)  # (severity, run, message)
    delta_sources: dict = field(default_factory=dict)   # run_id -> "theta_greeks" | "bs_fallback" | "mixed" | "unknown"

    def delta_source_summary(self) -> str:
        """One line for report headers: how many runs used which delta source."""
        if not self.delta_sources:
            return "unknown"
        n: dict = {}
        for v in self.delta_sources.values():
            n[v] = n.get(v, 0) + 1
        return ", ".join(f"{k} x{v}" for k, v in sorted(n.items()))


def read_delta_source(run_dir: Path) -> str:
    """meta.json -> delta_source_tag; derived from the older delta_source counts when the tag is absent."""
    try:
        meta = json.loads((Path(run_dir) / "meta.json").read_text())
    except Exception:  # noqa: BLE001
        return "unknown"
    tag = meta.get("delta_source_tag")
    if tag:
        return str(tag)
    c = meta.get("delta_source")
    if isinstance(c, dict):
        prov, bs_ = c.get("provider", 0), c.get("bs_mid_iv", 0)
        return "theta_greeks" if prov and not bs_ else ("mixed" if prov else "bs_fallback")
    return "unknown"


def _read_csv(path: Path, **kw) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, **kw)


def _ts(s: pd.Series) -> pd.Series:
    """ISO timestamps in US/Eastern -> naive wall-clock datetimes (offset text is dropped)."""
    return pd.to_datetime(s.str.slice(0, 19).str.replace("T", " "), errors="coerce")


def load_run(run_dir: Path):
    """Return (trades, marks, blocked, config, problems) for one run directory."""
    rid = run_dir.name
    problems: list = []
    errs = lambda m: problems.append(("error", rid, m))  # noqa: E731
    warn = lambda m: problems.append(("warn", rid, m))   # noqa: E731
    empty = pd.DataFrame()
    tp = run_dir / "trades.csv"
    if not tp.exists():
        errs("trades.csv missing")
        return empty, empty, empty, {}, problems
    t = _read_csv(tp)
    missing = [c for c in TRADE_COLS if c not in t.columns]
    if missing:
        errs(f"trades.csv missing columns: {missing}")
        return empty, empty, empty, {}, problems
    t = t.copy()
    for c in NUMERIC:
        t[c] = pd.to_numeric(t[c].replace("", np.nan), errors="coerce")
    t["entry_ts"] = _ts(t["entry_ts"])
    t["exit_ts"] = _ts(t["exit_ts"])
    t["expiry"] = pd.to_datetime(t["expiry"], errors="coerce")
    t["run_id"] = rid
    t["uid"] = rid + ":" + t["trade_id"]

    mp = run_dir / "marks.csv"
    if mp.exists():
        m = _read_csv(mp)
        mm = [c for c in MARK_COLS if c not in m.columns]
        if mm:
            errs(f"marks.csv missing columns: {mm}")
            m = pd.DataFrame(columns=MARK_COLS)
        m["date"] = pd.to_datetime(m["date"], errors="coerce")
        m["mark_pnl_per_contract"] = pd.to_numeric(m["mark_pnl_per_contract"], errors="coerce")
        m["uid"] = rid + ":" + m["trade_id"]
    else:
        warn("marks.csv missing (drawdowns will not include open-position losses)")
        m = pd.DataFrame(columns=MARK_COLS + ["uid"])
    bp = run_dir / "blocked_entries.csv"
    if bp.exists():
        b = _read_csv(bp)
        bm = [c for c in BLOCKED_COLS if c not in b.columns]
        if bm:
            errs(f"blocked_entries.csv missing columns: {bm}")
            b = pd.DataFrame(columns=BLOCKED_COLS)
        b["date"] = pd.to_datetime(b["date"], errors="coerce")
        b["run_id"] = rid
    else:
        warn("blocked_entries.csv missing")
        b = pd.DataFrame(columns=BLOCKED_COLS + ["run_id"])
    cfg = {}
    cp = run_dir / "config.json"
    if cp.exists():
        try:
            cfg = json.loads(cp.read_text())
        except Exception as e:  # noqa: BLE001
            errs(f"config.json unreadable: {e}")
    else:
        warn("config.json missing")
    problems += validate_frames(rid, t, m, b)
    return t, m, b, cfg, problems


def validate_frames(rid: str, t: pd.DataFrame, m: pd.DataFrame, b: pd.DataFrame) -> list:
    out: list = []
    E = lambda msg: out.append(("error", rid, msg))  # noqa: E731
    W = lambda msg: out.append(("warn", rid, msg))   # noqa: E731
    if t.empty:
        W("trades.csv has no rows")
        return out

    def bad(mask, what, sev=E, extra=None):
        n = int(mask.sum())
        if n:
            ids = ", ".join(t.loc[mask, "trade_id"].astype(str).head(5))
            sev(f"{n} trade(s) {what} (e.g. trade_id {ids})")

    for c in ("entry_ts", "exit_ts", "expiry"):
        bad(t[c].isna(), f"with unparseable {c}")
    for c in ("collateral_per_contract", "credit_per_contract", "pnl_per_contract", "fees_per_contract", "level"):
        bad(t[c].isna(), f"with non-numeric/blank {c}")
    bad(t["collateral_per_contract"] <= 0, "with non-positive collateral_per_contract")
    bad(t["credit_per_contract"] < 0, "with negative credit_per_contract", W)
    bad(t["exit_ts"] < t["entry_ts"], "with exit before entry")
    bad(~t["level"].isin([1, 2, 3, 4]), "with level outside 1-4")
    bad(~t["exit_reason"].isin(EXIT_REASONS), f"with exit_reason outside {sorted(EXIT_REASONS)}")
    bad(~t["vix_bucket"].isin(VIX_BUCKETS), f"with vix_bucket outside {sorted(VIX_BUCKETS)}")
    bad(t["expiry"] < t["entry_ts"].dt.normalize(), "with expiry before entry")
    bad(t["pnl_per_contract"] < -t["collateral_per_contract"] - 1e-6, "losing more than collateral (check fees/hedges)", W)
    dup = t["trade_id"].duplicated(keep=False)
    bad(dup, "with duplicate trade_id")
    if not m.empty:
        known = set(t["uid"])
        unk = ~m["uid"].isin(known)
        if unk.any():
            E(f"{int(unk.sum())} mark row(s) reference unknown trade_id (e.g. {m.loc[unk, 'trade_id'].iloc[0]})")
        bad_dates = m["date"].isna() | m["mark_pnl_per_contract"].isna()
        if bad_dates.any():
            E(f"{int(bad_dates.sum())} mark row(s) with unparseable date/value")
        j = m[~unk & ~bad_dates].merge(t[["uid", "entry_ts", "exit_ts"]], on="uid")
        out_life = (j["date"] < j["entry_ts"].dt.normalize()) | (j["date"] > j["exit_ts"].dt.normalize())
        if out_life.any():
            r = j[out_life].iloc[0]
            E(f"{int(out_life.sum())} mark row(s) outside trade life (e.g. trade_id {r['uid'].split(':', 1)[1]} on {r['date'].date()})")
    if not b.empty:
        nb = ~b["reason"].isin(BLOCK_REASONS)
        if nb.any():
            W(f"{int(nb.sum())} blocked_entries row(s) with unknown reason")
        if b["date"].isna().any():
            E("blocked_entries has unparseable dates")
    return out


def load_runs(root: str | Path) -> Logs:
    root = Path(root)
    if root.exists() and (root / "trades.csv").exists():      # a single run folder
        dirs = [root.resolve()]
    else:                                                      # a parent folder of runs
        dirs = sorted(p for p in root.iterdir() if p.is_dir() and (p / "trades.csv").exists()) if root.exists() else []
    problems: list = []
    if not root.exists():
        problems.append(("error", str(root), "runs directory does not exist"))
    elif not dirs:
        problems.append(("error", str(root), "no trades.csv here or in any subfolder (expected a run folder or a folder of runs)"))
    ts, ms, bs, cfgs, dsrc = [], [], [], {}, {}
    for d in dirs:
        t, m, b, cfg, pr = load_run(d)
        problems += pr
        cfgs[d.name] = cfg
        dsrc[d.name] = read_delta_source(d)
        if not t.empty:
            ts.append(t)
            ms.append(m)
            bs.append(b)
    trades = pd.concat(ts, ignore_index=True) if ts else pd.DataFrame(columns=TRADE_COLS + ["uid"])
    marks = pd.concat(ms, ignore_index=True) if ms else pd.DataFrame(columns=MARK_COLS + ["uid"])
    blocked = pd.concat(bs, ignore_index=True) if bs else pd.DataFrame(columns=BLOCKED_COLS + ["run_id"])
    synth = bool(cfgs) and any(bool(c.get("synthetic")) for c in cfgs.values())
    return Logs(trades, marks, blocked, cfgs, synth, problems, dsrc)


def validate_cli(root: str) -> int:
    lg = load_runs(root)
    errs = [p for p in lg.problems if p[0] == "error"]
    for sev, run, msg in lg.problems:
        print(f"[{sev.upper()}] {run}: {msg}")
    print(f"delta source: {lg.delta_source_summary()}")
    print(f"{len(lg.trades)} trades, {len(lg.marks)} marks, {len(lg.blocked)} blocked rows; "
          f"{len(errs)} error(s), {len(lg.problems) - len(errs)} warning(s)"
          + ("  [SYNTHETIC]" if lg.synthetic else ""))
    return 1 if errs else 0


def require_valid(lg: Logs) -> None:
    errs = [p for p in lg.problems if p[0] == "error"]
    if errs:
        raise SchemaError("; ".join(f"{r}: {m}" for _, r, m in errs[:10]))
