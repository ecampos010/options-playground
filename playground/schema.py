"""Trade-log schema (CONTRACT with the reporting/portfolio layer). Keep in sync with docs/TRADE_LOG_SCHEMA.md."""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

TRADE_COLUMNS = [
    "run_id", "trade_id", "ticker", "strategy", "level", "profile", "config_tag", "entry_ts", "exit_ts", "expiry",
    "dte_entry", "exit_reason", "width", "collateral_per_contract", "credit_per_contract", "pnl_per_contract",
    "fees_per_contract", "adj_count", "adj_cost_per_contract", "hedge_cost_per_contract",
    "cf_pnl_to_expiry_per_contract", "entry_spot", "exit_spot", "entry_vix", "vix_bucket", "vix_term_ratio",
    "short_put_delta", "short_call_delta", "exdiv_flag", "guards_on", "fill_model", "spot_source"]
MARK_COLUMNS = ["trade_id", "date", "mark_pnl_per_contract"]
BLOCKED_COLUMNS = ["date", "ticker", "reason"]
EXIT_REASONS = {"tp", "stop", "stop_dollar", "breach", "trigger_exit", "time", "expiry", "other"}
BLOCK_REASONS = {"gate", "upside", "event", "cb", "term", "spike", "ivr", "nodata", "noquote", "mincredit", "cadence",
                 "deltacap", "size"}     # last two are extensions documented in docs/TRADE_LOG_SCHEMA.md
VIX_BUCKETS = {"lt15", "15to20", "20to30", "gt30"}
BLANK_OK = {"cf_pnl_to_expiry_per_contract", "short_put_delta", "short_call_delta", "vix_term_ratio"}
REQUIRED_META = ["counts", "versions", "spot_coverage"]


def vix_bucket(v: float) -> str:
    return "lt15" if v < 15 else "15to20" if v < 20 else "20to30" if v < 30 else "gt30"


def _is_blank(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x)) or (isinstance(x, str) and x == "")


def validate_trades(df: pd.DataFrame) -> list[str]:
    """Return a list of problems (empty == conforming)."""
    errs: list[str] = []
    if list(df.columns) != TRADE_COLUMNS:
        missing = [c for c in TRADE_COLUMNS if c not in df.columns]
        extra = [c for c in df.columns if c not in TRADE_COLUMNS]
        errs.append(f"columns differ from contract: missing={missing} extra={extra} (order matters too)")
        if missing:
            return errs
    if df.empty:
        return errs
    if df["trade_id"].duplicated().any():
        errs.append("trade_id not unique")
    for c in TRADE_COLUMNS:
        if c not in BLANK_OK and df[c].map(_is_blank).any():
            errs.append(f"blank values in required column {c}")
    bad = set(df["exit_reason"].dropna()) - EXIT_REASONS
    if bad:
        errs.append(f"unknown exit_reason {sorted(bad)}")
    if set(df["level"].dropna().astype(int)) - {1, 2, 3, 4}:
        errs.append("level must be 1-4")
    if set(df["vix_bucket"].dropna()) - VIX_BUCKETS:
        errs.append("bad vix_bucket")
    if set(df["exdiv_flag"].dropna().astype(int)) - {0, 1}:
        errs.append("exdiv_flag must be 0/1")
    for c in ("collateral_per_contract", "credit_per_contract", "pnl_per_contract", "fees_per_contract",
              "adj_cost_per_contract", "hedge_cost_per_contract", "width", "entry_spot", "exit_spot", "entry_vix"):
        if not pd.api.types.is_numeric_dtype(df[c]):
            errs.append(f"{c} not numeric")
    if (df["collateral_per_contract"] <= 0).any():
        errs.append("collateral_per_contract must be > 0")
    if (df["fees_per_contract"] < 0).any():
        errs.append("fees_per_contract must be >= 0")
    if (df["adj_count"] < 0).any():
        errs.append("adj_count must be >= 0")
    try:
        et, xt = pd.to_datetime(df["entry_ts"]), pd.to_datetime(df["exit_ts"])
        if (xt < et).any():
            errs.append("exit_ts before entry_ts")
        ex = pd.to_datetime(df["expiry"])
        if (ex.dt.normalize() < et.dt.normalize()).any():
            errs.append("expiry before entry")
    except Exception as e:
        errs.append(f"timestamp parse error: {e}")
    # pnl cannot lose more than the exact max loss (+ fees/hedge slack for adjustments)
    slack = df["fees_per_contract"] + df["adj_cost_per_contract"].clip(lower=0) + df["hedge_cost_per_contract"].abs() + 1e-6
    unadj = df["adj_count"] == 0
    if ((-df["pnl_per_contract"] > df["collateral_per_contract"] + slack) & unadj & (df["hedge_cost_per_contract"] == 0)).any():
        errs.append("pnl_per_contract loses more than collateral_per_contract on an unadjusted trade")
    return errs


def validate_marks(m: pd.DataFrame, trades: pd.DataFrame | None = None) -> list[str]:
    errs = []
    if list(m.columns) != MARK_COLUMNS:
        return [f"marks columns {list(m.columns)} != {MARK_COLUMNS}"]
    if m.empty:
        return errs
    if m["mark_pnl_per_contract"].isna().any():
        errs.append("marks contain blanks")
    if m.duplicated(["trade_id", "date"]).any():
        errs.append("duplicate (trade_id, date) in marks")
    if trades is not None and set(m["trade_id"]) - set(trades["trade_id"]):
        errs.append("marks reference unknown trade_id")
    return errs


def validate_blocked(b: pd.DataFrame) -> list[str]:
    if list(b.columns) != BLOCKED_COLUMNS:
        return [f"blocked columns {list(b.columns)} != {BLOCKED_COLUMNS}"]
    bad = set(b["reason"]) - BLOCK_REASONS
    return [f"unknown block reason {sorted(bad)}"] if bad else []


def validate_run_dir(run_dir: str | Path) -> list[str]:
    """Validate runs/<run_id>/ as a whole. Returns problems (empty == conforming)."""
    import json
    d = Path(run_dir)
    errs: list[str] = []
    for f in ("trades.csv", "marks.csv", "blocked_entries.csv", "config.json", "meta.json"):
        if not (d / f).exists():
            errs.append(f"missing {f}")
    if errs:
        return errs
    t = pd.read_csv(d / "trades.csv")
    errs += validate_trades(t)
    errs += validate_marks(pd.read_csv(d / "marks.csv"), t)
    errs += validate_blocked(pd.read_csv(d / "blocked_entries.csv"))
    cfg, meta = json.loads((d / "config.json").read_text()), json.loads((d / "meta.json").read_text())
    for k in ("git_commit", "flags", "data_range"):
        if k not in cfg:
            errs.append(f"config.json missing {k}")
    for k in REQUIRED_META:
        if k not in meta:
            errs.append(f"meta.json missing {k}")
    if len(t) and set(t["run_id"]) != {d.name}:
        errs.append("run_id column does not match directory name")
    return errs
