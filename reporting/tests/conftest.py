import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest

from reporting import synth
from reporting.replay import build_variant


def mk_trades(rows, run_id="r1"):
    """rows: dicts with entry, exit, coll, pnl and optional ticker/strategy/expiry/exit_reason/credit/cf/..."""
    out = []
    for i, r in enumerate(rows):
        out.append({
            "run_id": run_id, "trade_id": r.get("id", f"T{i:03d}"), "uid": f"{run_id}:" + r.get("id", f"T{i:03d}"),
            "ticker": r.get("ticker", "SPY"), "strategy": r.get("strategy", "iron_condor"), "level": 1, "profile": r.get("profile", "green"),
            "config_tag": r.get("tag", "baseline"), "entry_ts": pd.Timestamp(r["entry"] + " 10:00"), "exit_ts": pd.Timestamp(r["exit"] + " 15:00"),
            "expiry": pd.Timestamp(r.get("expiry", r["exit"])), "dte_entry": 30, "exit_reason": r.get("reason", "expiry"), "width": 5,
            "collateral_per_contract": float(r["coll"]), "credit_per_contract": r.get("credit", 100.0), "pnl_per_contract": float(r["pnl"]),
            "fees_per_contract": r.get("fees", 0.0), "adj_count": r.get("adj", 0), "adj_cost_per_contract": 0.0, "hedge_cost_per_contract": 0.0,
            "cf_pnl_to_expiry_per_contract": r.get("cf", r["pnl"]), "entry_spot": 100.0, "exit_spot": 100.0, "entry_vix": 15.0,
            "vix_bucket": r.get("vix", "15to20"), "vix_term_ratio": 0.9, "short_put_delta": r.get("delta", 0.16), "short_call_delta": 0.16,
            "exdiv_flag": 0, "guards_on": "", "fill_model": "natural", "spot_source": "test",
        })
    return pd.DataFrame(out)


def mk_marks(trades, marks):
    """marks: {trade_id: {date: value}}"""
    rows = []
    for tid, d in marks.items():
        for dt_, v in d.items():
            rows.append({"trade_id": tid, "uid": f"r1:{tid}", "date": pd.Timestamp(dt_), "mark_pnl_per_contract": v})
    return pd.DataFrame(rows, columns=["trade_id", "uid", "date", "mark_pnl_per_contract"])


def vd_from(rows, marks=None):
    t = mk_trades(rows)
    m = mk_marks(t, marks or {})
    return build_variant(t, m)


@pytest.fixture(scope="session")
def synth_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("synth") / "runs_synth"
    keep = {"baseline", "mgmt_roll_breach", "wait1d", "wait10d", "nostop", "alloff", "delta25", "delta10", "stress"}
    vs = [v for v in synth.default_variants() if v.config_tag in keep and v.profile != "stress_inflation2022"]
    synth.generate(out, seed=5, start="2018-01-02", end="2020-12-30", cadence=10, variants=vs)
    return out


@pytest.fixture(scope="session")
def report_dir(synth_dir, tmp_path_factory):
    from reporting.pipeline import ReportOptions, run_report
    out = tmp_path_factory.mktemp("rep") / "report_synth"
    run_report(str(synth_dir), str(out), ReportOptions(paths=300, jobs=1, allocations=[5, 15, 50, 100]), log=lambda *_: None)
    return out
