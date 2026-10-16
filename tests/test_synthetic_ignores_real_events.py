"""A real data/events.csv (written by scripts/fetch_context_data.py) must not leak into synthetic-world runs."""
import argparse
import json

import pandas as pd

from playground.cli import build_parser, get_events
from playground.config import Config
from playground.provider import SyntheticProvider
from playground.sweep import run_sweep

NS = argparse.Namespace(provider="synthetic", data_dir="data", syn_start="2015-01-02", syn_end="2025-12-31", syn_seed=7, events="", limit=None)


def dense_events(path):
    """A calendar that blocks nearly every day of 2016 (FIXTURE dates, not real)."""
    days = pd.bdate_range("2016-01-04", "2016-12-30")
    pd.DataFrame({"date": days.date, "event": "CPI", "symbol": ""}).to_csv(path, index=False)


def test_get_events_uses_synthetic_calendar_for_synthetic_provider_even_if_file_exists(tmp_path):
    f = tmp_path / "events.csv"
    dense_events(f)
    args = build_parser().parse_args(["backtest", "--events", str(f), "--data-dir", str(tmp_path)])
    ev, note = get_events(args, Config(), SyntheticProvider())
    assert note.startswith("SYNTHETIC-calendar") and not (ev["event"].eq("CPI") & ev["date"].between(pd.Timestamp("2016-01-04").date(), pd.Timestamp("2016-12-30").date())).sum() > 200


def test_sweep_with_synthetic_provider_ignores_the_events_file(tmp_path):
    f = tmp_path / "events.csv"
    dense_events(f)
    spec = {"name": "g", "base": {"profile": "green", "resolution": "15m", "entry_cadence": "monthly", "start_date": "2016-01-04", "end_date": "2016-12-30"},
            "grid": {"ticker": ["GLD"]}}
    ns = argparse.Namespace(**{**vars(NS), "events": str(f)})
    res = run_sweep(spec, ns, tmp_path / "runs", log=lambda *a: None)
    rid = res["summary"].run_id.iloc[0]
    meta = json.loads((tmp_path / "runs" / rid / "meta.json").read_text())
    assert "SYNTHETIC-calendar" in json.dumps(meta) and res["summary"].n.iloc[0] > 0           # the only GLD entries are not blocked by real-calendar dates
