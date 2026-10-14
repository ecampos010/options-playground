import numpy as np
import pandas as pd
import pytest

from playground.config import build_config, diff_switches, parse_profile_text, profile_names
from playground.engine import run_backtest
from playground.events import EventsFileMissing, load_events
from playground.ladder import LADDER_PROFILES, run_ladder, run_matrix, switch_report
from playground.report import compact_summary


def test_profiles_exist():
    for n in LADDER_PROFILES:
        assert n in profile_names()


@pytest.mark.parametrize("i", range(1, 8))
def test_each_yellow_differs_from_green_by_exactly_one_switch(i):
    g, y = build_config("green"), build_config(f"yellow_E{i}")
    assert len(diff_switches(g, y)) == 1, diff_switches(g, y)
    assert y.tier == "yellow" and y.effective_budget() == 0.05 and g.effective_budget() == 0.15


def test_red_profile():
    r = build_config("red")
    assert r.effective_budget() == 0.01 and r.entry_requires_gate_trip and r.max_open_lots == 1 and not r.gate
    assert build_config("red", {"budget_pct": 0.02}).effective_budget() == 0.02


def test_profile_parser_rejects_bare_words():
    assert parse_profile_text('a: 1\nb: "x"  # c\nc: [1, 2]\nd: null') == {"a": 1, "b": "x", "c": [1, 2], "d": None}
    with pytest.raises(ValueError):
        parse_profile_text("a: delta")


def test_events_missing_is_a_clear_error(tmp_path):
    with pytest.raises(EventsFileMissing, match="never invent|Dates are never invented"):
        load_events(tmp_path / "nope.csv")


def test_events_csv_roundtrip(tmp_path):
    p = tmp_path / "e.csv"
    p.write_text("# c\ndate,event,symbol\n2024-01-31,fomc,\n2024-02-13,CPI,\n")
    df = load_events(p)
    assert list(df.event) == ["FOMC", "CPI"]
    p.write_text("date,event\n2024-01-31,BOGUS\n")
    with pytest.raises(ValueError):
        load_events(p)


def test_end_to_end_synthetic(syn, syn_events):
    cfg = build_config("green", {"resolution": "15m", "start_date": "2016-01-04", "end_date": "2017-12-29"})
    r1 = run_backtest(syn, cfg, syn_events, "t")
    r2 = run_backtest(syn, cfg, syn_events, "t")
    assert len(r1.trades) > 5
    pd.testing.assert_frame_equal(r1.trades, r2.trades)
    pd.testing.assert_frame_equal(r1.equity, r2.equity)
    assert (r1.equity["cash"] >= 0).all()
    t = r1.trades
    assert (t["risk_total_after"] <= t["budget"] + 1e-6).all()                 # 15% budget respected
    assert (t["risk_expiry_after"] <= 0.6 * t["budget"] + 1e-6).all()          # 60% per-expiry cap respected
    assert (t["credit"] >= 0.5).all()
    assert (t["entry_ts"].dt.time >= pd.Timestamp("09:45").time()).all()
    assert (t["entry_ts"].dt.time <= pd.Timestamp("15:30").time()).all()
    assert set(t["exit_reason"]) <= {"tp", "stop", "breach", "time", "expiry", "end"}
    assert (t["fill"] == "natural").all()
    # cash identity: cash ends = start + interest + sum(pnl); at least pnl moves it
    assert r1.equity["cash"].iloc[-1] > 0
    s = compact_summary(r1)
    assert len(s.splitlines()) <= 40 and "fill=natural" in s


def test_seed_changes_entry_times(syn, syn_events):
    a = build_config("green", {"resolution": "15m", "start_date": "2016-01-04", "end_date": "2016-12-30", "seed": 1})
    b = a.replace(seed=2)
    ta = run_backtest(syn, a, syn_events, "t").trades["entry_ts"]
    tb = run_backtest(syn, b, syn_events, "t").trades["entry_ts"]
    assert list(ta) != list(tb)


def test_mid_fill_beats_natural(syn, syn_events):
    c = build_config("green", {"resolution": "15m", "start_date": "2016-01-04", "end_date": "2017-06-30"})
    nat = run_backtest(syn, c, syn_events, "t").equity.equity.iloc[-1]
    mid = run_backtest(syn, c.replace(fill="mid"), syn_events, "t").equity.equity.iloc[-1]
    assert mid > nat


def test_full_use_does_not_shrink_size(syn, syn_events):
    c = build_config("green", {"resolution": "15m", "start_date": "2016-01-04", "end_date": "2016-12-30"})
    a = run_backtest(syn, c, syn_events, "t").trades
    b = run_backtest(syn, c.replace(full_use=True), syn_events, "t").trades
    assert b["contracts"].iloc[0] >= a["contracts"].iloc[0]


def test_matrix_and_ladder_run(syn, syn_events):
    ov = {"resolution": "15m", "start_date": "2016-01-04", "end_date": "2017-12-29"}
    m = run_matrix(syn, build_config("green", ov), syn_events, "t")
    assert m.count("\n") >= 9 and "gate+upside+cb21 / B tp50+21DTE" in m
    out = run_ladder(syn, ov, syn_events, "t", split_sample=True)
    for name in LADDER_PROFILES:
        assert name in out
    assert "first half" in out and "second half" in out
    assert "10-year tier only" in out


def test_guards_actually_trigger_in_synthetic(syn, syn_events):
    c = build_config("green", {"resolution": "15m"})
    r = run_backtest(syn, c, syn_events, "t")
    assert r.blocked.get("gate", 0) > 0
    reds = run_backtest(syn, build_config("red", {"resolution": "15m"}), syn_events, "t")
    assert len(reds.trades) >= 1                                   # red enters only on gate-trip days
    assert (reds.trades["contracts"] > 0).all()
