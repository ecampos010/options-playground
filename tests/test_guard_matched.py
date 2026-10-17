"""Matched guarded-vs-unguarded comparison: the arms differ only in the Green guard switches, the guarded entries are a
strict subset of the unguarded entries, and the metrics are right on a hand-built path."""
import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from helpers import market  # noqa: E402

from playground.config import build_config, diff_switches  # noqa: E402
from playground.guard_matched import (GUARD_KEYS, GUARD_REASONS, arm_metrics, entry_dates, guarded_config,  # noqa: E402
                                      unguarded_config, run_matched, skipped_trades)
from playground.lab import LabResult, run_id_for, run_lab  # noqa: E402

D = dt.date
START, END = "2016-01-04", "2016-06-30"
# SPY drops 5% on a few days (gate: day <= -1.5%, and the 5-day move stays <= -4% for the next days)
DROP_DAYS = {D(2016, 2, 8), D(2016, 2, 9), D(2016, 3, 22), D(2016, 5, 10)}
EVENTS = pd.DataFrame({"date": [D(2016, 3, 16), D(2016, 6, 15)], "event": ["FOMC", "CPI"], "symbol": ["", ""]})


def drop_market(**kw):
    return market(spot_fn=lambda d, k: 95.0 if d in DROP_DAYS else 100.0, start=START, end=END, vix=20.0, **kw)


# ---------------------------------------------------------------- the arms
def test_arms_differ_only_in_the_guard_switches():
    p, g = unguarded_config("monthly", START, END), guarded_config("monthly", START, END)
    assert diff_switches(p, g) == sorted(GUARD_KEYS)
    assert not any([p.gate, p.upside_guard, p.event_buffer, p.cb_days, p.earnings_filter])
    assert p.term_filter is False and p.post_spike_wait == 0 and p.min_ivr == 0 and p.delta_cap_pct is None
    assert p.max_open_lots == 0 and p.entry_requires_gate_trip is False and p.force_entry is False


def test_guarded_arm_carries_the_green_guard_values():
    green = build_config("green")
    g = guarded_config("monthly", START, END)
    for k in GUARD_KEYS:
        assert getattr(g, k) == getattr(green, k), k
    assert (g.gate_vix, g.gate_day, g.gate_5d, g.upside_5d, g.event_buffer_days) == (25.0, -0.015, -0.04, 0.04, 2)


@pytest.mark.parametrize("cfg_fn", [unguarded_config, guarded_config])
def test_both_arms_share_budget_strike_and_exits(cfg_fn):
    c = cfg_fn("weekly", START, END)
    assert c.effective_budget() == pytest.approx(0.15)                        # Green budget, not the experiment 5%
    assert c.strike_rule["mode"] == "delta" and c.strike_rule["put_delta"] == c.strike_rule["call_delta"] == 0.16
    assert (c.tp, c.stop_mult, c.time_exit, c.breach_exit, c.fill, c.dte_target) == (0.5, 2.0, 21, True, "natural", 45)


# ---------------------------------------------------------------- the subset, enforced
@pytest.mark.parametrize("cadence", ["monthly", "weekly"])
def test_guarded_entries_are_a_strict_subset_of_unguarded_entries(cadence):
    p = drop_market()
    m = run_matched(p, EVENTS, "test-events", cadence, START, END)
    unguarded_days, guarded_days = entry_dates(m.unguarded), entry_dates(m.guarded)
    assert guarded_days <= unguarded_days                                          # subset
    assert len(guarded_days) < len(unguarded_days)                                 # the guards really skipped something
    guard_hits = {r for r in m.guarded.blocked.reason} & set(GUARD_REASONS)
    assert guard_hits, "test market must trip at least one guard, or the test proves nothing"
    assert not ({r for r in m.unguarded.blocked.reason} & set(GUARD_REASONS)), "unguarded arm must never block on a guard"
    s = skipped_trades(m)
    assert len(s) == len(unguarded_days - guarded_days)
    assert "unexplained" not in set(s.guarded_reason)


def test_whitelist_does_not_change_an_arm_that_has_no_guards():
    p = drop_market()
    unguarded = run_lab(p, unguarded_config("weekly", START, END), EVENTS, "test-events")
    again = run_lab(p, unguarded_config("weekly", START, END), EVENTS, "test-events", entry_days=entry_dates(unguarded))
    assert list(again.trades.entry_ts) == list(unguarded.trades.entry_ts)
    assert list(again.trades.pnl_per_contract) == pytest.approx(list(unguarded.trades.pnl_per_contract))


def test_a_cache_gap_on_a_held_leg_is_counted_not_silent(monkeypatch):
    from playground.provider import MissingData
    p = drop_market()
    orig = p.leg_arrays

    def flaky(d, expiry, right, strike, ts_index, res):
        if d == D(2016, 3, 1):
            raise MissingData("gap")
        return orig(d, expiry, right, strike, ts_index, res)

    monkeypatch.setattr(p, "leg_arrays", flaky)
    res = run_lab(p, unguarded_config("weekly", START, END), EVENTS, "test-events")
    assert res.meta["leg_load_gaps"] > 0
    assert arm_metrics(res)["leg_load_gaps"] == res.meta["leg_load_gaps"]


def test_run_ids_of_existing_runs_are_unchanged_when_no_whitelist_is_given():
    cfg = unguarded_config("weekly", START, END)
    res = run_lab(drop_market(), cfg, EVENTS, "test-events")
    expanded = cfg.replace(dte=[38, 45, 52])                                   # _Lab applies dte_target before hashing
    assert res.run_id == run_id_for(expanded, {"ticker": cfg.ticker, "events": "test-events", "window": None})


# ---------------------------------------------------------------- metrics on a known path
def _hand_result() -> LabResult:
    cfg = unguarded_config("monthly", "2016-01-01", "2016-12-31")
    trades = pd.DataFrame({
        "trade_id": ["a", "b", "c"], "entry_ts": ["2016-01-11T10:00:00", "2016-02-08T10:00:00", "2016-02-22T10:00:00"],
        "exit_reason": ["tp", "stop", "tp"], "pnl_per_contract": [10.0, -20.0, 5.0],
        "collateral_per_contract": [100.0, 100.0, 100.0]})
    sizing = pd.DataFrame({"trade_id": ["a", "b", "c"], "contracts": [1, 2, 1], "equity_at_entry": [1e5] * 3})
    eq = pd.DataFrame({"date": pd.date_range("2016-01-01", periods=4), "equity": [100.0, 120.0, 90.0, 110.0]})
    meta = {"spot_coverage": {"days_without_spot": 0}, "open_excluded": {"trades": []}}
    return LabResult(cfg, "hand", trades, pd.DataFrame(), pd.DataFrame(columns=["date", "ticker", "reason"]), meta,
                     eq, sizing)


def test_arm_metrics_on_a_known_path():
    m = arm_metrics(_hand_result())
    assert m["trades"] == 3
    assert m["mean_ror"] == pytest.approx((0.10 - 0.20 + 0.05) / 3)
    assert m["win_rate"] == pytest.approx(2 / 3)
    assert (m["exit_tp"], m["exit_stop"], m["exit_time"], m["exit_breach"]) == (2, 1, 0, 0)
    assert m["worst_trade_ror"] == pytest.approx(-0.20)
    assert m["worst_5pct_ror"] == pytest.approx(-0.175)                       # linear interpolation, 3 trades
    assert m["total_pnl_usd"] == pytest.approx(10 * 1 - 20 * 2 + 5 * 1)       # contracts-weighted dollars
    assert m["max_dd"] == pytest.approx(90 / 120 - 1)                          # peak 120 -> trough 90
    assert m["worst_cycle"] == "2016-02" and m["worst_cycle_usd"] == pytest.approx(-40 + 5)
