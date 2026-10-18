"""Quote sanity on exit bars: a bar with out-of-order or crossed quotes fires no trigger, and the skip is counted."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from helpers import TI, cfg, market  # noqa: E402

from playground import lab  # noqa: E402
from playground.engine import Leg  # noqa: E402
from playground.lab import quote_sanity_faults, run_lab  # noqa: E402

CONDOR = [Leg("P", 95, -1), Leg("P", 90, 1), Leg("C", 105, -1), Leg("C", 110, 1)]
SANE = [(1.00, 1.20), (0.20, 0.30), (1.00, 1.20), (0.20, 0.30)]
EXP = ["2016-02-19"] * 4


def first(res):
    return res.trades.iloc[0]


# ---------------------------------------------------------------- the checks, on single bars
def test_sane_bar_has_no_fault():
    assert quote_sanity_faults(CONDOR, SANE, EXP) == []


def test_put_priced_above_higher_strike_is_a_fault():
    # the 554/559 incident shape: the lower-strike put's bid is above the higher-strike put's ask
    fly = [Leg("P", 559, -2), Leg("P", 554, 1), Leg("C", 559, -2), Leg("C", 564, 1)]
    q = [(3.0, 3.2), (3.6, 3.8), (3.0, 3.2), (0.1, 0.2)]
    assert "put_inverted" in quote_sanity_faults(fly, q, ["2025-04-17"] * 4)


def test_call_priced_below_higher_strike_is_a_fault():
    q = [(1.00, 1.20), (0.20, 0.30), (0.50, 0.60), (0.90, 1.00)]          # 110 call bid 0.9 above 105 call ask 0.6
    assert "call_inverted" in quote_sanity_faults(CONDOR, q, EXP)


def test_noise_inside_the_spreads_is_not_a_fault():
    q = [(0.00, 0.02), (0.00, 0.03), (0.00, 0.02), (0.00, 0.03)]          # mids cross, bids and asks do not
    assert quote_sanity_faults(CONDOR, q, EXP) == []


def test_crossed_leg_is_a_fault():
    q = [(1.50, 1.20), (0.20, 0.30), (1.00, 1.20), (0.20, 0.30)]
    assert "crossed" in quote_sanity_faults(CONDOR, q, EXP)


def test_close_cost_above_the_width_is_a_fault():
    q = [(9.00, 9.20), (0.20, 0.30), (9.00, 9.20), (0.20, 0.30)]
    assert "cost_range" in quote_sanity_faults(CONDOR, q, EXP)


def test_missing_quote_is_left_to_the_existing_path():
    q = [(float("nan"), 1.20), (0.20, 0.30), (1.00, 1.20), (0.20, 0.30)]
    assert quote_sanity_faults(CONDOR, q, EXP) == []


def test_different_expiries_are_not_compared():
    spread = [Leg("P", 554, -1), Leg("P", 559, 1)]
    q = [(3.5, 3.7), (0.4, 0.6)]                                             # 554 priced above 559: inverted
    assert "put_inverted" in quote_sanity_faults(spread, q, ["2025-04-17", "2025-04-17"])
    assert quote_sanity_faults(spread, q, ["2025-04-17", "2025-05-16"]) == []


# ---------------------------------------------------------------- the engine, on a market with a bad bar
def _inverted_puts_on(provider, day):
    """Every put on `day` is priced with the lower strike above the higher one, on every bar."""
    orig = provider.leg_arrays

    def wrapped(d, expiry, right, strike, ts_index, res):
        bid, ask = orig(d, expiry, right, strike, ts_index, res)
        if d == day and right == "P":
            px = 0.1 + (1000.0 - strike) * 0.01                                # lower strike priced higher
            bid, ask = np.full(len(ts_index), px), np.full(len(ts_index), px + 0.02)
        return bid, ask

    provider.leg_arrays = wrapped
    return provider


def test_out_of_order_puts_produce_no_take_profit_exit_on_that_bar():
    clean = first(run_lab(market(), cfg(tp=0.05), None, "", TI))
    assert clean.exit_reason == "tp"                                         # the clean market does take profit
    day = pd.Timestamp(clean.exit_ts).date()
    bad = run_lab(_inverted_puts_on(market(), day), cfg(tp=0.05), None, "", TI)
    exits_that_day = [t for t in bad.trades.itertuples()
                      if t.exit_reason == "tp" and pd.Timestamp(t.exit_ts).date() == day]
    assert exits_that_day == []                                              # no take-profit fired on the bad bars
    assert bad.meta["quote_sanity"]["skipped_position_bars"] > 0             # and the skips are counted
    assert {"put_inverted", "cost_range"} & set(bad.meta["quote_sanity"]["faults"])


def test_a_clean_market_skips_nothing_and_behaves_exactly_as_before(monkeypatch):
    with_check = run_lab(market(), cfg(tp=0.05), None, "", TI)
    monkeypatch.setattr(lab._Lab, "_quote_faults", lambda self, pos, b: [])  # the same run with the check switched off
    without = run_lab(market(), cfg(tp=0.05), None, "", TI)
    assert with_check.meta["quote_sanity"] == {"skipped_position_bars": 0, "faults": {}}
    pd.testing.assert_frame_equal(with_check.trades, without.trades)
    pd.testing.assert_frame_equal(with_check.marks, without.marks)


def test_skip_counter_counts_position_bars_and_names_the_reasons():
    p = _inverted_puts_on(market(), pd.Timestamp("2016-01-19").date())
    q = run_lab(p, cfg(tp=0.05), None, "", TI).meta["quote_sanity"]
    assert q["skipped_position_bars"] > 0 and q["faults"]
    assert set(q["faults"]) <= {"crossed", "put_inverted", "call_inverted", "cost_range", "zero_size"}
