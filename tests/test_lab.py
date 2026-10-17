"""Lab engine: strategies, strikes, management rules on hand-built markets, counterfactuals, logs."""
import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from helpers import TI, cfg, market  # noqa: E402

from playground import bs, mgmt, rules, schema  # noqa: E402
from playground.config import build_config  # noqa: E402
from playground.engine import Leg  # noqa: E402
from playground.lab import entry_schedule, run_lab, save_lab_run  # noqa: E402
from playground.strategies import (STRATEGIES, Chain, collateral_per_contract, max_loss_ps, payoff_ps)  # noqa: E402
from playground.tickers import TickerInfo, increment_near, spread_width  # noqa: E402

D = dt.date
DROP = D(2016, 1, 20)


def slide(rate=1.2, cap=9.0, start=DROP):
    return lambda d, k: 100.0 - (0 if d < start else min(cap, (d - start).days * rate + (k / 79 if d == start else 0)))


def first(res):
    return res.trades.iloc[0]


# ---------------------------------------------------------------- widths on odd strike grids
@pytest.mark.parametrize("inc,spot,pct,expect", [
    (1.0, 500.0, 0.007, 4.0), (0.5, 215.0, 0.007, 1.5), (2.5, 100.0, 0.007, 2.5), (5.0, 600.0, 0.007, 5.0),
    (1.0, 100.0, 0.02, 2.0)])
def test_width_pct_rounds_to_local_increment(inc, spot, pct, expect):
    strikes = [round(spot * 0.7 + inc * i, 3) for i in range(int(spot * 0.6 / inc))]
    w = spread_width(spot, strikes, build_config("green", {"width_mode": "pct", "width_pct": pct}))
    assert w == expect and (w / inc) == int(w / inc)


def test_width_fixed_from_ticker_and_mixed_grid():
    strikes = [400 + i for i in range(80)] + [480 + 5 * i for i in range(20)]
    assert increment_near(strikes, 440.0) == 1.0
    c = build_config("green")
    assert spread_width(440.0, strikes, c, TickerInfo("SPY", width_fixed=5.0)) == 5.0
    assert spread_width(440.0, strikes, c, TickerInfo("QQQ")) == 3.0          # 0.7% of 440 = 3.08 -> $3
    assert increment_near([100, 102.5, 105, 107.5, 110], 105.0) == 2.5


# ---------------------------------------------------------------- strategy max loss / payoff at expiry
def test_condor_max_loss_and_payoff():
    legs = [Leg("P", 95, -1), Leg("P", 90, 1), Leg("C", 105, -1), Leg("C", 110, 1)]
    assert collateral_per_contract(legs, 0.90, 100) == 410.0
    for S, pay in ((100, 0), (94, -1), (80, -5), (106, -1), (130, -5)):
        assert payoff_ps(legs, S) == pay
    assert max_loss_ps(legs, 0.0) == 5.0


def test_iron_fly_bull_put_bwb_cash_secured_covered_call_max_loss():
    fly = [Leg("P", 100, -1), Leg("P", 95, 1), Leg("C", 100, -1), Leg("C", 105, 1)]
    assert collateral_per_contract(fly, 3.0, 100) == 200.0 and payoff_ps(fly, 100) == 0 and payoff_ps(fly, 110) == -5
    bp = [Leg("P", 95, -1), Leg("P", 90, 1)]
    assert collateral_per_contract(bp, 1.0, 100) == 400.0
    bwb = [Leg("P", 100, 1), Leg("P", 95, -2), Leg("P", 70, 1)]       # near 5, far 25
    assert payoff_ps(bwb, 95) == 5 and payoff_ps(bwb, 100) == 0 and payoff_ps(bwb, 70) == -20 and payoff_ps(bwb, 40) == -20
    assert collateral_per_contract(bwb, 0.10, 100) == pytest.approx(1990.0)     # (far-near) - credit
    assert collateral_per_contract(bwb, -0.30, 100) == pytest.approx(2030.0)    # a debit adds to the max loss
    csp = [Leg("P", 95, -1)]
    assert collateral_per_contract(csp, 1.2, 100) == pytest.approx(9380.0)      # strike*100 - credit*100
    cc = [Leg("S", 0, 1), Leg("C", 105, -1)]
    assert collateral_per_contract(cc, 1.0, 100) == pytest.approx(9900.0)       # spot*100 - premium
    assert payoff_ps(cc, 120, 100) == 20 - 15
    with pytest.raises(ValueError):
        max_loss_ps([Leg("C", 100, -1)], 1.0)                                    # undefined risk refused
    with pytest.raises(ValueError):
        max_loss_ps([Leg("P", 95, -1), Leg("P", 90, 1), Leg("C", 105, -1)], 2.0)


# ---------------------------------------------------------------- chain / strike picker
def mk_chain(mode, deltas=True, spot=100.0, vix=15.3, **sr):
    cfgv = build_config("green", {"strike_rule": {"mode": mode, **sr}})
    strikes = [80 + i for i in range(41)]
    q, d = {}, {}
    for r in "PC":
        for k in strikes:
            q[(r, float(k))] = (0.10, 0.12)
            if deltas:
                m = 0.5 + 0.04 * (k - spot) if r == "P" else 0.5 - 0.04 * (k - spot)
                d[(r, float(k))] = (-1 if r == "P" else 1) * min(0.99, max(0.01, m))
    return Chain(cfgv, spot, vix, [float(k) for k in strikes], q, d, 5.0)


def test_delta_strike_picker():
    ch = mk_chain("delta", put_delta=0.16, call_delta=0.20)
    sp, sc = ch.short("P"), ch.short("C")
    assert abs(abs(ch.delta("P", sp)) - 0.16) <= 0.02 and abs(abs(ch.delta("C", sc)) - 0.20) <= 0.02
    assert sp < 100 < sc and ch.short("P", target_delta=0.30) > sp         # higher delta = closer to the money
    assert mk_chain("vix", put_pct=0.046, call_pct=0.04, vix_ref=15.3).short("P") == 95.0   # 100*(1-0.046)=95.4


def test_delta_picker_bs_fallback_and_sources():
    """No provider delta in the snapshot -> Black-Scholes delta from mid IV; logged in meta.delta_source."""
    sr = {"mode": "delta", "put_delta": 0.16, "call_delta": 0.16}
    p = market(lambda d, k: 100.0)
    r1 = run_lab(p, cfg(strike_rule=sr), None, "", TI)
    assert r1.meta["delta_source"]["bs_mid_iv"] > 0 and r1.meta["delta_source"]["provider"] == 0
    t = first(r1)
    assert -0.22 < t.short_put_delta < -0.10 and 0.10 < t.short_call_delta < 0.22
    pg = market(lambda d, k: 100.0, provider_greeks=True)
    r2 = run_lab(pg, cfg(strike_rule=sr), None, "", TI)
    assert r2.meta["delta_source"]["provider"] > 0
    assert abs(first(r2).short_put_delta - first(r1).short_put_delta) < 0.03    # the two sources agree
    # bs scalar roundtrip
    px = bs.price_s(100, 95, 0.1, 0.04, 0.2, "P")
    assert bs.delta_from_mid(px, 100, 95, 0.1, 0.04, "P") == pytest.approx(bs.delta_s(100, 95, 0.1, 0.04, 0.2, "P"), abs=1e-4)
    assert bs.delta_from_mid(10.0, 100, 120, 0.1, 0.04, "P") is None      # below intrinsic: no IV


# ---------------------------------------------------------------- strategies end to end
@pytest.mark.parametrize("strategy", ["iron_condor", "iron_butterfly", "bull_put_spread", "put_bwb"])
def test_strategies_open_and_log_exact_collateral(strategy):
    r = run_lab(market(), cfg(strategy=strategy, strike_rule={"mode": "delta", "put_delta": 0.16, "call_delta": 0.16}),
                None, "", TI)
    assert len(r.trades) >= 3
    t = first(r)
    assert t.strategy == strategy and t.collateral_per_contract > 0
    if strategy in ("iron_condor", "iron_butterfly", "bull_put_spread"):
        assert t.collateral_per_contract == pytest.approx(100 * t.width - t.credit_per_contract, abs=0.51)
    if strategy == "put_bwb":
        assert t.collateral_per_contract > 100 * t.width * 2           # far wing >> near wing (width = near wing)


def test_iron_fly_forced_close_and_default_tp():
    r = run_lab(market(), cfg(strategy="iron_butterfly", tp=9.0, ifly_tp=9.0), None, "", TI)
    t = first(r)
    assert t.exit_reason == "time" and (pd.Timestamp(t.expiry).date() - pd.Timestamp(t.exit_ts).date()).days <= 2
    assert build_config("green").ifly_tp == 0.25
    # 25% take-profit hit quickly when the market is flat (credit decays)
    r2 = run_lab(market(), cfg(strategy="iron_butterfly", tp=9.0, ifly_tp=0.05), None, "", TI)
    assert first(r2).exit_reason == "tp"


def test_experimental_strategies_unit_mode_collateral():
    r = run_lab(market(), cfg(strategy="cash_secured_put", strike_rule={"mode": "delta", "put_delta": 0.3, "call_delta": 0.3},
                              time_exit=21), None, "", TI)
    t = first(r)
    assert t.collateral_per_contract > 8000                              # ~ strike*100
    rc = run_lab(market(), cfg(strategy="covered_call", strike_rule={"mode": "delta", "put_delta": .3, "call_delta": 0.3}),
                 None, "", TI)
    assert first(rc).collateral_per_contract > 9000


# ---------------------------------------------------------------- management: next-bar natural fills
def test_stop_fills_on_next_bar_not_trigger_bar():
    f = lambda d, k: 93.0 if (d > DROP or (d == DROP and k >= 20)) else 100.0     # gap at 11:10 ET
    t = first(run_lab(market(f), cfg(stop_mult=2.0), None, "", TI))
    assert t.exit_reason == "stop" and t.exit_ts.endswith("11:30:00") and t.exit_spot == 93.0
    assert t.pnl_per_contract < -t.credit_per_contract                       # gap cost more than the stop level


def test_stop_dollar_and_tp_and_breach_and_time():
    f = lambda d, k: 93.0 if (d > DROP or (d == DROP and k >= 20)) else 100.0
    assert first(run_lab(market(f), cfg(stop_dollar=60.0), None, "", TI)).exit_reason == "stop_dollar"
    assert first(run_lab(market(f), cfg(breach_exit=True), None, "", TI)).exit_reason == "breach"
    assert first(run_lab(market(), cfg(tp=0.05), None, "", TI)).exit_reason == "tp"
    t = first(run_lab(market(), cfg(time_exit=21), None, "", TI))
    assert t.exit_reason == "time" and (pd.Timestamp(t.expiry).date() - pd.Timestamp(t.exit_ts).date()).days <= 21
    assert first(run_lab(market(), cfg(), None, "", TI)).exit_reason == "expiry"


@pytest.mark.parametrize("trig,val", [("distance", 0.01), ("delta", 0.25)])
def test_trigger_exit(trig, val):
    t = first(run_lab(market(slide()), cfg(trigger=trig, trigger_value=val, action="exit"), None, "", TI))
    assert t.exit_reason == "trigger_exit"
    base = first(run_lab(market(slide()), cfg(), None, "", TI))
    assert t.pnl_per_contract > base.pnl_per_contract              # exiting at the trigger beats riding it down


def test_trigger_none_is_inert_and_value_orders_firing():
    a = first(run_lab(market(slide()), cfg(trigger="distance", trigger_value=0.01, action="none"), None, "", TI))
    assert a.exit_reason == "expiry" and a.adj_count == 0
    early = first(run_lab(market(slide()), cfg(trigger="distance", trigger_value=0.03, action="exit"), None, "", TI))
    late = first(run_lab(market(slide()), cfg(trigger="distance", trigger_value=0.005, action="exit"), None, "", TI))
    assert pd.Timestamp(early.exit_ts) <= pd.Timestamp(late.exit_ts)


def adjustments(res):
    return res.meta["adjustments"]


def test_roll_tested_moves_tested_vertical_closer_same_expiry():
    r = run_lab(market(slide(1.0)), cfg(trigger="distance", trigger_value=0.01, action="roll_tested"), None, "", TI)
    t, a = first(r), adjustments(r)[0]
    assert t.adj_count >= 1 and a["kind"] == "roll_tested"
    old = [x for x in a["closed"] if x[2] < 0][0][1]
    new = [x for x in a["opened"] if x[2] < 0][0][1]
    assert all(x[0] == "P" for x in a["closed"] + a["opened"])         # only the tested (put) vertical
    spot = 100 - 1.0 * (a["day"] and (pd.Timestamp(a["day"]).date() - DROP).days)
    assert abs(spot - new) < abs(spot - old)                           # strictly closer to spot
    assert t.expiry == r.trades.iloc[0].expiry
    assert t.adj_cost_per_contract == pytest.approx(-100 * sum(x["net_credit_ps"] for x in adjustments(r)), abs=0.1)


def test_roll_untested_and_roll_condor_and_close_tested():
    g = slide(1.2)
    r = run_lab(market(g), cfg(trigger="distance", trigger_value=0.01, action="roll_untested", max_rolls=1), None, "", TI)
    a = adjustments(r)[0]
    assert {x[0] for x in a["closed"]} == {"C"} and min(x[1] for x in a["opened"]) < min(x[1] for x in a["closed"])
    r = run_lab(market(g), cfg(trigger="distance", trigger_value=0.01, action="roll_condor", max_rolls=1), None, "", TI)
    a = adjustments(r)[0]
    assert {x[0] for x in a["closed"]} == {"P", "C"} and len(a["opened"]) == 4
    assert all(n < o for n, o in zip(sorted(x[1] for x in a["opened"]), sorted(x[1] for x in a["closed"])))   # both followed the move down
    r = run_lab(market(g), cfg(trigger="distance", trigger_value=0.03, action="close_tested_spread"), None, "", TI)
    t, a = first(r), adjustments(r)[0]
    assert t.adj_count == 1 and a["opened"] == [] and {x[0] for x in a["closed"]} == {"P"} and a["net_credit_ps"] < 0


def test_max_rolls_and_credit_only_skip():
    g = slide(1.0, cap=12.0)
    free = run_lab(market(g), cfg(trigger="distance", trigger_value=0.01, action="roll_untested", max_rolls=3), None, "", TI)
    capped = run_lab(market(g), cfg(trigger="distance", trigger_value=0.01, action="roll_untested", max_rolls=1), None, "", TI)
    assert first(capped).adj_count == 1 and first(free).adj_count > 1
    zero = run_lab(market(g), cfg(trigger="distance", trigger_value=0.01, action="roll_untested", max_rolls=0), None, "", TI)
    assert first(zero).adj_count == 0
    # tested-side roll costs a net DEBIT in a falling market -> credit-only skips it
    paid = run_lab(market(g), cfg(trigger="distance", trigger_value=0.01, action="roll_tested", max_rolls=2), None, "", TI)
    skipped = run_lab(market(g), cfg(trigger="distance", trigger_value=0.01, action="roll_tested", max_rolls=2,
                                     roll_credit_only=True), None, "", TI)
    assert any(a["net_credit_ps"] < 0 for a in adjustments(paid))
    assert all(a["net_credit_ps"] >= 0 for a in adjustments(skipped))


def test_hedges():
    g = slide(1.2)
    r = run_lab(market(g), cfg(trigger="distance", trigger_value=0.02, action="hedge_put_call"), None, "", TI)
    t, a = first(r), adjustments(r)[0]
    assert a["kind"] == "hedge_put_call" and a["opened"][0][0] == "P" and a["opened"][0][2] == 1     # long put on the tested side
    assert t.hedge_cost_per_contract == pytest.approx(-100 * a["net_credit_ps"], abs=0.01) and t.adj_count == 0
    r = run_lab(market(g), cfg(trigger="distance", trigger_value=0.02, action="hedge_shares", stock_slip_bps=1.0), None, "", TI)
    t, a = first(r), adjustments(r)[0]
    shares = a["opened"][0][2]
    assert a["kind"] == "hedge_shares" and shares < 0                  # put-tested condor is long delta -> sell shares
    assert t.hedge_cost_per_contract > 0
    nohedge = first(run_lab(market(g), cfg(), None, "", TI))
    assert t.pnl_per_contract != nohedge.pnl_per_contract


# ---------------------------------------------------------------- counterfactual
def test_counterfactual_equals_holding_the_original_to_expiry():
    f = slide(0.5)
    early = first(run_lab(market(f), cfg(trigger="distance", trigger_value=0.02, action="exit"), None, "", TI))
    held = first(run_lab(market(f), cfg(), None, "", TI))
    assert early.exit_reason == "trigger_exit" and held.exit_reason == "expiry"
    assert early.cf_pnl_to_expiry_per_contract == pytest.approx(held.pnl_per_contract, abs=0.01)
    assert pd.isna(held.cf_pnl_to_expiry_per_contract)               # ran to expiry unadjusted -> blank
    rolled = first(run_lab(market(f), cfg(trigger="distance", trigger_value=0.02, action="roll_tested", max_rolls=1), None, "", TI))
    assert rolled.adj_count == 1 and rolled.cf_pnl_to_expiry_per_contract == pytest.approx(held.pnl_per_contract, abs=0.01)


# ---------------------------------------------------------------- fees / ex-div / levels
def test_fees_per_contract_per_leg():
    t = first(run_lab(market(), cfg(tp=0.05, fee_per_contract=0.05), None, "", TI))
    assert t.fees_per_contract == pytest.approx(0.05 * 4 * 2)           # 4 legs in, 4 legs out
    t0 = first(run_lab(market(), cfg(tp=0.05, fee_per_contract=0.0), None, "", TI))
    assert t0.fees_per_contract == 0 and t0.pnl_per_contract - t.pnl_per_contract == pytest.approx(0.4, abs=0.01)
    held = first(run_lab(market(), cfg(fee_per_contract=0.05), None, "", TI))
    assert held.fees_per_contract == pytest.approx(0.05 * 4)             # expiry settlement has no exit fee


def test_exdiv_flag_and_close_before_exdiv():
    p = market(lambda d, k: 100.0 if d < D(2016, 3, 10) else 112.0)        # short call ~105 goes ITM; synthetic ex-div 2016-03-18
    c = cfg(breach_exit=False)
    flagged = [t for t in run_lab(p, c, None, "", TI).trades.itertuples() if t.exdiv_flag == 1]
    assert flagged, "an ITM short call within 2 days of the synthetic ex-div date must be flagged"
    r = run_lab(p, cfg(close_before_exdiv=True), None, "", TI)
    assert (r.trades.exit_reason == "other").any()
    assert not run_lab(market(), cfg(), None, "", TI).trades.exdiv_flag.any()


def test_level_tags_and_guards_on():
    for prof, lvl in (("green", 1), ("yellow_E4", 2), ("red", 3), ("experiment_all_off", 4)):
        c = build_config(prof, {"sizing": "fixed1", "resolution": "15m", "min_credit_pct": 0.01, "entry_sample": "first",
                                "event_buffer": False})
        r = run_lab(market(), c.replace(entry_cadence="monthly"), None, "", TI)
        if len(r.trades):
            assert set(r.trades.level) == {lvl} and set(r.trades.profile) == {prof}
    g = run_lab(market(), cfg(gate=True, upside_guard=True, event_buffer=False), None, "", TI)
    assert first(g).guards_on == "gate;upside"
    assert first(run_lab(market(), cfg(gate=False, upside_guard=False), None, "", TI)).guards_on == "none"
    assert build_config("experiment_all_off").gate is False and build_config("experiment_all_off").cb_days == 0
    assert build_config("experiment_no_stop").stop_mult is None
    assert build_config("experiment_high_delta").strike_rule["put_delta"] == 0.30
    assert build_config("experiment_forced_entry").force_entry is True


# ---------------------------------------------------------------- entry filters
def test_entry_filters_log_reasons():
    p = market(vix=30.0)
    c = cfg(gate=True)
    r = run_lab(p, c, None, "", TI)
    assert len(r.trades) == 0 and set(r.blocked.reason) == {"gate"}
    spiky = market(vix=20.0)
    spiky._vix[:] = 20.0
    for i in range(len(spiky._vix)):                     # VIX +40% over a week mid-January
        if spiky.days[i] >= D(2016, 1, 8):
            spiky._vix.iloc[i] = 28.0
    r = run_lab(spiky, cfg(post_spike_wait=10), None, "", TI)
    assert "spike" in set(r.blocked.reason)
    r = run_lab(market(), cfg(term_filter=True), None, "", TI)
    assert len(r.trades) > 0                              # ratio == 1.0 is not backwardation
    inv = market()
    inv._vix3m[:] = 15.0                                  # VIX 20 / VIX3M 15 -> 1.33 backwardation
    r = run_lab(inv, cfg(term_filter=True), None, "", TI)
    assert len(r.trades) == 0 and set(r.blocked.reason) == {"term"}
    r = run_lab(market(), cfg(min_ivr=50), None, "", TI)
    assert set(r.blocked.reason) <= {"ivr", "cadence"}
    r = run_lab(market(), cfg(strategy="put_bwb", delta_cap_pct=1.0), None, "", TI)
    assert "deltacap" in set(r.blocked.reason)
    forced = run_lab(market(vix=30.0), cfg(gate=True, force_entry=True), None, "", TI)
    assert len(forced.trades) > 0 and first(forced).guards_on == "none"


def test_cadence_schedules():
    p = market(start="2016-01-04", end="2016-04-29")
    days = list(p.get_underlying_daily().index)
    m, _ = entry_schedule(days, 6, len(days) - 1, cfg(entry_cadence="monthly", dte=[28, 35, 42]))
    for d, exp in m.items():
        assert exp.weekday() == 4 and 15 <= exp.day <= 21 and abs((exp - d).days - 35) <= 3
    w, _ = entry_schedule(days, 6, len(days) - 1, cfg(entry_cadence="weekly"))
    tw, _ = entry_schedule(days, 6, len(days) - 1, cfg(entry_cadence="twice-weekly"))
    assert len(tw) > len(w) > len(m) and all(d.weekday() <= 4 for d in w)
    s4, _ = entry_schedule(days, 6, len(days) - 1, cfg(scale_in=4))
    exps = {}
    for d, e in s4.items():
        exps.setdefault(e, []).append(d)
    assert any(len(v) == 4 for v in exps.values())


def test_dte_target_skips_when_far_expiry_unlisted(tmp_path):
    p = market(start="2016-01-04", end="2016-12-30")
    r = run_lab(p, cfg(dte_target=90), None, "", TI)
    # synthetic lists ~125 days out, so 90 DTE monthlies exist; a 180 DTE target must be skipped and logged
    far = run_lab(p, cfg(dte_target=180), None, "", TI)
    assert len(far.trades) == 0 and set(far.blocked.reason) == {"cadence"}
    assert len(r.trades) > 0 and r.trades.dte_entry.between(80, 100).all()


# ---------------------------------------------------------------- logs / schema / determinism / equity mode
def test_unit_mode_logs_satisfy_schema(tmp_path):
    r = run_lab(market(slide(0.7)), cfg(trigger="distance", trigger_value=0.02, action="roll_tested",
                                         entry_cadence="weekly"), None, "", TI)
    out = save_lab_run(r, tmp_path, "test")
    assert schema.validate_run_dir(out) == []
    t = pd.read_csv(out / "trades.csv")
    assert list(t.columns) == schema.TRADE_COLUMNS and len(t) > 3
    m = pd.read_csv(out / "marks.csv")
    assert set(m.trade_id) == set(t.trade_id)
    last = m.sort_values("date").groupby("trade_id").tail(1).set_index("trade_id")["mark_pnl_per_contract"]
    assert all(abs(last[tid] - pnl) < 0.01 for tid, pnl in zip(t.trade_id, t.pnl_per_contract))   # exit-day mark == realised
    assert not any(c in " ".join(t.columns).lower() for c in ("bid", "ask"))                     # no raw quotes in the log
    bad = t.copy()
    bad["exit_reason"] = "nope"
    assert schema.validate_trades(bad)
    assert schema.validate_trades(t.drop(columns=["spot_source"]))


def test_deterministic_and_seed_changes_entry_minute():
    c = cfg(entry_sample="random", seed=1, entry_cadence="weekly", tp=0.3)
    a = run_lab(market(), c, None, "", TI)
    b = run_lab(market(), c, None, "", TI)
    pd.testing.assert_frame_equal(a.trades, b.trades)
    pd.testing.assert_frame_equal(a.marks, b.marks)
    other = run_lab(market(), c.replace(seed=2), None, "", TI)
    assert list(a.trades.entry_ts) != list(other.trades.entry_ts)


def test_equity_mode_sizes_within_budget_and_cap():
    c = cfg(sizing="equity-pct", entry_cadence="weekly", cb_days=21)
    r = run_lab(market(slide(0.8)), c, None, "", TI)
    assert len(r.trades) > 2 and r.equity is not None and (r.equity.cash > 0).all()
    sz = r.sizing.set_index("trade_id")
    for t in r.trades.itertuples():
        n, eq = sz.loc[t.trade_id, "contracts"], sz.loc[t.trade_id, "equity_at_entry"]
        assert n >= 1 and n * t.collateral_per_contract <= 0.15 * eq * 0.6 + 1e-6 + 0.0 or n * t.collateral_per_contract <= 0.15 * eq + 1e-6
    # CSP collateral rarely fits a 15% budget of 100k -> blocked as 'size', never oversized
    csp = run_lab(market(), cfg(sizing="equity-pct", strategy="cash_secured_put",
                                strike_rule={"mode": "delta", "put_delta": .3, "call_delta": .3}), None, "", TI)
    assert len(csp.trades) == 0 and "size" in set(csp.blocked.reason)


def test_dte_target_35_builds_28_to_42_window():
    p = market(start="2016-01-04", end="2016-12-30")
    r = run_lab(p, cfg(dte_target=35), None, "", TI)
    assert len(r.trades) > 0 and r.trades.dte_entry.between(28, 42).all()


@pytest.mark.parametrize("greeks", [False, True], ids=["black_scholes", "thetadata"])
def test_entry_log_has_strike_delta_and_source_per_entry(greeks):
    sr = {"mode": "delta", "put_delta": 0.16, "call_delta": 0.16}
    lines = []
    r = run_lab(market(lambda d, k: 100.0, provider_greeks=greeks), cfg(strike_rule=sr), None, "", TI,
                entry_log=lines.append)
    assert len(lines) == len(r.trades) > 0
    want = "src=thetadata" if greeks else "src=black_scholes"
    for line, t in zip(lines, r.trades.itertuples()):
        assert f"trade={t.trade_id} " in line
        legs = line.split(" | ")[1:]
        sp = next(x for x in legs if x.startswith("short P"))
        sc = next(x for x in legs if x.startswith("short C"))
        assert want in sp and want in sc and "delta=nan" not in sp + sc
        # the logged delta is the one the trade row carries for that side
        assert float(sp.split("delta=")[1].split()[0]) == pytest.approx(t.short_put_delta, abs=1e-4)
        assert float(sc.split("delta=")[1].split()[0]) == pytest.approx(t.short_call_delta, abs=1e-4)

