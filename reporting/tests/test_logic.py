import math

import numpy as np
import pandas as pd
import pytest

from reporting import grid, montecarlo as mc, stress, thermo, verdicts, ways, mgmt
from reporting.replay import ReplayConfig

from .conftest import mk_trades, vd_from


# ---------- ranking
def _row(**k):
    base = dict(scenario_id="x", broke=False, broke_date="", calmar=1.0, max_dd=0.1, too_few_trades=False, final_equity=1e5)
    base.update(k)
    return base


def test_ranking_excludes_broke_and_orders_by_calmar_not_final_equity():
    df = pd.DataFrame([_row(scenario_id="lottery", calmar=0.2, final_equity=1e9), _row(scenario_id="steady", calmar=2.0, final_equity=1.2e5),
                       _row(scenario_id="dead_early", broke=True, broke_date="2018-03-01", calmar=9), _row(scenario_id="dead_late", broke=True, broke_date="2024-05-01", calmar=-1),
                       _row(scenario_id="thin", calmar=50, too_few_trades=True)])
    ok, br = grid.rank(df)
    assert list(ok["scenario_id"]) == ["steady", "lottery", "thin"]          # no broke rows; thin rows after evidence-grade rows
    assert list(br["scenario_id"]) == ["dead_late", "dead_early"]            # later broke ranks first


# ---------- verdicts
CF = verdicts.load_config()


def test_verdict_thresholds():
    r = dict(broke=False, max_dd=0.10, worst_month=-0.031, worst_trade_pct=-0.02, allocation=15, too_few_trades=False, dd_trough_date="2020-03-20")
    assert verdicts.classify(r, CF) == "Steady"
    assert verdicts.verdict(r, CF) == "Steady: worst month -3.1%, worst trade -2.0% of equity."
    assert verdicts.classify({**r, "max_dd": 0.15}, CF) == "Bumpy"                 # boundary is strict
    assert verdicts.classify({**r, "worst_month": -0.06}, CF) == "Bumpy"
    assert verdicts.classify({**r, "max_dd": 0.30}, CF) == "Rough"
    assert verdicts.classify({**r, "max_dd": 0.50}, CF) == "Severe"
    assert verdicts.classify({**r, "broke": True}, CF) == "Broke"


def test_verdict_text_names_window_and_family_broke():
    r = dict(broke=False, max_dd=0.22, worst_month=-0.08, worst_trade_pct=-0.04, allocation=15, too_few_trades=False, dd_trough_date="2020-03-20")
    s = verdicts.verdict(r, CF, {50: "2018-03-09"})
    assert "Survived the 2020 window with a 22% drawdown" in s and "went broke at the 50% allocation in 2018" in s
    assert "TOO FEW TRADES - not evidence" in verdicts.verdict({**r, "too_few_trades": True}, CF)
    b = verdicts.verdict({**r, "broke": True, "broke_date": "2018-03-09", "allocation": 50}, CF)
    assert b.startswith("Went broke on 2018-03-09 at the 50% allocation")


# ---------- thermometer
TC = thermo.load_config()


def test_thermometer_zone_boundaries():
    z = lambda dd, ruin, broke=False: thermo.zone(dd, ruin, broke, TC)  # noqa: E731
    assert z(0.1499, 0.0099) == "Safe"
    assert z(0.15, 0.0) == "Mild" and z(0.10, 0.01) == "Mild"                      # strict limits
    assert z(0.2999, 0.0499) == "Mild" and z(0.30, 0.0) == "Danger" and z(0.1, 0.05) == "Danger"
    assert z(0.4999, 0.1999) == "Danger" and z(0.5, 0.0) == "Breaking point" and z(0.1, 0.20) == "Breaking point"
    assert z(0.01, 0.0, broke=True) == "Breaking point"
    assert z(0.10, None) == "Safe" and z(0.20, None) == "Mild"                     # no Monte Carlo: drawdown + broke only


# ---------- Monte Carlo
def test_mc_reproducible_and_seed_sensitive():
    rng = np.random.default_rng(0)
    ret = rng.normal(0.005, 0.04, 90); low = np.minimum(ret, 0) - 0.01
    a = mc.run_mc(ret, low, 1e5, 300, (12, 36), 2, 500, 11)
    mc._cache.clear()
    b = mc.run_mc(ret, low, 1e5, 300, (12, 36), 2, 500, 11)
    c = mc.run_mc(ret, low, 1e5, 300, (12, 36), 2, 500, 12)
    assert repr(a) == repr(b) and repr(a) != repr(c)


def test_mc_block_structure_and_same_month_across_tickers():
    idx = mc.month_indices(60, 36, 3, 200, 5)
    assert idx.shape == (200, 36)
    d = np.diff(idx[:, :36], axis=1)
    inside = np.array([i % 3 != 2 for i in range(35)])                             # positions inside a 3-month block
    assert (d[:, inside] == 1).all()                                               # consecutive months inside blocks
    R = np.random.default_rng(1).normal(size=(60, 2)); R[:, 1] = R[:, 0]           # two perfectly co-moving tickers
    S = R[idx]
    assert np.array_equal(S[:, :, 0], S[:, :, 1])                                  # same sampled months for all tickers
    assert mc.month_indices(60, 36, 3, 200, 5) is idx                              # same draws shared across scenarios (paired)


def test_mc_ruin_and_drawdown_logic():
    ret = np.array([-0.6, -0.6, 0.0]); low = ret.copy()
    idx = np.zeros((10, 3), dtype=np.int32)                                        # always month 0 (-60%) three times
    r = mc.simulate(ret, low, idx, 1e5, 500)
    assert r["p_dd50"] == 1.0 and r["p_dd90"] == 1.0 and r["p_below_start"] == 1.0
    ok = mc.simulate(np.array([0.01]), np.array([0.0]), np.zeros((5, 12), np.int32), 1e5, 500)
    assert ok["p_broke"] == 0 and ok["p_dd50"] == 0 and ok["final_p50"] > 1e5 and math.isnan(ok["median_months_to_ruin"])


# ---------- analytic table
def test_consecutive_loss_table():
    t = ways.consecutive_losses_table().set_index("allocation")
    assert t["n_rounded"].to_dict() == {5: 45, 15: 14, 20: 10, 50: 3, 100: 1}
    assert t.loc[15, "n_exact"] == pytest.approx(math.log(0.1) / math.log(0.85))
    assert t.loc[50, "n_to_reach"] == 4                                            # 3 losses leaves 12.5% - strictly 4 to pass -90%


# ---------- breaking point finder on a hand-built log
def test_breaking_point_known_answer():
    vd = vd_from([{"entry": "2020-01-06", "exit": "2020-02-07", "coll": 1000, "pnl": -1000, "reason": "stop", "id": "BAD"}],
                 {"BAD": {"2020-01-06": 0, "2020-01-20": -500}})
    cfg = ReplayConfig(apy=0.0, expiry_cap=None)
    r = stress.breaking_point(vd, np.ones(1, bool), cfg, dd_limit=0.5)
    assert r["found"] and r["allocation"] == 51                                    # loss = A x equity -> first A with dd > 50% is 51%
    assert r["causes"][0]["trade_id"] == "BAD"
    r2 = stress.breaking_point(vd, np.ones(1, bool), cfg, dd_limit=0.5, bisect=True)
    assert r2["allocation"] == 51
    none = stress.breaking_point(vd, np.ones(1, bool), cfg, dd_limit=1.5, a_max=60)
    assert not none["found"]


# ---------- stress slippage through stop
def test_slippage_through_stop_detected():
    rows = [{"entry": "2020-02-20", "exit": "2020-03-01", "coll": 500, "pnl": -297.2, "fees": 2.8, "reason": "stop", "credit": 100, "id": "A", "profile": "stress_x", "tag": "stress"},
            {"entry": "2020-02-21", "exit": "2020-03-02", "coll": 500, "pnl": -200, "fees": 2.8, "reason": "stop", "credit": 100, "id": "B", "profile": "stress_x", "tag": "stress"}]
    t = mk_trades(rows)
    t["width"] = 5
    tab = stress.stress_table(t, pd.DataFrame(columns=["uid", "date", "mark_pnl_per_contract"]), {"r1": {"stop_multiple": 2.0, "stop_floor_width_frac": 0.0}})
    assert tab.loc[0, "stops"] == 2 and tab.loc[0, "stops_through_trigger"] == 1  # A lost $300 vs a $200 stop; B filled at the stop level
    assert "1 of 2" in tab.loc[0, "slippage_through_stop"]


# ---------- management scorecard
def test_scorecard_pairs_on_entries_and_labels():
    base, rule = [], []
    for i in range(12):
        d = f"2020-{1 + i // 2:02d}-{3 + (i % 2) * 10:02d}"
        base.append({"entry": d, "exit": d[:8] + "28", "coll": 500, "pnl": -100 if i % 3 == 0 else 20, "id": f"B{i}", "tag": "baseline"})
        rule.append({"entry": d, "exit": d[:8] + "20", "coll": 500, "pnl": -40 if i % 3 == 0 else 20, "id": f"R{i}", "tag": "mgmt_x",
                     "reason": "stop" if i % 3 == 0 else "expiry", "cf": -100 if i % 3 == 0 else 20})
    t = pd.concat([mk_trades(base), mk_trades(rule)], ignore_index=True)
    sc = mgmt.scorecard(t)
    row = sc["table"].iloc[0]
    assert sc["n_configs"] == 1 and sc["baseline"] == "baseline" and row["matched_entries"] == 12
    assert row["saved_usd"] == pytest.approx(60 * 4) and row["cost_usd"] == 0 and row["whipsaw_rate"] == 0
    assert row["label"] == "holds up" and row["first_half"] > 0 and row["second_half"] > 0
