import json

import numpy as np
import pandas as pd
import pytest

from reporting import playbooks as PB
from reporting.schema import load_runs

CFG = PB.load_config()
D = pd.bdate_range("2016-10-03", periods=40)


def row(sid, **kw):
    r = dict(scenario_id=sid, profile="green", config_tag="baseline", strategy="iron_condor", tickers="SPY", allocation=15,
             level_tag="Green / Lvl 1", zone="Safe", final_equity=120000.0, cagr=0.02, max_dd=0.10, win_rate=0.8, ruin_odds=0.001,
             n_trades=60, calmar=0.2, broke=False, broke_date="", too_few_trades=False)
    r.update(kw)
    return r


def series_for(df, growth=None):
    out = {}
    for r in df.itertuples():
        eq = np.linspace(100000.0, r.final_equity, len(D)) if growth is None else growth
        out[r.scenario_id] = {"eq": eq, "days": D}
    return out


def world():
    rows = [
        row("safe_boring", zone="Safe", cagr=0.03, max_dd=0.05, calmar=0.6, final_equity=130000),
        row("safe_better", zone="Safe", cagr=0.04, max_dd=0.08, calmar=0.5, final_equity=140000, win_rate=0.9),
        row("safe_tiny_dd", zone="Safe", cagr=0.01, max_dd=0.02, calmar=0.5, final_equity=110000, ruin_odds=0.0),
        row("safe_loss", zone="Safe", cagr=-0.01, max_dd=0.01, calmar=-1, final_equity=95000),
        row("safe_ruiny", zone="Safe", cagr=0.02, max_dd=0.01, calmar=2, final_equity=120000, ruin_odds=0.009),
        row("mild_a", zone="Mild", cagr=0.07, max_dd=0.20, calmar=0.35, final_equity=190000, allocation=20, level_tag="Experiment / Lvl 4", ruin_odds=0.03),
        row("mild_b", zone="Mild", cagr=0.06, max_dd=0.12, calmar=0.50, final_equity=170000, ruin_odds=0.02),
        row("mild_c", zone="Mild", cagr=0.05, max_dd=0.25, calmar=0.20, final_equity=150000, ruin_odds=0.04),
        row("danger_g", zone="Danger", cagr=0.12, max_dd=0.45, calmar=0.27, final_equity=300000, ruin_odds=0.15),
        row("breaking_big", zone="Breaking point", cagr=0.30, max_dd=0.70, calmar=0.43, final_equity=900000, ruin_odds=0.4, win_rate=0.99),
        row("few_trades", zone="Safe", cagr=0.50, max_dd=0.01, calmar=50, final_equity=5000000, n_trades=12, win_rate=1.0, too_few_trades=True),
        row("broke_one", zone="Breaking point", cagr=-1.0, max_dd=1.0, calmar=-1, final_equity=0.0, broke=True, broke_date="2020-03-16",
            strategy="iron_butterfly", config_tag="alloff", profile="experiment_all_off"),
        row("long_put_stress", strategy="long_put", config_tag="lp", zone="Danger", final_equity=70000, cagr=-0.03, max_dd=0.35, win_rate=0.2,
            ruin_odds=0.0, calmar=-0.1, n_trades=40),
    ]
    df = pd.DataFrame(rows)
    return df, series_for(df)


def trades_for(df):
    t = []
    for i in range(6):
        t.append(dict(profile="experiment_all_off", config_tag="alloff", strategy="iron_butterfly", ticker="SPY",
                      pnl_per_contract=-900.0 if i == 2 else 50.0, exit_reason="stop" if i == 2 else "tp",
                      exit_ts=f"2020-03-1{i}", collateral_per_contract=1000.0))
    return pd.DataFrame(t)


def build(df=None, series=None, stress=None, spy=None):
    if df is None:
        df, series = world()
    return PB.build(df, series, trades_for(df), stress if stress is not None else pd.DataFrame(), spy, CFG, synthetic=True)


def get(res, key):
    return next(x for x in res["playbooks"] + res["awards"] if x["key"] == key)


# ---------------------------------------------------------------- pool / thresholds from the yaml
def test_thresholds_live_in_yaml_and_pool_excludes_few_trades_and_broke():
    assert CFG["min_trades"] == 30 and CFG["playbooks"]["capital_preservation"]["filters"]["max_ruin"] == 0.005
    df, _ = world()
    pool, info = PB.pool_of(df, CFG)
    assert "few_trades" not in set(pool.scenario_id) and "broke_one" not in set(pool.scenario_id)
    assert "long_put_stress" not in set(pool.scenario_id)                      # long premium is judged separately
    assert info["excluded_too_few_trades"] == 1 and info["excluded_broke"] == 1 and info["scenarios_total"] == len(df)


def test_final_equity_ranking_is_rejected_outside_lottery():
    bad = json.loads(json.dumps(CFG))
    bad["playbooks"]["growth"]["rank"] = {"by": "final_equity", "order": "desc"}
    with pytest.raises(ValueError):
        PB.validate_config(bad)
    bad2 = json.loads(json.dumps(CFG))
    bad2["awards"]["highest_roi"]["rank"] = {"by": "final_equity", "order": "desc"}
    with pytest.raises(ValueError):
        PB.validate_config(bad2)


# ---------------------------------------------------------------- one test per selector
def test_capital_preservation_lowest_dd_positive_return_ruin_under_half_percent():
    r = get(build(), "capital_preservation")
    assert r["scenario_id"] == "safe_tiny_dd"          # safe_ruiny has lower DD but 0.9% ruin; safe_loss has negative return


def test_conservative_safe_zone_best_calmar_and_spy_comparison():
    spy = {"t": [int(d.value // 10**6) for d in pd.date_range("2016-09-30", "2016-12-30", freq="ME")], "v": [100.0, 102.0, 105.0]}
    r = get(build(spy=spy), "conservative")
    assert r["scenario_id"] == "safe_ruiny" or r["zone"] == "Safe"
    assert r["calmar"] == max(x for x in [2, 0.6, 0.5, 0.5, -1])                # best return / max drawdown among Safe
    assert "SPY" in r["spy_text"] and "buy-and-hold" in r["spy_text"]
    assert r["spy"]["total_return"] == pytest.approx(0.02)       # 100 (Sep-30) -> 102 (Oct-31), last close on or before the window end
    none = get(build(), "conservative")
    assert "not available" in none["spy_text"]


def test_moderate_mild_zone_best_calmar():
    assert get(build(), "moderate")["scenario_id"] == "mild_b"


def test_max_safe_growth_highest_cagr_within_safe_limits():
    r = get(build(), "max_safe_growth")
    assert r["scenario_id"] == "safe_better"            # mild_* breach ruin<1% / dd<15%; safe_better has the highest CAGR inside


def test_growth_dd_under_50_and_ruin_under_20():
    r = get(build(), "growth")
    assert r["scenario_id"] == "danger_g"               # breaking_big has dd 70% and ruin 40%


def test_lottery_highest_final_equity_and_is_labelled_luck_dependent():
    res = build()
    r = get(res, "lottery")
    assert r["scenario_id"] == "breaking_big" and r["luck_dependent"] and "LUCK-DEPENDENT" in r["paragraph"]
    assert r["max_dd"] == 0.70 and r["ruin_odds"] == 0.4                       # shown beside its drawdown and ruin odds
    assert "few_trades" != r["scenario_id"]


def test_awards_win_rate_roi_happy_medium_worst():
    res = build()
    assert get(res, "highest_win_rate")["scenario_id"] == "breaking_big"       # 0.99 beats few_trades (excluded)
    assert get(res, "highest_roi")["scenario_id"] == "breaking_big"
    hm = get(res, "happy_medium")
    assert hm["zone"] == "Mild" and hm["scenario_id"] in {"mild_a", "mild_b"}
    w = get(res, "worst_performance")
    assert w["scenario_id"] == "broke_one"
    assert "went broke on 2020-03-16" in w["paragraph"] and "iron_butterfly" in w["paragraph"] and "alloff" in w["paragraph"]
    assert "worst single trade lost $900" in w["paragraph"]
    assert res["ways_to_lose_link"] == "#ways"


def test_big_short_ranks_on_stress_windows_and_states_bleed():
    st = pd.DataFrame([
        dict(window="stress_feb2018_volspike", config_tag="lp", strategy="long_put", pnl_total=800.0),
        dict(window="stress_covid_crash_2020", config_tag="lp", strategy="long_put", pnl_total=2500.0),
        dict(window="stress_h1_2022_drawdown", config_tag="lp", strategy="long_put", pnl_total=300.0),
        dict(window="stress_dec2018_selloff", config_tag="lp", strategy="long_put", pnl_total=9999.0),      # not a listed window
        dict(window="stress_covid_crash_2020", config_tag="ds", strategy="put_debit_spread", pnl_total=900.0),
        dict(window="stress_covid_crash_2020", config_tag="x", strategy="iron_condor", pnl_total=-5000.0),   # not long premium
    ])
    df, series = world()
    b = get(build(df, series, stress=st), "big_short")
    assert b["status"] == "ok" and b["strategy"] == "long_put" and b["stress_total_per_contract"] == pytest.approx(3600.0)
    assert set(b["windows"]) == {"feb2018_volspike", "covid_crash_2020", "h1_2022_drawdown"}
    assert "bleed premium in calm periods and pay in crashes" in b["paragraph"] and "never naked" in b["paragraph"]
    assert b["record"]["total_return"] == pytest.approx(-0.30)
    assert get(build(), "big_short")["status"] == "none"                         # no stress logs: honest 'not run'


# ---------------------------------------------------------------- required content on every card
def test_every_playbook_shows_configs_tried_halves_and_disclaimer():
    res = build()
    html = PB.render_html(res)
    assert html.count("backtest on one 10-year history, not a promise or advice") >= len(res["playbooks"]) + len(res["awards"])
    for it in res["playbooks"][:6] + res["awards"]:
        assert it["status"] == "ok" and it["n_configs_tried"] == 13
        assert it["first_half"] is not None and it["second_half"] is not None
        assert set(["config", "allocation", "tickers", "strategy", "level_tag", "total_return", "cagr", "max_dd", "win_rate",
                    "ruin_odds", "n_trades"]) <= set(it)
    assert "Configs tried" in html and "1st half" in html and "2nd half" in html
    assert "TOO FEW TRADES" in html and "Ways to Lose Money" in html


def test_no_monte_carlo_means_ruin_gated_playbooks_say_so():
    df, series = world()
    df["ruin_odds"] = np.nan
    res = build(df, series)
    cp = get(res, "capital_preservation")
    assert cp["status"] == "none" and "Monte Carlo" in cp["reason"]
    assert get(res, "conservative")["status"] == "ok"


def test_first_half_second_half_from_equity_curve():
    eq = np.r_[np.linspace(100000, 120000, 20), np.linspace(120000, 108000, 20)]
    h = PB.halves({"s": {"eq": eq, "days": D}}, "s")
    assert h["first_half"] == pytest.approx(0.2, abs=0.02) and h["second_half"] == pytest.approx(-0.10, abs=0.02)


# ---------------------------------------------------------------- end to end on the synthetic demo logs
def test_report_has_playbooks_section_json_and_md(report_dir):
    page = (report_dir / "index.html").read_text()
    assert 'id="playbooks"' in page and "Playbooks and Awards" in page and "Capital Preservation" in page and "Lottery Ticket" in page
    assert "backtest on one 10-year history, not a promise or advice" in page
    js = json.loads((report_dir.parent / "results" / "playbooks.json").read_text())
    assert js["synthetic"] is True and {p["key"] for p in js["playbooks"]} >= {"capital_preservation", "lottery", "big_short"}
    assert all(p["status"] == "ok" for p in js["playbooks"] if p["key"] not in ("big_short", "capital_preservation", "max_safe_growth", "conservative", "moderate", "growth"))
    assert (report_dir / "playbooks.md").exists()
