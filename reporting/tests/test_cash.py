"""Cash interest: daily fed funds accrual, flat fallback, with/without interest, vs cash."""
import numpy as np
import pandas as pd
import pytest

from reporting.cash import RateTable
from reporting.replay import ReplayConfig, replay

from .conftest import vd_from

DAYS = pd.DatetimeIndex(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"])      # Thu, Fri, Mon, Tue


def table(vals, start="2019-12-01", end="2020-02-29", flat=0.036):
    idx = pd.date_range(start, end)
    return RateTable(pd.Series(vals if np.ndim(vals) else np.full(len(idx), vals), index=idx), flat)


def test_missing_file_is_flat_fallback_and_says_so(tmp_path):
    t = RateTable.load(tmp_path / "nope.csv", 0.036)
    assert t.is_flat and "FLAT FALLBACK 3.6%" in t.describe() and "not found" in t.describe()
    f = t.factors(DAYS)
    assert f[0] == 1 and f[1] == pytest.approx(1.036 ** (1 / 365)) and f[2] == pytest.approx(1.036 ** (3 / 365))   # same maths as before
    assert RateTable.load(None, 0.036).is_flat


def test_daily_rate_accrues_actual_360_and_weekend_compounds():
    t = table(3.6)
    f = t.factors(DAYS)
    one = 1 + 0.036 / 360
    assert f[0] == 1 and f[1] == pytest.approx(one) and f[2] == pytest.approx(one ** 3) and f[3] == pytest.approx(one)
    assert t.cash_curve(DAYS, 100000.0)[-1] == pytest.approx(100000 * one ** 5)
    assert "daily effective fed funds rate" in t.describe(DAYS[0], DAYS[-1])


def test_rate_change_uses_the_rate_in_effect_each_night_and_carries_forward():
    idx = pd.date_range("2019-12-01", "2020-01-02")                      # file ends on the first day: later days carry the last rate
    vals = np.where(idx < "2020-01-01", 1.0, 5.0)
    t = RateTable(pd.Series(vals, index=idx), 0.036)
    f = t.factors(DAYS)
    assert f[1] == pytest.approx(1 + 0.05 / 360)                         # night of Jan 2 -> Jan 3 at 5%
    assert f[2] == pytest.approx((1 + 0.05 / 360) ** 3)                  # carried forward


def test_days_before_the_file_use_flat_and_coverage_reports_it():
    t = RateTable(pd.Series([2.0] * 3, index=pd.date_range("2020-01-04", periods=3)), 0.036)
    cov = t.coverage("2020-01-02", "2020-01-07")
    assert 0 < cov["share_real"] < 1 and "uses flat 3.6%" in t.describe("2020-01-02", "2020-01-07")


def test_regime_labels():
    idx = pd.date_range("2020-01-01", "2020-06-30")
    t = RateTable(pd.Series(np.where(idx < "2020-03-15", 1.5, 0.1), index=idx), 0.036)
    r = t.regime("2020-03-01", "2020-03-31")
    assert r["trend"] == "cutting" and r["level"] == "near zero" and r["change_bp"] == pytest.approx(-140)
    assert t.regime("2020-04-01", "2020-04-30")["trend"] == "on hold"


def test_replay_with_and_without_interest_on_idle_cash():
    vd = vd_from([{"entry": "2020-01-06", "exit": "2020-02-07", "coll": 1000, "pnl": 100}])
    rt = table(3.6)
    on = replay(vd, np.ones(1, bool), 0.10, ReplayConfig(apy=0.036, rates=rt, expiry_cap=None))
    off = replay(vd, np.ones(1, bool), 0.10, ReplayConfig(apy=0.0, rates=None, expiry_cap=None))
    assert off.metrics["final_equity"] == pytest.approx(100000 + 100 * 10)          # 10 contracts x $100, no interest
    assert on.metrics["final_equity"] > off.metrics["final_equity"]
    # interest accrues only on cash that is not reserved as collateral: 90k idle + 10k reserved while open
    days = vd.days
    f = rt.factors(days)
    cash = 100000.0
    reserved = 0.0
    for d in range(len(days)):
        if d and cash > reserved:
            cash += (cash - reserved) * (f[d] - 1)
        if days[d] == pd.Timestamp("2020-01-06"):
            reserved = 10000.0
        if days[d] == pd.Timestamp("2020-02-07"):
            cash += 1000.0
            reserved = 0.0
    assert on.metrics["final_equity"] == pytest.approx(cash, rel=1e-9)


def test_grid_columns_with_without_interest_and_vs_cash(report_ctx_dir):
    res = pd.read_csv(report_ctx_dir / "results.csv")
    need = {"final_equity_noint", "cagr_noint", "max_dd_noint", "interest_usd", "cash_final", "cash_cagr", "excess_cagr", "beats_cash"}
    assert need <= set(res.columns)
    ok = res[~res["broke"]]
    assert (ok["final_equity"] >= ok["final_equity_noint"] - 1.0).mean() > 0.9         # 1.5% rates: interest helps (sizing paths can differ slightly)
    assert (ok["cash_final"] > 100000).all()
    assert np.allclose(ok["excess_cagr"], ok["cagr"] - ok["cash_cagr"])
    assert (ok["beats_cash"] == (ok["final_equity"] > ok["cash_final"])).all()
    assert not res[res["broke"]]["beats_cash"].any()


def test_header_says_flat_fallback_when_rates_missing_and_real_rates_when_present(report_dir, report_ctx_dir):
    flat = (report_dir / "index.html").read_text()
    real = (report_ctx_dir / "index.html").read_text()
    assert "FLAT FALLBACK 3.6% APY" in flat and "not found" in flat
    assert "daily effective fed funds rate from" in real and "FLAT FALLBACK" not in real


def test_html_has_vs_cash_column_no_interest_series_and_cash_curve(report_ctx_dir):
    page = (report_ctx_dir / "index.html").read_text()
    assert "vs cash (CAGR pts)" in page and "Final, without interest" in page and "without interest" in page
    import json, re
    data = json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>', page, re.S).group(1))
    assert data["cash"] and data["meta"]["cash_label"].startswith("Cash @ fed funds")
    assert all(len(p) == 3 for p in data["series"])                     # equity, drawdown, equity without interest
    r = data["rows"][0]
    assert {"fn", "cn", "xc", "bc", "cf", "cc"} <= set(r)


def test_playbooks_require_beating_cash_and_show_both_returns(report_ctx_dir):
    import json
    pb = json.loads((report_ctx_dir.parent / "pb.json").read_text())
    assert pb["pool"]["require_beat_cash"] is True and "excluded_below_cash" in pb["pool"]
    ok = [i for i in pb["playbooks"] + pb["awards"] if i["status"] == "ok" and i["key"] != "big_short"]
    assert ok
    for it in ok:
        assert it["final_equity_noint"] is not None and it["cash_total_return"] is not None
        assert "without it the return would have been" in it["paragraph"] and "cash-only account" in it["paragraph"]
        if it["key"] != "worst_performance":                     # the worst scenario may legitimately sit below cash
            assert it["beats_cash"] is True and it["excess_cagr"] > 0
    md = (report_ctx_dir / "playbooks.md").read_text()
    assert "beat cash by" in md
    html = (report_ctx_dir / "index.html").read_text()
    assert "Return, no interest" in html and "vs cash" in html
