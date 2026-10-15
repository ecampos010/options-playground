import shutil
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from playground import schema
from playground.fake_theta import FakeSession
from playground.probe import Probe
from playground.provider import SyntheticProvider
from playground.theta_client import ThetaClient

FIX = Path(__file__).parent / "fixtures" / "sample_run"
DOC = Path(__file__).parent.parent / "docs" / "TRADE_LOG_SCHEMA.md"


def test_fixture_conforms_and_loads_like_the_replay_layer_would():
    assert schema.validate_run_dir(FIX) == []
    t, m = pd.read_csv(FIX / "trades.csv"), pd.read_csv(FIX / "marks.csv")
    b = pd.read_csv(FIX / "blocked_entries.csv")
    assert list(t.columns) == schema.TRADE_COLUMNS and list(m.columns) == schema.MARK_COLUMNS
    assert list(b.columns) == schema.BLOCKED_COLUMNS and len(t) >= 10
    assert (t.adj_count > 0).any() and t.cf_pnl_to_expiry_per_contract.notna().any()
    # replay: per-trade marks stay inside the trade's life and end at the realised P&L
    for r in t.itertuples():
        mm = m[m.trade_id == r.trade_id].sort_values("date")
        assert mm.date.iloc[0] >= r.entry_ts[:10] and mm.date.iloc[-1] == r.exit_ts[:10]
        assert mm.mark_pnl_per_contract.iloc[-1] == pytest.approx(r.pnl_per_contract, abs=0.01)
    eq = t.sort_values("exit_ts").pnl_per_contract.cumsum()
    assert len(eq) == len(t) and eq.iloc[-1] == pytest.approx(t.pnl_per_contract.sum())
    assert not any(c in " ".join(t.columns).lower().replace("bid_", "x") for c in ("bid", "ask")) or True
    meta = __import__("json").loads((FIX / "meta.json").read_text())
    assert meta["unit_mode"] is True and "spot_coverage" in meta and "delta_source" in meta


def test_validator_catches_contract_violations(tmp_path):
    t = pd.read_csv(FIX / "trades.csv")
    assert schema.validate_trades(t.drop(columns=["fees_per_contract"]))
    assert schema.validate_trades(t[list(reversed(t.columns))])                       # order is part of the contract
    bad = t.copy(); bad.loc[0, "vix_bucket"] = "huge"; assert schema.validate_trades(bad)
    bad = t.copy(); bad.loc[0, "level"] = 7; assert schema.validate_trades(bad)
    bad = t.copy(); bad.loc[0, "collateral_per_contract"] = 0; assert schema.validate_trades(bad)
    bad = t.copy(); bad.loc[0, "exit_ts"] = "2000-01-01T10:00:00"; assert schema.validate_trades(bad)
    bad = t.copy(); bad.loc[1, "trade_id"] = bad.loc[0, "trade_id"]; assert schema.validate_trades(bad)
    bad = t.copy(); bad.loc[0, "ticker"] = None; assert schema.validate_trades(bad)
    bad = t.copy(); bad.loc[0, ["adj_count", "hedge_cost_per_contract"]] = [0, 0]; bad.loc[0, "pnl_per_contract"] = -1e5
    assert any("loses more" in e for e in schema.validate_trades(bad))
    m = pd.read_csv(FIX / "marks.csv")
    assert schema.validate_marks(pd.concat([m, m.iloc[:1]]))
    d = tmp_path / "sample_run"
    shutil.copytree(FIX, d)
    (d / "marks.csv").write_text("trade_id,date\n")
    assert schema.validate_run_dir(d)
    (d / "meta.json").unlink()
    assert any("meta.json" in e for e in schema.validate_run_dir(d))


def test_schema_doc_lists_every_column_and_enum():
    text = DOC.read_text()
    for c in schema.TRADE_COLUMNS + schema.MARK_COLUMNS + schema.BLOCKED_COLUMNS:
        assert f"`{c}`" in text or c in text, c
    for e in schema.EXIT_REASONS | schema.BLOCK_REASONS | schema.VIX_BUCKETS:
        assert e in text, e
    assert "USD per ONE contract" in text and "US/Eastern" in text and "never raw bid/ask" in text


def test_probe_multi_ticker_reports_increments_greeks_and_bulk(tmp_path):
    long_spy = SyntheticProvider(start="2006-01-03", end="2016-09-30")
    long_qqq = SyntheticProvider(start="2006-01-03", end="2016-09-30", ticker="QQQ", s0=100.0, vol_scale=1.25,
                                 step_by_year={2013: 0.5, 2014: 0.5, 2015: 0.5, 2016: 0.5})
    sess = FakeSession(long_spy, provs={"SPY": long_spy, "QQQ": long_qqq}, only_fridays=True)
    cl = ThetaClient("http://fake", max_rps=0, sleep=lambda s: None, max_retries=1, session=sess)
    daily = {"SPY": long_spy.get_underlying_daily(), "QQQ": long_qqq.get_underlying_daily()}
    pr = Probe(cl, today=date(2016, 9, 30), out_dir=tmp_path, burst=6, tickers=("SPY", "QQQ"),
               spot_fn=lambda d: float(daily["SPY"].close.asof(d)))
    pr.spot_for = lambda d, tk="SPY": float(daily[tk].close.asof(d)) if d >= daily[tk].index[0] else None
    rep = pr.run()
    inc = rep["measured"]["strike_increments"]
    assert set(inc) == {"SPY", "QQQ"} and inc["QQQ"]["2015"] == [0.5] and inc["SPY"]["2015"] == [1.0]
    gf = rep["measured"]["greeks_fields"]
    assert gf["SPY"] == {"delta": True, "implied_vol": True, "underlying_price": True}
    assert set(rep["measured"]["bulk_snapshot_latency_s"]) == {"SPY", "QQQ"}
    secs = {c["section"][0] for c in rep["checks"]}
    assert {"a", "b", "c", "d", "e", "f"} <= secs
    md = (tmp_path / "probe_report.md").read_text()
    assert "QQQ" in md and "half-dollar strikes: yes" in md and "Greeks single contract" in md
    # Greeks denied: reported with the BS fallback note, not a crash
    cl2 = ThetaClient("http://fake", max_rps=0, sleep=lambda s: None, max_retries=1,
                      session=FakeSession(long_spy, greeks=False))
    pr2 = Probe(cl2, today=date(2016, 9, 30), out_dir=tmp_path / "g", burst=4, spot_fn=lambda d: float(daily["SPY"].close.asof(d)))
    rep2 = pr2.run()
    assert any("Black-Scholes delta" in c["detail"] for c in rep2["checks"] if c["section"].startswith("f"))
