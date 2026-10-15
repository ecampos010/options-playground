"""Spot sources: coverage, half days, Alpaca client (fake session), routing/fallbacks, parity, validation stats."""
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from helpers import cfg, market  # noqa: E402

from playground import spot as sp  # noqa: E402
from playground.exdiv import fetch_exdiv  # noqa: E402
from playground.lab import run_lab  # noqa: E402
from playground.provider import MissingData  # noqa: E402

D = date


def test_half_days_calendar():
    assert D(2019, 11, 29) in sp.half_days(2019) and D(2019, 12, 24) in sp.half_days(2019)
    assert D(2019, 7, 3) in sp.half_days(2019)                  # July 4 is a Thursday
    assert D(2020, 7, 3) not in sp.half_days(2020)              # July 3 2020 is the observed holiday, not a half day
    assert D(2021, 12, 24) not in sp.half_days(2021)            # Christmas Eve 2021 is a holiday (observed)
    assert sp.min_bars_needed(D(2019, 11, 29)) < sp.MIN_BARS == 330 == sp.min_bars_needed(D(2019, 11, 27))


def day_bars(n, d=D(2019, 11, 27), base=100.0):
    m = list(range(570, 570 + n))
    return pd.DataFrame({"ts": [pd.Timestamp(d) + pd.Timedelta(minutes=x) for x in m], "m": m,
                         "px": [base + 0.001 * i for i in range(n)], "c": [base + 0.001 * i + 0.0005 for i in range(n)]})


def test_coverage_flags_low_missing_and_accepts_half_days(tmp_path):
    st = sp.SpotStore(tmp_path)
    st.write_day("SPY", D(2019, 11, 25), day_bars(390, D(2019, 11, 25)))
    st.write_day("SPY", D(2019, 11, 26), day_bars(329, D(2019, 11, 26)))
    st.write_day("SPY", D(2019, 11, 29), day_bars(210, D(2019, 11, 29)))       # half day: fine
    cov = sp.coverage(st, "SPY", D(2019, 11, 25), D(2019, 11, 29)).set_index("date")
    assert cov.loc[D(2019, 11, 25), "status"] == "ok" and cov.loc[D(2019, 11, 26), "status"] == "LOW"
    assert cov.loc[D(2019, 11, 27), "status"] == "missing" and cov.loc[D(2019, 11, 29), "status"] == "ok"
    sp.write_coverage_md({"SPY": cov.reset_index()}, tmp_path / "cov.md")
    md = (tmp_path / "cov.md").read_text()
    assert "flagged: 2" in md and "never interpolated" in md


def test_regular_session_timezone_conversion():
    df = pd.DataFrame({"t": pd.to_datetime(["2019-07-01T13:29:00Z", "2019-07-01T13:30:00Z", "2019-07-01T19:59:00Z",
                                            "2019-07-01T20:00:00Z", "2019-12-02T14:30:00Z"], utc=True),
                       "o": [1, 2, 3, 4, 5], "h": 0, "l": 0, "c": [1, 2, 3, 4, 5], "v": 1})
    rs = sp.regular_session(df)
    assert list(rs.m) == [570, 959, 570] and list(rs.px) == [2, 3, 5]      # EDT and EST both map to 09:30 ET


def test_agreement_stats():
    a = pd.Series([100.0, 100.1, 100.3, 100.0], index=[1, 2, 3, 4])
    b = pd.Series([100.0, 100.0, 100.0, 99.0], index=[1, 2, 3, 4])
    st = sp.agreement_stats(a, b)
    assert st["n"] == 4 and st["median_abs"] == pytest.approx(0.2, abs=1e-9)
    assert st["within_005"] == 25.0 and st["within_025"] == 50.0 and st["p95_abs"] > 0.8
    assert sp.agreement_stats(pd.Series(dtype=float), pd.Series(dtype=float))["n"] == 0


class FakeAlpacaSession:
    def __init__(self):
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, dict(params or {}), dict(headers or {})))

        class R:
            status_code, text = 200, "ok"

            def __init__(s, js):
                s._js = js

            def json(s):
                return s._js
        if params.get("timeframe") == "1Day":
            return R({"bars": {"SPY": [{"t": "2019-11-26T05:00:00Z", "o": 1, "h": 2, "l": 1, "c": 2, "v": 5}]}})
        if params.get("page_token") is None:
            bars = [{"t": f"2019-11-26T14:{30 + i}:00Z", "o": 100 + i, "h": 0, "l": 0, "c": 100 + i, "v": 1} for i in range(5)]
            return R({"bars": {"SPY": bars}, "next_page_token": "p2"})
        bars = [{"t": f"2019-11-26T14:{35 + i}:00Z", "o": 105 + i, "h": 0, "l": 0, "c": 105 + i, "v": 1} for i in range(5)]
        return R({"bars": {"SPY": bars}, "next_page_token": None})


def test_alpaca_client_paginates_downloads_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "KEYID-123")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "SECRET-456")
    sess = FakeAlpacaSession()
    cl = sp.AlpacaClient(sess, per_min=10_000)
    st = sp.SpotStore(tmp_path)
    out = sp.download_spot(cl, st, ["SPY"], D(2019, 11, 26), D(2019, 11, 26), log=lambda *a: None)
    assert out["SPY"]["days_with_bars"] == 1
    minute_calls = [c for c in sess.calls if c[1]["timeframe"] == "1Min"]
    assert len(minute_calls) == 2 and minute_calls[0][1]["feed"] == "sip" and minute_calls[0][1]["adjustment"] == "raw"
    assert {c[1]["adjustment"] for c in sess.calls if c[1]["timeframe"] == "1Day"} == {"raw", "all"}
    assert len(st.read_day("SPY", D(2019, 11, 26))) == 10 and st.load_daily("SPY", "adj") is not None
    n = len(sess.calls)
    sp.download_spot(cl, st, ["SPY"], D(2019, 11, 26), D(2019, 11, 26), log=lambda *a: None)
    assert len([c for c in sess.calls[n:] if c[1]["timeframe"] == "1Min"]) == 0          # resumable: no refetch


def test_keys_never_leak_into_errors(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "KEYID-123")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "SECRET-456")

    class Bad:
        def get(self, *a, **k):
            class R:
                status_code, text = 401, "forbidden"
            return R()
    with pytest.raises(RuntimeError) as e:
        sp.AlpacaClient(Bad(), per_min=10_000).bars("SPY", "2019-01-01T00:00:00Z", "2019-01-02T00:00:00Z")
    assert "KEYID-123" not in str(e.value) and "SECRET-456" not in str(e.value) and "401" in str(e.value)
    monkeypatch.delenv("ALPACA_API_KEY")
    with pytest.raises(RuntimeError, match="not set"):
        sp.AlpacaClient(Bad(), per_min=10_000).bars("SPY", "a", "b")


def test_alpaca_spot_grid_close_print_and_bad_days(tmp_path):
    st = sp.SpotStore(tmp_path)
    d = D(2019, 11, 26)
    st.write_day("SPY", d, day_bars(390, d))
    a = sp.AlpacaSpot("SPY", st)
    b = a.bars_alpaca(d, "15m")
    assert b.index[0] == pd.Timestamp("2019-11-26 09:30") and b.index[-1] == pd.Timestamp("2019-11-26 16:00")
    assert b.spot.iloc[0] == pytest.approx(100.0) and b.spot.iloc[-1] == pytest.approx(100.0 + 0.001 * 389 + 0.0005)
    st.set_bad_days("SPY", {str(d)})
    with pytest.raises(MissingData, match="bad"):
        a.bars_alpaca(d, "15m")
    with pytest.raises(MissingData, match="missing"):
        a.bars_alpaca(D(2019, 11, 27), "15m")


def test_parity_spot_matches_true_spot_and_routing_fallbacks():
    p = market(lambda d, k: 100.0 + 0.1 * (k % 7), vix=18.0)         # parity-consistent chain (no skew), wiggly spot
    d = D(2016, 3, 1)
    truth = p.get_underlying_bars(d, "15m")["spot"]
    pb = sp.parity_bars(p, d, "15m", 0.04, 100.0)
    diff = (pb["spot"] - truth.reindex(pb.index)).abs()
    assert len(pb) > 20 and diff.median() < 0.08 and diff.max() < 0.2          # within the quoted spread
    bars, used = p.spot_bars(d, "15m", "parity")
    assert used == "parity"
    bars, used = p.spot_bars(d, "15m", "daily")
    assert used == "daily" and len(bars) == 1 and bars.index[0].hour == 16
    p.spot_gaps = {d}
    bars, used = p.spot_bars(d, "15m", "alpaca", "parity")
    assert used == "parity"                                                    # coverage failure -> parity fallback
    with pytest.raises(MissingData):
        p.spot_bars(d, "15m", "alpaca", "none")
    p.spot_gaps = set()


def test_engine_spot_sources_daily_parity_and_missing_days_block_entries():
    p = market(lambda d, k: 100.0, vix=20.0)
    c = cfg()
    r = run_lab(p, c, None, "", None)
    assert set(r.trades.spot_source) == {"synthetic"}
    r = run_lab(p, cfg(spot_source="daily"), None, "", None)
    assert len(r.trades) > 0 and set(r.trades.spot_source) == {"daily"} and r.trades.entry_ts.str.endswith("16:00:00").all()
    r = run_lab(p, cfg(spot_source="parity"), None, "", None)
    assert len(r.trades) > 0 and set(r.trades.spot_source) == {"parity"}
    # a day without spot coverage and no fallback: entry blocked and logged, never interpolated
    q = market(lambda d, k: 100.0, vix=20.0)
    first_day = run_lab(q, cfg(), None, "", None).trades.entry_ts.iloc[0][:10]
    q.spot_gaps = {D.fromisoformat(first_day)}
    r = run_lab(q, cfg(spot_fallback="none"), None, "", None)
    assert ((r.blocked.date == first_day) & (r.blocked.reason == "nodata")).any()
    assert first_day not in set(r.trades.entry_ts.str[:10])
    assert r.meta["spot_coverage"]["days_without_spot"] >= 1
    r2 = run_lab(q, cfg(spot_fallback="parity"), None, "", None)                 # fallback keeps the trade, labelled parity
    assert r2.trades.entry_ts.str[:10].eq(first_day).any()
    assert r2.trades[r2.trades.entry_ts.str[:10] == first_day].spot_source.iloc[0] == "parity"


class FakeMassive:
    def __init__(self, offset, store):
        self.offset, self.store = offset, store

    def minute_opens(self, t, d):
        a = sp.alpaca_opens(self.store, t, d)
        return None if a is None else a + self.offset


def test_massive_validation_stats_and_bad_day_marking(tmp_path, monkeypatch):
    st = sp.SpotStore(tmp_path)
    days = [D(2019, 11, 25), D(2019, 11, 26)]
    for d in days:
        st.write_day("SPY", d, day_bars(390, d))
    stats, per, got = sp.validate_massive(st, FakeMassive(0.01, st), "SPY", days)
    assert stats["n"] == 780 and stats["median_abs"] == pytest.approx(0.01) and stats["within_005"] == 100.0 and len(got) == 2
    stats, per, got = sp.validate_massive(st, FakeMassive(0.40, st), "SPY", days)
    assert stats["within_025"] == 0.0

    class Args:
        validate, sample_days, mark_bad, data_dir = "massive", 2, True, str(tmp_path)
    monkeypatch.setattr(sp, "MassiveClient", lambda: FakeMassive(0.40, st))
    # dates before 2024-10-16 are outside Massive's free-tier reach: skipped with a note, never queried
    rows0, notes0 = sp.run_validations(Args, st, ["SPY"], days[0], days[-1], None)
    assert rows0 == [] and any("Massive cross-check skipped" in n for n in notes0) and st.bad_days("SPY") == set()
    days = [D(2025, 11, 24), D(2025, 11, 25)]
    for d in days:
        st.write_day("SPY", d, day_bars(390, d))
    rows, notes = sp.run_validations(Args, st, ["SPY"], days[0], days[-1], None)
    assert rows and "Alpaca vs Massive" in rows[0] and st.bad_days("SPY") == {str(d) for d in days}
    sp.write_validation_md(rows, tmp_path / "v.md", notes)
    assert "within $0.25" in (tmp_path / "v.md").read_text()


def test_exdiv_without_any_source_warns(monkeypatch):
    import builtins
    real = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "yfinance":
            raise ImportError
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    msgs = []
    ds, src = fetch_exdiv("SPY", D(2019, 1, 1), D(2019, 12, 31), None, log=msgs.append)
    assert ds == [] and src == "none" and any("WARNING" in m for m in msgs)
