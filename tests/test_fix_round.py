"""Fix round: first_order Greeks + junk-row rejection, spot cross-checks, probe spot, plan without events,
delta-source tag, progress/ETA, and max-loss-from-actual-strikes."""
import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from playground import spot as sp
from playground import theta_client as tc
from playground.cli import build_parser, get_events
from playground.config import build_config
from playground.downloader import Progress, fmt_eta
from playground.engine import Leg
from playground.lab import delta_source_tag
from playground.probe import Probe
from playground.strategies import check_max_loss, expected_max_loss_ps, max_loss_ps
from playground.theta_client import ThetaClient

HDR = ("symbol,expiration,strike,right,timestamp,bid,ask,delta,theta,vega,rho,epsilon,lambda,implied_vol,iv_error,"
       "underlying_timestamp,underlying_price")
ROW_0930 = "SPY,2025-01-17,590.000,PUT,2025-01-06T09:30:00,0.0,0.0,-0.00041,0,0,0,0,0,0.0,100.00000,2025-01-06T09:30:00,595.1"
ROW_GOOD = "SPY,2025-01-17,590.000,PUT,2025-01-06T09:31:00,1.10,1.14,-0.2100,-0.1,0.3,-0.02,0.01,40,0.1650,0.00012,2025-01-06T09:31:00,595.4"


class Sess:
    def __init__(self, text):
        self.text, self.calls = text, []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        return type("R", (), {"status_code": 200, "text": self.text})()


def client(text):
    s = Sess(text)
    return ThetaClient("http://x", max_rps=0, session=s, sleep=lambda x: None), s


# ------------------------------------------------------------------ 1 + 2: endpoint, parsing, junk rows
def test_greeks_use_first_order_endpoint_and_iso_dates():
    c, s = client("\n".join([HDR, ROW_GOOD]))
    df, r = c.greeks_bulk(date(2025, 1, 17), date(2025, 1, 6), "1m", 570, 571, "SPY", strike=590.0, right="P")
    url, p = s.calls[0]
    assert url.endswith("/v3/option/history/greeks/first_order") and "greeks/all" not in url
    assert p["expiration"] == "2025-01-17" and p["date"] == "2025-01-06" and p["right"] == "put"
    assert list(df.columns) == ["right", "strike", "delta", "iv", "underlying_price"]
    row = df.iloc[0]
    assert row.right == "P" and row.delta == pytest.approx(-0.21) and row.iv == pytest.approx(0.165)
    assert row.underlying_price == pytest.approx(595.4)


def test_0930_junk_row_is_rejected_and_counted():
    c, _ = client("\n".join([HDR, ROW_0930, ROW_GOOD]))
    df, _ = c.greeks_bulk(date(2025, 1, 17), date(2025, 1, 6), "1m", 570, 572, "SPY")
    assert len(df) == 1 and df.delta.iloc[0] == pytest.approx(-0.21)         # the -0.00041 delta never gets through
    assert c.stats["greeks_rejected"] == 1
    only_junk, _ = client("\n".join([HDR, ROW_0930]))
    df2, _ = only_junk.greeks_bulk(date(2025, 1, 17), date(2025, 1, 6), "1m", 570, 571, "SPY")
    assert df2.empty


def test_junk_mask_rules():
    base = dict(bid=1.0, ask=1.1, delta=-0.2, iv_error=0.001, timestamp="2025-01-06T10:00:00",
                underlying_timestamp="2025-01-06T10:00:00")
    rows = [dict(base), {**base, "bid": 0.0}, {**base, "ask": 0.0}, {**base, "iv_error": 100.0},
            {**base, "iv_error": -0.06}, {**base, "iv_error": 0.05}, {**base, "delta": float("nan")},
            {**base, "underlying_timestamp": "2025-01-06T09:30:00"}]
    bad = tc.junk_greeks_mask(pd.DataFrame(rows)).tolist()
    assert bad == [False, True, True, True, True, False, True, True]
    assert tc.is_junk_quote(0.0, 0.0) and tc.is_junk_quote(0.0, 1.0) and tc.is_junk_quote(1.0, 0.0)
    assert not tc.is_junk_quote(1.0, 1.1) and tc.is_junk_quote(1.2, 1.1)


def test_option_quote_no_market_rows_dropped():
    hdr = "timestamp,bid_size,bid,ask_size,ask"
    txt = "\n".join([hdr, "2025-01-06T09:30:00,0,0.0,0,0.0", "2025-01-06T09:31:00,10,1.0,10,1.1", "2025-01-06T09:32:00,0,0.0,5,0.05"])
    c, _ = client(txt)
    df, _ = c.option_quotes(date(2025, 1, 17), "P", 590.0, date(2025, 1, 6), date(2025, 1, 6), "1m")
    assert len(df) == 2 and (df.ts.dt.minute == 30).sum() == 0               # bid==ask==0 dropped, one-sided wing kept


# ------------------------------------------------------------------ 3: spot cross-checks
def bars(d, closes):
    m = list(range(570, 570 + len(closes)))
    return pd.DataFrame({"ts": [pd.Timestamp(d) + pd.Timedelta(minutes=x) for x in m], "m": m, "px": closes, "c": closes})


def test_massive_dates_limited_to_free_tier_reach():
    ds = [date(2024, 10, 15), date(2024, 10, 16), date(2025, 3, 3)]
    assert sp.massive_days(ds) == ds[1:] and sp.MASSIVE_FLOOR == date(2024, 10, 16)


def test_daily_close_check_flags_diffs_over_tolerance(tmp_path):
    st = sp.SpotStore(tmp_path)
    d1, d2, d3 = date(2025, 1, 6), date(2025, 1, 7), date(2025, 1, 8)
    st.write_day("SPY", d1, bars(d1, [100.0, 100.5, 101.00]))
    st.write_day("SPY", d2, bars(d2, [100.0, 100.5, 101.10]))          # last bar close 101.10 vs yf 101.00 -> 0.10
    st.write_day("SPY", d3, bars(d3, [100.0, 100.5, 101.02]))          # 0.02 within tolerance
    yf = pd.Series({d1: 101.00, d2: 101.00, d3: 101.00})
    out = sp.daily_close_check(st, "SPY", [d1, d2, d3, date(2025, 1, 9)], yf)
    assert out["n"] == 3 and out["missing_alpaca"] == 1
    assert [f[0] for f in out["flagged"]] == ["2025-01-07"] and out["flagged"][0][3] == pytest.approx(0.10)


def test_side_by_side_note_shows_both_checks():
    a = {"n": 5, "median_abs": 0.02, "p95_abs": 0.1}
    b = {"n": 3, "median_abs": 0.05, "p95_abs": 0.2}
    note = sp.side_by_side_note("SPY", a, b)
    assert "parity" in note and "0.0200" in note and "underlying_price" in note and "0.0500" in note


# ------------------------------------------------------------------ 5: probe spot from Alpaca minute bars
def test_probe_spot_uses_alpaca_minute_bars_when_no_daily(tmp_path):
    st = sp.SpotStore(tmp_path / "spot")
    st.write_day("SPY", date(2025, 1, 6), bars(date(2025, 1, 6), [600.0, 601.0, 602.5]))
    p = Probe(ThetaClient("http://x", max_rps=0, session=Sess("")), data_dir=str(tmp_path), out_dir=str(tmp_path / "o"))
    assert p.spot_for(date(2025, 1, 7), "SPY") == pytest.approx(602.5)       # nearest earlier day, last bar close


# ------------------------------------------------------------------ 6: plan without events.csv
def test_get_events_missing_file_warns_when_not_required(tmp_path, capsys):
    args = build_parser().parse_args(["plan", "--events", str(tmp_path / "nope.csv"), "--data-dir", str(tmp_path)])
    cfg = build_config("green", {"event_buffer": True})
    prov = type("P", (), {"name": "theta-cache"})()
    ev, note = get_events(args, cfg, prov, required=False)
    assert ev is None and "missing" in note and "WARNING" in capsys.readouterr().err
    with pytest.raises(Exception):
        get_events(args, cfg, prov)                                           # other commands still fail loudly


def test_requirements_list_yfinance():
    req = (Path(__file__).parent.parent / "requirements.txt").read_text()
    assert any(l.strip().lower().startswith("yfinance") for l in req.splitlines())


# ------------------------------------------------------------------ 4: delta source tag
def test_delta_source_tag():
    assert delta_source_tag({"provider": 12, "bs_mid_iv": 0}) == "theta_greeks"
    assert delta_source_tag({"provider": 0, "bs_mid_iv": 7}) == "bs_fallback"
    assert delta_source_tag({"provider": 3, "bs_mid_iv": 2}) == "mixed"


# ------------------------------------------------------------------ 7: progress line
def test_progress_line_has_entry_date_counts_and_eta():
    lines = []

    class C:
        stats = {"requests": 0, "bytes": 0}
    c = C()
    pr = Progress(lines.append, 10, c, every=5, symbol="SPY")
    for i in range(10):
        c.stats["requests"] += 3
        pr.tick(date(2025, 1, 2 + i))
    assert lines and "entry 2025-01-" in lines[0] and "/10 entry days" in lines[0] and "ETA" in lines[0]
    assert "10/10" in lines[-1] and fmt_eta(3725) == "1:02:05"


# ------------------------------------------------------------------ 10: max loss from actual strike widths
def L(r, k, q):
    return Leg(r, k, q)


def test_max_loss_uses_actual_strike_widths_for_every_structure():
    ic = [L("P", 95, -1), L("P", 90, 1), L("C", 105, -1), L("C", 112, 1)]          # unequal wings 5 / 7
    assert max_loss_ps(ic, 1.5) == pytest.approx(7 - 1.5) == expected_max_loss_ps("iron_condor", ic, 1.5)
    bps = [L("P", 100, -1), L("P", 93, 1)]
    assert check_max_loss("bull_put_spread", bps, 1.2) == pytest.approx(7 - 1.2)
    ifly = [L("P", 100, -1), L("P", 94, 1), L("C", 100, -1), L("C", 103, 1)]
    assert check_max_loss("iron_butterfly", ifly, 2.0) == pytest.approx(6 - 2.0)
    # broken-wing butterfly: wider wing - narrower wing - credit, whatever the nominal width
    bwb = [L("P", 98, 1), L("P", 95, -2), L("P", 87, 1)]                          # near 3, far 8
    assert check_max_loss("put_bwb", bwb, 0.4) == pytest.approx((8 - 3) - 0.4)
    bwb2 = [L("P", 100, 1), L("P", 96, -2), L("P", 88, 1)]                        # near 4, far 8 -> different loss, same far wing
    assert check_max_loss("put_bwb", bwb2, 0.4) == pytest.approx((8 - 4) - 0.4)
    narrow_far = [L("P", 100, 1), L("P", 96, -2), L("P", 94, 1)]                  # far wing (2) narrower than near (4)
    assert check_max_loss("put_bwb", narrow_far, 0.5) == pytest.approx(0.0)       # no loss: the credit is all upside
    csp = [L("P", 90, -1)]
    assert check_max_loss("cash_secured_put", csp, 1.0) == pytest.approx(89.0)
    cc = [L("S", 0.0, 1), L("C", 105, -1)]
    assert check_max_loss("covered_call", cc, 0.8, S0=100.0) == pytest.approx(99.2)
