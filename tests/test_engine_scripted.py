"""Engine behaviour on a hand-scripted market (resolution 'eod': one bar per day)."""
from datetime import date

import pandas as pd
import pytest

from playground.calendar import trading_days
from playground.config import build_config
from playground.engine import run_backtest
from playground.provider import CLOSE_MIN, Provider

BASE_Q = {("P", 95): (1.00, 1.10), ("P", 90): (0.40, 0.50), ("C", 105): (0.80, 0.90), ("C", 110): (0.30, 0.40)}


class Scripted(Provider):
    name = "scripted"

    def __init__(self, scale=lambda i: 1.0, spot=lambda i: 100.0, n=45):
        self.days = trading_days(date(2024, 1, 2), date(2024, 12, 31))[:n]
        self.scale, self.spot = scale, spot
        self.expiry = self.days[6 + 25]       # ~35 calendar days after the first entry day

    def get_underlying_daily(self):
        s = [self.spot(i) for i in range(len(self.days))]
        return pd.DataFrame({"open": s, "high": s, "low": s, "close": s}, index=self.days)

    def get_vix_daily(self):
        return pd.Series(15.3, index=self.days)

    def get_underlying_bars(self, d, res):
        i = self.days.index(d)
        return pd.DataFrame({"spot": [self.spot(i)]}, index=pd.DatetimeIndex([pd.Timestamp(d) + pd.Timedelta(minutes=CLOSE_MIN)]))

    def list_expiries(self, d):
        return [self.expiry] if d < self.expiry else []

    def list_strikes(self, d, e):
        return [float(k) for k in range(80, 121)]

    def get_option_quotes(self, d, e, right, strikes, t0, t1, res):
        i = self.days.index(d)
        rows = []
        for k in strikes:
            if (right, int(k)) in BASE_Q:
                b, a = BASE_Q[(right, int(k))]
                s = self.scale(i)
                rows.append((pd.Timestamp(d) + pd.Timedelta(minutes=CLOSE_MIN), float(k), b * s, a * s, 10, 10))
        return pd.DataFrame(rows, columns=["ts", "strike", "bid", "ask", "bid_size", "ask_size"])


def cfg(**kw):
    return build_config("green", {"resolution": "eod", "event_buffer": False, "cb_days": 0, "gate": False,
                                  "upside_guard": False, "max_open_lots": 1, **kw})


def first_trade(p, c):
    r = run_backtest(p, c, None)
    return r, r.trades.iloc[0]


def test_entry_credit_and_sizing():
    r, t = first_trade(Scripted(), cfg(time_exit=0, tp=0.99))
    assert t.credit == pytest.approx(0.90)                    # 1.0-0.5+0.8-0.4 at natural
    # headroom = min(15% , 60% cap) -> 9000 ; floor(9000/410)=21
    assert t.contracts == 21
    assert (t.short_put, t.long_put, t.short_call, t.long_call) == (95, 90, 105, 110)
    assert t.risk_expiry_after <= 0.6 * t.budget


def test_full_use_rounds_up():
    r, t = first_trade(Scripted(), cfg(time_exit=0, full_use=True, tp=0.99))
    assert t.contracts == 22                                  # 9000/410 = 21.95 -> 22


def test_take_profit():
    p = Scripted(scale=lambda i: 0.3 if i >= 9 else 1.0)
    r, t = first_trade(p, cfg())
    assert t.exit_reason == "tp" and t.pnl > 0


def test_stop():
    p = Scripted(scale=lambda i: 1.6 if i >= 9 else 1.0)       # natural cost 1.3*1.6 = 2.08 >= 1.8
    r, t = first_trade(p, cfg())
    assert t.exit_reason == "stop" and t.pnl < 0


def test_breach():
    p = Scripted(spot=lambda i: 94.5 if i >= 9 else 100.0)
    r, t = first_trade(p, cfg())
    assert t.exit_reason == "breach"


def test_time_exit_and_disabled():
    r, t = first_trade(Scripted(), cfg(tp=0.99))
    assert t.exit_reason == "time"
    assert (t.expiry - pd.Timestamp(t.exit_ts).date()).days <= 21
    r, t = first_trade(Scripted(), cfg(tp=0.99, time_exit=0))
    assert t.exit_reason == "expiry" and t.exit_cost == 0 and t.pnl == pytest.approx(0.90 * 100 * t.contracts)


def test_expiry_settles_on_close_inside_wings():
    r, t = first_trade(Scripted(spot=lambda i: 92.0 if i >= 31 else 100.0), cfg(tp=0.99, time_exit=0, breach_exit=False))
    assert t.exit_reason == "expiry" and t.exit_cost == 3.0       # 95-92


def test_gate_blocks_entry():
    p = Scripted(spot=lambda i: 100.0)
    r = run_backtest(p, cfg(gate=True, gate_vix=10.0), None)
    assert len(r.trades) == 0 and r.blocked.get("gate", 0) > 0


def test_interest_accrual_on_idle_cash():
    p = Scripted()
    c = cfg(min_credit=99.0)
    r = run_backtest(p, c, None)
    days = list(r.equity["date"])
    exp = 100_000.0
    for a, b in zip(days[:-1], days[1:]):
        exp += exp * 0.036 / 365 * (b - a).days
    assert r.equity["cash"].iloc[-1] == pytest.approx(exp, rel=1e-6)
    assert len(r.trades) == 0


def test_circuit_breaker_pauses_entries():
    # losing month (stop) in January -> entries paused for 21 trading days from February
    p = Scripted(scale=lambda i: 1.6 if i >= 9 else 1.0, n=60)
    p.expiry = p.days[45]
    c = cfg(cb_days=21, max_open_lots=1)
    r = run_backtest(p, c, None)
    assert r.blocked.get("cb", 0) > 0
