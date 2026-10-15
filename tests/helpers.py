"""Hand-built markets for management-rule tests: a SyntheticProvider whose spot path we overwrite."""
import numpy as np
import pandas as pd

from playground.config import build_config
from playground.provider import SyntheticProvider
from playground.tickers import TickerInfo

TI = TickerInfo("SPY", width_fixed=5.0)


def market(spot_fn=lambda d, k: 100.0, start="2016-01-04", end="2016-06-30", vix=20.0, **kw):
    """spot_fn(date, bar_index 0..78 on the 5m grid) -> price. Constant VIX, no skew (clean parity)."""
    kw.setdefault("skew", False)
    p = SyntheticProvider(start=start, end=end, s0=100.0, **kw)
    for i in range(len(p.days)):
        p._spots[i, :] = [spot_fn(p.days[i], k) for k in range(79)]
    s = p._spots
    p._daily = pd.DataFrame({"open": s[:, 0], "high": s.max(1), "low": s.min(1), "close": s[:, -1]}, index=p.days)
    p._vix[:] = vix
    p._vix3m[:] = vix
    p._bar_cache.clear()
    p._leg_cache.clear()
    return p


def cfg(**kw):
    base = dict(sizing="fixed1", resolution="15m", event_buffer=False, gate=False, upside_guard=False, cb_days=0,
                min_credit_pct=0.01, entry_cadence="monthly", entry_sample="first", entry_window="09:45-15:30",
                time_exit=0, stop_mult=None, tp=9.0, breach_exit=False, fee_per_contract=0.05)
    base.update(kw)
    return build_config("green", base)


def first_day_index(p, cfgv):
    from playground.lab import entry_schedule
    days = list(p.get_underlying_daily().index)
    sched, _ = entry_schedule(days, 6, len(days) - 1, cfgv)
    return days, sorted(sched)
