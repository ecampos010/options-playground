"""Provider interface + deterministic SyntheticProvider (Black-Scholes with skew and bid/ask)."""
from __future__ import annotations

import bisect
import math
from collections import OrderedDict
from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import bs
from .calendar import trading_days as nyse_days

OPEN_MIN, CLOSE_MIN = 570, 960   # 09:30, 16:00 in minutes from midnight


class MissingData(RuntimeError):
    """Requested data is not in the cache (and fetching is disabled)."""


def bar_minutes(res: str) -> list[int]:
    if res == "eod":
        return [CLOSE_MIN]
    step = int(res.rstrip("m"))
    return list(range(OPEN_MIN, CLOSE_MIN + 1, step))


class Provider:
    name = "base"
    uses_real_data = False

    # --- daily
    def get_underlying_daily(self) -> pd.DataFrame:    # index: date; cols open high low close
        raise NotImplementedError

    def get_vix_daily(self) -> pd.Series:               # index: date; VIX close
        raise NotImplementedError

    def trading_days(self) -> list[date]:
        return list(self.get_underlying_daily().index)

    # --- intraday
    def get_underlying_bars(self, d: date, res: str) -> pd.DataFrame:   # index: ts; col spot
        raise NotImplementedError

    def list_expiries(self, d: date) -> list[date]:
        raise NotImplementedError

    def list_strikes(self, d: date, expiry: date) -> list[float]:
        raise NotImplementedError

    def get_option_quotes(self, d: date, expiry: date, right: str, strikes, t0: int, t1: int,
                          res: str) -> pd.DataFrame:
        """Long frame: ts, strike, bid, ask, bid_size, ask_size. t0/t1 minutes from midnight."""
        raise NotImplementedError

    def leg_arrays(self, d, expiry, right, strike, ts_index, res):
        """bid, ask arrays aligned (ffill within the day) to ts_index; NaN before first quote."""
        df = self.get_option_quotes(d, expiry, right, [strike], OPEN_MIN, CLOSE_MIN, res)
        df = df[df.strike == strike].drop_duplicates("ts").set_index("ts").sort_index()
        df = df[["bid", "ask"]].reindex(df.index.union(ts_index)).ffill().reindex(ts_index)
        return df["bid"].to_numpy(float), df["ask"].to_numpy(float)

    def synthetic_events(self):
        return None


# ======================================================================= synthetic
def _hash_u(d_ord: int, e_ord: int, right_i: int, strikes: np.ndarray, bars: np.ndarray) -> np.ndarray:
    """Deterministic uniform(0,1) per (day, expiry, right, strike, bar)."""
    with np.errstate(over="ignore"):
        base = np.uint64(d_ord * 1_000_003 + e_ord * 7_919 + right_i * 104_729)
        x = (base + (strikes * 100).astype(np.uint64)[:, None] * np.uint64(2654435761)
             + bars.astype(np.uint64)[None, :] * np.uint64(40503))
        x ^= x >> np.uint64(33)
        x *= np.uint64(0xFF51AFD7ED558CCD)
        x ^= x >> np.uint64(33)
        x *= np.uint64(0xC4CEB9FE1A85EC53)
        x ^= x >> np.uint64(33)
    return (x >> np.uint64(11)).astype(float) / float(2**53)


class SyntheticProvider(Provider):
    """Fake-but-plausible SPY/VIX path with >=3 vol spikes; options priced by Black-Scholes.

    vol = 0.8*VIX; put vol +20%, call vol -20%; r=4%; half-spreads such that the quoted
    spread is $0.02-0.06. Entirely synthetic: results say nothing about real markets.
    """
    name = "synthetic"
    SPIKE_FRACS = (0.25, 0.55, 0.80)
    SPIKE_LEN = 14

    def __init__(self, start="2015-01-02", end="2025-12-31", seed=7, s0=200.0, r=0.04,
                 warmup_days=30):
        self.r = r
        s = pd.Timestamp(start).date() - timedelta(days=int(warmup_days * 1.6))
        self.days = nyse_days(s, pd.Timestamp(end).date())
        n = len(self.days)
        self._pos = {d: i for i, d in enumerate(self.days)}
        rng = np.random.default_rng(seed)
        starts = {int(n * f): self.SPIKE_LEN for f in self.SPIKE_FRACS}
        x = math.log(15.0)
        vix = np.empty(n)
        close = np.empty(n)
        opens = np.empty(n)
        self._spots = np.empty((n, 79))     # 5-minute grid, 09:30..16:00
        prev = s0
        ep_start = None
        for i in range(n):
            if i in starts:
                ep_start = i
            k = i - ep_start if ep_start is not None and i - ep_start < self.SPIKE_LEN else None
            e_v, e_s = rng.standard_normal(2)
            x += 0.05 * (math.log(16.0) - x) + 0.05 * e_v
            extra = 0.0
            if k is not None:
                if k < 8:
                    x += 0.14
                    extra = -0.016
                else:
                    extra = 0.006
            v = float(np.clip(math.exp(x), 9.0, 80.0))
            sig = v / 100 * 0.85 / math.sqrt(252)
            z = -0.7 * e_v + math.sqrt(1 - 0.49) * e_s
            ret = 0.0004 + sig * z + extra
            c = prev * math.exp(ret)
            o = prev * math.exp(0.3 * ret + 0.1 * sig * rng.standard_normal())
            w = np.random.default_rng([seed, i]).standard_normal(78) * (sig * 0.8 / math.sqrt(78))
            W = np.concatenate([[0.0], np.cumsum(w)])
            bridge = W - (np.arange(79) / 78.0) * W[-1]
            path = o * np.exp(math.log(c / o) * np.arange(79) / 78.0 + bridge)
            path[-1] = c
            self._spots[i] = path
            vix[i], close[i], opens[i], prev = v, c, o, c
        self._daily = pd.DataFrame({"open": self._spots[:, 0], "high": self._spots.max(1),
                                    "low": self._spots.min(1), "close": close}, index=self.days)
        self._vix = pd.Series(vix, index=self.days, name="vix")
        self._first = bisect.bisect_left(self.days, pd.Timestamp(start).date())
        self._expiry_pool = [d for d in self.days if d.weekday() in (0, 2, 4)]
        self._leg_cache: OrderedDict = OrderedDict()
        self._bar_cache: dict = {}

    # ---- daily
    def get_underlying_daily(self):
        return self._daily.iloc[self._first - 6:]   # keep a few warm-up days

    def get_vix_daily(self):
        return self._vix.iloc[self._first - 6:]

    # ---- intraday
    def _grid(self, res):
        mins = [m for m in bar_minutes(res) if (m - OPEN_MIN) % 5 == 0 or m == CLOSE_MIN]
        if res != "eod" and int(res.rstrip("m")) < 5:
            mins = bar_minutes("5m")          # synthetic data is generated on a 5-minute grid
        return mins

    def get_underlying_bars(self, d, res):
        key = (d, res)
        if key not in self._bar_cache:
            mins = self._grid(res)
            idx = [(m - OPEN_MIN) // 5 for m in mins]
            ts = pd.DatetimeIndex([pd.Timestamp(d) + pd.Timedelta(minutes=m) for m in mins])
            self._bar_cache[key] = pd.DataFrame({"spot": self._spots[self._pos[d], idx]}, index=ts)
        return self._bar_cache[key]

    def list_expiries(self, d):
        i = bisect.bisect_right(self._expiry_pool, d)
        j = bisect.bisect_right(self._expiry_pool, d + timedelta(days=75))
        return self._expiry_pool[i:j]

    def list_strikes(self, d, expiry):
        s = float(self._daily.close.iloc[self._pos[d]])
        return [float(k) for k in range(int(0.7 * s), int(1.3 * s) + 2)]

    def _quote_block(self, d, expiry, right, strikes, mins):
        i = self._pos[d]
        strikes = np.asarray(strikes, float)
        mins = np.asarray(mins)
        spot = self._spots[i][(mins - OPEN_MIN) // 5]
        vix = float(self._vix.iloc[i])
        sig = vix / 100 * 0.8 * (1.2 if right == "P" else 0.8)
        T = (((expiry - d).days) * 1440.0 + (CLOSE_MIN - mins)) / (365.0 * 1440.0)
        mid = bs.price(spot[None, :], strikes[:, None], T[None, :], self.r, sig, right)
        u = _hash_u(d.toordinal(), expiry.toordinal(), 0 if right == "C" else 1, strikes, mins)
        half = (0.02 + 0.04 * u) / 2
        bid = np.round(np.maximum(mid - half, 0.0), 2)
        ask = np.round(np.maximum(mid + half, 0.01), 2)
        bid = np.where(mid < 0.01, 0.0, bid)
        return bid, ask, (10 + (u * 90)).astype(int)

    def get_option_quotes(self, d, expiry, right, strikes, t0, t1, res):
        mins = [m for m in self._grid(res) if t0 <= m <= t1]
        if not mins or not len(strikes):
            return pd.DataFrame(columns=["ts", "strike", "bid", "ask", "bid_size", "ask_size"])
        bid, ask, sz = self._quote_block(d, expiry, right, strikes, mins)
        ts = [pd.Timestamp(d) + pd.Timedelta(minutes=m) for m in mins]
        k = np.asarray(strikes, float)
        return pd.DataFrame({
            "ts": np.tile(np.array(ts, dtype="datetime64[ns]"), len(k)),
            "strike": np.repeat(k, len(mins)), "bid": bid.ravel(), "ask": ask.ravel(),
            "bid_size": sz.ravel(), "ask_size": sz.ravel()})

    def leg_arrays(self, d, expiry, right, strike, ts_index, res):
        key = (d, expiry, right, float(strike), res)
        hit = self._leg_cache.get(key)
        if hit is None:
            mins = [(t.hour * 60 + t.minute) for t in ts_index]
            bid, ask, _ = self._quote_block(d, expiry, right, [strike], mins)
            hit = (bid[0].astype(np.float32), ask[0].astype(np.float32))
            self._leg_cache[key] = hit
            if len(self._leg_cache) > 120_000:
                self._leg_cache.popitem(last=False)
        return hit[0].astype(float), hit[1].astype(float)

    def synthetic_events(self):
        """SYNTHETIC calendar (not real dates!): FOMC every 6 weeks on Wednesday, CPI/PPI/NFP monthly."""
        rows = []
        d = self.days[0]
        d += timedelta(days=(2 - d.weekday()) % 7)
        while d <= self.days[-1]:
            rows.append((d, "FOMC", ""))
            d += timedelta(weeks=6)
        for y in sorted({x.year for x in self.days}):
            for m in range(1, 13):
                f = date(y, m, 1)
                rows.append((f + timedelta(days=(3 - f.weekday()) % 7 + 7), "CPI", ""))
                rows.append((f + timedelta(days=(3 - f.weekday()) % 7 + 14), "PPI", ""))
                rows.append((f + timedelta(days=(4 - f.weekday()) % 7), "NFP", ""))
        return pd.DataFrame(rows, columns=["date", "event", "symbol"]).sort_values("date").reset_index(drop=True)
