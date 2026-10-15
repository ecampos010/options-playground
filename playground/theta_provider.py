"""ThetaProvider: reads the parquet cache; optionally fetches missing pieces on demand.

Contract selection (verified behaviour on a Standard account): the expirations list contains
expirations that are not yet tradable on a given date, and half-dollar strikes did not exist in
older years. So expiries are *confirmed by a trial quote* (cached per date/expiry) and the strike
grid is probed once per date; HTTP 472 (NO_DATA) is a cached negative, never an error.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from . import rules
from .cache import DataCache
from .provider import CLOSE_MIN, MissingData, Provider

MAX_TRIALS = 12   # expiry trial requests per (date, entry) before giving up


class ThetaProvider(Provider):
    name = "theta-cache"
    uses_real_data = True

    def __init__(self, cache: DataCache, fetcher=None, symbol="SPY"):
        self.cache, self.fetcher, self.symbol = cache, fetcher, symbol
        self._daily = self._vix = None
        self._ctx: dict[date, tuple[int, float, str]] = {}   # day -> (entry minute, target put strike, res)

    def _need(self, name):
        df = self.cache.load_daily(name)
        if df is None:
            raise MissingData(f"{name} daily history missing in {self.cache.root}; run `download --daily-only`")
        return df

    def get_underlying_daily(self):
        if self._daily is None:
            self._daily = self._need(self.symbol)
        return self._daily

    def get_vix_daily(self):
        if self._vix is None:
            self._vix = self._need("VIX")["close"]
        return self._vix

    def get_underlying_bars(self, d, res):
        if res == "eod":
            c = float(self.get_underlying_daily().loc[d, "close"])
            return pd.DataFrame({"spot": [c]}, index=pd.DatetimeIndex([pd.Timestamp(d) + pd.Timedelta(minutes=CLOSE_MIN)]))
        if not self.cache.has_bars(res, d):
            if self.fetcher is None:
                raise MissingData(f"underlying bars {self.symbol} {res} {d}")
            self.fetcher.fetch_bars(d, res)
        return self.cache.read_bars(res, d)

    def list_expiries(self, d):
        ex = self.cache.expirations()
        if ex is None:
            if self.fetcher is None:
                raise MissingData("expirations listing missing; run download")
            ex = self.fetcher.fetch_expirations()
        return [e for e in ex if d < e <= date.fromordinal(d.toordinal() + 120)]

    def _raw_strikes(self, expiry):
        ks = self.cache.strikes(expiry)
        if ks is None:
            if self.fetcher is None:
                raise MissingData(f"strike listing missing for {expiry}")
            ks = self.fetcher.fetch_strikes(expiry)
        return ks

    def list_strikes(self, d, expiry):
        """Listed strikes, minus half-dollar strikes on dates where a probe found none trading."""
        ks = self._raw_strikes(expiry)
        grid = self.cache.grid_get(d)
        if grid is None and self.fetcher is not None and d in self._ctx:
            grid = self._probe_grid(d, expiry, ks)
        return [k for k in ks if k == int(k)] if grid == "int" else ks

    def _probe_grid(self, d, expiry, ks):
        m, tgt, res = self._ctx[d]
        half = [k for k in ks if k != int(k)]
        if not half:
            grid = "int"
        else:
            k = min(half, key=lambda x: abs(x - tgt))
            grid = "half" if len(self.get_option_quotes(d, expiry, "P", [k], m, m, res)) else "int"
        self.cache.grid_set(d, grid)
        return grid

    def entry_expiries(self, d, cfg, minute, spot, vix):
        ranked = rules.rank_expiries(self.list_expiries(d), d, cfg)
        tgt, _ = rules.target_strikes(spot, vix, cfg)
        m = CLOSE_MIN if cfg.resolution == "eod" else minute
        self._ctx[d] = (m, tgt, cfg.resolution)
        out, trials = [], 0
        for e in ranked:
            if len(out) >= cfg.entry_expiries:
                break
            ok = self.cache.trial_get(d, e)
            if ok is None:
                if self.fetcher is None or trials >= MAX_TRIALS:
                    continue            # cache-only: unconfirmed == unusable
                trials += 1
                ok = self._trial(d, e, m, tgt, cfg.resolution)
            if ok:
                out.append(e)
        return out

    def _trial(self, d, e, m, tgt, res) -> bool:
        """Cheap trial: 1-minute quote of the integer put strike nearest the target. 472 -> not listed."""
        ints = [k for k in self._raw_strikes(e) if k == int(k)]
        listed = False
        if ints:
            k = min(ints, key=lambda x: abs(x - tgt))
            listed = len(self.get_option_quotes(d, e, "P", [k], m, m, res)) > 0
        self.cache.trial_set(d, e, listed)    # FetchError (real failures) propagates uncached
        return listed

    def get_option_quotes(self, d, expiry, right, strikes, t0, t1, res):
        if res == "eod":
            t0, t1 = CLOSE_MIN, CLOSE_MIN
        missing = [k for k in strikes if not self.cache.covered(res, d, expiry, right, k, t0, t1)]
        if missing:
            if self.fetcher is None:
                raise MissingData(f"option quotes {d} {expiry} {right} x{len(missing)} strikes {res}")
            self.fetcher.fetch_options(d, expiry, right, missing, t0, t1, res)
        return self.cache.read_options(res, d, expiry, right, strikes, t0, t1)
