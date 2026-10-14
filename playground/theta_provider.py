"""ThetaProvider: reads the parquet cache; optionally fetches missing pieces on demand.

*** UNVERIFIED against real ThetaData *** (see theta_client.py).
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from .cache import DataCache
from .provider import CLOSE_MIN, MissingData, Provider, bar_minutes


class ThetaProvider(Provider):
    name = "theta-cache"
    uses_real_data = True

    def __init__(self, cache: DataCache, fetcher=None, symbol="SPY"):
        self.cache, self.fetcher, self.symbol = cache, fetcher, symbol
        self._daily = self._vix = None

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

    def list_strikes(self, d, expiry):
        ks = self.cache.strikes(expiry)
        if ks is None:
            if self.fetcher is None:
                raise MissingData(f"strike listing missing for {expiry}")
            ks = self.fetcher.fetch_strikes(expiry)
        return ks

    def get_option_quotes(self, d, expiry, right, strikes, t0, t1, res):
        if res == "eod":
            t0, t1 = CLOSE_MIN, CLOSE_MIN
        missing = [k for k in strikes if not self.cache.covered(res, d, expiry, right, k, t0, t1)]
        if missing:
            if self.fetcher is None:
                raise MissingData(f"option quotes {d} {expiry} {right} x{len(missing)} strikes {res}")
            self.fetcher.fetch_options(d, expiry, right, missing, t0, t1, res)
        return self.cache.read_options(res, d, expiry, right, strikes, t0, t1)
