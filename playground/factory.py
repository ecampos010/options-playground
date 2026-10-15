"""Provider / config factories shared by run, sweep, stress, download and plan."""
from __future__ import annotations

from pathlib import Path

from .cache import DataCache
from .provider import SyntheticProvider
from .spot import SpotStore
from .theta_provider import ThetaProvider

# Synthetic stand-ins for the five underlyings (own spot path, shared VIX path); grids differ by ticker and era.
SYN = {
    "SPY": dict(s0=200.0, vol_scale=1.0, strike_step=1.0),
    "QQQ": dict(s0=100.0, vol_scale=1.25, strike_step=1.0),
    "IWM": dict(s0=100.0, vol_scale=1.3, strike_step=1.0, step_by_year={2015: 1.0, 2016: 1.0, 2017: 0.5}),
    "GLD": dict(s0=110.0, vol_scale=0.8, strike_step=1.0, step_by_year={2015: 2.5, 2016: 2.5}),
    "TLT": dict(s0=120.0, vol_scale=0.7, strike_step=0.5),
}
_syn_cache: dict = {}


def ticker_cache(data_dir, ticker: str) -> DataCache:
    root = Path(data_dir)
    return DataCache(root if ticker == "SPY" else root / "tickers" / ticker)


def synthetic(ticker: str, start="2015-01-02", end="2025-12-31", seed=7, **kw) -> SyntheticProvider:
    key = (ticker, start, end, seed, tuple(sorted(kw.items())))
    if key not in _syn_cache:
        params = {**SYN.get(ticker, SYN["SPY"]), **kw}
        _syn_cache[key] = SyntheticProvider(start=start, end=end, seed=seed, ticker=ticker, **params)
    return _syn_cache[key]


def theta(ticker: str, data_dir="data", fetcher=None, rate=0.036) -> ThetaProvider:
    return ThetaProvider(ticker_cache(data_dir, ticker), fetcher, ticker, vix_cache=DataCache(data_dir),
                         spot_store=SpotStore(Path(data_dir) / "spot"), rate=rate)


def make(kind: str, ticker: str, data_dir="data", syn_start="2015-01-02", syn_end="2025-12-31", syn_seed=7,
         fetcher=None, rate=0.036, **syn_kw):
    if kind == "synthetic":
        return synthetic(ticker, syn_start, syn_end, syn_seed, **syn_kw)
    return theta(ticker, data_dir, fetcher, rate)
