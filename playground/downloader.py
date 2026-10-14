"""Resumable two-stage downloader (Stage A entry snapshots, Stage B legs on demand).

*** UNVERIFIED against real ThetaData ***  Everything is idempotent: coverage is recorded per
(date, expiry, right, strike, window) in the cache, so a re-run fetches nothing already held.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from . import rules
from .cache import DataCache
from .config import Config
from .engine import entry_minutes
from .events import EventIndex
from .provider import CLOSE_MIN, OPEN_MIN, MissingData, bar_minutes
from .theta_client import ThetaClient


class FetchError(MissingData):
    pass


def candidate_days(cfg: Config, daily: pd.DataFrame, vix: pd.Series, evx: EventIndex | None) -> list[date]:
    """Entry days not *surely* blocked by gate/guards/event buffer (account-state rules like the
    circuit breaker are ignored). Move-based guards use the day's high/low envelope so the result is a
    superset of days the engine can actually enter."""
    days = list(daily.index)
    close, high, low = (daily[c].to_numpy(float) for c in ("close", "high", "low"))
    v = vix.reindex(daily.index).ffill().to_numpy(float)
    i0 = 6
    if cfg.start_date:
        i0 = max(i0, next((j for j, x in enumerate(days) if str(x) >= cfg.start_date), len(days)))
    i1 = max([j for j, x in enumerate(days) if not cfg.end_date or str(x) <= cfg.end_date] or [-1])
    w1 = cfg.window()[1]
    out = []
    for i in range(i0, i1 + 1):
        if (i - i0) % cfg.entry_every:
            continue
        if cfg.gate and v[i - 1] >= cfg.gate_vix:
            continue
        if cfg.gate and high[i] / close[i - 1] - 1 <= cfg.gate_day:
            continue
        if cfg.gate and high[i] / close[i - 5] - 1 <= cfg.gate_5d:
            continue
        if cfg.upside_guard and low[i] / close[i - 5] - 1 >= cfg.upside_5d:
            continue
        if cfg.entry_requires_gate_trip and not (v[i - 1] >= cfg.gate_vix or low[i] / close[i - 1] - 1 <= cfg.gate_day
                                                 or low[i] / close[i - 5] - 1 <= cfg.gate_5d):
            continue
        if evx is not None and rules.event_block(evx.upcoming(days[i]), w1, cfg):
            continue
        out.append(days[i])
    return out


class Fetcher:
    def __init__(self, client: ThetaClient, cache: DataCache, trading_days: list[date], threads: int = 2,
                 chunk_days: int = 10, entry_fetch: str = "per-strike", log=None, symbol="SPY"):
        self.client, self.cache, self.tdays = client, cache, trading_days
        self.threads, self.chunk, self.entry_fetch, self.symbol = threads, chunk_days, entry_fetch, symbol
        self.log = log or (lambda *a: None)
        self.failed: list[str] = []
        self.rows = 0
        self.parallel = True
        self._bars_lock = threading.Lock()

    def serial(self) -> "Fetcher":
        f = Fetcher(self.client, self.cache, self.tdays, 1, self.chunk, self.entry_fetch, self.log, self.symbol)
        f.failed = self.failed
        f.parallel = False
        f._bars_lock = self._bars_lock
        return f

    def _fail(self, what: str, r):
        self.failed.append(what)
        raise FetchError(f"{what}: {r.error or r.status} [{r.url} {r.params}]")

    # ---------------------------------------------------------- listings
    def fetch_expirations(self):
        ex, r = self.client.expirations(self.symbol)
        if not r.ok:
            self._fail("expirations", r)
        self.cache.save_expirations(ex)
        return ex

    def fetch_strikes(self, expiry):
        ks, r = self.client.strikes(expiry, self.symbol)
        if not r.ok:
            self._fail(f"strikes {expiry}", r)
        self.cache.save_strikes(expiry, ks)
        return ks

    # ---------------------------------------------------------- underlying bars
    def fetch_bars(self, d: date, res: str):
        with self._bars_lock:
            if not self.cache.has_bars(res, d):
                self._fetch_bars(d, res)

    def _fetch_bars(self, d: date, res: str):
        idx = self.tdays.index(d) if d in self.tdays else None
        end = self.tdays[min(len(self.tdays) - 1, idx + 19)] if idx is not None else d
        df, r = self.client.stock_spot_bars(self.symbol, d, end, res)
        if not r.ok:
            self._fail(f"bars {d}", r)
        grid = set(bar_minutes(res))
        df = df.assign(_d=df.ts.dt.date, _m=df.ts.dt.hour * 60 + df.ts.dt.minute)
        df = df[df._m.isin(grid)]
        for day in (self.tdays[idx: idx + 20] if idx is not None else [d]):
            if not self.cache.has_bars(res, day):
                self.cache.write_bars(res, day, df[df._d == day])
        self.rows += len(df)

    # ---------------------------------------------------------- options
    def _step(self, res):
        return 390 if res == "eod" else int(res.rstrip("m"))

    def fetch_options(self, d, expiry, right, strikes, t0, t1, res):
        if res == "eod":
            return self._fetch_range(d, expiry, right, strikes, res)
        if t0 == t1:
            return self._fetch_entry(d, expiry, right, strikes, t0, res)
        return self._fetch_range(d, expiry, right, strikes, res)

    def _map(self, fn, items):
        if self.parallel and self.threads > 1 and len(items) > 1:
            with ThreadPoolExecutor(self.threads) as ex:
                return list(ex.map(fn, items))
        return [fn(x) for x in items]

    def _fetch_entry(self, d, expiry, right, strikes, m, res):
        step = self._step(res)
        if self.entry_fetch == "bulk":
            df, r = self.client.option_quotes_bulk(expiry, d, res, m, m + step)
            if not r.ok:
                self._fail(f"bulk {d} {expiry}", r)
            df = df[(df.ts.dt.hour * 60 + df.ts.dt.minute) == m]
            universe = set(self.cache.strikes(expiry) or []) | set(strikes) | set(df.strike)
            for rt in ("P", "C"):
                part = df[df.right == rt]
                self.cache.write_options(res, d, expiry, rt, universe, m, m, part)
                self.rows += len(part)
            return

        def one(k):
            df, r = self.client.option_quotes(expiry, right, k, d, d, res, m, m + step)
            if not r.ok:
                self._fail(f"quote {d} {expiry} {right}{k}", r)
            return k, df[(df.ts.dt.hour * 60 + df.ts.dt.minute) == m]

        got = self._map(one, list(strikes))
        frames = [x.assign(strike=k) for k, x in got if len(x)]
        out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        self.cache.write_options(res, d, expiry, right, strikes, m, m, out)
        self.rows += len(out)

    def _fetch_range(self, d, expiry, right, strikes, res):
        end = min(expiry, d + timedelta(days=self.chunk))
        span = [x for x in self.tdays if d <= x <= end] or [d]
        t0, t1 = (CLOSE_MIN, CLOSE_MIN) if res == "eod" else (OPEN_MIN, CLOSE_MIN)

        def one(k):
            if res == "eod":
                df, r = self.client.option_eod(expiry, right, k, d, end)
            else:
                df, r = self.client.option_quotes(expiry, right, k, d, end, res, OPEN_MIN, CLOSE_MIN)
            if not r.ok:
                self._fail(f"quote {d}..{end} {expiry} {right}{k}", r)
            return k, df

        for k, df in self._map(one, list(strikes)):
            df = df.assign(strike=k) if len(df) else df
            dd = df.ts.dt.date if len(df) else pd.Series([], dtype=object)
            for day in span:
                if self.cache.covered(res, day, expiry, right, k, t0, t1):
                    continue
                part = df[dd == day] if len(df) else df
                self.cache.write_options(res, day, expiry, right, [k], t0, t1, part)
                self.rows += len(part)


def stage_a(provider, fetcher: Fetcher, cfg: Config, evx, log, threads: int = 2, limit: int | None = None) -> dict:
    """Prefetch the narrow entry snapshot for every candidate day (resumable)."""
    daily, vix = provider.get_underlying_daily(), provider.get_vix_daily()
    days = candidate_days(cfg, daily, vix, evx)[:limit]
    vx = vix.reindex(daily.index).ffill()
    pos = {d: i for i, d in enumerate(daily.index)}
    done, errors = [], []
    serial = fetcher.serial()
    sprov = type(provider)(provider.cache, serial, provider.symbol)
    t_start = time.time()

    def work(d):
        try:
            i = pos[d]
            bars = sprov.get_underlying_bars(d, cfg.resolution)
            mins = [t.hour * 60 + t.minute for t in bars.index]
            spot, v = float(daily["open"].iloc[i]), float(vx.iloc[i - 1])
            tp_, tc_ = rules.target_strikes(spot, v, cfg)
            for e in rules.candidate_expiries(sprov.list_expiries(d), d, cfg):
                ks = sprov.list_strikes(d, e)
                for right, tgt in (("P", tp_), ("C", tc_)):
                    _, need = rules.entry_strikes(ks, tgt, right, cfg)
                    for m in entry_minutes(d, cfg):
                        bm = next((x for x in mins if x >= m), None)
                        if bm is not None:
                            sprov.get_option_quotes(d, e, right, need, bm, bm, cfg.resolution)
            return d, None
        except MissingData as ex:
            return d, str(ex)

    with ThreadPoolExecutor(max(1, threads)) as pool:
        for n, (d, err) in enumerate(pool.map(work, days), 1):
            (errors if err else done).append((d, err) if err else d)
            if n % 25 == 0 or n == len(days):
                el = time.time() - t_start
                log(f"stage A {n}/{len(days)} days | {fetcher.client.stats['requests']} req | "
                    f"{fetcher.client.stats['bytes'] / 1e6:.1f} MB | {el:.0f}s")
    return {"candidates": len(days), "done": len(done), "errors": errors}


def write_manifest(cache: DataCache, fetcher: Fetcher, extra: dict) -> dict:
    st = fetcher.client.stats
    m = {"requests": st["requests"], "response_bytes": st["bytes"], "throttled": st["throttled"],
         "errors": st["errors"], "empty_responses": st["empty"], "rows_written": fetcher.rows,
         "cache_bytes_on_disk": cache.disk_bytes(), "failed": fetcher.failed[:200],
         "avg_latency_s": (st["latency_sum"] / st["requests"]) if st["requests"] else None, **extra}
    p = cache.root / "state" / "manifest.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(m, indent=2, default=str))
    return m
