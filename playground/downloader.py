"""Resumable two-stage downloader (Stage A entry snapshots, Stage B legs on demand).

*** UNVERIFIED against real ThetaData ***  Everything is idempotent: coverage is recorded per
(date, expiry, right, strike, window) in the cache, so a re-run fetches nothing already held.
"""
from __future__ import annotations

import json
import threading
from urllib.parse import urlsplit
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import pandas as pd

from . import rules
from .cache import DataCache
from .config import Config
from .engine import entry_minutes
from .events import EventIndex
from .provider import CLOSE_MIN, OPEN_MIN, MissingData, bar_minutes
from .theta_client import ThetaClient


def make_logger(log_path, echo=None):
    """Timestamped logger that appends to log_path and echoes to stderr. Never logs request params with credentials
    (the client holds none)."""
    import sys
    path = __import__("pathlib").Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def log(*a):
        msg = time.strftime("%Y-%m-%d %H:%M:%S ") + " ".join(map(str, a))
        with lock:
            print(msg, file=echo or sys.stderr, flush=True)
            with open(path, "a") as f:
                f.write(msg + "\n")
    return log


def fmt_eta(sec: float) -> str:
    sec = max(0, int(sec))
    return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


class Progress:
    """Prints a line every `every` requests (and at the end): entry date, entry days done / total, requests, ETA.
    ETA uses the rate of the most recent completions so cached (instant) days at the start of a resumed run do not
    make it look too optimistic for long."""

    def __init__(self, log, total: int, client, every: int = 50, symbol: str = "", window: int = 20):
        self.log, self.total, self.client, self.every, self.symbol = log, total, client, max(1, every), symbol
        self.t0 = time.time()
        self.req0 = client.stats["requests"]
        self.last_req = self.req0
        self.recent: list[tuple[float, int]] = []
        self.window = window
        self.done = 0

    def tick(self, d, force: bool = False):
        self.done += 1
        now = time.time()
        self.recent = (self.recent + [(now, self.done)])[-self.window:]
        req = self.client.stats["requests"]
        if not force and req - self.last_req < self.every and self.done != self.total:
            return
        self.last_req = req
        left = self.total - self.done
        if len(self.recent) >= 2 and self.recent[-1][0] > self.recent[0][0]:
            rate = (self.recent[-1][1] - self.recent[0][1]) / (self.recent[-1][0] - self.recent[0][0])
        else:
            rate = self.done / max(now - self.t0, 1e-9)
        eta = fmt_eta(left / rate) if rate > 0 else "?"
        self.log(f"stage A {self.symbol} entry {d} | {self.done}/{self.total} entry days | "
                 f"{req - self.req0} requests this run | {self.client.stats['bytes'] / 1e6:.1f} MB | ETA {eta}")


class FetchError(MissingData):
    pass


class BudgetExceeded(RuntimeError):
    """--max-gb / --max-hours hard stop (per ticker). The cache stays consistent; re-run to continue."""


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
        self.max_gb = self.max_hours = None
        self._t0 = time.monotonic()
        self._disk = (0.0, 0)
        self.greeks_supported: bool | None = None
        self.errlog = None                      # ErrorLog (playground/errorlog.py) or None: one row per failed request
        self.stage = ""                         # who is asking, written to errors.csv: listing | A | B | greeks

    def set_limits(self, max_gb=None, max_hours=None):
        self.max_gb, self.max_hours, self._t0 = max_gb, max_hours, time.monotonic()

    def _check(self):
        if self.max_hours is not None and (time.monotonic() - self._t0) / 3600.0 > self.max_hours:
            raise BudgetExceeded(f"--max-hours {self.max_hours} reached for {self.symbol}")
        if self.max_gb is not None:
            now = time.monotonic()
            if now - self._disk[0] > 20:
                self._disk = (now, self.cache.disk_bytes())
            if self._disk[1] / 1e9 > self.max_gb:
                raise BudgetExceeded(f"--max-gb {self.max_gb} reached for {self.symbol} ({self._disk[1] / 1e9:.2f} GB)")

    def serial(self) -> "Fetcher":
        f = Fetcher(self.client, self.cache, self.tdays, 1, self.chunk, self.entry_fetch, self.log, self.symbol)
        f.failed = self.failed
        f.errlog, f.stage = self.errlog, self.stage
        f.parallel = False
        f._bars_lock = self._bars_lock
        f.max_gb, f.max_hours, f._t0 = self.max_gb, self.max_hours, self._t0
        return f

    def _log_request(self, r, **ctx):
        """One errors.csv row for a failed request. Only failures are logged, so this never changes a fetch."""
        if self.errlog is not None:
            self.errlog.record(stage=self.stage, endpoint=urlsplit(r.url).path,
                               error_type="denied" if r.denied else "request_failed", http_status=r.status,
                               message=r.error or f"HTTP {r.status}", **ctx)

    def _fail(self, what: str, r, **ctx):
        self.failed.append(what)
        self._log_request(r, **ctx)
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
            self._fail(f"strikes {expiry}", r, expiry=expiry)
        self.cache.save_strikes(expiry, ks)
        return ks

    # ---------------------------------------------------------- underlying bars
    def fetch_bars(self, d: date, res: str):
        self._check()
        with self._bars_lock:
            if not self.cache.has_bars(res, d):
                self._fetch_bars(d, res)

    def _fetch_bars(self, d: date, res: str):
        idx = self.tdays.index(d) if d in self.tdays else None
        end = self.tdays[min(len(self.tdays) - 1, idx + 19)] if idx is not None else d
        df, r = self.client.stock_spot_bars(self.symbol, d, end, res)
        if not r.ok:
            self._fail(f"bars {d}", r, date=d)
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
        self._check()
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

    def fetch_bulk(self, d, expiry, m, res):
        """ONE bulk snapshot (all strikes, both rights) at one bar. TODO-VERIFY endpoint form (probe section c2)."""
        self._check()
        self._fetch_bulk(d, expiry, [], m, res)
        self.cache.bulk_set(res, d, expiry, m)

    def _fetch_bulk(self, d, expiry, strikes, m, res):
        step = self._step(res)
        df, r = self.client.option_quotes_bulk(expiry, d, res, m, m + step, self.symbol)
        if not r.ok:
            self._fail(f"bulk {d} {expiry}", r, date=d, expiry=expiry)
        df = df[(df.ts.dt.hour * 60 + df.ts.dt.minute) == m]
        universe = set(self.cache.strikes(expiry) or []) | set(strikes) | set(df.strike)
        for rt in ("P", "C"):
            part = df[df.right == rt]
            self.cache.write_options(res, d, expiry, rt, universe, m, m, part)
            self.rows += len(part)

    def fetch_greeks(self, d, expiry, m, res):
        """Greeks snapshot (delta/iv/underlying_price) or None if the plan/endpoint does not provide it."""
        if self.greeks_supported is False:
            return None
        got = self.cache.greeks_get(res, d, expiry, m)
        if got is not None:
            return got if len(got) else None
        self._check()
        df, r = self.client.greeks_bulk(expiry, d, res, m, m + self._step(res), self.symbol)
        if not r.ok:                      # 403/404/etc: endpoint unavailable -> stop asking, BS fallback
            self._log_request(r, date=d, expiry=expiry)
            self.greeks_supported = False
            self.failed.append(f"greeks unsupported: {r.error[:120]}")
            return None
        self.cache.greeks_put(res, d, expiry, m, df if len(df) else None)
        return df if len(df) else None

    def _fetch_entry(self, d, expiry, right, strikes, m, res):
        step = self._step(res)
        if self.entry_fetch == "bulk":
            return self._fetch_bulk(d, expiry, strikes, m, res)

        def one(k):
            df, r = self.client.option_quotes(expiry, right, k, d, d, res, m, m + step, self.symbol)
            if not r.ok:
                self._fail(f"quote {d} {expiry} {right}{k}", r, date=d, expiry=expiry, right=right, strike=k)
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
                df, r = self.client.option_eod(expiry, right, k, d, end, self.symbol)
            else:
                df, r = self.client.option_quotes(expiry, right, k, d, end, res, OPEN_MIN, CLOSE_MIN, self.symbol)
            if not r.ok:
                self._fail(f"quote {d}..{end} {expiry} {right}{k}", r, date=d, expiry=expiry, right=right, strike=k)
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


def stage_a(provider, fetcher: Fetcher, cfg: Config, evx, log, threads: int = 2, limit: int | None = None,
            lab: bool = False, tinfo=None, progress_every: int = 50) -> dict:
    """Prefetch the narrow entry snapshot for every candidate day (resumable).
    lab=True: only the cadence's entry days; delta mode fetches ONE bulk snapshot per expiry (no +/-10 strike band)."""
    from .lab import entry_schedule
    from .strategies import STRATEGIES
    from .tickers import load_tickers, spread_width
    daily, vix = provider.get_underlying_daily(), provider.get_vix_daily()
    ccfg = cfg.replace(entry_every=1) if lab else cfg
    days = candidate_days(ccfg, daily, vix, evx)
    sched = {}
    if lab:
        alld = list(daily.index)
        i0 = max(6, next((j for j, x in enumerate(alld) if not cfg.start_date or str(x) >= cfg.start_date), 6))
        i1 = max([j for j, x in enumerate(alld) if not cfg.end_date or str(x) <= cfg.end_date] or [len(alld) - 1])
        sched, _ = entry_schedule(alld, i0, i1, cfg.replace(dte=[cfg.dte_target - 7, cfg.dte_target, cfg.dte_target + 7])
                                  if cfg.dte_target else cfg)
        days = [d for d in days if d in sched]
        tinfo = tinfo or load_tickers().get(cfg.ticker)
    days = days[:limit]
    vx = vix.reindex(daily.index).ffill()
    pos = {d: i for i, d in enumerate(daily.index)}
    done, errors = [], []
    stopped = []
    serial = fetcher.serial()
    sprov = provider.clone(serial)
    t_start = time.time()
    delta_mode = cfg.strike_rule["mode"] == "delta"
    strat = STRATEGIES.get(cfg.strategy)

    def work(d):
        try:
            i = pos[d]
            # the entry bar grid is the engine's: daily spot -> the 16:00 print, otherwise the resolution grid
            # (never the Theta stock endpoint: stock history is 403 on Standard)
            mins = [CLOSE_MIN] if (cfg.spot_source == "daily" and lab) or cfg.resolution == "eod" \
                else bar_minutes(cfg.resolution)
            spot, v = float(daily["open"].iloc[i]), float(vx.iloc[i - 1])
            tp_, tc_ = rules.target_strikes(spot, v, cfg)
            for m in entry_minutes(d, cfg):
                bm = next((x for x in mins if x >= m), None)
                if bm is None:
                    continue
                hint = sched.get(d)
                if hint is not None:
                    exps = [hint] if sprov.confirm_expiry(d, hint, cfg, bm, spot, v) else []
                else:
                    exps = sprov.entry_expiries(d, cfg, bm, spot, v)
                for e in exps:
                    ks = sprov.list_strikes(d, e)
                    if lab and delta_mode:
                        sprov.chain_snapshot(d, e, bm, cfg.resolution, spot)
                        continue
                    if lab:
                        w = spread_width(spot, ks, cfg, tinfo)
                        need = strat.needed_strikes(ks, cfg, spot, v, w)
                        for right, kk in need.items():
                            sprov.get_option_quotes(d, e, right, kk, bm, bm, cfg.resolution)
                        continue
                    for right, tgt in (("P", tp_), ("C", tc_)):
                        _, need = rules.entry_strikes(ks, tgt, right, cfg)
                        sprov.get_option_quotes(d, e, right, need, bm, bm, cfg.resolution)
            return d, None
        except BudgetExceeded as ex:
            stopped.append(str(ex))
            return d, "budget"
        except MissingData as ex:
            return d, str(ex)

    # resumable: days whose items are all cached finish without requests, so a re-run skips straight through them
    prog = Progress(log, len(days), fetcher.client, progress_every, fetcher.symbol)
    log(f"stage A {fetcher.symbol}: {len(days)} entry days to cover (cached items are skipped); "
        f"progress line every {progress_every} requests")
    with ThreadPoolExecutor(max(1, min(threads, 4))) as pool:
        for d0, (d, err) in zip(days, pool.map(work, days)):
            if err == "budget":
                prog.tick(d0)
                continue
            if err:
                log(f"stage A {fetcher.symbol} entry {d0}: {err[:200]}")
                if fetcher.errlog is not None:
                    fetcher.errlog.record(stage=fetcher.stage, date=d0, endpoint="stage_a", error_type="day_error",
                                          message=err)
            (errors if err else done).append((d, err) if err else d)
            prog.tick(d0)
        log(f"stage A {fetcher.symbol} finished: {len(done)} ok, {len(errors)} with errors, "
            f"{fetcher.client.stats['requests']} requests, {time.time() - t_start:.0f}s")
    return {"candidates": len(days), "done": len(done), "errors": errors, "stopped": sorted(set(stopped))}


def write_manifest(cache: DataCache, fetcher: Fetcher, extra: dict) -> dict:
    st = fetcher.client.stats
    m = {"requests": st["requests"], "response_bytes": st["bytes"], "throttled": st["throttled"],
         "errors": st["errors"], "empty_responses": st["empty"], "no_data_472": st["no_data"],
         "denied_403": st["denied"], "rows_written": fetcher.rows,
         "cache_bytes_on_disk": cache.disk_bytes(), "failed": fetcher.failed[:200],
         "avg_latency_s": (st["latency_sum"] / st["requests"]) if st["requests"] else None, **extra}
    p = cache.root / "state" / "manifest.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(m, indent=2, default=str))
    return m
