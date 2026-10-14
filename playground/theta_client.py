"""Thin ThetaData Theta Terminal (v3 REST) adapter.

*** UNVERIFIED ***  Written from memory of the documented v3 API with no network access to the
docs. ALL endpoint paths and parameter names live in `EP` / `P` below; every guess is marked
TODO-VERIFY. Run `python -m playground probe` on the VPS and paste back any failure.

Credentials: the Theta Terminal holds them (start it with THETA_USER/THETA_PASS from the
environment or its own creds file). This client never sends or stores credentials.
"""
from __future__ import annotations

import io
import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

DEFAULT_BASE = "http://127.0.0.1:25503"   # TODO-VERIFY: v3 terminal default port


class EP:
    """Endpoint paths (single source of truth)."""
    EXPIRATIONS = "/v3/option/list/expirations"       # TODO-VERIFY
    STRIKES = "/v3/option/list/strikes"               # TODO-VERIFY
    OPT_QUOTE = "/v3/option/history/quote"            # TODO-VERIFY
    OPT_EOD = "/v3/option/history/eod"                # TODO-VERIFY
    STK_QUOTE = "/v3/stock/history/quote"             # TODO-VERIFY
    STK_EOD = "/v3/stock/history/eod"                 # TODO-VERIFY
    IDX_PRICE = "/v3/index/history/price"             # TODO-VERIFY
    IDX_EOD = "/v3/index/history/eod"                 # TODO-VERIFY


class P:
    """Parameter names / value formats (single source of truth)."""
    symbol, expiration, strike, right = "symbol", "expiration", "strike", "right"   # TODO-VERIFY
    date, start_date, end_date = "date", "start_date", "end_date"                   # TODO-VERIFY
    interval, start_time, end_time, fmt = "interval", "start_time", "end_time", "format"  # TODO-VERIFY
    BULK_STRIKE, BOTH = "*", "both"                                                 # TODO-VERIFY
    RIGHT = {"C": "call", "P": "put"}                                               # TODO-VERIFY

    @staticmethod
    def d(x: date) -> str:           # TODO-VERIFY: YYYYMMDD vs YYYY-MM-DD
        return x.strftime("%Y%m%d")

    @staticmethod
    def strike_val(k: float) -> str:  # TODO-VERIFY: dollars with decimals in v3
        return f"{k:.3f}"

    @staticmethod
    def t(minute: int) -> str:
        return f"{minute // 60:02d}:{minute % 60:02d}:00"


NO_DATA_STATUS = {472, 404}   # TODO-VERIFY: Theta v3 "no data" status codes


@dataclass
class ApiResult:
    ok: bool
    status: int
    text: str
    latency: float
    nbytes: int
    url: str
    params: dict
    error: str = ""
    empty: bool = False
    retries: int = 0
    throttled: int = 0


class RateLimiter:
    def __init__(self, max_rps: float):
        self.min_gap = 1.0 / max_rps if max_rps > 0 else 0.0
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self):
        with self._lock:
            now = time.monotonic()
            t = max(now, self._next)
            self._next = t + self.min_gap
        if t > now:
            time.sleep(t - now)


class ThetaClient:
    def __init__(self, base: str | None = None, max_rps: float = 5.0, session=None, timeout: float = 120.0,
                 max_retries: int = 6, sleep=time.sleep):
        self.base = (base or os.environ.get("THETA_BASE") or DEFAULT_BASE).rstrip("/")
        if session is None:
            import requests
            session = requests.Session()
        self.s = session
        self.limiter = RateLimiter(max_rps)
        self.timeout, self.max_retries, self._sleep = timeout, max_retries, sleep
        self.stats = {"requests": 0, "bytes": 0, "throttled": 0, "errors": 0, "empty": 0, "latency_sum": 0.0}
        self._lock = threading.Lock()

    # ------------------------------------------------------------ transport
    def get(self, path: str, params: dict) -> ApiResult:
        url = self.base + path
        params = {**params, P.fmt: "csv"}   # TODO-VERIFY csv is accepted/default
        throttled = 0
        err = ""
        for attempt in range(self.max_retries + 1):
            self.limiter.wait()
            t0 = time.monotonic()
            try:
                r = self.s.get(url, params=params, timeout=self.timeout)
                lat = time.monotonic() - t0
                status, text = r.status_code, r.text
            except Exception as e:   # connection errors: back off and retry
                lat, status, text = time.monotonic() - t0, -1, ""
                err = f"{type(e).__name__}: {e}"
            if status == 200 or status in NO_DATA_STATUS:
                empty = status != 200 or not text.strip() or len(text.strip().splitlines()) <= 1
                with self._lock:
                    self.stats["requests"] += 1
                    self.stats["bytes"] += len(text)
                    self.stats["latency_sum"] += lat
                    self.stats["empty"] += int(empty)
                    self.stats["throttled"] += throttled
                return ApiResult(True, status, text, lat, len(text), url, params, empty=empty,
                                 retries=attempt, throttled=throttled)
            if status in (429, 503) or "too many" in text.lower() or status == -1 or status >= 500:
                throttled += int(status == 429 or status == 503)
                self._sleep(min(2.0 ** attempt, 60.0))
                err = err or f"HTTP {status}: {text[:300]}"
                continue
            err = f"HTTP {status}: {text[:500]}"
            break
        with self._lock:
            self.stats["errors"] += 1
            self.stats["throttled"] += throttled
        return ApiResult(False, status, text, lat, len(text), url, params, error=err, retries=attempt,
                         throttled=throttled)

    # ------------------------------------------------------------ parsing
    @staticmethod
    def parse(text: str) -> pd.DataFrame:
        t = text.strip()
        if not t:
            return pd.DataFrame()
        if t[0] in "[{":
            obj = json.loads(t)
            if isinstance(obj, dict):
                obj = obj.get("response", obj.get("data", [obj]))
            return pd.DataFrame(obj)
        return pd.read_csv(io.StringIO(t))

    @staticmethod
    def norm_quotes(df: pd.DataFrame, strike: float | None = None, right: str | None = None) -> pd.DataFrame:
        """-> ts,strike,right,bid,ask,bid_size,ask_size   (TODO-VERIFY column names)."""
        cols = ["ts", "strike", "right", "bid", "ask", "bid_size", "ask_size"]
        if df.empty:
            return pd.DataFrame(columns=cols)
        df = df.rename(columns={c: c.lower() for c in df.columns})
        if "timestamp" in df:
            ts = pd.to_datetime(df["timestamp"])
        elif "datetime" in df:
            ts = pd.to_datetime(df["datetime"])
        else:   # v2-style fallback: date + ms_of_day
            ts = pd.to_datetime(df["date"].astype(str)) + pd.to_timedelta(df["ms_of_day"], unit="ms")
        out = pd.DataFrame({"ts": ts.dt.tz_localize(None) if getattr(ts.dt, "tz", None) else ts})
        out["strike"] = df["strike"].astype(float) if "strike" in df else (strike if strike is not None else float("nan"))
        if "right" in df:
            out["right"] = df["right"].astype(str).str.upper().str[0]
        else:
            out["right"] = right
        for c in ("bid", "ask"):
            out[c] = df[c].astype(float)
        for c in ("bid_size", "ask_size"):
            out[c] = df[c].astype(float) if c in df else float("nan")
        return out[cols]

    # ------------------------------------------------------------ endpoints
    def expirations(self, symbol="SPY") -> tuple[list[date], ApiResult]:
        r = self.get(EP.EXPIRATIONS, {P.symbol: symbol})
        if not r.ok or r.empty:
            return [], r
        df = self.parse(r.text)
        col = "expiration" if "expiration" in df else df.columns[0]
        return sorted(pd.to_datetime(df[col]).dt.date.unique()), r

    def strikes(self, expiry: date, symbol="SPY") -> tuple[list[float], ApiResult]:
        r = self.get(EP.STRIKES, {P.symbol: symbol, P.expiration: P.d(expiry)})
        if not r.ok or r.empty:
            return [], r
        df = self.parse(r.text)
        col = "strike" if "strike" in df else df.columns[0]
        return sorted(df[col].astype(float).unique().tolist()), r

    def option_quotes(self, expiry: date, right: str, strike: float, d0: date, d1: date, interval: str,
                      t0: int | None = None, t1: int | None = None, symbol="SPY"):
        p = {P.symbol: symbol, P.expiration: P.d(expiry), P.strike: P.strike_val(strike),
             P.right: P.RIGHT[right], P.interval: interval}
        if d0 == d1:
            p[P.date] = P.d(d0)
        else:
            p[P.start_date], p[P.end_date] = P.d(d0), P.d(d1)
        if t0 is not None:
            p[P.start_time], p[P.end_time] = P.t(t0), P.t(t1)
        r = self.get(EP.OPT_QUOTE, p)
        return (self.norm_quotes(self.parse(r.text), strike, right) if r.ok and not r.empty
                else self.norm_quotes(pd.DataFrame())), r

    def option_quotes_bulk(self, expiry: date, d: date, interval: str, t0: int, t1: int, symbol="SPY"):
        """All strikes, both rights, one day, narrow time window.  TODO-VERIFY strike=* + times."""
        p = {P.symbol: symbol, P.expiration: P.d(expiry), P.strike: P.BULK_STRIKE, P.right: P.BOTH,
             P.interval: interval, P.date: P.d(d), P.start_time: P.t(t0), P.end_time: P.t(t1)}
        r = self.get(EP.OPT_QUOTE, p)
        return (self.norm_quotes(self.parse(r.text)) if r.ok and not r.empty
                else self.norm_quotes(pd.DataFrame())), r

    def option_eod(self, expiry: date, right: str, strike: float, d0: date, d1: date, symbol="SPY"):
        p = {P.symbol: symbol, P.expiration: P.d(expiry), P.strike: P.strike_val(strike),
             P.right: P.RIGHT[right], P.start_date: P.d(d0), P.end_date: P.d(d1)}
        r = self.get(EP.OPT_EOD, p)
        if not r.ok or r.empty:
            return self.norm_quotes(pd.DataFrame()), r
        df = self.parse(r.text)
        df.columns = [c.lower() for c in df.columns]
        dcol = "created" if "created" in df else ("date" if "date" in df else "timestamp")
        df["timestamp"] = pd.to_datetime(df[dcol]).dt.normalize() + pd.Timedelta(hours=16)
        return self.norm_quotes(df, strike, right), r

    def stock_spot_bars(self, symbol: str, d0: date, d1: date, interval: str):
        """Quote-midpoint bars. -> DataFrame ts,spot."""
        p = {P.symbol: symbol, P.interval: interval}
        if d0 == d1:
            p[P.date] = P.d(d0)
        else:
            p[P.start_date], p[P.end_date] = P.d(d0), P.d(d1)
        r = self.get(EP.STK_QUOTE, p)
        if not r.ok or r.empty:
            return pd.DataFrame(columns=["ts", "spot"]), r
        q = self.norm_quotes(self.parse(r.text))
        q = q[(q.bid > 0) & (q.ask > 0)]
        return pd.DataFrame({"ts": q.ts, "spot": 0.5 * (q.bid + q.ask)}), r

    def eod(self, kind: str, symbol: str, d0: date, d1: date):
        """kind 'stock' or 'index'. -> DataFrame date,open,high,low,close."""
        path = EP.STK_EOD if kind == "stock" else EP.IDX_EOD
        r = self.get(path, {P.symbol: symbol, P.start_date: P.d(d0), P.end_date: P.d(d1)})
        if not r.ok or r.empty:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close"]), r
        df = self.parse(r.text)
        df.columns = [c.lower() for c in df.columns]
        dcol = next(c for c in ("created", "date", "timestamp") if c in df)
        out = df.assign(date=pd.to_datetime(df[dcol]).dt.date)[["date", "open", "high", "low", "close"]]
        return out.sort_values("date").reset_index(drop=True), r

    def index_price(self, symbol: str, d: date, interval: str):
        r = self.get(EP.IDX_PRICE, {P.symbol: symbol, P.date: P.d(d), P.interval: interval})
        return (self.parse(r.text) if r.ok and not r.empty else pd.DataFrame()), r
