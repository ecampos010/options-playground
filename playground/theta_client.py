"""Thin ThetaData Theta Terminal (v3 REST) adapter.

Status (ThetaData STANDARD options account, Terminal v3, checked on a VPS):
  VERIFIED   option expirations list, option strikes list, single-contract option history quote
             (columns bid_size, bid_exchange, bid, bid_condition, ask_size, ask_exchange, ask,
             ask_condition + a timestamp column; the parser tolerates several timestamp names).
  VERIFIED   HTTP 472 "No data found" (empty body) = contract not listed/tradable that date -> NO_DATA,
             not an error: never retried, cached as a negative result, counted separately.
  NOT AVAILABLE on this account: stock and index history (HTTP 403; Stock/Index = FREE tier).
  VERIFIED   /v3/option/history/greeks/first_order (Standard); /greeks/all is Pro-only and refused on Standard.
  UNVERIFIED (TODO-VERIFY): bulk strike=*, option EOD, v2 fallbacks.
ALL endpoint paths and parameter names live in `EP` / `P` below.

Credentials: the Theta Terminal holds them (start it with THETA_USER/THETA_PASS from the
environment or its own creds file). This client never sends or stores credentials.
"""
from __future__ import annotations

import io
import json
import os
import threading
import time
from dataclasses import dataclass
from datetime import date

import pandas as pd

DEFAULT_BASE = "http://127.0.0.1:25503"   # TODO-VERIFY: v3 terminal default port


class EP:
    """Endpoint paths (single source of truth)."""
    EXPIRATIONS = "/v3/option/list/expirations"       # VERIFIED (Standard)
    STRIKES = "/v3/option/list/strikes"               # VERIFIED (Standard)
    OPT_QUOTE = "/v3/option/history/quote"            # VERIFIED single contract (Standard); bulk form TODO-VERIFY
    OPT_EOD = "/v3/option/history/eod"                # TODO-VERIFY
    # VERIFIED (Standard): first-order Greeks. /greeks/all (2nd/3rd order) is Pro-only and is refused on Standard.
    OPT_GREEKS = "/v3/option/history/greeks/first_order"
    OPT_GREEKS_PRO = "/v3/option/history/greeks/all"  # Pro only; not used
    STK_QUOTE = "/v3/stock/history/quote"             # 403 on Standard account (Stock: FREE)
    STK_EOD = "/v3/stock/history/eod"                 # 403 on Standard account
    IDX_PRICE = "/v3/index/history/price"             # 403 on Standard account (Index: FREE)
    IDX_EOD = "/v3/index/history/eod"                 # 403 on Standard account


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


# VERIFIED first_order columns: symbol,expiration,strike,right,timestamp,bid,ask,delta,theta,vega,rho,epsilon,lambda,
# implied_vol,iv_error,underlying_timestamp,underlying_price.  Params: expiration/date as YYYY-MM-DD.
IV_ERROR_MAX = 0.05          # |iv_error| above this = the IV solver did not converge -> junk row
IV_ERROR_SENTINEL = 100.0    # ThetaData writes iv_error=100 when it could not solve (e.g. the 09:30 row, bid=ask=0)
STALE_UNDERLYING_S = 120.0   # underlying_timestamp older than the bar by more than this = stale opening bar


def iso(x: date) -> str:
    return x.strftime("%Y-%m-%d")


def junk_greeks_mask(df: pd.DataFrame) -> pd.Series:
    """True for rows that must not be used as a delta source: zero/missing bid or ask, IV solver error above
    IV_ERROR_MAX (incl. the 100.0 sentinel), non-finite delta, or a stale underlying timestamp. Columns that are
    absent from the frame are not tested."""
    bad = pd.Series(False, index=df.index)
    for c in ("bid", "ask"):
        if c in df:
            v = pd.to_numeric(df[c], errors="coerce")
            bad |= ~(v > 0)
    if "iv_error" in df:
        e = pd.to_numeric(df["iv_error"], errors="coerce")
        bad |= ~(e.abs() <= IV_ERROR_MAX)          # NaN, 100.0 sentinel and anything above the threshold
    if "delta" in df:
        d = pd.to_numeric(df["delta"], errors="coerce")
        bad |= ~d.abs().between(0.0, 1.0)          # NaN or outside [-1, 1]
    if "timestamp" in df and "underlying_timestamp" in df:
        lag = (pd.to_datetime(df["timestamp"], errors="coerce") -
               pd.to_datetime(df["underlying_timestamp"], errors="coerce")).dt.total_seconds()
        bad |= lag > STALE_UNDERLYING_S
    return bad


def is_junk_quote(bid: float, ask: float) -> bool:
    """A quote row with no market on at least one side (bid==0 or ask==0, NaN, or crossed)."""
    return not (bid > 0 and ask > 0 and bid <= ask)


CONCURRENCY_CAP = 4   # ThetaData Standard allows 4 concurrent requests (Value: 2); hard cap regardless of --threads

NO_DATA_STATUS = 472   # VERIFIED: "HTTP 472 No data found", empty body = not listed/tradable that date


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
    no_data: bool = False      # HTTP 472: contract not listed/tradable that date (NOT an error)
    denied: bool = False       # HTTP 403: endpoint not included in the subscription


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
        self._sem = threading.BoundedSemaphore(CONCURRENCY_CAP)
        self.stats = {"requests": 0, "bytes": 0, "throttled": 0, "errors": 0, "empty": 0, "no_data": 0,
                      "denied": 0, "latency_sum": 0.0, "ok_requests": 0}
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
                with self._sem:
                    r = self.s.get(url, params=params, timeout=self.timeout)
                lat = time.monotonic() - t0
                status, text = r.status_code, r.text
            except Exception as e:   # connection errors: back off and retry
                lat, status, text = time.monotonic() - t0, -1, ""
                err = f"{type(e).__name__}: {e}"
            if status == NO_DATA_STATUS:       # not an error, not retried; caller caches the negative
                with self._lock:
                    self.stats["requests"] += 1
                    self.stats["no_data"] += 1
                    self.stats["throttled"] += throttled
                return ApiResult(True, status, text, lat, len(text), url, params, empty=True,
                                 retries=attempt, throttled=throttled, no_data=True)
            if status == 200:
                empty = not text.strip() or len(text.strip().splitlines()) <= 1
                with self._lock:
                    self.stats["requests"] += 1
                    self.stats["ok_requests"] += 1
                    self.stats["bytes"] += len(text)
                    self.stats["latency_sum"] += lat
                    self.stats["empty"] += int(empty)
                    self.stats["throttled"] += throttled
                return ApiResult(True, status, text, lat, len(text), url, params, empty=empty,
                                 retries=attempt, throttled=throttled)
            if status == 403:                  # endpoint not in subscription: do not retry
                with self._lock:
                    self.stats["denied"] += 1
                return ApiResult(False, 403, text, lat, len(text), url, params, retries=attempt,
                                 error=f"HTTP 403 (not permitted on this subscription): {text[:300]}",
                                 denied=True)
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
        """-> ts,strike,right,bid,ask,bid_size,ask_size. bid/ask/bid_size/ask_size VERIFIED; timestamp name tolerant."""
        cols = ["ts", "strike", "right", "bid", "ask", "bid_size", "ask_size"]
        if df.empty:       # typed, so .dt / float ops work on NO_DATA (472) results
            return pd.DataFrame({"ts": pd.Series([], dtype="datetime64[ns]"), "strike": pd.Series([], dtype=float),
                                 "right": pd.Series([], dtype=object), "bid": pd.Series([], dtype=float),
                                 "ask": pd.Series([], dtype=float), "bid_size": pd.Series([], dtype=float),
                                 "ask_size": pd.Series([], dtype=float)})[cols]
        df = df.rename(columns={c: c.lower() for c in df.columns})
        tcol = next((c for c in ("timestamp", "datetime", "quote_time", "time", "ts", "created", "last_trade")
                     if c in df), None)
        if tcol is not None:
            ts = pd.to_datetime(df[tcol])
        elif "date" in df and "ms_of_day" in df:   # v2-style fallback
            ts = pd.to_datetime(df["date"].astype(str)) + pd.to_timedelta(df["ms_of_day"], unit="ms")
        else:
            raise KeyError(f"no timestamp column in quote response; columns={list(df.columns)}")
        if getattr(ts.dt, "tz", None) is not None:
            ts = ts.dt.tz_localize(None)
        out = pd.DataFrame({"ts": ts})
        out["strike"] = df["strike"].astype(float) if "strike" in df else (strike if strike is not None else float("nan"))
        if "right" in df:
            out["right"] = df["right"].astype(str).str.upper().str[0]
        else:
            out["right"] = right
        for c in ("bid", "ask"):
            out[c] = df[c].astype(float)
        for c in ("bid_size", "ask_size"):
            out[c] = df[c].astype(float) if c in df else float("nan")
        # no-market rows (bid==0 AND ask==0, e.g. the 09:30 opening print) carry no information: drop them.
        # One-sided rows (bid 0, ask > 0) are kept here - they are legitimate far-OTM wings; Greeks rows use the stricter rule.
        out = out[~((out["bid"].fillna(0) == 0) & (out["ask"].fillna(0) == 0))]
        return out[cols].reset_index(drop=True)

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

    def greeks_bulk(self, expiry: date, d: date, interval: str, t0: int, t1: int, symbol="SPY",
                    strike: float | None = None, right: str | None = None):
        """First-order Greeks at one bar: all strikes (strike=*) or one contract.
        VERIFIED endpoint/columns (see EP.OPT_GREEKS). Junk rows (bid/ask 0, |iv_error| > IV_ERROR_MAX, stale underlying)
        are dropped and counted in stats['greeks_rejected']. -> (DataFrame right,strike,delta,iv,underlying_price, r)."""
        p = {P.symbol: symbol, P.expiration: iso(expiry),
             P.strike: P.BULK_STRIKE if strike is None else P.strike_val(strike),
             P.right: P.BOTH if right is None else P.RIGHT[right], P.interval: interval, P.date: iso(d),
             P.start_time: P.t(t0), P.end_time: P.t(t1)}
        r = self.get(EP.OPT_GREEKS, p)
        cols = ["right", "strike", "delta", "iv", "underlying_price"]
        if not r.ok or r.empty:
            return pd.DataFrame({c: pd.Series([], dtype=float if c != "right" else object) for c in cols}), r
        df = self.parse(r.text)
        df.columns = [c.lower() for c in df.columns]
        junk = junk_greeks_mask(df)
        if junk.any():
            with self._lock:
                self.stats["greeks_rejected"] = self.stats.get("greeks_rejected", 0) + int(junk.sum())
            df = df[~junk]
        out = pd.DataFrame({
            "right": df["right"].astype(str).str.upper().str[0] if "right" in df else (right or ""),
            "strike": df["strike"].astype(float) if "strike" in df else strike,
            "delta": df["delta"].astype(float) if "delta" in df else float("nan"),
            "iv": next((df[c].astype(float) for c in ("implied_vol", "iv", "implied_volatility") if c in df), float("nan")),
            "underlying_price": df["underlying_price"].astype(float) if "underlying_price" in df else float("nan")})
        return out[cols].reset_index(drop=True), r

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
