"""Spot (underlying price) sources, coverage checks and cross-validation.

ThetaData Standard has no stock/index price history, so intraday spot comes from Alpaca (primary), with option-chain
put-call-parity spot as the fallback and daily closes as the last resort. Everything here that talks to a vendor is
UNVERIFIED until run on the VPS (TODO-VERIFY markers). Alpaca/Massive caches under data/spot/ are kept by
`purge-vendor-data` unless --include-spot. Keys come ONLY from env vars (ALPACA_API_KEY, ALPACA_SECRET_KEY,
MASSIVE_API_KEY) and are never printed or logged.
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .calendar import nyse_holidays, trading_days
from .provider import CLOSE_MIN, OPEN_MIN, MissingData, Provider, bar_minutes

MIN_BARS = 330                    # regular session has 390 one-minute bars; flag days below this
HALF_DAY_MIN_BARS = 180           # half days (13:00 close) have ~210 bars


# ---------------------------------------------------------------- NYSE half days (built-in calendar)
def half_days(year: int) -> set[date]:
    """Early-close days (13:00 ET): day after Thanksgiving, Christmas Eve (weekday), July 3 (when July 4 is a
    weekday other than Monday... NYSE rule: July 3 if it is a weekday and July 4 is not a Monday), plus a few
    historic one-offs. Approximate but explicit; edit if the exchange calendar differs."""
    out = set()
    nov = date(year, 11, 1)
    thanks = nov + timedelta(days=(3 - nov.weekday()) % 7 + 21)
    out.add(thanks + timedelta(days=1))
    c = date(year, 12, 24)
    if c.weekday() < 5:
        out.add(c)
    j3, j4 = date(year, 7, 3), date(year, 7, 4)
    if j3.weekday() < 5 and j4.weekday() != 0 and j3 not in nyse_holidays(year):
        out.add(j3)
    return {d for d in out if d.weekday() < 5 and d not in nyse_holidays(year)}


# ---------------------------------------------------------------- agreement statistics
def agreement_stats(a: pd.Series, b: pd.Series) -> dict:
    """Compare two price series on their common index: median/95th-pct abs diff, % within $0.05 / $0.25."""
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    if j.empty:
        return {"n": 0, "median_abs": None, "p95_abs": None, "within_005": None, "within_025": None}
    diff = (j["a"] - j["b"]).abs()
    return {"n": int(len(j)), "median_abs": float(diff.median()), "p95_abs": float(diff.quantile(0.95)),
            "within_005": float((diff <= 0.05).mean() * 100), "within_025": float((diff <= 0.25).mean() * 100)}


def stats_row(name: str, st: dict, note: str = "") -> str:
    if not st["n"]:
        return f"| {name} | 0 | - | - | - | - | {note or 'no overlap'} |"
    return (f"| {name} | {st['n']} | {st['median_abs']:.4f} | {st['p95_abs']:.4f} | {st['within_005']:.1f}% | "
            f"{st['within_025']:.1f}% | {note} |")


def write_validation_md(rows: list[str], path="results/spot_validation.md", notes: list[str] | None = None):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    md = ["# Spot validation (UNVERIFIED until run on the VPS with real keys)", "",
          "| comparison | n minutes | median abs diff | 95th pct abs diff | within $0.05 | within $0.25 | note |",
          "|---|---|---|---|---|---|---|"] + rows
    if notes:
        md += ["", *[f"- {n}" for n in notes]]
    p.write_text("\n".join(md) + "\n")


# ---------------------------------------------------------------- Alpaca (primary)
class RateLimiter:
    def __init__(self, per_min: float = 150.0):
        self.gap, self._next, self._lock = 60.0 / per_min, 0.0, threading.Lock()

    def wait(self):
        with self._lock:
            now = time.monotonic()
            t = max(now, self._next)
            self._next = t + self.gap
        if t > now:
            time.sleep(t - now)


class AlpacaClient:
    """Historical stock bars. TODO-VERIFY: endpoint, params (feed=sip, adjustment, page_token) and response shape."""
    BASE = "https://data.alpaca.markets"

    def __init__(self, session=None, per_min: float = 150.0, sleep=time.sleep):
        self.key, self.secret = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
        if session is None:
            import requests
            session = requests.Session()
        self.s, self.lim, self._sleep = session, RateLimiter(per_min), sleep

    def _get(self, path: str, params: dict) -> dict:
        if not self.key or not self.secret:
            raise RuntimeError("ALPACA_API_KEY / ALPACA_SECRET_KEY not set in the environment")
        hdr = {"APCA-API-KEY-ID": self.key, "APCA-API-SECRET-KEY": self.secret}
        for attempt in range(6):
            self.lim.wait()
            r = self.s.get(self.BASE + path, params=params, headers=hdr, timeout=60)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                self._sleep(min(2.0 ** attempt, 60.0))
                continue
            raise RuntimeError(f"Alpaca HTTP {r.status_code}: {r.text[:300]}")   # body never contains keys
        raise RuntimeError("Alpaca: retries exhausted")

    def bars(self, symbol: str, start: str, end: str, timeframe: str = "1Min", adjustment: str = "raw") -> pd.DataFrame:
        """Paginated; returns DataFrame t(UTC), o,h,l,c,v."""
        rows, tok = [], None
        while True:
            p = {"symbols": symbol, "timeframe": timeframe, "start": start, "end": end, "limit": 10000,
                 "adjustment": adjustment, "feed": "sip", "sort": "asc"}
            if tok:
                p["page_token"] = tok
            js = self._get("/v2/stocks/bars", p)
            rows += (js.get("bars") or {}).get(symbol, [])
            tok = js.get("next_page_token")
            if not tok:
                break
        if not rows:
            return pd.DataFrame(columns=["t", "o", "h", "l", "c", "v"])
        df = pd.DataFrame(rows)[["t", "o", "h", "l", "c", "v"]]
        df["t"] = pd.to_datetime(df["t"], utc=True)
        return df


def to_et(ts: pd.Series) -> pd.Series:
    return ts.dt.tz_convert("America/New_York").dt.tz_localize(None)


# ---------------------------------------------------------------- store (data/spot/)
class SpotStore:
    """data/spot/<T>/min1/date=YYYY-MM-DD.parquet (ET minute open prices, regular session) and daily series."""

    def __init__(self, root: str | Path = "data/spot"):
        self.root = Path(root)

    def _p(self, t, d):
        return self.root / t / "min1" / f"date={d}.parquet"

    def has_day(self, t, d) -> bool:
        return self._p(t, d).exists()

    def write_day(self, t, d, df: pd.DataFrame):
        p = self._p(t, d)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        df.to_parquet(tmp, index=False, compression="zstd")
        os.replace(tmp, p)

    def read_day(self, t, d) -> pd.DataFrame:
        return pd.read_parquet(self._p(t, d))

    def save_daily(self, t, kind, df):
        p = self.root / t / f"daily_{kind}.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p)

    def load_daily(self, t, kind="raw"):
        p = self.root / t / f"daily_{kind}.parquet"
        return pd.read_parquet(p) if p.exists() else None

    def bad_days(self, t) -> set[str]:
        p = self.root / t / "bad_days.json"
        return set(json.loads(p.read_text())) if p.exists() else set()

    def set_bad_days(self, t, days):
        p = self.root / t / "bad_days.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(sorted(days)))


def regular_session(df: pd.DataFrame) -> pd.DataFrame:
    """Keep 09:30-15:59 ET one-minute bars; adds `m` (minute of day) and `px` (bar open)."""
    t = to_et(df["t"])
    m = t.dt.hour * 60 + t.dt.minute
    out = df.assign(ts=t, m=m)
    return out[(out.m >= OPEN_MIN) & (out.m < CLOSE_MIN)][["ts", "m", "o", "c"]].rename(columns={"o": "px"}).reset_index(drop=True)


def min_bars_needed(d: date) -> int:
    return HALF_DAY_MIN_BARS if d in half_days(d.year) else MIN_BARS


def download_spot(client: AlpacaClient, store: SpotStore, tickers, start: date, end: date, log=print,
                  chunk_days: int = 20) -> dict:
    """Resumable: days already stored are skipped. Writes per-day minute-open parquet + daily raw/adjusted."""
    summary = {}
    days = trading_days(start, end)
    for t in tickers:
        todo = [d for d in days if not store.has_day(t, d)]
        got = 0
        for k in range(0, len(todo), chunk_days):
            chunk = todo[k:k + chunk_days]
            df = client.bars(t, f"{chunk[0]}T00:00:00Z", f"{chunk[-1] + timedelta(days=1)}T00:00:00Z")
            rs = regular_session(df) if len(df) else df
            for d in chunk:
                day = rs[rs.ts.dt.date == d] if len(rs) else rs
                store.write_day(t, d, day[["ts", "m", "px", "c"]] if len(day) else
                                pd.DataFrame({"ts": pd.Series([], dtype="datetime64[ns]"), "m": pd.Series([], dtype=int),
                                              "px": pd.Series([], dtype=float), "c": pd.Series([], dtype=float)}))
                got += len(day) > 0
            log(f"spot {t}: {min(k + chunk_days, len(todo))}/{len(todo)} days fetched")
        for kind, adj in (("raw", "raw"), ("adj", "all")):
            dd = client.bars(t, f"{start}T00:00:00Z", f"{end + timedelta(days=1)}T00:00:00Z", "1Day", adj)
            if len(dd):
                dd = dd.assign(date=pd.to_datetime(dd.t).dt.tz_convert("America/New_York").dt.date)
                store.save_daily(t, kind, dd.set_index("date")[["o", "h", "l", "c", "v"]]
                                 .rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}))
        summary[t] = {"days_requested": len(todo), "days_with_bars": got}
    return summary


def coverage(store: SpotStore, ticker: str, start: date, end: date) -> pd.DataFrame:
    rows = []
    for d in trading_days(start, end):
        if not store.has_day(ticker, d):
            rows.append((d, 0, min_bars_needed(d), "missing"))
            continue
        n = len(store.read_day(ticker, d))
        need = min_bars_needed(d)
        rows.append((d, n, need, "ok" if n >= need else "LOW"))
    return pd.DataFrame(rows, columns=["date", "bars", "needed", "status"])


def write_coverage_md(covs: dict[str, pd.DataFrame], path="results/spot_coverage.md"):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    md = ["# Spot coverage (bars per regular session; < 330 flagged, half days use a lower bar)", "",
          "Days flagged here are never interpolated: entries are blocked (logged as `nodata`) or the engine uses "
          "the parity fallback (`--spot-fallback`).", ""]
    for t, df in covs.items():
        bad = df[df.status != "ok"]
        md += [f"## {t}", f"- days checked: {len(df)}; ok: {int((df.status == 'ok').sum())}; flagged: {len(bad)}"]
        if len(bad):
            md += ["- flagged days (first 60): " + ", ".join(f"{r.date} ({r.bars})" for r in bad.head(60).itertuples())]
        md += [""]
    p.write_text("\n".join(md))


# ---------------------------------------------------------------- parity spot (option-implied)
def parity_bars(provider: Provider, d: date, res: str, rate: float, ref_spot: float,
                dte_range=(5, 20)) -> pd.DataFrame:
    """Spot from put-call parity at the strike nearest `ref_spot`: S = K*exp(-rT) + C_mid - P_mid.
    Dividends ignored (error ~ dividend*T; keep DTE short). Uses the provider's own quotes."""
    exps = [e for e in provider.list_expiries(d) if dte_range[0] <= (e - d).days <= dte_range[1]]
    if not exps:
        raise MissingData(f"parity spot: no expiry with DTE {dte_range} on {d}")
    e = min(exps, key=lambda x: abs((x - d).days - 10))
    ks = provider.list_strikes(d, e)
    K = min(ks, key=lambda k: abs(k - ref_spot))
    c = provider.get_option_quotes(d, e, "C", [K], OPEN_MIN, CLOSE_MIN, res)
    p = provider.get_option_quotes(d, e, "P", [K], OPEN_MIN, CLOSE_MIN, res)
    if c.empty or p.empty:
        raise MissingData(f"parity spot: no quotes for K={K} {e}")
    j = c.merge(p, on="ts", suffixes=("_c", "_p"))
    j = j[(j.ask_c > 0) & (j.ask_p > 0) & (j.bid_c <= j.ask_c) & (j.bid_p <= j.ask_p)]
    if j.empty:
        raise MissingData("parity spot: no valid pairs")
    mins = (j.ts.dt.hour * 60 + j.ts.dt.minute).to_numpy()
    T = ((e - d).days * 1440.0 + (CLOSE_MIN - mins)) / (365.0 * 1440.0)
    spot = K * np.exp(-rate * T) + 0.5 * (j.bid_c + j.ask_c).to_numpy() - 0.5 * (j.bid_p + j.ask_p).to_numpy()
    return pd.DataFrame({"spot": spot}, index=pd.DatetimeIndex(j.ts))


class AlpacaSpot:
    """Per-ticker spot router for ThetaProvider: alpaca -> (parity fallback) ; parity ; daily."""

    def __init__(self, ticker: str, store: SpotStore, rate: float = 0.036):
        self.ticker, self.store, self.rate = ticker, store, rate

    def bars_alpaca(self, d: date, res: str) -> pd.DataFrame:
        if str(d) in self.store.bad_days(self.ticker):
            raise MissingData(f"{self.ticker} {d}: flagged bad by validation")
        if not self.store.has_day(self.ticker, d):
            raise MissingData(f"{self.ticker} spot minute bars missing for {d}")
        df = self.store.read_day(self.ticker, d)
        if len(df) < min_bars_needed(d):
            raise MissingData(f"{self.ticker} {d}: only {len(df)} minute bars")
        px = pd.Series(df["px"].to_numpy(float), index=df["m"].to_numpy())
        grid = bar_minutes(res)
        vals, idx = [], []
        for m in grid:
            if m == CLOSE_MIN:
                v = float(df["c"].iloc[-1])                       # 16:00 print = close of the 15:59 bar
            else:
                v = px.get(m)
                if v is None or v != v:
                    continue
            vals.append(v)
            idx.append(pd.Timestamp(d) + pd.Timedelta(minutes=m))
        if not vals:
            raise MissingData(f"{self.ticker} {d}: no bars on the {res} grid")
        return pd.DataFrame({"spot": vals}, index=pd.DatetimeIndex(idx))


def daily_bar(daily: pd.DataFrame, d: date) -> pd.DataFrame:
    return pd.DataFrame({"spot": [float(daily.loc[d, "close"])]},
                        index=pd.DatetimeIndex([pd.Timestamp(d) + pd.Timedelta(minutes=CLOSE_MIN)]))


def route_spot(provider: Provider, d: date, res: str, source: str, fallback: str, primary, rate: float):
    """Shared routing used by ThetaProvider and the synthetic provider. primary(d,res) -> bars or MissingData."""
    daily = provider.get_underlying_daily()
    if source == "daily":
        return daily_bar(daily, d), "daily"
    pos = daily.index.get_loc(d)
    ref = float(daily["close"].iloc[pos - 1]) if pos > 0 else float(daily["close"].iloc[pos])
    if source == "parity":
        return parity_bars(provider, d, res, rate, ref), "parity"
    try:
        return primary(d, res), "alpaca" if provider.name != "synthetic" else "synthetic"
    except MissingData:
        if fallback == "parity":
            return parity_bars(provider, d, res, rate, ref), "parity"
        raise


# ---------------------------------------------------------------- Massive (cross-check) - UNVERIFIED
class MassiveClient:
    """Minute aggregates (Polygon-style REST). TODO-VERIFY base URL, path and response fields. Free tier reaches roughly
    two years back: whatever range comes back is used and reported."""
    BASE = "https://api.massive.com"

    def __init__(self, session=None):
        self.key = os.environ.get("MASSIVE_API_KEY")
        if session is None:
            import requests
            session = requests.Session()
        self.s = session

    def minute_opens(self, ticker: str, d: date) -> pd.Series | None:
        """Series (index = minute of day ET, value = bar open) or None if no data / not permitted."""
        if not self.key:
            raise RuntimeError("MASSIVE_API_KEY not set in the environment")
        r = self.s.get(f"{self.BASE}/v2/aggs/ticker/{ticker}/range/1/minute/{d}/{d}",
                       params={"adjusted": "false", "sort": "asc", "limit": 50000, "apiKey": self.key}, timeout=60)
        if r.status_code != 200:
            return None
        res = r.json().get("results") or []
        if not res:
            return None
        t = pd.to_datetime([x["t"] for x in res], unit="ms", utc=True).tz_convert("America/New_York")
        m = t.hour * 60 + t.minute
        s = pd.Series([x["o"] for x in res], index=m)
        return s[(s.index >= OPEN_MIN) & (s.index < CLOSE_MIN)]


def alpaca_opens(store: SpotStore, ticker: str, d: date) -> pd.Series | None:
    if not store.has_day(ticker, d):
        return None
    df = store.read_day(ticker, d)
    return pd.Series(df["px"].to_numpy(float), index=df["m"].to_numpy()) if len(df) else None


def sample_days(store: SpotStore, ticker: str, start: date, end: date, n: int) -> list[date]:
    ok = [d for d in trading_days(start, end) if store.has_day(ticker, d)
          and len(store.read_day(ticker, d)) >= min_bars_needed(d)]
    if len(ok) <= n:
        return ok
    idx = np.linspace(0, len(ok) - 1, n).astype(int)
    return [ok[i] for i in idx]


def pooled(pairs: list[tuple[pd.Series, pd.Series]]) -> dict:
    parts_a, parts_b = [], []
    for i, (x, y) in enumerate(pairs):
        j = pd.concat([x.rename("a"), y.rename("b")], axis=1, join="inner").dropna()
        parts_a.append(j["a"].reset_index(drop=True))
        parts_b.append(j["b"].reset_index(drop=True))
    if not parts_a:
        return agreement_stats(pd.Series(dtype=float), pd.Series(dtype=float))
    return agreement_stats(pd.concat(parts_a, ignore_index=True), pd.concat(parts_b, ignore_index=True))


def validate_massive(store, mc, ticker, days) -> tuple[dict, dict, list[str]]:
    pairs, per_day, got = [], {}, []
    for d in days:
        b = mc.minute_opens(ticker, d)
        a = alpaca_opens(store, ticker, d)
        if b is None or a is None:
            continue
        got.append(str(d))
        pairs.append((a, b))
        per_day[str(d)] = agreement_stats(a, b)["median_abs"]
    return pooled(pairs), per_day, got


def validate_parity(store, provider, ticker, days, rate, res="5m") -> tuple[dict, dict]:
    pairs, per_day = [], {}
    for d in days:
        a = alpaca_opens(store, ticker, d)
        daily = provider.get_underlying_daily()
        if a is None or d not in daily.index:
            continue
        pos = daily.index.get_loc(d)
        ref = float(daily["close"].iloc[pos - 1]) if pos else float(daily["close"].iloc[pos])
        try:
            pb = parity_bars(provider, d, res, rate, ref)
        except MissingData:
            continue
        pm = pd.Series(pb["spot"].to_numpy(float), index=(pb.index.hour * 60 + pb.index.minute))
        pairs.append((a, pm))
        per_day[str(d)] = agreement_stats(a, pm)["median_abs"]
    return pooled(pairs), per_day


def validate_greeks_field(store, client, ticker, samples) -> tuple[dict, bool]:
    """samples: [(day, expiry, strike, minute)]. Compares Alpaca spot with the Greeks response's underlying_price."""
    a_list, b_list, exists = [], [], False
    for d, e, k, m in samples:
        df, r = client.greeks_bulk(e, d, "1m", m, m + 1, ticker, strike=k, right="C")
        if r.ok and len(df) and df["underlying_price"].notna().any():
            exists = True
            a = alpaca_opens(store, ticker, d)
            if a is not None and m in a.index:
                a_list.append(float(a[m]))
                b_list.append(float(df["underlying_price"].dropna().iloc[0]))
    return agreement_stats(pd.Series(a_list), pd.Series(b_list)), exists


def run_validations(args, store, tickers, start, end, alpaca_client):
    """Orchestrates --validate massive,greeks,parity. Returns (markdown rows, notes). UNVERIFIED paths."""
    want = {x.strip() for x in args.validate.split(",") if x.strip()}
    rows, notes, bad = [], [], {}
    for t in tickers:
        days = sample_days(store, t, start, end, args.sample_days)
        if not days:
            notes.append(f"{t}: no covered days to validate")
            continue
        if "massive" in want:
            try:
                st, per, got = validate_massive(store, MassiveClient(), t, days)
                rows.append(stats_row(f"{t}: Alpaca vs Massive", st,
                                      f"Massive returned {len(got)}/{len(days)} sampled days ({', '.join(got[:2])}..)" if got
                                      else "Massive returned nothing (free tier ~2y reach; key/plan?)"))
                if got:
                    notes.append(f"{t}: Massive range actually returned: {min(got)}..{max(got)}")
                for d, v in per.items():
                    if v is not None and v > 0.25:
                        bad.setdefault(t, set()).add(d)
            except Exception as e:
                notes.append(f"{t}: Massive validation failed: {type(e).__name__}: {str(e)[:120]}")
        if want & {"greeks", "parity"}:
            try:
                from .downloader import Fetcher
                from . import factory
                from .theta_client import ThetaClient
                client = ThetaClient(max_rps=4.0)
                cache = factory.ticker_cache(args.data_dir, t)
                prov0 = factory.theta(t, args.data_dir, None, 0.036)
                daily = prov0.get_underlying_daily()
                fetcher = Fetcher(client, cache, list(daily.index), 2, symbol=t)
                prov = factory.theta(t, args.data_dir, fetcher, 0.036)
                if "parity" in want:
                    st, per = validate_parity(store, prov, t, days[: max(2, args.sample_days // 2)], 0.036)
                    rows.append(stats_row(f"{t}: Alpaca vs put-call-parity spot", st,
                                          "div ignored, ATM strike, 5-20 DTE; includes bid/ask noise"))
                    for d, v in per.items():
                        if v is not None and v > 0.25:
                            bad.setdefault(t, set()).add(d)
                if "greeks" in want:
                    samples = []
                    for d in days[:3]:
                        exps = [e for e in prov.list_expiries(d) if 7 <= (e - d).days <= 30]
                        if not exps:
                            continue
                        e = exps[0]
                        ks = prov.list_strikes(d, e)
                        c = float(daily.loc[d, "close"])
                        k = min([x for x in ks if x == int(x)] or ks, key=lambda x: abs(x - c))
                        samples += [(d, e, k, m) for m in (10 * 60, 12 * 60, 14 * 60)]
                    st, exists = validate_greeks_field(store, client, t, samples)
                    rows.append(stats_row(f"{t}: Alpaca vs ThetaData Greeks underlying_price", st,
                                          "field present" if exists else "underlying_price field NOT present / endpoint unavailable"))
            except Exception as e:
                notes.append(f"{t}: Theta-based validations failed: {type(e).__name__}: {str(e)[:120]}")
    if getattr(args, "mark_bad", True):
        for t, ds in bad.items():
            store.set_bad_days(t, store.bad_days(t) | ds)
            notes.append(f"{t}: {len(ds)} day(s) disagreed by > $0.25 (median) and were flagged; the engine will use the parity fallback there")
    return rows, notes
