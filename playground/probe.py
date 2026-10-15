"""`probe`: finds out what the local Theta Terminal / subscription really provides. Fails soft.

Verified on a Standard options account: expirations/strikes lists and single-contract 1m option
quotes work; HTTP 472 = NO_DATA (not an error); stock and index history are 403 (INFO only).
Contract selection mirrors the downloader: candidate expirations come from the list, Fridays first,
and each is CONFIRMED with a trial quote (1-minute window); strikes come from the strikes list.
Every record keeps endpoint, params, HTTP status and raw error text so failures can be pasted back.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path


from . import rules
from .config import build_config
from .theta_client import EP, ThetaClient

YEARS = (1, 3, 5, 8, 10)
ENTRY_MIN = 10 * 60   # 10:00 trial / snapshot minute (probe only; the engine samples its own entry minute)


def _wk(d: date) -> date:
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def _q(vals, p):
    vals = sorted(vals)
    return vals[min(len(vals) - 1, int(p * len(vals)))] if vals else None


class Probe:
    def __init__(self, client: ThetaClient, today: date | None = None, out_dir="results", burst=24,
                 data_dir="data", spot_fn=None, tickers=("SPY",)):
        self.c, self.today, self.out = client, today or date.today(), Path(out_dir)
        self.burst, self.data_dir, self.spot_fn = burst, data_dir, spot_fn
        self.tickers = tuple(tickers)
        self.checks, self.measured, self.samples = [], {}, []
        self.dte = build_config("green").dte

    # ------------------------------------------------------------------ records
    def rec(self, section, name, ok, detail="", r=None, level=None):
        lvl = level or ("PASS" if ok else "FAIL")
        self.checks.append({"section": section, "name": name, "ok": bool(ok) or lvl == "INFO", "level": lvl,
                            "detail": detail, "endpoint": r.url if r is not None else "",
                            "params": r.params if r is not None else {},
                            "status": r.status if r is not None else None,
                            "raw_error": (r.error or "")[:600] if r is not None else "",
                            "latency_s": round(r.latency, 3) if r is not None else None,
                            "bytes": r.nbytes if r is not None else None})

    def safe(self, section, name, fn):
        try:
            fn()
        except Exception as e:
            self.rec(section, name, False, f"EXCEPTION {type(e).__name__}: {e}")

    def spot_for(self, d: date, ticker: str = "SPY"):
        """Underlying close for d (stock history is 403 on Standard): test hook -> Alpaca daily cache -> cached
        daily -> yfinance."""
        if self.spot_fn and ticker == "SPY":
            return self.spot_fn(d)
        from .cache import DataCache
        from .spot import SpotStore
        store = SpotStore(Path(self.data_dir) / "spot")
        for df in (store.load_daily(ticker, "raw"), DataCache(self.data_dir).load_daily(ticker)):
            if df is not None and len(df):
                ix = df.index[df.index <= d]
                if len(ix) and (d - ix[-1]).days <= 5:
                    return float(df.loc[ix[-1], "close"])
        # Alpaca minute bars (data/spot/<T>/min1): last regular-session bar close on the nearest earlier cached day
        for back in range(0, 6):
            day = d - timedelta(days=back)
            if store.has_day(ticker, day):
                bars = store.read_day(ticker, day)
                if len(bars):
                    return float(bars.sort_values("m")["c"].iloc[-1])
        from .dailydata import spy_from_yfinance
        y = spy_from_yfinance(d - timedelta(days=3), d, ticker)
        if y is not None and len(y):
            return float(y["close"].iloc[-1])
        return None

    # ------------------------------------------------------------------ (a)
    def check_a(self):
        cfg = build_config("green")
        for tk in self.tickers:
            sec = f"a: expired {tk} contracts listable (candidates confirmed by trial quote)"
            exps, r = self.c.expirations(tk)
            self.rec(sec, f"{tk} list expirations", r.ok and bool(exps), f"{len(exps)} expirations"
                     + (f", {exps[0]}..{exps[-1]}" if exps else ""), r)
            for y in YEARS:
                self.safe(sec, f"{tk} {y}y", lambda y=y, tk=tk, sec=sec, exps=exps: self._one_year(sec, y, exps, cfg, tk))

    def _one_year(self, sec, y, exps, cfg, tk="SPY"):
        try:
            target = _wk(self.today.replace(year=self.today.year - y))
        except ValueError:
            target = _wk(self.today - timedelta(days=365 * y))
        spot = self.spot_for(target, tk)
        if spot is None:
            self.rec(sec, f"{y}y ago ({target})", False,
                     "no spot source (stock history is 403 on Standard; Theta is not used for spot): run `spot` (Alpaca "
                     "daily/minute bars), `download --daily-only`, or pip install yfinance")
            return
        ranked = rules.rank_expiries(exps, target, cfg)
        if not ranked:
            self.rec(sec, f"{y}y ago ({target})", False, "no expiration in the 30-45 DTE window in the list")
            return
        tgt_put = spot * 0.954
        confirmed, tried = None, 0
        for e in ranked[:8]:
            ks, r2 = self.c.strikes(e, tk)
            ints = [k for k in ks if k == int(k)]
            if not ints:
                self.rec(sec, f"{y}y {target} expiry {e} ({e:%a}) strikes", r2.ok, "no integer strikes in list", r2,
                         level="INFO")
                continue
            k = min(ints, key=lambda x: abs(x - tgt_put))
            df, r = self.c.option_quotes(e, "P", k, target, target, "1m", ENTRY_MIN, ENTRY_MIN + 1, tk)
            tried += 1
            if r.no_data:
                self.rec(sec, f"{y}y {target} trial expiry {e} ({e:%a}) P{k:g}", True,
                         "NO_DATA (HTTP 472): listed but not tradable that date - skipped, not an error", r, level="INFO")
                continue
            if r.ok and len(df):
                confirmed = (e, k, ks, ints)
                self.rec(sec, f"{tk} {y}y ago ({target}) expiry {e} ({e:%a}) P{k:g}", True,
                         f"confirmed after {tried} trial(s); {len(ks)} listed strikes", r)
                near = sorted(x for x in ks if abs(x - spot) <= 0.05 * spot)
                gaps = sorted({round(b - a, 4) for a, b in zip(near, near[1:])})
                self.measured.setdefault("strike_increments", {}).setdefault(tk, {})[str(target.year)] = gaps
                self.rec(sec, f"{tk} strike increments near the money in {target.year}", True,
                         f"distinct gaps within +/-5% of spot: {gaps} (half-dollar strikes: "
                         f"{'yes' if any(abs(g - round(g)) > 1e-9 for g in gaps) else 'no'}); listed strikes "
                         f"{len(near)} in band", None, level="INFO")
                break
            self.rec(sec, f"{y}y {target} trial expiry {e} P{k:g}", False, f"rows={len(df)}", r)
        if confirmed:
            e, k, ks, ints = confirmed
            self.samples.append((y, target, e, k, spot, ints, tk))
        else:
            self.rec(sec, f"{tk} {y}y ago ({target})", False, f"no listed expiry confirmed in {tried} trial(s)")

    # ------------------------------------------------------------------ (b)
    def check_b(self):
        sec = "b: historical option quotes (bid/ask/sizes)"
        if not self.samples:
            self.rec(sec, "quote history", False, "skipped: (a) confirmed no contract to test")
            return
        for y, target, e, k, spot, ints, tk in [x for x in self.samples if x[6] == self.tickers[0]][:5]:
            for itv in ("1m", "30s", "1s"):    # only 1m is required; finer intervals are INFO
                def one(itv=itv, y=y, target=target, e=e, k=k):
                    df, r = self.c.option_quotes(e, "P", k, target, target, itv, ENTRY_MIN, ENTRY_MIN + 2)
                    good = len(df) > 0 and df[["bid", "ask"]].notna().all().all()
                    sz = len(df) > 0 and df["bid_size"].notna().any()
                    msg = (f"rows={len(df)} bid/ask={'yes' if good else 'NO'} sizes={'yes' if sz else 'NO'}"
                           + (" NO_DATA(472)" if r.no_data else ""))
                    self.rec(sec, f"{y}y P{k:g} {e} interval={itv}", r.ok and good, msg, r,
                             level=None if itv == "1m" else "INFO")
                self.safe(sec, itv, one)

    # ------------------------------------------------------------------ (c)
    def check_c(self):
        sec = "c: size & latency, one day x ~40 integer strikes, 1m, full session"
        if not self.samples:
            self.rec(sec, "band", False, "no confirmed contract from (a)")
            return
        y, target, e, k0, spot, ints, tk = self.samples[0]
        i = min(range(len(ints)), key=lambda j: abs(ints[j] - k0))
        band = ints[max(0, i - 20): i + 20]
        lat, nbytes, rows, nodata, bad = [], 0, 0, 0, []
        t0 = datetime.now()
        for k in band:
            df, r = self.c.option_quotes(e, "P", k, target, target, "1m", 9 * 60 + 30, 16 * 60)
            if r.no_data:
                nodata += 1
            elif not r.ok:
                bad.append(r)
            elif len(df):
                lat.append(r.latency)       # only real, successful responses with rows
                nbytes += r.nbytes
                rows += len(df)
        wall = (datetime.now() - t0).total_seconds()
        if len(lat) >= 5:
            self.measured.update({"option_quote_latency_p50_s": _q(lat, 0.5),
                                  "option_quote_latency_p95_s": _q(lat, 0.95),
                                  "option_quote_samples": len(lat),
                                  "option_quote_rows_per_request": rows / len(lat),
                                  "option_quote_bytes_per_row": nbytes / max(rows, 1)})
        self.rec(sec, f"{len(band)} per-strike requests on {e} ({y}y, {target})", not bad and len(lat) >= 5,
                 f"ok_with_rows={len(lat)} NO_DATA={nodata} errors={len(bad)} rows={rows} bytes={nbytes} "
                 f"({nbytes / 1e6:.2f} MB) wall={wall:.1f}s"
                 + (f" latency p50={_q(lat, .5):.3f}s p95={_q(lat, .95):.3f}s max={max(lat):.3f}s" if lat else ""),
                 bad[0] if bad else None)

    def check_c2(self):
        sec = "c2: bulk entry snapshot (strike=*, both rights, one expiry, one minute)"
        if not self.samples:
            self.rec(sec, "bulk", False, "no confirmed contract from (a)")
            return
        y, target, e, k0, spot, ints, tk = self.samples[0]
        df, r = self.c.option_quotes_bulk(e, target, "1m", ENTRY_MIN, ENTRY_MIN + 1)
        if r.no_data:
            self.rec(sec, f"bulk {e} {target}", False, "NO_DATA (HTTP 472) for the bulk form", r)
        elif not r.ok:
            self.rec(sec, f"bulk {e} {target}", False,
                     f"UNSUPPORTED/ERROR status={r.status} raw={r.error[:300]!r}", r)
        else:
            try:
                raw_cols = list(self.c.parse(r.text).columns)
            except Exception as ex:
                raw_cols = [f"parse error {ex}"]
            self.rec(sec, f"bulk {e} {target}", len(df) > 0,
                     f"rows={len(df)} strikes={df.strike.nunique() if len(df) else 0} "
                     f"rights={sorted(df.right.dropna().unique()) if len(df) else []} bytes={r.nbytes} "
                     f"latency={r.latency:.3f}s columns={raw_cols}", r)
            self.measured["bulk_entry_supported"] = bool(len(df))
            if len(df):
                self.measured["bulk_entry_latency_s"] = r.latency
                self.measured["bulk_entry_rows"] = len(df)
        self.measured.setdefault("bulk_entry_supported", False)

    # ------------------------------------------------------------------ (d)
    def check_d(self):
        sec = "d: request-rate behaviour (limits: Standard 4 concurrent, Value 2)"
        c2 = ThetaClient(self.c.base, max_rps=0, session=self.c.s, max_retries=0)
        if self.samples:
            y, target, e, k, *_ = self.samples[0]
            call = lambda _: c2.option_quotes(e, "P", k, target, target, "1m", 9 * 60 + 30, 16 * 60)[1]
            what = f"quote P{k:g} {e}"
        else:
            call = lambda _: c2.get(EP.EXPIRATIONS, {"symbol": "SPY"})
            what = "expirations"
        t0 = datetime.now()
        with ThreadPoolExecutor(8) as ex:
            rs = list(ex.map(call, range(self.burst)))
        wall = (datetime.now() - t0).total_seconds()
        st = {}
        for r in rs:
            st[r.status] = st.get(r.status, 0) + 1
        msgs = sorted({(r.text or r.error)[:120] for r in rs if r.status != 200 and (r.text or r.error)})[:3]
        throttled = st.get(429, 0) + st.get(503, 0)
        self.measured["burst_statuses"] = {str(k): v for k, v in st.items()}
        self.rec(sec, f"burst of {self.burst} x {what}, 8 threads, no limiter, no retries", True,
                 f"statuses={st} wall={wall:.1f}s ({self.burst / max(wall, 1e-9):.1f} rps) throttled={throttled} "
                 f"messages={msgs}", rs[0], level="INFO")

    # ------------------------------------------------------------------ (e)
    def check_e(self):
        sec = "e: stock / index / VIX (INFO only - not needed: VIX from Cboe CSV, SPY daily from yfinance)"
        d0 = _wk(self.today - timedelta(days=365 * 3))
        denied = False
        for kind, sym in (("stock", "SPY"), ("index", "SPX"), ("index", "VIX")):
            df, r = self.c.eod(kind, sym, d0, d0 + timedelta(days=10))
            denied = denied or r.denied
            self.rec(sec, f"{kind} {sym} eod from {d0}", True,
                     ("NOT PERMITTED on this subscription (HTTP 403)" if r.denied else
                      f"rows={len(df)}" + (" NO_DATA(472)" if r.no_data else "")), r, level="INFO")
        for sym in ("VIX", "VIXW", "SPX"):
            ex, r = self.c.expirations(sym)
            self.rec(sec, f"option expirations for {sym}", True,
                     "NOT PERMITTED (403)" if r.denied else f"{len(ex)} expirations", r, level="INFO")
        if denied:
            self.rec(sec, "consequence", True,
                     "stock history is unavailable, so intraday SPY bars cannot come from ThetaData; the engine needs a "
                     "spot source for 1m/5m/15m runs (only `--resolution eod` is self-contained)", level="INFO")
        self.measured["stock_history_denied"] = denied

    # ------------------------------------------------------------------ (f) Greeks + per-ticker snapshot speed
    def check_f(self):
        sec = "f: Greeks fields and bulk chain snapshot per ticker (TODO-VERIFY endpoints)"
        seen = set()
        for y, target, e, k, spot, ints, tk in self.samples:
            if tk in seen:
                continue
            seen.add(tk)
            df, r = self.c.greeks_bulk(e, target, "1m", ENTRY_MIN, ENTRY_MIN + 1, tk, strike=k, right="P")
            if r.denied or (not r.ok):
                self.rec(sec, f"{tk} Greeks single contract {e} P{k:g}", False,
                         f"UNAVAILABLE status={r.status} raw={r.error[:200]!r} -> engine will use Black-Scholes delta from mid IV", r)
                self.measured.setdefault("greeks_fields", {})[tk] = None
                continue
            try:
                cols = list(self.c.parse(r.text).columns) if not r.empty else []
            except Exception:
                cols = []
            has = {c: (c in [x.lower() for x in cols]) for c in ("delta", "implied_vol", "underlying_price")}
            self.measured.setdefault("greeks_fields", {})[tk] = has
            self.rec(sec, f"{tk} Greeks single contract {e} P{k:g}", bool(len(df)),
                     f"rows={len(df)} fields: delta={has['delta']} implied_vol={has['implied_vol']} "
                     f"underlying_price={has['underlying_price']} columns={cols}", r)
            b, rb = self.c.option_quotes_bulk(e, target, "1m", ENTRY_MIN, ENTRY_MIN + 1, tk)
            if rb.ok and len(b):
                self.rec(sec, f"{tk} bulk chain snapshot {e}", True,
                         f"rows={len(b)} strikes={b.strike.nunique()} bytes={rb.nbytes} latency={rb.latency:.2f}s", rb)
                self.measured.setdefault("bulk_snapshot_latency_s", {})[tk] = rb.latency
            else:
                self.rec(sec, f"{tk} bulk chain snapshot {e}", False,
                         f"UNSUPPORTED/EMPTY status={rb.status} raw={rb.error[:200]!r}", rb)

    # ------------------------------------------------------------------
    def run(self) -> dict:
        for fn in (self.check_a, self.check_b, self.check_c, self.check_c2, self.check_d, self.check_e, self.check_f):
            self.safe(fn.__name__, fn.__name__, fn)
        return self.write()

    def write(self) -> dict:
        self.out.mkdir(parents=True, exist_ok=True)
        rep = {"when": datetime.now().isoformat(timespec="seconds"), "base": self.c.base,
               "client_stats": self.c.stats, "measured": self.measured, "checks": self.checks}
        (self.out / "probe_report.json").write_text(json.dumps(rep, indent=2, default=str))
        md = ["# ThetaData probe report", f"_run {rep['when']} against `{self.c.base}`_", "",
              "PASS/FAIL cover options data only. INFO lines are informational (HTTP 472 NO_DATA is not an error; "
              "stock/index 403 is expected on a Standard options account).", ""]
        sec = None
        for c in self.checks:
            if c["section"] != sec:
                sec = c["section"]
                md += ["", f"## {sec}"]
            md.append(f"- {c['level']} **{c['name']}** - {c['detail']}")
            if c["level"] == "FAIL":
                md.append(f"  - endpoint: `{c['endpoint']}` params: `{c['params']}` status: {c['status']}")
                if c["raw_error"]:
                    md.append(f"  - raw error: `{c['raw_error']}`")
        md += ["", "## Measured (used by `plan`; only successful requests with rows)", "```",
               json.dumps(self.measured, indent=1, default=str), "```"]
        (self.out / "probe_report.md").write_text("\n".join(md) + "\n")
        return rep
