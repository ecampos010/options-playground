"""`probe`: finds out what the local Theta Terminal / subscription really provides. Fails soft.

*** The adapter is UNVERIFIED until this has been run on the VPS. *** Every check records the exact
endpoint, parameters, HTTP status and raw error text so failures can be pasted back verbatim.
"""
from __future__ import annotations

import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from .theta_client import EP, ThetaClient

YEARS = (1, 3, 5, 8, 10)


def _wk(d: date) -> date:
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def _q(vals, p):
    vals = sorted(vals)
    return vals[min(len(vals) - 1, int(p * len(vals)))] if vals else None


class Probe:
    def __init__(self, client: ThetaClient, today: date | None = None, out_dir="results", burst=40):
        self.c, self.today, self.out = client, today or date.today(), Path(out_dir)
        self.burst, self.checks, self.measured = burst, [], {}

    def rec(self, section, name, ok, detail="", r=None):
        self.checks.append({"section": section, "name": name, "ok": bool(ok), "detail": detail,
                            "endpoint": r.url if r is not None else "", "params": r.params if r is not None else {},
                            "status": r.status if r is not None else None,
                            "raw_error": (r.error or "")[:600] if r is not None else "",
                            "latency_s": round(r.latency, 3) if r is not None else None,
                            "bytes": r.nbytes if r is not None else None})

    def safe(self, section, name, fn):
        try:
            fn()
        except Exception as e:
            self.rec(section, name, False, f"EXCEPTION {type(e).__name__}: {e}")

    # ------------------------------------------------------------------ (a)
    def check_a(self):
        sec = "a: expired SPY contracts listable"
        exps, r = self.c.expirations("SPY")
        self.rec(sec, "list expirations", r.ok and bool(exps), f"{len(exps)} expirations"
                 + (f", {exps[0]}..{exps[-1]}" if exps else ""), r)
        self.exps = exps
        for y in YEARS:
            def one(y=y):
                target = _wk(self.today.replace(year=self.today.year - y) if not (self.today.month == 2 and self.today.day == 29)
                             else self.today - timedelta(days=365 * y))
                cand = [e for e in exps if target + timedelta(days=30) <= e <= target + timedelta(days=45)]
                if not cand:
                    self.rec(sec, f"{y}y ago ({target})", False, "no expiration in 30-45 DTE window in listing")
                    return
                e = cand[len(cand) // 2]
                ks, r2 = self.c.strikes(e, "SPY")
                self.rec(sec, f"{y}y ago ({target}) expiry {e}", r2.ok and bool(ks),
                         f"{len(ks)} strikes" + (f" {ks[0]}..{ks[-1]}" if ks else ""), r2)
                if ks:
                    self.samples = getattr(self, "samples", []) + [(y, target, e, ks)]
            self.safe(sec, f"{y}y", one)

    # ------------------------------------------------------------------ (b)
    def check_b(self):
        sec = "b: historical option quotes (bid/ask/sizes)"
        if not getattr(self, "samples", []):
            self.rec(sec, "quote history", False, "skipped: (a) found no listable expired contract to test")
        for y, target, e, ks in getattr(self, "samples", [])[:5]:
            def one(y=y, target=target, e=e, ks=ks):
                eod, r0 = self.c.eod("stock", "SPY", target, target)
                spot = float(eod.close.iloc[0]) if len(eod) else ks[len(ks) // 2]
                self.rec(sec, f"SPY eod {target}", r0.ok and len(eod) > 0, f"close={spot}", r0)
                k = min(ks, key=lambda s: abs(s - spot))
                for itv in ("1m", "30s", "1s"):    # TODO-VERIFY accepted interval strings
                    df, r = self.c.option_quotes(e, "P", k, target, target, itv, 10 * 60, 10 * 60 + 2)
                    cols_ok = len(df) > 0 and df[["bid", "ask"]].notna().all().all()
                    sz = len(df) > 0 and df["bid_size"].notna().any()
                    self.rec(sec, f"{y}y P{k} {e} interval={itv}", r.ok and cols_ok,
                             f"rows={len(df)} bid/ask={'yes' if cols_ok else 'NO'} sizes={'yes' if sz else 'NO'}", r)
                    if itv == "1m" and r.ok:
                        self.measured.setdefault("sample_contract", (str(e), k, str(target)))
            self.safe(sec, f"{y}y", one)

    # ------------------------------------------------------------------ (c)
    def check_c(self):
        sec = "c: size & latency, one day x ~40-strike band, 1m"
        s = getattr(self, "samples", [])
        if not s:
            self.rec(sec, "band", False, "no sample expiry from (a)")
            return
        y, target, e, ks = s[0]
        eod, _ = self.c.eod("stock", "SPY", target, target)
        spot = float(eod.close.iloc[0]) if len(eod) else ks[len(ks) // 2]
        i = min(range(len(ks)), key=lambda j: abs(ks[j] - spot))
        band = ks[max(0, i - 20): i + 20]
        lat, nbytes, rows, bad = [], 0, 0, []
        t0 = time.monotonic()
        for k in band:
            df, r = self.c.option_quotes(e, "P", k, target, target, "1m", 9 * 60 + 30, 16 * 60)
            lat.append(r.latency)
            nbytes += r.nbytes
            rows += len(df)
            if not r.ok:
                bad.append(r)
        wall = time.monotonic() - t0
        self.measured.update({"option_quote_latency_p50_s": _q(lat, 0.5), "option_quote_latency_p95_s": _q(lat, 0.95)})
        self.rec(sec, f"{len(band)} per-strike requests (sequential)", not bad,
                 f"rows={rows} bytes={nbytes} ({nbytes / 1e6:.2f} MB) wall={wall:.1f}s latency p50={_q(lat, .5):.3f}s "
                 f"p95={_q(lat, .95):.3f}s max={max(lat):.3f}s rows/req~{rows / max(1, len(band)):.0f}", bad[0] if bad else None)
        df, r = self.c.option_quotes_bulk(e, target, "1m", 10 * 60, 10 * 60 + 1)
        self.rec(sec, "bulk strike=* both rights, 1 minute window", r.ok and len(df) > 0,
                 f"rows={len(df)} bytes={r.nbytes} latency={r.latency:.3f}s", r)
        self.measured["bulk_entry_supported"] = bool(r.ok and len(df) > 0)

    # ------------------------------------------------------------------ (d)
    def check_d(self):
        sec = "d: request-rate behaviour"
        c2 = ThetaClient(self.c.base, max_rps=0, session=self.c.s, max_retries=0)
        t0 = time.monotonic()
        with ThreadPoolExecutor(8) as ex:
            rs = list(ex.map(lambda _: c2.get(EP.EXPIRATIONS, {"symbol": "SPY"}), range(self.burst)))
        wall = time.monotonic() - t0
        st = {}
        for r in rs:
            st[r.status] = st.get(r.status, 0) + 1
        msgs = sorted({r.text[:120] for r in rs if r.status not in (200,) and r.text})[:3]
        throttled = st.get(429, 0) + st.get(503, 0)
        self.measured["burst_statuses"] = {str(k): v for k, v in st.items()}
        self.rec(sec, f"burst of {self.burst} requests, 8 threads, no limiter",
                 True, f"statuses={st} wall={wall:.1f}s ({self.burst / wall:.1f} rps achieved) "
                       f"throttled={throttled} messages={msgs}", rs[0])

    # ------------------------------------------------------------------ (e)
    def check_e(self):
        sec = "e: index & VIX data"
        d0 = _wk(self.today - timedelta(days=365 * 3))
        for sym in ("SPX", "VIX"):
            df, r = self.c.eod("index", sym, d0, d0 + timedelta(days=10))
            self.rec(sec, f"{sym} index eod from {d0}", r.ok and len(df) > 0, f"rows={len(df)}", r)
            px, r = self.c.index_price(sym, d0, "1m")
            self.rec(sec, f"{sym} index price 1m {d0}", r.ok and len(px) > 0, f"rows={len(px)} cols={list(px.columns)[:8]}", r)
        for sym in ("VIX", "VIXW", "SPX"):
            ex, r = self.c.expirations(sym)
            self.rec(sec, f"option expirations for {sym}", r.ok and bool(ex), f"{len(ex)} expirations", r)

    # ------------------------------------------------------------------
    def run(self) -> dict:
        for fn in (self.check_a, self.check_b, self.check_c, self.check_d, self.check_e):
            self.safe(fn.__name__, fn.__name__, fn)
        return self.write()

    def write(self) -> dict:
        self.out.mkdir(parents=True, exist_ok=True)
        rep = {"when": datetime.now().isoformat(timespec="seconds"), "base": self.c.base,
               "client_stats": self.c.stats, "measured": self.measured, "checks": self.checks}
        (self.out / "probe_report.json").write_text(json.dumps(rep, indent=2, default=str))
        md = ["# ThetaData probe report", f"_run {rep['when']} against `{self.c.base}`_", "",
              "Adapter status: **UNVERIFIED** until every section below passes on the VPS.", ""]
        sec = None
        for c in self.checks:
            if c["section"] != sec:
                sec = c["section"]
                md += ["", f"## {sec}"]
            md.append(f"- {'PASS' if c['ok'] else 'FAIL'} **{c['name']}** - {c['detail']}")
            if not c["ok"]:
                md.append(f"  - endpoint: `{c['endpoint']}` params: `{c['params']}` status: {c['status']}")
                if c["raw_error"]:
                    md.append(f"  - raw error: `{c['raw_error']}`")
        md += ["", "## Measured (used by `plan`)", "```", json.dumps(self.measured, indent=1, default=str), "```"]
        (self.out / "probe_report.md").write_text("\n".join(md) + "\n")
        return rep
