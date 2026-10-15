"""Offline stand-in for the Theta Terminal HTTP API, backed by SyntheticProvider.

It speaks the response SHAPES this repo ASSUMES for ThetaData v3 (see theta_client.EP/P). It exists
to test the downloader/cache/probe plumbing in a sandbox; it proves nothing about the real API.
"""
from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from .provider import CLOSE_MIN, OPEN_MIN, SyntheticProvider
from .theta_client import EP, P


class FakeResp:
    def __init__(self, status: int, text: str):
        self.status_code, self.text, self.headers = status, text, {}


def _d(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d" if "-" in s else "%Y%m%d").date()


def _min(s: str) -> int:
    h, m = s.split(":")[:2]
    return int(h) * 60 + int(m)


class FakeSession:
    def __init__(self, prov: SyntheticProvider | None = None, throttle_every: int = 0, fail_paths: tuple = (),
                 only_fridays: bool = False, half_strikes: bool = False, dead_expiries: tuple = (),
                 deny_stock: bool = False, provs: dict | None = None, greeks: bool = True):
        """only_fridays: 472 for non-Friday expirations; half_strikes: list .5 strikes but 472 on them before
        2020; dead_expiries: 472 for these expirations; deny_stock: 403 on stock/index history."""
        self.only_fridays, self.half_strikes = only_fridays, half_strikes
        self.dead, self.deny_stock = {str(x) for x in dead_expiries}, deny_stock
        self.provs, self.greeks = provs or {}, greeks
        self.prov = prov or SyntheticProvider()
        self.calls: list[tuple[str, dict]] = []
        self.throttle_every, self.fail_paths = throttle_every, fail_paths

    def get(self, url, params=None, timeout=None):
        path = "/" + url.split("//", 1)[1].split("/", 1)[1]
        p = dict(params or {})
        self.calls.append((path, p))
        if self.deny_stock and path in (EP.STK_QUOTE, EP.STK_EOD, EP.IDX_EOD, EP.IDX_PRICE):
            return FakeResp(403, "Your subscription does not include this endpoint")
        if path in self.fail_paths:
            return FakeResp(500, "synthetic failure")
        if self.throttle_every and len(self.calls) % self.throttle_every == 0:
            return FakeResp(429, "Too many requests")
        try:
            return FakeResp(200, self._route(path, p))
        except KeyError as e:
            return FakeResp(472, f"no data {e}")
        except PermissionError as e:
            return FakeResp(403, f"not permitted {e}")

    # ------------------------------------------------------------------
    def _days(self, p):
        if P.date in p:
            return [_d(p[P.date])] * 1
        a, b = _d(p[P.start_date]), _d(p[P.end_date])
        return [d for d in self.prov.days if a <= d <= b]

    def _route(self, path, p) -> str:
        pv = self.provs.get(p.get(P.symbol), self.prov)
        if path == EP.OPT_GREEKS:
            if not self.greeks:
                raise PermissionError("greeks not in plan")
            return self._greeks(p, pv)
        if path == EP.EXPIRATIONS:
            if p[P.symbol] != "SPY" and p[P.symbol] not in self.provs:
                return "expiration\n" if p[P.symbol] != "VIX" else "expiration\n2025-01-22\n"
            ex = sorted({e for d in pv.days[::5] for e in pv.list_expiries(d)})
            return "expiration\n" + "\n".join(str(e) for e in ex)
        if path == EP.STRIKES:
            e = _d(p[P.expiration])
            d = max(x for x in pv.days if x < e) if e > pv.days[0] else pv.days[0]
            ks = pv.list_strikes(d, e)
            if self.half_strikes:
                ks = sorted(set(ks) | {k + 0.5 for k in ks})
            return "strike\n" + "\n".join(f"{k:.3f}" for k in ks)
        if path == EP.OPT_QUOTE:
            return self._opt_quote(p)
        if path == EP.STK_QUOTE:
            res = p[P.interval]
            rows = ["timestamp,bid_size,bid,ask_size,ask"]
            for d in self._days(p):
                b = pv.get_underlying_bars(d, res)
                for t, s in zip(b.index, b.spot):
                    rows.append(f"{t.isoformat()},100,{s - 0.005:.4f},100,{s + 0.005:.4f}")
            return "\n".join(rows)
        if path in (EP.STK_EOD, EP.IDX_EOD):
            a, b = _d(p[P.start_date]), _d(p[P.end_date])
            sym = p[P.symbol]
            df = pv.get_underlying_daily() if sym == "SPY" else pv.get_vix_daily().to_frame("close").assign(
                open=lambda x: x.close, high=lambda x: x.close, low=lambda x: x.close)
            df = df[(df.index >= a) & (df.index <= b)]
            return "created,open,high,low,close\n" + "\n".join(
                f"{i},{r.open},{r.high},{r.low},{r.close}" for i, r in df.iterrows())
        if path == EP.IDX_PRICE:
            return "timestamp,price\n2025-01-02T09:30:00,100.0\n"
        if path == EP.OPT_EOD:
            e, k, r = _d(p[P.expiration]), float(p[P.strike]), "C" if p[P.right] == "call" else "P"
            rows = ["created,bid,ask,bid_size,ask_size"]
            for d in self._days(p):
                if d >= e:
                    continue
                q = pv.get_option_quotes(d, e, r, [k], CLOSE_MIN, CLOSE_MIN, "5m")
                rows += [f"{d},{x.bid},{x.ask},{x.bid_size},{x.ask_size}" for x in q.itertuples()]
            return "\n".join(rows)
        raise KeyError(path)

    def _greeks(self, p, pv) -> str:
        from . import bs
        e, res = _d(p[P.expiration]), p[P.interval]
        t0, t1 = _min(p[P.start_time]), _min(p[P.end_time])
        d = _d(p[P.date])
        i = pv._pos[d]
        spot_arr = pv._spots[i]
        rows = ["timestamp,symbol,expiration,strike,right,bid,ask,delta,implied_vol,iv_error,underlying_timestamp,underlying_price"]
        ks = pv.list_strikes(d, e)
        for r in ("C", "P"):
            q = pv.get_option_quotes(d, e, r, ks, t0, t1 - 1 if t1 > t0 else t0, res)
            for x in q.itertuples():
                m = pd.Timestamp(x.ts).hour * 60 + pd.Timestamp(x.ts).minute
                S = float(spot_arr[(m - OPEN_MIN) // 5])
                sig = float(pv._vix.iloc[i]) / 100 * 0.8 * pv.vol_scale * ((1.2 if r == "P" else 0.8) if pv.skew else 1.0)
                T = ((e - d).days * 1440.0 + (CLOSE_MIN - m)) / (365.0 * 1440.0)
                rows.append(f"{pd.Timestamp(x.ts).isoformat()},SPY,{e},{x.strike:.3f},{'CALL' if r == 'C' else 'PUT'},"
                            f"{x.bid},{x.ask},{bs.delta_s(S, x.strike, T, pv.r, sig, r):.5f},{sig:.4f},0.0001,"
                            f"{pd.Timestamp(x.ts).isoformat()},{S:.4f}")
        if len(rows) == 1:
            raise KeyError("no data")
        return "\n".join(rows)

    def _opt_quote(self, p) -> str:
        pv = self.provs.get(p.get(P.symbol), self.prov)
        e, res = _d(p[P.expiration]), p[P.interval]
        t0 = _min(p[P.start_time]) if P.start_time in p else OPEN_MIN
        t1 = _min(p[P.end_time]) if P.end_time in p else CLOSE_MIN
        bulk = p[P.strike] == P.BULK_STRIKE
        if (self.only_fridays and e.weekday() != 4) or str(e) in self.dead:
            raise KeyError("expiration not tradable")
        rights = ["C", "P"] if p[P.right] == P.BOTH else ["C" if p[P.right] == "call" else "P"]
        rows = ["timestamp,symbol,expiration,strike,right,bid_size,bid,ask_size,ask"]
        for d in self._days(p):
            if d >= e:
                continue
            ks = pv.list_strikes(d, e) if bulk else [float(p[P.strike])]
            if self.half_strikes and d.year < 2020:     # half-dollar strikes did not exist yet
                ks = [k for k in ks if k == int(k)]
            for r in rights:
                q = pv.get_option_quotes(d, e, r, ks, t0, t1, res)
                for x in q.itertuples():
                    rows.append(f"{pd.Timestamp(x.ts).isoformat()},SPY,{e},{x.strike:.3f},"
                                f"{'CALL' if r == 'C' else 'PUT'},{x.bid_size},{x.bid},{x.ask_size},{x.ask}")
        if len(rows) == 1:
            raise KeyError("no data")
        return "\n".join(rows)
