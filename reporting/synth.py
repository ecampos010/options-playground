"""Synthetic trade-log generator (FIXTURES ONLY - not market data, not option pricing of any vendor).

A crude latent-path world: correlated daily returns (SPY/QQQ/IWM crash together, GLD/TLT less so), a VIX proxy,
and a smooth-payoff "fair-value" model for each strategy so that marks, stops, take-profits, rolls and
counterfactual hold-to-expiry P&L are all consistent with one underlying path per entry.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
from statistics import NormalDist

TICKERS = ["SPY", "QQQ", "IWM", "GLD", "TLT"]
STRATEGIES = ["iron_condor", "iron_butterfly", "bull_put_spread", "put_broken_wing_butterfly"]
CRASHES = [  # (start, end, daily drift, daily vol) of the market factor
    ("2018-01-29", "2018-02-09", -0.0100, 0.0200),
    ("2018-10-03", "2018-12-24", -0.0030, 0.0135),
    ("2020-02-20", "2020-04-07", -0.0035, 0.0320),
    ("2022-01-03", "2022-10-14", -0.0012, 0.0150),
    ("2025-04-02", "2025-04-09", -0.0150, 0.0300),
]
STRESS_WINDOWS = {
    "stress_volmageddon2018": ("2018-01-24", "2018-02-14", 1),
    "stress_covid2020": ("2020-02-18", "2020-03-31", 1),
    "stress_inflation2022": ("2022-01-03", "2022-06-30", 5),
}
_nd = NormalDist()
_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(17)
_GH_W = _GH_W / _GH_W.sum()


def _nd_cdf(x):
    from math import erf, sqrt
    return 0.5 * (1 + np.vectorize(lambda z: erf(z / sqrt(2)))(x))


def _ppf(p: float) -> float:
    return _nd.inv_cdf(p)


@dataclass
class World:
    dates: pd.DatetimeIndex
    logp: dict
    vix: np.ndarray
    term: np.ndarray
    ivm: dict


def make_world(seed: int, start: str, end: str) -> World:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n = len(dates)
    drift = np.full(n, 0.0004)
    vol = np.full(n, 0.0085)
    # slow vol regime wobble
    reg = np.zeros(n)
    for i in range(1, n):
        reg[i] = 0.98 * reg[i - 1] + rng.normal(0, 0.07)
    vol = vol * np.exp(0.35 * reg)
    for s, e, d, v in CRASHES:
        m = (dates >= s) & (dates <= e)
        drift[m] = d
        vol[m] = v
    mkt = drift + vol * rng.standard_t(6, n) / np.sqrt(1.5)
    crash_mask = np.zeros(n, bool)
    for s, e, *_ in CRASHES:
        crash_mask |= (dates >= s) & (dates <= e)
    idio = lambda sd: rng.normal(0, sd, n)  # noqa: E731
    r = {
        "SPY": mkt,
        "QQQ": 1.18 * mkt + idio(0.0045),
        "IWM": 1.12 * mkt + idio(0.0055),
        "GLD": 0.04 * mkt + 0.0002 + idio(0.0085) + np.where(crash_mask, 0.0006, 0),
        "TLT": -0.12 * mkt + idio(0.0075) + np.where(crash_mask, 0.0003, 0),
    }
    # 2022: bonds also fell
    m22 = (dates >= "2022-01-03") & (dates <= "2022-10-14")
    r["TLT"] = r["TLT"] + np.where(m22, -0.0020, 0)
    logp = {t: np.log({"SPY": 300, "QQQ": 200, "IWM": 150, "GLD": 120, "TLT": 125}[t]) + np.cumsum(x) for t, x in r.items()}
    # VIX proxy: asymmetric EWMA of market variance
    ew = 0.0001
    vix = np.zeros(n)
    for i in range(n):
        x = mkt[i]
        w = 1.6 if x < 0 else 0.7
        ew = 0.90 * ew + 0.10 * (w * x * x)
        vix[i] = 0.5 + 1.0 * np.sqrt(ew * 252) * 100 + rng.normal(0, 0.35)
    vix = np.clip(vix, 9.5, 85)
    term = np.clip(0.88 + 0.004 * np.maximum(vix - 18, 0) * 3 + rng.normal(0, 0.03, n), 0.7, 1.4)
    ivm = {"SPY": (1.0, 0.0), "QQQ": (1.25, 0.0), "IWM": (1.22, 0.01), "GLD": (0.45, 0.10), "TLT": (0.42, 0.07)}
    return World(dates, logp, vix, term, ivm)


def vix_bucket(v: float) -> str:
    return "lt15" if v < 15 else "15to20" if v < 20 else "20to30" if v < 30 else "gt30"


@dataclass
class Params:
    delta: float = 0.16
    tp: float | None = 0.5
    stop_mult: float | None = 2.0
    stop_kind: str = "stop"
    stop_floor: float = 0.25          # stop level = max(stop_mult*credit, stop_floor*width$)
    breach_exit: bool = False
    vix_trigger: bool = False
    time_exit_td: int | None = None   # exit this many trading days before expiry
    roll: bool = False
    dte_td: int = 25
    haircut: float = 0.06
    edge: float = 1.12                # credit as a multiple of model fair value (1.12 = sold 12% rich)


# ---- payoff shapes: N(u) = P&L excluding credit, as function of move u in sigma_T units -----------------
def _shape(strategy, u, k, wn, W, kc=None):
    """u: (...,) array; k,wn,W broadcastable. Returns N (<=0 for condor-like, signed for BWB)."""
    if strategy == "iron_condor":
        kc = k if kc is None else kc
        lf = np.clip(np.maximum((-u - k) / wn, (u - kc) / wn), 0, 1)
        return -W * lf
    if strategy == "bull_put_spread":
        return -W * np.clip((-u - k) / wn, 0, 1)
    if strategy == "iron_butterfly":
        return -W * np.clip(np.abs(u) / wn, 0, 1)
    if strategy == "put_broken_wing_butterfly":
        w1 = wn
        w2 = 2.0 * wn
        d = -u
        W1 = W * 0.5
        W2 = W
        up = np.clip((d - k) / w1, 0, 1) * W1                 # rise to +W1 at the short strike
        dn = np.clip((d - k - w1) / w2, 0, 1) * W2            # fall W2 below the short strike
        return up - dn
    raise ValueError(strategy)


def _expected_N(strategy, u, tau, k, wn, W):
    z = _GH_X.reshape((1,) * u.ndim + (-1,))
    uu = u[..., None] + np.sqrt(np.maximum(tau, 0))[..., None] * z if np.ndim(tau) else u[..., None] + np.sqrt(max(tau, 0)) * z
    kk, ww, WW = (np.asarray(a)[..., None] if np.ndim(a) else a for a in (k, wn, W))
    return (_shape(strategy, uu, kk, ww, WW) * _GH_W).sum(-1)


WIDTH_PCT = {"iron_condor": 0.025, "bull_put_spread": 0.025, "iron_butterfly": 0.05, "put_broken_wing_butterfly": 0.03}
FEES_LEGS = {"iron_condor": 4, "iron_butterfly": 4, "bull_put_spread": 2, "put_broken_wing_butterfly": 4}


def sim_leg(strategy, rel, iv, spot, vix_path, P: Params, rng, allow_roll=False):
    """Simulate one leg over relative log-return path rel (m, L+1). Returns dict of arrays."""
    m, L1 = rel.shape
    L = P.dte_td
    sig = iv * np.sqrt(L / 252.0)                       # sigma_T (log units)
    width = np.maximum(1.0, np.round(WIDTH_PCT[strategy] * spot))
    W = width * 100.0
    wn = (width / spot) / sig                            # wing width in sigma_T units
    if strategy == "put_broken_wing_butterfly":
        # strike offset k solved per row so the fair credit is ~5% of the wide wing (a small-credit BWB)
        lo, hi = np.full(m, -1.0), np.full(m, 3.0)
        for _ in range(30):
            mid = (lo + hi) / 2
            f = -_expected_N(strategy, np.zeros(m), np.ones(m), mid, wn, W)
            big = f > 0.05 * W
            lo, hi = np.where(big, mid, lo), np.where(big, hi, mid)
        k = (lo + hi) / 2
        c = 0.04 * W * (P.edge / 1.12)             # received a bit LESS than the solved fair credit
    else:
        k = _ppf(1 - P.delta) if strategy != "iron_butterfly" else 0.0
        fair = -_expected_N(strategy, np.zeros(m), np.ones(m), k, wn, W)
        c = np.maximum(fair * P.edge, 0.01 * W)
    u = rel[:, : L + 1] / sig[:, None]
    j = np.arange(L + 1)
    tau = (L - j) / L
    marks = np.empty((m, L + 1))
    for jj in range(L + 1):
        EN = _expected_N(strategy, u[:, jj], np.full(m, tau[jj]), k, wn, W)
        marks[:, jj] = c + EN - P.haircut * np.maximum(c, 0.02 * W)
    final = c + _shape(strategy, u[:, L], k, wn, W)
    marks[:, L] = final
    maxloss = np.maximum(W - c, 1.0) if strategy != "put_broken_wing_butterfly" else np.maximum(W * 0.5 - c, 1.0) + 0
    if strategy == "put_broken_wing_butterfly":
        maxloss = np.maximum(W - W * 0.5 - c, 1.0)       # W2-W1-c
    INF = L + 5
    cond = []                                            # priority order
    reasons = []
    stop_lvl = None
    if P.stop_mult is not None:
        stop_lvl = -np.maximum(P.stop_mult * c, P.stop_floor * W)
        hit = marks[:, 1:] <= stop_lvl[:, None]
        cond.append(np.where(hit.any(1), hit.argmax(1) + 1, INF)); reasons.append(P.stop_kind)
    if P.breach_exit or allow_roll:
        bk = k if strategy != "iron_butterfly" else 0.8 * wn
        if strategy == "bull_put_spread" or strategy == "put_broken_wing_butterfly":
            bu = -u
        else:
            bu = np.abs(u)
        hit = bu[:, 1:] >= (bk if np.ndim(bk) == 0 else np.asarray(bk)[:, None])
        cond.append(np.where(hit.any(1), hit.argmax(1) + 1, INF)); reasons.append("breach")
    if P.vix_trigger:
        hit = vix_path[:, 1 : L + 1] >= vix_path[:, :1] * 1.7
        cond.append(np.where(hit.any(1), hit.argmax(1) + 1, INF)); reasons.append("trigger_exit")
    if P.tp is not None:
        tpref = c if strategy != "put_broken_wing_butterfly" else W
        hit = marks[:, 1:] >= P.tp * tpref[:, None]
        cond.append(np.where(hit.any(1), hit.argmax(1) + 1, INF)); reasons.append("tp")
    if P.time_exit_td is not None:
        cond.append(np.full(m, L - P.time_exit_td)); reasons.append("time")
    cond.append(np.full(m, L)); reasons.append("expiry")
    C = np.stack(cond, 1)
    first = C.min(1)
    which = C.argmin(1)                                  # earliest; ties -> priority order
    reason = np.array(reasons, dtype=object)[which]
    ej = first.astype(int)
    mj = marks[np.arange(m), ej]
    slip = c * rng.exponential(0.12, m) * (vix_path[np.arange(m), np.minimum(ej, vix_path.shape[1] - 1)] / 20.0)
    pnl = mj.copy()
    isstop = np.isin(reason, ["stop", "stop_dollar"])
    if stop_lvl is not None:
        gap = (stop_lvl - mj) > 0.25 * np.abs(stop_lvl)      # the day's mark jumped well past the trigger
        fill = np.where(gap, mj - slip, stop_lvl - 0.02 * c * rng.random(m))
        pnl = np.where(isstop, np.maximum(fill, -maxloss), pnl)
    if P.tp is not None:
        pnl = np.where(reason == "tp", P.tp * (c if strategy != "put_broken_wing_butterfly" else W), pnl)
    pnl = np.where(reason == "expiry", np.maximum(final, -maxloss), pnl)
    return dict(c=c, W=W, width=width, marks=marks, ej=ej, reason=reason, pnl=pnl, final=np.maximum(final, -maxloss),
                maxloss=maxloss, k=k, sig=sig, u=u, stop_lvl=stop_lvl)


def simulate_unit(world: World, ticker: str, strategy: str, P: Params, entry_idx: np.ndarray, rng, run_id: str,
                  profile: str, config_tag: str, level: int, guards: str, id_prefix: str):
    n = len(world.dates)
    L = P.dte_td
    lp = world.logp[ticker]
    spot_all = np.exp(lp)
    ent = entry_idx[entry_idx + 2 * L + 3 < n]
    if len(ent) == 0:
        return None, None
    m = len(ent)
    mult, add = world.ivm[ticker]
    iv = np.clip(mult * world.vix[ent] / 100.0 + add, 0.07, 1.2)
    jj = np.arange(2 * L + 2)
    rel = lp[ent[:, None] + jj[None, :]] - lp[ent][:, None]
    vixp = world.vix[ent[:, None] + jj[None, :]]
    spot = spot_all[ent]
    # profile applies roll only to leg 1
    leg = sim_leg(strategy, rel, iv, spot, vixp, P, rng, allow_roll=P.roll)
    legs = FEES_LEGS[strategy]
    fee_open = legs * 0.70
    marks_full: list[np.ndarray] = []
    exit_off = leg["ej"].copy()
    pnl = leg["pnl"].copy()
    reason = leg["reason"].copy()
    adj_n = np.zeros(m, int)
    adj_cost = np.zeros(m)
    # rolls: leg1 closed on breach (reason 'breach' under allow_roll) and re-opened
    roll_mask = (reason == "breach") & P.roll
    marks_list = []
    leg2 = None
    if P.roll and roll_mask.any():
        ri = np.where(roll_mask)[0]
        r = leg["ej"][ri]
        base = ent[ri] + r
        rel2 = lp[base[:, None] + np.arange(L + 2)[None, :]] - lp[base][:, None]
        vix2 = world.vix[base[:, None] + np.arange(L + 2)[None, :]]
        iv2 = np.clip(mult * world.vix[base] / 100.0 + add, 0.07, 1.2)
        P2 = replace(P, roll=False, breach_exit=False)
        leg2 = sim_leg(strategy, rel2, iv2, spot_all[base], vix2, P2, rng)
        l1 = leg["marks"][ri, r]                             # realised at roll (liquidation value)
        pnl[ri] = l1 + leg2["pnl"] - legs * 0.70             # extra open fees for the new legs
        exit_off[ri] = r + leg2["ej"]
        reason[ri] = leg2["reason"]
        adj_n[ri] = 1
        adj_cost[ri] = np.maximum(-l1, 0) + legs * 0.70
    exit_idx = ent + exit_off
    closed_early = ~np.isin(reason, ["expiry"])
    fees = fee_open + np.where(closed_early, legs * 0.70, 0.0) + np.where(adj_n > 0, legs * 0.70, 0.0)
    pnl_net = pnl - fees
    cf = leg["final"] - fee_open                              # hold leg 1 to expiry, no management
    # entry/exit timestamps
    entry_ts = world.dates[ent] + pd.Timedelta(hours=10, minutes=int(rng.integers(0, 30)))
    exit_ts = world.dates[exit_idx] + pd.Timedelta(hours=15, minutes=30)
    exp_idx = np.minimum(ent + L + np.where(adj_n > 0, leg["ej"], 0), n - 1)
    expiry = world.dates[exp_idx]
    credit = leg["c"]
    coll = np.maximum(leg["W"] - credit, 1.0) if strategy != "put_broken_wing_butterfly" else np.maximum(leg["W"] * 0.5 - credit, 1.0)
    # (BWB: max loss = W2-W1-c, with W1=0.5W, W2=W)
    kd = P.delta
    ids = np.array([f"{id_prefix}{i:06d}" for i in range(m)])
    exdiv = (rng.random(m) < 0.08) & np.isin(ticker, ["SPY", "QQQ", "IWM"])
    tdf = pd.DataFrame({
        "run_id": run_id, "trade_id": ids, "ticker": ticker, "strategy": strategy, "level": level, "profile": profile,
        "config_tag": config_tag, "entry_ts": entry_ts.strftime("%Y-%m-%dT%H:%M:%S-05:00"),
        "exit_ts": exit_ts.strftime("%Y-%m-%dT%H:%M:%S-05:00"), "expiry": expiry.strftime("%Y-%m-%d"),
        "dte_entry": (expiry - world.dates[ent]).days, "exit_reason": reason, "width": leg["width"],
        "collateral_per_contract": np.round(coll, 2), "credit_per_contract": np.round(credit, 2),
        "pnl_per_contract": np.round(pnl_net, 2), "fees_per_contract": np.round(fees, 2), "adj_count": adj_n,
        "adj_cost_per_contract": np.round(adj_cost, 2), "hedge_cost_per_contract": 0.0,
        "cf_pnl_to_expiry_per_contract": np.round(cf, 2), "entry_spot": np.round(spot, 2),
        "exit_spot": np.round(spot_all[exit_idx], 2), "entry_vix": np.round(world.vix[ent], 2),
        "vix_bucket": [vix_bucket(v) for v in world.vix[ent]], "vix_term_ratio": np.round(world.term[ent], 3),
        "short_put_delta": np.round(1 - _nd_cdf(leg["k"]), 3) if np.ndim(leg["k"]) else (kd if strategy != "iron_butterfly" else 0.5),
        "short_call_delta": kd if strategy in ("iron_condor",) else (0.5 if strategy == "iron_butterfly" else ""),
        "exdiv_flag": exdiv.astype(int), "guards_on": guards, "fill_model": "natural", "spot_source": "synthetic",
    })
    # marks: day j in 0..exit_off, date = dates[ent+j]
    rows_t, rows_d, rows_v = [], [], []
    for i in range(m):
        e = exit_off[i]
        js = np.arange(0, e + 1)
        if adj_n[i] > 0:
            ri = np.where(np.where(roll_mask)[0] == i)[0][0]
            r = leg["ej"][i]
            v1 = leg["marks"][i, : r + 1]
            l1 = leg["marks"][i, r]
            j2 = np.arange(1, leg2["ej"][ri] + 1)
            v = np.concatenate([v1, l1 + leg2["marks"][ri, j2] - legs * 0.0])
        else:
            v = leg["marks"][i, : e + 1]
        rows_t.append(np.full(len(js), ids[i], dtype=object))
        rows_d.append(world.dates[ent[i] + js].strftime("%Y-%m-%d").to_numpy())
        rows_v.append(np.round(v, 2))
    mdf = pd.DataFrame({"trade_id": np.concatenate(rows_t), "date": np.concatenate(rows_d),
                        "mark_pnl_per_contract": np.concatenate(rows_v)})
    return tdf, mdf


@dataclass
class Variant:
    profile: str
    config_tag: str
    level: int
    strategies: list
    tickers: list
    P: dict = field(default_factory=dict)            # per-strategy Params overrides
    base: Params = field(default_factory=Params)
    guards: str = "gate;spike;term"
    spike_wait: int | None = 5
    gate_vix: float | None = 35.0
    stress: tuple | None = None


def default_variants() -> list:
    cb = lambda **k: Params(**k)  # noqa: E731
    g = Params(delta=0.16, tp=0.5, stop_mult=2.0)
    V = []
    perstrat = lambda base: {  # noqa: E731
        "iron_condor": base,
        "bull_put_spread": replace(base, delta=base.delta + 0.04),
        "iron_butterfly": replace(base, delta=0.5, tp=0.25 if base.tp else None, stop_mult=1.0 if base.stop_mult else None),
        "put_broken_wing_butterfly": replace(base, delta=base.delta + 0.10, tp=0.5 if base.tp else None),
    }
    V.append(Variant("green", "baseline", 1, STRATEGIES, TICKERS, perstrat(g)))
    y = Params(delta=0.20, tp=0.5, stop_mult=2.5, vix_trigger=True)
    V.append(Variant("yellow_E4", "d20", 2, ["iron_condor", "bull_put_spread"], TICKERS, perstrat(y), guards="gate;spike", spike_wait=3))
    r = Params(delta=0.25, tp=0.7, stop_mult=3.0)
    V.append(Variant("red", "d25_wide", 3, ["iron_condor", "bull_put_spread"], TICKERS, perstrat(r), guards="gate", spike_wait=None))
    ns = replace(g, stop_mult=None)
    V.append(Variant("experiment_no_stop", "nostop", 4, ["iron_condor", "iron_butterfly"], TICKERS, perstrat(ns), guards="gate"))
    ao = Params(delta=0.30, tp=None, stop_mult=None, edge=0.92)   # guards off => sells vol that is NOT rich
    V.append(Variant("experiment_all_off", "alloff", 4, ["iron_condor"], TICKERS, perstrat(ao), guards="", spike_wait=None, gate_vix=None))
    t3 = ["SPY", "QQQ", "IWM"]
    cond = lambda p: {"iron_condor": p}  # noqa: E731
    for d in (10, 25, 30, 35):
        V.append(Variant("green", f"delta{d}", 1, ["iron_condor"], t3, cond(replace(g, delta=d / 100))))
    V.append(Variant("green", "mgmt_tp25", 1, ["iron_condor"], t3, cond(replace(g, tp=0.25))))
    V.append(Variant("green", "mgmt_stop3x", 1, ["iron_condor"], t3, cond(replace(g, stop_mult=3.0))))
    V.append(Variant("green", "mgmt_breach_exit", 1, ["iron_condor"], t3, cond(replace(g, breach_exit=True))))
    V.append(Variant("green", "mgmt_time21", 1, ["iron_condor"], t3, cond(replace(g, time_exit_td=10))))
    V.append(Variant("green", "mgmt_roll_breach", 1, ["iron_condor"], t3, cond(replace(g, roll=True))))
    V.append(Variant("green", "wait1d", 1, ["iron_condor"], t3, cond(g), spike_wait=1))
    V.append(Variant("green", "wait10d", 1, ["iron_condor"], t3, cond(g), spike_wait=10))
    for name, (s, e, step) in STRESS_WINDOWS.items():
        V.append(Variant(name, "stress", 4, STRATEGIES, t3, perstrat(g), guards="", spike_wait=None, gate_vix=None, stress=(s, e, step)))
    return V


def _entry_calendar(world: World, v: Variant, cadence: int):
    n = len(world.dates)
    if v.stress:
        s, e, step = v.stress
        idx = np.where((world.dates >= s) & (world.dates <= e))[0][::step]
        return idx, {}
    idx = np.arange(5, n, cadence)
    vix = world.vix
    blocked = {}
    ok = np.ones(len(idx), bool)
    reasons = np.full(len(idx), "", dtype=object)
    spike_day = np.zeros(n, bool)
    spike_day[5:] = vix[5:] / vix[:-5] > 1.25
    last_spike = np.full(n, -10_000)
    ls = -10_000
    for i in range(n):
        if spike_day[i]:
            ls = i
        last_spike[i] = ls
    for a, i in enumerate(idx):
        if v.gate_vix is not None and vix[i] > v.gate_vix:
            ok[a], reasons[a] = False, "gate"
        elif v.spike_wait is not None and i - last_spike[i] <= v.spike_wait:
            ok[a], reasons[a] = False, "spike"
        elif "term" in v.guards and world.term[i] > 1.12:
            ok[a], reasons[a] = False, "term"
    return idx[ok], {"idx": idx[~ok], "reasons": reasons[~ok]}


def generate(out: str | Path, seed: int = 7, start: str = "2016-10-03", end: str = "2026-09-30",
             cadence: int = 5, variants: list | None = None, verbose: bool = False) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    world = make_world(seed, start, end)
    variants = variants if variants is not None else default_variants()
    for vi, v in enumerate(variants):
        rid = f"{v.profile}__{v.config_tag}"
        rdir = out / rid
        rdir.mkdir(exist_ok=True)
        entries, blk = _entry_calendar(world, v, cadence)
        rng = np.random.default_rng([seed, vi, 99])
        T, M = [], []
        for ti, tk in enumerate(v.tickers):
            for si, st in enumerate(v.strategies):
                P = v.P.get(st, v.base)
                tdf, mdf = simulate_unit(world, tk, st, P, entries, rng, rid, v.profile, v.config_tag, v.level, v.guards,
                                         f"{tk[:2]}{si}{ti}-")
                if tdf is not None:
                    T.append(tdf); M.append(mdf)
        trades = pd.concat(T, ignore_index=True)
        marks = pd.concat(M, ignore_index=True)
        trades.to_csv(rdir / "trades.csv", index=False)
        marks.to_csv(rdir / "marks.csv", index=False)
        rows = []
        for tk in v.tickers:
            for i, rs in zip(blk.get("idx", []), blk.get("reasons", [])):
                rows.append((world.dates[i].strftime("%Y-%m-%d"), tk, rs))
        pd.DataFrame(rows, columns=["date", "ticker", "reason"]).to_csv(rdir / "blocked_entries.csv", index=False)
        P0 = v.P.get(v.strategies[0], v.base)
        cfg = {"synthetic": True, "label": "SYNTHETIC - fixtures only", "run_id": rid, "profile": v.profile,
               "config_tag": v.config_tag, "git_commit": "synthetic", "data_start": start, "data_end": end,
               "seed": seed, "cadence_trading_days": cadence, "stop_multiple": P0.stop_mult,
               "stop_floor_width_frac": P0.stop_floor, "guards": v.guards}
        (rdir / "config.json").write_text(json.dumps(cfg, indent=2))
        (rdir / "meta.json").write_text(json.dumps({
            "synthetic": True, "trades": int(len(trades)), "marks": int(len(marks)), "blocked": len(rows),
            "spot_coverage": "synthetic 100%", "version": "reporting-synth-1"}, indent=2))
        if verbose:
            print(f"  {rid}: {len(trades)} trades")
    return out
