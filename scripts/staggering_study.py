#!/usr/bin/env python3
"""Staggering study (MODELED, relative only): does splitting one condor risk budget R into staggered tranches
reduce risk vs entering all at once?  Output is comparison data, not a recommendation.

Pricing and trade management are copied from claude/guard-calibration-study.py.txt (Black-Scholes, Cboe VIX as vol, skew
+10% puts / -5% calls, ~16-delta shorts, $5-wide-equivalent condor, TP 50% of credit, stop 2x credit or short-strike touch,
21-DTE time exit).  The only deliberate differences: entry DTE is a parameter (spec: 45; the guard study used 35 -- run with
--dte 35 to reproduce it exactly, which the regression test does) and the simulation also returns the exit day.

Schedules (same R, same product, same exit rules per tranche; offsets are trading days after cycle day 0):
  A_ALL_AT_ONCE 100%@0 | B_TWO_TRANCHES 50%@0,50%@3 | C_FOUR_TRANCHES 25%@0,3,7,10 | D_TWO_TRANCHES_WIDE 50%@0,50%@7
Each tranche is sized by its own credit/width so its max loss is fraction*R; a tranche's return as a fraction of R is
fraction * ror (ror = P&L / (width - credit), as in the guard study).  Green hard gate (VIX>=25 or day<=-1.5% or 5d<=-4%)
is applied to every tranche on its own entry day; a blocked tranche is skipped (not delayed) and counted.

Usage:  python scripts/staggering_study.py [--px px.csv] [--vix vix.csv] [--out results/staggering_study]
        [--boot 500] [--synthetic]   (no data args = download ^GSPC via yfinance and Cboe VIX; --synthetic = smoke test)
"""
import argparse, io, math, sys
from pathlib import Path
from statistics import NormalDist
import numpy as np, pandas as pd

try:
    from scipy.special import erf as _erf
except Exception:
    _erf = np.vectorize(math.erf)

CBOE = "https://cdn-api.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="
R_, SKP, SKC, WIDTH = 0.02, 0.10, 0.05, 0.0064
ENTRY_SLIP, STOP_SLIP, TP_SLIP = 0.05, 0.05, 0.02
START = "1990-01-09"
BUDGET = 15000.0                 # R in dollars
SCALE_K = 21                     # daily-start variant: one monthly budget R spread over 21 daily cycle starts
EQUITY_RISK_FRAC = 0.10          # compounded equity curve: R = 10% of current equity (assumption)
GATE_VIX, GATE_D1, GATE_D5 = 25.0, -0.015, -0.04
SCHEDULES = {
    "A_ALL_AT_ONCE": [(0, 1.0)],
    "B_TWO_TRANCHES": [(0, .5), (3, .5)],
    "C_FOUR_TRANCHES": [(0, .25), (3, .25), (7, .25), (10, .25)],
    "D_TWO_TRANCHES_WIDE": [(0, .5), (7, .5)],
}
MAX_OFFSET = max(o for s in SCHEDULES.values() for o, _ in s)
CRASHES = {"1998": ("1998-07-15", "1998-10-31"), "2000-02": ("2000-03-01", "2002-10-31"), "2008": ("2008-01-01", "2008-12-31"),
           "2011": ("2011-07-15", "2011-10-31"), "2015": ("2015-08-10", "2015-10-15"), "2018Q4": ("2018-10-01", "2018-12-31"),
           "2020": ("2020-02-15", "2020-04-30"), "2022": ("2022-01-01", "2022-10-31")}

# ------------------------------------------------------------------ data (same loaders as the guard study)
def get(url):
    import requests
    r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"}); r.raise_for_status(); return r.text

def norm(df):
    df.columns = [str(c).strip().lower() for c in df.columns]
    dc = [c for c in df.columns if c in ("date", "observation_date")][0]
    df[dc] = pd.to_datetime(df[dc]); return df.set_index(dc).sort_index()

def load_vix(path):
    df = norm(pd.read_csv(path if path else io.StringIO(get(CBOE))))
    return pd.to_numeric(df["close"] if "close" in df else df.iloc[:, 0], errors="coerce").rename("vix")

def load_px(path):
    if path: df = norm(pd.read_csv(path))
    else:
        try:
            import yfinance as yf
            df = yf.download("^GSPC", start="1990-01-01", progress=False, auto_adjust=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df = norm(df.reset_index()); assert len(df) > 5000
        except Exception as e:
            print("yfinance failed (%s); falling back to FRED SP500 (closes only, ~10y)" % e, file=sys.stderr)
            df = norm(pd.read_csv(io.StringIO(get(FRED + "SP500"))))
            df["close"] = pd.to_numeric(df["sp500"], errors="coerce"); df["high"] = df["low"] = df["close"]
    return df[["high", "low", "close"]].apply(pd.to_numeric, errors="coerce").dropna()

def synthetic(n=3000, seed=7, start="2000-01-03"):
    """Seeded GARCH-ish daily path with a VIX-like series; for smoke tests only."""
    rng = np.random.default_rng(seed); vol = np.empty(n); r = np.empty(n); v = 0.15
    for i in range(n):
        v = float(np.clip(0.04 + 0.7 * v + 0.2 * abs(r[i - 1]) * math.sqrt(252) * 0.5 + 0.02 * rng.standard_normal(), 0.09, 0.80)) if i else v
        vol[i] = v; r[i] = rng.standard_normal() * v / math.sqrt(252) + 0.0003
    c = 1000 * np.exp(np.cumsum(r)); rg = np.abs(rng.standard_normal(n)) * vol / math.sqrt(252) * 0.6
    idx = pd.bdate_range(start, periods=n)
    px = pd.DataFrame({"high": c * (1 + rg), "low": c * (1 - rg), "close": c}, index=idx)
    return px, pd.Series(vol * 100 * 1.1, index=idx, name="vix")

def prepare(px, vix, start=START):
    d = px.join(vix, how="inner").dropna()
    d["d1"] = d.close.pct_change(); d["d5"] = d.close.pct_change(5); d = d.dropna()
    return d.loc[start:]

# ------------------------------------------------------------------ pricing / trade management (from the guard study)
def cdf(x): return 0.5 * (1 + _erf(x / math.sqrt(2)))
def opt(S, K, T, sig, call):
    if T <= 1e-9: return np.maximum(S - K, 0) if call else np.maximum(K - S, 0)
    d1 = (np.log(S / K) + (R_ + sig ** 2 / 2) * T) / (sig * np.sqrt(T)); d2 = d1 - sig * np.sqrt(T)
    return S * cdf(d1) - K * math.exp(-R_ * T) * cdf(d2) if call else K * math.exp(-R_ * T) * cdf(-d2) - S * cdf(-d1)
def condor(S, Ksp, Klp, Ksc, Klc, T, v):
    sp, sc = v * (1 + SKP), v * (1 - SKC)
    return opt(S, Ksp, T, sp, 0) - opt(S, Klp, T, sp, 0) + opt(S, Ksc, T, sc, 1) - opt(S, Klc, T, sc, 1)

def horizon(dte, exit_dte=21):
    """Trading-day hold horizon and time-exit day for an entry at `dte` (35 -> 24 / 10, as in the guard study; 45 -> 31 / 17)."""
    return round(dte * 252 / 365), round((dte - exit_dte) * 252 / 365)

def simulate(d, dte=45, call_delta=.16, put_delta=.16, tx=None, hold=None):
    """Guard-study simulate(), identical numerics, plus `k` = trading days held at exit.  ror = P&L / (width - credit)."""
    ht, tx0 = horizon(dte); hold = ht if hold is None else hold; tx = tx0 if tx is None else tx
    n = len(d); S0 = d.close.values; v0 = d.vix.values / 100; T0 = dte / 365
    sp, sc = v0 * (1 + SKP), v0 * (1 - SKC); nd = NormalDist()
    Ksp = S0 * np.exp((R_ + sp ** 2 / 2) * T0 - nd.inv_cdf(1 - put_delta) * sp * math.sqrt(T0))
    Ksc = S0 * np.exp((R_ + sc ** 2 / 2) * T0 + nd.inv_cdf(1 - call_delta) * sc * math.sqrt(T0))
    w = WIDTH * S0; Klp, Klc = Ksp - w, Ksc + w
    cr = condor(S0, Ksp, Klp, Ksc, Klc, T0, v0) * (1 - ENTRY_SLIP)
    alive = np.arange(n) + hold < n; pnl = np.full(n, np.nan); why = np.array([""] * n, dtype=object); kx = np.full(n, -1)
    for k in range(1, hold + 1):
        sh = lambda a: pd.Series(a).shift(-k).values
        S, hi, lo, v = sh(d.close.values), sh(d.high.values), sh(d.low.values), sh(d.vix.values) / 100
        T = max((dte - k * dte / ht) / 365, 0)
        with np.errstate(all="ignore"):
            Sb = np.where(S >= Ksc, S, np.where(S <= Ksp, S, np.where(hi >= Ksc, Ksc, Ksp)))
            br = alive & ((hi >= Ksc) | (lo <= Ksp)); cb = np.minimum(condor(Sb, Ksp, Klp, Ksc, Klc, T, v) * (1 + STOP_SLIP), w)
            mid = condor(S, Ksp, Klp, Ksc, Klc, T, v); nat = np.minimum(mid * (1 + STOP_SLIP), w)
            a2 = alive & ~br; st = a2 & (nat >= 2 * cr); tp = a2 & ~st & (mid <= 0.5 * cr)
            tm = a2 & ~st & ~tp & (k >= tx)
        for m, cost, lab in ((br, cb, "breach"), (st, nat, "stop"), (tp, mid * (1 + TP_SLIP), "tp"), (tm, mid if T == 0 else nat, "time")):
            pnl[m] = (cr - cost)[m]; why[m] = lab; kx[m] = k; alive &= ~m
    return pd.DataFrame({"ror": pnl / (w - cr), "why": why, "cr": cr / w, "k": kx}, index=d.index)

def gate_blocked(d):
    """Green hard gate on each entry day: True = tranche blocked."""
    return ((d.vix >= GATE_VIX) | (d.d1 <= GATE_D1) | (d.d5 <= GATE_D5)).values

# ------------------------------------------------------------------ schedule sizing and cycle construction
def tranche_sizing(budget, schedule, credits, width, mult=100):
    """Contracts and max loss per tranche.  `credits` = per-share credit at each tranche's own entry.  Sum of max loss == budget."""
    out = []
    for (off, frac), cr in zip(schedule, credits):
        per_contract = (width - cr) * mult
        contracts = frac * budget / per_contract
        out.append({"offset": off, "fraction": frac, "contracts": contracts, "max_loss": contracts * per_contract})
    return out

def cycle_results(sim, blocked, starts, schedule, gates_on):
    """Per cycle: return as fraction of R, tranche/blocked counts, stop-out days, and last exit index."""
    ror, kx = sim.ror.values, sim.k.values; stopped_all = sim.why.isin(["breach", "stop"]).values
    nt = len(schedule); ret = np.zeros(len(starts)); blk = np.zeros((len(starts), nt), bool)
    stop_day = np.full((len(starts), nt), np.nan); frac = np.array([f for _, f in schedule]); exit_idx = starts + schedule[-1][0]
    for j, (off, f) in enumerate(schedule):
        t = starts + off; b = blocked[t] if gates_on else np.zeros(len(t), bool)
        blk[:, j] = b; ret += np.where(b, 0.0, f * ror[t])
        st = stopped_all[t] & ~b; stop_day[:, j] = np.where(st, t + kx[t], np.nan)
        exit_idx = np.where(b, exit_idx, np.maximum(exit_idx, t + kx[t]))
    # more than half of the budget stopping out inside one 5-trading-day window
    mx = np.zeros(len(starts))
    for a in range(nt):
        sa = stop_day[:, a]
        w = sum(frac[j] * ((stop_day[:, j] >= sa) & (stop_day[:, j] <= sa + 4)) for j in range(nt))
        mx = np.where(np.isnan(sa), mx, np.maximum(mx, w))
    return {"ret": ret, "blocked": blk, "cluster": mx > 0.5 + 1e-12, "exit_idx": exit_idx}

def cycle_starts(d, sim, monthly):
    """Valid cycle-start indices: every tranche of every schedule has a completed trade (same set for all schedules/gates)."""
    n = len(d); ok = np.flatnonzero(~np.isnan(sim.ror.values)); last = ok.max() if len(ok) else -1
    idx = np.arange(n)
    valid = idx + MAX_OFFSET <= last
    if monthly:
        mo = d.index.to_period("M"); first = np.r_[True, mo[1:] != mo[:-1]]; valid &= first
    return idx[valid]

# ------------------------------------------------------------------ metrics
def tail(x, p):
    q = np.quantile(x, p); return q, x[x <= q].mean()

def equity_stats(ret, exit_idx, dates, scale):
    o = np.argsort(exit_idx, kind="stable"); inc = ret[o] * scale
    eq = np.cumsum(inc); dd = float(np.max(np.maximum.accumulate(np.r_[0, eq])[1:] - eq)) if len(eq) else 0.0
    ceq = np.cumprod(1 + EQUITY_RISK_FRAC * inc); pk = np.maximum.accumulate(np.r_[1.0, ceq])[1:]
    cdd = float(np.max(1 - ceq / pk)) if len(ceq) else 0.0
    yrs = pd.Series(inc, index=dates[np.minimum(exit_idx[o], len(dates) - 1)].year).groupby(level=0).sum()
    return dd, cdd, float(yrs.min()), int(yrs.idxmin())

def metrics(res, dates, scale):
    r = res["ret"]; q5, c5 = tail(r, .05); q1, c1 = tail(r, .01)
    dd, cdd, wy, wyy = equity_stats(r, res["exit_idx"], dates, scale)
    return {"n_cycles": len(r), "tranche_blocked_pct": res["blocked"].mean(), "cycles_flat_pct": res["blocked"].all(axis=1).mean(),
            "mean": r.mean(), "median": np.median(r), "std": r.std(ddof=1) if len(r) > 1 else np.nan, "worst": r.min(),
            "q05": q5, "cvar5": c5, "q01": q1, "cvar1": c1, "frac_loss_ge50": (r <= -0.5).mean(),
            "frac_gt_half_stop_5d": res["cluster"].mean(), "maxdd": dd, "maxdd_compounded": cdd, "worst_year": wy, "worst_year_label": wyy}

USD = ("mean", "median", "std", "worst", "q05", "cvar5", "q01", "cvar1", "maxdd", "worst_year")

def metrics_rows(allres, dates, variant, gates):
    scale = 1.0 / SCALE_K if variant == "daily" else 1.0; rows = []
    for name, res in allres.items():
        m = metrics(res, dates, scale); row = {"variant": variant, "gates": gates, "schedule": name, **m}
        for k in USD: row[k + "_usd"] = m[k] * BUDGET
        rows.append(row)
    return rows

def block_boot(allres, block, reps, seed=11):
    """Paired moving-block bootstrap over cycle starts: CI for mean / CVaR5 / CVaR1 / frac loss>=50% per schedule and, for the
    first three, the CI of the difference vs ALL_AT_ONCE (same resampled cycles for every schedule)."""
    names = list(allres); n = len(allres[names[0]]["ret"]); rng = np.random.default_rng(seed)
    nb = int(math.ceil(n / block)); k5, k1 = max(int(0.05 * n), 1), max(int(0.01 * n), 1)
    draws = {nm: {s: [] for s in ("mean", "cvar5", "cvar1", "frac_loss_ge50")} for nm in names}
    for _ in range(reps):
        st = rng.integers(0, max(n - block + 1, 1), nb); idx = (st[:, None] + np.arange(block)).ravel()[:n]
        for nm in names:
            x = np.sort(allres[nm]["ret"][idx]); dr = draws[nm]
            dr["mean"].append(x.mean()); dr["cvar5"].append(x[:k5].mean()); dr["cvar1"].append(x[:k1].mean()); dr["frac_loss_ge50"].append((x <= -0.5).mean())
    rows = []
    for nm in names:
        for s, v in draws[nm].items():
            lo, hi = np.percentile(v, [2.5, 97.5]); row = {"schedule": nm, "stat": s, "ci_lo": lo, "ci_hi": hi}
            dv = np.array(v) - np.array(draws[names[0]][s]); row["delta_ci_lo"], row["delta_ci_hi"] = np.percentile(dv, [2.5, 97.5])
            rows.append(row)
    return pd.DataFrame(rows)

def crash_table(allres, dates, starts, variant, gates):
    rows = []
    for ep, (a, b) in CRASHES.items():
        m = (dates[starts] >= a) & (dates[starts] <= b)
        for nm, res in allres.items():
            r = res["ret"][m]
            if len(r) == 0: continue
            sc = 1.0 / SCALE_K if variant == "daily" else 1.0
            rows.append({"variant": variant, "gates": gates, "episode": ep, "schedule": nm, "n_cycles": len(r), "mean": r.mean(),
                         "worst": r.min(), "total_budget_months": r.sum() * sc, "total_usd": r.sum() * sc * BUDGET,
                         "tranche_blocked_pct": res["blocked"][m].mean()})
    return rows

def tranche_corr(sim, blocked, starts, gates_on, variant):
    ror = sim.ror.values; st = sim.why.isin(["breach", "stop"]).values.astype(float); rows = []
    for off in sorted({o for s in SCHEDULES.values() for o, _ in s if o}):
        a, b = starts, starts + off; ok = np.ones(len(a), bool)
        if gates_on: ok = ~blocked[a] & ~blocked[b]
        if ok.sum() < 3: continue
        rows.append({"variant": variant, "gates": "on" if gates_on else "off", "delayed_offset": off, "n_pairs": int(ok.sum()),
                     "corr_return": np.corrcoef(ror[a][ok], ror[b][ok])[0, 1], "corr_stopped": np.corrcoef(st[a][ok], st[b][ok])[0, 1]})
    return rows

def split_sample(dfm):
    """Do schedule rankings (by CVaR5, higher = better; and by mean) hold in both halves?"""
    rows = []
    for (v, g), x in dfm.groupby(["variant", "gates"]):
        for h, y in x.groupby("half"):
            rk = y.sort_values("cvar5", ascending=False).schedule.tolist(); rm = y.sort_values("mean", ascending=False).schedule.tolist()
            rows.append({"variant": v, "gates": g, "half": h, "rank_by_cvar5": ">".join(s[0] for s in rk), "rank_by_mean": ">".join(s[0] for s in rm),
                         "dd_rank": ">".join(s[0] for s in y.sort_values("maxdd").schedule)})
    return pd.DataFrame(rows)

# ------------------------------------------------------------------ driver
def run(d, dte, boot, out):
    sim = simulate(d, dte=dte); blocked = gate_blocked(d); dates = d.index
    mrows, crows, trows, bdfs = [], [], [], []; cyc = {}
    for variant in ("daily", "monthly"):
        starts = cycle_starts(d, sim, variant == "monthly"); half = dates[starts[len(starts) // 2]] if len(starts) else None
        for gates_on in (True, False):
            g = "on" if gates_on else "off"
            allres = {nm: cycle_results(sim, blocked, starts, s, gates_on) for nm, s in SCHEDULES.items()}
            rows = metrics_rows(allres, dates, variant, g)
            for hn, m in (("first", dates[starts] < half), ("second", dates[starts] >= half)):
                sub = {nm: {k: v[m] for k, v in r.items()} for nm, r in allres.items()}
                for rr in metrics_rows(sub, dates, variant, g): rr["half"] = hn; mrows.append(rr)
            for rr in rows: rr["half"] = "full"
            mrows += rows
            crows += crash_table(allres, dates, starts, variant, g)
            trows += tranche_corr(sim, blocked, starts, gates_on, variant)
            b = block_boot(allres, 21 if variant == "daily" else 1, boot); b.insert(0, "gates", g); b.insert(0, "variant", variant); bdfs.append(b)
            cyc[(variant, g)] = pd.DataFrame({nm: r["ret"] for nm, r in allres.items()}, index=dates[starts])
    return sim, pd.DataFrame(mrows), pd.DataFrame(crows), pd.DataFrame(trows), pd.concat(bdfs), cyc

def pct(x): return "%.1f%%" % (100 * x)
def usd(x): return "${:,.0f}".format(x)

def md_table(df):
    cols = list(df.columns); s = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows(): s.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(s)

def main_table(m, variant, gates):
    x = m[(m.variant == variant) & (m.gates == gates) & (m.half == "full")]; rows = []
    for _, r in x.iterrows():
        rows.append({"schedule": r.schedule, "cycles": int(r.n_cycles), "blocked tranches": pct(r.tranche_blocked_pct),
                     "mean": "%s (%s)" % (pct(r["mean"]), usd(r.mean_usd)), "median": pct(r["median"]), "std": pct(r["std"]),
                     "worst": "%s (%s)" % (pct(r.worst), usd(r.worst_usd)), "CVaR5": "%s (%s)" % (pct(r.cvar5), usd(r.cvar5_usd)),
                     "CVaR1": pct(r.cvar1), "loss>=50%R": pct(r.frac_loss_ge50), ">half stopped in 5d": pct(r.frac_gt_half_stop_5d),
                     "maxDD %R": pct(r.maxdd), "maxDD compounded": pct(r.maxdd_compounded),
                     "worst year": "%s (%d)" % (pct(r.worst_year), r.worst_year_label)})
    return md_table(pd.DataFrame(rows))

def verdict(m, b, sp, d, dte):
    def g(v, gt, sch, col, half="full"):
        return float(m[(m.variant == v) & (m.gates == gt) & (m.schedule == sch) & (m.half == half)][col].iloc[0])
    def ci(v, gt, sch, stat):
        r = b[(b.variant == v) & (b.gates == gt) & (b.schedule == sch) & (b.stat == stat)].iloc[0]; return r.delta_ci_lo, r.delta_ci_hi
    A, B, C = "A_ALL_AT_ONCE", "B_TWO_TRANCHES", "C_FOUR_TRANCHES"
    a5 = g("daily", "on", A, "cvar5")
    out = []
    out.append("Modeled, relative only (%s to %s, %d trading days, %d-DTE condor, gates ON, daily-start cycles; R = %s)." % (
        d.index[0].date(), d.index[-1].date(), len(d), dte, usd(BUDGET)))
    parts = []
    for nm, lab in ((B, "2 tranches"), (C, "4 tranches")):
        dl = g("daily", "on", nm, "cvar5") - a5; lo, hi = ci("daily", "on", nm, "cvar5")
        rel = dl / abs(a5) if a5 else float("nan"); sig = (lo > 0 or hi < 0)
        tag = "meaningful" if (sig and abs(rel) >= 0.10 and dl > 0) else ("small" if dl > 0 else "none/worse")
        parts.append("%s: worst-5%% mean %s vs %s all-at-once (%+.1f pts of R, %s, 95%% block-bootstrap CI %+.1f..%+.1f pts; %s)" % (
            lab, pct(g("daily", "on", nm, "cvar5")), pct(a5), 100 * dl, usd(dl * BUDGET), 100 * lo, 100 * hi, tag))
    out.append("Tail risk (CVaR5): " + "; ".join(parts) + ". Monthly-start CVaR5: A %s, B %s, C %s." % (
        pct(g("monthly", "on", A, "cvar5")), pct(g("monthly", "on", B, "cvar5")), pct(g("monthly", "on", C, "cvar5"))))
    out.append("Worst cycle A/B/C: %s / %s / %s; loss>=50%% of R: %s / %s / %s; >half of budget stopped inside 5 days: %s / %s / %s (daily starts)." % (
        *[pct(g("daily", "on", s, "worst")) for s in (A, B, C)], *[pct(g("daily", "on", s, "frac_loss_ge50")) for s in (A, B, C)],
        *[pct(g("daily", "on", s, "frac_gt_half_stop_5d")) for s in (A, B, C)]))
    out.append("Equity max drawdown (%%R, daily-start, per-month-budget scale) A/B/C: %s / %s / %s; compounded monthly-start: %s / %s / %s; worst calendar year (monthly) A/B/C: %s / %s / %s." % (
        *[pct(g("daily", "on", s, "maxdd")) for s in (A, B, C)], *[pct(g("monthly", "on", s, "maxdd_compounded")) for s in (A, B, C)],
        *[pct(g("monthly", "on", s, "worst_year")) for s in (A, B, C)]))
    dm = g("daily", "on", C, "mean") - g("daily", "on", A, "mean"); lo, hi = ci("daily", "on", C, "mean")
    out.append("Return cost (mean cycle, C minus A): %+.2f pts of R (%s per cycle, CI %+.2f..%+.2f pts); B minus A: %+.2f pts. Expected ~0 by construction; any gap is gate/timing noise." % (
        100 * dm, usd(dm * BUDGET), 100 * lo, 100 * hi, 100 * (g("daily", "on", B, "mean") - g("daily", "on", A, "mean"))))
    holds = []
    for v in ("daily", "monthly"):
        for gt in ("on", "off"):
            y = sp[(sp.variant == v) & (sp.gates == gt)]
            holds.append(y.rank_by_cvar5.nunique() == 1)
    out.append("Split-sample: the CVaR5 ranking of schedules is the same in both halves for %d of %d variant/gate combinations (see split_sample.csv). Not a recommendation; real quotes, spread widening and early assignment are not modeled." % (
        sum(holds), len(holds)))
    return out

def write_summary(out, m, c, tc, b, sp, d, dte):
    v = verdict(m, b, sp, d, dte); L = ["# Staggering study (modeled, relative only)", "",
        "Data %s to %s (%d days). ^GSPC with Cboe VIX as vol, Black-Scholes, skew +10%%/-5%%, ~16-delta $5-wide-equivalent condor, %d DTE, TP 50%%, stop 2x credit or short-strike touch, 21-DTE time exit. R = %s per cycle." % (
            d.index[0].date(), d.index[-1].date(), len(d), dte, usd(BUDGET)), "", "## Verdict (6 lines)", ""]
    L += ["%d. %s" % (i + 1, s) for i, s in enumerate(v)]
    L += ["", "Cycle return is a fraction of R; a blocked tranche earns 0 and leaves its slice of R idle. Daily-start drawdown/yearly figures scale each cycle by 1/%d (one monthly budget across daily starts). Worst year = sum of cycle returns by exit year (monthly: 12 cycles of R, so it can exceed 100%%); maxDD %%R = non-compounded peak-to-trough of that cumulative sum; maxDD compounded = same curve with R = %d%% of equity." % (SCALE_K, round(100 * EQUITY_RISK_FRAC))]
    for variant, title in (("daily", "Daily-start cycles (overlapping; CIs by block bootstrap)"), ("monthly", "Monthly-start cycles (first trading day of month; non-overlapping starts)")):
        for gt in ("on", "off"):
            L += ["", "## %s, gates %s" % (title, gt.upper()), "", main_table(m, variant, gt)]
    L += ["", "## Block-bootstrap CIs (daily starts, gates ON; block = 21 days; delta = schedule minus ALL_AT_ONCE)", "",
          md_table(b[(b.variant == "daily") & (b.gates == "on")].round(4).drop(columns=["variant", "gates"])),
          "", "## Crash episodes (gates ON; mean/worst cycle as fraction of R; total_budget_months = sum of cycle returns, x 1/%d for daily starts; total_usd at R = %s)" % (SCALE_K, usd(BUDGET)), ""]
    cc = c[c.gates == "on"].copy()
    for col in ("mean", "worst", "tranche_blocked_pct"): cc[col] = cc[col].map(pct)
    cc["total_usd"] = cc.total_usd.map(usd); cc["total_budget_months"] = cc.total_budget_months.round(3)
    L += [md_table(cc.drop(columns=["gates"])), "", "## Tranche correlation (A-tranche vs delayed tranche, same cycle)", "", md_table(tc.round(3)),
          "", "## Split-sample (first vs second half of cycle starts); rankings best to worst, letters = schedule A/B/C/D", "", md_table(sp)]
    L += ["", "## Files", "", "metrics.csv (all stats, %R and $), bootstrap_ci.csv, crash_episodes.csv, tranche_correlation.csv, split_sample.csv, cycles_*.csv."]
    (out / "summary.md").write_text("\n".join(L) + "\n")

def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--px"); ap.add_argument("--vix"); ap.add_argument("--out", default="results/staggering_study")
    ap.add_argument("--dte", type=int, default=45); ap.add_argument("--boot", type=int, default=500); ap.add_argument("--start", default=START)
    ap.add_argument("--synthetic", action="store_true", help="smoke test on seeded synthetic data (no network)"); a = ap.parse_args(argv)
    px, vix = synthetic() if a.synthetic else (load_px(a.px), load_vix(a.vix))
    d = prepare(px, vix, "1900-01-01" if a.synthetic else a.start); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sim, m, c, tc, b, cyc = run(d, a.dte, a.boot, out)
    sp = split_sample(m[m.half != "full"])
    m.to_csv(out / "metrics.csv", index=False); c.to_csv(out / "crash_episodes.csv", index=False); tc.to_csv(out / "tranche_correlation.csv", index=False)
    b.to_csv(out / "bootstrap_ci.csv", index=False); sp.to_csv(out / "split_sample.csv", index=False)
    for (v, g), df in cyc.items(): df.to_csv(out / ("cycles_%s_gates_%s.csv" % (v, g)))
    write_summary(out, m, c, tc, b, sp, d, a.dte); print((out / "summary.md").read_text())

if __name__ == "__main__":
    main()
