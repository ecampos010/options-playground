"""The "why" layer: one explanation card per calendar month and per drawdown episode of the featured scenarios.

Everything is computed from data already on hand: the replayed account (re-run for the featured scenarios only), the trade
logs (exit reasons, `cf_pnl_to_expiry_per_contract`, `guards_on`), SPY/VIX closes, `events.csv` and `rates.csv`.
Wording is "coincided with": the cards line things up, they do not claim a cause. What-ifs are replays of the SAME logged
trades with one switch changed (no new trades are invented), so they are first-order answers, not new backtests.
"""
from __future__ import annotations

import html as _html
from dataclasses import dataclass, replace as dc_replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .grid import MIX
from .replay import build_variant, replay
from .thermo import CONFIG_DIR

VIX_NAMES = ["VIX < 15", "VIX 15-20", "VIX 20-30", "VIX > 30"]


def load_config(path=None) -> dict:
    return yaml.safe_load(Path(path or CONFIG_DIR / "why.yaml").read_text())


# ------------------------------------------------------------------ small formatters
def _pct(x, nd=1, sign=True):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:{'+' if sign else ''}.{nd}f}%"


def _usd(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"


def vix_regime(v, cuts) -> str | None:
    if v is None:
        return None
    return VIX_NAMES[sum(v >= c for c in cuts)]


def month_label(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m")


# ------------------------------------------------------------------ choosing and rebuilding scenarios
def pick_scenarios(ok: pd.DataFrame, pb: dict | None, n: int) -> list:
    """[(scenario_id, label)] - playbook and award picks first (labelled by title), then the best-ranked survivors."""
    picks: dict = {}
    for it in ((pb or {}).get("playbooks", []) + (pb or {}).get("awards", [])):
        sid = it.get("scenario_id")
        if it.get("status") == "ok" and sid:
            picks.setdefault(sid, []).append(it["title"])
    out = [(sid, " / ".join(t)) for sid, t in picks.items()]
    have = {s for s, _ in out}
    for r in ok[~ok["too_few_trades"]].itertuples():
        if len(out) >= n:
            break
        if r.scenario_id not in have:
            out.append((r.scenario_id, f"Rank {r.rank}"))
            have.add(r.scenario_id)
    return out[:n]


@dataclass
class Rebuilt:
    vd: object
    mask: np.ndarray
    res: object
    sub: pd.DataFrame
    mk: pd.DataFrame
    days: pd.DatetimeIndex
    acc: np.ndarray
    rcfg: object
    alloc: float


def rebuild(row, trades, marks, rcfg) -> Rebuilt:
    t = trades[(trades["profile"] == row["profile"]) & (trades["config_tag"] == row["config_tag"])]
    mk = marks[marks["uid"].isin(set(t["uid"]))] if len(marks) else marks
    parts = [t["entry_ts"].dt.normalize(), t["exit_ts"].dt.normalize()]
    if len(mk):
        parts.append(mk["date"])
    days = pd.DatetimeIndex(sorted(pd.concat(parts).dropna().unique()))
    sub = t if row["strategy"] == MIX else t[t["strategy"] == row["strategy"]]
    vd = build_variant(sub, mk, days)
    mask = np.isin(vd.trades["ticker"].to_numpy(), str(row["tickers"]).split("+"))
    res = replay(vd, mask, float(row["allocation"]) / 100.0, rcfg)
    return Rebuilt(vd, mask, res, sub, mk, days, np.where(res.contracts > 0)[0], rcfg, float(row["allocation"]) / 100.0)


def values_at(rb: Rebuilt, idx: np.ndarray):
    """(accepted trade indices, matrix [trade, day]) of each accepted trade's value (realised or marked) at the given day indices."""
    vd, res = rb.vd, rb.res
    acc = rb.acc
    c = res.contracts[acc]
    ei, xi, pnl = vd.ei[acc], vd.xi[acc], vd.pnl[acc]
    K = vd.Mc.shape[1]
    out = np.zeros((len(acc), len(idx)))
    for j, d in enumerate(idx):
        done = xi <= d
        run = (ei <= d) & ~done
        out[done, j] = c[done] * pnl[done]
        if run.any():
            out[run, j] = c[run] * vd.Mc[acc[run], np.minimum(d - ei[run], K - 1)]
    return acc, out


def event_window_days(ctx, cfg) -> set:
    """Business days within `event_filter_days` before (and on) any event: entries on these days would be skipped by the event filter."""
    win: set = set()
    if ctx.events is None:
        return win
    k = int(cfg["event_filter_days"])
    for d in ctx.events[ctx.events["event"].isin(cfg["event_types"])]["date"]:
        for j in range(k + 1):
            win.add((pd.Timestamp(d) - pd.offsets.BDay(j)).normalize())
    return win


# ------------------------------------------------------------------ trade tags
def tags_for(reason: str, pnl: float, coll: float, credit: float, closed: bool, cfg: dict) -> list:
    if not closed:
        return ["still open (mark-to-market loss)"]
    t = []
    if reason in ("stop", "stop_dollar"):
        t.append("stop hit")
    elif reason == "breach":
        t.append("short-strike breach")
    elif reason == "trigger_exit":
        t.append("management trigger exit")
    else:
        t.append(f"closed by {reason}")
    maxloss = coll - credit if credit > 0 else coll
    if pnl < 0 and maxloss > 0 and -pnl >= cfg["gap_through_wing_loss"] * maxloss:
        t.append("gap through a wing (lost >= %d%% of max loss)" % round(cfg["gap_through_wing_loss"] * 100))
    return t


def _trade_rec(rb, a, usd, closed, cfg):
    """a = position in the accepted-trade list."""
    vd = rb.vd
    i = rb.acc[a]
    r = vd.trades.iloc[i]
    return {"trade_id": str(r["trade_id"]), "ticker": r["ticker"], "strategy": r["strategy"], "usd": float(usd),
            "contracts": int(rb.res.contracts[i]), "closed": bool(closed),
            "tags": tags_for(str(r["exit_reason"]), float(r["pnl_per_contract"]), float(r["collateral_per_contract"]),
                             float(r["credit_per_contract"]), closed, cfg)}


# ------------------------------------------------------------------ cards
def _context_bits(ctx, cfg, d0, d1, prev_end, kind):
    """SPY move, VIX level/change/regime, events and fed funds regime between two dates (d0 exclusive start close, d1 end close)."""
    out = {}
    s0, s1 = ctx.last_close(ctx.spy, prev_end), ctx.last_close(ctx.spy, d1)
    out["spy"] = {"ret": s1 / s0 - 1.0, "src": ctx.spy_src} if s0 and s1 else None
    cuts = cfg["vix_cutoffs"]
    v0, v1 = ctx.last_close(ctx.vix, prev_end), ctx.last_close(ctx.vix, d1)
    if v1 is not None:
        win = ctx.vix[(ctx.vix.index >= pd.Timestamp(d0)) & (ctx.vix.index <= pd.Timestamp(d1))] if ctx.vix is not None else None
        out["vix"] = {"start": v0, "end": v1, "peak": float(win.max()) if win is not None and len(win) else None,
                      "regime_end": vix_regime(v1, cuts), "regime_start": vix_regime(v0, cuts), "src": ctx.vix_src}
    else:
        out["vix"] = None
    ev = ctx.events_between(d0, d1, cfg["event_types"])
    out["events"] = [{"date": str(pd.Timestamp(d).date()), "event": e} for d, e in ev]
    out["events_known"] = ctx.events is not None
    fr = cfg["fed_regime"]
    out["rate"] = ctx.rates.regime(prev_end, d1, fr["move_bp"], fr["low_rate_pct"], fr["high_rate_pct"]) if ctx.rates is not None else None
    return out


def _sentence(card) -> str:
    bits = []
    if card.get("spy"):
        bits.append(f"SPY {_pct(card['spy']['ret'])}")
    v = card.get("vix")
    if v and v.get("end") is not None:
        s = f"VIX {v['start']:.1f} to {v['end']:.1f}" if v.get("start") is not None else f"VIX {v['end']:.1f}"
        bits.append(s + f" ({v['regime_end']})")
    if card.get("events"):
        names = sorted({x["event"] for x in card["events"]})
        bits.append(" and ".join([", ".join(names[:-1]), names[-1]] if len(names) > 1 else names) + " inside the window")
    r = card.get("rate")
    if r:
        bits.append(f"fed funds {r['trend']} ({r['level']}, {r['start']:.2f}% to {r['end']:.2f}%)")
    word = "lost" if card["pnl"] < 0 else "made"
    what = f"The account {word} {_usd(abs(card['pnl']))} ({_pct(card['ret'])})"
    return what + (", which coincided with " + "; ".join(bits) + "." if bits else ". No market context data was available for this window.")


def _loss_summary(lost) -> str:
    if not lost:
        return "No trade had a negative contribution."
    cnt: dict = {}
    for t in lost:
        for g in t["tags"]:
            g = g.split(" (")[0] if g.startswith("gap") else g
            cnt[g] = cnt.get(g, 0) + 1
    return f"{len(lost)} trade(s) contributed negatively: " + ", ".join(f"{k} x{v}" for k, v in sorted(cnt.items(), key=lambda kv: -kv[1])) + "."


def build_scenario(row, label, trades, marks, rcfg, ctx, cfg, sidx: int) -> dict:
    rb = rebuild(row, trades, marks, rcfg)
    vd, res = rb.vd, rb.res
    n = vd.n
    eq, days = res.equity, vd.days
    s0 = res.start_equity
    # months
    mid = vd.month_id
    first = np.r_[True, mid[1:] != mid[:-1]]
    starts = np.where(first)[0]
    ends = np.r_[starts[1:], len(days)] - 1
    prev_eq = np.r_[s0, eq[ends[:-1]]]
    # episodes first, so one value matrix covers months and episode edges
    eps = _episodes(eq, s0, cfg["dd_episode_min"])
    idx = sorted({*ends.tolist(), *[e["peak_i"] for e in eps], *[e["trough_i"] for e in eps]})
    pos = {d: j for j, d in enumerate(idx)}
    acc, V = values_at(rb, np.array(idx))
    Vm = V[:, [pos[d] for d in ends]]
    C = np.diff(np.c_[np.zeros(len(acc)), Vm], axis=1)
    win = event_window_days(ctx, cfg)
    entry_day = vd.trades["entry_ts"].dt.normalize().to_numpy()[acc]
    in_win = np.array([pd.Timestamp(x) in win for x in entry_day]) if len(win) else np.zeros(len(acc), bool)
    guards = vd.trades["guards_on"].astype(str).to_numpy()[acc] if "guards_on" in vd.trades else np.array([], str)
    ev_guard = bool(len(guards)) and all("event" in g for g in guards)
    exit_reason = vd.trades["exit_reason"].astype(str).to_numpy()[acc]
    cf = vd.trades["cf_pnl_to_expiry_per_contract"].to_numpy(float)[acc] if "cf_pnl_to_expiry_per_contract" in vd.trades else np.full(len(acc), np.nan)
    pnl_u = vd.pnl[acc]
    contracts = res.contracts[acc]
    xmid = mid[vd.xi[acc]]
    early = np.isin(exit_reason, cfg["early_exit_reasons"])
    cards = []
    sid = row["scenario_id"]
    for m, (a, b) in enumerate(zip(starts, ends)):
        ms = pd.Timestamp(days[a].year, days[a].month, 1)
        me = ms + pd.offsets.MonthEnd(0)
        prev_end = ms - pd.Timedelta(days=1)
        ret = eq[b] / prev_eq[m] - 1.0 if prev_eq[m] > 0 else 0.0
        pnl = float(eq[b] - prev_eq[m])
        c = C[:, m]
        tr_pnl = float(c.sum())
        card = {"id": f"why-{sidx}-{month_label(ms)}", "kind": "month", "scenario_id": sid, "month": month_label(ms),
                "start": str(ms.date()), "end": str(me.date()), "ret": float(ret), "pnl": pnl, "equity_start": float(prev_eq[m]),
                "equity_end": float(eq[b]), "trade_pnl": tr_pnl, "interest_usd": pnl - tr_pnl,
                "red": bool(ret <= cfg["red_month_return"])}
        card.update(_context_bits(ctx, cfg, ms, me, prev_end, "month"))
        closed_here = xmid == mid[a]
        neg = np.where(c < -0.5)[0]
        neg = neg[np.argsort(c[neg])]
        lost = [_trade_rec(rb, j, c[j], closed_here[j], cfg) for j in neg]
        card["lost"] = lost[:3]
        card["lost_summary"] = _loss_summary(lost)
        big = None
        if len(c) and np.abs(c).max() > 0.5:
            j = int(np.argmin(c)) if pnl < 0 else int(np.argmax(c))
            big = _trade_rec(rb, j, c[j], closed_here[j], cfg)
        card["biggest"] = big
        # month-level what-ifs (static: the month's own trades, sizing held fixed)
        w = []
        if abs(card["interest_usd"]) >= 0.5:
            w.append(f"Interest on idle cash added about {_usd(card['interest_usd'])} this month; without it the month would have been "
                     f"{_pct((pnl - card['interest_usd']) / prev_eq[m]) if prev_eq[m] > 0 else 'n/a'} instead of {_pct(ret)} (sizing held fixed).")
        e_idx = np.where(closed_here & early & ~np.isnan(cf))[0]
        if len(e_idx):
            act = float((contracts[e_idx] * pnl_u[e_idx]).sum())
            cfv = float((contracts[e_idx] * cf[e_idx]).sum())
            w.append(f"{len(e_idx)} early exit(s) closed this month realised {_usd(act)}; holding the same trades to expiry would have realised "
                     f"{_usd(cfv)} ({_usd(cfv - act)} difference, settled in a later month).")
        if ctx.events is None:
            w.append("Event filter what-if: n/a (no events.csv).")
        elif ev_guard:
            w.append("Event filter: already ON in these logs (guards_on includes 'event').")
        else:
            wi = np.where(in_win & (np.abs(c) > 0.5))[0]
            if len(wi):
                w.append(f"{len(wi)} trade(s) active this month were entered within {cfg['event_filter_days']} business days before an event; they contributed "
                         f"{_usd(float(c[wi].sum()))} to the month, which an event filter would have skipped (other trades and sizing not re-run).")
            else:
                w.append("Event filter: none of this month's active trades were entered inside an event window, so it would not have changed the month.")
        card["whatif"] = w
        if res.broke_idx is not None and b >= res.broke_idx:
            card["broke"] = True
        card["summary"] = _sentence(card)
        cards.append(card)
    # drawdown episodes
    episodes = []
    for k, ep in enumerate(eps, 1):
        p, t, r = ep["peak_i"], ep["trough_i"], ep.get("rec_i")
        d0, d1 = days[p], days[t]
        dcol = V[:, pos[t]] - V[:, pos[p]]
        closed_in = (vd.xi[acc] > p) & (vd.xi[acc] <= t)
        neg = np.where(dcol < -0.5)[0]
        neg = neg[np.argsort(dcol[neg])]
        lost = [_trade_rec(rb, j, dcol[j], closed_in[j], cfg) for j in neg]
        card = {"id": f"why-{sidx}-dd{k}", "kind": "episode", "scenario_id": sid, "n": k, "peak_date": str(d0.date()),
                "trough_date": str(d1.date()), "recovery_date": str(days[r].date()) if r is not None else None,
                "depth": float(ep["depth"]), "pnl": float(ep["trough"] - ep["peak"]), "ret": float(-ep["depth"]),
                "days_to_trough": int((d1 - d0).days), "days_to_recover": int((days[r] - d1).days) if r is not None else None,
                "lost": lost[:3], "lost_summary": _loss_summary(lost)}
        card.update(_context_bits(ctx, cfg, d0, d1, d0, "episode"))
        card["events_total"] = len(card["events"])
        card["events"] = card["events"][:cfg["max_episode_events"]]
        card["biggest"] = _trade_rec(rb, int(np.argmin(dcol)), dcol.min(), closed_in[int(np.argmin(dcol))], cfg) if len(dcol) and dcol.min() < -0.5 else None
        interest = float(ep["trough"] - ep["peak"]) - float(dcol.sum())
        card["interest_usd"] = interest
        card["whatif"] = []
        card["summary"] = _sentence(card).replace("The account lost", "The account fell").replace("The account made", "The account moved")
        episodes.append(card)
    return {"scenario_id": sid, "label": label, "sidx": sidx, "row": _row_facts(row), "months": cards, "episodes": episodes,
            "whatifs": whatifs(row, rb, ctx, cfg, ev_guard, win), "broke": res.broke_idx is not None}


def _row_facts(row) -> dict:
    keys = ("profile", "config_tag", "strategy", "tickers", "allocation", "level_tag", "zone", "final_equity", "cagr", "max_dd")
    return {k: (row[k].item() if hasattr(row[k], "item") else row[k]) for k in keys if k in row}


def _episodes(eq, start_eq, min_depth):
    peak, peak_i, ep, out = start_eq, 0, None, []
    for d, e in enumerate(eq):
        if e >= peak:
            if ep is not None:
                ep["rec_i"] = d
                out.append(ep)
                ep = None
            peak, peak_i = e, d
        elif ep is None:
            ep = {"peak_i": peak_i, "peak": peak, "trough_i": d, "trough": e}
        elif e < ep["trough"]:
            ep["trough_i"], ep["trough"] = d, e
    if ep is not None:
        out.append(ep)
    for ep in out:
        ep["depth"] = 1.0 - ep["trough"] / ep["peak"] if ep["peak"] > 0 else 0.0
    return [ep for ep in out if ep["depth"] >= min_depth]


# ------------------------------------------------------------------ scenario-level what-ifs
def _m(res) -> dict:
    m = res.metrics
    return {"final": m["final_equity"], "cagr": m["cagr"], "max_dd": m["max_dd"], "trades": m["n_trades"]}


def _cmp(on, off, a, b):
    return (f"{a}: final {_usd(on['final'])}, CAGR {_pct(on['cagr'])}, max drawdown {_pct(on['max_dd'], sign=False)}, {on['trades']} trades; "
            f"{b}: final {_usd(off['final'])}, CAGR {_pct(off['cagr'])}, max drawdown {_pct(off['max_dd'], sign=False)}, {off['trades']} trades.")


def whatifs(row, rb: Rebuilt, ctx, cfg, ev_guard, win) -> list:
    """Three switches, each a replay of the same logged trades: interest on cash, early exits, event filter."""
    vd, rcfg, s0 = rb.vd, rb.rcfg, rb.res.start_equity
    base = _m(rb.res)
    out = []
    # 1. interest on idle cash
    off = _m(replay(vd, rb.mask, rb.alloc, dc_replace(rcfg, apy=0.0, rates=None)))
    rt = rcfg.rates
    cash_final = float((rt.cash_curve(vd.days, s0) if rt is not None else s0 * np.cumprod(np.r_[1.0, (1 + rcfg.apy) ** (np.diff(vd.days.values).astype("timedelta64[D]").astype(int) / 365.0)]))[-1])
    yrs = max((vd.days[-1] - vd.days[0]).days / 365.25, 1e-9)
    cash_cagr = (cash_final / s0) ** (1 / yrs) - 1.0
    gap = base["cagr"] - cash_cagr
    out.append({"key": "interest", "title": "Interest on idle cash", "status": "ok", "on": base, "off": off,
                "cash_final": cash_final, "cash_cagr": cash_cagr,
                "text": _cmp(base, off, "Interest ON (as replayed)", "Interest OFF") + f" Interest added {_usd(base['final'] - off['final'])}. A cash-only account would have ended at "
                        f"{_usd(cash_final)} (CAGR {_pct(cash_cagr)}), so this scenario " + (f"beat cash by {gap * 100:.1f} CAGR points." if gap > 0 else f"did NOT beat cash (short by {-gap * 100:.1f} CAGR points).")})
    # 2. early exits vs hold to expiry
    alt = rb.sub.copy()
    sel = alt["exit_reason"].isin(cfg["early_exit_reasons"]) & alt["cf_pnl_to_expiry_per_contract"].notna()
    if not sel.any():
        out.append({"key": "early_exit", "title": "Early exits", "status": "n/a", "text": "No trade in this scenario exited early with a hold-to-expiry counterfactual in the logs."})
    else:
        alt.loc[sel, "pnl_per_contract"] = alt.loc[sel, "cf_pnl_to_expiry_per_contract"]
        newx = pd.to_datetime(alt.loc[sel, "expiry"]) + pd.Timedelta(hours=16)
        alt.loc[sel, "exit_ts"] = np.maximum(newx.to_numpy(), alt.loc[sel, "exit_ts"].to_numpy())
        alt.loc[sel, "exit_reason"] = "expiry"
        days2 = rb.days.union(pd.DatetimeIndex(alt["exit_ts"].dt.normalize().unique()))
        vd2 = build_variant(alt, rb.mk, days2)
        mask2 = np.isin(vd2.trades["ticker"].to_numpy(), str(row["tickers"]).split("+"))
        hold = _m(replay(vd2, mask2, rb.alloc, rcfg))
        out.append({"key": "early_exit", "title": "Early exits", "status": "ok", "on": base, "off": hold, "n_trades": int(sel.sum()),
                    "text": f"{int(sel.sum())} of the {len(alt)} trades in this configuration's log exited early (the replay only takes those it can afford). " + _cmp(base, hold, "Early exits ON (as logged)", "Early exits OFF (those trades held to expiry, from cf_pnl_to_expiry)")
                            + " The in-between marks of held trades carry the last logged mark, so only the end result is exact."})
    # 3. event filter
    if ctx.events is None:
        out.append({"key": "event_filter", "title": "Event filter", "status": "n/a", "text": "n/a: no events.csv (run scripts/fetch_context_data.py)."})
    elif ev_guard:
        out.append({"key": "event_filter", "title": "Event filter", "status": "n/a", "on": base,
                    "text": f"Event filter is already ON in these logs (guards_on includes 'event'): final {_usd(base['final'])}, max drawdown {_pct(base['max_dd'], sign=False)}. "
                            "The OFF case needs a sibling run with --event-buffer off; it cannot be computed from this log."})
    else:
        skip = np.array([pd.Timestamp(x) in win for x in vd.trades["entry_ts"].dt.normalize()])
        mask2 = rb.mask & ~skip
        n_skip = int((rb.mask & skip).sum())
        on = _m(replay(vd, mask2, rb.alloc, rcfg))
        out.append({"key": "event_filter", "title": "Event filter", "status": "ok", "on": on, "off": base, "n_skipped": n_skip,
                    "text": _cmp(on, base, f"Event filter ON (skip entries within {cfg['event_filter_days']} business days before {'/'.join(cfg['event_types'])}; {n_skip} logged trades removed)",
                                 "Event filter OFF (as logged)") + " Skipped entries are not replaced by later ones."})
    return out


# ------------------------------------------------------------------ rendering
def _e(x) -> str:
    return _html.escape("" if x is None else str(x), quote=True)


def _events_txt(c) -> str:
    if not c.get("events_known", True):
        return "n/a (no events.csv)"
    if not c["events"]:
        return "none"
    s = ", ".join(f"{x['event']} {x['date'][5:]}" for x in c["events"])
    extra = c.get("events_total", len(c["events"])) - len(c["events"])
    return s + (f" (+{extra} more)" if extra > 0 else "")


def _trade_txt(t) -> str:
    return f"{t['trade_id']} ({t['ticker']} {t['strategy']}, {t['contracts']} ct): {_usd(t['usd'])} - {'; '.join(t['tags'])}"


def _rate_txt(r) -> str:
    return "n/a (no rates.csv)" if not r else f"{r['trend']}, {r['level']} ({r['start']:.2f}% to {r['end']:.2f}%, {r['change_bp']:+.0f} bp)"


def _vix_txt(v) -> str:
    if not v or v.get("end") is None:
        return "n/a"
    a = f"{v['start']:.1f} to {v['end']:.1f}" if v.get("start") is not None else f"{v['end']:.1f}"
    pk = f", peak {v['peak']:.1f}" if v.get("peak") is not None else ""
    rg = f"; regime {v['regime_start']} to {v['regime_end']}" if v.get("regime_start") and v["regime_start"] != v["regime_end"] else f"; regime {v['regime_end']}"
    return a + pk + rg


def card_fields(c) -> list:
    f = []
    if c["kind"] == "month":
        f.append(("P&L", f"{_usd(c['pnl'])} ({_pct(c['ret'])}): trades {_usd(c['trade_pnl'])}, interest on cash {_usd(c['interest_usd'])}"))
    else:
        f.append(("Drawdown", f"{_pct(-c['depth'], sign=False)} ({_usd(c['pnl'])}) from {c['peak_date']} to the {c['trough_date']} low; "
                              + (f"recovered {c['recovery_date']} ({c['days_to_recover']} days after the low)" if c["recovery_date"] else "NOT recovered by the end of the data")))
    f.append(("SPY move", _pct(c["spy"]["ret"]) if c.get("spy") else "n/a (no SPY data)"))
    f.append(("VIX", _vix_txt(c.get("vix"))))
    f.append(("Events inside", _events_txt(c)))
    f.append(("Fed funds regime", _rate_txt(c.get("rate"))))
    f.append(("Losing trades", c["lost_summary"] + ("" if not c["lost"] else " Worst: " + " | ".join(_trade_txt(t) for t in c["lost"]))))
    f.append(("Biggest single contributor", _trade_txt(c["biggest"]) if c.get("biggest") else "none (no trade moved the account)"))
    return f


def card_html(c) -> str:
    cls = "why-card" + (" red" if c.get("red") else "")
    if c["kind"] == "month":
        head = f"<b>{_e(c['month'])}</b> &middot; {_pct(c['ret'])} &middot; {_usd(c['pnl'])}"
        if c.get("spy"):
            head += f" &middot; SPY {_pct(c['spy']['ret'])}"
    else:
        head = (f"<b>Drawdown {c['n']}</b>: {_e(c['peak_date'])} to {_e(c['trough_date'])} &middot; {_pct(-c['depth'], sign=False)} &middot; "
                + ("recovered " + _e(c["recovery_date"]) if c["recovery_date"] else "not recovered"))
    body = "".join(f"<dt>{_e(k)}</dt><dd>{_e(v)}</dd>" for k, v in card_fields(c))
    wi = "".join(f"<li>{_e(w)}</li>" for w in c.get("whatif", []))
    return (f'<details class="{cls}" id="{_e(c["id"])}"><summary>{head}</summary><p>{_e(c["summary"])}</p><dl>{body}</dl>'
            + (f'<p class="small muted">What-if lines (first-order, from existing data):</p><ul class="small">{wi}</ul>' if wi else "") + "</details>")


def scenario_html(sc) -> str:
    r = sc["row"]
    head = (f"<b>{_e(sc['label'])}</b> &middot; {_e(r.get('strategy'))} {_e(r.get('tickers'))} @ {_e(r.get('allocation'))}% "
            f"({_e(r.get('profile'))} / {_e(r.get('config_tag'))})")
    wi = "".join(f"<li><b>{_e(w['title'])}:</b> {_e(w['text'])}</li>" for w in sc["whatifs"])
    eps = "".join(card_html(c) for c in sc["episodes"]) or "<p class='small'>No drawdown episode at or above the minimum depth.</p>"
    months = "".join(card_html(c) for c in sc["months"])
    return (f'<details class="why-scn" id="why-s{sc["sidx"]}"><summary>{head}</summary>'
            f"<h4>What-ifs for the whole scenario</h4><ul class='small'>{wi}</ul>"
            f"<h4>Drawdown episodes ({len(sc['episodes'])})</h4>{eps}<h4>Months ({len(sc['months'])})</h4>{months}</details>")


def red_rows(scenarios) -> list:
    rows = []
    for sc in scenarios:
        r = sc["row"]
        for c in sc["months"]:
            if not c["red"]:
                continue
            rows.append({"scenario": f"{sc['label']} | {r.get('strategy')} {r.get('tickers')} @ {r.get('allocation')}% ({r.get('profile')}/{r.get('config_tag')})",
                         "scenario_id": sc["scenario_id"], "month": c["month"], "ret": c["ret"], "pnl": c["pnl"],
                         "spy": c["spy"]["ret"] if c.get("spy") else None, "vix_end": c["vix"]["end"] if c.get("vix") else None,
                         "events": _events_txt(c), "fed": _rate_txt(c.get("rate")),
                         "biggest": _trade_txt(c["biggest"]) if c.get("biggest") else "", "card": c["id"]})
    rows.sort(key=lambda x: (x["ret"], x["month"]))
    return rows


def render_html(scenarios, omitted, ctx_desc, cfg, synthetic=False) -> tuple:
    """-> (html, red_rows_rendered). Cards past the size budget are left out of the page (but stay in why.json / why.md)."""
    from .htmlreport import e, money, pct as hpct, raw, table
    budget = int(cfg["html_budget_bytes"])
    shown, parts, used = [], [], 0
    for sc in scenarios:
        h = scenario_html(sc)
        if shown and used + len(h) > budget:
            omitted.append(sc["label"])
            continue
        shown.append(sc)
        parts.append(h)
        used += len(h)
    rr = red_rows(shown)
    trs = []
    for r in rr:
        trs.append([r["scenario"], r["month"], hpct(r["ret"]), money(r["pnl"]),
                    (_pct(r["spy"]), r["spy"] if r["spy"] is not None else float("nan")) if r["spy"] is not None else "n/a",
                    (f"{r['vix_end']:.1f}", r["vix_end"]) if r["vix_end"] is not None else "n/a", r["events"], r["biggest"],
                    raw(f'<a href="#{e(r["card"])}" class="why-link">card</a>')])
    H = ['<h2 id="why">Why: explanation cards</h2>',
         '<p class="small">Every calendar month and every drawdown episode of the featured scenarios (the playbook and award picks, then the best-ranked survivors) '
         'gets a card: P&amp;L, the SPY move, VIX level and regime, FOMC/CPI/PPI/NFP/election dates inside the window, the fed funds regime, which trades lost '
         '(stop hit, short-strike breach, gap through a wing) and the biggest single contributor. Cards say what <b>coincided with</b> a result; they never claim a cause. '
         'What-if lines replay the same logged trades with one switch changed (event filter, early exits, interest on cash). '
         f'Data used: {e(ctx_desc)}. A "red" month is one with a return at or below {_pct(cfg["red_month_return"], 0, sign=False)}.</p>']
    if synthetic:
        H.append('<div class="synth">SYNTHETIC: the cards describe generated fixture logs.</div>')
    H.append('<h3 id="red-months">Red months</h3><div class="controls"><label>Search<input id="red-q" type="search" placeholder="month, scenario, event, trade"></label></div>'
             '<p class="small muted" id="red-count"></p>')
    H.append(table(["Scenario", "Month", "Return", "P&L", "SPY", "VIX (end)", "Events inside", "Biggest contributor", "Card"], trs, num=(2, 3, 4, 5), tid="red-table")
             if trs else "<p>No red months in the featured scenarios.</p>")
    if omitted:
        H.append(f'<p class="small flag">Left out of this page for size: {e(", ".join(omitted))} (all cards are in why.json / why.md).</p>')
    H.append("".join(parts))
    return "".join(H), rr


def render_md(scenarios, ctx_desc) -> str:
    L = ["# Why: explanation cards", "", "Cards list what *coincided with* each result; they do not claim a cause.", f"Data used: {ctx_desc}", ""]
    for sc in scenarios:
        r = sc["row"]
        L += [f"## {sc['label']} - {r.get('strategy')} {r.get('tickers')} @ {r.get('allocation')}% ({r.get('profile')}/{r.get('config_tag')})", "", "### What-ifs"]
        L += [f"- **{w['title']}**: {w['text']}" for w in sc["whatifs"]] + ["", "### Drawdown episodes"]
        for c in sc["episodes"] + [None] + sc["months"]:
            if c is None:
                L += ["", "### Months"]
                continue
            L += ["", f"#### {c['month'] if c['kind'] == 'month' else 'Drawdown ' + str(c['n'])}{' (RED)' if c.get('red') else ''}", c["summary"]]
            L += [f"- {k}: {v}" for k, v in card_fields(c)] + [f"- What-if: {w}" for w in c.get("whatif", [])]
        L.append("")
    return "\n".join(L) + "\n"


def build(df, ok, pb, trades, marks, rcfg, ctx, cfg=None, synthetic=False, scenarios_override=None) -> dict:
    cfg = cfg or load_config()
    picks = scenarios_override or pick_scenarios(ok, pb, int(cfg["max_scenarios"]))
    by = df.set_index("scenario_id", drop=False)
    scs = []
    for k, (sid, label) in enumerate(picks):
        if sid in by.index:
            scs.append(build_scenario(by.loc[sid], label, trades, marks, rcfg, ctx, cfg, k))
    omitted: list = []
    desc = ctx.describe()
    html, rr = render_html(scs, omitted, desc, cfg, synthetic)
    return {"scenarios": scs, "html": html, "md": render_md(scs, desc), "red_all": red_rows(scs), "red_html": rr, "omitted": omitted, "desc": desc}
