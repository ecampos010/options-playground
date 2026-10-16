"""Playbooks and Awards: picks named scenarios from the existing replay grid. All thresholds come from
config/playbooks.yaml. Nothing here re-runs a replay; it only selects, compares and explains."""
from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import START_EQUITY
from .thermo import CONFIG_DIR

FINAL_EQUITY_ONLY = {"lottery"}      # the only selectors allowed to rank on final equity


def load_config(path=None) -> dict:
    cfg = yaml.safe_load(Path(path or CONFIG_DIR / "playbooks.yaml").read_text())
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    for key, spec in {**cfg.get("playbooks", {}), **cfg.get("awards", {})}.items():
        if spec.get("rank", {}).get("by") == "final_equity" and key not in FINAL_EQUITY_ONLY:
            raise ValueError(f"playbooks.yaml: {key} ranks on final equity; only {sorted(FINAL_EQUITY_ONLY)} may")


# ------------------------------------------------------------------ helpers
def _num(x):
    try:
        v = float(x)
        return None if np.isnan(v) else v
    except (TypeError, ValueError):
        return None


def pct(x, nd=1):
    return "n/a" if x is None else f"{x * 100:.{nd}f}%"


def halves(series: dict, sid: str, start_eq: float = START_EQUITY) -> dict:
    """Return of the first and second half of the equity curve (split at the middle observation)."""
    s = series.get(sid)
    if not s:
        return {"first_half": None, "second_half": None}
    eq = np.asarray(s["eq"], dtype=float)
    if len(eq) < 4:
        return {"first_half": None, "second_half": None}
    mid = len(eq) // 2
    base = float(start_eq)
    return {"first_half": float(eq[mid] / base - 1.0), "second_half": float(eq[-1] / eq[mid] - 1.0) if eq[mid] > 0 else None}


def pool_of(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Eligible rows: >= min_trades, not broke, not a long-premium strategy. Counts what was excluded and why."""
    mt = int(cfg["min_trades"])
    lp = set(cfg.get("long_premium", []))
    few = df["n_trades"] < mt
    info = {"scenarios_total": int(len(df)), "configs_total": int(df[["profile", "config_tag", "strategy"]].drop_duplicates().shape[0]),
            "excluded_too_few_trades": int(few.sum()), "excluded_broke": int((df["broke"] & ~few).sum()),
            "min_trades": mt, "ruin_available": bool(df["ruin_odds"].notna().any()) if "ruin_odds" in df else False}
    keep = ~few & ~df["broke"] & ~df["strategy"].isin(lp)
    info["require_beat_cash"] = bool(cfg.get("require_beat_cash", True) and "beats_cash" in df)
    info["excluded_below_cash"] = 0
    if info["require_beat_cash"]:
        below = keep & ~df["beats_cash"].astype(bool)         # must beat a cash-only account earning the risk-free rate
        info["excluded_below_cash"] = int(below.sum())
        keep &= ~below
    p = df[keep].copy()
    info["eligible"] = int(len(p))
    return p, info


def select(pool: pd.DataFrame, spec: dict, start_eq: float = START_EQUITY) -> tuple[pd.Series | None, int, str]:
    """-> (best row or None, candidates passing the filters, reason when None)."""
    f = spec.get("filters", {})
    m = pd.Series(True, index=pool.index)
    if f.get("zones"):
        m &= pool["zone"].isin(f["zones"])
    if f.get("positive_return"):
        m &= pool["final_equity"] > start_eq
    if "max_dd" in f:
        m &= pool["max_dd"] < f["max_dd"]
    if "max_ruin" in f:
        if "ruin_odds" not in pool or pool["ruin_odds"].notna().sum() == 0:
            return None, 0, "Monte Carlo was not run, so ruin odds are unavailable"
        m &= pool["ruin_odds"] < f["max_ruin"]          # NaN compares False
    c = pool[m]
    if c.empty:
        return None, 0, "no scenario meets the limits"
    by, asc = spec["rank"]["by"], spec["rank"].get("order", "desc") == "asc"
    keys, order = [by], [asc]
    for k, a in (("calmar", False), ("max_dd", True), ("scenario_id", True)):
        if k != by:
            keys.append(k)
            order.append(a)
    c = c.sort_values(keys, ascending=order, kind="stable")
    return c.iloc[0], int(len(c)), ""


def spy_compare(spy: dict | None, d0, d1) -> dict | None:
    """Buy-and-hold SPY over [d0, d1] from the monthly closes supplied via --spy-csv."""
    if not spy or not spy.get("t"):
        return None
    s = pd.Series(spy["v"], index=pd.to_datetime(spy["t"], unit="ms")).sort_index()
    d0, d1 = pd.Timestamp(d0), pd.Timestamp(d1)
    a = s[s.index <= d0]
    a_val = float(a.iloc[-1]) if len(a) else float(s.iloc[0])
    b = s[s.index <= d1]
    if b.empty:
        return None
    b_val = float(b.iloc[-1])
    yrs = max((d1 - d0).days / 365.25, 1e-9)
    sub = s[(s.index >= (a.index[-1] if len(a) else s.index[0])) & (s.index <= b.index[-1])]
    dd = float((1 - sub / sub.cummax()).max()) if len(sub) else None
    return {"total_return": b_val / a_val - 1.0, "cagr": (b_val / a_val) ** (1 / yrs) - 1.0, "max_dd_monthly": dd,
            "covers_whole_window": bool(len(a) > 0 and s.index[-1] >= d1 - pd.Timedelta(days=45))}


def record(row: pd.Series, series: dict, start_eq: float) -> dict:
    r = {k: (None if k not in row or (isinstance(row[k], float) and np.isnan(row[k])) else row[k])
         for k in ("scenario_id", "profile", "config_tag", "strategy", "tickers", "level_tag", "zone", "allocation")}
    r.update(config=f"{row['profile']} / {row['config_tag']}", allocation=int(row["allocation"]),
             total_return=float(row["final_equity"] / start_eq - 1.0), final_equity=float(row["final_equity"]),
             cagr=_num(row["cagr"]), max_dd=_num(row["max_dd"]), win_rate=_num(row["win_rate"]),
             ruin_odds=_num(row.get("ruin_odds")), n_trades=int(row["n_trades"]), calmar=_num(row["calmar"]),
             broke=bool(row["broke"]))
    if "final_equity_noint" in row:
        r.update(final_equity_noint=_num(row["final_equity_noint"]), cagr_noint=_num(row["cagr_noint"]),
                 total_return_noint=float(row["final_equity_noint"] / start_eq - 1.0), cash_final=_num(row["cash_final"]),
                 cash_cagr=_num(row["cash_cagr"]), cash_total_return=float(row["cash_final"] / start_eq - 1.0),
                 excess_cagr=_num(row["excess_cagr"]), beats_cash=bool(row["beats_cash"]), interest_usd=_num(row["interest_usd"]))
    r.update(halves(series, row["scenario_id"], start_eq))
    return r


def _cash_txt(r: dict) -> str:
    if r.get("cash_cagr") is None:
        return ""
    gap = r["excess_cagr"]
    return (f"Interest on idle cash is included: without it the return would have been {pct(r['total_return_noint'], 0)} ({pct(r['cagr_noint'])} a year) "
            f"instead of {pct(r['total_return'], 0)}. A cash-only account earning the fed funds rate would have returned {pct(r['cash_total_return'], 0)} "
            f"({pct(r['cash_cagr'])} a year), so it " + (f"beat cash by {gap * 100:.1f} CAGR points. " if gap > 0 else f"did NOT beat cash (short by {-gap * 100:.1f} points). "))


def _halves_txt(r: dict) -> str:
    a, b = r.get("first_half"), r.get("second_half")
    if a is None or b is None:
        return "first-half vs second-half result unavailable"
    both = "positive in both halves" if a > 0 and b > 0 else "NOT positive in both halves, so treat it as period-dependent"
    return f"first half {pct(a)}, second half {pct(b)} ({both})"


def _ruin_txt(r: dict) -> str:
    return "ruin odds not computed (Monte Carlo off)" if r["ruin_odds"] is None else f"ruin odds {pct(r['ruin_odds'], 2)}"


def paragraph(key: str, r: dict, ctx: dict) -> str:
    base = (f"{r['strategy']} on {r['tickers']} at a {r['allocation']}% allocation ({r['config']}, {r['level_tag']}) returned "
            f"{pct(r['total_return'], 0)} in total ({pct(r['cagr'])} a year), with a worst drawdown of {pct(r['max_dd'])}, "
            f"a {pct(r['win_rate'], 0)} win rate over {r['n_trades']} trades and {_ruin_txt(r)}. ")
    tried = f"It is the best of {ctx['n_candidates']} scenarios that met the limits, out of {ctx['pool']['scenarios_total']} tried. "
    tail = _cash_txt(r) + _halves_txt(r) + ". "
    extra = {
        "capital_preservation": "This is the setup that hurt least while still making money; the price of that is usually a modest return. ",
        "conservative": "The boring way: stays in the Safe zone and aims for the best return per unit of drawdown. ",
        "moderate": "A balanced choice: accepts Mild-zone drawdowns in exchange for more return. ",
        "max_safe_growth": "The highest annual growth that still stays inside the Safe limits. ",
        "growth": "Accepts large drawdowns (but under the Growth limits) for the highest annual growth. ",
        "lottery": ("LUCK-DEPENDENT: this has the highest final equity of everything tried, so it is also the most likely to be the lucky "
                    "outlier of a large search. Look at the drawdown and ruin odds next to it before believing it. "),
    }.get(key, "")
    return base + tried + extra + tail


# ------------------------------------------------------------------ Big Short Style (long premium)
def big_short(df: pd.DataFrame, stress_df: pd.DataFrame, spec: dict, cfg: dict, series: dict, start_eq: float) -> dict:
    strats = set(spec["strategies"])
    pats = [re.compile(p, re.I) for p in spec["stress_windows"]]
    out = {"key": "big_short", "title": spec["title"], "nickname": spec.get("nickname", ""), "status": "none", "reason": "", "windows": {}}
    st = stress_df[stress_df["strategy"].isin(strats)] if len(stress_df) else stress_df
    if st is None or st.empty:
        out["reason"] = ("not run: no long_put / put_debit_spread / long_strangle stress-window logs found "
                         "(run `python -m playground stress` with sweeps/big_short.yaml)")
        return out
    st = st.assign(win=st["window"].str.replace(r"^stress_", "", regex=True))
    st = st[[any(p.search(w) for p in pats) for w in st["win"]]]
    if st.empty:
        out["reason"] = "not run: stress logs exist for these strategies but none in the Feb 2018 / Mar 2020 / 2022 windows"
        return out
    g = st.groupby(["strategy", "config_tag"]).agg(stress_total=("pnl_total", "sum"), windows=("win", "nunique")).reset_index()
    g = g.sort_values(["stress_total", "windows"], ascending=False, kind="stable")
    best = g.iloc[0]
    per = st[(st["strategy"] == best["strategy"]) & (st["config_tag"] == best["config_tag"])].groupby("win")["pnl_total"].sum()
    out.update(status="ok", strategy=best["strategy"], config_tag=best["config_tag"], stress_total_per_contract=float(best["stress_total"]),
               windows={k: float(v) for k, v in per.items()}, n_configs_tried=int(len(g)))
    calm = df[(df["strategy"] == best["strategy"]) & (df["config_tag"] == best["config_tag"]) & (df["n_trades"] >= cfg["min_trades"])] if len(df) else df
    if len(calm):
        row = calm.sort_values("final_equity", ascending=False, kind="stable").iloc[0]
        out["record"] = record(row, series, start_eq)
    wtxt = "; ".join(f"{k}: {v:+,.0f}$/contract" for k, v in out["windows"].items())
    if "record" in out:
        r = out["record"]
        calm_txt = (f"Over the whole 10-year log it returned {pct(r['total_return'], 0)} in total (max drawdown {pct(r['max_dd'])}, "
                    f"win rate {pct(r['win_rate'], 0)}, {r['n_trades']} trades): the bleed between crashes is visible there. ")
    else:
        calm_txt = "No calm-period (full-history) log exists for it, so the bleed between crashes was NOT measured here. "
    out["paragraph"] = (f"{best['strategy']} ({best['config_tag']}) had the best crash payoff of the long-premium structures tried "
                        f"({int(len(g))} configurations): {wtxt}; {best['stress_total']:+,.0f}$ per contract in total across those windows. "
                        + calm_txt +
                        "These are defined-risk structures (max loss is the debit paid, never naked), and they bleed premium in calm periods and "
                        "pay in crashes; one good crash window does not make them a strategy. ")
    out["n_configs_tried"] = int(len(g))
    return out


# ------------------------------------------------------------------ worst performance: why it lost
def why_lost(row: pd.Series, trades: pd.DataFrame) -> str:
    bits = []
    if row["broke"]:
        bits.append(f"the account went broke on {row['broke_date']} at the {int(row['allocation'])}% allocation")
    else:
        bits.append(f"it ended at ${row['final_equity']:,.0f} with a {pct(row['max_dd'], 0)} drawdown")
    t = trades[(trades["profile"] == row["profile"]) & (trades["config_tag"] == row["config_tag"])]
    if row["strategy"] != "all_strategies_equal":
        t = t[t["strategy"] == row["strategy"]]
    t = t[t["ticker"].isin(str(row["tickers"]).split("+"))]
    if len(t):
        w = t.sort_values("pnl_per_contract").iloc[0]
        losers = int((t["pnl_per_contract"] < 0).sum())
        reasons = t.loc[t["pnl_per_contract"] < 0, "exit_reason"].value_counts()
        top = ", ".join(f"{k} x{v}" for k, v in reasons.head(2).items())
        bits.append(f"worst single trade lost ${-w['pnl_per_contract']:,.0f} per contract ({w['ticker']}, exited {str(w['exit_ts'])[:10]} by {w['exit_reason']}) "
                    f"against ${w['collateral_per_contract']:,.0f} of collateral")
        bits.append(f"{losers} of {len(t)} trades lost money" + (f", mostly exits by {top}" if top else ""))
    bits.append(f"zone {row['zone']}")
    return "; ".join(bits) + "."


# ------------------------------------------------------------------ main
def build(df: pd.DataFrame, series: dict, trades: pd.DataFrame, stress_df: pd.DataFrame, spy: dict | None, cfg: dict,
          synthetic: bool = False, start_eq: float = START_EQUITY) -> dict:
    pool, info = pool_of(df, cfg)
    res = {"generated": dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "synthetic": synthetic, "disclaimer": cfg["disclaimer"],
           "pool": info, "playbooks": [], "awards": [], "ways_to_lose_link": "#ways"}
    for key, spec in cfg["playbooks"].items():
        if spec.get("kind") == "stress":
            b = big_short(df, stress_df, spec, cfg, series, start_eq)
            b["n_configs_tried"] = b.get("n_configs_tried", 0)
            res["playbooks"].append(b)
            continue
        row, n, why = select(pool, spec, start_eq)
        item = {"key": key, "title": spec["title"], "nickname": spec.get("nickname", ""), "n_configs_tried": info["scenarios_total"],
                "n_configs_distinct": info["configs_total"], "n_candidates": n, "luck_dependent": bool(spec.get("luck_dependent"))}
        if row is None:
            item.update(status="none", reason=why)
        else:
            r = record(row, series, start_eq)
            item.update(status="ok", **r)
            item["paragraph"] = paragraph(key, r, {"n_candidates": n, "pool": info})
            if spec.get("compare_spy"):
                days = series[row["scenario_id"]]["days"]
                sc = spy_compare(spy, days[0], days[-1])
                item["spy"] = sc
                if sc is None:
                    item["spy_text"] = "SPY comparison not available: no --spy-csv was supplied (or it does not cover this window)."
                else:
                    beat = r["total_return"] > sc["total_return"]
                    item["spy_text"] = (f"Holding SPY over the same window returned {pct(sc['total_return'], 0)} ({pct(sc['cagr'])} a year, "
                                        f"monthly-close max drawdown {pct(sc['max_dd_monthly'])}) versus {pct(r['total_return'], 0)} for this playbook "
                                        f"({pct(r['cagr'])} a year, max drawdown {pct(r['max_dd'])}). "
                                        + ("It beat buy-and-hold on return" if beat else "It did NOT beat buy-and-hold on return")
                                        + (" and with a smaller drawdown." if r["max_dd"] < (sc["max_dd_monthly"] or 1) else
                                           " and did not have a smaller drawdown."
                                           if not beat else ", but with a drawdown no smaller than SPY's.")
                                        + ("" if sc["covers_whole_window"] else " (SPY file does not cover the whole window.)"))
                item["paragraph"] += item["spy_text"] + " "
        res["playbooks"].append(item)
    res["awards"] = awards(df, pool, info, series, trades, cfg, start_eq)
    return res


def awards(df, pool, info, series, trades, cfg, start_eq) -> list:
    out = []
    A = cfg["awards"]

    def card(key, row, extra="", **kw):
        a = {"key": key, "title": A[key]["title"], "n_configs_tried": info["scenarios_total"], **kw}
        if row is None:
            return {**a, "status": "none", "reason": extra}
        r = record(row, series, start_eq)
        return {**a, "status": "ok", **r, "paragraph": extra + _cash_txt(r)}

    for key in ("highest_win_rate", "highest_roi"):
        row, n, why = select(pool, A[key], start_eq)
        txt = ""
        if row is not None:
            by = A[key]["rank"]["by"]
            txt = (f"{row['strategy']} on {row['tickers']} at {int(row['allocation'])}% ({row['profile']} / {row['config_tag']}) has the best "
                   f"{'win rate' if by == 'win_rate' else 'annual return (CAGR)'} ({pct(row[by], 1 if by == 'win_rate' else 1)}) among {n} scenarios "
                   f"with at least {info['min_trades']} trades. ")
            if by == "win_rate":
                txt += f"A high win rate can hide rare large losses: its worst drawdown was {pct(row['max_dd'])}, and its zone is {row['zone']}. "
            else:
                txt += f"Its worst drawdown was {pct(row['max_dd'])} (zone {row['zone']}); high return is not the same as safe. "
        out.append(card(key, row, txt or why))
    # Happy Medium
    h = A["happy_medium"]
    mild = pool[pool["zone"] == h["zone"]]
    if mild.empty or len(pool) == 0:
        out.append(card("happy_medium", None, "no scenario in the Mild zone with enough trades"))
    else:
        q = mild["calmar"].quantile(h["calmar_quantile"])
        keep = mild[mild["calmar"] >= q]
        med_c, med_d = pool["cagr"].median(), pool["max_dd"].median()
        sc_c, sc_d = max(pool["cagr"].std(), 1e-9), max(pool["max_dd"].std(), 1e-9)
        dist = ((keep["cagr"] - med_c) / sc_c) ** 2 + ((keep["max_dd"] - med_d) / sc_d) ** 2
        row = keep.loc[dist.idxmin()]
        out.append(card("happy_medium", row, (
            f"Among the {len(mild)} Mild-zone scenarios, this is the one closest to the median config (median CAGR {pct(med_c)}, median drawdown {pct(med_d)}) "
            f"that still sits in the better half of Mild by return / max drawdown. Neither the best nor the worst: a typical, reasonable setup. ")))
    # Worst performance
    cand = df[df["n_trades"] >= cfg["min_trades"]]
    if cand.empty:
        out.append(card("worst_performance", None, "no scenario with enough trades"))
    else:
        w = cand.sort_values(["final_equity", "max_dd"], ascending=[True, False], kind="stable").iloc[0]
        out.append(card("worst_performance", w, f"{w['strategy']} on {w['tickers']} at {int(w['allocation'])}% ({w['profile']} / {w['config_tag']}) lost the most: "
                        + why_lost(w, trades) + " See the Ways to Lose Money section for how the same mistakes compound."))
    return out


# ------------------------------------------------------------------ rendering
def _fmt_item(it: dict) -> list[tuple[str, str]]:
    return [("Config", it["config"]), ("Allocation", f"{it['allocation']}%"), ("Tickers", it["tickers"]), ("Strategy", it["strategy"]),
            ("Level", it["level_tag"]), ("Zone", it["zone"]), ("Return", pct(it["total_return"], 0)), ("CAGR", pct(it["cagr"])),
            ("Max drawdown", pct(it["max_dd"])), ("Win rate", pct(it["win_rate"], 0)),
            ("Ruin odds", "n/a" if it["ruin_odds"] is None else pct(it["ruin_odds"], 2)), ("Trades", str(it["n_trades"])),
            ("Configs tried", f"{it.get('n_configs_tried', '?')}"),
            ("1st half", pct(it["first_half"])), ("2nd half", pct(it["second_half"]))] + (
        [("Return, no interest", pct(it["total_return_noint"], 0)), ("Cash-only", pct(it["cash_total_return"], 0)),
         ("vs cash", f"{it['excess_cagr'] * 100:+.1f} pts" + ("" if it["beats_cash"] else " (below cash)"))] if it.get("cash_cagr") is not None else [])


def render_html(res: dict, e=lambda s: __import__("html").escape(str(s))) -> str:
    p = res["pool"]
    H = ['<h2 id="playbooks">Playbooks and Awards</h2>',
         f'<p class="small">Each card is one scenario picked from the {p["scenarios_total"]:,} replayed ({p["configs_total"]} distinct configs). '
         f'Scenarios with fewer than {p["min_trades"]} trades are flagged <b>TOO FEW TRADES</b> and excluded ({p["excluded_too_few_trades"]} of them); '
         f'scenarios that went broke ({p["excluded_broke"]}) can only win "Worst Performance". '
         + (f'Scenarios that did not beat a cash-only account earning the fed funds rate are excluded too ({p.get("excluded_below_cash", 0)}). ' if p.get("require_beat_cash") else "")
         + f'Picking the best of many tries flatters the winner: '
         'read the first-half vs second-half numbers. Rankings never use final equity alone, except Lottery Ticket.</p>']
    if res["synthetic"]:
        H.append('<div class="synth">SYNTHETIC: these picks come from generated fixture logs.</div>')
    for it in res["playbooks"]:
        sub = f' <span class="muted small">({e(it["nickname"])})</span>' if it.get("nickname") else ""
        H.append(f'<div class="card"><h3>{e(it["title"])}{sub}</h3>')
        if it["status"] != "ok":
            H.append(f'<p>No pick: {e(it.get("reason", ""))}. Configs tried: {it.get("n_configs_tried", 0)}.</p>')
        elif it["key"] == "big_short":
            rows = "".join(f"<tr><td>{e(k)}</td><td class='n'>{v:+,.0f}</td></tr>" for k, v in it["windows"].items())
            H.append(f'<p><b>{e(it["strategy"])}</b> / {e(it["config_tag"])} - configs tried: {it["n_configs_tried"]}</p>'
                     f'<div class="scroll"><table><thead><tr><th>Stress window</th><th class="n">P&amp;L $/contract</th></tr></thead><tbody>{rows}</tbody></table></div>')
            if "record" in it:
                H.append(_kv(_fmt_item(it["record"]), e))
            H.append(f'<p>{e(it["paragraph"])}</p>')
        else:
            if it.get("luck_dependent"):
                H.append('<p class="caveat">LUCK-DEPENDENT: highest final equity of everything tried; shown beside its drawdown and ruin odds.</p>')
            H.append(_kv(_fmt_item(it), e))
            H.append(f'<p>{e(it["paragraph"])}</p>')
        H.append(f'<p class="small muted">{e(res["disclaimer"])}.</p></div>')
    H.append("<h3>Awards</h3>")
    for it in res["awards"]:
        H.append(f'<div class="card"><h3>{e(it["title"])}</h3>')
        if it["status"] != "ok":
            H.append(f'<p>No pick: {e(it.get("reason", ""))}.</p>')
        else:
            H.append(_kv(_fmt_item(it), e) + f'<p>{e(it["paragraph"])}</p>')
        H.append(f'<p class="small muted">{e(res["disclaimer"])}.</p></div>')
    H.append(f'<p>See also <a href="{res["ways_to_lose_link"]}">How to become an expert at losing money</a> (Ways to Lose Money).</p>'
             f'<p class="small muted">{e(res["disclaimer"])}.</p>')
    return "".join(H)


def _kv(pairs, e) -> str:
    return '<div class="tiles">' + "".join(f'<div class="tile"><span class="muted small">{e(k)}</span><b style="font-size:1rem">{e(v)}</b></div>' for k, v in pairs) + "</div>"


def render_md(res: dict) -> str:
    L = ["# Playbooks and Awards", ""]
    for it in res["playbooks"] + res["awards"]:
        L.append(f"## {it['title']}")
        L.append(it["paragraph"] if it["status"] == "ok" else f"No pick: {it.get('reason', '')}.")
        L += [f"*{res['disclaimer']}*", ""]
    L.append("See also: ways_to_lose_money.md")
    return "\n".join(L) + "\n"


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if (np.isnan(o) or np.isinf(o)) else float(o)
    if isinstance(o, (pd.Timestamp, dt.date)):
        return str(o)
    return o


def write_json(res: dict, path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_clean(res), indent=2))
    return p
