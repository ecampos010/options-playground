"""WAYS TO LOSE MONEY: swap one mistake at a time into the base setup, Monte Carlo the result."""
from __future__ import annotations

import math
import re
from functools import lru_cache

import numpy as np
import pandas as pd
import yaml

from . import montecarlo as mcm
from .replay import ReplayConfig, build_variant, monthly_stats, replay
from .thermo import CONFIG_DIR

CAVEAT = mcm.CAVEAT_MC


def load_config(path=None) -> dict:
    return yaml.safe_load(open(path or CONFIG_DIR / "ways_to_lose.yaml"))


def consecutive_losses_table(allocs=(5, 15, 20, 50, 100), lose=0.90) -> pd.DataFrame:
    """Number of consecutive FULL losses of the allocated slice needed to lose `lose` of the account:
    n = ln(1-lose)/ln(1-A). No stop, no credit received: the worst case."""
    rows = []
    for a in allocs:
        f = a / 100.0
        n = 1.0 if f >= 1 else math.log(1 - lose) / math.log(1 - f)
        rows.append({"allocation": a, "n_exact": n, "n_rounded": int(round(n)), "n_to_reach": int(math.ceil(n - 1e-9))})
    return pd.DataFrame(rows)


def _match(series: pd.Series, pat: str | None) -> pd.Series:
    if not pat:
        return pd.Series(True, index=series.index)
    rx = re.compile(pat, re.I)
    return series.astype(str).map(lambda x: bool(rx.search(x)))


def _sel(trades, spec):
    m = _match(trades["profile"], spec.get("profile")) & _match(trades["config_tag"], spec.get("config_tag")) & \
        _match(trades["strategy"], f"^{spec['strategy']}$" if spec.get("strategy") else None)
    return trades[m]


class Ways:
    def __init__(self, trades, marks, cfg: dict, rcfg: ReplayConfig | None = None, paths=10000, block=2, seed=7, horizon=36):
        self.trades, self.marks, self.cfg = trades, marks, cfg
        self.rcfg = rcfg or ReplayConfig()
        self.paths, self.block, self.seed, self.horizon = paths, block, seed, horizon
        self._vd = {}

    def _build(self, sub):
        key = (tuple(sorted(sub["profile"].unique())), tuple(sorted(sub["config_tag"].unique())), tuple(sorted(sub["strategy"].unique())), len(sub))
        if key not in self._vd:
            mk = self.marks[self.marks["uid"].isin(set(sub["uid"]))] if len(self.marks) else self.marks
            self._vd[key] = build_variant(sub, mk)
        return self._vd[key]

    def run(self, sub: pd.DataFrame, tickers: list, alloc: float, override=None):
        """Replay + MC for one setup. Returns dict or None when a ticker has no trades."""
        if sub.empty or not set(tickers) <= set(sub["ticker"]):
            return None
        vd = self._build(sub)
        mask = np.isin(vd.trades["ticker"].to_numpy(), tickers)
        res = replay(vd, mask, alloc / 100.0, self.rcfg)
        ret, low, mid, _ = monthly_stats(res.equity, vd.month_id, self.rcfg.start_equity)
        if override is not None:
            ret, low = override(ret, low, mid)
        mc = mcm.run_mc(ret, low, self.rcfg.start_equity, res.metrics["min_collateral"], (self.horizon,), self.block, self.paths, self.seed)
        h = self.horizon
        return {"p_broke": mc[f"mc{h}_p_broke"], "p_dd50": mc[f"mc{h}_p_dd50"], "p_dd90": mc[f"mc{h}_p_dd90"],
                "median_final": mc[f"mc{h}_final_p50"], "replay_max_dd": res.metrics["max_dd"], "replay_broke": res.metrics["broke"],
                "trades": res.metrics["n_trades"], "ret": ret, "low": low, "mid": mid}

    def table(self) -> pd.DataFrame:
        c, t = self.cfg, self.trades
        floor = float(self.cfg.get("base_floor", 0.001))
        b = c["base"]
        base_sub = _sel(t, {"profile": b["profile"], "config_tag": b.get("config_tag"), "strategy": b["strategy"]})
        base = self.run(base_sub, [b["ticker"]], b["allocation"])
        rows = []

        def add(name, r, how):
            if r is None:
                rows.append({"mistake": name, "how": how, "status": "not run", "p_broke": np.nan, "p_dd50": np.nan, "multiple": np.nan,
                             "median_final": np.nan, "replay_max_dd": np.nan})
                return
            mult = r["p_broke"] / max(base["p_broke"], floor) if base else np.nan
            rows.append({"mistake": name, "how": how, "status": "ok", "p_broke": r["p_broke"], "p_dd50": r["p_dd50"], "multiple": mult,
                         "median_final": r["median_final"], "replay_max_dd": r["replay_max_dd"]})

        if base is None:
            return pd.DataFrame([{"mistake": "BASE (not run)", "how": "no Level-1 Green SPY condor log found", "status": "not run"}])
        rows.append({"mistake": "BASE: Green SPY condor", "how": f"{b['allocation']}% allocation", "status": "ok", "p_broke": base["p_broke"],
                     "p_dd50": base["p_dd50"], "multiple": 1.0, "median_final": base["median_final"], "replay_max_dd": base["replay_max_dd"]})
        for a in c.get("oversize_allocations", [50, 100]):
            add(f"Oversizing ({a}%)", self.run(base_sub, [b["ticker"]], a), f"same logs, {a}% allocation")
        s = c["no_stop"]; sub = _sel(t, s)
        add("No stop", self.run(sub, [b["ticker"]], b["allocation"]), "experiment_no_stop profile")
        s = c["high_delta"]; sub = _sel(t, s)
        if sub.empty:
            add("High delta (25-35)", None, "")
        for tag, g in sub.groupby("config_tag"):
            add(f"High delta ({tag})", self.run(g, [b["ticker"]], b["allocation"]), f"config_tag {tag}")
        add("All guards off", self.run(_sel(t, c["all_guards_off"]), [b["ticker"]], b["allocation"]), "experiment_all_off profile")
        # forced entries into crash windows: base months overwritten by the stress-window months
        st = _sel(t[t["profile"].str.match(c["crash_forced"]["profile"], case=False)], {"strategy": b["strategy"]})
        if st.empty or b["ticker"] not in set(st["ticker"]):
            add("Forced entries into crash windows", None, "")
        else:
            vds = self._build(st)
            res = replay(vds, np.isin(vds.trades["ticker"].to_numpy(), [b["ticker"]]), b["allocation"] / 100.0, self.rcfg)
            r2, l2, m2, _ = monthly_stats(res.equity, vds.month_id, self.rcfg.start_equity)
            smap = {int(m): (a, b_) for m, a, b_ in zip(m2, r2, l2)}

            def override(ret, low, mid, smap=smap):
                ret, low = ret.copy(), low.copy()
                for i, m in enumerate(mid):
                    if int(m) in smap:
                        ret[i], low[i] = smap[int(m)]
                return ret, low
            add("Forced entries into crash windows", self.run(base_sub, [b["ticker"]], b["allocation"], override),
                "base months replaced by stress-log months (guards ignored in those windows)")
        tk = c["correlated"]["tickers"]
        add(f"Stacking correlated tickers ({'+'.join(tk)})", self.run(base_sub, tk, b["allocation"]), f"equal split of the same {b['allocation']}% vs {b['ticker']} alone")
        s = c["roll_losers"]; sub = _sel(t, s)
        sub = sub[sub["adj_count"].groupby(sub["config_tag"]).transform("max") > 0] if len(sub) else sub
        if sub.empty:
            add("Rolling losers", None, "")
        for tag, g in sub.groupby("config_tag"):
            add(f"Rolling losers ({tag})", self.run(g, [b["ticker"]], b["allocation"]), f"config_tag {tag} (adj_count>0)")
        add("Butterfly held without a stop", self.run(_sel(t, c["butterfly_no_stop"]), [b["ticker"]], b["allocation"]), "iron_butterfly, experiment_no_stop")
        s = c["post_spike"]; sub = _sel(t, s)
        waits = {}
        for tag in sub["config_tag"].unique():
            mm = re.search(s["config_tag"], tag, re.I)
            if mm:
                waits[tag] = int(mm.group(1))
        if not waits:
            add("Post-spike entries", None, "")
        else:
            lo, hi = min(waits, key=waits.get), max(waits, key=waits.get)
            add(f"Post-spike entries (shortest wait: {lo})", self.run(sub[sub["config_tag"] == lo], [b["ticker"]], b["allocation"]), f"wait {waits[lo]}d after a VIX spike")
            if hi != lo:
                add(f"Post-spike entries (longest wait: {hi}, for contrast)", self.run(sub[sub["config_tag"] == hi], [b["ticker"]], b["allocation"]), f"wait {waits[hi]}d after a VIX spike")
        tab = pd.DataFrame(rows)
        base_row = tab.iloc[[0]]
        rest = tab.iloc[1:].copy()
        rest["_s"] = rest["multiple"].fillna(-1)
        rest["_d"] = rest["p_dd50"].fillna(-1)
        rest = rest.sort_values(["_s", "_d"], ascending=False, kind="stable").drop(columns=["_s", "_d"])
        return pd.concat([base_row, rest], ignore_index=True)


def ways_markdown(tab: pd.DataFrame, analytic: pd.DataFrame, horizon=36, synthetic=False) -> str:
    from . import CAVEAT as HDR
    L = ["# Ways to lose money", "", f"> {HDR}", f"> {CAVEAT}"]
    if synthetic:
        L.append("> **SYNTHETIC** - built from generated fixtures, not market data.")
    L += ["", f"P(broke) over {horizon} months, block-bootstrap Monte Carlo. Multiple = P(broke) / base P(broke) (base floored at 0.1%). "
          "Rows with no log are printed as **not run**.", "",
          "| Mistake | How | P(broke) | P(DD>50%) | x base | Median final equity | Replay max DD |", "|---|---|---|---|---|---|---|"]
    for r in tab.to_dict("records"):
        if r.get("status") != "ok":
            L.append(f"| {r['mistake']} | {r.get('how', '')} | not run | not run | not run | not run | not run |")
        else:
            L.append(f"| {r['mistake']} | {r['how']} | {r['p_broke']:.1%} | {r['p_dd50']:.1%} | {r['multiple']:.1f}x | "
                     f"${r['median_final']:,.0f} | {r['replay_max_dd']:.0%} |")
    L += ["", "## Analytic: consecutive full losses to lose 90% of the account", "",
          "n = ln(0.10) / ln(1 - A). **No stop and no credit received - the worst case.**", "",
          "| Allocation | n (exact) | n (rounded) | Losses needed to reach -90% |", "|---|---|---|---|"]
    for r in analytic.to_dict("records"):
        L.append(f"| {r['allocation']}% | {r['n_exact']:.2f} | {r['n_rounded']} | {r['n_to_reach']} |")
    return "\n".join(L) + "\n"
