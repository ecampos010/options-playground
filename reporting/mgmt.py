"""Management-rule scorecard: saved / cost / net vs holding to expiry, paired against a baseline config."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .stress import PROTECTIVE

STRESS = r"^stress_"


def pick_baseline(trades: pd.DataFrame, tag: str | None = None) -> str | None:
    tags = set(trades["config_tag"])
    if tag:
        return tag if tag in tags else None
    for cand in ("baseline", "base", "hold"):
        if cand in tags:
            return cand
    l1 = trades[trades["level"] == 1]
    return l1["config_tag"].value_counts().index[0] if len(l1) else None


def scorecard(trades: pd.DataFrame, baseline_tag: str | None = None) -> dict:
    t = trades[~trades["profile"].str.match(STRESS, case=False)].copy()
    base_tag = pick_baseline(t, baseline_tag)
    if base_tag is None:
        return {"baseline": None, "n_configs": 0, "table": pd.DataFrame()}
    t["key"] = t["ticker"] + "|" + t["strategy"] + "|" + t["entry_ts"].dt.strftime("%Y-%m-%d")
    b = t[t["config_tag"] == base_tag].drop_duplicates("key").set_index("key")
    rows = []
    excluded: list = []
    for (p, tag), g in t.groupby(["profile", "config_tag"]):
        if tag == base_tag and p == b["profile"].iloc[0]:
            continue
        early = (g["exit_reason"] != "expiry") | (g["adj_count"] > 0)
        cf = g["cf_pnl_to_expiry_per_contract"]
        e = g[early & cf.notna()]
        diff = e["pnl_per_contract"] - e["cf_pnl_to_expiry_per_contract"]
        prot = e[e["exit_reason"].isin(PROTECTIVE)]
        gm = g.drop_duplicates("key").set_index("key")
        common = gm.index.intersection(b.index)
        if len(common):
            dd_ = (gm.loc[common, "short_put_delta"] - b.loc[common, "short_put_delta"]).abs().median()
            if pd.notna(dd_) and dd_ > 0.005:      # different strikes = a different strategy, not a management rule
                excluded.append(f"{p}/{tag}")
                continue
        d = (gm.loc[common, "pnl_per_contract"] - b.loc[common, "pnl_per_contract"]) if len(common) else pd.Series(dtype=float)
        half = np.nan
        first = second = np.nan
        verdict = "no matched entries"
        if len(common):
            ts = gm.loc[common, "entry_ts"]
            cut = ts.sort_values().iloc[len(ts) // 2]
            f, s = d[ts < cut], d[ts >= cut]
            first = float(f.mean()) if len(f) else np.nan
            second = float(s.mean()) if len(s) else np.nan
            beats = d.mean() > 0
            if not beats:
                verdict = "worse than baseline"
            elif first > 0 and second > 0:
                verdict = "holds up"
            else:
                verdict = "unproven"
        rows.append({
            "profile": p, "config_tag": tag, "level": int(g["level"].mode().iloc[0]), "trades": len(g),
            "early_exits": int(early.sum()), "adjusted": int((g["adj_count"] > 0).sum()),
            "saved_usd": float(diff[diff > 0].sum()), "cost_usd": float(-diff[diff < 0].sum()), "net_usd": float(diff.sum()),
            "whipsaw_rate": float((prot["cf_pnl_to_expiry_per_contract"] > 0).mean()) if len(prot) else np.nan,
            "protective_exits": len(prot), "matched_entries": len(common),
            "vs_base_mean": float(d.mean()) if len(d) else np.nan, "first_half": first, "second_half": second,
            "label": verdict,
        })
    tab = pd.DataFrame(rows)
    if len(tab):
        tab = tab.sort_values(["label", "vs_base_mean"], ascending=[True, False]).reset_index(drop=True)
    return {"baseline": base_tag, "n_configs": len(tab), "table": tab, "excluded_other_strikes": excluded}
