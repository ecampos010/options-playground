"""BWB body short: the same delta rule as the iron condor short, and a per-entry record of each short's delta source."""
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from helpers import TI, cfg, market  # noqa: E402

from playground import factory  # noqa: E402
from playground.config import build_config  # noqa: E402
from playground.lab import ENTRY_COLUMNS, run_lab, save_lab_run  # noqa: E402
from playground.strategies import Chain, IronCondor, PutBWB  # noqa: E402

SPOT, WIDTH = 450.0, 5.0


def _delta_cfg():
    rule = {**build_config("green").strike_rule, "mode": "delta", "put_delta": 0.16, "call_delta": 0.16}
    return build_config("green", {"strike_rule": rule, "strategy": "iron_condor"})


def _chain(put_delta_of, call_delta_of, ks=range(400, 501), no_quote=()):
    """A chain where every strike has a bid, an ask, a delta, and a quoted wing WIDTH away on its own side."""
    quotes, deltas = {}, {}
    for k in ks:
        for right, dfn in (("P", put_delta_of), ("C", call_delta_of)):
            if (right, float(k)) in no_quote:
                continue
            quotes[(right, float(k))] = (1.0, 1.1)
            d = dfn(float(k))
            if d is not None:
                deltas[(right, float(k))] = d
    return Chain(_delta_cfg(), SPOT, 15.0, list(map(float, ks)), quotes, deltas, WIDTH)


def _abs_delta_put(k):                       # |delta| rises toward spot as the put strike rises
    return -1.0 / (1.0 + math.exp(-(k - 430.0) / 3.0))


def _abs_delta_call(k):
    return 1.0 / (1.0 + math.exp((k - 470.0) / 3.0))


def test_bwb_body_is_the_nearest_put_to_the_0_16_delta_target_like_the_condor():
    ch = _chain(_abs_delta_put, _abs_delta_call)
    expected = min((k for (r, k) in ch.q if r == "P" and ch.wing("P", k, -1) is not None and ch.q[("P", k)][0] > 0),
                   key=lambda k: abs(abs(ch.delta("P", k)) - 0.16))
    legs = PutBWB().build(ch)
    body = next(l for l in legs if l.qty == -2)
    assert body.strike == expected
    condor = IronCondor().build(ch)
    assert next(l for l in condor if l.right == "P" and l.qty < 0).strike == expected


def test_bwb_body_skips_the_nearest_strike_when_it_has_no_quote_and_takes_the_next_nearest():
    full = _chain(_abs_delta_put, _abs_delta_call)
    best = min((k for (r, k) in full.q if r == "P"), key=lambda k: abs(abs(full.delta("P", k)) - 0.16))
    ch = _chain(_abs_delta_put, _abs_delta_call, no_quote={("P", best)})     # the same chain, minus that one quote
    expected = min((k for (r, k) in ch.q if r == "P" and ch.wing("P", k, -1) is not None),
                   key=lambda k: abs(abs(ch.delta("P", k)) - 0.16))
    assert expected != best
    legs = PutBWB().build(ch)
    assert next(l for l in legs if l.qty == -2).strike == expected
    assert next(l for l in IronCondor().build(ch) if l.right == "P" and l.qty < 0).strike == expected


def test_no_delta_on_any_put_means_no_bwb_and_no_condor():
    ch = _chain(lambda k: None, _abs_delta_call)
    assert PutBWB().build(ch) is None
    assert IronCondor().build(ch) is None


def test_the_bwb_wings_keep_their_fixed_width_rule():
    ch = _chain(_abs_delta_put, _abs_delta_call)
    legs = PutBWB().build(ch)
    body = next(l.strike for l in legs if l.qty == -2)
    near = next(l.strike for l in legs if l.qty == 1 and l.strike > body)
    far = next(l.strike for l in legs if l.qty == 1 and l.strike < body)
    assert abs(near - (body + ch.cfg.bwb_near_pct * SPOT)) <= 0.5 * WIDTH + 1e-9
    assert abs(far - (body - ch.cfg.bwb_far_pct * SPOT)) <= 0.5 * WIDTH + 1e-9


def test_entries_csv_records_the_delta_source_of_every_bwb_entry(tmp_path):
    prov = factory.synthetic("SPY", "2015-01-02", "2016-12-31", 7)
    c = cfg(strategy="put_bwb", entry_cadence="weekly", dte_target=45, sizing="equity-pct",
            strike_rule={**build_config("green").strike_rule, "mode": "delta", "put_delta": 0.16, "call_delta": 0.16},
            resolution="5m", start_date="2016-01-04", end_date="2016-06-30")
    res = run_lab(prov, c, prov.synthetic_events(), "SYNTHETIC")
    out = save_lab_run(res, tmp_path)
    assert list(pd.read_csv(out / "entries.csv").columns) == ENTRY_COLUMNS
    entries = pd.read_csv(out / "entries.csv")
    trades = pd.read_csv(out / "trades.csv")
    assert len(trades) > 0
    assert set(trades.trade_id) <= set(entries.trade_id)                  # every finished trade has its entry row
    assert len(entries) - len(trades) == res.meta["open_excluded"]["count"]   # the rest are still open at data end
    assert entries.short_put_delta_source.isin(["thetadata", "black_scholes"]).all()
    merged = entries.merge(trades[["trade_id", "short_put_delta"]], on="trade_id", suffixes=("", "_trade"))
    assert ((merged.short_put_delta - merged.short_put_delta_trade).abs() <= 5e-5).all()   # trades.csv rounds to 4 dp
    assert list(trades.columns) == list(pd.read_csv(out / "trades.csv").columns)   # trades.csv contract unchanged


def test_a_condor_run_has_the_entry_record_with_the_same_columns():
    res = run_lab(market(), cfg(strategy="iron_condor"), None, "", TI)
    assert res.entries is not None and list(res.entries.columns) == ENTRY_COLUMNS
