"""RESERVED (stubs, not built).

Design hook: the engine's Position is a list of signed Legs with a `kind`; cash, marking, closing and
equity are leg-generic. A hedge is just another Position with kind="hedge_put" and its own exit rules.
"""
from __future__ import annotations


def overlay(*_a, **_k):
    # TODO: always-on SPY put overlay sized 0.5-1% of equity (cfg.overlay_pct). Plan:
    #  - each month (or at roll DTE), buy a Leg("P", strike ~ spot*(1-otm), +1) paid at ask (natural fill);
    #  - register as Position(kind="hedge_put"); engine._close/last_mid/equity already handle it;
    #  - add exit rules for kind="hedge_put" in engine.run (roll at N DTE; sell at bid);
    #  - exclude hedge premium from the Green risk budget (it is paid cash, not defined-risk collateral);
    #  - report hedge P&L separately in trades.csv (kind column already exists).
    raise NotImplementedError("overlay is reserved - see TODOs in playground/reserve.py")


def spike_study(*_a, **_k):
    # TODO: post-VIX-spike study. Plan:
    #  - find days VIX crosses above N (e.g. 30) after being < 20 within M days;
    #  - for each, enter Green/Red condors N days after the peak (peak = max VIX in window, no look-ahead:
    #    confirm peak only after VIX falls X% from it);
    #  - compare outcomes vs unconditional entries using the ladder machinery (same fills/seed).
    raise NotImplementedError("spike-study is reserved - see TODOs in playground/reserve.py")
