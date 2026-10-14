from datetime import date

import pytest

from playground import rules
from playground.config import build_config
from playground.engine import Leg

G = build_config("green")
LEGS = [Leg("P", 95, -1), Leg("P", 90, 1), Leg("C", 105, -1), Leg("C", 110, 1)]
Q = [(1.00, 1.10), (0.40, 0.50), (0.80, 0.90), (0.30, 0.40)]   # sp, lp, sc, lc (bid, ask)


# ---------------------------------------------------------------- sizing
def test_contracts_floor_vs_full_use():
    # headroom 10,000, credit 0.90 -> 410 per contract -> 24.39
    assert rules.contracts(10_000, 0.90, G) == 24
    assert rules.contracts(10_000, 0.90, G.replace(full_use=True)) == 24
    # 24.6 rounds to 25 under full-use but floors to 24
    assert rules.contracts(10_090, 0.90, G) == 24
    assert rules.contracts(10_090, 0.90, G.replace(full_use=True)) == 25
    assert rules.contracts(0, 0.9, G) == 0 and rules.contracts(-5, 0.9, G) == 0


def test_headroom_caps():
    # budget 15% of 100k = 15k; expiry cap 60% = 9k
    assert rules.headrooms(100_000, 0, 0, G) == 9_000
    assert rules.headrooms(100_000, 9_000, 9_000, G) == 0
    assert rules.headrooms(100_000, 9_000, 0, G) == 6_000     # total budget binds
    assert rules.headrooms(100_000, 0, 0, G.replace(expiry_cap=False)) == 15_000


def test_strike_targets():
    p, c = rules.target_strikes(500.0, 15.3, G)
    assert p == pytest.approx(500 * (1 - 0.046)) and c == pytest.approx(500 * 1.04)
    p2, c2 = rules.target_strikes(500.0, 30.6, G)
    assert p2 == pytest.approx(500 * (1 - 0.092)) and c2 == pytest.approx(500 * 1.08)
    assert rules.nearest_listed(476.9, [475, 476, 477, 478]) == 477


# ---------------------------------------------------------------- filters
def test_gate_and_guards():
    assert rules.gate_block(25.0, 0.0, 0.0, G) == "gate"
    assert rules.gate_block(20.0, -0.015, 0.0, G) == "gate"
    assert rules.gate_block(20.0, 0.0, -0.04, G) == "gate"
    assert rules.gate_block(24.9, -0.0149, -0.0399, G) is None
    assert rules.gate_block(30.0, 0.0, 0.0, G.replace(gate=False)) is None
    assert rules.upside_block(0.04, G) == "upside"
    assert rules.upside_block(0.0399, G) is None
    assert rules.upside_block(0.10, G.replace(upside_guard=False)) is None
    assert rules.upside_block(0.05, G.replace(upside_5d=0.06)) is None


def test_event_buffer():
    assert rules.event_block([("FOMC", 2)], 600, G) == "event"
    assert rules.event_block([("CPI", 0)], 600, G) == "event"
    assert rules.event_block([("CPI", 3)], 600, G) is None
    assert rules.event_block([("ELECTION", 1)], 600, G) == "event"
    assert rules.event_block([("CPI", 1)], 600, G.replace(event_buffer=False)) is None
    # E1: day-of CPI allowed after 08:30 release, but not the days before
    e1 = build_config("yellow_E1")
    assert rules.event_block([("CPI", 0)], 600, e1) is None
    assert rules.event_block([("CPI", 1)], 600, e1) == "event"
    assert rules.event_block([("FOMC", 0)], 13 * 60, e1) == "event"     # before the 14:00 release
    assert rules.event_block([("FOMC", 0)], 14 * 60 + 5, e1) is None
    # E2: FOMC only blocked day-of
    e2 = build_config("yellow_E2")
    assert rules.event_block([("FOMC", 2)], 600, e2) is None
    assert rules.event_block([("FOMC", 0)], 600, e2) == "event"
    assert rules.event_block([("CPI", 2)], 600, e2) == "event"


def test_circuit_breaker():
    assert rules.cb_trigger(100.0, 99.99) and not rules.cb_trigger(100.0, 100.0)
    assert rules.cb_block(10, 31, G) == "cb"
    assert rules.cb_block(31, 31, G) is None
    assert rules.cb_block(10, 31, G.replace(cb_days=0)) is None


def test_red_requires_trip_and_lots():
    red = build_config("red")
    assert rules.notrip_block(15, 0.0, 0.0, red) == "notrip"
    assert rules.notrip_block(26, 0.0, 0.0, red) is None
    assert rules.gate_block(26, 0.0, 0.0, red) is None          # red is not blocked by the gate
    assert rules.lots_block(1, red) == "lots" and rules.lots_block(0, red) is None


# ---------------------------------------------------------------- fills
def test_fill_sign_conventions():
    nat = rules.entry_credit(LEGS, Q, G)
    mid = rules.entry_credit(LEGS, Q, G.replace(fill="mid"))
    hc = rules.entry_credit(LEGS, Q, G.replace(fill="haircut", haircut=0.05))
    assert nat == pytest.approx(1.00 - 0.50 + 0.80 - 0.40)          # sell bids, buy asks
    assert mid == pytest.approx(1.05 - 0.45 + 0.85 - 0.35)
    assert hc == pytest.approx(mid - 0.05)
    assert nat < hc < mid
    cn = rules.close_cost(LEGS, Q, G)                                # buy shorts at ask, sell longs at bid
    cm = rules.close_cost(LEGS, Q, G.replace(fill="mid"))
    ch = rules.close_cost(LEGS, Q, G.replace(fill="haircut"))
    assert cn == pytest.approx(1.10 - 0.40 + 0.90 - 0.30)
    assert cm == pytest.approx(1.05 - 0.45 + 0.85 - 0.35)
    assert ch == pytest.approx(cm + 0.05)
    assert cn > ch > cm
    # round trip at natural always loses the spread
    assert nat - cn < 0 < mid - cm + 1e-9


# ---------------------------------------------------------------- exits on hand-built sequences
def first_exit(seq, cfg=G, credit=0.90, entry_dte=35):
    for k, (nat, fill, spot, dte) in enumerate(seq):
        w = rules.check_exit(credit, nat, fill, spot, 95.0, 105.0, dte, entry_dte, cfg)
        if w:
            return k, w
    return None


def test_exit_tp_stop_breach_time():
    calm = (1.0, 0.9, 100.0, 30)
    assert first_exit([calm, calm, (0.5, 0.44, 100.0, 29)]) == (2, "tp")          # fill cost <= 0.5*credit
    assert first_exit([calm, (0.5, 0.46, 100, 29)]) is None
    assert first_exit([calm, (1.8, 1.8, 100.0, 29)]) == (1, "stop")               # natural >= 2x credit
    assert first_exit([calm, (1.79, 1.5, 100.0, 29)]) is None
    assert first_exit([calm, (1.2, 1.1, 95.0, 29)]) == (1, "breach")
    assert first_exit([calm, (1.2, 1.1, 105.5, 29)]) == (1, "breach")
    assert first_exit([calm, (1.2, 1.1, 100.0, 22), (1.2, 1.1, 100.0, 21)]) == (2, "time")
    assert first_exit([(1.2, 1.1, 100.0, 21)], G.replace(time_exit=0)) is None
    # tp 0.75 needs a deeper decline
    assert first_exit([(0.5, 0.44, 100, 29)], G.replace(tp=0.75)) is None
    assert first_exit([(0.3, 0.22, 100, 29)], G.replace(tp=0.75)) == (0, "tp")
    # E7: no stop
    assert first_exit([(5.0, 5.0, 100.0, 29)], build_config("yellow_E7")) is None
    # E3: 7-14 DTE entry -> time exit never applies
    assert first_exit([(1.2, 1.1, 100.0, 5)], build_config("yellow_E3"), entry_dte=10) is None


def test_settle_cost():
    assert rules.settle_cost(95, 90, 105, 110, 100) == 0
    assert rules.settle_cost(95, 90, 105, 110, 92) == 3
    assert rules.settle_cost(95, 90, 105, 110, 80) == 5
    assert rules.settle_cost(95, 90, 105, 110, 107) == 2
