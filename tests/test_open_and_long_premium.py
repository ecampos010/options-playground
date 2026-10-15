import json
import shutil
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from playground.engine import Leg
from playground.lab import is_open_at, run_lab, save_lab_run
from playground.strategies import LONG_PREMIUM, STRATEGIES, check_max_loss, collateral_per_contract, max_loss_ps
from reporting.schema import load_runs, validate_cli

from helpers import TI, cfg, market

FIX = Path(__file__).parent / "fixtures" / "sample_run"


# ---------------------------------------------------------------- item 0: open trades
def test_entry_oct_6_expiring_nov_20_is_open_when_data_ends_oct_30():
    assert is_open_at(date(2026, 11, 20), date(2026, 10, 30))
    assert not is_open_at(date(2026, 10, 30), date(2026, 10, 30))      # expires on the last data day: settled
    assert not is_open_at(date(2026, 10, 16), date(2026, 10, 30))


def test_lab_excludes_still_open_trades_and_reports_them(tmp_path):
    p = market(start="2026-07-01", end="2026-12-15")                 # data exists past the cut-off, the run ends 2026-10-30
    c = cfg(entry_cadence="weekly", dte_target=45, end_date="2026-10-30")
    r = run_lab(p, c, None, "", TI)
    last = pd.Timestamp("2026-10-30")
    assert len(r.trades) >= 1 and (pd.to_datetime(r.trades.expiry) <= last).all()    # only resolved trades are logged
    oe = r.meta["open_excluded"]
    assert oe["count"] == len(oe["trades"]) and oe["count"] >= 1
    assert all(pd.Timestamp(t["expiry"]) > last for t in oe["trades"])
    # never in trades.csv, marks or counts as win/loss
    opened = {t["entry_ts"] for t in oe["trades"]}
    assert not opened & set(r.trades.entry_ts.astype(str))
    assert r.meta["counts"]["trades"] == len(r.trades)
    out = save_lab_run(r, tmp_path)
    assert json.loads((out / "meta.json").read_text())["open_excluded"]["count"] == oe["count"]


def test_reporting_header_and_validate_show_open_excluded(tmp_path, capsys):
    run = tmp_path / "runs" / "r1"
    shutil.copytree(FIX, run)
    meta = json.loads((run / "meta.json").read_text())
    meta["open_excluded"] = {"count": 3, "trades": []}
    (run / "meta.json").write_text(json.dumps(meta))
    lg = load_runs(run.parent)
    assert lg.open_excluded == {"r1": 3} and lg.open_excluded_total() == 3
    validate_cli(str(run.parent))
    assert "open trades excluded (expiry after last data date): 3" in capsys.readouterr().out


# ---------------------------------------------------------------- item: long-premium strategies
def L(r, k, q):
    return Leg(r, k, q)


def test_long_premium_max_loss_is_the_debit_never_naked():
    assert set(LONG_PREMIUM) <= set(STRATEGIES)
    lp = [L("P", 95, 1)]
    assert check_max_loss("long_put", lp, -2.0) == pytest.approx(2.0)
    pds = [L("P", 100, 1), L("P", 94, -1)]
    assert check_max_loss("put_debit_spread", pds, -1.5) == pytest.approx(1.5)
    ls = [L("P", 95, 1), L("C", 105, 1)]
    assert check_max_loss("long_strangle", ls, -3.0) == pytest.approx(3.0)
    assert collateral_per_contract(ls, -3.0, 100.0) == pytest.approx(300.0)
    with pytest.raises(ValueError):
        max_loss_ps([L("C", 100, -1)], 1.0)                                 # a naked short call is still refused


@pytest.mark.parametrize("strategy", ["long_put", "put_debit_spread", "long_strangle"])
def test_long_premium_on_cached_spy_puts_opens_with_debit_as_collateral(strategy):
    r = run_lab(market(), cfg(strategy=strategy, strike_rule={"mode": "delta", "put_delta": 0.30, "call_delta": 0.16}),
                None, "", TI)
    assert len(r.trades) >= 3
    t = r.trades.iloc[0]
    assert t.strategy == strategy and t.collateral_per_contract > 0
    assert t.credit_per_contract < 0                                         # a debit was paid
    assert t.collateral_per_contract == pytest.approx(-t.credit_per_contract, abs=0.51) or t.collateral_per_contract >= -t.credit_per_contract - 0.51
    assert (r.trades.pnl_per_contract >= -r.trades.collateral_per_contract - 1.0).all()   # can never lose more than the debit (+fees)
