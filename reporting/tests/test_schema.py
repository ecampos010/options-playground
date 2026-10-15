import json
import shutil

import pandas as pd

from reporting import schema, synth
from reporting.schema import load_runs, validate_cli

from .conftest import mk_trades  # noqa: F401


def _write(tmp, trades_mut=None, marks_mut=None, drop_col=None):
    src = tmp / "good"
    synth.generate(src, seed=1, start="2018-01-02", end="2018-12-31", cadence=20,
                   variants=[v for v in synth.default_variants() if v.config_tag == "baseline"][:1])
    run = next(p for p in src.iterdir() if p.is_dir())
    t = pd.read_csv(run / "trades.csv", dtype=str, keep_default_na=False)
    m = pd.read_csv(run / "marks.csv", dtype=str, keep_default_na=False)
    if trades_mut:
        trades_mut(t)
    if marks_mut:
        m = marks_mut(m)
    if drop_col:
        t = t.drop(columns=[drop_col])
    t.to_csv(run / "trades.csv", index=False)
    m.to_csv(run / "marks.csv", index=False)
    return src


def errors(src):
    return [m for s, _, m in load_runs(src).problems if s == "error"]


def test_good_fixture_is_valid(tmp_path):
    src = _write(tmp_path)
    assert errors(src) == []
    assert validate_cli(str(src)) == 0


def test_missing_column_is_clear(tmp_path):
    src = _write(tmp_path, drop_col="collateral_per_contract")
    e = errors(src)
    assert any("missing columns" in m and "collateral_per_contract" in m for m in e)


def test_negative_collateral(tmp_path):
    def mut(t):
        t.loc[0, "collateral_per_contract"] = "-5"
    assert any("non-positive collateral" in m for m in errors(_write(tmp_path, trades_mut=mut)))


def test_exit_before_entry(tmp_path):
    def mut(t):
        t.loc[0, "exit_ts"] = "2000-01-01T10:00:00-05:00"
    assert any("exit before entry" in m for m in errors(_write(tmp_path, trades_mut=mut)))


def test_marks_outside_trade_life(tmp_path):
    def mut(m):
        m.loc[0, "date"] = "2030-01-01"
        return m
    assert any("outside trade life" in m for m in errors(_write(tmp_path, marks_mut=mut)))


def test_bad_enum_and_unknown_trade(tmp_path):
    def mut(t):
        t.loc[0, "exit_reason"] = "yolo"
    assert any("exit_reason" in m for m in errors(_write(tmp_path, trades_mut=mut)))

    def mm(m):
        m.loc[0, "trade_id"] = "NOPE"
        return m
    assert any("unknown trade_id" in m for m in errors(_write(tmp_path, marks_mut=mm)))


def test_missing_dir(tmp_path):
    assert any("does not exist" in m or "no runs" in m for m in errors(tmp_path / "nope"))
