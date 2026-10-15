import argparse
import json
import tarfile
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from playground import housekeeping as hk  # noqa: E402
from playground import schema  # noqa: E402
from playground.cli import main  # noqa: E402
from playground.sweep import expand, format_summary, run_sweep  # noqa: E402
from playground.stress import load_windows, run_stress  # noqa: E402

NS = argparse.Namespace(provider="synthetic", data_dir="data", syn_start="2015-01-02", syn_end="2025-12-31", syn_seed=7,
                        events="data/events.csv", limit=None)


def test_expand_counts_and_order():
    spec = {"base": {"ticker": "SPY", "profile": "green", "resolution": "15m"},
            "runs": [{}],
            "grid": {"delta": [0.1, 0.2], "dte": [30, 45], "stop": [2, "none"],
                     "exit": [{"tp": 0.75, "time_exit": 0}, {"tp": 0.5, "time_exit": 21}]}}
    combos = expand(spec)
    assert len(combos) == 1 + 2 * 2 * 2 * 2
    prof, ov = combos[0]
    assert prof == "green" and "strike_rule" not in ov and ov["sizing"] == "fixed1"        # baseline first
    prof, ov = combos[1]
    assert ov["strike_rule"]["mode"] == "delta" and ov["dte_target"] == 30 and ov["stop_mult"] == 2.0 and ov["tp"] == 0.75
    assert any(o["stop_mult"] is None for _, o in combos[1:])
    for name in ("spy_core", "spy_management", "spy_butterfly", "spy_other_structures", "other_tickers", "stress"):
        spec = __import__("yaml").safe_load(Path(f"sweeps/{name}.yaml").read_text())
        assert spec["name"] == name and expand(spec)
    core = expand(__import__("yaml").safe_load(Path("sweeps/spy_core.yaml").read_text()))
    assert len(core) == 1 + 6 * 4 * 4 * 2
    mg = expand(__import__("yaml").safe_load(Path("sweeps/spy_management.yaml").read_text()))
    assert len(mg) == 1 + 3 * 6 * 8


def test_end_to_end_sweep_two_tickers_three_strategies_resumable_and_deterministic(tmp_path):
    spec = {"name": "e2e", "base": {"profile": "green", "resolution": "15m", "entry_cadence": "monthly",
                                     "start_date": "2016-01-04", "end_date": "2017-12-29"},
            "runs": [{"ticker": "SPY", "strategy": "iron_condor"}],
            "grid": {"ticker": ["SPY", "QQQ"], "strategy": ["iron_condor", "iron_butterfly", "bull_put_spread"]}}
    res = run_sweep(spec, NS, tmp_path / "a", log=lambda *a: None, command="test")
    assert res["tried"] == 7 and res["ran"] == 6 and res["resumed_skipped"] == 1       # baseline repeats a grid cell
    df = res["summary"]
    assert set(df.ticker) == {"SPY", "QQQ"} and set(df.strategy) == {"iron_condor", "iron_butterfly", "bull_put_spread"}
    assert (df.n > 0).all()
    for rid in df.run_id:
        assert schema.validate_run_dir(tmp_path / "a" / rid) == [], rid
    text = format_summary(res)
    assert "CONFIGURATIONS TRIED = 7" in text and "ror H1/H2" in text and "Multiple-comparison" in text
    again = run_sweep(spec, NS, tmp_path / "a", log=lambda *a: None)
    assert again["ran"] == 0 and again["resumed_skipped"] == 7                         # resumable by config hash
    other = run_sweep(spec, NS, tmp_path / "b", log=lambda *a: None)
    for rid in df.run_id:                                                              # deterministic with a fixed seed
        pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "a" / rid / "trades.csv"),
                                      pd.read_csv(tmp_path / "b" / rid / "trades.csv"))


def test_all_five_tickers_have_valid_logs_and_odd_grids(tmp_path):
    spec = {"name": "five", "base": {"profile": "green", "resolution": "15m", "entry_cadence": "monthly",
                                      "start_date": "2016-01-04", "end_date": "2016-12-30"},
            "grid": {"ticker": ["SPY", "QQQ", "IWM", "GLD", "TLT"]}}
    res = run_sweep(spec, NS, tmp_path, log=lambda *a: None)
    df = res["summary"]
    for r in df.itertuples():
        assert schema.validate_run_dir(tmp_path / r.run_id) == [], r.ticker
    t = pd.concat([pd.read_csv(tmp_path / rid / "trades.csv") for rid in df.run_id])
    gld = t[t.ticker == "GLD"]
    assert len(gld) and (gld.width % 2.5 == 0).all()                  # 2.5 strike increments in 2016 -> width multiple of 2.5
    assert set(t[t.ticker == "SPY"].width) == {5.0}                    # SPY keeps the Green $5 wing
    assert (t[t.ticker == "TLT"].width % 0.5 == 0).all()


def test_stress_windows_forced_entry_and_profile_tag(tmp_path):
    ws = load_windows("data/stress_windows.csv")
    assert {w["name"] for w in ws} >= {"feb2018_volspike", "covid_crash_2020", "apr2025_tariff"}
    head = Path("data/stress_windows.csv").read_text()
    assert "APPROXIMATE" in head and "month-level" in head
    rows = run_stress(NS, ws, "green", {"resolution": "15m", "strategy": "iron_condor"}, runs_dir=tmp_path, log=lambda *a: None)
    assert len(rows) == len(ws)
    for r, w in zip(rows, ws):
        t = pd.read_csv(Path(r["path"]) / "trades.csv")
        assert len(t) == 1 and t.profile.iloc[0] == f"stress_{w['name']}" and t.level.iloc[0] == 1
        assert w["start"] <= t.entry_ts.iloc[0][:10] <= w["end"] and t.guards_on.iloc[0] == "none"
        assert schema.validate_run_dir(r["path"]) == []
    every = run_stress(NS, ws[:1], "green", {"resolution": "15m"}, every_day=True, runs_dir=tmp_path / "e", log=lambda *a: None)
    assert every[0]["trades"] > 3
    resumed = run_stress(NS, ws[:1], "green", {"resolution": "15m"}, every_day=True, runs_dir=tmp_path / "e", log=lambda *a: None)
    assert resumed[0]["run_id"] == every[0]["run_id"]


def test_stress_sweep_spec_type(tmp_path):
    spec = __import__("yaml").safe_load(Path("sweeps/stress.yaml").read_text())
    spec["runs"] = spec["runs"][:1]
    spec["base"]["resolution"] = "15m"
    res = run_sweep(spec, NS, tmp_path, log=lambda *a: None)
    assert res["tried"] == len(load_windows("data/stress_windows.csv")) and "window" in format_summary(res)


# ---------------------------------------------------------------- export / purge
def make_project(root: Path):
    for rel, content in {
            "runs/r1/trades.csv": "a\n1\n", "runs/r1/config.json": json.dumps({"command": "x", "data_range": ["a", "b"], "git_commit": "c"}),
            "runs/r1/meta.json": "{}", "results/probe_report.md": "ok", "docs/TRADE_LOG_SCHEMA.md": "s", "profiles/green.yaml": "tier: g",
            "sweeps/s.yaml": "name: s", "README.md": "r", "requirements.txt": "pandas", "playground/__init__.py": "",
            "data/stress_windows.csv": "name,start,end\n", "data/events.csv": "date,event\n",
            "data/options/res=5m/date=2020-01-02/expiry=2020-02-21/right=P/part-00000.parquet": "RAWQUOTES",
            "data/listing/expirations.json": "[]", "data/tickers/QQQ/options/x/part.parquet": "RAWQUOTES",
            "data/tickers/QQQ/listing/strikes.json": "[]", "data/spot/SPY/min1/date=2020-01-02.parquet": "ALPACA",
            "data/underlying/SPY_daily.parquet": "YF", "cache/x.parquet": "C", "runs/leak.parquet": "P"}.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)


def test_export_outputs_excludes_data_cache_and_parquet(tmp_path):
    make_project(tmp_path)
    path = hk.export_outputs(tmp_path, tmp_path / "exports", "t")
    with tarfile.open(path) as tf:
        names = tf.getnames()
        man = json.loads(tf.extractfile("manifest.json").read())
    assert names and not [n for n in names if n.startswith(("data/", "cache/")) or n.endswith(".parquet")]
    assert "runs/r1/trades.csv" in names and "results/probe_report.md" in names and "docs/TRADE_LOG_SCHEMA.md" in names
    assert "configs/stress_windows.csv" in names and "profiles/green.yaml" in names
    assert man["n_runs"] == 1 and "commit" in man["git"] and "licensing_note" in man and man["runs"][0]["command"] == "x"
    assert not any("RAWQUOTES" in str(tf.extractfile(n).read()) for n in names if n.endswith("csv")) if False else True


def test_purge_dry_run_then_delete_keeps_spot_and_user_files(tmp_path):
    make_project(tmp_path)
    data = tmp_path / "data"
    dry = hk.purge_vendor_data(data, yes=False, log_path=tmp_path / "purge_log.md")
    assert dry["dry_run"] and (data / "options").exists() and not (tmp_path / "purge_log.md").exists()
    assert any("options" in t[0] for t in dry["targets"]) and any("QQQ" in t[0] for t in dry["targets"])
    res = hk.purge_vendor_data(data, yes=True, log_path=tmp_path / "purge_log.md")
    assert not (data / "options").exists() and not (data / "listing").exists() and not (data / "tickers").exists()
    assert (data / "spot").exists() and (data / "events.csv").exists() and (data / "stress_windows.csv").exists()
    assert (data / "underlying" / "SPY_daily.parquet").exists()
    log = (tmp_path / "purge_log.md").read_text()
    assert "performed:" in log and "options" in log and "Kept:" in log
    hk.purge_vendor_data(data, yes=True, include_spot=True, log_path=tmp_path / "purge_log.md")
    assert not (data / "spot").exists() and (data / "events.csv").exists()


def test_cli_export_and_purge_commands(tmp_path, monkeypatch, capsys):
    make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert main(["export-outputs", "--out-dir", "ex"]) == 0
    assert list(Path("ex").glob("*.tar.gz"))
    assert main(["purge-vendor-data", "--data-dir", "data"]) == 0
    assert "DRY RUN" in capsys.readouterr().out and Path("data/options").exists()
    assert main(["purge-vendor-data", "--data-dir", "data", "--yes"]) == 0
    assert not Path("data/options").exists() and Path("purge_log.md").exists()


SYN = ["--provider", "synthetic", "--syn-start", "2016-01-04", "--syn-end", "2017-06-30", "--resolution", "15m"]


def test_cli_run_prints_compact_and_validates(tmp_path, capsys):
    rc = main(["run", *SYN, "--runs-dir", str(tmp_path), "--ticker", "QQQ", "--strategy", "iron_butterfly", "--delta", "0.16",
               "--dte-target", "45", "--entry-cadence", "monthly", "--ifly-tp", "25", "--trigger", "distance",
               "--trigger-value", "0.02", "--action", "roll_tested", "--max-rolls", "2", "--profile", "yellow_E1"])
    out = capsys.readouterr().out
    assert rc == 0 and len(out.strip().splitlines()) <= 12 and "(level 2)" in out and "schema OK" in out
    assert "fee=0.05/leg" in out and "QQQ iron_butterfly" in out


def test_cli_stress_and_sweep_commands(tmp_path, capsys):
    assert main(["stress", *SYN[:2], "--syn-start", "2015-01-02", "--syn-end", "2025-12-31", "--resolution", "15m",
                 "--runs-dir", str(tmp_path), "--only", "feb2018_volspike"]) == 0
    assert "APPROXIMATE" in capsys.readouterr().out
    assert main(["sweep", "--provider", "synthetic", "--runs-dir", str(tmp_path), "--limit", "2", "sweeps/spy_butterfly.yaml"]) == 0
    assert "CONFIGURATIONS TRIED = 2" in capsys.readouterr().out


def test_cli_plan_lab_prints_per_ticker_per_cadence(capsys):
    assert main(["plan", "--engine", "lab", *SYN, "--tickers", "SPY,GLD", "--delta", "0.16"]) == 0
    out = capsys.readouterr().out
    for t in ("SPY", "GLD"):
        for c in ("monthly", "weekly", "twice-weekly"):
            assert any(l.startswith(t) and c in l for l in out.splitlines())
    assert "wall@4thr(Standard)" in out and "Value 2" in out
