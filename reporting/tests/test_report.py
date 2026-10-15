import glob
import hashlib
import re
import subprocess
import sys

import pandas as pd
import pytest

from reporting import synth


def test_synthetic_is_deterministic(tmp_path):
    vs = [v for v in synth.default_variants() if v.config_tag == "baseline"][:1]
    for n in ("a", "b"):
        synth.generate(tmp_path / n, seed=4, start="2018-01-02", end="2018-12-31", cadence=15, variants=vs)
    h = lambda n: hashlib.md5((tmp_path / n / "green__baseline" / "trades.csv").read_bytes()).hexdigest()  # noqa: E731
    assert h("a") == h("b")
    synth.generate(tmp_path / "c", seed=5, start="2018-01-02", end="2018-12-31", cadence=15, variants=vs)
    assert h("a") != h("c")


def test_outputs_exist_and_marked_synthetic(report_dir):
    for f in ("index.html", "results.csv", "verdicts.md", "ways_to_lose_money.md", "stress.md"):
        assert (report_dir / f).exists()
    res = pd.read_csv(report_dir / "results.csv")
    assert res["synthetic"].all()
    assert {"level", "tickers", "strategy", "profile", "config_tag", "allocation", "zone", "verdict", "level_tag"} <= set(res.columns)
    assert "SYNTHETIC" in (report_dir / "verdicts.md").read_text()
    assert glob.glob(str(report_dir / "equity" / "*.parquet"))
    ok = res[res["list"] == "ranked"]
    assert not ok["broke"].any() and res[res["list"] == "broke"]["broke"].all()
    nt = res.groupby(["profile", "config_tag", "strategy", "allocation"]).size()
    assert nt.max() >= 1 and set(res["allocation"]) == {5, 15, 50, 100}
    assert set(res["level_tag"]) <= {"Green / Lvl 1", "Yellow / Lvl 2", "Red / Lvl 3", "Experiment / Lvl 4"}


def test_ways_md_has_not_run_rows_and_analytic(report_dir):
    t = (report_dir / "ways_to_lose_money.md").read_text()
    assert "consecutive full losses" in t and "| 15% | 14.17 | 14 | 15 |" in t
    assert "worst case" in t and "understates tails" in t


def test_html_single_file_no_external_urls(report_dir):
    page = (report_dir / "index.html").read_text()
    assert len(page.encode()) < 8_000_000
    assert not re.search(r"https?://", page) and "<link" not in page and not re.search(r"<script[^>]+src=", page)
    assert "10-year real-quote sample; results are relative comparisons, not forecasts." in page
    assert "SYNTHETIC" in page
    css = page[page.index("<style>"):page.index("</style>")]
    assert ":root{" in css and "prefers-color-scheme:dark" in css and ':root[data-theme="dark"]' in css
    assert "--bg:" in css and "--fg:" in css


def _launch(p):
    try:
        return p.chromium.launch(args=["--no-sandbox"])
    except Exception:  # noqa: BLE001
        exe = next(iter(glob.glob("/opt/pw-browsers/chromium-*/chrome-linux/chrome")), "/opt/pw-browsers/chromium")
        return p.chromium.launch(executable_path=exe, args=["--no-sandbox"])


def test_html_loads_in_chromium_without_errors_and_no_overflow(report_dir):
    pw = pytest.importorskip("playwright.sync_api")
    with pw.sync_playwright() as p:
        try:
            b = _launch(p)
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"chromium unavailable: {e}")
        for scheme in ("light", "dark"):
            for w in (400, 1100):
                pg = b.new_page(viewport={"width": w, "height": 900}, color_scheme=scheme)
                errs = []
                pg.on("pageerror", lambda e: errs.append(str(e)))
                pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
                pg.goto("file://" + str(report_dir / "index.html"))
                pg.wait_for_selector("body[data-ready]")
                assert pg.evaluate("document.body.dataset.ready") == "1", pg.evaluate("window.__errors")
                assert errs == []
                assert pg.evaluate("document.documentElement.scrollWidth") <= w
                assert pg.evaluate("getComputedStyle(document.body).backgroundColor") != ""
                assert pg.query_selector("#eq-chart svg path") is not None and pg.query_selector("#dd-chart svg path") is not None
                assert pg.query_selector("#thermo .gauge") is not None
                if w == 1100 and scheme == "light":
                    box = pg.query_selector("#eq-chart .hit").bounding_box()
                    pg.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
                    assert pg.query_selector("#eq-chart .tip").is_visible()
                    pg.select_option("#f-view", "br")
                    pg.click("#btn-log")
                    assert pg.evaluate("window.__errors") in (None, [])
                pg.close()
        b.close()


def test_cli_validate_and_exit_codes(synth_dir, tmp_path):
    ok = subprocess.run([sys.executable, "-m", "reporting", "validate", str(synth_dir)], capture_output=True, text=True)
    assert ok.returncode == 0 and "SYNTHETIC" in ok.stdout
    bad = subprocess.run([sys.executable, "-m", "reporting", "validate", str(tmp_path / "missing")], capture_output=True, text=True)
    assert bad.returncode == 1
