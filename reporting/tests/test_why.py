"""Why layer: month / drawdown cards, trade tags, what-ifs, Red months table. Fixture data only (nothing here is market data)."""
import json
import re

import numpy as np
import pandas as pd
import pytest

from reporting import why
from reporting.cash import RateTable
from reporting.context import Context, load_context, read_events
from reporting.replay import ReplayConfig

from .conftest import mk_marks, mk_trades

CFG = why.load_config()
RC = ReplayConfig(apy=0.0, rates=None, expiry_cap=None)


def scenario(extra_rows=None, guards=""):
    rows = [
        dict(id="T000", entry="2020-01-06", exit="2020-01-20", coll=1000, pnl=100, credit=200, reason="tp", cf=-300, expiry="2020-02-14"),
        dict(id="T001", entry="2020-02-03", exit="2020-02-10", coll=1000, pnl=-500, credit=200, reason="stop", cf=100, expiry="2020-02-28"),
        dict(id="T002", entry="2020-02-11", exit="2020-02-14", coll=1000, pnl=-780, credit=200, reason="breach", expiry="2020-02-14"),
    ] + (extra_rows or [])
    t = mk_trades(rows)
    t["guards_on"] = guards
    m = mk_marks(t, {})
    row = pd.Series(dict(scenario_id="s1", profile="green", config_tag="baseline", strategy="iron_condor", tickers="SPY", allocation=5,
                         level_tag="Green / Lvl 1", zone="Safe", final_equity=0.0, cagr=0.0, max_dd=0.0))
    return row, t, m


def ctx(events=True, rates=True):
    c = Context(spy=pd.Series([100.0, 90.0, 95.0], index=pd.to_datetime(["2020-01-31", "2020-02-14", "2020-02-28"])), spy_src="fixture",
                vix=pd.Series([14.0, 18.0, 31.0], index=pd.to_datetime(["2020-01-31", "2020-02-14", "2020-02-28"])), vix_src="fixture")
    if events:
        c.events = pd.DataFrame({"date": pd.to_datetime(["2020-02-05", "2020-02-12"]), "event": ["FOMC", "CPI"], "symbol": ""})
        c.events_src = "fixture"
    if rates:
        idx = pd.date_range("2019-12-01", "2020-03-31")
        c.rates = RateTable(pd.Series(np.where(idx < "2020-02-20", 1.5, 1.0), index=idx), 0.036, source="fixture")
    return c


@pytest.fixture(scope="module")
def sc():
    row, t, m = scenario()
    return why.build_scenario(row, "Test", t, m, RC, ctx(), CFG, 0)


def month(sc, ym):
    return next(c for c in sc["months"] if c["month"] == ym)


def test_every_month_gets_a_card_and_pnl_ties_out(sc):
    assert [c["month"] for c in sc["months"]] == ["2020-01", "2020-02"]
    jan, feb = month(sc, "2020-01"), month(sc, "2020-02")
    assert jan["pnl"] == pytest.approx(500) and feb["pnl"] == pytest.approx(-2500 - 3120)       # 5 ct x +100; then 5 ct x -500 and 4 ct x -780
    assert feb["trade_pnl"] == pytest.approx(feb["pnl"]) and feb["interest_usd"] == pytest.approx(0)
    assert feb["ret"] == pytest.approx(-5620 / 100500)


def test_losing_trades_are_tagged_stop_breach_and_gap(sc):
    feb = month(sc, "2020-02")
    by = {t["trade_id"]: t for t in feb["lost"]}
    assert "stop hit" in by["T001"]["tags"] and not any("gap" in g for g in by["T001"]["tags"])
    assert "short-strike breach" in by["T002"]["tags"] and any("gap through a wing" in g for g in by["T002"]["tags"])     # lost 780 of an 800 max loss
    assert feb["biggest"]["trade_id"] == "T002" and feb["biggest"]["usd"] == pytest.approx(-3120)
    assert "stop hit x1" in feb["lost_summary"] and "short-strike breach x1" in feb["lost_summary"]


def test_tags_unit():
    assert why.tags_for("stop_dollar", -100, 1000, 200, True, CFG) == ["stop hit"]
    assert why.tags_for("expiry", -790, 1000, 200, True, CFG)[1].startswith("gap through a wing")
    assert why.tags_for("expiry", -300, 1000, 200, True, CFG) == ["closed by expiry"]
    assert why.tags_for("breach", -5, 1000, 200, False, CFG) == ["still open (mark-to-market loss)"]


def test_context_fields_spy_vix_events_and_fed_regime(sc):
    feb = month(sc, "2020-02")
    assert feb["spy"]["ret"] == pytest.approx(95 / 100 - 1)
    assert feb["vix"]["start"] == 14.0 and feb["vix"]["end"] == 31.0 and feb["vix"]["regime_end"] == "VIX > 30" and feb["vix"]["regime_start"] == "VIX < 15"
    assert [(x["date"], x["event"]) for x in feb["events"]] == [("2020-02-05", "FOMC"), ("2020-02-12", "CPI")]
    assert feb["rate"]["trend"] == "cutting" and feb["rate"]["change_bp"] == pytest.approx(-50)
    assert month(sc, "2020-01")["events"] == []
    s = feb["summary"]
    assert "coincided with" in s and "caused" not in s.lower() and "FOMC" in s and "SPY -5.0%" in s


def test_red_flag_and_red_months_rows(sc):
    assert month(sc, "2020-02")["red"] and not month(sc, "2020-01")["red"]
    rows = why.red_rows([sc])
    assert len(rows) == 1 and rows[0]["month"] == "2020-02" and rows[0]["card"] == month(sc, "2020-02")["id"]
    assert "T002" in rows[0]["biggest"] and "FOMC 02-05" in rows[0]["events"]


def test_drawdown_episode_card(sc):
    assert len(sc["episodes"]) == 1
    ep = sc["episodes"][0]
    assert ep["peak_date"] == "2020-02-03" and ep["trough_date"] == "2020-02-14" and ep["recovery_date"] is None
    assert ep["depth"] == pytest.approx(5620 / 100500) and ep["pnl"] == pytest.approx(-5620)
    assert ep["biggest"]["trade_id"] == "T002" and ep["events_total"] == 2
    assert "coincided with" in ep["summary"]


def test_episode_detection_recovery_and_min_depth():
    eq = np.array([100, 98, 95, 99, 101, 100, 99.5, 100.2], float)
    eps = why._episodes(eq, 100.0, 0.03)
    assert len(eps) == 1 and eps[0]["trough_i"] == 2 and eps[0]["rec_i"] == 4 and eps[0]["depth"] == pytest.approx(0.05)
    both = why._episodes(eq, 100.0, 0.001)
    assert len(both) == 2 and both[1]['trough_i'] == 6 and 'rec_i' not in both[1]            # 101 is never regained: still open


def test_whatif_interest_matches_replay_and_reports_cash(sc):
    w = {x["key"]: x for x in sc["whatifs"]}
    i = w["interest"]
    assert i["on"]["final"] == pytest.approx(i["off"]["final"]) and "did NOT beat cash" in i["text"]     # RC has interest off, cash still earns the fixture rate
    row, t, m = scenario()
    rc = ReplayConfig(apy=0.036, rates=RateTable(pd.Series(2.0, index=pd.date_range("2019-12-01", "2020-03-31")), 0.036), expiry_cap=None)
    s2 = why.build_scenario(row, "T", t, m, rc, ctx(), CFG, 0)
    i2 = {x["key"]: x for x in s2["whatifs"]}["interest"]
    assert i2["on"]["final"] > i2["off"]["final"] and "Interest added" in i2["text"]


def test_whatif_event_filter_skips_pre_event_entries(sc):
    ef = {x["key"]: x for x in sc["whatifs"]}["event_filter"]
    assert ef["status"] == "ok" and ef["n_skipped"] == 2        # T001 entered Mon 02-03 (2 business days before the 02-05 FOMC), T002 on 02-11 (day before the 02-12 CPI)
    assert ef["off"]["final"] == pytest.approx(100000 + 500 - 2500 - 3120)
    assert ef["on"]["final"] == pytest.approx(100000 + 500)                    # both losers skipped, only T000 remains
    row, t, m = scenario(guards="gate;event")
    s2 = why.build_scenario(row, "T", t, m, RC, ctx(), CFG, 0)
    assert "already ON" in {x["key"]: x for x in s2["whatifs"]}["event_filter"]["text"]
    s3 = why.build_scenario(row, "T", t, m, RC, ctx(events=False), CFG, 0)
    assert {x["key"]: x for x in s3["whatifs"]}["event_filter"]["status"] == "n/a"
    assert month(s3, "2020-02")["events"] == [] and "n/a (no events.csv)" in why.card_fields(month(s3, "2020-02"))[3][1]


def test_whatif_early_exit_hold_to_expiry(sc):
    ee = {x["key"]: x for x in sc["whatifs"]}["early_exit"]
    assert ee["status"] == "ok" and ee["n_trades"] == 3
    assert ee["on"]["final"] == pytest.approx(100000 + 500 - 2500 - 3120)
    assert ee["off"]["final"] == pytest.approx(100000 - 1500)                  # T000 held to its 02-14 expiry loses 5 x 300; it ties up the cash so T001/T002 cannot be taken
    row, t, m = scenario()
    t["cf_pnl_to_expiry_per_contract"] = np.nan
    none = why.build_scenario(row, "T", t, m, RC, ctx(), CFG, 0)
    assert {x["key"]: x for x in none["whatifs"]}["early_exit"]["status"] == "n/a"


def test_month_card_whatif_lines(sc):
    feb = month(sc, "2020-02")
    txt = " ".join(feb["whatif"])
    assert "early exit(s) closed this month realised" in txt and "entered within 2 business days before an event" in txt
    assert not any("caused by" in w for w in feb["whatif"])


def test_context_loader_fallbacks(tmp_path):
    t = mk_trades([dict(entry="2020-01-06", exit="2020-01-20", coll=1000, pnl=1)])
    c = load_context(tmp_path, tmp_path / "events.csv", RateTable(None, 0.036), t)
    assert c.events is None and "trade logs" in c.spy_src and c.notes and "not found" in c.notes[0]
    (tmp_path / "events.csv").write_text("# x\ndate,event,symbol\n2020-02-05,fomc,\n")
    assert read_events(tmp_path / "events.csv")["event"].tolist() == ["FOMC"]


# ------------------------------------------------------------------ report-level
@pytest.fixture(scope="module")
def whyjson(report_ctx_dir):
    return json.loads((report_ctx_dir / "why.json").read_text())


def test_report_writes_why_files_and_covers_every_month(report_ctx_dir, whyjson):
    for f in ("why.md", "why.json", "red_months.csv"):
        assert (report_ctx_dir / f).exists()
    assert 1 <= len(whyjson["scenarios"]) <= 4
    res = pd.read_csv(report_ctx_dir / "results.csv").set_index("scenario_id")
    for s in whyjson["scenarios"]:
        months = [c["month"] for c in s["months"]]
        assert months == sorted(set(months)) and len(months) >= 30                     # one card per calendar month of the sample
        tot = sum(c["pnl"] for c in s["months"])
        assert tot == pytest.approx(res.loc[s["scenario_id"], "final_equity"] - 100000, abs=1.0)    # the rebuilt replay equals the grid replay
        assert {w["key"] for w in s["whatifs"]} == {"interest", "early_exit", "event_filter"}
        for c in s["months"] + s["episodes"]:
            assert c["summary"] and "lost_summary" in c


def test_report_cards_use_context_data(report_ctx_dir, whyjson):
    mar = [c for s in whyjson["scenarios"] for c in s["months"] if c["month"] == "2020-03"]
    assert mar
    for c in mar:
        assert c["vix"]["regime_end"] == "VIX > 30" and c["spy"]["ret"] < -0.05
        assert {"FOMC", "CPI"} <= {x["event"] for x in c["events"]}
        assert c["rate"]["trend"] == "cutting" and c["rate"]["level"] in ("near zero", "moderate")
        assert "fixture" not in json.dumps(c["vix"]) and c["vix"]["src"].endswith("VIX_daily.parquet")


def test_html_red_months_table_search_and_links(report_ctx_dir):
    page = (report_ctx_dir / "index.html").read_text()
    assert 'id="why"' in page and 'id="red-months"' in page and 'id="red-q"' in page and 'type="search"' in page
    tbl = page[page.index('id="red-table"'):]
    tbl = tbl[:tbl.index("</table>")]
    hrefs = re.findall(r'class="why-link">card', tbl)
    links = re.findall(r'href="#(why-[^"]+)" class="why-link"', tbl)
    assert links and len(links) == len(hrefs)
    for l in links:
        assert f'id="{l}"' in page                                                     # every Red months row links to an existing card
    assert "Red months" in page and page.count('class="why-card') >= 30
    assert "coincided with" in page and "caused by" not in page.lower()


def test_wording_never_says_caused_by(report_ctx_dir):
    for f in ("why.md", "why.json", "index.html"):
        t = (report_ctx_dir / f).read_text().lower()
        assert "caused by" not in t and "because of" not in t


def test_red_months_csv_matches_threshold(report_ctx_dir, whyjson):
    red = pd.read_csv(report_ctx_dir / "red_months.csv")
    assert len(red) and (red["ret"] <= CFG["red_month_return"]).all()
    assert red["ret"].is_monotonic_increasing                                             # worst first
    n = sum(1 for s in whyjson["scenarios"] for c in s["months"] if c["red"])
    assert len(red) == n


def test_report_without_context_files_says_na(report_dir):
    page = (report_dir / "index.html").read_text()
    assert "n/a (no events.csv)" in page or "Event filter what-if: n/a" in page or "n/a: no events.csv" in page
    assert 'id="red-months"' in page and "sparse" in page                                # SPY/VIX fell back to the logs, labelled sparse


def test_browser_red_months_search_link_opens_card_and_no_js_errors(report_ctx_dir):
    pw = pytest.importorskip("playwright.sync_api")
    import glob
    with pw.sync_playwright() as p:
        try:
            try:
                b = p.chromium.launch(args=["--no-sandbox"])
            except Exception:  # noqa: BLE001
                exe = next(iter(glob.glob("/opt/pw-browsers/chromium-*/chrome-linux/chrome")), "/opt/pw-browsers/chromium")
                b = p.chromium.launch(executable_path=exe, args=["--no-sandbox"])
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"chromium unavailable: {e}")
        pg = b.new_page(viewport={"width": 420, "height": 900})
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto("file://" + str(report_ctx_dir / "index.html"))
        pg.wait_for_selector("body[data-ready]")
        assert pg.evaluate("document.body.dataset.ready") == "1" and errs == []
        assert pg.evaluate("document.documentElement.scrollWidth") <= 420
        total = pg.locator("#red-table tbody tr").count()
        assert total > 0
        key = pg.locator("#red-table tbody tr").first.locator("td").nth(1).inner_text()      # the first row's month, e.g. 2019-08
        pg.fill("#red-q", key)
        shown = pg.locator("#red-table tbody tr:visible").count()
        assert 0 < shown < total and "shown" in pg.inner_text("#red-count")
        pg.fill("#red-q", "zzzz-no-such")
        assert pg.locator("#red-table tbody tr:visible").count() == 0
        pg.fill("#red-q", "")
        pg.locator("#red-table tbody tr:visible a.why-link").first.click()
        target = pg.evaluate("location.hash.slice(1)")
        assert pg.evaluate(f"document.getElementById('{target}').open") is True
        assert pg.evaluate(f"document.getElementById('{target}').closest('details.why-scn').open") is True
        assert errs == []
        b.close()
