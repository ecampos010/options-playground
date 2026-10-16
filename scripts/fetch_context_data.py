#!/usr/bin/env python3
"""Download the market-context calendars and the risk-free rate from official public pages.

    python scripts/fetch_context_data.py                      # 2016-2026 -> data/events.csv + data/rates.csv
    python scripts/fetch_context_data.py --check              # re-validate the files already in data/ (no network)
    python scripts/fetch_context_data.py --html-dir saved/    # parse pre-downloaded pages instead of fetching

Sources (public; the URL patterns below are the ones tried, in this order):
  FOMC        federalreserve.gov/monetarypolicy/fomccalendars.htm, fomchistorical<year>.htm for 2016-2020 (decision day = last day of each scheduled
              meeting; unscheduled / emergency meetings such as March 2020 are included; notation votes are not)
  CPI/PPI/NFP 1. bls.gov/bls/news-release/<cpi|ppi|empsit>.htm   archived releases; link names embed the date (empsit_08072026.htm)
              2. bls.gov/schedule/news_release/<cpi|ppi|empsit>.htm   reference month -> release date
              3. bls.gov/schedule/<year>/<MM>_sched_list.htm   monthly lists (recent years only; a 404 is non-fatal)
              4. only if FRED_API_KEY is set: FRED fred/release/dates for releases 10 (CPI), 46 (PPI), 50 (Employment Situation)
              Each (event, year) uses the first source whose count is as expected, else the fullest one; the output says which.
  ELECTION    federal general-election day, computed from the statute (Tuesday after the first Monday of November,
              even years); nothing is scraped for it
  rates       fred.stlouisfed.org/graph/fredgraph.csv?id=DFF (or EFFR): daily effective federal funds rate, percent (no key)

Dates are NEVER invented: every FOMC/CPI/PPI/NFP date comes from a downloaded page or the FRED API. The run never aborts on one
bad source: it writes what it found (an earlier good file fills holes), prints a coverage table (event x year) and the source used,
and exits 0 only if FOMC, CPI and NFP exist for every year (else 4; --strict makes count anomalies exit 2). The sandbox this was
written in could not reach the sites, so the parsers are tested on saved fixtures only; use --html-dir with pages saved in a browser
if a site blocks the script. FRED_API_KEY is read from the environment only (set -a; source .env; set +a) and is never printed.

Output: data/events.csv (date,event,symbol; EARNINGS rows already in the file are kept) and data/rates.csv
(date,rate_pct). Needs internet; standard library only.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import html as htmllib
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

EVENT_TYPES = ("FOMC", "CPI", "PPI", "NFP", "ELECTION")
KNOWN_TYPES = EVENT_TYPES + ("EARNINGS",)
WEEKDAY_ONLY = ("CPI", "PPI", "NFP")      # FOMC is exempt: emergency meetings can fall on a weekend (Sunday 15 March 2020)
FOMC_PER_YEAR = (7, 9)                  # "about 8": the scheduled calendar has 8 every year in 2016-2026
MONTHLY_PER_YEAR = 12                   # CPI, PPI, NFP: exactly 12 unless explicitly allowed (e.g. 2025 shutdown)
BLS_RELEASES = {"CPI": "Consumer Price Index", "PPI": "Producer Price Index", "NFP": "Employment Situation"}
FOMC_CAL_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
FOMC_HIST_URL = "https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm"
FOMC_HIST_LAST = 2020                   # fomchistorical<year>.htm is requested for 2016-2020; 2021+ comes from the calendars page
# BLS, in the order tried per series (slug = cpi | ppi | empsit):
BLS_ARCHIVE_URL = "https://www.bls.gov/bls/news-release/{slug}.htm"            # archived news releases by year; file names embed the release date (empsit_08072026.htm)
BLS_SCHED_URL = "https://www.bls.gov/schedule/news_release/{slug}.htm"         # reference month -> release date, current / upcoming / prior years
BLS_MONTH_URL = "https://www.bls.gov/schedule/{year}/{month:02d}_sched_list.htm"   # monthly list; exists for recent years only (2016 is a 404)
BLS_SLUG = {"CPI": "cpi", "PPI": "ppi", "NFP": "empsit"}
FRED_RELEASE_IDS = {"CPI": 10, "PPI": 46, "NFP": 50}                           # FRED release ids: Consumer Price Index, Producer Price Index, Employment Situation
FRED_REL_URL = ("https://api.stlouisfed.org/fred/release/dates?release_id={rid}&api_key={key}&file_type=json&sort_order=asc"
                "&realtime_start=1776-07-04&realtime_end=9999-12-31&include_release_dates_with_no_data=true&limit=10000")
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd={start}&coed={end}"
DEFAULT_UA = "options-playground-context-fetch/1.0 (personal research; set FETCH_USER_AGENT to add contact info)"

MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_MON = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"


class FetchError(RuntimeError):
    pass


class SanityError(RuntimeError):
    pass


def month_num(word: str) -> int:
    return MONTHS[word.strip(".").lower()[:3]]


# ------------------------------------------------------------------ fetching
def http_get(url: str, ua: str | None = None, timeout: int = 60, tries: int = 4) -> str:
    ua = ua or os.environ.get("FETCH_USER_AGENT") or DEFAULT_UA
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": ua, "Accept": "text/html,text/csv,*/*"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
            if e.code in (400, 401, 403, 404, 410):
                break
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
        time.sleep(2 ** k)
    raise FetchError(redact(f"GET {url} failed: {last}"))


def slug(url: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", re.sub(r"^https?://", "", url)).strip("_")


def dir_fetcher(folder: str):
    """Fetch from saved pages: <folder>/<slug(url)>.html (or .csv). Lets you download with a browser if a site blocks scripts."""
    base = Path(folder)

    def fetch(url: str) -> str:
        for ext in (".html", ".htm", ".csv", ".txt"):
            p = base / (slug(url) + ext)
            if p.exists():
                return p.read_text(encoding="utf-8", errors="replace")
        raise FetchError(f"{url}: no saved copy {base / (slug(url) + '.html')}")
    return fetch


def polite(fetch, delay: float):
    last = [0.0]

    def f(url):
        wait = delay - (time.time() - last[0])
        if wait > 0:
            time.sleep(wait)
        try:
            return fetch(url)
        finally:
            last[0] = time.time()
    return f


# ------------------------------------------------------------------ html -> lines
def html_lines(markup: str) -> list[str]:
    s = re.sub(r"(?is)<(script|style|noscript)\b.*?</\1>", " ", markup)
    s = re.sub(r"(?i)<br\s*/?>|</?(?:p|div|tr|li|ul|ol|table|thead|tbody|section|article|header|footer|td|th|h[1-6])\b[^>]*>", "\n", s)
    s = re.sub(r"<[^>]+>", "", s)
    s = htmllib.unescape(s).replace("\xa0", " ")
    out = []
    for ln in s.splitlines():
        ln = re.sub(r"\s+", " ", ln).strip()
        if ln:
            out.append(ln)
    return out


# ------------------------------------------------------------------ FOMC
_FOMC_MONTH_LINE = re.compile(rf"^({_MON})(?:\s*/\s*({_MON}))?$", re.I)
_FOMC_DAY_LINE = re.compile(r"^(\d{1,2})(?:\s*[-–]\s*(\d{1,2}))?\s*\*?(.*)$")
_FOMC_HEAD = re.compile(r"\b(20\d\d) FOMC Meetings\b", re.I)
_FOMC_HIST = re.compile(rf"\b({_MON})\.?\s+(\d{{1,2}})(?:\s*[-–]\s*(?:({_MON})\.?\s+)?(\d{{1,2}}))?\s+Meeting\b(?:\s*[-–]\s*(20\d\d))?", re.I)


_UNSCHED = re.compile(r"unscheduled|emergency|conference call", re.I)
_NOTATION = re.compile(r"notation|\bvote\b", re.I)
_CALL_A = re.compile(rf"(?:Conference Call|Unscheduled(?: Meeting)?)\W{{0,6}}({_MON})\.?\s+(\d{{1,2}})(?:,?\s+(20\d\d))?(?!\W{{0,3}}(?:notation|vote))", re.I)
_CALL_B = re.compile(rf"\b({_MON})\.?\s+(\d{{1,2}})(?:,?\s+(20\d\d))?\s*\W{{0,3}}\s*(?:Conference Call|\(?Unscheduled)", re.I)


def parse_fomc_detail(markup: str, year_hint: int | None = None) -> dict:
    """{decision date: scheduled?} for every FOMC meeting on a Fed calendar or historical page.

    Scheduled meetings use their last day. Unscheduled / emergency meetings and conference calls (e.g. March 2020) are included with
    scheduled=False. Notation votes are not meetings and are skipped."""
    lines = html_lines(markup)
    found: dict = {}

    def add(d: dt.date, sched: bool):
        found[d] = found.get(d, False) or sched

    year = year_hint
    i = 0
    while i < len(lines):
        ln = lines[i]
        h = _FOMC_HEAD.search(ln)
        if h:
            year = int(h.group(1))
        m = _FOMC_MONTH_LINE.match(ln)
        if m and year and i + 1 < len(lines):
            d = _FOMC_DAY_LINE.match(lines[i + 1])
            if d:
                tail = d.group(3) + " " + (lines[i + 2] if i + 2 < len(lines) and not _FOMC_MONTH_LINE.match(lines[i + 2]) else "")
                if not _NOTATION.search(tail):
                    m1 = month_num(m.group(1))
                    m2 = month_num(m.group(2)) if m.group(2) else m1
                    d1, d2 = int(d.group(1)), int(d.group(2) or d.group(1))
                    mm = m2 if (m.group(2) and m2 != m1) else m1
                    if d.group(2) and d2 < d1 and not m.group(2):
                        raise SanityError(f"FOMC: day range {d1}-{d2} goes backwards under month '{ln}'")
                    try:
                        add(dt.date(year, mm, d2), not _UNSCHED.search(tail))
                    except ValueError as e:
                        raise SanityError(f"FOMC: bad date {year}-{mm}-{d2} ({e})")
                    i += 2
                    continue
        i += 1
    text = " ".join(lines)
    for hm in _FOMC_HIST.finditer(text):
        yr = int(hm.group(5)) if hm.group(5) else year_hint
        if not yr:
            continue
        before = text[max(0, hm.start() - 20):hm.start()]
        after = text[hm.end():hm.end() + 30]
        tag = r"(?:unscheduled|emergency)"          # not "conference call": that belongs to the NEXT heading
        if re.search(r"(?:notation|\bvote\b)\W{0,3}$", before, re.I) or re.match(r"\W{0,4}(?:notation|vote)", after, re.I):
            continue                                  # keywords only count when attached to THIS heading, never a neighbour's
        sched = not (re.search(rf"{tag}\W{{0,3}}$", before, re.I) or re.match(rf"\W{{0,4}}{tag}", after, re.I))
        m1 = month_num(hm.group(1))
        m2 = month_num(hm.group(3)) if hm.group(3) else m1
        day = int(hm.group(4) or hm.group(2))
        try:
            add(dt.date(yr, m2, day), sched)
        except ValueError as e:
            raise SanityError(f"FOMC: bad date {yr}-{m2}-{day} ({e})")
    for rx in (_CALL_A, _CALL_B):                    # "Conference Call - March 3, 2020" / "March 15 Conference Call"
        for cm in rx.finditer(text):
            yr = int(cm.group(3)) if cm.group(3) else year_hint
            if not yr:
                continue
            try:
                add(dt.date(yr, month_num(cm.group(1)), int(cm.group(2))), False)
            except ValueError:
                continue
    return found


def parse_fomc(markup: str, year_hint: int | None = None, unscheduled: bool = True) -> list[dt.date]:
    """Decision dates found on the page (unscheduled / emergency meetings included unless unscheduled=False)."""
    return sorted(d for d, sched in parse_fomc_detail(markup, year_hint).items() if sched or unscheduled)


# ------------------------------------------------------------------ BLS
_DATE_FULL = re.compile(rf"({_MON})\.?\s+(\d{{1,2}}),?\s+(20\d\d)", re.I)
_DATE_SHORT = re.compile(rf"\b({_MON})\.?\s+(\d{{1,2}})\b(?!,?\s*\d{{4}})", re.I)


def parse_bls_month(markup: str, year: int, month: int) -> dict[str, list[dt.date]]:
    """{'CPI': [...], 'PPI': [...], 'NFP': [...]} release dates on one BLS monthly schedule-list page."""
    lines = html_lines(markup)
    out: dict[str, set] = {k: set() for k in BLS_RELEASES}
    full = any(_DATE_FULL.search(ln) for ln in lines)
    cur = None
    for ln in lines:
        dm = _DATE_FULL.search(ln) if full else _DATE_SHORT.search(ln)
        if dm:
            try:
                cur = dt.date(int(dm.group(3)) if full else year, month_num(dm.group(1)), int(dm.group(2)))
            except ValueError:
                cur = None
        if cur is None:
            continue
        for key, name in BLS_RELEASES.items():
            if re.search(rf"\b{name}\b", ln, re.I) and not re.search(r"state|county|metropolitan|cost index|import|export|real earnings", ln, re.I):
                if (cur.year, cur.month) == (year, month):
                    out[key].add(cur)
    return {k: sorted(v) for k, v in out.items()}


def election_days(y0: int, y1: int) -> list[dt.date]:
    """Federal general-election day: Tuesday after the first Monday in November of every even year (2 U.S.C. 7)."""
    out = []
    for y in range(y0, y1 + 1):
        if y % 2:
            continue
        d = dt.date(y, 11, 1)
        d += dt.timedelta(days=(0 - d.weekday()) % 7)       # first Monday
        out.append(d + dt.timedelta(days=1))
    return out


# ------------------------------------------------------------------ BLS archive / schedule / FRED release dates
_ARCH_HREF = re.compile(r"(cpi|ppi|empsit)_(\d{2})(\d{2})(\d{4})\.(?:htm|html|pdf|txt)", re.I)
_REF_REL = re.compile(rf"\b({_MON})\.?\s+(20\d\d)\s+({_MON})\.?\s+(\d{{1,2}}),?\s+(20\d\d)", re.I)


def parse_bls_archive(markup: str, slug_: str) -> list[dt.date]:
    """Release dates from the archived-news-release page: the links embed the date as <series>_MMDDYYYY.htm (read from the raw HTML)."""
    out = set()
    for m in _ARCH_HREF.finditer(markup):
        if m.group(1).lower() != slug_:
            continue
        try:
            out.add(dt.date(int(m.group(4)), int(m.group(2)), int(m.group(3))))
        except ValueError:
            continue
    return sorted(out)


def parse_bls_schedule(markup: str) -> list[dt.date]:
    """Release dates from a schedule page whose rows read '<reference month> <release date> <time>' (e.g. 'Oct 2024 Nov. 01, 2024 08:30 AM')."""
    text = " ".join(html_lines(markup))
    out = set()
    for m in _REF_REL.finditer(text):
        ry, rm = int(m.group(2)), month_num(m.group(1))
        try:
            rel = dt.date(int(m.group(5)), month_num(m.group(3)), int(m.group(4)))
        except ValueError:
            continue
        lag = (rel.year - ry) * 12 + rel.month - rm
        if 0 <= lag <= 3:                       # a release comes 0-3 months after its reference month; anything else is a false match
            out.add(rel)
    return sorted(out)


def parse_fred_release_dates(text: str) -> list[dt.date]:
    import json
    try:
        data = json.loads(text)
        return sorted({dt.date.fromisoformat(r["date"]) for r in data["release_dates"]})
    except (ValueError, KeyError, TypeError) as e:
        raise SanityError(f"FRED release/dates: unexpected response ({type(e).__name__}); the key may be invalid")


def redact(msg: str) -> str:
    return re.sub(r"(api_key=)[^&\s]+", r"\1***", str(msg))


# ------------------------------------------------------------------ FRED
def parse_fred(text: str, series: str) -> list[tuple[dt.date, float]]:
    rows = list(csv.reader(text.strip().splitlines()))
    if not rows or len(rows[0]) < 2:
        raise SanityError("FRED: empty or malformed CSV (did the id or the network policy change?)")
    head = [h.strip().lower() for h in rows[0]]
    if head[0] not in ("date", "observation_date"):
        raise SanityError(f"FRED: unexpected header {rows[0]}; expected observation_date,{series}")
    if series.lower() not in head[1]:
        raise SanityError(f"FRED: expected column {series}, got {rows[0][1]}")
    out = []
    for r in rows[1:]:
        if len(r) < 2 or r[1].strip() in ("", "."):        # FRED marks missing values with "."
            continue
        try:
            out.append((dt.date.fromisoformat(r[0].strip()), float(r[1])))
        except ValueError as e:
            raise SanityError(f"FRED: unparseable row {r}: {e}")
    return out


# ------------------------------------------------------------------ sanity checks
def parse_allow(items: list[str]) -> dict:
    """['2025:CPI=11'] -> {(2025, 'CPI'): 11}"""
    out = {}
    for it in items or []:
        m = re.fullmatch(r"(\d{4}):([A-Za-z]+)=(\d+)", it.strip())
        if not m:
            raise SystemExit(f"--allow expects YEAR:EVENT=COUNT (e.g. 2025:CPI=11), got {it!r}")
        out[(int(m.group(1)), m.group(2).upper())] = int(m.group(3))
    return out


def check_events(rows: list[tuple], y0: int, y1: int, allow: dict | None = None, require_sorted: bool = True) -> dict:
    """rows: (date, event, symbol). Raises SanityError listing EVERY problem. Returns {(year, event): count}."""
    allow = allow or {}
    bad: list[str] = []
    for r in rows:
        if not isinstance(r[0], dt.date):
            bad.append(f"date does not parse: {r!r}")
        if r[1] not in KNOWN_TYPES:
            bad.append(f"unknown event type {r[1]!r} on {r[0]}")
    if bad:
        raise SanityError("; ".join(bad[:20]))
    keys = [(r[0], r[1], r[2] if r[1] == "EARNINGS" else "") for r in rows]
    dups = [k for k, n in Counter(keys).items() if n > 1]
    if dups:
        bad.append("duplicate rows: " + ", ".join(f"{d} {e} {s}".strip() for d, e, s in sorted(dups)[:10]))
    if require_sorted and [(r[0], r[1]) for r in rows] != sorted((r[0], r[1]) for r in rows):
        bad.append("rows are not sorted by (date, event)")
    for d, e, *_ in rows:
        if e in WEEKDAY_ONLY and d.weekday() >= 5:
            bad.append(f"{e} on a weekend ({d}): parse error?")
        if e == "ELECTION" and d.weekday() != 1:
            bad.append(f"ELECTION {d} is not a Tuesday")
    counts: dict = defaultdict(int)
    for d, e, *_ in rows:
        counts[(d.year, e)] += 1
    for y in range(y0, y1 + 1):
        n = counts.get((y, "FOMC"), 0)
        lo, hi = FOMC_PER_YEAR if (y, "FOMC") not in allow else (allow[(y, "FOMC")],) * 2
        hi += 0 if (y, "FOMC") in allow else 3                     # --check cannot tell scheduled from emergency rows: allow a few extras
        if not lo <= n <= hi:
            bad.append(f"{y}: {n} FOMC meetings, expected {lo}-{hi}: " + ", ".join(str(d) for d, e, *_ in rows if e == "FOMC" and d.year == y))
        for e in ("CPI", "PPI", "NFP"):
            n = counts.get((y, e), 0)
            want = allow.get((y, e), MONTHLY_PER_YEAR)
            if n != want:
                bad.append(f"{y}: {n} {e} releases, expected {want}" + (" (allowed override)" if (y, e) in allow else
                           f" (genuine schedule disruptions such as the 2025 shutdown need an explicit --allow {y}:{e}={n}): "
                           + ", ".join(str(d) for d, ee, *_ in rows if ee == e and d.year == y)))
        if y % 2 == 0 and counts.get((y, "ELECTION"), 0) != 1:
            bad.append(f"{y}: expected exactly 1 ELECTION day")
    if bad:
        raise SanityError("events.csv sanity check FAILED:\n  - " + "\n  - ".join(bad))
    return dict(counts)


def check_rates(rows: list[tuple], y0: int, today: dt.date, max_gap_days: int = 7, stale_days: int = 14) -> None:
    bad = []
    if not rows:
        raise SanityError("rates.csv sanity check FAILED: no rows")
    ds = [r[0] for r in rows]
    if len(set(ds)) != len(ds):
        bad.append("duplicate dates")
    if ds != sorted(ds):
        bad.append("dates are not sorted")
    for d, v in rows:
        if not 0.0 <= v <= 25.0:
            bad.append(f"rate {v} on {d} outside 0-25 percent (units wrong? expected percent)")
            break
    if ds[0] > dt.date(y0, 1, 1) + dt.timedelta(days=max_gap_days):
        bad.append(f"first observation {ds[0]} is after {y0}-01-0{max_gap_days}")
    if (today - ds[-1]).days > stale_days:
        bad.append(f"last observation {ds[-1]} is more than {stale_days} days old")
    gaps = [(a, b) for a, b in zip(ds, ds[1:]) if (b - a).days > max_gap_days]
    if gaps:
        bad.append(f"gaps longer than {max_gap_days} days, e.g. {gaps[0][0]} -> {gaps[0][1]}")
    if bad:
        raise SanityError("rates.csv sanity check FAILED:\n  - " + "\n  - ".join(bad))


# ------------------------------------------------------------------ csv io
def read_events_csv(path: Path) -> list[tuple]:
    rows = []
    with open(path, newline="") as f:
        lines = [ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    rd = csv.DictReader(lines)
    if not rd.fieldnames or not {"date", "event"} <= set(rd.fieldnames):
        raise SanityError(f"{path}: needs columns date,event[,symbol]")
    for r in rd:
        try:
            d = dt.date.fromisoformat(r["date"].strip())
        except ValueError:
            raise SanityError(f"{path}: date does not parse: {r['date']!r}")
        rows.append((d, r["event"].strip().upper(), (r.get("symbol") or "").strip()))
    return rows


def read_rates_csv(path: Path) -> list[tuple]:
    rows = []
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        if rd.fieldnames != ["date", "rate_pct"]:
            raise SanityError(f"{path}: header must be date,rate_pct, got {rd.fieldnames}")
        for r in rd:
            try:
                rows.append((dt.date.fromisoformat(r["date"]), float(r["rate_pct"])))
            except ValueError:
                raise SanityError(f"{path}: unparseable row {r}")
    return rows


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def events_text(rows: list[tuple], stamp: str) -> str:
    out = [f"# Generated {stamp} by scripts/fetch_context_data.py from federalreserve.gov (FOMC), bls.gov (CPI/PPI/NFP) and the",
           "# statutory election-day rule. EARNINGS rows (if any) are kept from the previous file. Never edit dates by guesswork.",
           "date,event,symbol"]
    out += [f"{d.isoformat()},{e},{s}" for d, e, s in rows]
    return "\n".join(out) + "\n"


def rates_text(rows: list[tuple]) -> str:
    return "date,rate_pct\n" + "".join(f"{d.isoformat()},{v:g}\n" for d, v in rows)


# ------------------------------------------------------------------ orchestration (never aborts: per source, per year)
def try_fetch(fetch, url: str, failures: list):
    try:
        return fetch(url)
    except FetchError as e:
        failures.append(redact(str(e)))
        return None


def _by_year(dates) -> dict:
    out: dict = defaultdict(set)
    for d in dates:
        out[d.year].add(d)
    return out


def gather_events(fetch, y0: int, y1: int, allow: dict, fred_key: str | None, failures: list, log=print, unsched: set | None = None) -> dict:
    """-> {(event, year): (sorted dates, source label)}. Each (event, year) takes the first source with the expected count, else the fullest one."""
    chosen: dict = {}
    unsched = unsched if unsched is not None else set()
    years = range(y0, y1 + 1)
    # ---- FOMC: calendars page, plus fomchistorical<year> for 2016-2020 (scheduled and emergency meetings merged)
    cal = try_fetch(fetch, FOMC_CAL_URL, failures)
    det = parse_fomc_detail(cal) if cal else {}
    for y in years:
        cur = {d: sc for d, sc in det.items() if d.year == y}
        src = "fomccalendars.htm"
        if y <= FOMC_HIST_LAST:                      # historical pages for 2016-2020 are always requested and merged with the calendars page
            page = try_fetch(fetch, FOMC_HIST_URL.format(year=y), failures)
            if page:
                more = {d: sc for d, sc in parse_fomc_detail(page, y).items() if d.year == y}
                if more:
                    src = f"fomchistorical{y}.htm" + (" + fomccalendars.htm" if cur else "")
                    cur = {d: more.get(d, False) or cur.get(d, False) for d in {*cur, *more}}
        if cur:
            chosen[("FOMC", y)] = (sorted(cur), src)
            unsched.update(d for d, sc in cur.items() if not sc)
    # ---- CPI / PPI / NFP
    pages: dict = {}

    def page_dates(kind, event):
        key = (kind, event)
        if key not in pages:
            slug_ = BLS_SLUG[event]
            url = (BLS_ARCHIVE_URL if kind == "archive" else BLS_SCHED_URL).format(slug=slug_)
            html = try_fetch(fetch, url, failures)
            pages[key] = _by_year((parse_bls_archive(html, slug_) if kind == "archive" else parse_bls_schedule(html)) if html else [])
        return pages[key]

    month_cache: dict = {}

    def month_dates(event, y):
        if y not in month_cache:
            acc: dict = {e: set() for e in BLS_SLUG}
            for m in range(1, 13):
                html = try_fetch(fetch, BLS_MONTH_URL.format(year=y, month=m), failures)
                if html:
                    for e, ds in parse_bls_month(html, y, m).items():
                        acc[e] |= set(ds)
            month_cache[y] = acc
        return month_cache[y][event]

    fred_cache: dict = {}

    def fred_dates(event):
        if event not in fred_cache:
            html = try_fetch(fetch, FRED_REL_URL.format(rid=FRED_RELEASE_IDS[event], key=fred_key), failures)
            try:
                fred_cache[event] = _by_year(parse_fred_release_dates(html)) if html else {}
            except SanityError as e:
                failures.append(str(e))
                fred_cache[event] = {}
        return fred_cache[event]

    for event in BLS_SLUG:
        for y in years:
            want = allow.get((y, event), MONTHLY_PER_YEAR)
            best: tuple = ([], "")
            sources = [("bls archive", lambda: page_dates("archive", event).get(y, set())),
                       ("bls schedule page", lambda: page_dates("sched", event).get(y, set())),
                       ("bls monthly lists", lambda: month_dates(event, y))]
            if fred_key:
                sources.append(("FRED release dates (API key)", lambda: fred_dates(event).get(y, set())))
            for name, getter in sources:
                ds = sorted(d for d in getter() if d.weekday() < 5)
                if len(ds) == want:
                    best = (ds, name)
                    break
                if len(ds) > len(best[0]):
                    best = (ds, name)
            if best[0]:
                chosen[(event, y)] = best
    for d in election_days(y0, y1):
        chosen[("ELECTION", d.year)] = ([d], "statutory rule (2 U.S.C. 7)")
    return chosen


def coverage(rows: list, y0: int, y1: int, allow: dict, unsched: set | None = None) -> tuple:
    """-> (counts, warnings). Warnings are anomalies; absence is shown as MISSING in the table."""
    counts: dict = defaultdict(int)
    for d, e, *_ in rows:
        counts[(d.year, e)] += 1
    warns = []
    extra: dict = defaultdict(int)
    for d in (unsched or ()):
        extra[d.year] += 1
    for y in range(y0, y1 + 1):
        n = counts.get((y, "FOMC"), 0) - extra.get(y, 0)           # the ~8 check counts SCHEDULED meetings; emergency ones are extra rows
        if n and not FOMC_PER_YEAR[0] <= n <= FOMC_PER_YEAR[1] and (y, "FOMC") not in allow:
            warns.append(f"{y}: {n} scheduled FOMC meetings (expected {FOMC_PER_YEAR[0]}-{FOMC_PER_YEAR[1]})")
        for e in ("CPI", "PPI", "NFP"):
            n = counts.get((y, e), 0)
            want = allow.get((y, e), MONTHLY_PER_YEAR)
            if n and n != want:
                warns.append(f"{y}: {n} {e} releases (expected {want}); if this is a genuine disruption re-run with --allow {y}:{e}={n}")
    return dict(counts), warns


def table_text(counts: dict, y0: int, y1: int, allow: dict, unsched: set | None = None) -> str:
    ys = list(range(y0, y1 + 1))
    L = ["event      " + "".join(f"{y:>9}" for y in ys)]
    for e in EVENT_TYPES:
        cells = []
        for y in ys:
            n = counts.get((y, e), 0)
            ex = sum(1 for d in (unsched or ()) if d.year == y) if e == "FOMC" else 0
            n -= ex
            exp = (FOMC_PER_YEAR if e == "FOMC" else (1, 1) if e == "ELECTION" else (allow.get((y, e), MONTHLY_PER_YEAR),) * 2)
            if e == "ELECTION" and y % 2:
                cells.append("-")
            elif n + ex == 0:
                cells.append("MISSING")
            else:
                cells.append(f"{n}" + (f"+{ex}" if ex else "") + ("" if exp[0] <= n <= exp[1] else "!"))
        L.append(f"{e:<10} " + "".join(f"{c:>9}" for c in cells))
    return "\n".join(L)


def source_text(chosen: dict, y0: int, y1: int) -> str:
    L = []
    for e in EVENT_TYPES:
        runs: list = []
        for y in range(y0, y1 + 1):
            if (e, y) not in chosen:
                if e == "ELECTION" and y % 2:
                    continue
                s = "MISSING"
            else:
                s = chosen[(e, y)][1]
            if runs and runs[-1][2] == s and runs[-1][1] == y - 1:
                runs[-1][1] = y
            else:
                runs.append([y, y, s])
        L.append(f"  {e:<9}" + "; ".join(f"{a if a == b else f'{a}-{b}'}: {s}" for a, b, s in runs))
    return "\n".join(L)


def run(fetch, out_dir: Path, y0: int, y1: int, series: str, allow: dict, today: dt.date, log=print, fred_key: str | None = None, strict: bool = False) -> dict:
    """Fetch what can be fetched, write what succeeded, print the coverage table. Returns {'exit': code, ...}."""
    out_dir = Path(out_dir)
    ev_path, rt_path = out_dir / "events.csv", out_dir / "rates.csv"
    failures: list = []
    log(f"fetching FOMC / BLS calendars {y0}-{y1}" + (" (FRED release dates enabled as fallback)" if fred_key else " (no FRED_API_KEY: FRED release-date fallback off)") + " ...")
    unsched: set = set()
    chosen = gather_events(fetch, y0, y1, allow, fred_key, failures, log, unsched)
    rows = {(d, e, "") for (e, y), (ds, _) in chosen.items() for d in ds}
    old = read_events_csv(ev_path) if ev_path.exists() else []
    have = {(e, y) for (e, y) in chosen}
    kept = 0
    for d, e, sym in old:
        if e == "EARNINGS":
            rows.add((d, e, sym))
        elif e in EVENT_TYPES and (e, d.year) not in have and y0 <= d.year <= y1:
            rows.add((d, e, sym))                               # an earlier good file beats a hole in this run
            chosen.setdefault((e, d.year), ([], "existing events.csv"))
            chosen[(e, d.year)][0].append(d)
            kept += 1
    rows = sorted(rows, key=lambda r: (r[0], r[1], r[2]))
    counts, warns = coverage(rows, y0, y1, allow, unsched)
    if rows:
        write_atomic(ev_path, events_text(rows, today.isoformat()))
        log(f"wrote {ev_path} ({len(rows)} rows" + (f", {kept} kept from the previous file for holes in this run" if kept else "") + ")")
    else:
        log(f"nothing to write to {ev_path}")
    # ---- rates
    rate_msg = ""
    text = try_fetch(fetch, FRED_URL.format(series=series, start=f"{y0}-01-01", end=f"{y1}-12-31"), failures)
    rwarn: list = []
    if text is None:
        rate_msg = f"rates.csv NOT written (FRED {series} fetch failed); the report will use the flat --apy and say so"
    else:
        try:
            rates = [r for r in parse_fred(text, series) if y0 <= r[0].year <= y1]
            if not rates or any(not 0.0 <= v <= 25.0 for _, v in rates):
                raise SanityError("no rows, or values outside 0-25 percent")
            try:
                check_rates(rates, y0, today)
            except SanityError as e:
                rwarn.append(str(e).replace("\n", " "))      # gaps / stale / late start: still written, but loudly flagged
            write_atomic(rt_path, rates_text(rates))
            rate_msg = f"wrote {rt_path} ({len(rates)} rows, {rates[0][0]}..{rates[-1][0]})"
        except SanityError as e:
            rate_msg = f"rates.csv NOT written ({e})"
    log(rate_msg)
    log("\nCOVERAGE (count per event type and year; ! = unexpected count, MISSING = nothing found):\n" + table_text(counts, y0, y1, allow, unsched))
    log("\nSOURCE USED PER EVENT AND YEARS:\n" + source_text(chosen, y0, y1))
    for w in warns + rwarn:
        log(f"WARNING: {w}")
    if failures:
        log(f"\n{len(failures)} fetch failure(s) (non-fatal):")
        for f in sorted(set(failures))[:40]:
            log(f"  - {f}")
        log("  Save any of those pages in a browser and re-run with --html-dir (file names = the URL with non-alphanumerics replaced by '_', plus .html).")
    need = [(e, y) for y in range(y0, y1 + 1) for e in ("FOMC", "CPI", "NFP") if counts.get((y, e), 0) == 0]
    code = 0 if not need else 4
    if need:
        log("\nINCOMPLETE: missing " + ", ".join(f"{e} {y}" for e, y in need[:20]) + (" ..." if len(need) > 20 else "") + " (partial files were still written)")
    elif strict and (warns or rwarn):
        code = 2
    log(f"exit code {code}")
    return {"exit": code, "counts": counts, "warnings": warns + rwarn, "failures": failures, "chosen": chosen, "events": len(rows)}


def check_existing(out_dir: Path, y0: int, y1: int, allow: dict, today: dt.date, log=print) -> None:
    ev = read_events_csv(Path(out_dir) / "events.csv")
    counts = check_events(ev, y0, y1, allow)
    check_rates(read_rates_csv(Path(out_dir) / "rates.csv"), y0, today)
    log(f"OK: {len(ev)} events and the rates file pass all checks")
    return counts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data", help="output folder (default data/)")
    ap.add_argument("--start-year", type=int, default=2016)
    ap.add_argument("--end-year", type=int, default=2026)
    ap.add_argument("--rate-series", default="DFF", choices=["DFF", "EFFR"], help="FRED series for the daily effective fed funds rate")
    ap.add_argument("--allow", action="append", default=[], metavar="YEAR:EVENT=N",
                    help="accept N releases for a year/event (e.g. 2025:CPI=11 after the 2025 shutdown); repeatable. Printed, never silent.")
    ap.add_argument("--html-dir", default=None, help="parse saved pages from this folder instead of fetching (files named by URL slug)")
    ap.add_argument("--delay", type=float, default=0.7, help="seconds between requests (be polite to bls.gov)")
    ap.add_argument("--strict", action="store_true", help="exit 2 when any count anomaly is flagged (files are still written)")
    ap.add_argument("--check", action="store_true", help="only re-validate the existing data/events.csv and data/rates.csv (no network)")
    a = ap.parse_args(argv)
    allow = parse_allow(a.allow)
    today = dt.date.today()
    try:
        if a.check:
            check_existing(Path(a.out), a.start_year, a.end_year, allow, today)
            return 0
        fetch = dir_fetcher(a.html_dir) if a.html_dir else polite(http_get, a.delay)
        key = os.environ.get("FRED_API_KEY") or None            # from your shell / .env only; never printed (URLs are redacted)
        return run(fetch, Path(a.out), a.start_year, a.end_year, a.rate_series, allow, today, fred_key=key, strict=a.strict)["exit"]
    except SanityError as e:
        print(f"SANITY CHECK FAILED.\n{e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
