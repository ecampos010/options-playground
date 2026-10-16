#!/usr/bin/env python3
"""Download the market-context calendars and the risk-free rate from official public pages.

    python scripts/fetch_context_data.py                      # 2016-2026 -> data/events.csv + data/rates.csv
    python scripts/fetch_context_data.py --check              # re-validate the files already in data/ (no network)
    python scripts/fetch_context_data.py --html-dir saved/    # parse pre-downloaded pages instead of fetching

Sources (all public, no API key):
  FOMC        federalreserve.gov/monetarypolicy/fomccalendars.htm + fomchistorical<year>.htm   (decision day = last day of
              each SCHEDULED meeting; unscheduled calls and notation votes are not events)
  CPI/PPI/NFP bls.gov/schedule/<year>/<MM>_sched_list.htm   ("Consumer Price Index", "Producer Price Index",
              "Employment Situation" rows; NFP = the Employment Situation release)
  ELECTION    federal general-election day, computed from the statute (Tuesday after the first Monday of November,
              even years); nothing is scraped for it
  rates       fred.stlouisfed.org/graph/fredgraph.csv?id=DFF (or EFFR): the daily effective federal funds rate, percent

Dates are NEVER invented: every FOMC/CPI/PPI/NFP date comes from a downloaded page, and the run FAILS LOUDLY (exit
code 2, nothing written) if a sanity check fails. Parsers were written from the public page layouts and are tested
against saved fixtures only; if a page layout changed, the error names the URL - save that page and use --html-dir.

Output: data/events.csv (date,event,symbol; EARNINGS rows already in the file are kept) and data/rates.csv
(date,rate_pct). Needs internet; standard library only. Keys never needed (a .env is not read).
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
WEEKDAY_ONLY = ("FOMC", "CPI", "PPI", "NFP")
FOMC_PER_YEAR = (7, 9)                  # "about 8": the scheduled calendar has 8 every year in 2016-2026
MONTHLY_PER_YEAR = 12                   # CPI, PPI, NFP: exactly 12 unless explicitly allowed (e.g. 2025 shutdown)
BLS_RELEASES = {"CPI": "Consumer Price Index", "PPI": "Producer Price Index", "NFP": "Employment Situation"}
FOMC_CAL_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
FOMC_HIST_URL = "https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm"
BLS_MONTH_URL = "https://www.bls.gov/schedule/{year}/{month:02d}_sched_list.htm"
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
    raise FetchError(f"GET {url} failed: {last}")


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


def parse_fomc(markup: str, year_hint: int | None = None) -> list[dt.date]:
    """Decision dates (last day) of SCHEDULED meetings found in a Fed calendar or historical page."""
    lines = html_lines(markup)
    found: set[dt.date] = set()
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
                tail = (d.group(3) + " " + (lines[i + 2] if i + 2 < len(lines) else "")).lower()
                if not re.search(r"unscheduled|notation|conference call|\bvote\b", tail):
                    m1 = month_num(m.group(1))
                    m2 = month_num(m.group(2)) if m.group(2) else m1
                    d1, d2 = int(d.group(1)), int(d.group(2) or d.group(1))
                    mm = m2 if (m.group(2) and m2 != m1) else m1
                    if d.group(2) and d2 < d1 and not m.group(2):
                        raise SanityError(f"FOMC: day range {d1}-{d2} goes backwards under month '{ln}'")
                    try:
                        found.add(dt.date(year, mm, d2))
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
        if re.search(r"unscheduled", text[max(0, hm.start() - 40):hm.end() + 40], re.I):
            continue
        m1 = month_num(hm.group(1))
        m2 = month_num(hm.group(3)) if hm.group(3) else m1
        day = int(hm.group(4) or hm.group(2))
        try:
            found.add(dt.date(yr, m2, day))
        except ValueError as e:
            raise SanityError(f"FOMC: bad date {yr}-{m2}-{day} ({e})")
    return sorted(found)


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


# ------------------------------------------------------------------ orchestration
def collect_events(fetch, y0: int, y1: int, log=print) -> list[tuple]:
    rows: set[tuple] = set()
    fomc: set[dt.date] = set()
    fomc |= set(parse_fomc(fetch(FOMC_CAL_URL)))
    for y in range(y0, y1 + 1):
        url = FOMC_HIST_URL.format(year=y)
        try:
            page = fetch(url)
        except FetchError as e:
            log(f"  note: {e} (historical page optional; the calendar page may cover {y})")
            continue
        fomc |= set(parse_fomc(page, y))
    for d in fomc:
        if y0 <= d.year <= y1:
            rows.add((d, "FOMC", ""))
    for y in range(y0, y1 + 1):
        for mth in range(1, 13):
            url = BLS_MONTH_URL.format(year=y, month=mth)
            page = fetch(url)
            got = parse_bls_month(page, y, mth)
            if not any(got.values()):
                raise SanityError(f"no CPI/PPI/Employment Situation rows found on {url}: the page layout may have changed. "
                                  f"Save the page and re-run with --html-dir (file name {slug(url)}.html).")
            for key, ds in got.items():
                for d in ds:
                    rows.add((d, key, ""))
    for d in election_days(y0, y1):
        rows.add((d, "ELECTION", ""))
    return sorted(rows, key=lambda r: (r[0], r[1]))


def run(fetch, out_dir: Path, y0: int, y1: int, series: str, allow: dict, today: dt.date, log=print) -> dict:
    out_dir = Path(out_dir)
    ev_path, rt_path = out_dir / "events.csv", out_dir / "rates.csv"
    log(f"fetching FOMC / BLS calendars {y0}-{y1} ...")
    rows = collect_events(fetch, y0, y1, log)
    if ev_path.exists():                          # keep per-symbol earnings that the user entered by hand
        keep = [r for r in read_events_csv(ev_path) if r[1] == "EARNINGS"]
        rows = sorted(set(rows) | set(keep), key=lambda r: (r[0], r[1], r[2]))
    counts = check_events(rows, y0, y1, allow)
    log(f"fetching FRED {series} ...")
    text = fetch(FRED_URL.format(series=series, start=f"{y0}-01-01", end=f"{y1}-12-31"))
    rates = parse_fred(text, series)
    rates = [r for r in rates if y0 <= r[0].year <= y1]
    check_rates(rates, y0, today)
    stamp = today.isoformat()
    write_atomic(ev_path, events_text(rows, stamp))      # both files only after BOTH checks passed
    write_atomic(rt_path, rates_text(rates))
    log(f"wrote {ev_path} ({len(rows)} rows) and {rt_path} ({len(rates)} rows, {rates[0][0]}..{rates[-1][0]})")
    log("per-year counts (FOMC/CPI/PPI/NFP/ELECTION): " + "; ".join(
        f"{y}: " + "/".join(str(counts.get((y, e), 0)) for e in EVENT_TYPES) for y in range(y0, y1 + 1)))
    return {"events": len(rows), "rates": len(rates), "counts": counts}


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
    ap.add_argument("--check", action="store_true", help="only re-validate the existing data/events.csv and data/rates.csv (no network)")
    a = ap.parse_args(argv)
    allow = parse_allow(a.allow)
    today = dt.date.today()
    try:
        if a.check:
            check_existing(Path(a.out), a.start_year, a.end_year, allow, today)
            return 0
        fetch = dir_fetcher(a.html_dir) if a.html_dir else polite(http_get, a.delay)
        run(fetch, Path(a.out), a.start_year, a.end_year, a.rate_series, allow, today)
        return 0
    except SanityError as e:
        print(f"SANITY CHECK FAILED - nothing written.\n{e}", file=sys.stderr)
        return 2
    except FetchError as e:
        print(f"FETCH FAILED - nothing written.\n{e}\nIf the site blocks scripts, save the pages in a browser and use --html-dir.", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
