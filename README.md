# options-playground

ThetaData-backed backtester for the **Green** SPY iron-condor rules, with a probe, a tight two-stage
downloader, a plan (dry-run) command, a rule matrix and the traffic-light **ladder**
(Green / Yellow E1-E7 / Red). Everything runs offline against a built-in **synthetic** provider.

> **Status: the ThetaData adapter is UNVERIFIED.** It was written without network access to ThetaData's
> docs. Run `probe` on the VPS first and paste back any FAIL lines. Synthetic results say nothing about
> real markets (the synthetic world is only there to exercise the code).

License: placeholder - the owner will choose a license later. No third-party code is copied in.

## VPS quickstart (Linux, Python 3.10+)

```bash
git clone <repo> && cd options-playground
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # optional: pip install yfinance (SPY daily, tried first)
cp .env.example .env                       # placeholders only; .env is git-ignored. Never commit credentials.
set -a; source .env; set +a                # THETA_USER / THETA_PASS (+ THETA_BASE, default http://127.0.0.1:25503)
# start Theta Terminal (v3) with those credentials, see ThetaData docs, e.g. java -jar ThetaTerminal.jar
python -m pytest -q                        # sanity (offline)

python -m playground probe                 # -> results/probe_report.md (+ .json). Paste FAIL lines back.
python -m playground download --daily-only # SPY daily (yfinance, else Theta) + VIX daily (Cboe CSV) -> data/
cp templates/events.csv.template data/events.csv   # then fill REAL dates (see below)
python -m playground plan --start-date 2016-10-01 --end-date 2026-09-30 --resolution 5m
python -m playground download --start-date 2016-10-01 --end-date 2026-09-30 --resolution 5m \
        --threads 2 --max-rps 5 --cover-matrix       # resumable: stop (Ctrl-C) and re-run any time
python -m playground backtest --profile green
python -m playground matrix                # 4 guard sets x 2 exits (8 runs), compact table
python -m playground matrix --fill-compare # natural vs mid side by side
python -m playground ladder --split-sample
```

Offline demo (no ThetaData needed): add `--provider synthetic` to `backtest`, `matrix`, `ladder`, `plan`
(`probe --fake` / `download --fake` use an offline *simulator of the assumed API shapes*, plumbing test only).

## Commands

| command | purpose |
|---|---|
| `probe` | (a) expired SPY contracts listable ~1/3/5/8/10y ago, (b) bid/ask+sizes at 1m and finer, (c) size/latency of a ~40-strike day at 1m (+ bulk `strike=*` test), (d) request-rate burst / 429 behaviour, (e) SPX/VIX index history and VIX options. Fails soft: prints endpoint, params, status, raw error. |
| `plan` | candidate days, expected trades, requests/rows/GB at 1m/5m/15m/eod, wall-clock at 2 and 4 threads (latency from the last probe), plus the naive full-chain cost. Writes nothing to `data/`. |
| `download` | daily data, then Stage A (narrow entry snapshot: 3 expiries nearest the target DTE, +/-10 listed strikes around the rule strikes, at the entry bar) for every candidate day, then Stage B (the engine runs in fetch mode and pulls full-life quotes for only the 4 chosen legs, in `--chunk-days` pieces). Parquet in `data/options/res=<r>/date=<d>/expiry=<e>/right=<P|C>/`; coverage is recorded so re-runs fetch nothing. `--max-rps`, exponential backoff on 429/5xx, `--entry-fetch bulk` (one `strike=*` request per expiry; only if probe (c) shows it works), prints a manifest. `--cover-matrix` makes the cache cover all 8 matrix variants. |
| `backtest` | one run; prints <=40 lines; writes `results/<run_id>/{trades.csv,equity.csv,config.json,summary.md}` |
| `matrix` | standard 8-run grid (+ `--fill-compare`) |
| `ladder` | green + yellow_E1..E7 + red on the same data/seed/fill; `--split-sample`, `--equal-budget` (give everyone Green's budget to isolate the guardrail effect) |
| `overlay`, `spike-study` | **reserved stubs** (see `playground/reserve.py`; the engine's Position = signed legs + `kind`, so a hedge leg can be added without a rewrite) |

Every rule is a config default overridable by CLI flag: `--gate on|off --upside-guard on|off --event-buffer on|off
--cb 0|21 --tp --stop-mult --time-exit --fill natural|mid|haircut --haircut --entry-window 09:45-15:30
--entry-sample first|random|every-30m --seed --start 100000 --apy --budget-pct --full-use --strike-mode vix|delta
--short-delta --dte 30,35,45 --min-credit --expiry-cap-frac --resolution --profile ...` (`backtest --help`).

## events.csv

`data/events.csv`, columns `date,event[,symbol]`; event in `FOMC, CPI, PPI, NFP, ELECTION, EARNINGS`
(`symbol` only matters for EARNINGS). Lines starting with `#` are ignored. See `templates/events.csv.template`.
**Fill it from official calendars; this repo never invents dates.** A missing file is a clear error (use
`--event-buffer off` to run without it). With `--provider synthetic` and no file, a clearly-labelled SYNTHETIC
event calendar is used.

## Rule interpretations I had to choose (check these)

* Entry cadence is not in the brief: the engine tries one entry per trading day (`--entry-every N`) and fills
  whatever budget headroom exists; per-expiry cap and budget limit it. Equity = cash + mid-marked open positions.
* Gate/guards use **prior-day VIX close**, SPY spot at the entry bar vs prior close (day change) and vs close 5
  trading days earlier (5-day change). No look-ahead.
* "Within 2 trading days before" an event blocks offsets 0,1,2 (event day included). E1 allows day-of entries after
  the release time; E2 blocks FOMC day-of only.
* Stop uses the NATURAL buyback; take-profit and realised exit price use the selected fill model. Exit order per bar:
  stop > breach > take-profit > time. Time exit only applies if entry DTE > `--time-exit` (so E3 has none).
* Circuit breaker: if a calendar month ends below its start equity, no entries for 21 trading days.
* Interest: simple daily accrual on the whole cash balance (`apy/365` per calendar day). Commission default 0.
* Red = gate switched off + `entry_requires_gate_trip` + one open lot + 1% budget. Optional per-flag ignores:
  `--red-ignore event,expiry-cap,time-exit,stop`.
* Yellow profiles use the Yellow budget (5%); Green 15%; Red 1% (profile `tier`). Compare guardrail effects
  cleanly with `ladder --equal-budget`. E5 (earnings) and E6 (concentration) are implemented but N/A for SPY-only.
* E3 (7-14 DTE) with the unchanged VIX-distance strike rule and 0.50 minimum credit may produce almost no trades.
* Stage A is intentionally per-strike by default (slow but least assumption-heavy); bulk is opt-in.

## Verified vs unverified

Verified (offline, `pytest`): sizing (floor vs `--full-use`), gate/guards/event/circuit breaker, fill sign
conventions, TP/stop/breach/time/expiry on scripted markets, interest accrual, 15%/60% caps, determinism,
ladder profiles differ from Green by exactly one switch, cache coverage/idempotent re-runs, probe/plan plumbing,
backoff - all against synthetic data or an offline simulator of the *assumed* API.

**Unverified until `probe` runs on the VPS:** every ThetaData endpoint path, parameter name and response column
(`playground/theta_client.py`, marked `TODO-VERIFY`: v3 paths, port 25503, `format=csv`, `strike=*`/`right=both`,
`start_time/end_time`, status 472 for "no data", interval strings, quote timestamp semantics, expired-contract
listing, VIX/SPX index endpoints, Cboe CSV URL, rate limits and subscription tier). Also unverified: real-market
behaviour of any strategy result.
