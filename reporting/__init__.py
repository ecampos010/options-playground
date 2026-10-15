"""Reporting, portfolio replay and stress-lab layer for the options playground.

Consumes standardized trade logs (see docs/TRADE_LOG_SCHEMA.md or README) only.
Research software, not advice.
"""
CAVEAT = "10-year real-quote sample; results are relative comparisons, not forecasts."
LEVEL_TAGS = {1: "Green / Lvl 1", 2: "Yellow / Lvl 2", 3: "Red / Lvl 3", 4: "Experiment / Lvl 4"}
ALLOCATIONS = [5, 15, 20, 50, 100]
START_EQUITY = 100000.0
GREEN_MAX_ALLOCATION = 15   # Green / Lvl 1 only up to this share of equity; anything larger is an Experiment


def effective_level(log_level: int, allocation: float) -> int:
    """Level shown in the report: the log's level, except that allocations above GREEN_MAX_ALLOCATION are Experiment (4)
    whatever profile produced the trades. Green (1) additionally requires the log to come from the Green rules."""
    return 4 if allocation > GREEN_MAX_ALLOCATION else int(log_level)
