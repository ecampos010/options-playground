"""Reporting, portfolio replay and stress-lab layer for the options playground.

Consumes standardized trade logs (see docs/TRADE_LOG_SCHEMA.md or README) only.
Research software, not advice.
"""
CAVEAT = "10-year real-quote sample; results are relative comparisons, not forecasts."
LEVEL_TAGS = {1: "Green / Lvl 1", 2: "Yellow / Lvl 2", 3: "Red / Lvl 3", 4: "Experiment / Lvl 4"}
ALLOCATIONS = [5, 15, 20, 50, 100]
START_EQUITY = 100000.0
