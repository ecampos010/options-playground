"""export-outputs and purge-vendor-data (licensing hygiene).

ThetaData's raw quotes are licensed for the owner's personal use: they must not be archived, shared or kept after
cancelling. Derived per-trade logs (runs/, results/) contain no raw quotes and may be exported. `export-outputs`
therefore NEVER includes data/, cache/ or any parquet file.
"""
from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime
from pathlib import Path

INCLUDE_DIRS = ["runs", "results", "reports", "profiles", "sweeps", "docs", "templates", "playground", "tests"]
INCLUDE_FILES = ["README.md", "requirements.txt", ".env.example", ".gitignore"]
EXTRA_CONFIGS = {"data/stress_windows.csv": "configs/stress_windows.csv"}      # user config, not vendor data
EXCLUDED_TOP = {"data", "cache", "exports", ".git", ".venv", "venv"}


def _skip(rel: Path) -> bool:
    parts = rel.parts
    return (parts[0] in EXCLUDED_TOP or "__pycache__" in parts or rel.suffix in (".parquet", ".pyc")
            or any(p.startswith(".pytest") for p in parts))


def git_info(root: Path) -> dict:
    def g(*a):
        try:
            return subprocess.run(["git", *a], cwd=root, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return ""
    return {"commit": g("rev-parse", "HEAD") or "unknown", "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(g("status", "--porcelain"))}


def build_manifest(root: Path) -> dict:
    runs = []
    for cfgp in sorted((root / "runs").glob("*/config.json")) if (root / "runs").exists() else []:
        try:
            c = json.loads(cfgp.read_text())
            m = json.loads((cfgp.parent / "meta.json").read_text()) if (cfgp.parent / "meta.json").exists() else {}
            runs.append({"run_id": cfgp.parent.name, "command": c.get("command"), "data_range": c.get("data_range"),
                         "window": c.get("window"), "git_commit": c.get("git_commit"),
                         "spot_source": m.get("spot_coverage", {}).get("days_by_source"),
                         "fill_model": m.get("fill_model"), "level": m.get("level")})
        except Exception:
            continue
    return {"created": datetime.now().isoformat(timespec="seconds"), "git": git_info(root),
            "versions": {"python": sys.version.split()[0], "platform": platform.platform()},
            "n_runs": len(runs), "runs": runs,
            "licensing_note": "Contains only derived per-trade numbers. No raw vendor quotes: data/, cache/ and all "
                              "parquet files are excluded by construction."}


def export_outputs(root: str | Path = ".", out_dir: str | Path = "exports", name: str | None = None) -> Path:
    root = Path(root).resolve()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = name or f"playground_outputs_{datetime.now():%Y%m%d_%H%M%S}"
    path = out_dir / f"{name}.tar.gz"
    manifest = build_manifest(root)
    with tarfile.open(path, "w:gz") as tf:
        for d in INCLUDE_DIRS:
            base = root / d
            if not base.exists():
                continue
            for f in sorted(base.rglob("*")):
                if f.is_file() and not _skip(f.relative_to(root)):
                    tf.add(f, arcname=str(f.relative_to(root)))
        for fn in INCLUDE_FILES:
            if (root / fn).exists():
                tf.add(root / fn, arcname=fn)
        for src, dst in EXTRA_CONFIGS.items():
            if (root / src).exists():
                tf.add(root / src, arcname=dst)
        data = json.dumps(manifest, indent=2).encode()
        import io
        ti = tarfile.TarInfo("manifest.json")
        ti.size, ti.mtime = len(data), time.time()
        tf.addfile(ti, io.BytesIO(data))
    return path


# ---------------------------------------------------------------- purge
VENDOR_SUBDIRS = ["options", "listing", "greeks", "underlying_bars", "state"]


def vendor_targets(data_dir: Path, include_spot: bool = False) -> list[Path]:
    """ThetaData-derived caches: options/listing/greeks/underlying_bars/state (also under tickers/*/).
    Kept: data/spot (Alpaca/Massive, unless include_spot), data/underlying (yfinance/Cboe daily), events.csv,
    stress_windows.csv and any other user file."""
    out = []
    for base in [data_dir] + sorted((data_dir / "tickers").glob("*")) if (data_dir / "tickers").exists() else [data_dir]:
        for sub in VENDOR_SUBDIRS:
            p = base / sub
            if p.exists():
                out.append(p)
    if (data_dir / "tickers").exists():
        pass
    if include_spot and (data_dir / "spot").exists():
        out.append(data_dir / "spot")
    return out


def dir_stats(p: Path) -> tuple[int, int]:
    files = [f for f in p.rglob("*") if f.is_file()]
    return len(files), sum(f.stat().st_size for f in files)


def purge_vendor_data(data_dir="data", yes=False, include_spot=False, log_path="purge_log.md") -> dict:
    data_dir = Path(data_dir)
    targets = vendor_targets(data_dir, include_spot)
    rows = [(str(t), *dir_stats(t)) for t in targets]
    res = {"dry_run": not yes, "targets": rows, "deleted": []}
    if yes:
        for t in targets:
            shutil.rmtree(t, ignore_errors=False)
            res["deleted"].append(str(t))
        # drop now-empty per-ticker folders
        tick = data_dir / "tickers"
        if tick.exists():
            for d in tick.glob("*"):
                if d.is_dir() and not any(d.iterdir()):
                    d.rmdir()
            if not any(tick.iterdir()):
                tick.rmdir()
        lines = ["# Vendor data purge log", f"- performed: {datetime.now().isoformat(timespec='seconds')}",
                 f"- host: {platform.node()}", f"- data dir: {data_dir.resolve()}",
                 f"- include spot caches: {include_spot}", "", "| path | files | bytes |", "|---|---|---|"]
        lines += [f"| {p} | {n} | {b} |" for p, n, b in rows]
        lines += ["", "Kept: " + ", ".join(sorted(x.name for x in data_dir.iterdir())) if data_dir.exists() else "Kept: (none)",
                  "", "Derived outputs (runs/, results/) contain no raw vendor quotes and were not touched."]
        Path(log_path).write_text("\n".join(lines) + "\n")
        res["log"] = str(log_path)
    return res
