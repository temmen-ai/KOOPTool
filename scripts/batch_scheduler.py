#!/usr/bin/env python3
"""
Eenvoudige scheduler die elke dag om 22:00 de KOOP-batch draait.

Gebruik:
    - Pas INPUT_DIR / OUTPUT_DIR / PYTHON_BIN aan indien nodig.
    - Start dit script één keer (bijvoorbeeld via `nohup python scripts/batch_scheduler.py &`)
      en laat het proces draaien.
"""

from __future__ import annotations

import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

# === Config ===
PROJECT_DIR = Path("/Users/denizayhan/Desktop/KOOPTool")
PYTHON_BIN = Path("/Users/denizayhan/opt/anaconda3/envs/kooptool/bin/python")
INPUT_DIR = PROJECT_DIR / "inputmap"
OUTPUT_DIR = PROJECT_DIR / "outputmap"
LOG_FILE = Path("/Users/denizayhan/koop_batch_scheduler.log")

RUN_HOUR = 22
RUN_MINUTE = 0


def next_run(after: datetime | None = None) -> datetime:
    """Bepaal het eerstvolgende moment waarop we moeten draaien."""
    after = after or datetime.now()
    candidate = after.replace(hour=RUN_HOUR, minute=RUN_MINUTE, second=0, microsecond=0)
    if candidate <= after:
        candidate += timedelta(days=1)
    return candidate


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write(f"[{timestamp}] {message}\n")


def run_batch() -> None:
    cmd = [
        str(PYTHON_BIN),
        str(PROJECT_DIR / "KOOP_batch.py"),
        str(INPUT_DIR),
        str(OUTPUT_DIR),
    ]
    log(f"Start batch: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        log(f"Batch klaar (exit {result.returncode}). Output:\n{result.stdout}")
    except Exception as exc:
        log(f"Batchfout: {exc}")


def main() -> None:
    log("Scheduler gestart.")
    wait_until = next_run()
    log(f"Eerste run gepland op {wait_until}.")
    while True:
        now = datetime.now()
        if now >= wait_until:
            run_batch()
            wait_until = next_run(now)
            log(f"Volgende run gepland op {wait_until}.")
        time.sleep(30)


if __name__ == "__main__":
    main()
