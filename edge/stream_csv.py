#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="Stream a CSV line-by-line to stdout (edge helper)")
    ap.add_argument("--input", required=True, help="CSV file to stream")
    ap.add_argument("--rate", type=float, default=1.0, help="Rows per second (Hz)")
    ap.add_argument("--loop", action="store_true", help="Loop forever")
    ap.add_argument("--no-header", action="store_true", help="Do not emit header")
    args = ap.parse_args()

    p = Path(args.input)
    if not p.exists():
        raise SystemExit(f"File not found: {p}")

    period = 1.0 / max(args.rate, 1e-6)

    # read once into memory as list of dicts (tiny demo files)
    with p.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        header = reader.fieldnames

    if not rows or not header:
        return

    def emit_header():
        print(",".join(header), flush=True)

    def emit_row(row: dict):
        vals = [str(row.get(c, "")) for c in header]
        print(",".join(vals), flush=True)

    if not args.no_header:
        emit_header()

    while True:
        for row in rows:
            t0 = time.time()
            emit_row(row)
            dt = time.time() - t0
            sleep_s = period - dt
            if sleep_s > 0:
                time.sleep(sleep_s)
        if not args.loop:
            break

if __name__ == "__main__":
    main()
