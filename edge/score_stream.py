#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

EPS = 1e-12

def robust_z(x: float, med: float, mad: float) -> float:
    denom = 1.4826 * mad + EPS
    return (x - med) / denom

def main():
    ap = argparse.ArgumentParser(description="Edge stream anomaly scoring (reads CSV from stdin)")
    ap.add_argument("--baseline", required=True, help="Path to baseline.json")
    ap.add_argument("--threshold", type=float, default=None, help="Override threshold")
    ap.add_argument("--agg", choices=["max", "mean"], default=None, help="Override aggregation")
    ap.add_argument("--only-anomalies", action="store_true", help="Print only anomalous rows")
    args = ap.parse_args()

    cfg = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    columns = cfg["columns"]
    med = {k: float(v) for k, v in cfg["med"].items()}
    mad = {k: float(v) for k, v in cfg["mad"].items()}
    threshold = float(cfg["threshold"]) if args.threshold is None else float(args.threshold)
    agg = cfg.get("agg", "max") if args.agg is None else args.agg

    reader = csv.DictReader(sys.stdin)
    if reader.fieldnames is None:
        raise SystemExit("No CSV header received on stdin.")

    missing = [c for c in columns if c not in reader.fieldnames]
    if missing:
        raise SystemExit(f"Missing columns in stream: {missing}")

    # Output header
    out_fields = list(reader.fieldnames) + ["anomaly_score", "is_anomaly"]
    writer = csv.DictWriter(sys.stdout, fieldnames=out_fields)
    writer.writeheader()

    for row in reader:
        zs = []
        for c in columns:
            try:
                x = float(row[c])
            except Exception:
                x = float("nan")
            z = robust_z(x, med[c], mad[c])
            zs.append(abs(z))

        if agg == "mean":
            score = float(np.mean(zs))
        else:
            score = float(np.max(zs))

        is_anom = 1 if score >= threshold else 0

        if args.only_anomalies and not is_anom:
            continue

        row_out = dict(row)
        row_out["anomaly_score"] = f"{score:.6f}"
        row_out["is_anomaly"] = str(is_anom)
        writer.writerow(row_out)
        sys.stdout.flush()

if __name__ == "__main__":
    main()
