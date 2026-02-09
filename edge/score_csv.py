#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-12

def robust_z(x: np.ndarray, med: float, mad: float) -> np.ndarray:
    denom = 1.4826 * mad + EPS
    return (x - med) / denom

def main():
    ap = argparse.ArgumentParser(description="Edge anomaly scoring (baseline robust-z)")
    ap.add_argument("--baseline", required=True, help="Path to baseline.json")
    ap.add_argument("--input", required=True, help="Input CSV (must contain the sensor columns)")
    ap.add_argument("--out", default="predictions.csv", help="Output CSV path")
    ap.add_argument("--threshold", type=float, default=None, help="Override threshold (float)")
    ap.add_argument("--agg", choices=["max", "mean"], default=None, help="Override aggregation mode")
    args = ap.parse_args()

    baseline_path = Path(args.baseline)
    cfg = json.loads(baseline_path.read_text(encoding="utf-8"))

    columns = cfg["columns"]
    med = cfg["med"]
    mad = cfg["mad"]
    threshold = float(cfg["threshold"]) if args.threshold is None else float(args.threshold)
    agg = cfg.get("agg", "max") if args.agg is None else args.agg

    df = pd.read_csv(args.input)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in input CSV: {missing}")

    X = df[columns].to_numpy(dtype=np.float32)

    zs = []
    for j, col in enumerate(columns):
        z = robust_z(X[:, j], float(med[col]), float(mad[col]))
        zs.append(np.abs(z))
    Z = np.stack(zs, axis=1)

    score = np.mean(Z, axis=1) if agg == "mean" else np.max(Z, axis=1)
    pred = (score >= threshold).astype(int)

    out = df.copy()
    out["anomaly_score"] = score
    out["is_anomaly"] = pred
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    # small summary
    topk = min(5, len(out))
    idx = np.argsort(-score)[:topk]
    print(f"OK: wrote {out_path} | threshold={threshold:.3f} agg={agg}")
    print("Top anomalies:")
    for i in idx:
        t = out.iloc[int(i)].get("time", "")
        print(f"  score={score[int(i)]:.3f} is_anomaly={int(pred[int(i)])} time={t}")

if __name__ == "__main__":
    main()
