from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
import numpy as np
from rich.console import Console

from .baseline import fit_baseline, choose_threshold_from_train

console = Console()

def train_main(data: str, out: str, model: str, device: str):
    outp = Path(out)
    outp.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data)
    # Expect a time column; keep only sensor columns
    cols = [c for c in df.columns if c != "time"]
    if not cols:
        raise ValueError("No sensor columns found (expected columns besides 'time').")

    X = df[cols].to_numpy(dtype=np.float32)

    meta = {"model": model, "device": device, "n_rows": int(len(df)), "columns": cols}
    (outp / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if model.lower() != "baseline":
        raise ValueError("Only model=baseline implemented right now.")

    bm = fit_baseline(X, columns=cols, agg="max")
    bm.threshold = choose_threshold_from_train(bm, X, q=0.995)

    baseline_payload = {
        "type": "baseline_robust_z",
        "columns": bm.columns,
        "med": bm.med,
        "mad": bm.mad,
        "threshold": bm.threshold,
        "agg": bm.agg,
        "train_quantile": 0.995,
    }
    (outp / "baseline.json").write_text(json.dumps(baseline_payload, indent=2), encoding="utf-8")

    console.print(f"[green]OK[/green] baseline trained -> {outp/'baseline.json'}")
    console.print(f"Threshold set from train quantile 0.995: {bm.threshold:.3f}")
