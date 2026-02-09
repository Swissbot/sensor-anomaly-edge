from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from .baseline import BaselineModel

console = Console()

def _load_baseline(path: Path) -> BaselineModel:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    return BaselineModel(
        columns=cfg["columns"],
        med=cfg["med"],
        mad=cfg["mad"],
        threshold=float(cfg["threshold"]),
        agg=cfg.get("agg", "max"),
    )

def infer_main(model_path: str, input_csv: str, out_csv: str = "", threshold: float = -1.0, agg: str = ""):
    mp = Path(model_path)
    if mp.is_dir():
        mp = mp / "baseline.json"

    if mp.suffix.lower() != ".json":
        raise ValueError("For baseline inference, provide baseline.json or a run folder containing it.")

    model = _load_baseline(mp)

    # overrides
    if threshold is not None and threshold >= 0:
        model.threshold = float(threshold)
    if agg:
        if agg not in ("max", "mean"):
            raise ValueError("--agg must be 'max' or 'mean'")
        model.agg = agg

    df = pd.read_csv(input_csv)
    cols = [c for c in df.columns if c in model.columns]
    if cols != model.columns:
        raise ValueError(f"Input columns mismatch. Expected {model.columns}, got {cols}")

    X = df[model.columns].to_numpy(dtype=np.float32)
    score, pred = model.predict(X)

    out = df.copy()
    out["anomaly_score"] = score
    out["is_anomaly"] = pred

    if out_csv:
        out_path = Path(out_csv)
    else:
        out_path = mp.parent / "predictions.csv"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    # show top anomalies
    topk = min(10, len(out))
    idx = np.argsort(-score)[:topk]
    t = Table(title=f"Top {topk} anomalies (baseline)")
    t.add_column("#", justify="right")
    t.add_column("time")
    t.add_column("score", justify="right")
    t.add_column("is_anomaly", justify="right")
    for k, i in enumerate(idx, start=1):
        tm = str(out.iloc[int(i)].get("time", ""))
        t.add_row(str(k), tm, f"{score[int(i)]:.3f}", str(int(pred[int(i)])))
    console.print(t)

    console.print(f"[green]OK[/green] wrote {out_path} | threshold={model.threshold:.3f} agg={model.agg}")
