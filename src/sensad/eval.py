from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from .baseline import BaselineModel

console = Console()

def _load_baseline(runp: Path) -> BaselineModel:
    cfg = json.loads((runp / "baseline.json").read_text(encoding="utf-8"))
    return BaselineModel(
        columns=cfg["columns"],
        med=cfg["med"],
        mad=cfg["mad"],
        threshold=float(cfg["threshold"]),
        agg=cfg.get("agg", "max"),
    )

def _prf(y_true: np.ndarray, y_pred: np.ndarray):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    f1 = (2 * prec * rec) / (prec + rec + 1e-12)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1}

def eval_main(run: str):
    runp = Path(run)
    model = _load_baseline(runp)

    # We expect test.csv alongside data demo; store a pointer in meta.json later if desired.
    # For now: try common locations.
    candidates = [
        runp.parent / "data" / "demo" / "test.csv",
        Path("data/demo/test.csv"),
    ]
    test_path = None
    for c in candidates:
        if c.exists():
            test_path = c
            break
    if test_path is None:
        raise FileNotFoundError("Could not find test.csv. Run: sensad synth --out data/demo first.")

    df = pd.read_csv(test_path)
    if "anomaly" not in df.columns:
        raise ValueError("test.csv must contain 'anomaly' label column (0/1).")

    X = df[model.columns].to_numpy(dtype=np.float32)
    y_true = df["anomaly"].to_numpy(dtype=int)

    score = model.score(X)

    # evaluate at trained threshold
    y_pred = (score >= model.threshold).astype(int)
    m = _prf(y_true, y_pred)

    # threshold sweep (small)
    thresholds = np.quantile(score, np.linspace(0.80, 0.999, 20))
    best = None
    for th in thresholds:
        yy = (score >= th).astype(int)
        mm = _prf(y_true, yy)
        mm["threshold"] = float(th)
        if best is None or mm["f1"] > best["f1"]:
            best = mm

    out = {
        "run": str(runp),
        "test_csv": str(test_path),
        "trained_threshold": float(model.threshold),
        "metrics_at_trained_threshold": m,
        "best_sweep": best,
    }
    (runp / "eval.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    t = Table(title="Baseline evaluation")
    t.add_column("setting")
    t.add_column("precision", justify="right")
    t.add_column("recall", justify="right")
    t.add_column("f1", justify="right")
    t.add_row("trained_threshold", f"{m['precision']:.3f}", f"{m['recall']:.3f}", f"{m['f1']:.3f}")
    t.add_row("best_sweep", f"{best['precision']:.3f}", f"{best['recall']:.3f}", f"{best['f1']:.3f}")
    console.print(t)
    console.print(f"[green]OK[/green] wrote {runp/'eval.json'}")
