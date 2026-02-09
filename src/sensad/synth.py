from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from rich.console import Console

console = Console()

def synth_main(out: str, minutes: int, freq: str, seed: int):
    """
    Generates synthetic multi-sensor data with injected anomalies.
    Writes: train.csv, test.csv (with anomaly labels)
    """
    outp = Path(out)
    outp.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    # timeline
    dt = pd.date_range("2026-01-01", periods=int((minutes*60)/pd.Timedelta(freq).total_seconds()), freq=freq)
    n = len(dt)

    # 3 sensors: temp, pressure, vibration (toy)
    t = np.arange(n)
    temp = 50 + 2*np.sin(2*np.pi*t/600) + rng.normal(0, 0.25, n)
    press = 5 + 0.2*np.sin(2*np.pi*t/300) + rng.normal(0, 0.05, n)
    vib = 0.8 + 0.1*np.sin(2*np.pi*t/120) + rng.normal(0, 0.03, n)

    y = np.zeros(n, dtype=int)

    # anomalies: spikes, drift, dropouts
    for _ in range(6):
        idx = int(rng.integers(50, n-50))
        temp[idx:idx+3] += rng.uniform(6, 12)
        y[idx:idx+3] = 1

    drift_start = int(rng.integers(n//3, n//2))
    press[drift_start:] += np.linspace(0, rng.uniform(0.6, 1.2), n-drift_start)
    y[drift_start:drift_start+60] = 1  # mark early drift window as anomaly-ish label

    for _ in range(3):
        idx = int(rng.integers(100, n-100))
        vib[idx:idx+20] = np.nan  # dropout
        y[idx:idx+20] = 1

    df = pd.DataFrame({"time": dt, "temp": temp, "pressure": press, "vibration": vib, "anomaly": y})
    df = df.interpolate(limit_direction="both")

    # split: last 20% test
    cut = int(n*0.8)
    train = df.iloc[:cut].drop(columns=["anomaly"])
    test = df.iloc[cut:]

    train.to_csv(outp/"train.csv", index=False)
    test.to_csv(outp/"test.csv", index=False)

    console.print(f"[green]OK[/green] wrote {outp/'train.csv'} and {outp/'test.csv'}  (test includes anomaly labels)")
