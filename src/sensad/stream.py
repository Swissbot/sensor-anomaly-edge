from __future__ import annotations

import time
from pathlib import Path
import pandas as pd
from rich.console import Console

console = Console()

def stream_main(input_csv: str, rate_hz: float = 1.0, loop: bool = False, no_header: bool = False):
    """
    Stream a CSV row-by-row to stdout at a fixed rate (simulates edge sensor feed).
    """
    p = Path(input_csv)
    if not p.exists():
        raise FileNotFoundError(p)

    df = pd.read_csv(p)
    if len(df) == 0:
        return

    # Ensure time col exists; if not, still stream
    cols = list(df.columns)

    period = 1.0 / max(rate_hz, 1e-6)

    def emit_header():
        print(",".join(cols), flush=True)

    def emit_row(row):
        # Keep as CSV line
        vals = []
        for c in cols:
            v = row[c]
            vals.append("" if pd.isna(v) else str(v))
        print(",".join(vals), flush=True)

    if not no_header:
        emit_header()

    while True:
        t0 = time.time()
        for _, row in df.iterrows():
            emit_row(row)
            # timing
            dt = time.time() - t0
            sleep_s = period - dt
            if sleep_s > 0:
                time.sleep(sleep_s)
            t0 = time.time()
        if not loop:
            break

    console.print("[green]OK[/green] stream finished")
