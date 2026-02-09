from __future__ import annotations
from pathlib import Path
import pandas as pd
from rich.console import Console

console = Console()

def infer_main(model_path: str, input_csv: str):
    mp = Path(model_path)
    df = pd.read_csv(input_csv)
    console.print(f"[yellow]Stub[/yellow] would load model {mp} and score {len(df)} rows")
