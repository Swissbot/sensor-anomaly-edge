from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
from rich.console import Console

console = Console()

def train_main(data: str, out: str, model: str, device: str):
    """
    MVP: placeholder training. Next step we implement baseline + AE.
    """
    outp = Path(out)
    outp.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data)
    meta = {"model": model, "device": device, "n_rows": int(len(df)), "columns": list(df.columns)}
    (outp/"meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # TODO: implement baseline scorer + AE training
    (outp/"model.pt").write_text("placeholder", encoding="utf-8")

    console.print(f"[yellow]Stub[/yellow] train saved meta.json and placeholder model -> {outp}")
