from __future__ import annotations
from pathlib import Path
import json
from rich.console import Console

console = Console()

def eval_main(run: str):
    runp = Path(run)
    meta = json.loads((runp/"meta.json").read_text(encoding="utf-8"))
    console.print("[green]OK[/green] Loaded run:")
    console.print(meta)
