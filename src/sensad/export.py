from __future__ import annotations
from pathlib import Path
from rich.console import Console

console = Console()

def export_main(run: str, format: str):
    runp = Path(run)
    console.print(f"[yellow]Stub[/yellow] export {format} from {runp} (next step)")
