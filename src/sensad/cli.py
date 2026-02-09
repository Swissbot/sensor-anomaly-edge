from __future__ import annotations

import typer
from rich.console import Console

from .synth import synth_main
from .train import train_main
from .eval import eval_main
from .export import export_main
from .infer import infer_main
from .stream import stream_main

app = typer.Typer(add_completion=False)
console = Console()

@app.command()
def synth(
    out: str = typer.Option("data/demo", "--out", help="Output folder"),
    minutes: int = typer.Option(60, "--minutes", help="Duration in minutes"),
    freq: str = typer.Option("1s", "--freq", help="Sampling frequency (e.g., 1s, 200ms)"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
):
    synth_main(out=out, minutes=minutes, freq=freq, seed=seed)

@app.command()
def train(
    data: str = typer.Option(..., "--data", help="CSV file (time, sensor1..N)"),
    out: str = typer.Option("runs/demo", "--out", help="Run output folder"),
    model: str = typer.Option("baseline", "--model", help="baseline|ae"),
    device: str = typer.Option("cpu", "--device", help="cpu|cuda (for AE)"),
):
    train_main(data=data, out=out, model=model, device=device)

@app.command()
def eval(
    run: str = typer.Option(..., "--run", help="Run folder with saved artifacts"),
):
    eval_main(run=run)

@app.command()
def export(
    run: str = typer.Option(..., "--run", help="Run folder"),
    format: str = typer.Option("torchscript", "--format", help="torchscript|onnx"),
):
    export_main(run=run, format=format)

@app.command()
def infer(
    model: str = typer.Option(..., "--model", help="baseline.json OR run folder containing it"),
    input: str = typer.Option(..., "--input", help="CSV file to score"),
    out: str = typer.Option("", "--out", help="Output CSV (default: <run>/predictions.csv)"),
    threshold: float = typer.Option(-1.0, "--threshold", help="Override threshold (use -1 to keep trained)"),
    agg: str = typer.Option("", "--agg", help="Override aggregation: max|mean (empty keeps trained)"),
):
    infer_main(model_path=model, input_csv=input, out_csv=out, threshold=threshold, agg=agg)

@app.command()
def stream(
    input: str = typer.Option(..., "--input", help="CSV to stream line-by-line"),
    rate: float = typer.Option(1.0, "--rate", help="Rows per second (Hz)"),
    loop: bool = typer.Option(False, "--loop", help="Loop forever"),
    no_header: bool = typer.Option(False, "--no-header", help="Do not emit CSV header"),
):
    """
    Stream a CSV as if it was a live sensor feed (stdout).
    """
    stream_main(input_csv=input, rate_hz=rate, loop=loop, no_header=no_header)

def main():
    app()

if __name__ == "__main__":
    main()
