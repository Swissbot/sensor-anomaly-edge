# Sensor Anomaly Detection (Edge-Ready)

**DE:** Zeitreihen-Anomalieerkennung für Sensordaten (IoT/Industrial). Enthält reproduzierbare Datensynthese, Baseline-Detector und Edge-Ready Inferenz (Batch + Stream).  
**EN:** Time-series anomaly detection for sensors (IoT/industrial). Includes reproducible data synthesis, a baseline detector, and edge-ready inference (batch + stream).

<p>
  <strong>Author / Autor:</strong> Roger Seeberger (Swissbot)<br>
  <img src="docs/author_icon.png" alt="Author icon" width="64" />
</p>

---

## Features
- Synthetic multi-sensor **time-series generator** with injected anomalies (spikes, drift, dropouts)
- **Baseline detector** (robust z-score using MAD) with automatic threshold from train quantile
- **Batch inference**: score CSV → `predictions.csv`
- **Edge scripts** for portable inference:
  - `edge/score_csv.py` for batch scoring
  - `edge/score_stream.py` for real-time scoring from stdin
  - `edge/stream_csv.py` to simulate a live CSV feed without `sensad`
- **Streaming demo**: `sensad stream` simulates MQTT/serial-like feeds

---

## Installation (Ubuntu 24.04, Python 3.12)

### 1) Create & activate venv
```bash
cd sensor-anomaly-edge
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
```

### 2) Install dependencies + project (editable)
```bash
pip install -r requirements.txt
pip install -e .
```

### 3) Verify CLI
```bash
sensad --help
```

---

## Quickstart (CLI)

### 1) Generate synthetic dataset
```bash
mkdir -p data
sensad synth --out data/demo --minutes 5 --freq 1s --seed 42
```

This writes:
```text
data/demo/train.csv
data/demo/test.csv   (includes column: anomaly = 0/1)
```

### 2) Train baseline detector
```bash
sensad train --data data/demo/train.csv --out runs/demo --model baseline
```

Artifacts:
```text
runs/demo/meta.json
runs/demo/baseline.json
```

### 3) Batch inference (score test.csv)
```bash
sensad infer --model runs/demo --input data/demo/test.csv
```

Default output:
```text
runs/demo/predictions.csv
```

Optional overrides:
```bash
# stricter threshold
sensad infer --model runs/demo --input data/demo/test.csv --threshold 5.0 --out runs/demo/pred_thr5.csv

# aggregation across sensors: max | mean
sensad infer --model runs/demo --input data/demo/test.csv --agg mean --out runs/demo/pred_mean.csv
```

### 4) Evaluate against labels
```bash
sensad eval --run runs/demo
```

Writes:
```text
runs/demo/eval.json
```

---

## Edge-ready scripts (batch)

These scripts do **not** require installing the package (`pip install -e .`), but they still require Python deps like `numpy`/`pandas`.

### Batch scoring (edge)
```bash
python3 edge/score_csv.py \
  --baseline runs/demo/baseline.json \
  --input data/demo/test.csv \
  --out runs/demo/edge_predictions.csv
```

You can override threshold / aggregation:
```bash
python3 edge/score_csv.py \
  --baseline runs/demo/baseline.json \
  --input data/demo/test.csv \
  --out runs/demo/out.csv \
  --threshold 5.0 \
  --agg mean
```

---

## Edge Demo (2 Terminals)

There are **two ways** to run the live demo:

- **Dev mode (recommended):** uses the `sensad` CLI inside your project venv.
- **Edge mode:** uses only scripts in `edge/` (no `sensad` install), but still requires Python deps (`numpy`, `pandas`).

### A) Dev mode (project venv)

**Terminal A — stream sensor rows**
```bash
source .venv/bin/activate
sensad synth --out data/demo --minutes 5 --freq 1s --seed 42
sensad train --data data/demo/train.csv --out runs/demo --model baseline

# stream test.csv as live feed (rows/sec)
sensad stream --input data/demo/test.csv --rate 5
```

**Terminal B — score stream**
```bash
source .venv/bin/activate
sensad stream --input data/demo/test.csv --rate 5 | \
  python3 edge/score_stream.py --baseline runs/demo/baseline.json --only-anomalies
```

### B) Edge mode (scripts only)

This mode does **not** require installing the package (`sensad`), but you still need Python deps:

```bash
python3 -m venv edge_venv
source edge_venv/bin/activate
pip install -U pip
pip install numpy pandas
```

**Terminal A — stream CSV (edge helper)**
```bash
source edge_venv/bin/activate
python3 edge/stream_csv.py --input data/demo/test.csv --rate 5
```

**Terminal B — score stream**
```bash
source edge_venv/bin/activate
python3 edge/stream_csv.py --input data/demo/test.csv --rate 5 | \
  python3 edge/score_stream.py --baseline runs/demo/baseline.json --only-anomalies
```

---

## What’s inside (Baseline model)

The baseline detector computes a **robust z-score per sensor** using median + MAD:

```text
z = (x - median) / (1.4826 * MAD + eps)

per-row anomaly score = max(|z_i|)  (or mean(|z_i|) with --agg mean)

anomaly decision: score >= threshold
```

The threshold is set automatically from a high quantile of training scores (default: `0.995`).

---

## Makefile shortcuts (optional)

```bash
make setup
make synth
make train
make eval
```

---

## Notes
- This repo uses synthetic data, so it can be public and shareable.
- Next upgrade (optional): add an Autoencoder path (`--model ae`) and export to TorchScript/ONNX for even smaller edge runtimes.
