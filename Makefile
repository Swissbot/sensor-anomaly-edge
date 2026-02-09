# Sensor Anomaly Detection (Edge-Ready)
# Makefile helpers for setup, data synthesis, training, evaluation, and edge demos.

.PHONY: setup synth train eval infer stream edge-venv edge-demo-dev edge-demo-edge

setup:
	python3.12 -m venv .venv
	. .venv/bin/activate && python -m pip install -U pip setuptools wheel
	. .venv/bin/activate && pip install -r requirements.txt
	. .venv/bin/activate && pip install -e .

synth:
	. .venv/bin/activate && sensad synth --out data/demo --minutes 5 --freq 1s --seed 42

train:
	. .venv/bin/activate && sensad train --data data/demo/train.csv --out runs/demo --model baseline

eval:
	. .venv/bin/activate && sensad eval --run runs/demo

infer:
	. .venv/bin/activate && sensad infer --model runs/demo --input data/demo/test.csv

stream:
	. .venv/bin/activate && sensad stream --input data/demo/test.csv --rate 5

# Minimal venv for edge scripts (no 'sensad' CLI). Still needs numpy+pandas.
edge-venv:
	python3 -m venv edge_venv
	. edge_venv/bin/activate && python -m pip install -U pip
	. edge_venv/bin/activate && pip install numpy pandas

# Dev-mode: uses sensad stream (requires .venv)
edge-demo-dev:
	. .venv/bin/activate && sensad stream --input data/demo/test.csv --rate 5 | \
	python3 edge/score_stream.py --baseline runs/demo/baseline.json --only-anomalies

# Edge-mode: scripts only (requires edge_venv; no sensad usage)
edge-demo-edge:
	. edge_venv/bin/activate && python3 edge/stream_csv.py --input data/demo/test.csv --rate 5 | \
	python3 edge/score_stream.py --baseline runs/demo/baseline.json --only-anomalies
