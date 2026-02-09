.PHONY: setup synth train eval export infer

setup:
	python3.12 -m venv .venv
	. .venv/bin/activate && python -m pip install -U pip setuptools wheel
	. .venv/bin/activate && pip install -r requirements.txt
	. .venv/bin/activate && pip install -e .

synth:
	. .venv/bin/activate && sensad synth --out data/demo --minutes 30 --freq 1s --seed 42

train:
	. .venv/bin/activate && sensad train --data data/demo/train.csv --out runs/demo --model baseline

eval:
	. .venv/bin/activate && sensad eval --run runs/demo

export:
	. .venv/bin/activate && sensad export --run runs/demo --format torchscript

infer:
	. .venv/bin/activate && sensad infer --model runs/demo/model.pt --input data/demo/test.csv
