# HNRD

This repository contains a clean PyTorch implementation of **HNRD**, a
hypergraph neural reaction-diffusion model. The default entry point reproduces
the main ModelNet40 experiment.

## ModelNet40 Main Result

The default configuration uses 4 HNRD layers, hidden dimension 64, Adam,
learning rate 0.01, weight decay 0.001, input dropout 0.2, hidden dropout 0.2,
and a learnable sigmoid-constrained step size initialized at 0.5.

| Dataset | Model | Seeds | Test Accuracy |
| --- | --- | --- | --- |
| ModelNet40 | HNRD | 20 | 98.74 +/- 0.22 |

## Installation

```bash
conda create -n hnrd python=3.10 -y
conda activate hnrd
pip install -r requirements.txt
```

Install `torch` and `torch-scatter` with wheels matching your CUDA version if
the default `pip` resolution does not match your environment.

## Data

Place the LE-style ModelNet40 files under:

```text
data/ModelNet40/ModelNet40.content
data/ModelNet40/ModelNet40.edges
```

Alternatively, pass `--data-root /path/to/data` or set:

```bash
export HNRD_DATA_ROOT=/path/to/data
```

## Run

```bash
python train_modelnet40.py --config configs/modelnet40.json --data-root /path/to/data --device 0
```

The default configuration runs 20 random seeds, `0` through `19`. The script writes:

```text
runs/modelnet40/modelnet40_hnrd_raw.csv
runs/modelnet40/modelnet40_hnrd_summary.json
```

To run a quick smoke test:

```bash
python train_modelnet40.py --data-root /path/to/data --epochs 5 --seeds 0
```

## HNRD Update

HNRD uses HND-style attention diffusion as the diffusion backbone:

```text
X_{k+1} = X_k + h [D_HND(X_k) + R(X_k) + c(X_k) Q_phi X_k]
```

where `R` is the reaction term and `c(X)` combines instantaneous dissipation
compensation with bounded feedback. The step size is parameterized by a sigmoid
so that the discrete update remains in the stable range `(0, 1)`.
