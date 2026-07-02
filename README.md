# HNRD

This repository contains a clean PyTorch implementation of **HNRD**, a
hypergraph neural reaction-diffusion model.
The diffusion part is implemented with incidence-domain gradient and divergence
operators in `hnrd/operators.py`, corresponding to `-G^T A_theta(X) G X`.

## Supported Datasets

The parser supports the following 11 datasets:

```text
Cora
Citeseer
Pubmed
Cora-CA
DBLP-CA
Zoo
NTU2012
ModelNet40
Walmart
Senate
House
```

## Installation

```bash
conda create -n hnrd python=3.10 -y
conda activate hnrd
pip install -r requirements.txt
```

Install `torch` and `torch-scatter` with wheels matching your CUDA version if
the default `pip` resolution does not match your environment.

## Data

See [data/README.md](data/README.md) for the expected layout. Dataset files are
ignored by Git by default.

## HNRD Update

HNRD uses HND-style attention diffusion as the diffusion backbone:

```text
X_{k+1} = X_k + h [-G^T A_theta(X_k) G X_k + R_eta(X_k)]
```

where `R_eta(X)` combines instantaneous dissipation compensation with bounded
feedback along the transverse component `Q_phi X`. The step size is
parameterized by a sigmoid so that the discrete update remains in the stable
range `(0, 1)`.
