import os
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from .utils import normalize_edges


SUPPORTED_DATASETS = (
    "Cora",
    "Citeseer",
    "Pubmed",
    "Cora-CA",
    "DBLP-CA",
    "Zoo",
    "NTU2012",
    "ModelNet40",
    "Walmart",
    "Senate",
    "House",
)

DHG_DATASETS = {
    "Cora": "CocitationCora",
    "Citeseer": "CocitationCiteseer",
    "Pubmed": "CocitationPubmed",
    "Cora-CA": "CoauthorshipCora",
    "DBLP-CA": "CoauthorshipDBLP",
    "Walmart": "WalmartTrips",
}


def _data_root(data_root=None):
    return Path(data_root or os.environ.get("HNRD_DATA_ROOT", "data"))


def _make_result(features, labels, edge_list):
    labels = labels.long().view(-1)
    labels = labels - labels.min()
    num_nodes = labels.numel()
    return {
        "features": features.float()[:num_nodes],
        "labels": labels,
        "edge_list": normalize_edges(edge_list, num_nodes),
        "num_nodes": num_nodes,
        "num_classes": int(labels.max().item() + 1),
    }


def load_dhg_dataset(name):
    from dhg.data import (
        CoauthorshipCora,
        CoauthorshipDBLP,
        CocitationCiteseer,
        CocitationCora,
        CocitationPubmed,
        WalmartTrips,
    )

    class_map = {
        "CocitationCora": CocitationCora,
        "CocitationCiteseer": CocitationCiteseer,
        "CocitationPubmed": CocitationPubmed,
        "CoauthorshipCora": CoauthorshipCora,
        "CoauthorshipDBLP": CoauthorshipDBLP,
        "WalmartTrips": WalmartTrips,
    }
    dataset = class_map[name]()
    labels = dataset["labels"]
    if not torch.is_tensor(labels):
        labels = torch.as_tensor(labels)
    num_nodes = int(dataset["num_vertices"])
    num_classes = int(dataset["num_classes"])

    if hasattr(dataset, "content") and "features" in dataset.content:
        features = dataset["features"].float()
    else:
        features = torch.zeros((num_nodes, max(num_classes, 100)), dtype=torch.float32)
        labels_tmp = labels.long().view(-1) - labels.min()
        features[torch.arange(num_nodes), labels_tmp] = 1.0
        features = torch.normal(features, 1.0)

    edge_list = [tuple(sorted(set(map(int, edge)))) for edge in dataset["edge_list"] if len(edge) > 0]
    return _make_result(features, labels, edge_list)


def _find_le_paths(root, dataset):
    candidates = {
        "Zoo": [("zoo", "zoo"), ("Zoo", "Zoo")],
        "NTU2012": [("NTU2012", "NTU2012")],
        "ModelNet40": [("ModelNet40", "ModelNet40")],
    }[dataset]
    for directory, stem in candidates:
        path = root / directory
        content_path = path / f"{stem}.content"
        edge_path = path / f"{stem}.edges"
        if content_path.exists() and edge_path.exists():
            return content_path, edge_path
    tried = ", ".join(str(root / directory / f"{stem}.content") for directory, stem in candidates)
    raise FileNotFoundError(f"{dataset} files were not found. Tried: {tried}")


def load_le_dataset(dataset, data_root=None):
    """Load LE-style hypergraph files.

    Expected layout:
        data_root/<dataset>/<dataset>.content
        data_root/<dataset>/<dataset>.edges
    """
    content_path, edge_path = _find_le_paths(_data_root(data_root), dataset)
    content = np.genfromtxt(content_path, dtype=np.dtype(str))
    features = sp.csr_matrix(content[:, 1:-1], dtype=np.float32)
    labels = torch.as_tensor(content[:, -1].astype(float), dtype=torch.long)

    node_ids = np.asarray(content[:, 0], dtype=np.int64)
    id_map = {old_id: new_id for new_id, old_id in enumerate(node_ids)}
    raw_edges = np.genfromtxt(edge_path, dtype=np.int64)
    if raw_edges.ndim == 1:
        raw_edges = raw_edges.reshape(1, -1)
    mapped = np.array(list(map(id_map.get, raw_edges.flatten())), dtype=np.int64).reshape(raw_edges.shape)
    edge_index = mapped.T
    num_nodes = int(edge_index[0].max() + 1)

    edge_list = []
    for edge_id in sorted(np.unique(edge_index[1])):
        nodes = np.unique(edge_index[0, edge_index[1] == edge_id]).tolist()
        edge_list.append(tuple(int(v) for v in nodes))

    return _make_result(torch.as_tensor(features[:num_nodes].todense(), dtype=torch.float32), labels[:num_nodes], edge_list)


def load_cornell_dataset(dataset, data_root=None, feature_dim=100, feature_noise=1.0):
    directory = {"House": "house-committees", "Senate": "senate-committees"}[dataset]
    path = _data_root(data_root) / directory
    label_path = path / f"node-labels-{directory}.txt"
    edge_path = path / f"hyperedges-{directory}.txt"
    if not label_path.exists() or not edge_path.exists():
        raise FileNotFoundError(f"{dataset} files were not found under {path}.")

    labels = torch.as_tensor(np.genfromtxt(label_path, dtype=np.int64), dtype=torch.long).view(-1)
    labels = labels - labels.min()
    num_nodes = labels.numel()
    num_classes = int(labels.max().item() + 1)
    features = torch.zeros((num_nodes, max(feature_dim, num_classes)), dtype=torch.float32)
    features[torch.arange(num_nodes), labels] = 1.0
    features = torch.normal(features, feature_noise)

    edge_list = []
    with edge_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            nodes = [int(value) for value in line.strip().split(",") if value]
            if nodes and min(nodes) == 1:
                nodes = [node - 1 for node in nodes]
            if nodes:
                edge_list.append(tuple(sorted(set(nodes))))
    return _make_result(features, labels, edge_list)


def load_dataset(name: str, data_root=None):
    aliases = {
        "CocitationCora": "Cora",
        "CocitationCiteseer": "Citeseer",
        "CocitationPubmed": "Pubmed",
        "CoauthorshipCora": "Cora-CA",
        "CoauthorshipDBLP": "DBLP-CA",
        "WalmartTrips": "Walmart",
    }
    name = aliases.get(name, name)
    if name in DHG_DATASETS:
        return load_dhg_dataset(DHG_DATASETS[name])
    if name in {"Zoo", "NTU2012", "ModelNet40"}:
        return load_le_dataset(name, data_root)
    if name in {"House", "Senate"}:
        return load_cornell_dataset(name, data_root)
    raise ValueError(f"Unsupported dataset: {name}. Choose from {', '.join(SUPPORTED_DATASETS)}.")
