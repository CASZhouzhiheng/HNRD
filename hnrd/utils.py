import random
from typing import Dict

import numpy as np
import torch


def fix_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def balanced_split(
    labels: torch.Tensor,
    train_prop: float = 0.5,
    valid_prop: float = 0.25,
) -> Dict[str, torch.Tensor]:
    labels = labels.view(-1).cpu()
    num_classes = int(labels.max().item() + 1)
    per_class_train = int(train_prop / num_classes * labels.numel())
    class_indices = []
    for c in range(num_classes):
        idx = torch.where(labels == c)[0]
        idx = idx[torch.randperm(idx.numel())]
        class_indices.append(idx)

    train_idx = torch.cat([idx[:per_class_train] for idx in class_indices], dim=0)
    rest = torch.cat([idx[per_class_train:] for idx in class_indices], dim=0)
    rest = rest[torch.randperm(rest.numel())]
    val_count = int(valid_prop * labels.numel())
    return {
        "train": train_idx,
        "valid": rest[:val_count],
        "test": rest[val_count:],
    }


def normalize_edges(edge_list, num_nodes: int):
    clean = []
    for edge in edge_list:
        nodes = sorted({int(v) for v in edge if 0 <= int(v) < num_nodes})
        if nodes:
            clean.append(tuple(nodes))
    return clean


def incidence_index(edge_list, num_nodes: int, device: torch.device):
    vertices = []
    edges = []
    clean_edges = []
    for edge in edge_list:
        nodes = sorted({int(v) for v in edge if 0 <= int(v) < num_nodes})
        if not nodes:
            continue
        edge_id = len(clean_edges)
        clean_edges.append(tuple(nodes))
        for node in nodes:
            vertices.append(node)
            edges.append(edge_id)

    if not vertices:
        vertices = list(range(num_nodes))
        edges = list(range(num_nodes))
        clean_edges = [(i,) for i in range(num_nodes)]

    return (
        torch.tensor(vertices, dtype=torch.long, device=device),
        torch.tensor(edges, dtype=torch.long, device=device),
        len(clean_edges),
    )

