from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class IncidenceOperator:
    """Incidence-domain implementation of G and G^T for hypergraphs."""

    node_ids: torch.Tensor
    edge_ids: torch.Tensor
    inverse_sqrt_node_degree: torch.Tensor
    node_degree: torch.Tensor
    edge_sizes: torch.Tensor
    num_nodes: int
    num_edges: int

    @classmethod
    def from_edge_index(cls, edge_index: torch.Tensor, num_nodes: int, device: torch.device) -> "IncidenceOperator":
        node_ids = edge_index[0].long().to(device)
        _, edge_ids = torch.unique(edge_index[1].long().to(device), sorted=True, return_inverse=True)
        num_edges = int(edge_ids.max().item()) + 1
        node_degree = torch.bincount(node_ids, minlength=num_nodes).float().clamp_min(1.0)
        edge_sizes = torch.bincount(edge_ids, minlength=num_edges).float().clamp_min(1.0)
        return cls(node_ids, edge_ids, node_degree.rsqrt(), node_degree, edge_sizes, num_nodes, num_edges)

    def edge_mean(self, features: torch.Tensor) -> torch.Tensor:
        output = features.new_zeros((self.num_edges, features.size(-1)))
        output.index_add_(0, self.edge_ids, features[self.node_ids])
        return output / self.edge_sizes.unsqueeze(-1)

    def gradient(self, features: torch.Tensor) -> torch.Tensor:
        normalized_features = features * self.inverse_sqrt_node_degree.unsqueeze(-1)
        normalized_edge_mean = self.edge_mean(normalized_features)
        return normalized_features[self.node_ids] - normalized_edge_mean[self.edge_ids]

    def divergence(self, incidence_features: torch.Tensor) -> torch.Tensor:
        scale = self.inverse_sqrt_node_degree[self.node_ids].unsqueeze(-1)
        direct = incidence_features.new_zeros((self.num_nodes, incidence_features.size(-1)))
        direct.index_add_(0, self.node_ids, incidence_features * scale)

        edge_sum = incidence_features.new_zeros((self.num_edges, incidence_features.size(-1)))
        edge_sum.index_add_(0, self.edge_ids, incidence_features)

        correction = incidence_features.new_zeros((self.num_nodes, incidence_features.size(-1)))
        correction.index_add_(
            0,
            self.node_ids,
            edge_sum[self.edge_ids] / self.edge_sizes[self.edge_ids].unsqueeze(-1) * scale,
        )
        return direct - correction


def edge_list_to_index(edge_list, num_nodes: int, device: torch.device) -> torch.Tensor:
    nodes = []
    edges = []
    for edge_id, edge in enumerate(edge_list):
        clean = sorted({int(node) for node in edge if 0 <= int(node) < num_nodes})
        for node in clean:
            nodes.append(node)
            edges.append(edge_id)
    if not nodes:
        nodes = list(range(num_nodes))
        edges = list(range(num_nodes))
    return torch.tensor([nodes, edges], dtype=torch.long, device=device)
