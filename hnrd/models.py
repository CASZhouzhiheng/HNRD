import torch
import torch.nn as nn
import torch.nn.functional as F

from .operators import IncidenceOperator


class IncidenceModulation(nn.Module):
    """Learn normalized diagonal weights A_theta(X) on the incidence space."""

    def __init__(self, hidden_dim: int, epsilon: float = 1e-4):
        super().__init__()
        self.epsilon = epsilon
        self.node_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.score = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor, operator: IncidenceOperator) -> torch.Tensor:
        edge_features = operator.edge_mean(features)
        context = torch.cat(
            (
                self.node_projection(features[operator.node_ids]),
                self.edge_projection(edge_features[operator.edge_ids]),
            ),
            dim=-1,
        )
        raw = self.score(context).view(-1)
        exp_raw = raw.exp()
        denom = features.new_zeros(operator.num_nodes)
        denom.index_add_(0, operator.node_ids, exp_raw)
        normalized = exp_raw / denom[operator.node_ids].clamp_min(1e-12)
        return (1.0 - self.epsilon) * normalized + self.epsilon / operator.node_degree[operator.node_ids]


class HNRD(nn.Module):
    """Hypergraph Neural Reaction-Diffusion network.

    The diffusion term follows the incidence-level operator
    -G^T A_theta(X) G X. The reaction term acts only on the transverse
    component Q_phi X through instantaneous dissipation compensation and
    bounded feedback.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        layers: int = 4,
        input_dropout: float = 0.2,
        hidden_dropout: float = 0.2,
        step_size: float = 0.5,
        learnable_step: bool = True,
    ):
        super().__init__()
        self.encoder = nn.Linear(in_dim, hidden_dim)
        self.modulation = IncidenceModulation(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, out_dim)
        self.skip = nn.Linear(in_dim, out_dim)
        self.eta = nn.Parameter(torch.tensor(1.0))
        init_step = torch.tensor(float(step_size)).clamp(1e-4, 1.0 - 1e-4)
        init_logit = torch.logit(init_step)
        if learnable_step:
            self.step_logit = nn.Parameter(init_logit)
        else:
            self.register_buffer("step_logit", init_logit)
        self.layers = layers
        self.input_dropout = input_dropout
        self.hidden_dropout = hidden_dropout

    @property
    def step_size(self):
        return torch.sigmoid(self.step_logit)

    def incidence_diffusion(self, h: torch.Tensor, operator: IncidenceOperator) -> torch.Tensor:
        gradient = operator.gradient(h)
        weights = self.modulation(h, operator)
        return -operator.divergence(weights[:, None] * gradient)

    def transverse_component(self, h: torch.Tensor, operator: IncidenceOperator) -> torch.Tensor:
        phi = operator.node_degree.sqrt()
        denom = torch.sum(phi * phi).clamp_min(1e-12)
        coefficient = torch.sum(phi[:, None] * h, dim=0, keepdim=True) / denom
        return h - phi[:, None] * coefficient

    def forward(self, x, operator: IncidenceOperator, return_representation=False):
        h = F.relu(self.encoder(F.dropout(x, self.input_dropout, self.training)))
        for _ in range(self.layers):
            diffusion = self.incidence_diffusion(h, operator)

            q_phi_x = self.transverse_component(h, operator)
            q_norm2 = torch.sum(q_phi_x * q_phi_x).clamp_min(1e-12)
            dissipation_rate = -torch.sum(diffusion * q_phi_x) / q_norm2
            bounded_feedback = torch.tanh(F.softplus(self.eta) - q_norm2)

            h = h + self.step_size * (diffusion + (dissipation_rate + bounded_feedback) * q_phi_x)
            h = F.dropout(F.relu(h), self.hidden_dropout, self.training)

        logits = self.classifier(h) + self.skip(x)
        if return_representation:
            return logits, h
        return logits
