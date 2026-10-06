from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

class RouteAhead(nn.Module):
    """Predict only; never connects predicted IDs to the language model dispatch."""
    def __init__(self, d_model: int, experts: int, capacity_layers: int, cfg: dict):
        super().__init__(); self.cfg = cfg; self.experts = experts; self.layers = capacity_layers
        features = 2 * d_model + cfg['history'] * capacity_layers * experts
        self.network = nn.Sequential(nn.Linear(features, cfg['hidden']), nn.SiLU(), nn.Linear(cfg['hidden'], cfg['horizons'] * capacity_layers * experts))
    def forward(self, hidden: torch.Tensor, draft: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        x = torch.cat((hidden, draft, history.flatten(-3)), -1)
        return self.network(x).view(*hidden.shape[:-1], self.cfg['horizons'], self.layers, self.experts)
    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Targets are multi-hot, supporting Top2 rather than discarding its second expert.
        return F.binary_cross_entropy_with_logits(logits, targets.float())

def prediction_metrics(predicted: list[list[int]], actual: list[list[int]], mapping: dict[int, int]) -> dict:
    correct = total_actual = total_predicted = page_correct = page_total = 0
    for p, a in zip(predicted, actual):
        ps, ac = set(p), set(a); correct += len(ps & ac); total_actual += len(ac); total_predicted += len(ps)
        pp, ap = {mapping[e] for e in ps}, {mapping[e] for e in ac}; page_correct += len(pp & ap); page_total += len(ap)
    return {'expert_recall': correct / max(total_actual, 1), 'precision': correct / max(total_predicted, 1), 'page_recall': page_correct / max(page_total, 1)}
