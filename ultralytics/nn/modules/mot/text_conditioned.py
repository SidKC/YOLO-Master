"""Text-conditioned sparse two-expert routing for detector feature maps.

This block is intentionally small and explicit: a frozen text embedding conditions an
image-level router, and exactly one of two feature experts is executed for each sample.
The block is an ``nn.Module`` that can be placed on a detector feature path from YAML.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.routing_protocol import (
    current_aux_step,
    export_capabilities as _export_routing_capabilities,
    publish_aux_loss,
    routing_snapshot as _routing_snapshot,
)

from .router import _MoTRouter


class TextConditionedMoT(nn.Module):
    """Route a detector feature map through one of two text-conditioned experts.

    ``forward`` accepts a detached text representation for the current batch. A zero
    vector is used when no condition is supplied.

    Args:
        c1: Input feature channels.
        c2: Output feature channels.
        text_dim: Frozen text embedding width.
        hidden_dim: Router and condition projection width.
        balance_loss_coeff: Weight of the graph-connected routing auxiliary loss.
    """

    NUM_EXPERTS = 2
    publishes_aux_loss = True

    def __init__(
        self,
        c1: int,
        c2: int,
        text_dim: int = 512,
        hidden_dim: int = 64,
        balance_loss_coeff: float = 0.01,
    ):
        super().__init__()
        if c1 <= 0 or c2 <= 0:
            raise ValueError(f"feature channels must be positive, got c1={c1}, c2={c2}")
        if text_dim <= 0 or hidden_dim <= 0:
            raise ValueError(f"text_dim and hidden_dim must be positive, got {text_dim}, {hidden_dim}")
        if balance_loss_coeff < 0:
            raise ValueError(f"balance_loss_coeff must be non-negative, got {balance_loss_coeff}")

        self.text_dim = int(text_dim)
        self.hidden_dim = int(hidden_dim)
        self.balance_loss_coeff = float(balance_loss_coeff)
        self._top_k = 1

        self.input_projection = nn.Identity() if c1 == c2 else nn.Conv2d(c1, c2, 1, bias=False)
        self.condition_projection = nn.Linear(self.text_dim, self.hidden_dim)
        self.visual_projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c2, self.hidden_dim),
            nn.SiLU(inplace=False),
        )
        self.router = nn.Linear(self.hidden_dim * 2, self.NUM_EXPERTS)
        self.experts = nn.ModuleList([self._make_expert(c2) for _ in range(self.NUM_EXPERTS)])
        self.output_projection = nn.Conv2d(c2, c2, 1, bias=False)

        # The persistent zero fallback keeps model construction/device movement
        # deterministic without making a caller-owned condition part of state_dict.
        self.register_buffer("zero_condition", torch.zeros(1, self.text_dim), persistent=True)
        self.last_aux_loss: torch.Tensor | None = None
        self.last_routing_snapshot: dict[str, Any] = {}
        self.last_routing_logits: torch.Tensor | None = None

    @staticmethod
    def _make_expert(channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(1, channels),
            nn.SiLU(inplace=False),
            nn.Conv2d(channels, channels, 1, bias=False),
        )

    @property
    def num_experts(self) -> int:
        return self.NUM_EXPERTS

    @property
    def top_k(self) -> int:
        return self._top_k

    def _condition_for_batch(
        self,
        condition: torch.Tensor | None,
        batch: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Validate and detach a caller-owned [D] or [B,D] condition tensor."""

        parameter = next(self.condition_projection.parameters())
        if condition is None:
            return self.zero_condition.to(device=device, dtype=parameter.dtype).expand(batch, -1)
        if not isinstance(condition, torch.Tensor):
            raise TypeError(f"condition must be a Tensor or None, got {type(condition)!r}")
        if condition.ndim == 1:
            condition = condition.unsqueeze(0)
        if condition.ndim != 2 or condition.shape[-1] != self.text_dim:
            raise ValueError(f"condition must have shape [D] or [B, {self.text_dim}], got {tuple(condition.shape)}")
        if not torch.isfinite(condition).all():
            raise ValueError("condition must contain only finite values")
        if condition.shape[0] == 1:
            condition = condition.expand(batch, -1)
        elif condition.shape[0] != batch:
            raise ValueError(f"condition batch {condition.shape[0]} does not match feature batch {batch}")
        return condition.detach().to(device=device, dtype=parameter.dtype)

    def forward(self, x: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        """Route each sample through one expert."""

        if x.ndim != 4:
            raise ValueError(f"expected feature map [B,C,H,W], got shape {tuple(x.shape)}")
        features = self.input_projection(x)
        text = self._condition_for_batch(condition, features.shape[0], features.device)
        condition = self.condition_projection(text)
        visual = self.visual_projection(features)
        logits = self.router(torch.cat((visual, condition), dim=1)).float()
        probabilities = F.softmax(logits, dim=-1)
        assignments = probabilities.argmax(dim=-1)
        selected_probability = probabilities.gather(1, assignments.unsqueeze(1)).squeeze(1)

        # Compute only the selected expert for each sample. ``index_copy`` keeps
        # the selected expert output connected to the detector loss graph.
        routed = torch.zeros_like(features)
        for expert_index, expert in enumerate(self.experts):
            sample_indices = (assignments == expert_index).nonzero(as_tuple=True)[0]
            if sample_indices.numel() == 0:
                continue
            selected = features.index_select(0, sample_indices)
            selected = expert(selected)
            routed = routed.index_copy(0, sample_indices, selected)
        routed = routed * selected_probability.to(dtype=routed.dtype).view(-1, 1, 1, 1)
        output = self.output_projection(routed) + features

        usage = F.one_hot(assignments, num_classes=self.NUM_EXPERTS).float().mean(dim=0)
        importance = probabilities.float().mean(dim=0)
        balance = self.NUM_EXPERTS * torch.sum(importance * usage)
        # Use the existing MoT balance term together with its router z-loss.
        z_loss = _MoTRouter.z_loss_from_logits(logits)
        raw_aux_loss = self.balance_loss_coeff * (balance + z_loss)
        self.last_aux_loss = raw_aux_loss if self.training else raw_aux_loss.detach().new_zeros(())
        publish_aux_loss(
            self,
            self.last_aux_loss,
            step=current_aux_step(),
            kind="mot",
            training=self.training,
        )

        self.last_routing_logits = logits.detach()
        with torch.no_grad():
            actual_expert_calls = int(assignments.numel())
            skipped_expert_calls = actual_expert_calls * (self.NUM_EXPERTS - self.top_k)
            self.last_routing_snapshot = {
                "num_experts": self.NUM_EXPERTS,
                "top_k": self.top_k,
                "expert_usage": usage.detach(),
                "mean_router_probs": importance.detach(),
                "route_indices": assignments.detach(),
                "executed_expert": assignments.detach().cpu().tolist(),
                "actual_expert_calls": actual_expert_calls,
                "skipped_expert_calls": skipped_expert_calls,
                "skipped_expert_count": skipped_expert_calls,
                "aux_loss": float(self.last_aux_loss.detach()),
                "dispatch": {
                    "policy": "sample_top1_sparse",
                    "selected_samples": int(assignments.numel()),
                    "skipped_experts": int((usage == 0).sum()),
                    "actual_expert_calls": actual_expert_calls,
                    "skipped_expert_calls": skipped_expert_calls,
                    "nontrivial_action_count": skipped_expert_calls,
                    "dense_top1_action_count": 0,
                },
            }
        return output

    @property
    def aux_loss(self) -> torch.Tensor:
        """Return the current graph-connected routing auxiliary scalar."""

        if self.last_aux_loss is not None:
            return self.last_aux_loss
        return self.zero_condition.new_zeros(())

    def publish_aux_loss(self, *, step: int, training: bool) -> torch.Tensor:
        return publish_aux_loss(self, self.aux_loss, step=step, kind="mot", training=training)

    def routing_snapshot(self) -> dict[str, Any]:
        return _routing_snapshot(self)

    def export_capabilities(self) -> dict[str, Any]:
        capabilities = _export_routing_capabilities(self)
        capabilities.update(
            routing_kind="mot",
            sparse_dispatch=True,
            eager_sparse_dispatch=True,
            training_sparse_dispatch=True,
            sparse_train=True,
            dispatch_policy="sample_top1_sparse",
            sparse_export_limitation="Data-dependent sample top-1 dispatch is eager-only.",
        )
        return capabilities


__all__ = ("TextConditionedMoT",)
