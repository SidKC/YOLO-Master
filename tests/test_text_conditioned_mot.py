"""Tests for the detector-connected frozen-condition two-expert block."""

from pathlib import Path

import torch

from ultralytics.nn.mixture_loss import CompositeCriterion, has_routed_modules
from ultralytics.nn.modules import TextConditionedMoT
from ultralytics.nn.modules.routing_protocol import (
    clear_aux_records,
    collect_aux_loss,
    current_aux_step,
)
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
from ultralytics.utils.export_capabilities import classify_routed_module


MODEL_CFG = Path(__file__).parents[1] / "ultralytics/cfg/models/master/v0_15/det/yolo-master-b1-tiny.yaml"


def _configure_router(module: TextConditionedMoT) -> None:
    """Make the first condition coordinate deterministically select expert 0/1."""

    with torch.no_grad():
        module.condition_projection.weight.zero_()
        module.condition_projection.bias.zero_()
        module.condition_projection.weight[0, 0] = 1.0
        module.router.weight.zero_()
        module.router.bias.zero_()
        module.router.weight[0, module.hidden_dim] = 1.0
        module.router.weight[1, module.hidden_dim] = -1.0


def _mot(model: DetectionModel) -> TextConditionedMoT:
    modules = [module for module in model.modules() if isinstance(module, TextConditionedMoT)]
    assert len(modules) == 1
    return modules[0]


def _synthetic_batch(images: torch.Tensor, condition: torch.Tensor | None) -> dict[str, torch.Tensor | None]:
    return {
        "img": images,
        "text_condition": condition,
        "batch_idx": torch.arange(images.shape[0], dtype=torch.long),
        "cls": torch.zeros(images.shape[0], 1),
        "bboxes": torch.full((images.shape[0], 4), 0.5),
    }


def test_text_conditioned_mot_condition_shapes_aux_and_sparse_telemetry():
    torch.manual_seed(0)
    module = TextConditionedMoT(8, 8, text_dim=4, hidden_dim=4).train()
    _configure_router(module)
    visual = torch.randn(2, 8, 4, 4, requires_grad=True)
    condition = torch.tensor([[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]])

    clear_aux_records(step=41)
    routed = module(visual, condition=condition)
    assert module.num_experts == 2
    assert module.top_k == 1
    assert has_routed_modules(module)
    assert routed.shape == visual.shape
    assert routed.grad_fn is not None
    assert module.last_routing_snapshot["route_indices"].tolist() == [0, 1]
    assert module.last_routing_snapshot["executed_expert"] == [0, 1]
    assert module.last_routing_snapshot["actual_expert_calls"] == 2
    assert module.last_routing_snapshot["skipped_expert_calls"] == 2
    assert module.last_routing_snapshot["dispatch"]["nontrivial_action_count"] == 2
    assert module.last_routing_snapshot["dispatch"]["dense_top1_action_count"] == 0

    aux, diagnostics = collect_aux_loss(module, step=41, include_kinds=("mot",), return_diagnostics=True)
    assert diagnostics["counts_by_kind"]["mot"] == 1
    assert diagnostics["stale_skipped"] == 0
    assert diagnostics["eval_skipped"] == 0
    assert diagnostics["duplicate_skipped"] == 0
    assert aux.requires_grad

    aux.backward(retain_graph=True)
    for parameter in (module.condition_projection.weight, module.router.weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().max() > 0

    routed.square().mean().backward()
    for expert in module.experts:
        gradients = [parameter.grad for parameter in expert.parameters() if parameter.grad is not None]
        assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
        assert max(float(gradient.abs().max()) for gradient in gradients) > 0

    # A caller-owned condition is detached by the module and never becomes optimizer state.
    external = condition.clone().requires_grad_()
    external_before = external.detach().clone()
    module(external.new_zeros(2, 8, 4, 4), condition=external)
    assert external.grad is None
    assert torch.equal(external, external_before)

    # [D] broadcasts to the batch and None uses the module's default condition.
    assert module(torch.randn(2, 8, 4, 4), condition=torch.ones(4)).shape == (2, 8, 4, 4)
    assert module(torch.randn(2, 8, 4, 4), condition=None).shape == (2, 8, 4, 4)


def test_text_conditioned_mot_eval_publication_is_zero_and_detached():
    module = TextConditionedMoT(4, 4, text_dim=3, hidden_dim=4).eval()
    clear_aux_records(step=51)
    output = module(torch.randn(1, 4, 3, 3), condition=torch.ones(3))
    assert output.shape == (1, 4, 3, 3)
    assert module.zero_condition.requires_grad is False
    assert module.aux_loss.requires_grad is False
    eval_aux, diagnostics = collect_aux_loss(
        module,
        step=51,
        include_kinds=("mot",),
        return_diagnostics=True,
    )
    assert eval_aux.requires_grad is False
    assert diagnostics["counts_by_kind"]["mot"] == 0
    assert diagnostics["eval_skipped"] == 1
    assert diagnostics["stale_skipped"] == 0


def test_real_detection_model_has_one_router_and_condition_reaches_detector_loss():
    torch.manual_seed(1)
    model = DetectionModel(MODEL_CFG, ch=3, nc=2, verbose=False).train()
    model.args = IterableSimpleNamespace(**DEFAULT_CFG_DICT)
    module = _mot(model)
    _configure_router(module)
    routed_families = [family for m in model.modules() if (family := classify_routed_module(m)) is not None]
    assert routed_families == ["MoT"]
    assert has_routed_modules(model)
    assert module.num_experts == 2 and module.top_k == 1

    images = torch.rand(2, 3, 64, 64)
    positive_negative = torch.tensor([[1.0] + [0.0] * 511, [-1.0] + [0.0] * 511])
    zero = torch.zeros(512)

    # Run batch size 1 once for each expert.
    for semantic, expected in ((positive_negative[:1], [0]), (positive_negative[1:], [1])):
        model.predict(images[:1], condition=semantic)
        assert module.last_routing_snapshot["executed_expert"] == expected

    # The detector head consumes the routed P5 feature; condition changes its actual output.
    conditioned = model.predict(images, condition=positive_negative)["boxes"]
    conditioned_routes = module.last_routing_snapshot["executed_expert"]
    control = model.predict(images, condition=zero)["boxes"]
    control_routes = module.last_routing_snapshot["executed_expert"]
    assert conditioned.requires_grad
    assert not torch.allclose(conditioned.detach(), control.detach())
    assert conditioned_routes == [0, 1]
    assert control_routes == [0, 0]

    batch = _synthetic_batch(images, positive_negative)
    clear_aux_records(step=71)
    loss, _ = model.loss(batch)
    assert loss.requires_grad and loss.grad_fn is not None
    control_loss, _ = model.loss(_synthetic_batch(images, zero))
    assert not torch.allclose(loss.detach(), control_loss.detach())
    clear_aux_records(step=71)
    loss, items = model.loss(batch)
    assert isinstance(model.criterion, CompositeCriterion)
    assert isinstance(items, torch.Tensor)
    assert items.ndim == 1 and items.numel() >= 4
    assert isinstance(model._last_mixture_aux_loss, torch.Tensor)
    assert torch.isfinite(model._last_mixture_aux_loss)
    assert model._last_mixture_aux_loss.detach().abs() > 0
    assert torch.allclose(items[-1], model._last_mixture_aux_loss)
    aux, diagnostics = collect_aux_loss(
        model,
        step=current_aux_step(),
        include_kinds=("mot",),
        return_diagnostics=True,
    )
    assert aux.requires_grad
    assert diagnostics["counts_by_kind"]["mot"] == 1
    assert diagnostics["stale_skipped"] == 0
    assert diagnostics["eval_skipped"] == 0
    assert diagnostics["duplicate_skipped"] == 0

    model.zero_grad(set_to_none=True)
    aux.backward(retain_graph=True)
    for parameter in (module.condition_projection.weight, module.router.weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().max() > 0

    optimizer = torch.optim.SGD(module.parameters(), lr=1e-3)
    expert_before = [[parameter.detach().clone() for parameter in expert.parameters()] for expert in module.experts]
    model.zero_grad(set_to_none=True)
    loss.sum().backward()
    assert all(
        any(parameter.grad is not None and parameter.grad.abs().max() > 0 for parameter in expert.parameters())
        for expert in module.experts
    )
    optimizer.step()
    for before, expert in zip(expert_before, module.experts):
        deltas = [(parameter.detach() - initial).abs().max() for initial, parameter in zip(before, expert.parameters())]
        assert all(torch.isfinite(delta) for delta in deltas)
        assert max(float(delta) for delta in deltas) > 0
    assert positive_negative.grad is None
