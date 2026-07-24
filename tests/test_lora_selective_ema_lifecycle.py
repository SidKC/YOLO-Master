import math
import io
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from ultralytics.engine.extensions.adapters import AdapterRuntimeController
from ultralytics.engine.extensions.recovery import TrainingRecoveryController
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils.lora.fallback import ManualLoRAConv
from ultralytics.utils.torch_utils import ModelEMA


class TinyFallbackGraph(nn.Module):
    def __init__(self, count: int = 1, *, use_rslora: bool = True):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ManualLoRAConv(
                    nn.Conv2d(4, 4, 1),
                    r=8,
                    alpha=16,
                    dropout=0.05,
                    use_rslora=use_rslora,
                )
                for _ in range(count)
            ]
        )
        self.lora_enabled = True
        self.lora_config = SimpleNamespace(r=8, alpha=16)
        self.lora_runtime_metadata = {
            "effective_backend": "fallback",
            "requested_use_rslora": use_rslora,
            "effective_use_rslora": use_rslora,
        }
        for name, parameter in self.named_parameters():
            parameter.requires_grad = name.endswith((".lora_A", ".lora_B"))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def _trainer(model: nn.Module, *, warmup: int) -> SimpleNamespace:
    optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.1)
    return SimpleNamespace(
        model=model,
        optimizer=optimizer,
        epochs=7,
        ema=None,
        lora_strategy=None,
        args=SimpleNamespace(
            lora_type="lora",
            lora_alpha_warmup=warmup,
            lora_layer_decay=0.0,
            lora_ortho_weight=0.0,
            lora_ortho_frequency=10,
            lora_dropout=0.05,
            lora_dropout_end=0.05,
            lora_dropout_start_ratio=1.0,
        ),
    )


def test_adapter_only_ema_uses_recurrence_for_adapter_and_exact_copy_for_frozen_state():
    model = TinyFallbackGraph()
    ema = ModelEMA(model)
    adapter_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert adapter_names == {"layers.0.lora_A", "layers.0.lora_B"}
    ema.configure_adapter_only(model, adapter_names)

    ema_before = {name: value.clone() for name, value in ema.ema.state_dict().items()}
    with torch.no_grad():
        for name, value in model.state_dict().items():
            value.add_(1.0 if name in adapter_names else 2.0)
    decay = ema.decay(1)
    ema.update(model)

    for name, value in ema.ema.state_dict().items():
        online = model.state_dict()[name]
        if name in adapter_names:
            expected = ema_before[name] * decay + online * (1.0 - decay)
            torch.testing.assert_close(value, expected)
        else:
            torch.testing.assert_close(value, online, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("use_rslora", "warmup"),
    ((False, 0), (False, 5), (True, 0), (True, 5)),
)
def test_realized_scaling_and_ema_attributes_follow_v4_schedule(use_rslora: bool, warmup: int):
    model = TinyFallbackGraph(count=62, use_rslora=use_rslora)
    trainer = _trainer(model, warmup=warmup)
    controller = AdapterRuntimeController(trainer)
    controller.configure_optimizer(trainer.optimizer)
    trainer.ema = ModelEMA(model)
    controller.configure_ema(trainer.ema, trainer.optimizer)

    assert trainer.ema.adapter_state_names == {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert len(trainer.ema.adapter_state_names) == 124
    assert trainer.args.effective_lora_ema_policy == "adapter_only_exact_copy_v1"
    full_scale = 16 / math.sqrt(8) if use_rslora else 16 / 8

    for epoch in range(7):
        controller.begin_epoch(epoch)
        expected_factor = (
            0.5 * (1 - math.cos(math.pi * min(epoch / 5, 1.0)))
            if warmup
            else 1.0
        )
        online_modules = dict(model.named_modules())
        ema_modules = dict(trainer.ema.ema.named_modules())
        wrappers = [name for name, module in online_modules.items() if isinstance(module, ManualLoRAConv)]
        assert len(wrappers) == 62
        for name in wrappers:
            online = online_modules[name]
            averaged = ema_modules[name]
            assert online.scaling == pytest.approx(full_scale * expected_factor)
            assert averaged.scaling == pytest.approx(online.scaling)
            assert (averaged.use_rslora, averaged.r, averaged.alpha) == (
                online.use_rslora,
                online.r,
                online.alpha,
            )


def test_resume_after_warmup_restores_full_scale_online_and_ema():
    model = TinyFallbackGraph(use_rslora=True)
    trainer = _trainer(model, warmup=5)
    controller = AdapterRuntimeController(trainer)
    controller.configure_optimizer(trainer.optimizer)
    trainer.ema = ModelEMA(model)
    controller.configure_ema(trainer.ema, trainer.optimizer)

    assert model.layers[0].scaling == 0.0
    assert trainer.ema.ema.layers[0].scaling == 0.0
    controller.restore_after_resume(start_epoch=6)

    expected = 16 / math.sqrt(8)
    assert model.layers[0].scaling == pytest.approx(expected)
    assert trainer.ema.ema.layers[0].scaling == pytest.approx(expected)


def test_explicit_treatment_sync_repairs_non_state_ema_attributes():
    model = TinyFallbackGraph(use_rslora=True)
    trainer = _trainer(model, warmup=0)
    controller = AdapterRuntimeController(trainer)
    controller.configure_optimizer(trainer.optimizer)
    trainer.ema = ModelEMA(model)
    controller.configure_ema(trainer.ema, trainer.optimizer)
    trainer.ema.ema.layers[0].scaling = 0.0
    trainer.ema.ema.layers[0].use_rslora = False

    assert controller.sync_ema_treatment() == 1
    assert trainer.ema.ema.layers[0].scaling == model.layers[0].scaling
    assert trainer.ema.ema.layers[0].use_rslora is True


def test_validation_syncs_ema_treatment_before_selecting_ema_model():
    model = TinyFallbackGraph(use_rslora=True)
    trainer = _trainer(model, warmup=0)
    controller = AdapterRuntimeController(trainer)
    trainer.adapter_controller = controller
    controller.configure_optimizer(trainer.optimizer)
    trainer.ema = ModelEMA(model)
    controller.configure_ema(trainer.ema, trainer.optimizer)
    trainer.ema.ema.layers[0].scaling = 0.0
    trainer._sync_ema_buffers_for_validation = lambda: None
    trainer._state_is_finite = lambda _value: True
    trainer.best_fitness = None
    trainer.loss = torch.tensor(1.0)

    def validator(runtime_trainer):
        assert runtime_trainer.ema.ema.layers[0].scaling == model.layers[0].scaling
        return {"fitness": 1.0}

    trainer.validator = validator
    metrics, fitness = BaseTrainer.validate(trainer)

    assert metrics == {}
    assert fitness == 1.0


def test_checkpoint_serialization_syncs_non_state_ema_treatment():
    model = TinyFallbackGraph(use_rslora=True)
    trainer = _trainer(model, warmup=0)
    controller = AdapterRuntimeController(trainer)
    trainer.adapter_controller = controller
    controller.configure_optimizer(trainer.optimizer)
    trainer.ema = ModelEMA(model)
    controller.configure_ema(trainer.ema, trainer.optimizer)
    trainer.ema.ema.layers[0].scaling = 0.0
    trainer.scaler = SimpleNamespace(state_dict=lambda: {})
    trainer.epoch = 0
    trainer.start_epoch = 0
    trainer.best_fitness = 0.0
    trainer.metrics = {}
    trainer.fitness = 0.0
    trainer.read_results_csv = lambda: {}

    serialized = TrainingRecoveryController(trainer).serialize_checkpoint(include_online_model=True)
    checkpoint = torch.load(io.BytesIO(serialized), map_location="cpu", weights_only=False)

    assert checkpoint["ema"].layers[0].scaling == model.layers[0].scaling
    assert checkpoint["ema"].layers[0].use_rslora is True
