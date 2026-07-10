"""Issue #52 regression tests for MoE dynamic scheduling and pruning metrics."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch import nn

from ultralytics.engine.trainer import BaseTrainer
from ultralytics.nn.modules.moe.modules import ES_MOE, OptimizedMOE
from ultralytics.nn.modules.moe._common import _record_moe_snapshot, _should_record_snapshot
from ultralytics.nn.modules.moe.pruning import MoEPruner
from ultralytics.nn.modules.moe.schedule import (
    GiniBalanceScheduler,
    apply_balance_loss_coeff,
    usage_gini,
    usage_ginis_with_observation_counts_from_model,
)
from ultralytics.nn.modules.moe.utils import (
    is_core_moe_block,
    model_has_core_moe,
    reset_moe_usage_accumulator,
    set_core_moe_balance_loss_coeff,
)
from ultralytics.utils import DEFAULT_CFG_DICT
from ultralytics.nn.modules.moe.analysis import ExpertStats, ExpertUsageTracker
from scripts.diagnose_moe_pruning_signal import prepare_fraction_dataset, summarize_usage_stats
from scripts.assemble_moe_pruning_results import canonical_layer_name, percentile
from scripts.moe_pruning_sweep import compare_expert_signatures, load_signal
from scripts.plot_moe_pruning_sweep import group_plot_rows, pareto_front, plot_group_label
from scripts.run_moe_dynamic_schedule_ablation import (
    VARIANTS,
    audit_dynamic_trace,
    audit_initializations,
    model_state_sha256,
    record_initialization,
    require_fresh_run_dir,
)


def test_usage_gini_uniform_and_collapsed():
    """Gini is zero for uniform usage and high for collapsed routing."""
    assert usage_gini([0.25, 0.25, 0.25, 0.25]) == pytest.approx(0.0)
    assert usage_gini([1.0, 0.0, 0.0, 0.0]) == pytest.approx(0.75)


def test_dynamic_variants_can_audit_identical_initialization():
    """State hashes prove same-seed variants start identically and distinguish a changed seed."""
    torch.manual_seed(42)
    first = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), nn.Linear(8, 2))
    torch.manual_seed(42)
    repeated = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), nn.Linear(8, 2))
    torch.manual_seed(43)
    changed = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), nn.Linear(8, 2))

    assert model_state_sha256(first) == model_state_sha256(repeated)
    assert model_state_sha256(first) != model_state_sha256(changed)


def test_variant_initializations_use_independent_atomic_records(tmp_path):
    args = SimpleNamespace(model=Path("model.yaml"), data=Path("data.yaml"), seed=42, deterministic=True)
    expected_hash = "a" * 64
    for key in ("baseline", "dynamic", "ablation"):
        record_initialization(tmp_path, VARIANTS[key], expected_hash, args)

    audit = audit_initializations(tmp_path, list(VARIANTS.values()))

    assert audit["valid"] is True
    assert len(list(tmp_path.glob("dynamic_schedule_initialization.*.json"))) == 3
    assert {record["model_state_sha256"] for record in audit["variants"].values()} == {expected_hash}


def test_initialization_audit_accepts_legacy_combined_manifest(tmp_path):
    expected_hash = "b" * 64
    payload = {
        "model_source": "model.yaml",
        "data": "data.yaml",
        "seed": 42,
        "deterministic": True,
        "variants": {
            key: {"run_name": variant.name, "model_state_sha256": expected_hash}
            for key, variant in VARIANTS.items()
        },
    }
    (tmp_path / "dynamic_schedule_initialization.json").write_text(json.dumps(payload), encoding="utf-8")

    audit = audit_initializations(tmp_path, list(VARIANTS.values()))

    assert audit["valid"] is True
    assert audit["errors"] == []


def test_dynamic_runner_rejects_stale_fixed_name_output(tmp_path):
    run_dir = tmp_path / VARIANTS["dynamic"].name
    run_dir.mkdir()
    (run_dir / "moe_dynamic_trace.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="use a new --project"):
        require_fresh_run_dir(tmp_path, VARIANTS["dynamic"])


def _write_dynamic_trace_fixture(project: Path, epochs: list[int]) -> list[dict[str, str]]:
    run = project / "visdrone_issue52_gini_balance"
    run.mkdir(parents=True)
    records = []
    for index, epoch in enumerate(epochs, start=1):
        records.append(
            {
                "epoch": epoch,
                "layer_event_count": 4,
                "routing_observation_count": 40,
                "min_layer_observation_count": 10,
                "opportunity_count": index,
                "event_count": index,
                "nontrivial_action_count": index,
            }
        )
    (run / "moe_dynamic_trace.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    return [{"epoch": str(epoch)} for epoch in range(1, 4)]


def test_dynamic_trace_audit_accepts_one_record_per_completed_epoch(tmp_path):
    rows = _write_dynamic_trace_fixture(tmp_path, [1, 2, 3])

    audit = audit_dynamic_trace(tmp_path, rows)

    assert audit["valid"] is True
    assert audit["trace_record_count"] == 3
    assert json.loads((tmp_path / "dynamic_schedule_trace_audit.json").read_text())["valid"] is True


def test_dynamic_trace_audit_rejects_retried_epoch_duplicate(tmp_path):
    rows = _write_dynamic_trace_fixture(tmp_path, [1, 2, 2, 3])

    audit = audit_dynamic_trace(tmp_path, rows)

    assert audit["valid"] is False
    assert audit["recorded_epochs"] == [1, 2, 2, 3]
    assert "do not match accepted epochs" in audit["errors"][0]


def test_result_assembler_normalizes_peft_paths_and_interpolates_percentiles():
    assert canonical_layer_name("model.3.routing") == "model.3"
    assert canonical_layer_name("model.base_model.model.3") == "model.3"
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == pytest.approx(2.5)


def test_plot_groups_duplicate_thresholds_without_hiding_unique_structure():
    rows = [
        {"threshold": "0.05", "status": "no_op", "signature": "A", "recovery": "direct"},
        {"threshold": "0.10", "status": "duplicate", "signature": "A", "recovery": "direct"},
        {"threshold": "0.20", "status": "duplicate", "signature": "A", "recovery": "direct"},
        {"threshold": "0.30", "status": "unique", "signature": "B", "recovery": "direct"},
    ]

    groups = group_plot_rows(rows)

    assert len(groups) == 2
    assert plot_group_label(groups[0]) == "0.05-0.20/direct/no-op"
    assert plot_group_label(groups[1]) == "0.30/direct/unique"


def test_pareto_front_does_not_force_dominated_no_op_anchor():
    rows = [
        {
            "threshold": "0.05",
            "status": "no_op",
            "recovery": "direct",
            "mAP50-95": "0.20",
            "latency_p95_ms": "13",
            "gflops": "8",
        },
        {
            "threshold": "0.05",
            "status": "no_op",
            "recovery": "lora10",
            "mAP50-95": "0.18",
            "latency_p95_ms": "22",
            "gflops": "10",
        },
        {
            "threshold": "0.30",
            "status": "unique",
            "recovery": "direct",
            "mAP50-95": "0.12",
            "latency_p95_ms": "12",
            "gflops": "7",
        },
    ]

    front = pareto_front(rows)

    assert rows[0] in front
    assert rows[1] not in front
    assert rows[2] in front


def test_prepare_fraction_dataset_materializes_seeded_subset(tmp_path):
    """Calibration fractions use an explicit deterministic list instead of the ignored val fraction option."""
    images = tmp_path / "images" / "train"
    images.mkdir(parents=True)
    for index in range(10):
        (images / f"{index:02d}.jpg").touch()
    data = tmp_path / "dataset.yaml"
    data.write_text(
        "path: .\ntrain: images/train\nval: images/train\nnames: {0: object}\n",
        encoding="utf-8",
    )

    subset_yaml, available, selected = prepare_fraction_dataset(data, "train", 0.3, 42, tmp_path / "signal.json")

    assert available == 10
    assert selected == 3
    subset = yaml.safe_load(subset_yaml.read_text(encoding="utf-8"))
    image_list = Path(subset["train"])
    first_selection = image_list.read_text(encoding="utf-8")
    _, _, repeated_selected = prepare_fraction_dataset(data, "train", 0.3, 42, tmp_path / "repeat.json")
    repeated = (tmp_path / "repeat.images.txt").read_text(encoding="utf-8")
    assert repeated_selected == 3
    assert first_selection == repeated


def test_prepare_fraction_dataset_preserves_image_symlink_path(tmp_path):
    """Image lists retain the logical images path so Ultralytics can derive the converted labels path."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "frame.jpg").touch()
    (source / "other.jpg").touch()
    images = tmp_path / "images" / "train"
    images.mkdir(parents=True)
    (images / "frame.jpg").symlink_to(source / "frame.jpg")
    (images / "other.jpg").symlink_to(source / "other.jpg")
    data = tmp_path / "dataset.yaml"
    data.write_text(
        "path: .\ntrain: images/train\nval: images/train\nnames: {0: object}\n",
        encoding="utf-8",
    )

    _, _, selected = prepare_fraction_dataset(data, "train", 0.5, 42, tmp_path / "signal.json")

    listed = (tmp_path / "signal.images.txt").read_text(encoding="utf-8").strip()
    assert selected == 1
    assert listed in {str(images / "frame.jpg"), str(images / "other.jpg")}


def test_sweep_rejects_signal_without_positive_event_gates(tmp_path):
    """A structurally valid signal cannot enter the sweep unless routing was actually observed."""
    signal = tmp_path / "signal.json"
    signal.write_text(
        json.dumps(
            {
                "opportunity_count": 1,
                "event_count": 1,
                "nontrivial_signal_layer_count": 0,
                "layers": {"model.3.routing": {"experts": [{"expert_id": 0, "soft_contribution": 1.0}]}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="positive-event gates"):
        load_signal(signal)


def test_gini_balance_scheduler_clamps_and_updates_modules():
    """High Gini increases balance loss while clamp bounds are respected."""
    scheduler = GiniBalanceScheduler(base=1.0, target=0.25, alpha=2.0, beta=0.0, min_coeff=0.5, max_coeff=2.0)
    high = scheduler.update(0.75)
    low = scheduler.update(0.0)
    assert high == pytest.approx(2.0)
    assert 0.5 <= low < 1.0

    module = OptimizedMOE(32, 32, num_experts=4, top_k=2)
    updated = apply_balance_loss_coeff(module, 1.5)
    assert updated >= 1
    assert module.balance_loss_coeff == pytest.approx(1.5)
    assert module.moe_loss_fn.balance_loss_coeff == pytest.approx(1.5)


def test_dynamic_schedule_is_disabled_by_default():
    """The dynamic schedule is opt-in for backward compatibility."""
    assert DEFAULT_CFG_DICT["moe_dynamic_schedule"] == "none"
    assert DEFAULT_CFG_DICT["moe_dynamic_gini_target"] == pytest.approx(0.25)
    assert DEFAULT_CFG_DICT["preserve_checkpoint_structure"] is False


def test_forced_snapshot_bypasses_sampling_interval_once():
    """Dynamic scheduling can request one observation without changing global sampling."""
    module = nn.Identity()
    module._force_moe_snapshot = True

    assert _should_record_snapshot(module)
    assert module._force_moe_snapshot is False


def test_dynamic_epoch_usage_accumulates_every_forward():
    """Dynamic scheduling uses an epoch aggregate even when visible snapshots remain interval-sampled."""
    module = nn.Identity()
    reset_moe_usage_accumulator(module)
    module._force_moe_snapshot = True

    _record_moe_snapshot(module, expert_usage=torch.tensor([1.0, 0.0, 0.0]))
    _record_moe_snapshot(module, expert_usage=torch.tensor([0.0, 1.0, 0.0]))

    values = usage_ginis_with_observation_counts_from_model(module)
    assert len(values) == 1
    assert values[0][0] == pytest.approx(1 / 3)
    assert values[0][1] == 2


def test_trainer_dynamic_schedule_emits_nontrivial_event_trace(tmp_path):
    """A routing snapshot must create an observable scheduler event and action."""

    class SnapshotModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.balance_loss_coeff = 1.0
            self.last_routing_snapshot = {"expert_usage": torch.tensor([1.0, 0.0, 0.0, 0.0])}

    layer = SnapshotModule()
    trainer = SimpleNamespace(
        args=SimpleNamespace(
            moe_dynamic_schedule="gini_balance",
            moe_dynamic_gini_target=0.25,
            moe_dynamic_gini_alpha=1.0,
            moe_dynamic_gini_beta=0.8,
            moe_dynamic_balance_min=0.5,
            moe_dynamic_balance_max=2.0,
        ),
        model=nn.Sequential(layer),
        save_dir=tmp_path,
        moe_dynamic_scheduler=None,
        moe_dynamic_metrics={},
        moe_dynamic_counts={"opportunity_count": 0, "event_count": 0, "nontrivial_action_count": 0},
        moe_dynamic_previous_coeff=None,
        _arm_moe_dynamic_snapshots=lambda: 0,
    )

    BaseTrainer._setup_moe_dynamic_scheduler(trainer, base_balance_loss=1.0)
    BaseTrainer._update_moe_dynamic_scheduler(trainer, epoch=0)

    assert trainer.moe_dynamic_counts == {
        "opportunity_count": 1,
        "event_count": 1,
        "nontrivial_action_count": 1,
    }
    assert layer.balance_loss_coeff > 1.0
    record = json.loads((tmp_path / "moe_dynamic_trace.jsonl").read_text().strip())
    assert record["layer_event_count"] == 1
    assert record["routing_observation_count"] == 1
    assert record["min_layer_observation_count"] == 1
    assert record["max_layer_observation_count"] == 1
    assert record["updated_modules"] == 1
    assert record["balance_loss_coeff"] == pytest.approx(layer.balance_loss_coeff)


def test_dynamic_schedule_rejects_unaggregated_ddp_usage():
    trainer = SimpleNamespace(
        args=SimpleNamespace(moe_dynamic_schedule="gini_balance"),
        world_size=2,
        moe_dynamic_scheduler=None,
        moe_dynamic_metrics={},
    )

    with pytest.raises(RuntimeError, match="not aggregated across DDP ranks"):
        BaseTrainer._setup_moe_dynamic_scheduler(trainer, base_balance_loss=1.0)


def test_recovered_epoch_resets_observations_without_updating_scheduler():
    """A retried epoch must not advance the scheduler or append a duplicate event."""
    calls = {"arm": 0, "update": 0}
    trainer = SimpleNamespace(
        moe_dynamic_scheduler=object(),
        moe_dynamic_metrics={"stale": 1},
        _arm_moe_dynamic_snapshots=lambda: calls.__setitem__("arm", calls["arm"] + 1),
        _update_moe_dynamic_scheduler=lambda epoch: calls.__setitem__("update", calls["update"] + 1),
    )

    accepted = BaseTrainer._finalize_moe_dynamic_epoch(trainer, epoch=8, recovered=True)

    assert accepted is False
    assert trainer.moe_dynamic_metrics == {}
    assert calls == {"arm": 1, "update": 0}


def test_accepted_epoch_updates_scheduler_once():
    """An accepted epoch advances the dynamic scheduler exactly once."""
    calls = {"arm": 0, "update": 0}
    trainer = SimpleNamespace(
        moe_dynamic_scheduler=object(),
        moe_dynamic_metrics={},
        _arm_moe_dynamic_snapshots=lambda: calls.__setitem__("arm", calls["arm"] + 1),
        _update_moe_dynamic_scheduler=lambda epoch: calls.__setitem__("update", calls["update"] + 1),
    )

    accepted = BaseTrainer._finalize_moe_dynamic_epoch(trainer, epoch=8, recovered=False)

    assert accepted is True
    assert calls == {"arm": 0, "update": 1}


def test_es_moe_get_gflops_reports_nonzero_total():
    """The pruning script can collect ES_MOE FLOPs via get_gflops()."""
    module = ES_MOE(32, 32, num_experts=3, top_k=2)
    gflops = module.get_gflops((1, 32, 16, 16))
    assert isinstance(gflops, dict)
    assert gflops["total_gflops"] > 0
    routing_weights = torch.tensor([0.8, 0.1, 0.1]).view(1, 3, 1, 1)
    baseline_loss = module._compute_load_balancing_loss(routing_weights)
    assert apply_balance_loss_coeff(module, 1.5) >= 1
    assert module.balance_loss_coeff == pytest.approx(1.5)
    scaled_loss = module._compute_load_balancing_loss(routing_weights)
    torch.testing.assert_close(scaled_loss, 1.5 * baseline_loss)


def test_es_moe_repairs_legacy_balance_loss_coeff_on_forward():
    """Release checkpoints predating the coefficient remain loadable."""
    module = ES_MOE(32, 32, num_experts=3).eval()
    del module.balance_loss_coeff

    output = module(torch.randn(1, 32, 8, 8))

    assert output.shape == (1, 32, 8, 8)
    assert module.balance_loss_coeff == pytest.approx(1.0)


def test_trainer_config_injection_repairs_legacy_core_moe_state():
    """Legacy release checkpoints count as injected even when the pickled block lacks a new attribute."""
    module = ES_MOE(8, 8, num_experts=3, top_k=2)
    module.moe_loss_fn = SimpleNamespace(balance_loss_coeff=0.5)
    del module.balance_loss_coeff

    assert set_core_moe_balance_loss_coeff(module, 1.25)
    assert module.balance_loss_coeff == pytest.approx(1.25)
    assert module.moe_loss_fn.balance_loss_coeff == pytest.approx(1.25)


def test_es_moe_pruning_resizes_router_and_usage_buffer():
    """A 3-to-2 surgery remains dimensionally valid in train and eval."""
    module = ES_MOE(32, 32, num_experts=3)
    module.requires_grad_(False)
    pruner = MoEPruner("dummy.pt")

    pruner._prune_experts(module, [0, 2])
    assert pruner._prune_router_weights(module.routing, [0, 2], 3)
    pruner._validate_pruned_module(module)
    assert not any(parameter.requires_grad for parameter in module.routing.parameters())

    module.train()
    assert module(torch.randn(1, 32, 8, 8)).shape == (1, 32, 8, 8)
    assert module.expert_usage_counts.shape == (2,)
    module.eval()
    assert module(torch.randn(1, 32, 8, 8)).shape == (1, 32, 8, 8)


def test_split_es_moe_is_detected_as_core_block():
    """Trainer integration must recognize classes after modules.py was split."""
    module = ES_MOE(32, 32, num_experts=3, top_k=2)
    assert is_core_moe_block(module)
    assert model_has_core_moe(nn.Sequential(module))


def test_moe_pruner_usage_weight_score_preserves_usage_default():
    """The optional weighted score distinguishes equally selected experts."""
    stats = [
        SimpleNamespace(hits=10.0, avg_weight=0.2),
        SimpleNamespace(hits=10.0, avg_weight=0.4),
    ]

    usage_pruner = MoEPruner("dummy.pt")
    weighted_pruner = MoEPruner("dummy.pt", importance_mode="usage_weight", keep_top_m=1)

    assert usage_pruner._expert_score(stats[0], 20.0) == pytest.approx(0.5)
    assert usage_pruner._expert_score(stats[1], 20.0) == pytest.approx(0.5)
    assert weighted_pruner._expert_score(stats[0], 20.0) == pytest.approx(0.1)
    assert weighted_pruner._expert_score(stats[1], 20.0) == pytest.approx(0.2)


def test_soft_contribution_exposes_signal_hidden_by_hard_hits():
    """Equal hard hits can still carry unequal accumulated routing mass."""
    stats = {
        "model.3.routing": {
            0: ExpertStats(hits=10.0, weighted_sum=2.0),
            1: ExpertStats(hits=10.0, weighted_sum=8.0),
        }
    }

    layer = summarize_usage_stats(stats, {"model.3.routing": 2})["model.3.routing"]

    assert layer["hard_usage_gini"] == pytest.approx(0.0)
    assert layer["soft_contribution_gini"] == pytest.approx(0.3)
    assert [expert["soft_contribution"] for expert in layer["experts"]] == pytest.approx([0.2, 0.8])


def test_pruner_reuses_hash_matched_signal_json(tmp_path):
    """All thresholds can reuse one immutable calibration artifact."""
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    signal = tmp_path / "signal.json"
    signal.write_text(
        json.dumps(
            {
                "model_sha256": MoEPruner._sha256_file(checkpoint),
                "layers": {
                    "0.routing": {
                        "experts": [
                            {"expert_id": 0, "soft_contribution": 0.35},
                            {"expert_id": 1, "soft_contribution": 0.25},
                            {"expert_id": 2, "soft_contribution": 0.40},
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    pruner = MoEPruner(str(checkpoint), threshold=0.30, importance_mode="soft_contribution", signal_json=str(signal))
    pruner.model = SimpleNamespace(model=nn.Sequential(ES_MOE(32, 32, num_experts=3)))

    pruner._load_pruning_plan_from_signal_json()

    assert pruner.pruning_plan == {"0.routing": [0, 2]}
    assert pruner.signal_sha256 == MoEPruner._sha256_file(signal)


def test_tracker_does_not_hook_lora_children_under_router():
    """PEFT containers below a router are not independent routing events."""
    tracker = ExpertUsageTracker.__new__(ExpertUsageTracker)

    assert not tracker._is_router_module("model.3.routing.lora_A.default", nn.Dropout())


def test_compare_expert_signatures_detects_lora_structure_mismatch():
    """LoRA recovery should report when a pruned structure is rebuilt."""
    pruned = [("model.3", 2, 2), ("model.6", 2, 2)]
    preserved = [("model.3", 2, 2), ("model.6", 2, 2)]
    rebuilt = [("model.3", 3, 3), ("model.6", 3, 3)]

    assert compare_expert_signatures(pruned, preserved) == ("preserved", "")

    status, note = compare_expert_signatures(pruned, rebuilt)
    assert status == "structure_mismatch"
    assert "reference=2:2/2:2" in note
    assert "candidate=3:3/3:3" in note
