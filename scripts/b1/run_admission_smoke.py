"""Run the fixture-free B1 admission smoke tests without requiring pytest."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TEST_FILES = (
    ROOT / "tests/test_mixture_loss_composition.py",
    ROOT / "tests/test_text_conditioned_mot.py",
)


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"b1_smoke_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load test module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_artifacts() -> dict[str, bool]:
    split = json.loads((ROOT / "b1/config/coco_48_17_split.json").read_text())
    groups = [split[name] for name in ("base", "new", "unused")]
    expected_counts = (48, 17, 15)
    for group, expected in zip(groups, expected_counts):
        assert group["count"] == expected
        assert len(group["category_ids"]) == expected
        assert len(group["coco80_ids"]) == expected
        assert len(group["names"]) == expected

    for field in ("category_ids", "coco80_ids", "names"):
        sets = [set(group[field]) for group in groups]
        assert all(not sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3))
    assert set().union(*(set(group["coco80_ids"]) for group in groups)) == set(range(80))

    model_config = (
        ROOT / "ultralytics/cfg/models/master/v0_15/det/yolo-master-b1-tiny.yaml"
    ).read_text()
    assert model_config.count("TextConditionedMoT") == 1
    result = json.loads((ROOT / "b1/results/admission_result.json").read_text())
    assert result["status"] == "PASS_ADMISSION_SMOKE"
    return {
        "split_counts_48_17_15": True,
        "split_groups_pairwise_disjoint": True,
        "split_coco80_union_is_0_to_79": True,
        "model_config_has_exactly_one_text_router": True,
        "result_status_is_pass": True,
    }


def main() -> None:
    """Execute every fixture-free B1 test and print a machine-readable summary."""
    passed = []
    for path in TEST_FILES:
        module = _load_module(path)
        tests = sorted(name for name in vars(module) if name.startswith("test_"))
        for name in tests:
            getattr(module, name)()
            passed.append(f"{path.name}::{name}")

    result = {
        "status": "PASS",
        "passed_count": len(passed),
        "failed_count": 0,
        "tests": passed,
        "artifact_checks": _validate_artifacts(),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
