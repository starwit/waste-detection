from __future__ import annotations

"""Project-level params/config integration checks.

These tests stay intentionally narrow:
- validate the real project params schema,
- ensure every configured model resolves against the trainer API,
- ensure the project declares at least one model for every supported backend.

Generic pipeline contract coverage lives in `object-detector-trainer`.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from object_detector_trainer.backends.registry import (
    normalize_backend_name,
    required_resolved_fields,
    supported_backend_names,
)
from object_detector_trainer.backends.training_config import resolve_training_config
from object_detector_trainer.config.loader import load_config


_PARAMS_YAML = Path(__file__).parents[2] / "params.yaml"

def _discover_project_model_expectations() -> list[tuple[str, str, list[str]]]:
    cfg = load_config(_PARAMS_YAML)
    expectations: list[tuple[str, str, list[str]]] = []
    for model_key in sorted(cfg.models):
        model_cfg = cfg.models[model_key]
        backend = normalize_backend_name(model_cfg["backend"])
        expectations.append(
            (
                str(model_key),
                backend,
                list(required_resolved_fields(backend)),
            )
        )
    return expectations


# ---------------------------------------------------------------------------
# Test group 1 — Config resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_key,expected_backend,required_keys",
    _discover_project_model_expectations(),
)
def test_project_model_config_resolves(
    model_key: str, expected_backend: str, required_keys: list[str]
) -> None:
    """Every model key in params.yaml resolves to the correct backend with required fields."""
    cfg = load_config(_PARAMS_YAML)
    args = SimpleNamespace(model=model_key, seed=42, set=[])
    resolved = resolve_training_config(args, cfg)

    assert resolved["backend"] == expected_backend, (
        f"Expected backend {expected_backend!r} for {model_key!r}, got {resolved['backend']!r}"
    )
    for key in required_keys:
        assert resolved.get(key) is not None, (
            f"Missing or None required key {key!r} for model {model_key!r}"
        )


def test_project_params_schema_validates() -> None:
    """The real params.yaml loads without error and has the expected project structure."""
    cfg = load_config(_PARAMS_YAML)
    configured_backends = {
        normalize_backend_name(model_cfg["backend"])
        for model_cfg in cfg.models.values()
    }

    assert cfg.data.custom_classes, "custom_classes must be non-empty"
    assert not cfg.data.use_coco_classes, "use_coco_classes must be False"
    assert cfg.data.class_mapping, "class_mapping must be non-empty"
    assert cfg.train.model in cfg.models, (
        f"train.model={cfg.train.model!r} is not present in the models dict"
    )
    assert cfg.prepare.folder_subsets, "folder_subsets must be non-empty"
    assert configured_backends == set(supported_backend_names()), (
        "params.yaml must declare at least one model for every supported backend. "
        f"Configured={sorted(configured_backends)}, supported={sorted(supported_backend_names())}"
    )
