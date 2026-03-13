from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--heavy",
        action="store_true",
        default=False,
        help="Run heavy project integration tests (slow DVC/pipeline wiring checks).",
    )


def _explicit_heavy_selection(config: pytest.Config, items: list[pytest.Item]) -> bool:
    markexpr = str(getattr(config.option, "markexpr", "") or "").strip()
    if markexpr == "heavy":
        return True
    return bool(items) and all("heavy" in item.keywords for item in items)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--heavy"):
        return
    if _explicit_heavy_selection(config, items):
        raise pytest.UsageError(
            "Heavy tests require the explicit `--heavy` flag. "
            "Use `pytest --heavy` (and optionally add a nodeid), not `pytest -m heavy`."
        )

    skip_heavy = pytest.mark.skip(reason="Need --heavy option to run heavy tests.")
    for item in items:
        if "heavy" in item.keywords:
            item.add_marker(skip_heavy)
