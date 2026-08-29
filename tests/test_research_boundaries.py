"""Prevent the independent research tracks from silently recombining."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _imports(folder: str) -> set[str]:
    imported: set[str] = set()
    for path in (ROOT / "research" / folder).rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
    return imported


def _assert_no_prefix(imports: set[str], forbidden: tuple[str, ...]) -> None:
    violations = sorted(
        module for module in imports if any(module.startswith(prefix) for prefix in forbidden)
    )
    assert not violations, f"cross-track imports found: {violations}"


def test_conductance_gat_does_not_import_cycle_or_combined_tracks() -> None:
    _assert_no_prefix(
        _imports("conductance_gat"),
        (
            "research.cycle_pe",
            "research.tree_augmentation",
            "research.combined_later",
            "chartgat.completion",
            "chartgat.layers",
        ),
    )


def test_cycle_pe_does_not_import_conductance_or_combined_tracks() -> None:
    _assert_no_prefix(
        _imports("cycle_pe"),
        (
            "research.conductance_gat",
            "research.tree_augmentation",
            "research.combined_later",
            "chartgat.completion",
            "chartgat.layers",
        ),
    )


def test_tree_augmentation_depends_on_neither_conductance_nor_combined_track() -> None:
    _assert_no_prefix(
        _imports("tree_augmentation"),
        (
            "research.conductance_gat",
            "research.combined_later",
            "chartgat.completion",
            "chartgat.layers",
        ),
    )
