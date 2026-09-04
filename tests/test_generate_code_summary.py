"""Regression tests for the generated GPT handoff source inventory."""

from __future__ import annotations

from pathlib import Path

from scripts.generate_code_summary import discover_sources, render_summary


def test_pytest_temporary_trees_are_excluded_from_handoff(tmp_path: Path) -> None:
    included = tmp_path / "keep.py"
    excluded = tmp_path / ".pytest-tmp-run" / "fixture.py"
    included.write_text("VALUE = 1\n", encoding="utf-8")
    excluded.parent.mkdir()
    excluded.write_text("LEAKED_FIXTURE = True\n", encoding="utf-8")

    sources = [path.relative_to(tmp_path).as_posix() for path in discover_sources(tmp_path)]
    rendered, rendered_sources = render_summary(tmp_path)

    assert sources == ["keep.py"]
    assert rendered_sources == ["keep.py"]
    assert "VALUE = 1" in rendered
    assert "LEAKED_FIXTURE" not in rendered
