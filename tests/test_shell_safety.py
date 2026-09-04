from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash")
SHELL_SCRIPTS = tuple(
    sorted(
        path
        for base in (ROOT / "scripts", ROOT / "research")
        for path in base.rglob("*.sh")
    )
)
SOURCE_LIBRARY = ROOT / "scripts" / "conda_env.sh"
ENTRYPOINTS = tuple(path for path in SHELL_SCRIPTS if path != SOURCE_LIBRARY)


@pytest.mark.parametrize("script", SHELL_SCRIPTS, ids=lambda path: str(path.relative_to(ROOT)))
def test_shell_scripts_have_no_session_terminators_or_option_pollution(script: Path) -> None:
    source = script.read_text(encoding="utf-8")
    forbidden = {
        "strict shell options": r"(?m)^\s*set\s+-[^\n#]*(?:e|u)|\bpipefail\b",
        "explicit shell exit": r"(?m)(?:^|[;&|])\s*exit(?:\s|;|$)",
        "process replacement": r"(?m)(?:^|[;&|])\s*exec\s+",
        "session or server termination": (
            r"(?im)(?:^|[;&|])\s*(?:logout|shutdown|reboot|poweroff|halt|killall)\b"
            r"|\bsystemctl\s+(?:reboot|poweroff)\b|\btmux\s+kill-server\b"
            r"|\bkill\s+-9\s+-1\b|\bpkill\s+-(?:u|f)\b"
        ),
        "destructive recursive removal": r"(?im)(?:^|[;&|])\s*rm\s+-[^\s]*r[^\s]*f",
        "destructive git reset or clean": r"(?im)\bgit\s+(?:reset\s+--hard|clean\s+-[^\s]*f)",
    }
    matches = {
        label: match.group(0)
        for label, pattern in forbidden.items()
        if (match := re.search(pattern, source)) is not None
    }
    assert not matches, f"{script.relative_to(ROOT)} contains unsafe shell constructs: {matches}"


@pytest.mark.parametrize("script", ENTRYPOINTS, ids=lambda path: str(path.relative_to(ROOT)))
def test_shell_entrypoints_are_main_guarded_against_sourcing(script: Path) -> None:
    source = script.read_text(encoding="utf-8")
    assert "main()" in source
    assert '[[ "${BASH_SOURCE[0]}" != "$0" ]]' in source
    assert "must be executed, not sourced" in source
    assert source.rstrip().endswith('main "$@"')


@pytest.mark.skipif(
    sys.platform != "linux" or BASH is None,
    reason="Dynamic source-safety checks require Linux and Bash.",
)
@pytest.mark.parametrize("script", SHELL_SCRIPTS, ids=lambda path: str(path.relative_to(ROOT)))
def test_sourcing_shell_scripts_does_not_end_or_reconfigure_parent_shell(script: Path) -> None:
    shell_program = """
before_options="$-"
before_directory="$PWD"
source "$TARGET_SCRIPT" >/dev/null
source_status=$?
after_options="$-"
after_directory="$PWD"
printf '%s\\n' \
    "alive|$source_status|$before_options|$after_options|$before_directory|$after_directory"
"""
    environment = os.environ.copy()
    environment["TARGET_SCRIPT"] = str(script)
    completed = subprocess.run(
        [BASH, "-c", shell_program],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert completed.returncode == 0, completed.stderr
    marker, source_status, before_options, after_options, before_directory, after_directory = (
        completed.stdout.strip().split("|", maxsplit=5)
    )
    assert marker == "alive"
    assert source_status == "2"
    assert after_options == before_options
    assert after_directory == before_directory
