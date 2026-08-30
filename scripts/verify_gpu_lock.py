#!/usr/bin/env python3
"""Verify the exact GPU research stack selected by ``setup_gpu.sh``."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

try:
    from scripts.gpu_profiles import CUDA_RUNTIMES
except ModuleNotFoundError:
    from gpu_profiles import CUDA_RUNTIMES


class LockVerificationError(RuntimeError):
    """The installed environment does not satisfy the selected GPU lock."""


REQUIRED_RESEARCH_PACKAGES = {
    "networkx",
    "numpy",
    "ogb",
    "pandas",
    "pyyaml",
    "scikit-learn",
    "scipy",
    "torch",
    "torch-geometric",
}
IMPORT_NAMES = {
    "networkx": "networkx",
    "numpy": "numpy",
    "ogb": "ogb",
    "pandas": "pandas",
    "pyyaml": "yaml",
    "scikit-learn": "sklearn",
    "scipy": "scipy",
    "torch-geometric": "torch_geometric",
}


def canonical_name(name: str) -> str:
    """Return the distribution-name normalization used for lock comparison."""

    return re.sub(r"[-_.]+", "-", name).lower()


def read_exact_pins(path: Path) -> dict[str, str]:
    """Read a deliberately simple constraints file containing only ``name==version``."""

    pins: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.count("==") != 1:
            raise LockVerificationError(
                f"{path}:{line_number} is not an exact name==version pin: {line!r}"
            )
        raw_name, version = (part.strip() for part in line.split("==", 1))
        name = canonical_name(raw_name)
        if not name or not version or any(character.isspace() for character in version):
            raise LockVerificationError(f"{path}:{line_number} has an invalid exact pin")
        if name in pins:
            raise LockVerificationError(f"{path}:{line_number} duplicates {name}")
        pins[name] = version
    missing = sorted(REQUIRED_RESEARCH_PACKAGES - pins.keys())
    if missing:
        raise LockVerificationError(f"{path} is missing required pins: {', '.join(missing)}")
    return pins


def assert_same_pins(lock_path: Path, constraints_path: Path) -> dict[str, str]:
    """Reject drift between the portable lock and a CUDA-specific constraints file."""

    lock_pins = read_exact_pins(lock_path)
    constraint_pins = read_exact_pins(constraints_path)
    if lock_pins != constraint_pins:
        missing = sorted(lock_pins.keys() - constraint_pins.keys())
        extra = sorted(constraint_pins.keys() - lock_pins.keys())
        changed = sorted(
            name
            for name in lock_pins.keys() & constraint_pins.keys()
            if lock_pins[name] != constraint_pins[name]
        )
        raise LockVerificationError(
            f"CUDA constraints drift from {lock_path.name}: "
            f"missing={missing}, extra={extra}, changed={changed}"
        )
    return constraint_pins


def version_matches(
    name: str,
    expected: str,
    actual: str,
    *,
    cuda_tag: str | None = None,
) -> bool:
    """Allow only the official CUDA local suffix on the pinned torch version."""

    if name == "torch":
        if cuda_tag is not None:
            return actual == f"{expected}+{cuda_tag}"
        return re.fullmatch(re.escape(expected) + r"(?:\+cu[0-9]+)?", actual) is not None
    return actual == expected


def verify_numpy_bridge(torch: Any, numpy: Any) -> None:
    """Check the binary conversion boundary using an empty, non-GPU array."""
    try:
        torch.from_numpy(numpy.empty(0, dtype=numpy.float32)).numpy()
    except Exception as error:
        raise LockVerificationError(
            f"NumPy/Torch interoperability failed: {type(error).__name__}: {error}"
        ) from error


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def verify_environment(*, lock_path: Path, constraints_path: Path, cuda_tag: str) -> dict[str, Any]:
    """Check package versions, import-time ABI health, and the CUDA runtime."""

    if cuda_tag not in CUDA_RUNTIMES:
        raise LockVerificationError(f"unsupported CUDA wheel tag: {cuda_tag}")
    expected_pins = assert_same_pins(lock_path, constraints_path)

    installed: dict[str, str] = {}
    mismatches: list[str] = []
    for name, expected in sorted(expected_pins.items()):
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            mismatches.append(f"{name}: missing (expected {expected})")
            continue
        installed[name] = actual
        if not version_matches(name, expected, actual, cuda_tag=cuda_tag):
            expected_label = f"{expected}+{cuda_tag}" if name == "torch" else expected
            mismatches.append(f"{name}: installed {actual}, expected {expected_label}")
    if mismatches:
        raise LockVerificationError("exact package assertion failed: " + "; ".join(mismatches))

    import_errors: list[str] = []
    for distribution, module in IMPORT_NAMES.items():
        try:
            importlib.import_module(module)
        except Exception as error:  # binary dependencies can fail at import time
            import_errors.append(f"{distribution}: {type(error).__name__}: {error}")
    if import_errors:
        raise LockVerificationError("paper dependency import failed: " + "; ".join(import_errors))

    torch = importlib.import_module("torch")
    expected_runtime = CUDA_RUNTIMES[cuda_tag]
    actual_runtime = str(torch.version.cuda)
    if actual_runtime != expected_runtime:
        raise LockVerificationError(
            f"torch CUDA runtime is {actual_runtime}, expected {expected_runtime} for {cuda_tag}"
        )
    verify_numpy_bridge(torch, importlib.import_module("numpy"))
    if not torch.cuda.is_available():
        raise LockVerificationError("torch.cuda.is_available() is false")

    return {
        "status": "passed",
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cuda_wheel_tag": cuda_tag,
        "torch_cuda_runtime": actual_runtime,
        "numpy_torch_interop": "passed",
        "constraints_path": str(constraints_path.resolve()),
        "constraints_sha256": hashlib.sha256(constraints_path.read_bytes()).hexdigest(),
        "lock_path": str(lock_path.resolve()),
        "lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        "installed_top_level_versions": installed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--constraints", type=Path, required=True)
    parser.add_argument("--cuda-tag", choices=sorted(CUDA_RUNTIMES), required=True)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    try:
        report = verify_environment(
            lock_path=args.lock,
            constraints_path=args.constraints,
            cuda_tag=args.cuda_tag,
        )
    except (LockVerificationError, OSError) as error:
        print(f"GPU LOCK VERIFICATION FAILED: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.json_out is not None:
        _write_json_atomic(args.json_out, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
