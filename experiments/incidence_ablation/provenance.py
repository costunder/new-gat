"""Independent immutable sources; historical manifests never include this package."""

import hashlib
from pathlib import Path

from scripts.training_resource_plan import source_snapshot as core_snapshot


def source_snapshot():
    root = Path(__file__).resolve().parents[2]
    sources = core_snapshot()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        sources[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return sources


def require_source_compatibility(saved, current, *, scope):
    if not isinstance(saved, dict) or not saved or saved != current:
        raise ValueError(
            f"independent incidence {scope} source mismatch; retain results and use a new run ID"
        )
    return None
