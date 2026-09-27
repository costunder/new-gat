"""Immutable data origin and explicit in-process synthetic verification scope.

Production command lines never expose this scope. Debug artifacts remain marked
and cannot be approved/exported by an ordinary production invocation.
"""

from contextlib import contextmanager
from contextvars import ContextVar

_VERIFY_SYNTHETIC = ContextVar("aggregation_synthetic_verification", default=False)


def origin(protocol):
    debug = protocol.get("explicit_synthetic_debug", False)
    if type(debug) is not bool:
        raise ValueError("explicit_synthetic_debug must be a boolean")
    return {"kind": "synthetic_debug" if debug else "benchmark", "debug": debug}


def require_approval(protocol):
    value = origin(protocol)
    if value["debug"] and not _VERIFY_SYNTHETIC.get():
        raise ValueError("synthetic debug evidence cannot be approved as a benchmark result")
    return value


@contextmanager
def synthetic_verification():
    """For isolated verification code, not a runtime fallback or CLI option."""
    token = _VERIFY_SYNTHETIC.set(True)
    try:
        yield
    finally:
        _VERIFY_SYNTHETIC.reset(token)
