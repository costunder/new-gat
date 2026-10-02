"""Shared real DEBUG artifacts for frozen wedge evaluation tests."""

from argparse import Namespace

import pytest


@pytest.fixture(scope="session")
def wedge_completed_debug_run(tmp_path_factory):
    """Run the complete separate 4-epoch DEBUG contract, never full training."""
    import torch
    from threadpoolctl import threadpool_limits

    from research.wedge_propagation.learned.train import run

    output = tmp_path_factory.mktemp("completed-wedge-debug")
    previous_threads = torch.get_num_threads()
    try:
        with threadpool_limits(limits=1):
            run(
                Namespace(
                    config=None,
                    profile="debug",
                    device="cpu",
                    workers="1",
                    batch_size="12",
                    resume_from=None,
                ),
                output,
            )
    finally:
        torch.set_num_threads(previous_threads)
    return output


@pytest.fixture(scope="session")
def wedge_completed_feature_debug_run(wedge_completed_debug_run, tmp_path_factory):
    """Complete the real separate Experiment 3 DEBUG contract once."""
    import torch
    from threadpoolctl import threadpool_limits

    from research.wedge_propagation.generalization.study import run

    output = tmp_path_factory.mktemp("completed-wedge-feature-debug")
    previous_threads = torch.get_num_threads()
    try:
        with threadpool_limits(limits=1):
            run(
                Namespace(
                    run_dir=wedge_completed_debug_run,
                    profile="debug",
                    config=None,
                    device="cpu",
                    workers="1",
                    batch_size="36",
                ),
                output,
            )
    finally:
        torch.set_num_threads(previous_threads)
    return output
