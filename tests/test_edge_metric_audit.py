"""Independent DEBUG checks for fixed observability and individual q/E/J."""
import numpy as np
import pytest
import torch

from research.local_energy_relations.topology import build_topology
from research.edge_metric_relations.audit.numerics import null_projector, quadratic_witness, conjugate_gradient
from research.edge_metric_relations.audit.targets import dense_target_matrices, evaluate_targets, observation_masks
from research.edge_metric_relations.audit.actions import build_actions
from research.edge_metric_relations.audit.dense import dense_matrix, observability_rows, reconstruction_errors
from research.edge_metric_relations.audit.recovery import dense_reconstruct, sparse_reconstruct, TargetAccumulator
from research.edge_metric_relations.audit.contract import profile_config, read_config, OPERATOR_PLAN, expected_target_rows, validate_completion


def top():
    edges = np.asarray([[0, 0, 1, 1, 2, 3], [1, 2, 2, 3, 4, 4]], dtype=np.int64)
    return build_topology(5, edges).to("cpu")


@pytest.mark.parametrize("recipe", ["unit", "local_degree"])
def test_individual_target_matrices(recipe):
    topology = top()
    h = torch.arange(15, dtype=torch.float64).reshape(5, 3)/7
    matrices = dense_target_matrices(topology, recipe)
    actual = evaluate_targets(topology, h, recipe)
    torch.testing.assert_close(matrices["q"]@h, actual["q"])
    for name in ("E", "J_shared", "J_node", "J_distinct"):
        expected = torch.einsum("nf,tnm,mf->t", h, matrices[name], h)
        torch.testing.assert_close(expected, actual[name])
        torch.testing.assert_close(matrices[name], matrices[name].transpose(-2, -1))


def test_quadratic_collision_uses_cross_term_not_only_null_energy():
    a = torch.tensor([[1., 0.]], dtype=torch.float64)
    k = torch.tensor([[0., 1.], [1., 0.]], dtype=torch.float64)
    null = null_projector(a)
    witness = quadratic_witness(a, k, null["projector"])
    assert witness["observation_residual"] == 0
    assert witness["target_difference"] > 0
    torch.testing.assert_close(witness["target_difference"], witness["predicted_difference"])
    # n.T K n=0 would incorrectly declare this quadratic observable.
    n = witness["direction"]
    assert n@k@n == 0


def test_zero_matrix_null_rank_and_inverse_explicit():
    matrix = torch.zeros((3, 5), dtype=torch.float64)
    result = null_projector(matrix)
    assert result["rank"] == 0
    torch.testing.assert_close(result["projector"], torch.eye(5, dtype=torch.float64))
    assert torch.isfinite(result["pseudoinverse"]).all()


@pytest.mark.parametrize("kind", ["target", "one_hop"])
def test_observation_chunks_cover_every_original_node(kind):
    topology = top()
    actual = torch.cat([observation_masks(topology, kind, s, min(s+2, topology.n))
                        for s in range(0, topology.n, 2)])
    whole = observation_masks(topology, kind)
    torch.testing.assert_close(actual, whole)
    assert (actual.sum(1) >= 1).all()


def test_independent_packed_cg_and_unconverged_status():
    diagonal = torch.tensor([1., 2., 3., 4.], dtype=torch.float64)
    rhs = torch.arange(24, dtype=torch.float64).reshape(2, 4, 3)
    recovered, status = conjugate_gradient(lambda x: diagonal[None, :, None]*x, rhs,
                                           maximum_iterations=16, check_every=1, tolerance=1e-11)
    torch.testing.assert_close(recovered, rhs/diagonal[None, :, None])
    assert status["converged"].all()
    _, short = conjugate_gradient(lambda x: diagonal[None, :, None]*x, rhs,
                                  maximum_iterations=1, tolerance=1e-11)
    assert not short["converged"].all()


@pytest.mark.parametrize("recipe", ["unit", "local_degree"])
def test_fixed_coefficients_change_value_without_recomputing_gate(recipe):
    topology = top()
    reference = torch.arange(15, dtype=torch.float64).reshape(1, 5, 3)/7
    actions = build_actions(topology, recipe, reference, pair_chunk=2)
    for name in ("Q_F0", "P_F0", "Q_reference", "P_reference", "copy_on"):
        operator = actions[name]
        frozen = None if operator.coefficient is None else operator.coefficient.clone()
        matrix = dense_matrix(operator, 2)
        torch.testing.assert_close(operator(reference, 2), matrix@reference)
        torch.testing.assert_close(operator.transpose(reference, 2), matrix.transpose(-2, -1)@reference)
        if frozen is not None:
            torch.testing.assert_close(frozen, operator.coefficient)


@pytest.mark.parametrize("kind", ["full", "target", "one_hop"])
def test_certificate_coverage_and_actual_reconstruction(kind):
    topology = top()
    reference = torch.arange(10, dtype=torch.float64).reshape(1, 5, 2)
    actions = build_actions(topology, "unit", reference)
    matrix, baseline = dense_matrix(actions["Q_F0"]), dense_matrix(actions["Q_diag"])
    rows = list(observability_rows(topology, matrix, baseline, "unit", kind=kind, target_chunk=2))
    assert len(rows) == 2*topology.n+3*topology.num_pairs
    assert sum(row["target_kind"] == "q" for row in rows) == topology.n
    assert sum(row["target_kind"] == "E" for row in rows) == topology.n
    mask = observation_masks(topology, kind)
    obs = mask[:, None, :, None]*matrix[None]
    svd = null_projector(obs)
    estimated = svd["pseudoinverse"]@(obs@reference[None])
    errors = list(reconstruction_errors(topology, reference, estimated, "unit", kind, 0, topology.n))
    assert len(errors) == len(rows)
    if kind == "full":
        assert max(row["absolute_error"] for row in errors) < 1e-8


@pytest.mark.parametrize("kind", ["full", "target", "one_hop"])
def test_vector_trace_channel_accumulation_is_exact(kind):
    topology = top()
    reference = torch.sin(torch.arange(35, dtype=torch.float64).reshape(1, 5, 7)*.41)
    batch = 1 if kind == "full" else topology.n
    restored = reference[None].expand(batch, -1, -1, -1)+torch.cos(torch.arange(batch*35, dtype=torch.float64).reshape(batch, 1, 5, 7))*.1
    all_channels = TargetAccumulator(topology, "local_degree", kind, 0, topology.n, 1, reference)
    all_channels.add(reference, restored)
    chunks = TargetAccumulator(topology, "local_degree", kind, 0, topology.n, 1, reference)
    for start in range(0, 7, 2):
        chunks.add(reference[..., start:start+2], restored[..., start:start+2])
    expected, actual = list(all_channels.rows()), list(chunks.rows())
    assert len(actual) == expected_target_rows(topology, 1)
    for old, new in zip(expected, actual, strict=True):
        assert old["target"] == new["target"]
        assert new["absolute_error"] == pytest.approx(old["absolute_error"], abs=1e-12)
        assert new["target_norm"] == pytest.approx(old["target_norm"], abs=1e-12)


@pytest.mark.parametrize("name", ["Q_reference", "P_reference", "copy_on", "smooth_Ld"])
@pytest.mark.parametrize("kind", ["full", "target", "one_hop"])
def test_iterative_and_dense_minimum_norm_agree(name, kind):
    topology = top()
    source = torch.sin(torch.arange(15, dtype=torch.float64).reshape(1, 5, 3)*.73)
    action = build_actions(topology, "local_degree", source, pair_chunk=2)[name]
    mask = observation_masks(topology, kind)
    direct, _ = dense_reconstruct(dense_matrix(action), source, mask, 0.)
    sparse, status = sparse_reconstruct(action, 1, source, mask, noise_level=0., maximum_iterations=80, tolerance=1e-10)
    torch.testing.assert_close(direct, sparse, atol=1e-7, rtol=1e-7)
    assert status["converged"].all()
    assert status["numerical_operator_scale"] >= 1


def test_noise_is_relative_and_irreducible_residual_is_separate_from_stationarity():
    topology = top()
    source = torch.sin(torch.arange(15, dtype=torch.float64).reshape(1, 5, 3)*.73)
    action = build_actions(topology, "unit", source)["Ld"]
    _, status = sparse_reconstruct(action, 1, source, observation_masks(topology, "full"), noise_level=1e-3,
                                  maximum_iterations=50)
    torch.testing.assert_close(status["noise_norm"], status["clean_rhs_norm"]*1e-3)
    assert status["converged"].all()
    assert not status["observation_within_tolerance"].all()
    assert (status["observation_residual_norm"] > 0).all()
    assert status["normal_relative_residual"].max() < 1e-8


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
@pytest.mark.parametrize("condition", ["Q_F0", "copy_on"])
@pytest.mark.parametrize("kind", ["full", "one_hop"])
def test_noisy_rank_deficient_primal_cg_matches_independent_dense_pinv(device, condition, kind):
    from research.edge_metric_relations.audit.recovery import operator_scale
    # Two nontrivial components and two isolates make the null space explicit.
    topology = build_topology(9, np.asarray([[0, 1, 2, 0, 4, 5], [1, 2, 3, 3, 5, 6]], dtype=np.int64)).to(device)
    source = torch.sin(torch.arange(54, dtype=torch.float64, device=device).reshape(2, 9, 3)*.713+.37)
    action = build_actions(topology, "local_degree", source, pair_chunk=3, channel_chunk=2)[condition]
    mask = observation_masks(topology, kind, 0, 4) if kind != "full" else observation_masks(topology, kind)
    matrix = dense_matrix(action)/operator_scale(action, 1)
    observed = mask[:, None, :, None]*matrix[None]
    clean = observed@source[None]
    batch, replicas, nodes, channels = clean.shape
    a = torch.arange(batch*replicas, dtype=source.dtype, device=device)
    b = torch.arange(nodes, dtype=source.dtype, device=device)
    c = torch.arange(channels, dtype=source.dtype, device=device)
    independent_probe = torch.sin((a[:, None, None]+1)*.731+(b[None, :, None]+1)*1.213+(c[None, None, :]+1)*.417)
    independent_probe = independent_probe.reshape(batch, replicas, nodes, channels)*mask[:, None, :, None]
    probe_norm = independent_probe.square().sum(-2).sqrt()
    signal_norm = clean.square().sum(-2).sqrt()
    noise = 1e-3*independent_probe*(signal_norm/torch.where(probe_norm > 0, probe_norm, 1.))[..., None, :]
    rhs = clean+noise
    expected = torch.linalg.pinv(observed, rtol=1e-12)@rhs
    recovered, status = sparse_reconstruct(action, 1, source, mask, noise_level=1e-3,
                                          tolerance=1e-11, maximum_iterations=160)
    torch.testing.assert_close(recovered, expected, atol=2e-8, rtol=2e-8)
    normal_residual = observed.transpose(-2, -1)@(observed@recovered-rhs)
    torch.testing.assert_close(status["normal_residual_norm"].reshape(batch, replicas, channels),
                               normal_residual.square().sum(-2).sqrt(), atol=1e-12, rtol=1e-6)
    assert status["converged"].shape == (batch*replicas, channels)
    assert status["converged"].all() and not status["breakdown"].any()
    if condition == "Q_F0" and kind == "full":
        assert not status["observation_within_tolerance"].all()
        assert (status["observation_residual_norm"] > 1e-8).any()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_zero_normal_rhs_and_unresolved_iterates_have_explicit_status(device):
    topology = top().to(device)
    zero = torch.zeros((2, 5, 3), device=device, dtype=torch.float64)
    action = build_actions(topology, "unit", zero, pair_chunk=3)["Q_F0"]
    mask = observation_masks(topology, "full")
    recovered, status = sparse_reconstruct(action, 1, zero, mask, noise_level=1e-3, maximum_iterations=50)
    assert not status["normal_rhs_nonzero"].any() and not status["observation_rhs_nonzero"].any()
    assert status["converged"].all() and (recovered == 0).all()
    assert (status["normal_residual_norm"] == 0).all()
    source = torch.sin(torch.arange(30, device=device, dtype=torch.float64).reshape(2, 5, 3)*.41)
    short, incomplete = sparse_reconstruct(action, 1, source, mask, noise_level=1e-3,
                                           tolerance=1e-12, maximum_iterations=1)
    assert not incomplete["converged"].all()
    assert torch.isfinite(short).all() and short.abs().max() > 0
    assert (incomplete["normal_relative_residual"] > 1e-12).any()


def test_direct_svd_noisy_least_squares_passes_stationarity_with_nonzero_observation_residual():
    topology = top()
    source = torch.sin(torch.arange(15, dtype=torch.float64).reshape(1, 5, 3)*.73)
    action = build_actions(topology, "unit", source)["Ld"]
    _, status = dense_reconstruct(dense_matrix(action), source, observation_masks(topology, "full"), 1e-3)
    assert status["converged"].all()
    assert not status["observation_within_tolerance"].all()
    assert status["normal_relative_residual"].max() < 1e-10


def test_success_summary_excludes_entire_original_vector_if_any_channel_unresolved(tmp_path):
    from research.edge_metric_relations.audit.study import Results, _solve_aggregate, _update_solver
    topology = top()
    source = torch.sin(torch.arange(15, dtype=torch.float64).reshape(1, 5, 3)*.73)
    action = build_actions(topology, "unit", source)["Q_F0"]
    mask = observation_masks(topology, "full")
    good, good_status = sparse_reconstruct(action, 1, source, mask, noise_level=1e-3, maximum_iterations=80)
    short, short_status = sparse_reconstruct(action, 1, source, mask, noise_level=1e-3, maximum_iterations=1, tolerance=1e-12)
    assert good_status["converged"].all() and not short_status["converged"].all()
    partial = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in good_status.items()}
    partial["converged"][0, 1] = False  # one actual channel must disqualify the full vector row
    state = _solve_aggregate((1, 1), source)
    _update_solver(state, partial, 1, 1, 3)
    accumulator = TargetAccumulator(topology, "unit", "full", 0, 5, 1, source)
    accumulator.add(source, good)
    rejected = list(accumulator.rows(state))
    assert rejected and all(row["solver_converged"] is False for row in rejected)
    resolved = _solve_aggregate((1, 1), source)
    _update_solver(resolved, good_status, 1, 1, 3)
    accepted = list(accumulator.rows(resolved))
    metadata = {"family": "DEBUG", "recipe": "unit", "operator": "Q_F0", "repetitions": 1,
                "observation": "full", "noise_level": 1e-3}
    writer = Results(tmp_path, 0)
    writer.write("reconstruction", rejected, metadata)
    writer.write("reconstruction", accepted, metadata)
    files, summary = writer.close()
    assert sum(row["unresolved_rows"] for row in summary) == len(rejected)
    assert sum(row["successful_rows"] for row in summary) == len(accepted)
    assert sum(row["rows"] for row in summary) == len(rejected)+len(accepted)
    assert files[0]["rows"] == len(rejected)+len(accepted)


def test_completion_requires_correct_least_squares_scope_and_separate_diagnostics():
    config = profile_config("debug")
    complete = {"completed": True, "status": "complete", "experiment": config["experiment"], "profile": "debug",
                "graphs": 21, "math_checks_passed": True, "all_original_nodes_edges_channels": True,
                "optimizer_updates": 0, "classifier_training_run": False, "trained_model_audit_run": False,
                "source_digest": "a"*64, "least_squares_solver": "primal_CG_zero_start_or_direct_SVD",
                "convergence_criterion": "true_relative_normal_stationarity_residual_and_no_breakdown",
                "successful_summary_scope": "all_constituent_channels_solver_converged_only",
                "stationarity_unresolved_channel_solves": 2,
                "observation_residual_outside_tolerance_channel_solves": 100, "unresolved_target_rows": 10}
    assert validate_completion(complete, config) is complete
    historical = {key: value for key, value in complete.items() if key != "successful_summary_scope"}
    with pytest.raises(RuntimeError, match="mismatched"):
        validate_completion(historical, config)
    malformed = {**complete, "stationarity_unresolved_channel_solves": -1}
    with pytest.raises(RuntimeError, match="invalid"):
        validate_completion(malformed, config)


def test_exact_contract_has21_views_and_rejects_subset(tmp_path):
    import json
    assert sum(map(len, OPERATOR_PLAN.values())) == 21
    assert profile_config("full")["source_graphs"] == 201
    changed = profile_config("full")
    changed["source_graphs"] = 200
    destination = tmp_path/"changed.json"
    destination.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        read_config("full", destination)


@pytest.mark.parametrize("profile,graphs", [("full", 201), ("debug", 21)])
def test_actual_snapshot_reader_contract_profile_is_keyword(profile, graphs):
    from research.edge_metric_relations.audit.study import input_contract
    # The adapter's first positional argument is a filename, not the profile.
    # Exercise the actual JSON-backed contract reader used before source load.
    source_contract = input_contract(profile=profile)
    assert source_contract["profile"] == profile
    assert source_contract["source"]["profile"] == profile
    assert source_contract["source"]["graphs"] == graphs
    assert source_contract["source"]["reuse_complete_input_and_topology_snapshots"]


def test_batched_physical_matrices_equal_isolated_graphs():
    from types import SimpleNamespace
    from research.edge_metric_relations.audit.study import _dense_group
    topology = top()
    other = build_topology(5, np.asarray([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64))
    source = torch.sin(torch.arange(20, dtype=torch.float64).reshape(5, 4))
    first = SimpleNamespace(num_nodes=5, features=source, feature_mode="independent_scalar_columns")
    second = SimpleNamespace(num_nodes=5, features=source*.3, feature_mode="independent_scalar_columns")
    items = [(first, topology), (second, other)]
    packed = _dense_group(items, "local_degree", torch.device("cpu"), 2, 1)
    for graph, item in enumerate(items):
        alone = _dense_group([item], "local_degree", torch.device("cpu"), 2, 1)
        for key in packed:
            torch.testing.assert_close(packed[key][graph], alone[key][0], atol=1e-10, rtol=1e-10)


def test_worker_schedule_all_graph_recipe_pairs_exactly_once():
    from types import SimpleNamespace
    from research.edge_metric_relations.audit.study import _jobs
    topology = top()
    scalar = SimpleNamespace(graph_id="synthetic", num_nodes=5, num_features=4, feature_mode="independent_scalar_columns")
    vector = SimpleNamespace(graph_id="citationDEBUG", num_nodes=5, num_features=3, feature_mode="vector_trace")
    queues = _jobs([scalar, vector], [topology, topology], ["unit", "local_degree"], ["cuda:0", "cuda:1"])
    actual = [(case.graph_id, recipe) for queue in queues for _, recipe, items in queue for case, _ in items]
    assert set(actual) == {(name, recipe) for name in ("synthetic", "citationDEBUG") for recipe in ("unit", "local_degree")}
    assert len(actual) == 4 and all(queues)


@pytest.mark.parametrize("kind", ["target", "one_hop"])
def test_actual_q_collision_matches_exported_witness(kind):
    topology = top()
    source = torch.sin(torch.arange(15, dtype=torch.float64).reshape(1, 5, 3))
    action = build_actions(topology, "unit", source)["Q_F0"]
    matrix = dense_matrix(action)
    records = list(observability_rows(topology, matrix, matrix, "unit", kind=kind, target_chunk=2))
    count = 0
    for row in records:
        if row["target_kind"] != "q" or row["collision_plus"] is None:
            continue
        plus = torch.tensor(row["collision_plus"], dtype=torch.float64)[:, None]
        minus = torch.tensor(row["collision_minus"], dtype=torch.float64)[:, None]
        target = row["target"]
        mask = observation_masks(topology, kind, target, target+1)[0]
        assert (mask[:, None]*(matrix[0]@(plus-minus))).norm().item() == pytest.approx(row["collision_observation_residual"], abs=1e-10)
        qplus, qminus = evaluate_targets(topology, plus, "unit")["q"], evaluate_targets(topology, minus, "unit")["q"]
        assert (qplus-qminus)[topology.local_edge_center == target].norm().item() == pytest.approx(row["collision_target_difference"], abs=1e-10)
        count += 1
    assert count > 0


@pytest.mark.parametrize("feature_mode", ["independent_scalar_columns", "vector_trace"])
def test_complete_case_engine_exports_exact_expected_targets(tmp_path, feature_mode):
    from types import SimpleNamespace
    from research.edge_metric_relations.audit.actions import feature_fields
    from research.edge_metric_relations.audit.study import _case_views, _dense_group, Results
    import time
    topology = build_topology(3, np.asarray([[0, 1], [1, 2]], dtype=np.int64))
    case = SimpleNamespace(graph_id="DEBUG_only", family="DEBUG_fixture", num_nodes=3,
                           features=torch.sin(torch.arange(6, dtype=torch.float64).reshape(3, 2)*.73),
                           actual_data=False, feature_mode=feature_mode)
    output = Results(tmp_path, 0)
    output.last_progress = time.perf_counter()
    calibration = {"target_chunk": 2, "channel_chunk": 1, "pair_chunk": 2}
    source = feature_fields(case, "cpu")
    kwargs = {"matrices": {key: matrix[0] for key, matrix in _dense_group([(case, topology)], "unit", torch.device("cpu"), 2, 1).items()}} if feature_mode.startswith("independent") else {"actions": build_actions(topology, "unit", source, pair_chunk=2, channel_chunk=1)}
    coverage = _case_views(case, topology.to("cpu"), "unit", torch.device("cpu"), profile_config("debug"), calibration, output, **kwargs)
    files, summary = output.close()
    assert coverage["target_rows"] == expected_target_rows(topology, source.shape[0])*21*3*3
    records = next(row for row in files if row["table"] == "reconstruction")
    assert records["rows"] == coverage["target_rows"]
    assert coverage["channels"] == (1 if feature_mode.startswith("independent") else 2)
    assert files and summary and all(row["bytes"] > 0 for row in files)
