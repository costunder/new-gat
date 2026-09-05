"""Archive safety on synthetic directories only; no research results or GPUs."""

from __future__ import annotations

import errno
import json
import os
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import archive_failed_rich_run as archive

RUN_ID = "v5-cycle-se-pe-a6000-gpu3-seed0-v1"


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@pytest.fixture
def failed_run(tmp_path_factory):
    # Short fixture names keep Windows mocked-Linux journal paths under MAX_PATH.
    tmp_path = tmp_path_factory.mktemp("a")
    results = tmp_path / "results"
    parent = results / "rich_scaling" / RUN_ID
    paths = {"rich": parent}
    jobs = []
    for track, version in (("conductance", "v5"), ("cycle", "v2")):
        child_id = archive._child_run_id(RUN_ID, track)
        child = results / archive.TRACK_SPECS[track]["results_subdir"] / child_id
        paths[track] = child
        manifest = {
            "schema_version": 1 if track == "conductance" else 2,
            "run_id": child_id,
            "status": "failed",
            "jobs": [
                {"version": version, "status": "failed"},
                {"version": version, "status": "pending"},
            ],
        }
        if track == "conductance":
            manifest.update(
                suite="conductance_architecture_scaling_v1_v5", config={"versions": ["v5"]}
            )
        else:
            manifest.update(
                scope="cycle_pe_v1_v2_larger_model_scaling",
                versions=["v2"],
                output_dir=str(child),
                test_evaluation_jobs=[],
            )
        _write(child / "manifest.json", manifest)
        _write(child / "summary.json", {"status": "failed"})
        checkpoint = child / "test-fixture-only.pt"
        checkpoint.write_bytes(b"synthetic checkpoint bytes: never model training")
        jobs.append(
            {
                "track": track,
                "status": "failed",
                "child_run_id": child_id,
                "output_dir": str(child),
                "summary_path": str(child / "summary.json"),
                "requested_matrix": {"versions": [version]},
            }
        )
    _write(
        parent / "manifest.json",
        {
            "schema_version": 1,
            "suite": "rich_scaling",
            "status": "failed",
            "run_id": RUN_ID,
            "config": {
                "tracks": ["conductance", "cycle"],
                "conductance_versions": ["v5"],
                "cycle_versions": ["v2"],
                "results_root": str(results),
            },
            "jobs": jobs,
        },
    )
    _write(parent / "summary.json", {"status": "failed"})
    untouched = tmp_path / "data" / "verified-cache.pt"
    untouched.parent.mkdir()
    untouched.write_bytes(b"untouched cached dataset")
    other = results / "conductance_gat" / "scaling" / "unrelated-v1-run" / "best.pt"
    other.parent.mkdir(parents=True)
    other.write_bytes(b"unrelated V1 unchanged")
    return results, paths


def _mutate(path, change):
    payload = json.loads(path.read_text(encoding="utf-8"))
    change(payload)
    _write(path, payload)


@pytest.fixture
def mock_linux(monkeypatch):
    monkeypatch.setattr(archive.sys, "platform", "linux")
    monkeypatch.setattr(archive, "active_processes", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(archive.uuid, "uuid4", lambda: SimpleNamespace(hex="fixture-uuid"))

    def rename(source, destination):
        assert not destination.exists()
        source.rename(destination)

    monkeypatch.setattr(archive, "_rename_no_replace", rename)


def test_default_is_read_only_plan_and_does_not_hash_checkpoints(failed_run, monkeypatch, capsys):
    results, paths = failed_run
    original = {track: (path / "manifest.json").read_bytes() for track, path in paths.items()}
    read_bytes = Path.read_bytes

    def metadata_only(path):
        assert path.suffix != ".pt", "planning must not read tensor/checkpoint contents"
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", metadata_only)
    assert archive.main(["--run-id", RUN_ID, "--results-root", str(results)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "dry_run"
    assert len(output["targets"]) == 3
    assert not (results / archive.ARCHIVE_DIRECTORY).exists()
    assert all(
        read_bytes(paths[track] / "manifest.json") == value for track, value in original.items()
    )


def test_apply_preserves_all_bytes_and_only_moves_bound_directories(failed_run, mock_linux):
    results, paths = failed_run
    originals = {
        track: {
            file.relative_to(path): file.read_bytes() for file in path.rglob("*") if file.is_file()
        }
        for track, path in paths.items()
    }
    result = archive.apply_archive(archive.plan_archive(RUN_ID, results))
    assert result["status"] == "archived"
    assert len(result["moves"]) == 3
    for move in result["moves"]:
        assert not Path(move["original"]).exists()
        destination = Path(move["destination"])
        assert {
            file.relative_to(destination): file.read_bytes()
            for file in destination.rglob("*")
            if file.is_file()
        } == originals[move["track"]]
    assert (results.parent / "data/verified-cache.pt").read_bytes() == b"untouched cached dataset"
    assert (
        results / "conductance_gat/scaling/unrelated-v1-run/best.pt"
    ).read_bytes() == b"unrelated V1 unchanged"
    events = [
        json.loads(line)
        for line in Path(result["journal"]).read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event"] for event in events].count("move_completed") == 3
    assert events[-1]["event"] == "archive_completed"
    assert not (results / archive.ARCHIVE_DIRECTORY / "_locks" / f"{RUN_ID}.lock").exists()
    with pytest.raises(archive.ArchiveError, match="cannot read manifest"):
        archive.plan_archive(RUN_ID, results)


@pytest.mark.parametrize("status", ["running", "passed", "pending", None])
def test_nonfailed_root_refused(failed_run, status):
    results, paths = failed_run
    _mutate(paths["rich"] / "manifest.json", lambda obj: obj.update(status=status))
    with pytest.raises(archive.ArchiveError, match="failed/interrupted"):
        archive.plan_archive(RUN_ID, results)


@pytest.mark.parametrize("site", ["root_job", "child", "leaf"])
def test_running_or_ambiguous_children_refused(failed_run, site):
    results, paths = failed_run
    path = paths["rich" if site == "root_job" else "conductance"] / "manifest.json"

    def change(obj):
        if site == "child":
            obj["status"] = "running"
        else:
            obj["jobs"][0]["status"] = "running"

    _mutate(path, change)
    with pytest.raises(archive.ArchiveError, match="terminal|running"):
        archive.plan_archive(RUN_ID, results)


def test_failed_root_may_include_a_passed_track(failed_run):
    results, paths = failed_run
    _mutate(paths["rich"] / "manifest.json", lambda obj: obj["jobs"][0].update(status="passed"))
    _mutate(paths["conductance"] / "manifest.json", lambda obj: obj.update(status="passed"))
    assert len(archive.plan_archive(RUN_ID, results)["targets"]) == 3


@pytest.mark.parametrize(
    "change",
    [
        lambda obj: obj["config"].update(conductance_versions=["v1", "v5"]),
        lambda obj: obj["jobs"][0]["requested_matrix"].update(versions=["v1", "v5"]),
        lambda obj: obj["jobs"][0].update(track="tree"),
        lambda obj: obj["jobs"].append(obj["jobs"][0].copy()),
        lambda obj: obj["config"].update(results_root="/unrelated"),
    ],
)
def test_legacy_tree_duplicate_or_wrong_root_refused(failed_run, change):
    results, paths = failed_run
    _mutate(paths["rich"] / "manifest.json", change)
    with pytest.raises(archive.ArchiveError):
        archive.plan_archive(RUN_ID, results)
    assert all(path.is_dir() for path in paths.values())


@pytest.mark.parametrize("case", ["outside", "root", "parent", "other_child", "traversal"])
def test_manifest_output_path_boundary_refused(failed_run, case):
    results, paths = failed_run
    values = {
        "outside": results.parent / "data",
        "root": results,
        "parent": paths["rich"],
        "other_child": paths["cycle"],
        "traversal": paths["conductance"] / ".." / "elsewhere",
    }
    _mutate(
        paths["rich"] / "manifest.json",
        lambda obj: obj["jobs"][0].update(output_dir=str(values[case])),
    )
    with pytest.raises(archive.ArchiveError):
        archive.plan_archive(RUN_ID, results)


@pytest.mark.parametrize("run_id", ["../other", "/absolute", ".", "bad/id", "bad\\id", ""])
def test_invalid_run_ids_cannot_escape_results_root(failed_run, run_id):
    with pytest.raises(archive.ArchiveError, match="run ID"):
        archive.plan_archive(run_id, failed_run[0])


def test_legacy_directory_hidden_inside_v5_is_not_moved(failed_run):
    results, paths = failed_run
    (paths["conductance"] / "v4").mkdir()
    with pytest.raises(archive.ArchiveError, match="legacy-version directories"):
        archive.plan_archive(RUN_ID, results)


def test_unreferenced_child_is_not_inferred(failed_run):
    results, paths = failed_run

    def only_conductance(obj):
        obj["jobs"] = obj["jobs"][:1]
        obj["config"]["tracks"] = ["conductance"]

    _mutate(paths["rich"] / "manifest.json", only_conductance)
    plan = archive.plan_archive(RUN_ID, results)
    assert [target["track"] for target in plan["targets"]] == ["rich", "conductance"]


def test_reparse_target_is_refused_without_following_it(failed_run, monkeypatch):
    results, paths = failed_run
    original = Path.lstat

    def metadata(path):
        if path == paths["conductance"]:
            return SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400)
        return original(path)

    monkeypatch.setattr(Path, "lstat", metadata)
    with pytest.raises(archive.ArchiveError, match="reparse"):
        archive.plan_archive(RUN_ID, results)


def test_apply_without_linux_process_evidence_refused(failed_run, monkeypatch):
    results, _ = failed_run
    plan = archive.plan_archive(RUN_ID, results)
    monkeypatch.setattr(archive.sys, "platform", "win32")
    with pytest.raises(archive.ArchiveError, match="Linux /proc"):
        archive.apply_archive(plan)
    assert not (results / archive.ARCHIVE_DIRECTORY).exists()


def test_live_matching_process_prevents_any_rename(failed_run, mock_linux, monkeypatch):
    results, paths = failed_run
    monkeypatch.setattr(
        archive, "active_processes", lambda *_: [{"pid": 123, "command": ["python"]}]
    )
    with pytest.raises(archive.ArchiveError, match="active matching"):
        archive.apply_archive(archive.plan_archive(RUN_ID, results))
    assert all(path.exists() for path in paths.values())
    assert not (results / archive.ARCHIVE_DIRECTORY).exists()


def test_metadata_change_after_plan_refused(failed_run, mock_linux):
    results, paths = failed_run
    plan = archive.plan_archive(RUN_ID, results)
    _mutate(paths["cycle"] / "manifest.json", lambda obj: obj.update(extra="changed"))
    with pytest.raises(archive.ArchiveError, match="changed after planning"):
        archive.apply_archive(plan)
    assert all(path.exists() for path in paths.values())


def test_partial_move_failure_preserves_journal_and_exact_remaining_paths(
    failed_run, mock_linux, monkeypatch, capsys
):
    results, paths = failed_run
    calls = []

    def fail_second(source, destination):
        calls.append(source)
        if len(calls) == 2:
            raise OSError(errno.EACCES, "synthetic rename denial")
        source.rename(destination)

    monkeypatch.setattr(archive, "_rename_no_replace", fail_second)
    with pytest.raises(OSError, match="synthetic rename denial"):
        archive.apply_archive(archive.plan_archive(RUN_ID, results))
    failure = json.loads(capsys.readouterr().err)
    assert len(failure["completed_moves"]) == 1
    assert {entry["track"] for entry in failure["unmoved"]} == {"conductance", "cycle"}
    assert not paths["rich"].exists()
    assert paths["conductance"].exists() and paths["cycle"].exists()
    archive_dir = Path(failure["archive"])
    assert (archive_dir / "rich/manifest.json").is_file()
    events = [
        json.loads(line)
        for line in (archive_dir / "archive_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert events[-1]["event"] == "archive_failed"


def _process(proc, pid, command, uid=123):
    directory = proc / str(pid)
    directory.mkdir(parents=True)
    (directory / "status").write_text(
        f"Name:\tpython\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\n", encoding="utf-8"
    )
    (directory / "cmdline").write_bytes(b"\0".join(os.fsencode(value) for value in command) + b"\0")
    return directory


def test_proc_detects_exact_run_ids_and_nested_output_not_prefix_or_other_uid(failed_run, tmp_path):
    results, paths = failed_run
    plan = archive.plan_archive(RUN_ID, results)
    proc = tmp_path / "proc"
    _process(proc, 101, ["python", "--run-id", RUN_ID])
    _process(proc, 102, ["python", f"--run-id={RUN_ID}-cycle"])
    _process(proc, 103, ["python", "--output-dir", str(paths["conductance"] / "v5/large")])
    _process(proc, 104, ["python", "--run-id", RUN_ID + "-unrelated"])
    _process(proc, 105, ["python", "--run-id", RUN_ID], uid=999)
    _process(proc, 106, ["archive", "--run-id", RUN_ID])
    matches = archive.active_processes(plan, proc_root=proc, uid=123, own_pid=106)
    assert {match["pid"] for match in matches} == {101, 102, 103}


def test_inaccessible_same_uid_cmdline_fails_closed(failed_run, tmp_path, monkeypatch):
    plan = archive.plan_archive(RUN_ID, failed_run[0])
    proc = tmp_path / "proc"
    process = _process(proc, 101, ["python", "unknown"])
    original = Path.read_bytes

    def deny(path):
        if path == process / "cmdline":
            raise PermissionError("synthetic same UID restriction")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", deny)
    with pytest.raises(archive.ArchiveError, match="same-UID"):
        archive.active_processes(plan, proc_root=proc, uid=123, own_pid=999)


def test_atomic_rename_uses_no_replace_and_preserves_existing_target_on_error(
    monkeypatch, tmp_path
):
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    calls = []

    class Rename:
        def __call__(self, *args):
            calls.append(args)
            return -1

    monkeypatch.setattr(archive.ctypes.util, "find_library", lambda _: "libc-fixture")
    monkeypatch.setattr(
        archive.ctypes, "CDLL", lambda *_args, **_kwargs: SimpleNamespace(renameat2=Rename())
    )
    monkeypatch.setattr(archive.ctypes, "get_errno", lambda: errno.EEXIST)
    with pytest.raises(FileExistsError):
        archive._rename_no_replace(source, destination)
    assert calls[0][0] == calls[0][2] == -100
    assert calls[0][-1] == 1
    assert source.is_dir() and destination.is_dir()


@pytest.mark.parametrize("bad_track", [[], {}, None, 5])
def test_malformed_track_metadata_has_clear_archive_error(failed_run, bad_track):
    results, paths = failed_run
    _mutate(paths["rich"] / "manifest.json", lambda obj: obj["jobs"][0].update(track=bad_track))
    with pytest.raises(archive.ArchiveError, match="distinct explicitly"):
        archive.plan_archive(RUN_ID, results)


@pytest.mark.parametrize("status,version", [("running", "v2"), ("unknown", "v2"), ("passed", "v1")])
def test_cycle_test_evaluation_jobs_must_be_safe_terminal_or_unstarted(failed_run, status, version):
    results, paths = failed_run
    _mutate(
        paths["cycle"] / "manifest.json",
        lambda obj: obj.update(test_evaluation_jobs=[{"status": status, "version": version}]),
    )
    with pytest.raises(archive.ArchiveError, match="test_evaluation_jobs"):
        archive.plan_archive(RUN_ID, results)


@pytest.mark.parametrize("mode,attributes", [(stat.S_IFIFO, 0), (stat.S_IFREG, 0x400)])
def test_manifest_special_file_is_rejected_before_any_content_read(
    failed_run, monkeypatch, mode, attributes
):
    results, paths = failed_run
    target = paths["rich"] / "manifest.json"
    original_stat, original_read = Path.lstat, Path.read_text

    def metadata(path):
        if path == target:
            return SimpleNamespace(st_mode=mode, st_file_attributes=attributes)
        return original_stat(path)

    def read(path, *args, **kwargs):
        assert path != target, "must not read or block on a special/indirect manifest"
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", metadata)
    monkeypatch.setattr(Path, "read_text", read)
    with pytest.raises(archive.ArchiveError, match="regular file|reparse"):
        archive.plan_archive(RUN_ID, results)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="native Linux renameat2 contract")
def test_native_linux_rename_success_and_no_replace(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.mkdir()
    (source / "checkpoint.pt").write_bytes(b"synthetic native rename fixture")
    archive._rename_no_replace(source, destination)
    assert not source.exists()
    assert (destination / "checkpoint.pt").read_bytes() == b"synthetic native rename fixture"
    source.mkdir()
    with pytest.raises(FileExistsError):
        archive._rename_no_replace(source, destination)
    assert source.is_dir()
    assert (destination / "checkpoint.pt").read_bytes() == b"synthetic native rename fixture"
    empty_destination = tmp_path / "empty-destination"
    empty_destination.mkdir()
    (source / "another.pt").write_bytes(b"must not replace an existing empty directory")
    with pytest.raises(FileExistsError):
        archive._rename_no_replace(source, empty_destination)
    assert (source / "another.pt").is_file()
    assert empty_destination.is_dir() and not list(empty_destination.iterdir())
