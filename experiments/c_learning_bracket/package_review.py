"""Build an immutable revision package with tests and real-data calibration evidence."""

import argparse
import ast
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import zipfile
from pathlib import Path

from experiments.aggregation_comparison import engine


def sha(data):
    return hashlib.sha256(data).hexdigest()


def build(destination):
    root = engine.ROOT
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("review packages are immutable; use a new path")
    destination.mkdir(parents=True)

    def put(relative, data):
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(data if isinstance(data, bytes) else data.encode("utf-8"))

    def put_json(relative, value):
        put(relative, json.dumps(value, indent=2, ensure_ascii=False))

    sources = engine.implementation_source_hashes()
    for path in (root / "experiments/c_learning_only").glob("*.py"):
        sources[path.relative_to(root).as_posix()] = sha(path.read_bytes())
    own = sorted((root / "experiments/c_learning_bracket").glob("*.py"))
    for path in own:
        sources[path.relative_to(root).as_posix()] = sha(path.read_bytes())
    tracked = subprocess.check_output(
        ["git", "-c", f"safe.directory={root.as_posix()}", "ls-files", "-z"],
        cwd=root,
        text=True,
    ).split("\0")
    available = {path for path in set(tracked) | set(sources) if path.endswith(".py")}
    modules = {}
    for relative in available:
        name = relative.removeprefix("src/").removesuffix(".py").replace("/", ".")
        modules[name.removesuffix(".__init__")] = relative
    selected, scanned = set(sources), set()
    queue = [path for path in selected if path.endswith(".py")]
    while queue:
        relative = queue.pop()
        if relative in scanned:
            continue
        scanned.add(relative)
        module = relative.removeprefix("src/").removesuffix(".py").replace("/", ".")
        package = (
            module.removesuffix(".__init__")
            if module.endswith(".__init__")
            else module.rpartition(".")[0]
        )
        candidates = []
        for node in ast.walk(ast.parse((root / relative).read_text(encoding="utf-8-sig"))):
            if isinstance(node, ast.Import):
                candidates.extend(item.name for item in node.names)
            elif isinstance(node, ast.ImportFrom):
                prefix = (
                    package.split(".")[: len(package.split(".")) - node.level + 1]
                    if node.level
                    else []
                )
                base = ".".join(prefix + ([node.module] if node.module else []))
                candidates.append(base)
                candidates.extend(base + "." + item.name for item in node.names if item.name != "*")
        for candidate in candidates:
            parts = candidate.split(".")
            for end in range(1, len(parts) + 1):
                target = modules.get(".".join(parts[:end]))
                if target:
                    selected.add(target)
                    if target not in scanned:
                        queue.append(target)
    selected.update(path.relative_to(root).as_posix() for path in root.glob("requirements*.txt"))
    selected.update(path.relative_to(root).as_posix() for path in root.glob("constraints*.txt"))
    docs = ["README.md", "MODEL_MATH.md", "REVIEW_FIXES.md", "RUN.md", "GPT_REVIEW_PROMPT.md"]
    for name in docs:
        selected.add("experiments/c_learning_bracket/" + name)
    for relative in sorted(selected):
        content = (root / relative).read_bytes()
        if relative.endswith(".py"):
            compile(content.decode("utf-8-sig"), relative, "exec")
        put("code/" + relative, content)
    put(
        "BRACKET_REVIEW.md",
        "\n\n---\n\n".join(
            (root / "experiments/c_learning_bracket" / name).read_text(encoding="utf-8")
            for name in docs
        ),
    )
    put(
        "BRACKET_CODE.txt",
        "\n\n".join(
            f"FILE: {path.relative_to(root).as_posix()}\nSHA256: {sha(path.read_bytes())}\n"
            + path.read_text(encoding="utf-8-sig")
            for path in own
        ),
    )
    put_json("SOURCE_CONTRACT.json", sources)
    put_json(
        "ENVIRONMENT.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "libraries": {
                name: importlib.metadata.version(name)
                for name in (
                    "torch",
                    "torch-geometric",
                    "numpy",
                    "scipy",
                    "ogb",
                    "pytest",
                    "psutil",
                )
            },
            "scope": "observed environment; not a dependency installer",
        },
    )
    evidence = root / "results/bracket-coverage-debug-20260929-01.xml"
    put("evidence/DEBUG_TESTS.xml", evidence.read_bytes())
    smoke = root / "results/bracket-coverage-debug-20260929-01"
    for path in sorted(smoke.rglob("*.json")):
        put("evidence/synthetic_smoke/" + path.relative_to(smoke).as_posix(), path.read_bytes())
    preserved = json.loads(
        (root / "experiments/information_flow_v2/V1_PRESERVATION.json").read_text(encoding="utf-8")
    )
    for relative, expected in preserved["files"].items():
        if sha((root / relative).read_bytes()) != expected["sha256"]:
            raise ValueError(f"original source changed: {relative}")
    old = root / "gpt_handoff_v2_20260928/code/experiments/information_flow_v2"
    old_count = 0
    for path in old.iterdir():
        if path.is_file():
            if (
                path.read_bytes()
                != (root / "experiments/information_flow_v2" / path.name).read_bytes()
            ):
                raise ValueError(f"previous v2 candidate changed: {path.name}")
            old_count += 1
    prior = root / "gpt_handoff_v2_0_20260928/code/experiments/c_learning_only"
    c_only_count = 0
    for path in prior.iterdir():
        if path.is_file():
            if path.read_bytes() != (root / "experiments/c_learning_only" / path.name).read_bytes():
                raise ValueError(f"C-only source changed: {path.name}")
            c_only_count += 1
    prior_bracket = root / "gpt_handoff_bracket_20260929/CGAT_C_LEARNING_BRACKET.zip"
    if sha(prior_bracket.read_bytes()) != (
        "49f7a79e7705adc2ab85e049884d2317720e94b3a33040c94a6a3dd2989e02ea"
    ):
        raise ValueError("previous bracket package changed")
    previous_review = root / "gpt_handoff_bracket_reviewfix_20260929/CGAT_C_LEARNING_BRACKET.zip"
    if sha(previous_review.read_bytes()) != (
        "9f81a5b5643c90b01e25fc952fdf4141c5a29501f38b9c7b0c0a95e616ff97a4"
    ):
        raise ValueError("previous reviewed bracket package changed")
    calibration = root / "results/bracket-signal-arxiv-20260929-01"
    report = json.loads((calibration / "calibration.json").read_text(encoding="utf-8"))
    previous_signal = root / "gpt_handoff_bracket_signal_20260929/CGAT_C_LEARNING_BRACKET.zip"
    if sha(previous_signal.read_bytes()) != (
        "2b2bfaf3f2349d6a279cba4f840090688483fb0b52ecf9a0a880758be6e4d316"
    ):
        raise ValueError("previous signal audit package changed")
    previous_sources = report["contract"]["source_sha256"]
    with zipfile.ZipFile(previous_signal) as prior_zip:
        if json.loads(prior_zip.read("SOURCE_CONTRACT.json")) != previous_sources:
            raise ValueError("historical calibration does not match its original package")
    changes = {
        name: {"previous": previous_sources.get(name), "current": sources.get(name)}
        for name in sorted(set(previous_sources) | set(sources))
        if previous_sources.get(name) != sources.get(name)
    }
    allowed = {
        "experiments/c_learning_bracket/" + name
        for name in ("train.py", "test_debug.py", "package_review.py")
    }
    if not set(changes) <= allowed:
        raise ValueError("coverage follow-up must preserve model, input and observer sources")
    put_json("CHANGES_FROM_SIGNAL_AUDIT_2.json", changes)
    for path in calibration.glob("*.json"):
        put("evidence/prior_signal_audit_2_calibration/" + path.name, path.read_bytes())
    comparison = root / "results/bracket-reviewfix-inputs-20260929-02/comparison.json"
    put("evidence/INPUT_EQUIVALENCE.json", comparison.read_bytes())
    put_json(
        "PACKAGE_CHECKS.json",
        {
            "syntax_checked_python_files": len(scanned),
            "copied_source_files": len(selected),
            "original_tracked_files_unchanged": len(preserved["files"]),
            "previous_v2_candidate_files_unchanged": old_count,
            "full_training_executed": False,
            "real_data_calibration_executed_for_current_sources": False,
            "historical_calibration_rows": len(report["measurements"]),
            "historical_calibration_matches_current_sources": not bool(changes),
            "fresh_calibration_required_before_production_training": True,
            "c_learning_only_files_unchanged": c_only_count,
            "previous_bracket_zip_unchanged": True,
            "previous_review_fixes_zip_unchanged": True,
            "previous_signal_audit_zip_unchanged": True,
            "evidence_scope": "current synthetic tests; historical actual-data calibration",
        },
    )
    for relative, expected in sources.items():
        if sha((root / relative).read_bytes()) != expected:
            raise RuntimeError("source changed while packaging")
    manifest = {
        path.relative_to(destination).as_posix(): sha(path.read_bytes())
        for path in destination.rglob("*")
        if path.is_file()
    }
    put_json("MANIFEST.json", manifest)
    archive = destination / "CGAT_C_LEARNING_BRACKET.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as handle:
        for relative in [*sorted(manifest), "MANIFEST.json"]:
            handle.write(destination / relative, relative)
    with zipfile.ZipFile(archive) as handle:
        if handle.testzip() is not None:
            raise RuntimeError("ZIP CRC verification failed")
        for relative, expected in manifest.items():
            if sha(handle.read(relative)) != expected:
                raise RuntimeError(f"ZIP entry differs: {relative}")
    print(
        json.dumps(
            {
                "archive": str(archive),
                "bytes": archive.stat().st_size,
                "sha256": sha(archive.read_bytes()),
                "files": len(manifest),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    build(parser.parse_args().destination)
