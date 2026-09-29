"""Build a NEW review ZIP with exact sources and explicit debug-only evidence."""

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
    own = sorted((root / "experiments/c_learning_only").glob("*.py"))
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
    docs = ["README.md", "MODEL_MATH.md", "RUN.md", "VERIFICATION.md", "GPT_REVIEW_PROMPT.md"]
    for name in docs:
        selected.add("experiments/c_learning_only/" + name)
    for relative in sorted(selected):
        content = (root / relative).read_bytes()
        if relative.endswith(".py"):
            compile(content.decode("utf-8-sig"), relative, "exec")
        put("code/" + relative, content)
    put(
        "V2_0_REVIEW.md",
        "\n\n---\n\n".join(
            (root / "experiments/c_learning_only" / name).read_text(encoding="utf-8")
            for name in docs
        ),
    )
    put(
        "V2_0_CODE.txt",
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
    evidence = root / "results/c-learning-debug-20260928-02.xml"
    put("evidence/DEBUG_TESTS.xml", evidence.read_bytes())
    smoke = root / "results/c-learning-debug-20260928-02"
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
    put_json(
        "PACKAGE_CHECKS.json",
        {
            "syntax_checked_python_files": len(scanned),
            "copied_source_files": len(selected),
            "original_tracked_files_unchanged": len(preserved["files"]),
            "previous_v2_candidate_files_unchanged": old_count,
            "full_training_executed": False,
            "real_data_experiment_executed": False,
            "evidence_scope": "synthetic unit and CUDA orchestration tests only",
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
    archive = destination / "CGAT_V2_0_C_LEARNING.zip"
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
