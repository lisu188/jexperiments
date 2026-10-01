#!/usr/bin/env python3
"""Compare two prebuilt JNeuro class trees without downloading dependencies or changing either tree."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess


def tree_hash(directory):
    digest = hashlib.sha256()
    for file in sorted(directory.rglob("*.class")):
        digest.update(str(file.relative_to(directory)).encode())
        digest.update(hashlib.sha256(file.read_bytes()).digest())
    return digest.hexdigest()


def run(command, output, error, timeout):
    with output.open("w") as stdout, error.open("w") as stderr:
        subprocess.run(command, stdout=stdout, stderr=stderr, timeout=timeout, check=True,
                       env=dict(os.environ, TF_CPP_MIN_LOG_LEVEL="2"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--java", type=Path, required=True, help="Linux JDK27 bin/java")
    parser.add_argument("--compiler-classpath", required=True, help="Kotlin2.4.20 compiler and its dependencies")
    parser.add_argument("--runtime-classpath", required=True, help="TensorFlow1.2 CPU, Kotlin and FlatLaf dependencies")
    parser.add_argument("--baseline-classes", type=Path, required=True)
    parser.add_argument("--candidate-classes", type=Path, required=True,
                        help="Complete candidate classes, or TensorFlowMath overlay on the baseline")
    parser.add_argument("--work-dir", type=Path, required=True, help="New task-owned directory; must not exist")
    parser.add_argument("--native-cache", type=Path, help="Existing compatible JavaCPP cache, otherwise use work-dir/native")
    parser.add_argument("--forks", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=180, help="Seconds per compiler or benchmark process")
    args = parser.parse_args()
    if not 1 <= args.forks <= 5 or args.timeout <= 0:
        parser.error("forks must be in 1..5 and timeout must be positive")
    for name in ("java", "baseline_classes", "candidate_classes"):
        value = getattr(args, name).resolve(strict=True)
        setattr(args, name, value)
    for entry in (args.compiler_classpath + os.pathsep + args.runtime_classpath).split(os.pathsep):
        if not Path(entry).exists():
            parser.error(f"Missing classpath entry: {entry}")
    args.work_dir.mkdir(parents=True, exist_ok=False)
    work = args.work_dir.resolve()
    classes = work / "harness-classes"
    temporary = work / "tmp"
    temporary.mkdir()
    cache = args.native_cache.resolve() if args.native_cache else work / "native"
    baseline_hash = tree_hash(args.baseline_classes)
    candidate_hash = tree_hash(args.candidate_classes)
    source = Path(__file__).with_name("EpochBenchmark.kt").resolve()
    compiler = [str(args.java), "-Xmx1500m", f"-Djava.io.tmpdir={temporary}", "--enable-native-access=ALL-UNNAMED",
                "-cp", args.compiler_classpath, "org.jetbrains.kotlin.cli.jvm.K2JVMCompiler",
                "-no-stdlib", "-no-reflect", "-jvm-target", "26", "-language-version", "2.4", "-api-version", "2.4",
                "-Werror", "-Xadd-modules=jdk.incubator.vector", "-module-name", "jneuro_epoch_benchmark",
                "-classpath", str(args.baseline_classes) + os.pathsep + args.runtime_classpath,
                f"-Xfriend-paths={args.baseline_classes}", "-d", str(classes), str(source)]
    run(compiler, work / "compile.stdout.log", work / "compile.stderr.log", args.timeout)
    records = {"baseline": [], "candidate": []}
    commands = []
    state_count = 0
    double_count = 0
    differences = []
    for fork in range(args.forks):
        # Alternating process order reduces, but does not eliminate, workstation drift.
        variants = ("baseline", "candidate") if fork % 2 == 0 else ("candidate", "baseline")
        for variant in variants:
            label = f"{variant}-fork{fork}"
            classpath = [str(classes), str(args.baseline_classes), args.runtime_classpath]
            if variant == "candidate":
                classpath.insert(0, str(args.candidate_classes))
            command = [str(args.java), "-Xmx1500m", "--add-modules", "jdk.incubator.vector",
                       "--enable-native-access=ALL-UNNAMED", "-Djava.awt.headless=true",
                       f"-Djava.io.tmpdir={temporary}", f"-Dorg.bytedeco.javacpp.cachedir={cache}",
                       "-Djneuro.log.level=OFF", "-Djneuro.log.file=false", "-cp", os.pathsep.join(classpath),
                       "com.lis.neuro.EpochBenchmark", str(work / f"{label}-states")]
            commands.append(command)
            output = work / f"{label}.jsonl"
            run(command, output, work / f"{label}.stderr.log", args.timeout)
            rows = [dict(json.loads(line), fork=fork) for line in output.read_text().splitlines()]
            if len(rows) != 41:
                raise RuntimeError(f"Incomplete benchmark: {label}, expected 41 rows, got {len(rows)}")
            records[variant].extend(rows)
        before_dir = work / f"baseline-fork{fork}-states"
        after_dir = work / f"candidate-fork{fork}-states"
        before_files = sorted(before_dir.glob("*.json"))
        if len(before_files) != 51 or {p.name for p in before_files} != {p.name for p in after_dir.glob("*.json")}:
            raise RuntimeError("Expected 51 matching checkpoint files per version")
        for file in before_files:
            before = json.loads(file.read_text())
            after = json.loads((after_dir / file.name).read_text())
            state_count += 1
            if before != after:
                differences.append(f"fork{fork}:{file.name}")
            for key in ("weights", "biases", "weightVelocity", "biasVelocity"):
                double_count += sum(len(row) for row in before[key])
                if any(x.hex() != y.hex() for a, b in zip(before[key], after[key]) for x, y in zip(a, b)):
                    differences.append(f"fork{fork}:{file.name}:{key}")
    if tree_hash(args.baseline_classes) != baseline_hash or tree_hash(args.candidate_classes) != candidate_hash:
        raise RuntimeError("Input classes changed during the benchmark")
    timings = []
    for shape in ("2x6x1", "2x8x8x8x1"):
        for api in ("epoch", "chunk"):
            for batch in (1, 16, 32):
                values = {name: statistics.median(row["epochMs"] for row in rows if row["kind"] == "training"
                          and row["shape"] == shape and row["api"] == api and row["batch"] == batch)
                          for name, rows in records.items()}
                timings.append(dict(shape=shape, api=api, batch=batch, baselineMs=values["baseline"],
                                    candidateMs=values["candidate"], speedup=values["baseline"] / values["candidate"]))
    result = dict(forksPerVersion=args.forks, baselineClassesSha256=baseline_hash,
                  candidateClassesSha256=candidate_hash, harnessSha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  checkpoints=state_count, parameterAndMomentumValues=double_count, differences=differences,
                  training=timings, commands=commands)
    (work / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(checkpoints=state_count, differences=differences, report=str(work / "summary.json"))))
    if differences:
        raise SystemExit("Numerical equivalence failed; inspect retained checkpoint evidence")


if __name__ == "__main__":
    main()
