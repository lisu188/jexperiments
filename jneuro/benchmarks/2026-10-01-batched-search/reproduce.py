#!/usr/bin/env python3
"""Bounded local whole-search comparison. Reuses supplied classes/dependencies; never downloads."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def tree(path):
    return {str(item.relative_to(path)): digest(item) for item in sorted(path.rglob("*")) if item.is_file()}


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def describe(values):
    return {"median": statistics.median(values), "minimum": min(values), "maximum": max(values), "observations": len(values)}


def analyze(directory):
    summary = {"fullBudget": {}, "quality": {}, "parity": {"checkpoints": 0, "parameters": 0, "maxParameterDifference": 0.0, "maxScoreDifference": 0.0}}
    for case in ("five-seeds", "mixed-population"):
        sets = {kind: [row for fork in (1, 2) for row in records(directory / f"{kind}-fork{fork}.jsonl") if row.get("case") == case]
                for kind in ("legacy", "batched")}
        assert all(len(rows) == 4 for rows in sets.values()), (case, "missing full-budget observations")
        timing = {kind: describe([row["elapsedMs"] for row in values]) for kind, values in sets.items()}
        timing["speedup"] = timing["legacy"]["median"] / timing["batched"]["median"]
        for metric in ("firstEligibleMs", "firstDiverseMs", "aggregateEpochsPerSecond"):
            timing[metric] = {kind: describe([row[metric] for row in values if row[metric] is not None])
                              for kind, values in sets.items() if any(row[metric] is not None for row in values)}
        summary["fullBudget"][case] = timing
        for before, after in zip(sets["legacy"], sets["batched"]):
            assert before["totalCommittedEpochs"] == after["totalCommittedEpochs"]
            left = {(tuple(item["shape"]), item["seed"]): item for item in before["trials"]}
            right = {(tuple(item["shape"]), item["seed"]): item for item in after["trials"]}
            assert left.keys() == right.keys()
            for key, expected in left.items():
                actual = right[key]
                assert expected["state"] == actual["state"] == "COMPLETED"
                assert expected["epochs"] == actual["epochs"] and expected["bestEpoch"] == actual["bestEpoch"]
                for field in ("bestRmse", "finalRmse"):
                    error = abs(expected[field] - actual[field])
                    summary["parity"]["maxScoreDifference"] = max(summary["parity"]["maxScoreDifference"], error)
                    assert error <= 1e-10, (case, key, field, error)
                assert len(expected["parameters"]) == len(actual["parameters"])
                for old, new in zip(expected["parameters"], actual["parameters"]):
                    error = abs(old - new)
                    summary["parity"]["maxParameterDifference"] = max(summary["parity"]["maxParameterDifference"], error)
                    assert error <= 1e-9, (case, key, error)
                    summary["parity"]["parameters"] += 1
                summary["parity"]["checkpoints"] += 1
    for policy in ("quality-full", "quality-prune"):
        path = directory / f"{policy}.jsonl"
        if path.exists():
            summary["quality"][policy] = [{key: value for key, value in row.items() if key != "trials"}
                for row in records(path) if row.get("case") == "quality"]
    (directory / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--java", type=Path, required=True)
    parser.add_argument("--compiler-classpath", required=True)
    parser.add_argument("--runtime-classpath", required=True)
    parser.add_argument("--baseline-classes", type=Path, required=True)
    parser.add_argument("--candidate-classes", type=Path, required=True)
    parser.add_argument("--native-cache", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--skip-quality", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    directory = args.work_dir.resolve()
    if args.analyze_only:
        print(json.dumps(analyze(directory), indent=2)); return
    directory.mkdir(parents=True, exist_ok=False)
    scratch = directory / "tmp"; scratch.mkdir()
    classes = directory / "harness-classes"; classes.mkdir()
    source = Path(__file__).with_name("PopulationBenchmark.kt").resolve()
    baseline = args.baseline_classes.resolve(); candidate = args.candidate_classes.resolve()
    before = {"baseline": tree(baseline), "candidate": tree(candidate)}
    provenance = {"source": str(source), "sourceSha256": digest(source), "baselineClasses": str(baseline),
                  "candidateClasses": str(candidate), "classHashes": before,
                  "runtime": {str(Path(item).resolve()): digest(Path(item)) for item in args.runtime_classpath.split(os.pathsep)},
                  "java": str(args.java.resolve()), "javaVersion": subprocess.run([str(args.java), "-version"], capture_output=True, text=True, check=True).stderr,
                  "nativeCache": str(args.native_cache.resolve()), "commands": [],
                  "environment": {key: os.environ.get(key) for key in ("TF_ENABLE_ONEDNN_OPTS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
                  "method": "Two warmed JVM forks/version, legacy/candidate/candidate/legacy; two repeats/case/fork; separate quality policies"}
    checkout = source.parents[3]
    provenance["candidateGitHead"] = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    provenance["candidateSources"] = {str(item.relative_to(checkout)): digest(item)
        for item in sorted((checkout / "jneuro/src/main").rglob("*")) if item.is_file()}
    command = [str(args.java), "-Xmx1200m", "--enable-native-access=ALL-UNNAMED", f"-Djava.io.tmpdir={scratch}",
               "-cp", args.compiler_classpath, "org.jetbrains.kotlin.cli.jvm.K2JVMCompiler", "-no-stdlib", "-no-reflect",
               "-jvm-target", "26", "-language-version", "2.4", "-api-version", "2.4", "-Werror", "-module-name", "jneuro",
               "-classpath", str(baseline) + os.pathsep + args.runtime_classpath, "-Xfriend-paths=" + str(baseline),
               "-d", str(classes), str(source)]
    provenance["compile"] = command
    with (directory / "compile.log").open("w") as output:
        subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=True, timeout=90)
    runs = [("legacy", "legacy-fork1", baseline), ("batched", "batched-fork1", candidate),
            ("batched", "batched-fork2", candidate), ("legacy", "legacy-fork2", baseline)]
    if not args.skip_quality:
        runs += [("quality-full", "quality-full", candidate), ("quality-prune", "quality-prune", candidate)]
    for mode, label, implementation in runs:
        command = [str(args.java), "-Xmx1500m", "--enable-native-access=ALL-UNNAMED", "-Djava.awt.headless=true",
                   "-Djneuro.log.level=OFF", "-Djneuro.log.file=false", f"-Djava.io.tmpdir={scratch}",
                   f"-Dorg.bytedeco.javacpp.cachedir={args.native_cache.resolve()}", "-cp",
                   os.pathsep.join((str(classes), str(implementation), args.runtime_classpath)),
                   "com.lis.neuro.PopulationBenchmark", mode, str(directory / f"{label}.jsonl")]
        provenance["commands"].append(command)
        (directory / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        print(f"Running {label}", flush=True)
        with (directory / f"{label}.log").open("w") as output:
            subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=True, timeout=240)
        print(f"Finished {label}", flush=True)
    assert before == {"baseline": tree(baseline), "candidate": tree(candidate)}, "Supplied class trees changed during measurement"
    print(json.dumps(analyze(directory), indent=2))


if __name__ == "__main__":
    main()
