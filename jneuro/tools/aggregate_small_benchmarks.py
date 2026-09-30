#!/usr/bin/env python3
"""Aggregate SMALL evidence without hiding failed cases or counting copied files as forks.

Example: python3 jneuro/tools/aggregate_small_benchmarks.py --jsonpaths fork*.json
    --csvpaths cohort*.csv quality.csv --output build/reports/small-summary

Writes JSON and Markdown. Exit 0 means a native candidate passed every required
workload; exit 1 means the gate was not met. A report is still written on failure.
Only distinct (processId, jvmStartMillis) pairs verify independent JVM forks.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


JVM = {"CPU", "CPU_LEGACY", "SMALL_SCALAR_FP64", "SMALL_128_FP64", "SMALL_256_FP64",
       "REFERENCE_MATRIX", "REFERENCE_PARALLEL", "SMALL_SEQUENTIAL", "SMALL_CPU"}
NATIVE = {"NATIVE_FP64": "NATIVE", "NATIVE": "NATIVE", "NATIVE_PARALLEL": "NATIVE_PARALLEL"}
VARIANTS = ("FP64/EXACT", "FP64/FAST", "FP32/EXACT", "FP32/FAST")


def distribution(values):
    values = sorted(values)
    if not values:
        return None
    return {"count": len(values), "min": values[0], "median": statistics.median(values),
            "p95": values[math.ceil(0.95 * len(values)) - 1]}


def integer(value, minimum=0):
    if isinstance(value, bool) or str(value) != str(int(value)) or int(value) < minimum:
        raise ValueError(f"Expected integer >= {minimum}, got {value!r}")
    return int(value)


def number(value, minimum=0.0):
    value = float(value)
    if not math.isfinite(value) or value < minimum:
        raise ValueError("Expected finite nonnegative number")
    return value


def process_identity(environment):
    try:
        return f"{integer(environment['processId'], 1)}:{integer(environment['jvmStartMillis'], 1)}"
    except (KeyError, ValueError, TypeError):
        return None


def provenance(environment):
    result = {"java": environment.get("javaVersion", environment.get("java")), "os": environment.get("os"),
              "cpu": environment.get("cpu"), "revision": environment.get("sourceRevision", environment.get("revision"))}
    for field in ("benchmarkClassSha256", "jvmArguments", "seed", "learningRate", "momentum", "beta",
                  "maximumHeapBytes", "loggingLevel", "cpuKernel"):
        result[field] = environment.get(field)
    return result


def native_hash(environment, device):
    value = environment.get("nativeLibrarySha256") or device.get("kernelVersion", "").rsplit("/", 1)[-1]
    return value.lower() if re.fullmatch(r"[a-fA-F0-9]{64}", value) else None


def sidecar(path):
    filename = Path(str(path) + ".environment.txt")
    if not filename.exists():
        return {}, {}
    environment = dict(line.split("=", 1) for line in filename.read_text(encoding="utf-8-sig").splitlines() if "=" in line)
    # Java Map.toString separates entries with ', '; topology commas contain no spaces.
    options = dict(re.findall(r"(?:^|, )([^=, ]+)=([^=]*?)(?=, [^=, ]+=|$)",
                              environment.get("options", "").strip("{}")))
    return environment, options


def timing(row, names):
    opened, trained, closed, total = (integer(row[name]) for name in names)
    if trained == 0 or total != opened + trained + closed:
        raise ValueError("Timing must include positive training and open+training+close == total")
    return {"openNanos": opened, "trainingNanos": trained, "closeNanos": closed, "totalNanos": total}


def json_workload(config, case):
    shape = "x".join(str(integer(width, 1)) for width in case["topology"])
    return (f"json:{shape}:s{integer(case['samples'], 1)}:e{integer(config['epochs'], 1)}"
            f":b{integer(case['batchSize'], 1)}:{config.get('mode', 'MINIBATCH')}"
            f":{config.get('sigmoid', 'EXACT')}:retained={str(config.get('retainedDevice', False)).lower()}")


def read_json(path, report, records):
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    config, environment = data["configuration"], data["environment"]
    fork = process_identity(environment)
    cases = {case["name"]: case for case in data["workloads"]}
    if not cases or len(cases) != len(data["workloads"]) or not config["backends"] or len(set(config["backends"])) != len(config["backends"]):
        raise ValueError("Workloads and configured engines must be nonempty and unique")
    # Keep failed/empty cases in the denominator even if the first measured row is invalid.
    for case in cases.values():
        report["requiredWorkloads"].add(json_workload(config, case))
    repetitions = integer(config["repeats"], 1)
    expected = {(name, engine, repeat) for name in cases for engine in config["backends"]
                for repeat in range(1, repetitions + 1)}
    seen = set()
    if data.get("status") != "passed" or data.get("failure") is not None:
        report["issues"].append(f"{path}: benchmark did not complete successfully")
    for row in data["rounds"]:
        marker = row["workload"], row["backend"], integer(row["round"], 1)
        if marker in seen or marker not in expected:
            raise ValueError(f"Duplicate or unexpected benchmark round {marker}")
        seen.add(marker)
        case = cases[row["workload"]]
        key = json_workload(config, case)
        if integer(row["epochs"]) != integer(config["epochs"], 1) or integer(row["samplesSeen"]) != integer(case["samples"], 1) * integer(config["epochs"], 1):
            raise ValueError("Published epoch/sample counts do not match the workload")
        if number(row["validation"]["maxScaledError"]) > 1:
            raise ValueError("Full-state numerical parity failed")
        elements = 1 + 2 * sum((left + 1) * right for left, right in zip(case["topology"], case["topology"][1:]))
        if integer(row["validation"]["elements"], 1) != elements:
            raise ValueError("Full-state validation element count differs from topology")
        number(row["rmse"])
        records.append({"workload": key, "engine": row["backend"], "precision": row["device"]["precision"],
                        "requestedPrecision": row["device"]["precision"], "actualCpuWorkers": 1 if row["device"].get("backend") == "CPU" else None,
                        "deviceProvenance": {field: row["device"].get(field) for field in
                                             ("backend", "identity", "kernelVersion", "engine", "simdBits", "sigmoid")},
                        "nativeLibrarySha256": native_hash(environment, row["device"]) if row["backend"] in NATIVE else None,
                        "round": marker[2], "fork": fork, "file": str(path),
                        "provenance": provenance(environment),
                        **timing(row, ("openNanos", "trainingNanos", "closeNanos", "totalNanos"))})
    if expected != seen:
        report["issues"].append(f"{path}: missing {len(expected - seen)} expected measured rounds")
    return environment, fork


def read_csv(path, report, records, quality):
    environment, options = sidecar(path)
    fork = process_identity(environment)
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fields = set(reader.fieldnames or ())
        rows = list(reader)
    if not rows:
        raise ValueError("CSV contains no measured rows")
    if {"engine", "total_ns", "max_scaled_error"} <= fields:
        if not environment:
            report["issues"].append(f"{path}: missing environment sidecar (shape/data/fork provenance unavailable)")
        shape = options.get("topology", "2,8,8,8,1").replace(",", "x") if environment else "unknown"
        samples = integer(options.get("samples", 128), 1) if environment else "unknown"
        observed = {}
        for row in rows:
            if environment:
                for column, option, default in (("models", "models", 32), ("epochs", "epochs", 64),
                                                ("batch", "batch", 64), ("workers", "parallelism", 4)):
                    if integer(row[column], 1) != integer(options.get(option, default), 1):
                        raise ValueError(f"Cohort {column} disagrees with environment options")
                if row["precision"] != options.get("precision", "FP64"):
                    raise ValueError("Cohort precision disagrees with environment options")
            key = (f"cohort:{shape}:s{samples}:e{integer(row['epochs'], 1)}:b{integer(row['batch'], 1)}"
                   f":m{integer(row['models'], 1)}:w{integer(row['workers'], 1)}:{row['precision']}:EXACT")
            reference = row["engine"] in ("REFERENCE_MATRIX", "REFERENCE_PARALLEL")
            actual_precision = row.get("actual_precision") or ("FP64" if reference else row["precision"])
            if actual_precision not in ("FP64", "FP32") or reference and actual_precision != "FP64":
                raise ValueError("Invalid actual precision: reference cohort trainers must report FP64")
            workers = integer(row["actual_cpu_workers"]) if row.get("actual_cpu_workers") else None
            if workers is not None and workers > integer(row["workers"], 1):
                raise ValueError("Actual CPU workers exceed the configured maximum")
            repeat = integer(row["round"])
            pair = key, row["engine"]
            if repeat in observed.setdefault(pair, set()):
                raise ValueError(f"Duplicate cohort round {pair}/{repeat}")
            observed[pair].add(repeat)
            if number(row["max_scaled_error"]) > 1:
                raise ValueError("Full-state numerical parity failed")
            records.append({"workload": key, "engine": row["engine"], "precision": actual_precision,
                            "requestedPrecision": row["precision"], "actualCpuWorkers": workers,
                            "deviceProvenance": {},
                            "nativeLibrarySha256": native_hash(environment, {}) if row["engine"] in NATIVE else None,
                            "round": repeat, "fork": fork, "file": str(path),
                            "provenance": provenance(environment),
                            **timing(row, ("open_ns", "training_ns", "close_ns", "total_ns"))})
            if row["precision"] == "FP64":
                report["requiredWorkloads"].add(key)
        repeats = integer(options.get("repeats", 8), 1) if environment else max(integer(row["round"]) for row in rows) + 1
        expected_engines = options.get("engines", "REFERENCE_MATRIX,REFERENCE_PARALLEL,SMALL_SEQUENTIAL,SMALL_CPU,SMALL_CUDA").split(",") if environment else sorted({row["engine"] for row in rows})
        for key in {pair[0] for pair in observed}:
            for engine in expected_engines:
                if observed.get((key, engine), set()) != set(range(repeats)):
                    report["issues"].append(f"{path}: incomplete cohort rounds for {key}/{engine}")
    elif {"seed", "sigmoid", "converged", "nanos", "target", "check_every"} <= fields:
        for row in rows:
            variant = row["precision"] + "/" + row["sigmoid"]
            if variant not in VARIANTS or row["converged"].lower() not in ("true", "false"):
                raise ValueError("Invalid quality variant/convergence flag")
            target, rmse = number(row["target"]), number(row["rmse"])
            converged = row["converged"].lower() == "true"
            if converged != (rmse <= target):
                raise ValueError("Quality convergence flag disagrees with RMSE/target")
            quality.append({"file": str(path), "fork": fork, "variant": variant, "seed": integer(row["seed"]),
                            "epochs": integer(row["epochs"]), "nanos": integer(row["nanos"]),
                            "rmse": rmse, "target": target, "checkEvery": integer(row["check_every"], 1),
                            "budgetEpochs": integer(options.get("epochs", 10000), 1),
                            "expectedSeeds": integer(options.get("seeds", 32), 1),
                            "converged": converged})
    else:
        raise ValueError("Unsupported CSV schema; expected cohort or paired quality report")
    return environment, fork


def summarize_records(records):
    result = {}
    for engine in sorted({row["engine"] for row in records}):
        rows = [row for row in records if row["engine"] == engine]
        forks = []
        for identity in sorted({row["fork"] or row["file"] for row in rows}):
            selected = [row for row in rows if (row["fork"] or row["file"]) == identity]
            forks.append({"files": sorted({row["file"] for row in selected}), "processIdentity": selected[0]["fork"],
                          "rounds": len(selected), "totalNanos": distribution(row["totalNanos"] for row in selected),
                          "trainingNanos": distribution(row["trainingNanos"] for row in selected)})
        result[engine] = {"precision": sorted({row["precision"] for row in rows}),
                          "requestedPrecision": sorted({row["requestedPrecision"] for row in rows}),
                          "actualCpuWorkers": sorted({row["actualCpuWorkers"] for row in rows if row["actualCpuWorkers"] is not None}),
                          "nativeLibrarySha256": sorted({row["nativeLibrarySha256"] for row in rows if row["nativeLibrarySha256"]}),
                          "missingNativeHash": sum(row["nativeLibrarySha256"] is None for row in rows) if engine in NATIVE else 0,
                          "verifiedForks": len({row["fork"] for row in rows if row["fork"]}),
                          "files": len({row["file"] for row in rows}), "perFork": forks,
                          "unverifiedRounds": sum(row["fork"] is None for row in rows),
                          "totalNanos": distribution(row["totalNanos"] for row in rows),
                          "trainingNanos": distribution(row["trainingNanos"] for row in rows),
                          "rawRounds": [{key: row[key] for key in ("file", "fork", "round", "totalNanos", "trainingNanos")} for row in rows]}
    return result


def summarize_quality(rows):
    results = []
    for key in sorted({(row["target"], row["checkEvery"], row["budgetEpochs"]) for row in rows}):
        group = [row for row in rows if (row["target"], row["checkEvery"], row["budgetEpochs"]) == key]
        runs = {}
        issues = []
        for row in group:
            pair = row["fork"] or row["file"], row["seed"]
            variants = runs.setdefault(pair, {})
            if row["variant"] in variants:
                issues.append(f"Duplicate paired observation {pair}/{row['variant']}")
            variants[row["variant"]] = row
        complete = [value for value in runs.values() if set(value) == set(VARIANTS)]
        seeds = {seed for (_, seed), value in runs.items() if set(value) == set(VARIANTS)}
        expected_by_fork = {}
        for row in group:
            identity = row["fork"] or row["file"]
            expected_by_fork[identity] = max(expected_by_fork.get(identity, 0), row["expectedSeeds"])
        expected_runs = max(len(runs), sum(expected_by_fork.values()))
        if len(complete) != len(runs):
            issues.append("Missing math variants in paired seed runs")
        if expected_runs != len(runs):
            issues.append("Missing complete seed runs from the configured seed count")
        if any(row["epochs"] > row["budgetEpochs"] or not row["converged"] and row["epochs"] != row["budgetEpochs"] for row in group):
            issues.append("Quality run did not honor its epoch budget")
        if len(seeds) < 32:
            issues.append("Fewer than 32 distinct fully paired seeds")
        variants = {}
        for variant in VARIANTS:
            observed = [value[variant] for value in runs.values() if variant in value]
            success = [row for row in observed if row["converged"]]
            censored = [row for row in observed if not row["converged"]]
            variants[variant] = {"observedSeedRuns": len(observed), "expectedSeedRuns": expected_runs,
                                 "converged": len(success), "censoredFailures": len(censored),
                                 "missing": expected_runs - len(observed),
                                 "convergenceFraction": len(success) / expected_runs if expected_runs else None,
                                 "conditionalTimeToTargetNanos": distribution(row["nanos"] for row in success),
                                 "conditionalEpochsToTarget": distribution(row["epochs"] for row in success),
                                 "censoredElapsedNanos": distribution(row["nanos"] for row in censored)}
        results.append({"target": key[0], "checkEvery": key[1], "budgetEpochs": key[2],
                        "pairedSeedRuns": len(complete), "distinctPairedSeeds": len(seeds),
                        "complete": not issues, "issues": issues, "variants": variants})
    return results


def aggregate(jsonpaths=(), csvpaths=(), required_workloads=()):
    report = {"schemaVersion": 1, "issues": [], "inputs": [], "requiredWorkloads": set(),
              "policy": {"minimumVerifiedForks": 3, "nativeMedianRatioMaximum": 0.8,
                         "nativeP95RatioMaximum": 1.0, "p95": "nearest rank of all raw measured rounds",
                         "forkIdentity": "distinct processId:jvmStartMillis, at least three separate files",
                         "qualityTimes": "conditional on convergence; failures remain in the seed-run denominator"}}
    records, quality, seen_paths = [], [], set()
    for value, kind in [(path, "json") for path in jsonpaths] + [(path, "csv") for path in csvpaths]:
        path = Path(value).resolve()
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if path in seen_paths:
                raise ValueError("Duplicate input path")
            seen_paths.add(path)
            environment, fork = (read_json(path, report, records) if kind == "json"
                                 else read_csv(path, report, records, quality))
            report["inputs"].append({"path": str(path), "sha256": digest, "kind": kind,
                                     "processIdentity": fork, "environment": environment})
        except (OSError, ValueError, TypeError, KeyError, OverflowError) as failure:
            report["issues"].append(f"{path}: {failure}")
    if not report["inputs"]:
        report["issues"].append("No readable benchmark evidence")
    required = sorted(set(required_workloads) or report["requiredWorkloads"])
    report["requiredWorkloads"] = required
    work = {}
    duplicated = set()
    for row in records:
        marker = row["workload"], row["engine"], row["fork"] or row["file"], row["round"]
        if marker in duplicated:
            report["issues"].append(f"Repeated round from the same JVM: {marker}")
        duplicated.add(marker)
        work.setdefault(row["workload"], []).append(row)
    for key, rows in work.items():
        for field in rows[0]["provenance"]:
            values = {json.dumps(row["provenance"][field], sort_keys=True) for row in rows if row["provenance"][field] is not None}
            if len(values) > 1:
                report["issues"].append(f"{key}: incompatible {field} provenance across benchmark inputs")
        for engine in {row["engine"] for row in rows}:
            selected = [row for row in rows if row["engine"] == engine]
            for field in ("deviceProvenance", "nativeLibrarySha256", "actualCpuWorkers", "precision"):
                values = {json.dumps(row[field], sort_keys=True) for row in selected if row[field] is not None}
                if len(values) > 1:
                    report["issues"].append(f"{key}/{engine}: incompatible {field} across benchmark forks")
    candidates = sorted({NATIVE[row["engine"]] for row in records if row["engine"] in NATIVE} or {"NATIVE"})
    report["workloads"] = {}
    for key in sorted(set(work) | set(required)):
        engines = summarize_records(work.get(key, []))
        baselines = [engine for engine in engines if engine in JVM and engines[engine]["precision"] == ["FP64"]]
        baseline = min(baselines, key=lambda engine: (engines[engine]["totalNanos"]["median"], engine)) if baselines else None
        gates = {}
        for candidate in candidates:
            names = [name for name in engines if NATIVE.get(name) == candidate and engines[name]["precision"] == ["FP64"]]
            selected = names[0] if len(names) == 1 else None
            reasons = []
            if baseline is None:
                reasons.append("Missing applicable FP64 JVM baseline")
            if selected is None:
                reasons.append("Missing or ambiguous native candidate")
            median_ratio = p95_ratio = None
            for name in (baseline, selected):
                if name and (engines[name]["verifiedForks"] < 3 or engines[name]["files"] < 3):
                    reasons.append(f"{name} needs >=3 separate files with distinct verified JVM identities")
                if name and engines[name]["unverifiedRounds"]:
                    reasons.append(f"{name} contains rounds without a verified JVM identity")
            if selected and engines[selected]["missingNativeHash"]:
                reasons.append("Native library SHA-256 is missing or invalid")
            if baseline and selected:
                median_ratio = engines[selected]["totalNanos"]["median"] / engines[baseline]["totalNanos"]["median"]
                p95_ratio = engines[selected]["totalNanos"]["p95"] / engines[baseline]["totalNanos"]["p95"]
                if median_ratio > 0.8:
                    reasons.append("Native median improvement is below 20%")
                if p95_ratio > 1.0:
                    reasons.append("Native p95 regressed")
            gates[candidate] = {"engine": selected, "passed": not reasons and not report["issues"],
                                "medianRatio": median_ratio, "p95Ratio": p95_ratio, "reasons": reasons}
        report["workloads"][key] = {"required": key in required, "baseline": baseline, "engines": engines, "candidateGates": gates}
    report["candidateGates"] = {candidate: {"passed": bool(required) and not report["issues"] and all(
        report["workloads"][key]["candidateGates"][candidate]["passed"] for key in required),
        "failedWorkloads": [key for key in required if not report["workloads"][key]["candidateGates"][candidate]["passed"]]}
        for candidate in candidates}
    report["nativeGatePassed"] = any(gate["passed"] for gate in report["candidateGates"].values())
    report["quality"] = summarize_quality(quality)
    report["qualityEvidenceComplete"] = bool(report["quality"]) and all(group["complete"] for group in report["quality"])
    return report


def markdown(report):
    lines = ["# SMALL benchmark evidence", "", f"Native retention gate: **{'passed' if report['nativeGatePassed'] else 'not passed'}**.",
             "", "Each native variant must pass every required workload. Totals include open, training and close.",
             "", "| Workload | FP64 JVM baseline | Native | Median ratio | p95 ratio | Gate |",
             "| --- | --- | --- | ---: | ---: | --- |"]
    for key, work in report["workloads"].items():
        if not work["required"]:
            continue
        for candidate, gate in work["candidateGates"].items():
            ratio = lambda value: "—" if value is None else f"{value:.3f}"
            lines.append(f"| `{key}` | {work['baseline'] or 'missing'} | {candidate} | {ratio(gate['medianRatio'])} | {ratio(gate['p95Ratio'])} | {'pass' if gate['passed'] else 'not passed'} |")
    if report["issues"]:
        lines += ["", "Evidence issues:", ""] + [f"- {issue}" for issue in report["issues"]]
    lines += ["", "Raw rounds, per-fork median/p95, verified JVM identities and failure reasons are retained in the JSON."]
    for group in report["quality"]:
        lines += ["", f"## Paired quality: target {group['target']}, check every {group['checkEvery']}", "",
                  f"Fully paired distinct seeds: {group['distinctPairedSeeds']}; evidence complete: {group['complete']}.", "",
                  "Times below are conditional on convergence. Unsuccessful runs are censored and remain in the denominator.", "",
                  "| Variant | Converged / expected | Censored | Missing | Conditional median ms | Conditional p95 ms |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for variant, stats in group["variants"].items():
            times = stats["conditionalTimeToTargetNanos"]
            median, p95 = (f"{times['median'] / 1e6:.3f}", f"{times['p95'] / 1e6:.3f}") if times else ("—", "—")
            lines.append(f"| {variant} | {stats['converged']} / {stats['expectedSeedRuns']} | {stats['censoredFailures']} | {stats['missing']} | {median} | {p95} |")
        lines += [""] + [f"- {issue}" for issue in group["issues"]]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonpaths", nargs="+", default=[])
    parser.add_argument("--csvpaths", nargs="+", default=[])
    parser.add_argument("--required-workloads", nargs="+", default=[])
    parser.add_argument("--output", type=Path, required=True, help="Output prefix for .json and .md")
    args = parser.parse_args(argv)
    report = aggregate(args.jsonpaths, args.csvpaths, args.required_workloads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Path(str(args.output) + ".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    Path(str(args.output) + ".md").write_text(markdown(report), encoding="utf-8")
    print(f"Native gate {'passed' if report['nativeGatePassed'] else 'not passed'}; reports: {args.output}.json / .md")
    return 0 if report["nativeGatePassed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
