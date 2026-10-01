#!/usr/bin/env python3
"""Compare complete, paired fixed-search forks with matching provenance and work."""
import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


MODES = ("REFERENCE", "OPTIMIZED")
DYNAMIC_ENVIRONMENT = {"started", "pid", "orderOffset"}
PROVENANCE = ("sourceRevision", "java", "vm", "os", "osVersion", "arch", "cpuModel", "heapMaxBytes")
DEVICE_FIELDS = ("deviceIdentity", "backend", "precision", "sigmoid", "engine")


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def finite(value, minimum=None):
    return type(value) in (int, float) and math.isfinite(value) and (minimum is None or value >= minimum)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def environment_protocol(environment):
    require(integer(environment.get("pid"), 1) and isinstance(environment.get("started"), str)
            and bool(environment["started"].strip()), "Missing process identity")
    for key in ("epochsPerSeed", "trainingSamples", "validationSamples", "batchSize", "checkEvery", "repeats"):
        require(integer(environment.get(key), 1), "Invalid environment " + key)
    require(integer(environment.get("warmups")), "Invalid warmup count")
    require(bool(environment.get("datasetFingerprint")), "Missing dataset fingerprint")
    manifest = environment.get("manifest", [])
    require(bool(manifest) and all(isinstance(shape, list) and shape and
            all(integer(width, 1) for width in shape) for shape in manifest), "Invalid manifest")
    require(len({tuple(shape) for shape in manifest}) == len(manifest), "Duplicate manifest shape")
    seeds = environment.get("seeds", [])
    require(bool(seeds) and all(type(seed) is int for seed in seeds) and len(set(seeds)) == len(seeds), "Invalid seed manifest")
    return {key: value for key, value in environment.items() if key not in DYNAMIC_ENVIRONMENT}


def parse_case(case):
    parts = case.split("/")
    require(len(parts) == 5 and parts[2] in MODES, "Invalid comparison case")
    engine, backend, execution, workers, seed = parts
    require(workers.startswith("workers=") and int(workers.split("=")[1]) > 0 and
            seed.startswith("searchSeed="), "Invalid worker or search seed")
    int(seed.split("=")[1])
    return engine, backend, execution, workers, seed


def validate_round(record, protocol, engine, backend):
    require(record.get("workComplete") is True and record.get("termination") in ("COMPLETED", "TRIAL_BUDGET"),
            "Incomplete or failed work")
    for key in ("failedTrials", "partialCandidates"):
        require(type(record.get(key)) is int and record[key] == 0, "Incomplete or failed work")
    require(integer(record.get("totalNanos"), 1) and record["totalNanos"] <= 2**63 - 1, "Invalid elapsed time")
    shapes = {tuple(shape) for shape in protocol["manifest"]}
    candidates = record.get("candidates", [])
    require(len(candidates) == len(shapes) and {tuple(c["hidden"]) for c in candidates} == shapes,
            "Unequal work: candidate manifest mismatch")
    seeds, epochs, samples = set(protocol["seeds"]), protocol["epochsPerSeed"], protocol["trainingSamples"]
    total = len(shapes) * len(seeds)
    expected_counts = dict(completeTrials=total, evaluatedCandidates=len(shapes), committedEpochs=total * epochs,
                           sampleUpdates=total * epochs * samples)
    require(all(type(record.get(key)) is int and record[key] == count for key, count in expected_counts.items()),
            "Unequal work: aggregate counters mismatch")
    trials = {}
    for candidate in candidates:
        hidden = tuple(candidate["hidden"])
        topology = (2,) + hidden + (1,)
        parameters = sum((source + 1) * target for source, target in zip(topology, topology[1:]))
        require(candidate.get("complete") is True and candidate.get("parameters") == parameters,
                "Incomplete candidate or wrong parameter count")
        rows = candidate.get("trials", [])
        require(len(rows) == len(seeds) and {trial["seed"] for trial in rows} == seeds,
                "Unequal work: seed manifest mismatch")
        for trial in rows:
            require(type(trial["seed"]) is int and trial.get("state") == "COMPLETED" and
                    type(trial.get("epochs")) is int and trial["epochs"] == epochs and
                    type(trial.get("sampleUpdates")) is int and trial["sampleUpdates"] == epochs * samples and
                    not trial.get("failure"), "Incomplete trial")
            best_epoch = trial.get("bestEpoch")
            require(integer(best_epoch) and best_epoch <= epochs and
                    (best_epoch == epochs or best_epoch % protocol["checkEvery"] == 0), "Invalid best epoch")
            values = trial.get("parametersAtBest")
            require(isinstance(values, list) and len(values) == parameters and all(finite(value) for value in values),
                    "Invalid best parameters")
            require(all(finite(trial.get(key), 0) for key in ("bestRmse", "finalRmse")), "Invalid RMSE")
            for key, expected in (("backend", backend), ("engine", engine), ("precision", protocol["precision"]),
                                  ("sigmoid", protocol["sigmoid"])):
                require(key not in trial or trial[key] == expected, "Effective device configuration mismatch")
            trials[(hidden, trial["seed"])] = trial
    return trials



def valid_execution_route(engine, backend, mode, protocol, trial):
    """Audit TensorFlow execution routes independently of topology-family hints."""
    route, kernel = (trial.get(key) for key in ("route", "kernel"))
    if engine not in ("SMALL", "REFERENCE") or backend not in ("CPU", "GPU"):
        return False
    if mode == "BATCHED":
        return route == "TENSOR_BATCH" and isinstance(kernel, str) and bool(
            re.fullmatch(r"tensorflow-.+-batched-v1-t[1-9][0-9]*", kernel))
    prefix, suffix = "tensorflow-", "-dense-loop-v2"
    if not isinstance(kernel, str) or not kernel.startswith(prefix) or not kernel.endswith(suffix) or len(kernel) <= len(prefix) + len(suffix):
        return False
    if mode not in MODES:
        return False
    if engine == "REFERENCE":
        return route == "SESSION"
    if mode == "REFERENCE":
        return route in ("SESSION", "COHORT")
    if backend == "CPU":
        return route == "SESSION"
    if backend == "GPU":
        return route == "TENSORFLOW_QUEUE"
    return False


def summarize(paths):
    groups, failures, seen_paths, excluded_forks = {}, [], set(), set()
    for path in paths:
        path = Path(path).resolve()
        if path in seen_paths:
            continue
        seen_paths.add(path)
        records = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        environments = [record for record in records if record.get("type") == "environment"]
        require(len(environments) == 1 and records[0] == environments[0], "Expected one initial environment: " + str(path))
        environment = environments[0]
        if environment.get("mode") != "fixed":
            continue
        protocol = environment_protocol(environment)
        protocol_key = canonical(protocol)
        fork = (environment["started"], environment["pid"])
        # A failed or budget-truncated process cannot supply selectively surviving fast rounds.
        for record in records:
            if record.get("type") in ("failure", "budget"):
                failures.append({"path": str(path), "failure": record})
                excluded_forks.add((protocol_key, fork))
        for record in records:
            if record.get("type") != "round":
                continue
            engine, backend, execution, workers, seed = parse_case(record["case"])
            key = (engine, backend, workers, seed, protocol_key)
            group = groups.setdefault(key, {"protocol": protocol, "rows": defaultdict(dict), "invalid": set()})
            number = record.get("round")
            try:
                require(type(number) is int and -protocol["warmups"] <= number < protocol["repeats"], "Invalid round index")
                validate_round(record, protocol, engine, backend)
                # Reject non-finite numbers anywhere, including metadata not used in parity.
                canonical(record)
            except (KeyError, TypeError, ValueError, OverflowError) as failure:
                failures.append({"path": str(path), "case": record["case"], "round": number, "reason": str(failure)})
                group["invalid"].add(fork)
                continue
            measurement = (fork, number)
            previous = group["rows"][execution].get(measurement)
            require(previous is None or previous == record, "Conflicting duplicate measurement: " + record["case"])
            group["rows"][execution][measurement] = record
    comparisons = []
    for (engine, backend, workers, seed, protocol_key), group in sorted(groups.items()):
        rows, protocol = group["rows"], group["protocol"]
        forks = {fork for mode in rows.values() for fork, _ in mode}
        expected_rounds = set(range(-protocol["warmups"], protocol["repeats"]))
        paired_forks = {fork for fork in forks if fork not in group["invalid"] and
                        (protocol_key, fork) not in excluded_forks and all(
                            {number for process, number in rows[mode] if process == fork} == expected_rounds for mode in MODES)}
        for fork in forks - paired_forks:
            failures.append({"case": "/".join((engine, backend, workers, seed)), "fork": fork,
                             "reason": "Excluded entire unpaired, incomplete or failed fork"})
        selected = {mode: [row for (fork, number), row in sorted(rows[mode].items()) if fork in paired_forks and number >= 0]
                    for mode in MODES}
        distributions = {}
        for mode, samples in selected.items():
            times = [sample["totalNanos"] / 1e6 for sample in samples]
            distributions[mode] = {"medianMs": statistics.median(times) if times else None,
                                   "p95Ms": percentile(times, .95) if times else None,
                                   "rounds": len(times), "forks": len(paired_forks)}
        reference, optimized = distributions["REFERENCE"], distributions["OPTIMIZED"]
        scaled_error, epoch_match, stable_device, device_complete = 0.0, True, True, True
        valid_routes = True
        observed_routes = {mode: Counter() for mode in MODES}
        if selected["REFERENCE"]:
            anchor = validate_round(selected["REFERENCE"][0], protocol, engine, backend)
            hardware = {key: tuple(trial.get(field) for field in DEVICE_FIELDS) for key, trial in anchor.items()}
            for mode, samples in selected.items():
                route_signatures = defaultdict(set)
                for record in samples:
                    trials = validate_round(record, protocol, engine, backend)
                    for key, trial in trials.items():
                        routing = tuple(trial.get(field) for field in ("route", "kernel"))
                        observed_routes[mode][routing] += 1
                        # Preserve each shape/route's TensorFlow kernel, without fixing which seed takes that route.
                        route_signatures[(key[0], routing[0])].add(routing[1:])
                        valid_routes &= valid_execution_route(engine, backend, mode, protocol, trial)
                        actual_device = tuple(trial.get(field) for field in DEVICE_FIELDS)
                        stable_device &= actual_device == hardware[key]
                        device_complete &= all(value is not None and value != "" for value in actual_device)
                        expected = anchor[key]
                        epoch_match &= trial["bestEpoch"] == expected["bestEpoch"]
                        actual = trial["parametersAtBest"] + [trial["bestRmse"], trial["finalRmse"]]
                        expected_values = expected["parametersAtBest"] + [expected["bestRmse"], expected["finalRmse"]]
                        for value, expected_value in zip(actual, expected_values):
                            error = abs(value - expected_value) / (1e-10 + 1e-8 * abs(expected_value))
                            # Extreme finite inputs can overflow their difference; still emit valid JSON and fail parity.
                            scaled_error = max(scaled_error, error if math.isfinite(error) else float.fromhex("0x1.fffffffffffffp+1023"))
                valid_routes &= all(len(signatures) == 1 for signatures in route_signatures.values())
        enough = all(d["forks"] >= 3 and d["rounds"] >= 9 for d in distributions.values())
        provenance = all(protocol.get(key) not in (None, "", "unspecified", "unknown") for key in PROVENANCE)
        eligible = bool(selected["REFERENCE"])
        speedup = reference["medianMs"] / optimized["medianMs"] if eligible else None
        gain = 1 - optimized["medianMs"] / reference["medianMs"] if eligible else None
        p95_regression = optimized["p95Ms"] > reference["p95Ms"] if eligible else None
        comparisons.append({"case": "/".join((engine, backend, workers, seed)), "protocol": protocol,
                            "reference": reference, "optimized": optimized, "speedup": speedup,
                            "medianReductionFraction": gain, "p95Regression": p95_regression,
                            "matchingBestEpochs": epoch_match if eligible else None,
                            "maximumScaledParameterOrScoreError": scaled_error if eligible else None,
                            "threeForkEvidence": enough, "provenanceComplete": provenance and device_complete,
                            "stableEffectiveDevice": stable_device, "validExecutionRoutes": valid_routes,
                            "observedExecutionRoutes": {mode: [dict(route=route, kernel=kernel, trials=count)
                                for (route, kernel), count in sorted(counts.items(), key=lambda item: str(item[0]))]
                                for mode, counts in observed_routes.items()},
                            "qualifies": enough and provenance and device_complete and stable_device and valid_routes and
                            protocol["precision"] == "FP64" and epoch_match and scaled_error <= 1 and
                            optimized["medianMs"] <= .90 * reference["medianMs"] and not p95_regression})
    return {"comparisons": comparisons, "incompleteOrFailed": failures,
            "pairingScope": "Same process, same round, same complete manifest and seeds; all configured warmups and rounds required.",
            "parityScope": "Every measured reference and optimized best snapshot, best/final RMSE and best epoch; optimizer/shuffle parity is covered by tests."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(summarize(args.reports), indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
