#!/usr/bin/env python3
"""Read flushed quality JSONL only; never run a model or pool seeds across searches."""
import argparse
import collections
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import statistics
import sys

SEEDS = (1, 42, 123, 999, 2026)
EPOCHS = 1_000_000
TARGET = 0.01
MAX_TRIALS = 60
DATASET_FINGERPRINT = "f8fc0ceb97466e339301faa42b05b45df6aee3c878a98986e65771cdac1d4dc9"
CASE = re.compile(r"SMALL/(CPU|CUDA)/(REFERENCE|OPTIMIZED)/workers=(\d+)/searchSeed=(42|123)$")


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def unopened_partial(t):
    """A cancelled-before-open or failed-open trial has no device or checkpoint."""
    return (t.get("state") in ("CANCELLED", "FAILED")
            and all(integer(t.get(k)) and t[k] == 0 for k in ("epochs", "bestEpoch", "sampleUpdates"))
            and all(t.get(k) is None for k in ("backend", "engine", "precision", "sigmoid", "kernel", "deviceIdentity", "device", "simdBits",
                                               "bestRmse", "finalRmse", "parametersAtBest", "phaseNanos"))
            and t.get("historySize") in (None, 0)
            and (t.get("state") != "FAILED" or bool(t.get("failure"))))


def candidate(raw, samples):
    issues = []
    hidden = raw.get("hidden")
    shape_ok = isinstance(hidden, list) and 1 <= len(hidden) <= 4 and all(integer(x) and x in (4, 8, 16) for x in hidden)
    parameter_count = sum((a + 1) * b for a, b in zip([2] + hidden, hidden + [1])) if shape_ok else None
    if not shape_ok or raw.get("parameters") != parameter_count:
        issues.append("Invalid SMALL topology or parameter count")
    raw_trials = raw.get("trials", [])
    if not isinstance(raw_trials, list):
        raw_trials = []; issues.append("Trials field is not an array")
    reported_seeds = [t.get("seed") for t in raw_trials]
    exact_seeds = len(reported_seeds) == 5 and all(integer(s) for s in reported_seeds) and sorted(reported_seeds) == sorted(SEEDS)
    if not exact_seeds:
        issues.append("Requires exactly the five distinct protocol training seeds")
    trials = []
    for t in raw_trials:
        unopened = unopened_partial(t)
        score_ok = all(finite(t.get(k)) and t[k] >= 0 for k in ("bestRmse", "finalRmse")) and t["bestRmse"] <= t["finalRmse"]
        epochs = t.get("epochs")
        best_epoch = t.get("bestEpoch")
        counters_ok = integer(epochs) and 0 <= epochs <= EPOCHS and integer(best_epoch) and 0 <= best_epoch <= epochs and best_epoch % 25 == 0
        updates_ok = counters_ok and integer(t.get("sampleUpdates")) and t["sampleUpdates"] == epochs * samples
        declared = t.get("state") == "COMPLETED"
        complete = declared and epochs == EPOCHS and updates_ok and score_ok and not t.get("failure")
        trial_issues = []
        if not counters_ok or not updates_ok: trial_issues.append("Invalid epoch/best-epoch/sample-update counters")
        if not score_ok and not unopened: trial_issues.append("Best/final RMSE must be finite, nonnegative, and best must not exceed final")
        if declared and not complete: trial_issues.append("Declared completed trial fails full-budget/finite-score/failure checks")
        if not unopened and (t.get("precision") != "FP64" or t.get("sigmoid") != "EXACT" or t.get("engine") != "SMALL"):
            trial_issues.append("Trial execution metadata differs from FP64/EXACT/SMALL")
        complete = complete and not trial_issues
        trials.append({k: t.get(k) for k in ("seed", "state", "epochs", "bestEpoch", "bestRmse", "finalRmse", "sampleUpdates", "failure", "backend", "route", "precision", "sigmoid", "engine", "kernel", "deviceIdentity")} |
                      {"unopenedZeroWorkPartial": unopened,
                       "fullBudgetComplete": complete, "qualifyingSuccess": complete and t["bestRmse"] <= TARGET,
                       "partialObservedThresholdCrossing": not complete and finite(t.get("bestRmse")) and t["bestRmse"] <= TARGET,
                       "issues": trial_issues})
    all_complete = shape_ok and exact_seeds and not issues and all(t["fullBudgetComplete"] for t in trials)
    successes = sum(t["qualifyingSuccess"] for t in trials)
    finite_best = [t["bestRmse"] for t in trials if finite(t["bestRmse"])]
    return {"hidden": hidden, "parameters": parameter_count, "reportedComplete": raw.get("complete"),
            "fullyCompletedFiveSeeds": all_complete, "successesAtFullBudget": successes,
            "reliableCompletedCandidate": all_complete and successes >= 4,
            "reportedSuccesses": raw.get("successes"), "reportedMedianRmse": raw.get("medianRmse"),
            "completedFiveSeedMedianBestRmse": statistics.median(finite_best) if all_complete else None,
            "stateCounts": dict(collections.Counter(t["state"] for t in trials)),
            "trials": trials, "issues": issues}


def analyze_round(raw, env, protocol_issues):
    issues = list(protocol_issues)
    name = raw.get("case", "")
    match = CASE.fullmatch(name)
    if not match: issues.append("Unexpected case or search seed")
    samples = env.get("trainingSamples", 0)
    if not integer(samples) or samples <= 0:
        samples = 0; issues.append("Missing valid training sample count")
    items = [candidate(c, samples) for c in raw.get("candidates", [])]
    shapes = [tuple(c["hidden"]) if isinstance(c["hidden"], list) else None for c in items]
    if len(set(shapes)) != len(shapes): issues.append("Duplicate candidate topology in one search")
    trials = [t for c in items for t in c["trials"]]
    if len(trials) > MAX_TRIALS: issues.append("Search exceeds the 60-trial protocol budget")
    if match and (int(match[3]) not in env.get("workers", []) or match[1] not in env.get("backends", []) or match[2] not in env.get("executions", [])):
        issues.append("Case differs from environment selection")
    if match and any(t["backend"] != match[1] and not t["unopenedZeroWorkPartial"] for t in trials):
        issues.append("Trial backend differs from case or is missing outside an unopened zero-work partial trial")
    for field, actual in (("committedEpochs", sum(t["epochs"] for t in trials if integer(t["epochs"]))),
                          ("sampleUpdates", sum(t["sampleUpdates"] for t in trials if integer(t["sampleUpdates"]))),
                          ("completeTrials", sum(t["state"] == "COMPLETED" for t in trials))):
        if raw.get(field) != actual: issues.append("Reported " + field + " differs from trial records")
    full = not issues and len(trials) == MAX_TRIALS and len(items) == MAX_TRIALS // 5 and all(c["fullyCompletedFiveSeeds"] for c in items) and raw.get("termination") in ("COMPLETED", "TRIAL_BUDGET")
    if raw.get("workComplete") != full: issues.append("Harness workComplete differs from recomputed protocol completion")
    # Qualification is local to an intact record. Never combine trial seeds across cases/files.
    if issues:
        full = False
        for c in items: c["reliableCompletedCandidate"] = False
    reliable = sorted((c for c in items if c["reliableCompletedCandidate"]), key=lambda c: (c["parameters"], c["completedFiveSeedMedianBestRmse"], c["hidden"]))
    scope = "within_completed_60_trial_search" if full else "within_completed_candidates_of_partial_search"
    qualified = [{"hidden": c["hidden"], "parameters": c["parameters"], "successes": c["successesAtFullBudget"]} for c in reliable]
    winner = raw.get("winner")
    independent = raw.get("independentTest")
    independent_summary = None
    if independent is not None:
        selected = next((c for c in reliable if c["hidden"] == winner), None)
        representative = None
        if selected:
            median = selected["completedFiveSeedMedianBestRmse"]
            representative = min(selected["trials"], key=lambda t: (abs(t["bestRmse"] - median), t["bestRmse"], t["seed"]))["seed"]
        independent_summary = {"scope": "harness winner representative seed only; not five-seed reliability or independent architecture selection",
                               "winner": winner, "representativeSeed": representative,
                               "winnerQualifiesUnderProtocol": selected is not None, "harnessResult": independent}
    return {"case": name, "round": raw.get("round"), "termination": raw.get("termination"),
            "status": "invalid_record" if issues else "full_work_complete" if full else "partial_search",
            "fullWorkComplete": full, "reportedWorkComplete": raw.get("workComplete"),
            "elapsedSeconds": raw.get("totalNanos", 0) / 1e9, "trialCount": len(trials),
            "fullBudgetCompletedTrials": sum(t["fullBudgetComplete"] for t in trials),
            "stateCounts": dict(collections.Counter(t["state"] for t in trials)),
            "committedEpochs": raw.get("committedEpochs"), "sampleUpdates": raw.get("sampleUpdates"),
            "fullyCompletedCandidates": sum(c["fullyCompletedFiveSeeds"] for c in items),
            "reliableCompletedCandidates": qualified, "reliabilityScope": scope,
            "smallestObservedReliableCandidate": qualified[0] if qualified else None,
            "globalMinimumEstablished": False, "harnessWinner": winner,
            "harnessTargetMet": raw.get("targetMet"), "independentTest": independent_summary,
            "candidates": items, "issues": issues}


def analyze_file(label, filename, expected_search_seeds):
    path = Path(filename)
    base = {"label": label, "path": str(path), "comparisonScope": "Separate host/process result; no cross-host throughput inference"}
    if "quality-superseded" in path.parts:
        return base | {"status": "excluded_superseded", "reason": "Cancelled before the progress-snapshot race fix; no qualification", "runs": []}
    if not path.is_file(): return base | {"status": "missing", "runs": []}
    data = path.read_bytes(); rows = []; parse_issues = []; trailing = False
    lines = data.splitlines()
    for number, line in enumerate(lines, 1):
        try: rows.append(json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value))))
        except (ValueError, UnicodeDecodeError) as error:
            if number == len(lines) and not data.endswith(b"\n"):
                trailing = True
            else: parse_issues.append(f"Invalid JSON at line {number}: {error}")
    envs = [r for r in rows if r.get("type") == "environment"]
    env = envs[0] if len(envs) == 1 else {}
    issues = list(parse_issues)
    if len(envs) != 1: issues.append("Expected exactly one environment record")
    expected = {"mode": "quality", "epochsPerSeed": EPOCHS, "seeds": list(SEEDS), "targetRmse": TARGET,
                "requiredSuccesses": 4, "batchSize": 1, "precision": "FP64", "sigmoid": "EXACT", "engine": "SMALL", "checkEvery": 25,
                "trainingSamples": 176, "validationSamples": 44, "datasetFingerprint": DATASET_FINGERPRINT}
    for key, value in expected.items():
        if env.get(key) != value: issues.append(f"Environment {key} must equal {value!r}")
    if not env.get("datasetFingerprint") or env.get("sourceRevision") in (None, "", "unspecified"):
        issues.append("Missing dataset/source provenance")
    raw_runs = [r for r in rows if r.get("type") == "round"]
    run_cases = [r.get("case") for r in raw_runs]
    if len(set(run_cases)) != len(run_cases): issues.append("Quality case occurs more than once in one process")
    runs = [analyze_round(r, env, issues) for r in raw_runs]
    failures = [r for r in rows if r.get("type") == "failure"]
    if failures:
        for run in runs:
            if any(f.get("case") == run["case"] for f in failures):
                run["issues"].append("Harness failure record also exists for this case")
                run["status"] = "invalid_record"; run["fullWorkComplete"] = False
                run["reliableCompletedCandidates"] = []; run["smallestObservedReliableCandidate"] = None
                for c in run["candidates"]: c["reliableCompletedCandidate"] = False
                if run["independentTest"]: run["independentTest"]["winnerQualifiesUnderProtocol"] = False
    expected_cases = [f"SMALL/{backend}/{execution}/workers={workers}/searchSeed={seed}" for workers, backend, execution, seed in itertools.product(
        env.get("workers", []), env.get("backends", []), env.get("executions", []), expected_search_seeds)]
    progress = {}
    for r in rows:
        if r.get("type") == "progress": progress[r.get("case")] = r
    absent = [{"case": case, "status": "progress_only_unqualified" if case in progress else "not_observed_no_result", "latestProgress": progress.get(case)}
              for case in expected_cases if case not in run_cases]
    all_complete = bool(expected_cases) and set(run_cases) == set(expected_cases) and all(r["fullWorkComplete"] for r in runs) and not failures
    return base | {"status": "invalid_input" if issues else "full_protocol_complete" if all_complete else "partial_run" if runs else "progress_only_unqualified" if progress else "no_results",
                   "sha256AtRead": hashlib.sha256(data).hexdigest(), "bytesAtRead": len(data), "trailingPartialLineIgnored": trailing,
                   "allExpectedSearchCasesComplete": all_complete,
                   "searchCaseCounts": {"expected": len(expected_cases), "reported": len(runs), "fullWorkComplete": sum(r["fullWorkComplete"] for r in runs), "unreported": len(absent)},
                   "environment": env, "issues": issues, "runs": runs, "unreportedExpectedCases": absent,
                   "latestProgressByCase": progress, "failures": failures, "budgetRecords": [r for r in rows if r.get("type") == "budget"]}


def markdown(report):
    text = ["# Spiral quality evidence", "", "Target: validation RMSE ≤0.01 in at least 4/5 specified seeds; all five trials must finish 1,000,000 epochs. Each search has a 60-trial budget. No global minimum is established.", "",
            "Host/process results remain separate. Concurrent local CPU/GPU runs support quality observations, not throughput comparisons. Progress and partial threshold crossings never establish reliability.", "",
            "| Source | Case | Status | Full-budget trials | Reliable completed candidates |", "|---|---|---|---:|---|"]
    for item in report["inputs"]:
        for run in item["runs"]:
            reliable = "; ".join(f"{c['hidden']} ({c['parameters']} params, {c['successes']}/5)" for c in run["reliableCompletedCandidates"]) or "None"
            text.append(f"| {item['label']} | {run['case']} | {run['status']} | {run['fullBudgetCompletedTrials']}/60 | {reliable} |")
        if not item["runs"]: text.append(f"| {item['label']} | — | {item['status']} | — | Unqualified |")
    for item in report["inputs"]:
        text += ["", "## " + item["label"], "", "Source: `" + item["path"] + "`"]
        if item.get("issues"): text += ["", "Input issues: " + "; ".join(item["issues"])]
        for run in item["runs"]:
            text += ["", "### " + run["case"], "", "Reliability scope: " + run["reliabilityScope"] + "."]
            if run["issues"]: text += ["", "Record issues: " + "; ".join(run["issues"])]
            for c in run["candidates"]:
                text += ["", f"Candidate {c['hidden']} / {c['parameters']} parameters: fully completed={c['fullyCompletedFiveSeeds']}; reliable={c['reliableCompletedCandidate']}; successes={c['successesAtFullBudget']}/5.", "",
                         "| Seed | State | Epochs | Best epoch | Best validation RMSE | Final validation RMSE | Failure |", "|---:|---|---:|---:|---:|---:|---|"]
                for t in c["trials"]:
                    failure = str(t["failure"] or "").replace("|", "/").replace("\n", " ")
                    text.append(f"| {t['seed']} | {t['state']} | {t['epochs']} | {t['bestEpoch']} | {t['bestRmse']} | {t['finalRmse']} | {failure} |")
            if run["independentTest"]:
                text += ["", "Independent test is only the harness winner's representative seed: `" + json.dumps(run["independentTest"], separators=(",", ":")) + "`."]
        for case in item.get("unreportedExpectedCases", []): text += ["", "Unreported case: `" + case["case"] + "` — " + case["status"] + "."]
    return "\n".join(text) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="LABEL=PATH.jsonl; repeat for separate hosts/processes")
    parser.add_argument("--output", required=True, help="Output basename; writes .json and .md")
    parser.add_argument("--search-seeds", default="42,123", help="Expected SEARCH seeds, not the five training seeds")
    args = parser.parse_args()
    search_seeds = [int(s) for s in args.search_seeds.split(",")]
    if not search_seeds or len(set(search_seeds)) != len(search_seeds) or any(s not in (42, 123) for s in search_seeds): parser.error("Search seeds must be a unique subset of 42,123")
    inputs = []
    for value in args.input:
        if "=" not in value: parser.error("--input requires LABEL=PATH")
        label, filename = value.split("=", 1)
        if not label or any(x["label"] == label for x in inputs): parser.error("Input labels must be nonempty and unique")
        inputs.append(analyze_file(label, filename, search_seeds))
    report = {"schemaVersion": 1, "protocol": {"trainingSeeds": list(SEEDS), "expectedSearchSeeds": search_seeds,
              "epochsPerSeed": EPOCHS, "targetValidationRmse": TARGET, "requiredSuccesses": 4, "maximumTrialsPerSearch": MAX_TRIALS,
              "trainingSamples": 176, "validationSamples": 44, "datasetFingerprint": DATASET_FINGERPRINT,
              "searchSeedAndTrialBudgetSource": "Analysis contract; current harness environment omits these two CLI options"},
              "globalMinimumEstablished": False, "inputs": inputs}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    Path(str(output) + ".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    Path(str(output) + ".md").write_text(markdown(report))
    print(json.dumps({"outputs": [str(output) + ".json", str(output) + ".md"], "inputs": [{"label": x["label"], "status": x["status"], "runs": len(x["runs"])} for x in inputs]}))
    return int(any(x["status"] == "invalid_input" or any(r["issues"] for r in x["runs"]) for x in inputs))


if __name__ == "__main__":
    sys.exit(main())
