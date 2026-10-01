import itertools
import json
from pathlib import Path
import tempfile
import unittest

import analyze_quality as q


def env():
    return {"type": "environment", "mode": "quality", "epochsPerSeed": q.EPOCHS, "seeds": list(q.SEEDS),
            "targetRmse": .01, "requiredSuccesses": 4, "batchSize": 1, "precision": "FP64", "sigmoid": "EXACT",
            "engine": "SMALL", "checkEvery": 25, "workers": [8], "backends": ["CPU"], "executions": ["OPTIMIZED"],
            "sourceRevision": "fixture", "datasetFingerprint": q.DATASET_FINGERPRINT, "trainingSamples": 176, "validationSamples": 44}


def candidate(hidden=(8,), completed=5, scores=(.001, .002, .003, .004, .1)):
    hidden = list(hidden)
    return {"hidden": hidden, "parameters": sum((a + 1) * b for a, b in zip([2] + hidden, hidden + [1])),
            "complete": completed == 5, "successes": min(completed, 4), "medianRmse": .003 if completed == 5 else None,
            "trials": [{"seed": s, "state": "COMPLETED" if i < completed else "CANCELLED",
                        "epochs": q.EPOCHS if i < completed else 500_000, "bestEpoch": 200_000,
                        "bestRmse": scores[i], "finalRmse": scores[i] + .0001,
                        "sampleUpdates": (q.EPOCHS if i < completed else 500_000) * 176, "failure": "",
                        "backend": "CPU", "route": "SESSION", "precision": "FP64", "sigmoid": "EXACT", "engine": "SMALL"}
                       for i, s in enumerate(q.SEEDS)]}


def record(candidates=None, full=False):
    candidates = candidates or [candidate()]
    trials = [t for c in candidates for t in c["trials"]]
    return {"type": "round", "case": "SMALL/CPU/OPTIMIZED/workers=8/searchSeed=42", "round": 0,
            "termination": "TRIAL_BUDGET" if full else "CANCELLED", "workComplete": full,
            "candidates": candidates, "completeTrials": sum(t["state"] == "COMPLETED" for t in trials),
            "committedEpochs": sum(t["epochs"] for t in trials), "sampleUpdates": sum(t["sampleUpdates"] for t in trials),
            "winner": [8], "targetMet": True, "independentTest": None}


class QualityAnalysisTest(unittest.TestCase):
    def test_completed_reliable_candidate_survives_partial_search_without_global_claim(self):
        result = q.analyze_round(record(), env(), [])
        self.assertEqual("partial_search", result["status"])
        self.assertEqual(1, len(result["reliableCompletedCandidates"]))
        self.assertFalse(result["globalMinimumEstablished"])
        self.assertEqual("within_completed_candidates_of_partial_search", result["reliabilityScope"])

    def test_four_complete_successes_and_one_partial_never_qualifies(self):
        result = q.analyze_round(record([candidate(completed=4)]), env(), [])
        self.assertEqual([], result["reliableCompletedCandidates"])
        self.assertEqual(4, result["candidates"][0]["successesAtFullBudget"])

    def test_duplicate_seed_and_short_budget_never_qualify(self):
        for mutation in (lambda c: c["trials"][4].update(seed=1), lambda c: c["trials"][4].update(epochs=999_999, sampleUpdates=999_999 * 176)):
            c = candidate(); mutation(c)
            self.assertFalse(q.analyze_round(record([c]), env(), [])["candidates"][0]["reliableCompletedCandidate"])

    def test_failures_and_nonfinite_scores_never_qualify(self):
        for mutation in (lambda t: t.update(state="FAILED", failure="numerical"), lambda t: t.update(bestRmse=None),
                         lambda t: t.update(precision="FP32"), lambda t: t.update(sampleUpdates=0)):
            c = candidate(); mutation(c["trials"][4])
            result = q.analyze_round(record([c]), env(), [])
            self.assertEqual([], result["reliableCompletedCandidates"])

    def test_three_successes_fail_even_all_five_complete(self):
        result = q.analyze_round(record([candidate(scores=(.001, .002, .003, .011, .1))]), env(), [])
        self.assertEqual([], result["reliableCompletedCandidates"])

    def test_full_search_requires_twelve_distinct_complete_candidates(self):
        shapes = [list(s) for n in range(1, 5) for s in itertools.product((4, 8, 16), repeat=n)][:12]
        result = q.analyze_round(record([candidate(s) for s in shapes], True), env(), [])
        self.assertEqual("full_work_complete", result["status"])
        self.assertEqual(60, result["fullBudgetCompletedTrials"])
        self.assertFalse(result["globalMinimumEstablished"])

    def test_reported_complete_summary_cannot_override_partial_data(self):
        raw = record([candidate(completed=4)], True)
        result = q.analyze_round(raw, env(), [])
        self.assertEqual("invalid_record", result["status"])
        self.assertFalse(result["fullWorkComplete"])

    def test_independent_result_is_only_harness_winner_representative(self):
        raw = record(); raw["independentTest"] = {"scope": "representative winning seed, not five-seed reliability", "samples": 218, "rmse": .03}
        result = q.analyze_round(raw, env(), [])
        self.assertEqual(123, result["independentTest"]["representativeSeed"])
        self.assertEqual(.03, result["independentTest"]["harnessResult"]["rmse"])

    def test_progress_only_and_not_started_search_seed_remain_unqualified(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "quality.jsonl"
            path.write_text(json.dumps(env()) + "\n" + json.dumps({"type": "progress", "case": record()["case"], "bestObservedRmse": .0001}) + "\n")
            result = q.analyze_file("local", path, [42, 123])
            self.assertEqual("progress_only_unqualified", result["status"])
            self.assertEqual([], result["runs"])
            self.assertEqual(["progress_only_unqualified", "not_observed_no_result"], [x["status"] for x in result["unreportedExpectedCases"]])

    def test_partial_last_line_not_used_and_malformed_middle_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "quality.jsonl"
            path.write_text(json.dumps(env()) + "\n" + json.dumps(record()) + '\n{"type":')
            result = q.analyze_file("local", path, [42, 123])
            self.assertTrue(result["trailingPartialLineIgnored"])
            self.assertEqual("partial_run", result["status"])
            path.write_text(json.dumps(env()) + '\nINVALID\n' + json.dumps(record()) + "\n")
            result = q.analyze_file("local", path, [42, 123])
            self.assertEqual("invalid_input", result["status"])
            self.assertEqual([], result["runs"][0]["reliableCompletedCandidates"])

    def test_wrong_target_and_fixed_mode_rejected(self):
        for change in ({"targetRmse": .05}, {"mode": "fixed"}, {"epochsPerSeed": 2000}):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "quality.jsonl"
                path.write_text(json.dumps(env() | change) + "\n" + json.dumps(record()) + "\n")
                result = q.analyze_file("local", path, [42])
                self.assertEqual("invalid_input", result["status"])
                self.assertEqual([], result["runs"][0]["reliableCompletedCandidates"])

    def test_wrong_spiral_dataset_and_sample_counts_rejected(self):
        for change in ({"datasetFingerprint": "different-dataset"}, {"trainingSamples": 175}, {"validationSamples": 45}):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "quality.jsonl"
                path.write_text(json.dumps(env() | change) + "\n" + json.dumps(record()) + "\n")
                result = q.analyze_file("local", path, [42])
                self.assertEqual("invalid_input", result["status"])
                self.assertEqual([], result["runs"][0]["reliableCompletedCandidates"])

    def test_off_boundary_best_epoch_and_invalid_score_ordering_never_qualify(self):
        for change in ({"bestEpoch": 200_001}, {"bestRmse": .009, "finalRmse": .008}):
            c = candidate(); c["trials"][0].update(change)
            result = q.analyze_round(record([c]), env(), [])
            self.assertEqual([], result["reliableCompletedCandidates"])
            self.assertTrue(result["candidates"][0]["trials"][0]["issues"])

    def test_superseded_input_excluded_even_if_file_missing(self):
        self.assertEqual("excluded_superseded", q.analyze_file("old", "quality-superseded/quality.jsonl", [42, 123])["status"])


if __name__ == "__main__":
    unittest.main()
