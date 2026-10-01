import copy
import json
from pathlib import Path
import tempfile
import unittest
import summarize_search_benchmark as summary


class SearchSummaryTest(unittest.TestCase):
    def records(self, fork):
        environment = dict(type="environment", mode="fixed", datasetFingerprint="spiral", epochsPerSeed=2000,
                           trainingSamples=176, validationSamples=44, manifest=[[8]], seeds=[42],
                           batchSize=1, precision="FP64", sigmoid="EXACT", checkEvery=25,
                           warmups=1, repeats=3, pid=fork + 1, started=str(fork), sourceRevision="abc123",
                           java="27", vm="OpenJDK", os="Linux", osVersion="6.8", arch="amd64",
                           cpuModel="test CPU", heapMaxBytes=1000000, jvmArguments=["--add-modules=jdk.incubator.vector"])
        trial = dict(seed=42, state="COMPLETED", epochs=2000, sampleUpdates=352000, bestEpoch=25,
                     parametersAtBest=[1.] * 33, bestRmse=.4, finalRmse=.5, failure="",
                     deviceIdentity="jvm-cpu", backend="CPU", precision="FP64", sigmoid="EXACT", engine="SMALL",
                     kernel="small-cpu-v1", route="SESSION", simdBits=256)
        common = dict(type="round", workComplete=True, termination="TRIAL_BUDGET", failedTrials=0,
                      partialCandidates=0, evaluatedCandidates=1, completeTrials=1, committedEpochs=2000,
                      sampleUpdates=352000, candidates=[dict(hidden=[8], parameters=33, complete=True, trials=[trial])])
        records = [environment]
        for number in range(-1, 3):
            for execution, duration in (("REFERENCE", 2000000), ("OPTIMIZED", 1000000)):
                row = dict(copy.deepcopy(common), case=f"SMALL/CPU/{execution}/workers=4/searchSeed=42",
                           round=number, totalNanos=duration)
                # Hardware is the same; different execution kernels and names are legitimate.
                if execution == "REFERENCE":
                    row["candidates"][0]["trials"][0].update(kernel="small-cpu-cohort-v1", route="COHORT", device="CPU model lanes")
                else:
                    row["candidates"][0]["trials"][0]["device"] = "CPU"
                records.append(row)
        return records

    def write(self, directory, fork, records):
        path = Path(directory) / f"fork-{fork}.jsonl"
        path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
        return path

    def run_reports(self, change=None, count=3):
        with tempfile.TemporaryDirectory() as directory:
            reports = [self.records(fork) for fork in range(count)]
            if change:
                change(reports)
            return summary.summarize([self.write(directory, index, records) for index, records in enumerate(reports)])

    def assert_not_qualified(self, result):
        self.assertFalse(any(row["qualifies"] for row in result["comparisons"]))

    def test_three_complete_paired_forks_and_nine_rounds_qualify(self):
        result = self.run_reports()["comparisons"][0]
        self.assertTrue(result["qualifies"])
        self.assertEqual(2, result["speedup"])
        self.assertEqual(.5, result["medianReductionFraction"])
        self.assertEqual(dict(medianMs=2., p95Ms=2., rounds=9, forks=3), result["reference"])
        self.assert_not_qualified(self.run_reports(count=1))
        # Even three actual JVMs do not qualify with only six measured rounds.
        def fewer_rounds(reports):
            for records in reports:
                records[0]["repeats"] = 2
                records[:] = [record for record in records if record.get("round") != 2]
        self.assert_not_qualified(self.run_reports(fewer_rounds))

    def test_duplicate_paths_copies_and_rounds_cannot_create_forks_or_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            records = self.records(0)
            path = self.write(directory, 0, records)
            clone = self.write(directory, 1, records)
            duplicated = self.write(directory, 2, records + records[1:])
            result = summary.summarize([path, path, clone, duplicated])["comparisons"][0]
            self.assertEqual(3, result["reference"]["rounds"])
            self.assertEqual(1, result["reference"]["forks"])
            self.assertFalse(result["qualifies"])
            bad = copy.deepcopy(records)
            bad[-1]["totalNanos"] += 1
            with self.assertRaisesRegex(ValueError, "Conflicting duplicate"):
                summary.summarize([path, self.write(directory, 3, bad)])

    def test_disjoint_forks_and_missing_warmup_or_round_are_excluded(self):
        def disjoint(reports):
            for index, records in enumerate(reports):
                execution = "REFERENCE" if index < 3 else "OPTIMIZED"
                records[:] = [record for record in records if record["type"] != "round" or f"/{execution}/" in record["case"]]
        self.assert_not_qualified(self.run_reports(disjoint, 6))
        for removed in (-1, 2):
            def missing(reports):
                reports[0][:] = [record for record in reports[0] if not
                                (record.get("round") == removed and "/OPTIMIZED/" in record.get("case", ""))]
            result = self.run_reports(missing)
            self.assert_not_qualified(result)
            self.assertEqual(2, result["comparisons"][0]["reference"]["forks"])

    def test_incomplete_work_and_failed_processes_cannot_supply_surviving_rounds(self):
        mutations = [lambda record: record.update(workComplete=False),
                     lambda record: record.update(termination="MEMORY_LIMIT"),
                     lambda record: record.update(failedTrials=1),
                     lambda record: record.update(sampleUpdates=1),
                     lambda record: record.update(candidates=[]),
                     lambda record: record["candidates"][0].update(complete=False),
                     lambda record: record["candidates"][0]["trials"][0].update(seed=123),
                     lambda record: record["candidates"][0]["trials"][0].update(epochs=1999),
                     lambda record: record["candidates"][0]["trials"][0].update(state="FAILED"),
                     lambda record: record["candidates"][0]["trials"].append(copy.deepcopy(record["candidates"][0]["trials"][0]))]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                result = self.run_reports(lambda reports: mutate(reports[0][-1]))
                self.assert_not_qualified(result)
                self.assertTrue(result["incompleteOrFailed"])
        for kind in ("failure", "budget"):
            self.assert_not_qualified(self.run_reports(lambda reports: reports[0].append(dict(type=kind, error="failure"))))

    def test_nonfinite_or_missing_scores_parameters_and_durations_never_pass(self):
        for value in (float("nan"), float("inf"), float("-inf"), None, True):
            for field in ("bestRmse", "finalRmse", "parametersAtBest", "totalNanos"):
                def invalid(reports):
                    row = reports[0][-1]
                    trial = row["candidates"][0]["trials"][0]
                    if field == "totalNanos":
                        row[field] = value
                    elif field == "parametersAtBest":
                        trial[field][0] = value
                    else:
                        trial[field] = value
                with self.subTest(value=value, field=field):
                    self.assert_not_qualified(self.run_reports(invalid))
        for duration in (0, -1):
            self.assert_not_qualified(self.run_reports(lambda reports: reports[0][-1].update(totalNanos=duration)))

    def test_all_reference_and_optimized_rows_must_pass_parity(self):
        for execution in ("REFERENCE", "OPTIMIZED"):
            def changed(reports):
                row = next(record for record in reports[2][1:] if record["round"] == 2 and f"/{execution}/" in record["case"])
                row["candidates"][0]["trials"][0]["parametersAtBest"][0] += 1e-5
            result = self.run_reports(changed)["comparisons"][0]
            self.assertTrue(result["threeForkEvidence"])
            self.assertFalse(result["qualifies"])
            self.assertGreater(result["maximumScaledParameterOrScoreError"], 1)
        def changed_epoch(reports):
            reports[2][-1]["candidates"][0]["trials"][0]["bestEpoch"] = 50
        self.assertFalse(self.run_reports(changed_epoch)["comparisons"][0]["matchingBestEpochs"])
        def within_tolerance(reports):
            reports[2][-1]["candidates"][0]["trials"][0]["parametersAtBest"][0] += 1e-9
        self.assertTrue(self.run_reports(within_tolerance)["comparisons"][0]["qualifies"])

    def test_extreme_finite_parity_failure_still_serializes_as_valid_json(self):
        def extreme(reports):
            for records in reports:
                for row in records[1:]:
                    row["candidates"][0]["trials"][0]["parametersAtBest"][0] = 1e308
            reports[2][-1]["candidates"][0]["trials"][0]["parametersAtBest"][0] = -1e308
        result = self.run_reports(extreme)
        self.assert_not_qualified(result)
        json.dumps(result, allow_nan=False)

    def test_mixed_provenance_hardware_and_protocols_cannot_be_pooled(self):
        changes = dict(sourceRevision="def456", java="28", cpuModel="other CPU", osVersion="7.0",
                       jvmArguments=["-Xmx32m"], heapMaxBytes=2000000, datasetFingerprint="other-data")
        for field, value in changes.items():
            with self.subTest(field=field):
                result = self.run_reports(lambda reports: reports[0][0].update({field: value}))
                self.assert_not_qualified(result)
                self.assertEqual(2, len(result["comparisons"]))
        def missing_provenance(reports):
            for records in reports:
                records[0]["sourceRevision"] = "unspecified"
        self.assert_not_qualified(self.run_reports(missing_provenance))
        for field, value in (("deviceIdentity", "other-device"), ("precision", "FP32"), ("kernel", "different-kernel")):
            self.assert_not_qualified(self.run_reports(lambda reports: reports[0][-1]["candidates"][0]["trials"][0].update({field: value})))

    def test_reference_routes_may_change_with_available_slots_but_remain_auditable(self):
        def rerouted(reports):
            for index, records in enumerate(reports):
                for row in records[1:]:
                    if "/REFERENCE/" in row["case"] and (row["round"] + index) % 2 == 0:
                        # Session and model-lane cohort kernels can legitimately use different widths.
                        row["candidates"][0]["trials"][0].update(route="SESSION", kernel="small-cpu-v1", simdBits=128)
        result = self.run_reports(rerouted)["comparisons"][0]
        self.assertTrue(result["qualifies"])
        self.assertTrue(result["validExecutionRoutes"])
        routes = result["observedExecutionRoutes"]["REFERENCE"]
        self.assertEqual({"SESSION", "COHORT"}, {row["route"] for row in routes})
        self.assertEqual(9, sum(row["trials"] for row in routes))
        for route, kernel, bits in (("COHORT", "small-cpu-v1", 256), ("SESSION", "unknown", 256),
                                    ("SESSION", "small-cpu-v1", 512)):
            result = self.run_reports(lambda reports: reports[0][-1]["candidates"][0]["trials"][0].update(
                route=route, kernel=kernel, simdBits=bits))["comparisons"][0]
            self.assertFalse(result["qualifies"])
            self.assertFalse(result["validExecutionRoutes"])

    def test_cuda_and_general_reference_route_families(self):
        def cuda(reports):
            for records in reports:
                for row in records[1:]:
                    row["case"] = row["case"].replace("/CPU/", "/CUDA/")
                    reference = "/REFERENCE/" in row["case"]
                    route = ("SESSION" if row["round"] % 2 else "COHORT") if reference else "CUDA_QUEUE"
                    prefix = "small-v2" if reference else "small-search-v3"
                    row["candidates"][0]["trials"][0].update(backend="CUDA", deviceIdentity="gpu-0", simdBits=0,
                        route=route, kernel=prefix + "/packed-fp64/driver-sha")
        self.assertTrue(self.run_reports(cuda)["comparisons"][0]["qualifies"])
        def general(reports):
            for records in reports:
                for row in records[1:]:
                    row["case"] = row["case"].replace("SMALL/", "REFERENCE/")
                    row["candidates"][0]["trials"][0].update(engine="REFERENCE", route="SESSION", kernel="cpu-v1", simdBits=0)
        self.assertTrue(self.run_reports(general)["comparisons"][0]["qualifies"])

    def test_literal_ten_percent_median_reduction_and_no_p95_regression(self):
        for duration, qualifies in ((1800000, True), (1810000, False)):
            def timing(reports):
                for records in reports:
                    for row in records[1:]:
                        if "/OPTIMIZED/" in row["case"]:
                            row["totalNanos"] = duration
            self.assertEqual(qualifies, self.run_reports(timing)["comparisons"][0]["qualifies"])
        def tail_regression(reports):
            reports[0][-1]["totalNanos"] = 3000000
        result = self.run_reports(tail_regression)["comparisons"][0]
        self.assertEqual(2, result["speedup"])
        self.assertTrue(result["p95Regression"])
        self.assertFalse(result["qualifies"])

    def test_multiple_environment_blocks_or_missing_process_identity_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = self.records(0)
            with self.assertRaisesRegex(ValueError, "one initial environment"):
                summary.summarize([self.write(directory, 0, rows + rows)])
            rows[0].pop("pid")
            with self.assertRaisesRegex(ValueError, "process identity"):
                summary.summarize([self.write(directory, 0, rows)])
        def quality(reports):
            for rows in reports:
                rows[0]["mode"] = "quality"
        self.assertEqual([], self.run_reports(quality)["comparisons"])


if __name__ == "__main__":
    unittest.main()
