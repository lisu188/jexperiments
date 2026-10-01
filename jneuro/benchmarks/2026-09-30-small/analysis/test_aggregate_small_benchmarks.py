"""Small deterministic evidence fixtures; no training, native library or GPU required."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

import aggregate_small_benchmarks as aggregate


class AggregationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="jneuro-small-reports-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def json_report(self, fork, times=None, batches=(64,), identity=True):
        times = times or {"CPU": [100, 110, 120], "SMALL_256_FP64": [80, 90, 100], "NATIVE_FP64": [50, 60, 70]}
        cases = [{"name": f"batch-{batch}", "topology": [2, 8, 8, 8, 1], "samples": 128, "batchSize": batch} for batch in batches]
        config = {"epochs": 5, "repeats": len(next(iter(times.values()))), "backends": list(times), "mode": "CHUNK", "sigmoid": "EXACT", "retainedDevice": True}
        environment = {"javaVersion": "27", "os": "test OS", "cpu": "test CPU", "sourceRevision": "abc",
                       "benchmarkClassSha256": "b" * 64, "jvmArguments": ["-Xmx2g"], "seed": 1234,
                       "learningRate": 0.05, "momentum": 0.1, "beta": 1.0}
        if identity:
            environment.update(processId=100 + fork, jvmStartMillis=1000 + fork)
        rows = []
        for case in cases:
            for engine, values in times.items():
                for index, value in enumerate(values, 1):
                    total = value + fork
                    rows.append({"workload": case["name"], "backend": engine, "round": index,
                                 "epochs": 5, "samplesSeen": 640, "device": {"precision": "FP64", "backend": "CPU",
                                     "identity": "test CPU", "kernelVersion": "native-small-abi1/" + "a" * 64 if engine in aggregate.NATIVE else "cpu-v1",
                                     "engine": "SMALL", "simdBits": 256, "sigmoid": "EXACT"},
                                 "validation": {"maxScaledError": 0.01, "elements": 355}, "rmse": 0.3, "openNanos": 1,
                                 "trainingNanos": total - 2, "closeNanos": 1, "totalNanos": total})
        data = {"status": "passed", "failure": None, "configuration": config, "workloads": cases,
                "environment": environment, "rounds": rows}
        path = self.root / f"fork-{fork}.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def change(self, path, action):
        data = json.loads(path.read_text(encoding="utf-8"))
        action(data)
        path.write_text(json.dumps(data), encoding="utf-8")

    def cohort(self, fork, parallel=False):
        path = self.root / f"cohort-{fork}.csv"
        engines = {"REFERENCE_PARALLEL": 100, "SMALL_CPU": 80, "SMALL_SEQUENTIAL": 90,
                   "NATIVE_PARALLEL" if parallel else "NATIVE": 50}
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow("round,engine,models,precision,epochs,batch,workers,open_ns,training_ns,close_ns,total_ns,max_scaled_error,actual_cpu_workers,actual_precision".split(","))
            for repeat in range(3):
                for engine, time in engines.items():
                    total = time + fork + repeat
                    workers = 4 if engine in ("REFERENCE_PARALLEL", "SMALL_CPU", "NATIVE_PARALLEL") else 1
                    writer.writerow([repeat, engine, 32, "FP64", 64, 64, 4, 1, total - 2, 1, total, 0.01, workers, "FP64"])
        self.environment(path, fork, "samples=128, topology=2,8,8,8,1, repeats=3, engines=" + ",".join(engines))
        return path

    def environment(self, path, fork, options):
        Path(str(path) + ".environment.txt").write_text(
            f"java=27\nos=test OS\ncpu=test CPU\nrevision=abc\nprocessId={100 + fork}\njvmStartMillis={1000 + fork}\n"
            f"benchmarkClassSha256={'b' * 64}\nnativeLibrarySha256={'a' * 64}\njvmArguments=[-Xmx2g]\noptions={{{options}}}\n", encoding="utf-8")

    def quality(self, seeds=32, failure_variant="FP32/FAST", missing=None):
        path = self.root / "quality.csv"
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow("seed,precision,sigmoid,epochs,rmse,converged,nanos,target,check_every".split(","))
            for seed in range(seeds):
                for variant in aggregate.VARIANTS:
                    if (seed, variant) == missing:
                        continue
                    precision, sigmoid = variant.split("/")
                    success = variant != failure_variant or seed != 0
                    writer.writerow([seed, precision, sigmoid, 25 if success else 100,
                                     0.04 if success else 0.2, str(success).lower(),
                                     1000 + seed if success else 9999, 0.05, 25])
        self.environment(path, 20, "seeds=32, epochs=100, mode=quality")
        return path

    def test_three_verified_forks_compare_against_fastest_jvm_and_keep_raw_rounds(self):
        paths = [self.json_report(fork) for fork in range(3)]
        report = aggregate.aggregate(paths)
        self.assertTrue(report["nativeGatePassed"], report["issues"])
        work = next(iter(report["workloads"].values()))
        self.assertEqual("SMALL_256_FP64", work["baseline"])
        native = work["engines"]["NATIVE_FP64"]
        self.assertEqual(3, native["verifiedForks"])
        self.assertEqual(3, len(native["perFork"]))
        self.assertEqual(9, len(native["rawRounds"]))
        self.assertEqual(61, native["totalNanos"]["median"])
        self.assertEqual(72, native["totalNanos"]["p95"])
        self.assertIn("NATIVE", aggregate.markdown(report))

    def test_one_slow_workload_fails_without_global_cherry_picking(self):
        paths = [self.json_report(fork, batches=(16, 64)) for fork in range(3)]
        for path in paths:
            def slow(data):
                for row in data["rounds"]:
                    if row["workload"] == "batch-16" and row["backend"] == "NATIVE_FP64":
                        row["totalNanos"] += 100; row["trainingNanos"] += 100
            self.change(path, slow)
        report = aggregate.aggregate(paths)
        self.assertFalse(report["nativeGatePassed"])
        self.assertEqual(2, len(report["requiredWorkloads"]))
        self.assertEqual(1, len(report["candidateGates"]["NATIVE"]["failedWorkloads"]))

    def test_fast_median_does_not_hide_bad_p95(self):
        times = {"CPU": [100, 100, 100], "NATIVE_FP64": [50, 50, 150]}
        report = aggregate.aggregate([self.json_report(fork, times) for fork in range(3)])
        gate = next(iter(report["workloads"].values()))["candidateGates"]["NATIVE"]
        self.assertLess(gate["medianRatio"], 0.8)
        self.assertIn("Native p95 regressed", gate["reasons"])
        self.assertFalse(report["nativeGatePassed"])

    def test_missing_identity_or_same_jvm_or_only_two_files_cannot_pass(self):
        for identity, same_jvm in ((False, False), (True, True)):
            paths = [self.json_report(fork, identity=identity) for fork in range(3)]
            if same_jvm:
                for path in paths:
                    self.change(path, lambda data: data["environment"].update(processId=99, jvmStartMillis=123))
            self.assertFalse(aggregate.aggregate(paths)["nativeGatePassed"])
        self.assertFalse(aggregate.aggregate([self.json_report(fork) for fork in range(2)])["nativeGatePassed"])

    def test_missing_round_failed_status_parity_counts_and_negative_time_fail(self):
        changes = [lambda data: data["rounds"].pop(), lambda data: data.update(status="failed"),
                   lambda data: data["rounds"][0]["validation"].update(maxScaledError=1.1),
                   lambda data: data["rounds"][0].update(epochs=4),
                   lambda data: data["rounds"][0].update(trainingNanos=-1),
                   lambda data: data["rounds"][0].update(rmse=float("nan")),
                   lambda data: data["rounds"][0]["validation"].update(elements=1),
                   lambda data: data["rounds"].append(data["rounds"][0])]
        for change in changes:
            paths = [self.json_report(fork) for fork in range(3)]
            self.change(paths[0], change)
            report = aggregate.aggregate(paths)
            self.assertFalse(report["nativeGatePassed"])
            self.assertTrue(report["issues"])
            self.assertEqual(1, len(report["requiredWorkloads"]))

    def test_missing_workload_candidate_baseline_and_changed_provenance_fail(self):
        paths = [self.json_report(fork) for fork in range(3)]
        self.assertFalse(aggregate.aggregate(paths, required_workloads=["missing-case"])["nativeGatePassed"])
        self.change(paths[0], lambda data: data["environment"].update(cpu="different CPU"))
        self.assertFalse(aggregate.aggregate(paths)["nativeGatePassed"])
        for times in ({"CPU": [100]}, {"NATIVE_FP64": [50]}):
            report = aggregate.aggregate([self.json_report(fork, times) for fork in range(3)])
            self.assertFalse(report["nativeGatePassed"])

    def test_duplicate_input_does_not_create_a_fork(self):
        path = self.json_report(0)
        report = aggregate.aggregate([path, path, path])
        self.assertFalse(report["nativeGatePassed"])
        self.assertTrue(any("Duplicate input path" in issue for issue in report["issues"]))

    def test_mixed_native_artifact_device_simd_and_jvm_hyperparameters_are_rejected(self):
        changes = [lambda data: data["environment"].update(seed=7),
                   lambda data: data["environment"].update(momentum=0.0),
                   lambda data: data["environment"].update(jvmArguments=["-Xmx4g"]),
                   lambda data: data["environment"].update(benchmarkClassSha256="c" * 64)]
        for field, value in (("kernelVersion", "native-small-abi1/" + "c" * 64),
                             ("simdBits", 0), ("identity", "another CPU")):
            def change_device(data, name=field, replacement=value):
                for row in data["rounds"]:
                    if row["backend"] == "NATIVE_FP64":
                        row["device"][name] = replacement
            changes.append(change_device)
        for change in changes:
            paths = [self.json_report(fork) for fork in range(3)]
            self.change(paths[0], change)
            report = aggregate.aggregate(paths)
            self.assertFalse(report["nativeGatePassed"])
            self.assertTrue(any("incompatible" in issue for issue in report["issues"]))

    def test_cohort_csv_reads_sidecar_shape_and_native_variants(self):
        paths = [self.cohort(fork, parallel=True) for fork in range(3)]
        report = aggregate.aggregate(csvpaths=paths)
        self.assertTrue(report["nativeGatePassed"], report["issues"])
        key = report["requiredWorkloads"][0]
        self.assertIn("cohort:2x8x8x8x1:s128", key)
        self.assertEqual("SMALL_CPU", report["workloads"][key]["baseline"])
        self.assertTrue(report["candidateGates"]["NATIVE_PARALLEL"]["passed"])
        Path(str(paths[0]) + ".environment.txt").unlink()
        self.assertFalse(aggregate.aggregate(csvpaths=paths)["nativeGatePassed"])

    def test_csv_native_hash_is_required_and_must_match_across_forks(self):
        paths = [self.cohort(fork) for fork in range(3)]
        environment = Path(str(paths[0]) + ".environment.txt")
        text = environment.read_text()
        environment.write_text(text.replace("a" * 64, "c" * 64))
        self.assertFalse(aggregate.aggregate(csvpaths=paths)["nativeGatePassed"])
        for path in paths:
            sidecar = Path(str(path) + ".environment.txt")
            sidecar.write_text(sidecar.read_text().replace("a" * 64, "none").replace("c" * 64, "none"))
        report = aggregate.aggregate(csvpaths=paths)
        self.assertFalse(report["nativeGatePassed"])
        gate = next(iter(report["workloads"].values()))["candidateGates"]["NATIVE"]
        self.assertIn("Native library SHA-256 is missing or invalid", gate["reasons"])

    def test_actual_worker_counts_are_reported_without_partitioning_the_workload(self):
        report = aggregate.aggregate(csvpaths=[self.cohort(fork) for fork in range(3)])
        self.assertEqual(1, len(report["workloads"]))
        engines = next(iter(report["workloads"].values()))["engines"]
        self.assertEqual([4], engines["SMALL_CPU"]["actualCpuWorkers"])
        self.assertEqual([1], engines["NATIVE"]["actualCpuWorkers"])
        self.assertTrue(report["nativeGatePassed"])

    def test_actual_reference_precision_is_fp64_even_with_fp32_request(self):
        path = self.cohort(0)
        lines = path.read_text().splitlines()
        modified = [lines[0]]
        for line in lines[1:]:
            row = line.split(",")
            row[3] = "FP32"
            row[-1] = "FP64" if row[1].startswith("REFERENCE") else "FP32"
            modified.append(",".join(row))
        path.write_text("\n".join(modified) + "\n")
        sidecar = Path(str(path) + ".environment.txt")
        sidecar.write_text(sidecar.read_text().replace("options={", "options={precision=FP32, "))
        report = aggregate.aggregate(csvpaths=[path])
        self.assertFalse(report["issues"], report["issues"])
        engines = next(iter(report["workloads"].values()))["engines"]
        self.assertEqual(["FP64"], engines["REFERENCE_PARALLEL"]["precision"])
        self.assertEqual(["FP32"], engines["REFERENCE_PARALLEL"]["requestedPrecision"])
        self.assertEqual(["FP32"], engines["SMALL_CPU"]["precision"])
        self.assertEqual([], report["requiredWorkloads"])

    def test_incomplete_cohort_rows_fail(self):
        paths = [self.cohort(fork) for fork in range(3)]
        lines = paths[0].read_text(encoding="utf-8").splitlines()
        paths[0].write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
        report = aggregate.aggregate(csvpaths=paths)
        self.assertFalse(report["nativeGatePassed"])
        self.assertTrue(any("incomplete cohort" in issue for issue in report["issues"]))

    def test_quality_keeps_censored_runs_in_denominator_and_times_are_conditional(self):
        report = aggregate.aggregate(csvpaths=[self.quality()])
        self.assertTrue(report["qualityEvidenceComplete"], report["issues"])
        self.assertFalse(report["nativeGatePassed"])
        quality = report["quality"][0]
        self.assertEqual(32, quality["distinctPairedSeeds"])
        fast = quality["variants"]["FP32/FAST"]
        self.assertEqual(31, fast["converged"])
        self.assertEqual(1, fast["censoredFailures"])
        self.assertEqual(32, fast["expectedSeedRuns"])
        self.assertEqual(31 / 32, fast["convergenceFraction"])
        self.assertEqual(1030, fast["conditionalTimeToTargetNanos"]["p95"])
        self.assertEqual(9999, fast["censoredElapsedNanos"]["median"])
        self.assertIn("31 / 32", aggregate.markdown(report))

    def test_quality_missing_variant_or_entire_seed_is_incomplete(self):
        report = aggregate.aggregate(csvpaths=[self.quality(missing=(3, "FP64/FAST"))])
        self.assertFalse(report["qualityEvidenceComplete"])
        self.assertEqual(1, report["quality"][0]["variants"]["FP64/FAST"]["missing"])
        report = aggregate.aggregate(csvpaths=[self.quality(seeds=31)])
        self.assertFalse(report["qualityEvidenceComplete"])
        self.assertEqual(32, report["quality"][0]["variants"]["FP64/EXACT"]["expectedSeedRuns"])

    def test_nonfinite_and_unrecognized_csv_cannot_pass(self):
        path = self.json_report(0)
        self.change(path, lambda data: data["rounds"][0]["validation"].update(maxScaledError=float("nan")))
        self.assertTrue(aggregate.aggregate([path])["issues"])
        csvpath = self.root / "bad.csv"
        csvpath.write_text("wrong,header\n1,2\n", encoding="utf-8")
        self.assertTrue(aggregate.aggregate(csvpaths=[csvpath])["issues"])
        self.assertFalse(aggregate.aggregate()["nativeGatePassed"])

    def test_cli_writes_compact_reports_and_returns_gate_status(self):
        paths = [self.json_report(fork) for fork in range(3)]
        output = self.root / "summary"
        self.assertEqual(0, aggregate.main(["--jsonpaths", *map(str, paths), "--output", str(output)]))
        self.assertTrue(json.loads(output.with_suffix(".json").read_text())["nativeGatePassed"])
        self.assertTrue(output.with_suffix(".md").exists())
        self.assertEqual(1, aggregate.main(["--output", str(output)]))


if __name__ == "__main__":
    unittest.main()
