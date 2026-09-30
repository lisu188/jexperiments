package com.lis.neuro

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import java.nio.file.Files
import java.nio.file.Path

class NeuroCudaBenchmarkHarnessTest {
    @TempDir lateinit var temporary: Path

    @Test fun defaultsAreBoundedAndArgumentsAreExplicitAndValidated() {
        val defaults = NeuroCudaBenchmarkConfig.parse(emptyArray())
        assertEquals("smoke", defaults.profile)
        assertEquals(listOf(16, 64), defaults.batches)
        assertEquals(NeuroCudaBenchmarkBackend.entries, defaults.backends)
        assertEquals(listOf(128, 128), NeuroCudaBenchmarkHarness.workloads(defaults).map { it.samples })
        val parsed = NeuroCudaBenchmarkConfig.parse(arrayOf("--profile", "matrix", "--epochs", "3", "--warmups", "0",
            "--repeats", "2", "--batches", "8,32", "--backends", "cpu,cuda", "--output", temporary.resolve("result").toString()))
        assertEquals(3, parsed.epochs)
        assertEquals(0, parsed.warmups)
        assertEquals(2, parsed.repeats)
        assertEquals(listOf(NeuroCudaBenchmarkBackend.CPU, NeuroCudaBenchmarkBackend.CUDA), parsed.backends)
        assertEquals(4, NeuroCudaBenchmarkHarness.workloads(parsed).size)
        assertEquals("smoke", NeuroCudaBenchmarkConfig.parse(arrayOf("--profile", "matrix", "--smoke")).profile)
        for (args in listOf(arrayOf("--epochs"), arrayOf("--unknown", "value"), arrayOf("--epochs", "bad"),
            arrayOf("--epochs", "0"), arrayOf("--warmups", "-1"), arrayOf("--repeats", "0"),
            arrayOf("--profile", "huge"), arrayOf("--batches", "0"), arrayOf("--batches", "16,16"),
            arrayOf("--backends", "AUTO"), arrayOf("--backends", "CPU,CPU"),
            arrayOf("--topology", "8"), arrayOf("--topology", "8,0,8"), arrayOf("--topology", "8,2049,8"),
            arrayOf("--topology", List(17) { "8" }.joinToString(",")), arrayOf("--topology", "2048,2048"),
            arrayOf("--samples", "0"), arrayOf("--samples", "8193"))) {
            assertThrows(IllegalArgumentException::class.java, { NeuroCudaBenchmarkConfig.parse(args) }, args.joinToString(" "))
        }
        assertThrows(IllegalArgumentException::class.java) { defaults.copy(backends = emptyList()) }
        assertThrows(IllegalArgumentException::class.java) { defaults.copy(batches = emptyList()) }
    }

    @Test fun customTopologyReplacesPresetsAndSamplesAreExplicitInModelsAndReports() {
        val config = NeuroCudaBenchmarkConfig.parse(arrayOf("--topology", "2,8,8,8,1", "--samples", "16", "--epochs", "1",
            "--warmups", "0", "--repeats", "1", "--backends", "CPU", "--batches", "4"))
        val workload = NeuroCudaBenchmarkHarness.workloads(config).single()
        assertEquals("custom-batch-4", workload.name)
        assertEquals(listOf(2, 8, 8, 8, 1), workload.topology)
        assertEquals(16, workload.samples)
        val model = NeuroCudaBenchmarkHarness.prepared(workload)
        assertArrayEquals(intArrayOf(2, 8, 8, 8, 1), model.topology())
        assertEquals(16, model.trainingSampleCount())
        assertEquals(177, model.parameterCount())
        val report = NeuroCudaBenchmarkHarness.run(config, environment = emptyMap())
        assertTrue(NeuroCudaBenchmarkReports.console(report).contains("topology=2x8x8x8x1, samples=16"))
        val json = NeuroCudaBenchmarkReports.json(report)
        assertTrue(json.contains("\"topology\":[2,8,8,8,1]"))
        assertTrue(json.contains("\"samples\":16"))
        assertEquals(16L, report.rounds.single().samplesSeen)
        assertEquals(128, config.copy(samples = null).sampleCount)
        assertEquals(1024, config.copy(profile = "matrix", samples = null).sampleCount)
        assertEquals(1, NeuroCudaBenchmarkHarness.workloads(config.copy(profile = "matrix")).size)
        assertEquals(listOf(16, 16), NeuroCudaBenchmarkHarness.workloads(config.copy(profile = "matrix", topology = null)).map { it.samples })
    }

    @Test fun commandLineHelpDescribesCustomShapeAndReturnsWithoutTraining() {
        val previous = System.out
        val output = java.io.ByteArrayOutputStream()
        try {
            System.setOut(java.io.PrintStream(output, true, Charsets.UTF_8))
            NeuroCudaTrainingBenchmark.main(arrayOf("--help"))
            NeuroCudaTrainingBenchmark.main(arrayOf("-h"))
        } finally { System.setOut(previous) }
        val text = output.toString(Charsets.UTF_8)
        assertTrue(text.contains("--topology 2,8,8,8,1"))
        assertTrue(text.contains("--samples N"))
        assertTrue(text.contains("without CPU fallback"))
        assertFalse(text.contains("Starting CPU/GPU benchmark"))
    }

    @Test fun warmupsAreExcludedFreshModelsRotateOrderAndTimingsHaveComparableScopes() {
        val config = NeuroCudaBenchmarkConfig(epochs = 1, warmups = 1, repeats = 3, batches = listOf(16))
        val requests = ArrayList<NeuroCudaBenchmarkBackend>()
        val modelIds = HashSet<String>()
        var closeCount = 0
        var nanos = 0L
        val progress = ArrayList<Triple<String, Int, Int>>()
        val report = NeuroCudaBenchmarkHarness.run(config, clock = { nanos += 10; nanos }, sessionFactory = { model, engine, batch ->
            assertEquals(0L, model.statistics().epochsTrained)
            assertEquals(128, model.trainingSampleCount())
            assertTrue(modelIds.add(model.logId), "Every warmup and measured round requires a fresh model")
            requests += engine
            val cpu = model.newTrainingSession(TrainingBackend.CPU, batchSize = batch)
            object : NeuroTrainingSession by cpu {
                override val info = info(engine)
                override fun close() { closeCount++; cpu.close() }
            }
        }, environment = mapOf("loggingLevel" to "WARNING"), onWorkloadCompleted = { workload, completed, total ->
            assertEquals(16, closeCount, "Progress must follow all warmup/measured session cleanup")
            progress += Triple(workload.name, completed, total)
            nanos += 1_000_000 // Callback work must not enter any measured phase.
        })
        assertEquals(listOf(Triple("smoke-batch-16", 1, 1)), progress)
        val engines = config.backends
        assertEquals(engines + engines + engines.drop(1) + engines.take(1) + engines.drop(2) + engines.take(2), requests)
        assertEquals(16, closeCount)
        assertEquals(12, report.rounds.size)
        assertEquals(listOf(1, 2, 3), report.rounds.map { it.round }.distinct())
        assertTrue(report.rounds.all { it.openNanos == 10L && it.trainingNanos == 10L && it.closeNanos == 10L && it.totalNanos == 30L })
        assertTrue(report.rounds.all { it.validation.maximumAbsoluteError == 0.0 && it.epochs == 1L && it.samplesSeen == 128L })
        assertEquals(4, report.summaries.size)
        assertTrue(report.summaries.all { it.training.median == 10.0 && it.trainingSpeedup == 1.0 && it.totalSpeedup == 1.0 })
        assertNull(report.failure)
        assertEquals(0L, NeuroCudaBenchmarkHarness.prepared(NeuroCudaBenchmarkHarness.workloads(config).first()).statistics().epochsTrained)
    }

    @Test fun distributionsIncludeEvenMediansAndNearestRankP95() {
        val result = NeuroCudaBenchmarkHarness.distribution((1L..100L).toList().reversed())
        assertEquals(1.0, result.minimum)
        assertEquals(50.5, result.median)
        assertEquals(95.0, result.p95)
        assertEquals(5.0, NeuroCudaBenchmarkHarness.distribution(listOf(9, 1, 5)).median)
        assertEquals(0.0, NeuroCudaBenchmarkHarness.distribution(listOf(0)).p95)
        assertThrows(IllegalArgumentException::class.java) { NeuroCudaBenchmarkHarness.distribution(emptyList()) }
        assertThrows(IllegalArgumentException::class.java) { NeuroCudaBenchmarkHarness.distribution(listOf(-1)) }
    }

    @Test fun fullStateValidationIncludesMomentumPrecisionAndRmse() {
        val model = NeuroCudaBenchmarkHarness.prepared(NeuroCudaBenchmarkWorkload("test", listOf(2, 2, 1), 2, 1))
        val expected = model.exportTrainingState()
        val actual = model.exportTrainingState()
        actual.weightVelocity[0][0] += 0.01
        val mismatch = NeuroCudaBenchmarkHarness.validate(expected, actual, 0.1, 0.1, Neuro.TrainingPrecision.FP64)
        assertEquals(0.01, mismatch.maximumAbsoluteError)
        assertTrue(mismatch.maximumScaledError > 1.0)
        assertEquals(model.parameterCount() * 2 + 1, mismatch.elements)
        actual.weightVelocity[0][0] = 1e-5
        assertTrue(NeuroCudaBenchmarkHarness.validate(expected, actual, 0.1, 0.1, Neuro.TrainingPrecision.FP32).maximumScaledError < 1.0)
        assertTrue(NeuroCudaBenchmarkHarness.validate(expected, expected, 0.1, 0.2, Neuro.TrainingPrecision.FP64).maximumScaledError > 1.0)
        assertThrows(IllegalStateException::class.java) {
            NeuroCudaBenchmarkHarness.validate(expected, actual, 0.1, Double.NaN, Neuro.TrainingPrecision.FP32)
        }
        assertThrows(IllegalStateException::class.java) {
            NeuroCudaBenchmarkHarness.validate(expected, actual.copy(topology = intArrayOf(2, 3, 1)), 0.1, 0.1, Neuro.TrainingPrecision.FP64)
        }
        assertThrows(IllegalStateException::class.java) {
            NeuroCudaBenchmarkHarness.validate(expected, actual.copy(weights = emptyArray()), 0.1, 0.1, Neuro.TrainingPrecision.FP64)
        }
        assertThrows(IllegalStateException::class.java) {
            NeuroCudaBenchmarkHarness.validate(expected, actual.copy(weights = arrayOf(DoubleArray(1), actual.weights[1])), 0.1, 0.1, Neuro.TrainingPrecision.FP64)
        }
    }

    @Test fun selectedGpuFailureIsExplicitAndKeepsCompletedEvidenceWithoutFallback() {
        val config = NeuroCudaBenchmarkConfig(epochs = 1, warmups = 0, repeats = 1, batches = listOf(16),
            backends = listOf(NeuroCudaBenchmarkBackend.CPU, NeuroCudaBenchmarkBackend.CUDA))
        val unavailable = UnsatisfiedLinkError("CUDA is unavailable")
        val failure = assertThrows(NeuroCudaBenchmarkFailure::class.java) {
            NeuroCudaBenchmarkHarness.run(config, sessionFactory = { model, engine, batch ->
                if (engine == NeuroCudaBenchmarkBackend.CUDA) throw unavailable
                model.newTrainingSession(batchSize = batch)
            }, environment = emptyMap())
        }
        assertSame(unavailable, failure.cause)
        assertEquals("measured", failure.report.failure!!["stage"])
        assertEquals("CUDA", failure.report.failure["backend"])
        assertEquals(1, failure.report.rounds.size)
        assertEquals(NeuroCudaBenchmarkBackend.CPU, failure.report.rounds.single().backend)
        assertTrue(NeuroCudaBenchmarkReports.json(failure.report).contains("\"status\":\"failed\""))
    }

    @Test fun wrongResolvedDeviceAndFailedTrainingAreRejectedAndCleanupNeverMasksFailure() {
        val config = NeuroCudaBenchmarkConfig(epochs = 1, warmups = 1, repeats = 1, batches = listOf(16),
            backends = listOf(NeuroCudaBenchmarkBackend.CUDA))
        var closed = 0
        val wrong = assertThrows(NeuroCudaBenchmarkFailure::class.java) {
            NeuroCudaBenchmarkHarness.run(config, sessionFactory = { model, _, batch ->
                val cpu = model.newTrainingSession(batchSize = batch)
                object : NeuroTrainingSession by cpu { override fun close() { closed++; cpu.close() } }
            }, environment = emptyMap())
        }
        assertTrue(wrong.cause!!.message!!.contains("resolved CPU"))
        assertEquals(1, closed)
        val trainFailure = IllegalStateException("native training failed")
        val closeFailure = IllegalStateException("native cleanup failed")
        val failed = assertThrows(NeuroCudaBenchmarkFailure::class.java) {
            NeuroCudaBenchmarkHarness.run(config, sessionFactory = { model, engine, batch ->
                val cpu = model.newTrainingSession(batchSize = batch)
                object : NeuroTrainingSession by cpu {
                    override val info = info(engine)
                    override fun trainMiniBatch(epochs: Int, batchSize: Int, parallelism: Int) { throw trainFailure }
                    override fun close() { cpu.close(); throw closeFailure }
                }
            }, environment = emptyMap())
        }
        assertSame(trainFailure, failed.cause)
        assertArrayEquals(arrayOf(closeFailure), trainFailure.suppressed)
        assertTrue(failed.report.rounds.isEmpty())
    }

    @Test fun reportsEscapeMetadataAndPersistRoundAndSummaryValues() {
        val config = NeuroCudaBenchmarkConfig(epochs = 1, warmups = 0, repeats = 1, batches = listOf(16),
            backends = listOf(NeuroCudaBenchmarkBackend.CPU), output = temporary.resolve("nested/benchmark"))
        val report = NeuroCudaBenchmarkHarness.run(config, environment = mapOf("sourceRevision" to "abc", "special" to "a\"b\\c\n\r\t\u0001"))
        NeuroCudaBenchmarkReports.write(report, config.output)
        val json = Files.readString(Path.of("${config.output}.json"))
        val csv = Files.readString(Path.of("${config.output}.csv"))
        assertTrue(json.contains("\"status\":\"passed\""))
        assertTrue(json.contains("\"sourceRevision\":\"abc\""))
        assertTrue(json.contains("a\\\"b\\\\c\\n\\r\\t\\u0001"))
        assertTrue(json.contains("\"absoluteTolerance\":1.0E-10"))
        assertTrue(json.contains("\"endToEndSpeedup\":1.0"))
        assertEquals(2, csv.lineSequence().filter { it.isNotBlank() }.count())
        assertTrue(csv.contains("training_p95_ns"))
        assertTrue(csv.contains("report_status"))
        assertTrue(csv.lineSequence().first { it.startsWith("\"smoke-") }.endsWith("\"passed\""))
        val round = report.rounds.single()
        val quoted = report.copy(rounds = listOf(round.copy(device = round.device.copy(name = "GPU,\"quoted\""))))
        assertTrue(NeuroCudaBenchmarkReports.csv(quoted).contains("\"GPU,\"\"quoted\"\"\""))
    }

    @Test fun missingCpuTimingAndZeroDurationProduceNoInventedSpeedup() {
        val config = NeuroCudaBenchmarkConfig(epochs = 1, warmups = 0, repeats = 1, batches = listOf(16),
            backends = listOf(NeuroCudaBenchmarkBackend.CUDA))
        val report = NeuroCudaBenchmarkHarness.run(config, clock = { 0L }, sessionFactory = { model, engine, batch ->
            val cpu = model.newTrainingSession(batchSize = batch)
            object : NeuroTrainingSession by cpu { override val info = info(engine) }
        }, environment = emptyMap())
        assertNull(report.summaries.single().trainingSpeedup)
        assertNull(report.summaries.single().totalSpeedup)
        assertEquals(0.0, report.summaries.single().total.median)
    }

    @Test fun failureReportWritesNeverMaskTheOriginalTrainingFailure() {
        val cause = IllegalStateException("device synchronization failed")
        val config = NeuroCudaBenchmarkConfig(output = temporary.resolve("failed-report"))
        val report = NeuroCudaBenchmarkReport(config, emptyList(), emptyList(), emptyList(), emptyMap(),
            mapOf("stage" to "measured", "message" to cause.message))
        val failure = NeuroCudaBenchmarkFailure(report, cause)
        assertTrue(NeuroCudaBenchmarkReports.writeFailure(failure, config.output))
        assertTrue(Files.readString(Path.of("${config.output}.json")).contains("\"status\":\"failed\""))
        val writeFailure = java.io.IOException("report directory is full")
        assertFalse(NeuroCudaBenchmarkReports.writeFailure(failure, config.output) { _, _ -> throw writeFailure })
        assertSame(cause, failure.cause)
        assertArrayEquals(arrayOf(writeFailure), failure.suppressed)
        assertFalse(NeuroCudaBenchmarkReports.writeFailure(failure, config.output) { _, _ -> throw failure })
        assertEquals(1, failure.suppressed.size, "A rethrown primary failure must not be suppressed on itself")
    }

    @Test fun consoleComparisonNamesTimingScopesAndDoesNotInventSpeedupsOrSuccess() {
        val duration = NeuroCudaBenchmarkDistribution(1_000_000.0, 2_500_000.0, 3_000_000.0)
        val config = NeuroCudaBenchmarkConfig()
        val summary = NeuroCudaBenchmarkSummary("smoke-batch-16", NeuroCudaBenchmarkBackend.CUDA,
            duration, duration, duration, duration, 0.0002, null)
        val report = NeuroCudaBenchmarkReport(config, emptyList(), emptyList(), listOf(summary), emptyMap())
        val output = NeuroCudaBenchmarkReports.console(report)
        assertTrue(output.contains("epochs=2, warmups=1, repeats=3, CPU parallelism=1"))
        assertTrue(output.contains("Train median ms"))
        assertTrue(output.contains("Total median ms"))
        assertTrue(output.contains("2.500"))
        assertTrue(output.contains("<0.001x"))
        assertTrue(output.contains("n/a"))
        assertTrue(output.contains(">1 means faster than CPU"))
        assertTrue(output.contains("Numerical verification: passed"))
        val failed = report.copy(failure = mapOf("stage" to "warmup"))
        assertTrue(NeuroCudaBenchmarkReports.console(failed).contains("Numerical verification: incomplete"))
        assertFalse(NeuroCudaBenchmarkReports.console(failed).contains("verification: passed"))
        assertTrue(NeuroCudaBenchmarkReports.console(report.copy(summaries = listOf(summary.copy(trainingSpeedup = 1.0))))
            .contains("1.000x"))
    }

    private fun info(engine: NeuroCudaBenchmarkBackend) = TrainingDeviceInfo(engine.backend, "test device", "test-id",
        engine.precision.name, "test-kernels")
}
