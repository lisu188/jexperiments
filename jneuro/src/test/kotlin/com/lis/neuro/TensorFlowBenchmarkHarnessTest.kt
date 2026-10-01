package com.lis.neuro

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import java.nio.file.Files
import java.nio.file.Path

class TensorFlowBenchmarkHarnessTest {
    @TempDir lateinit var temporary: Path

    @Test fun onlyExplicitTensorFlowDeviceAndPrecisionPresetsAreAccepted() {
        val parsed = TensorFlowBenchmarkConfig.parse(arrayOf("--backends", "CPU_FP64,CPU_FP32,GPU_FP64,GPU_FP32"))
        assertEquals(listOf(TrainingBackend.CPU to Neuro.TrainingPrecision.FP64,
            TrainingBackend.CPU to Neuro.TrainingPrecision.FP32,
            TrainingBackend.GPU to Neuro.TrainingPrecision.FP64,
            TrainingBackend.GPU to Neuro.TrainingPrecision.FP32), parsed.backends.map { it.backend to it.precision })
        for (obsolete in listOf("CPU", "GPU", "CUDA", "CUBLAS_FP64", "CUBLAS_FP32", "CPU_LEGACY",
            "SMALL_SCALAR_FP64", "SMALL_SCALAR_FP32", "SMALL_128_FP64", "SMALL_256_FP64",
            "SMALL_128_FP32", "SMALL_256_FP32", "SMALL_CUDA_FP64", "SMALL_CUDA_FP32", "NATIVE_FP64")) {
            assertThrows(IllegalArgumentException::class.java, {
                TensorFlowBenchmarkConfig.parse(arrayOf("--backends", obsolete))
            }, obsolete)
        }
        assertEquals(Path.of("build", "reports", "tensorflow-benchmark", "benchmark"), parsed.output)
        assertTrue(TensorFlowBenchmarkConfig.usage().contains("CPU_FP64,CPU_FP32,GPU_FP64,GPU_FP32"))
    }

    @Test fun cpuPrecisionsAndPublicationModesVerifyCompleteStateAndReportActualPrecision() {
        val engines = listOf(TensorFlowBenchmarkBackend.CPU_FP64, TensorFlowBenchmarkBackend.CPU_FP32)
        for (mode in NeuroBenchmarkMode.entries) for (sigmoid in Neuro.SigmoidMode.entries) {
            val config = TensorFlowBenchmarkConfig(epochs = 2, warmups = 0, repeats = 1,
                topology = listOf(2, 4, 8, 1), samples = 9, batches = listOf(7), backends = engines,
                mode = mode, sigmoid = sigmoid)
            val report = TensorFlowBenchmarkHarness.run(config, environment = emptyMap())
            assertEquals(engines.size, report.rounds.size)
            assertTrue(report.rounds.all { it.validation.maximumScaledError <= 1.0 })
            assertEquals(sigmoid.name, report.rounds.last().device.sigmoid)
            assertEquals(listOf("FP64", "FP32"), report.rounds.map { it.device.precision })
            assertTrue(report.rounds.all { it.device.backend == TrainingBackend.CPU })
            assertTrue(TensorFlowBenchmarkReports.json(report).contains("\"mode\":\"$mode\""))
        }
        val parsed = TensorFlowBenchmarkConfig.parse(arrayOf("--mode", "CHUNK", "--sigmoid", "FAST", "--retained-device", "true"))
        assertEquals(NeuroBenchmarkMode.CHUNK, parsed.mode)
        assertEquals(Neuro.SigmoidMode.FAST, parsed.sigmoid)
        assertTrue(parsed.retainedDevice)
        assertThrows(IllegalArgumentException::class.java) { TensorFlowBenchmarkConfig.parse(arrayOf("--retained-device", "perhaps")) }
        val actual = NeuroTest.prepared(intArrayOf(2, 4, 1))
        NeuroTrainingDeviceService().use { service ->
            TensorFlowBenchmarkBackend.CPU_FP64.open(actual, 7, service).use { assertTrue(it.trainEpoch().isFinite()) }
        }
    }

    @Test fun defaultsAreBoundedAndArgumentsAreExplicitAndValidated() {
        val defaults = TensorFlowBenchmarkConfig.parse(emptyArray())
        assertEquals("smoke", defaults.profile)
        assertEquals(listOf(16, 64), defaults.batches)
        assertEquals(TensorFlowBenchmarkBackend.entries, defaults.backends)
        assertEquals(listOf(128, 128), TensorFlowBenchmarkHarness.workloads(defaults).map { it.samples })
        val parsed = TensorFlowBenchmarkConfig.parse(arrayOf("--profile", "matrix", "--epochs", "3", "--warmups", "0",
            "--repeats", "2", "--batches", "8,32", "--backends", "cpu_fp64,gpu_fp64", "--output", temporary.resolve("result").toString()))
        assertEquals(3, parsed.epochs)
        assertEquals(0, parsed.warmups)
        assertEquals(2, parsed.repeats)
        assertEquals(listOf(TensorFlowBenchmarkBackend.CPU_FP64, TensorFlowBenchmarkBackend.GPU_FP64), parsed.backends)
        assertEquals(4, TensorFlowBenchmarkHarness.workloads(parsed).size)
        assertEquals("smoke", TensorFlowBenchmarkConfig.parse(arrayOf("--profile", "matrix", "--smoke")).profile)
        for (args in listOf(arrayOf("--epochs"), arrayOf("--unknown", "value"), arrayOf("--epochs", "bad"),
            arrayOf("--epochs", "0"), arrayOf("--warmups", "-1"), arrayOf("--repeats", "0"),
            arrayOf("--profile", "huge"), arrayOf("--batches", "0"), arrayOf("--batches", "16,16"),
            arrayOf("--backends", "AUTO"), arrayOf("--backends", "CPU_FP64,CPU_FP64"),
            arrayOf("--topology", "8"), arrayOf("--topology", "8,0,8"), arrayOf("--topology", "8,2049,8"),
            arrayOf("--topology", List(17) { "8" }.joinToString(",")), arrayOf("--topology", "2048,2048"),
            arrayOf("--samples", "0"), arrayOf("--samples", "8193"))) {
            assertThrows(IllegalArgumentException::class.java, { TensorFlowBenchmarkConfig.parse(args) }, args.joinToString(" "))
        }
        assertThrows(IllegalArgumentException::class.java) { defaults.copy(backends = emptyList()) }
        assertThrows(IllegalArgumentException::class.java) { defaults.copy(batches = emptyList()) }
    }

    @Test fun customTopologyReplacesPresetsAndSamplesAreExplicitInModelsAndReports() {
        val config = TensorFlowBenchmarkConfig.parse(arrayOf("--topology", "2,8,8,8,1", "--samples", "16", "--epochs", "1",
            "--warmups", "0", "--repeats", "1", "--backends", "CPU_FP64", "--batches", "4"))
        val workload = TensorFlowBenchmarkHarness.workloads(config).single()
        assertEquals("custom-batch-4", workload.name)
        assertEquals(listOf(2, 8, 8, 8, 1), workload.topology)
        assertEquals(16, workload.samples)
        val model = TensorFlowBenchmarkHarness.prepared(workload)
        assertArrayEquals(intArrayOf(2, 8, 8, 8, 1), model.topology())
        assertEquals(16, model.trainingSampleCount())
        assertEquals(177, model.parameterCount())
        val report = TensorFlowBenchmarkHarness.run(config, environment = emptyMap())
        assertTrue(TensorFlowBenchmarkReports.console(report).contains("topology=2x8x8x8x1, samples=16"))
        val json = TensorFlowBenchmarkReports.json(report)
        assertTrue(json.contains("\"topology\":[2,8,8,8,1]"))
        assertTrue(json.contains("\"samples\":16"))
        assertEquals(16L, report.rounds.single().samplesSeen)
        assertEquals(128, config.copy(samples = null).sampleCount)
        assertEquals(1024, config.copy(profile = "matrix", samples = null).sampleCount)
        assertEquals(1, TensorFlowBenchmarkHarness.workloads(config.copy(profile = "matrix")).size)
        assertEquals(listOf(16, 16), TensorFlowBenchmarkHarness.workloads(config.copy(profile = "matrix", topology = null)).map { it.samples })
    }

    @Test fun commandLineHelpDescribesCustomShapeAndReturnsWithoutTraining() {
        val previous = System.out
        val output = java.io.ByteArrayOutputStream()
        try {
            System.setOut(java.io.PrintStream(output, true, Charsets.UTF_8))
            TensorFlowTrainingBenchmark.main(arrayOf("--help"))
            TensorFlowTrainingBenchmark.main(arrayOf("-h"))
        } finally { System.setOut(previous) }
        val text = output.toString(Charsets.UTF_8)
        assertTrue(text.contains("--topology 2,8,8,8,1"))
        assertTrue(text.contains("--samples N"))
        assertTrue(text.contains("without CPU fallback"))
        assertFalse(text.contains("Starting CPU/GPU benchmark"))
    }

    @Test fun warmupsAreExcludedFreshModelsRotateOrderAndTimingsHaveComparableScopes() {
        val config = TensorFlowBenchmarkConfig(epochs = 1, warmups = 1, repeats = 3, batches = listOf(16))
        val requests = ArrayList<TensorFlowBenchmarkBackend>()
        val modelIds = HashSet<String>()
        var closeCount = 0
        var nanos = 0L
        val progress = ArrayList<Triple<String, Int, Int>>()
        val report = TensorFlowBenchmarkHarness.run(config, clock = { nanos += 10; nanos }, sessionFactory = { model, engine, batch ->
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
        assertEquals(0L, TensorFlowBenchmarkHarness.prepared(TensorFlowBenchmarkHarness.workloads(config).first()).statistics().epochsTrained)
    }

    @Test fun distributionsIncludeEvenMediansAndNearestRankP95() {
        val result = TensorFlowBenchmarkHarness.distribution((1L..100L).toList().reversed())
        assertEquals(1.0, result.minimum)
        assertEquals(50.5, result.median)
        assertEquals(95.0, result.p95)
        assertEquals(5.0, TensorFlowBenchmarkHarness.distribution(listOf(9, 1, 5)).median)
        assertEquals(0.0, TensorFlowBenchmarkHarness.distribution(listOf(0)).p95)
        assertThrows(IllegalArgumentException::class.java) { TensorFlowBenchmarkHarness.distribution(emptyList()) }
        assertThrows(IllegalArgumentException::class.java) { TensorFlowBenchmarkHarness.distribution(listOf(-1)) }
    }

    @Test fun fullStateValidationIncludesMomentumPrecisionAndRmse() {
        val model = TensorFlowBenchmarkHarness.prepared(TensorFlowBenchmarkWorkload("test", listOf(2, 2, 1), 2, 1))
        val expected = model.exportTrainingState()
        val actual = model.exportTrainingState()
        actual.weightVelocity[0][0] += 0.01
        val mismatch = TensorFlowBenchmarkHarness.validate(expected, actual, 0.1, 0.1, Neuro.TrainingPrecision.FP64)
        assertEquals(0.01, mismatch.maximumAbsoluteError)
        assertTrue(mismatch.maximumScaledError > 1.0)
        assertEquals(model.parameterCount() * 2 + 1, mismatch.elements)
        actual.weightVelocity[0][0] = 1e-5
        assertTrue(TensorFlowBenchmarkHarness.validate(expected, actual, 0.1, 0.1, Neuro.TrainingPrecision.FP32).maximumScaledError < 1.0)
        assertTrue(TensorFlowBenchmarkHarness.validate(expected, expected, 0.1, 0.2, Neuro.TrainingPrecision.FP64).maximumScaledError > 1.0)
        assertThrows(IllegalStateException::class.java) {
            TensorFlowBenchmarkHarness.validate(expected, actual, 0.1, Double.NaN, Neuro.TrainingPrecision.FP32)
        }
        assertThrows(IllegalStateException::class.java) {
            TensorFlowBenchmarkHarness.validate(expected, actual.copy(topology = intArrayOf(2, 3, 1)), 0.1, 0.1, Neuro.TrainingPrecision.FP64)
        }
        assertThrows(IllegalStateException::class.java) {
            TensorFlowBenchmarkHarness.validate(expected, actual.copy(weights = emptyArray()), 0.1, 0.1, Neuro.TrainingPrecision.FP64)
        }
        assertThrows(IllegalStateException::class.java) {
            TensorFlowBenchmarkHarness.validate(expected, actual.copy(weights = arrayOf(DoubleArray(1), actual.weights[1])), 0.1, 0.1, Neuro.TrainingPrecision.FP64)
        }
    }

    @Test fun selectedGpuFailureIsExplicitAndKeepsCompletedEvidenceWithoutFallback() {
        val config = TensorFlowBenchmarkConfig(epochs = 1, warmups = 0, repeats = 1, batches = listOf(16),
            backends = listOf(TensorFlowBenchmarkBackend.CPU_FP64, TensorFlowBenchmarkBackend.GPU_FP64))
        val unavailable = UnsatisfiedLinkError("GPU is unavailable")
        val failure = assertThrows(TensorFlowBenchmarkFailure::class.java) {
            TensorFlowBenchmarkHarness.run(config, sessionFactory = { model, engine, batch ->
                if (engine == TensorFlowBenchmarkBackend.GPU_FP64) throw unavailable
                model.newTrainingSession(batchSize = batch)
            }, environment = emptyMap())
        }
        assertSame(unavailable, failure.cause)
        assertEquals("measured", failure.report.failure!!["stage"])
        assertEquals("GPU_FP64", failure.report.failure["backend"])
        assertEquals(1, failure.report.rounds.size)
        assertEquals(TensorFlowBenchmarkBackend.CPU_FP64, failure.report.rounds.single().backend)
        assertTrue(TensorFlowBenchmarkReports.json(failure.report).contains("\"status\":\"failed\""))
    }

    @Test fun wrongResolvedDeviceAndFailedTrainingAreRejectedAndCleanupNeverMasksFailure() {
        val config = TensorFlowBenchmarkConfig(epochs = 1, warmups = 1, repeats = 1, batches = listOf(16),
            backends = listOf(TensorFlowBenchmarkBackend.GPU_FP64))
        var closed = 0
        val wrong = assertThrows(TensorFlowBenchmarkFailure::class.java) {
            TensorFlowBenchmarkHarness.run(config, sessionFactory = { model, _, batch ->
                val cpu = model.newTrainingSession(batchSize = batch)
                object : NeuroTrainingSession by cpu { override fun close() { closed++; cpu.close() } }
            }, environment = emptyMap())
        }
        assertTrue(wrong.cause!!.message!!.contains("resolved CPU"))
        assertEquals(1, closed)
        val trainFailure = IllegalStateException("native training failed")
        val closeFailure = IllegalStateException("native cleanup failed")
        val failed = assertThrows(TensorFlowBenchmarkFailure::class.java) {
            TensorFlowBenchmarkHarness.run(config, sessionFactory = { model, engine, batch ->
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
        val config = TensorFlowBenchmarkConfig(epochs = 1, warmups = 0, repeats = 1, batches = listOf(16),
            backends = listOf(TensorFlowBenchmarkBackend.CPU_FP64), output = temporary.resolve("nested/benchmark"))
        val report = TensorFlowBenchmarkHarness.run(config, environment = mapOf("sourceRevision" to "abc", "special" to "a\"b\\c\n\r\t\u0001"))
        TensorFlowBenchmarkReports.write(report, config.output)
        val json = Files.readString(Path.of("${config.output}.json"))
        val csv = Files.readString(Path.of("${config.output}.csv"))
        assertTrue(json.contains("\"status\":\"passed\""))
        assertTrue(json.contains("\"sourceRevision\":\"abc\""))
        assertFalse(json.contains("simdBits"))
        assertTrue(json.contains("a\\\"b\\\\c\\n\\r\\t\\u0001"))
        assertTrue(json.contains("\"absoluteTolerance\":1.0E-10"))
        assertTrue(json.contains("\"endToEndSpeedup\":1.0"))
        assertEquals(2, csv.lineSequence().filter { it.isNotBlank() }.count())
        assertTrue(csv.contains("training_p95_ns"))
        assertTrue(csv.contains("report_status"))
        assertTrue(csv.lineSequence().first { it.startsWith("\"smoke-") }.endsWith("\"passed\""))
        val round = report.rounds.single()
        val quoted = report.copy(rounds = listOf(round.copy(device = round.device.copy(name = "GPU,\"quoted\""))))
        assertTrue(TensorFlowBenchmarkReports.csv(quoted).contains("\"GPU,\"\"quoted\"\"\""))
    }

    @Test fun missingCpuTimingAndZeroDurationProduceNoInventedSpeedup() {
        val config = TensorFlowBenchmarkConfig(epochs = 1, warmups = 0, repeats = 1, batches = listOf(16),
            backends = listOf(TensorFlowBenchmarkBackend.GPU_FP64))
        val report = TensorFlowBenchmarkHarness.run(config, clock = { 0L }, sessionFactory = { model, engine, batch ->
            val cpu = model.newTrainingSession(batchSize = batch)
            object : NeuroTrainingSession by cpu { override val info = info(engine) }
        }, environment = emptyMap())
        assertNull(report.summaries.single().trainingSpeedup)
        assertNull(report.summaries.single().totalSpeedup)
        assertEquals(0.0, report.summaries.single().total.median)
    }

    @Test fun failureReportWritesNeverMaskTheOriginalTrainingFailure() {
        val cause = IllegalStateException("device synchronization failed")
        val config = TensorFlowBenchmarkConfig(output = temporary.resolve("failed-report"))
        val report = TensorFlowBenchmarkReport(config, emptyList(), emptyList(), emptyList(), emptyMap(),
            mapOf("stage" to "measured", "message" to cause.message))
        val failure = TensorFlowBenchmarkFailure(report, cause)
        assertTrue(TensorFlowBenchmarkReports.writeFailure(failure, config.output))
        assertTrue(Files.readString(Path.of("${config.output}.json")).contains("\"status\":\"failed\""))
        val writeFailure = java.io.IOException("report directory is full")
        assertFalse(TensorFlowBenchmarkReports.writeFailure(failure, config.output) { _, _ -> throw writeFailure })
        assertSame(cause, failure.cause)
        assertArrayEquals(arrayOf(writeFailure), failure.suppressed)
        assertFalse(TensorFlowBenchmarkReports.writeFailure(failure, config.output) { _, _ -> throw failure })
        assertEquals(1, failure.suppressed.size, "A rethrown primary failure must not be suppressed on itself")
    }

    @Test fun consoleComparisonNamesTimingScopesAndDoesNotInventSpeedupsOrSuccess() {
        val duration = TensorFlowBenchmarkDistribution(1_000_000.0, 2_500_000.0, 3_000_000.0)
        val config = TensorFlowBenchmarkConfig()
        val summary = TensorFlowBenchmarkSummary("smoke-batch-16", TensorFlowBenchmarkBackend.GPU_FP64,
            duration, duration, duration, duration, 0.0002, null)
        val report = TensorFlowBenchmarkReport(config, emptyList(), emptyList(), listOf(summary), emptyMap())
        val output = TensorFlowBenchmarkReports.console(report)
        assertTrue(output.contains("epochs=2, warmups=1, repeats=3, CPU parallelism=1"))
        assertTrue(output.contains("Train median ms"))
        assertTrue(output.contains("Total median ms"))
        assertTrue(output.contains("2.500"))
        assertTrue(output.contains("<0.001x"))
        assertTrue(output.contains("n/a"))
        assertTrue(output.contains(">1 means faster than CPU_FP64"))
        assertTrue(output.contains("Numerical verification: passed"))
        val failed = report.copy(failure = mapOf("stage" to "warmup"))
        assertTrue(TensorFlowBenchmarkReports.console(failed).contains("Numerical verification: incomplete"))
        assertFalse(TensorFlowBenchmarkReports.console(failed).contains("verification: passed"))
        assertTrue(TensorFlowBenchmarkReports.console(report.copy(summaries = listOf(summary.copy(trainingSpeedup = 1.0))))
            .contains("1.000x"))
    }

    private fun info(engine: TensorFlowBenchmarkBackend) = TrainingDeviceInfo(engine.backend, "test device", "test-id",
        engine.precision.name, "test-kernels")
}
