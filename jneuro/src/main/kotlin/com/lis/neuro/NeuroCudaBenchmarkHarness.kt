package com.lis.neuro

import java.nio.file.Files
import java.nio.file.Path
import java.security.MessageDigest
import java.time.Instant
import java.util.logging.Logger
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.max

internal data class NeuroCudaBenchmarkWorkload(val name: String, val topology: List<Int>, val samples: Int, val batchSize: Int)
internal data class NeuroCudaBenchmarkValidation(val elements: Int, val absoluteTolerance: Double, val relativeTolerance: Double,
                                             val maximumAbsoluteError: Double, val maximumScaledError: Double, val rmseError: Double)
internal data class NeuroCudaBenchmarkRound(val workload: String, val backend: NeuroCudaBenchmarkBackend, val round: Int,
    val openNanos: Long, val trainingNanos: Long, val closeNanos: Long, val totalNanos: Long,
    val rmse: Double, val epochs: Long, val samplesSeen: Long, val device: TrainingDeviceInfo, val validation: NeuroCudaBenchmarkValidation)
internal data class NeuroCudaBenchmarkDistribution(val minimum: Double, val median: Double, val p95: Double)
internal data class NeuroCudaBenchmarkSummary(val workload: String, val backend: NeuroCudaBenchmarkBackend,
    val opening: NeuroCudaBenchmarkDistribution, val training: NeuroCudaBenchmarkDistribution,
    val closing: NeuroCudaBenchmarkDistribution, val total: NeuroCudaBenchmarkDistribution,
    val trainingSpeedup: Double?, val totalSpeedup: Double?)
internal data class NeuroCudaBenchmarkReport(val config: NeuroCudaBenchmarkConfig, val workloads: List<NeuroCudaBenchmarkWorkload>,
    val rounds: List<NeuroCudaBenchmarkRound>, val summaries: List<NeuroCudaBenchmarkSummary>,
    val environment: Map<String, Any?>, val failure: Map<String, Any?>? = null)
internal class NeuroCudaBenchmarkFailure(val report: NeuroCudaBenchmarkReport, cause: Throwable) :
    IllegalStateException("CPU/GPU benchmark failed: ${cause.message}", cause)

internal object NeuroCudaBenchmarkHarness {
    private const val SEED = 1234L
    fun workloads(config: NeuroCudaBenchmarkConfig): List<NeuroCudaBenchmarkWorkload> {
        val shapes = if (config.topology != null) listOf("custom" to config.topology)
            else if (config.profile == "smoke") listOf("smoke" to listOf(32, 64, 32, 4)) else listOf(
            "medium" to listOf(128, 256, 128, 32), "large" to listOf(256, 512, 256, 32))
        return shapes.flatMap { (name, shape) -> config.batches.map { batch ->
            NeuroCudaBenchmarkWorkload("$name-batch-$batch", shape, config.sampleCount, batch)
        } }
    }

    fun run(config: NeuroCudaBenchmarkConfig, clock: () -> Long = System::nanoTime,
            sessionFactory: (Neuro, NeuroCudaBenchmarkBackend, Int) -> NeuroTrainingSession = { model, engine, batch ->
                engine.open(model, batch)
            }, environment: Map<String, Any?> = environment(config),
            onWorkloadCompleted: (NeuroCudaBenchmarkWorkload, Int, Int) -> Unit = { _, _, _ -> }): NeuroCudaBenchmarkReport {
        val cases = workloads(config)
        val rounds = ArrayList<NeuroCudaBenchmarkRound>()
        var stage = "reference"
        var currentWorkload: String? = null
        var currentBackend: NeuroCudaBenchmarkBackend? = null
        try {
            for ((workloadIndex, workload) in cases.withIndex()) {
                currentWorkload = workload.name
                currentBackend = NeuroCudaBenchmarkBackend.CPU
                stage = "reference"
                val reference = prepared(workload, config.sigmoid)
                // The reference is always CPU, including GPU-only selections, and is outside measured data.
                reference.newTrainingSession(TrainingBackend.CPU, batchSize = workload.batchSize).use {
                    train(it, config.epochs, workload.batchSize, config.mode)
                }
                val state = reference.exportTrainingState()
                val error = reference.trainingError()
                stage = "warmup"
                repeat(config.warmups) {
                    for (engine in config.backends) {
                        currentBackend = engine
                        measure(workload, engine, -1, config, reference.statistics(), state, error, clock, sessionFactory)
                    }
                }
                stage = "measured"
                repeat(config.repeats) { round ->
                    val offset = round % config.backends.size
                    val order = config.backends.drop(offset) + config.backends.take(offset)
                    for (engine in order) {
                        currentBackend = engine
                        rounds += measure(workload, engine, round + 1, config, reference.statistics(), state, error, clock, sessionFactory)
                    }
                }
                // Progress runs after every timed session has closed and its state has been verified.
                stage = "progress"
                onWorkloadCompleted(workload, workloadIndex + 1, cases.size)
            }
        } catch (failure: Throwable) {
            throw NeuroCudaBenchmarkFailure(NeuroCudaBenchmarkReport(config, cases, rounds.toList(), summaries(rounds), environment,
                mapOf("stage" to stage, "workload" to currentWorkload, "backend" to currentBackend?.name,
                    "type" to failure.javaClass.name, "message" to failure.message)), failure)
        }
        return NeuroCudaBenchmarkReport(config, cases, rounds.toList(), summaries(rounds), environment)
    }

    private fun measure(workload: NeuroCudaBenchmarkWorkload, engine: NeuroCudaBenchmarkBackend, round: Int, config: NeuroCudaBenchmarkConfig,
                        referenceStatistics: Neuro.Statistics, reference: NeuroTrainingState, referenceError: Double,
                        clock: () -> Long, factory: (Neuro, NeuroCudaBenchmarkBackend, Int) -> NeuroTrainingSession): NeuroCudaBenchmarkRound {
        val epochs = config.epochs
        val model = prepared(workload, config.sigmoid)
        NeuroLog.debug("benchmark", "benchmark.round.started") { mapOf("workload" to workload.name,
            "backend" to engine, "round" to round, "epochs" to epochs, "batchSize" to workload.batchSize) }
        val start = clock()
        val session = factory(model, engine, workload.batchSize)
        val opened = clock()
        var trainingEnd = opened
        var primary: Throwable? = null
        try {
            check(session.info.backend == engine.backend && session.info.precision == engine.precision.name) {
                "Requested $engine but session resolved ${session.info.backend}/${session.info.precision}"
            }
            train(session, epochs, workload.batchSize, config.mode)
            trainingEnd = clock()
        } catch (failure: Throwable) {
            primary = failure
            throw failure
        } finally {
            try { session.close() } catch (cleanup: Throwable) {
                if (primary == null) throw cleanup else primary.addSuppressed(cleanup)
            }
        }
        val closed = clock()
        val statistics = model.statistics()
        check(statistics.epochsTrained == referenceStatistics.epochsTrained && statistics.samplesSeen == referenceStatistics.samplesSeen) {
            "Backend $engine completed different epoch/sample counts from CPU"
        }
        val rmse = model.trainingError()
        val validation = validate(reference, model.exportTrainingState(), referenceError, rmse, engine.precision)
        check(validation.maximumScaledError <= 1.0) {
            "Backend $engine state differs from CPU beyond tolerance: scaled=${validation.maximumScaledError}, absolute=${validation.maximumAbsoluteError}"
        }
        return NeuroCudaBenchmarkRound(workload.name, engine, round, opened - start, trainingEnd - opened,
            closed - trainingEnd, closed - start, rmse, statistics.epochsTrained, statistics.samplesSeen, session.info, validation)
    }

    internal fun train(session: NeuroTrainingSession, epochs: Int, batch: Int, mode: NeuroBenchmarkMode) {
        when (mode) {
            NeuroBenchmarkMode.MINIBATCH -> session.trainMiniBatch(epochs, batch, 1)
            NeuroBenchmarkMode.EPOCH -> repeat(epochs) { session.trainEpoch() }
            NeuroBenchmarkMode.CHUNK -> {
                var completed = 0
                while (completed < epochs) {
                    val result = session.trainChunk(TrainingChunkRequest(epochs - completed))
                    check(result.committedEpochs > 0) { "Benchmark chunk made no progress" }
                    completed += result.committedEpochs
                }
            }
        }
    }

    internal fun validate(expected: NeuroTrainingState, actual: NeuroTrainingState, expectedError: Double,
                          actualError: Double, precision: Neuro.TrainingPrecision): NeuroCudaBenchmarkValidation {
        check(expected.topology.contentEquals(actual.topology)) { "Backend changed topology" }
        val absolute = if (precision == Neuro.TrainingPrecision.FP64) 1e-10 else 5e-5
        val relative = if (precision == Neuro.TrainingPrecision.FP64) 1e-8 else 2e-3
        val expectedArrays = expected.weights + expected.biases + expected.weightVelocity + expected.biasVelocity
        val actualArrays = actual.weights + actual.biases + actual.weightVelocity + actual.biasVelocity
        check(expectedArrays.size == actualArrays.size) { "Backend changed parameter buffers" }
        var count = 0
        var maxAbsolute = 0.0
        var maxScaled = 0.0
        fun compare(left: Double, right: Double) {
            check(left.isFinite() && right.isFinite()) { "Benchmark produced non-finite state or RMSE" }
            val difference = abs(left - right)
            maxAbsolute = max(maxAbsolute, difference)
            maxScaled = max(maxScaled, difference / (absolute + relative * abs(left)))
            count++
        }
        for (index in expectedArrays.indices) {
            check(expectedArrays[index].size == actualArrays[index].size) { "Backend changed parameter shape" }
            for (element in expectedArrays[index].indices) compare(expectedArrays[index][element], actualArrays[index][element])
        }
        compare(expectedError, actualError)
        return NeuroCudaBenchmarkValidation(count, absolute, relative, maxAbsolute, maxScaled, abs(expectedError - actualError))
    }

    internal fun distribution(values: List<Long>): NeuroCudaBenchmarkDistribution {
        require(values.isNotEmpty() && values.all { it >= 0 }) { "Timings must be nonempty and nonnegative" }
        val sorted = values.sorted()
        val median = if (sorted.size % 2 == 0) sorted[sorted.size / 2 - 1] / 2.0 + sorted[sorted.size / 2] / 2.0
            else sorted[sorted.size / 2].toDouble()
        return NeuroCudaBenchmarkDistribution(sorted.first().toDouble(), median, sorted[ceil(sorted.size * 0.95).toInt() - 1].toDouble())
    }

    private fun summaries(rounds: List<NeuroCudaBenchmarkRound>): List<NeuroCudaBenchmarkSummary> {
        val grouped = rounds.groupBy { it.workload to it.backend }
        return grouped.map { (key, values) ->
            val cpu = grouped[key.first to NeuroCudaBenchmarkBackend.CPU]
            val training = distribution(values.map { it.trainingNanos })
            val total = distribution(values.map { it.totalNanos })
            NeuroCudaBenchmarkSummary(key.first, key.second, distribution(values.map { it.openNanos }), training,
                distribution(values.map { it.closeNanos }), total,
                if (cpu == null || training.median == 0.0) null else distribution(cpu.map { it.trainingNanos }).median / training.median,
                if (cpu == null || total.median == 0.0) null else distribution(cpu.map { it.totalNanos }).median / total.median)
        }
    }

    internal fun prepared(workload: NeuroCudaBenchmarkWorkload, sigmoid: Neuro.SigmoidMode = Neuro.SigmoidMode.EXACT): Neuro =
        Neuro(workload.topology.toIntArray(), Neuro.HyperParameters(0.05, 0.1, 1.0, SEED, Neuro.Kernel.VECTOR, sigmoid)).also { model ->
            repeat(workload.samples) { sample ->
                model.addTrainingSample(DoubleArray(workload.topology.first()) { ((sample * 17 + it * 13) and 255) / 255.0 },
                    DoubleArray(workload.topology.last()) { ((sample + it) and 1).toDouble() })
            }
        }

    private fun environment(config: NeuroCudaBenchmarkConfig): Map<String, Any?> {
        NeuroLog.initialize()
        fun hash(resource: String): String? = NeuroCudaBenchmarkHarness::class.java.getResourceAsStream(resource)?.use {
            MessageDigest.getInstance("SHA-256").digest(it.readAllBytes()).joinToString("") { value -> "%02x".format(value) }
        }
        val cudaBuild = java.util.Properties().also { properties ->
            NeuroCudaBenchmarkHarness::class.java.getResourceAsStream("/com/lis/neuro/cuda/train.properties")?.use(properties::load)
        }.entries.associate { it.key.toString() to it.value.toString() }
        return linkedMapOf("timestamp" to Instant.now().toString(), "javaVersion" to System.getProperty("java.runtime.version"),
            "processId" to ProcessHandle.current().pid(), "jvmStartMillis" to java.lang.management.ManagementFactory.getRuntimeMXBean().startTime,
            "jvmArguments" to java.lang.management.ManagementFactory.getRuntimeMXBean().inputArguments,
            "javaVm" to System.getProperty("java.vm.name"), "os" to System.getProperty("os.name"),
            "osVersion" to System.getProperty("os.version"), "architecture" to System.getProperty("os.arch"),
            "availableProcessors" to Runtime.getRuntime().availableProcessors(), "cpu" to System.getenv("PROCESSOR_IDENTIFIER"),
            "maximumHeapBytes" to Runtime.getRuntime().maxMemory(), "loggingLevel" to Logger.getLogger("com.lis.neuro").level?.name,
            "sourceRevision" to (System.getProperty("jneuro.benchmark.sourceRevision") ?: "unspecified"),
            "benchmarkClassSha256" to hash("/com/lis/neuro/NeuroCudaBenchmarkHarness.class"),
            "ptxSha256" to hash("/com/lis/neuro/cuda/train.ptx"), "cudaBuildPropertiesSha256" to hash("/com/lis/neuro/cuda/train.properties"),
            "cublasKernelSourceSha256" to hash("/cuda/jneuro.cu"), "cudaBuild" to cudaBuild, "seed" to SEED,
            "learningRate" to 0.05, "momentum" to 0.1, "beta" to 1.0, "sigmoid" to config.sigmoid.name, "cpuKernel" to "VECTOR",
            "timingScope" to "fresh model/data preparation and numerical validation excluded; open, ${config.mode} training, close included; CPU parallelism=1",
            "cublasTiming" to "Runtime allocation, NVRTC compilation, transfers and cleanup occur within the training call; open separately probes device metadata",
            "order" to "measured backend order rotates once per repetition; all warmups excluded", "p95Definition" to "nearest rank")
    }
}

internal object NeuroCudaBenchmarkReports {
    fun write(report: NeuroCudaBenchmarkReport, prefix: Path) {
        prefix.toAbsolutePath().parent?.let(Files::createDirectories)
        Files.writeString(Path.of("$prefix.json"), json(report) + "\n")
        Files.writeString(Path.of("$prefix.csv"), csv(report))
    }

    /** Failure evidence is best effort; a report write must never replace the native/training failure. */
    fun writeFailure(failure: NeuroCudaBenchmarkFailure, prefix: Path,
                     writer: (NeuroCudaBenchmarkReport, Path) -> Unit = ::write): Boolean = try {
        writer(failure.report, prefix)
        true
    } catch (reportFailure: Throwable) {
        if (reportFailure !== failure) failure.addSuppressed(reportFailure)
        NeuroLog.error("benchmark", "benchmark.failure.report.failed", reportFailure, "prefix" to prefix.toString())
        false
    }

    fun console(report: NeuroCudaBenchmarkReport): String {
        fun format(pattern: String, vararg values: Any?) = String.format(java.util.Locale.ROOT, pattern, *values)
        fun speedup(value: Double?): String = when {
            value == null -> "n/a"
            value > 0.0 && value < 0.001 -> "<0.001x"
            else -> format("%.3fx", value)
        }
        val columns = "%-23s %-20s %15s %15s %13s %13s"
        return buildString {
            appendLine("CPU/GPU training comparison: profile=${report.config.profile}, epochs=${report.config.epochs}, " +
                "warmups=${report.config.warmups}, repeats=${report.config.repeats}, CPU parallelism=1, " +
                "topology=${report.config.topology?.joinToString("x") ?: "profile-default"}, samples=${report.config.sampleCount}")
            appendLine(format(columns, "Workload", "Backend", "Train median ms", "Total median ms", "Train speedup", "Total speedup"))
            for (summary in report.summaries) {
                appendLine(format(columns, summary.workload, summary.backend.name,
                    format("%.3f", summary.training.median / 1_000_000.0),
                    format("%.3f", summary.total.median / 1_000_000.0),
                    speedup(summary.trainingSpeedup), speedup(summary.totalSpeedup)))
            }
            appendLine("Speedup = CPU median / backend median; >1 means faster than CPU.")
            appendLine("Train includes the selected training API; total includes session open, train and close. Mode=${report.config.mode}, sigmoid=${report.config.sigmoid}, retained-device=${report.config.retainedDevice}.")
            if (report.failure == null) append("Numerical verification: passed (${report.rounds.size} measured rounds; weights, biases, momentum buffers and RMSE).")
            else append("Numerical verification: incomplete (benchmark failed).")
        }
    }

    fun json(report: NeuroCudaBenchmarkReport): String {
        fun distribution(value: NeuroCudaBenchmarkDistribution) = mapOf("minNanos" to value.minimum, "medianNanos" to value.median, "p95Nanos" to value.p95)
        return encode(linkedMapOf("schemaVersion" to 1, "status" to if (report.failure == null) "passed" else "failed",
            "environment" to report.environment, "configuration" to mapOf("profile" to report.config.profile,
                "epochs" to report.config.epochs, "warmups" to report.config.warmups, "repeats" to report.config.repeats,
                "batches" to report.config.batches, "backends" to report.config.backends.map { it.name },
                "topology" to report.config.topology, "samples" to report.config.sampleCount,
                "mode" to report.config.mode, "sigmoid" to report.config.sigmoid, "retainedDevice" to report.config.retainedDevice),
            "workloads" to report.workloads.map { mapOf("name" to it.name, "topology" to it.topology, "samples" to it.samples, "batchSize" to it.batchSize) },
            "rounds" to report.rounds.map { round -> mapOf("workload" to round.workload, "backend" to round.backend.name,
                "round" to round.round, "openNanos" to round.openNanos, "trainingNanos" to round.trainingNanos,
                "closeNanos" to round.closeNanos, "totalNanos" to round.totalNanos, "rmse" to round.rmse,
                "epochs" to round.epochs, "samplesSeen" to round.samplesSeen,
                "device" to mapOf("name" to round.device.name, "identity" to round.device.identity,
                    "backend" to round.device.backend.name, "precision" to round.device.precision, "kernelVersion" to round.device.kernelVersion,
                    "engine" to round.device.engine, "simdBits" to round.device.simdBits, "sigmoid" to round.device.sigmoid),
                "validation" to mapOf("elements" to round.validation.elements, "absoluteTolerance" to round.validation.absoluteTolerance,
                    "relativeTolerance" to round.validation.relativeTolerance, "maxAbsoluteError" to round.validation.maximumAbsoluteError,
                    "maxScaledError" to round.validation.maximumScaledError, "rmseError" to round.validation.rmseError)) },
            "summaries" to report.summaries.map { mapOf("workload" to it.workload, "backend" to it.backend.name,
                "open" to distribution(it.opening), "training" to distribution(it.training), "close" to distribution(it.closing),
                "total" to distribution(it.total), "trainingSpeedup" to it.trainingSpeedup, "endToEndSpeedup" to it.totalSpeedup) },
            "failure" to report.failure))
    }

    fun csv(report: NeuroCudaBenchmarkReport): String {
        val header = listOf("workload", "backend", "round", "open_ns", "training_ns", "close_ns", "total_ns", "rmse",
            "epochs", "samples_seen", "device", "identity", "precision", "kernel", "max_absolute_error", "max_scaled_error",
            "training_median_ns", "training_min_ns", "training_p95_ns", "total_median_ns", "training_speedup", "end_to_end_speedup", "report_status")
        val summaries = report.summaries.associateBy { it.workload to it.backend }
        return buildString {
            appendLine(header.joinToString(","))
            for (round in report.rounds) {
                val summary = summaries.getValue(round.workload to round.backend)
                appendLine(listOf(round.workload, round.backend.name, round.round, round.openNanos, round.trainingNanos,
                    round.closeNanos, round.totalNanos, round.rmse, round.epochs, round.samplesSeen, round.device.name,
                    round.device.identity, round.device.precision, round.device.kernelVersion, round.validation.maximumAbsoluteError,
                    round.validation.maximumScaledError, summary.training.median, summary.training.minimum, summary.training.p95,
                    summary.total.median, summary.trainingSpeedup, summary.totalSpeedup,
                    if (report.failure == null) "passed" else "failed").joinToString(",") { csvValue(it) })
            }
        }
    }

    private fun csvValue(value: Any?): String = "\"" + (value?.toString() ?: "").replace("\"", "\"\"") + "\""
    internal fun encode(value: Any?): String = when (value) {
        null -> "null"
        is Boolean, is Number -> value.toString()
        is Map<*, *> -> value.entries.joinToString(",", "{", "}") { encode(it.key.toString()) + ":" + encode(it.value) }
        is Iterable<*> -> value.joinToString(",", "[", "]") { encode(it) }
        else -> buildString {
            append('"')
            for (character in value.toString()) when (character) {
                '"' -> append("\\\""); '\\' -> append("\\\\"); '\n' -> append("\\n"); '\r' -> append("\\r"); '\t' -> append("\\t")
                else -> if (character.code < 32) append("\\u%04x".format(java.util.Locale.ROOT, character.code)) else append(character)
            }
            append('"')
        }
    }
}
