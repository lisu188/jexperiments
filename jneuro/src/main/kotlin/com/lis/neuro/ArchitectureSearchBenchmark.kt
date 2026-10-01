package com.lis.neuro

import java.lang.management.ManagementFactory
import java.nio.file.Files
import java.nio.file.Path
import java.time.Instant
import kotlin.math.ceil

/** End-to-end search measurements. Each invocation is one JVM fork. */
object ArchitectureSearchBenchmark {
    @JvmStatic fun main(args: Array<String>) = NeuroLog.application("ArchitectureSearchBenchmark") {
        if (args.contentEquals(arrayOf("--help"))) { println(SearchBenchmarkOptions.usage()); return@application }
        val options = SearchBenchmarkOptions.parse(args)
        val output = options.output.toAbsolutePath()
        Files.createDirectories(output.parent)
        Files.newBufferedWriter(output).use { writer ->
            SearchBenchmarkHarness.run(options) { record ->
                writer.write(NeuroCudaBenchmarkReports.encode(record)); writer.newLine(); writer.flush()
                println(NeuroCudaBenchmarkReports.encode(record.filterKeys { it in setOf("type", "case", "round", "totalNanos", "termination", "completeTrials", "targetMet", "error") }))
            }
        }
        println("Search benchmark report: $output")
    }
}

internal data class SearchBenchmarkOptions(
    val mode: String = "fixed", val manifest: String = "small", val epochs: Int = 2_000,
    val workers: List<Int> = listOf(4), val backends: List<TrainingBackend> = listOf(TrainingBackend.CPU),
    val executions: List<ArchitectureExecution> = listOf(ArchitectureExecution.REFERENCE, ArchitectureExecution.OPTIMIZED),
    val warmups: Int = 1, val repeats: Int = 3, val orderOffset: Int = 0, val limitSeconds: Long = 0,
    val searchSeeds: List<Long> = listOf(42, 123), val trials: Int = 60,
    val output: Path = Path.of("build/reports/search-benchmark/search.jsonl"),
    val engine: TrainingEngine = TrainingEngine.SMALL
) {
    init {
        require(mode in setOf("fixed", "quality") && manifest in setOf("small", "general"))
        require(mode != "quality" || manifest == "small") { "Quality protocol uses the SMALL family." }
        require(mode != "quality" || engine == TrainingEngine.SMALL)
        require(TrainingBackend.CUDA !in backends || engine == TrainingEngine.SMALL) { "GPU search benchmark requires SMALL." }
        require(epochs > 0 && warmups in 0..10 && repeats in 1..100 && orderOffset >= 0 && limitSeconds >= 0)
        require(workers.isNotEmpty() && workers.distinct().size == workers.size && workers.all { it in 1..64 })
        require(backends.isNotEmpty() && backends.distinct().size == backends.size && backends.all { it in setOf(TrainingBackend.CPU, TrainingBackend.CUDA) })
        require(manifest != "general" || backends == listOf(TrainingBackend.CPU)) { "General-width regressions use CPU REFERENCE engine." }
        require(executions.isNotEmpty() && executions.distinct().size == executions.size)
        require(searchSeeds.isNotEmpty() && searchSeeds.distinct().size == searchSeeds.size)
        require(trials >= 5 && trials % 5 == 0 && trials <= 1000)
        require(output.fileName != null && output.toString().isNotBlank())
    }
    companion object {
        fun usage() = """
            architectureSearchBenchmark --mode fixed|quality --manifest small|general
              --epochs N (fixed default 2000; quality default 1000000) --engine SMALL|REFERENCE
              --workers 1,4,8,16,32 --backends CPU,CUDA --executions REFERENCE,OPTIMIZED,BATCHED
              --warmups 1 --repeats 3 --order-offset 0 --limit-seconds 0
              --search-seeds 42,123 --trials 60 --output PATH.jsonl
            Each process is one JVM fork; run three forks with rotated order offsets.
            Fixed mode evaluates the same manifest and all five seeds to the requested epoch budget.
            Quality mode uses adaptive SMALL search, validation RMSE <=0.01 and >=4/5 seeds.
            Quality mode runs each selected case/search seed once; warmups/repeats apply only to fixed mode.
            Time limits preserve partial evidence; incomplete candidates cannot qualify.
            GPU is explicit and requires CUDA; no fallback. All output is flushed incrementally.
        """.trimIndent()
        fun parse(args: Array<String>): SearchBenchmarkOptions {
            require(args.size % 2 == 0) { "Every option requires a value; use --help." }
            val values = args.toList().chunked(2).associate { it[0] to it[1] }
            require(values.size * 2 == args.size) { "Duplicate options are not allowed." }
            require(values.keys.all { it in setOf("--mode", "--manifest", "--epochs", "--workers", "--backends", "--executions", "--warmups", "--repeats", "--order-offset", "--limit-seconds", "--search-seeds", "--trials", "--output", "--engine") }) { "Unknown option; use --help." }
            val mode = values["--mode"] ?: "fixed"
            return SearchBenchmarkOptions(mode, values["--manifest"] ?: "small",
                values["--epochs"]?.toInt() ?: if (mode == "quality") 1_000_000 else 2_000,
                values["--workers"]?.split(',')?.map(String::toInt) ?: listOf(4),
                values["--backends"]?.split(',')?.map { TrainingBackend.valueOf(it) } ?: listOf(TrainingBackend.CPU),
                values["--executions"]?.split(',')?.map { ArchitectureExecution.valueOf(it) }
                    ?: listOf(ArchitectureExecution.REFERENCE, ArchitectureExecution.OPTIMIZED),
                values["--warmups"]?.toInt() ?: 1, values["--repeats"]?.toInt() ?: 3,
                values["--order-offset"]?.toInt() ?: 0, values["--limit-seconds"]?.toLong() ?: 0,
                values["--search-seeds"]?.split(',')?.map(String::toLong) ?: listOf(42, 123),
                values["--trials"]?.toInt() ?: 60, values["--output"]?.let(Path::of) ?: Path.of("build/reports/search-benchmark/search.jsonl"),
                values["--engine"]?.let(TrainingEngine::valueOf) ?: TrainingEngine.SMALL)
        }
    }
}

internal object SearchBenchmarkHarness {
    val seeds = listOf(1L, 42L, 123L, 999L, 2026L)
    fun manifest(kind: String): List<NetworkArchitecture> = (if (kind == "small") listOf(
        listOf(8), listOf(8, 8), listOf(8, 8, 8), listOf(8, 8, 8, 8),
        listOf(4, 8), listOf(16, 4), listOf(4, 8, 16), listOf(16, 8, 4, 8))
        else listOf(listOf(3), listOf(6, 6), listOf(7, 5, 9), listOf(12, 6, 3, 5)))
        .map(::NetworkArchitecture)

    fun configuration(options: SearchBenchmarkOptions, workers: Int, backend: TrainingBackend,
                      execution: ArchitectureExecution, searchSeed: Long): ArchitectureSearchConfig = ArchitectureSearchConfig(
        minLayers = 1, maxLayers = 4, minWidth = if (options.manifest == "small") 4 else 1, maxWidth = 16,
        maxParameters = 881, seeds = seeds, maxEpochs = options.epochs, checkEvery = minOf(25, options.epochs),
        targetRmse = 0.01, requiredSuccesses = 4, parallelism = workers,
        maxTrials = if (options.mode == "fixed") manifest(options.manifest).size * seeds.size else options.trials,
        strategy = if (options.mode == "fixed") ArchitectureSearchStrategy.EXHAUSTIVE else ArchitectureSearchStrategy.ADAPTIVE,
        initialHidden = listOf(8, 8, 8), searchSeed = searchSeed, backend = backend,
        precision = Neuro.TrainingPrecision.FP64, batchSize = 1,
        engine = if (options.manifest == "small") options.engine else TrainingEngine.REFERENCE, execution = execution)

    fun run(options: SearchBenchmarkOptions, emit: (Map<String, Any?>) -> Unit) {
        val data = ArchitectureSearchData.split(NeuroLearningSets.create(NeuroLearningSets.Kind.SPIRAL, 42), 0.2, 42, "Spiral")
        val begin = System.nanoTime()
        fun expired() = options.limitSeconds > 0 && (System.nanoTime() - begin) / 1_000_000_000 >= options.limitSeconds
        val cases = options.workers.flatMap { workers -> options.backends.flatMap { backend -> options.executions.map { execution ->
            Triple(workers, backend, execution)
        } } }
        emit(linkedMapOf("type" to "environment", "started" to Instant.now().toString(),
            "sourceRevision" to System.getProperty("jneuro.benchmark.sourceRevision", "unspecified"),
            "java" to System.getProperty("java.runtime.version"), "vm" to System.getProperty("java.vm.name"),
            "os" to System.getProperty("os.name"), "arch" to System.getProperty("os.arch"),
            "osVersion" to System.getProperty("os.version"), "cpuModel" to cpuModel(),
            "processors" to Runtime.getRuntime().availableProcessors(), "pid" to ProcessHandle.current().pid(),
            "heapMaxBytes" to Runtime.getRuntime().maxMemory(), "workers" to options.workers,
            "backends" to options.backends.map { it.name }, "executions" to options.executions.map { it.name },
            "engine" to if (options.manifest == "small") options.engine.name else TrainingEngine.REFERENCE.name,
            "jvmArguments" to ManagementFactory.getRuntimeMXBean().inputArguments,
            "datasetFingerprint" to data.fingerprint, "trainingSamples" to data.training.size,
            "validationSamples" to data.validation.size, "mode" to options.mode, "manifest" to manifest(options.manifest).map { it.hidden },
            "epochsPerSeed" to options.epochs, "seeds" to seeds, "targetRmse" to 0.01, "requiredSuccesses" to 4,
            "batchSize" to 1, "precision" to "FP64", "sigmoid" to "EXACT", "checkEvery" to minOf(25, options.epochs),
            "limitSeconds" to options.limitSeconds, "warmups" to if (options.mode == "quality") 0 else options.warmups,
            "repeats" to if (options.mode == "quality") 1 else options.repeats,
            "orderOffset" to options.orderOffset, "allocationScope" to "JVM total allocated bytes, includes all threads"))
        val measured = ArrayList<Map<String, Any?>>()
        val roundIndices = if (options.mode == "quality") 0..0 else -options.warmups until options.repeats
        for (round in roundIndices) {
            val offset = (options.orderOffset + maxOf(0, round)) % cases.size
            for ((workers, backend, execution) in cases.drop(offset) + cases.take(offset)) {
                for (seed in if (options.mode == "quality") options.searchSeeds else listOf(42L)) {
                    if (expired()) { emit(mapOf("type" to "budget", "status" to "expired", "elapsedNanos" to System.nanoTime() - begin)); return }
                    val config = configuration(options, workers, backend, execution, seed)
                    val name = "${config.engine}/$backend/$execution/workers=$workers/searchSeed=$seed"
                    val allocationStart = allocatedBytes()
                    val start = System.nanoTime()
                    val result = try {
                        val search = NeuroArchitectureSearch()
                        if (options.mode == "fixed") search.searchManifest(data, config, manifest(options.manifest), cancelled = ::expired)
                        else {
                            var published = start
                            search.search(data, config, { progress ->
                                val now = System.nanoTime()
                                if (now - published >= 30_000_000_000L) {
                                    emit(mapOf("type" to "progress", "case" to name, "elapsedNanos" to now - start,
                                        "finishedTrials" to progress.finishedTrials, "activeModels" to progress.running.size,
                                        "maximumEpoch" to progress.running.maxOfOrNull { it.epoch },
                                        "bestObservedRmse" to (progress.running.map { it.bestRmse } +
                                            progress.candidates.flatMap { it.trials }.map { it.bestRmse }).filter(Double::isFinite).minOrNull()))
                                    published = now
                                }
                            }, ::expired)
                        }
                    } catch (failure: Exception) {
                        try { emit(mapOf("type" to "failure", "case" to name, "round" to round, "error" to failure.toString())) }
                        catch (reporting: Exception) { failure.addSuppressed(reporting) }
                        throw failure
                    }
                    val elapsed = System.nanoTime() - start
                    val allocations = allocatedBytes()?.let { end -> allocationStart?.let { end - it } }
                    val record = report(result, name, round, elapsed, allocations)
                    emit(record)
                    if (round >= 0) measured += record
                }
            }
        }
        for ((name, rounds) in measured.groupBy { it.getValue("case") }) {
            val full = rounds.filter { it["workComplete"] == true }
            val times = full.map { (it.getValue("totalNanos") as Long).toDouble() }.sorted()
            emit(mapOf("type" to "summary", "case" to name, "completeRounds" to times.size,
                "medianNanos" to if (times.isEmpty()) null else percentile(times, 0.5),
                "p95Nanos" to if (times.isEmpty()) null else percentile(times, 0.95)))
        }
    }

    fun report(result: ArchitectureSearchResult, name: String, round: Int, elapsed: Long, allocations: Long?): Map<String, Any?> {
        val trials = result.candidates.flatMap { it.trials }
        val winner = result.selection.smallestMeetingTarget
        return linkedMapOf("type" to "round", "case" to name, "round" to round,
            "totalNanos" to elapsed, "searchNanos" to result.elapsedNanos, "allocatedBytes" to allocations,
            "termination" to result.termination.name, "completeTrials" to trials.count { it.state == ArchitectureTrialState.COMPLETED },
            "workComplete" to (trials.size == result.config.maxTrials && result.partial == 0 &&
                result.termination in setOf(ArchitectureTermination.COMPLETED, ArchitectureTermination.TRIAL_BUDGET) &&
                result.candidates.all { it.trials.map { trial -> trial.seed } == result.config.seeds } &&
                trials.all { it.state == ArchitectureTrialState.COMPLETED && it.epochs == result.config.maxEpochs &&
                    it.sampleUpdates == result.config.maxEpochs.toLong() * result.data.training.size }),
            "failedTrials" to result.numericalFailures, "partialCandidates" to result.partial, "evaluatedCandidates" to result.evaluated,
            "committedEpochs" to trials.sumOf { it.epochs.toLong() }, "sampleUpdates" to trials.sumOf { it.sampleUpdates },
            "trialsPerSecond" to trials.count { it.state == ArchitectureTrialState.COMPLETED } * 1e9 / elapsed.coerceAtLeast(1),
            "peakModels" to result.peakParallelTrials, "peakWorkers" to result.peakWorkers,
            "peakResidentModels" to result.peakResidentModels, "gpuBatches" to result.gpuBatches,
            "nativeTrainingCalls" to result.nativeTrainingCalls, "modelsPerBatch" to result.modelsPerBatch,
            "aggregateEpochsPerSecond" to result.aggregateEpochsPerSecond, "budgetPolicy" to result.config.budgetPolicy.name,
            "targetMet" to (winner != null), "winner" to winner?.architecture?.hidden,
            "independentTest" to winner?.representative?.snapshot?.let(::independentScore),
            "candidates" to result.candidates.map { candidate -> mapOf("hidden" to candidate.architecture.hidden,
                "parameters" to candidate.architecture.parameters, "complete" to candidate.fullyEvaluated,
                "successes" to candidate.successes, "medianRmse" to candidate.medianRmse.takeIf(Double::isFinite),
                "trials" to candidate.trials.map { trial -> mapOf("seed" to trial.seed, "state" to trial.state.name,
                    "epochs" to trial.epochs, "bestEpoch" to trial.bestEpoch, "bestRmse" to trial.bestRmse.takeIf(Double::isFinite),
                    "finalRmse" to trial.finalRmse.takeIf(Double::isFinite), "sampleUpdates" to trial.sampleUpdates,
                    "elapsedNanos" to trial.elapsedNanos, "historySize" to trial.history.size,
                    "phaseNanos" to trial.timings?.let { mapOf("open" to it.openNanos, "training" to it.trainingNanos,
                        "scoring" to it.scoringNanos, "snapshot" to it.snapshotNanos, "close" to it.closeNanos) },
                    "kernel" to trial.deviceInfo?.kernelVersion, "simdBits" to trial.deviceInfo?.simdBits,
                    "device" to trial.deviceInfo?.name, "deviceIdentity" to trial.deviceInfo?.identity,
                    "precision" to trial.deviceInfo?.precision, "sigmoid" to trial.deviceInfo?.sigmoid,
                    "engine" to trial.deviceInfo?.engine?.name, "backend" to trial.deviceInfo?.backend?.name,
                    "route" to trial.route.name,
                    "failure" to trial.failure, "parametersAtBest" to trial.snapshot?.parameters()?.toList()) }) })
    }

    /** These points are evaluated only after a reliable winner has been selected. */
    fun independentSpiral(): List<NeuroLearningSets.Sample> = List(218) { index ->
        val target = index / 109
        val fraction = (index % 109 + 0.5) / 109.0
        val radius = 0.05 + 0.43 * fraction
        val angle = target * Math.PI + fraction * Math.PI * 3.25
        NeuroLearningSets.Sample(0.5 + radius * Math.cos(angle), 0.5 + radius * Math.sin(angle), target.toDouble())
    }
    fun independentScore(snapshot: NeuroXorDiagnostics.Snapshot): Map<String, Any> {
        val data = independentSpiral()
        val inputs = DoubleArray(data.size * 2) { index -> if (index % 2 == 0) data[index / 2].x else data[index / 2].y }
        val targets = DoubleArray(data.size) { data[it].target }
        val outputs = snapshot.predictBatch(inputs, data.size)
        var correct = 0
        for (index in data.indices) {
            if ((outputs[index] >= 0.5) == (targets[index] >= 0.5)) correct++
        }
        return mapOf("scope" to "representative winning seed, not five-seed reliability", "samples" to data.size,
            "rmse" to snapshot.error(inputs, targets), "classificationAccuracy" to correct.toDouble() / data.size)
    }
    fun percentile(sorted: List<Double>, fraction: Double): Double {
        require(sorted.isNotEmpty() && fraction in 0.0..1.0)
        if (fraction == 0.5 && sorted.size % 2 == 0) return (sorted[sorted.size / 2 - 1] + sorted[sorted.size / 2]) / 2
        return sorted[(ceil(fraction * sorted.size).toInt() - 1).coerceIn(sorted.indices)]
    }
    private fun allocatedBytes(): Long? = try { (ManagementFactory.getThreadMXBean() as? com.sun.management.ThreadMXBean)?.let {
        if (it.isThreadAllocatedMemorySupported) {
            if (!it.isThreadAllocatedMemoryEnabled) it.isThreadAllocatedMemoryEnabled = true
            it.totalThreadAllocatedBytes.takeIf { value -> value >= 0 }
        } else null
    } } catch (_: UnsupportedOperationException) { null } catch (_: SecurityException) { null }

    private fun cpuModel(): String = try {
        val path = Path.of("/proc/cpuinfo")
        if (Files.isReadable(path)) Files.readAllLines(path).firstOrNull { it.startsWith("model name") }
            ?.substringAfter(':')?.trim() ?: "unspecified"
        else System.getenv("PROCESSOR_IDENTIFIER") ?: "unspecified"
    } catch (_: java.io.IOException) { "unspecified" }
}
