package com.lis.neuro

import java.nio.file.Files
import java.nio.file.Path

/** One bounded JVM fork; all timings include model initialization, TensorFlow opening, scoring and cleanup. */
object PopulationBenchmark {
    private val seeds = listOf(1L, 42L, 123L, 999L, 2026L)
    private val shapes = listOf(listOf(2), listOf(4), listOf(6), listOf(8), listOf(4, 4), listOf(8, 4))
    private val data = ArchitectureSearchData.split(NeuroLearningSets.create(NeuroLearningSets.Kind.SPIRAL, 42), 0.2, 42, "Spiral")

    @JvmStatic fun main(args: Array<String>) {
        require(args.size == 2) { "PopulationBenchmark legacy|batched|quality-full|quality-prune output.jsonl" }
        val mode = args[0]
        require(mode in setOf("legacy", "batched", "quality-full", "quality-prune"))
        val path = Path.of(args[1]); Files.createDirectories(path.toAbsolutePath().parent)
        val execution = ArchitectureExecution.valueOf(if (mode == "legacy") "OPTIMIZED" else "BATCHED")
        val warm = settings(execution, "FULL", epochs = 3, trials = 5, models = 5)
        NeuroArchitectureSearch().searchManifest(data, warm, listOf(NetworkArchitecture(listOf(6))))
        Files.newBufferedWriter(path).use { writer ->
            fun record(value: Map<String, Any?>) {
                writer.write(NeuroCudaBenchmarkReports.encode(value)); writer.newLine(); writer.flush()
                println(NeuroCudaBenchmarkReports.encode(value.filterKeys { it in setOf("case", "round", "elapsedMs", "firstEligibleMs", "firstDiverseMs", "aggregateEpochsPerSecond", "bestMedianRmse") }))
            }
            record(mapOf("case" to "environment", "mode" to mode, "jvm" to System.getProperty("java.version"),
                "os" to System.getProperty("os.name"), "cpuCount" to Runtime.getRuntime().availableProcessors(),
                "trainingSamples" to data.training.size, "validationSamples" to data.validation.size,
                "datasetFingerprint" to data.fingerprint, "seedSet" to seeds, "precision" to "FP64", "sigmoid" to "EXACT",
                "learningRate" to 0.6, "momentum" to 0.2, "beta" to 1.0, "sampleBatch" to 1,
                "warmupEpochs" to 3, "legacyWorkers" to 5))
            if (mode.startsWith("quality")) {
                val policy = if (mode == "quality-prune") "SUCCESSIVE_HALVING" else "FULL"
                for (searchSeed in listOf(42L, 123L)) {
                    val setup = settings(execution, policy, epochs = 150, trials = 60, models = 30, adaptive = true, searchSeed = searchSeed)
                    record(run("quality", searchSeed.toInt(), setup, null))
                }
            } else repeat(2) { round ->
                record(run("five-seeds", round, settings(execution, "FULL", 50, 5, 5), listOf(NetworkArchitecture(listOf(6)))))
                record(run("mixed-population", round, settings(execution, "FULL", 50, 30, 30), shapes.map(::NetworkArchitecture)))
            }
        }
        TensorFlowMath.clearInferenceCache()
    }

    private fun settings(execution: ArchitectureExecution, policy: String, epochs: Int,
                         trials: Int, models: Int, adaptive: Boolean = false, searchSeed: Long = 42): ArchitectureSearchConfig {
        // The original26 arguments are unchanged. Reflection keeps this exact harness binary compatible with PR64.
        val args = mutableListOf<Any?>(1, 2, 1, 8, 256, seeds, epochs, minOf(25, epochs), 0.05, 4, 0.005,
            ArchitecturePolicy.SMALLEST_MEETING_TARGET, Neuro.HyperParameters(0.6, 0.2, 1.0, 42), 5, trials, 0L,
            if (adaptive) ArchitectureSearchStrategy.ADAPTIVE else ArchitectureSearchStrategy.EXHAUSTIVE,
            listOf(6), searchSeed, 12, 4, TrainingBackend.CPU, Neuro.TrainingPrecision.FP64, 1, TrainingEngine.REFERENCE, execution)
        val constructor = ArchitectureSearchConfig::class.java.constructors.single { it.parameterCount in setOf(26, 30) }
        if (constructor.parameterCount == 30) {
            args += constructor.parameterTypes[26].enumConstants.single { (it as Enum<*>).name == policy }
            args.addAll(listOf(3, 25, models))
        } else check(policy == "FULL")
        return constructor.newInstance(*args.toTypedArray()) as ArchitectureSearchConfig
    }

    private fun optional(target: Any, name: String): Any? = target.javaClass.methods.firstOrNull { it.name == name && it.parameterCount == 0 }?.invoke(target)

    private fun run(name: String, round: Int, config: ArchitectureSearchConfig,
                    manifest: List<NetworkArchitecture>?): Map<String, Any?> {
        var firstEligible: Long? = null
        var firstDiverse: Long? = null
        var firstPruned: Long? = null
        var peakArchitectures = 0
        val start = System.nanoTime()
        val progress: (ArchitectureSearchProgress) -> Unit = { update ->
            val elapsed = System.nanoTime() - start
            val active = update.running.map { it.architecture }.distinct().size
            peakArchitectures = maxOf(peakArchitectures, active)
            if (firstDiverse == null && active > 1) firstDiverse = elapsed
            if (firstEligible == null && update.candidates.any { it.valid }) firstEligible = elapsed
            if (firstPruned == null && update.candidates.any { it.trials.any { trial -> trial.state.name == "PRUNED" } }) firstPruned = elapsed
        }
        val engine = NeuroArchitectureSearch()
        val report = if (manifest == null) engine.search(data, config, progress, { false })
            else engine.searchManifest(data, config, manifest, progress, { false })
        val elapsed = System.nanoTime() - start
        check(report.numericalFailures == 0)
        val trials = report.candidates.flatMap { it.trials }
        val policy = (optional(config, "getBudgetPolicy") as? Enum<*>)?.name ?: "FULL"
        if (policy == "FULL")
            check(trials.all { it.state == ArchitectureTrialState.COMPLETED && it.epochs == config.maxEpochs })
        return mapOf("case" to name, "round" to round, "execution" to config.execution.name,
            "budgetPolicy" to policy, "epochCap" to config.maxEpochs, "modelsPerBatchLimit" to optional(config, "getModelsPerBatch"),
            "elapsedMs" to elapsed / 1_000_000.0, "firstEligibleMs" to firstEligible?.div(1_000_000.0),
            "firstDiverseMs" to firstDiverse?.div(1_000_000.0), "firstPrunedMs" to firstPruned?.div(1_000_000.0),
            "peakActiveArchitectures" to peakArchitectures, "architecturesSampled" to report.generated,
            "totalCommittedEpochs" to trials.sumOf { it.epochs.toLong() },
            "aggregateEpochsPerSecond" to trials.sumOf { it.epochs.toLong() } * 1_000_000_000.0 / elapsed,
            "nativeTrainingCalls" to optional(report, "getNativeTrainingCalls"), "actualMaximumBatch" to optional(report, "getModelsPerBatch"),
            "validArchitectures" to report.candidates.count { it.valid },
            "prunedTrials" to trials.count { it.state.name == "PRUNED" },
            "bestMedianRmse" to report.selection.bestError?.medianRmse,
            "targetMet" to (report.selection.recommended != null), "termination" to report.termination.name,
            "trials" to report.candidates.flatMap { candidate -> candidate.trials.map { trial ->
                mapOf("shape" to candidate.architecture.topology().toList(), "seed" to trial.seed, "state" to trial.state.name,
                    "epochs" to trial.epochs, "bestEpoch" to trial.bestEpoch, "bestRmse" to trial.bestRmse,
                    "finalRmse" to trial.finalRmse, "parameters" to trial.snapshot?.parameters()?.toList(),
                    "kernel" to trial.deviceInfo?.kernelVersion, "rebinds" to maxOf(0, ((optional(trial, "getBatchSegments") as? List<*>)?.size ?: 0) - 1))
            } })
    }
}
