package com.lis.neuro

import java.nio.ByteBuffer
import java.security.MessageDigest
import java.util.SplittableRandom
import java.util.concurrent.ExecutorCompletionService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.abs
import kotlin.math.roundToInt

internal enum class ArchitectureEvaluation(val label: String) {
    TRAINING_FIT("Training RMSE"), VALIDATION("Validation RMSE");
    override fun toString() = label
}

internal enum class ArchitecturePolicy(val label: String) {
    SMALLEST_MEETING_TARGET("Smallest meeting target"), SMALLEST_NEAR_BEST("Smallest near best"), LOWEST_RMSE("Lowest RMSE");
    override fun toString() = label
}

internal enum class ArchitectureTermination { COMPLETED, TRIAL_BUDGET, TIME_LIMIT, CANCELLED, NEIGHBOURHOODS_EXHAUSTED, MEMORY_LIMIT }
internal enum class ArchitectureTrialState { COMPLETED, FAILED, CANCELLED }

internal class NetworkArchitecture(hidden: List<Int>) {
    val hidden: List<Int> = java.util.List.copyOf(hidden)
    private val shape = NeuroTopologyConfig.topology(this.hidden.toIntArray())
    val parameters: Int = NumericInputs.parameterCount(shape)
    fun topology(): IntArray = shape.copyOf()
    override fun equals(other: Any?): Boolean = other is NetworkArchitecture && hidden == other.hidden
    override fun hashCode(): Int = hidden.hashCode()
    override fun toString(): String = NeuroTopologyConfig.label(shape)
}

internal class ArchitectureSearchConfig(
    val minLayers: Int = 1,
    val maxLayers: Int = 3,
    val minWidth: Int = 1,
    val maxWidth: Int = 8,
    val maxParameters: Int = 256,
    seeds: List<Long> = listOf(1, 42, 123, 999, 2026),
    val maxEpochs: Int = 10_000,
    val checkEvery: Int = 25,
    val targetRmse: Double = 0.05,
    val requiredSuccesses: Int = 4,
    val nearBestTolerance: Double = 0.005,
    val policy: ArchitecturePolicy = ArchitecturePolicy.SMALLEST_MEETING_TARGET,
    val hyperParameters: Neuro.HyperParameters = Neuro.HyperParameters(0.6, 0.2, 1.0, 42),
    val parallelism: Int = minOf(4, maxOf(1, Runtime.getRuntime().availableProcessors() - 1)),
    val maxTrials: Int = 10_000,
    val timeLimitSeconds: Long = 0,
    val strategy: ArchitectureSearchStrategy = ArchitectureSearchStrategy.ADAPTIVE,
    initialHidden: List<Int>? = null,
    val searchSeed: Long = 42,
    val restartAfter: Int = 12,
    val maxRestarts: Int = 4,
    val backend: TrainingBackend = TrainingBackend.CPU
) {
    val seeds: List<Long> = java.util.List.copyOf(seeds)
    val initialHidden: List<Int>? = initialHidden?.let { java.util.List.copyOf(it) }
    init {
        require(minLayers > 0 && maxLayers >= minLayers) { "Hidden layer bounds must be positive and ordered." }
        require(minWidth > 0 && maxWidth >= minWidth) { "Width bounds must be positive and ordered." }
        require(maxParameters > 0) { "Parameter budget must be positive." }
        require(this.seeds.isNotEmpty() && this.seeds.distinct().size == this.seeds.size) { "Use a nonempty list of distinct seeds." }
        require(maxEpochs > 0 && checkEvery in 1..maxEpochs) { "Check interval must be positive and within the epoch budget." }
        require(targetRmse.isFinite() && targetRmse >= 0.0) { "Target RMSE must be finite and non-negative." }
        require(requiredSuccesses in 1..this.seeds.size) { "Required successes must be between 1 and the seed count." }
        require(nearBestTolerance.isFinite() && nearBestTolerance >= 0.0) { "Near-best tolerance must be finite and non-negative." }
        require(parallelism > 0) { "Parallelism must be positive." }
        require(maxTrials >= this.seeds.size) { "Trial budget must fit at least one full seed group." }
        require(restartAfter > 0 && maxRestarts >= 0) { "Restart interval must be positive and restart count non-negative." }
        this.initialHidden?.let { NeuroTopologyConfig.topology(it.toIntArray()) }
        require(timeLimitSeconds >= 0) { "Time limit must be non-negative; 0 means unlimited." }
    }

    fun minimumArchitecture(): NetworkArchitecture {
        val base = 4L * minWidth + 1
        val added = (minWidth.toLong() + 1) * minWidth
        require(base <= maxParameters && minLayers.toLong() - 1 <= (maxParameters - base) / added) {
            "No architecture fits these bounds and parameter limit."
        }
        val runtime = Runtime.getRuntime()
        val available = runtime.maxMemory() - runtime.totalMemory() + runtime.freeMemory()
        require((base + (minLayers - 1) * added) * 64 <= available / 2) { "Insufficient JVM heap for the minimum topology." }
        return NetworkArchitecture(List(minLayers) { minWidth })
    }

    fun startingArchitecture(): NetworkArchitecture {
        val minimum = minimumArchitecture()
        val initial = initialHidden?.let { NetworkArchitecture(it) } ?: return minimum
        return initial.takeIf { it.hidden.size in minLayers..maxLayers && it.hidden.all { width -> width in minWidth..maxWidth } &&
            it.parameters <= maxParameters } ?: minimum
    }

    fun plannedTrials(): Int = if (strategy == ArchitectureSearchStrategy.EXHAUSTIVE) {
        minOf(architectures().size, maxTrials / seeds.size) * seeds.size
    } else {
        require(startingArchitecture().parameters.toLong() * seeds.size <= 8_000_000L) { "Initial checkpoint storage exceeds the parameter budget." }
        maxTrials / seeds.size * seeds.size
    }

    fun architectures(): List<NetworkArchitecture> {
        val result = ArrayList<NetworkArchitecture>()
        val widths = ArrayList<Int>()
        var visited = 0
        fun visit(previous: Int, accumulated: Long) {
            check(++visited <= 100_000) { "Search space is too large to enumerate. Narrow the depth, width or parameter limit." }
            if (widths.size >= minLayers && accumulated + previous + 1 <= maxParameters) {
                require(result.size < 4096) { "More than 4,096 architectures. Narrow the search space." }
                result += NetworkArchitecture(widths)
            }
            if (widths.size == maxLayers) return
            for (width in minWidth..maxWidth) {
                val next = accumulated + (previous.toLong() + 1) * width
                if (next + width + 1 > maxParameters) break
                widths += width
                visit(width, next)
                widths.removeAt(widths.lastIndex)
            }
        }
        visit(2, 0L)
        require(result.isNotEmpty()) { "No architecture fits these bounds and parameter limit." }
        val sorted = result.sortedWith(ARCHITECTURE_ORDER)
        val planned = sorted.take(maxTrials / seeds.size)
        require(planned.sumOf { it.parameters.toLong() } * seeds.size <= 8_000_000L) {
            "Checkpoint storage would exceed 8 million parameters. Reduce the space, seeds or trial budget."
        }
        return sorted
    }

    companion object {
        val ARCHITECTURE_ORDER: Comparator<NetworkArchitecture> = compareBy<NetworkArchitecture> { it.parameters }
            .thenBy { it.hidden.size }.thenBy { it.hidden.joinToString(",") { width -> "%03d".format(java.util.Locale.ROOT, width) } }
    }
}

internal class ArchitectureSearchData private constructor(
    training: List<NeuroLearningSets.Sample>,
    validation: List<NeuroLearningSets.Sample>,
    val evaluation: ArchitectureEvaluation,
    val label: String,
    val splitSeed: Long,
    val validationFraction: Double
) {
    val training: List<NeuroLearningSets.Sample> = java.util.List.copyOf(training)
    val validation: List<NeuroLearningSets.Sample> = java.util.List.copyOf(validation)
    val fingerprint: String
    init {
        require(this.training.isNotEmpty()) { "Add training samples before starting a search." }
        require(this.training.size + this.validation.size <= 100_000) { "Search supports at most 100,000 samples." }
        require(evaluation == ArchitectureEvaluation.TRAINING_FIT || this.validation.isNotEmpty()) { "Validation samples are required." }
        val digest = MessageDigest.getInstance("SHA-256")
        digest.update(evaluation.name.toByteArray(Charsets.UTF_8))
        val buffer = ByteBuffer.allocate(24)
        for (group in listOf(this.training, this.validation)) {
            digest.update(ByteBuffer.allocate(4).putInt(group.size).array())
            for (sample in group) {
                buffer.clear()
                buffer.putDouble(sample.x).putDouble(sample.y).putDouble(sample.target)
                digest.update(buffer.array())
            }
        }
        fingerprint = digest.digest().joinToString("") { "%02x".format(java.util.Locale.ROOT, it.toInt() and 255) }
    }

    fun newNetwork(architecture: NetworkArchitecture, parameters: Neuro.HyperParameters, seed: Long): Neuro =
        Neuro(architecture.topology(), parameters.withSeed(seed)).also { model ->
            NeuroLearningSets.addTo(model, training)
            for (sample in validation) model.addTestSample(doubleArrayOf(sample.x, sample.y), doubleArrayOf(sample.target))
        }

    fun score(network: Neuro): Double = if (evaluation == ArchitectureEvaluation.TRAINING_FIT) network.trainingError() else network.testError()

    companion object {
        fun fitting(samples: List<NeuroLearningSets.Sample>, label: String = "Custom") =
            ArchitectureSearchData(samples, emptyList(), ArchitectureEvaluation.TRAINING_FIT, label, 0, 0.0)

        fun split(samples: List<NeuroLearningSets.Sample>, validationFraction: Double = 0.2, seed: Long = 42, label: String = "Custom"): ArchitectureSearchData {
            require(validationFraction.isFinite() && validationFraction > 0.0 && validationFraction < 1.0) { "Validation fraction must be strictly between 0 and 1." }
            val groups = samples.groupBy { coordinate(it) }.values.toList()
            require(groups.size >= 10) { "Validation requires at least 10 distinct input points. Use Training fit for truth tables or tiny datasets." }
            val strata = groups.indices.groupBy { index -> if (groups[index].map { it.target }.average() >= 0.5) 1 else 0 }
            val random = SplittableRandom(seed)
            val heldOut = HashSet<Pair<Double, Double>>()
            for (indices in strata.values) {
                require(indices.size >= 2) { "Each class needs at least two distinct input points for validation." }
                val shuffled = indices.toMutableList()
                for (index in shuffled.lastIndex downTo 1) {
                    val other = random.nextInt(index + 1)
                    val value = shuffled[index]; shuffled[index] = shuffled[other]; shuffled[other] = value
                }
                val count = (shuffled.size * validationFraction).roundToInt().coerceIn(1, shuffled.size - 1)
                for (index in shuffled.take(count)) heldOut += coordinate(groups[index].first())
            }
            return ArchitectureSearchData(samples.filter { coordinate(it) !in heldOut }, samples.filter { coordinate(it) in heldOut },
                ArchitectureEvaluation.VALIDATION, label, seed, validationFraction)
        }

        private fun coordinate(sample: NeuroLearningSets.Sample): Pair<Double, Double> =
            (if (sample.x == 0.0) 0.0 else sample.x) to (if (sample.y == 0.0) 0.0 else sample.y)
    }
}

internal data class ArchitectureCheckpoint(val epoch: Int, val trainingRmse: Double, val score: Double)
internal class ArchitectureTrial(
    val seed: Long,
    val state: ArchitectureTrialState,
    val epochs: Int,
    val bestEpoch: Int,
    val bestRmse: Double,
    val trainingRmseAtBest: Double,
    val finalRmse: Double,
    val sampleUpdates: Long,
    val elapsedNanos: Long,
    history: List<ArchitectureCheckpoint>,
    val snapshot: NeuroXorDiagnostics.Snapshot?,
    val failure: String = "",
    val deviceInfo: TrainingDeviceInfo? = null
) {
    val history: List<ArchitectureCheckpoint> = java.util.List.copyOf(history)
}

internal class ArchitectureCandidate(val architecture: NetworkArchitecture, trials: List<ArchitectureTrial>, val expectedSeeds: Int, val targetRmse: Double) {
    val trials: List<ArchitectureTrial> = java.util.List.copyOf(trials)
    val fullyEvaluated: Boolean = trials.size == expectedSeeds && trials.none { it.state == ArchitectureTrialState.CANCELLED }
    val valid: Boolean = fullyEvaluated && trials.all { it.state == ArchitectureTrialState.COMPLETED && it.bestRmse.isFinite() }
    val successes: Int = trials.count { it.state == ArchitectureTrialState.COMPLETED && it.bestRmse <= targetRmse }
    private val errors = trials.map { if (it.state == ArchitectureTrialState.COMPLETED) it.bestRmse else Double.POSITIVE_INFINITY }.sorted()
    val observedMedianRmse: Double = if (errors.isEmpty()) Double.POSITIVE_INFINITY else median(errors)
    val medianRmse: Double = if (!valid) Double.POSITIVE_INFINITY else observedMedianRmse
    val bestRmse: Double = errors.firstOrNull() ?: Double.POSITIVE_INFINITY
    val worstRmse: Double = errors.lastOrNull() ?: Double.POSITIVE_INFINITY
    val representative: ArchitectureTrial? = if (!valid) null else trials.minWithOrNull(
        compareBy<ArchitectureTrial> { abs(it.bestRmse - medianRmse) }.thenBy { it.bestRmse }.thenBy { it.seed })
    fun meetsTarget(required: Int): Boolean = valid && medianRmse <= targetRmse && successes >= required

    companion object {
        private fun median(values: List<Double>): Double = if (values.size % 2 == 1) values[values.size / 2]
            else values[values.size / 2 - 1] / 2.0 + values[values.size / 2] / 2.0
    }
}

internal data class ArchitectureSelection(
    val recommended: ArchitectureCandidate?,
    val bestError: ArchitectureCandidate?,
    val smallestMeetingTarget: ArchitectureCandidate?,
    val paretoFrontier: List<ArchitectureCandidate>,
    val reliableFrontier: List<ArchitectureCandidate>
)

internal object ArchitectureRanking {
    fun select(candidates: List<ArchitectureCandidate>, config: ArchitectureSearchConfig, policy: ArchitecturePolicy = config.policy): ArchitectureSelection {
        val valid = candidates.filter { it.valid }
        val bySize = compareBy<ArchitectureCandidate> { it.architecture.parameters }.thenBy { it.medianRmse }
            .thenComparator { a, b -> ArchitectureSearchConfig.ARCHITECTURE_ORDER.compare(a.architecture, b.architecture) }
        val byError = compareBy<ArchitectureCandidate> { it.medianRmse }.then(bySize)
        val best = valid.minWithOrNull(byError)
        val reliable = valid.filter { it.meetsTarget(config.requiredSuccesses) }
        val smallest = reliable.minWithOrNull(bySize)
        val recommended = when (policy) {
            ArchitecturePolicy.SMALLEST_MEETING_TARGET -> smallest
            ArchitecturePolicy.LOWEST_RMSE -> best
            ArchitecturePolicy.SMALLEST_NEAR_BEST -> best?.let { winner ->
                valid.filter { it.medianRmse <= winner.medianRmse + config.nearBestTolerance }.minWithOrNull(bySize)
            }
        }
        return ArchitectureSelection(recommended, best, smallest, frontier(valid), frontier(reliable))
    }

    fun frontier(candidates: List<ArchitectureCandidate>): List<ArchitectureCandidate> {
        val sorted = candidates.filter { it.valid }.sortedWith(compareBy<ArchitectureCandidate> { it.architecture.parameters }
            .thenBy { it.medianRmse }.thenBy { it.architecture.toString() })
        val frontier = ArrayList<ArchitectureCandidate>()
        var bestError = Double.POSITIVE_INFINITY
        for (group in sorted.groupBy { it.architecture.parameters }.values) {
            val error = group.first().medianRmse
            if (error < bestError) frontier.addAll(group.takeWhile { it.medianRmse == error })
            bestError = minOf(bestError, error)
        }
        return java.util.List.copyOf(frontier)
    }
}

internal data class ArchitectureRunningTrial(val architecture: NetworkArchitecture, val seed: Long, val epoch: Int, val bestRmse: Double)
internal class ArchitectureSearchProgress(
    val generated: Int,
    val plannedTrials: Int,
    val finishedTrials: Int,
    candidates: List<ArchitectureCandidate>,
    running: List<ArchitectureRunningTrial>,
    val elapsedNanos: Long,
    lineage: List<ArchitectureProposal> = emptyList(),
    val peakParallelTrials: Int = 0
) {
    val lineage: List<ArchitectureProposal> = java.util.List.copyOf(lineage)
    val candidates: List<ArchitectureCandidate> = java.util.List.copyOf(candidates)
    val running: List<ArchitectureRunningTrial> = java.util.List.copyOf(running)
    val fullyEvaluated: Int get() = candidates.count { it.fullyEvaluated }
}

internal class ArchitectureSearchResult(
    val data: ArchitectureSearchData,
    val config: ArchitectureSearchConfig,
    val termination: ArchitectureTermination,
    val generated: Int,
    candidates: List<ArchitectureCandidate>,
    val elapsedNanos: Long,
    lineage: List<ArchitectureProposal> = emptyList(),
    val peakParallelTrials: Int = 0
) {
    val lineage: List<ArchitectureProposal> = java.util.List.copyOf(lineage)
    val candidates: List<ArchitectureCandidate> = java.util.List.copyOf(candidates)
    val evaluated: Int get() = candidates.count { it.fullyEvaluated }
    val untested: Int get() = generated - candidates.size
    val partial: Int get() = candidates.count { !it.fullyEvaluated }
    val numericalFailures: Int get() = candidates.sumOf { candidate -> candidate.trials.count { it.state == ArchitectureTrialState.FAILED } }
    val selection: ArchitectureSelection = ArchitectureRanking.select(this.candidates, config)
    val environment: String = "Kotlin ${KotlinVersion.CURRENT}; JVM ${System.getProperty("java.version")}; ${System.getProperty("os.name")}/${System.getProperty("os.arch")}"
}

internal fun interface ArchitectureSearcher {
    fun search(data: ArchitectureSearchData, config: ArchitectureSearchConfig,
               onProgress: (ArchitectureSearchProgress) -> Unit, cancelled: () -> Boolean): ArchitectureSearchResult
}

internal class NeuroArchitectureSearch(
    private val openSession: (Neuro, TrainingBackend) -> NeuroTrainingSession = { model, backend -> model.newTrainingSession(backend) }
) : ArchitectureSearcher {
    override fun search(data: ArchitectureSearchData, config: ArchitectureSearchConfig,
                        onProgress: (ArchitectureSearchProgress) -> Unit, cancelled: () -> Boolean): ArchitectureSearchResult {
        return if (config.strategy == ArchitectureSearchStrategy.ADAPTIVE) searchAdaptive(data, config, onProgress, cancelled)
        else searchExhaustive(data, config, onProgress, cancelled)
    }

    private fun searchExhaustive(data: ArchitectureSearchData, config: ArchitectureSearchConfig,
                                 onProgress: (ArchitectureSearchProgress) -> Unit, cancelled: () -> Boolean): ArchitectureSearchResult {
        val architectures = config.architectures()
        val planned = architectures.take(config.maxTrials / config.seeds.size)
        val requests = planned.flatMap { architecture -> config.seeds.map { architecture to it } }
        val start = System.nanoTime()
        val stopping = AtomicBoolean(false)
        val activity = TrialActivity()
        val workerId = AtomicInteger()
        val pool = Executors.newFixedThreadPool(config.parallelism) { runnable ->
            Thread(runnable, "jneuro-search-${workerId.incrementAndGet()}").apply { isDaemon = true }
        }
        data class Finished(val architecture: NetworkArchitecture, val trial: ArchitectureTrial)
        val completions = ExecutorCompletionService<Finished>(pool)
        val trials = LinkedHashMap<NetworkArchitecture, MutableMap<Long, ArchitectureTrial>>()
        var submitted = 0
        var received = 0
        var lastPublish = Long.MIN_VALUE
        var termination: ArchitectureTermination? = null
        fun candidates(): List<ArchitectureCandidate> = trials.entries.map { (architecture, values) ->
            ArchitectureCandidate(architecture, config.seeds.mapNotNull { values[it] }, config.seeds.size, config.targetRmse)
        }.sortedWith { a, b -> ArchitectureSearchConfig.ARCHITECTURE_ORDER.compare(a.architecture, b.architecture) }
        fun publish(force: Boolean = false) {
            val now = System.nanoTime()
            if (force || lastPublish == Long.MIN_VALUE || now - lastPublish >= 100_000_000L) {
                onProgress(ArchitectureSearchProgress(architectures.size, requests.size, received, candidates(),
                    activity.snapshot(), now - start, peakParallelTrials = activity.peak))
                lastPublish = now
            }
        }
        fun checkStop() {
            if (termination != null) return
            if (cancelled() || Thread.currentThread().isInterrupted) termination = ArchitectureTermination.CANCELLED
            else if (config.timeLimitSeconds > 0 && (System.nanoTime() - start) / 1_000_000_000L >= config.timeLimitSeconds) {
                termination = ArchitectureTermination.TIME_LIMIT
            }
            if (termination != null) stopping.set(true)
        }
        try {
            publish(true)
            while (received < submitted || submitted < requests.size && termination == null) {
                checkStop()
                while (termination == null && submitted < requests.size && submitted - received < config.parallelism) {
                    val index = submitted++
                    val (architecture, seed) = requests[index]
                    completions.submit {
                        activity.track(index, architecture, seed) { progress ->
                            Finished(architecture, evaluate(data, config, architecture, seed,
                                { stopping.get() || Thread.currentThread().isInterrupted }, progress))
                        }
                    }
                }
                if (received < submitted) {
                    val completed = completions.poll(50, TimeUnit.MILLISECONDS)
                    if (completed != null) {
                        val finished = completed.get()
                        trials.getOrPut(finished.architecture) { LinkedHashMap() }[finished.trial.seed] = finished.trial
                        received++
                    }
                }
                publish()
            }
            checkStop()
            val end = termination ?: if (planned.size < architectures.size) ArchitectureTermination.TRIAL_BUDGET else ArchitectureTermination.COMPLETED
            publish(true)
            return ArchitectureSearchResult(data, config, end, architectures.size, candidates(), System.nanoTime() - start, peakParallelTrials = activity.peak)
        } finally {
            stopping.set(true)
            pool.shutdownNow()
            try { pool.awaitTermination(5, TimeUnit.SECONDS) } catch (_: InterruptedException) { Thread.currentThread().interrupt() }
        }
    }

    private fun searchAdaptive(data: ArchitectureSearchData, config: ArchitectureSearchConfig,
                               onProgress: (ArchitectureSearchProgress) -> Unit, cancelled: () -> Boolean): ArchitectureSearchResult =
        AdaptiveTrialScheduler(data, config) { architecture, seed, stop, progress ->
            evaluate(data, config, architecture, seed, stop, progress)
        }.search(onProgress, cancelled)

    fun search(data: ArchitectureSearchData, config: ArchitectureSearchConfig): ArchitectureSearchResult = search(data, config, {}, { false })

    internal fun evaluate(data: ArchitectureSearchData, config: ArchitectureSearchConfig, architecture: NetworkArchitecture, seed: Long,
                          cancelled: () -> Boolean, progress: (Int, Double) -> Unit): ArchitectureTrial {
        val start = System.nanoTime()
        val history = ArrayList<ArchitectureCheckpoint>()
        var epoch = 0
        var bestEpoch = 0
        var best = Double.POSITIVE_INFINITY
        var trainingAtBest = Double.POSITIVE_INFINITY
        var finalScore = Double.POSITIVE_INFINITY
        var bestSnapshot: NeuroXorDiagnostics.Snapshot? = null
        var deviceInfo: TrainingDeviceInfo? = null
        fun result(state: ArchitectureTrialState, message: String = "") = ArchitectureTrial(seed, state, epoch, bestEpoch, best,
            trainingAtBest, finalScore, epoch.toLong() * data.training.size, System.nanoTime() - start, history, bestSnapshot, message, deviceInfo)
        if (cancelled()) return result(ArchitectureTrialState.CANCELLED)
        return try {
            val model = data.newNetwork(architecture, config.hyperParameters, seed)
            fun checkPoint() {
                val training = model.trainingError()
                finalScore = if (data.evaluation == ArchitectureEvaluation.TRAINING_FIT) training else data.score(model)
                check(training.isFinite() && finalScore.isFinite()) { "Training produced a non-finite RMSE." }
                if (finalScore < best) {
                    best = finalScore; bestEpoch = epoch; trainingAtBest = training
                    bestSnapshot = NeuroXorDiagnostics.capture(model, epoch, training)
                }
                history += ArchitectureCheckpoint(epoch, training, finalScore)
                if (history.size > 128) {
                    val retained = history.filterIndexed { index, point -> index == 0 || index % 2 == 0 || index == history.lastIndex || point.epoch == bestEpoch }
                    history.clear(); history.addAll(retained)
                }
                progress(epoch, best)
            }
            openSession(model, config.backend).use { session ->
                deviceInfo = session.info
                checkPoint()
                while (epoch < config.maxEpochs) {
                    if (cancelled()) return result(ArchitectureTrialState.CANCELLED)
                    val training = session.trainEpoch()
                    epoch++
                    check(training.isFinite()) { "Training produced a non-finite RMSE." }
                    if (epoch % config.checkEvery == 0 || epoch == config.maxEpochs) checkPoint()
                }
                result(ArchitectureTrialState.COMPLETED)
            }
        } catch (exception: Exception) {
            if (exception is InterruptedException) { Thread.currentThread().interrupt(); result(ArchitectureTrialState.CANCELLED) }
            else result(ArchitectureTrialState.FAILED, exception.message ?: exception.javaClass.simpleName)
        }
    }
}

internal class ArchitectureSearchSession {
    class Token internal constructor(val id: Long) { val cancelled = AtomicBoolean(false) }
    private var next = 0L
    @Volatile private var active: Token? = null
    @Synchronized fun begin(): Token {
        active?.cancelled?.set(true)
        return Token(++next).also { active = it }
    }
    fun cancel() { active?.cancelled?.set(true) }
    @Synchronized fun invalidate() { cancel(); active = null }
    fun isCurrent(token: Token): Boolean = active === token
}
