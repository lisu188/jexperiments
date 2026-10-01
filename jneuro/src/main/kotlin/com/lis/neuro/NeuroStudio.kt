package com.lis.neuro

import java.awt.image.BufferedImage
import java.awt.image.DataBufferInt
import kotlin.math.sqrt

internal data class StudioConfig(
    val hidden: String = "6",
    val dataset: NeuroLearningSets.Kind = NeuroLearningSets.Kind.XOR,
    val seed: Long = 42,
    val maxEpochs: Int = 10_000,
    val targetError: Double = 0.05,
    val learningRate: Double = 0.6,
    val momentum: Double = 0.2,
    val backend: TrainingBackend = TrainingBackend.CPU,
    val precision: Neuro.TrainingPrecision = Neuro.TrainingPrecision.FP64,
    val batchSize: Int = 1,
    val engine: TrainingEngine = TrainingEngine.REFERENCE,
    val sigmoid: Neuro.SigmoidMode = Neuro.SigmoidMode.EXACT
) {
    init {
        val layers = NeuroTopologyConfig.parseHidden(hidden)
        require(engine != TrainingEngine.SMALL || SmallNetworkShape.supports(NeuroTopologyConfig.topology(layers))) {
            "SMALL requires 1–4 hidden layers of width 4, 8 or 16; select REFERENCE for other topologies."
        }
        require(maxEpochs > 0) { "Epoch limit must be positive." }
        require(targetError.isFinite() && targetError >= 0.0) { "Target RMSE must be finite and non-negative." }
        Neuro.HyperParameters(learningRate, momentum, 1.0, seed)
        require(batchSize > 0) { "Batch size must be positive." }
    }
    fun topology(): IntArray = NeuroTopologyConfig.topology(NeuroTopologyConfig.parseHidden(hidden))
    fun description(): String = NeuroTopologyConfig.label(topology())
}

internal enum class StudioState(val label: String) {
    READY("Ready"), RUNNING("Training"), PAUSED("Paused"), CONVERGED("Target reached"),
    LIMIT_REACHED("Epoch limit reached"), EMPTY("Add training points"), FAILED("Training failed")
}

internal class StudioHistory(
    val epoch: Int,
    val error: Double,
    predictions: DoubleArray,
    parameters: DoubleArray,
    norms: DoubleArray
) {
    private val predictions = predictions.copyOf()
    private val parameters = parameters.copyOf()
    private val norms = norms.copyOf()
    val parameterCount: Int get() = parameters.size
    val normCount: Int get() = norms.size
    fun prediction(index: Int): Double = predictions[index]
    fun parameter(index: Int): Double = parameters[index]
    fun norm(index: Int): Double = norms[index]
}

internal data class StudioCheckpoint(val epoch: Int, val error: Double, val image: BufferedImage)
internal data class StudioSeed(val seed: Long, val epochs: Int, val error: Double, val converged: Boolean, val image: BufferedImage)
internal data class StudioFrame(
    val config: StudioConfig,
    val state: StudioState,
    val diagnostics: NeuroXorDiagnostics.Snapshot,
    val samples: List<NeuroLearningSets.Sample>,
    val image: BufferedImage,
    val beforeImage: BufferedImage,
    val beforeEpoch: Int,
    val differenceImage: BufferedImage,
    val hiddenImages: List<BufferedImage>,
    val hiddenLayer: Int,
    val hiddenStart: Int,
    val history: List<StudioHistory>,
    val checkpoints: List<StudioCheckpoint>,
    val seeds: List<StudioSeed>,
    val message: String = "",
    val replayNote: String = "",
    val deviceInfo: TrainingDeviceInfo? = null
)

/** closeTraining() releases a resumable session; AutoCloseable.close() ends this Studio and its owned service. */
internal class NeuroStudio(
    config: StudioConfig = StudioConfig(), custom: List<NeuroLearningSets.Sample> = emptyList(),
    private val windowId: String? = null,
    private val openCohort: ((List<Neuro>, TrainingBackend, Neuro.TrainingPrecision, Int, Int) -> NeuroTrainingCohort)? = null,
    private val openSearchCuda: ((Neuro.TrainingPrecision, Int) -> SearchCudaService)? = null,
    private val openSession: ((Neuro, TrainingBackend, Neuro.TrainingPrecision, Int, TrainingEngine) -> NeuroTrainingSession)? = null
) : AutoCloseable {
    private val ownedDeviceService = lazy { NeuroTrainingDeviceService() }
    private var closed = false
    private fun ensureOpen() { check(!closed) { "Studio is closed; create a new Studio to resume training." } }
    private fun openTrainingSession(model: Neuro, backend: TrainingBackend, precision: Neuro.TrainingPrecision,
                                    batch: Int, engine: TrainingEngine): NeuroTrainingSession {
        ensureOpen()
        return openSession?.invoke(model, backend, precision, batch, engine)
            ?: ownedDeviceService.value.openSession(model, backend, precision, batch, engine)
    }

    private fun openTrainingCohort(models: List<Neuro>, backend: TrainingBackend, precision: Neuro.TrainingPrecision,
                                   batch: Int, parallelism: Int): NeuroTrainingCohort {
        ensureOpen()
        return openCohort?.invoke(models, backend, precision, batch, parallelism)
            ?: ownedDeviceService.value.openCohort(models, backend, precision, batch, parallelism)
    }

    private var runId = NeuroLog.id("run")
    private var config = config
    private var custom = custom.toList()
    private var samples = samplesFor(config, custom)
    private var network = createNetwork(config, config.seed, samples)
    private var session: NeuroTrainingSession? = null
    private var deviceInfo: TrainingDeviceInfo? = null
    private var epoch = 0
    private var error = network.trainingError()
    private var automatic = false
    private var pendingEpochs = 0
    private var stepped = false
    private val history = ArrayList<StudioHistory>()
    private val checkpoints = ArrayList<StudioCheckpoint>()
    private var seedResults: List<StudioSeed> = emptyList()
    private var selectedLayer = 0
    private var hiddenStart = 0
    private var render: Render? = null
    private var previousEpoch = 0
    private var previousImage: BufferedImage? = null
    private var previousValues: DoubleArray? = null
    private var difference: BufferedImage? = null
    private var layerImages: List<BufferedImage>? = null
    private var failure = ""
    private var replayNote = ""
    private val historyCapacity: Int get() = minOf(1024, maxOf(8, 2_000_000 / network.parameterCount()))

    private data class Render(val diagnostics: NeuroXorDiagnostics.Snapshot, val image: BufferedImage, val values: DoubleArray)

    val activeConfig: StudioConfig get() = config
    val epochs: Int get() = epoch
    val currentError: Double get() = error
    val hasWork: Boolean get() = state == StudioState.RUNNING || pendingEpochs > 0 && canTrain()
    val state: StudioState get() = when {
        failure.isNotEmpty() -> StudioState.FAILED
        samples.isEmpty() -> StudioState.EMPTY
        error <= config.targetError -> StudioState.CONVERGED
        epoch >= config.maxEpochs -> StudioState.LIMIT_REACHED
        automatic -> StudioState.RUNNING
        stepped || epoch > 0 -> StudioState.PAUSED
        else -> StudioState.READY
    }

    init { logRun("studio.run.created") }

    private fun logRun(event: String) {
        NeuroLog.info("studio", event, "windowId" to windowId, "runId" to runId, "model" to network.logId, "topology" to config.description(),
            "dataset" to config.dataset, "samples" to samples.size, "seed" to config.seed,
            "requestedBackend" to config.backend, "requestedPrecision" to config.precision, "engine" to config.engine, "sigmoid" to config.sigmoid,
            "batchSize" to config.batchSize, "maxEpochs" to config.maxEpochs, "targetRmse" to config.targetError, "automatic" to automatic)
    }

    fun apply(next: StudioConfig, run: Boolean = false) {
        ensureOpen()
        val nextSamples = samplesFor(next, custom)
        val nextNetwork = createNetwork(next, next.seed, nextSamples)
        closeTraining()
        deviceInfo = null
        runId = NeuroLog.id("run")
        config = next
        samples = nextSamples
        network = nextNetwork
        epoch = 0
        error = network.trainingError()
        automatic = run && samples.isNotEmpty()
        pendingEpochs = 0
        stepped = false
        failure = ""
        replayNote = ""
        history.clear()
        checkpoints.clear()
        seedResults = emptyList()
        selectedLayer = 0
        hiddenStart = 0
        render = null
        previousEpoch = 0
        previousImage = null
        previousValues = null
        difference = null
        layerImages = null
        logRun("studio.configuration.applied")
    }

    fun setRunning(value: Boolean) {
        ensureOpen()
        automatic = value && canTrain()
        if (!value) pendingEpochs = 0
        NeuroLog.info("studio", if (automatic) "studio.training.started" else "studio.training.paused",
            "runId" to runId, "epoch" to epoch, "state" to state)
    }

    fun step(count: Int) {
        ensureOpen()
        require(count > 0) { "Step count must be positive." }
        automatic = false
        if (!canTrain()) { pendingEpochs = 0; return }
        pendingEpochs = minOf((config.maxEpochs - epoch).toLong(), pendingEpochs.toLong() + count).toInt()
        stepped = true
        NeuroLog.info("studio", "studio.step.requested", "runId" to runId, "epochs" to count, "pendingEpochs" to pendingEpochs)
    }

    fun addSample(x: Double, y: Double, target: Double) {
        require(config.dataset == NeuroLearningSets.Kind.CUSTOM) { "Select Custom before editing training points." }
        require(custom.size < 4096) { "Custom datasets support at most 4096 points." }
        custom = custom + NeuroLearningSets.Sample(x, y, target)
        NeuroLog.info("studio", "studio.sample.added", "runId" to runId, "samples" to custom.size)
        apply(config, false)
    }

    fun undoSample() {
        require(config.dataset == NeuroLearningSets.Kind.CUSTOM) { "Select Custom before editing training points." }
        custom = custom.dropLast(1)
        NeuroLog.info("studio", "studio.sample.undone", "runId" to runId, "samples" to custom.size)
        apply(config, false)
    }

    fun clearSamples() {
        custom = emptyList()
        NeuroLog.info("studio", "studio.samples.cleared", "runId" to runId)
        apply(config.copy(dataset = NeuroLearningSets.Kind.CUSTOM), false)
    }

    fun selectHidden(layer: Int, start: Int = 0) {
        val topology = network.topology()
        require(layer in 0 until topology.size - 2) { "Hidden layer is out of range." }
        require(start in 0 until topology[layer + 1]) { "Neuron offset is out of range." }
        selectedLayer = layer
        hiddenStart = start
        layerImages = null
    }

    fun advance(speed: Int = 10, cancelled: () -> Boolean = { false }): Int {
        ensureOpen()
        require(speed > 0) { "Speed must be positive." }
        if (!hasWork || cancelled()) return 0
        ensureRender()
        val before = render!!
        previousEpoch = epoch
        previousImage = before.image
        previousValues = before.values
        val requested = if (pendingEpochs > 0) minOf(speed, pendingEpochs) else speed
        var advanced = 0
        val started = System.nanoTime()
        while (advanced < requested && canTrain() && !cancelled()) {
            var termination = TrainingTermination.COMPLETED
            val committed = try {
                if (config.engine == TrainingEngine.SMALL) {
                    val remainingNanos = 25_000_000L - (System.nanoTime() - started)
                    if (advanced > 0 && remainingNanos <= 0) break
                    val milestone = MILESTONES.firstOrNull { it > epoch } ?: config.maxEpochs
                    val count = minOf(requested - advanced, config.maxEpochs - epoch, milestone - epoch)
                    val result = trainingSession().trainChunk(TrainingChunkRequest(count, targetError = config.targetError,
                        checkEvery = 1, cancelled = cancelled, maxNanos = maxOf(1L, remainingNanos)))
                    error = result.rmse
                    termination = result.termination
                    check(result.committedEpochs in 0..count) { "Training chunk exceeded its requested epoch boundary." }
                    result.committedEpochs
                } else {
                    error = trainConfiguredEpoch(network, trainingSession(), config.batchSize)
                    1
                }
            } catch (exception: Exception) {
                NeuroLog.error("studio", "studio.epoch.failed", exception, "runId" to runId, "epoch" to epoch)
                throw exception
            }
            check(error.isFinite()) { "Training produced a non-finite RMSE. Reset with different settings." }
            epoch += committed
            advanced += committed
            if (pendingEpochs > 0) pendingEpochs -= committed
            if (committed > 0 && (epoch in MILESTONES || !canTrain())) captureCheckpoint()
            if (committed == 0 || termination in setOf(TrainingTermination.CANCELLED, TrainingTermination.BUDGET)) break
        }
        if (advanced > 0) {
            render = null
            layerImages = null
            difference = null
        }
        if (!canTrain()) {
            automatic = false; pendingEpochs = 0
            NeuroLog.info("studio", "studio.training.completed", "runId" to runId, "epoch" to epoch, "rmse" to error, "state" to state)
        }
        NeuroLog.debug("studio", "studio.training.progress") { mapOf("runId" to runId, "epoch" to epoch, "advanced" to advanced, "rmse" to error) }
        return advanced
    }

    fun compareSeeds(limit: Int = config.maxEpochs, cancelled: () -> Boolean = { false }): List<StudioSeed> {
        ensureOpen()
        require(limit > 0) { "Seed study epoch limit must be positive." }
        if (samples.isEmpty()) return emptyList()
        closeTraining()
        val studyId = NeuroLog.id("study")
        NeuroLog.info("studio", "studio.study.started", "runId" to runId, "studyId" to studyId, "maxEpochs" to limit, "backend" to config.backend)
        fun isCancelled(): Boolean = cancelled().also { stopped ->
            if (stopped) NeuroLog.info("studio", "studio.study.cancelled", "runId" to runId, "studyId" to studyId)
        }
        try {
            val results = ArrayList<StudioSeed>()
            for (seed in STUDY_SEEDS) {
                if (isCancelled()) return emptyList()
                val model = createNetwork(config, seed, samples)
                var trained = 0
                var rmse = model.trainingError()
                openTrainingSession(model, config.backend, config.precision, config.batchSize, config.engine).use { training ->
                    NeuroLog.info("studio", "studio.study.seed.started", "runId" to runId, "studyId" to studyId,
                        "seed" to seed, "model" to model.logId, "session" to model.trainingSessionLogId,
                        "effectiveBackend" to training.info.backend, "effectivePrecision" to training.info.precision)
                    while (trained < limit && rmse > config.targetError) {
                        if (isCancelled()) return emptyList()
                        if (config.engine == TrainingEngine.SMALL) {
                            val result = training.trainChunk(TrainingChunkRequest(limit - trained, targetError = config.targetError,
                                checkEvery = 1, cancelled = ::isCancelled))
                            trained += result.committedEpochs
                            rmse = result.rmse
                            if (result.termination == TrainingTermination.CANCELLED) return emptyList()
                        } else {
                            rmse = trainConfiguredEpoch(model, training, config.batchSize)
                            trained++
                        }
                    }
                }
                NeuroLog.info("studio", "studio.study.seed.completed", "runId" to runId, "studyId" to studyId,
                    "seed" to seed, "epochs" to trained, "rmse" to rmse)
                val diagnostics = NeuroXorDiagnostics.capture(model, trained, rmse)
                results += StudioSeed(seed, trained, rmse, rmse <= config.targetError,
                    NeuroXorDiagnostics.renderOutputMap(diagnostics, resolution(model.parameterCount(), 160)))
            }
            NeuroLog.info("studio", "studio.study.completed", "runId" to runId, "studyId" to studyId, "seeds" to results.size)
            seedResults = results.toList()
            return seedResults
        } catch (exception: Exception) {
            NeuroLog.error("studio", "studio.study.failed", exception, "runId" to runId, "studyId" to studyId)
            throw exception
        }
    }

    fun searchData(evaluation: ArchitectureEvaluation, fraction: Double = 0.2, splitSeed: Long = 42): ArchitectureSearchData =
        if (evaluation == ArchitectureEvaluation.TRAINING_FIT) ArchitectureSearchData.fitting(samples, config.dataset.toString())
        else ArchitectureSearchData.split(samples, fraction, splitSeed, config.dataset.toString())

    fun applyArchitecture(report: ArchitectureSearchResult, candidate: ArchitectureCandidate) {
        require(candidate.valid && report.candidates.any { it === candidate }) { "Choose a fully evaluated architecture from this search." }
        val hp = report.config.hyperParameters
        apply(config.copy(hidden = candidate.architecture.hidden.joinToString(","), maxEpochs = report.config.maxEpochs,
            targetError = report.config.targetRmse, learningRate = hp.learningRate, momentum = hp.momentum,
            backend = report.config.backend, precision = report.config.precision, batchSize = report.config.batchSize,
            engine = report.config.engine, sigmoid = hp.sigmoidMode))
        NeuroLog.info("studio", "studio.architecture.applied", "runId" to runId, "searchId" to report.logId, "topology" to candidate.architecture)
    }

    fun replayArchitecture(report: ArchitectureSearchResult, candidate: ArchitectureCandidate, trial: ArchitectureTrial,
                           cancelled: () -> Boolean = { false }): Boolean {
        ensureOpen()
        require(candidate.valid && report.candidates.any { it === candidate } && candidate.trials.any { it === trial } &&
            trial.state == ArchitectureTrialState.COMPLETED) { "Choose a completed seed run from this search." }
        val replayId = NeuroLog.id("replay")
        NeuroLog.info("studio", "studio.replay.started", "runId" to runId, "replayId" to replayId,
            "searchId" to report.logId, "trialId" to trial.logId, "seed" to trial.seed, "bestEpoch" to trial.bestEpoch)
        fun isCancelled(): Boolean = cancelled().also { stopped ->
            if (stopped) NeuroLog.info("studio", "studio.replay.cancelled", "runId" to runId, "replayId" to replayId)
        }
        try {
            if (isCancelled()) return false
            closeTraining()
            val model = report.data.newNetwork(candidate.architecture, report.config.hyperParameters, trial.seed)
            val recordedDevice = requireNotNull(trial.deviceInfo) { "This trial has no recorded training backend; replay is unavailable." }
            val recordedPrecision = Neuro.TrainingPrecision.valueOf(recordedDevice.precision)
            fun verifyDevice(actual: TrainingDeviceInfo) {
                if (actual != recordedDevice) NeuroLog.warn("studio", "studio.replay.provenance.rejected", null,
                    "runId" to runId, "replayId" to replayId, "recordedDevice" to recordedDevice, "actualDevice" to actual)
                check(actual == recordedDevice) { "Replay requires the recorded training device, precision and kernel version: ${recordedDevice.name}." }
                NeuroLog.info("studio", "studio.replay.provenance.accepted", "runId" to runId, "replayId" to replayId,
                    "model" to model.logId, "session" to model.trainingSessionLogId,
                    "backend" to recordedDevice.backend, "device" to recordedDevice.name, "precision" to recordedDevice.precision,
                    "deviceIdentity" to recordedDevice.identity, "kernelVersion" to recordedDevice.kernelVersion,
                    "engine" to recordedDevice.engine, "simdBits" to recordedDevice.simdBits, "sigmoid" to recordedDevice.sigmoid,
                    "cohort" to trial.cohort, "execution" to trial.execution, "route" to trial.route)
            }
            fun replayChunks(train: (TrainingChunkRequest) -> SearchAdvanceResult): Boolean {
                var replayed = 0
                while (replayed < trial.bestEpoch) {
                    if (isCancelled()) return false
                    val boundary = minOf(trial.bestEpoch - replayed, report.config.checkEvery - replayed % report.config.checkEvery)
                    val result = train(TrainingChunkRequest(boundary, checkEvery = boundary, cancelled = ::isCancelled))
                    check(result.committedEpochs in 0..minOf(64, boundary)) { "Replay crossed a checkpoint boundary." }
                    replayed += result.committedEpochs
                    if (result.termination == TrainingTermination.CANCELLED) return false
                    check(result.committedEpochs > 0) { "Replay made no training progress." }
                }
                return true
            }
            if (trial.route == ArchitectureTrialRoute.CUDA_QUEUE) {
                (openSearchCuda?.invoke(recordedPrecision, report.config.batchSize)
                    ?: SearchCudaService(recordedPrecision, report.config.batchSize, maximumModels = 1)).use { service ->
                    service.openSession(model).use { training ->
                        verifyDevice(training.info)
                        if (!replayChunks(training::advanceForSearch)) return false
                    }
                }
            } else if (trial.cohort) {
                openTrainingCohort(listOf(model), recordedDevice.backend, recordedPrecision, report.config.batchSize, 1).use { cohort ->
                    verifyDevice(cohort.info)
                    if (!replayChunks { request -> cohort.trainChunk(request).single().let {
                        SearchAdvanceResult(it.committedEpochs, it.rmse, it.termination)
                    } }) return false
                }
            } else {
                openTrainingSession(model, recordedDevice.backend, recordedPrecision, report.config.batchSize, recordedDevice.engine).use { training ->
                    verifyDevice(training.info)
                    if (trial.execution == ArchitectureExecution.OPTIMIZED) {
                        if (!replayChunks { request -> advanceTrainingForSearch(training, request) }) return false
                    } else if (recordedDevice.engine == TrainingEngine.SMALL) {
                        if (!replayChunks { request -> training.trainChunk(request).let {
                            SearchAdvanceResult(it.committedEpochs, it.rmse, it.termination)
                        } }) return false
                    } else repeat(trial.bestEpoch) {
                        if (isCancelled()) return false
                        trainConfiguredEpoch(model, training, report.config.batchSize)
                    }
                }
            }
            val score = report.data.score(model)
            check(score.isFinite() && kotlin.math.abs(score - trial.bestRmse) <= 1e-10) { "Replay did not reproduce the scored checkpoint." }
            if (isCancelled()) return false
            applyArchitecture(report, candidate)
            config = config.copy(seed = trial.seed, backend = recordedDevice.backend, precision = recordedPrecision,
                engine = recordedDevice.engine, sigmoid = Neuro.SigmoidMode.valueOf(recordedDevice.sigmoid))
            deviceInfo = recordedDevice
            samples = report.data.training
            network = model
            epoch = trial.bestEpoch
            error = model.trainingError()
            stepped = true
            replayNote = "Search replay · ${report.data.evaluation.label} %.5f · %d held-out samples".format(java.util.Locale.ROOT, score, report.data.validation.size)
            NeuroLog.info("studio", "studio.replay.completed", "runId" to runId, "replayId" to replayId,
                "searchId" to report.logId, "trialId" to trial.logId, "epoch" to epoch, "score" to score)
            return true
        } catch (exception: Exception) {
            NeuroLog.error("studio", "studio.replay.failed", exception, "runId" to runId, "replayId" to replayId,
                "searchId" to report.logId, "trialId" to trial.logId)
            throw exception
        }
    }

    fun fail(message: String) {
        require(message.isNotBlank()) { "Failure message must not be empty." }
        automatic = false
        pendingEpochs = 0
        failure = message
        NeuroLog.warn("studio", "studio.training.failed", null, "runId" to runId, "epoch" to epoch, "reason" to message)
        // A backend may publish an epoch before reporting a native-resource cleanup failure.
        epoch = network.statistics().epochsTrained.toInt()
        error = network.trainingError()
        render = null
        layerImages = null
        difference = null
        try { closeTraining() } catch (exception: Exception) {
            NeuroLog.error("studio", "studio.cleanup.failed", exception, "runId" to runId)
            failure += " Resource cleanup: ${exception.message ?: exception.javaClass.simpleName}"
        }
    }

    fun frame(): StudioFrame {
        val current = ensureRender()
        val images = layerImages ?: (if (current.diagnostics.hiddenLayerCount() == 0) emptyList()
            else NeuroXorDiagnostics.renderHiddenMaps(current.diagnostics,
                resolution(network.parameterCount(), 96), selectedLayer, hiddenStart,
                minOf(8, current.diagnostics.layerOutputCount(selectedLayer) - hiddenStart)).toList()).also { layerImages = it }
        val delta = difference ?: renderDifference(previousValues ?: current.values, current.values,
            current.image.width).also { difference = it }
        return StudioFrame(config, state, current.diagnostics, samples, current.image,
            previousImage ?: current.image, previousEpoch, delta, images, selectedLayer, hiddenStart,
            history.toList(), checkpoints.toList(), seedResults, failure, replayNote, deviceInfo)
    }

    private fun trainingSession(): NeuroTrainingSession = session ?: openTrainingSession(network, config.backend, config.precision, config.batchSize, config.engine).also {
        session = it
        deviceInfo = it.info
        NeuroLog.info("studio", "studio.backend.ready", "runId" to runId, "model" to network.logId,
            "session" to network.trainingSessionLogId, "requestedBackend" to config.backend,
            "requestedPrecision" to config.precision, "effectiveBackend" to it.info.backend, "effectivePrecision" to it.info.precision,
            "engine" to it.info.engine, "simdBits" to it.info.simdBits, "sigmoid" to it.info.sigmoid,
            "batchSize" to config.batchSize, "device" to it.info.name, "deviceIdentity" to it.info.identity, "kernelVersion" to it.info.kernelVersion)
    }

    /** Releases the current session while retaining parameters, momentum, shuffle state and the device owner. */
    internal fun closeTraining() {
        val current = session
        session = null
        if (current != null) {
            try { current.close() } catch (exception: Exception) {
                NeuroLog.error("studio", "studio.session.release.failed", exception, "runId" to runId, "epoch" to epoch)
                throw exception
            }
            NeuroLog.info("studio", "studio.session.released", "runId" to runId, "epoch" to epoch)
        }
    }

    override fun close() {
        closed = true
        var failure: Throwable? = null
        try { closeTraining() } catch (exception: Throwable) { failure = exception }
        try {
            if (ownedDeviceService.isInitialized()) ownedDeviceService.value.close()
        } catch (exception: Throwable) {
            if (failure == null) failure = exception else failure.addSuppressed(exception)
        }
        failure?.let { throw it }
    }

    private fun canTrain(): Boolean = samples.isNotEmpty() && epoch < config.maxEpochs &&
        error > config.targetError && failure.isEmpty()

    private fun ensureRender(): Render {
        render?.let { return it }
        val diagnostics = NeuroXorDiagnostics.capture(network, epoch, error)
        val size = resolution(network.parameterCount(), 192)
        val values = DoubleArray(size * size)
        val image = NeuroXorGrid.render(network, size, NeuroXorGrid.createInputs(size), values)
        val result = Render(diagnostics, image, values)
        render = result
        if (history.lastOrNull()?.epoch != epoch) {
            val predictions = doubleArrayOf(values[(size - 1) * size], values[0], values.last(), values[size - 1])
            val norms = DoubleArray(diagnostics.layerCount() * 2) {
                if (it % 2 == 0) NeuroXorDiagnostics.weightNorm(diagnostics, it / 2)
                else NeuroXorDiagnostics.biasNorm(diagnostics, it / 2)
            }
            history += StudioHistory(epoch, error, predictions, diagnostics.parameters(), norms)
            if (history.size > historyCapacity) {
                val retained = history.filterIndexed { index, _ -> index == 0 || index % 2 == 0 || index == history.lastIndex }
                history.clear()
                history.addAll(retained)
            }
        }
        if (checkpoints.isEmpty()) checkpoints += StudioCheckpoint(epoch, error, image)
        return result
    }

    private fun captureCheckpoint() {
        if (checkpoints.lastOrNull()?.epoch == epoch) return
        val diagnostics = NeuroXorDiagnostics.capture(network, epoch, error)
        checkpoints += StudioCheckpoint(epoch, error,
            NeuroXorDiagnostics.renderOutputMap(diagnostics, resolution(network.parameterCount(), 128)))
        NeuroLog.info("studio", "studio.checkpoint.created", "runId" to runId, "epoch" to epoch, "rmse" to error)
    }

    companion object {
        private val MILESTONES = setOf(10, 50, 100, 250, 500, 1000, 2000, 5000, 10_000, 100_000, 1_000_000)
        private val STUDY_SEEDS = longArrayOf(1, 42, 123, 999)
        private fun samplesFor(config: StudioConfig, custom: List<NeuroLearningSets.Sample>): List<NeuroLearningSets.Sample> =
            if (config.dataset == NeuroLearningSets.Kind.CUSTOM) custom.toList()
            else NeuroLearningSets.create(config.dataset, 0xC0FFEE42L).toList()
        private fun createNetwork(config: StudioConfig, seed: Long, samples: List<NeuroLearningSets.Sample>): Neuro =
            Neuro(config.topology(), Neuro.HyperParameters(config.learningRate, config.momentum, 1.0, seed, sigmoidMode = config.sigmoid)).also {
                NeuroLearningSets.addTo(it, samples)
            }
        fun resolution(parameters: Int, maximum: Int): Int {
            require(parameters > 0 && maximum >= 8)
            return sqrt(2_000_000.0 / parameters).toInt().coerceIn(8, maximum)
        }
        fun renderDifference(before: DoubleArray, after: DoubleArray, size: Int): BufferedImage {
            require(size >= 2 && size.toLong() * size == before.size.toLong() && before.size == after.size) { "Difference grid dimensions do not match." }
            val image = BufferedImage(size, size, BufferedImage.TYPE_INT_RGB)
            val pixels = (image.raster.dataBuffer as DataBufferInt).data
            for (index in pixels.indices) pixels[index] = NeuroXorDiagnostics.differenceRgb(after[index] - before[index])
            return image
        }
    }
}

/** Keep cancellation, diagnostics and momentum publication at complete-epoch boundaries for every backend. */
internal fun trainConfiguredEpoch(model: Neuro, session: NeuroTrainingSession, batchSize: Int): Double =
    if (batchSize == 1) session.trainEpoch() else {
        session.trainMiniBatch(1, batchSize)
        model.trainingError()
    }
