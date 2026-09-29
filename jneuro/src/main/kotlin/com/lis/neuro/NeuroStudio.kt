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
    val momentum: Double = 0.2
) {
    init {
        NeuroTopologyConfig.parseHidden(hidden)
        require(maxEpochs in 1..1_000_000) { "Epoch limit must be between 1 and 1,000,000." }
        require(targetError.isFinite() && targetError > 0.0 && targetError < 1.0) { "Target RMSE must be between 0 and 1." }
        Neuro.HyperParameters(learningRate, momentum, 1.0, seed)
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
    val replayNote: String = ""
)

internal class NeuroStudio(config: StudioConfig = StudioConfig(), custom: List<NeuroLearningSets.Sample> = emptyList()) {
    private var config = config
    private var custom = custom.toList()
    private var samples = samplesFor(config, custom)
    private var network = createNetwork(config, config.seed, samples)
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

    fun apply(next: StudioConfig, run: Boolean = false) {
        config = next
        samples = samplesFor(next, custom)
        network = createNetwork(next, next.seed, samples)
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
    }

    fun setRunning(value: Boolean) {
        automatic = value && canTrain()
        if (!value) pendingEpochs = 0
    }

    fun step(count: Int) {
        require(count in 1..10_000) { "Step count must be between 1 and 10,000." }
        automatic = false
        if (!canTrain()) { pendingEpochs = 0; return }
        pendingEpochs = minOf(config.maxEpochs - epoch, pendingEpochs + count)
        stepped = true
    }

    fun addSample(x: Double, y: Double, target: Double) {
        require(config.dataset == NeuroLearningSets.Kind.CUSTOM) { "Select Custom before editing training points." }
        require(custom.size < 4096) { "Custom datasets support at most 4096 points." }
        custom = custom + NeuroLearningSets.Sample(x, y, target)
        apply(config, false)
    }

    fun undoSample() {
        require(config.dataset == NeuroLearningSets.Kind.CUSTOM) { "Select Custom before editing training points." }
        custom = custom.dropLast(1)
        apply(config, false)
    }

    fun clearSamples() {
        custom = emptyList()
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
        require(speed in 1..1000) { "Speed must be between 1 and 1000 epochs per refresh." }
        if (!hasWork || cancelled()) return 0
        ensureRender()
        val before = render!!
        previousEpoch = epoch
        previousImage = before.image
        previousValues = before.values
        val requested = if (pendingEpochs > 0) minOf(speed, pendingEpochs) else speed
        var advanced = 0
        while (advanced < requested && canTrain() && !cancelled()) {
            error = network.trainEpoch()
            check(error.isFinite()) { "Training produced a non-finite RMSE. Reset with different settings." }
            epoch++
            advanced++
            if (pendingEpochs > 0) pendingEpochs--
            if (epoch in MILESTONES || !canTrain()) captureCheckpoint()
        }
        if (advanced > 0) {
            render = null
            layerImages = null
            difference = null
        }
        if (!canTrain()) { automatic = false; pendingEpochs = 0 }
        return advanced
    }

    fun compareSeeds(limit: Int = config.maxEpochs, cancelled: () -> Boolean = { false }): List<StudioSeed> {
        require(limit in 1..1_000_000) { "Seed study epoch limit is invalid." }
        if (samples.isEmpty()) return emptyList()
        val results = ArrayList<StudioSeed>()
        for (seed in STUDY_SEEDS) {
            if (cancelled()) return emptyList()
            val model = createNetwork(config, seed, samples)
            var trained = 0
            var rmse = model.trainingError()
            while (trained < limit && rmse > config.targetError) {
                if (cancelled()) return emptyList()
                rmse = model.trainEpoch()
                trained++
            }
            val diagnostics = NeuroXorDiagnostics.capture(model, trained, rmse)
            results += StudioSeed(seed, trained, rmse, rmse <= config.targetError,
                NeuroXorDiagnostics.renderOutputMap(diagnostics, resolution(model.parameterCount(), 160)))
        }
        seedResults = results.toList()
        return seedResults
    }

    fun searchData(evaluation: ArchitectureEvaluation, fraction: Double = 0.2, splitSeed: Long = 42): ArchitectureSearchData =
        if (evaluation == ArchitectureEvaluation.TRAINING_FIT) ArchitectureSearchData.fitting(samples, config.dataset.toString())
        else ArchitectureSearchData.split(samples, fraction, splitSeed, config.dataset.toString())

    fun applyArchitecture(report: ArchitectureSearchResult, candidate: ArchitectureCandidate) {
        require(candidate.valid && report.candidates.any { it === candidate }) { "Choose a fully evaluated architecture from this search." }
        val hp = report.config.hyperParameters
        apply(config.copy(hidden = candidate.architecture.hidden.joinToString(","), maxEpochs = report.config.maxEpochs,
            targetError = report.config.targetRmse, learningRate = hp.learningRate, momentum = hp.momentum))
    }

    fun replayArchitecture(report: ArchitectureSearchResult, candidate: ArchitectureCandidate, trial: ArchitectureTrial,
                           cancelled: () -> Boolean = { false }): Boolean {
        require(candidate.valid && report.candidates.any { it === candidate } && candidate.trials.any { it === trial } &&
            trial.state == ArchitectureTrialState.COMPLETED) { "Choose a completed seed run from this search." }
        if (cancelled()) return false
        val model = report.data.newNetwork(candidate.architecture, report.config.hyperParameters, trial.seed)
        repeat(trial.bestEpoch) {
            if (cancelled()) return false
            model.trainEpoch()
        }
        val score = report.data.score(model)
        check(score.isFinite() && kotlin.math.abs(score - trial.bestRmse) <= 1e-10) { "Replay did not reproduce the scored checkpoint." }
        if (cancelled()) return false
        applyArchitecture(report, candidate)
        config = config.copy(seed = trial.seed)
        samples = report.data.training
        network = model
        epoch = trial.bestEpoch
        error = model.trainingError()
        stepped = true
        replayNote = "Search replay · ${report.data.evaluation.label} %.5f · %d held-out samples".format(java.util.Locale.ROOT, score, report.data.validation.size)
        return true
    }

    fun fail(message: String) {
        require(message.isNotBlank()) { "Failure message must not be empty." }
        automatic = false
        pendingEpochs = 0
        failure = message
    }

    fun frame(): StudioFrame {
        val current = ensureRender()
        val images = layerImages ?: NeuroXorDiagnostics.renderHiddenMaps(current.diagnostics,
            resolution(network.parameterCount(), 96), selectedLayer, hiddenStart,
            minOf(8, current.diagnostics.layerOutputCount(selectedLayer) - hiddenStart)).toList().also { layerImages = it }
        val delta = difference ?: renderDifference(previousValues ?: current.values, current.values,
            current.image.width).also { difference = it }
        return StudioFrame(config, state, current.diagnostics, samples, current.image,
            previousImage ?: current.image, previousEpoch, delta, images, selectedLayer, hiddenStart,
            history.toList(), checkpoints.toList(), seedResults, failure, replayNote)
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
    }

    companion object {
        private val MILESTONES = setOf(10, 50, 100, 250, 500, 1000, 2000, 5000, 10_000, 100_000, 1_000_000)
        private val STUDY_SEEDS = longArrayOf(1, 42, 123, 999)
        private fun samplesFor(config: StudioConfig, custom: List<NeuroLearningSets.Sample>): List<NeuroLearningSets.Sample> =
            if (config.dataset == NeuroLearningSets.Kind.CUSTOM) custom.toList()
            else NeuroLearningSets.create(config.dataset, 0xC0FFEE42L).toList()
        private fun createNetwork(config: StudioConfig, seed: Long, samples: List<NeuroLearningSets.Sample>): Neuro =
            Neuro(config.topology(), Neuro.HyperParameters(config.learningRate, config.momentum, 1.0, seed)).also {
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
