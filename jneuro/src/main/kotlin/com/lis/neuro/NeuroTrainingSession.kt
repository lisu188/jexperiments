package com.lis.neuro

enum class TrainingBackend { CPU, CUDA, CUBLAS, AUTO }
enum class TrainingEngine { REFERENCE, SMALL }

enum class TrainingTermination { COMPLETED, CONVERGED, CANCELLED, BUDGET }

/** Explicitly permits deferred publication within a bounded chunk. Cancellation is checked between launches. */
data class TrainingChunkRequest(
    val maxEpochs: Int,
    val targetError: Double? = null,
    val checkEvery: Int = 1,
    val cancelled: () -> Boolean = { false },
    val maxNanos: Long = 25_000_000
) {
    init {
        require(maxEpochs >= 0 && checkEvery > 0 && maxNanos > 0)
        require(targetError == null || targetError.isFinite() && targetError >= 0.0)
    }
}

data class TrainingChunkResult(val committedEpochs: Int, val rmse: Double, val termination: TrainingTermination)

data class TrainingDeviceInfo(
    val backend: TrainingBackend,
    val name: String,
    val identity: String,
    val precision: String = "FP64",
    val kernelVersion: String = "tensorflow",
    val engine: TrainingEngine = TrainingEngine.REFERENCE,
    val simdBits: Int = 0,
    val sigmoid: String = "EXACT"
)

/** A single writer for a model. Closing keeps the last completed epoch available on the CPU. */
interface NeuroTrainingSession : AutoCloseable {
    override fun close()
    val info: TrainingDeviceInfo
    val currentRmse: Double get() = Double.NaN
    fun trainEpoch(): Double
    fun train(epochs: Int)
    fun trainUntil(targetError: Double, maxEpochs: Int, checkEvery: Int = 1): Neuro.TrainingResult
    fun trainMiniBatch(epochs: Int, batchSize: Int, parallelism: Int = 1)
    fun trainChunk(request: TrainingChunkRequest): TrainingChunkResult {
        val started = System.nanoTime()
        var error = currentRmse
        var completed = 0
        if (request.targetError != null && error <= request.targetError)
            return TrainingChunkResult(0, error, TrainingTermination.CONVERGED)
        while (completed < minOf(64, request.maxEpochs)) {
            if (request.cancelled()) return TrainingChunkResult(completed, error, TrainingTermination.CANCELLED)
            error = trainEpoch()
            completed++
            if (request.targetError != null && (completed % request.checkEvery == 0 || completed == request.maxEpochs) && error <= request.targetError)
                return TrainingChunkResult(completed, error, TrainingTermination.CONVERGED)
            if (completed < request.maxEpochs && System.nanoTime() - started >= request.maxNanos)
                return TrainingChunkResult(completed, error, TrainingTermination.BUDGET)
        }
        return TrainingChunkResult(completed, error,
            if (completed == request.maxEpochs) TrainingTermination.COMPLETED else TrainingTermination.BUDGET)
    }
}

internal interface SmallTrainingKernel : AutoCloseable {
    val info: TrainingDeviceInfo
    fun train(orders: Array<IntArray>, batchSize: Int, online: Boolean): NeuroTrainingState
    override fun close() {}
}

internal data class NeuroTrainingState(
    val topology: IntArray,
    val weights: Array<DoubleArray>,
    val biases: Array<DoubleArray>,
    val weightVelocity: Array<DoubleArray>,
    val biasVelocity: Array<DoubleArray>,
    val inputs: DoubleArray,
    val targets: DoubleArray,
    /** Internal read-only dataset ownership; parameter and momentum buffers remain independent snapshots. */
    val sharedDataset: Boolean = false
) {
    val samples: Int get() = inputs.size / topology[0]
}

internal fun openTrainingSession(network: Neuro, backend: TrainingBackend): NeuroTrainingSession =
    openConfiguredTrainingSession(network, backend, Neuro.TrainingPrecision.FP64, 1)

/** Legacy backend and engine selections share TensorFlow's numerical implementation. */
internal fun openConfiguredTrainingSession(
    network: Neuro, backend: TrainingBackend, precision: Neuro.TrainingPrecision, batchSize: Int,
    engine: TrainingEngine = TrainingEngine.REFERENCE,
    kernelFactory: (NeuroTrainingState) -> SmallTrainingKernel = { state ->
        TensorFlowMath.trainingKernel(state, network.hyperParameters(), precision, backend, engine)
    }
): NeuroTrainingSession {
    NeuroLog.debug("training", "session.requested") { mapOf("model" to network.logId,
        "requestedBackend" to backend, "requestedPrecision" to precision, "batchSize" to batchSize) }
    return try {
        require(batchSize > 0) { "batchSize must be > 0" }
        require(engine != TrainingEngine.SMALL || SmallNetworkShape.supports(network.topology())) {
            "SMALL requires 2 inputs, 1 output, and 1–4 hidden layers of width 4, 8 or 16."
        }
        val session = SmallTrainingSession(network, batchSize, kernelFactory)
        NeuroLog.info("training", "session.resolved", "model" to network.logId,
            "session" to network.trainingSessionLogId, "requestedBackend" to backend,
            "backend" to session.info.backend, "requestedPrecision" to precision,
            "precision" to session.info.precision, "requestedBatchSize" to batchSize,
            "batchSize" to minOf(batchSize, network.trainingSampleCount()),
            "device" to session.info.name, "deviceIdentity" to session.info.identity,
            "kernel" to session.info.kernelVersion)
        session
    } catch (failure: Throwable) {
        val fields = arrayOf("model" to network.logId, "requestedBackend" to backend,
            "requestedPrecision" to precision, "requestedBatchSize" to batchSize)
        if (failure is IllegalArgumentException) NeuroLog.warn("training", "session.open.rejected", failure, *fields)
        else NeuroLog.error("training", "session.open.failed", failure, *fields)
        throw failure
    }
}

/** Logs completed work only after the training call has returned and published its host state. */
internal fun <T> loggedTraining(network: Neuro, sessionId: String?, info: TrainingDeviceInfo,
                               operation: String, requestedEpochs: Int, batchSize: Int,
                               parallelism: Int = 1, details: Map<String, Any?> = emptyMap(), initialEpoch: Long = 0,
                               action: () -> T): T {
    val logProgress = NeuroLog.isEnabled("training", java.util.logging.Level.INFO)
    val searchAdvance = operation == "advanceForSearch"
    val beforeEpoch = network.trainingEpochCount()
    val beforeSamples = network.processedSampleCount()
    val started = System.nanoTime()
    var runId: String? = null
    fun fields(after: Neuro.Statistics) = linkedMapOf<String, Any?>(
        "model" to network.logId, "session" to sessionId,
        "run" to (runId ?: NeuroLog.id("run").also { runId = it }), "operation" to operation,
        "backend" to info.backend, "device" to info.name, "deviceIdentity" to info.identity,
        "precision" to info.precision, "kernel" to info.kernelVersion,
        "requestedEpochs" to requestedEpochs, "requestedBatchSize" to batchSize,
        "batchSize" to minOf(batchSize, network.trainingSampleCount()), "requestedParallelism" to parallelism,
        "cpuParallelism" to if (info.backend == TrainingBackend.CPU) parallelism else null,
        "trainingSamples" to network.trainingSampleCount(), "completedEpochs" to (after.epochsTrained - beforeEpoch),
        "totalEpochs" to after.epochsTrained, "samplesProcessed" to (after.samplesSeen - beforeSamples),
        "totalSamplesSeen" to after.samplesSeen, "lastRecordedRmse" to after.lastTrainingError,
        "durationMs" to ((System.nanoTime() - started) / 1_000_000.0)).apply { putAll(details) }
    if (logProgress) {
        val before = network.statistics()
        if (!searchAdvance && (requestedEpochs > 1 || operation == "trainUntil"))
            NeuroLog.info("training", "training.started") { fields(before) }
        else NeuroLog.debug("training", "training.started") { fields(before) }
    }
    try {
        val result = action()
        if (!logProgress) return result
        val after = network.statistics()
        fun completedFields() = fields(after).apply {
            put("gpuWorkCompleted", (info.backend == TrainingBackend.CUDA || info.backend == TrainingBackend.CUBLAS) &&
                after.epochsTrained > beforeEpoch)
            when (result) {
                is Double -> put("rmse", result)
                is Neuro.TrainingResult -> { put("rmse", result.error); put("converged", result.converged) }
                is TrainingChunkResult -> { put("rmse", result.rmse); put("termination", result.termination) }
                is SearchAdvanceResult -> { put("rmse", result.rmse); put("termination", result.termination) }
            }
        }
        // Search chunks are streaming progress just like repeated single-epoch calls.
        // Only published work can sample the initial session or a crossed hundred-epoch boundary.
        val sampledProgress = beforeEpoch == initialEpoch || after.epochsTrained / 100 > beforeEpoch / 100
        val infoCompletion = if (searchAdvance) after.epochsTrained > beforeEpoch && sampledProgress
            else requestedEpochs != 1 || operation == "trainUntil" || sampledProgress
        if (infoCompletion) {
            NeuroLog.info("training", "training.completed") { completedFields() }
        } else NeuroLog.debug("training", "training.completed") { completedFields() }
        return result
    } catch (failure: Throwable) {
        val details = fields(network.statistics()).apply {
            put("retainedState", if (info.backend == TrainingBackend.CPU) "current-model" else "last-completed-epoch")
            put("gpuWorkCompleted", false)
        }.toList().toTypedArray()
        if (failure is IllegalArgumentException) NeuroLog.warn("training", "training.rejected", failure, *details)
        else NeuroLog.error("training", "training.failed", failure, *details)
        throw failure
    }
}

internal fun logSessionOpened(network: Neuro, sessionId: String, info: TrainingDeviceInfo, batchSize: Int) {
    val parameters = network.hyperParameters()
    NeuroLog.info("training", "session.opened", "model" to network.logId, "session" to sessionId,
        "backend" to info.backend, "device" to info.name, "deviceIdentity" to info.identity,
        "precision" to info.precision, "kernel" to info.kernelVersion,
        "requestedBatchSize" to batchSize, "batchSize" to minOf(batchSize, network.trainingSampleCount()),
        "topology" to network.topology().joinToString("x"), "parameters" to network.parameterCount(),
        "trainingSamples" to network.trainingSampleCount(), "testSamples" to network.testSampleCount(),
        "learningRate" to parameters.learningRate, "momentum" to parameters.momentum, "beta" to parameters.beta,
        "seed" to parameters.seed, "cpuKernel" to parameters.kernel, "sigmoid" to parameters.sigmoidMode)
}

internal fun logSessionClosed(network: Neuro, sessionId: String, info: TrainingDeviceInfo,
                              openedNanos: Long, failed: Boolean) {
    val state = network.statistics()
    NeuroLog.info("training", "session.closed", "model" to network.logId, "session" to sessionId,
        "backend" to info.backend, "device" to info.name, "precision" to info.precision,
        "failed" to failed, "completedEpochs" to state.epochsTrained, "samplesSeen" to state.samplesSeen,
        "lastRecordedRmse" to state.lastTrainingError, "retainedState" to "current-model",
        "durationMs" to ((System.nanoTime() - openedNanos) / 1_000_000.0))
}
