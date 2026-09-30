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
    val kernelVersion: String = "cpu-v1",
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
    val targets: DoubleArray
) {
    val samples: Int get() = inputs.size / topology[0]
}

internal fun openTrainingSession(network: Neuro, backend: TrainingBackend,
                                 driverFactory: () -> CudaDriver = { NativeCudaDriver() }): NeuroTrainingSession =
    openConfiguredTrainingSession(network, backend, Neuro.TrainingPrecision.FP64, 1, driverFactory)

internal fun openConfiguredTrainingSession(
    network: Neuro, backend: TrainingBackend, precision: Neuro.TrainingPrecision, batchSize: Int,
    driverFactory: () -> CudaDriver = { NativeCudaDriver() },
    cublasFactory: (Neuro, Neuro.TrainingPrecision, Int) -> NeuroTrainingSession =
        { model, format, batch -> CublasTrainingSession(model, format, batch) },
    cublasAvailable: () -> Boolean = { NeuroCuda.isAvailable() },
    engine: TrainingEngine = TrainingEngine.REFERENCE
): NeuroTrainingSession {
    NeuroLog.debug("training", "session.requested") { mapOf("model" to network.logId,
        "requestedBackend" to backend, "requestedPrecision" to precision, "batchSize" to batchSize) }
    var effective: TrainingBackend? = null
    return try {
        require(batchSize > 0) { "batchSize must be > 0" }
        require(engine != TrainingEngine.SMALL || backend != TrainingBackend.CUBLAS) {
            "SMALL supports CPU, CUDA and AUTO. Choose REFERENCE for cuBLAS."
        }
        require(engine != TrainingEngine.SMALL || SmallNetworkShape.supports(network.topology())) {
            "SMALL requires 2 inputs, 1 output, and 1–4 hidden layers of width 4, 8 or 16."
        }
        require(engine == TrainingEngine.SMALL || backend != TrainingBackend.CUDA || precision == Neuro.TrainingPrecision.FP64) {
            "Driver CUDA training uses FP64. Select CUBLAS for FP32 training."
        }
        val resolved = if (backend == TrainingBackend.AUTO) {
            val batch = minOf(batchSize, maxOf(1, network.trainingSampleCount()))
            if (engine != TrainingEngine.SMALL && NeuroCuda.resolveBackend(Neuro.BatchBackend.AUTO, network.topology(), batch, true) == Neuro.BatchBackend.CUDA && cublasAvailable())
                TrainingBackend.CUBLAS else TrainingBackend.CPU
        } else backend
        effective = resolved
        val session = if (engine == TrainingEngine.SMALL) SmallTrainingSession(network, batchSize, kernelFactory = { state ->
                if (resolved == TrainingBackend.CPU) SmallCpuTraining(state, network.hyperParameters(), precision)
                else SmallCudaTraining(state, network.hyperParameters(), precision, driverFactory())
            })
            else if (resolved == TrainingBackend.CUBLAS) cublasFactory(network, precision, batchSize)
            else DefaultTrainingSession(network, resolved, driverFactory, batchSize)
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
            "resolvedBackend" to effective, "requestedPrecision" to precision, "requestedBatchSize" to batchSize)
        if (failure is IllegalArgumentException) NeuroLog.warn("training", "session.open.rejected", failure, *fields)
        else NeuroLog.error("training", "session.open.failed", failure, *fields)
        throw failure
    }
}

private class DefaultTrainingSession(private val network: Neuro, backend: TrainingBackend,
                                     driverFactory: () -> CudaDriver,
                                     private val batchSize: Int) : NeuroTrainingSession {
    private val sessionId = NeuroLog.id("session")
    private val openedNanos = System.nanoTime()
    private val initialEpoch: Long
    private var closed = false
    private var failed = false
    private val cuda: CudaTraining?
    override val info: TrainingDeviceInfo
    override val currentRmse: Double get() = network.trainingError()

    init {
        network.acquireTraining(this, sessionId)
        initialEpoch = network.statistics().epochsTrained
        try {
            cuda = if (backend == TrainingBackend.CUDA) CudaTraining(network, network.exportTrainingState(), driverFactory()) else null
            info = cuda?.info?.copy(sigmoid = network.hyperParameters().sigmoidMode.name) ?: TrainingDeviceInfo(TrainingBackend.CPU, "CPU", "jvm-cpu",
                kernelVersion = if (batchSize > 1) "cpu-matrix-v1" else "cpu-v1", sigmoid = network.hyperParameters().sigmoidMode.name)
            logSessionOpened(network, sessionId, info, batchSize)
        } catch (failure: Throwable) {
            network.releaseTraining(this)
            throw failure
        }
    }

    private fun <T> run(operation: String, epochs: Int, batch: Int = batchSize,
                        parallelism: Int = 1, details: Map<String, Any?> = emptyMap(),
                        action: () -> T): T = network.withTraining(this) {
        loggedTraining(network, sessionId, info, operation, epochs, batch, parallelism, details, initialEpoch) {
            check(!closed) { "Training session is closed." }
            check(!failed) { "CUDA training failed. Close this session before continuing from the last completed epoch." }
            try { action() } catch (failure: Throwable) {
                if (cuda != null && failure !is IllegalArgumentException) failed = true
                throw failure
            }
        }
    }

    private fun epoch(): Double = cuda?.epoch(batchSize, batchSize == 1) ?: if (batchSize == 1) network.trainEpoch() else {
        network.trainMiniBatch(1, batchSize, 1, Neuro.BatchBackend.CPU)
        network.statistics().lastTrainingError
    }

    override fun trainEpoch(): Double = run("trainEpoch", 1) { epoch() }

    override fun train(epochs: Int) = run("train", epochs) {
        require(epochs >= 0) { "epochs must be >= 0" }
        if (cuda == null && batchSize == 1) network.train(epochs) else repeat(epochs) { epoch() }
    }

    override fun trainUntil(targetError: Double, maxEpochs: Int, checkEvery: Int): Neuro.TrainingResult = run("trainUntil", maxEpochs,
        details = mapOf("targetError" to targetError, "checkEvery" to checkEvery)) {
        require(targetError.isFinite() && targetError >= 0.0) { "targetError must be finite and >= 0" }
        require(maxEpochs >= 0) { "maxEpochs must be >= 0" }
        require(checkEvery > 0) { "checkEvery must be > 0" }
        if (cuda == null && batchSize == 1) network.trainUntil(targetError, maxEpochs, checkEvery) else {
            check(network.trainingSampleCount() > 0) { "no training samples" }
            var error = network.trainingError()
            if (error <= targetError) network.recordTrainingError(error)
            var epochs = 0
            while (epochs < maxEpochs && error > targetError) {
                val current = epoch()
                epochs++
                if (epochs % checkEvery == 0 || epochs == maxEpochs) error = current
            }
            Neuro.TrainingResult(epochs, error, error <= targetError)
        }
    }

    override fun trainMiniBatch(epochs: Int, batchSize: Int, parallelism: Int) = run("trainMiniBatch", epochs, batchSize, parallelism) {
        require(epochs >= 0) { "epochs must be >= 0" }
        require(batchSize > 0) { "batchSize must be > 0" }
        require(parallelism > 0) { "parallelism must be > 0" }
        if (cuda == null) {
            if (this.batchSize > 1) network.trainMiniBatch(epochs, batchSize, parallelism, Neuro.BatchBackend.CPU)
            else network.trainMiniBatch(epochs, batchSize, parallelism)
        } else repeat(epochs) { cuda.epoch(batchSize, false) }
    }

    @Synchronized override fun close() {
        if (closed) return
        network.withTraining(this) {
            closed = true
            try {
                cuda?.close()
                logSessionClosed(network, sessionId, info, openedNanos, failed)
            } catch (failure: Throwable) {
                NeuroLog.error("training", "session.close.failed", failure, "model" to network.logId,
                    "session" to sessionId, "backend" to info.backend,
                    "completedEpochs" to network.statistics().epochsTrained)
                throw failure
            } finally { network.releaseTraining(this) }
        }
    }
}

/** Logs completed work only after the training call has returned and published its host state. */
internal fun <T> loggedTraining(network: Neuro, sessionId: String?, info: TrainingDeviceInfo,
                               operation: String, requestedEpochs: Int, batchSize: Int,
                               parallelism: Int = 1, details: Map<String, Any?> = emptyMap(), initialEpoch: Long = 0,
                               action: () -> T): T {
    val before = network.statistics()
    val started = System.nanoTime()
    val runId = NeuroLog.id("run")
    fun fields(after: Neuro.Statistics) = linkedMapOf<String, Any?>(
        "model" to network.logId, "session" to sessionId, "run" to runId, "operation" to operation,
        "backend" to info.backend, "device" to info.name, "deviceIdentity" to info.identity,
        "precision" to info.precision, "kernel" to info.kernelVersion,
        "requestedEpochs" to requestedEpochs, "requestedBatchSize" to batchSize,
        "batchSize" to minOf(batchSize, network.trainingSampleCount()), "requestedParallelism" to parallelism,
        "cpuParallelism" to if (info.backend == TrainingBackend.CPU) parallelism else null,
        "trainingSamples" to network.trainingSampleCount(), "completedEpochs" to (after.epochsTrained - before.epochsTrained),
        "totalEpochs" to after.epochsTrained, "samplesProcessed" to (after.samplesSeen - before.samplesSeen),
        "totalSamplesSeen" to after.samplesSeen, "lastRecordedRmse" to after.lastTrainingError,
        "durationMs" to ((System.nanoTime() - started) / 1_000_000.0)).apply { putAll(details) }
    if (requestedEpochs > 1 || operation == "trainUntil")
        NeuroLog.info("training", "training.started") { fields(before) }
    else NeuroLog.debug("training", "training.started") { fields(before) }
    try {
        val result = action()
        val after = network.statistics()
        fun completedFields() = fields(after).apply {
            put("gpuWorkCompleted", (info.backend == TrainingBackend.CUDA || info.backend == TrainingBackend.CUBLAS) &&
                after.epochsTrained > before.epochsTrained)
            when (result) {
                is Double -> put("rmse", result)
                is Neuro.TrainingResult -> { put("rmse", result.error); put("converged", result.converged) }
                is TrainingChunkResult -> { put("rmse", result.rmse); put("termination", result.termination) }
            }
        }
        // UI/architecture loops call a single epoch repeatedly. INFO remains useful and bounded.
        if (requestedEpochs != 1 || operation == "trainUntil" ||
            before.epochsTrained == initialEpoch || after.epochsTrained / 100 > before.epochsTrained / 100) {
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
