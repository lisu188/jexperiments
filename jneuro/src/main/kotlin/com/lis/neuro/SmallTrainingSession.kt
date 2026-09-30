package com.lis.neuro

/** Private compute state; publication and shuffle advancement share the same host checkpoint. */
internal class SmallTrainingSession(
    private val network: Neuro,
    private val batchSize: Int,
    kernelFactory: (NeuroTrainingState) -> SmallTrainingKernel,
    private val clock: () -> Long = System::nanoTime
) : NeuroTrainingSession {
    private val sessionId = NeuroLog.id("session")
    private val openedNanos = clock()
    private val initialEpoch = network.statistics().epochsTrained
    private var closed = false
    private var failed = false
    private var nanosPerEpoch = 0L
    private val kernel: SmallTrainingKernel
    override val info: TrainingDeviceInfo get() = kernel.info
    override val currentRmse: Double get() = network.statistics().lastTrainingError.let {
        if (it.isNaN()) network.trainingError() else it
    }

    init {
        require(batchSize > 0)
        network.acquireTraining(this, sessionId)
        try {
            kernel = kernelFactory(network.exportTrainingState())
            logSessionOpened(network, sessionId, info, batchSize)
        } catch (failure: Throwable) {
            network.releaseTraining(this)
            throw failure
        }
    }

    private fun <T> run(operation: String, epochs: Int, batch: Int = batchSize, action: () -> T): T =
        network.withTraining(this) {
            check(!closed) { "Training session is closed." }
            check(!failed) { "Training failed. Close this session before resuming from the last committed checkpoint." }
            loggedTraining(network, sessionId, info, operation, epochs, batch, initialEpoch = initialEpoch, action = action)
        }

    private fun execute(epochs: Int, batch: Int, online: Boolean): Double {
        val orders = network.reserveTrainingOrders(epochs)
        val started = clock()
        try {
            val state = kernel.train(orders, batch, online)
            val error = network.commitTrainingChunk(state, epochs)
            val elapsed = maxOf(1L, clock() - started)
            nanosPerEpoch = maxOf(1L, elapsed / epochs)
            return error
        } catch (failure: Throwable) {
            // The kernel may have advanced private state. Reopening imports only the validated host checkpoint.
            failed = true
            throw failure
        }
    }

    override fun trainEpoch(): Double = run("trainEpoch", 1) { execute(1, batchSize, batchSize == 1) }
    override fun train(epochs: Int) = run("train", epochs) {
        require(epochs >= 0)
        repeat(epochs) { execute(1, batchSize, batchSize == 1) }
    }
    override fun trainMiniBatch(epochs: Int, batchSize: Int, parallelism: Int) = run("trainMiniBatch", epochs, batchSize) {
        require(epochs >= 0 && batchSize > 0 && parallelism > 0)
        repeat(epochs) { execute(1, batchSize, false) }
    }

    override fun trainUntil(targetError: Double, maxEpochs: Int, checkEvery: Int): Neuro.TrainingResult = run("trainUntil", maxEpochs) {
        require(targetError.isFinite() && targetError >= 0.0 && maxEpochs >= 0 && checkEvery > 0)
        check(network.trainingSampleCount() > 0) { "no training samples" }
        var error = currentRmse
        var completed = 0
        if (error <= targetError) network.recordTrainingError(error)
        while (completed < maxEpochs && error > targetError) {
            val current = execute(1, batchSize, batchSize == 1)
            completed++
            if (completed % checkEvery == 0 || completed == maxEpochs) error = current
        }
        Neuro.TrainingResult(completed, error, error <= targetError)
    }

    override fun trainChunk(request: TrainingChunkRequest): TrainingChunkResult = run("trainChunk", request.maxEpochs) {
        val started = clock()
        var error = currentRmse
        var completed = 0
        var termination = TrainingTermination.COMPLETED
        if (request.targetError != null && error <= request.targetError) {
            network.recordTrainingError(error)
            return@run TrainingChunkResult(0, error, TrainingTermination.CONVERGED)
        }
        val limit = minOf(64, request.maxEpochs)
        while (completed < limit) {
            if (request.cancelled()) { termination = TrainingTermination.CANCELLED; break }
            val elapsed = clock() - started
            if (completed > 0 && elapsed >= request.maxNanos) { termination = TrainingTermination.BUDGET; break }
            val remainingNanos = maxOf(1L, request.maxNanos - elapsed)
            // Begin with one epoch to measure this model/device. Never claim a hard real-time deadline.
            val adaptive = if (nanosPerEpoch == 0L) 1 else (remainingNanos / nanosPerEpoch).coerceIn(1, 64).toInt()
            val scoring = if (request.targetError == null) limit else request.checkEvery - completed % request.checkEvery
            val count = minOf(limit - completed, scoring, adaptive)
            error = execute(count, batchSize, batchSize == 1)
            completed += count
            if (request.targetError != null && (completed % request.checkEvery == 0 || completed == request.maxEpochs) && error <= request.targetError) {
                termination = TrainingTermination.CONVERGED; break
            }
        }
        if (termination == TrainingTermination.COMPLETED && completed < request.maxEpochs) termination = TrainingTermination.BUDGET
        TrainingChunkResult(completed, error, termination)
    }

    @Synchronized override fun close() {
        if (closed) return
        network.withTraining(this) {
            closed = true
            try { kernel.close() }
            finally {
                network.releaseTraining(this)
                logSessionClosed(network, sessionId, info, openedNanos, failed)
            }
        }
    }
}
