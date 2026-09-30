package com.lis.neuro

enum class TrainingBackend { CPU, CUDA }

data class TrainingDeviceInfo(
    val backend: TrainingBackend,
    val name: String,
    val identity: String,
    val precision: String = "FP64",
    val kernelVersion: String = "cpu-v1"
)

/** A single writer for a model. Closing keeps the last completed epoch available on the CPU. */
interface NeuroTrainingSession : AutoCloseable {
    val info: TrainingDeviceInfo
    fun trainEpoch(): Double
    fun train(epochs: Int)
    fun trainUntil(targetError: Double, maxEpochs: Int, checkEvery: Int = 1): Neuro.TrainingResult
    fun trainMiniBatch(epochs: Int, batchSize: Int, parallelism: Int = 1)
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
    DefaultTrainingSession(network, backend, driverFactory)

private class DefaultTrainingSession(private val network: Neuro, backend: TrainingBackend,
                                     driverFactory: () -> CudaDriver) : NeuroTrainingSession {
    private var closed = false
    private var failed = false
    private val cuda: CudaTraining?
    override val info: TrainingDeviceInfo

    init {
        network.acquireTraining(this)
        try {
            cuda = if (backend == TrainingBackend.CUDA) CudaTraining(network, network.exportTrainingState(), driverFactory()) else null
            info = cuda?.info ?: TrainingDeviceInfo(TrainingBackend.CPU, "CPU", "jvm-cpu")
        } catch (failure: Throwable) {
            network.releaseTraining(this)
            throw failure
        }
    }

    private fun <T> run(action: () -> T): T {
        return network.withTraining(this) {
            check(!closed) { "Training session is closed." }
            check(!failed) { "CUDA training failed. Close this session before continuing from the last completed epoch." }
            try { action() } catch (failure: Throwable) {
                if (cuda != null && failure !is IllegalArgumentException) failed = true
                throw failure
            }
        }
    }

    override fun trainEpoch(): Double = run { cuda?.epoch(1, true) ?: network.trainEpoch() }

    override fun train(epochs: Int) = run {
        require(epochs >= 0) { "epochs must be >= 0" }
        if (cuda == null) network.train(epochs) else repeat(epochs) { cuda.epoch(1, true) }
    }

    override fun trainUntil(targetError: Double, maxEpochs: Int, checkEvery: Int): Neuro.TrainingResult = run {
        require(targetError.isFinite() && targetError >= 0.0) { "targetError must be finite and >= 0" }
        require(maxEpochs >= 0) { "maxEpochs must be >= 0" }
        require(checkEvery > 0) { "checkEvery must be > 0" }
        if (cuda == null) network.trainUntil(targetError, maxEpochs, checkEvery) else {
            check(network.trainingSampleCount() > 0) { "no training samples" }
            var error = network.trainingError()
            if (error <= targetError) network.recordTrainingError(error)
            var epochs = 0
            while (epochs < maxEpochs && error > targetError) {
                val current = cuda.epoch(1, true)
                epochs++
                if (epochs % checkEvery == 0 || epochs == maxEpochs) error = current
            }
            Neuro.TrainingResult(epochs, error, error <= targetError)
        }
    }

    override fun trainMiniBatch(epochs: Int, batchSize: Int, parallelism: Int) = run {
        require(epochs >= 0) { "epochs must be >= 0" }
        require(batchSize > 0) { "batchSize must be > 0" }
        require(parallelism > 0) { "parallelism must be > 0" }
        if (cuda == null) network.trainMiniBatch(epochs, batchSize, parallelism)
        else repeat(epochs) { cuda.epoch(batchSize, false) }
    }

    @Synchronized override fun close() {
        if (closed) return
        network.withTraining(this) {
            closed = true
            try { cuda?.close() } finally { network.releaseTraining(this) }
        }
    }
}
