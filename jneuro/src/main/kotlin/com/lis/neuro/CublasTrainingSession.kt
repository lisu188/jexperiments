package com.lis.neuro

import java.security.MessageDigest

/** Optional matrix backend; every successful epoch is published atomically by NeuroCudaBatchBackend. */
internal class CublasTrainingSession(
    private val network: Neuro,
    private val precision: Neuro.TrainingPrecision,
    private val batchSize: Int,
    infoFactory: (Neuro.TrainingPrecision) -> TrainingDeviceInfo = { cublasDeviceInfo(it) },
    private val batchTrain: (Neuro, Int, Int, Neuro.TrainingPrecision) -> Unit = { model, epochs, batch, format ->
        NeuroCudaBatchBackend.train(model, model.backendTrainingData(), epochs, batch, format)
    }
) : NeuroTrainingSession {
    private val sessionId = NeuroLog.id("session")
    private val openedNanos = System.nanoTime()
    private val initialEpoch: Long
    private var closed = false
    private var failed = false
    override val info: TrainingDeviceInfo
    override val currentRmse: Double get() = network.trainingError()

    init {
        require(batchSize > 0) { "batchSize must be > 0" }
        network.acquireTraining(this, sessionId)
        initialEpoch = network.statistics().epochsTrained
        try {
            info = infoFactory(precision).copy(sigmoid = network.hyperParameters().sigmoidMode.name)
            check(info.backend == TrainingBackend.CUBLAS && info.precision == precision.name) {
                "cuBLAS device information does not match the requested backend and precision."
            }
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
            check(!failed) { "cuBLAS training failed. Close the session before resuming from the last completed epoch." }
            try { action() } catch (failure: Throwable) {
                if (failure !is IllegalArgumentException) failed = true
                throw failure
            }
        }
    }

    private fun trainBatches(epochs: Int, batch: Int) {
        if (epochs == 0) return
        check(network.trainingSampleCount() > 0) { "no training samples" }
        batchTrain(network, epochs, minOf(batch, network.trainingSampleCount()), precision)
    }

    override fun trainEpoch(): Double = run("trainEpoch", 1) {
        trainBatches(1, batchSize)
        network.statistics().lastTrainingError.let { if (it.isNaN()) network.trainingError() else it }
    }
    override fun train(epochs: Int) = run("train", epochs) {
        require(epochs >= 0) { "epochs must be >= 0" }
        trainBatches(epochs, batchSize)
    }
    override fun trainUntil(targetError: Double, maxEpochs: Int, checkEvery: Int): Neuro.TrainingResult = run("trainUntil", maxEpochs,
        details = mapOf("targetError" to targetError, "checkEvery" to checkEvery)) {
        require(targetError.isFinite() && targetError >= 0.0) { "targetError must be finite and >= 0" }
        require(maxEpochs >= 0) { "maxEpochs must be >= 0" }
        require(checkEvery > 0) { "checkEvery must be > 0" }
        check(network.trainingSampleCount() > 0) { "no training samples" }
        var error = network.trainingError()
        if (error <= targetError) network.recordTrainingError(error)
        var epochs = 0
        while (epochs < maxEpochs && error > targetError) {
            val count = minOf(checkEvery, maxEpochs - epochs)
            trainBatches(count, batchSize)
            epochs += count
            error = network.trainingError()
        }
        Neuro.TrainingResult(epochs, error, error <= targetError)
    }
    override fun trainMiniBatch(epochs: Int, batchSize: Int, parallelism: Int) = run("trainMiniBatch", epochs, batchSize, parallelism) {
        require(epochs >= 0) { "epochs must be >= 0" }
        require(batchSize > 0) { "batchSize must be > 0" }
        require(parallelism > 0) { "parallelism must be > 0" }
        trainBatches(epochs, batchSize)
    }
    @Synchronized override fun close() {
        if (closed) return
        network.withTraining(this) {
            closed = true
            network.releaseTraining(this)
            logSessionClosed(network, sessionId, info, openedNanos, failed)
        }
    }
}

internal fun cublasDeviceInfo(
    precision: Neuro.TrainingPrecision,
    loader: NeuroLibraryLoader = NeuroNativeLibrary::open,
    driverInfo: () -> TrainingDeviceInfo = { NativeCudaDriver().use { it.info } }
): TrainingDeviceInfo {
    val device = driverInfo()
    return NeuroCudaRuntime.create(loader).use { runtime ->
        check(runtime.deviceCount() > 0) { "No CUDA-capable devices are available" }
        runtime.setDevice(0)
        val capability = runtime.computeCapability(0)
        NeuroCublas.create(loader).use { cublas ->
            NeuroNvrtc.create(loader).use { nvrtc ->
                val source = checkNotNull(CublasTrainingSession::class.java.getResourceAsStream("/cuda/jneuro.cu")) {
                    "Packaged cuBLAS activation kernels are missing."
                }.use { it.readAllBytes() }
                val hash = MessageDigest.getInstance("SHA-256").digest(source).joinToString("") { "%02x".format(it) }
                val version = nvrtc.version()
                val options = NeuroNvrtc.compilerOptions(capability).joinToString(",")
                device.copy(backend = TrainingBackend.CUBLAS, precision = precision.name,
                    kernelVersion = "cublas-${cublas.version() ?: cublas.libraryName}" +
                        "/nvrtc-${version.first}.${version.second}/compute-${capability.first}.${capability.second}" +
                        "/options=$options/source-$hash")
            }
        }
    }
}
