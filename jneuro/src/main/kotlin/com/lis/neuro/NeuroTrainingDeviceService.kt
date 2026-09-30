package com.lis.neuro

/** Explicit application owner for retained CUDA contexts, modules and streams. Close sessions before the service. */
class NeuroTrainingDeviceService internal constructor(private val cuda: SmallCudaDeviceService) : AutoCloseable {
    @JvmOverloads constructor(maximumSessions: Int = 1) : this(SmallCudaDeviceService(maximumSessions))
    private var closed = false

    @Synchronized @JvmOverloads
    fun openSession(model: Neuro, backend: TrainingBackend = TrainingBackend.CPU,
                    precision: Neuro.TrainingPrecision = Neuro.TrainingPrecision.FP64,
                    batchSize: Int = 1, engine: TrainingEngine = TrainingEngine.REFERENCE): NeuroTrainingSession {
        check(!closed) { "Training device service is closed" }
        return openConfiguredTrainingSession(model, backend, precision, batchSize,
            driverFactory = cuda::openDriver, engine = engine)
    }

    @Synchronized @JvmOverloads
    fun openCohort(models: List<Neuro>, backend: TrainingBackend = TrainingBackend.CPU,
                   precision: Neuro.TrainingPrecision = Neuro.TrainingPrecision.FP64,
                   batchSize: Int = 1, parallelism: Int = 4): NeuroTrainingCohort {
        check(!closed) { "Training device service is closed" }
        return NeuroTrainingCohort.openConfigured(models, backend, precision, batchSize, parallelism, cuda::openDriver)
    }

    @Synchronized override fun close() {
        if (closed) return
        cuda.close()
        closed = true
    }
}
