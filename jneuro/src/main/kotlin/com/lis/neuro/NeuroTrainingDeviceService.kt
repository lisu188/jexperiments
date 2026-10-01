package com.lis.neuro

/** Application admission owner. Every numerical session owns its TensorFlow resources. */
class NeuroTrainingDeviceService @JvmOverloads constructor(maximumSessions: Int = 1) : AutoCloseable {
    init { require(maximumSessions > 0) }
    private var closed = false
    private var owners = 0

    @Synchronized @JvmOverloads
    fun openSession(model: Neuro, backend: TrainingBackend = TrainingBackend.CPU,
                    precision: Neuro.TrainingPrecision = Neuro.TrainingPrecision.FP64,
                    batchSize: Int = 1, engine: TrainingEngine = TrainingEngine.REFERENCE): NeuroTrainingSession {
        check(!closed) { "Training device service is closed" }
        val delegate = openConfiguredTrainingSession(model, backend, precision, batchSize, engine)
        owners++
        return OwnedSession(delegate) { released() }
    }

    @Synchronized @JvmOverloads
    fun openCohort(models: List<Neuro>, backend: TrainingBackend = TrainingBackend.CPU,
                   precision: Neuro.TrainingPrecision = Neuro.TrainingPrecision.FP64,
                   batchSize: Int = 1, parallelism: Int = 4): NeuroTrainingCohort {
        check(!closed) { "Training device service is closed" }
        val cohort = NeuroTrainingCohort.openConfigured(models, backend, precision, batchSize, parallelism)
        owners++
        cohort.onClose = { released() }
        return cohort
    }

    @Synchronized private fun released() { owners-- }
    @Synchronized override fun close() {
        if (closed) return
        check(owners == 0) { "Close all training sessions and cohorts before their service." }
        closed = true
    }

    private class OwnedSession(private val delegate: NeuroTrainingSession, private val released: () -> Unit) :
        NeuroTrainingSession by delegate, SearchEpochAdvancer {
        private var closed = false
        override fun advanceForSearch(request: TrainingChunkRequest) = advanceTrainingForSearch(delegate, request)
        @Synchronized override fun close() {
            if (closed) return
            closed = true
            try { delegate.close() } finally { released() }
        }
    }
}
