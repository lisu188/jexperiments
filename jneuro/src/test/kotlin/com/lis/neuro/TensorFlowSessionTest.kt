package com.lis.neuro

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

class TensorFlowSessionTest {
    @Test fun referenceAndSmallEnginesHonorBothCpuPrecisionsAndAutoReportsCpu() {
        for (engine in TrainingEngine.entries) for (precision in Neuro.TrainingPrecision.entries) {
            val model = preparedSmall(intArrayOf(2, 4, 1), Neuro.HyperParameters.defaults(), 7)
            model.newTrainingSession(TrainingBackend.AUTO, precision, 3, engine).use { session ->
                assertEquals(TrainingBackend.CPU, session.info.backend)
                assertEquals(precision.name, session.info.precision)
                assertEquals(engine, session.info.engine)
                assertEquals(0, session.info.simdBits)
                assertTrue(session.info.name.startsWith("TensorFlow "))
                assertTrue(session.info.kernelVersion.startsWith("tensorflow-"))
                assertEquals(2, session.trainChunk(TrainingChunkRequest(2, maxNanos = Long.MAX_VALUE)).committedEpochs)
                assertThrows(IllegalStateException::class.java) { model.newTrainingSession() }
            }
            assertEquals(2L, model.statistics().epochsTrained)
            assertEquals(14L, model.statistics().samplesSeen)
            model.newTrainingSession(precision = precision, batchSize = 3, engine = engine).use { assertTrue(it.trainEpoch().isFinite()) }
        }
    }

    @Test fun searchConfigurationAcceptsAllTensorFlowAliasesAndPrecisions() {
        for (backend in TrainingBackend.entries) for (engine in TrainingEngine.entries) for (precision in Neuro.TrainingPrecision.entries) {
            val config = ArchitectureSearchConfig(backend = backend, precision = precision, engine = engine)
            assertEquals(backend, config.backend)
            assertEquals(precision, config.precision)
        }
        assertEquals(Neuro.BatchBackend.CPU, NeuroCuda.resolveBackend(Neuro.BatchBackend.AUTO, intArrayOf(4096, 4096), 1024, true))
        assertEquals(Neuro.BatchBackend.CPU, NeuroCuda.resolveBackend(Neuro.BatchBackend.CPU, intArrayOf(2, 1), 1))
        assertEquals(Neuro.BatchBackend.CUDA, NeuroCuda.resolveBackend(Neuro.BatchBackend.CUDA, intArrayOf(2, 1), 1))
    }
}
