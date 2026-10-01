package com.lis.neuro

import java.util.concurrent.TimeUnit
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.Timeout

/** Real mixed-topology CUDA arithmetic; driver absence/failure must fail this task. */
@Tag("cuda")
@Timeout(240)
class SearchCudaAcceptanceTest {
    @Test fun mixedTopologiesAndPrecisionsMatchAllParametersMomentumAndContinuation() {
        for (precision in Neuro.TrainingPrecision.entries) for (mode in Neuro.SigmoidMode.entries) {
            SearchCudaService(precision, 7).use { service ->
                for (shapes in shapes().chunked(64)) {
                    val models = shapes.mapIndexed { index, shape -> model(shape, index.toLong(), mode) }
                    val references = shapes.mapIndexed { index, shape -> model(shape, index.toLong(), mode) }
                    val sessions = models.map(service::openSession)
                    try {
                        val results = sessions.map { it.advanceForSearchAsync(request(3)) }.map { it.get(30, TimeUnit.SECONDS) }
                        sessions.indices.forEach { index ->
                            assertEquals(3, results[index].committedEpochs)
                            assertNull(results[index].rmse)
                            references[index].newTrainingSession(TrainingBackend.CPU, precision, 7, TrainingEngine.SMALL).use { it.train(3) }
                            assertSmallState(references[index].exportTrainingState(), models[index].exportTrainingState(), tolerance(precision))
                            assertTrue(models[index].statistics().lastTrainingError.isNaN())
                        }
                        // Unequal epoch requests remain independent within one heterogeneous launch.
                        val more = sessions.mapIndexed { index, session -> session.advanceForSearchAsync(request(index % 3 + 1)) }
                        more.forEachIndexed { index, result ->
                            assertEquals(index % 3 + 1, result.get(30, TimeUnit.SECONDS).committedEpochs)
                            references[index].newTrainingSession(TrainingBackend.CPU, precision, 7, TrainingEngine.SMALL).use { it.train(index % 3 + 1) }
                            assertSmallState(references[index].exportTrainingState(), models[index].exportTrainingState(), tolerance(precision))
                        }
                    } finally { sessions.forEach { it.close() } }
                    models.indices.forEach { index ->
                        models[index].trainEpoch(); references[index].trainEpoch()
                        assertSmallState(references[index].exportTrainingState(), models[index].exportTrainingState(), tolerance(precision))
                    }
                }
            }
        }
    }

    @Test fun onlineMaximumChunksRaggedDatasetAndSameRouteReopeningPreserveShuffle() {
        val shapes = listOf(intArrayOf(2, 4, 1), intArrayOf(2, 4, 16, 8, 16, 1), intArrayOf(2, 16, 16, 16, 16, 1))
        for (precision in Neuro.TrainingPrecision.entries) for (batch in listOf(1, 64)) {
            SearchCudaService(precision, batch).use { service ->
                val models = shapes.mapIndexed { index, shape -> model(shape, index.toLong(), Neuro.SigmoidMode.EXACT) }
                val references = shapes.mapIndexed { index, shape -> model(shape, index.toLong(), Neuro.SigmoidMode.EXACT) }
                val sessions = models.map(service::openSession)
                val info = sessions.first().info
                try {
                    val results = sessions.map { it.advanceForSearchAsync(request(65)) }
                    results.forEachIndexed { index, result ->
                        assertEquals(SearchAdvanceResult(64, null, TrainingTermination.BUDGET), result.get(30, TimeUnit.SECONDS))
                        references[index].newTrainingSession(TrainingBackend.CPU, precision, batch, TrainingEngine.SMALL).use { it.train(64) }
                        assertSmallState(references[index].exportTrainingState(), models[index].exportTrainingState(), tolerance(precision))
                    }
                } finally { sessions.forEach { it.close() } }
                service.openSession(models.first()).use { reopened ->
                    assertEquals(info, reopened.info)
                    assertEquals(2, reopened.advanceForSearch(request(2)).committedEpochs)
                }
                references.first().newTrainingSession(TrainingBackend.CPU, precision, batch, TrainingEngine.SMALL).use { it.train(2) }
                assertSmallState(references.first().exportTrainingState(), models.first().exportTrainingState(), tolerance(precision))
                models.first().addTrainingSample(doubleArrayOf(0.1, 0.9), doubleArrayOf(1.0))
                assertThrows(IllegalArgumentException::class.java) { service.openSession(models.first()) }
            }
        }
    }

    private fun request(epochs: Int) = TrainingChunkRequest(epochs, maxNanos = Long.MAX_VALUE)
    private fun tolerance(precision: Neuro.TrainingPrecision) = if (precision == Neuro.TrainingPrecision.FP64) 1e-9 else 3e-5
    private fun model(shape: IntArray, seed: Long, mode: Neuro.SigmoidMode) = preparedSmall(shape,
        Neuro.HyperParameters(0.07, 0.21, 1.1, seed, Neuro.Kernel.SCALAR, mode), 19)
    private fun shapes(): List<IntArray> = buildList {
        fun visit(hidden: List<Int>) {
            if (hidden.isNotEmpty()) add((listOf(2) + hidden + 1).toIntArray())
            if (hidden.size < 4) for (width in listOf(4, 8, 16)) visit(hidden + width)
        }
        visit(emptyList())
    }
}
