package com.lis.neuro

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

class NeuroTrainingSessionTest {
    @Test fun cpuSessionPreservesEveryTrainingEntrypointAndContinuation() {
        val expected = NeuroTest.prepared(intArrayOf(3, 5, 2), Neuro.Kernel.SCALAR)
        val actual = NeuroTest.prepared(intArrayOf(3, 5, 2), Neuro.Kernel.SCALAR)
        actual.newTrainingSession().use { session ->
            assertEquals(TrainingBackend.CPU, session.info.backend)
            assertTrue(session.info.name.isNotBlank())
            assertEquals("FP64", session.info.precision)
            assertEquals(expected.trainEpoch(), session.trainEpoch(), 0.0)
            expected.train(3); session.train(3)
            assertModelsEqual(expected, actual)
            val result = expected.trainUntil(0.0, 5, 2)
            assertEquals(result, session.trainUntil(0.0, 5, 2))
            expected.trainMiniBatch(3, 7, 2); session.trainMiniBatch(3, 7, 2)
            assertModelsEqual(expected, actual)
            expected.train(0); session.train(0)
            expected.trainMiniBatch(0, 1); session.trainMiniBatch(0, 1)
            assertModelsEqual(expected, actual)
        }
        expected.train(2); actual.train(2)
        assertModelsEqual(expected, actual)
    }

    @Test fun cpuSessionRetainsValidationAndConvergenceContracts() {
        Neuro(intArrayOf(2, 1)).newTrainingSession(TrainingBackend.CPU).use { session ->
            session.train(0)
            session.trainMiniBatch(0, 1)
            assertThrows(IllegalStateException::class.java) { session.trainEpoch() }
            assertThrows(IllegalArgumentException::class.java) { session.train(-1) }
            assertThrows(IllegalArgumentException::class.java) { session.trainMiniBatch(-1, 1) }
            assertThrows(IllegalArgumentException::class.java) { session.trainMiniBatch(0, 0) }
            assertThrows(IllegalArgumentException::class.java) { session.trainMiniBatch(0, 1, 0) }
            assertThrows(IllegalArgumentException::class.java) { session.trainUntil(Double.NaN, 0) }
            assertThrows(IllegalArgumentException::class.java) { session.trainUntil(0.0, -1) }
            assertThrows(IllegalArgumentException::class.java) { session.trainUntil(0.0, 0, 0) }
        }
        val model = NeuroTest.prepared(intArrayOf(3, 2))
        model.newTrainingSession().use { session ->
            val converged = session.trainUntil(1.0, 0)
            assertEquals(0, converged.epochs)
            assertTrue(converged.converged)
            assertEquals(model.trainingError(), converged.error)
            assertEquals(0, session.trainUntil(0.0, 0).epochs)
        }
    }

    @Test fun closedSessionCannotBeReusedButModelRemainsUsable() {
        val model = NeuroTest.prepared(intArrayOf(3, 2))
        val session = model.newTrainingSession()
        session.close()
        session.close()
        assertThrows(IllegalStateException::class.java) { session.trainEpoch() }
        assertThrows(IllegalStateException::class.java) { session.train(1) }
        assertThrows(IllegalStateException::class.java) { session.trainUntil(0.0, 1) }
        assertThrows(IllegalStateException::class.java) { session.trainMiniBatch(1, 1) }
        assertTrue(model.trainEpoch().isFinite())
    }

    private fun assertModelsEqual(expected: Neuro, actual: Neuro) {
        for (layer in 0 until expected.topology().lastIndex) {
            assertArrayEquals(expected.backendWeights(layer), actual.backendWeights(layer), 0.0)
            assertArrayEquals(expected.backendBiases(layer), actual.backendBiases(layer), 0.0)
        }
        assertEquals(expected.statistics(), actual.statistics())
        assertEquals(expected.trainingError(), actual.trainingError(), 0.0)
    }
}
