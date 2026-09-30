package com.lis.neuro

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class NeuroCpuBatchTrainerTest {
    @Test fun batchSizeOneTracksOnlineTraining() {
        val online = NeuroTest.prepared(intArrayOf(7, 9, 5, 2), Neuro.Kernel.SCALAR)
        val matrix = NeuroTest.prepared(intArrayOf(7, 9, 5, 2), Neuro.Kernel.SCALAR)
        online.train(5)
        matrix.trainMiniBatch(5, 1)
        val input = NeuroTest.input(7)
        assertArrayEquals(online.predict(input), matrix.predict(input), 1e-12)
        assertEquals(online.statistics().epochsTrained, matrix.statistics().epochsTrained)
        assertEquals(online.statistics().samplesSeen, matrix.statistics().samplesSeen)
    }

    @Test fun sequentialAndParallelMatrixTrainingAgree() {
        val sequential = NeuroTest.prepared(intArrayOf(33, 65, 17, 3))
        val parallel = NeuroTest.prepared(intArrayOf(33, 65, 17, 3))
        sequential.trainMiniBatch(4, 7, 1)
        parallel.trainMiniBatch(4, 7, 4)
        val input = NeuroTest.input(33)
        assertArrayEquals(sequential.predict(input), parallel.predict(input), 1e-12)
        assertEquals(sequential.trainingError(), parallel.trainingError(), 1e-12)
    }

    @Test fun matrixTrainingHandlesPartialAndOversizedBatches() {
        val network = NeuroTest.prepared(intArrayOf(5, 8, 4, 2))
        val before = network.trainingError()
        network.trainMiniBatch(6, 13, 3)
        assertTrue(network.trainingError() < before)
        assertEquals(6L, network.statistics().epochsTrained)
        assertEquals(192L, network.statistics().samplesSeen)
        val oversized = NeuroTest.prepared(intArrayOf(5, 8, 4, 2))
        oversized.trainMiniBatch(2, 128, 2)
        assertEquals(64L, oversized.statistics().samplesSeen)
    }
}
