package com.lis.neuro

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assumptions.assumeTrue
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test

@Tag("cuda")
class NeuroCudaIntegrationTest {
    @Test fun cudaFp64TrainingMatchesCpuMatrixTraining() {
        assumeTrue(NeuroCuda.isAvailable(), NeuroCuda.status().reason)
        val cpu = NeuroTest.prepared(intArrayOf(32, 64, 16, 4), Neuro.Kernel.SCALAR)
        val gpu = NeuroTest.prepared(intArrayOf(32, 64, 16, 4), Neuro.Kernel.SCALAR)
        cpu.trainMiniBatch(4, 16, 1, Neuro.TrainingBackend.CPU)
        gpu.trainMiniBatch(4, 16, 1, Neuro.TrainingBackend.CUDA)
        val input = NeuroTest.input(32)
        assertArrayEquals(cpu.predict(input), gpu.predict(input), 1e-9)
        assertEquals(cpu.trainingError(), gpu.trainingError(), 1e-9)
        assertEquals(cpu.statistics().epochsTrained, gpu.statistics().epochsTrained)
        assertEquals(cpu.statistics().samplesSeen, gpu.statistics().samplesSeen)
    }

    @Test fun cudaFp32TracksCpuWithinSinglePrecisionTolerance() {
        assumeTrue(NeuroCuda.isAvailable(), NeuroCuda.status().reason)
        val cpu = NeuroTest.prepared(intArrayOf(32, 64, 16, 4), Neuro.Kernel.SCALAR)
        val gpu = NeuroTest.prepared(intArrayOf(32, 64, 16, 4), Neuro.Kernel.SCALAR)
        cpu.trainMiniBatch(4, 16, 1, Neuro.TrainingBackend.CPU)
        gpu.trainMiniBatch(4, 16, 1, Neuro.TrainingBackend.CUDA, Neuro.TrainingPrecision.FP32)
        assertArrayEquals(cpu.predict(NeuroTest.input(32)), gpu.predict(NeuroTest.input(32)), 5e-4)
        assertEquals(cpu.trainingError(), gpu.trainingError(), 5e-4)
    }

    @Test fun cudaHandlesPartialFinalBatchAndFastSigmoid() {
        assumeTrue(NeuroCuda.isAvailable(), NeuroCuda.status().reason)
        val gpu = NeuroTest.prepared(intArrayOf(17, 33, 9, 2), Neuro.Kernel.AUTO, Neuro.SigmoidMode.FAST)
        gpu.trainMiniBatch(3, 13, 1, Neuro.TrainingBackend.CUDA)
        assertEquals(3L, gpu.statistics().epochsTrained)
        assertEquals(96L, gpu.statistics().samplesSeen)
    }
}
