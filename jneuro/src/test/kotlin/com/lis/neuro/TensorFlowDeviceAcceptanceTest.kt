package com.lis.neuro

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.Timeout

/** This suite requires a real GPU device. Missing hardware is a failure, never a skipped test. */
@Tag("tensorflow-gpu")
@Timeout(120)
class TensorFlowDeviceAcceptanceTest {
    @Test fun onlineGpuTrainingMatchesCpuParametersMomentumAndContinuation() {
        for (mode in Neuro.SigmoidMode.entries) {
            for (shape in listOf(intArrayOf(2, 1), intArrayOf(3, 5, 2), intArrayOf(2, 33, 17, 2), intArrayOf(3, 257, 2))) {
                val expected = NeuroTest.prepared(shape, Neuro.Kernel.SCALAR, mode)
                val actual = NeuroTest.prepared(shape, Neuro.Kernel.SCALAR, mode)
                expected.train(2); actual.train(2)
                actual.newTrainingSession(TrainingBackend.GPU).use { session ->
                    assertGpu(session.info)
                    assertEquals(expected.trainEpoch(), session.trainEpoch(), TOLERANCE)
                    assertModelsEqual(expected, actual)
                    expected.train(3); session.train(3)
                    assertModelsEqual(expected, actual)
                    val input = DoubleArray(shape.first()) { 0.25 }
                    val target = DoubleArray(shape.last()) { 0.75 }
                    assertThrows(IllegalStateException::class.java) { actual.addTrainingSample(input, target) }
                }
                // Continuing on the CPU detects lost device momentum or incorrect shuffle-state ownership.
                expected.train(2); actual.train(2)
                assertModelsEqual(expected, actual)
                val input = DoubleArray(shape.first()) { 0.25 }
                val target = DoubleArray(shape.last()) { 0.75 }
                expected.addTrainingSample(input, target); actual.addTrainingSample(input, target)
                actual.newTrainingSession(TrainingBackend.GPU).use { session ->
                    assertEquals(expected.trainEpoch(), session.trainEpoch(), TOLERANCE)
                    assertModelsEqual(expected, actual)
                }
            }
        }
    }

    @Test fun gpuMiniBatchesHandleSingletonRemaindersAndWholeDataset() {
        for (batchSize in listOf(1, 7, 64)) {
            val expected = NeuroTest.prepared(intArrayOf(3, 7, 2), Neuro.Kernel.SCALAR)
            val actual = NeuroTest.prepared(intArrayOf(3, 7, 2), Neuro.Kernel.SCALAR)
            actual.newTrainingSession(TrainingBackend.GPU).use { session ->
                assertGpu(session.info)
                expected.trainMiniBatch(3, batchSize)
                session.trainMiniBatch(3, batchSize, 2)
                assertModelsEqual(expected, actual)
                val before = actual.statistics()
                session.train(0)
                session.trainMiniBatch(0, batchSize)
                assertEquals(before, actual.statistics())
                expected.trainMiniBatch(1, batchSize)
                session.trainMiniBatch(1, batchSize)
                assertModelsEqual(expected, actual)
            }
            expected.train(1); actual.train(1)
            assertModelsEqual(expected, actual)
        }
    }

    @Test fun gpuRespectsNondefaultBetaAndZeroOrHighMomentum() {
        for (momentum in listOf(0.0, 0.8)) {
            val parameters = Neuro.HyperParameters(0.2, momentum, 1.7, 456, Neuro.Kernel.SCALAR)
            fun model() = Neuro(intArrayOf(2, 7, 1), parameters).also {
                NeuroLearningSets.addTo(it, NeuroLearningSets.create(NeuroLearningSets.Kind.XOR, 0))
            }
            val expected = model()
            val actual = model()
            actual.newTrainingSession(TrainingBackend.GPU).use { session ->
                expected.train(3); session.train(3)
                assertModelsEqual(expected, actual)
                expected.trainMiniBatch(2, 3); session.trainMiniBatch(2, 3)
                assertModelsEqual(expected, actual)
            }
        }
    }

    @Test fun gpuConvergesWithSparseChecksAndCanBeReopened() {
        val expected = NeuroTest.xor(Neuro.Kernel.SCALAR)
        val actual = NeuroTest.xor(Neuro.Kernel.SCALAR)
        val session = actual.newTrainingSession(TrainingBackend.GPU)
        session.use {
            assertGpu(it.info)
            val cpuStart = System.nanoTime()
            val cpuResult = expected.trainUntil(0.08, 10_000, 8)
            val cpuMillis = (System.nanoTime() - cpuStart) / 1_000_000
            val gpuStart = System.nanoTime()
            val gpuResult = it.trainUntil(0.08, 10_000, 8)
            val gpuMillis = (System.nanoTime() - gpuStart) / 1_000_000
            assertTrue(gpuResult.converged)
            assertEquals(cpuResult.epochs, gpuResult.epochs)
            assertEquals(cpuResult.error, gpuResult.error, TOLERANCE)
            assertModelsEqual(expected, actual)
            println("GPU acceptance: ${it.info}; epochs=${gpuResult.epochs}; samples=${actual.statistics().samplesSeen}; " +
                "rmse=${gpuResult.error}; cpuTrainingMs=$cpuMillis; gpuTrainingMs=$gpuMillis")
        }
        session.close()
        assertThrows(IllegalStateException::class.java) { session.trainEpoch() }
        actual.newTrainingSession(TrainingBackend.GPU).use {
            assertEquals(expected.trainEpoch(), it.trainEpoch(), TOLERANCE)
            assertModelsEqual(expected, actual)
        }
    }

    private fun assertGpu(info: TrainingDeviceInfo) {
        assertEquals(TrainingBackend.GPU, info.backend)
        assertEquals("FP64", info.precision)
        assertTrue(info.name.isNotBlank())
        assertTrue(info.identity.isNotBlank())
        assertTrue(info.kernelVersion.isNotBlank())
    }

    private fun assertModelsEqual(expected: Neuro, actual: Neuro) {
        val reference = expected.exportTrainingState()
        val result = actual.exportTrainingState()
        assertArrayEquals(reference.topology, result.topology)
        for (layer in reference.weights.indices) {
            assertArrayEquals(reference.weights[layer], result.weights[layer], TOLERANCE, "weights, layer $layer")
            assertArrayEquals(reference.biases[layer], result.biases[layer], TOLERANCE, "biases, layer $layer")
            assertArrayEquals(reference.weightVelocity[layer], result.weightVelocity[layer], TOLERANCE, "momentum, layer $layer")
            assertArrayEquals(reference.biasVelocity[layer], result.biasVelocity[layer], TOLERANCE, "bias momentum, layer $layer")
        }
        assertEquals(expected.statistics().epochsTrained, actual.statistics().epochsTrained)
        assertEquals(expected.statistics().samplesSeen, actual.statistics().samplesSeen)
        assertEquals(expected.statistics().lastTrainingError, actual.statistics().lastTrainingError, TOLERANCE)
        assertEquals(expected.trainingError(), actual.trainingError(), TOLERANCE)
        assertArrayEquals(expected.predict(DoubleArray(reference.topology.first()) { 0.3 }),
            actual.predict(DoubleArray(reference.topology.first()) { 0.3 }), TOLERANCE)
    }

    companion object { private const val TOLERANCE = 1e-10 }
}
