package com.lis.neuro

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.Timeout

/** Strict real-device TensorFlow GPU acceptance. The matrix is intentionally small enough for ordinary developer GPUs. */
@Tag("tensorflow-gpu")
@Timeout(120)
class TensorFlowGpuIntegrationTest {
    @Test fun tensorflowGpuMatchesFullStateAcrossPrecisionsSigmoidsAndTopologiesWithCpuContinuation() {
        requireTensorFlowGpu()
        val shapes = listOf(intArrayOf(2, 1), intArrayOf(3, 5, 2), intArrayOf(3, 7, 5, 2))
        for (precision in Neuro.TrainingPrecision.entries) for (mode in Neuro.SigmoidMode.entries) for (shape in shapes) {
            val cpu = NeuroTest.prepared(shape, Neuro.Kernel.SCALAR, mode)
            val gpu = NeuroTest.prepared(shape, Neuro.Kernel.SCALAR, mode)
            cpu.train(2); gpu.train(2) // Enter TensorFlow GPU with non-zero optimizer momentum.
            gpu.newTrainingSession(TrainingBackend.GPU, precision, 7).use { session ->
                assertEquals(TrainingBackend.GPU, session.info.backend)
                assertEquals(precision.name, session.info.precision)
                assertTrue(session.info.identity.isNotBlank())
                assertTrue(session.info.kernelVersion.startsWith("tensorflow-"))
                cpu.trainMiniBatch(2, 7, 1, Neuro.BatchBackend.CPU)
                session.train(2)
                assertState(cpu, gpu, precision)
                println("TensorFlow GPU full-state parity: ${session.info}; shape=${shape.contentToString()}; sigmoid=$mode; batch=7; epochs=2")
            }
            // No GPU helper participates here: CPU continuation independently detects lost or misordered velocities.
            cpu.train(2); gpu.train(2)
            assertState(cpu, gpu, precision)
        }
    }

    @Test fun tensorflowGpuExplicitBatchApiHandlesSingletonRemaindersAndOversizedBatches() {
        requireTensorFlowGpu()
        for (precision in Neuro.TrainingPrecision.entries) for (batch in listOf(1, 7, 64)) {
            val cpu = NeuroTest.prepared(intArrayOf(3, 7, 2), Neuro.Kernel.SCALAR)
            val gpu = NeuroTest.prepared(intArrayOf(3, 7, 2), Neuro.Kernel.SCALAR)
            cpu.trainMiniBatch(2, batch, 1, Neuro.BatchBackend.CPU)
            gpu.trainMiniBatch(2, batch, 1, Neuro.BatchBackend.GPU, precision)
            assertState(cpu, gpu, precision)
            assertEquals(2L, gpu.statistics().epochsTrained)
            assertEquals(64L, gpu.statistics().samplesSeen)
            cpu.trainEpoch(); gpu.trainEpoch()
            assertState(cpu, gpu, precision)
        }
    }

    @Test fun tensorflowGpuReopensAfterDatasetGrowthAndPreservesSparseCheckResults() {
        requireTensorFlowGpu()
        for (precision in Neuro.TrainingPrecision.entries) {
            val cpu = NeuroTest.prepared(intArrayOf(3, 5, 2), Neuro.Kernel.SCALAR)
            val gpu = NeuroTest.prepared(intArrayOf(3, 5, 2), Neuro.Kernel.SCALAR)
            gpu.newTrainingSession(TrainingBackend.GPU, precision, 7).use { session ->
                cpu.trainMiniBatch(1, 7, 1, Neuro.BatchBackend.CPU)
                session.trainEpoch()
                assertState(cpu, gpu, precision)
            }
            val input = doubleArrayOf(0.25, 0.5, 0.75)
            val target = doubleArrayOf(1.0, 0.0)
            cpu.addTrainingSample(input, target); gpu.addTrainingSample(input, target)
            gpu.newTrainingSession(TrainingBackend.GPU, precision, 7).use { session ->
                val result = session.trainUntil(0.0, 3, 2)
                cpu.trainMiniBatch(3, 7, 1, Neuro.BatchBackend.CPU)
                assertEquals(3, result.epochs)
                assertFalse(result.converged)
                assertEquals(cpu.trainingError(), result.error, tolerance(precision))
                assertState(cpu, gpu, precision)
                val before = gpu.statistics()
                session.train(0); session.trainMiniBatch(0, 7)
                assertEquals(before, gpu.statistics())
            }
            assertEquals(131L, gpu.statistics().samplesSeen)
        }
    }

    private fun requireTensorFlowGpu() {
        assertTrue(TensorFlowMath.isGpuAvailable(),
            "TensorFlow GPU acceptance requires a GPU-enabled TensorFlow runtime and a real GPU: " + TensorFlowMath.gpuFailure())
    }

    private fun tolerance(precision: Neuro.TrainingPrecision): Double =
        if (precision == Neuro.TrainingPrecision.FP64) 1e-9 else 5e-5

    private fun assertState(cpu: Neuro, gpu: Neuro, precision: Neuro.TrainingPrecision) {
        val expected = cpu.exportTrainingState()
        val actual = gpu.exportTrainingState()
        val tolerance = tolerance(precision)
        assertArrayEquals(expected.topology, actual.topology)
        for (layer in expected.weights.indices) {
            assertArrayEquals(expected.weights[layer], actual.weights[layer], tolerance, "weights: layer $layer, $precision")
            assertArrayEquals(expected.biases[layer], actual.biases[layer], tolerance, "biases: layer $layer, $precision")
            assertArrayEquals(expected.weightVelocity[layer], actual.weightVelocity[layer], tolerance, "weight momentum: layer $layer, $precision")
            assertArrayEquals(expected.biasVelocity[layer], actual.biasVelocity[layer], tolerance, "bias momentum: layer $layer, $precision")
        }
        assertEquals(cpu.statistics().epochsTrained, gpu.statistics().epochsTrained)
        assertEquals(cpu.statistics().samplesSeen, gpu.statistics().samplesSeen)
        assertEquals(cpu.statistics().lastTrainingError, gpu.statistics().lastTrainingError, tolerance)
        assertEquals(cpu.trainingError(), gpu.trainingError(), tolerance)
        assertArrayEquals(cpu.predict(NeuroTest.input(expected.topology.first())),
            gpu.predict(NeuroTest.input(actual.topology.first())), tolerance)
    }
}
