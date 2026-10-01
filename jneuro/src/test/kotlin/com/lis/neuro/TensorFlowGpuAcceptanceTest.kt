package com.lis.neuro

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.Timeout

/** A missing TensorFlow GPU must fail this explicit hardware task. */
@Tag("cuda")
@Timeout(180)
class TensorFlowGpuAcceptanceTest {
    @Test fun gpuTensorCohortPreservesMixedWidthStateScoringAndRaggedContinuation() {
        assertTrue(TensorFlowMath.isGpuAvailable(), "A GPU-enabled TensorFlow runtime and accessible GPU are required")
        for (precision in Neuro.TrainingPrecision.entries) for (mode in Neuro.SigmoidMode.entries) {
            val parameters = Neuro.HyperParameters(0.1, 0.2, 1.1, 42, sigmoidMode = mode)
            val states = arrayOf(intArrayOf(2, 4, 8, 1), intArrayOf(2, 8, 4, 1), intArrayOf(2, 3, 5, 1)).mapIndexed { lane, shape ->
                preparedSmall(shape, parameters.withSeed(42L + lane), 7).exportTrainingState()
            }.toTypedArray()
            fun open(backend: TrainingBackend) = TensorFlowMath.searchCohort(states, parameters, precision, backend,
                states[0].inputs.copyOfRange(0, 6), doubleArrayOf(1.0, 0.0, 1.0), 3)
            open(TrainingBackend.CPU).use { cpu -> open(TrainingBackend.CUDA).use { gpu ->
                assertEquals(TrainingBackend.CUDA, gpu.info.backend)
                assertEquals("tensorflow:/device:GPU:0", gpu.info.identity)
                assertTrue(gpu.info.kernelVersion.endsWith("-batched-v1-t1"))
                val tolerance = if (precision == Neuro.TrainingPrecision.FP64) 1e-9 else 5e-5
                for (count in intArrayOf(1, 2)) {
                    val orders = Array(states.size) { lane -> Array(count) { epoch -> IntArray(7) { (it + lane + epoch) % 7 } } }
                    assertArrayEquals(BooleanArray(states.size), cpu.advance(orders))
                    assertArrayEquals(BooleanArray(states.size), gpu.advance(orders))
                    val expected = cpu.score(); val actual = gpu.score()
                    assertArrayEquals(expected.epochs, actual.epochs)
                    assertArrayEquals(expected.trainingRmse, actual.trainingRmse, tolerance)
                    assertArrayEquals(expected.validationRmse, actual.validationRmse, tolerance)
                    assertArrayEquals(expected.bestScore, actual.bestScore, tolerance)
                    assertArrayEquals(expected.bestEpoch, actual.bestEpoch)
                    for (best in listOf(false, true)) {
                        val left = cpu.exportStates(intArrayOf(0, 1, 2), best)
                        val right = gpu.exportStates(intArrayOf(0, 1, 2), best)
                        left.indices.forEach { assertSmallState(left[it], right[it], tolerance) }
                    }
                }
            } }
        }
    }

    @Test fun gpuAliasesUseRealTensorFlowDevicesAndPreserveWeightsMomentumAndContinuation() {
        assertTrue(TensorFlowMath.isGpuAvailable(), "A GPU-enabled TensorFlow runtime and accessible GPU are required")
        for (backend in listOf(TrainingBackend.CUDA, TrainingBackend.CUBLAS))
            for (engine in TrainingEngine.entries) for (precision in Neuro.TrainingPrecision.entries) {
                fun model() = preparedSmall(intArrayOf(2, 4, 8, 1), Neuro.HyperParameters(0.1, 0.2, 1.1, 42), 7)
                val cpu = model(); val gpu = model()
                cpu.newTrainingSession(TrainingBackend.CPU, precision, 3, engine).use { it.train(3) }
                gpu.newTrainingSession(backend, precision, 3, engine).use { session ->
                    assertEquals(TrainingBackend.CUDA, session.info.backend)
                    assertEquals("tensorflow:/device:GPU:0", session.info.identity)
                    assertTrue(session.info.kernelVersion.startsWith("tensorflow-"))
                    assertEquals(precision.name, session.info.precision)
                    session.train(3)
                }
                val tolerance = if (precision == Neuro.TrainingPrecision.FP64) 1e-9 else 5e-5
                assertSmallState(cpu.exportTrainingState(), gpu.exportTrainingState(), tolerance)
                cpu.trainEpoch(); gpu.trainEpoch()
                assertSmallState(cpu.exportTrainingState(), gpu.exportTrainingState(), tolerance)
                assertEquals(cpu.statistics().epochsTrained, gpu.statistics().epochsTrained)
            }
    }
}
