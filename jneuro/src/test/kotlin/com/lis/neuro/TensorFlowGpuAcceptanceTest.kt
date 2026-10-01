package com.lis.neuro

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.Timeout

/** A missing TensorFlow GPU must fail this explicit hardware task. */
@Tag("cuda")
@Timeout(180)
class TensorFlowGpuAcceptanceTest {
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
