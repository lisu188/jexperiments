package com.lis.neuro

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.Timeout

/** Explicit native experiment acceptance; a missing library is a failure, never a skip. */
@Tag("native-small")
@Timeout(120)
class NativeSmallAcceptanceTest {
    @Test fun scalarAvx2AndTailModelsMatchJvmForOnlineRaggedAndMaximumChunks() {
        for (mode in Neuro.SigmoidMode.entries) for (models in listOf(1, 4, 5, 16)) {
            val hp = Neuro.HyperParameters(0.11, 0.31, 0.75, 42, Neuro.Kernel.AUTO, mode)
            val initial = Array(models) { model ->
                val state = preparedSmall(intArrayOf(2, 8, 8, 8, 1), hp.copy(seed = hp.seed + model)).exportTrainingState()
                SmallCpuTraining(state, hp, Neuro.TrainingPrecision.FP64, 0).use { it.train(smallOrders(2, 19), 7, false) }
            }
            for ((batch, online) in listOf(1 to true, 1 to false, 7 to false, 64 to false)) {
                val epochs = if (batch == 64) 64 else 3
                val orders = Array(models) { model -> Array(epochs) { epoch -> smallOrders(model + epoch + 1, 19).last() } }
                NativeSmallCohort(initial, hp).use { native ->
                    val actual = native.train(orders, batch, online)
                    for (model in initial.indices) SmallCpuTraining(initial[model], hp, Neuro.TrainingPrecision.FP64, 0).use { reference ->
                        assertSmallState(reference.train(orders[model], batch, online), actual[model], 1e-10)
                    }
                    val continuation = Array(models) { smallOrders(2, 19) }
                    val continued = native.train(continuation, 1, true)
                    for (model in actual.indices) SmallCpuTraining(actual[model], hp, Neuro.TrainingPrecision.FP64, 0).use { reference ->
                        assertSmallState(reference.train(continuation[model], 1, true), continued[model], 1e-10)
                    }
                }
            }
        }
    }

    @Test fun forcedScalarAndRuntimeDispatchProduceIdenticalFullState() {
        val hp = Neuro.HyperParameters(0.11, 0.31, 0.75, 71, sigmoidMode = Neuro.SigmoidMode.FAST)
        val states = Array(5) { model -> preparedSmall(intArrayOf(2, 8, 8, 8, 1), hp.copy(seed = model + 1L)).exportTrainingState() }
        val orders = Array(5) { smallOrders(7, 19) }
        NativeSmallCohort(states, hp, allowSimd = false).use { scalar ->
            NativeSmallCohort(states, hp).use { dispatched ->
                assertEquals(0, scalar.info.simdBits)
                val expected = scalar.train(orders, 7, false)
                val actual = dispatched.train(orders, 7, false)
                states.indices.forEach { assertSmallState(expected[it], actual[it], 0.0) }
            }
        }
    }
}
