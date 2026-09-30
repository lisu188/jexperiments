package com.lis.neuro

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.Timeout

/** Strict device arithmetic acceptance: absent hardware is a failure, never a skipped test. */
@Tag("cuda")
@Timeout(240)
class SmallCudaAcceptanceTest {
    @Test fun everySmallTopologyMatchesScalarStateMomentumAndCpuContinuation() {
        SmallCudaDeviceService().use { service ->
            for (precision in Neuro.TrainingPrecision.entries) for (mode in Neuro.SigmoidMode.entries) {
                for (shape in shapes()) {
                    val hp = parameters(mode)
                    val original = preparedSmall(shape, hp).exportTrainingState()
                    val warm = SmallCpuTraining(original, hp, precision, 0).use { it.train(smallOrders(2, 19), 11, false) }
                    SmallCpuTraining(warm, hp, precision, 0).use { reference ->
                        val result = SmallCudaTraining(warm, hp, precision, service.openDriver()).use { gpu ->
                            assertEquals(TrainingBackend.CUDA, gpu.info.backend)
                            assertEquals(TrainingEngine.SMALL, gpu.info.engine)
                            assertEquals(precision.name, gpu.info.precision)
                            assertEquals(mode.name, gpu.info.sigmoid)
                            val expected = reference.train(smallOrders(3, 19), 11, false)
                            gpu.train(smallOrders(3, 19), 11, false).also { actual -> assertSmallState(expected, actual, tolerance(precision)) }
                        }
                        SmallCpuTraining(result, hp, precision, 0).use { resumed ->
                            assertSmallState(reference.train(smallOrders(2, 19), 1, true),
                                resumed.train(smallOrders(2, 19), 1, true), tolerance(precision))
                        }
                    }
                }
            }
        }
    }

    @Test fun onlineSingletonOversizedAndMaximumChunksPreserveUpdateSemantics() {
        SmallCudaDeviceService().use { service ->
            val shapes = listOf(intArrayOf(2, 4, 1), intArrayOf(2, 8, 8, 8, 1), intArrayOf(2, 16, 16, 16, 16, 1))
            for (precision in Neuro.TrainingPrecision.entries) for (shape in shapes) {
                for ((batch, online) in listOf(1 to true, 1 to false, 7 to false, 64 to false)) {
                    val hp = parameters(Neuro.SigmoidMode.EXACT)
                    val initial = preparedSmall(shape, hp).exportTrainingState()
                    SmallCpuTraining(initial, hp, precision, 0).use { reference ->
                        SmallCudaTraining(initial, hp, precision, service.openDriver()).use { gpu ->
                            val orders = smallOrders(if (shape.size == 3) 64 else 3, 19)
                            assertSmallState(reference.train(orders, batch, online), gpu.train(orders, batch, online), tolerance(precision))
                        }
                    }
                }
                val hp = parameters(Neuro.SigmoidMode.FAST)
                val one = preparedSmall(shape, hp, 1).exportTrainingState()
                SmallCpuTraining(one, hp, precision, 0).use { reference ->
                    SmallCudaTraining(one, hp, precision, service.openDriver()).use { gpu ->
                        assertSmallState(reference.train(smallOrders(2, 1), 64, false),
                            gpu.train(smallOrders(2, 1), 64, false), tolerance(precision))
                    }
                }
            }
        }
    }

    @Test fun cohortsKeepIndependentMomentumOrdersAndInactiveModels() {
        SmallCudaDeviceService().use { service ->
            for (precision in Neuro.TrainingPrecision.entries) {
                val hp = parameters(Neuro.SigmoidMode.FAST)
                val shape = intArrayOf(2, 4, 16, 8, 16, 1)
                val states = Array(5) { preparedSmall(shape, hp.copy(seed = hp.seed + it)).exportTrainingState() }
                val order = Array(5) { model -> if (model == 2) emptyArray() else
                    Array(3) { epoch -> smallOrders(epoch + model + 1, 19).last() } }
                SmallCudaCohort(states, hp, precision, service.openDriver()).use { gpu ->
                    val actual = gpu.train(order, 11, false, booleanArrayOf(true, true, false, true, true))
                    for (model in states.indices) {
                        val expected = if (model == 2) states[model] else SmallCpuTraining(states[model], hp, precision, 0).use {
                            it.train(order[model], 11, false)
                        }
                        assertSmallState(expected, actual[model], tolerance(precision))
                    }
                    val resumed = gpu.train(Array(5) { smallOrders(2, 19) }, 1, true)
                    for (model in states.indices) SmallCpuTraining(actual[model], hp, precision, 0).use { reference ->
                        assertSmallState(reference.train(smallOrders(2, 19), 1, true), resumed[model], tolerance(precision))
                    }
                }
            }
        }
    }

    private fun parameters(mode: Neuro.SigmoidMode) = Neuro.HyperParameters(0.11, 0.31, 0.75, 42, Neuro.Kernel.SCALAR, mode)
    private fun tolerance(precision: Neuro.TrainingPrecision) = if (precision == Neuro.TrainingPrecision.FP64) 1e-9 else 5e-5
    private fun shapes(): List<IntArray> = buildList {
        fun add(prefix: List<Int>, remaining: Int) {
            if (remaining == 0) add((listOf(2) + prefix + 1).toIntArray())
            else for (width in listOf(4, 8, 16)) add(prefix + width, remaining - 1)
        }
        for (hidden in 1..4) add(emptyList(), hidden)
    }
}
