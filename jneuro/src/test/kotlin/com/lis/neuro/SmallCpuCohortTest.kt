package com.lis.neuro

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test
import com.lis.neuro.SmallCpuTrainingTest.Companion.assertState
import com.lis.neuro.SmallCpuTrainingTest.Companion.model
import com.lis.neuro.SmallCpuTrainingTest.Companion.orders

class SmallCpuCohortTest {
    @Test fun modelLanesMatchIndependentKernelsForEveryShapeAndPrecision() {
        for (shape in SmallCpuTrainingTest.shapes()) for (precision in Neuro.TrainingPrecision.entries) for (online in listOf(true, false)) {
            val models = Array(3) { model(shape, 19L + it) }
            val states = Array(models.size) { models[it].exportTrainingState() }
            val orders = Array(models.size) { orders(states[it].samples, 2, it) }
            val hp = models[0].hyperParameters().copy(kernel = Neuro.Kernel.VECTOR)
            val expected = Array(models.size) { model ->
                SmallCpuTraining(states[model], hp, precision, 0).use { it.train(orders[model], 3, online) }
            }
            for (bits in listOf(0, 128, 256)) SmallCpuCohort(states, hp, precision, bits).use { cohort ->
                val actual = cohort.train(orders, 3, online)
                for (model in actual.indices) assertState(expected[model], actual[model], context = "${shape.contentToString()} $precision $online $bits model=$model")
                assertEquals(smallVectorBits(hp, bits), cohort.info.simdBits)
                assertEquals(TrainingEngine.SMALL, cohort.info.engine)
            }
        }
    }

    @Test fun tailsInactiveLanesAndContinuationPreserveWeightsMomentumAndPerModelData() {
        for (precision in Neuro.TrainingPrecision.entries) for (mode in Neuro.SigmoidMode.entries) for (online in listOf(true, false)) {
            val models = Array(11) { model(intArrayOf(2, 4, 8, 1), it.toLong(), mode) }
            val states = Array(models.size) { index -> models[index].exportTrainingState().let { state ->
                state.copy(inputs = DoubleArray(state.inputs.size) { state.inputs[it] + index * 0.01 })
            } }
            val hp = models[0].hyperParameters().copy(kernel = Neuro.Kernel.VECTOR)
            val orders = Array(models.size) { orders(states[it].samples, 2, it) }
            val active = BooleanArray(models.size) { it % 3 != 1 }
            val firstOrders = Array(models.size) { if (active[it]) arrayOf(orders[it][0]) else emptyArray() }
            for (bits in listOf(0, 128, 256)) SmallCpuCohort(states, hp, precision, bits).use { cohort ->
                val first = cohort.train(firstOrders, 64, online, active)
                for (i in states.indices) if (!active[i]) assertState(states[i], first[i])
                first[0].weights[0].fill(999.0)
                first[0].inputs.fill(999.0)
                val second = cohort.train(Array(models.size) { arrayOf(orders[it][1]) }, 64, online)
                for (i in states.indices) {
                    val requested = if (active[i]) orders[i] else arrayOf(orders[i][1])
                    val expected = SmallCpuTraining(states[i], hp, precision, 0).use { it.train(requested, 64, online) }
                    assertState(expected, second[i])
                    assertArrayEquals(states[i].inputs, second[i].inputs)
                }
                val idle = cohort.train(Array(models.size) { emptyArray() }, 1, false, BooleanArray(models.size))
                for (i in states.indices) assertState(second[i], idle[i])
            }
        }
    }

    @Test fun invalidCohortsAndOrdersFailBeforeAnyModelAdvances() {
        val m = model(intArrayOf(2, 8, 1))
        val state = m.exportTrainingState()
        val hp = m.hyperParameters()
        assertThrows(IllegalArgumentException::class.java) { SmallCpuCohort(emptyArray(), hp, Neuro.TrainingPrecision.FP64) }
        assertThrows(IllegalArgumentException::class.java) {
            SmallCpuCohort(arrayOf(state, model(intArrayOf(2, 4, 1)).exportTrainingState()), hp, Neuro.TrainingPrecision.FP64)
        }
        assertThrows(IllegalArgumentException::class.java) {
            SmallCpuCohort(arrayOf(state, state.copy(inputs = state.inputs.copyOf(4), targets = state.targets.copyOf(2))), hp, Neuro.TrainingPrecision.FP64)
        }
        for (precision in Neuro.TrainingPrecision.entries) {
            val cohort = SmallCpuCohort(arrayOf(state, state), hp, precision)
            assertEquals(0, cohort.info.simdBits)
            assertThrows(IllegalArgumentException::class.java) { cohort.train(emptyArray(), 1, false) }
            assertThrows(IllegalArgumentException::class.java) { cohort.train(arrayOf(emptyArray(), emptyArray()), 0, false) }
            assertThrows(IllegalArgumentException::class.java) { cohort.train(arrayOf(emptyArray(), emptyArray()), 1, false, booleanArrayOf(true)) }
            assertThrows(IllegalArgumentException::class.java) { cohort.train(arrayOf(orders(state.samples, 1), emptyArray()), 1, false) }
            assertThrows(IllegalArgumentException::class.java) { cohort.train(arrayOf(orders(state.samples, 1), arrayOf(IntArray(state.samples))), 1, false) }
            val unchanged = cohort.train(arrayOf(emptyArray(), emptyArray()), 1, false)
            unchanged.forEach { assertState(state, it) }
            cohort.close(); cohort.close()
            assertThrows(IllegalStateException::class.java) { cohort.train(arrayOf(emptyArray(), emptyArray()), 1, false) }
        }
        assertThrows(IllegalArgumentException::class.java) {
            SmallCpuCohort(arrayOf(state), hp.copy(learningRate = Double.MAX_VALUE), Neuro.TrainingPrecision.FP32)
        }
        assertThrows(IllegalArgumentException::class.java) {
            SmallCpuCohort(arrayOf(state.copy(targets = state.targets.copyOf().also { it[0] = Double.MAX_VALUE })), hp, Neuro.TrainingPrecision.FP32)
        }
    }
}
