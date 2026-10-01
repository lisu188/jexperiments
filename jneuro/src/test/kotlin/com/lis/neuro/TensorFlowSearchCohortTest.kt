package com.lis.neuro

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

class TensorFlowSearchCohortTest {
    private val hp = Neuro.HyperParameters(0.07, 0.4, 1.3, 42)
    private val first = intArrayOf(4, 0, 3, 1, 2)
    private val second = intArrayOf(2, 1, 4, 0, 3)

    @Test fun mixedWidthsPreserveIndependentUpdatesScoringAndBestStates() {
        for (precision in Neuro.TrainingPrecision.entries) for (mode in Neuro.SigmoidMode.entries) for (batch in intArrayOf(1, 3, Int.MAX_VALUE)) {
            val parameters = hp.withSigmoidMode(mode)
            val states = arrayOf(fixture(1), fixture(3), fixture(2))
            val validation = states[0].targets.map { 1.0 - it }.toDoubleArray()
            val tolerance = if (precision == Neuro.TrainingPrecision.FP64) 2e-13 else 3e-6
            val kernels = states.map { TensorFlowMath.trainingKernel(it, parameters, precision, TrainingBackend.CPU) }
            try {
                TensorFlowMath.searchCohort(states, parameters, precision, TrainingBackend.CPU, states[0].inputs, validation, batch).use { cohort ->
                    assertArrayEquals(intArrayOf(2, 3, 2), cohort.paddedTopology)
                    assertEquals(3, cohort.size)
                    assertTrue(cohort.info.kernelVersion.endsWith("-batched-v1-t1"))
                    assertEquals(precision.name, cohort.info.precision)
                    val best = Array(states.size) { kernels[it].train(emptyArray(), batch, batch == 1) }
                    val bestScores = cohort.score().bestScore
                    val bestEpochs = IntArray(states.size)
                    var epoch = 0
                    for (chunk in listOf(arrayOf(first), arrayOf(second, first))) {
                        val orders = Array(states.size) { lane -> Array(chunk.size) { chunk[it].let { row -> if (lane == 1) row.reversedArray() else row } } }
                        assertArrayEquals(BooleanArray(states.size), cohort.advance(orders))
                        epoch += chunk.size
                        val metrics = cohort.score()
                        val current = cohort.exportStates(intArrayOf(2, 0, 1))
                        for (lane in states.indices) {
                            val expected = kernels[lane].train(orders[lane], batch, batch == 1)
                            assertState(expected, current[intArrayOf(1, 2, 0)[lane]], tolerance)
                            val training = TensorFlowMath.error(expected.topology, expected.weights, expected.biases, parameters,
                                expected.inputs, expected.targets, precision)
                            val score = TensorFlowMath.error(expected.topology, expected.weights, expected.biases, parameters,
                                expected.inputs, validation, precision)
                            assertEquals(training, metrics.trainingRmse[lane], tolerance)
                            assertEquals(score, metrics.validationRmse[lane], tolerance)
                            if (score < bestScores[lane]) { bestScores[lane] = score; best[lane] = expected; bestEpochs[lane] = epoch }
                            assertEquals(bestScores[lane], metrics.bestScore[lane], tolerance)
                            assertEquals(bestEpochs[lane], metrics.bestEpoch[lane])
                            assertEquals(epoch, metrics.epochs[lane])
                        }
                        val snapshots = cohort.exportStates(intArrayOf(0, 1, 2), best = true)
                        snapshots.indices.forEach { assertState(best[it], snapshots[it], tolerance) }
                    }
                    assertTrue(cohort.exportStates(intArrayOf()).isEmpty())
                }
            } finally { kernels.forEach { it.close() } }
        }
    }

    @Test fun inactiveLanesFreezeAllOptimizerStateAndScoreOnlyAtRequestedBoundaries() {
        val states = arrayOf(fixture(1), fixture(3))
        TensorFlowMath.searchCohort(states, hp, Neuro.TrainingPrecision.FP64, TrainingBackend.CPU, doubleArrayOf(), doubleArrayOf(), 3).use { cohort ->
            val initial = cohort.exportStates(intArrayOf(0, 1))
            val firstScore = cohort.score(booleanArrayOf(true, false))
            assertEquals(Double.POSITIVE_INFINITY, firstScore.bestScore[1])
            assertArrayEquals(BooleanArray(2), cohort.advance(arrayOf(arrayOf(first), emptyArray()), booleanArrayOf(true, false)))
            assertState(initial[1], cohort.exportState(1), 0.0)
            val scored = cohort.score(booleanArrayOf(true, false))
            assertArrayEquals(intArrayOf(1, 0), scored.epochs)
            assertEquals(Double.POSITIVE_INFINITY, scored.bestScore[1])
            assertArrayEquals(scored.trainingRmse, scored.validationRmse)
            val paused = cohort.exportState(0)
            cohort.advance(arrayOf(emptyArray(), arrayOf(second)), booleanArrayOf(false, true))
            assertState(paused, cohort.exportState(0), 0.0)
            assertTrue(cohort.score(booleanArrayOf(false, true)).bestScore[1].isFinite())
            assertArrayEquals(BooleanArray(2), cohort.advance(arrayOf(emptyArray(), emptyArray()), booleanArrayOf(false, false)))
            assertArrayEquals(intArrayOf(1, 1), cohort.score().epochs)
        }
    }

    @Test fun deepMixedWidthsKeepEveryBackpropagationLayerIndependent() {
        val states = arrayOf(fixture(1, 3, 2, 4), fixture(4, 1, 3, 2), fixture(2, 4, 1, 3))
        for (precision in Neuro.TrainingPrecision.entries) for (mode in Neuro.SigmoidMode.entries) {
            val parameters = hp.withSigmoidMode(mode)
            TensorFlowMath.searchCohort(states, parameters, precision, TrainingBackend.CPU, doubleArrayOf(), doubleArrayOf(), 3).use { cohort ->
                val orders = Array(states.size) { arrayOf(first, second) }
                cohort.advance(orders)
                val actual = cohort.exportStates(intArrayOf(0, 1, 2))
                for (lane in states.indices) TensorFlowMath.trainingKernel(states[lane], parameters, precision, TrainingBackend.CPU).use { independent ->
                    assertState(independent.train(orders[lane], 3, false), actual[lane],
                        if (precision == Neuro.TrainingPrecision.FP64) 2e-13 else 3e-6)
                }
            }
        }
    }

    @Test fun lateNonFiniteLaneRollsBackWholeChunkWithoutPoisoningSuccessfulSibling() {
        val bad = NeuroTrainingState(intArrayOf(1, 1), arrayOf(doubleArrayOf(0.0)), arrayOf(doubleArrayOf(0.0)),
            arrayOf(doubleArrayOf(0.0)), arrayOf(doubleArrayOf(0.0)), doubleArrayOf(0.0, 1.0), doubleArrayOf(0.5, Double.MAX_VALUE))
        val good = bad.copy(weights = arrayOf(doubleArrayOf(1000.0)), biases = arrayOf(doubleArrayOf(1000.0)),
            weightVelocity = arrayOf(doubleArrayOf(1.0)), biasVelocity = arrayOf(doubleArrayOf(2.0)))
        val parameters = Neuro.HyperParameters(Double.MAX_VALUE, 0.2, 1.0, 42)
        val orders = arrayOf(intArrayOf(0, 1), intArrayOf(1, 0))
        TensorFlowMath.trainingKernel(good, parameters, Neuro.TrainingPrecision.FP64, TrainingBackend.CPU).use { reference ->
            TensorFlowMath.searchCohort(arrayOf(bad, good), parameters, Neuro.TrainingPrecision.FP64, TrainingBackend.CPU,
                doubleArrayOf(), doubleArrayOf()).use { cohort ->
                assertArrayEquals(booleanArrayOf(true, false), cohort.advance(arrayOf(orders, orders)))
                assertState(bad, cohort.exportState(0), 0.0)
                assertState(bad, cohort.exportState(0, best = true), 0.0)
                assertState(reference.train(orders, 1, true), cohort.exportState(1), 0.0)
                assertArrayEquals(booleanArrayOf(true, false), cohort.advance(arrayOf(emptyArray(), arrayOf(intArrayOf(0, 1)))))
                assertState(reference.train(arrayOf(intArrayOf(0, 1)), 1, true), cohort.exportState(1), 0.0)
            }
        }
    }

    @Test fun explicitPaddingPreservesReplayAndConcurrentCallsSerializeCompleteChunks() {
        val state = fixture(1)
        val geometry = intArrayOf(2, 5, 2)
        val cohort = TensorFlowMath.searchCohort(arrayOf(state, fixture(3)), hp, Neuro.TrainingPrecision.FP64, TrainingBackend.CPU,
            doubleArrayOf(), doubleArrayOf(), 3, geometry)
        try {
            geometry[1] = 7
            assertArrayEquals(intArrayOf(2, 5, 2), cohort.paddedTopology)
            val copy = cohort.paddedTopology; copy[1] = 9
            val executor = Executors.newFixedThreadPool(2)
            try {
                val futures = (0..1).map { executor.submit<BooleanArray> { cohort.advance(arrayOf(arrayOf(first), emptyArray()), booleanArrayOf(true, false)) } }
                futures.forEach { assertArrayEquals(BooleanArray(2), it.get(30, TimeUnit.SECONDS)) }
            } finally { executor.shutdownNow() }
            TensorFlowMath.searchCohort(arrayOf(state, state), hp, Neuro.TrainingPrecision.FP64, TrainingBackend.CPU,
                doubleArrayOf(), doubleArrayOf(), 3, intArrayOf(2, 5, 2)).use { replay ->
                replay.advance(arrayOf(arrayOf(first, first), emptyArray()), booleanArrayOf(true, false))
                assertState(cohort.exportState(0), replay.exportState(0), 0.0)
                assertArrayEquals(intArrayOf(2, 0), replay.score().epochs)
            }
        } finally { cohort.close(); cohort.close() }
        assertThrows(IllegalStateException::class.java) { cohort.score() }
        assertThrows(IllegalStateException::class.java) { cohort.exportState(0) }
        assertThrows(IllegalStateException::class.java) { cohort.advance(arrayOf(emptyArray(), emptyArray())) }
    }

    @Test fun invalidPreflightIsRecoverableAndBadDatasetsAndShapesAreRejected() {
        val state = fixture(1)
        fun create(states: Array<NeuroTrainingState> = arrayOf(state), precision: Neuro.TrainingPrecision = Neuro.TrainingPrecision.FP64,
                   x: DoubleArray = doubleArrayOf(), y: DoubleArray = doubleArrayOf(), batch: Int = 1, shape: IntArray? = null) =
            TensorFlowMath.searchCohort(states, hp, precision, TrainingBackend.CPU, x, y, batch, shape)
        assertThrows(IllegalArgumentException::class.java) { create(emptyArray()) }
        assertThrows(IllegalArgumentException::class.java) { create(Array(1025) { state }) }
        assertThrows(IllegalArgumentException::class.java) { create(batch = 0) }
        assertThrows(IllegalArgumentException::class.java) { create(arrayOf(state.copy(inputs = doubleArrayOf(), targets = doubleArrayOf()))) }
        assertThrows(IllegalArgumentException::class.java) { create(arrayOf(state, state.copy(inputs = state.inputs.reversedArray()))) }
        assertThrows(IllegalArgumentException::class.java) { create(x = doubleArrayOf(1.0)) }
        assertThrows(IllegalArgumentException::class.java) { create(x = doubleArrayOf(Double.NaN, 0.0), y = doubleArrayOf(0.0, 0.0)) }
        assertThrows(IllegalArgumentException::class.java) { create(precision = Neuro.TrainingPrecision.FP32,
            x = doubleArrayOf(Double.MAX_VALUE, 0.0), y = doubleArrayOf(0.0, 0.0)) }
        assertThrows(IllegalArgumentException::class.java) { create(shape = intArrayOf(2, 0, 2)) }
        assertThrows(IllegalArgumentException::class.java) { create(shape = intArrayOf(2, 3)) }
        assertThrows(IllegalArgumentException::class.java) { create(shape = intArrayOf(2, Int.MAX_VALUE, 2)) }
        create(arrayOf(state, fixture(3))).use { cohort ->
            assertThrows(IllegalArgumentException::class.java) { cohort.advance(emptyArray()) }
            assertThrows(IllegalArgumentException::class.java) { cohort.advance(arrayOf(arrayOf(first), emptyArray())) }
            assertThrows(IllegalArgumentException::class.java) { cohort.advance(arrayOf(arrayOf(intArrayOf(0, 0, 1, 2, 3)), arrayOf(first))) }
            assertThrows(IllegalArgumentException::class.java) { cohort.advance(arrayOf(arrayOf(intArrayOf(0, 1, 2, 3, 5)), arrayOf(first))) }
            assertThrows(IllegalArgumentException::class.java) { cohort.advance(Array(2) { Array(65) { first } }) }
            assertThrows(IllegalArgumentException::class.java) { cohort.advance(arrayOf(arrayOf(first), arrayOf(first, first)), booleanArrayOf(true, false)) }
            assertThrows(IllegalArgumentException::class.java) { cohort.score(booleanArrayOf()) }
            assertThrows(IllegalArgumentException::class.java) { cohort.exportState(-1) }
            assertArrayEquals(BooleanArray(2), cohort.advance(arrayOf(arrayOf(first), arrayOf(second))))
            assertArrayEquals(intArrayOf(1, 1), cohort.score().epochs)
        }
    }

    private fun fixture(width: Int, vararg more: Int): NeuroTrainingState {
        val shape = intArrayOf(2, width, *more, 2)
        return NeuroTrainingState(shape, Array(shape.size - 1) { layer -> DoubleArray(shape[layer] * shape[layer + 1]) { ((it + width) % 7 - 3) * 0.07 } },
            Array(shape.size - 1) { layer -> DoubleArray(shape[layer + 1]) { (it - 1) * 0.03 } },
            Array(shape.size - 1) { layer -> DoubleArray(shape[layer] * shape[layer + 1]) { (it % 3 - 1) * 0.001 } },
            Array(shape.size - 1) { layer -> DoubleArray(shape[layer + 1]) { (it % 3 - 1) * 0.002 } },
            doubleArrayOf(0.1, 0.9, 0.2, 0.8, 0.5, 0.3, 0.9, 0.2, 0.7, 0.4),
            doubleArrayOf(0.0, 1.0, 0.2, 0.8, 1.0, 0.0, 0.5, 0.5, 0.9, 0.1))
    }

    private fun assertState(expected: NeuroTrainingState, actual: NeuroTrainingState, tolerance: Double) {
        assertArrayEquals(expected.topology, actual.topology)
        for ((left, right) in listOf(expected.weights to actual.weights, expected.biases to actual.biases,
            expected.weightVelocity to actual.weightVelocity, expected.biasVelocity to actual.biasVelocity))
            left.indices.forEach { assertArrayEquals(left[it], right[it], tolerance) }
        assertArrayEquals(expected.inputs, actual.inputs)
        assertArrayEquals(expected.targets, actual.targets)
    }
}
