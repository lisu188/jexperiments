package com.lis.neuro

import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test
import kotlin.math.exp
import kotlin.math.sqrt

/** Independent small analytic fixtures exercise real TensorFlow sessions and tensors. */
class TensorFlowMathTest {
    private val parameters = Neuro.HyperParameters(0.07, 0.4, 1.3, 42)

    @Test fun forwardIntermediatesAndRmseMatchIndependentMultiOutputCalculation() {
        val state = fixture()
        val expected = forward(state, parameters, state.inputs, state.samples)
        val expectedRmse = sqrt(expected.last().indices.sumOf { index ->
            val residual = state.targets[index] - expected.last()[index]
            residual * residual
        } / state.targets.size)
        for (precision in Neuro.TrainingPrecision.entries) {
            val tolerance = tolerance(precision)
            val actual = TensorFlowMath.evaluate(state.topology, state.weights, state.biases, parameters,
                state.inputs, state.samples, precision)
            assertEquals(state.topology.size, actual.size)
            for (layer in actual.indices) {
                assertEquals(state.samples * state.topology[layer], actual[layer].size)
                assertArrayEquals(expected[layer], actual[layer], tolerance, "layer $layer ($precision)")
            }
            assertArrayEquals(expected.last(), TensorFlowMath.predict(state.topology, state.weights, state.biases,
                parameters, state.inputs, state.samples, precision), tolerance)
            assertEquals(expectedRmse, TensorFlowMath.error(state.topology, state.weights, state.biases,
                parameters, state.inputs, state.targets, precision), tolerance)
        }
    }

    @Test fun onlineRaggedAndOversizedBatchesPreserveGradientsAndMomentumInBothPrecisions() {
        val initial = fixture()
        val orders = arrayOf(intArrayOf(4, 0, 3, 1, 2), intArrayOf(2, 1, 4, 0, 3))
        for (precision in Neuro.TrainingPrecision.entries) for ((batch, online) in listOf(4 to true, 3 to false, 100 to false)) {
            val expected = advance(initial, parameters, orders, batch, online)
            TensorFlowMath.trainingKernel(initial, parameters, precision, TrainingBackend.CPU).use { kernel ->
                val actual = kernel.train(orders, batch, online)
                assertState(expected, actual, tolerance(precision))
                assertEquals(precision.name, kernel.info.precision)
                assertEquals(TrainingBackend.CPU, kernel.info.backend)
                assertTrue(kernel.info.name.startsWith("TensorFlow "))
                if (precision == Neuro.TrainingPrecision.FP32) {
                    assertTrue(actual.weights.all { row -> row.all { it == it.toFloat().toDouble() } })
                    assertTrue(actual.weightVelocity.all { row -> row.all { it == it.toFloat().toDouble() } })
                }
                TensorFlowMath.trainingKernel(initial, parameters, precision, TrainingBackend.CPU).use { split ->
                    split.train(arrayOf(orders[0]), batch, online)
                    assertState(actual, split.train(arrayOf(orders[1]), batch, online), 0.0)
                }
            }
        }
    }

    @Test fun fastActivationRemainsExplicitAndStableForExtremeInputsAndTraining() {
        var approximationDiffers = false
        for (value in doubleArrayOf(-1000.0, -745.0, -104.0, -16.0, -1.2, -0.5, 0.0, 0.5, 1.2, 16.0, 104.0, 745.0, 1000.0)) {
            val exact = TensorFlowMath.sigmoid(value, Neuro.SigmoidMode.EXACT)
            val fast = TensorFlowMath.sigmoid(value, Neuro.SigmoidMode.FAST)
            assertEquals(1.0 / (1.0 + exp(-value)), exact, 1e-14)
            assertEquals(exact, fast, 2e-4)
            assertTrue(fast.isFinite() && fast in 0.0..1.0)
            if (exact != fast) approximationDiffers = true
        }
        assertTrue(approximationDiffers, "FAST must retain its explicitly selected approximation")
        val state = fixture()
        val fast = parameters.withSigmoidMode(Neuro.SigmoidMode.FAST)
        for (precision in Neuro.TrainingPrecision.entries) {
            val exactPrediction = TensorFlowMath.predict(state.topology, state.weights, state.biases, parameters,
                state.inputs, state.samples, precision)
            val fastPrediction = TensorFlowMath.predict(state.topology, state.weights, state.biases, fast,
                state.inputs, state.samples, precision)
            assertArrayEquals(exactPrediction, fastPrediction, 2e-4)
            val orders = arrayOf(intArrayOf(4, 3, 2, 1, 0))
            TensorFlowMath.trainingKernel(state, fast, precision, TrainingBackend.CPU).use { kernel ->
                assertState(advance(state, parameters, orders, 3, false), kernel.train(orders, 3, false), 2e-4)
                assertEquals("FAST", kernel.info.sigmoid)
            }
        }
    }

    @Test fun multiEpochChunksPreserveRaggedBoundariesAcrossBatchAndPrecisionChanges() {
        val initial = fixture()
        val orders = arrayOf(intArrayOf(4, 0, 3, 1, 2), intArrayOf(2, 1, 4, 0, 3),
            intArrayOf(1, 3, 0, 4, 2), intArrayOf(0, 4, 2, 3, 1))
        for (precision in Neuro.TrainingPrecision.entries) for (mode in Neuro.SigmoidMode.entries) {
            val hp = parameters.withSigmoidMode(mode)
            TensorFlowMath.trainingKernel(initial, hp, precision, TrainingBackend.CPU).use { chunked ->
                TensorFlowMath.trainingKernel(initial, hp, precision, TrainingBackend.CPU).use { separate ->
                    var expected = initial
                    var next = 0
                    for ((epochs, batch, online) in listOf(Triple(2, 3, false), Triple(1, 2, false), Triple(1, 4, true))) {
                        val group = orders.copyOfRange(next, next + epochs)
                        val actual = chunked.train(group, batch, online)
                        for (order in group) expected = separate.train(arrayOf(order), batch, online)
                        assertState(expected, actual, 0.0)
                        assertState(actual, chunked.train(emptyArray(), batch, online), 0.0)
                        next += epochs
                    }
                    assertEquals(orders.size, next)
                }
            }
        }
    }

    @Test fun emptyDatasetSnapshotsDoNotEnterTheTrainingLoop() {
        val state = fixture().copy(inputs = doubleArrayOf(), targets = doubleArrayOf())
        for (precision in Neuro.TrainingPrecision.entries) {
            TensorFlowMath.trainingKernel(state, parameters, precision, TrainingBackend.CPU).use { kernel ->
                assertState(state, kernel.train(emptyArray(), 3, false), tolerance(precision))
                assertThrows(IllegalArgumentException::class.java) { kernel.train(arrayOf(intArrayOf()), 1, true) }
                assertState(state, kernel.train(emptyArray(), 1, true), tolerance(precision))
            }
        }
    }

    @Test fun failureAfterAnEarlierFiniteBatchKeepsTheImportedCheckpointUnchanged() {
        val state = NeuroTrainingState(intArrayOf(2, 1), arrayOf(doubleArrayOf(0.0, 0.0)), arrayOf(doubleArrayOf(0.0)),
            arrayOf(doubleArrayOf(0.001, 0.002)), arrayOf(doubleArrayOf(0.003)),
            doubleArrayOf(0.2, 0.3, 0.4, 0.5), doubleArrayOf(0.5, Double.MAX_VALUE))
        val saved = copy(state)
        TensorFlowMath.trainingKernel(state, Neuro.HyperParameters(Double.MAX_VALUE, 0.1, 1.0, 42),
            Neuro.TrainingPrecision.FP64, TrainingBackend.CPU).use { kernel ->
            assertThrows(IllegalStateException::class.java) { kernel.train(arrayOf(intArrayOf(0, 1)), 1, false) }
            assertThrows(IllegalStateException::class.java) { kernel.train(emptyArray(), 1, false) }
            assertState(saved, state, 0.0)
            assertArrayEquals(saved.inputs, state.inputs)
            assertArrayEquals(saved.targets, state.targets)
        }
    }

    @Test fun emptyBatchesAndDiagnosticPrimitivesHaveDefinedResults() {
        val state = fixture()
        for (precision in Neuro.TrainingPrecision.entries) {
            assertArrayEquals(doubleArrayOf(), TensorFlowMath.predict(state.topology, state.weights, state.biases,
                parameters, doubleArrayOf(), 0, precision))
            val layers = TensorFlowMath.evaluate(state.topology, state.weights, state.biases, parameters,
                doubleArrayOf(), 0, precision)
            assertEquals(state.topology.size, layers.size)
            assertTrue(layers.all { it.isEmpty() })
            assertTrue(TensorFlowMath.error(state.topology, state.weights, state.biases, parameters,
                doubleArrayOf(), doubleArrayOf(), precision).isNaN())
        }
        assertArrayEquals(doubleArrayOf(6.0, -8.0, 0.0),
            TensorFlowMath.contributions(doubleArrayOf(2.0, -4.0, 0.0), doubleArrayOf(3.0, 2.0, 7.0)))
        assertEquals(-2.0, TensorFlowMath.sum(doubleArrayOf(6.0, -8.0, 0.0)))
        assertEquals(25.0, TensorFlowMath.squaredNorm(doubleArrayOf(3.0, -4.0)))
        assertEquals(5.0, TensorFlowMath.norm(doubleArrayOf(3.0, -4.0)))
        assertArrayEquals(doubleArrayOf(), TensorFlowMath.contributions(doubleArrayOf(), doubleArrayOf()))
        assertEquals(0.0, TensorFlowMath.sum(doubleArrayOf()))
        assertEquals(0.0, TensorFlowMath.squaredNorm(doubleArrayOf()))
        assertEquals(0.0, TensorFlowMath.norm(doubleArrayOf()))
        assertThrows(IllegalArgumentException::class.java) {
            TensorFlowMath.contributions(doubleArrayOf(1.0), doubleArrayOf())
        }
    }

    @Test fun cachedGraphsDoNotRetainMutableShapeParametersInputsOrOutputBuffers() {
        TensorFlowMath.clearInferenceCache()
        val state = fixture()
        val saved = copy(state)
        val expected = forward(saved, parameters, saved.inputs, saved.samples).last()
        val first = TensorFlowMath.predict(state.topology, state.weights, state.biases, parameters, state.inputs, state.samples)
        state.topology[1] = 97
        state.weights[0].fill(Double.NaN); state.biases[0].fill(Double.NaN); state.inputs.fill(Double.NaN)
        first.fill(Double.NaN)
        assertArrayEquals(expected, TensorFlowMath.predict(saved.topology, saved.weights, saved.biases, parameters,
            saved.inputs, saved.samples), 1e-13)
        // More than eight graph keys evicts old entries; each still uses the current model's feeds.
        for (width in 1..12) {
            val current = fixture(width)
            assertArrayEquals(forward(current, parameters, current.inputs, current.samples).last(),
                TensorFlowMath.predict(current.topology, current.weights, current.biases, parameters,
                    current.inputs, current.samples), 1e-13)
        }
        TensorFlowMath.clearInferenceCache()
        TensorFlowMath.clearInferenceCache()
        assertArrayEquals(expected, TensorFlowMath.predict(saved.topology, saved.weights, saved.biases, parameters,
            saved.inputs, saved.samples), 1e-13)
    }

    @Test fun trainingOwnsItsImportedStateAndEveryUnsharedSnapshotIsDetached() {
        for (precision in Neuro.TrainingPrecision.entries) {
            val state = fixture()
            val saved = copy(state)
            TensorFlowMath.trainingKernel(state, parameters, precision, TrainingBackend.CPU).use { kernel ->
                state.topology[1] = 97
                state.inputs.fill(Double.NaN); state.targets.fill(Double.NaN)
                state.weights[0].fill(Double.NaN); state.biasVelocity[0].fill(Double.NaN)
                val first = kernel.train(emptyArray(), 1, true)
                assertState(saved, first, tolerance(precision))
                assertArrayEquals(saved.inputs, first.inputs)
                assertArrayEquals(saved.targets, first.targets)
                first.topology[1] = 98; first.weights[0].fill(Double.NaN)
                first.inputs.fill(Double.NaN); first.targets.fill(Double.NaN)
                val second = kernel.train(emptyArray(), 1, true)
                assertState(saved, second, tolerance(precision))
                assertArrayEquals(saved.inputs, second.inputs)
                assertArrayEquals(saved.targets, second.targets)
                assertNotSame(first.inputs, second.inputs)
                assertNotSame(first.targets, second.targets)
            }
            val shared = fixture().copy(sharedDataset = true)
            TensorFlowMath.trainingKernel(shared, parameters, precision, TrainingBackend.CPU).use { kernel ->
                val snapshot = kernel.train(emptyArray(), 1, true)
                assertSame(shared.inputs, snapshot.inputs)
                assertSame(shared.targets, snapshot.targets)
                assertNotSame(shared.topology, snapshot.topology)
            }
        }
    }

    @Test fun rejectedArgumentsDoNotAdvanceOrPoisonTheKernelAndCloseIsIdempotent() {
        val state = fixture()
        val kernel = TensorFlowMath.trainingKernel(state, parameters, Neuro.TrainingPrecision.FP64, TrainingBackend.CPU)
        try {
            assertThrows(IllegalArgumentException::class.java) { kernel.train(arrayOf(intArrayOf(0, 1, 2, 3, 4)), 0, true) }
            for (invalid in listOf(intArrayOf(0), intArrayOf(-1, 1, 2, 3, 4), intArrayOf(0, 1, 2, 3, 5), intArrayOf(0, 0, 2, 3, 4))) {
                assertThrows(IllegalArgumentException::class.java) { kernel.train(arrayOf(invalid), 2, false) }
                assertState(state, kernel.train(emptyArray(), 1, true), 0.0)
            }
            val order = arrayOf(intArrayOf(4, 2, 0, 1, 3))
            assertState(advance(state, parameters, order, 2, false), kernel.train(order, 2, false), 1e-13)
        } finally { kernel.close(); kernel.close() }
        assertThrows(IllegalStateException::class.java) { kernel.train(emptyArray(), 1, true) }
    }

    @Test fun nonfiniteTrainingPoisonsTheKernelWithoutMutatingItsInputCheckpoint() {
        val state = NeuroTrainingState(intArrayOf(2, 1), arrayOf(doubleArrayOf(0.0, 0.0)), arrayOf(doubleArrayOf(0.0)),
            arrayOf(doubleArrayOf(0.0, 0.0)), arrayOf(doubleArrayOf(0.0)), doubleArrayOf(0.2, 0.3), doubleArrayOf(Double.MAX_VALUE))
        val saved = copy(state)
        TensorFlowMath.trainingKernel(state, Neuro.HyperParameters(Double.MAX_VALUE, 0.1, 1.0, 42),
            Neuro.TrainingPrecision.FP64, TrainingBackend.CPU).use { kernel ->
            assertThrows(IllegalStateException::class.java) { kernel.train(arrayOf(intArrayOf(0)), 1, true) }
            assertThrows(IllegalStateException::class.java) { kernel.train(emptyArray(), 1, true) }
            assertState(saved, state, 0.0)
            assertArrayEquals(saved.inputs, state.inputs)
            assertArrayEquals(saved.targets, state.targets)
        }
        assertThrows(IllegalArgumentException::class.java) {
            TensorFlowMath.trainingKernel(state, parameters, Neuro.TrainingPrecision.FP32, TrainingBackend.CPU).close()
        }
        assertThrows(IllegalArgumentException::class.java) {
            TensorFlowMath.trainingKernel(fixture(), parameters.withBeta(Double.MAX_VALUE),
                Neuro.TrainingPrecision.FP32, TrainingBackend.CPU).close()
        }
    }

    @Test fun concurrentInferenceTrainingAndCacheCleanupKeepModelResourcesIndependent() {
        val pool = Executors.newFixedThreadPool(4)
        val start = CountDownLatch(1)
        try {
            val futures = (0 until 4).map { index -> pool.submit {
                check(start.await(10, TimeUnit.SECONDS))
                val state = fixture().also { it.weights[0][0] += index * 0.1 }
                val expected = forward(state, parameters, state.inputs, state.samples).last()
                val order = arrayOf(intArrayOf(4, 2, 0, 1, 3))
                TensorFlowMath.trainingKernel(state, parameters, Neuro.TrainingPrecision.FP64, TrainingBackend.CPU).use { kernel ->
                    repeat(4) {
                        assertArrayEquals(expected, TensorFlowMath.predict(state.topology, state.weights, state.biases,
                            parameters, state.inputs, state.samples), 1e-13)
                        assertEquals(5.0, TensorFlowMath.norm(doubleArrayOf(3.0, 4.0)))
                        if (index == 0) TensorFlowMath.clearInferenceCache()
                    }
                    assertState(advance(state, parameters, order, 3, false), kernel.train(order, 3, false), 1e-13)
                }
            } }
            start.countDown()
            futures.forEach { it.get(30, TimeUnit.SECONDS) }
        } finally {
            start.countDown(); pool.shutdownNow()
            assertTrue(pool.awaitTermination(10, TimeUnit.SECONDS))
            TensorFlowMath.clearInferenceCache()
        }
    }

    @Test fun deviceMetadataPreservesPrecisionAndExplicitGpuRequestsNeverFallBack() {
        for (precision in Neuro.TrainingPrecision.entries) for (engine in TrainingEngine.entries) {
            val cpu = TensorFlowMath.info(TrainingBackend.CPU, precision, engine, Neuro.SigmoidMode.FAST)
            val automatic = TensorFlowMath.info(TrainingBackend.AUTO, precision, engine, Neuro.SigmoidMode.FAST)
            assertEquals(cpu, automatic)
            assertEquals(TrainingBackend.CPU, cpu.backend)
            assertEquals(precision.name, cpu.precision)
            assertEquals(engine, cpu.engine)
            assertEquals("FAST", cpu.sigmoid)
            assertEquals(0, cpu.simdBits)
        }
        val state = fixture()
        val available = TensorFlowMath.isGpuAvailable()
        for (backend in listOf(TrainingBackend.CUDA, TrainingBackend.CUBLAS)) {
            if (available) {
                assertEquals("", TensorFlowMath.gpuFailure())
                assertEquals(TrainingBackend.CUDA, TensorFlowMath.info(backend, Neuro.TrainingPrecision.FP32).backend)
                assertArrayEquals(forward(state, parameters, state.inputs, state.samples).last(),
                    TensorFlowMath.predict(state.topology, state.weights, state.biases, parameters,
                        state.inputs, state.samples, Neuro.TrainingPrecision.FP32, backend), 5e-7)
                TensorFlowMath.trainingKernel(state, parameters, Neuro.TrainingPrecision.FP32, backend).use {
                    assertEquals(TrainingBackend.CUDA, it.info.backend)
                    val order = arrayOf(intArrayOf(4, 2, 0, 1, 3))
                    assertState(advance(state, parameters, order, 3, false), it.train(order, 3, false), 5e-7)
                }
            } else {
                assertTrue(TensorFlowMath.gpuFailure().isNotBlank())
                assertThrows(IllegalStateException::class.java) { TensorFlowMath.info(backend, Neuro.TrainingPrecision.FP32) }
                assertThrows(IllegalStateException::class.java) {
                    TensorFlowMath.predict(state.topology, state.weights, state.biases, parameters,
                        state.inputs, state.samples, Neuro.TrainingPrecision.FP32, backend)
                }
                assertThrows(IllegalStateException::class.java) {
                    TensorFlowMath.trainingKernel(state, parameters, Neuro.TrainingPrecision.FP32, backend).close()
                }
            }
        }
    }

    private fun fixture(width: Int = 3): NeuroTrainingState {
        val shape = intArrayOf(2, width, 2)
        val weights = Array(2) { layer -> DoubleArray(shape[layer] * shape[layer + 1]) { (it + 1) * 0.03 - 0.08 } }
        val biases = Array(2) { layer -> DoubleArray(shape[layer + 1]) { it * 0.02 - 0.03 } }
        return NeuroTrainingState(shape, weights, biases,
            Array(2) { DoubleArray(weights[it].size) { index -> (index % 3 - 1) * 0.004 } },
            Array(2) { DoubleArray(biases[it].size) { index -> (index % 3 - 1) * 0.003 } },
            doubleArrayOf(0.1, 0.7, 0.4, 0.2, 0.9, 0.8, 0.0, 1.0, 0.5, 0.3),
            doubleArrayOf(0.0, 1.0, 0.2, 0.8, 1.0, 0.0, 0.9, 0.1, 0.3, 0.7))
    }

    private fun copy(state: NeuroTrainingState) = state.copy(topology = state.topology.copyOf(),
        weights = state.weights.map { it.copyOf() }.toTypedArray(), biases = state.biases.map { it.copyOf() }.toTypedArray(),
        weightVelocity = state.weightVelocity.map { it.copyOf() }.toTypedArray(), biasVelocity = state.biasVelocity.map { it.copyOf() }.toTypedArray(),
        inputs = state.inputs.copyOf(), targets = state.targets.copyOf())

    private fun forward(state: NeuroTrainingState, hp: Neuro.HyperParameters, input: DoubleArray, count: Int): Array<DoubleArray> {
        val activations = Array(state.topology.size) { DoubleArray(count * state.topology[it]) }
        input.copyInto(activations[0])
        for (layer in state.weights.indices) {
            val width = state.topology[layer]
            val next = state.topology[layer + 1]
            for (sample in 0 until count) for (output in 0 until next) {
                var sum = state.biases[layer][output]
                for (index in 0 until width) sum += activations[layer][sample * width + index] * state.weights[layer][output * width + index]
                activations[layer + 1][sample * next + output] = 1.0 / (1.0 + exp(-hp.beta * sum))
            }
        }
        return activations
    }

    private fun advance(initial: NeuroTrainingState, hp: Neuro.HyperParameters, orders: Array<IntArray>, batchSize: Int, online: Boolean): NeuroTrainingState {
        val state = copy(initial)
        val batch = if (online) 1 else batchSize
        for (order in orders) for (start in order.indices step batch) {
            val count = minOf(batch, order.size - start)
            val gradient = Array(state.weights.size) { DoubleArray(state.weights[it].size) }
            val biasGradient = Array(state.biases.size) { DoubleArray(state.biases[it].size) }
            for (position in start until start + count) {
                val sample = order[position]
                val activations = forward(state, hp, state.inputs.copyOfRange(sample * state.topology[0], (sample + 1) * state.topology[0]), 1)
                val deltas = Array(state.weights.size) { DoubleArray(state.topology[it + 1]) }
                for (layer in state.weights.indices.reversed()) for (output in deltas[layer].indices) {
                    val activation = activations[layer + 1][output]
                    val residual = if (layer == state.weights.lastIndex) state.targets[sample * state.topology.last() + output] - activation
                        else deltas[layer + 1].indices.sumOf { next -> deltas[layer + 1][next] * state.weights[layer + 1][next * state.topology[layer + 1] + output] }
                    val delta = residual * hp.beta * activation * (1.0 - activation)
                    deltas[layer][output] = delta
                    biasGradient[layer][output] += delta
                    for (input in activations[layer].indices) gradient[layer][output * state.topology[layer] + input] += delta * activations[layer][input]
                }
            }
            for (layer in state.weights.indices) {
                for (index in state.weights[layer].indices) {
                    val next = hp.momentum * state.weightVelocity[layer][index] + hp.learningRate * gradient[layer][index] / count
                    state.weightVelocity[layer][index] = next
                    state.weights[layer][index] += next
                }
                for (index in state.biases[layer].indices) {
                    val next = hp.momentum * state.biasVelocity[layer][index] + hp.learningRate * biasGradient[layer][index] / count
                    state.biasVelocity[layer][index] = next
                    state.biases[layer][index] += next
                }
            }
        }
        return state
    }

    private fun tolerance(precision: Neuro.TrainingPrecision) = if (precision == Neuro.TrainingPrecision.FP64) 1e-13 else 5e-7

    private fun assertState(expected: NeuroTrainingState, actual: NeuroTrainingState, tolerance: Double) {
        assertArrayEquals(expected.topology, actual.topology)
        for ((left, right) in listOf(expected.weights to actual.weights, expected.biases to actual.biases,
            expected.weightVelocity to actual.weightVelocity, expected.biasVelocity to actual.biasVelocity)) {
            assertEquals(left.size, right.size)
            for (layer in left.indices) assertArrayEquals(left[layer], right[layer], tolerance)
        }
    }
}
