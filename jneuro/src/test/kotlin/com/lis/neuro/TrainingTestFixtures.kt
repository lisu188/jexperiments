package com.lis.neuro

import org.junit.jupiter.api.Assertions.*

internal fun preparedSmall(shape: IntArray, parameters: Neuro.HyperParameters, samples: Int = 19): Neuro = Neuro(shape, parameters).also { model ->
    repeat(samples) { sample -> model.addTrainingSample(doubleArrayOf((sample % 7) / 7.0, (sample % 5) / 5.0), doubleArrayOf((sample % 2).toDouble())) }
}
internal fun smallOrders(epochs: Int, samples: Int): Array<IntArray> = Array(epochs) { epoch ->
    IntArray(samples) { index -> (samples - 1 - index + epoch * 3) % samples }
}
internal fun assertSmallState(expected: NeuroTrainingState, actual: NeuroTrainingState, tolerance: Double) {
    assertArrayEquals(expected.topology, actual.topology)
    val left = expected.weights + expected.biases + expected.weightVelocity + expected.biasVelocity
    val right = actual.weights + actual.biases + actual.weightVelocity + actual.biasVelocity
    for (index in left.indices) assertArrayEquals(left[index], right[index], tolerance, "Parameter/momentum buffer $index")
    assertArrayEquals(expected.inputs, actual.inputs); assertArrayEquals(expected.targets, actual.targets)
}

/** Real TensorFlow arithmetic with deterministic failure/cancellation hooks for lifecycle tests. */
internal class SearchTensorFlowKernels(
    private val precision: Neuro.TrainingPrecision = Neuro.TrainingPrecision.FP64,
    private val identity: String = "tensorflow-test"
) {
    val threads = java.util.concurrent.ConcurrentHashMap.newKeySet<String>()
    val failedSlots = HashSet<Int>()
    val corruptSlots = HashSet<Int>()
    var beforeTrain: () -> Unit = {}
    var failClose = false
    var opened = 0
        private set
    var closed = 0
        private set
    fun open(state: NeuroTrainingState, parameters: Neuro.HyperParameters): SmallTrainingKernel {
        val index = opened++
        val delegate = TensorFlowMath.trainingKernel(state, parameters, precision, TrainingBackend.CPU, TrainingEngine.SMALL)
        return object : SmallTrainingKernel {
            override val info = delegate.info.copy(identity = identity)
            override fun train(orders: Array<IntArray>, batchSize: Int, online: Boolean): NeuroTrainingState {
                threads += Thread.currentThread().name
                beforeTrain()
                check(index !in failedSlots) { "injected TensorFlow failure" }
                return delegate.train(orders, batchSize, online).also {
                    if (index in corruptSlots) it.weights[0][0] = Double.NaN
                }
            }
            override fun close() { delegate.close(); closed++; check(!failClose) { "injected cleanup failure" } }
        }
    }
}
