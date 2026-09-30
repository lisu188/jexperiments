package com.lis.neuro

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

class SmallCpuTrainingTest {
    @Test fun everySupportedShapeMatchesReferenceIncludingMomentumAndPartialBatches() {
        assertEquals(120, shapes().size)
        for (shape in shapes()) for (online in listOf(true, false)) {
            val reference = model(shape)
            val state = reference.exportTrainingState()
            val orders = reference.reserveTrainingOrders(2).map { it.copyOf() }.toTypedArray()
            if (online) reference.train(2) else reference.trainMiniBatch(2, 3, 1, Neuro.BatchBackend.CPU)
            val expected = reference.exportTrainingState()
            for (bits in listOf(0, 128, 256)) SmallCpuTraining(state, reference.hyperParameters().copy(kernel = Neuro.Kernel.VECTOR),
                Neuro.TrainingPrecision.FP64, bits).use { kernel ->
                assertState(expected, kernel.train(orders, 3, online), 0.0, "${shape.contentToString()} bits=$bits online=$online")
            }
        }
    }

    @Test fun fp32UsesFloatStateAndMatchesScalarAcrossEveryShapeAndVectorWidth() {
        for (shape in shapes()) for (online in listOf(true, false)) {
            val model = model(shape)
            val initial = model.exportTrainingState()
            val orders = orders(initial.samples, 2)
            val parameters = model.hyperParameters().copy(kernel = Neuro.Kernel.VECTOR)
            val expected = SmallCpuTraining(initial, parameters, Neuro.TrainingPrecision.FP32, 0).use { it.train(orders, 4, online) }
            for (bits in listOf(128, 256)) SmallCpuTraining(initial, parameters, Neuro.TrainingPrecision.FP32, bits).use {
                val actual = it.train(orders, 4, online)
                assertState(expected, actual)
                assertTrue(actual.weights.all { row -> row.all { value -> value == value.toFloat().toDouble() } })
                assertEquals("FP32", it.info.precision)
                assertEquals(if (shape.drop(1).dropLast(1).all { it == 4 }) minOf(128, smallVectorBits(parameters, bits))
                    else smallVectorBits(parameters, bits), it.info.simdBits)
                assertEquals(TrainingEngine.SMALL, it.info.engine)
            }
            val exact = SmallCpuTraining(initial, parameters, Neuro.TrainingPrecision.FP64, 0).use { it.train(orders, 4, online) }
            assertState(exact, expected, 2e-6)
        }
    }

    @Test fun chunksRetainMomentumAndSnapshotsAreDetachedForBothPrecisionsAndSigmoids() {
        for (precision in Neuro.TrainingPrecision.entries) for (mode in Neuro.SigmoidMode.entries) for (online in listOf(true, false)) {
            val model = model(intArrayOf(2, 4, 8, 16, 4, 1), mode = mode)
            val state = model.exportTrainingState()
            val order = orders(state.samples, 3)
            val hp = model.hyperParameters().copy(kernel = Neuro.Kernel.VECTOR)
            val expected = SmallCpuTraining(state, hp, precision).use { it.train(order, 64, online) }
            SmallCpuTraining(state, hp, precision).use { chunked ->
                val first = chunked.train(arrayOf(order[0]), 64, online)
                first.weights[0].fill(999.0)
                first.inputs.fill(999.0)
                val actual = chunked.train(arrayOf(order[1], order[2]), 64, online)
                assertState(expected, actual)
                assertArrayEquals(state.inputs, actual.inputs)
                assertArrayEquals(state.targets, actual.targets)
                val empty = chunked.train(emptyArray(), 1, false)
                assertState(actual, empty)
                empty.biasVelocity[0].fill(100.0)
                assertState(actual, chunked.train(emptyArray(), 1, false))
            }
            assertState(state, model.exportTrainingState())
        }
    }

    @Test fun activationModesAreStableAtTailsAndExtremeInputs() {
        val values = doubleArrayOf(-1000.0, -745.0, -104.0, -16.0, -1.2, -0.5, 0.0, 0.5, 1.2, 16.0, 104.0, 745.0, 1000.0)
        for (mode in Neuro.SigmoidMode.entries) for (bits in listOf(0, 128, 256)) {
            val doubles = values.copyOf()
            SmallDoubleActivation(mode, bits, values.size).apply(doubles, doubles.size)
            val expected = DoubleArray(values.size) { Neuro.activate(values[it], mode) }
            assertArrayEquals(expected, doubles, 0.0)
            val floats = smallFloats(values)
            SmallFloatActivation(mode, bits, values.size).apply(floats, floats.size)
            for (i in floats.indices) {
                assertTrue(floats[i].isFinite() && floats[i] in 0.0f..1.0f)
                assertEquals(expected[i], floats[i].toDouble(), 2e-7)
            }
        }
    }

    @Test fun rejectsUnsupportedShapesMalformedStateOrdersAndClosedUseWithoutMutatingPublishedState() {
        for (shape in listOf(intArrayOf(), intArrayOf(2), intArrayOf(2, 1), intArrayOf(3, 8, 1),
            intArrayOf(2, 8, 2), intArrayOf(2, 7, 1), intArrayOf(2, 4, 4, 4, 4, 4, 1))) assertFalse(SmallNetworkShape.supports(shape))
        val model = model(intArrayOf(2, 8, 1))
        val state = model.exportTrainingState()
        val hp = model.hyperParameters()
        assertThrows(IllegalArgumentException::class.java) { SmallCpuTraining(state, hp, Neuro.TrainingPrecision.FP64, 512) }
        val malformed = listOf(state.copy(topology = intArrayOf(2, 7, 1)), state.copy(weights = emptyArray()),
            state.copy(biases = emptyArray()), state.copy(weightVelocity = emptyArray()), state.copy(biasVelocity = emptyArray()),
            state.copy(weights = arrayOf(doubleArrayOf(1.0), state.weights[1])),
            state.copy(inputs = doubleArrayOf(1.0)), state.copy(inputs = doubleArrayOf()),
            state.copy(targets = doubleArrayOf()), state.copy(inputs = state.inputs.copyOf().also { it[0] = Double.NaN }),
            state.copy(targets = state.targets.copyOf().also { it[0] = Double.POSITIVE_INFINITY }))
        for (invalid in malformed) assertThrows(IllegalArgumentException::class.java) { SmallCpuTraining(invalid, hp, Neuro.TrainingPrecision.FP64) }
        for (buffer in listOf(state.weights, state.biases, state.weightVelocity, state.biasVelocity)) {
            val previous = buffer[0][0]
            buffer[0][0] = Double.NaN
            assertThrows(IllegalArgumentException::class.java) { SmallCpuTraining(state, hp, Neuro.TrainingPrecision.FP64) }
            buffer[0][0] = previous
        }
        for (precision in Neuro.TrainingPrecision.entries) {
            val kernel = SmallCpuTraining(state, hp, precision)
            assertEquals(0, kernel.info.simdBits) // Explicit SCALAR wins over the constructor's preferred width.
            val before = kernel.train(emptyArray(), 1, false)
            for (invalid in listOf(intArrayOf(), IntArray(state.samples) { 0 }, IntArray(state.samples) { it + 1 }))
                assertThrows(IllegalArgumentException::class.java) { kernel.train(arrayOf(invalid), 1, false) }
            assertThrows(IllegalArgumentException::class.java) { kernel.train(emptyArray(), 0, false) }
            assertState(before, kernel.train(emptyArray(), 1, false))
            kernel.close(); kernel.close()
            assertThrows(IllegalStateException::class.java) { kernel.train(emptyArray(), 1, false) }
        }
        assertThrows(IllegalArgumentException::class.java) {
            SmallCpuTraining(state, hp.copy(beta = Double.MAX_VALUE), Neuro.TrainingPrecision.FP32)
        }
        state.weights[0][0] = Double.MAX_VALUE
        assertThrows(IllegalArgumentException::class.java) { SmallCpuTraining(state, hp, Neuro.TrainingPrecision.FP32) }
    }

    companion object {
        fun shapes(): List<IntArray> {
            val result = mutableListOf<IntArray>()
            fun append(hidden: List<Int>) {
                if (hidden.isNotEmpty()) result += (listOf(2) + hidden + 1).toIntArray()
                if (hidden.size < 4) for (width in listOf(4, 8, 16)) append(hidden + width)
            }
            append(emptyList())
            return result
        }

        fun model(shape: IntArray, seed: Long = 1234, mode: Neuro.SigmoidMode = Neuro.SigmoidMode.EXACT): Neuro =
            Neuro(shape, Neuro.HyperParameters(0.1, 0.23, 1.15, seed, Neuro.Kernel.SCALAR, mode)).also { model ->
                repeat(7) { sample -> model.addTrainingSample(doubleArrayOf(sample / 7.0, (sample * 3 % 7) / 7.0),
                    doubleArrayOf((sample % 2).toDouble())) }
            }

        fun orders(samples: Int, epochs: Int, offset: Int = 0): Array<IntArray> =
            Array(epochs) { epoch -> IntArray(samples) { (samples - 1 - it + epoch + offset) % samples } }

        internal fun assertState(expected: NeuroTrainingState, actual: NeuroTrainingState, tolerance: Double = 0.0, context: String = "") {
            assertArrayEquals(expected.topology, actual.topology, context)
            for ((left, right) in listOf(expected.weights to actual.weights, expected.biases to actual.biases,
                expected.weightVelocity to actual.weightVelocity, expected.biasVelocity to actual.biasVelocity)) {
                assertEquals(left.size, right.size, context)
                for (layer in left.indices) assertArrayEquals(left[layer], right[layer], tolerance, "$context layer=$layer")
            }
        }
    }
}
