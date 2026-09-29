package com.lis.neuro

import java.lang.reflect.InvocationTargetException
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

class NeuroTest {
    @Test fun validatesConfigurationAndSamples() {
        for (shape in listOf(intArrayOf(2), intArrayOf(2, 0, 1), intArrayOf(-1, 1)))
            assertThrows(IllegalArgumentException::class.java) { Neuro(shape) }
        for (value in doubleArrayOf(0.0, -1.0, Double.NaN, Double.POSITIVE_INFINITY)) {
            assertThrows(IllegalArgumentException::class.java) { Neuro.HyperParameters(value, 0.1, 1.0, 1) }
            assertThrows(IllegalArgumentException::class.java) { Neuro.HyperParameters(0.1, 0.1, value, 1) }
        }
        for (value in doubleArrayOf(-0.1, 1.0, Double.NaN, Double.POSITIVE_INFINITY))
            assertThrows(IllegalArgumentException::class.java) { Neuro.HyperParameters(0.1, value, 1.0, 1) }
        val model = Neuro(intArrayOf(2, 3, 1))
        assertThrows(IllegalArgumentException::class.java) { model.addTrainingSample(doubleArrayOf(0.0), doubleArrayOf(0.0)) }
        assertThrows(IllegalArgumentException::class.java) { model.addTrainingSample(doubleArrayOf(0.0, 0.0), doubleArrayOf(0.0, 1.0)) }
        assertThrows(IllegalArgumentException::class.java) { model.addTrainingSample(doubleArrayOf(Double.NaN, 0.0), doubleArrayOf(0.0)) }
        assertThrows(IllegalArgumentException::class.java) { model.addTestSample(doubleArrayOf(0.0, 0.0), doubleArrayOf(Double.POSITIVE_INFINITY)) }
        assertThrows(IllegalStateException::class.java) { model.trainEpoch() }
        assertThrows(IllegalArgumentException::class.java) { model.train(-1) }
        model.train(0)
        assertThrows(IllegalArgumentException::class.java) { model.trainUntil(-1.0, 1) }
        assertThrows(IllegalArgumentException::class.java) { model.trainUntil(0.1, -1) }
        assertThrows(IllegalArgumentException::class.java) { model.trainUntil(0.1, 1, 0) }
        assertThrows(IllegalArgumentException::class.java) { model.trainMiniBatch(1, 0) }
        assertThrows(IllegalArgumentException::class.java) { model.trainMiniBatch(-1, 1) }
        assertThrows(IllegalArgumentException::class.java) { model.trainMiniBatch(1, 1, 0) }
        model.trainMiniBatch(0, 1)
        assertTrue(model.trainingError().isNaN())
    }

    @Test fun javaNullContractRemainsFailFast() {
        val constructor = Neuro::class.java.getConstructor(IntArray::class.java)
        assertInstanceOf(NullPointerException::class.java,
            assertThrows(InvocationTargetException::class.java) { constructor.newInstance(null) }.cause)
        val hyper = Neuro.HyperParameters::class.java.getConstructor(Double::class.javaPrimitiveType, Double::class.javaPrimitiveType,
            Double::class.javaPrimitiveType, Long::class.javaPrimitiveType, Neuro.Kernel::class.java, Neuro.SigmoidMode::class.java)
        assertInstanceOf(NullPointerException::class.java,
            assertThrows(InvocationTargetException::class.java) { hyper.newInstance(0.1, 0.1, 1.0, 1L, null, Neuro.SigmoidMode.EXACT) }.cause)
        assertInstanceOf(NullPointerException::class.java,
            assertThrows(InvocationTargetException::class.java) { hyper.newInstance(0.1, 0.1, 1.0, 1L, Neuro.Kernel.AUTO, null) }.cause)
        val method = Neuro::class.java.getMethod("addTrainingSample", DoubleArray::class.java, DoubleArray::class.java)
        val model = Neuro(intArrayOf(2, 1))
        for (args in listOf(arrayOf(null, doubleArrayOf(0.0)), arrayOf(doubleArrayOf(0.0, 0.0), null))) {
            assertInstanceOf(NullPointerException::class.java,
                assertThrows(InvocationTargetException::class.java) { method.invoke(model, *args) }.cause)
        }
    }

    @Test fun preservesDeterminismAndDefensiveCopies() {
        val parameters = Neuro.HyperParameters.defaults().withLearningRate(0.3).withMomentum(0.1).withBeta(1.2)
            .withSeed(12345).withKernel(Neuro.Kernel.SCALAR).withSigmoidMode(Neuro.SigmoidMode.EXACT)
        assertEquals(0.3, parameters.learningRate); assertEquals(0.1, parameters.momentum)
        assertEquals(1.2, parameters.beta); assertEquals(12345L, parameters.seed)
        assertEquals(Neuro.Kernel.SCALAR, parameters.kernel)
        val topology = intArrayOf(3, 5, 2)
        val first = Neuro(topology, parameters)
        val second = Neuro(topology, parameters)
        topology[1] = 99; first.topology()[0] = 99
        assertArrayEquals(intArrayOf(3, 5, 2), first.topology())
        assertEquals(parameters, first.hyperParameters()); assertEquals(32, first.parameterCount())
        val input = doubleArrayOf(0.2, 0.4, 0.6)
        assertArrayEquals(first.predict(input), second.predict(input), 0.0)
        val sample = doubleArrayOf(0.1, 0.2, 0.3); val target = doubleArrayOf(0.0, 1.0)
        first.addTrainingSample(sample, target)
        sample[0] = 9.0; target[0] = 9.0
        val reference = Neuro(intArrayOf(3, 5, 2), parameters).addTrainingSample(doubleArrayOf(0.1, 0.2, 0.3), doubleArrayOf(0.0, 1.0))
        assertEquals(reference.trainingError(), first.trainingError(), 0.0)
        assertEquals(1, first.trainingSampleCount()); assertEquals(0, first.testSampleCount()); assertTrue(first.testError().isNaN())
        val legacyConstructor = Neuro(intArrayOf(2, 3, 1), 0.1, 1.2, 0.3)
        assertEquals(0.3, legacyConstructor.hyperParameters().learningRate)
    }

    @Test fun scalarVectorSessionsBatchAndFloatAgree() {
        for (shape in listOf(intArrayOf(32, 64, 32, 8), intArrayOf(3, 7, 2), intArrayOf(33, 65, 17), intArrayOf(2, 1))) {
            val scalar = prepared(shape, Neuro.Kernel.SCALAR)
            val vector = prepared(shape, Neuro.Kernel.VECTOR)
            val automatic = prepared(shape, Neuro.Kernel.AUTO)
            val input = input(shape[0]); val output = vector.predict(input)
            assertArrayEquals(scalar.predict(input), output, 1e-12)
            assertArrayEquals(scalar.predict(input), automatic.predict(input), 1e-12)
            val reused = DoubleArray(shape.last())
            vector.predictInto(input, reused); assertArrayEquals(output, reused, 0.0)
            val session = vector.newInferenceSession()
            reused.fill(Double.NaN); session.predictInto(input, reused); assertArrayEquals(output, reused, 0.0)
            val batchSize = 5
            val batchInputs = DoubleArray(input.size * batchSize) { input[it % input.size] }
            val batch = DoubleArray(shape.last() * batchSize); val parallel = DoubleArray(batch.size)
            session.predictBatch(batchInputs, batchSize, batch)
            vector.predictBatchParallel(batchInputs, batchSize, parallel, 2)
            assertArrayEquals(batch, parallel, 1e-12)
            vector.newParallelInferenceSession(8).use { parallelSession ->
                parallelSession.predictBatch(batchInputs, batchSize, parallel)
                assertArrayEquals(batch, parallel, 1e-12)
                parallelSession.predictBatch(DoubleArray(0), 0, DoubleArray(0))
            }
            for (sample in 0 until batchSize) assertArrayEquals(output, batch.copyOfRange(sample * output.size, (sample + 1) * output.size), 1e-12)
            val floatModel = vector.toFloatModel()
            val floatInput = FloatArray(input.size) { input[it].toFloat() }
            val floatOutput = floatModel.predict(floatInput)
            val floatReused = FloatArray(floatOutput.size)
            assertArrayEquals(shape, floatModel.topology())
            floatModel.topology()[0] = 99
            assertArrayEquals(shape, floatModel.topology())
            floatModel.predictInto(floatInput, floatReused); assertArrayEquals(floatOutput, floatReused, 0.0f)
            for (i in output.indices) assertEquals(output[i], floatOutput[i].toDouble(), 1e-5)
            val floatBatch = FloatArray(batch.size)
            val floatBatchInputs = FloatArray(batchInputs.size) { batchInputs[it].toFloat() }
            floatModel.predictBatch(floatBatchInputs, batchSize, floatBatch)
            assertArrayEquals(floatOutput, floatBatch.copyOfRange(0, floatOutput.size), 0.0f)
            assertThrows(IllegalArgumentException::class.java) { vector.predictInto(DoubleArray(1), reused) }
            assertThrows(IllegalArgumentException::class.java) { vector.predictInto(input, DoubleArray(shape.last() + 1)) }
            assertThrows(IllegalArgumentException::class.java) { vector.predictBatch(DoubleArray(1), 1, batch) }
            assertThrows(IllegalArgumentException::class.java) { vector.predictBatch(batchInputs, -1, batch) }
            assertThrows(IllegalArgumentException::class.java) { vector.predictBatch(batchInputs, 1, DoubleArray(0)) }
            assertThrows(IllegalArgumentException::class.java) { vector.predictBatch(DoubleArray(1), Int.MAX_VALUE, DoubleArray(1)) }
            assertThrows(IllegalArgumentException::class.java) { vector.newParallelInferenceSession(0) }
            assertThrows(IllegalArgumentException::class.java) { floatModel.predictInto(FloatArray(1), floatReused) }
            assertThrows(IllegalArgumentException::class.java) { floatModel.predictInto(floatInput, FloatArray(0)) }
            assertThrows(IllegalArgumentException::class.java) { floatModel.predictBatch(FloatArray(1), 1, floatBatch) }
            assertThrows(IllegalArgumentException::class.java) { floatModel.predictBatch(floatBatchInputs, -1, floatBatch) }
            assertThrows(IllegalArgumentException::class.java) { floatModel.predictBatch(FloatArray(1), Int.MAX_VALUE, FloatArray(1)) }
        }
    }

    @Test fun onlineTrainingConvergesAndTracksStatistics() {
        for (kernel in Neuro.Kernel.entries) {
            val model = xor(kernel)
            val before = model.trainingError()
            val result = model.trainUntil(0.08, 10_000)
            assertTrue(result.converged, kernel.name); assertTrue(result.error < before)
            assertTrue(model.predict(doubleArrayOf(0.0, 0.0))[0] < 0.2)
            assertTrue(model.predict(doubleArrayOf(0.0, 1.0))[0] > 0.8)
            assertTrue(model.predict(doubleArrayOf(1.0, 0.0))[0] > 0.8)
            assertTrue(model.predict(doubleArrayOf(1.0, 1.0))[0] < 0.2)
            assertEquals(result.epochs.toLong(), model.statistics().epochsTrained)
            assertEquals(result.epochs * 4L, model.statistics().samplesSeen)
        }
    }

    @Test fun exactXorMatchesPinnedJavaBaseline() {
        val model = xor()
        assertEquals(0.5129051250129316, model.trainingError(), 1e-14)
        val result = model.trainUntil(0.05, 10_000)
        assertEquals(1144, result.epochs)
        assertEquals(0.04997028695977735, result.error, 1e-12)
        val expected = doubleArrayOf(0.045355701043297426, 0.9499573164770715, 0.9514754011138365, 0.055426273796837275)
        val corners = doubleArrayOf(0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0)
        val actual = DoubleArray(4); model.predictBatch(corners, 4, actual)
        assertArrayEquals(expected, actual, 1e-12)
    }

    @Test fun fastSigmoidAndSparseErrorChecksWork() {
        val exact = prepared(intArrayOf(16, 32, 4), Neuro.Kernel.AUTO)
        val fast = prepared(intArrayOf(16, 32, 4), Neuro.Kernel.AUTO, Neuro.SigmoidMode.FAST)
        assertArrayEquals(exact.predict(input(16)), fast.predict(input(16)), 0.04)
        val model = xor(mode = Neuro.SigmoidMode.FAST)
        assertTrue(model.trainUntil(0.12, 10_000, 8).converged)
        for (mode in Neuro.SigmoidMode.entries) {
            val extreme = Neuro(intArrayOf(1, 1), Neuro.HyperParameters.defaults().withSigmoidMode(mode))
            for (value in doubleArrayOf(-1e300, 1e300)) {
                val output = extreme.predict(doubleArrayOf(value))[0]
                assertTrue(output.isFinite() && output in 0.0..1.0)
            }
            val floating = fast.toFloatModel().predict(FloatArray(16) { if (it % 2 == 0) -1000f else 1000f })
            assertTrue(floating.all { it.isFinite() })
        }
    }

    @Test fun fixedEpochAndMiniBatchTrainingUsePackedData() {
        for (kernel in Neuro.Kernel.entries) for (shape in listOf(intArrayOf(32, 64, 32, 8), intArrayOf(3, 5, 2))) {
            val online = prepared(shape, kernel)
            val before = online.trainingError(); online.train(4)
            assertEquals(4L, online.statistics().epochsTrained); assertEquals(128L, online.statistics().samplesSeen)
            assertTrue(online.statistics().lastTrainingError.isFinite()); assertTrue(online.trainingError() <= before + 0.05)
            for (parallelism in listOf(1, 2)) {
                val batch = prepared(shape, kernel)
                val original = batch.trainingError()
                batch.trainMiniBatch(4, 7, parallelism)
                assertEquals(4L, batch.statistics().epochsTrained); assertEquals(128L, batch.statistics().samplesSeen)
                assertTrue(batch.trainingError() < original)
            }
            assertTrue(prepared(shape, kernel).trainEpoch().isFinite())
        }
        val singleton = Neuro(intArrayOf(2, 1)).addTrainingSample(doubleArrayOf(0.0, 1.0), doubleArrayOf(1.0))
        singleton.trainMiniBatch(1, 8, 4)
        singleton.addTrainingSample(doubleArrayOf(1.0, 0.0), doubleArrayOf(1.0))
        singleton.trainMiniBatch(2, 1)
        assertEquals(5L, singleton.statistics().samplesSeen)
        singleton.trainUntil(1e-15, 0)
    }

    @Test fun testSetErrorAndEpochLimitsRemainCorrect() {
        val network = Neuro(intArrayOf(2, 4, 1), Neuro.HyperParameters.defaults().withLearningRate(0.6).withSeed(99))
        NeuroLearningSets.addTo(network, NeuroLearningSets.create(NeuroLearningSets.Kind.OR, 0))
        network.train(1000)
        network.addTestSample(doubleArrayOf(0.0, 0.0), doubleArrayOf(1.0))
        network.addTestSample(doubleArrayOf(1.0, 1.0), doubleArrayOf(0.0))
        val p00 = network.predict(doubleArrayOf(0.0, 0.0))[0]; val p11 = network.predict(doubleArrayOf(1.0, 1.0))[0]
        assertEquals(Math.sqrt(((1 - p00) * (1 - p00) + p11 * p11) / 2), network.testError(), 1e-12)
        assertEquals(2, network.testSampleCount())
        val limited = Neuro(intArrayOf(2, 2, 1), Neuro.HyperParameters.defaults().withSeed(17))
        NeuroLearningSets.addTo(limited, NeuroLearningSets.create(NeuroLearningSets.Kind.XOR, 0))
        val result = limited.trainUntil(1e-15, 3, 2)
        assertEquals(3, result.epochs); assertFalse(result.converged)
        val good = Neuro(intArrayOf(1, 1))
        good.addTrainingSample(doubleArrayOf(0.0), good.predict(doubleArrayOf(0.0)))
        assertEquals(0, good.trainUntil(1.0, 10).epochs)
        assertTrue(good.trainUntil(1.0, 10).converged)
    }

    companion object {
        fun prepared(shape: IntArray, kernel: Neuro.Kernel = Neuro.Kernel.AUTO, mode: Neuro.SigmoidMode = Neuro.SigmoidMode.EXACT): Neuro {
            val model = Neuro(shape, Neuro.HyperParameters(0.1, 0.1, 1.0, 1234, kernel, mode))
            for (sample in 0 until 32) model.addTrainingSample(DoubleArray(shape[0]) { ((sample + it) and 7) / 7.0 }, DoubleArray(shape.last()) { ((sample + it) and 1).toDouble() })
            return model
        }
        fun input(size: Int) = DoubleArray(size) { (it and 7) / 7.0 }
        fun xor(kernel: Neuro.Kernel = Neuro.Kernel.AUTO, mode: Neuro.SigmoidMode = Neuro.SigmoidMode.EXACT): Neuro =
            Neuro(intArrayOf(2, 6, 1), Neuro.HyperParameters(0.6, 0.2, 1.0, 42, kernel, mode)).also {
                NeuroLearningSets.addTo(it, NeuroLearningSets.create(NeuroLearningSets.Kind.XOR, 0))
            }
    }
}
