package com.lis.neuro

import kotlin.math.abs

object NeuroVerification {
    @JvmStatic fun main(args: Array<String>) {
        for (topology in listOf(intArrayOf(2), intArrayOf(2, 0, 1), intArrayOf(-1, 1))) {
            expect<IllegalArgumentException> { Neuro(topology) }
        }
        val empty = Neuro(intArrayOf(2, 1))
        expect<IllegalStateException> { empty.trainEpoch() }
        expect<IllegalArgumentException> { empty.addTrainingSample(doubleArrayOf(1.0), doubleArrayOf(1.0)) }
        expect<IllegalArgumentException> { empty.addTrainingSample(doubleArrayOf(1.0, 0.0), doubleArrayOf(1.0, 0.0)) }
        expect<IllegalArgumentException> { empty.addTrainingSample(doubleArrayOf(Double.NaN, 0.0), doubleArrayOf(1.0)) }
        val parameters = Neuro.HyperParameters.defaults().withSeed(123456789)
        val first = Neuro(intArrayOf(3, 5, 2), parameters)
        val second = Neuro(intArrayOf(3, 5, 2), parameters)
        val input = doubleArrayOf(0.25, 0.5, 0.75)
        check(first.predict(input).contentEquals(second.predict(input))) { "Seed determinism" }
        val topology = intArrayOf(2, 3, 1)
        val copied = Neuro(topology)
        topology[1] = 99
        check(copied.topology().contentEquals(intArrayOf(2, 3, 1)))
        copied.topology()[0] = 99
        check(copied.topology()[0] == 2)
        val sample = doubleArrayOf(0.0, 0.0)
        val target = doubleArrayOf(0.0)
        copied.addTrainingSample(sample, target)
        sample[0] = 1.0; target[0] = 1.0
        val reference = Neuro(intArrayOf(2, 3, 1)).addTrainingSample(doubleArrayOf(0.0, 0.0), doubleArrayOf(0.0))
        check(copied.trainingError() == reference.trainingError())
        check(copied.parameterCount() == 13)
        val multipleOutput = Neuro(intArrayOf(2, 4, 2), parameters.withSeed(7))
        val testInput = doubleArrayOf(0.2, 0.8)
        val result = DoubleArray(2)
        multipleOutput.predictInto(testInput, result)
        check(result.contentEquals(multipleOutput.predict(testInput)))
        expect<IllegalArgumentException> { multipleOutput.predictInto(testInput, DoubleArray(1)) }
        for ((kind, shape, seed) in listOf(
            Triple(NeuroLearningSets.Kind.OR, intArrayOf(2, 3, 1), 11L),
            Triple(NeuroLearningSets.Kind.XOR, intArrayOf(2, 6, 1), 42L))) {
            val model = Neuro(shape, Neuro.HyperParameters(0.6, if (kind == NeuroLearningSets.Kind.OR) 0.1 else 0.2, 1.0, seed))
            val samples = NeuroLearningSets.create(kind, 0)
            NeuroLearningSets.addTo(model, samples)
            val before = model.trainingError()
            val training = model.trainUntil(0.08, 10_000)
            check(training.converged && training.error < before) { "$kind convergence" }
            for (point in samples) {
                val output = model.predict(doubleArrayOf(point.x, point.y))[0]
                check(if (point.target == 0.0) output < 0.2 else output > 0.8) { "$kind $point" }
            }
        }
        val testModel = Neuro(intArrayOf(2, 4, 1), Neuro.HyperParameters.defaults().withLearningRate(0.6).withSeed(99))
        NeuroLearningSets.addTo(testModel, NeuroLearningSets.create(NeuroLearningSets.Kind.OR, 0))
        testModel.train(1000)
        testModel.addTestSample(doubleArrayOf(0.0, 0.0), doubleArrayOf(1.0))
        testModel.addTestSample(doubleArrayOf(1.0, 1.0), doubleArrayOf(0.0))
        val zero = testModel.predict(doubleArrayOf(0.0, 0.0))[0]
        val one = testModel.predict(doubleArrayOf(1.0, 1.0))[0]
        val expected = Math.sqrt(((1 - zero) * (1 - zero) + one * one) / 2)
        check(abs(testModel.testError() - expected) <= 1e-12 && testModel.testSampleCount() == 2)
        val limited = Neuro(intArrayOf(2, 2, 1), parameters.withSeed(17))
        NeuroLearningSets.addTo(limited, NeuroLearningSets.create(NeuroLearningSets.Kind.XOR, 0))
        val training = limited.trainUntil(1e-15, 3)
        check(training.epochs == 3 && !training.converged)
        check(limited.statistics().epochsTrained == 3L && limited.statistics().samplesSeen == 12L)
        val extreme = Neuro(intArrayOf(1, 1), parameters.withSeed(3))
        for (value in doubleArrayOf(-1e300, 1e300)) {
            val output = extreme.predict(doubleArrayOf(value))[0]
            check(output.isFinite() && output in 0.0..1.0)
        }
        check(extreme.testError().isNaN())
        println("JNeuro Kotlin verification passed")
    }
    private inline fun <reified T : Throwable> expect(action: () -> Unit) {
        try { action() } catch (exception: Throwable) { if (exception is T) return else throw exception }
        throw AssertionError("Expected ${T::class.java.simpleName}")
    }
}
