package com.lis.neuro

object NeuroExample {
    @JvmStatic fun main(args: Array<String>) = NeuroLog.application("NeuroExample") {
        val network = Neuro(intArrayOf(2, 6, 1), Neuro.HyperParameters.defaults()
            .withLearningRate(0.6).withMomentum(0.2).withSeed(42))
        val samples = NeuroLearningSets.create(NeuroLearningSets.Kind.XOR, 42)
        NeuroLearningSets.addTo(network, samples)
        println("before = ${network.trainingError()}")
        println("training = ${network.trainUntil(0.05, 10_000)}")
        for (sample in samples) {
            val input = doubleArrayOf(sample.x, sample.y)
            println("${input.contentToString()} -> ${network.predict(input).contentToString()}")
        }
        println("statistics = ${network.statistics()}")
    }
}
