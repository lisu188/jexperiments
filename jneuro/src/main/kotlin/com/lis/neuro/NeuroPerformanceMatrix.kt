package com.lis.neuro

object NeuroPerformanceMatrix {
    @JvmStatic fun main(args: Array<String>) = NeuroLog.application("NeuroPerformanceMatrix") {
        val iterations = args.getOrNull(0)?.toInt() ?: 2_000
        val trainingEpochs = args.getOrNull(1)?.toInt() ?: 20
        require(iterations > 0 && trainingEpochs > 0)
        val scenarios = linkedMapOf("tiny" to intArrayOf(2, 6, 1), "small" to intArrayOf(32, 64, 32, 8),
            "medium" to intArrayOf(128, 256, 128, 32), "deep" to intArrayOf(64, 128, 128, 64, 32, 8))
        println("TensorFlow CPU FP64/EXACT matrix: %,d inference iterations, %d training epochs".format(iterations, trainingEpochs))
        println("Inference uses a retained facade; training includes session setup/close. Use JMH for warmed multi-fork comparisons.")
        for ((name, topology) in scenarios) {
            val network = NeuroBenchmark.preparedNetwork(topology)
            val input = NeuroBenchmark.input(topology[0])
            val output = DoubleArray(topology.last())
            val session = network.newInferenceSession()
            repeat(minOf(iterations, 200)) { session.predictInto(input, output) }
            val inferenceStart = System.nanoTime()
            var sum = 0.0
            repeat(iterations) { session.predictInto(input, output); sum += output[it % output.size] }
            val inferenceNanos = System.nanoTime() - inferenceStart
            val trainingStart = System.nanoTime()
            network.train(trainingEpochs)
            val trainingNanos = System.nanoTime() - trainingStart
            println("%-8s params=%8d predict=%9.1f ns/op train=%9.3f ms/epoch error=%8.5f checksum=%f".format(
                name, network.parameterCount(), inferenceNanos.toDouble() / iterations,
                trainingNanos / 1_000_000.0 / trainingEpochs, network.trainingError(), sum))
        }
    }
}
