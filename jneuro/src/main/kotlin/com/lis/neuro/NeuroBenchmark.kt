package com.lis.neuro

object NeuroBenchmark {
    @Volatile private var blackhole = 0.0

    @JvmStatic fun main(args: Array<String>) = NeuroLog.application("NeuroBenchmark") {
        val iterations = args.getOrNull(0)?.toInt() ?: 2_000
        val repetitions = args.getOrNull(1)?.toInt() ?: 5
        require(iterations > 0 && repetitions > 0)
        repeat(2) { run(iterations) }
        val samples = linkedMapOf<String, MutableList<Double>>()
        repeat(repetitions) {
            for ((name, value) in run(iterations)) samples.getOrPut(name) { ArrayList() }.add(value)
        }
        println("TensorFlow CPU benchmark: %,d predictions per inference case; training includes session setup/close".format(iterations))
        for ((name, values) in samples) {
            val ordered = values.sorted()
            val median = (ordered[(ordered.size - 1) / 2] + ordered[ordered.size / 2]) / 2
            println("%-30s %10.3f ms median".format(name, median))
        }
        println("blackhole=$blackhole")
    }

    private fun run(iterations: Int): Map<String, Double> {
        val network = preparedNetwork(intArrayOf(32, 64, 32, 8))
        val input = input(32)
        val output = DoubleArray(8)
        val session = network.newInferenceSession()
        val fp32 = network.toFloatModel()
        val floatInput = FloatArray(input.size) { input[it].toFloat() }
        val floatOutput = FloatArray(output.size)
        return linkedMapOf(
            "tensorflow-fp64-allocating" to millis {
                var sum = 0.0
                repeat(iterations) { sum += network.predict(input)[it and 7] }
                blackhole += sum
            },
            "tensorflow-fp64-direct" to millis {
                var sum = 0.0
                repeat(iterations) { network.predictInto(input, output); sum += output[it and 7] }
                blackhole += sum
            },
            "tensorflow-fp64-session" to millis {
                var sum = 0.0
                repeat(iterations) { session.predictInto(input, output); sum += output[it and 7] }
                blackhole += sum
            },
            "tensorflow-fp32" to millis {
                var sum = 0.0
                repeat(iterations) { fp32.predictInto(floatInput, floatOutput); sum += floatOutput[it and 7] }
                blackhole += sum
            },
            "tensorflow-train-100" to millis {
                val training = preparedNetwork(intArrayOf(32, 64, 32, 8))
                training.train(100)
                blackhole += training.trainingError()
            }
        )
    }

    @JvmStatic fun preparedNetwork(topology: IntArray): Neuro {
        val network = Neuro(topology, Neuro.HyperParameters.defaults().withLearningRate(0.1).withMomentum(0.1).withSeed(1234))
        repeat(32) { sample ->
            network.addTrainingSample(DoubleArray(topology[0]) { ((sample + it) and 7) / 7.0 },
                DoubleArray(topology.last()) { ((sample + it) and 1).toDouble() })
        }
        return network
    }
    @JvmStatic fun input(size: Int): DoubleArray = DoubleArray(size) { (it and 7) / 7.0 }
    private inline fun millis(action: () -> Unit): Double {
        val start = System.nanoTime()
        action()
        return (System.nanoTime() - start) / 1_000_000.0
    }
}
