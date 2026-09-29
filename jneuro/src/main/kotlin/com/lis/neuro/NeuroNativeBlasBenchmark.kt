package com.lis.neuro

import kotlin.math.abs

object NeuroNativeBlasBenchmark {
    private data class Scenario(val name: String, val topology: IntArray, val batchSize: Int, val iterations: Int)
    @Volatile private var blackhole = 0.0

    @JvmStatic fun main(args: Array<String>) {
        val scenarios = arrayOf(Scenario("medium", intArrayOf(128, 256, 128, 32), 64, 100),
            Scenario("large", intArrayOf(512, 1024, 512, 128), 128, 30))
        for (scenario in scenarios) run(scenario)
        println("blackhole=$blackhole")
    }
    private fun run(scenario: Scenario) {
        val network = Neuro(scenario.topology, Neuro.HyperParameters.defaults().withSeed(1234).withKernel(Neuro.Kernel.VECTOR))
        val inputSize = scenario.topology[0]
        val outputs = DoubleArray(scenario.topology.last() * scenario.batchSize)
        val nativeOutputs = DoubleArray(outputs.size)
        val inputs = DoubleArray(inputSize * scenario.batchSize)
        repeat(scenario.batchSize) { NeuroBenchmark.input(inputSize).copyInto(inputs, it * inputSize) }
        val optional = NeuroNativeBlas.tryCreate(network)
        if (optional.isEmpty) {
            println("%-8s native BLAS unavailable".format(scenario.name))
            return
        }
        optional.orElseThrow().use { blas ->
            val session = network.newInferenceSession()
            session.predictBatch(inputs, scenario.batchSize, outputs)
            blas.predictBatch(inputs, scenario.batchSize, nativeOutputs)
            val difference = outputs.indices.maxOf { abs(outputs[it] - nativeOutputs[it]) }
            check(difference <= 1.0e-10) { "native BLAS output differs by $difference" }
            repeat(10) {
                session.predictBatch(inputs, scenario.batchSize, outputs)
                blas.predictBatch(inputs, scenario.batchSize, nativeOutputs)
            }
            val javaNanos = measure(scenario.iterations, outputs) { session.predictBatch(inputs, scenario.batchSize, outputs) }
            val nativeNanos = measure(scenario.iterations, nativeOutputs) { blas.predictBatch(inputs, scenario.batchSize, nativeOutputs) }
            println("%-8s library=%-18s batch=%3d java=%9.3f ms native=%9.3f ms speedup=%6.2fx diff=%g".format(
                scenario.name, blas.libraryName(), scenario.batchSize, javaNanos / 1_000_000.0 / scenario.iterations,
                nativeNanos / 1_000_000.0 / scenario.iterations, javaNanos.toDouble() / nativeNanos, difference))
        }
    }
    private inline fun measure(iterations: Int, output: DoubleArray, action: () -> Unit): Long {
        val start = System.nanoTime()
        repeat(iterations) { action(); blackhole += output[it % output.size] }
        return System.nanoTime() - start
    }
}
