package com.lis.neuro

import java.util.SplittableRandom

internal class LegacyNeuroBaseline(topology: IntArray) {
    private val topology = topology.copyOf()
    private val random = SplittableRandom(1234)
    private val weights = Array(topology.size - 1) { layer ->
        val limit = Math.sqrt(6.0 / (topology[layer] + topology[layer + 1]))
        DoubleArray(topology[layer] * topology[layer + 1]) { random.nextDouble(-limit, limit) }
    }
    private val biases = Array(topology.size - 1) { DoubleArray(topology[it + 1]) }
    private val workspace = ThreadLocal.withInitial { Array(this.topology.size) { DoubleArray(this.topology[it]) } }
    fun predictInto(input: DoubleArray, output: DoubleArray) {
        val activations = workspace.get()
        input.copyInto(activations[0])
        for (layer in weights.indices) {
            val source = activations[layer]
            val destination = activations[layer + 1]
            for (out in destination.indices) {
                val offset = out * source.size
                var sum = biases[layer][out]
                var i = 0
                val limit = source.size - (source.size and 3)
                while (i < limit) {
                    sum = Math.fma(source[i], weights[layer][offset + i], sum)
                    sum = Math.fma(source[i + 1], weights[layer][offset + i + 1], sum)
                    sum = Math.fma(source[i + 2], weights[layer][offset + i + 2], sum)
                    sum = Math.fma(source[i + 3], weights[layer][offset + i + 3], sum)
                    i += 4
                }
                while (i < source.size) { sum = Math.fma(source[i], weights[layer][offset + i], sum); i++ }
                destination[out] = if (sum >= 0) 1.0 / (1.0 + Math.exp(-sum)) else {
                    val exp = Math.exp(sum); exp / (1.0 + exp)
                }
            }
        }
        activations.last().copyInto(output)
    }
}
