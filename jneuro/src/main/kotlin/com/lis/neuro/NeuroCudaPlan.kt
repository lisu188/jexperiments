package com.lis.neuro

internal class NeuroCudaPlan(topology: IntArray, val batchSize: Int) {
    val topology: IntArray = topology.copyOf()
    val layerCount: Int = topology.size - 1
    val maximumWidth: Int
    val activationElements: Long
    val deltaElements: Long
    val parameterElements: Long
    val estimatedFlops: Long

    init {
        require(topology.size >= 2 && topology.all { it > 0 })
        require(batchSize > 0)
        maximumWidth = topology.max()
        activationElements = topology.sumOf { Math.multiplyExact(batchSize.toLong(), it.toLong()) }
        deltaElements = topology.drop(1).sumOf { Math.multiplyExact(batchSize.toLong(), it.toLong()) }
        parameterElements = (0 until topology.lastIndex).sumOf {
            Math.multiplyExact(topology[it].toLong(), topology[it + 1].toLong()) + topology[it + 1]
        }
        estimatedFlops = (0 until topology.lastIndex).sumOf {
            val product = Math.multiplyExact(batchSize.toLong(),
                Math.multiplyExact(topology[it].toLong(), topology[it + 1].toLong()))
            Math.multiplyExact(product, 6L)
        }
    }
}
