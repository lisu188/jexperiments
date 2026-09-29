package com.lis.neuro

internal object NeuroTopologyConfig {

    fun parseHidden(text: String): IntArray {
        if (text.isBlank()) return intArrayOf()
        val tokens = text.trim().split(',')
        return IntArray(tokens.size) { index ->
            val token = tokens[index].trim()
            require(token.isNotEmpty()) { "Layer ${index + 1} has no size. Remove the extra comma or enter a number." }
            val size = token.toIntOrNull()
            require(size != null) { "Layer ${index + 1} must be an integer." }
            require(size > 0) { "Layer ${index + 1} must have a positive number of neurons." }
            size
        }
    }

    fun topology(hidden: IntArray): IntArray {
        require(hidden.all { it > 0 }) { "Each layer must have a positive number of neurons." }
        return intArrayOf(2) + hidden + intArrayOf(1)
    }

    fun format(hidden: IntArray): String = hidden.joinToString(",")
    fun label(topology: IntArray): String = topology.joinToString(" → ")
    fun addLayer(hidden: IntArray, defaultSize: Int): IntArray {
        return (hidden + defaultSize).also { topology(it) }
    }
    fun removeLayer(hidden: IntArray): IntArray {
        require(hidden.isNotEmpty()) { "There is no hidden layer to remove." }
        return hidden.copyOf(hidden.size - 1).also { topology(it) }
    }
}
