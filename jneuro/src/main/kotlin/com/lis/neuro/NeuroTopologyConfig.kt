package com.lis.neuro

internal object NeuroTopologyConfig {
    const val MAX_HIDDEN_LAYERS = 8
    const val MAX_NEURONS_PER_LAYER = 128

    fun parseHidden(text: String): IntArray {
        require(text.isNotBlank()) { "Enter at least one hidden layer, for example 2 or 6,4,2." }
        val tokens = text.trim().split(',')
        require(tokens.size <= MAX_HIDDEN_LAYERS) { "Use at most $MAX_HIDDEN_LAYERS hidden layers." }
        return IntArray(tokens.size) { index ->
            val token = tokens[index].trim()
            require(token.isNotEmpty()) { "Layer ${index + 1} has no size. Remove the extra comma or enter a number." }
            val size = token.toIntOrNull()
            require(size != null) { "Layer ${index + 1} must be an integer." }
            require(size in 1..MAX_NEURONS_PER_LAYER) { "Layer ${index + 1} must have 1–$MAX_NEURONS_PER_LAYER neurons." }
            size
        }
    }

    fun topology(hidden: IntArray): IntArray {
        require(hidden.size in 1..MAX_HIDDEN_LAYERS) { "Use 1–$MAX_HIDDEN_LAYERS hidden layers." }
        require(hidden.all { it in 1..MAX_NEURONS_PER_LAYER }) { "Each layer must have 1–$MAX_NEURONS_PER_LAYER neurons." }
        return intArrayOf(2) + hidden + intArrayOf(1)
    }

    fun format(hidden: IntArray): String = hidden.joinToString(",")
    fun label(topology: IntArray): String = topology.joinToString(" → ")
    fun addLayer(hidden: IntArray, defaultSize: Int): IntArray {
        require(hidden.size < MAX_HIDDEN_LAYERS) { "Maximum hidden layer count reached." }
        return (hidden + defaultSize).also { topology(it) }
    }
    fun removeLayer(hidden: IntArray): IntArray {
        require(hidden.size > 1) { "Keep at least one hidden layer." }
        return hidden.copyOf(hidden.size - 1).also { topology(it) }
    }
}
