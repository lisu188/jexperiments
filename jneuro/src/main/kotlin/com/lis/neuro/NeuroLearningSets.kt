package com.lis.neuro

import java.util.SplittableRandom

internal object NeuroLearningSets {
    enum class Kind(private val label: String, val description: String) {
        XOR("XOR", "Opposite corners share a class. A non-linear decision boundary is required."),
        AND("AND", "Only (1, 1) belongs to class 1. A line can separate the classes."),
        OR("OR", "Only (0, 0) belongs to class 0. A line can separate the classes."),
        NAND("NAND", "The complement of AND: all corners except (1, 1) belong to class 1."),
        XNOR("XNOR", "Matching inputs belong to class 1. The complement of XOR."),
        NOISY_XOR("Noisy XOR", "Labeled clouds around the four corners; positions are jittered, labels are unchanged."),
        CIRCLE("Circle", "Learn a curved boundary around a central disk from 180 labeled samples."),
        SPIRAL("Spiral", "Two intertwined arms. A challenging problem for a small sigmoid network."),
        CUSTOM("Custom", "Place your own labeled points. Changes restart the model on the edited set.");
        override fun toString(): String = label
    }

    @JvmRecord data class Sample(val x: Double, val y: Double, val target: Double) {
        init {
            require(x.isFinite() && y.isFinite() && target.isFinite()) { "sample values must be finite" }
            require(x in 0.0..1.0 && y in 0.0..1.0) { "sample coordinates must be in [0, 1]" }
            require(target in 0.0..1.0) { "target must be in [0, 1]" }
        }
    }

    fun create(kind: Kind, seed: Long): List<Sample> = when (kind) {
        Kind.XOR -> truthTable(0, 1, 1, 0)
        Kind.AND -> truthTable(0, 0, 0, 1)
        Kind.OR -> truthTable(0, 1, 1, 1)
        Kind.NAND -> truthTable(1, 1, 1, 0)
        Kind.XNOR -> truthTable(1, 0, 0, 1)
        Kind.NOISY_XOR -> noisyXor(seed)
        Kind.CIRCLE -> circle(seed)
        Kind.SPIRAL -> spiral()
        Kind.CUSTOM -> emptyList()
    }
    fun truthTable(value00: Int, value01: Int, value10: Int, value11: Int): List<Sample> = listOf(
        Sample(0.0, 0.0, value00.toDouble()), Sample(0.0, 1.0, value01.toDouble()),
        Sample(1.0, 0.0, value10.toDouble()), Sample(1.0, 1.0, value11.toDouble())
    )
    fun addTo(network: Neuro, samples: List<Sample>) {
        for (sample in samples) network.addTrainingSample(doubleArrayOf(sample.x, sample.y), doubleArrayOf(sample.target))
    }
    private fun noisyXor(seed: Long): List<Sample> {
        val random = SplittableRandom(seed)
        return List(160) { index ->
            val corner = index and 3
            val baseX = if (corner >= 2) 1.0 else 0.0
            val baseY = if (corner and 1 == 1) 1.0 else 0.0
            Sample((baseX + gaussian(random) * 0.115).coerceIn(0.0, 1.0),
                (baseY + gaussian(random) * 0.115).coerceIn(0.0, 1.0),
                if (corner == 1 || corner == 2) 1.0 else 0.0)
        }
    }
    private fun circle(seed: Long): List<Sample> {
        val random = SplittableRandom(seed)
        return List(180) {
            val x = random.nextDouble()
            val y = random.nextDouble()
            val dx = x - 0.5
            val dy = y - 0.5
            Sample(x, y, if (dx * dx + dy * dy <= 0.26 * 0.26) 1.0 else 0.0)
        }
    }
    private fun spiral(): List<Sample> = List(220) { index ->
        val target = index / 110
        val fraction = (index % 110) / 109.0
        val radius = 0.05 + 0.43 * fraction
        val angle = target * Math.PI + fraction * Math.PI * 3.25
        Sample((0.5 + radius * Math.cos(angle)).coerceIn(0.0, 1.0),
            (0.5 + radius * Math.sin(angle)).coerceIn(0.0, 1.0), target.toDouble())
    }
    private fun gaussian(random: SplittableRandom): Double {
        val u1 = maxOf(java.lang.Double.MIN_NORMAL, random.nextDouble())
        return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * random.nextDouble())
    }
}
