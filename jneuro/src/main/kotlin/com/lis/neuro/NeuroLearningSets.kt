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
        CHECKERBOARD("Checkerboard 8×8", "64 alternating cells with repeated sharp boundaries. 1,024 spatially stratified samples."),
        CONCENTRIC_RINGS("Concentric rings", "Six alternating radial bands inside a disk. Several nested boundaries, not just one circle."),
        TIGHT_SPIRAL("Tight spiral", "Dense spiral regions with four windings to radius 0.5. 1,024 samples, without label noise."),
        PINWHEEL("Twisted pinwheel", "Eight alternating angular sectors that twist with radius. Narrow, curved class regions."),
        INTERFERENCE("Wave interference", "Crossing warped waves create many small curved regions. High-frequency structure in both inputs."),
        ISLANDS("16 islands", "Sixteen disconnected positive disks in a negative background. Learn local features across the square."),
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
        Kind.CHECKERBOARD -> spatial(seed) { x, y -> ((x * 8).toInt() + (y * 8).toInt()) % 2 == 1 }
        Kind.CONCENTRIC_RINGS -> spatial(seed) { x, y ->
            val radius = Math.hypot(x - 0.5, y - 0.5)
            radius < 0.5 && (radius * 12).toInt() % 2 == 0
        }
        Kind.TIGHT_SPIRAL -> twisted(seed, 1, 16 * Math.PI)
        Kind.PINWHEEL -> twisted(seed, 4, 12 * Math.PI)
        Kind.INTERFERENCE -> spatial(seed) { x, y -> wave(x, y) * wave(y, x) >= 0.0 }
        Kind.ISLANDS -> spatial(seed) { x, y ->
            val dx = (x * 4) % 1.0 - 0.5
            val dy = (y * 4) % 1.0 - 0.5
            dx * dx + dy * dy < 0.09
        }
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
    private fun spatial(seed: Long, positive: (Double, Double) -> Boolean): List<Sample> {
        val random = SplittableRandom(seed)
        val side = 32
        return List(side * side) { index ->
            val x = (index % side + 0.1 + 0.8 * random.nextDouble()) / side
            val y = (index / side + 0.1 + 0.8 * random.nextDouble()) / side
            Sample(x, y, if (positive(x, y)) 1.0 else 0.0)
        }
    }
    private fun twisted(seed: Long, angularFrequency: Int, radialFrequency: Double): List<Sample> =
        spatial(seed) { x, y ->
            val dx = x - 0.5
            val dy = y - 0.5
            val phase = angularFrequency * Math.atan2(dy, dx) - radialFrequency * Math.hypot(dx, dy)
            Math.sin(phase) >= 0.0
        }
    private fun wave(x: Double, y: Double): Double = Math.sin(10 * Math.PI * x + 2 * Math.sin(4 * Math.PI * y))

    private fun gaussian(random: SplittableRandom): Double {
        val u1 = maxOf(java.lang.Double.MIN_NORMAL, random.nextDouble())
        return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * random.nextDouble())
    }
}
