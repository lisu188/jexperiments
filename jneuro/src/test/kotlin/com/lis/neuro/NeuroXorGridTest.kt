package com.lis.neuro

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

class NeuroXorGridTest {
    @Test fun mapsPixelsWithYIncreasingUpward() {
        val inputs = NeuroXorGrid.createInputs(3)
        assertArrayEquals(doubleArrayOf(0.0, 1.0), inputs.copyOfRange(0, 2))
        assertArrayEquals(doubleArrayOf(1.0, 1.0), inputs.copyOfRange(4, 6))
        assertArrayEquals(doubleArrayOf(0.0, 0.0), inputs.copyOfRange(12, 14))
        assertArrayEquals(doubleArrayOf(1.0, 0.0), inputs.copyOfRange(16, 18))
        assertArrayEquals(doubleArrayOf(0.5, 0.5), inputs.copyOfRange(8, 10))
    }
    @Test fun convertsClampedGrayscale() {
        assertEquals(0, NeuroXorGrid.grayRgb(-1.0)); assertEquals(0, NeuroXorGrid.grayRgb(0.0))
        assertEquals(0x808080, NeuroXorGrid.grayRgb(0.5)); assertEquals(0xffffff, NeuroXorGrid.grayRgb(1.0))
        assertEquals(0xffffff, NeuroXorGrid.grayRgb(2.0))
        assertThrows(IllegalArgumentException::class.java) { NeuroXorGrid.grayRgb(Double.NaN) }
        assertThrows(IllegalArgumentException::class.java) { NeuroXorGrid.grayRgb(Double.POSITIVE_INFINITY) }
    }
    @Test fun rendersExactBatchResults() {
        val network = Neuro(intArrayOf(2, 3, 1))
        val outputs = DoubleArray(16)
        val inputs = NeuroXorGrid.createInputs(4)
        val image = NeuroXorGrid.render(network, 4, inputs, outputs)
        assertEquals(4, image.width); assertEquals(4, image.height)
        for (y in 0..3) for (x in 0..3) assertEquals(NeuroXorGrid.grayRgb(outputs[y * 4 + x]), image.getRGB(x, y) and 0xffffff)
        assertThrows(IllegalArgumentException::class.java) { NeuroXorGrid.createInputs(1) }
        assertThrows(ArithmeticException::class.java) { NeuroXorGrid.createInputs(Int.MAX_VALUE) }
        assertThrows(IllegalArgumentException::class.java) { NeuroXorGrid.render(network, 1, DoubleArray(2), DoubleArray(1)) }
        assertThrows(IllegalArgumentException::class.java) { NeuroXorGrid.render(network, 2, DoubleArray(7), DoubleArray(4)) }
        assertThrows(IllegalArgumentException::class.java) { NeuroXorGrid.render(network, 2, DoubleArray(8), DoubleArray(3)) }
    }
}
