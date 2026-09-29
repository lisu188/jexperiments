package com.lis.neuro

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Assumptions.assumeTrue
import org.junit.jupiter.api.Test

class NeuroNativeBlasTest {
    @Test fun nativeBackendMatchesJavaAcrossBatchGrowthAndSnapshots() {
        val model = NeuroTest.prepared(intArrayOf(17, 33, 9, 2))
        val optional = NeuroNativeBlas.tryCreate(model)
        assumeTrue(optional.isPresent, "Optional CBLAS library is not installed")
        optional.orElseThrow().use { native ->
            assertTrue(native.libraryName().isNotBlank())
            for (size in listOf(0, 1, 5, 20, 3)) {
                val inputs = DoubleArray(size * 17) { (it % 11) / 11.0 }
                val outputs = DoubleArray(size * 2); val expected = DoubleArray(size * 2)
                model.predictBatch(inputs, size, expected)
                native.predictBatch(inputs, size, outputs)
                assertArrayEquals(expected, outputs, 1e-12)
            }
            assertThrows(IllegalArgumentException::class.java) { native.predictBatch(DoubleArray(1), -1, DoubleArray(1)) }
            assertThrows(IllegalArgumentException::class.java) { native.predictBatch(DoubleArray(1), 1, DoubleArray(1)) }
            val inputs = DoubleArray(17) { 0.5 }; val before = DoubleArray(2); val after = DoubleArray(2)
            native.predictBatch(inputs, 1, before); model.train(10); native.predictBatch(inputs, 1, after)
            assertArrayEquals(before, after, 0.0)
        }
    }
}
