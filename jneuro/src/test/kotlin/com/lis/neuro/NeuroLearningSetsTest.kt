package com.lis.neuro

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

class NeuroLearningSetsTest {
    @Test fun booleanLearningSetsHaveExpectedTruthTables() {
        val cases = mapOf(NeuroLearningSets.Kind.XOR to listOf(0, 1, 1, 0), NeuroLearningSets.Kind.AND to listOf(0, 0, 0, 1),
            NeuroLearningSets.Kind.OR to listOf(0, 1, 1, 1), NeuroLearningSets.Kind.NAND to listOf(1, 1, 1, 0), NeuroLearningSets.Kind.XNOR to listOf(1, 0, 0, 1))
        for ((kind, expected) in cases) {
            val samples = NeuroLearningSets.create(kind, 1)
            assertEquals(expected.map { it.toDouble() }, samples.map { it.target })
            assertEquals(listOf(0.0 to 0.0, 0.0 to 1.0, 1.0 to 0.0, 1.0 to 1.0), samples.map { it.x to it.y })
        }
    }
    @Test fun generatedSetsAreDeterministicBoundedAndHaveBothClasses() {
        for (kind in listOf(NeuroLearningSets.Kind.NOISY_XOR, NeuroLearningSets.Kind.CIRCLE, NeuroLearningSets.Kind.SPIRAL)) {
            val first = NeuroLearningSets.create(kind, 1234)
            assertEquals(first, NeuroLearningSets.create(kind, 1234))
            assertFalse(first.isEmpty())
            assertEquals(setOf(0.0, 1.0), first.map { it.target }.toSet())
            for (sample in first) assertTrue(sample.x in 0.0..1.0 && sample.y in 0.0..1.0)
        }
        assertNotEquals(NeuroLearningSets.create(NeuroLearningSets.Kind.NOISY_XOR, 1), NeuroLearningSets.create(NeuroLearningSets.Kind.NOISY_XOR, 2))
        assertTrue(NeuroLearningSets.create(NeuroLearningSets.Kind.CUSTOM, 42).isEmpty())
        val model = Neuro(intArrayOf(2, 3, 1))
        NeuroLearningSets.addTo(model, NeuroLearningSets.create(NeuroLearningSets.Kind.XOR, 42))
        assertEquals(4, model.trainingSampleCount()); assertTrue(model.trainingError().isFinite())
    }
    @Test fun validatesSamples() {
        for (values in listOf(doubleArrayOf(-0.1, 0.5, 0.0), doubleArrayOf(0.5, 1.1, 0.0), doubleArrayOf(0.5, 0.5, 2.0),
            doubleArrayOf(Double.NaN, 0.5, 0.0), doubleArrayOf(0.5, Double.POSITIVE_INFINITY, 0.0), doubleArrayOf(0.5, 0.5, Double.NaN))) {
            assertThrows(IllegalArgumentException::class.java) { NeuroLearningSets.Sample(values[0], values[1], values[2]) }
        }
        assertEquals("Noisy XOR", NeuroLearningSets.Kind.NOISY_XOR.toString())
        assertEquals("Custom", NeuroLearningSets.Kind.CUSTOM.toString())
        for (kind in NeuroLearningSets.Kind.entries) assertTrue(kind.description.isNotBlank())
    }
}
