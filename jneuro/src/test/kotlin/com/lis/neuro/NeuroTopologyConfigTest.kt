package com.lis.neuro

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

class NeuroTopologyConfigTest {
    @Test fun parsesAndCopiesTopology() {
        assertArrayEquals(intArrayOf(1), NeuroTopologyConfig.parseHidden("1"))
        assertArrayEquals(intArrayOf(6, 4, 2), NeuroTopologyConfig.parseHidden(" 6, 4,2 "))
        val hidden = intArrayOf(6, 4, 2)
        val topology = NeuroTopologyConfig.topology(hidden)
        assertArrayEquals(intArrayOf(2, 6, 4, 2, 1), topology)
        assertEquals("6,4,2", NeuroTopologyConfig.format(hidden))
        assertEquals("2 → 6 → 4 → 2 → 1", NeuroTopologyConfig.label(topology))
        topology[1] = 99; assertEquals(6, hidden[0])
        assertEquals("", NeuroTopologyConfig.format(intArrayOf()))
    }
    @Test fun editsLayers() {
        val original = intArrayOf(3)
        val added = NeuroTopologyConfig.addLayer(original, 3)
        assertArrayEquals(intArrayOf(3, 3), added); assertArrayEquals(original, NeuroTopologyConfig.removeLayer(added))
        assertArrayEquals(intArrayOf(3), original)
    }
    @Test fun rejectsMalformedOrExcessiveTopology() {
        for (text in listOf("0", "-2", "2,,3", "a", "2,", ",2", "9999999999999"))
            assertThrows(IllegalArgumentException::class.java) { NeuroTopologyConfig.parseHidden(text) }
        for (hidden in listOf(intArrayOf(0), intArrayOf(-1)))
            assertThrows(IllegalArgumentException::class.java) { NeuroTopologyConfig.topology(hidden) }
        assertThrows(IllegalArgumentException::class.java) { NeuroTopologyConfig.removeLayer(intArrayOf()) }
        assertEquals(9, NeuroTopologyConfig.addLayer(IntArray(8) { 1 }, 1).size)
        assertThrows(IllegalArgumentException::class.java) { NeuroTopologyConfig.addLayer(intArrayOf(2), 0) }
    }
}
