package com.lis.neuro;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

final class NeuroTopologySpecTest {
    @Test
    void parsesSingleAndMultipleHiddenLayers() {
        assertArrayEquals(new int[]{2, 1, 1}, NeuroTopologySpec.parseHiddenLayers("1"));
        assertArrayEquals(new int[]{2, 2, 1}, NeuroTopologySpec.parseHiddenLayers(" 2 "));
        assertArrayEquals(new int[]{2, 3, 2, 1}, NeuroTopologySpec.parseHiddenLayers("3,2"));
        assertArrayEquals(new int[]{2, 8, 4, 2, 1}, NeuroTopologySpec.parseHiddenLayers("8, 4, 2"));
    }

    @Test
    void emptySpecificationMeansNoHiddenLayer() {
        assertArrayEquals(new int[]{2, 1}, NeuroTopologySpec.parseHiddenLayers(""));
        assertArrayEquals(new int[]{2, 1}, NeuroTopologySpec.parseHiddenLayers("   "));
        assertEquals("", NeuroTopologySpec.hiddenLayersText(new int[]{2, 1}));
    }

    @Test
    void formatsTopologyAndCountsParameters() {
        var topology = new int[]{2, 3, 2, 1};

        assertEquals("3,2", NeuroTopologySpec.hiddenLayersText(topology));
        assertEquals("2 → 3 → 2 → 1", NeuroTopologySpec.display(topology));
        assertEquals(5, NeuroTopologySpec.hiddenNeuronCount(topology));
        assertEquals(20, NeuroTopologySpec.parameterCount(topology));
        assertEquals(3, NeuroTopologySpec.parameterCount(new int[]{2, 1}));
    }

    @Test
    void rejectsInvalidSpecifications() {
        assertThrows(IllegalArgumentException.class, () -> NeuroTopologySpec.parseHiddenLayers(null));
        assertThrows(IllegalArgumentException.class, () -> NeuroTopologySpec.parseHiddenLayers("0"));
        assertThrows(IllegalArgumentException.class, () -> NeuroTopologySpec.parseHiddenLayers("-1"));
        assertThrows(IllegalArgumentException.class, () -> NeuroTopologySpec.parseHiddenLayers("2,,3"));
        assertThrows(IllegalArgumentException.class, () -> NeuroTopologySpec.parseHiddenLayers("abc"));
        assertThrows(IllegalArgumentException.class, () -> NeuroTopologySpec.parseHiddenLayers("257"));
        assertThrows(
                IllegalArgumentException.class,
                () -> NeuroTopologySpec.parseHiddenLayers("1,1,1,1,1,1,1,1,1"));
        assertThrows(
                IllegalArgumentException.class,
                () -> NeuroTopologySpec.parseHiddenLayers("32,33"));
    }

    @Test
    void rejectsInvalidVisualTopologies() {
        assertThrows(IllegalArgumentException.class, () -> NeuroTopologySpec.display(null));
        assertThrows(IllegalArgumentException.class, () -> NeuroTopologySpec.display(new int[]{3, 1}));
        assertThrows(IllegalArgumentException.class, () -> NeuroTopologySpec.display(new int[]{2, 2}));
        assertThrows(IllegalArgumentException.class, () -> NeuroTopologySpec.display(new int[]{2, 0, 1}));
    }
}
