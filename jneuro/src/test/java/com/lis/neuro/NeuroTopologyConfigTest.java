package com.lis.neuro;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

final class NeuroTopologyConfigTest {
    @Test
    void parsesSingleAndMultipleHiddenLayers() {
        assertArrayEquals(new int[]{1}, NeuroTopologyConfig.parseHidden("1"));
        assertArrayEquals(new int[]{6, 4, 2}, NeuroTopologyConfig.parseHidden(" 6, 4,2 "));
        assertArrayEquals(new int[]{2, 6, 4, 2, 1}, NeuroTopologyConfig.topology(new int[]{6, 4, 2}));
        assertEquals("6,4,2", NeuroTopologyConfig.format(new int[]{6, 4, 2}));
    }

    @Test
    void addsAndRemovesLayers() {
        var added = NeuroTopologyConfig.addLayer(new int[]{3}, 3);
        assertArrayEquals(new int[]{3, 3}, added);
        assertArrayEquals(new int[]{3}, NeuroTopologyConfig.removeLayer(added));
    }

    @Test
    void validatesConfiguration() {
        assertThrows(NullPointerException.class, () -> NeuroTopologyConfig.parseHidden(null));
        assertThrows(IllegalArgumentException.class, () -> NeuroTopologyConfig.parseHidden(""));
        assertThrows(IllegalArgumentException.class, () -> NeuroTopologyConfig.parseHidden("0"));
        assertThrows(IllegalArgumentException.class, () -> NeuroTopologyConfig.parseHidden("129"));
        assertThrows(IllegalArgumentException.class, () -> NeuroTopologyConfig.parseHidden("2,,3"));
        assertThrows(IllegalArgumentException.class, () -> NeuroTopologyConfig.parseHidden("a"));
        assertThrows(IllegalArgumentException.class, () -> NeuroTopologyConfig.parseHidden("1,1,1,1,1,1,1,1,1"));
        assertThrows(NullPointerException.class, () -> NeuroTopologyConfig.topology(null));
        assertThrows(IllegalArgumentException.class, () -> NeuroTopologyConfig.topology(new int[0]));
        assertThrows(IllegalArgumentException.class, () -> NeuroTopologyConfig.topology(new int[]{0}));
        assertThrows(NullPointerException.class, () -> NeuroTopologyConfig.addLayer(null, 2));
        assertThrows(IllegalArgumentException.class, () -> NeuroTopologyConfig.removeLayer(new int[]{2}));
        assertThrows(NullPointerException.class, () -> NeuroTopologyConfig.removeLayer(null));
        assertThrows(NullPointerException.class, () -> NeuroTopologyConfig.format(null));
    }
}
