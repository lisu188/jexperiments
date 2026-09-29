package com.lis.neuro;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import org.junit.jupiter.api.Test;

final class NeuroLearningSetsTest {
    @Test
    void booleanLearningSetsHaveExpectedTruthTables() {
        assertTargets(NeuroLearningSets.create(NeuroLearningSets.Kind.XOR, 1), 0, 1, 1, 0);
        assertTargets(NeuroLearningSets.create(NeuroLearningSets.Kind.AND, 1), 0, 0, 0, 1);
        assertTargets(NeuroLearningSets.create(NeuroLearningSets.Kind.OR, 1), 0, 1, 1, 1);
        assertTargets(NeuroLearningSets.create(NeuroLearningSets.Kind.NAND, 1), 1, 1, 1, 0);
        assertTargets(NeuroLearningSets.create(NeuroLearningSets.Kind.XNOR, 1), 1, 0, 0, 1);
    }

    @Test
    void generatedSetsAreDeterministicAndBounded() {
        for (var kind : new NeuroLearningSets.Kind[]{
                NeuroLearningSets.Kind.NOISY_XOR,
                NeuroLearningSets.Kind.CIRCLE,
                NeuroLearningSets.Kind.SPIRAL}) {
            var first = NeuroLearningSets.create(kind, 1234);
            var second = NeuroLearningSets.create(kind, 1234);

            assertEquals(first, second);
            assertFalse(first.isEmpty());
            for (var sample : first) {
                assertTrue(sample.x() >= 0.0 && sample.x() <= 1.0);
                assertTrue(sample.y() >= 0.0 && sample.y() <= 1.0);
                assertTrue(sample.target() == 0.0 || sample.target() == 1.0);
            }
        }
    }

    @Test
    void customSetStartsEmpty() {
        assertTrue(NeuroLearningSets.create(NeuroLearningSets.Kind.CUSTOM, 42).isEmpty());
    }

    @Test
    void addToCopiesSamplesIntoNetwork() {
        var network = new Neuro(new int[]{2, 3, 1});
        var samples = NeuroLearningSets.create(NeuroLearningSets.Kind.XOR, 42);

        NeuroLearningSets.addTo(network, samples);

        assertEquals(4, network.trainingSampleCount());
        assertTrue(Double.isFinite(network.trainingError()));
    }

    @Test
    void validatesSamplesAndArguments() {
        assertThrows(NullPointerException.class, () -> NeuroLearningSets.create(null, 1));
        assertThrows(IllegalArgumentException.class, () -> new NeuroLearningSets.Sample(-0.1, 0.5, 0));
        assertThrows(IllegalArgumentException.class, () -> new NeuroLearningSets.Sample(0.5, 1.1, 0));
        assertThrows(IllegalArgumentException.class, () -> new NeuroLearningSets.Sample(0.5, 0.5, 2));
        assertThrows(IllegalArgumentException.class, () -> new NeuroLearningSets.Sample(Double.NaN, 0.5, 0));
        assertThrows(NullPointerException.class, () -> NeuroLearningSets.addTo(null, new ArrayList<>()));
        assertThrows(
                NullPointerException.class,
                () -> NeuroLearningSets.addTo(new Neuro(new int[]{2, 1}), null));
    }

    @Test
    void labelsAreHumanReadable() {
        assertEquals("Noisy XOR", NeuroLearningSets.Kind.NOISY_XOR.toString());
        assertEquals("Custom", NeuroLearningSets.Kind.CUSTOM.toString());
    }

    private static void assertTargets(
            java.util.List<NeuroLearningSets.Sample> samples,
            int value00,
            int value01,
            int value10,
            int value11) {
        assertEquals(4, samples.size());
        assertEquals(value00, samples.get(0).target());
        assertEquals(value01, samples.get(1).target());
        assertEquals(value10, samples.get(2).target());
        assertEquals(value11, samples.get(3).target());
    }
}
