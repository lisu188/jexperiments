package com.lis.neuro;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

final class NeuroXorDiagnosticsTest {
    @Test
    void capturesTwoHiddenOneTopologyAndMatchesNetworkOutput() {
        var network = xorNetwork();
        network.train(25);

        var snapshot = NeuroXorDiagnostics.capture(network, 25, network.trainingError());
        var probe = NeuroXorDiagnostics.probe(snapshot, 0.23, 0.81);

        assertEquals(25, snapshot.epoch());
        assertEquals(network.trainingError(), snapshot.error(), 0.0);
        assertEquals(6, snapshot.hiddenCount());
        assertEquals(network.predict(new double[]{0.23, 0.81})[0], probe.output(), 1.0e-12);
        assertEquals(6, probe.hidden().length);
        assertEquals(6, probe.contributions().length);

        var sum = snapshot.outputBias();
        for (var contribution : probe.contributions()) {
            sum += contribution;
        }
        assertEquals(sum, probe.outputPreActivation(), 1.0e-12);
    }

    @Test
    void probeDefensivelyCopiesArrays() {
        var snapshot = NeuroXorDiagnostics.capture(xorNetwork(), 0, 0.5);
        var probe = NeuroXorDiagnostics.probe(snapshot, 0.5, 0.5);

        var hidden = probe.hidden();
        var contributions = probe.contributions();
        var originalHidden = hidden.clone();
        var originalContributions = contributions.clone();
        hidden[0] = -100;
        contributions[0] = -100;

        assertArrayEquals(originalHidden, probe.hidden());
        assertArrayEquals(originalContributions, probe.contributions());
    }

    @Test
    void rendersOneActivationMapPerHiddenNeuron() {
        var snapshot = NeuroXorDiagnostics.capture(xorNetwork(), 0, 0.5);
        var images = NeuroXorDiagnostics.renderHiddenMaps(snapshot, 5);

        assertEquals(6, images.length);
        for (var image : images) {
            assertEquals(5, image.getWidth());
            assertEquals(5, image.getHeight());
        }

        var probeTopLeft = NeuroXorDiagnostics.probe(snapshot, 0.0, 1.0);
        for (int neuron = 0; neuron < images.length; neuron++) {
            assertEquals(
                    NeuroXorGrid.grayRgb(probeTopLeft.hidden()[neuron]),
                    images[neuron].getRGB(0, 0) & 0xFFFFFF);
        }
    }

    @Test
    void decisionBoundariesLieOnHiddenPreActivationZero() {
        var snapshot = NeuroXorDiagnostics.capture(xorNetwork(), 0, 0.5);
        var found = 0;

        for (int neuron = 0; neuron < snapshot.hiddenCount(); neuron++) {
            var boundary = NeuroXorDiagnostics.boundary(snapshot, neuron);
            if (boundary == null) {
                continue;
            }
            found++;
            assertInsideUnitSquare(boundary.x1(), boundary.y1());
            assertInsideUnitSquare(boundary.x2(), boundary.y2());
            assertEquals(0.0, hiddenZ(snapshot, neuron, boundary.x1(), boundary.y1()), 1.0e-9);
            assertEquals(0.0, hiddenZ(snapshot, neuron, boundary.x2(), boundary.y2()), 1.0e-9);
        }

        assertTrue(found > 0);
    }

    @Test
    void reportsMaximumAbsoluteWeight() {
        var snapshot = NeuroXorDiagnostics.capture(xorNetwork(), 0, 0.5);
        var expected = 0.0;
        for (int neuron = 0; neuron < snapshot.hiddenCount(); neuron++) {
            expected = Math.max(expected, Math.abs(snapshot.inputWeight(neuron, 0)));
            expected = Math.max(expected, Math.abs(snapshot.inputWeight(neuron, 1)));
            expected = Math.max(expected, Math.abs(snapshot.outputWeight(neuron)));
        }

        assertEquals(expected, NeuroXorDiagnostics.maxAbsWeight(snapshot), 0.0);
        assertTrue(expected > 0.0);
    }

    @Test
    void validatesDiagnosticInputs() {
        var valid = NeuroXorDiagnostics.capture(xorNetwork(), 0, 0.5);

        assertThrows(NullPointerException.class, () -> NeuroXorDiagnostics.capture(null, 0, 0.0));
        assertThrows(
                IllegalArgumentException.class,
                () -> NeuroXorDiagnostics.capture(new Neuro(new int[]{2, 1}), 0, 0.0));
        assertThrows(NullPointerException.class, () -> NeuroXorDiagnostics.probe(null, 0.0, 0.0));
        assertThrows(NullPointerException.class, () -> NeuroXorDiagnostics.renderHiddenMaps(null, 4));
        assertThrows(
                IllegalArgumentException.class,
                () -> NeuroXorDiagnostics.renderHiddenMaps(valid, 1));
        assertThrows(NullPointerException.class, () -> NeuroXorDiagnostics.boundary(null, 0));
        assertThrows(IllegalArgumentException.class, () -> NeuroXorDiagnostics.boundary(valid, -1));
        assertThrows(
                IllegalArgumentException.class,
                () -> NeuroXorDiagnostics.boundary(valid, valid.hiddenCount()));
        assertThrows(NullPointerException.class, () -> NeuroXorDiagnostics.maxAbsWeight(null));
    }

    @Test
    void snapshotAccessorsExposeCapturedParameters() {
        var network = xorNetwork();
        var snapshot = NeuroXorDiagnostics.capture(network, 7, 0.25);
        var inputWeights = network.backendWeights(0);
        var hiddenBiases = network.backendBiases(0);
        var outputWeights = network.backendWeights(1);
        var outputBiases = network.backendBiases(1);

        assertEquals(7, snapshot.epoch());
        assertEquals(0.25, snapshot.error(), 0.0);
        assertEquals(6, snapshot.hiddenCount());
        for (int neuron = 0; neuron < snapshot.hiddenCount(); neuron++) {
            assertEquals(inputWeights[neuron * 2], snapshot.inputWeight(neuron, 0), 0.0);
            assertEquals(inputWeights[neuron * 2 + 1], snapshot.inputWeight(neuron, 1), 0.0);
            assertEquals(hiddenBiases[neuron], snapshot.hiddenBias(neuron), 0.0);
            assertEquals(outputWeights[neuron], snapshot.outputWeight(neuron), 0.0);
        }
        assertEquals(outputBiases[0], snapshot.outputBias(), 0.0);
        assertNotNull(NeuroXorDiagnostics.probe(snapshot, 0.0, 0.0));
    }

    private static Neuro xorNetwork() {
        var network = new Neuro(
                new int[]{2, 6, 1},
                Neuro.HyperParameters.defaults()
                        .withLearningRate(0.6)
                        .withMomentum(0.2)
                        .withSeed(42));
        network.addTrainingSample(new double[]{0, 0}, new double[]{0});
        network.addTrainingSample(new double[]{0, 1}, new double[]{1});
        network.addTrainingSample(new double[]{1, 0}, new double[]{1});
        network.addTrainingSample(new double[]{1, 1}, new double[]{0});
        return network;
    }

    private static double hiddenZ(
            NeuroXorDiagnostics.Snapshot snapshot,
            int neuron,
            double x,
            double y) {
        return snapshot.inputWeight(neuron, 0) * x
                + snapshot.inputWeight(neuron, 1) * y
                + snapshot.hiddenBias(neuron);
    }

    private static void assertInsideUnitSquare(double x, double y) {
        assertTrue(x >= 0.0 && x <= 1.0);
        assertTrue(y >= 0.0 && y <= 1.0);
    }
}
