package com.lis.neuro;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

final class NeuroXorGridTest {
    @Test
    void mapsPixelsToXyCoordinatesWithYIncreasingUpward() {
        var inputs = NeuroXorGrid.createInputs(3);

        assertPoint(inputs, 3, 0, 0, 0.0, 1.0);
        assertPoint(inputs, 3, 2, 0, 1.0, 1.0);
        assertPoint(inputs, 3, 0, 2, 0.0, 0.0);
        assertPoint(inputs, 3, 2, 2, 1.0, 0.0);
        assertPoint(inputs, 3, 1, 1, 0.5, 0.5);
    }

    @Test
    void convertsOutputToClampedGrayscale() {
        assertEquals(0x000000, NeuroXorGrid.grayRgb(-1.0));
        assertEquals(0x000000, NeuroXorGrid.grayRgb(0.0));
        assertEquals(0x808080, NeuroXorGrid.grayRgb(0.5));
        assertEquals(0xFFFFFF, NeuroXorGrid.grayRgb(1.0));
        assertEquals(0xFFFFFF, NeuroXorGrid.grayRgb(2.0));
        assertThrows(IllegalArgumentException.class, () -> NeuroXorGrid.grayRgb(Double.NaN));
        assertThrows(IllegalArgumentException.class, () -> NeuroXorGrid.grayRgb(Double.POSITIVE_INFINITY));
    }

    @Test
    void rendersBatchInferenceIntoImage() {
        var network = new Neuro(new int[]{2, 3, 1}, Neuro.HyperParameters.defaults().withSeed(42));
        var inputs = NeuroXorGrid.createInputs(4);
        var outputs = new double[16];

        var image = NeuroXorGrid.render(network, 4, inputs, outputs);

        assertEquals(4, image.getWidth());
        assertEquals(4, image.getHeight());
        for (int y = 0; y < 4; y++) {
            for (int x = 0; x < 4; x++) {
                assertEquals(NeuroXorGrid.grayRgb(outputs[y * 4 + x]), image.getRGB(x, y) & 0xFFFFFF);
            }
        }
    }

    @Test
    void validatesGridBuffers() {
        var network = new Neuro(new int[]{2, 1});

        assertThrows(IllegalArgumentException.class, () -> NeuroXorGrid.createInputs(1));
        assertThrows(NullPointerException.class, () -> NeuroXorGrid.render(null, 2, new double[8], new double[4]));
        assertThrows(NullPointerException.class, () -> NeuroXorGrid.render(network, 2, null, new double[4]));
        assertThrows(NullPointerException.class, () -> NeuroXorGrid.render(network, 2, new double[8], null));
        assertThrows(IllegalArgumentException.class, () -> NeuroXorGrid.render(network, 1, new double[2], new double[1]));
        assertThrows(IllegalArgumentException.class, () -> NeuroXorGrid.render(network, 2, new double[7], new double[4]));
        assertThrows(IllegalArgumentException.class, () -> NeuroXorGrid.render(network, 2, new double[8], new double[3]));
    }

    private static void assertPoint(
            double[] inputs,
            int size,
            int xPixel,
            int yPixel,
            double expectedX,
            double expectedY) {
        var offset = (yPixel * size + xPixel) * 2;
        assertEquals(expectedX, inputs[offset], 1.0e-12);
        assertEquals(expectedY, inputs[offset + 1], 1.0e-12);
    }
}
