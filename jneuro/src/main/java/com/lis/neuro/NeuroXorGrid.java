package com.lis.neuro;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.util.Objects;

final class NeuroXorGrid {
    private NeuroXorGrid() {
    }

    static double[] createInputs(int size) {
        if (size < 2) {
            throw new IllegalArgumentException("size must be >= 2");
        }

        var inputs = new double[size * size * 2];
        var scale = 1.0 / (size - 1);
        for (int yPixel = 0; yPixel < size; yPixel++) {
            var y = 1.0 - yPixel * scale;
            for (int xPixel = 0; xPixel < size; xPixel++) {
                var sample = yPixel * size + xPixel;
                inputs[sample * 2] = xPixel * scale;
                inputs[sample * 2 + 1] = y;
            }
        }
        return inputs;
    }

    static int grayRgb(double output) {
        if (!Double.isFinite(output)) {
            throw new IllegalArgumentException("output must be finite");
        }
        var clamped = Math.max(0.0, Math.min(1.0, output));
        var gray = (int) Math.round(clamped * 255.0);
        return gray << 16 | gray << 8 | gray;
    }

    static BufferedImage render(Neuro network, int size, double[] inputs, double[] outputs) {
        Objects.requireNonNull(network, "network");
        Objects.requireNonNull(inputs, "inputs");
        Objects.requireNonNull(outputs, "outputs");

        var samples = size * size;
        if (size < 2) {
            throw new IllegalArgumentException("size must be >= 2");
        }
        if (inputs.length != samples * 2) {
            throw new IllegalArgumentException("inputs length must equal size * size * 2");
        }
        if (outputs.length != samples) {
            throw new IllegalArgumentException("outputs length must equal size * size");
        }

        network.predictBatch(inputs, samples, outputs);

        var image = new BufferedImage(size, size, BufferedImage.TYPE_INT_RGB);
        var pixels = ((DataBufferInt) image.getRaster().getDataBuffer()).getData();
        for (int i = 0; i < pixels.length; i++) {
            pixels[i] = grayRgb(outputs[i]);
        }
        return image;
    }
}
