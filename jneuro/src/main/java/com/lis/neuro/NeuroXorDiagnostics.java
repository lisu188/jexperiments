package com.lis.neuro;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

final class NeuroXorDiagnostics {
    record Boundary(double x1, double y1, double x2, double y2) {
    }

    record Probe(double[] hidden, double output, double[] contributions, double outputPreActivation) {
        Probe {
            hidden = hidden.clone();
            contributions = contributions.clone();
        }

        @Override
        public double[] hidden() {
            return hidden.clone();
        }

        @Override
        public double[] contributions() {
            return contributions.clone();
        }
    }

    static final class Snapshot {
        private final int epoch;
        private final double error;
        private final double beta;
        private final double[] inputWeights;
        private final double[] hiddenBiases;
        private final double[] outputWeights;
        private final double outputBias;

        private Snapshot(
                int epoch,
                double error,
                double beta,
                double[] inputWeights,
                double[] hiddenBiases,
                double[] outputWeights,
                double outputBias) {
            this.epoch = epoch;
            this.error = error;
            this.beta = beta;
            this.inputWeights = inputWeights.clone();
            this.hiddenBiases = hiddenBiases.clone();
            this.outputWeights = outputWeights.clone();
            this.outputBias = outputBias;
        }

        int epoch() {
            return epoch;
        }

        double error() {
            return error;
        }

        int hiddenCount() {
            return hiddenBiases.length;
        }

        double inputWeight(int hidden, int input) {
            return inputWeights[hidden * 2 + input];
        }

        double hiddenBias(int hidden) {
            return hiddenBiases[hidden];
        }

        double outputWeight(int hidden) {
            return outputWeights[hidden];
        }

        double outputBias() {
            return outputBias;
        }

        double[] parameters() {
            var result = new double[inputWeights.length + hiddenBiases.length + outputWeights.length + 1];
            var offset = 0;
            System.arraycopy(inputWeights, 0, result, offset, inputWeights.length);
            offset += inputWeights.length;
            System.arraycopy(hiddenBiases, 0, result, offset, hiddenBiases.length);
            offset += hiddenBiases.length;
            System.arraycopy(outputWeights, 0, result, offset, outputWeights.length);
            result[result.length - 1] = outputBias;
            return result;
        }
    }

    private NeuroXorDiagnostics() {
    }

    static Snapshot capture(Neuro network, int epoch, double error) {
        Objects.requireNonNull(network, "network");
        var topology = network.topology();
        if (topology.length != 3 || topology[0] != 2 || topology[2] != 1) {
            throw new IllegalArgumentException("diagnostics require topology 2-hidden-1");
        }

        var inputWeights = network.backendWeights(0);
        var hiddenBiases = network.backendBiases(0);
        var outputWeights = network.backendWeights(1);
        var outputBiases = network.backendBiases(1);
        return new Snapshot(
                epoch,
                error,
                network.hyperParameters().beta(),
                inputWeights,
                hiddenBiases,
                outputWeights,
                outputBiases[0]);
    }

    static Probe probe(Snapshot snapshot, double x, double y) {
        Objects.requireNonNull(snapshot, "snapshot");
        var hidden = new double[snapshot.hiddenCount()];
        var contributions = new double[hidden.length];
        var outputZ = snapshot.outputBias();

        for (int neuron = 0; neuron < hidden.length; neuron++) {
            var z = Math.fma(snapshot.inputWeight(neuron, 0), x, snapshot.hiddenBias(neuron));
            z = Math.fma(snapshot.inputWeight(neuron, 1), y, z);
            hidden[neuron] = sigmoid(snapshot.beta * z);
            contributions[neuron] = hidden[neuron] * snapshot.outputWeight(neuron);
            outputZ += contributions[neuron];
        }

        return new Probe(hidden, sigmoid(snapshot.beta * outputZ), contributions, outputZ);
    }

    static BufferedImage renderOutputMap(Snapshot snapshot, int size) {
        Objects.requireNonNull(snapshot, "snapshot");
        validateSize(size);
        var image = new BufferedImage(size, size, BufferedImage.TYPE_INT_RGB);
        var pixels = ((DataBufferInt) image.getRaster().getDataBuffer()).getData();
        var scale = 1.0 / (size - 1);

        for (int yPixel = 0; yPixel < size; yPixel++) {
            var y = 1.0 - yPixel * scale;
            for (int xPixel = 0; xPixel < size; xPixel++) {
                var x = xPixel * scale;
                pixels[yPixel * size + xPixel] = NeuroXorGrid.grayRgb(probe(snapshot, x, y).output());
            }
        }
        return image;
    }

    static BufferedImage renderDifferenceMap(Snapshot before, Snapshot after, int size) {
        Objects.requireNonNull(before, "before");
        Objects.requireNonNull(after, "after");
        validateSize(size);
        var image = new BufferedImage(size, size, BufferedImage.TYPE_INT_RGB);
        var pixels = ((DataBufferInt) image.getRaster().getDataBuffer()).getData();
        var scale = 1.0 / (size - 1);

        for (int yPixel = 0; yPixel < size; yPixel++) {
            var y = 1.0 - yPixel * scale;
            for (int xPixel = 0; xPixel < size; xPixel++) {
                var x = xPixel * scale;
                var difference = probe(after, x, y).output() - probe(before, x, y).output();
                pixels[yPixel * size + xPixel] = differenceRgb(difference);
            }
        }
        return image;
    }

    static int differenceRgb(double difference) {
        if (!Double.isFinite(difference)) {
            throw new IllegalArgumentException("difference must be finite");
        }
        var magnitude = Math.min(1.0, Math.abs(difference) * 4.0);
        var neutral = 32;
        var strong = 230;
        var weak = (int) Math.round(neutral + (1.0 - magnitude) * 55.0);
        var channel = (int) Math.round(neutral + magnitude * (strong - neutral));
        return difference >= 0.0
                ? channel << 16 | weak << 8 | weak
                : weak << 16 | weak << 8 | channel;
    }

    static BufferedImage[] renderHiddenMaps(Snapshot snapshot, int size) {
        Objects.requireNonNull(snapshot, "snapshot");
        validateSize(size);

        var images = new BufferedImage[snapshot.hiddenCount()];
        var pixels = new int[images.length][];
        for (int neuron = 0; neuron < images.length; neuron++) {
            images[neuron] = new BufferedImage(size, size, BufferedImage.TYPE_INT_RGB);
            pixels[neuron] = ((DataBufferInt) images[neuron].getRaster().getDataBuffer()).getData();
        }

        var scale = 1.0 / (size - 1);
        for (int yPixel = 0; yPixel < size; yPixel++) {
            var y = 1.0 - yPixel * scale;
            for (int xPixel = 0; xPixel < size; xPixel++) {
                var x = xPixel * scale;
                var index = yPixel * size + xPixel;
                for (int neuron = 0; neuron < images.length; neuron++) {
                    var z = Math.fma(snapshot.inputWeight(neuron, 0), x, snapshot.hiddenBias(neuron));
                    z = Math.fma(snapshot.inputWeight(neuron, 1), y, z);
                    pixels[neuron][index] = NeuroXorGrid.grayRgb(sigmoid(snapshot.beta * z));
                }
            }
        }
        return images;
    }

    static Boundary boundary(Snapshot snapshot, int neuron) {
        Objects.requireNonNull(snapshot, "snapshot");
        if (neuron < 0 || neuron >= snapshot.hiddenCount()) {
            throw new IllegalArgumentException("hidden neuron index out of range");
        }

        var wx = snapshot.inputWeight(neuron, 0);
        var wy = snapshot.inputWeight(neuron, 1);
        var bias = snapshot.hiddenBias(neuron);
        var intersections = new ArrayList<double[]>(4);

        if (Math.abs(wy) > 1.0e-12) {
            addIntersection(intersections, 0.0, -bias / wy);
            addIntersection(intersections, 1.0, -(wx + bias) / wy);
        }
        if (Math.abs(wx) > 1.0e-12) {
            addIntersection(intersections, -bias / wx, 0.0);
            addIntersection(intersections, -(wy + bias) / wx, 1.0);
        }

        if (intersections.size() < 2) {
            return null;
        }

        var first = intersections.get(0);
        var second = intersections.get(1);
        return new Boundary(first[0], first[1], second[0], second[1]);
    }

    static double maxAbsWeight(Snapshot snapshot) {
        Objects.requireNonNull(snapshot, "snapshot");
        var max = 0.0;
        for (int neuron = 0; neuron < snapshot.hiddenCount(); neuron++) {
            max = Math.max(max, Math.abs(snapshot.inputWeight(neuron, 0)));
            max = Math.max(max, Math.abs(snapshot.inputWeight(neuron, 1)));
            max = Math.max(max, Math.abs(snapshot.outputWeight(neuron)));
        }
        return max;
    }

    static double weightNorm(Snapshot snapshot, int layer) {
        Objects.requireNonNull(snapshot, "snapshot");
        var sum = 0.0;
        if (layer == 0) {
            for (int neuron = 0; neuron < snapshot.hiddenCount(); neuron++) {
                sum += square(snapshot.inputWeight(neuron, 0));
                sum += square(snapshot.inputWeight(neuron, 1));
            }
        } else if (layer == 1) {
            for (int neuron = 0; neuron < snapshot.hiddenCount(); neuron++) {
                sum += square(snapshot.outputWeight(neuron));
            }
        } else {
            throw new IllegalArgumentException("layer must be 0 or 1");
        }
        return Math.sqrt(sum);
    }

    static double biasNorm(Snapshot snapshot, int layer) {
        Objects.requireNonNull(snapshot, "snapshot");
        var sum = 0.0;
        if (layer == 0) {
            for (int neuron = 0; neuron < snapshot.hiddenCount(); neuron++) {
                sum += square(snapshot.hiddenBias(neuron));
            }
        } else if (layer == 1) {
            sum = square(snapshot.outputBias());
        } else {
            throw new IllegalArgumentException("layer must be 0 or 1");
        }
        return Math.sqrt(sum);
    }

    private static void validateSize(int size) {
        if (size < 2) {
            throw new IllegalArgumentException("size must be >= 2");
        }
    }

    private static void addIntersection(List<double[]> intersections, double x, double y) {
        if (x < -1.0e-9 || x > 1.0 + 1.0e-9 || y < -1.0e-9 || y > 1.0 + 1.0e-9) {
            return;
        }

        var clampedX = Math.max(0.0, Math.min(1.0, x));
        var clampedY = Math.max(0.0, Math.min(1.0, y));
        for (var point : intersections) {
            if (Math.abs(point[0] - clampedX) < 1.0e-9 && Math.abs(point[1] - clampedY) < 1.0e-9) {
                return;
            }
        }
        intersections.add(new double[]{clampedX, clampedY});
    }

    private static double square(double value) {
        return value * value;
    }

    private static double sigmoid(double value) {
        if (value >= 0.0) {
            return 1.0 / (1.0 + Math.exp(-value));
        }
        var exp = Math.exp(value);
        return exp / (1.0 + exp);
    }
}
