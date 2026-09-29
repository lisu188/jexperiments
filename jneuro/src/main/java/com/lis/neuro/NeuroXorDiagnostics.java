package com.lis.neuro;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

final class NeuroXorDiagnostics {
    record Boundary(double x1, double y1, double x2, double y2) {
    }

    record Probe(
            double[][] hiddenLayers,
            double output,
            double[] contributions,
            double outputPreActivation) {
        Probe {
            hiddenLayers = deepClone(hiddenLayers);
            contributions = contributions.clone();
        }

        @Override
        public double[][] hiddenLayers() {
            return deepClone(hiddenLayers);
        }

        @Override
        public double[] contributions() {
            return contributions.clone();
        }

        double[] hiddenLayer(int index) {
            return hiddenLayers[index].clone();
        }

        double[] hidden() {
            return hiddenLayers[hiddenLayers.length - 1].clone();
        }

        private static double[][] deepClone(double[][] source) {
            var result = new double[source.length][];
            for (int i = 0; i < source.length; i++) {
                result[i] = source[i].clone();
            }
            return result;
        }
    }

    static final class Snapshot {
        private final int epoch;
        private final double error;
        private final double beta;
        private final int[] topology;
        private final double[][] weights;
        private final double[][] biases;

        private Snapshot(
                int epoch,
                double error,
                double beta,
                int[] topology,
                double[][] weights,
                double[][] biases) {
            this.epoch = epoch;
            this.error = error;
            this.beta = beta;
            this.topology = topology.clone();
            this.weights = deepClone(weights);
            this.biases = deepClone(biases);
        }

        int epoch() {
            return epoch;
        }

        double error() {
            return error;
        }

        int[] topology() {
            return topology.clone();
        }

        int hiddenLayerCount() {
            return topology.length - 2;
        }

        int hiddenCount() {
            return topology[1];
        }

        int layerCount() {
            return weights.length;
        }

        int layerInputCount(int layer) {
            return topology[layer];
        }

        int layerOutputCount(int layer) {
            return topology[layer + 1];
        }

        double weight(int layer, int output, int input) {
            return weights[layer][output * topology[layer] + input];
        }

        double bias(int layer, int output) {
            return biases[layer][output];
        }

        double inputWeight(int hidden, int input) {
            return weight(0, hidden, input);
        }

        double hiddenBias(int hidden) {
            return bias(0, hidden);
        }

        double outputWeight(int hidden) {
            var layer = weights.length - 1;
            return weight(layer, 0, hidden);
        }

        double outputBias() {
            return bias(weights.length - 1, 0);
        }

        double[] parameters() {
            var count = 0;
            for (int layer = 0; layer < weights.length; layer++) {
                count += weights[layer].length + biases[layer].length;
            }

            var result = new double[count];
            var offset = 0;
            for (int layer = 0; layer < weights.length; layer++) {
                System.arraycopy(weights[layer], 0, result, offset, weights[layer].length);
                offset += weights[layer].length;
                System.arraycopy(biases[layer], 0, result, offset, biases[layer].length);
                offset += biases[layer].length;
            }
            return result;
        }

        int parameterCount() {
            return parameters().length;
        }

        private static double[][] deepClone(double[][] source) {
            var result = new double[source.length][];
            for (int i = 0; i < source.length; i++) {
                result[i] = source[i].clone();
            }
            return result;
        }
    }

    private NeuroXorDiagnostics() {
    }

    static Snapshot capture(Neuro network, int epoch, double error) {
        Objects.requireNonNull(network, "network");
        var topology = network.topology();
        if (topology.length < 3 || topology[0] != 2 || topology[topology.length - 1] != 1) {
            throw new IllegalArgumentException("diagnostics require topology 2-hidden...-1");
        }

        var weights = new double[topology.length - 1][];
        var biases = new double[topology.length - 1][];
        for (int layer = 0; layer < weights.length; layer++) {
            weights[layer] = network.backendWeights(layer);
            biases[layer] = network.backendBiases(layer);
        }
        return new Snapshot(
                epoch,
                error,
                network.hyperParameters().beta(),
                topology,
                weights,
                biases);
    }

    static Probe probe(Snapshot snapshot, double x, double y) {
        Objects.requireNonNull(snapshot, "snapshot");
        var topology = snapshot.topology;
        var hiddenLayers = new double[topology.length - 2][];
        var source = new double[]{x, y};

        for (int layer = 0; layer < snapshot.weights.length; layer++) {
            var outputCount = topology[layer + 1];
            var destination = new double[outputCount];
            for (int output = 0; output < outputCount; output++) {
                var sum = snapshot.bias(layer, output);
                for (int input = 0; input < source.length; input++) {
                    sum = Math.fma(source[input], snapshot.weight(layer, output, input), sum);
                }
                destination[output] = sigmoid(snapshot.beta * sum);
            }

            if (layer < hiddenLayers.length) {
                hiddenLayers[layer] = destination;
            }
            source = destination;
        }

        var lastHidden = hiddenLayers[hiddenLayers.length - 1];
        var outputLayer = snapshot.layerCount() - 1;
        var contributions = new double[lastHidden.length];
        var outputZ = snapshot.bias(outputLayer, 0);
        for (int input = 0; input < lastHidden.length; input++) {
            contributions[input] = lastHidden[input] * snapshot.weight(outputLayer, 0, input);
            outputZ += contributions[input];
        }
        return new Probe(hiddenLayers, source[0], contributions, outputZ);
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
        var neutral = 45;
        var strong = 235;
        var channel = (int) Math.round(neutral + magnitude * (strong - neutral));
        return difference >= 0.0
                ? channel << 16 | neutral << 8 | neutral
                : neutral << 16 | neutral << 8 | channel;
    }

    static BufferedImage[] renderHiddenMaps(Snapshot snapshot, int size) {
        Objects.requireNonNull(snapshot, "snapshot");
        validateSize(size);

        var images = new BufferedImage[Math.min(snapshot.hiddenCount(), 12)];
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
        for (int layer = 0; layer < snapshot.layerCount(); layer++) {
            for (int output = 0; output < snapshot.layerOutputCount(layer); output++) {
                for (int input = 0; input < snapshot.layerInputCount(layer); input++) {
                    max = Math.max(max, Math.abs(snapshot.weight(layer, output, input)));
                }
            }
        }
        return max;
    }

    static double weightNorm(Snapshot snapshot, int layer) {
        Objects.requireNonNull(snapshot, "snapshot");
        validateLayer(snapshot, layer);
        var sum = 0.0;
        for (int output = 0; output < snapshot.layerOutputCount(layer); output++) {
            for (int input = 0; input < snapshot.layerInputCount(layer); input++) {
                sum += square(snapshot.weight(layer, output, input));
            }
        }
        return Math.sqrt(sum);
    }

    static double biasNorm(Snapshot snapshot, int layer) {
        Objects.requireNonNull(snapshot, "snapshot");
        validateLayer(snapshot, layer);
        var sum = 0.0;
        for (int output = 0; output < snapshot.layerOutputCount(layer); output++) {
            sum += square(snapshot.bias(layer, output));
        }
        return Math.sqrt(sum);
    }

    private static void validateLayer(Snapshot snapshot, int layer) {
        if (layer < 0 || layer >= snapshot.layerCount()) {
            throw new IllegalArgumentException("layer index out of range");
        }
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
