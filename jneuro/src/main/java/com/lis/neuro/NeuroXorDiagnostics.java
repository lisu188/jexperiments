package com.lis.neuro;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

final class NeuroXorDiagnostics {
    record Boundary(double x1, double y1, double x2, double y2) {
    }

    record Probe(double[][] activations, double output, double[] contributions, double outputPreActivation) {
        Probe {
            activations = deepCopy(activations);
            contributions = contributions.clone();
        }

        @Override
        public double[][] activations() {
            return deepCopy(activations);
        }

        double[] layerActivations(int layer) {
            return activations[layer].clone();
        }

        double[] hidden() {
            return activations.length > 1 ? activations[0].clone() : new double[0];
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
            this.weights = deepCopy(weights);
            this.biases = deepCopy(biases);
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

        int layerCount() {
            return weights.length;
        }

        int hiddenLayerCount() {
            return Math.max(0, topology.length - 2);
        }

        int hiddenCount() {
            return hiddenLayerCount() == 0 ? 0 : hiddenLayerSize(0);
        }

        int hiddenLayerSize(int hiddenLayer) {
            validateHiddenLayer(hiddenLayer);
            return topology[hiddenLayer + 1];
        }

        int hiddenNeuronCount() {
            var count = 0;
            for (int layer = 1; layer < topology.length - 1; layer++) {
                count += topology[layer];
            }
            return count;
        }

        int layerInputSize(int layer) {
            validateLayer(layer);
            return topology[layer];
        }

        int layerOutputSize(int layer) {
            validateLayer(layer);
            return topology[layer + 1];
        }

        double layerWeight(int layer, int output, int input) {
            validateLayer(layer);
            var inputs = topology[layer];
            return weights[layer][output * inputs + input];
        }

        double layerBias(int layer, int output) {
            validateLayer(layer);
            return biases[layer][output];
        }

        double inputWeight(int hidden, int input) {
            requireSingleFirstHiddenLayer();
            return layerWeight(0, hidden, input);
        }

        double hiddenBias(int hidden) {
            requireSingleFirstHiddenLayer();
            return layerBias(0, hidden);
        }

        double outputWeight(int hidden) {
            if (weights.length != 2) {
                throw new IllegalStateException("outputWeight is only defined for one hidden layer");
            }
            return layerWeight(1, 0, hidden);
        }

        double outputBias() {
            return layerBias(weights.length - 1, 0);
        }

        double[] layerWeights(int layer) {
            validateLayer(layer);
            return weights[layer].clone();
        }

        double[] layerBiases(int layer) {
            validateLayer(layer);
            return biases[layer].clone();
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
            var count = 0;
            for (int layer = 0; layer < weights.length; layer++) {
                count += weights[layer].length + biases[layer].length;
            }
            return count;
        }

        int layerParameterOffset(int layer) {
            validateLayer(layer);
            var offset = 0;
            for (int index = 0; index < layer; index++) {
                offset += weights[index].length + biases[index].length;
            }
            return offset;
        }

        int layerParameterCount(int layer) {
            validateLayer(layer);
            return weights[layer].length + biases[layer].length;
        }

        private void requireSingleFirstHiddenLayer() {
            if (hiddenLayerCount() < 1) {
                throw new IllegalStateException("network has no hidden layer");
            }
        }

        private void validateLayer(int layer) {
            if (layer < 0 || layer >= weights.length) {
                throw new IllegalArgumentException("layer index out of range");
            }
        }

        private void validateHiddenLayer(int hiddenLayer) {
            if (hiddenLayer < 0 || hiddenLayer >= hiddenLayerCount()) {
                throw new IllegalArgumentException("hidden layer index out of range");
            }
        }
    }

    private NeuroXorDiagnostics() {
    }

    static Snapshot capture(Neuro network, int epoch, double error) {
        Objects.requireNonNull(network, "network");
        var topology = network.topology();
        if (topology.length < 2 || topology[0] != 2 || topology[topology.length - 1] != 1) {
            throw new IllegalArgumentException("diagnostics require topology 2-...-1");
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
        var activations = forward(snapshot, x, y);
        var outputLayer = snapshot.layerCount() - 1;
        var source = outputLayer == 0
                ? new double[]{x, y}
                : activations[outputLayer - 1];
        var contributions = new double[source.length];
        var outputZ = snapshot.layerBias(outputLayer, 0);
        for (int input = 0; input < source.length; input++) {
            contributions[input] = source[input] * snapshot.layerWeight(outputLayer, 0, input);
            outputZ += contributions[input];
        }
        return new Probe(activations, activations[activations.length - 1][0], contributions, outputZ);
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

        var images = new BufferedImage[snapshot.hiddenNeuronCount()];
        var pixels = new int[images.length][];
        for (int index = 0; index < images.length; index++) {
            images[index] = new BufferedImage(size, size, BufferedImage.TYPE_INT_RGB);
            pixels[index] = ((DataBufferInt) images[index].getRaster().getDataBuffer()).getData();
        }

        var scale = 1.0 / (size - 1);
        for (int yPixel = 0; yPixel < size; yPixel++) {
            var y = 1.0 - yPixel * scale;
            for (int xPixel = 0; xPixel < size; xPixel++) {
                var x = xPixel * scale;
                var activations = forward(snapshot, x, y);
                var mapIndex = 0;
                for (int hiddenLayer = 0; hiddenLayer < snapshot.hiddenLayerCount(); hiddenLayer++) {
                    for (var activation : activations[hiddenLayer]) {
                        pixels[mapIndex++][yPixel * size + xPixel] = NeuroXorGrid.grayRgb(activation);
                    }
                }
            }
        }
        return images;
    }

    static int hiddenMapOffset(Snapshot snapshot, int hiddenLayer) {
        Objects.requireNonNull(snapshot, "snapshot");
        if (hiddenLayer < 0 || hiddenLayer >= snapshot.hiddenLayerCount()) {
            throw new IllegalArgumentException("hidden layer index out of range");
        }
        var offset = 0;
        for (int layer = 0; layer < hiddenLayer; layer++) {
            offset += snapshot.hiddenLayerSize(layer);
        }
        return offset;
    }

    static Boundary boundary(Snapshot snapshot, int neuron) {
        Objects.requireNonNull(snapshot, "snapshot");
        if (snapshot.hiddenLayerCount() == 0) {
            return null;
        }
        if (neuron < 0 || neuron >= snapshot.hiddenLayerSize(0)) {
            throw new IllegalArgumentException("hidden neuron index out of range");
        }

        var wx = snapshot.layerWeight(0, neuron, 0);
        var wy = snapshot.layerWeight(0, neuron, 1);
        var bias = snapshot.layerBias(0, neuron);
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
            for (var weight : snapshot.layerWeights(layer)) {
                max = Math.max(max, Math.abs(weight));
            }
        }
        return max;
    }

    static double weightNorm(Snapshot snapshot, int layer) {
        Objects.requireNonNull(snapshot, "snapshot");
        var sum = 0.0;
        for (var value : snapshot.layerWeights(layer)) {
            sum += square(value);
        }
        return Math.sqrt(sum);
    }

    static double biasNorm(Snapshot snapshot, int layer) {
        Objects.requireNonNull(snapshot, "snapshot");
        var sum = 0.0;
        for (var value : snapshot.layerBiases(layer)) {
            sum += square(value);
        }
        return Math.sqrt(sum);
    }

    private static double[][] forward(Snapshot snapshot, double x, double y) {
        var activations = new double[snapshot.layerCount()][];
        double[] source = {x, y};

        for (int layer = 0; layer < snapshot.layerCount(); layer++) {
            var outputs = snapshot.layerOutputSize(layer);
            var destination = new double[outputs];
            for (int output = 0; output < outputs; output++) {
                var sum = snapshot.layerBias(layer, output);
                for (int input = 0; input < source.length; input++) {
                    sum = Math.fma(
                            source[input],
                            snapshot.layerWeight(layer, output, input),
                            sum);
                }
                destination[output] = sigmoid(snapshot.beta * sum);
            }
            activations[layer] = destination;
            source = destination;
        }
        return activations;
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

    private static double[][] deepCopy(double[][] values) {
        return Arrays.stream(values).map(double[]::clone).toArray(double[][]::new);
    }
}
