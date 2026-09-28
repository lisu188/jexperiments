package com.lis.neuro;

import java.util.SplittableRandom;

final class LegacyNeuroBaseline {
    private static final class Layer {
        private final int inputs;
        private final int outputs;
        private final double[] weights;
        private final double[] biases;

        private Layer(int inputs, int outputs, SplittableRandom random) {
            this.inputs = inputs;
            this.outputs = outputs;
            weights = new double[inputs * outputs];
            biases = new double[outputs];
            var limit = Math.sqrt(6.0 / (inputs + outputs));
            for (int index = 0; index < weights.length; index++) {
                weights[index] = random.nextDouble(-limit, limit);
            }
        }
    }

    private static final class Workspace {
        private final double[][] activations;

        private Workspace(int[] topology) {
            activations = new double[topology.length][];
            for (int layer = 0; layer < topology.length; layer++) {
                activations[layer] = new double[topology[layer]];
            }
        }
    }

    private final int[] topology;
    private final Layer[] layers;
    private final ThreadLocal<Workspace> workspace;

    LegacyNeuroBaseline(int[] topology) {
        this.topology = topology.clone();
        var random = new SplittableRandom(1234);
        layers = new Layer[topology.length - 1];
        for (int layer = 0; layer < layers.length; layer++) {
            layers[layer] = new Layer(topology[layer], topology[layer + 1], random);
        }
        workspace = ThreadLocal.withInitial(() -> new Workspace(this.topology));
    }

    void predictInto(double[] input, double[] output) {
        var local = workspace.get();
        System.arraycopy(input, 0, local.activations[0], 0, input.length);
        for (int layerIndex = 0; layerIndex < layers.length; layerIndex++) {
            var layer = layers[layerIndex];
            var source = local.activations[layerIndex];
            var destination = local.activations[layerIndex + 1];
            for (int out = 0; out < layer.outputs; out++) {
                var offset = out * layer.inputs;
                var sum = layer.biases[out];
                var in = 0;
                var limit = layer.inputs - (layer.inputs & 3);
                for (; in < limit; in += 4) {
                    sum = Math.fma(source[in], layer.weights[offset + in], sum);
                    sum = Math.fma(source[in + 1], layer.weights[offset + in + 1], sum);
                    sum = Math.fma(source[in + 2], layer.weights[offset + in + 2], sum);
                    sum = Math.fma(source[in + 3], layer.weights[offset + in + 3], sum);
                }
                for (; in < layer.inputs; in++) {
                    sum = Math.fma(source[in], layer.weights[offset + in], sum);
                }
                destination[out] = sigmoid(sum);
            }
        }
        System.arraycopy(local.activations[topology.length - 1], 0, output, 0, output.length);
    }

    private static double sigmoid(double value) {
        if (value >= 0.0) {
            return 1.0 / (1.0 + Math.exp(-value));
        }
        var exp = Math.exp(value);
        return exp / (1.0 + exp);
    }
}
