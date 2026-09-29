package com.lis.neuro;

import java.util.Arrays;

final class NeuroTopologySpec {
    private NeuroTopologySpec() {
    }

    static int[] parseHiddenLayers(String text) {
        if (text == null) {
            throw new IllegalArgumentException("hidden layer specification is required");
        }

        var trimmed = text.trim();
        if (trimmed.isEmpty()) {
            return new int[]{2, 1};
        }

        var parts = trimmed.split(",");
        if (parts.length > 8) {
            throw new IllegalArgumentException("at most 8 hidden layers are supported by the visualizer");
        }
        var topology = new int[parts.length + 2];
        topology[0] = 2;
        topology[topology.length - 1] = 1;

        var hiddenTotal = 0;
        for (int index = 0; index < parts.length; index++) {
            var token = parts[index].trim();
            if (token.isEmpty()) {
                throw new IllegalArgumentException("hidden layer sizes must be comma-separated positive integers");
            }

            final int size;
            try {
                size = Integer.parseInt(token);
            } catch (NumberFormatException exception) {
                throw new IllegalArgumentException("hidden layer sizes must be integers", exception);
            }

            if (size <= 0) {
                throw new IllegalArgumentException("hidden layer sizes must be > 0");
            }
            if (size > 256) {
                throw new IllegalArgumentException("hidden layer sizes must be <= 256");
            }
            topology[index + 1] = size;
            hiddenTotal += size;
        }
        if (hiddenTotal > 64) {
            throw new IllegalArgumentException("at most 64 hidden neurons are supported by the visualizer");
        }
        return topology;
    }

    static String hiddenLayersText(int[] topology) {
        validateTopology(topology);
        if (topology.length == 2) {
            return "";
        }

        var builder = new StringBuilder();
        for (int index = 1; index < topology.length - 1; index++) {
            if (builder.length() > 0) {
                builder.append(',');
            }
            builder.append(topology[index]);
        }
        return builder.toString();
    }

    static String display(int[] topology) {
        validateTopology(topology);
        var builder = new StringBuilder();
        for (var size : topology) {
            if (builder.length() > 0) {
                builder.append(" → ");
            }
            builder.append(size);
        }
        return builder.toString();
    }

    static int hiddenNeuronCount(int[] topology) {
        validateTopology(topology);
        return Arrays.stream(topology, 1, topology.length - 1).sum();
    }

    static int parameterCount(int[] topology) {
        validateTopology(topology);
        var count = 0;
        for (int layer = 0; layer < topology.length - 1; layer++) {
            count += topology[layer] * topology[layer + 1] + topology[layer + 1];
        }
        return count;
    }

    private static void validateTopology(int[] topology) {
        if (topology == null || topology.length < 2 || topology[0] != 2 || topology[topology.length - 1] != 1) {
            throw new IllegalArgumentException("visualizer topology must start with 2 inputs and end with 1 output");
        }
        for (var size : topology) {
            if (size <= 0) {
                throw new IllegalArgumentException("topology sizes must be > 0");
            }
        }
    }
}
