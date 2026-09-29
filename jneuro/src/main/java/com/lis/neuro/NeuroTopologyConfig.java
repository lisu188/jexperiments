package com.lis.neuro;

import java.util.Arrays;

final class NeuroTopologyConfig {
    private static final int MAX_HIDDEN_LAYERS = 8;
    private static final int MAX_NEURONS_PER_LAYER = 128;

    private NeuroTopologyConfig() {
    }

    static int[] parseHidden(String text) {
        if (text == null) {
            throw new NullPointerException("text");
        }
        var trimmed = text.trim();
        if (trimmed.isEmpty()) {
            throw new IllegalArgumentException("at least one hidden layer is required");
        }

        var parts = trimmed.split(",");
        if (parts.length > MAX_HIDDEN_LAYERS) {
            throw new IllegalArgumentException("at most " + MAX_HIDDEN_LAYERS + " hidden layers are supported");
        }

        var hidden = new int[parts.length];
        for (int i = 0; i < parts.length; i++) {
            var token = parts[i].trim();
            if (token.isEmpty()) {
                throw new IllegalArgumentException("hidden layer size is missing");
            }

            final int size;
            try {
                size = Integer.parseInt(token);
            } catch (NumberFormatException exception) {
                throw new IllegalArgumentException("hidden layer sizes must be integers", exception);
            }

            if (size < 1 || size > MAX_NEURONS_PER_LAYER) {
                throw new IllegalArgumentException(
                        "hidden layer size must be in [1, " + MAX_NEURONS_PER_LAYER + "]");
            }
            hidden[i] = size;
        }
        return hidden;
    }

    static int[] topology(int[] hidden) {
        if (hidden == null) {
            throw new NullPointerException("hidden");
        }
        if (hidden.length == 0 || hidden.length > MAX_HIDDEN_LAYERS) {
            throw new IllegalArgumentException("hidden layer count must be in [1, " + MAX_HIDDEN_LAYERS + "]");
        }

        var topology = new int[hidden.length + 2];
        topology[0] = 2;
        topology[topology.length - 1] = 1;
        for (int i = 0; i < hidden.length; i++) {
            if (hidden[i] < 1 || hidden[i] > MAX_NEURONS_PER_LAYER) {
                throw new IllegalArgumentException(
                        "hidden layer size must be in [1, " + MAX_NEURONS_PER_LAYER + "]");
            }
            topology[i + 1] = hidden[i];
        }
        return topology;
    }

    static String format(int[] hidden) {
        if (hidden == null) {
            throw new NullPointerException("hidden");
        }
        if (hidden.length == 0) {
            return "";
        }
        var builder = new StringBuilder();
        for (int i = 0; i < hidden.length; i++) {
            if (i > 0) {
                builder.append(',');
            }
            builder.append(hidden[i]);
        }
        return builder.toString();
    }

    static int[] addLayer(int[] hidden, int defaultSize) {
        if (hidden == null) {
            throw new NullPointerException("hidden");
        }
        if (hidden.length >= MAX_HIDDEN_LAYERS) {
            throw new IllegalArgumentException("maximum hidden layer count reached");
        }
        var result = Arrays.copyOf(hidden, hidden.length + 1);
        result[result.length - 1] = defaultSize;
        topology(result);
        return result;
    }

    static int[] removeLayer(int[] hidden) {
        if (hidden == null) {
            throw new NullPointerException("hidden");
        }
        if (hidden.length <= 1) {
            throw new IllegalArgumentException("at least one hidden layer is required");
        }
        return Arrays.copyOf(hidden, hidden.length - 1);
    }
}
