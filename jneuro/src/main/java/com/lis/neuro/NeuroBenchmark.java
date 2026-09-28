package com.lis.neuro;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;

public final class NeuroBenchmark {
    private static volatile double blackhole;

    public static void main(String[] args) {
        var iterations = args.length > 0 ? Integer.parseInt(args[0]) : 100_000;
        var repetitions = args.length > 1 ? Integer.parseInt(args[1]) : 5;

        for (int i = 0; i < 2; i++) {
            run(iterations, false);
        }

        var samples = new LinkedHashMap<String, double[]>();
        for (int repetition = 0; repetition < repetitions; repetition++) {
            var result = run(iterations, true);
            var index = repetition;
            result.forEach((name, value) ->
                    samples.computeIfAbsent(name, ignored -> new double[repetitions])[index] = value);
        }

        System.out.printf("JNeuro benchmark: %,d predictions%n", iterations);
        samples.forEach((name, values) -> {
            var sorted = values.clone();
            Arrays.sort(sorted);
            System.out.printf("%-24s %10.3f ms median%n", name, sorted[sorted.length / 2]);
        });
        System.out.println("blackhole=" + blackhole);
    }

    private static Map<String, Double> run(int iterations, boolean measured) {
        var result = new LinkedHashMap<String, Double>();
        var network = preparedNetwork(new int[]{32, 64, 32, 8});
        var input = input(32);
        var output = new double[8];

        result.put("predict-allocating", millis(() -> {
            double sum = 0.0;
            for (int i = 0; i < iterations; i++) {
                sum += network.predict(input)[i & 7];
            }
            blackhole += sum;
        }));

        result.put("predict-into", millis(() -> {
            double sum = 0.0;
            for (int i = 0; i < iterations; i++) {
                network.predictInto(input, output);
                sum += output[i & 7];
            }
            blackhole += sum;
        }));

        result.put("train-100-epochs", millis(() -> {
            var training = preparedNetwork(new int[]{32, 64, 32, 8});
            training.train(100);
            blackhole += training.trainingError();
        }));

        if (!measured) {
            result.clear();
        }
        return result;
    }

    static Neuro preparedNetwork(int[] topology) {
        var network = new Neuro(
                topology,
                Neuro.HyperParameters.defaults()
                        .withLearningRate(0.1)
                        .withMomentum(0.1)
                        .withSeed(1234));
        var outputSize = topology[topology.length - 1];
        for (int sample = 0; sample < 32; sample++) {
            var input = new double[topology[0]];
            var target = new double[outputSize];
            for (int i = 0; i < input.length; i++) {
                input[i] = ((sample + i) & 7) / 7.0;
            }
            for (int i = 0; i < target.length; i++) {
                target[i] = ((sample + i) & 1);
            }
            network.addTrainingSample(input, target);
        }
        return network;
    }

    static double[] input(int size) {
        var input = new double[size];
        for (int i = 0; i < size; i++) {
            input[i] = (i & 7) / 7.0;
        }
        return input;
    }

    private static double millis(Runnable action) {
        var start = System.nanoTime();
        action.run();
        return (System.nanoTime() - start) / 1_000_000.0;
    }
}
