package com.lis.neuro;

public final class NeuroPerformanceMatrix {
    private record Scenario(String name, int[] topology) {
    }

    public static void main(String[] args) {
        var inferenceIterations = args.length > 0 ? Integer.parseInt(args[0]) : 20_000;
        var trainingEpochs = args.length > 1 ? Integer.parseInt(args[1]) : 20;

        var scenarios = new Scenario[]{
                new Scenario("tiny", new int[]{2, 6, 1}),
                new Scenario("small", new int[]{32, 64, 32, 8}),
                new Scenario("medium", new int[]{128, 256, 128, 32}),
                new Scenario("deep", new int[]{64, 128, 128, 64, 32, 8})
        };

        System.out.printf(
                "JNeuro performance matrix: %,d inference iterations, %d training epochs%n",
                inferenceIterations,
                trainingEpochs);

        for (var scenario : scenarios) {
            run(scenario, inferenceIterations, trainingEpochs);
        }
    }

    private static void run(Scenario scenario, int iterations, int trainingEpochs) {
        var network = NeuroBenchmark.preparedNetwork(scenario.topology());
        var input = NeuroBenchmark.input(scenario.topology()[0]);
        var output = new double[scenario.topology()[scenario.topology().length - 1]];

        for (int i = 0; i < 2_000; i++) {
            network.predictInto(input, output);
        }

        var inferenceStart = System.nanoTime();
        double sum = 0.0;
        for (int i = 0; i < iterations; i++) {
            network.predictInto(input, output);
            sum += output[i % output.length];
        }
        var inferenceNanos = System.nanoTime() - inferenceStart;

        var trainingStart = System.nanoTime();
        network.train(trainingEpochs);
        var trainingNanos = System.nanoTime() - trainingStart;

        System.out.printf(
                "%-8s params=%8d predict=%9.1f ns/op train=%9.3f ms/epoch error=%8.5f checksum=%f%n",
                scenario.name(),
                network.parameterCount(),
                inferenceNanos / (double) iterations,
                trainingNanos / 1_000_000.0 / trainingEpochs,
                network.trainingError(),
                sum);
    }
}
