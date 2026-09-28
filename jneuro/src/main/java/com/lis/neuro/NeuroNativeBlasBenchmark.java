package com.lis.neuro;

public final class NeuroNativeBlasBenchmark {
    private record Scenario(String name, int[] topology, int batchSize, int iterations) {
    }

    private static volatile double blackhole;

    private NeuroNativeBlasBenchmark() {
    }

    public static void main(String[] args) {
        var scenarios = new Scenario[]{
                new Scenario("medium", new int[]{128, 256, 128, 32}, 64, 100),
                new Scenario("large", new int[]{512, 1024, 512, 128}, 128, 30)
        };
        for (var scenario : scenarios) {
            run(scenario);
        }
        System.out.println("blackhole=" + blackhole);
    }

    private static void run(Scenario scenario) {
        var network = new Neuro(
                scenario.topology(),
                Neuro.HyperParameters.defaults()
                        .withSeed(1234)
                        .withKernel(Neuro.Kernel.VECTOR));
        var inputSize = scenario.topology()[0];
        var outputSize = scenario.topology()[scenario.topology().length - 1];
        var inputs = new double[inputSize * scenario.batchSize()];
        var outputs = new double[outputSize * scenario.batchSize()];
        var nativeOutputs = new double[outputs.length];
        for (int sample = 0; sample < scenario.batchSize(); sample++) {
            var input = NeuroBenchmark.input(inputSize);
            System.arraycopy(input, 0, inputs, sample * inputSize, inputSize);
        }

        var nativeSession = NeuroNativeBlas.tryCreate(network);
        if (nativeSession.isEmpty()) {
            System.out.printf("%-8s native BLAS unavailable%n", scenario.name());
            return;
        }

        try (var blas = nativeSession.orElseThrow()) {
            var session = network.newInferenceSession();
            session.predictBatch(inputs, scenario.batchSize(), outputs);
            blas.predictBatch(inputs, scenario.batchSize(), nativeOutputs);
            var maxDifference = 0.0;
            for (int i = 0; i < outputs.length; i++) {
                maxDifference = Math.max(maxDifference, Math.abs(outputs[i] - nativeOutputs[i]));
            }
            if (maxDifference > 1.0e-10) {
                throw new AssertionError("native BLAS output differs by " + maxDifference);
            }

            for (int i = 0; i < 10; i++) {
                session.predictBatch(inputs, scenario.batchSize(), outputs);
                blas.predictBatch(inputs, scenario.batchSize(), nativeOutputs);
            }

            var javaNanos = measure(
                    scenario.iterations(),
                    () -> session.predictBatch(inputs, scenario.batchSize(), outputs),
                    outputs);
            var nativeNanos = measure(
                    scenario.iterations(),
                    () -> blas.predictBatch(inputs, scenario.batchSize(), nativeOutputs),
                    nativeOutputs);

            System.out.printf(
                    "%-8s library=%-18s batch=%3d java=%9.3f ms native=%9.3f ms speedup=%6.2fx diff=%g%n",
                    scenario.name(),
                    blas.libraryName(),
                    scenario.batchSize(),
                    javaNanos / 1_000_000.0 / scenario.iterations(),
                    nativeNanos / 1_000_000.0 / scenario.iterations(),
                    (double) javaNanos / nativeNanos,
                    maxDifference);
        }
    }

    private static long measure(int iterations, Runnable action, double[] output) {
        var start = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            action.run();
            blackhole += output[i % output.length];
        }
        return System.nanoTime() - start;
    }
}
