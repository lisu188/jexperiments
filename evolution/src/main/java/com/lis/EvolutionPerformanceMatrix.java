package com.lis;

public final class EvolutionPerformanceMatrix {
    public static void main(String[] args) {
        var generations = args.length > 0 ? Integer.parseInt(args[0]) : 500;
        var populations = new int[]{64, 256, 1_024, 4_096};
        var genes = new int[]{8, 32, 128, 512};

        System.out.printf("Evolution performance matrix: %,d generations%n", generations);
        for (var population : populations) {
            for (var geneCount : genes) {
                run(population, geneCount, generations);
            }
        }
    }

    private static void run(int populationSize, int geneCount, int generations) {
        var evolution = EvolutionBenchmark.create(populationSize, geneCount, populationSize * 31L + geneCount);

        for (int warmup = 0; warmup < Math.min(20, generations); warmup++) {
            evolution.evolve();
        }

        var startError = evolution.bestError();
        var start = System.nanoTime();
        evolution.evolve(generations);
        var elapsed = System.nanoTime() - start;

        System.out.printf(
                "population=%5d genes=%4d ns/generation=%12.1f ns/candidate-gene=%8.3f best=%10.7f improvement=%10.7f%n",
                populationSize,
                geneCount,
                elapsed / (double) generations,
                elapsed / ((double) generations * populationSize * geneCount),
                evolution.bestError(),
                startError - evolution.bestError());
    }
}
