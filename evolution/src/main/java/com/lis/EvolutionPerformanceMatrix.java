package com.lis;

import java.util.Arrays;

public final class EvolutionPerformanceMatrix {
    public static void main(String[] args) {
        var generations = args.length > 0 ? Integer.parseInt(args[0]) : 500;
        sizeMatrix(generations);
        operatorMatrix(Math.max(10, generations / 5));
        parallelFitnessMatrix(Math.max(10, generations / 10));
        islandMatrix(Math.max(10, generations / 10));
    }

    private static void sizeMatrix(int generations) {
        var populations = new int[]{64, 256, 1_024, 4_096};
        var genes = new int[]{8, 32, 128, 512};

        System.out.printf("Evolution size matrix: %,d generations%n", generations);
        for (var population : populations) {
            for (var geneCount : genes) {
                try (var evolution = EvolutionBenchmark.create(population, geneCount, population * 31L + geneCount)) {
                    for (int warmup = 0; warmup < Math.min(20, generations); warmup++) {
                        evolution.evolve();
                    }
                    var start = System.nanoTime();
                    evolution.evolve(generations);
                    var elapsed = System.nanoTime() - start;
                    System.out.printf(
                            "population=%5d genes=%4d ns/gen=%12.1f ns/candidate-gene=%8.3f best=%10.7f diversity=%9.6f%n",
                            population,
                            geneCount,
                            elapsed / (double) generations,
                            elapsed / ((double) generations * population * geneCount),
                            evolution.bestError(),
                            evolution.statistics().geneDiversity());
                }
            }
        }
    }

    private static void operatorMatrix(int generations) {
        System.out.printf("Evolution operator matrix: %,d generations%n", generations);
        for (var selection : Evolution.SelectionType.values()) {
            for (var crossover : Evolution.CrossoverType.values()) {
                var config = Evolution.Config.defaults()
                        .withPopulationSize(256)
                        .withSeed(selection.ordinal() * 100L + crossover.ordinal())
                        .withSelection(Evolution.SelectionPolicy.defaults().withType(selection))
                        .withCrossover(Evolution.CrossoverPolicy.defaults().withType(crossover))
                        .withTrackDiversity(false);
                var goal = new double[32];
                Arrays.fill(goal, 1.0);
                try (var evolution = new Evolution(goal, config)) {
                    var start = System.nanoTime();
                    evolution.evolve(generations);
                    var elapsed = System.nanoTime() - start;
                    System.out.printf(
                            "%-10s %-12s ns/gen=%10.1f best=%10.7f%n",
                            selection,
                            crossover,
                            elapsed / (double) generations,
                            evolution.bestError());
                }
            }
        }
    }

    private static void parallelFitnessMatrix(int generations) {
        System.out.printf("Evolution parallel expensive-fitness matrix: %,d generations%n", generations);
        EvolutionFitness expensive = (genome, offset, length) -> {
            var sum = 0.0;
            for (int repeat = 0; repeat < 64; repeat++) {
                for (int gene = 0; gene < length; gene++) {
                    var value = genome[offset + gene] - 0.5;
                    sum = Math.fma(value, value, sum);
                }
            }
            return sum;
        };

        for (var parallelism : new int[]{1, 2, 4, 8}) {
            var config = Evolution.Config.defaults()
                    .withPopulationSize(512)
                    .withFitnessParallelism(parallelism)
                    .withTrackDiversity(false)
                    .withSeed(parallelism);
            try (var evolution = new Evolution(64, expensive, config)) {
                var start = System.nanoTime();
                evolution.evolve(generations);
                var elapsed = System.nanoTime() - start;
                System.out.printf(
                        "parallelism=%d ns/gen=%12.1f eval=%d%n",
                        parallelism,
                        elapsed / (double) generations,
                        evolution.statistics().fitnessEvaluations());
            }
        }
    }

    private static void islandMatrix(int generations) {
        System.out.printf("Evolution island matrix: %,d generations%n", generations);
        var goal = new double[32];
        Arrays.fill(goal, 1.0);
        var base = Evolution.Config.defaults().withPopulationSize(128).withTrackDiversity(false);
        for (var parallel : new boolean[]{false, true}) {
            try (var islands = new IslandEvolution(
                    goal,
                    base,
                    new IslandEvolution.Config(4, 20, 2, parallel))) {
                var start = System.nanoTime();
                islands.evolve(generations);
                var elapsed = System.nanoTime() - start;
                System.out.printf(
                        "parallelIslands=%-5s ns/gen=%12.1f best=%10.7f migrations=%d%n",
                        parallel,
                        elapsed / (double) generations,
                        islands.bestError(),
                        islands.statistics().migrations());
            }
        }
    }
}
