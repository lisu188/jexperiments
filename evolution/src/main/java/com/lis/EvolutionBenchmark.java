package com.lis;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;

public final class EvolutionBenchmark {
    private static volatile double blackhole;

    public static void main(String[] args) {
        var generations = args.length > 0 ? Integer.parseInt(args[0]) : 2_000;
        var repetitions = args.length > 1 ? Integer.parseInt(args[1]) : 5;

        for (int i = 0; i < 2; i++) {
            run(generations, false);
        }

        var samples = new LinkedHashMap<String, double[]>();
        for (int repetition = 0; repetition < repetitions; repetition++) {
            var result = run(generations, true);
            var index = repetition;
            result.forEach((name, value) ->
                    samples.computeIfAbsent(name, ignored -> new double[repetitions])[index] = value);
        }

        System.out.printf("Evolution benchmark: %,d generations%n", generations);
        samples.forEach((name, values) -> {
            var sorted = values.clone();
            Arrays.sort(sorted);
            System.out.printf("%-26s %10.3f ms median%n", name, sorted[sorted.length / 2]);
        });
        System.out.println("blackhole=" + blackhole);
    }

    private static Map<String, Double> run(int generations, boolean measured) {
        var result = new LinkedHashMap<String, Double>();

        result.put("p64-g16", millis(() -> {
            var evolution = create(64, 16, 1);
            evolution.evolve(generations);
            blackhole += evolution.bestError();
        }));

        result.put("p512-g16", millis(() -> {
            var evolution = create(512, 16, 2);
            evolution.evolve(generations);
            blackhole += evolution.bestError();
        }));

        result.put("p512-g128", millis(() -> {
            var evolution = create(512, 128, 3);
            evolution.evolve(generations);
            blackhole += evolution.bestError();
        }));

        if (!measured) {
            result.clear();
        }
        return result;
    }

    static Evolution create(int populationSize, int geneCount, long seed) {
        var goal = new double[geneCount];
        Arrays.fill(goal, 1.0);
        var config = Evolution.Config.defaults()
                .withPopulationSize(populationSize)
                .withMutation(Evolution.MutationPolicy.defaults()
                        .withProbability(0.05)
                        .withAdaptation(false, 1.0, 1.0, 100, 0.0))
                .withSeed(seed);
        return new Evolution(goal, config);
    }

    private static double millis(Runnable action) {
        var start = System.nanoTime();
        action.run();
        return (System.nanoTime() - start) / 1_000_000.0;
    }
}
