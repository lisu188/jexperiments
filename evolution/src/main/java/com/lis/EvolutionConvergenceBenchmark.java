package com.lis;

import java.util.Arrays;
import java.util.function.LongFunction;

public final class EvolutionConvergenceBenchmark {
    private record Sample(long generations, long evaluations, long nanos, boolean success) {
    }

    public static void main(String[] args) {
        var seedCount = args.length > 0 ? Integer.parseInt(args[0]) : 100;
        var maxGenerations = args.length > 1 ? Integer.parseInt(args[1]) : 5_000;
        var geneCount = 16;
        var target = 1.0e-4;

        System.out.printf(
                "Evolution convergence benchmark: seeds=%d genes=%d target=%g maxGenerations=%d%n",
                seedCount, geneCount, target, maxGenerations);

        run("reset+single", seedCount, seed -> createBaseline(geneCount, seed), target, maxGenerations);
        run("gaussian+arithmetic", seedCount, seed -> createGaussianArithmetic(geneCount, seed), target, maxGenerations);
        run("adaptive+blx", seedCount, seed -> createAdaptiveBlx(geneCount, seed), target, maxGenerations);
        runIslands("islands", seedCount, geneCount, target, maxGenerations);
    }

    private static void run(
            String name,
            int seedCount,
            LongFunction<Evolution> factory,
            double target,
            int maxGenerations) {
        var samples = new Sample[seedCount];
        for (int seed = 0; seed < seedCount; seed++) {
            try (var evolution = factory.apply(seed)) {
                var before = evolution.statistics().fitnessEvaluations();
                var start = System.nanoTime();
                var result = evolution.evolveUntil(target, maxGenerations);
                var nanos = System.nanoTime() - start;
                samples[seed] = new Sample(
                        result.generations(),
                        evolution.statistics().fitnessEvaluations() - before,
                        nanos,
                        result.converged());
            }
        }
        print(name, samples);
    }

    private static void runIslands(
            String name,
            int seedCount,
            int geneCount,
            double target,
            int maxGenerations) {
        var samples = new Sample[seedCount];
        var goal = new double[geneCount];
        Arrays.fill(goal, 1.0);

        for (int seed = 0; seed < seedCount; seed++) {
            var base = adaptiveConfig(seed).withPopulationSize(64);
            try (var islands = new IslandEvolution(
                    goal,
                    base,
                    new IslandEvolution.Config(4, 25, 2, true))) {
                var start = System.nanoTime();
                var result = islands.evolveUntil(target, maxGenerations);
                samples[seed] = new Sample(
                        result.generations(),
                        result.fitnessEvaluations(),
                        System.nanoTime() - start,
                        result.converged());
            }
        }
        print(name, samples);
    }

    private static Evolution createBaseline(int geneCount, long seed) {
        var goal = goal(geneCount);
        var mutation = Evolution.MutationPolicy.defaults()
                .withType(Evolution.MutationType.RANDOM_RESET)
                .withProbability(0.05)
                .withAdaptation(false, 1.0, 1.0, 100, 0.0);
        var crossover = Evolution.CrossoverPolicy.defaults()
                .withType(Evolution.CrossoverType.SINGLE_POINT);
        return new Evolution(goal, Evolution.Config.defaults()
                .withPopulationSize(256)
                .withEliteCount(1)
                .withSeed(seed)
                .withMutation(mutation)
                .withCrossover(crossover));
    }

    private static Evolution createGaussianArithmetic(int geneCount, long seed) {
        var mutation = Evolution.MutationPolicy.defaults()
                .withProbability(0.05)
                .withAdaptation(false, 1.0, 1.0, 100, 0.0);
        var crossover = Evolution.CrossoverPolicy.defaults()
                .withType(Evolution.CrossoverType.ARITHMETIC);
        return new Evolution(goal(geneCount), Evolution.Config.defaults()
                .withPopulationSize(256)
                .withEliteCount(2)
                .withSeed(seed)
                .withMutation(mutation)
                .withCrossover(crossover));
    }

    private static Evolution createAdaptiveBlx(int geneCount, long seed) {
        return new Evolution(goal(geneCount), adaptiveConfig(seed));
    }

    private static Evolution.Config adaptiveConfig(long seed) {
        return Evolution.Config.defaults()
                .withPopulationSize(256)
                .withEliteCount(4)
                .withSeed(seed)
                .withSelection(Evolution.SelectionPolicy.defaults().withTournamentSize(6))
                .withCrossover(Evolution.CrossoverPolicy.defaults()
                        .withType(Evolution.CrossoverType.BLX_ALPHA)
                        .withBlxAlpha(0.25))
                .withMutation(Evolution.MutationPolicy.defaults()
                        .withProbability(0.05)
                        .withSigma(0.08)
                        .withAdaptation(true, 0.995, 1.75, 100, 0.35));
    }

    private static double[] goal(int geneCount) {
        var goal = new double[geneCount];
        Arrays.fill(goal, 1.0);
        return goal;
    }

    private static void print(String name, Sample[] samples) {
        var successful = Arrays.stream(samples).filter(Sample::success).toArray(Sample[]::new);
        var successRate = successful.length * 100.0 / samples.length;
        if (successful.length == 0) {
            System.out.printf("%-22s success=%6.2f%%%n", name, successRate);
            return;
        }

        var generations = Arrays.stream(successful).mapToLong(Sample::generations).sorted().toArray();
        var evaluations = Arrays.stream(successful).mapToLong(Sample::evaluations).sorted().toArray();
        var nanos = Arrays.stream(successful).mapToLong(Sample::nanos).sorted().toArray();

        System.out.printf(
                "%-22s success=%6.2f%% gen[p50=%d p90=%d p99=%d] eval[p50=%d] time[p50=%.3fms p90=%.3fms]%n",
                name,
                successRate,
                percentile(generations, 0.50),
                percentile(generations, 0.90),
                percentile(generations, 0.99),
                percentile(evaluations, 0.50),
                percentile(nanos, 0.50) / 1_000_000.0,
                percentile(nanos, 0.90) / 1_000_000.0);
    }

    private static long percentile(long[] sorted, double percentile) {
        var index = Math.min(sorted.length - 1, Math.max(0, (int) Math.ceil(sorted.length * percentile) - 1));
        return sorted[index];
    }
}
