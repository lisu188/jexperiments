package com.lis;

import java.util.Arrays;
import java.util.Objects;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class IslandEvolution implements AutoCloseable {
    public record Config(
            int islandCount,
            int migrationInterval,
            int migrants,
            boolean parallelIslands) {
        public Config {
            if (islandCount < 2) {
                throw new IllegalArgumentException("islandCount must be >= 2");
            }
            if (migrationInterval < 1) {
                throw new IllegalArgumentException("migrationInterval must be >= 1");
            }
            if (migrants < 1) {
                throw new IllegalArgumentException("migrants must be >= 1");
            }
        }

        public static Config defaults() {
            return new Config(4, 25, 2, true);
        }
    }

    public record Statistics(
            long generation,
            int islandCount,
            double bestError,
            double averageIslandBest,
            long migrations) {
    }

    private final Evolution[] islands;
    private final Config config;
    private final ExecutorService executor;
    private final int geneCount;
    private long generation;
    private long migrations;

    public IslandEvolution(double[] goal, Evolution.Config evolutionConfig, Config config) {
        Objects.requireNonNull(goal, "goal");
        this.config = Objects.requireNonNull(config, "config");
        geneCount = goal.length;
        islands = new Evolution[config.islandCount()];
        for (int island = 0; island < islands.length; island++) {
            islands[island] = new Evolution(
                    goal,
                    evolutionConfig.withSeed(mixSeed(evolutionConfig.seed(), island)));
        }
        executor = config.parallelIslands()
                ? Executors.newFixedThreadPool(
                        config.islandCount(),
                        Thread.ofPlatform().daemon(true).name("evolution-island-", 0).factory())
                : null;
    }

    public IslandEvolution(
            int geneCount,
            EvolutionFitness fitness,
            Evolution.Config evolutionConfig,
            Config config) {
        if (geneCount <= 0) {
            throw new IllegalArgumentException("geneCount must be positive");
        }
        this.config = Objects.requireNonNull(config, "config");
        this.geneCount = geneCount;
        islands = new Evolution[config.islandCount()];
        for (int island = 0; island < islands.length; island++) {
            islands[island] = new Evolution(
                    geneCount,
                    fitness,
                    evolutionConfig.withSeed(mixSeed(evolutionConfig.seed(), island)));
        }
        executor = config.parallelIslands()
                ? Executors.newFixedThreadPool(
                        config.islandCount(),
                        Thread.ofPlatform().daemon(true).name("evolution-island-", 0).factory())
                : null;
    }

    public Statistics evolve() {
        if (executor == null) {
            for (var island : islands) {
                island.evolve();
            }
        } else {
            var futures = Arrays.stream(islands)
                    .map(island -> java.util.concurrent.CompletableFuture.runAsync(island::evolve, executor))
                    .toArray(java.util.concurrent.CompletableFuture[]::new);
            java.util.concurrent.CompletableFuture.allOf(futures).join();
        }

        generation++;
        if (generation % config.migrationInterval() == 0) {
            migrate();
        }
        return statistics();
    }

    public Statistics evolve(int generations) {
        if (generations < 0) {
            throw new IllegalArgumentException("generations must be >= 0");
        }
        var result = statistics();
        for (int index = 0; index < generations; index++) {
            result = evolve();
        }
        return result;
    }

    public Evolution.SearchResult evolveUntil(double targetError, long maxGenerations) {
        if (targetError < 0.0 || !Double.isFinite(targetError)) {
            throw new IllegalArgumentException("targetError must be finite and >= 0");
        }
        if (maxGenerations < 0) {
            throw new IllegalArgumentException("maxGenerations must be >= 0");
        }

        var startEvaluations = totalFitnessEvaluations();
        var startRestarts = totalRestarts();
        var evolved = 0L;
        while (evolved < maxGenerations && bestError() > targetError) {
            evolve();
            evolved++;
        }
        return new Evolution.SearchResult(
                evolved,
                bestError(),
                bestError() <= targetError,
                totalFitnessEvaluations() - startEvaluations,
                totalRestarts() - startRestarts);
    }

    public Statistics statistics() {
        var best = Double.POSITIVE_INFINITY;
        var sum = 0.0;
        for (var island : islands) {
            var value = island.bestError();
            best = Math.min(best, value);
            sum += value;
        }
        return new Statistics(generation, islands.length, best, sum / islands.length, migrations);
    }

    public double bestError() {
        return statistics().bestError();
    }

    public double[] bestGenome() {
        var best = islands[0];
        for (int island = 1; island < islands.length; island++) {
            if (islands[island].bestError() < best.bestError()) {
                best = islands[island];
            }
        }
        return best.bestGenome();
    }

    private void migrate() {
        var migrantCount = Math.min(config.migrants(), islands[0].populationSize());
        var migrants = new double[islands.length][migrantCount][geneCount];

        for (int island = 0; island < islands.length; island++) {
            for (int migrant = 0; migrant < migrantCount; migrant++) {
                islands[island].copyRankedGenomeInto(migrant, migrants[island][migrant]);
            }
        }

        for (int island = 0; island < islands.length; island++) {
            var destination = islands[(island + 1) % islands.length];
            for (int migrant = 0; migrant < migrantCount; migrant++) {
                destination.injectGenome(migrants[island][migrant]);
            }
        }
        migrations++;
    }

    private long totalFitnessEvaluations() {
        var total = 0L;
        for (var island : islands) {
            total += island.statistics().fitnessEvaluations();
        }
        return total;
    }

    private long totalRestarts() {
        var total = 0L;
        for (var island : islands) {
            total += island.statistics().restarts();
        }
        return total;
    }

    private static long mixSeed(long seed, int island) {
        var value = seed + 0x9E3779B97F4A7C15L * (island + 1L);
        value ^= value >>> 30;
        value *= 0xBF58476D1CE4E5B9L;
        value ^= value >>> 27;
        value *= 0x94D049BB133111EBL;
        return value ^ (value >>> 31);
    }

    @Override
    public void close() {
        for (var island : islands) {
            island.close();
        }
        if (executor != null) {
            executor.close();
        }
    }
}
