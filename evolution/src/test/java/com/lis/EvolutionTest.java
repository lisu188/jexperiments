package com.lis;

import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

class EvolutionTest {
    @Test
    void configValidationAndFluentCopiesAreCovered() {
        var defaults = Evolution.Config.defaults();

        assertEquals(256, defaults.populationSize());
        assertEquals(4, defaults.tournamentSize());
        assertEquals(0.05, defaults.mutationProbability());
        assertEquals(0.9, defaults.crossoverProbability());
        assertEquals(0.0, defaults.minGene());
        assertEquals(1.0, defaults.maxGene());

        assertThrows(IllegalArgumentException.class,
                () -> new Evolution.Config(1, 2, 0.1, 0.9, 0.0, 1.0, 1));
        assertThrows(IllegalArgumentException.class,
                () -> new Evolution.Config(10, 1, 0.1, 0.9, 0.0, 1.0, 1));
        assertThrows(IllegalArgumentException.class,
                () -> new Evolution.Config(10, 11, 0.1, 0.9, 0.0, 1.0, 1));
        assertThrows(IllegalArgumentException.class,
                () -> new Evolution.Config(10, 2, -0.1, 0.9, 0.0, 1.0, 1));
        assertThrows(IllegalArgumentException.class,
                () -> new Evolution.Config(10, 2, Double.NaN, 0.9, 0.0, 1.0, 1));
        assertThrows(IllegalArgumentException.class,
                () -> new Evolution.Config(10, 2, 0.1, 1.1, 0.0, 1.0, 1));
        assertThrows(IllegalArgumentException.class,
                () -> new Evolution.Config(10, 2, 0.1, Double.NaN, 0.0, 1.0, 1));
        assertThrows(IllegalArgumentException.class,
                () -> new Evolution.Config(10, 2, 0.1, 0.9, 1.0, 1.0, 1));
        assertThrows(IllegalArgumentException.class,
                () -> new Evolution.Config(10, 2, 0.1, 0.9, Double.NEGATIVE_INFINITY, 1.0, 1));

        var config = defaults
                .withPopulationSize(32)
                .withTournamentSize(3)
                .withMutationProbability(0.2)
                .withCrossoverProbability(0.7)
                .withGeneBounds(-2.0, 3.0)
                .withSeed(1234L);

        assertEquals(32, config.populationSize());
        assertEquals(3, config.tournamentSize());
        assertEquals(0.2, config.mutationProbability());
        assertEquals(0.7, config.crossoverProbability());
        assertEquals(-2.0, config.minGene());
        assertEquals(3.0, config.maxGene());
        assertEquals(1234L, config.seed());
        assertEquals(2, defaults.withPopulationSize(2).tournamentSize());
    }

    @Test
    void constructionAccessorsAndDefensiveCopiesAreCovered() {
        var goal = new double[]{0.25, 0.75};
        var evolution = new Evolution(goal);

        assertSame(Evolution.Config.class, evolution.config().getClass());
        assertEquals(256, evolution.populationSize());
        assertEquals(2, evolution.geneCount());
        assertEquals(0, evolution.generation());
        assertTrue(Double.isFinite(evolution.bestError()));
        assertTrue(Double.isFinite(evolution.averageError()));
        assertEquals(evolution.bestError(), evolution.statistics().bestError());

        goal[0] = 1.0;
        assertArrayEquals(new double[]{0.25, 0.75}, evolution.goal());

        var returnedGoal = evolution.goal();
        returnedGoal[1] = 0.0;
        assertArrayEquals(new double[]{0.25, 0.75}, evolution.goal());

        var best = evolution.bestGenome();
        var copied = new double[2];
        evolution.copyBestInto(copied);
        assertArrayEquals(best, copied);

        best[0] = -100.0;
        assertNotEquals(-100.0, evolution.bestGenome()[0]);

        assertThrows(NullPointerException.class, () -> evolution.copyBestInto(null));
        assertThrows(IllegalArgumentException.class, () -> evolution.copyBestInto(new double[1]));
        assertThrows(NullPointerException.class, () -> new Evolution(null));
        assertThrows(NullPointerException.class,
                () -> new Evolution(new double[]{0.5}, null));
        assertThrows(IllegalArgumentException.class, () -> new Evolution(new double[]{}));
        assertThrows(IllegalArgumentException.class, () -> new Evolution(new double[]{Double.NaN}));
        assertThrows(IllegalArgumentException.class, () -> new Evolution(new double[]{-0.1}));
        assertThrows(IllegalArgumentException.class, () -> new Evolution(new double[]{1.1}));

        var bounded = new Evolution(
                new double[]{-1.5, 1.5},
                Evolution.Config.defaults().withGeneBounds(-2.0, 2.0));
        assertArrayEquals(new double[]{-1.5, 1.5}, bounded.goal());
    }

    @Test
    void deterministicEvolutionCoversNoCrossoverAndNoMutationPaths() {
        var config = Evolution.Config.defaults()
                .withPopulationSize(32)
                .withTournamentSize(3)
                .withMutationProbability(0.0)
                .withCrossoverProbability(0.0)
                .withSeed(7);
        var goal = new double[]{1.0, 0.5, 0.0, 0.25};
        var first = new Evolution(goal, config);
        var second = new Evolution(goal, config);

        var previousBest = first.bestError();
        for (int generation = 0; generation < 20; generation++) {
            var firstStats = first.evolve();
            var secondStats = second.evolve();

            assertEquals(firstStats, secondStats);
            assertArrayEquals(first.bestGenome(), second.bestGenome());
            assertTrue(first.bestError() <= previousBest);
            previousBest = first.bestError();
        }

        assertEquals(20, first.generation());
        assertEquals(32, first.statistics().populationSize());
        assertEquals(4, first.statistics().geneCount());
    }

    @Test
    void crossoverAndMutationPathsStayWithinBounds() {
        var config = Evolution.Config.defaults()
                .withPopulationSize(48)
                .withTournamentSize(4)
                .withMutationProbability(1.0)
                .withCrossoverProbability(1.0)
                .withGeneBounds(-2.0, 2.0)
                .withSeed(99);
        var evolution = new Evolution(new double[]{1.5, -1.0, 0.25}, config);

        var stats = evolution.evolve(25);

        assertEquals(25, stats.generation());
        assertEquals(25, evolution.generation());
        for (var value : evolution.bestGenome()) {
            assertTrue(value >= -2.0 && value <= 2.0);
        }
    }

    @Test
    void evolveCountValidationAndZeroGenerationPathAreCovered() {
        var evolution = new Evolution(new double[]{0.5});

        assertThrows(IllegalArgumentException.class, () -> evolution.evolve(-1));

        var before = evolution.statistics();
        assertEquals(before, evolution.evolve(0));

        var after = evolution.evolve(3);
        assertEquals(3, after.generation());
    }

    @Test
    void evolveUntilValidationImmediateSuccessAndLimitAreCovered() {
        var evolution = new Evolution(
                new double[]{1.0, 1.0, 1.0, 1.0},
                Evolution.Config.defaults()
                        .withPopulationSize(16)
                        .withMutationProbability(0.0)
                        .withCrossoverProbability(0.0)
                        .withSeed(2));

        assertThrows(IllegalArgumentException.class, () -> evolution.evolveUntil(-1.0, 1));
        assertThrows(IllegalArgumentException.class, () -> evolution.evolveUntil(Double.NaN, 1));
        assertThrows(IllegalArgumentException.class, () -> evolution.evolveUntil(0.0, -1));

        var immediate = evolution.evolveUntil(evolution.bestError(), 100);
        assertTrue(immediate.converged());
        assertEquals(0, immediate.generations());

        var limited = evolution.evolveUntil(0.0, 3);
        assertFalse(limited.converged());
        assertEquals(3, limited.generations());
        assertEquals(3, evolution.generation());

        var zeroLimit = evolution.evolveUntil(0.0, 0);
        assertFalse(zeroLimit.converged());
        assertEquals(0, zeroLimit.generations());
    }

    @Test
    void meanSquaredErrorSupportsOffsets() {
        var genome = new double[]{99.0, 99.0, 1.0, 2.0, 3.0};
        var goal = new double[]{1.0, 1.0, 1.0};

        assertEquals(5.0 / 3.0, Evolution.meanSquaredError(genome, 2, goal), 1.0e-15);
    }

    @Test
    void recordsRetainTheirValues() {
        var stats = new Evolution.GenerationStats(4, 8, 2, 0.1, 0.2);
        var result = new Evolution.SearchResult(5, 0.01, true);

        assertEquals(4, stats.generation());
        assertEquals(8, stats.populationSize());
        assertEquals(2, stats.geneCount());
        assertEquals(0.1, stats.bestError());
        assertEquals(0.2, stats.averageError());
        assertEquals(5, result.generations());
        assertEquals(0.01, result.bestError());
        assertTrue(result.converged());
        assertEquals(stats, new Evolution.GenerationStats(4, 8, 2, 0.1, 0.2));
        assertEquals(result, new Evolution.SearchResult(5, 0.01, true));
        assertTrue(Arrays.toString(evolutionArray(stats, result)).contains("GenerationStats"));
    }

    private static Object[] evolutionArray(
            Evolution.GenerationStats stats,
            Evolution.SearchResult result) {
        return new Object[]{stats, result};
    }
}
