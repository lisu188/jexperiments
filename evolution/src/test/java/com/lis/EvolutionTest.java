package com.lis;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class EvolutionTest {
    @Test
    void evolvesPopulationAndExposesStatistics() {
        var config = Evolution.Config.defaults()
                .withPopulationSize(32)
                .withTournamentSize(3)
                .withMutationProbability(1.0)
                .withCrossoverProbability(1.0)
                .withGeneBounds(-1.0, 1.0)
                .withSeed(42L);
        var evolution = new Evolution(new double[]{0.25, -0.5, 0.75}, config);

        assertEquals(config, evolution.config());
        assertArrayEquals(new double[]{0.25, -0.5, 0.75}, evolution.goal());
        assertEquals(32, evolution.populationSize());
        assertEquals(3, evolution.geneCount());
        assertEquals(0, evolution.generation());
        assertTrue(Double.isFinite(evolution.bestError()));
        assertTrue(Double.isFinite(evolution.averageError()));

        var best = evolution.bestGenome();
        assertEquals(3, best.length);
        var destination = new double[3];
        evolution.copyBestInto(destination);
        assertArrayEquals(best, destination);

        var first = evolution.evolve();
        assertEquals(1, first.generation());
        assertEquals(32, first.populationSize());
        assertEquals(3, first.geneCount());
        assertEquals(evolution.bestError(), first.bestError());

        var afterThree = evolution.evolve(2);
        assertEquals(3, afterThree.generation());
        assertEquals(afterThree, evolution.statistics());

        var immediate = evolution.evolveUntil(evolution.bestError(), 10);
        assertEquals(0, immediate.generations());
        assertTrue(immediate.converged());

        var bounded = evolution.evolveUntil(0.0, 1);
        assertTrue(bounded.generations() <= 1);
        assertTrue(Double.isFinite(bounded.bestError()));
    }

    @Test
    void noCrossoverAndNoMutationPathsAreCovered() {
        var config = Evolution.Config.defaults()
                .withPopulationSize(8)
                .withTournamentSize(2)
                .withMutationProbability(0.0)
                .withCrossoverProbability(0.0)
                .withSeed(7L);
        var evolution = new Evolution(new double[]{0.2, 0.8}, config);

        var before = evolution.bestError();
        evolution.evolve();
        assertEquals(1, evolution.generation());
        assertTrue(evolution.bestError() <= before);
    }

    @Test
    void configurationValidationAndWithersAreCovered() {
        var defaults = Evolution.Config.defaults();
        assertEquals(256, defaults.populationSize());
        assertEquals(4, defaults.tournamentSize());

        var resized = defaults.withPopulationSize(2);
        assertEquals(2, resized.populationSize());
        assertEquals(2, resized.tournamentSize());

        assertThrows(IllegalArgumentException.class,
                () -> new Evolution.Config(1, 2, 0.1, 0.5, 0, 1, 1));
        assertThrows(IllegalArgumentException.class,
                () -> new Evolution.Config(4, 1, 0.1, 0.5, 0, 1, 1));
        assertThrows(IllegalArgumentException.class,
                () -> new Evolution.Config(4, 5, 0.1, 0.5, 0, 1, 1));
        assertThrows(IllegalArgumentException.class,
                () -> new Evolution.Config(4, 2, -0.1, 0.5, 0, 1, 1));
        assertThrows(IllegalArgumentException.class,
                () -> new Evolution.Config(4, 2, 1.1, 0.5, 0, 1, 1));
        assertThrows(IllegalArgumentException.class,
                () -> new Evolution.Config(4, 2, Double.NaN, 0.5, 0, 1, 1));
        assertThrows(IllegalArgumentException.class,
                () -> new Evolution.Config(4, 2, 0.1, -0.1, 0, 1, 1));
        assertThrows(IllegalArgumentException.class,
                () -> new Evolution.Config(4, 2, 0.1, 1.1, 0, 1, 1));
        assertThrows(IllegalArgumentException.class,
                () -> new Evolution.Config(4, 2, 0.1, Double.NaN, 0, 1, 1));
        assertThrows(IllegalArgumentException.class,
                () -> new Evolution.Config(4, 2, 0.1, 0.5, 1, 1, 1));
        assertThrows(IllegalArgumentException.class,
                () -> new Evolution.Config(4, 2, 0.1, 0.5, Double.NaN, 1, 1));
        assertThrows(IllegalArgumentException.class,
                () -> new Evolution.Config(4, 2, 0.1, 0.5, 0, Double.POSITIVE_INFINITY, 1));
    }

    @Test
    void rejectsInvalidGoalsAndArguments() {
        assertThrows(NullPointerException.class, () -> new Evolution(null));
        assertThrows(NullPointerException.class, () -> new Evolution(new double[]{0.5}, null));
        assertThrows(IllegalArgumentException.class, () -> new Evolution(new double[0]));
        assertThrows(IllegalArgumentException.class, () -> new Evolution(new double[]{Double.NaN}));
        assertThrows(IllegalArgumentException.class, () -> new Evolution(new double[]{2.0}));

        var evolution = new Evolution(new double[]{0.5, 0.5});
        assertThrows(NullPointerException.class, () -> evolution.copyBestInto(null));
        assertThrows(IllegalArgumentException.class, () -> evolution.copyBestInto(new double[1]));
        assertThrows(IllegalArgumentException.class, () -> evolution.evolve(-1));
        assertThrows(IllegalArgumentException.class, () -> evolution.evolveUntil(-0.1, 1));
        assertThrows(IllegalArgumentException.class, () -> evolution.evolveUntil(Double.NaN, 1));
        assertThrows(IllegalArgumentException.class, () -> evolution.evolveUntil(0.0, -1));
        assertEquals(evolution.statistics(), evolution.evolve(0));
    }

    @Test
    void meanSquaredErrorUsesRequestedGenomeOffset() {
        assertEquals(0.0, Evolution.meanSquaredError(
                new double[]{9, 9, 1, 2, 3}, 2, new double[]{1, 2, 3}), 1.0e-12);
        assertEquals(1.0, Evolution.meanSquaredError(
                new double[]{0, 1, 2}, 0, new double[]{1, 2, 3}), 1.0e-12);
    }
}
