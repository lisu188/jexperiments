package com.lis;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.concurrent.Executors;

import static org.junit.jupiter.api.Assertions.*;

class EvolutionTest {
    @Test
    void verificationHarnessPasses() {
        EvolutionVerification.main(new String[0]);
    }

    @Test
    void configurationBuildersPreserveAndReplaceExpectedValues() {
        var selection = Evolution.SelectionPolicy.defaults()
                .withType(Evolution.SelectionType.RANK)
                .withTournamentSize(3)
                .withTruncationFraction(0.25);
        var crossover = Evolution.CrossoverPolicy.defaults()
                .withType(Evolution.CrossoverType.ARITHMETIC)
                .withProbability(0.75)
                .withBlxAlpha(0.4);
        var mutation = Evolution.MutationPolicy.defaults()
                .withType(Evolution.MutationType.RANDOM_RESET)
                .withProbability(0.2)
                .withSigma(0.2)
                .withSigmaBounds(0.01, 0.5)
                .withGeometricSkipping(false);

        var config = Evolution.Config.defaults()
                .withPopulationSize(32)
                .withEliteCount(3)
                .withGeneBounds(-1, 2)
                .withSeed(123)
                .withSelection(selection)
                .withCrossover(crossover)
                .withMutation(mutation)
                .withFitnessParallelism(2)
                .withVectorizedGoalFitness(false)
                .withTrackDiversity(false);

        assertEquals(32, config.populationSize());
        assertEquals(3, config.eliteCount());
        assertEquals(-1, config.minGene());
        assertEquals(2, config.maxGene());
        assertEquals(123, config.seed());
        assertEquals(Evolution.SelectionType.RANK, config.selection().type());
        assertEquals(Evolution.CrossoverType.ARITHMETIC, config.crossover().type());
        assertEquals(Evolution.MutationType.RANDOM_RESET, config.mutation().type());
        assertEquals(2, config.fitnessParallelism());
        assertFalse(config.vectorizedGoalFitness());
        assertFalse(config.trackDiversity());
    }

    @Test
    void callerOwnedFitnessExecutorIsNotClosedByEvolution() {
        var executor = Executors.newFixedThreadPool(2);
        try {
            var config = Evolution.Config.defaults()
                    .withPopulationSize(32)
                    .withFitnessParallelism(2)
                    .withTrackDiversity(false);
            try (var evolution = new Evolution(
                    8,
                    (genome, offset, length) -> {
                        var sum = 0.0;
                        for (int i = 0; i < length; i++) {
                            sum += genome[offset + i];
                        }
                        return sum;
                    },
                    config,
                    executor)) {
                evolution.evolve(2);
            }
            assertFalse(executor.isShutdown());
        } finally {
            executor.close();
        }
    }

    @Test
    void genericIslandEvolutionMigratesCustomFitness() {
        EvolutionFitness fitness = (genome, offset, length) -> {
            var sum = 0.0;
            for (int i = 0; i < length; i++) {
                var difference = genome[offset + i] - 0.25;
                sum = Math.fma(difference, difference, sum);
            }
            return sum;
        };

        try (var islands = new IslandEvolution(
                4,
                fitness,
                Evolution.Config.defaults().withPopulationSize(32).withSeed(44),
                new IslandEvolution.Config(3, 1, 1, false))) {
            var before = islands.bestError();
            islands.evolve(5);
            assertTrue(islands.bestError() <= before);
            assertEquals(5, islands.statistics().migrations());
            assertEquals(4, islands.bestGenome().length);
        }
    }

    @Test
    void paretoFrontReturnsDefensiveGenomesAndRejectsNonFiniteObjectives() {
        MultiObjectiveFitness objectives = (genome, offset, length, output, objectiveOffset) -> {
            var x = genome[offset];
            output[objectiveOffset] = x * x;
            var inverse = 1.0 - x;
            output[objectiveOffset + 1] = inverse * inverse;
        };

        var pareto = new ParetoEvolution(
                1,
                2,
                objectives,
                Evolution.Config.defaults().withPopulationSize(32).withSeed(55));
        pareto.evolve(10);

        var front = pareto.paretoFront();
        assertFalse(front.isEmpty());
        var original = front.get(0).clone();
        front.get(0)[0] = -100;
        assertFalse(Arrays.equals(front.get(0), original));
        assertTrue(pareto.paretoFront().stream().allMatch(genome -> genome[0] >= 0));

        MultiObjectiveFitness invalid = (genome, offset, length, output, objectiveOffset) -> {
            output[objectiveOffset] = Double.NaN;
            output[objectiveOffset + 1] = 0;
        };
        assertThrows(
                IllegalStateException.class,
                () -> new ParetoEvolution(
                        1,
                        2,
                        invalid,
                        Evolution.Config.defaults().withPopulationSize(8)));
    }

    @Test
    void vectorAndScalarGoalSearchesProduceFiniteResults() {
        var goal = new double[129];
        Arrays.fill(goal, 1.0);

        for (var vectorized : new boolean[]{false, true}) {
            var config = Evolution.Config.defaults()
                    .withPopulationSize(32)
                    .withVectorizedGoalFitness(vectorized)
                    .withSeed(vectorized ? 2 : 1);
            try (var evolution = new Evolution(goal, config)) {
                evolution.evolve(3);
                assertTrue(Double.isFinite(evolution.bestError()));
            }
        }
    }
}
