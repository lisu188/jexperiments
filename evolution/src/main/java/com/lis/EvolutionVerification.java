package com.lis;

import java.util.Arrays;

public final class EvolutionVerification {
    public static void main(String[] args) {
        verifyConfigValidation();
        verifyGoalValidation();
        verifyDefensiveCopies();
        verifyDeterministicSeed();
        verifyFixedPopulation();
        verifyElitism();
        verifyGeneBounds();
        verifyConvergence();
        verifySingleGeneSearch();
        verifyGenerationLimit();
        verifyDistanceKernel();
        System.out.println("Evolution verification passed");
    }

    private static void verifyConfigValidation() {
        expectIllegalArgument(() -> new Evolution.Config(1, 2, 0.1, 0.9, 0, 1, 1));
        expectIllegalArgument(() -> new Evolution.Config(10, 1, 0.1, 0.9, 0, 1, 1));
        expectIllegalArgument(() -> new Evolution.Config(10, 11, 0.1, 0.9, 0, 1, 1));
        expectIllegalArgument(() -> new Evolution.Config(10, 2, -0.1, 0.9, 0, 1, 1));
        expectIllegalArgument(() -> new Evolution.Config(10, 2, 1.1, 0.9, 0, 1, 1));
        expectIllegalArgument(() -> new Evolution.Config(10, 2, 0.1, 1.1, 0, 1, 1));
        expectIllegalArgument(() -> new Evolution.Config(10, 2, 0.1, 0.9, 1, 1, 1));
    }

    private static void verifyGoalValidation() {
        expectIllegalArgument(() -> new Evolution(new double[]{}));
        expectIllegalArgument(() -> new Evolution(new double[]{Double.NaN}));
        expectIllegalArgument(() -> new Evolution(new double[]{-0.1}));
        expectIllegalArgument(() -> new Evolution(new double[]{1.1}));

        var config = Evolution.Config.defaults().withGeneBounds(-2, 2);
        var evolution = new Evolution(new double[]{-1.5, 1.5}, config);
        require(evolution.geneCount() == 2, "custom gene bounds");
    }

    private static void verifyDefensiveCopies() {
        var goal = new double[]{0.25, 0.75};
        var evolution = new Evolution(goal, Evolution.Config.defaults().withSeed(7));
        goal[0] = 1.0;
        require(Arrays.equals(evolution.goal(), new double[]{0.25, 0.75}), "constructor copies goal");

        var returned = evolution.goal();
        returned[1] = 0.0;
        require(evolution.goal()[1] == 0.75, "goal accessor copies");

        var best = evolution.bestGenome();
        best[0] = -100;
        require(evolution.bestGenome()[0] >= 0.0, "best genome accessor copies");
    }

    private static void verifyDeterministicSeed() {
        var config = Evolution.Config.defaults()
                .withPopulationSize(128)
                .withMutationProbability(0.08)
                .withSeed(123456789L);
        var goal = new double[]{1, 0.25, 0.5, 0.75, 0};
        var first = new Evolution(goal, config);
        var second = new Evolution(goal, config);

        for (int generation = 0; generation < 250; generation++) {
            var firstStats = first.evolve();
            var secondStats = second.evolve();
            require(firstStats.equals(secondStats), "same seed must produce identical statistics");
        }
        require(Arrays.equals(first.bestGenome(), second.bestGenome()), "same seed must produce identical best genome");
    }

    private static void verifyFixedPopulation() {
        var config = Evolution.Config.defaults().withPopulationSize(37).withSeed(9);
        var evolution = new Evolution(new double[]{1, 1, 1}, config);
        for (int generation = 0; generation < 100; generation++) {
            var stats = evolution.evolve();
            require(stats.populationSize() == 37, "population size stays fixed");
        }
    }

    private static void verifyElitism() {
        var evolution = new Evolution(
                new double[]{1, 1, 1, 1, 1, 1},
                Evolution.Config.defaults()
                        .withPopulationSize(96)
                        .withMutationProbability(1.0)
                        .withSeed(13));
        var previous = evolution.bestError();
        for (int generation = 0; generation < 500; generation++) {
            var current = evolution.evolve().bestError();
            require(current <= previous, "elitism must make best error non-increasing");
            previous = current;
        }
    }

    private static void verifyGeneBounds() {
        var config = Evolution.Config.defaults()
                .withGeneBounds(-3, 5)
                .withPopulationSize(128)
                .withMutationProbability(1.0)
                .withSeed(15);
        var evolution = new Evolution(new double[]{4, -2, 0}, config);
        evolution.evolve(100);
        for (var value : evolution.bestGenome()) {
            require(value >= -3 && value <= 5, "best genome remains inside bounds");
        }
    }

    private static void verifyConvergence() {
        var evolution = new Evolution(
                new double[]{1, 1, 1, 1},
                Evolution.Config.defaults()
                        .withPopulationSize(512)
                        .withTournamentSize(6)
                        .withMutationProbability(0.08)
                        .withSeed(42));
        var before = evolution.bestError();
        var result = evolution.evolveUntil(0.001, 10_000);
        require(result.converged(), "four-gene target should converge");
        require(result.bestError() < before, "best error decreases");
        require(evolution.bestError() <= 0.001, "target error reached");
    }

    private static void verifySingleGeneSearch() {
        var evolution = new Evolution(
                new double[]{0.12345},
                Evolution.Config.defaults()
                        .withPopulationSize(128)
                        .withMutationProbability(0.2)
                        .withSeed(99));
        var result = evolution.evolveUntil(1.0e-5, 5_000);
        require(result.converged(), "single-gene search converges");
    }

    private static void verifyGenerationLimit() {
        var evolution = new Evolution(
                new double[]{1, 1, 1, 1, 1, 1, 1, 1},
                Evolution.Config.defaults()
                        .withPopulationSize(32)
                        .withMutationProbability(0.0)
                        .withCrossoverProbability(0.0)
                        .withSeed(2));
        var result = evolution.evolveUntil(0.0, 3);
        require(result.generations() == 3, "generation limit respected");
        require(evolution.generation() == 3, "generation statistic");
    }

    private static void verifyDistanceKernel() {
        var genome = new double[]{99, 99, 1, 2, 3};
        var goal = new double[]{1, 1, 1};
        var error = Evolution.meanSquaredError(genome, 2, goal);
        require(Math.abs(error - (0.0 + 1.0 + 4.0) / 3.0) < 1.0e-15, "MSE kernel");
    }

    private static void expectIllegalArgument(Runnable action) {
        try {
            action.run();
            throw new AssertionError("IllegalArgumentException expected");
        } catch (IllegalArgumentException expected) {
        }
    }

    private static void require(boolean condition, String message) {
        if (!condition) {
            throw new AssertionError(message);
        }
    }
}
