package com.lis;

import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;

public final class EvolutionVerification {
    public static void main(String[] args) {
        verifyPolicyValidation();
        verifyGoalAndGenericFitness();
        verifyScalarAndVectorFitness();
        verifySelectionPolicies();
        verifyCrossoverPolicies();
        verifyMutationPolicies();
        verifyGeometricSkipping();
        verifyAdaptiveRestart();
        verifyMultipleElites();
        verifyParallelFitness();
        verifyInjectionAndRanking();
        verifyConvergence();
        verifyIslandEvolution();
        verifyParetoEvolution();
        verifyFailureCases();
        System.out.println("Evolution verification passed");
    }

    private static void verifyPolicyValidation() {
        expectIllegalArgument(() -> new Evolution.SelectionPolicy(
                Evolution.SelectionType.TOURNAMENT, 1, 0.5));
        expectIllegalArgument(() -> new Evolution.SelectionPolicy(
                Evolution.SelectionType.TOURNAMENT, 2, 0.0));
        expectIllegalArgument(() -> new Evolution.CrossoverPolicy(
                Evolution.CrossoverType.BLX_ALPHA, -0.1, 0.2));
        expectIllegalArgument(() -> new Evolution.CrossoverPolicy(
                Evolution.CrossoverType.BLX_ALPHA, 0.9, -0.1));
        expectIllegalArgument(() -> new Evolution.MutationPolicy(
                Evolution.MutationType.GAUSSIAN, 0.1, 0.0, 0.001, 1, 0.99, 2, 10, 0.2, true, true));
        expectIllegalArgument(() -> Evolution.Config.defaults().withEliteCount(256));
        expectIllegalArgument(() -> Evolution.Config.defaults().withFitnessParallelism(0));
    }

    private static void verifyGoalAndGenericFitness() {
        var goal = new double[]{0.25, 0.75};
        try (var evolution = new Evolution(goal, Evolution.Config.defaults().withSeed(1))) {
            goal[0] = 1.0;
            require(Arrays.equals(evolution.goal(), new double[]{0.25, 0.75}), "goal copied");
            require(evolution.hasGoal(), "goal-backed search");
        }

        EvolutionFitness sumFitness = (genome, offset, length) -> {
            var sum = 0.0;
            for (int index = 0; index < length; index++) {
                sum += genome[offset + index];
            }
            return sum;
        };
        try (var evolution = new Evolution(3, sumFitness, Evolution.Config.defaults().withSeed(2))) {
            require(!evolution.hasGoal(), "custom fitness has no goal");
            expectIllegalState(evolution::goal);
            evolution.evolve(5);
            require(Double.isFinite(evolution.bestError()), "custom fitness evaluates");
        }
    }

    private static void verifyScalarAndVectorFitness() {
        var goal = new double[257];
        var genome = new double[300];
        for (int index = 0; index < goal.length; index++) {
            goal[index] = (index & 15) / 15.0;
            genome[17 + index] = ((index * 3) & 15) / 15.0;
        }

        var scalar = EvolutionFitness.meanSquaredError(goal).evaluate(genome, 17, goal.length);
        var vector = EvolutionFitness.vectorMeanSquaredError(goal).evaluate(genome, 17, goal.length);
        require(Math.abs(scalar - vector) < 1.0e-12, "scalar/vector MSE match");
        require(EvolutionVectorFitness.preferredLaneCount() > 0, "vector species available");
    }

    private static void verifySelectionPolicies() {
        for (var type : Evolution.SelectionType.values()) {
            var selection = Evolution.SelectionPolicy.defaults()
                    .withType(type)
                    .withTournamentSize(3)
                    .withTruncationFraction(0.4);
            var config = compactConfig(10 + type.ordinal())
                    .withSelection(selection);
            try (var evolution = new Evolution(new double[]{1, 1, 1, 1}, config)) {
                var before = evolution.bestError();
                evolution.evolve(50);
                require(evolution.bestError() <= before, "selection " + type + " preserves elite");
            }
        }
    }

    private static void verifyCrossoverPolicies() {
        for (var type : Evolution.CrossoverType.values()) {
            var crossover = Evolution.CrossoverPolicy.defaults()
                    .withType(type)
                    .withProbability(1.0)
                    .withBlxAlpha(0.5);
            var config = compactConfig(20 + type.ordinal())
                    .withCrossover(crossover);
            try (var evolution = new Evolution(new double[]{0.2, 0.4, 0.6, 0.8}, config)) {
                evolution.evolve(20);
                assertBounds(evolution.bestGenome(), config);
            }
        }
    }

    private static void verifyMutationPolicies() {
        for (var type : Evolution.MutationType.values()) {
            var mutation = Evolution.MutationPolicy.defaults()
                    .withType(type)
                    .withProbability(1.0)
                    .withAdaptation(false, 1.0, 1.0, 10, 0.0);
            var config = compactConfig(30 + type.ordinal())
                    .withMutation(mutation);
            try (var evolution = new Evolution(new double[]{0.5, 0.5, 0.5}, config)) {
                evolution.evolve(10);
                assertBounds(evolution.bestGenome(), config);
            }
        }
    }

    private static void verifyGeometricSkipping() {
        var base = Evolution.MutationPolicy.defaults()
                .withProbability(0.05)
                .withAdaptation(false, 1.0, 1.0, 10, 0.0);
        for (var skipping : new boolean[]{false, true}) {
            var config = compactConfig(40 + (skipping ? 1 : 0))
                    .withMutation(base.withGeometricSkipping(skipping));
            try (var evolution = new Evolution(new double[64], config)) {
                evolution.evolve(20);
                require(evolution.statistics().fitnessEvaluations() > 0, "mutation path executes");
            }
        }
    }

    private static void verifyAdaptiveRestart() {
        var mutation = Evolution.MutationPolicy.defaults()
                .withProbability(0.1)
                .withSigma(0.05)
                .withAdaptation(true, 0.9, 2.0, 1, 0.5);
        var config = compactConfig(50).withMutation(mutation);
        EvolutionFitness constant = (genome, offset, length) -> 1.0;

        try (var evolution = new Evolution(8, constant, config)) {
            var initialSigma = evolution.statistics().mutationSigma();
            evolution.evolve();
            require(evolution.statistics().restarts() == 1, "stagnation restart");
            require(evolution.statistics().mutationSigma() > initialSigma, "sigma boosted");
        }
    }

    private static void verifyMultipleElites() {
        var config = compactConfig(60)
                .withEliteCount(4)
                .withMutation(Evolution.MutationPolicy.defaults().withProbability(1.0));
        try (var evolution = new Evolution(new double[]{1, 1, 1, 1}, config)) {
            var previous = evolution.bestError();
            for (int generation = 0; generation < 100; generation++) {
                var current = evolution.evolve().bestError();
                require(current <= previous, "multiple elitism preserves best");
                previous = current;
            }
        }
    }

    private static void verifyParallelFitness() {
        var calls = new AtomicInteger();
        EvolutionFitness expensive = (genome, offset, length) -> {
            calls.incrementAndGet();
            var sum = 0.0;
            for (int repeat = 0; repeat < 8; repeat++) {
                for (int gene = 0; gene < length; gene++) {
                    var value = genome[offset + gene] - 0.5;
                    sum = Math.fma(value, value, sum);
                }
            }
            return sum;
        };

        var config = compactConfig(70).withFitnessParallelism(4);
        try (var evolution = new Evolution(32, expensive, config)) {
            evolution.evolve(3);
            require(calls.get() >= config.populationSize(), "parallel fitness called");
            require(evolution.statistics().fitnessEvaluations() >= config.populationSize(), "fitness count");
        }
    }

    private static void verifyInjectionAndRanking() {
        var config = compactConfig(80).withGeneBounds(-1, 1);
        try (var evolution = new Evolution(new double[]{0, 0, 0}, config)) {
            var destination = new double[3];
            evolution.copyRankedGenomeInto(0, destination);
            require(Arrays.equals(destination, evolution.bestGenome()), "rank zero is best");
            require(evolution.injectGenome(new double[]{0, 0, 0}), "perfect migrant accepted");
            require(evolution.bestError() == 0.0, "injected optimum becomes best");
            require(!evolution.injectGenome(new double[]{1, 1, 1}), "bad migrant rejected");
            expectIllegalArgument(() -> evolution.copyRankedGenomeInto(-1, destination));
            expectIllegalArgument(() -> evolution.injectGenome(new double[]{2, 0, 0}));
        }
    }

    private static void verifyConvergence() {
        var config = Evolution.Config.defaults()
                .withPopulationSize(256)
                .withEliteCount(4)
                .withSelection(Evolution.SelectionPolicy.defaults().withTournamentSize(6))
                .withMutation(Evolution.MutationPolicy.defaults().withProbability(0.08))
                .withSeed(42);

        try (var evolution = new Evolution(new double[]{1, 1, 1, 1, 1, 1, 1, 1}, config)) {
            var result = evolution.evolveUntil(1.0e-4, 5_000);
            require(result.converged(), "adaptive real-valued GA converges");
            require(result.fitnessEvaluations() > 0, "search reports evaluations");
            require(evolution.statistics().geneDiversity() >= 0.0, "diversity tracked");
            require(evolution.statistics().fitnessStdDev() >= 0.0, "fitness deviation tracked");
        }
    }

    private static void verifyIslandEvolution() {
        var config = compactConfig(90).withPopulationSize(64);
        for (var parallel : new boolean[]{false, true}) {
            var islandConfig = new IslandEvolution.Config(3, 2, 2, parallel);
            try (var islands = new IslandEvolution(new double[]{1, 1, 1, 1}, config, islandConfig)) {
                islands.evolve(4);
                require(islands.statistics().migrations() == 2, "island migration cadence");
                require(islands.bestGenome().length == 4, "island best genome");
                var result = islands.evolveUntil(0.01, 500);
                require(result.bestError() <= islands.statistics().averageIslandBest(), "island best");
            }
        }
    }

    private static void verifyParetoEvolution() {
        MultiObjectiveFitness objectives = (genome, offset, length, output, objectiveOffset) -> {
            var first = genome[offset];
            output[objectiveOffset] = first * first;
            var difference = 1.0 - first;
            output[objectiveOffset + 1] = difference * difference;
        };

        var config = compactConfig(100)
                .withPopulationSize(64)
                .withEliteCount(4)
                .withMutation(Evolution.MutationPolicy.defaults().withProbability(0.1));
        var pareto = new ParetoEvolution(1, 2, objectives, config);
        pareto.evolve(50);
        require(pareto.statistics().paretoFrontSize() > 1, "Pareto front contains trade-offs");
        var front = pareto.paretoFront();
        require(!front.isEmpty(), "Pareto front exported");
        var values = pareto.objectives(front.get(0));
        require(values.length == 2, "two objectives");
        expectIllegalArgument(() -> pareto.objectives(new double[]{0, 1}));
    }

    private static void verifyFailureCases() {
        expectIllegalArgument(() -> new Evolution(0, (g, o, l) -> 0, Evolution.Config.defaults()));
        expectIllegalArgument(() -> new Evolution(new double[]{}));
        expectIllegalArgument(() -> new Evolution(new double[]{Double.NaN}));
        expectIllegalArgument(() -> new IslandEvolution.Config(1, 1, 1, false));
        expectIllegalArgument(() -> new ParetoEvolution(
                1, 1, (g, o, l, d, p) -> {}, compactConfig(111)));

        EvolutionFitness bad = (genome, offset, length) -> Double.NaN;
        expectIllegalState(() -> {
            try (var ignored = new Evolution(2, bad, compactConfig(112))) {
            }
        });
    }

    private static Evolution.Config compactConfig(long seed) {
        return Evolution.Config.defaults()
                .withPopulationSize(48)
                .withEliteCount(2)
                .withSeed(seed)
                .withMutation(Evolution.MutationPolicy.defaults()
                        .withProbability(0.08)
                        .withAdaptation(false, 1.0, 1.0, 20, 0.0));
    }

    private static void assertBounds(double[] genome, Evolution.Config config) {
        for (var value : genome) {
            require(value >= config.minGene() && value <= config.maxGene(), "gene bounds");
        }
    }

    private static void expectIllegalArgument(Runnable action) {
        try {
            action.run();
            throw new AssertionError("IllegalArgumentException expected");
        } catch (IllegalArgumentException expected) {
        }
    }

    private static void expectIllegalState(Runnable action) {
        try {
            action.run();
            throw new AssertionError("IllegalStateException expected");
        } catch (IllegalStateException expected) {
        }
    }

    private static void require(boolean condition, String message) {
        if (!condition) {
            throw new AssertionError(message);
        }
    }
}
