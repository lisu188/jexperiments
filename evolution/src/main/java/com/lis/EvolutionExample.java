package com.lis;

import java.util.Arrays;

public final class EvolutionExample {
    public static void main(String[] args) {
        var goal = new double[]{1, 1, 1, 1, 1, 1, 1, 1};
        var config = Evolution.Config.defaults()
                .withPopulationSize(512)
                .withEliteCount(4)
                .withSelection(Evolution.SelectionPolicy.defaults().withTournamentSize(6))
                .withCrossover(Evolution.CrossoverPolicy.defaults()
                        .withType(Evolution.CrossoverType.BLX_ALPHA))
                .withMutation(Evolution.MutationPolicy.defaults()
                        .withProbability(0.08))
                .withSeed(42);

        try (var evolution = new Evolution(goal, config)) {
            var before = evolution.statistics();
            var result = evolution.evolveUntil(0.0001, 10_000);

            System.out.println("before = " + before);
            System.out.println("search = " + result);
            System.out.println("best = " + Arrays.toString(evolution.bestGenome()));
            System.out.println("after = " + evolution.statistics());
        }

        try (var islands = new IslandEvolution(
                goal,
                config.withPopulationSize(128),
                IslandEvolution.Config.defaults())) {
            System.out.println("islands = " + islands.evolveUntil(0.0001, 5_000));
        }
    }
}
