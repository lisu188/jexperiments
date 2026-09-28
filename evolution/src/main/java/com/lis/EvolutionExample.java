package com.lis;

import java.util.Arrays;

public final class EvolutionExample {
    public static void main(String[] args) {
        var goal = new double[]{1, 1, 1, 1, 1, 1, 1, 1};
        var evolution = new Evolution(
                goal,
                Evolution.Config.defaults()
                        .withPopulationSize(512)
                        .withTournamentSize(6)
                        .withMutationProbability(0.08)
                        .withSeed(42));

        var before = evolution.statistics();
        var result = evolution.evolveUntil(0.001, 10_000);

        System.out.println("before = " + before);
        System.out.println("search = " + result);
        System.out.println("best = " + Arrays.toString(evolution.bestGenome()));
        System.out.println("after = " + evolution.statistics());
    }
}
