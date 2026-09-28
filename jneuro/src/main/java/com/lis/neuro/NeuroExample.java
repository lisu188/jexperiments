package com.lis.neuro;

import java.util.Arrays;

public final class NeuroExample {
    public static void main(String[] args) {
        var network = new Neuro(
                new int[]{2, 6, 1},
                Neuro.HyperParameters.defaults()
                        .withLearningRate(0.6)
                        .withMomentum(0.2)
                        .withSeed(42));

        addXorSamples(network);
        var before = network.trainingError();
        var result = network.trainUntil(0.05, 10_000);

        System.out.println("before = " + before);
        System.out.println("training = " + result);
        for (var input : new double[][]{
                {0, 0},
                {0, 1},
                {1, 0},
                {1, 1}}) {
            System.out.println(Arrays.toString(input) + " -> " + Arrays.toString(network.predict(input)));
        }
        System.out.println("statistics = " + network.statistics());
    }

    private static void addXorSamples(Neuro network) {
        network.addTrainingSample(new double[]{0, 0}, new double[]{0});
        network.addTrainingSample(new double[]{0, 1}, new double[]{1});
        network.addTrainingSample(new double[]{1, 0}, new double[]{1});
        network.addTrainingSample(new double[]{1, 1}, new double[]{0});
    }
}
