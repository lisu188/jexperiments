package com.lis.neuro;

import java.util.Arrays;

public final class NeuroVerification {
    public static void main(String[] args) {
        verifyTopologyValidation();
        verifySampleValidation();
        verifyDeterministicSeed();
        verifyDefensiveCopies();
        verifyPredictInto();
        verifyParameterCount();
        verifyOrConvergence();
        verifyXorConvergence();
        verifyTestErrorUsesTestSet();
        verifyEpochLimit();
        verifyStableSigmoid();
        System.out.println("JNeuro verification passed");
    }

    private static void verifyTopologyValidation() {
        expectIllegalArgument(() -> new Neuro(new int[]{2}));
        expectIllegalArgument(() -> new Neuro(new int[]{2, 0, 1}));
        expectIllegalArgument(() -> new Neuro(new int[]{-1, 1}));
    }

    private static void verifySampleValidation() {
        var network = new Neuro(new int[]{2, 1});
        expectIllegalArgument(() -> network.addTrainingSample(new double[]{1}, new double[]{1}));
        expectIllegalArgument(() -> network.addTrainingSample(new double[]{1, 0}, new double[]{1, 0}));
        expectIllegalArgument(() -> network.addTrainingSample(new double[]{Double.NaN, 0}, new double[]{1}));
        expectIllegalState(network::trainEpoch);
    }

    private static void verifyDeterministicSeed() {
        var parameters = Neuro.HyperParameters.defaults().withSeed(123456789L);
        var first = new Neuro(new int[]{3, 5, 2}, parameters);
        var second = new Neuro(new int[]{3, 5, 2}, parameters);
        var input = new double[]{0.25, 0.5, 0.75};
        require(Arrays.equals(first.predict(input), second.predict(input)), "same seed must create identical model");
    }

    private static void verifyDefensiveCopies() {
        var topology = new int[]{2, 3, 1};
        var network = new Neuro(topology);
        topology[1] = 99;
        require(Arrays.equals(network.topology(), new int[]{2, 3, 1}), "constructor must copy topology");

        var returned = network.topology();
        returned[0] = 99;
        require(network.topology()[0] == 2, "topology accessor must copy");

        var input = new double[]{0, 0};
        var target = new double[]{0};
        network.addTrainingSample(input, target);
        input[0] = 1;
        target[0] = 1;

        var reference = new Neuro(new int[]{2, 3, 1});
        reference.addTrainingSample(new double[]{0, 0}, new double[]{0});
        require(close(network.trainingError(), reference.trainingError(), 0.0), "samples must be copied");
    }

    private static void verifyPredictInto() {
        var network = new Neuro(new int[]{2, 4, 2}, Neuro.HyperParameters.defaults().withSeed(7));
        var input = new double[]{0.2, 0.8};
        var allocated = network.predict(input);
        var reused = new double[2];
        network.predictInto(input, reused);
        require(Arrays.equals(allocated, reused), "predictInto must match predict");
        expectIllegalArgument(() -> network.predictInto(input, new double[1]));
    }

    private static void verifyParameterCount() {
        var network = new Neuro(new int[]{2, 3, 1});
        require(network.parameterCount() == 13, "parameter count includes biases");
    }

    private static void verifyOrConvergence() {
        var network = new Neuro(
                new int[]{2, 3, 1},
                Neuro.HyperParameters.defaults()
                        .withLearningRate(0.6)
                        .withMomentum(0.1)
                        .withSeed(11));
        addOrSamples(network);

        var before = network.trainingError();
        var result = network.trainUntil(0.08, 5_000);
        require(result.converged(), "OR must converge");
        require(result.error() < before, "OR error must decrease");
        require(network.predict(new double[]{0, 0})[0] < 0.2, "OR 00");
        require(network.predict(new double[]{0, 1})[0] > 0.8, "OR 01");
        require(network.predict(new double[]{1, 0})[0] > 0.8, "OR 10");
        require(network.predict(new double[]{1, 1})[0] > 0.8, "OR 11");
    }

    private static void verifyXorConvergence() {
        var network = new Neuro(
                new int[]{2, 6, 1},
                Neuro.HyperParameters.defaults()
                        .withLearningRate(0.6)
                        .withMomentum(0.2)
                        .withSeed(42));
        addXorSamples(network);

        var result = network.trainUntil(0.08, 10_000);
        require(result.converged(), "XOR must converge");
        require(network.predict(new double[]{0, 0})[0] < 0.2, "XOR 00");
        require(network.predict(new double[]{0, 1})[0] > 0.8, "XOR 01");
        require(network.predict(new double[]{1, 0})[0] > 0.8, "XOR 10");
        require(network.predict(new double[]{1, 1})[0] < 0.2, "XOR 11");
    }

    private static void verifyTestErrorUsesTestSet() {
        var network = new Neuro(
                new int[]{2, 4, 1},
                Neuro.HyperParameters.defaults().withLearningRate(0.6).withSeed(99));
        addOrSamples(network);
        network.train(1_000);

        network.addTestSample(new double[]{0, 0}, new double[]{1});
        network.addTestSample(new double[]{1, 1}, new double[]{0});

        var p00 = network.predict(new double[]{0, 0})[0];
        var p11 = network.predict(new double[]{1, 1})[0];
        var expected = Math.sqrt((square(1.0 - p00) + square(p11)) / 2.0);
        require(close(network.testError(), expected, 1.0e-12), "test error denominator must use test set");
        require(network.testSampleCount() == 2, "test sample count");
    }

    private static void verifyEpochLimit() {
        var network = new Neuro(new int[]{2, 2, 1}, Neuro.HyperParameters.defaults().withSeed(17));
        addXorSamples(network);
        var result = network.trainUntil(1.0e-15, 3);
        require(result.epochs() == 3, "max epoch limit");
        require(!result.converged(), "unreachable target should report not converged");
        require(network.statistics().epochsTrained() == 3, "epoch statistics");
        require(network.statistics().samplesSeen() == 12, "sample statistics");
    }

    private static void verifyStableSigmoid() {
        var network = new Neuro(new int[]{1, 1}, Neuro.HyperParameters.defaults().withSeed(3));
        for (var value : new double[]{-1.0e300, 1.0e300}) {
            var output = network.predict(new double[]{value})[0];
            require(Double.isFinite(output), "sigmoid must stay finite");
            require(output >= 0.0 && output <= 1.0, "sigmoid range");
        }
        require(Double.isNaN(network.testError()), "empty test set returns NaN");
    }

    private static void addOrSamples(Neuro network) {
        network.addTrainingSample(new double[]{0, 0}, new double[]{0});
        network.addTrainingSample(new double[]{0, 1}, new double[]{1});
        network.addTrainingSample(new double[]{1, 0}, new double[]{1});
        network.addTrainingSample(new double[]{1, 1}, new double[]{1});
    }

    private static void addXorSamples(Neuro network) {
        network.addTrainingSample(new double[]{0, 0}, new double[]{0});
        network.addTrainingSample(new double[]{0, 1}, new double[]{1});
        network.addTrainingSample(new double[]{1, 0}, new double[]{1});
        network.addTrainingSample(new double[]{1, 1}, new double[]{0});
    }

    private static double square(double value) {
        return value * value;
    }

    private static boolean close(double first, double second, double tolerance) {
        return Math.abs(first - second) <= tolerance;
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
