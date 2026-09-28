package com.lis.neuro;

import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

class NeuroTest {
    @Test
    void hyperParameterValidationAndCopiesAreCovered() {
        var defaults = Neuro.HyperParameters.defaults();

        assertEquals(0.5, defaults.learningRate());
        assertEquals(0.2, defaults.momentum());
        assertEquals(1.0, defaults.beta());

        assertThrows(IllegalArgumentException.class,
                () -> new Neuro.HyperParameters(0.0, 0.2, 1.0, 1));
        assertThrows(IllegalArgumentException.class,
                () -> new Neuro.HyperParameters(Double.NaN, 0.2, 1.0, 1));
        assertThrows(IllegalArgumentException.class,
                () -> new Neuro.HyperParameters(0.5, -0.1, 1.0, 1));
        assertThrows(IllegalArgumentException.class,
                () -> new Neuro.HyperParameters(0.5, 1.0, 1.0, 1));
        assertThrows(IllegalArgumentException.class,
                () -> new Neuro.HyperParameters(0.5, Double.NaN, 1.0, 1));
        assertThrows(IllegalArgumentException.class,
                () -> new Neuro.HyperParameters(0.5, 0.2, 0.0, 1));
        assertThrows(IllegalArgumentException.class,
                () -> new Neuro.HyperParameters(0.5, 0.2, Double.NaN, 1));

        var changed = defaults
                .withLearningRate(0.3)
                .withMomentum(0.4)
                .withBeta(1.5)
                .withSeed(1234);

        assertEquals(0.3, changed.learningRate());
        assertEquals(0.4, changed.momentum());
        assertEquals(1.5, changed.beta());
        assertEquals(1234, changed.seed());
    }

    @Test
    void topologyConstructionAndAccessorsAreCovered() {
        var topology = new int[]{2, 3, 1};
        var neuro = new Neuro(topology);

        topology[0] = 99;
        assertArrayEquals(new int[]{2, 3, 1}, neuro.topology());

        var returned = neuro.topology();
        returned[1] = 99;
        assertArrayEquals(new int[]{2, 3, 1}, neuro.topology());

        assertEquals(13, neuro.parameterCount());
        assertEquals(0, neuro.trainingSampleCount());
        assertEquals(0, neuro.testSampleCount());
        assertEquals(0, neuro.statistics().epochsTrained());
        assertEquals(0, neuro.statistics().samplesSeen());
        assertTrue(Double.isNaN(neuro.statistics().lastTrainingError()));
        assertNotNull(neuro.hyperParameters());

        var legacy = new Neuro(new int[]{1, 1}, 0.3, 1.2, 0.4);
        assertEquals(0.3, legacy.hyperParameters().momentum());
        assertEquals(1.2, legacy.hyperParameters().beta());
        assertEquals(0.4, legacy.hyperParameters().learningRate());

        assertThrows(NullPointerException.class, () -> new Neuro(null));
        assertThrows(NullPointerException.class,
                () -> new Neuro(new int[]{1, 1}, (Neuro.HyperParameters) null));
        assertThrows(IllegalArgumentException.class, () -> new Neuro(new int[]{1}));
        assertThrows(IllegalArgumentException.class, () -> new Neuro(new int[]{1, 0}));
        assertThrows(IllegalArgumentException.class, () -> new Neuro(new int[]{-1, 1}));
    }

    @Test
    void sampleValidationAndDefensiveCopiesAreCovered() {
        var neuro = new Neuro(new int[]{2, 2, 1});
        var input = new double[]{0.0, 1.0};
        var target = new double[]{1.0};

        assertSame(neuro, neuro.addTrainingSample(input, target));
        assertSame(neuro, neuro.addTestSample(input, target));
        assertEquals(1, neuro.trainingSampleCount());
        assertEquals(1, neuro.testSampleCount());

        input[0] = Double.NaN;
        target[0] = Double.NaN;
        assertTrue(Double.isFinite(neuro.trainingError()));
        assertTrue(Double.isFinite(neuro.testError()));

        assertThrows(NullPointerException.class, () -> neuro.addTrainingSample(null, new double[]{1}));
        assertThrows(NullPointerException.class, () -> neuro.addTrainingSample(new double[]{1, 2}, null));
        assertThrows(IllegalArgumentException.class,
                () -> neuro.addTrainingSample(new double[]{1}, new double[]{1}));
        assertThrows(IllegalArgumentException.class,
                () -> neuro.addTrainingSample(new double[]{1, 2}, new double[]{1, 2}));
        assertThrows(IllegalArgumentException.class,
                () -> neuro.addTrainingSample(new double[]{Double.NaN, 0}, new double[]{1}));
        assertThrows(IllegalArgumentException.class,
                () -> neuro.addTrainingSample(new double[]{0, 0}, new double[]{Double.POSITIVE_INFINITY}));
    }

    @Test
    void predictionPathsAndValidationAreCovered() {
        var neuro = new Neuro(
                new int[]{5, 4, 2},
                Neuro.HyperParameters.defaults().withSeed(7).withBeta(1.5));

        var prediction = neuro.predict(new double[]{1, -1, 0.5, 0.25, -0.75});
        assertEquals(2, prediction.length);
        for (var value : prediction) {
            assertTrue(value > 0.0 && value < 1.0);
        }

        var output = new double[2];
        neuro.predictInto(new double[]{-1, 1, -0.5, 0.75, 0.1}, output);
        assertFalse(Arrays.equals(new double[2], output));

        assertThrows(NullPointerException.class, () -> neuro.predict(null));
        assertThrows(NullPointerException.class,
                () -> neuro.predictInto(new double[]{1, 2, 3, 4, 5}, null));
        assertThrows(IllegalArgumentException.class,
                () -> neuro.predictInto(new double[]{1, 2, 3, 4, 5}, new double[1]));
        assertThrows(IllegalArgumentException.class,
                () -> neuro.predict(new double[]{1, 2}));
    }

    @Test
    void trainingCoversForwardBackpropagationMomentumAndStatistics() {
        var neuro = xorNetwork();

        var initialError = neuro.trainingError();
        var firstEpoch = neuro.trainEpoch();
        assertTrue(Double.isFinite(firstEpoch));

        neuro.train(25);

        var stats = neuro.statistics();
        assertEquals(26, stats.epochsTrained());
        assertEquals(26L * 4, stats.samplesSeen());
        assertEquals(neuro.trainingError(), stats.lastTrainingError(), 1.0e-12);
        assertTrue(Double.isFinite(neuro.testError()));
        assertTrue(neuro.trainingError() <= initialError || neuro.trainingError() < 0.75);

        var prediction = neuro.predict(new double[]{0, 1});
        assertEquals(1, prediction.length);
    }

    @Test
    void trainingValidationAndEmptyDataPathsAreCovered() {
        var neuro = new Neuro(new int[]{1, 1});

        assertTrue(Double.isNaN(neuro.trainingError()));
        assertTrue(Double.isNaN(neuro.testError()));
        assertThrows(IllegalStateException.class, neuro::trainEpoch);
        assertThrows(IllegalArgumentException.class, () -> neuro.train(-1));
        assertDoesNotThrow(() -> neuro.train(0));
        assertThrows(IllegalArgumentException.class, () -> neuro.trainUntil(-1.0, 1));
        assertThrows(IllegalArgumentException.class, () -> neuro.trainUntil(Double.NaN, 1));
        assertThrows(IllegalArgumentException.class, () -> neuro.trainUntil(0.1, -1));
        assertThrows(IllegalStateException.class, () -> neuro.trainUntil(0.1, 1));
    }

    @Test
    void trainUntilCoversImmediateSuccessLimitAndConvergenceResults() {
        var neuro = xorNetwork();

        var immediate = neuro.trainUntil(neuro.trainingError(), 100);
        assertTrue(immediate.converged());
        assertEquals(0, immediate.epochs());

        var limited = neuro.trainUntil(0.0, 2);
        assertFalse(limited.converged());
        assertEquals(2, limited.epochs());
        assertTrue(Double.isFinite(limited.error()));

        var easy = new Neuro(
                new int[]{1, 2, 1},
                Neuro.HyperParameters.defaults()
                        .withLearningRate(0.8)
                        .withMomentum(0.1)
                        .withSeed(11));
        easy.addTrainingSample(new double[]{0}, new double[]{0});
        easy.addTrainingSample(new double[]{1}, new double[]{1});
        var result = easy.trainUntil(1.0, 10);
        assertTrue(result.converged());
        assertEquals(0, result.epochs());
    }

    @Test
    void recordsRetainValues() {
        var result = new Neuro.TrainingResult(3, 0.2, true);
        var stats = new Neuro.Statistics(4, 20, 0.1);

        assertEquals(3, result.epochs());
        assertEquals(0.2, result.error());
        assertTrue(result.converged());
        assertEquals(4, stats.epochsTrained());
        assertEquals(20, stats.samplesSeen());
        assertEquals(0.1, stats.lastTrainingError());
    }

    private static Neuro xorNetwork() {
        var neuro = new Neuro(
                new int[]{2, 4, 1},
                Neuro.HyperParameters.defaults()
                        .withLearningRate(0.7)
                        .withMomentum(0.3)
                        .withSeed(42));

        neuro.addTrainingSample(new double[]{0, 0}, new double[]{0});
        neuro.addTrainingSample(new double[]{0, 1}, new double[]{1});
        neuro.addTrainingSample(new double[]{1, 0}, new double[]{1});
        neuro.addTrainingSample(new double[]{1, 1}, new double[]{0});

        neuro.addTestSample(new double[]{0, 0}, new double[]{0});
        neuro.addTestSample(new double[]{0, 1}, new double[]{1});
        neuro.addTestSample(new double[]{1, 0}, new double[]{1});
        neuro.addTestSample(new double[]{1, 1}, new double[]{0});
        return neuro;
    }
}
