package com.lis.neuro;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class NeuroTest {
    @Test
    void trainsPredictsAndTracksStatistics() {
        var parameters = Neuro.HyperParameters.defaults()
                .withLearningRate(0.6)
                .withMomentum(0.1)
                .withBeta(1.2)
                .withSeed(42L);
        var neuro = new Neuro(new int[]{2, 4, 1}, parameters);

        assertArrayEquals(new int[]{2, 4, 1}, neuro.topology());
        assertEquals(parameters, neuro.hyperParameters());
        assertEquals(17, neuro.parameterCount());
        assertEquals(0, neuro.trainingSampleCount());
        assertEquals(0, neuro.testSampleCount());
        assertTrue(Double.isNaN(neuro.trainingError()));
        assertTrue(Double.isNaN(neuro.testError()));

        neuro.addTrainingSample(new double[]{0, 0}, new double[]{0})
                .addTrainingSample(new double[]{0, 1}, new double[]{1})
                .addTrainingSample(new double[]{1, 0}, new double[]{1})
                .addTrainingSample(new double[]{1, 1}, new double[]{0})
                .addTestSample(new double[]{0, 1}, new double[]{1});

        assertEquals(4, neuro.trainingSampleCount());
        assertEquals(1, neuro.testSampleCount());

        var before = neuro.predict(new double[]{0, 1});
        assertEquals(1, before.length);
        assertTrue(before[0] > 0.0 && before[0] < 1.0);

        var output = new double[1];
        neuro.predictInto(new double[]{1, 0}, output);
        assertTrue(output[0] > 0.0 && output[0] < 1.0);

        var firstError = neuro.trainEpoch();
        assertTrue(Double.isFinite(firstError));
        neuro.train(2);

        var statistics = neuro.statistics();
        assertEquals(3, statistics.epochsTrained());
        assertEquals(12, statistics.samplesSeen());
        assertEquals(neuro.trainingError(), statistics.lastTrainingError(), 1.0e-12);
        assertTrue(Double.isFinite(neuro.testError()));

        var result = neuro.trainUntil(0.0, 1);
        assertEquals(1, result.epochs());
        assertFalse(result.converged());
        assertTrue(Double.isFinite(result.error()));

        var alreadyGood = neuro.trainUntil(1.0, 10);
        assertEquals(0, alreadyGood.epochs());
        assertTrue(alreadyGood.converged());
    }

    @Test
    void constructorsAndHyperParameterValidationAreCovered() {
        var defaults = Neuro.HyperParameters.defaults();
        assertEquals(0.5, defaults.learningRate());
        assertEquals(0.2, defaults.momentum());
        assertEquals(1.0, defaults.beta());

        var first = new Neuro(new int[]{1, 1});
        var second = new Neuro(new int[]{1, 2, 1}, 0.1, 1.0, 0.4);
        assertEquals(2, first.parameterCount());
        assertEquals(7, second.parameterCount());

        assertThrows(IllegalArgumentException.class,
                () -> new Neuro.HyperParameters(0.0, 0.1, 1.0, 1));
        assertThrows(IllegalArgumentException.class,
                () -> new Neuro.HyperParameters(Double.NaN, 0.1, 1.0, 1));
        assertThrows(IllegalArgumentException.class,
                () -> new Neuro.HyperParameters(0.1, -0.1, 1.0, 1));
        assertThrows(IllegalArgumentException.class,
                () -> new Neuro.HyperParameters(0.1, 1.0, 1.0, 1));
        assertThrows(IllegalArgumentException.class,
                () -> new Neuro.HyperParameters(0.1, Double.NaN, 1.0, 1));
        assertThrows(IllegalArgumentException.class,
                () -> new Neuro.HyperParameters(0.1, 0.1, 0.0, 1));
        assertThrows(IllegalArgumentException.class,
                () -> new Neuro.HyperParameters(0.1, 0.1, Double.NaN, 1));
    }

    @Test
    void rejectsInvalidTopologySamplesTrainingAndPredictionArguments() {
        assertThrows(NullPointerException.class, () -> new Neuro(null));
        assertThrows(IllegalArgumentException.class, () -> new Neuro(new int[]{1}));
        assertThrows(IllegalArgumentException.class, () -> new Neuro(new int[]{1, 0}));
        assertThrows(NullPointerException.class,
                () -> new Neuro(new int[]{1, 1}, (Neuro.HyperParameters) null));

        var neuro = new Neuro(new int[]{2, 1});

        assertThrows(NullPointerException.class,
                () -> neuro.addTrainingSample(null, new double[]{0}));
        assertThrows(NullPointerException.class,
                () -> neuro.addTrainingSample(new double[]{0, 0}, null));
        assertThrows(IllegalArgumentException.class,
                () -> neuro.addTrainingSample(new double[]{0}, new double[]{0}));
        assertThrows(IllegalArgumentException.class,
                () -> neuro.addTrainingSample(new double[]{0, 0}, new double[]{0, 1}));
        assertThrows(IllegalArgumentException.class,
                () -> neuro.addTrainingSample(new double[]{Double.NaN, 0}, new double[]{0}));
        assertThrows(IllegalArgumentException.class,
                () -> neuro.addTrainingSample(new double[]{0, 0}, new double[]{Double.POSITIVE_INFINITY}));

        assertThrows(NullPointerException.class, () -> neuro.predict(null));
        assertThrows(NullPointerException.class,
                () -> neuro.predictInto(new double[]{0, 0}, null));
        assertThrows(IllegalArgumentException.class,
                () -> neuro.predictInto(new double[]{0, 0}, new double[2]));
        assertThrows(IllegalArgumentException.class,
                () -> neuro.predict(new double[]{0}));

        assertThrows(IllegalStateException.class, neuro::trainEpoch);
        assertThrows(IllegalArgumentException.class, () -> neuro.train(-1));
        neuro.train(0);
        assertThrows(IllegalArgumentException.class, () -> neuro.trainUntil(-1.0, 1));
        assertThrows(IllegalArgumentException.class, () -> neuro.trainUntil(Double.NaN, 1));
        assertThrows(IllegalArgumentException.class, () -> neuro.trainUntil(0.1, -1));
        assertThrows(IllegalStateException.class, () -> neuro.trainUntil(0.1, 1));
    }

    @Test
    void sigmoidCoversPositiveAndNegativeNumericalBranches() throws Exception {
        var sigmoid = Neuro.class.getDeclaredMethod("sigmoid", double.class);
        sigmoid.setAccessible(true);

        var positive = (double) sigmoid.invoke(null, 2.0);
        var negative = (double) sigmoid.invoke(null, -2.0);

        assertEquals(1.0 - negative, positive, 1.0e-12);
        assertEquals(0.5, (double) sigmoid.invoke(null, 0.0), 1.0e-12);
    }
}
