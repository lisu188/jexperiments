package com.lis.neuro;

import java.util.Arrays;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class NeuroTest {
    @Test
    void validatesConfigurationAndSamples() {
        assertThrows(NullPointerException.class, () -> new Neuro(null));
        assertThrows(IllegalArgumentException.class, () -> new Neuro(new int[]{2}));
        assertThrows(IllegalArgumentException.class, () -> new Neuro(new int[]{2, 0, 1}));
        assertThrows(IllegalArgumentException.class, () -> new Neuro(new int[]{-1, 1}));

        assertThrows(IllegalArgumentException.class, () -> new Neuro.HyperParameters(0.0, 0.1, 1.0, 1));
        assertThrows(IllegalArgumentException.class, () -> new Neuro.HyperParameters(0.1, -0.1, 1.0, 1));
        assertThrows(IllegalArgumentException.class, () -> new Neuro.HyperParameters(0.1, 1.0, 1.0, 1));
        assertThrows(IllegalArgumentException.class, () -> new Neuro.HyperParameters(0.1, 0.1, 0.0, 1));
        assertThrows(NullPointerException.class, () -> new Neuro.HyperParameters(
                0.1, 0.1, 1.0, 1, null, Neuro.SigmoidMode.EXACT));
        assertThrows(NullPointerException.class, () -> new Neuro.HyperParameters(
                0.1, 0.1, 1.0, 1, Neuro.Kernel.AUTO, null));

        var network = new Neuro(new int[]{2, 3, 1});
        assertThrows(NullPointerException.class, () -> network.addTrainingSample(null, new double[]{0}));
        assertThrows(NullPointerException.class, () -> network.addTrainingSample(new double[]{0, 0}, null));
        assertThrows(IllegalArgumentException.class, () -> network.addTrainingSample(new double[]{0}, new double[]{0}));
        assertThrows(IllegalArgumentException.class, () -> network.addTrainingSample(new double[]{0, 0}, new double[]{0, 1}));
        assertThrows(IllegalArgumentException.class, () -> network.addTrainingSample(
                new double[]{Double.NaN, 0}, new double[]{0}));
        assertThrows(IllegalArgumentException.class, () -> network.addTestSample(
                new double[]{0, 0}, new double[]{Double.POSITIVE_INFINITY}));

        assertThrows(IllegalStateException.class, network::trainEpoch);
        assertThrows(IllegalArgumentException.class, () -> network.train(-1));
        assertDoesNotThrow(() -> network.train(0));
        assertThrows(IllegalArgumentException.class, () -> network.trainUntil(-1, 1));
        assertThrows(IllegalArgumentException.class, () -> network.trainUntil(0.1, -1));
        assertThrows(IllegalArgumentException.class, () -> network.trainUntil(0.1, 1, 0));
        assertThrows(IllegalArgumentException.class, () -> network.trainMiniBatch(1, 0));
        assertThrows(IllegalArgumentException.class, () -> network.trainMiniBatch(-1, 1));
        assertThrows(IllegalArgumentException.class, () -> network.trainMiniBatch(1, 1, 0));
        assertDoesNotThrow(() -> network.trainMiniBatch(0, 1));
    }

    @Test
    void preservesDeterminismAndDefensiveCopies() {
        var parameters = Neuro.HyperParameters.defaults()
                .withLearningRate(0.3)
                .withMomentum(0.1)
                .withBeta(1.2)
                .withSeed(12345)
                .withKernel(Neuro.Kernel.SCALAR)
                .withSigmoidMode(Neuro.SigmoidMode.EXACT);
        assertEquals(0.3, parameters.learningRate());
        assertEquals(0.1, parameters.momentum());
        assertEquals(1.2, parameters.beta());
        assertEquals(12345, parameters.seed());
        assertEquals(Neuro.Kernel.SCALAR, parameters.kernel());

        var topology = new int[]{3, 5, 2};
        var first = new Neuro(topology, parameters);
        var second = new Neuro(topology, parameters);
        topology[1] = 99;
        assertArrayEquals(new int[]{3, 5, 2}, first.topology());

        var returned = first.topology();
        returned[0] = 99;
        assertArrayEquals(new int[]{3, 5, 2}, first.topology());
        assertEquals(parameters, first.hyperParameters());
        assertEquals(32, first.parameterCount());

        var input = new double[]{0.2, 0.4, 0.6};
        assertArrayEquals(first.predict(input), second.predict(input), 0.0);

        var sampleInput = new double[]{0.1, 0.2, 0.3};
        var sampleTarget = new double[]{0.0, 1.0};
        first.addTrainingSample(sampleInput, sampleTarget);
        sampleInput[0] = 9.0;
        sampleTarget[0] = 9.0;

        var reference = new Neuro(new int[]{3, 5, 2}, parameters);
        reference.addTrainingSample(new double[]{0.1, 0.2, 0.3}, new double[]{0.0, 1.0});
        assertEquals(reference.trainingError(), first.trainingError(), 0.0);
        assertEquals(1, first.trainingSampleCount());
        assertEquals(0, first.testSampleCount());
        assertTrue(Double.isNaN(first.testError()));
    }

    @Test
    void scalarVectorSessionsBatchAndFloatAgree() {
        var scalar = prepared(new int[]{32, 64, 32, 8}, Neuro.Kernel.SCALAR, Neuro.SigmoidMode.EXACT);
        var vector = prepared(new int[]{32, 64, 32, 8}, Neuro.Kernel.VECTOR, Neuro.SigmoidMode.EXACT);
        var automatic = prepared(new int[]{32, 64, 32, 8}, Neuro.Kernel.AUTO, Neuro.SigmoidMode.EXACT);
        var input = input(32);

        var scalarOutput = scalar.predict(input);
        var vectorOutput = vector.predict(input);
        var automaticOutput = automatic.predict(input);
        assertArrayEquals(scalarOutput, vectorOutput, 1.0e-12);
        assertArrayEquals(scalarOutput, automaticOutput, 1.0e-12);

        var reused = new double[8];
        vector.predictInto(input, reused);
        assertArrayEquals(vectorOutput, reused, 0.0);

        var session = vector.newInferenceSession();
        Arrays.fill(reused, Double.NaN);
        session.predictInto(input, reused);
        assertArrayEquals(vectorOutput, reused, 0.0);

        var batchSize = 5;
        var batchInputs = new double[32 * batchSize];
        for (int sample = 0; sample < batchSize; sample++) {
            System.arraycopy(input, 0, batchInputs, sample * 32, 32);
        }
        var batch = new double[8 * batchSize];
        var parallel = new double[8 * batchSize];
        var reusableParallel = new double[8 * batchSize];
        vector.predictBatch(batchInputs, batchSize, batch);
        vector.predictBatchParallel(batchInputs, batchSize, parallel, 2);
        try (var parallelSession = vector.newParallelInferenceSession(2)) {
            parallelSession.predictBatch(batchInputs, batchSize, reusableParallel);
            assertArrayEquals(batch, reusableParallel, 0.0);
        }
        assertArrayEquals(batch, parallel, 0.0);
        for (int sample = 0; sample < batchSize; sample++) {
            assertArrayEquals(vectorOutput, Arrays.copyOfRange(batch, sample * 8, sample * 8 + 8), 0.0);
        }

        var floatModel = vector.toFloatModel();
        assertArrayEquals(vector.topology(), floatModel.topology());
        var floatInput = new float[32];
        for (int i = 0; i < floatInput.length; i++) {
            floatInput[i] = (float) input[i];
        }
        var floatOutput = floatModel.predict(floatInput);
        var floatReused = new float[8];
        floatModel.predictInto(floatInput, floatReused);
        assertArrayEquals(floatOutput, floatReused, 0.0f);
        for (int i = 0; i < floatOutput.length; i++) {
            assertEquals(vectorOutput[i], floatOutput[i], 1.0e-5);
        }

        var floatBatchInputs = new float[floatInput.length * batchSize];
        for (int sample = 0; sample < batchSize; sample++) {
            System.arraycopy(floatInput, 0, floatBatchInputs, sample * floatInput.length, floatInput.length);
        }
        var floatBatch = new float[floatOutput.length * batchSize];
        floatModel.predictBatch(floatBatchInputs, batchSize, floatBatch);
        assertArrayEquals(floatOutput, Arrays.copyOfRange(floatBatch, 0, floatOutput.length), 0.0f);

        assertThrows(IllegalArgumentException.class, () -> vector.predictInto(new double[1], new double[8]));
        assertThrows(IllegalArgumentException.class, () -> vector.predictInto(input, new double[1]));
        assertThrows(IllegalArgumentException.class, () -> vector.predictBatch(new double[1], 1, new double[8]));
        assertThrows(IllegalArgumentException.class, () -> vector.predictBatch(batchInputs, -1, batch));
        assertThrows(IllegalArgumentException.class, () -> vector.predictBatch(batchInputs, 1, new double[1]));
        assertThrows(IllegalArgumentException.class, () -> vector.predictBatchParallel(batchInputs, 1, batch, 0));
        assertThrows(IllegalArgumentException.class, () -> vector.newParallelInferenceSession(0));
        assertDoesNotThrow(() -> vector.predictBatchParallel(new double[0], 0, new double[0], 1));

        assertThrows(IllegalArgumentException.class, () -> floatModel.predictInto(new float[1], new float[8]));
        assertThrows(IllegalArgumentException.class, () -> floatModel.predictInto(floatInput, new float[1]));
        assertThrows(IllegalArgumentException.class, () -> floatModel.predictBatch(new float[1], 1, new float[8]));
        assertThrows(IllegalArgumentException.class, () -> floatModel.predictBatch(floatBatchInputs, -1, floatBatch));
    }

    @Test
    void onlineTrainingConvergesAndTracksStatistics() {
        for (var kernel : Neuro.Kernel.values()) {
            var network = new Neuro(
                    new int[]{2, 6, 1},
                    Neuro.HyperParameters.defaults()
                            .withLearningRate(0.6)
                            .withMomentum(0.2)
                            .withSeed(42)
                            .withKernel(kernel));
            addXor(network);
            var before = network.trainingError();
            var result = network.trainUntil(0.08, 10_000);
            assertTrue(result.converged(), kernel.name());
            assertTrue(result.error() < before);
            assertTrue(network.predict(new double[]{0, 0})[0] < 0.2);
            assertTrue(network.predict(new double[]{0, 1})[0] > 0.8);
            assertTrue(network.predict(new double[]{1, 0})[0] > 0.8);
            assertTrue(network.predict(new double[]{1, 1})[0] < 0.2);
            assertEquals(result.epochs(), network.statistics().epochsTrained());
            assertEquals(result.epochs() * 4L, network.statistics().samplesSeen());
        }
    }

    @Test
    void fastSigmoidAndSparseErrorChecksWork() {
        var exact = prepared(new int[]{16, 32, 4}, Neuro.Kernel.AUTO, Neuro.SigmoidMode.EXACT);
        var fast = prepared(new int[]{16, 32, 4}, Neuro.Kernel.AUTO, Neuro.SigmoidMode.FAST);
        var input = input(16);
        var exactOutput = exact.predict(input);
        var fastOutput = fast.predict(input);
        for (int i = 0; i < exactOutput.length; i++) {
            assertEquals(exactOutput[i], fastOutput[i], 0.04);
        }

        var network = new Neuro(
                new int[]{2, 6, 1},
                Neuro.HyperParameters.defaults()
                        .withLearningRate(0.6)
                        .withMomentum(0.2)
                        .withSeed(42)
                        .withSigmoidMode(Neuro.SigmoidMode.FAST));
        addXor(network);
        var result = network.trainUntil(0.12, 10_000, 8);
        assertTrue(result.converged());

        var extreme = new Neuro(
                new int[]{1, 1},
                Neuro.HyperParameters.defaults().withSigmoidMode(Neuro.SigmoidMode.FAST));
        assertEquals(0.0, extreme.predict(new double[]{-1.0e300})[0], 0.0);
        assertEquals(1.0, extreme.predict(new double[]{1.0e300})[0], 0.0);
    }

    @Test
    void fixedEpochAndMiniBatchTrainingUsePackedData() {
        var online = prepared(new int[]{32, 64, 32, 8}, Neuro.Kernel.AUTO, Neuro.SigmoidMode.EXACT);
        var beforeOnline = online.trainingError();
        online.train(4);
        assertEquals(4, online.statistics().epochsTrained());
        assertEquals(128, online.statistics().samplesSeen());
        assertTrue(Double.isFinite(online.statistics().lastTrainingError()));
        assertTrue(online.trainingError() <= beforeOnline + 0.05);

        var miniBatch = prepared(new int[]{32, 64, 32, 8}, Neuro.Kernel.AUTO, Neuro.SigmoidMode.EXACT);
        var beforeBatch = miniBatch.trainingError();
        miniBatch.trainMiniBatch(4, 8);
        assertEquals(4, miniBatch.statistics().epochsTrained());
        assertEquals(128, miniBatch.statistics().samplesSeen());
        assertTrue(miniBatch.trainingError() < beforeBatch);

        var parallel = prepared(new int[]{32, 64, 32, 8}, Neuro.Kernel.AUTO, Neuro.SigmoidMode.EXACT);
        parallel.trainMiniBatch(2, 8, 2);
        assertEquals(2, parallel.statistics().epochsTrained());
        assertEquals(64, parallel.statistics().samplesSeen());
        assertTrue(Double.isFinite(parallel.trainingError()));

        var epoch = prepared(new int[]{8, 12, 2}, Neuro.Kernel.AUTO, Neuro.SigmoidMode.EXACT);
        assertTrue(Double.isFinite(epoch.trainEpoch()));
    }

    @Test
    void testSetErrorAndEpochLimitsRemainCorrect() {
        var network = new Neuro(
                new int[]{2, 4, 1},
                Neuro.HyperParameters.defaults().withLearningRate(0.6).withSeed(99));
        addOr(network);
        network.train(1_000);

        network.addTestSample(new double[]{0, 0}, new double[]{1});
        network.addTestSample(new double[]{1, 1}, new double[]{0});
        var p00 = network.predict(new double[]{0, 0})[0];
        var p11 = network.predict(new double[]{1, 1})[0];
        var expected = Math.sqrt((square(1.0 - p00) + square(p11)) / 2.0);
        assertEquals(expected, network.testError(), 1.0e-12);
        assertEquals(2, network.testSampleCount());

        var limited = new Neuro(new int[]{2, 2, 1}, Neuro.HyperParameters.defaults().withSeed(17));
        addXor(limited);
        var result = limited.trainUntil(1.0e-15, 3, 2);
        assertEquals(3, result.epochs());
        assertFalse(result.converged());

        var alreadyGood = new Neuro(new int[]{1, 1});
        alreadyGood.addTrainingSample(new double[]{0}, alreadyGood.predict(new double[]{0}));
        var immediate = alreadyGood.trainUntil(1.0, 10);
        assertEquals(0, immediate.epochs());
        assertTrue(immediate.converged());
    }

    private static Neuro prepared(int[] topology, Neuro.Kernel kernel, Neuro.SigmoidMode sigmoidMode) {
        var network = new Neuro(
                topology,
                Neuro.HyperParameters.defaults()
                        .withLearningRate(0.1)
                        .withMomentum(0.1)
                        .withSeed(1234)
                        .withKernel(kernel)
                        .withSigmoidMode(sigmoidMode));
        var outputSize = topology[topology.length - 1];
        for (int sample = 0; sample < 32; sample++) {
            var input = new double[topology[0]];
            var target = new double[outputSize];
            for (int i = 0; i < input.length; i++) {
                input[i] = ((sample + i) & 7) / 7.0;
            }
            for (int i = 0; i < target.length; i++) {
                target[i] = ((sample + i) & 1);
            }
            network.addTrainingSample(input, target);
        }
        return network;
    }

    private static double[] input(int size) {
        var input = new double[size];
        for (int i = 0; i < size; i++) {
            input[i] = (i & 7) / 7.0;
        }
        return input;
    }

    private static void addOr(Neuro network) {
        network.addTrainingSample(new double[]{0, 0}, new double[]{0});
        network.addTrainingSample(new double[]{0, 1}, new double[]{1});
        network.addTrainingSample(new double[]{1, 0}, new double[]{1});
        network.addTrainingSample(new double[]{1, 1}, new double[]{1});
    }

    private static void addXor(Neuro network) {
        network.addTrainingSample(new double[]{0, 0}, new double[]{0});
        network.addTrainingSample(new double[]{0, 1}, new double[]{1});
        network.addTrainingSample(new double[]{1, 0}, new double[]{1});
        network.addTrainingSample(new double[]{1, 1}, new double[]{0});
    }

    private static double square(double value) {
        return value * value;
    }
}
