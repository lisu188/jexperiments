package com.lis.neuro;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

public final class Neuro {
    private static final long DEFAULT_SEED = 0x5EEDL;

    public record HyperParameters(
            double learningRate,
            double momentum,
            double beta,
            long seed) {
        public HyperParameters {
            if (!(learningRate > 0.0) || !Double.isFinite(learningRate)) {
                throw new IllegalArgumentException("learningRate must be finite and > 0");
            }
            if (momentum < 0.0 || momentum >= 1.0 || !Double.isFinite(momentum)) {
                throw new IllegalArgumentException("momentum must be finite and in [0, 1)");
            }
            if (!(beta > 0.0) || !Double.isFinite(beta)) {
                throw new IllegalArgumentException("beta must be finite and > 0");
            }
        }

        public static HyperParameters defaults() {
            return new HyperParameters(0.5, 0.2, 1.0, DEFAULT_SEED);
        }

        public HyperParameters withLearningRate(double value) {
            return new HyperParameters(value, momentum, beta, seed);
        }

        public HyperParameters withMomentum(double value) {
            return new HyperParameters(learningRate, value, beta, seed);
        }

        public HyperParameters withBeta(double value) {
            return new HyperParameters(learningRate, momentum, value, seed);
        }

        public HyperParameters withSeed(long value) {
            return new HyperParameters(learningRate, momentum, beta, value);
        }
    }

    public record TrainingResult(
            int epochs,
            double error,
            boolean converged) {
    }

    public record Statistics(
            long epochsTrained,
            long samplesSeen,
            double lastTrainingError) {
    }

    private static final class Sample {
        private final double[] input;
        private final double[] target;

        private Sample(double[] input, double[] target) {
            this.input = input.clone();
            this.target = target.clone();
        }
    }

    private static final class Layer {
        private final int inputs;
        private final int outputs;
        private final double[] weights;
        private final double[] biases;
        private final double[] weightVelocity;
        private final double[] biasVelocity;

        private Layer(int inputs, int outputs, SplittableRandom random) {
            this.inputs = inputs;
            this.outputs = outputs;
            weights = new double[inputs * outputs];
            biases = new double[outputs];
            weightVelocity = new double[weights.length];
            biasVelocity = new double[outputs];

            var limit = Math.sqrt(6.0 / (inputs + outputs));
            for (int index = 0; index < weights.length; index++) {
                weights[index] = random.nextDouble(-limit, limit);
            }
        }
    }

    private static final class Workspace {
        private final double[][] activations;
        private final double[][] deltas;

        private Workspace(int[] topology) {
            activations = new double[topology.length][];
            for (int layer = 0; layer < topology.length; layer++) {
                activations[layer] = new double[topology[layer]];
            }

            deltas = new double[topology.length - 1][];
            for (int layer = 0; layer < topology.length - 1; layer++) {
                deltas[layer] = new double[topology[layer + 1]];
            }
        }
    }

    private final int[] topology;
    private final HyperParameters hyperParameters;
    private final Layer[] layers;
    private final List<Sample> trainingSamples = new ArrayList<>();
    private final List<Sample> testSamples = new ArrayList<>();
    private final Workspace trainingWorkspace;
    private final ThreadLocal<Workspace> inferenceWorkspace;
    private final SplittableRandom shuffleRandom;

    private int[] trainingOrder = new int[0];
    private long epochsTrained;
    private long samplesSeen;
    private double lastTrainingError = Double.NaN;

    public Neuro(int[] topology) {
        this(topology, HyperParameters.defaults());
    }

    public Neuro(int[] topology, double momentum, double beta, double learningRate) {
        this(topology, new HyperParameters(learningRate, momentum, beta, DEFAULT_SEED));
    }

    public Neuro(int[] topology, HyperParameters hyperParameters) {
        this.topology = validateTopology(topology);
        this.hyperParameters = Objects.requireNonNull(hyperParameters, "hyperParameters");

        var initializationRandom = new SplittableRandom(hyperParameters.seed());
        layers = new Layer[this.topology.length - 1];
        for (int layer = 0; layer < layers.length; layer++) {
            layers[layer] = new Layer(
                    this.topology[layer],
                    this.topology[layer + 1],
                    initializationRandom);
        }

        trainingWorkspace = new Workspace(this.topology);
        inferenceWorkspace = ThreadLocal.withInitial(() -> new Workspace(this.topology));
        shuffleRandom = new SplittableRandom(hyperParameters.seed() ^ 0x9E3779B97F4A7C15L);
    }

    public int[] topology() {
        return topology.clone();
    }

    public HyperParameters hyperParameters() {
        return hyperParameters;
    }

    public int parameterCount() {
        var count = 0;
        for (var layer : layers) {
            count += layer.weights.length + layer.biases.length;
        }
        return count;
    }

    public int trainingSampleCount() {
        return trainingSamples.size();
    }

    public int testSampleCount() {
        return testSamples.size();
    }

    public Statistics statistics() {
        return new Statistics(epochsTrained, samplesSeen, lastTrainingError);
    }

    public Neuro addTrainingSample(double[] input, double[] target) {
        validateSample(input, target);
        trainingSamples.add(new Sample(input, target));
        return this;
    }

    public Neuro addTestSample(double[] input, double[] target) {
        validateSample(input, target);
        testSamples.add(new Sample(input, target));
        return this;
    }

    public double[] predict(double[] input) {
        var result = new double[topology[topology.length - 1]];
        predictInto(input, result);
        return result;
    }

    public void predictInto(double[] input, double[] output) {
        Objects.requireNonNull(output, "output");
        if (output.length != topology[topology.length - 1]) {
            throw new IllegalArgumentException(
                    "output length " + output.length + " != expected " + topology[topology.length - 1]);
        }
        var workspace = inferenceWorkspace.get();
        forward(input, workspace);
        System.arraycopy(workspace.activations[topology.length - 1], 0, output, 0, output.length);
    }

    public double trainEpoch() {
        requireTrainingSamples();
        ensureTrainingOrder();
        shuffleTrainingOrder();

        var event = NeuroJfr.trainingEpoch(epochsTrained + 1, trainingSamples.size());
        for (var sampleIndex : trainingOrder) {
            var sample = trainingSamples.get(sampleIndex);
            forward(sample.input, trainingWorkspace);
            backpropagate(sample.target, trainingWorkspace);
            applyGradient(trainingWorkspace);
            samplesSeen++;
        }

        epochsTrained++;
        lastTrainingError = error(trainingSamples, trainingWorkspace);
        NeuroJfr.commitTrainingEpoch(event, lastTrainingError);
        return lastTrainingError;
    }

    public void train(int epochs) {
        if (epochs < 0) {
            throw new IllegalArgumentException("epochs must be >= 0");
        }
        for (int epoch = 0; epoch < epochs; epoch++) {
            trainEpoch();
        }
    }

    public TrainingResult trainUntil(double targetError, int maxEpochs) {
        if (!(targetError >= 0.0) || !Double.isFinite(targetError)) {
            throw new IllegalArgumentException("targetError must be finite and >= 0");
        }
        if (maxEpochs < 0) {
            throw new IllegalArgumentException("maxEpochs must be >= 0");
        }
        requireTrainingSamples();

        var initialError = trainingError();
        if (initialError <= targetError) {
            return new TrainingResult(0, initialError, true);
        }

        var event = NeuroJfr.trainingRun(targetError, maxEpochs);
        var error = initialError;
        var epochs = 0;
        while (epochs < maxEpochs && error > targetError) {
            error = trainEpoch();
            epochs++;
        }

        var converged = error <= targetError;
        NeuroJfr.commitTrainingRun(event, epochs, error, converged);
        return new TrainingResult(epochs, error, converged);
    }

    public double trainingError() {
        return error(trainingSamples, trainingWorkspace);
    }

    public double testError() {
        return error(testSamples, trainingWorkspace);
    }

    private void forward(double[] input, Workspace workspace) {
        Objects.requireNonNull(input, "input");
        if (input.length != topology[0]) {
            throw new IllegalArgumentException(
                    "input length " + input.length + " != expected " + topology[0]);
        }

        System.arraycopy(input, 0, workspace.activations[0], 0, input.length);
        for (int layerIndex = 0; layerIndex < layers.length; layerIndex++) {
            var layer = layers[layerIndex];
            var source = workspace.activations[layerIndex];
            var destination = workspace.activations[layerIndex + 1];

            for (int output = 0; output < layer.outputs; output++) {
                var offset = output * layer.inputs;
                var sum = layer.biases[output];
                var inputIndex = 0;
                var unrolledLimit = layer.inputs - (layer.inputs & 3);

                for (; inputIndex < unrolledLimit; inputIndex += 4) {
                    sum = Math.fma(source[inputIndex], layer.weights[offset + inputIndex], sum);
                    sum = Math.fma(source[inputIndex + 1], layer.weights[offset + inputIndex + 1], sum);
                    sum = Math.fma(source[inputIndex + 2], layer.weights[offset + inputIndex + 2], sum);
                    sum = Math.fma(source[inputIndex + 3], layer.weights[offset + inputIndex + 3], sum);
                }
                for (; inputIndex < layer.inputs; inputIndex++) {
                    sum = Math.fma(source[inputIndex], layer.weights[offset + inputIndex], sum);
                }
                destination[output] = sigmoid(sum * hyperParameters.beta());
            }
        }
    }

    private void backpropagate(double[] target, Workspace workspace) {
        var lastLayerIndex = layers.length - 1;
        var outputActivation = workspace.activations[topology.length - 1];
        var outputDelta = workspace.deltas[lastLayerIndex];

        for (int output = 0; output < outputDelta.length; output++) {
            var activation = outputActivation[output];
            outputDelta[output] =
                    (target[output] - activation) * sigmoidDerivativeFromActivation(activation);
        }

        for (int layerIndex = lastLayerIndex - 1; layerIndex >= 0; layerIndex--) {
            var currentDelta = workspace.deltas[layerIndex];
            var currentActivation = workspace.activations[layerIndex + 1];
            var nextLayer = layers[layerIndex + 1];
            var nextDelta = workspace.deltas[layerIndex + 1];

            Arrays.fill(currentDelta, 0.0);
            for (int nextOutput = 0; nextOutput < nextLayer.outputs; nextOutput++) {
                var weightedDelta = nextDelta[nextOutput];
                var offset = nextOutput * nextLayer.inputs;
                for (int current = 0; current < nextLayer.inputs; current++) {
                    currentDelta[current] = Math.fma(
                            weightedDelta,
                            nextLayer.weights[offset + current],
                            currentDelta[current]);
                }
            }

            for (int current = 0; current < currentDelta.length; current++) {
                currentDelta[current] *= sigmoidDerivativeFromActivation(currentActivation[current]);
            }
        }
    }

    private void applyGradient(Workspace workspace) {
        var learningRate = hyperParameters.learningRate();
        var momentum = hyperParameters.momentum();

        for (int layerIndex = 0; layerIndex < layers.length; layerIndex++) {
            var layer = layers[layerIndex];
            var source = workspace.activations[layerIndex];
            var delta = workspace.deltas[layerIndex];

            for (int output = 0; output < layer.outputs; output++) {
                var offset = output * layer.inputs;
                var outputDelta = delta[output];

                for (int input = 0; input < layer.inputs; input++) {
                    var weightIndex = offset + input;
                    var velocity = Math.fma(
                            momentum,
                            layer.weightVelocity[weightIndex],
                            learningRate * outputDelta * source[input]);
                    layer.weightVelocity[weightIndex] = velocity;
                    layer.weights[weightIndex] += velocity;
                }

                var biasVelocity = Math.fma(
                        momentum,
                        layer.biasVelocity[output],
                        learningRate * outputDelta);
                layer.biasVelocity[output] = biasVelocity;
                layer.biases[output] += biasVelocity;
            }
        }
    }

    private double error(List<Sample> samples, Workspace workspace) {
        if (samples.isEmpty()) {
            return Double.NaN;
        }

        var squaredError = 0.0;
        var outputCount = topology[topology.length - 1];
        for (var sample : samples) {
            forward(sample.input, workspace);
            var output = workspace.activations[topology.length - 1];
            for (int index = 0; index < outputCount; index++) {
                var difference = sample.target[index] - output[index];
                squaredError = Math.fma(difference, difference, squaredError);
            }
        }
        return Math.sqrt(squaredError / ((double) samples.size() * outputCount));
    }

    private double sigmoidDerivativeFromActivation(double activation) {
        return hyperParameters.beta() * activation * (1.0 - activation);
    }

    private static double sigmoid(double value) {
        if (value >= 0.0) {
            return 1.0 / (1.0 + Math.exp(-value));
        }
        var exp = Math.exp(value);
        return exp / (1.0 + exp);
    }

    private void ensureTrainingOrder() {
        if (trainingOrder.length == trainingSamples.size()) {
            return;
        }
        trainingOrder = new int[trainingSamples.size()];
        for (int index = 0; index < trainingOrder.length; index++) {
            trainingOrder[index] = index;
        }
    }

    private void shuffleTrainingOrder() {
        for (int index = trainingOrder.length - 1; index > 0; index--) {
            var other = shuffleRandom.nextInt(index + 1);
            var value = trainingOrder[index];
            trainingOrder[index] = trainingOrder[other];
            trainingOrder[other] = value;
        }
    }

    private void validateSample(double[] input, double[] target) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(target, "target");
        if (input.length != topology[0]) {
            throw new IllegalArgumentException(
                    "input length " + input.length + " != expected " + topology[0]);
        }
        if (target.length != topology[topology.length - 1]) {
            throw new IllegalArgumentException(
                    "target length " + target.length + " != expected " + topology[topology.length - 1]);
        }
        requireFinite(input, "input");
        requireFinite(target, "target");
    }

    private void requireTrainingSamples() {
        if (trainingSamples.isEmpty()) {
            throw new IllegalStateException("no training samples");
        }
    }

    private static int[] validateTopology(int[] topology) {
        Objects.requireNonNull(topology, "topology");
        if (topology.length < 2) {
            throw new IllegalArgumentException("topology must contain at least input and output layers");
        }
        var copy = topology.clone();
        for (var size : copy) {
            if (size <= 0) {
                throw new IllegalArgumentException("all layer sizes must be positive");
            }
        }
        return copy;
    }

    private static void requireFinite(double[] values, String name) {
        for (var value : values) {
            if (!Double.isFinite(value)) {
                throw new IllegalArgumentException(name + " contains a non-finite value");
            }
        }
    }
}
