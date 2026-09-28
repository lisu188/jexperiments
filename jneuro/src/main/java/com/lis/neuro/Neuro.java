package com.lis.neuro;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.Future;

public final class Neuro {
    private static final long DEFAULT_SEED = 0x5EEDL;

    public enum Kernel {
        AUTO,
        SCALAR,
        VECTOR
    }

    public enum SigmoidMode {
        EXACT,
        FAST
    }

    public record HyperParameters(
            double learningRate,
            double momentum,
            double beta,
            long seed,
            Kernel kernel,
            SigmoidMode sigmoidMode) {
        public HyperParameters(double learningRate, double momentum, double beta, long seed) {
            this(learningRate, momentum, beta, seed, Kernel.AUTO, SigmoidMode.EXACT);
        }

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
            Objects.requireNonNull(kernel, "kernel");
            Objects.requireNonNull(sigmoidMode, "sigmoidMode");
        }

        public static HyperParameters defaults() {
            return new HyperParameters(0.5, 0.2, 1.0, DEFAULT_SEED);
        }

        public HyperParameters withLearningRate(double value) {
            return new HyperParameters(value, momentum, beta, seed, kernel, sigmoidMode);
        }

        public HyperParameters withMomentum(double value) {
            return new HyperParameters(learningRate, value, beta, seed, kernel, sigmoidMode);
        }

        public HyperParameters withBeta(double value) {
            return new HyperParameters(learningRate, momentum, value, seed, kernel, sigmoidMode);
        }

        public HyperParameters withSeed(long value) {
            return new HyperParameters(learningRate, momentum, beta, value, kernel, sigmoidMode);
        }

        public HyperParameters withKernel(Kernel value) {
            return new HyperParameters(learningRate, momentum, beta, seed, value, sigmoidMode);
        }

        public HyperParameters withSigmoidMode(SigmoidMode value) {
            return new HyperParameters(learningRate, momentum, beta, seed, kernel, value);
        }
    }

    public record TrainingResult(int epochs, double error, boolean converged) {
    }

    public record Statistics(long epochsTrained, long samplesSeen, double lastTrainingError) {
    }

    private static final class Sample {
        private final double[] input;
        private final double[] target;

        private Sample(double[] input, double[] target) {
            this.input = input.clone();
            this.target = target.clone();
        }
    }

    private static final class PackedDataset {
        private final int size;
        private final int inputSize;
        private final int outputSize;
        private final double[] inputs;
        private final double[] targets;

        private PackedDataset(List<Sample> samples, int inputSize, int outputSize) {
            size = samples.size();
            this.inputSize = inputSize;
            this.outputSize = outputSize;
            inputs = new double[size * inputSize];
            targets = new double[size * outputSize];
            for (int sample = 0; sample < size; sample++) {
                var value = samples.get(sample);
                System.arraycopy(value.input, 0, inputs, sample * inputSize, inputSize);
                System.arraycopy(value.target, 0, targets, sample * outputSize, outputSize);
            }
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
            for (int layer = 1; layer < topology.length; layer++) {
                activations[layer] = new double[topology[layer]];
            }

            deltas = new double[topology.length - 1][];
            for (int layer = 0; layer < topology.length - 1; layer++) {
                deltas[layer] = new double[topology[layer + 1]];
            }
        }
    }

    private static final class BatchWorkspace {
        private final int[] topology;
        private final double[][] activations;
        private int capacity;

        private BatchWorkspace(int[] topology) {
            this.topology = topology;
            activations = new double[topology.length][];
        }

        private void ensureCapacity(int batchSize) {
            if (batchSize <= capacity) {
                return;
            }
            capacity = Math.max(batchSize, Math.max(8, capacity * 2));
            for (int layer = 1; layer < topology.length - 1; layer++) {
                activations[layer] = new double[capacity * topology[layer]];
            }
        }
    }

    private static final class GradientBuffer {
        private final double[][] weights;
        private final double[][] biases;

        private GradientBuffer(Layer[] layers) {
            weights = new double[layers.length][];
            biases = new double[layers.length][];
            for (int layer = 0; layer < layers.length; layer++) {
                weights[layer] = new double[layers[layer].weights.length];
                biases[layer] = new double[layers[layer].biases.length];
            }
        }

        private void clear() {
            for (var values : weights) {
                Arrays.fill(values, 0.0);
            }
            for (var values : biases) {
                Arrays.fill(values, 0.0);
            }
        }
    }

    private static final class WorkerState {
        private final Workspace workspace;
        private final GradientBuffer gradient;

        private WorkerState(int[] topology, Layer[] layers) {
            workspace = new Workspace(topology);
            gradient = new GradientBuffer(layers);
        }
    }

    public final class InferenceSession {
        private final Workspace workspace = new Workspace(topology);
        private final BatchWorkspace batchWorkspace = new BatchWorkspace(topology);

        private InferenceSession() {
        }

        public void predictInto(double[] input, double[] output) {
            validatePrediction(input, output);
            forward(input, 0, workspace, output, 0);
        }

        public void predictBatch(double[] inputs, int batchSize, double[] outputs) {
            validateBatch(inputs, batchSize, outputs);
            batchWorkspace.ensureCapacity(batchSize);
            forwardBatch(inputs, batchSize, outputs, batchWorkspace);
        }
    }

    public final class ParallelInferenceSession implements AutoCloseable {
        private final int parallelism;
        private final ForkJoinPool pool;
        private final InferenceSession[] sessions;

        private ParallelInferenceSession(int parallelism) {
            if (parallelism <= 0) {
                throw new IllegalArgumentException("parallelism must be > 0");
            }
            this.parallelism = parallelism;
            pool = new ForkJoinPool(parallelism);
            sessions = new InferenceSession[parallelism];
            for (int i = 0; i < parallelism; i++) {
                sessions[i] = newInferenceSession();
            }
        }

        public void predictBatch(double[] inputs, int batchSize, double[] outputs) {
            validateBatch(inputs, batchSize, outputs);
            if (batchSize == 0) {
                return;
            }
            var workerCount = Math.min(parallelism, batchSize);
            var inputSize = topology[0];
            var outputSize = topology[topology.length - 1];
            Future<?>[] futures = new Future<?>[workerCount];
            for (int worker = 0; worker < workerCount; worker++) {
                var start = worker * batchSize / workerCount;
                var end = (worker + 1) * batchSize / workerCount;
                var session = sessions[worker];
                futures[worker] = pool.submit(() -> {
                    for (int sample = start; sample < end; sample++) {
                        forward(
                                inputs,
                                sample * inputSize,
                                session.workspace,
                                outputs,
                                sample * outputSize);
                    }
                });
            }
            await(futures);
        }

        @Override
        public void close() {
            pool.shutdown();
        }
    }

    public static final class FloatModel {
        private final int[] topology;
        private final float[][] weights;
        private final float[][] biases;
        private final Kernel kernel;
        private final SigmoidMode sigmoidMode;
        private final double beta;
        private final ThreadLocal<float[][]> workspace;

        private FloatModel(Neuro source) {
            topology = source.topology.clone();
            kernel = source.hyperParameters.kernel();
            sigmoidMode = source.hyperParameters.sigmoidMode();
            beta = source.beta;
            weights = new float[source.layers.length][];
            biases = new float[source.layers.length][];
            for (int layer = 0; layer < source.layers.length; layer++) {
                var original = source.layers[layer];
                weights[layer] = new float[original.weights.length];
                biases[layer] = new float[original.biases.length];
                for (int i = 0; i < original.weights.length; i++) {
                    weights[layer][i] = (float) original.weights[i];
                }
                for (int i = 0; i < original.biases.length; i++) {
                    biases[layer][i] = (float) original.biases[i];
                }
            }
            workspace = ThreadLocal.withInitial(() -> {
                var result = new float[topology.length][];
                for (int layer = 1; layer < topology.length; layer++) {
                    result[layer] = new float[topology[layer]];
                }
                return result;
            });
        }

        public int[] topology() {
            return topology.clone();
        }

        public float[] predict(float[] input) {
            var output = new float[topology[topology.length - 1]];
            predictInto(input, output);
            return output;
        }

        public void predictInto(float[] input, float[] output) {
            Objects.requireNonNull(input, "input");
            Objects.requireNonNull(output, "output");
            if (input.length != topology[0]) {
                throw new IllegalArgumentException("input length " + input.length + " != expected " + topology[0]);
            }
            if (output.length != topology[topology.length - 1]) {
                throw new IllegalArgumentException(
                        "output length " + output.length + " != expected " + topology[topology.length - 1]);
            }
            forward(input, 0, output, 0, workspace.get());
        }

        public void predictBatch(float[] inputs, int batchSize, float[] outputs) {
            Objects.requireNonNull(inputs, "inputs");
            Objects.requireNonNull(outputs, "outputs");
            if (batchSize < 0) {
                throw new IllegalArgumentException("batchSize must be >= 0");
            }
            var inputSize = topology[0];
            var outputSize = topology[topology.length - 1];
            if (inputs.length < batchSize * inputSize || outputs.length < batchSize * outputSize) {
                throw new IllegalArgumentException("batch arrays are too small");
            }
            var activations = workspace.get();
            for (int sample = 0; sample < batchSize; sample++) {
                forward(inputs, sample * inputSize, outputs, sample * outputSize, activations);
            }
        }

        private void forward(float[] input, int inputOffset, float[] output, int outputOffset, float[][] activations) {
            for (int layerIndex = 0; layerIndex < weights.length; layerIndex++) {
                var inputs = topology[layerIndex];
                var outputs = topology[layerIndex + 1];
                var sourceArray = layerIndex == 0 ? input : activations[layerIndex];
                var sourceOffset = layerIndex == 0 ? inputOffset : 0;
                var destination = layerIndex == weights.length - 1 ? output : activations[layerIndex + 1];
                var destinationOffset = layerIndex == weights.length - 1 ? outputOffset : 0;
                var vector = useFloatVector(kernel, inputs);
                for (int out = 0; out < outputs; out++) {
                    var offset = out * inputs;
                    var sum = biases[layerIndex][out];
                    if (vector) {
                        sum += NeuroVectorOps.dot(sourceArray, sourceOffset, weights[layerIndex], offset, inputs);
                    } else {
                        for (int in = 0; in < inputs; in++) {
                            sum = Math.fma(sourceArray[sourceOffset + in], weights[layerIndex][offset + in], sum);
                        }
                    }
                    destination[destinationOffset + out] = (float) activate(sum * beta, sigmoidMode);
                }
            }
        }
    }

    private final int[] topology;
    private final HyperParameters hyperParameters;
    private final Layer[] layers;
    private final List<Sample> trainingSamples = new ArrayList<>();
    private final List<Sample> testSamples = new ArrayList<>();
    private final Workspace trainingWorkspace;
    private final ThreadLocal<InferenceSession> inferenceSession;
    private final SplittableRandom shuffleRandom;
    private final GradientBuffer batchGradient;
    private final double learningRate;
    private final double momentum;
    private final double beta;

    private PackedDataset packedTraining;
    private PackedDataset packedTests;
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
        learningRate = hyperParameters.learningRate();
        momentum = hyperParameters.momentum();
        beta = hyperParameters.beta();

        var initializationRandom = new SplittableRandom(hyperParameters.seed());
        layers = new Layer[this.topology.length - 1];
        for (int layer = 0; layer < layers.length; layer++) {
            layers[layer] = new Layer(this.topology[layer], this.topology[layer + 1], initializationRandom);
        }

        trainingWorkspace = new Workspace(this.topology);
        inferenceSession = ThreadLocal.withInitial(this::newInferenceSession);
        shuffleRandom = new SplittableRandom(hyperParameters.seed() ^ 0x9E3779B97F4A7C15L);
        batchGradient = new GradientBuffer(layers);
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
        packedTraining = null;
        return this;
    }

    public Neuro addTestSample(double[] input, double[] target) {
        validateSample(input, target);
        testSamples.add(new Sample(input, target));
        packedTests = null;
        return this;
    }

    public double[] predict(double[] input) {
        var result = new double[topology[topology.length - 1]];
        predictInto(input, result);
        return result;
    }

    public void predictInto(double[] input, double[] output) {
        inferenceSession.get().predictInto(input, output);
    }

    public InferenceSession newInferenceSession() {
        return new InferenceSession();
    }

    public void predictBatch(double[] inputs, int batchSize, double[] outputs) {
        inferenceSession.get().predictBatch(inputs, batchSize, outputs);
    }

    public void predictBatchParallel(double[] inputs, int batchSize, double[] outputs, int parallelism) {
        try (var session = newParallelInferenceSession(parallelism)) {
            session.predictBatch(inputs, batchSize, outputs);
        }
    }

    public ParallelInferenceSession newParallelInferenceSession(int parallelism) {
        return new ParallelInferenceSession(parallelism);
    }

    public FloatModel toFloatModel() {
        return new FloatModel(this);
    }

    public double trainEpoch() {
        requireTrainingSamples();
        return trainOnlineEpoch(trainingData(), true);
    }

    public void train(int epochs) {
        if (epochs < 0) {
            throw new IllegalArgumentException("epochs must be >= 0");
        }
        if (epochs == 0) {
            return;
        }
        requireTrainingSamples();
        var data = trainingData();
        for (int epoch = 0; epoch < epochs - 1; epoch++) {
            trainOnlineEpoch(data, false);
        }
        trainOnlineEpoch(data, true);
    }

    public TrainingResult trainUntil(double targetError, int maxEpochs) {
        return trainUntil(targetError, maxEpochs, 1);
    }

    public TrainingResult trainUntil(double targetError, int maxEpochs, int checkEvery) {
        if (!(targetError >= 0.0) || !Double.isFinite(targetError)) {
            throw new IllegalArgumentException("targetError must be finite and >= 0");
        }
        if (maxEpochs < 0) {
            throw new IllegalArgumentException("maxEpochs must be >= 0");
        }
        if (checkEvery <= 0) {
            throw new IllegalArgumentException("checkEvery must be > 0");
        }
        requireTrainingSamples();

        var data = trainingData();
        var error = error(data, trainingWorkspace);
        if (error <= targetError) {
            lastTrainingError = error;
            return new TrainingResult(0, error, true);
        }

        var event = NeuroJfr.trainingRun(targetError, maxEpochs);
        var epochs = 0;
        while (epochs < maxEpochs && error > targetError) {
            trainOnlineEpoch(data, false);
            epochs++;
            if (epochs % checkEvery == 0 || epochs == maxEpochs) {
                error = error(data, trainingWorkspace);
                lastTrainingError = error;
            }
        }

        var converged = error <= targetError;
        NeuroJfr.commitTrainingRun(event, epochs, error, converged);
        return new TrainingResult(epochs, error, converged);
    }

    public void trainMiniBatch(int epochs, int batchSize) {
        trainMiniBatch(epochs, batchSize, 1);
    }

    public void trainMiniBatch(int epochs, int batchSize, int parallelism) {
        if (epochs < 0) {
            throw new IllegalArgumentException("epochs must be >= 0");
        }
        if (batchSize <= 0) {
            throw new IllegalArgumentException("batchSize must be > 0");
        }
        if (parallelism <= 0) {
            throw new IllegalArgumentException("parallelism must be > 0");
        }
        if (epochs == 0) {
            return;
        }
        requireTrainingSamples();
        var data = trainingData();
        ForkJoinPool pool = parallelism > 1 ? new ForkJoinPool(parallelism) : null;
        WorkerState[] workers = parallelism > 1 ? createWorkers(parallelism) : null;
        try {
            for (int epoch = 0; epoch < epochs; epoch++) {
                trainMiniBatchEpoch(data, batchSize, parallelism, pool, workers);
            }
        } finally {
            if (pool != null) {
                pool.shutdown();
            }
        }
        lastTrainingError = error(data, trainingWorkspace);
    }

    public double trainingError() {
        return trainingSamples.isEmpty() ? Double.NaN : error(trainingData(), trainingWorkspace);
    }

    public double testError() {
        return testSamples.isEmpty() ? Double.NaN : error(testData(), trainingWorkspace);
    }

    private double trainOnlineEpoch(PackedDataset data, boolean evaluateError) {
        ensureTrainingOrder(data.size);
        shuffleTrainingOrder();

        var event = NeuroJfr.trainingEpoch(epochsTrained + 1, data.size);
        for (var sampleIndex : trainingOrder) {
            var inputOffset = sampleIndex * data.inputSize;
            var targetOffset = sampleIndex * data.outputSize;
            forward(data.inputs, inputOffset, trainingWorkspace, null, 0);
            backpropagate(data.targets, targetOffset, trainingWorkspace);
            applyOnlineGradient(data.inputs, inputOffset, trainingWorkspace);
        }

        samplesSeen += data.size;
        epochsTrained++;
        var error = evaluateError ? error(data, trainingWorkspace) : Double.NaN;
        if (evaluateError) {
            lastTrainingError = error;
        }
        NeuroJfr.commitTrainingEpoch(event, error);
        return error;
    }

    private void trainMiniBatchEpoch(
            PackedDataset data,
            int batchSize,
            int parallelism,
            ForkJoinPool pool,
            WorkerState[] workers) {
        ensureTrainingOrder(data.size);
        shuffleTrainingOrder();
        for (int start = 0; start < data.size; start += batchSize) {
            var end = Math.min(data.size, start + batchSize);
            var count = end - start;
            batchGradient.clear();
            if (parallelism == 1 || count == 1) {
                for (int position = start; position < end; position++) {
                    accumulateSampleGradient(data, trainingOrder[position], trainingWorkspace, batchGradient);
                }
            } else {
                var activeWorkers = Math.min(parallelism, count);
                @SuppressWarnings("unchecked")
                Future<?>[] futures = new Future<?>[activeWorkers];
                for (int worker = 0; worker < activeWorkers; worker++) {
                    var state = workers[worker];
                    state.gradient.clear();
                    var workerStart = start + worker * count / activeWorkers;
                    var workerEnd = start + (worker + 1) * count / activeWorkers;
                    futures[worker] = pool.submit(() -> {
                        for (int position = workerStart; position < workerEnd; position++) {
                            accumulateSampleGradient(
                                    data,
                                    trainingOrder[position],
                                    state.workspace,
                                    state.gradient);
                        }
                        return null;
                    });
                }
                await(futures);
                for (int worker = 0; worker < activeWorkers; worker++) {
                    addGradient(batchGradient, workers[worker].gradient);
                }
            }
            applyBatchGradient(batchGradient, count);
        }
        samplesSeen += data.size;
        epochsTrained++;
    }

    private WorkerState[] createWorkers(int count) {
        var result = new WorkerState[count];
        for (int i = 0; i < count; i++) {
            result[i] = new WorkerState(topology, layers);
        }
        return result;
    }

    private void accumulateSampleGradient(
            PackedDataset data,
            int sampleIndex,
            Workspace workspace,
            GradientBuffer gradient) {
        var inputOffset = sampleIndex * data.inputSize;
        var targetOffset = sampleIndex * data.outputSize;
        forward(data.inputs, inputOffset, workspace, null, 0);
        backpropagate(data.targets, targetOffset, workspace);
        for (int layerIndex = 0; layerIndex < layers.length; layerIndex++) {
            var layer = layers[layerIndex];
            var source = layerIndex == 0 ? data.inputs : workspace.activations[layerIndex];
            var sourceOffset = layerIndex == 0 ? inputOffset : 0;
            var delta = workspace.deltas[layerIndex];
            for (int output = 0; output < layer.outputs; output++) {
                var offset = output * layer.inputs;
                var scale = delta[output];
                if (useVector(layer.inputs)) {
                    NeuroVectorOps.addOuterProduct(
                            gradient.weights[layerIndex],
                            offset,
                            source,
                            sourceOffset,
                            layer.inputs,
                            scale);
                } else {
                    for (int input = 0; input < layer.inputs; input++) {
                        gradient.weights[layerIndex][offset + input] = Math.fma(
                                source[sourceOffset + input],
                                scale,
                                gradient.weights[layerIndex][offset + input]);
                    }
                }
                gradient.biases[layerIndex][output] += scale;
            }
        }
    }

    private void addGradient(GradientBuffer destination, GradientBuffer source) {
        for (int layer = 0; layer < layers.length; layer++) {
            if (useVector(destination.weights[layer].length)) {
                NeuroVectorOps.add(destination.weights[layer], source.weights[layer], destination.weights[layer].length);
            } else {
                for (int i = 0; i < destination.weights[layer].length; i++) {
                    destination.weights[layer][i] += source.weights[layer][i];
                }
            }
            for (int i = 0; i < destination.biases[layer].length; i++) {
                destination.biases[layer][i] += source.biases[layer][i];
            }
        }
    }

    private void applyBatchGradient(GradientBuffer gradient, int sampleCount) {
        var gradientScale = learningRate / sampleCount;
        for (int layerIndex = 0; layerIndex < layers.length; layerIndex++) {
            var layer = layers[layerIndex];
            for (int output = 0; output < layer.outputs; output++) {
                var offset = output * layer.inputs;
                if (useVector(layer.inputs)) {
                    NeuroVectorOps.update(
                            layer.weights,
                            layer.weightVelocity,
                            offset,
                            gradient.weights[layerIndex],
                            offset,
                            layer.inputs,
                            momentum,
                            gradientScale);
                } else {
                    for (int input = 0; input < layer.inputs; input++) {
                        var index = offset + input;
                        var velocity = Math.fma(
                                momentum,
                                layer.weightVelocity[index],
                                gradientScale * gradient.weights[layerIndex][index]);
                        layer.weightVelocity[index] = velocity;
                        layer.weights[index] += velocity;
                    }
                }
                var biasVelocity = Math.fma(
                        momentum,
                        layer.biasVelocity[output],
                        gradientScale * gradient.biases[layerIndex][output]);
                layer.biasVelocity[output] = biasVelocity;
                layer.biases[output] += biasVelocity;
            }
        }
    }

    private void forward(
            double[] input,
            int inputOffset,
            Workspace workspace,
            double[] externalOutput,
            int externalOutputOffset) {
        for (int layerIndex = 0; layerIndex < layers.length; layerIndex++) {
            var layer = layers[layerIndex];
            var source = layerIndex == 0 ? input : workspace.activations[layerIndex];
            var sourceOffset = layerIndex == 0 ? inputOffset : 0;
            var finalLayer = layerIndex == layers.length - 1;
            var destination = finalLayer && externalOutput != null
                    ? externalOutput
                    : workspace.activations[layerIndex + 1];
            var destinationOffset = finalLayer && externalOutput != null ? externalOutputOffset : 0;
            var vector = useVector(layer.inputs);

            for (int output = 0; output < layer.outputs; output++) {
                var weightOffset = output * layer.inputs;
                var sum = layer.biases[output];
                if (vector) {
                    sum += NeuroVectorOps.dot(source, sourceOffset, layer.weights, weightOffset, layer.inputs);
                } else {
                    var inputIndex = 0;
                    var unrolledLimit = layer.inputs - (layer.inputs & 3);
                    for (; inputIndex < unrolledLimit; inputIndex += 4) {
                        sum = Math.fma(source[sourceOffset + inputIndex], layer.weights[weightOffset + inputIndex], sum);
                        sum = Math.fma(
                                source[sourceOffset + inputIndex + 1],
                                layer.weights[weightOffset + inputIndex + 1],
                                sum);
                        sum = Math.fma(
                                source[sourceOffset + inputIndex + 2],
                                layer.weights[weightOffset + inputIndex + 2],
                                sum);
                        sum = Math.fma(
                                source[sourceOffset + inputIndex + 3],
                                layer.weights[weightOffset + inputIndex + 3],
                                sum);
                    }
                    for (; inputIndex < layer.inputs; inputIndex++) {
                        sum = Math.fma(source[sourceOffset + inputIndex], layer.weights[weightOffset + inputIndex], sum);
                    }
                }
                destination[destinationOffset + output] = activate(sum * beta, hyperParameters.sigmoidMode());
            }
        }
    }

    private void forwardBatch(double[] inputs, int batchSize, double[] outputs, BatchWorkspace workspace) {
        for (int layerIndex = 0; layerIndex < layers.length; layerIndex++) {
            var layer = layers[layerIndex];
            var source = layerIndex == 0 ? inputs : workspace.activations[layerIndex];
            var destination = layerIndex == layers.length - 1
                    ? outputs
                    : workspace.activations[layerIndex + 1];
            var vector = useVector(layer.inputs);
            for (int output = 0; output < layer.outputs; output++) {
                var weightOffset = output * layer.inputs;
                for (int sample = 0; sample < batchSize; sample++) {
                    var sourceOffset = sample * layer.inputs;
                    var sum = layer.biases[output];
                    if (vector) {
                        sum += NeuroVectorOps.dot(source, sourceOffset, layer.weights, weightOffset, layer.inputs);
                    } else {
                        for (int input = 0; input < layer.inputs; input++) {
                            sum = Math.fma(
                                    source[sourceOffset + input],
                                    layer.weights[weightOffset + input],
                                    sum);
                        }
                    }
                    destination[sample * layer.outputs + output] = activate(
                            sum * beta,
                            hyperParameters.sigmoidMode());
                }
            }
        }
    }

    private void backpropagate(double[] target, int targetOffset, Workspace workspace) {
        var lastLayerIndex = layers.length - 1;
        var outputActivation = workspace.activations[topology.length - 1];
        var outputDelta = workspace.deltas[lastLayerIndex];

        if (useVector(outputDelta.length)) {
            NeuroVectorOps.outputDelta(target, targetOffset, outputActivation, outputDelta, outputDelta.length, beta);
        } else {
            for (int output = 0; output < outputDelta.length; output++) {
                var activation = outputActivation[output];
                outputDelta[output] = (target[targetOffset + output] - activation)
                        * beta
                        * activation
                        * (1.0 - activation);
            }
        }

        for (int layerIndex = lastLayerIndex - 1; layerIndex >= 0; layerIndex--) {
            var currentDelta = workspace.deltas[layerIndex];
            var currentActivation = workspace.activations[layerIndex + 1];
            var nextLayer = layers[layerIndex + 1];
            var nextDelta = workspace.deltas[layerIndex + 1];
            var vector = useVector(nextLayer.inputs);

            if (vector) {
                NeuroVectorOps.initScaled(
                        currentDelta,
                        nextLayer.weights,
                        0,
                        nextLayer.inputs,
                        nextDelta[0]);
                for (int nextOutput = 1; nextOutput < nextLayer.outputs; nextOutput++) {
                    NeuroVectorOps.addScaled(
                            currentDelta,
                            nextLayer.weights,
                            nextOutput * nextLayer.inputs,
                            nextLayer.inputs,
                            nextDelta[nextOutput]);
                }
                NeuroVectorOps.applyDerivative(currentDelta, currentActivation, currentDelta.length, beta);
            } else {
                for (int current = 0; current < nextLayer.inputs; current++) {
                    currentDelta[current] = nextDelta[0] * nextLayer.weights[current];
                }
                for (int nextOutput = 1; nextOutput < nextLayer.outputs; nextOutput++) {
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
                    var activation = currentActivation[current];
                    currentDelta[current] *= beta * activation * (1.0 - activation);
                }
            }
        }
    }

    private void applyOnlineGradient(double[] input, int inputOffset, Workspace workspace) {
        for (int layerIndex = 0; layerIndex < layers.length; layerIndex++) {
            var layer = layers[layerIndex];
            var source = layerIndex == 0 ? input : workspace.activations[layerIndex];
            var sourceOffset = layerIndex == 0 ? inputOffset : 0;
            var delta = workspace.deltas[layerIndex];
            var vector = useVector(layer.inputs);

            for (int output = 0; output < layer.outputs; output++) {
                var offset = output * layer.inputs;
                var scale = learningRate * delta[output];
                if (vector) {
                    NeuroVectorOps.update(
                            layer.weights,
                            layer.weightVelocity,
                            offset,
                            source,
                            sourceOffset,
                            layer.inputs,
                            momentum,
                            scale);
                } else {
                    for (int inputIndex = 0; inputIndex < layer.inputs; inputIndex++) {
                        var weightIndex = offset + inputIndex;
                        var velocity = Math.fma(
                                momentum,
                                layer.weightVelocity[weightIndex],
                                scale * source[sourceOffset + inputIndex]);
                        layer.weightVelocity[weightIndex] = velocity;
                        layer.weights[weightIndex] += velocity;
                    }
                }

                var biasVelocity = Math.fma(momentum, layer.biasVelocity[output], learningRate * delta[output]);
                layer.biasVelocity[output] = biasVelocity;
                layer.biases[output] += biasVelocity;
            }
        }
    }

    private double error(PackedDataset data, Workspace workspace) {
        var squaredError = 0.0;
        for (int sample = 0; sample < data.size; sample++) {
            var inputOffset = sample * data.inputSize;
            var targetOffset = sample * data.outputSize;
            forward(data.inputs, inputOffset, workspace, null, 0);
            var output = workspace.activations[topology.length - 1];
            for (int index = 0; index < data.outputSize; index++) {
                var difference = data.targets[targetOffset + index] - output[index];
                squaredError = Math.fma(difference, difference, squaredError);
            }
        }
        return Math.sqrt(squaredError / ((double) data.size * data.outputSize));
    }

    private PackedDataset trainingData() {
        if (packedTraining == null) {
            packedTraining = new PackedDataset(trainingSamples, topology[0], topology[topology.length - 1]);
        }
        return packedTraining;
    }

    private PackedDataset testData() {
        if (packedTests == null) {
            packedTests = new PackedDataset(testSamples, topology[0], topology[topology.length - 1]);
        }
        return packedTests;
    }

    private boolean useVector(int length) {
        return switch (hyperParameters.kernel()) {
            case SCALAR -> false;
            case VECTOR -> length >= NeuroVectorOps.doubleLanes();
            case AUTO -> length >= NeuroVectorOps.doubleLanes() * 2;
        };
    }

    private static boolean useFloatVector(Kernel kernel, int length) {
        return switch (kernel) {
            case SCALAR -> false;
            case VECTOR -> length >= NeuroVectorOps.floatLanes();
            case AUTO -> length >= NeuroVectorOps.floatLanes() * 2;
        };
    }

    private static double activate(double value, SigmoidMode mode) {
        if (mode == SigmoidMode.FAST) {
            return fastSigmoid(value);
        }
        return 1.0 / (1.0 + Math.exp(-value));
    }

    private static double fastSigmoid(double value) {
        if (value <= -8.0) {
            return 0.0;
        }
        if (value >= 8.0) {
            return 1.0;
        }
        var x = value * 0.5;
        var square = x * x;
        var tanh = x * (27.0 + square) / (27.0 + 9.0 * square);
        tanh = Math.max(-1.0, Math.min(1.0, tanh));
        return 0.5 * (tanh + 1.0);
    }

    private void validatePrediction(double[] input, double[] output) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(output, "output");
        if (input.length != topology[0]) {
            throw new IllegalArgumentException("input length " + input.length + " != expected " + topology[0]);
        }
        if (output.length != topology[topology.length - 1]) {
            throw new IllegalArgumentException(
                    "output length " + output.length + " != expected " + topology[topology.length - 1]);
        }
    }

    private void validateBatch(double[] inputs, int batchSize, double[] outputs) {
        Objects.requireNonNull(inputs, "inputs");
        Objects.requireNonNull(outputs, "outputs");
        if (batchSize < 0) {
            throw new IllegalArgumentException("batchSize must be >= 0");
        }
        if (inputs.length < batchSize * topology[0]) {
            throw new IllegalArgumentException("inputs array is too small for batchSize");
        }
        if (outputs.length < batchSize * topology[topology.length - 1]) {
            throw new IllegalArgumentException("outputs array is too small for batchSize");
        }
    }

    private void validateSample(double[] input, double[] target) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(target, "target");
        if (input.length != topology[0]) {
            throw new IllegalArgumentException("input length " + input.length + " != expected " + topology[0]);
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

    private void ensureTrainingOrder(int size) {
        if (trainingOrder.length == size) {
            return;
        }
        trainingOrder = new int[size];
        for (int index = 0; index < size; index++) {
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

    private static void await(Future<?>[] futures) {
        for (var future : futures) {
            try {
                future.get();
            } catch (InterruptedException exception) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException("parallel operation interrupted", exception);
            } catch (java.util.concurrent.ExecutionException exception) {
                throw new IllegalStateException("parallel operation failed", exception.getCause());
            }
        }
    }
}
