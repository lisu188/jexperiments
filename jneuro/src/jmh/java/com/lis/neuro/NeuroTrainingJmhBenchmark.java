package com.lis.neuro;

import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(2)
@State(Scope.Thread)
public class NeuroTrainingJmhBenchmark {
    @Param({"small", "medium"})
    public String topology;

    private int[] shape;
    private Neuro scalar;
    private Neuro vector;
    private Neuro miniBatch;
    private Neuro parallelMiniBatch;

    @Setup(Level.Invocation)
    public void setup() {
        shape = NeuroInferenceJmhBenchmark.shape(topology);
        scalar = prepared(Neuro.Kernel.SCALAR);
        vector = prepared(Neuro.Kernel.VECTOR);
        miniBatch = prepared(Neuro.Kernel.VECTOR);
        parallelMiniBatch = prepared(Neuro.Kernel.VECTOR);
    }

    @Benchmark
    public double scalarTenEpochs() {
        scalar.train(10);
        return scalar.statistics().lastTrainingError();
    }

    @Benchmark
    public double vectorTenEpochs() {
        vector.train(10);
        return vector.statistics().lastTrainingError();
    }

    @Benchmark
    public double vectorMiniBatchTenEpochs() {
        miniBatch.trainMiniBatch(10, 16);
        return miniBatch.statistics().lastTrainingError();
    }

    @Benchmark
    public double parallelMiniBatchTenEpochs() {
        parallelMiniBatch.trainMiniBatch(10, 16, 2);
        return parallelMiniBatch.statistics().lastTrainingError();
    }

    private Neuro prepared(Neuro.Kernel kernel) {
        var network = new Neuro(
                shape,
                Neuro.HyperParameters.defaults()
                        .withLearningRate(0.1)
                        .withMomentum(0.1)
                        .withSeed(1234)
                        .withKernel(kernel));
        var outputSize = shape[shape.length - 1];
        for (int sample = 0; sample < 32; sample++) {
            var input = new double[shape[0]];
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
}
