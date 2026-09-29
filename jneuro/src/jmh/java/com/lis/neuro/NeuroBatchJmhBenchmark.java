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
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Warmup;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(2)
@State(Scope.Thread)
public class NeuroBatchJmhBenchmark {
    @Param({"small", "medium"})
    public String topology;

    @Param({"8", "32", "128"})
    public int batchSize;

    private Neuro network;
    private Neuro.InferenceSession session;
    private Neuro.ParallelInferenceSession parallelSession;
    private Neuro.FloatModel floatModel;
    private double[] inputs;
    private double[] outputs;
    private float[] floatInputs;
    private float[] floatOutputs;

    @Setup(Level.Trial)
    public void setup() {
        var shape = NeuroInferenceJmhBenchmark.shape(topology);
        network = new Neuro(
                shape,
                Neuro.HyperParameters.defaults()
                        .withSeed(1234)
                        .withKernel(Neuro.Kernel.VECTOR));
        session = network.newInferenceSession();
        parallelSession = network.newParallelInferenceSession(2);
        floatModel = network.toFloatModel();

        var input = NeuroBenchmark.input(shape[0]);
        inputs = new double[input.length * batchSize];
        outputs = new double[shape[shape.length - 1] * batchSize];
        floatInputs = new float[inputs.length];
        floatOutputs = new float[outputs.length];
        for (int sample = 0; sample < batchSize; sample++) {
            System.arraycopy(input, 0, inputs, sample * input.length, input.length);
        }
        for (int i = 0; i < inputs.length; i++) {
            floatInputs[i] = (float) inputs[i];
        }
    }

    @TearDown(Level.Trial)
    public void tearDown() {
        parallelSession.close();
    }

    @Benchmark
    public double vectorBatch() {
        session.predictBatch(inputs, batchSize, outputs);
        return outputs[0];
    }

    @Benchmark
    public double parallelVectorBatch() {
        parallelSession.predictBatch(inputs, batchSize, outputs);
        return outputs[0];
    }

    @Benchmark
    public float floatVectorBatch() {
        floatModel.predictBatch(floatInputs, batchSize, floatOutputs);
        return floatOutputs[0];
    }
}
