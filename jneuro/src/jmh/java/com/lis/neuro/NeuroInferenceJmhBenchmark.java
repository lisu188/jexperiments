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
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 10, time = 1)
@Fork(3)
@State(Scope.Thread)
public class NeuroInferenceJmhBenchmark {
    @Param({"small", "medium", "deep"})
    public String topology;

    private LegacyNeuroBaseline legacy;
    private Neuro scalar;
    private Neuro vector;
    private Neuro automatic;
    private Neuro fast;
    private Neuro.InferenceSession vectorSession;
    private Neuro.InferenceSession fastSession;
    private Neuro.FloatModel floatModel;
    private double[] input;
    private double[] legacyOutput;
    private double[] scalarOutput;
    private double[] vectorOutput;
    private double[] autoOutput;
    private double[] fastOutput;
    private float[] floatInput;
    private float[] floatOutput;

    @Setup(Level.Trial)
    public void setup() {
        var shape = shape(topology);
        legacy = new LegacyNeuroBaseline(shape);
        scalar = network(shape, Neuro.Kernel.SCALAR, Neuro.SigmoidMode.EXACT);
        vector = network(shape, Neuro.Kernel.VECTOR, Neuro.SigmoidMode.EXACT);
        automatic = network(shape, Neuro.Kernel.AUTO, Neuro.SigmoidMode.EXACT);
        fast = network(shape, Neuro.Kernel.VECTOR, Neuro.SigmoidMode.FAST);
        vectorSession = vector.newInferenceSession();
        fastSession = fast.newInferenceSession();
        floatModel = vector.toFloatModel();

        input = NeuroBenchmark.input(shape[0]);
        legacyOutput = new double[shape[shape.length - 1]];
        scalarOutput = new double[legacyOutput.length];
        vectorOutput = new double[legacyOutput.length];
        autoOutput = new double[legacyOutput.length];
        fastOutput = new double[legacyOutput.length];
        floatInput = new float[input.length];
        floatOutput = new float[legacyOutput.length];
        for (int i = 0; i < input.length; i++) {
            floatInput[i] = (float) input[i];
        }
    }

    @Benchmark
    public double legacyPredictInto() {
        legacy.predictInto(input, legacyOutput);
        return legacyOutput[0];
    }

    @Benchmark
    public double scalarPredictInto() {
        scalar.predictInto(input, scalarOutput);
        return scalarOutput[0];
    }

    @Benchmark
    public double vectorPredictInto() {
        vector.predictInto(input, vectorOutput);
        return vectorOutput[0];
    }

    @Benchmark
    public double autoPredictInto() {
        automatic.predictInto(input, autoOutput);
        return autoOutput[0];
    }

    @Benchmark
    public double vectorSession() {
        vectorSession.predictInto(input, vectorOutput);
        return vectorOutput[0];
    }

    @Benchmark
    public double fastVectorSession() {
        fastSession.predictInto(input, fastOutput);
        return fastOutput[0];
    }

    @Benchmark
    public float floatVectorPredictInto() {
        floatModel.predictInto(floatInput, floatOutput);
        return floatOutput[0];
    }

    private static Neuro network(int[] shape, Neuro.Kernel kernel, Neuro.SigmoidMode sigmoidMode) {
        return new Neuro(
                shape,
                Neuro.HyperParameters.defaults()
                        .withSeed(1234)
                        .withKernel(kernel)
                        .withSigmoidMode(sigmoidMode));
    }

    static int[] shape(String name) {
        return switch (name) {
            case "small" -> new int[]{32, 64, 32, 8};
            case "medium" -> new int[]{128, 256, 128, 32};
            case "deep" -> new int[]{64, 128, 128, 64, 32, 8};
            default -> throw new IllegalArgumentException(name);
        };
    }
}
