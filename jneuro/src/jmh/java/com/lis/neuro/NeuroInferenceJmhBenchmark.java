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

    private Neuro network;
    private double[] input;
    private double[] output;

    @Setup(Level.Trial)
    public void setup() {
        var shape = switch (topology) {
            case "small" -> new int[]{32, 64, 32, 8};
            case "medium" -> new int[]{128, 256, 128, 32};
            case "deep" -> new int[]{64, 128, 128, 64, 32, 8};
            default -> throw new IllegalArgumentException(topology);
        };
        network = NeuroBenchmark.preparedNetwork(shape);
        input = NeuroBenchmark.input(shape[0]);
        output = new double[shape[shape.length - 1]];
    }

    @Benchmark
    public double[] predictAllocating() {
        return network.predict(input);
    }

    @Benchmark
    public double predictInto() {
        network.predictInto(input, output);
        return output[0];
    }
}
