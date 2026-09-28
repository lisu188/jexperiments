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

    private Neuro network;

    @Setup(Level.Iteration)
    public void setup() {
        var shape = switch (topology) {
            case "small" -> new int[]{32, 64, 32, 8};
            case "medium" -> new int[]{128, 256, 128, 32};
            default -> throw new IllegalArgumentException(topology);
        };
        network = NeuroBenchmark.preparedNetwork(shape);
    }

    @Benchmark
    public double trainEpoch() {
        return network.trainEpoch();
    }
}
