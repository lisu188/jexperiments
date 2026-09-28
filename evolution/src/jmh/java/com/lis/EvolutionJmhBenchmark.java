package com.lis;

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
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 10, time = 1)
@Fork(3)
@State(Scope.Thread)
public class EvolutionJmhBenchmark {
    @Param({"64", "512", "4096"})
    public int populationSize;

    @Param({"16", "128"})
    public int geneCount;

    private Evolution evolution;

    @Setup(Level.Iteration)
    public void setup() {
        evolution = EvolutionBenchmark.create(populationSize, geneCount, 12345);
    }

    @Benchmark
    public double evolveGeneration() {
        return evolution.evolve().bestError();
    }
}
