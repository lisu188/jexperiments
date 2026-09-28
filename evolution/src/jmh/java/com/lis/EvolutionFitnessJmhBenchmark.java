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
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 10, time = 1)
@Fork(3)
@State(Scope.Thread)
public class EvolutionFitnessJmhBenchmark {
    @Param({"16", "128", "512", "2048"})
    public int geneCount;

    private double[] genome;
    private EvolutionFitness scalar;
    private EvolutionFitness vector;

    @Setup(Level.Trial)
    public void setup() {
        var goal = new double[geneCount];
        genome = new double[geneCount + 8];
        for (int index = 0; index < geneCount; index++) {
            goal[index] = (index & 31) / 31.0;
            genome[4 + index] = ((index * 7) & 31) / 31.0;
        }
        scalar = EvolutionFitness.meanSquaredError(goal);
        vector = EvolutionFitness.vectorMeanSquaredError(goal);
    }

    @Benchmark
    public double scalar() {
        return scalar.evaluate(genome, 4, geneCount);
    }

    @Benchmark
    public double vector() {
        return vector.evaluate(genome, 4, geneCount);
    }
}
