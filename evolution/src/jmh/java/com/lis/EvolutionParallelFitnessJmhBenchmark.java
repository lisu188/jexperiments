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
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Warmup;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(2)
@State(Scope.Thread)
public class EvolutionParallelFitnessJmhBenchmark {
    @Param({"1", "2", "4", "8"})
    public int parallelism;

    private Evolution evolution;

    @Setup(Level.Iteration)
    public void setup() {
        EvolutionFitness expensive = (genome, offset, length) -> {
            var sum = 0.0;
            for (int repeat = 0; repeat < 128; repeat++) {
                for (int gene = 0; gene < length; gene++) {
                    var value = genome[offset + gene] - 0.5;
                    sum = Math.fma(value, value, sum);
                }
            }
            return sum;
        };
        evolution = new Evolution(
                128,
                expensive,
                Evolution.Config.defaults()
                        .withPopulationSize(512)
                        .withFitnessParallelism(parallelism)
                        .withTrackDiversity(false)
                        .withSeed(1234));
    }

    @TearDown(Level.Iteration)
    public void tearDown() {
        evolution.close();
    }

    @Benchmark
    public double evolveGeneration() {
        return evolution.evolve().bestError();
    }
}
