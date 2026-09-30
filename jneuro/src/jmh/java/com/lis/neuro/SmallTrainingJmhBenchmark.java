package com.lis.neuro;

import java.util.Arrays;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OperationsPerInvocation;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Warmup;

/**
 * Identical seeded trajectories and public epoch publication/RMSE boundaries.
 * Setup and close are excluded; the command-line harness measures those separately.
 * Reports time per epoch, not per ten-epoch invocation. Use -prof gc for allocation.
 * The topology parameter accepts every supported shape, for example 2-4-16-8-1.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(value = 3, jvmArgsAppend = {
    "--add-modules=jdk.incubator.vector", "-Djneuro.log.level=WARNING", "-Djneuro.log.file=false"
})
@State(Scope.Thread)
public class SmallTrainingJmhBenchmark {
    @Param({"2-8-8-8-1"}) public String topology;
    @Param({"REFERENCE_SCALAR", "REFERENCE_MATRIX", "SMALL_SCALAR", "SMALL_128", "SMALL_256",
        "SMALL_FP32_SCALAR", "SMALL_FP32_128", "SMALL_FP32_256"}) public String engine;
    @Param({"ONLINE", "MINI_BATCH"}) public String workload;
    @Param({"EXACT", "FAST"}) public String sigmoid;
    @Param({"128"}) public int samples;
    @Param({"16"}) public int batchSize;
    private Neuro model;
    private NeuroTrainingSession session;
    private boolean online;

    @Setup(Level.Invocation)
    public void setup() {
        int[] shape = Arrays.stream(topology.split("-")).mapToInt(Integer::parseInt).toArray();
        if (!SmallNetworkShape.INSTANCE.supports(shape) || samples < 1 || batchSize < 1) {
            throw new IllegalArgumentException("Unsupported shape or dataset/batch size");
        }
        online = switch (workload) {
            case "ONLINE" -> true;
            case "MINI_BATCH" -> false;
            default -> throw new IllegalArgumentException("Unknown workload: " + workload);
        };
        int bits = switch (engine) {
            case "REFERENCE_SCALAR", "REFERENCE_MATRIX", "SMALL_SCALAR", "SMALL_FP32_SCALAR" -> 0;
            case "SMALL_128", "SMALL_FP32_128" -> 128;
            case "SMALL_256", "SMALL_FP32_256" -> 256;
            default -> throw new IllegalArgumentException("Unknown engine: " + engine);
        };
        Neuro.HyperParameters hp = new Neuro.HyperParameters(0.1, 0.23, 1.15, 1234L,
            bits == 0 ? Neuro.Kernel.SCALAR : Neuro.Kernel.VECTOR, Neuro.SigmoidMode.valueOf(sigmoid));
        model = new Neuro(shape, hp);
        for (int sample = 0; sample < samples; sample++) {
            model.addTrainingSample(new double[] {(sample % 17) / 17.0, (sample * 7 % 19) / 19.0},
                new double[] {(sample % 2)});
        }
        session = null;
        if (engine.startsWith("SMALL")) {
            Neuro.TrainingPrecision precision = engine.contains("FP32") ? Neuro.TrainingPrecision.FP32 : Neuro.TrainingPrecision.FP64;
            session = new SmallTrainingSession(model, online ? 1 : batchSize,
                state -> new SmallCpuTraining(state, hp, precision, bits), System::nanoTime);
        }
    }

    @Benchmark
    @OperationsPerInvocation(10)
    public double publicEpochs() {
        for (int epoch = 0; epoch < 10; epoch++) {
            if (session != null) {
                if (online) session.trainEpoch();
                else session.trainMiniBatch(1, batchSize, 1);
            } else if (engine.equals("REFERENCE_MATRIX")) {
                model.trainMiniBatch(1, online ? 1 : batchSize, 1, Neuro.BatchBackend.CPU);
            } else if (online) model.trainEpoch();
            else model.trainMiniBatch(1, batchSize, 1);
        }
        return model.statistics().lastTrainingError();
    }

    @TearDown(Level.Invocation)
    public void close() throws Exception {
        if (session != null) session.close();
    }
}
