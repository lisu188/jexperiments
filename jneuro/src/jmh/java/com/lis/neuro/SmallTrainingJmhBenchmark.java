package com.lis.neuro;

import java.util.Arrays;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.*;

/**
 * TensorFlow CPU training with identical seeded models and public epoch scoring boundaries.
 * RETAINED excludes session setup/close; PER_CALL includes one session lifetime per ten epochs.
 * OperationsPerInvocation reports time per epoch. FP32 and FAST are distinct arithmetic modes.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(value = 3, jvmArgsAppend = {"-Djneuro.log.level=WARNING", "-Djneuro.log.file=false"})
@State(Scope.Thread)
public class SmallTrainingJmhBenchmark {
    @Param({"2-8-8-8-1"}) public String topology;
    @Param({"CPU_FP64", "CPU_FP32"}) public String engine;
    @Param({"ONLINE", "MINI_BATCH"}) public String workload;
    @Param({"EXACT", "FAST"}) public String sigmoid;
    @Param({"RETAINED", "PER_CALL"}) public String lifetime;
    @Param({"128"}) public int samples;
    @Param({"16"}) public int batchSize;
    private Neuro model;
    private NeuroTrainingSession session;
    private Neuro.TrainingPrecision precision;
    private int effectiveBatch;

    @Setup(Level.Invocation)
    public void setup() {
        int[] shape = Arrays.stream(topology.split("-")).mapToInt(Integer::parseInt).toArray();
        if (!SmallNetworkShape.INSTANCE.supports(shape) || samples < 1 || batchSize < 1)
            throw new IllegalArgumentException("Unsupported shape or dataset/batch size");
        effectiveBatch = switch (workload) {
            case "ONLINE" -> 1;
            case "MINI_BATCH" -> batchSize;
            default -> throw new IllegalArgumentException("Unknown workload: " + workload);
        };
        precision = switch (engine) {
            case "CPU_FP64" -> Neuro.TrainingPrecision.FP64;
            case "CPU_FP32" -> Neuro.TrainingPrecision.FP32;
            default -> throw new IllegalArgumentException("Unknown TensorFlow engine: " + engine);
        };
        if (!lifetime.equals("RETAINED") && !lifetime.equals("PER_CALL"))
            throw new IllegalArgumentException("Unknown session lifetime: " + lifetime);
        var hp = new Neuro.HyperParameters(0.1, 0.23, 1.15, 1234L).withSigmoidMode(Neuro.SigmoidMode.valueOf(sigmoid));
        model = new Neuro(shape, hp);
        for (int sample = 0; sample < samples; sample++)
            model.addTrainingSample(new double[] {(sample % 17) / 17.0, (sample * 7 % 19) / 19.0},
                new double[] {sample % 2});
        session = lifetime.equals("RETAINED") ? open() : null;
    }
    private NeuroTrainingSession open() {
        return model.newTrainingSession(TrainingBackend.CPU, precision, effectiveBatch, TrainingEngine.SMALL);
    }
    @Benchmark
    @OperationsPerInvocation(10)
    public double publicEpochs() {
        if (session != null) advance(session);
        else try (NeuroTrainingSession opened = open()) { advance(opened); }
        return model.statistics().lastTrainingError();
    }
    private void advance(NeuroTrainingSession training) {
        for (int epoch = 0; epoch < 10; epoch++) training.trainEpoch();
    }
    @TearDown(Level.Invocation)
    public void close() {
        if (session != null) { session.close(); session = null; }
    }
}
