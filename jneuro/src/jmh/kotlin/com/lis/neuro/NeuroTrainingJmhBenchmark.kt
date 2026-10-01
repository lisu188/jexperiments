package com.lis.neuro

import java.util.concurrent.TimeUnit
import org.openjdk.jmh.annotations.*

/** PER_CALL includes session setup/close; RETAINED excludes both from the measured ten epochs. */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(2)
@State(Scope.Thread)
open class NeuroTrainingJmhBenchmark {
    @Param("small", "medium") @JvmField var topology = "small"
    @Param("FP64", "FP32") @JvmField var precision = "FP64"
    @Param("EXACT", "FAST") @JvmField var sigmoid = "EXACT"
    @Param("1", "16") @JvmField var batchSize = 1
    @Param("RETAINED", "PER_CALL") @JvmField var lifetime = "RETAINED"
    private lateinit var model: Neuro
    private var session: NeuroTrainingSession? = null
    @Setup(Level.Invocation) fun setup() {
        val shape = NeuroInferenceJmhBenchmark.shape(topology)
        model = Neuro(shape, Neuro.HyperParameters(0.1, 0.1, 1.0, 1234, sigmoidMode = Neuro.SigmoidMode.valueOf(sigmoid)))
        for (sample in 0 until 32) model.addTrainingSample(
            DoubleArray(shape[0]) { ((sample + it) and 7) / 7.0 },
            DoubleArray(shape.last()) { ((sample + it) and 1).toDouble() })
        require(lifetime in setOf("RETAINED", "PER_CALL"))
        session = if (lifetime == "RETAINED") open() else null
    }
    private fun open() = model.newTrainingSession(TrainingBackend.CPU, Neuro.TrainingPrecision.valueOf(precision), batchSize)
    @Benchmark fun tensorflowTenEpochs(): Double {
        val retained = session
        if (retained == null) open().use { it.train(10) } else retained.train(10)
        return model.statistics().lastTrainingError
    }
    @TearDown(Level.Invocation) fun close() { session?.close(); session = null }
}
