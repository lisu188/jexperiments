package com.lis.neuro

import java.util.concurrent.TimeUnit
import org.openjdk.jmh.annotations.*

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(2)
@State(Scope.Thread)
open class NeuroTrainingJmhBenchmark {
    @Param("small", "medium") @JvmField var topology = "small"
    private lateinit var scalar: Neuro
    private lateinit var vector: Neuro
    private lateinit var miniBatch: Neuro
    private lateinit var parallelMiniBatch: Neuro
    @Setup(Level.Invocation) fun setup() {
        scalar = prepared(Neuro.Kernel.SCALAR); vector = prepared(Neuro.Kernel.VECTOR)
        miniBatch = prepared(Neuro.Kernel.VECTOR); parallelMiniBatch = prepared(Neuro.Kernel.VECTOR)
    }
    @Benchmark fun scalarTenEpochs(): Double { scalar.train(10); return scalar.statistics().lastTrainingError }
    @Benchmark fun vectorTenEpochs(): Double { vector.train(10); return vector.statistics().lastTrainingError }
    @Benchmark fun vectorMiniBatchTenEpochs(): Double { miniBatch.trainMiniBatch(10, 16); return miniBatch.statistics().lastTrainingError }
    @Benchmark fun parallelMiniBatchTenEpochs(): Double { parallelMiniBatch.trainMiniBatch(10, 16, 2); return parallelMiniBatch.statistics().lastTrainingError }
    private fun prepared(kernel: Neuro.Kernel): Neuro {
        val shape = NeuroInferenceJmhBenchmark.shape(topology)
        val model = Neuro(shape, Neuro.HyperParameters(0.1, 0.1, 1.0, 1234, kernel))
        for (sample in 0 until 32) model.addTrainingSample(
            DoubleArray(shape[0]) { ((sample + it) and 7) / 7.0 },
            DoubleArray(shape.last()) { ((sample + it) and 1).toDouble() })
        return model
    }
}
