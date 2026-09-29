package com.lis.neuro

import java.util.concurrent.TimeUnit
import org.openjdk.jmh.annotations.*

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 10, time = 1)
@Fork(3)
@State(Scope.Thread)
open class NeuroInferenceJmhBenchmark {
    @Param("small", "medium", "deep") @JvmField var topology = "small"
    private lateinit var legacy: LegacyNeuroBaseline
    private lateinit var scalar: Neuro
    private lateinit var vector: Neuro
    private lateinit var automatic: Neuro
    private lateinit var vectorSession: Neuro.InferenceSession
    private lateinit var fastSession: Neuro.InferenceSession
    private lateinit var floatModel: Neuro.FloatModel
    private lateinit var input: DoubleArray
    private lateinit var legacyOutput: DoubleArray
    private lateinit var scalarOutput: DoubleArray
    private lateinit var vectorOutput: DoubleArray
    private lateinit var autoOutput: DoubleArray
    private lateinit var fastOutput: DoubleArray
    private lateinit var floatInput: FloatArray
    private lateinit var floatOutput: FloatArray
    @Setup(Level.Trial) fun setup() {
        val shape = shape(topology)
        legacy = LegacyNeuroBaseline(shape)
        scalar = network(shape, Neuro.Kernel.SCALAR, Neuro.SigmoidMode.EXACT)
        vector = network(shape, Neuro.Kernel.VECTOR, Neuro.SigmoidMode.EXACT)
        automatic = network(shape, Neuro.Kernel.AUTO, Neuro.SigmoidMode.EXACT)
        val fast = network(shape, Neuro.Kernel.VECTOR, Neuro.SigmoidMode.FAST)
        vectorSession = vector.newInferenceSession()
        fastSession = fast.newInferenceSession()
        floatModel = vector.toFloatModel()
        input = DoubleArray(shape[0]) { (it and 7) / 7.0 }
        legacyOutput = DoubleArray(shape.last()); scalarOutput = DoubleArray(shape.last())
        vectorOutput = DoubleArray(shape.last()); autoOutput = DoubleArray(shape.last()); fastOutput = DoubleArray(shape.last())
        floatInput = FloatArray(input.size) { input[it].toFloat() }; floatOutput = FloatArray(shape.last())
    }
    @Benchmark fun legacyPredictInto(): Double { legacy.predictInto(input, legacyOutput); return legacyOutput[0] }
    @Benchmark fun scalarPredictInto(): Double { scalar.predictInto(input, scalarOutput); return scalarOutput[0] }
    @Benchmark fun vectorPredictInto(): Double { vector.predictInto(input, vectorOutput); return vectorOutput[0] }
    @Benchmark fun autoPredictInto(): Double { automatic.predictInto(input, autoOutput); return autoOutput[0] }
    @Benchmark fun vectorSession(): Double { vectorSession.predictInto(input, vectorOutput); return vectorOutput[0] }
    @Benchmark fun fastVectorSession(): Double { fastSession.predictInto(input, fastOutput); return fastOutput[0] }
    @Benchmark fun floatVectorPredictInto(): Float { floatModel.predictInto(floatInput, floatOutput); return floatOutput[0] }
    companion object {
        private fun network(shape: IntArray, kernel: Neuro.Kernel, mode: Neuro.SigmoidMode) =
            Neuro(shape, Neuro.HyperParameters.defaults().withSeed(1234).withKernel(kernel).withSigmoidMode(mode))
        fun shape(name: String): IntArray = when (name) {
            "small" -> intArrayOf(32, 64, 32, 8)
            "medium" -> intArrayOf(128, 256, 128, 32)
            "deep" -> intArrayOf(64, 128, 128, 64, 32, 8)
            else -> throw IllegalArgumentException(name)
        }
    }
}
