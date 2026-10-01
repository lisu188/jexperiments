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
    @Param("EXACT", "FAST") @JvmField var sigmoid = "EXACT"
    private lateinit var model: Neuro
    private lateinit var session: Neuro.InferenceSession
    private lateinit var floatModel: Neuro.FloatModel
    private lateinit var input: DoubleArray
    private lateinit var output: DoubleArray
    private lateinit var floatInput: FloatArray
    private lateinit var floatOutput: FloatArray
    @Setup(Level.Trial) fun setup() {
        val shape = shape(topology)
        model = Neuro(shape, Neuro.HyperParameters.defaults().withSeed(1234).withSigmoidMode(Neuro.SigmoidMode.valueOf(sigmoid)))
        session = model.newInferenceSession(); floatModel = model.toFloatModel()
        input = NeuroBenchmark.input(shape[0]); output = DoubleArray(shape.last())
        floatInput = FloatArray(input.size) { input[it].toFloat() }; floatOutput = FloatArray(shape.last())
    }
    @Benchmark fun tensorflowAllocating(): Double = model.predict(input)[0]
    @Benchmark fun tensorflowDirect(): Double { model.predictInto(input, output); return output[0] }
    @Benchmark fun tensorflowSession(): Double { session.predictInto(input, output); return output[0] }
    @Benchmark fun tensorflowFp32(): Float { floatModel.predictInto(floatInput, floatOutput); return floatOutput[0] }
    companion object {
        fun shape(name: String): IntArray = when (name) {
            "small" -> intArrayOf(32, 64, 32, 8)
            "medium" -> intArrayOf(128, 256, 128, 32)
            "deep" -> intArrayOf(64, 128, 128, 64, 32, 8)
            else -> throw IllegalArgumentException(name)
        }
    }
}
