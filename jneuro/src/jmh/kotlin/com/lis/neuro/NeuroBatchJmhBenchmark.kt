package com.lis.neuro

import java.util.concurrent.TimeUnit
import org.openjdk.jmh.annotations.*

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(2)
@State(Scope.Thread)
open class NeuroBatchJmhBenchmark {
    @Param("small", "medium") @JvmField var topology = "small"
    @Param("8", "32", "128") @JvmField var batchSize = 8
    private lateinit var session: Neuro.InferenceSession
    private lateinit var parallelSession: Neuro.ParallelInferenceSession
    private lateinit var floatModel: Neuro.FloatModel
    private lateinit var inputs: DoubleArray
    private lateinit var outputs: DoubleArray
    private lateinit var floatInputs: FloatArray
    private lateinit var floatOutputs: FloatArray
    @Setup(Level.Trial) fun setup() {
        val shape = NeuroInferenceJmhBenchmark.shape(topology)
        val network = Neuro(shape, Neuro.HyperParameters.defaults().withSeed(1234).withKernel(Neuro.Kernel.VECTOR))
        session = network.newInferenceSession(); parallelSession = network.newParallelInferenceSession(2); floatModel = network.toFloatModel()
        val input = DoubleArray(shape[0]) { (it and 7) / 7.0 }
        inputs = DoubleArray(input.size * batchSize)
        for (sample in 0 until batchSize) input.copyInto(inputs, sample * input.size)
        outputs = DoubleArray(shape.last() * batchSize)
        floatInputs = FloatArray(inputs.size) { inputs[it].toFloat() }; floatOutputs = FloatArray(outputs.size)
    }
    @TearDown(Level.Trial) fun tearDown() = parallelSession.close()
    @Benchmark fun vectorBatch(): Double { session.predictBatch(inputs, batchSize, outputs); return outputs[0] }
    @Benchmark fun parallelVectorBatch(): Double { parallelSession.predictBatch(inputs, batchSize, outputs); return outputs[0] }
    @Benchmark fun floatVectorBatch(): Float { floatModel.predictBatch(floatInputs, batchSize, floatOutputs); return floatOutputs[0] }
}
