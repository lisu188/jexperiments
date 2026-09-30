package com.lis.neuro

object NeuroCudaTrainingBenchmark {
    @JvmStatic fun main(args: Array<String>) {
        if (!NeuroCuda.isAvailable()) {
            println("CUDA unavailable: " + NeuroCuda.status().reason)
            return
        }
        val scenarios = listOf(
            "medium" to intArrayOf(128, 256, 128, 32),
            "large" to intArrayOf(512, 1024, 512, 128),
            "huge" to intArrayOf(1024, 2048, 2048, 512)
        )
        val batches = intArrayOf(32, 128, 512, 2048)
        for ((name, topology) in scenarios) {
            for (batch in batches) {
                val samples = maxOf(batch * 2, 2048)
                val cpu = prepared(topology, samples)
                val gpu64 = prepared(topology, samples)
                val gpu32 = prepared(topology, samples)
                val cpuNanos = measure {
                    cpu.trainMiniBatch(2, batch, Runtime.getRuntime().availableProcessors().coerceAtMost(8),
                        Neuro.TrainingBackend.CPU)
                }
                val gpu64Nanos = measure {
                    gpu64.trainMiniBatch(2, batch, 1, Neuro.TrainingBackend.CUDA, Neuro.TrainingPrecision.FP64)
                }
                val gpu32Nanos = measure {
                    gpu32.trainMiniBatch(2, batch, 1, Neuro.TrainingBackend.CUDA, Neuro.TrainingPrecision.FP32)
                }
                println("%-7s batch=%4d params=%9d cpu=%9.3f ms fp64=%9.3f ms fp32=%9.3f ms fp64=%6.2fx fp32=%6.2fx diff64=%g diff32=%g".format(
                    name, batch, cpu.parameterCount(), cpuNanos / 1_000_000.0, gpu64Nanos / 1_000_000.0,
                    gpu32Nanos / 1_000_000.0, cpuNanos.toDouble() / gpu64Nanos, cpuNanos.toDouble() / gpu32Nanos,
                    kotlin.math.abs(cpu.trainingError() - gpu64.trainingError()),
                    kotlin.math.abs(cpu.trainingError() - gpu32.trainingError())))
            }
        }
    }
    private fun prepared(topology: IntArray, samples: Int): Neuro =
        Neuro(topology, Neuro.HyperParameters(0.05, 0.1, 1.0, 1234, Neuro.Kernel.VECTOR)).also { model ->
            repeat(samples) { sample ->
                model.addTrainingSample(
                    DoubleArray(topology[0]) { ((sample * 17 + it * 13) and 255) / 255.0 },
                    DoubleArray(topology.last()) { ((sample + it) and 1).toDouble() })
            }
        }
    private inline fun measure(action: () -> Unit): Long {
        val start = System.nanoTime()
        action()
        return System.nanoTime() - start
    }
}
