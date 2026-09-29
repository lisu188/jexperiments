package com.lis.neuro

import java.io.File
import java.net.URLClassLoader
import kotlin.math.abs

object NeuroMigrationParity {
    @JvmStatic fun main(args: Array<String>) {
        require(args.size == 1) { "Pass the compiled Java baseline directory." }
        val directory = File(args[0])
        require(directory.isDirectory) { "Java baseline directory does not exist: $directory" }
        URLClassLoader(arrayOf(directory.toURI().toURL()), ClassLoader.getPlatformClassLoader()).use { loader ->
            val type = loader.loadClass("com.lis.neuro.Neuro")
            val hyper = loader.loadClass("com.lis.neuro.Neuro\$HyperParameters")
            val kernelType = loader.loadClass("com.lis.neuro.Neuro\$Kernel")
            val sigmoidType = loader.loadClass("com.lis.neuro.Neuro\$SigmoidMode")
            val constructor = hyper.getConstructor(Double::class.javaPrimitiveType, Double::class.javaPrimitiveType,
                Double::class.javaPrimitiveType, Long::class.javaPrimitiveType, kernelType, sigmoidType)
            var scenarios = 0
            for (shape in listOf(intArrayOf(2, 6, 1), intArrayOf(3, 5, 2), intArrayOf(17, 33, 9),
                intArrayOf(2, 4, 3, 2, 1), intArrayOf(32, 64, 8))) {
                for (seed in listOf(0L, 42L, 1234L)) for (kernel in Neuro.Kernel.entries) for (mode in Neuro.SigmoidMode.entries) {
                    val label = "${shape.contentToString()} seed=$seed $kernel $mode"
                    val javaHyper = constructor.newInstance(0.1, 0.1, 1.0, seed,
                        kernelType.enumConstants.first { it.toString() == kernel.name },
                        sigmoidType.enumConstants.first { it.toString() == mode.name })
                    val java = type.getConstructor(IntArray::class.java, hyper).newInstance(shape, javaHyper)
                    val kotlin = Neuro(shape, Neuro.HyperParameters(0.1, 0.1, 1.0, seed, kernel, mode))
                    for (sample in 0 until 7) {
                        val input = DoubleArray(shape[0]) { ((sample + it) % 7) / 7.0 }
                        val target = DoubleArray(shape.last()) { ((sample + it) % 2).toDouble() }
                        invoke(java, "addTrainingSample", input, target); kotlin.addTrainingSample(input, target)
                        if (sample < 2) { invoke(java, "addTestSample", input, target); kotlin.addTestSample(input, target) }
                    }
                    compare(java, kotlin, label)
                    invoke(java, "train", 4); kotlin.train(4); compare(java, kotlin, "$label online")
                    invoke(java, "trainEpoch"); kotlin.trainEpoch(); compare(java, kotlin, "$label epoch")
                    invoke(java, "trainMiniBatch", 3, 4, 1); kotlin.trainMiniBatch(3, 4, 1); compare(java, kotlin, "$label minibatch")
                    invoke(java, "trainMiniBatch", 2, 4, 3); kotlin.trainMiniBatch(2, 4, 3); compare(java, kotlin, "$label parallel minibatch")
                    scenarios++
                }
            }
            println("Java/Kotlin differential parity passed: $scenarios scenarios, 450 model states; double tolerance 1e-12, float tolerance 1e-6.")
        }
    }

    private fun compare(java: Any, kotlin: Neuro, label: String) {
        val shape = kotlin.topology()
        check(invoke(java, "parameterCount") == kotlin.parameterCount()) { "$label parameter count" }
        near(invoke(java, "trainingError") as Double, kotlin.trainingError(), 1e-12, "$label training error")
        near(invoke(java, "testError") as Double, kotlin.testError(), 1e-12, "$label test error")
        val statistics = invoke(java, "statistics")!!
        check(invoke(statistics, "epochsTrained") == kotlin.statistics().epochsTrained) { "$label epoch count" }
        check(invoke(statistics, "samplesSeen") == kotlin.statistics().samplesSeen) { "$label sample count" }
        val inputs = DoubleArray(shape[0] * 3) { (it % 11) / 11.0 }
        val javaOutput = DoubleArray(shape.last() * 3); val kotlinOutput = DoubleArray(javaOutput.size)
        invoke(java, "predictBatch", inputs, 3, javaOutput)
        kotlin.predictBatch(inputs, 3, kotlinOutput)
        for (i in javaOutput.indices) near(javaOutput[i], kotlinOutput[i], 1e-12, "$label batch[$i]")
        val point = inputs.copyOfRange(0, shape[0])
        val single = invoke(java, "predict", point) as DoubleArray
        val actual = kotlin.predict(point)
        for (i in single.indices) near(single[i], actual[i], 1e-12, "$label predict[$i]")
        val javaFloat = invoke(java, "toFloatModel")!!
        val floatInput = FloatArray(shape[0]) { point[it].toFloat() }
        val expectedFloat = invoke(javaFloat, "predict", floatInput) as FloatArray
        val actualFloat = kotlin.toFloatModel().predict(floatInput)
        for (i in expectedFloat.indices) near(expectedFloat[i].toDouble(), actualFloat[i].toDouble(), 1e-6, "$label float[$i]")
    }

    private fun invoke(receiver: Any, name: String, vararg args: Any): Any? =
        receiver.javaClass.methods.first { it.name == name && it.parameterCount == args.size }.invoke(receiver, *args)

    private fun near(expected: Double, actual: Double, tolerance: Double, label: String) {
        check(abs(expected - actual) <= tolerance) { "$label expected $expected, actual $actual" }
    }
}
