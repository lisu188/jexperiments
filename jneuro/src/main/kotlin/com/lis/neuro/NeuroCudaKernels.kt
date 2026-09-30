package com.lis.neuro

import java.lang.foreign.MemorySegment

internal class NeuroCudaKernels(
    nvrtc: NeuroNvrtc,
    private val driver: NeuroCudaDriver,
    computeCapability: Pair<Int, Int>
) : AutoCloseable {
    private val module = driver.loadModule(nvrtc.compile(source(), "jneuro.cu", computeCapability))
    private val functions = try {
        listOf("gatherRows", "addBiasAndSigmoid", "outputDelta", "applySigmoidDerivative",
            "reduceBiasGradient", "momentumUpdate", "gatherRowsFloat", "addBiasAndSigmoidFloat",
            "outputDeltaFloat", "applySigmoidDerivativeFloat", "reduceBiasGradientFloat", "momentumUpdateFloat")
            .associateWith(module::function)
    } catch (failure: Throwable) {
        NeuroLog.error("cuda", "kernel.lookup.failed", failure)
        try { module.close() } catch (cleanup: Throwable) { failure.addSuppressed(cleanup) }
        throw failure
    }
    private val gatherRows = functions.getValue("gatherRows")
    private val activate = functions.getValue("addBiasAndSigmoid")
    private val outputDelta = functions.getValue("outputDelta")
    private val applyDerivative = functions.getValue("applySigmoidDerivative")
    private val reduceBias = functions.getValue("reduceBiasGradient")
    private val momentumUpdate = functions.getValue("momentumUpdate")
    private val gatherRowsFloat = functions.getValue("gatherRowsFloat")
    private val activateFloat = functions.getValue("addBiasAndSigmoidFloat")
    private val outputDeltaFloat = functions.getValue("outputDeltaFloat")
    private val applyDerivativeFloat = functions.getValue("applySigmoidDerivativeFloat")
    private val reduceBiasFloat = functions.getValue("reduceBiasGradientFloat")
    private val momentumUpdateFloat = functions.getValue("momentumUpdateFloat")

    fun gatherRows(source: MemorySegment, width: Int, order: MemorySegment, start: Int, count: Int,
                   destination: MemorySegment) {
        launch("gatherRows", gatherRows, Math.multiplyExact(count, width),
            pointer(source), int(width), pointer(order), int(start), int(count), pointer(destination))
    }
    fun activate(values: MemorySegment, biases: MemorySegment, batch: Int, width: Int, beta: Double,
                 mode: Neuro.SigmoidMode) {
        launch("addBiasAndSigmoid", activate, Math.multiplyExact(batch, width),
            pointer(values), pointer(biases), int(batch), int(width), double(beta),
            int(if (mode == Neuro.SigmoidMode.FAST) 1 else 0))
    }
    fun outputDelta(targets: MemorySegment, activations: MemorySegment, deltas: MemorySegment,
                    elements: Int, beta: Double) {
        launch("outputDelta", outputDelta, elements, pointer(targets), pointer(activations), pointer(deltas),
            int(elements), double(beta))
    }
    fun applyDerivative(deltas: MemorySegment, activations: MemorySegment, elements: Int, beta: Double) {
        launch("applySigmoidDerivative", applyDerivative, elements, pointer(deltas), pointer(activations), int(elements), double(beta))
    }
    fun reduceBiasGradient(deltas: MemorySegment, gradient: MemorySegment, batch: Int, width: Int) {
        launch("reduceBiasGradient", reduceBias, width, pointer(deltas), pointer(gradient), int(batch), int(width))
    }
    fun momentumUpdate(values: MemorySegment, velocity: MemorySegment, gradient: MemorySegment,
                       elements: Int, momentum: Double, scale: Double) {
        launch("momentumUpdate", momentumUpdate, elements, pointer(values), pointer(velocity), pointer(gradient),
            int(elements), double(momentum), double(scale))
    }
    fun gatherRowsFloat(source: MemorySegment, width: Int, order: MemorySegment, start: Int, count: Int,
                        destination: MemorySegment) {
        launch("gatherRowsFloat", gatherRowsFloat, Math.multiplyExact(count, width),
            pointer(source), int(width), pointer(order), int(start), int(count), pointer(destination))
    }
    fun activateFloat(values: MemorySegment, biases: MemorySegment, batch: Int, width: Int, beta: Float,
                      mode: Neuro.SigmoidMode) {
        launch("addBiasAndSigmoidFloat", activateFloat, Math.multiplyExact(batch, width),
            pointer(values), pointer(biases), int(batch), int(width), float(beta),
            int(if (mode == Neuro.SigmoidMode.FAST) 1 else 0))
    }
    fun outputDeltaFloat(targets: MemorySegment, activations: MemorySegment, deltas: MemorySegment,
                         elements: Int, beta: Float) {
        launch("outputDeltaFloat", outputDeltaFloat, elements, pointer(targets), pointer(activations), pointer(deltas),
            int(elements), float(beta))
    }
    fun applyDerivativeFloat(deltas: MemorySegment, activations: MemorySegment, elements: Int, beta: Float) {
        launch("applySigmoidDerivativeFloat", applyDerivativeFloat, elements, pointer(deltas), pointer(activations), int(elements), float(beta))
    }
    fun reduceBiasGradientFloat(deltas: MemorySegment, gradient: MemorySegment, batch: Int, width: Int) {
        launch("reduceBiasGradientFloat", reduceBiasFloat, width, pointer(deltas), pointer(gradient), int(batch), int(width))
    }
    fun momentumUpdateFloat(values: MemorySegment, velocity: MemorySegment, gradient: MemorySegment,
                            elements: Int, momentum: Float, scale: Float) {
        launch("momentumUpdateFloat", momentumUpdateFloat, elements, pointer(values), pointer(velocity), pointer(gradient),
            int(elements), float(momentum), float(scale))
    }

    private fun launch(name: String, function: MemorySegment, elements: Int, vararg arguments: NeuroCudaDriver.Argument) {
        NeuroLog.trace("cuda", "kernel.submitted") { mapOf("kernel" to name, "workItems" to elements,
            "stream" to "default", "arguments" to arguments.size) }
        driver.launch(function, elements, *arguments)
    }

    override fun close() = module.close()
    private fun pointer(value: MemorySegment) = NeuroCudaDriver.Argument.Pointer(value)
    private fun int(value: Int) = NeuroCudaDriver.Argument.IntValue(value)
    private fun double(value: Double) = NeuroCudaDriver.Argument.DoubleValue(value)
    private fun float(value: Float) = NeuroCudaDriver.Argument.FloatValue(value)

    companion object {
        private fun source(): String = checkNotNull(
            NeuroCudaKernels::class.java.getResourceAsStream("/cuda/jneuro.cu")) {
            "Missing CUDA kernel resource /cuda/jneuro.cu"
        }.bufferedReader().use { it.readText() }
    }
}
