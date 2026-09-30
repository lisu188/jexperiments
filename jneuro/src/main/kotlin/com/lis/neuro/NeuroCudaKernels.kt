package com.lis.neuro

import java.lang.foreign.MemorySegment

internal class NeuroCudaKernels(
    nvrtc: NeuroNvrtc,
    private val driver: NeuroCudaDriver,
    computeCapability: Pair<Int, Int>
) : AutoCloseable {
    private val module = driver.loadModule(nvrtc.compile(source(), "jneuro.cu", computeCapability))
    private val gatherRows = module.function("gatherRows")
    private val activate = module.function("addBiasAndSigmoid")
    private val outputDelta = module.function("outputDelta")
    private val applyDerivative = module.function("applySigmoidDerivative")
    private val reduceBias = module.function("reduceBiasGradient")
    private val momentumUpdate = module.function("momentumUpdate")
    private val gatherRowsFloat = module.function("gatherRowsFloat")
    private val activateFloat = module.function("addBiasAndSigmoidFloat")
    private val outputDeltaFloat = module.function("outputDeltaFloat")
    private val applyDerivativeFloat = module.function("applySigmoidDerivativeFloat")
    private val reduceBiasFloat = module.function("reduceBiasGradientFloat")
    private val momentumUpdateFloat = module.function("momentumUpdateFloat")

    fun gatherRows(source: MemorySegment, width: Int, order: MemorySegment, start: Int, count: Int,
                   destination: MemorySegment) {
        driver.launch(gatherRows, Math.multiplyExact(count, width),
            pointer(source), int(width), pointer(order), int(start), int(count), pointer(destination))
    }
    fun activate(values: MemorySegment, biases: MemorySegment, batch: Int, width: Int, beta: Double,
                 mode: Neuro.SigmoidMode) {
        driver.launch(activate, Math.multiplyExact(batch, width),
            pointer(values), pointer(biases), int(batch), int(width), double(beta),
            int(if (mode == Neuro.SigmoidMode.FAST) 1 else 0))
    }
    fun outputDelta(targets: MemorySegment, activations: MemorySegment, deltas: MemorySegment,
                    elements: Int, beta: Double) {
        driver.launch(outputDelta, elements, pointer(targets), pointer(activations), pointer(deltas),
            int(elements), double(beta))
    }
    fun applyDerivative(deltas: MemorySegment, activations: MemorySegment, elements: Int, beta: Double) {
        driver.launch(applyDerivative, elements, pointer(deltas), pointer(activations), int(elements), double(beta))
    }
    fun reduceBiasGradient(deltas: MemorySegment, gradient: MemorySegment, batch: Int, width: Int) {
        driver.launch(reduceBias, width, pointer(deltas), pointer(gradient), int(batch), int(width))
    }
    fun momentumUpdate(values: MemorySegment, velocity: MemorySegment, gradient: MemorySegment,
                       elements: Int, momentum: Double, scale: Double) {
        driver.launch(momentumUpdate, elements, pointer(values), pointer(velocity), pointer(gradient),
            int(elements), double(momentum), double(scale))
    }
    fun gatherRowsFloat(source: MemorySegment, width: Int, order: MemorySegment, start: Int, count: Int,
                        destination: MemorySegment) {
        driver.launch(gatherRowsFloat, Math.multiplyExact(count, width),
            pointer(source), int(width), pointer(order), int(start), int(count), pointer(destination))
    }
    fun activateFloat(values: MemorySegment, biases: MemorySegment, batch: Int, width: Int, beta: Float,
                      mode: Neuro.SigmoidMode) {
        driver.launch(activateFloat, Math.multiplyExact(batch, width),
            pointer(values), pointer(biases), int(batch), int(width), float(beta),
            int(if (mode == Neuro.SigmoidMode.FAST) 1 else 0))
    }
    fun outputDeltaFloat(targets: MemorySegment, activations: MemorySegment, deltas: MemorySegment,
                         elements: Int, beta: Float) {
        driver.launch(outputDeltaFloat, elements, pointer(targets), pointer(activations), pointer(deltas),
            int(elements), float(beta))
    }
    fun applyDerivativeFloat(deltas: MemorySegment, activations: MemorySegment, elements: Int, beta: Float) {
        driver.launch(applyDerivativeFloat, elements, pointer(deltas), pointer(activations), int(elements), float(beta))
    }
    fun reduceBiasGradientFloat(deltas: MemorySegment, gradient: MemorySegment, batch: Int, width: Int) {
        driver.launch(reduceBiasFloat, width, pointer(deltas), pointer(gradient), int(batch), int(width))
    }
    fun momentumUpdateFloat(values: MemorySegment, velocity: MemorySegment, gradient: MemorySegment,
                            elements: Int, momentum: Float, scale: Float) {
        driver.launch(momentumUpdateFloat, elements, pointer(values), pointer(velocity), pointer(gradient),
            int(elements), float(momentum), float(scale))
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
