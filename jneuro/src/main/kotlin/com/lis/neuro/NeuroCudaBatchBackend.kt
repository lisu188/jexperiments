package com.lis.neuro

internal object NeuroCudaBatchBackend {
    private data class LayerBuffers(
        val weights: NeuroCudaBuffer,
        val biases: NeuroCudaBuffer,
        val weightVelocity: NeuroCudaBuffer,
        val biasVelocity: NeuroCudaBuffer,
        val weightGradient: NeuroCudaBuffer,
        val biasGradient: NeuroCudaBuffer
    )
    private class Resources(private val memory: NeuroCudaMemory) : AutoCloseable {
        private val buffers = ArrayList<NeuroCudaBuffer>()
        fun doubles(elements: Int): NeuroCudaBuffer = NeuroCudaBuffer.doubles(memory, elements).also { buffers += it }
        fun floats(elements: Int): NeuroCudaBuffer = NeuroCudaBuffer.floats(memory, elements).also { buffers += it }
        fun ints(elements: Int): NeuroCudaBuffer = NeuroCudaBuffer.ints(memory, elements).also { buffers += it }
        override fun close() {
            closeAll(listOf(AutoCloseable { memory.synchronize() }) + buffers.asReversed())
        }
    }

    fun train(network: Neuro, data: Neuro.PackedDataset, epochs: Int, batchSize: Int,
              precision: Neuro.TrainingPrecision, loader: NeuroLibraryLoader = NeuroNativeLibrary::open) {
        require(epochs >= 0 && batchSize > 0)
        if (epochs == 0) return
        check(data.size > 0) { "no training samples" }
        val resources = ArrayList<AutoCloseable>()
        var failure: Throwable? = null
        try {
            val runtime = NeuroCudaRuntime.create(loader).also { resources += it }
            check(runtime.deviceCount() > 0) { "No CUDA-capable devices are available" }
            runtime.setDevice(0)
            val cublas = NeuroCublas.create(loader).also { resources += it }
            val nvrtc = NeuroNvrtc.create(loader).also { resources += it }
            val driver = NeuroCudaDriver.create(loader).also { resources += it }
            val kernels = NeuroCudaKernels(nvrtc, driver, runtime.computeCapability(0)).also { resources += it }
            val batch = minOf(batchSize, data.size)
            when (precision) {
                Neuro.TrainingPrecision.FP64 -> trainResidentDouble(network, data, epochs, batch, runtime, cublas, kernels)
                Neuro.TrainingPrecision.FP32 -> trainResidentFloat(network, data, epochs, batch, runtime, cublas, kernels)
            }
        } catch (problem: Throwable) {
            failure = problem
            throw problem
        } finally {
            closeAll(resources.asReversed(), failure)
        }
    }

    private fun closeAll(resources: List<AutoCloseable>, original: Throwable? = null) {
        var failure = original
        for (resource in resources) {
            try { resource.close() } catch (problem: Throwable) {
                if (failure == null) failure = problem else failure.addSuppressed(problem)
            }
        }
        if (original == null) failure?.let { throw it }
    }

    private fun trainResidentFloat(network: Neuro, data: Neuro.PackedDataset, epochs: Int, batchSize: Int,
                                   runtime: NeuroCudaRuntime, cublas: NeuroCublas, kernels: NeuroCudaKernels) {
        val layers = network.backendLayers()
        val topology = network.topology()
        NeuroCudaPlan(topology, batchSize)
        Resources(runtime).use { resources ->
            fun floats(values: DoubleArray) = FloatArray(values.size) { values[it].toFloat() }
            fun doubles(values: FloatArray) = DoubleArray(values.size) { values[it].toDouble() }

            val allInputs = resources.floats(data.inputs.size).also { it.upload(floats(data.inputs)) }
            val allTargets = resources.floats(data.targets.size).also { it.upload(floats(data.targets)) }
            val order = resources.ints(data.size)
            val activations = Array(topology.size) { resources.floats(Math.multiplyExact(batchSize, topology[it])) }
            val batchTargets = resources.floats(Math.multiplyExact(batchSize, topology.last()))
            val deltas = Array(layers.size) { resources.floats(Math.multiplyExact(batchSize, layers[it].outputs)) }
            val layerBuffers = Array(layers.size) { index ->
                val layer = layers[index]
                LayerBuffers(
                    resources.floats(layer.weights.size).also { it.upload(floats(layer.weights)) },
                    resources.floats(layer.biases.size).also { it.upload(floats(layer.biases)) },
                    resources.floats(layer.weightVelocity.size).also { it.upload(floats(layer.weightVelocity)) },
                    resources.floats(layer.biasVelocity.size).also { it.upload(floats(layer.biasVelocity)) },
                    resources.floats(layer.weights.size),
                    resources.floats(layer.biases.size)
                )
            }
            val beta = network.hyperParameters().beta.toFloat()
            val momentum = network.hyperParameters().momentum.toFloat()
            val learningRate = network.hyperParameters().learningRate.toFloat()
            val mode = network.hyperParameters().sigmoidMode

            repeat(epochs) {
                order.upload(network.backendNextTrainingOrder(data.size))
                var start = 0
                while (start < data.size) {
                    val count = minOf(batchSize, data.size - start)
                    kernels.gatherRowsFloat(allInputs.pointer, data.inputSize, order.pointer, start, count, activations[0].pointer)
                    kernels.gatherRowsFloat(allTargets.pointer, data.outputSize, order.pointer, start, count, batchTargets.pointer)
                    for (layerIndex in layers.indices) {
                        val layer = layers[layerIndex]
                        cublas.forwardFloat(activations[layerIndex].pointer, layerBuffers[layerIndex].weights.pointer,
                            activations[layerIndex + 1].pointer, count, layer.inputs, layer.outputs)
                        kernels.activateFloat(activations[layerIndex + 1].pointer, layerBuffers[layerIndex].biases.pointer,
                            count, layer.outputs, beta, mode)
                    }
                    kernels.outputDeltaFloat(batchTargets.pointer, activations.last().pointer, deltas.last().pointer,
                        Math.multiplyExact(count, layers.last().outputs), beta)
                    for (layerIndex in layers.lastIndex - 1 downTo 0) {
                        val nextLayer = layers[layerIndex + 1]
                        val currentWidth = layers[layerIndex].outputs
                        cublas.backwardFloat(deltas[layerIndex + 1].pointer, layerBuffers[layerIndex + 1].weights.pointer,
                            deltas[layerIndex].pointer, count, currentWidth, nextLayer.outputs)
                        kernels.applyDerivativeFloat(deltas[layerIndex].pointer, activations[layerIndex + 1].pointer,
                            Math.multiplyExact(count, currentWidth), beta)
                    }
                    val scale = learningRate / count
                    for (layerIndex in layers.indices) {
                        val layer = layers[layerIndex]
                        val buffers = layerBuffers[layerIndex]
                        cublas.weightGradientFloat(activations[layerIndex].pointer, deltas[layerIndex].pointer,
                            buffers.weightGradient.pointer, count, layer.inputs, layer.outputs)
                        kernels.reduceBiasGradientFloat(deltas[layerIndex].pointer, buffers.biasGradient.pointer,
                            count, layer.outputs)
                        kernels.momentumUpdateFloat(buffers.weights.pointer, buffers.weightVelocity.pointer,
                            buffers.weightGradient.pointer, layer.weights.size, momentum, scale)
                        kernels.momentumUpdateFloat(buffers.biases.pointer, buffers.biasVelocity.pointer,
                            buffers.biasGradient.pointer, layer.biases.size, momentum, scale)
                    }
                    start += count
                }
                runtime.synchronize()
                val state = network.exportTrainingState()
                for (index in layers.indices) {
                    val layer = layers[index]
                    val buffers = layerBuffers[index]
                    doubles(buffers.weights.downloadFloats(layer.weights.size)).copyInto(state.weights[index])
                    doubles(buffers.biases.downloadFloats(layer.biases.size)).copyInto(state.biases[index])
                    doubles(buffers.weightVelocity.downloadFloats(layer.weightVelocity.size)).copyInto(state.weightVelocity[index])
                    doubles(buffers.biasVelocity.downloadFloats(layer.biasVelocity.size)).copyInto(state.biasVelocity[index])
                }
                network.commitDeviceEpoch(state)
            }
        }
    }

    private fun trainResidentDouble(network: Neuro, data: Neuro.PackedDataset, epochs: Int, batchSize: Int,
                              runtime: NeuroCudaRuntime, cublas: NeuroCublas, kernels: NeuroCudaKernels) {
        val layers = network.backendLayers()
        val topology = network.topology()
        NeuroCudaPlan(topology, batchSize)
        Resources(runtime).use { resources ->
            val allInputs = resources.doubles(data.inputs.size).also { it.upload(data.inputs) }
            val allTargets = resources.doubles(data.targets.size).also { it.upload(data.targets) }
            val order = resources.ints(data.size)
            val activations = Array(topology.size) { resources.doubles(Math.multiplyExact(batchSize, topology[it])) }
            val batchTargets = resources.doubles(Math.multiplyExact(batchSize, topology.last()))
            val deltas = Array(layers.size) { resources.doubles(Math.multiplyExact(batchSize, layers[it].outputs)) }
            val layerBuffers = Array(layers.size) { index ->
                val layer = layers[index]
                LayerBuffers(
                    resources.doubles(layer.weights.size).also { it.upload(layer.weights) },
                    resources.doubles(layer.biases.size).also { it.upload(layer.biases) },
                    resources.doubles(layer.weightVelocity.size).also { it.upload(layer.weightVelocity) },
                    resources.doubles(layer.biasVelocity.size).also { it.upload(layer.biasVelocity) },
                    resources.doubles(layer.weights.size),
                    resources.doubles(layer.biases.size)
                )
            }
            val beta = network.hyperParameters().beta
            val momentum = network.hyperParameters().momentum
            val learningRate = network.hyperParameters().learningRate
            val mode = network.hyperParameters().sigmoidMode

            repeat(epochs) {
                order.upload(network.backendNextTrainingOrder(data.size))
                var start = 0
                while (start < data.size) {
                    val count = minOf(batchSize, data.size - start)
                    kernels.gatherRows(allInputs.pointer, data.inputSize, order.pointer, start, count, activations[0].pointer)
                    kernels.gatherRows(allTargets.pointer, data.outputSize, order.pointer, start, count, batchTargets.pointer)
                    for (layerIndex in layers.indices) {
                        val layer = layers[layerIndex]
                        cublas.forward(activations[layerIndex].pointer, layerBuffers[layerIndex].weights.pointer,
                            activations[layerIndex + 1].pointer, count, layer.inputs, layer.outputs)
                        kernels.activate(activations[layerIndex + 1].pointer, layerBuffers[layerIndex].biases.pointer,
                            count, layer.outputs, beta, mode)
                    }
                    kernels.outputDelta(batchTargets.pointer, activations.last().pointer, deltas.last().pointer,
                        Math.multiplyExact(count, layers.last().outputs), beta)
                    for (layerIndex in layers.lastIndex - 1 downTo 0) {
                        val nextLayer = layers[layerIndex + 1]
                        val currentWidth = layers[layerIndex].outputs
                        cublas.backward(deltas[layerIndex + 1].pointer, layerBuffers[layerIndex + 1].weights.pointer,
                            deltas[layerIndex].pointer, count, currentWidth, nextLayer.outputs)
                        kernels.applyDerivative(deltas[layerIndex].pointer, activations[layerIndex + 1].pointer,
                            Math.multiplyExact(count, currentWidth), beta)
                    }
                    val scale = learningRate / count
                    for (layerIndex in layers.indices) {
                        val layer = layers[layerIndex]
                        val buffers = layerBuffers[layerIndex]
                        cublas.weightGradient(activations[layerIndex].pointer, deltas[layerIndex].pointer,
                            buffers.weightGradient.pointer, count, layer.inputs, layer.outputs)
                        kernels.reduceBiasGradient(deltas[layerIndex].pointer, buffers.biasGradient.pointer,
                            count, layer.outputs)
                        kernels.momentumUpdate(buffers.weights.pointer, buffers.weightVelocity.pointer,
                            buffers.weightGradient.pointer, layer.weights.size, momentum, scale)
                        kernels.momentumUpdate(buffers.biases.pointer, buffers.biasVelocity.pointer,
                            buffers.biasGradient.pointer, layer.biases.size, momentum, scale)
                    }
                    start += count
                }
                runtime.synchronize()
                val state = network.exportTrainingState()
                for (index in layers.indices) {
                    val layer = layers[index]
                    val buffers = layerBuffers[index]
                    buffers.weights.downloadDoubles(layer.weights.size).copyInto(state.weights[index])
                    buffers.biases.downloadDoubles(layer.biases.size).copyInto(state.biases[index])
                    buffers.weightVelocity.downloadDoubles(layer.weightVelocity.size).copyInto(state.weightVelocity[index])
                    buffers.biasVelocity.downloadDoubles(layer.biasVelocity.size).copyInto(state.biasVelocity[index])
                }
                network.commitDeviceEpoch(state)
            }
        }
    }
}
