package com.lis.neuro

import java.util.concurrent.ExecutionException
import java.util.concurrent.ForkJoinPool
import java.util.concurrent.Future

internal object NeuroCpuBatchTrainer {
    private class Workspace(topology: IntArray, layers: Array<Neuro.Layer>, batchSize: Int) {
        val activations = Array(topology.size) { DoubleArray(Math.multiplyExact(batchSize, topology[it])) }
        val targets = DoubleArray(Math.multiplyExact(batchSize, topology.last()))
        val deltas = Array(layers.size) { DoubleArray(Math.multiplyExact(batchSize, layers[it].outputs)) }
        val weightGradients = Array(layers.size) { DoubleArray(layers[it].weights.size) }
        val biasGradients = Array(layers.size) { DoubleArray(layers[it].biases.size) }
    }

    fun train(network: Neuro, data: Neuro.PackedDataset, epochs: Int, batchSize: Int, parallelism: Int) {
        val layers = network.backendLayers()
        val topology = network.topology()
        val workspace = Workspace(topology, layers, batchSize)
        val pool = if (parallelism > 1) ForkJoinPool(parallelism) else null
        try {
            repeat(epochs) {
                val order = network.backendNextTrainingOrder(data.size)
                var start = 0
                while (start < data.size) {
                    val count = minOf(batchSize, data.size - start)
                    gather(data, order, start, count, workspace, topology)
                    forward(network, layers, workspace, count, parallelism, pool)
                    backward(network, layers, workspace, count, parallelism, pool)
                    update(network, layers, workspace, count, parallelism, pool)
                    start += count
                }
                network.backendCompleteEpoch(data.size)
            }
        } finally {
            pool?.shutdown()
        }
    }

    private fun gather(data: Neuro.PackedDataset, order: IntArray, start: Int, count: Int,
                       workspace: Workspace, topology: IntArray) {
        val inputs = workspace.activations[0]
        for (position in 0 until count) {
            val sample = order[start + position]
            data.inputs.copyInto(inputs, position * topology[0], sample * data.inputSize,
                sample * data.inputSize + data.inputSize)
            data.targets.copyInto(workspace.targets, position * topology.last(), sample * data.outputSize,
                sample * data.outputSize + data.outputSize)
        }
    }

    private fun forward(network: Neuro, layers: Array<Neuro.Layer>, workspace: Workspace, count: Int,
                        parallelism: Int, pool: ForkJoinPool?) {
        val beta = network.hyperParameters().beta
        val mode = network.hyperParameters().sigmoidMode
        for (layerIndex in layers.indices) {
            val layer = layers[layerIndex]
            val source = workspace.activations[layerIndex]
            val destination = workspace.activations[layerIndex + 1]
            parallelFor(Math.multiplyExact(count, layer.outputs), parallelism, pool) { from, to ->
                for (index in from until to) {
                    val sample = index / layer.outputs
                    val output = index - sample * layer.outputs
                    val sourceOffset = sample * layer.inputs
                    val weightOffset = output * layer.inputs
                    var sum = layer.biases[output]
                    for (input in 0 until layer.inputs) {
                        sum = Math.fma(source[sourceOffset + input], layer.weights[weightOffset + input], sum)
                    }
                    destination[index] = Neuro.activate(sum * beta, mode)
                }
            }
        }
    }

    private fun backward(network: Neuro, layers: Array<Neuro.Layer>, workspace: Workspace, count: Int,
                         parallelism: Int, pool: ForkJoinPool?) {
        val beta = network.hyperParameters().beta
        val outputActivation = workspace.activations.last()
        val outputDelta = workspace.deltas.last()
        parallelFor(Math.multiplyExact(count, layers.last().outputs), parallelism, pool) { from, to ->
            for (index in from until to) {
                val activation = outputActivation[index]
                outputDelta[index] = (workspace.targets[index] - activation) * beta * activation * (1.0 - activation)
            }
        }
        for (layerIndex in layers.lastIndex - 1 downTo 0) {
            val currentWidth = layers[layerIndex].outputs
            val nextLayer = layers[layerIndex + 1]
            val currentActivation = workspace.activations[layerIndex + 1]
            val currentDelta = workspace.deltas[layerIndex]
            val nextDelta = workspace.deltas[layerIndex + 1]
            parallelFor(Math.multiplyExact(count, currentWidth), parallelism, pool) { from, to ->
                for (index in from until to) {
                    val sample = index / currentWidth
                    val current = index - sample * currentWidth
                    val nextOffset = sample * nextLayer.outputs
                    var sum = 0.0
                    for (output in 0 until nextLayer.outputs) {
                        sum = Math.fma(nextDelta[nextOffset + output],
                            nextLayer.weights[output * nextLayer.inputs + current], sum)
                    }
                    val activation = currentActivation[index]
                    currentDelta[index] = sum * beta * activation * (1.0 - activation)
                }
            }
        }
    }

    private fun update(network: Neuro, layers: Array<Neuro.Layer>, workspace: Workspace, count: Int,
                       parallelism: Int, pool: ForkJoinPool?) {
        val learningRate = network.hyperParameters().learningRate
        val momentum = network.hyperParameters().momentum
        val scale = learningRate / count
        for (layerIndex in layers.indices) {
            val layer = layers[layerIndex]
            val source = workspace.activations[layerIndex]
            val delta = workspace.deltas[layerIndex]
            val weightGradient = workspace.weightGradients[layerIndex]
            val biasGradient = workspace.biasGradients[layerIndex]
            parallelFor(layer.weights.size, parallelism, pool) { from, to ->
                for (weightIndex in from until to) {
                    val output = weightIndex / layer.inputs
                    val input = weightIndex - output * layer.inputs
                    var gradient = 0.0
                    for (sample in 0 until count) {
                        gradient = Math.fma(delta[sample * layer.outputs + output],
                            source[sample * layer.inputs + input], gradient)
                    }
                    weightGradient[weightIndex] = gradient
                    val velocity = Math.fma(momentum, layer.weightVelocity[weightIndex], scale * gradient)
                    layer.weightVelocity[weightIndex] = velocity
                    layer.weights[weightIndex] += velocity
                }
            }
            parallelFor(layer.outputs, parallelism, pool) { from, to ->
                for (output in from until to) {
                    var gradient = 0.0
                    for (sample in 0 until count) gradient += delta[sample * layer.outputs + output]
                    biasGradient[output] = gradient
                    val velocity = Math.fma(momentum, layer.biasVelocity[output], scale * gradient)
                    layer.biasVelocity[output] = velocity
                    layer.biases[output] += velocity
                }
            }
        }
    }

    private fun parallelFor(size: Int, parallelism: Int, pool: ForkJoinPool?, action: (Int, Int) -> Unit) {
        if (size == 0) return
        if (pool == null || parallelism <= 1 || size < 256) {
            action(0, size)
            return
        }
        val workers = minOf(parallelism, size)
        val futures = Array<Future<*>>(workers) { worker ->
            val from = worker * size / workers
            val to = (worker + 1) * size / workers
            pool.submit { action(from, to) }
        }
        for (future in futures) {
            try {
                future.get()
            } catch (exception: InterruptedException) {
                Thread.currentThread().interrupt()
                throw IllegalStateException("parallel matrix training interrupted", exception)
            } catch (exception: ExecutionException) {
                throw IllegalStateException("parallel matrix training failed", exception.cause)
            }
        }
    }
}
