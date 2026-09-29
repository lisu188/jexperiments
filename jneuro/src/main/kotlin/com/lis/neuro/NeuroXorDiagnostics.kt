package com.lis.neuro

import java.awt.image.BufferedImage
import java.awt.image.DataBufferInt
import kotlin.math.abs

internal object NeuroXorDiagnostics {
    data class Boundary(val x1: Double, val y1: Double, val x2: Double, val y2: Double)

    class Probe(hiddenLayers: Array<DoubleArray>, private val result: Double,
                contributions: DoubleArray, private val preActivation: Double) {
        private val layers = Array(hiddenLayers.size) { hiddenLayers[it].copyOf() }
        private val parts = contributions.copyOf()
        fun hiddenLayers(): Array<DoubleArray> = Array(layers.size) { layers[it].copyOf() }
        fun hiddenLayer(index: Int): DoubleArray = layers[index].copyOf()
        fun hidden(): DoubleArray = layers.last().copyOf()
        fun output(): Double = result
        fun contributions(): DoubleArray = parts.copyOf()
        fun outputPreActivation(): Double = preActivation
    }

    class Snapshot(epoch: Int, error: Double, private val beta: Double, private val mode: Neuro.SigmoidMode,
                   topology: IntArray, weights: Array<DoubleArray>, biases: Array<DoubleArray>) {
        private val epochValue = epoch
        private val errorValue = error
        private val shape = topology.copyOf()
        private val weights = Array(weights.size) { weights[it].copyOf() }
        private val biases = Array(biases.size) { biases[it].copyOf() }
        private val count = weights.sumOf { it.size } + biases.sumOf { it.size }

        fun epoch(): Int = epochValue
        fun error(): Double = errorValue
        fun topology(): IntArray = shape.copyOf()
        fun hiddenLayerCount(): Int = shape.size - 2
        fun hiddenCount(): Int = shape[1]
        fun layerCount(): Int = weights.size
        fun layerInputCount(layer: Int): Int = shape[layer]
        fun layerOutputCount(layer: Int): Int = shape[layer + 1]
        fun weight(layer: Int, output: Int, input: Int): Double = weights[layer][output * shape[layer] + input]
        fun bias(layer: Int, output: Int): Double = biases[layer][output]
        fun inputWeight(hidden: Int, input: Int): Double = weight(0, hidden, input)
        fun hiddenBias(hidden: Int): Double = bias(0, hidden)
        fun outputWeight(hidden: Int): Double = weight(weights.lastIndex, 0, hidden)
        fun outputBias(): Double = bias(weights.lastIndex, 0)
        fun parameterCount(): Int = count
        fun parameterOffset(layer: Int): Int {
            require(layer in weights.indices) { "layer index out of range" }
            var offset = 0
            for (previous in 0 until layer) offset += weights[previous].size + biases[previous].size
            return offset
        }
        fun parameters(): DoubleArray {
            val result = DoubleArray(count)
            var offset = 0
            for (layer in weights.indices) {
                weights[layer].copyInto(result, offset)
                offset += weights[layer].size
                biases[layer].copyInto(result, offset)
                offset += biases[layer].size
            }
            return result
        }
        fun newWorkspace(): Array<DoubleArray> = Array(shape.size) { DoubleArray(shape[it]) }
        fun evaluate(x: Double, y: Double, workspace: Array<DoubleArray>, throughLayer: Int = weights.lastIndex): Double {
            workspace[0][0] = x
            workspace[0][1] = y
            for (layer in 0..throughLayer) {
                val source = workspace[layer]
                val destination = workspace[layer + 1]
                for (output in destination.indices) {
                    var sum = biases[layer][output]
                    val offset = output * source.size
                    for (input in source.indices) sum = Math.fma(source[input], weights[layer][offset + input], sum)
                    destination[output] = Neuro.activate(sum * beta, mode)
                }
            }
            return workspace[throughLayer + 1][0]
        }
    }

    fun capture(network: Neuro, epoch: Int, error: Double): Snapshot {
        val topology = network.topology()
        require(topology.size >= 3 && topology[0] == 2 && topology.last() == 1) { "diagnostics require topology 2-hidden...-1" }
        return Snapshot(epoch, error, network.hyperParameters().beta, network.hyperParameters().sigmoidMode,
            topology, Array(topology.size - 1) { network.backendWeights(it) },
            Array(topology.size - 1) { network.backendBiases(it) })
    }

    fun probe(snapshot: Snapshot, x: Double, y: Double): Probe {
        require(x.isFinite() && y.isFinite()) { "probe coordinates must be finite" }
        val workspace = snapshot.newWorkspace()
        val output = snapshot.evaluate(x, y, workspace)
        val lastHidden = workspace[workspace.lastIndex - 1]
        val contributions = DoubleArray(lastHidden.size) { lastHidden[it] * snapshot.outputWeight(it) }
        var preActivation = snapshot.outputBias()
        for (part in contributions) preActivation += part
        return Probe(Array(snapshot.hiddenLayerCount()) { workspace[it + 1] }, output, contributions, preActivation)
    }

    fun renderOutputMap(snapshot: Snapshot, size: Int): BufferedImage {
        validateSize(size)
        val image = BufferedImage(size, size, BufferedImage.TYPE_INT_RGB)
        val pixels = (image.raster.dataBuffer as DataBufferInt).data
        val workspace = snapshot.newWorkspace()
        val scale = 1.0 / (size - 1)
        for (row in 0 until size) {
            for (column in 0 until size) {
                pixels[row * size + column] = NeuroXorGrid.grayRgb(snapshot.evaluate(column * scale, 1.0 - row * scale, workspace))
            }
        }
        return image
    }

    fun renderDifferenceMap(before: Snapshot, after: Snapshot, size: Int): BufferedImage {
        validateSize(size)
        val image = BufferedImage(size, size, BufferedImage.TYPE_INT_RGB)
        val pixels = (image.raster.dataBuffer as DataBufferInt).data
        val first = before.newWorkspace()
        val second = after.newWorkspace()
        val scale = 1.0 / (size - 1)
        for (row in 0 until size) {
            for (column in 0 until size) {
                val x = column * scale
                val y = 1.0 - row * scale
                pixels[row * size + column] = differenceRgb(after.evaluate(x, y, second) - before.evaluate(x, y, first))
            }
        }
        return image
    }

    fun differenceRgb(difference: Double): Int {
        require(difference.isFinite()) { "difference must be finite" }
        val magnitude = minOf(1.0, abs(difference) * 4.0)
        val neutral = 45
        val channel = Math.round(neutral + magnitude * 190.0).toInt()
        return if (difference >= 0.0) (channel shl 16) or (neutral shl 8) or neutral
        else (neutral shl 16) or (neutral shl 8) or channel
    }

    fun renderHiddenMaps(snapshot: Snapshot, size: Int): Array<BufferedImage> =
        renderHiddenMaps(snapshot, size, 0, 0, minOf(snapshot.hiddenCount(), 12))

    fun renderHiddenMaps(snapshot: Snapshot, size: Int, layer: Int, start: Int, count: Int): Array<BufferedImage> {
        validateSize(size)
        require(layer in 0 until snapshot.hiddenLayerCount()) { "hidden layer index out of range" }
        require(start >= 0 && count > 0 && start.toLong() + count <= snapshot.layerOutputCount(layer)) { "hidden neuron range is invalid" }
        val images = Array(count) { BufferedImage(size, size, BufferedImage.TYPE_INT_RGB) }
        val pixels = Array(count) { (images[it].raster.dataBuffer as DataBufferInt).data }
        val workspace = snapshot.newWorkspace()
        val scale = 1.0 / (size - 1)
        for (row in 0 until size) {
            for (column in 0 until size) {
                snapshot.evaluate(column * scale, 1.0 - row * scale, workspace, layer)
                for (neuron in 0 until count) pixels[neuron][row * size + column] = NeuroXorGrid.grayRgb(workspace[layer + 1][start + neuron])
            }
        }
        return images
    }

    fun boundary(snapshot: Snapshot, neuron: Int): Boundary? {
        require(neuron in 0 until snapshot.hiddenCount()) { "hidden neuron index out of range" }
        val wx = snapshot.inputWeight(neuron, 0)
        val wy = snapshot.inputWeight(neuron, 1)
        val bias = snapshot.hiddenBias(neuron)
        val points = ArrayList<Pair<Double, Double>>(4)
        if (abs(wy) > 1.0e-12) {
            addIntersection(points, 0.0, -bias / wy)
            addIntersection(points, 1.0, -(wx + bias) / wy)
        }
        if (abs(wx) > 1.0e-12) {
            addIntersection(points, -bias / wx, 0.0)
            addIntersection(points, -(wy + bias) / wx, 1.0)
        }
        if (points.size < 2) return null
        return Boundary(points[0].first, points[0].second, points[1].first, points[1].second)
    }
    fun maxAbsWeight(snapshot: Snapshot): Double {
        var max = 0.0
        for (layer in 0 until snapshot.layerCount()) {
            for (output in 0 until snapshot.layerOutputCount(layer)) {
                for (input in 0 until snapshot.layerInputCount(layer)) max = maxOf(max, abs(snapshot.weight(layer, output, input)))
            }
        }
        return max
    }
    fun weightNorm(snapshot: Snapshot, layer: Int): Double {
        require(layer in 0 until snapshot.layerCount()) { "layer index out of range" }
        var sum = 0.0
        for (output in 0 until snapshot.layerOutputCount(layer)) {
            for (input in 0 until snapshot.layerInputCount(layer)) {
                val value = snapshot.weight(layer, output, input)
                sum += value * value
            }
        }
        return Math.sqrt(sum)
    }
    fun biasNorm(snapshot: Snapshot, layer: Int): Double {
        require(layer in 0 until snapshot.layerCount()) { "layer index out of range" }
        var sum = 0.0
        for (output in 0 until snapshot.layerOutputCount(layer)) {
            val value = snapshot.bias(layer, output)
            sum += value * value
        }
        return Math.sqrt(sum)
    }
    private fun validateSize(size: Int) = require(size in 2..1024) { "map size must be in [2, 1024]" }
    private fun addIntersection(points: MutableList<Pair<Double, Double>>, x: Double, y: Double) {
        if (x < -1.0e-9 || x > 1.0 + 1.0e-9 || y < -1.0e-9 || y > 1.0 + 1.0e-9) return
        val point = x.coerceIn(0.0, 1.0) to y.coerceIn(0.0, 1.0)
        if (points.none { abs(it.first - point.first) < 1.0e-9 && abs(it.second - point.second) < 1.0e-9 }) points.add(point)
    }
}
