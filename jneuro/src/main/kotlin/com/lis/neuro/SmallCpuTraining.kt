package com.lis.neuro

import jdk.incubator.vector.DoubleVector
import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorSpecies

/** The deliberately bounded family for the specialized streaming kernels. */
internal object SmallNetworkShape {
    fun supports(topology: IntArray): Boolean = topology.size in 3..6 && topology.first() == 2 &&
        topology.last() == 1 && (1 until topology.lastIndex).all { topology[it] == 4 || topology[it] == 8 || topology[it] == 16 }
}

internal fun validateSmallState(state: NeuroTrainingState) {
    require(SmallNetworkShape.supports(state.topology)) { "SMALL requires 2 inputs, 1 output and 1..4 hidden layers of 4, 8 or 16 neurons" }
    val layers = state.topology.size - 1
    require(state.weights.size == layers && state.biases.size == layers &&
        state.weightVelocity.size == layers && state.biasVelocity.size == layers) { "Invalid state layer count" }
    for (layer in 0 until layers) {
        val weights = state.topology[layer] * state.topology[layer + 1]
        val outputs = state.topology[layer + 1]
        require(state.weights[layer].size == weights && state.weightVelocity[layer].size == weights &&
            state.biases[layer].size == outputs && state.biasVelocity[layer].size == outputs) { "Invalid layer state shape" }
        require(state.weights[layer].all { it.isFinite() } && state.biases[layer].all { it.isFinite() } &&
            state.weightVelocity[layer].all { it.isFinite() } && state.biasVelocity[layer].all { it.isFinite() }) { "Non-finite initial state" }
    }
    require(state.inputs.size % 2 == 0 && state.targets.size == state.samples) { "Invalid training dataset" }
    require(state.inputs.all { it.isFinite() } && state.targets.all { it.isFinite() }) { "Non-finite training dataset" }
}

internal fun validateSmallOrders(orders: Array<IntArray>, samples: Int) {
    val seen = BooleanArray(samples)
    for (order in orders) {
        require(order.size == samples) { "Each epoch order must contain every sample" }
        seen.fill(false)
        for (sample in order) {
            require(sample in 0 until samples && !seen[sample]) { "Epoch order must be a permutation" }
            seen[sample] = true
        }
    }
}

internal fun smallVectorBits(parameters: Neuro.HyperParameters, vectorBits: Int): Int {
    require(vectorBits == 0 || vectorBits == 128 || vectorBits == 256) { "vectorBits must be 0, 128 or 256" }
    val available = DoubleVector.SPECIES_PREFERRED.vectorBitSize()
    return if (parameters.kernel == Neuro.Kernel.SCALAR || available < 128) 0 else minOf(vectorBits, available)
}

/** Owns primitive compute state; snapshots are allocated only at publication boundaries. */
internal class SmallCpuTraining(
    state: NeuroTrainingState,
    parameters: Neuro.HyperParameters,
    precision: Neuro.TrainingPrecision,
    vectorBits: Int = 256
) : SmallTrainingKernel {
    private val bits = smallVectorBits(parameters, vectorBits).let { selected ->
        if (precision == Neuro.TrainingPrecision.FP32 && selected == 256 &&
            state.topology.drop(1).dropLast(1).all { it == 4 }) 128 else selected
    }
    private val samples: Int
    private val compute: SmallCpuCompute
    private var closed = false
    override val info = TrainingDeviceInfo(TrainingBackend.CPU, "CPU", "jvm-cpu",
        precision.name, "small-cpu-v1", engine = TrainingEngine.SMALL, simdBits = bits, sigmoid = parameters.sigmoidMode.name)

    init {
        validateSmallState(state)
        samples = state.samples
        compute = if (precision == Neuro.TrainingPrecision.FP32) SmallFloatCompute(state, parameters, bits)
            else SmallDoubleCompute(state, parameters, bits)
    }

    override fun train(orders: Array<IntArray>, batchSize: Int, online: Boolean): NeuroTrainingState {
        check(!closed) { "Small CPU kernel is closed" }
        require(batchSize > 0) { "batchSize must be > 0" }
        validateSmallOrders(orders, samples)
        compute.train(orders, batchSize, online)
        return compute.snapshot()
    }

    override fun close() { closed = true }
}

private interface SmallCpuCompute {
    fun train(orders: Array<IntArray>, batchSize: Int, online: Boolean)
    fun snapshot(): NeuroTrainingState
}

private class SmallDoubleCompute(state: NeuroTrainingState, private val parameters: Neuro.HyperParameters, private val bits: Int) : SmallCpuCompute {
    private val topology = state.topology.copyOf()
    private val originalInputs = state.inputs.copyOf()
    private val originalTargets = state.targets.copyOf()
    private val weights = Array(state.weights.size) { i -> smallDoubles(state.weights[i]) }
    private val biases = Array(state.biases.size) { i -> smallDoubles(state.biases[i]) }
    private val velocity = Array(state.weightVelocity.size) { i -> smallDoubles(state.weightVelocity[i]) }
    private val biasVelocity = Array(state.biasVelocity.size) { i -> smallDoubles(state.biasVelocity[i]) }
    private val inputs = smallDoubles(state.inputs)
    private val targets = smallDoubles(state.targets)
    private val activations = Array(topology.size) { DoubleArray(topology[it]) }
    private val deltas = Array(weights.size) { DoubleArray(topology[it + 1]) }
    private val gradients = Array(weights.size) { DoubleArray(weights[it].size) }
    private val biasGradients = Array(weights.size) { DoubleArray(biases[it].size) }
    private val transposed = Array(weights.size) { DoubleArray(weights[it].size) }
    private val beta = parameters.beta
    private val rate = parameters.learningRate
    private val momentum = parameters.momentum
    private val activation = SmallDoubleActivation(parameters.sigmoidMode, bits, 16)

    init {
        require(beta.isFinite() && rate.isFinite() && momentum.isFinite()) { "Hyperparameters exceed compute precision" }
        require(inputs.all { it.isFinite() } && targets.all { it.isFinite() } &&
            weights.all { row -> row.all { it.isFinite() } } && biases.all { row -> row.all { it.isFinite() } } &&
            velocity.all { row -> row.all { it.isFinite() } } && biasVelocity.all { row -> row.all { it.isFinite() } }) {
            "State exceeds compute precision"
        }
        transpose()
    }

    private fun transpose() {
        for (layer in weights.indices) {
            val width = topology[layer + 1]
            for (input in 0 until topology[layer]) for (output in 0 until width)
                transposed[layer][input * width + output] = weights[layer][output * topology[layer] + input]
        }
    }

    override fun train(orders: Array<IntArray>, batchSize: Int, online: Boolean) {
        for (order in orders) {
            var start = 0
            while (start < order.size) {
                val count = if (online) 1 else minOf(batchSize, order.size - start)
                if (!online) {
                    gradients.forEach { it.fill(0.0) }
                    biasGradients.forEach { it.fill(0.0) }
                }
                for (position in start until start + count) {
                    val sample = order[position]
                    inputs.copyInto(activations[0], 0, sample * 2, sample * 2 + 2)
                    forward()
                    backward(targets[sample], online)
                    if (online) updateOnline() else accumulate()
                }
                if (!online) updateBatch(count)
                transpose()
                start += count
            }
        }
    }

    private fun forward() {
        for (layer in weights.indices) {
            val source = activations[layer]
            val destination = activations[layer + 1]
            val width = destination.size
            var output = smallDoubleVectors(width, bits) { laneOffset, species ->
                var sum = DoubleVector.fromArray(species, biases[layer], laneOffset)
                smallInputOffsets(source.size) { input ->
                    val w = DoubleVector.fromArray(species, transposed[layer], input * width + laneOffset)
                    sum = DoubleVector.broadcast(species, source[input]).fma(w, sum)
                }
                sum.mul(beta).intoArray(destination, laneOffset)
            }
            while (output < width) {
                var sum = biases[layer][output]
                smallInputOffsets(source.size) { input ->
                    sum = Math.fma(source[input], weights[layer][output * source.size + input], sum)
                }
                destination[output] = sum * beta
                output++
            }
            activation.apply(destination, width)
        }
    }

    private fun backward(target: Double, online: Boolean) {
        val last = activations.last()[0]
        deltas.last()[0] = (target - last) * beta * last * (1.0 - last)
        for (layer in weights.lastIndex - 1 downTo 0) {
            val a = activations[layer + 1]
            val delta = deltas[layer]
            val next = deltas[layer + 1]
            var input = smallDoubleVectors(a.size, bits) { laneOffset, species ->
                var sum = if (online) DoubleVector.fromArray(species, weights[layer + 1], laneOffset).mul(next[0])
                    else DoubleVector.zero(species)
                for (output in (if (online) 1 else 0) until next.size) {
                    val w = DoubleVector.fromArray(species, weights[layer + 1], output * a.size + laneOffset)
                    sum = DoubleVector.broadcast(species, next[output]).fma(w, sum)
                }
                val value = DoubleVector.fromArray(species, a, laneOffset)
                val complement = DoubleVector.broadcast(species, 1.0).sub(value)
                val derivative = if (online) sum.mul(value.mul(beta).mul(complement))
                    else sum.mul(beta).mul(value).mul(complement)
                derivative.intoArray(delta, laneOffset)
            }
            while (input < a.size) {
                var sum = if (online) next[0] * weights[layer + 1][input] else 0.0
                for (output in (if (online) 1 else 0) until next.size)
                    sum = Math.fma(next[output], weights[layer + 1][output * a.size + input], sum)
                val value = a[input]
                delta[input] = if (online) sum * (beta * value * (1.0 - value))
                    else sum * beta * value * (1.0 - value)
                input++
            }
        }
    }

    private fun accumulate() {
        for (layer in weights.indices) {
            val source = activations[layer]
            for (output in deltas[layer].indices) {
                val delta = deltas[layer][output]
                biasGradients[layer][output] += delta
                val offset = output * source.size
                var input = smallDoubleVectors(source.size, bits) { laneOffset, species ->
                    DoubleVector.fromArray(species, source, laneOffset).fma(DoubleVector.broadcast(species, delta),
                        DoubleVector.fromArray(species, gradients[layer], offset + laneOffset)).intoArray(gradients[layer], offset + laneOffset)
                }
                while (input < source.size) {
                    gradients[layer][offset + input] = Math.fma(source[input], delta, gradients[layer][offset + input])
                    input++
                }
            }
        }
    }

    private fun updateOnline() {
        for (layer in weights.indices) {
            val source = activations[layer]
            for (output in deltas[layer].indices) {
                val scale = rate * deltas[layer][output]
                val offset = output * source.size
                var input = smallDoubleVectors(source.size, bits) { laneOffset, species ->
                    val v = DoubleVector.fromArray(species, velocity[layer], offset + laneOffset)
                    val next = DoubleVector.broadcast(species, momentum).fma(v, DoubleVector.fromArray(species, source, laneOffset).mul(scale))
                    next.intoArray(velocity[layer], offset + laneOffset)
                    DoubleVector.fromArray(species, weights[layer], offset + laneOffset).add(next).intoArray(weights[layer], offset + laneOffset)
                }
                while (input < source.size) {
                    val index = offset + input
                    val next = Math.fma(momentum, velocity[layer][index], scale * source[input])
                    velocity[layer][index] = next
                    weights[layer][index] += next
                    input++
                }
                val next = Math.fma(momentum, biasVelocity[layer][output], scale)
                biasVelocity[layer][output] = next
                biases[layer][output] += next
            }
        }
    }

    private fun updateBatch(count: Int) {
        val scale = rate / count
        for (layer in weights.indices) {
            update(weights[layer], velocity[layer], gradients[layer], scale)
            update(biases[layer], biasVelocity[layer], biasGradients[layer], scale)
        }
    }

    private fun update(values: DoubleArray, velocities: DoubleArray, gradients: DoubleArray, scale: Double) {
        var index = smallDoubleVectors(values.size, bits) { laneOffset, species ->
            val next = DoubleVector.broadcast(species, momentum).fma(DoubleVector.fromArray(species, velocities, laneOffset),
                DoubleVector.fromArray(species, gradients, laneOffset).mul(scale))
            next.intoArray(velocities, laneOffset)
            DoubleVector.fromArray(species, values, laneOffset).add(next).intoArray(values, laneOffset)
        }
        while (index < values.size) {
            val next = Math.fma(momentum, velocities[index], scale * gradients[index])
            velocities[index] = next
            values[index] += next
            index++
        }
    }

    override fun snapshot() = NeuroTrainingState(topology.copyOf(),
        Array(weights.size) { i -> smallDoubles(weights[i]) }, Array(biases.size) { i -> smallDoubles(biases[i]) },
        Array(velocity.size) { i -> smallDoubles(velocity[i]) }, Array(biasVelocity.size) { i -> smallDoubles(biasVelocity[i]) },
        originalInputs.copyOf(), originalTargets.copyOf())
}

/** EXACT keeps Math.exp per lane. FAST vectorizes the explicitly selected range-reduced polynomial. */
internal class SmallDoubleActivation(private val mode: Neuro.SigmoidMode, private val bits: Int, capacity: Int) {
    private val remainder = DoubleArray(capacity)
    private val exponents = IntArray(capacity)
    private val polynomial = DoubleArray(capacity)

    fun apply(values: DoubleArray, length: Int) {
        if (mode == Neuro.SigmoidMode.EXACT) {
            for (i in 0 until length) values[i] = 1.0 / (1.0 + Math.exp(-values[i]))
            return
        }
        for (i in 0 until length) {
            val negative = -kotlin.math.abs(values[i])
            val exponent = (negative * 1.4426950408889634).toInt()
            exponents[i] = exponent
            remainder[i] = if (negative <= -745.0) 0.0 else negative - exponent * 0.6931471805599453
        }
        var i = smallDoubleVectors(length, bits) { laneOffset, species ->
            val r = DoubleVector.fromArray(species, remainder, laneOffset)
            val p = r.mul(0.008333333333333333).add(0.041666666666666664).mul(r).add(0.16666666666666666).mul(r).add(0.5)
            r.mul(r).mul(p).add(r.add(1.0)).intoArray(polynomial, laneOffset)
        }
        while (i < length) {
            val r = remainder[i]
            polynomial[i] = 1.0 + r + r * r * (0.5 + r * (0.16666666666666666 + r * (0.041666666666666664 + r * 0.008333333333333333)))
            i++
        }
        for (index in 0 until length) {
            val value = values[index]
            val exp = if (-kotlin.math.abs(value) <= -745.0) 0.0 else Math.scalb(polynomial[index], exponents[index])
            values[index] = if (value >= 0.0) 1.0 / (1.0 + exp) else exp / (1.0 + exp)
        }
    }
}

private class SmallFloatCompute(state: NeuroTrainingState, private val parameters: Neuro.HyperParameters, private val bits: Int) : SmallCpuCompute {
    private val topology = state.topology.copyOf()
    private val originalInputs = state.inputs.copyOf()
    private val originalTargets = state.targets.copyOf()
    private val weights = Array(state.weights.size) { i -> smallFloats(state.weights[i]) }
    private val biases = Array(state.biases.size) { i -> smallFloats(state.biases[i]) }
    private val velocity = Array(state.weightVelocity.size) { i -> smallFloats(state.weightVelocity[i]) }
    private val biasVelocity = Array(state.biasVelocity.size) { i -> smallFloats(state.biasVelocity[i]) }
    private val inputs = smallFloats(state.inputs)
    private val targets = smallFloats(state.targets)
    private val activations = Array(topology.size) { FloatArray(topology[it]) }
    private val deltas = Array(weights.size) { FloatArray(topology[it + 1]) }
    private val gradients = Array(weights.size) { FloatArray(weights[it].size) }
    private val biasGradients = Array(weights.size) { FloatArray(biases[it].size) }
    private val transposed = Array(weights.size) { FloatArray(weights[it].size) }
    private val beta = parameters.beta.toFloat()
    private val rate = parameters.learningRate.toFloat()
    private val momentum = parameters.momentum.toFloat()
    private val activation = SmallFloatActivation(parameters.sigmoidMode, bits, 16)

    init {
        require(beta.isFinite() && rate.isFinite() && momentum.isFinite()) { "Hyperparameters exceed compute precision" }
        require(inputs.all { it.isFinite() } && targets.all { it.isFinite() } &&
            weights.all { row -> row.all { it.isFinite() } } && biases.all { row -> row.all { it.isFinite() } } &&
            velocity.all { row -> row.all { it.isFinite() } } && biasVelocity.all { row -> row.all { it.isFinite() } }) {
            "State exceeds compute precision"
        }
        transpose()
    }

    private fun transpose() {
        for (layer in weights.indices) {
            val width = topology[layer + 1]
            for (input in 0 until topology[layer]) for (output in 0 until width)
                transposed[layer][input * width + output] = weights[layer][output * topology[layer] + input]
        }
    }

    override fun train(orders: Array<IntArray>, batchSize: Int, online: Boolean) {
        for (order in orders) {
            var start = 0
            while (start < order.size) {
                val count = if (online) 1 else minOf(batchSize, order.size - start)
                if (!online) {
                    gradients.forEach { it.fill(0.0f) }
                    biasGradients.forEach { it.fill(0.0f) }
                }
                for (position in start until start + count) {
                    val sample = order[position]
                    inputs.copyInto(activations[0], 0, sample * 2, sample * 2 + 2)
                    forward()
                    backward(targets[sample], online)
                    if (online) updateOnline() else accumulate()
                }
                if (!online) updateBatch(count)
                transpose()
                start += count
            }
        }
    }

    private fun forward() {
        for (layer in weights.indices) {
            val source = activations[layer]
            val destination = activations[layer + 1]
            val width = destination.size
            var output = smallFloatVectors(width, bits) { laneOffset, species ->
                var sum = FloatVector.fromArray(species, biases[layer], laneOffset)
                smallInputOffsets(source.size) { input ->
                    val w = FloatVector.fromArray(species, transposed[layer], input * width + laneOffset)
                    sum = FloatVector.broadcast(species, source[input]).fma(w, sum)
                }
                sum.mul(beta).intoArray(destination, laneOffset)
            }
            while (output < width) {
                var sum = biases[layer][output]
                smallInputOffsets(source.size) { input ->
                    sum = Math.fma(source[input], weights[layer][output * source.size + input], sum)
                }
                destination[output] = sum * beta
                output++
            }
            activation.apply(destination, width)
        }
    }

    private fun backward(target: Float, online: Boolean) {
        val last = activations.last()[0]
        deltas.last()[0] = (target - last) * beta * last * (1.0f - last)
        for (layer in weights.lastIndex - 1 downTo 0) {
            val a = activations[layer + 1]
            val delta = deltas[layer]
            val next = deltas[layer + 1]
            var input = smallFloatVectors(a.size, bits) { laneOffset, species ->
                var sum = if (online) FloatVector.fromArray(species, weights[layer + 1], laneOffset).mul(next[0])
                    else FloatVector.zero(species)
                for (output in (if (online) 1 else 0) until next.size) {
                    val w = FloatVector.fromArray(species, weights[layer + 1], output * a.size + laneOffset)
                    sum = FloatVector.broadcast(species, next[output]).fma(w, sum)
                }
                val value = FloatVector.fromArray(species, a, laneOffset)
                val complement = FloatVector.broadcast(species, 1.0f).sub(value)
                val derivative = if (online) sum.mul(value.mul(beta).mul(complement))
                    else sum.mul(beta).mul(value).mul(complement)
                derivative.intoArray(delta, laneOffset)
            }
            while (input < a.size) {
                var sum = if (online) next[0] * weights[layer + 1][input] else 0.0f
                for (output in (if (online) 1 else 0) until next.size)
                    sum = Math.fma(next[output], weights[layer + 1][output * a.size + input], sum)
                val value = a[input]
                delta[input] = if (online) sum * (beta * value * (1.0f - value))
                    else sum * beta * value * (1.0f - value)
                input++
            }
        }
    }

    private fun accumulate() {
        for (layer in weights.indices) {
            val source = activations[layer]
            for (output in deltas[layer].indices) {
                val delta = deltas[layer][output]
                biasGradients[layer][output] += delta
                val offset = output * source.size
                var input = smallFloatVectors(source.size, bits) { laneOffset, species ->
                    FloatVector.fromArray(species, source, laneOffset).fma(FloatVector.broadcast(species, delta),
                        FloatVector.fromArray(species, gradients[layer], offset + laneOffset)).intoArray(gradients[layer], offset + laneOffset)
                }
                while (input < source.size) {
                    gradients[layer][offset + input] = Math.fma(source[input], delta, gradients[layer][offset + input])
                    input++
                }
            }
        }
    }

    private fun updateOnline() {
        for (layer in weights.indices) {
            val source = activations[layer]
            for (output in deltas[layer].indices) {
                val scale = rate * deltas[layer][output]
                val offset = output * source.size
                var input = smallFloatVectors(source.size, bits) { laneOffset, species ->
                    val v = FloatVector.fromArray(species, velocity[layer], offset + laneOffset)
                    val next = FloatVector.broadcast(species, momentum).fma(v, FloatVector.fromArray(species, source, laneOffset).mul(scale))
                    next.intoArray(velocity[layer], offset + laneOffset)
                    FloatVector.fromArray(species, weights[layer], offset + laneOffset).add(next).intoArray(weights[layer], offset + laneOffset)
                }
                while (input < source.size) {
                    val index = offset + input
                    val next = Math.fma(momentum, velocity[layer][index], scale * source[input])
                    velocity[layer][index] = next
                    weights[layer][index] += next
                    input++
                }
                val next = Math.fma(momentum, biasVelocity[layer][output], scale)
                biasVelocity[layer][output] = next
                biases[layer][output] += next
            }
        }
    }

    private fun updateBatch(count: Int) {
        val scale = rate / count
        for (layer in weights.indices) {
            update(weights[layer], velocity[layer], gradients[layer], scale)
            update(biases[layer], biasVelocity[layer], biasGradients[layer], scale)
        }
    }

    private fun update(values: FloatArray, velocities: FloatArray, gradients: FloatArray, scale: Float) {
        var index = smallFloatVectors(values.size, bits) { laneOffset, species ->
            val next = FloatVector.broadcast(species, momentum).fma(FloatVector.fromArray(species, velocities, laneOffset),
                FloatVector.fromArray(species, gradients, laneOffset).mul(scale))
            next.intoArray(velocities, laneOffset)
            FloatVector.fromArray(species, values, laneOffset).add(next).intoArray(values, laneOffset)
        }
        while (index < values.size) {
            val next = Math.fma(momentum, velocities[index], scale * gradients[index])
            velocities[index] = next
            values[index] += next
            index++
        }
    }

    override fun snapshot() = NeuroTrainingState(topology.copyOf(),
        Array(weights.size) { i -> smallExport(weights[i]) }, Array(biases.size) { i -> smallExport(biases[i]) },
        Array(velocity.size) { i -> smallExport(velocity[i]) }, Array(biasVelocity.size) { i -> smallExport(biasVelocity[i]) },
        originalInputs.copyOf(), originalTargets.copyOf())
}

/** EXACT keeps Math.exp per lane. FAST vectorizes the explicitly selected range-reduced polynomial. */
internal class SmallFloatActivation(private val mode: Neuro.SigmoidMode, private val bits: Int, capacity: Int) {
    private val remainder = FloatArray(capacity)
    private val exponents = IntArray(capacity)
    private val polynomial = FloatArray(capacity)

    fun apply(values: FloatArray, length: Int) {
        if (mode == Neuro.SigmoidMode.EXACT) {
            for (i in 0 until length) values[i] = (1.0f / (1.0f + Math.exp(-values[i].toDouble()))).toFloat()
            return
        }
        for (i in 0 until length) {
            val negative = -kotlin.math.abs(values[i])
            val exponent = (negative * 1.4426950408889634f).toInt()
            exponents[i] = exponent
            remainder[i] = if (negative <= -104.0f) 0.0f else negative - exponent * 0.6931471805599453f
        }
        var i = smallFloatVectors(length, bits) { laneOffset, species ->
            val r = FloatVector.fromArray(species, remainder, laneOffset)
            val p = r.mul(0.008333333333333333f).add(0.041666666666666664f).mul(r).add(0.16666666666666666f).mul(r).add(0.5f)
            r.mul(r).mul(p).add(r.add(1.0f)).intoArray(polynomial, laneOffset)
        }
        while (i < length) {
            val r = remainder[i]
            polynomial[i] = 1.0f + r + r * r * (0.5f + r * (0.16666666666666666f + r * (0.041666666666666664f + r * 0.008333333333333333f)))
            i++
        }
        for (index in 0 until length) {
            val value = values[index]
            val exp = if (-kotlin.math.abs(value) <= -104.0f) 0.0f else Math.scalb(polynomial[index], exponents[index])
            values[index] = if (value >= 0.0f) 1.0f / (1.0f + exp) else exp / (1.0f + exp)
        }
    }
}

internal fun smallDoubles(values: DoubleArray): DoubleArray = values.copyOf()
internal fun smallFloats(values: DoubleArray): FloatArray = FloatArray(values.size) { values[it].toFloat() }
internal fun smallExport(values: FloatArray): DoubleArray = DoubleArray(values.size) { values[it].toDouble() }

/** Inline dispatch leaves a static species constant at every Vector API intrinsic call site. */
internal inline fun smallDoubleVectors(length: Int, bits: Int, action: (Int, VectorSpecies<Double>) -> Unit): Int {
    var offset = 0
    if (bits == 128) {
        val limit = DoubleVector.SPECIES_128.loopBound(length)
        while (offset < limit) { action(offset, DoubleVector.SPECIES_128); offset += 2 }
    } else if (bits == 256) {
        val limit = DoubleVector.SPECIES_256.loopBound(length)
        while (offset < limit) { action(offset, DoubleVector.SPECIES_256); offset += 4 }
    }
    return offset
}

internal inline fun smallFloatVectors(length: Int, bits: Int, action: (Int, VectorSpecies<Float>) -> Unit): Int {
    var offset = 0
    if (bits == 128) {
        val limit = FloatVector.SPECIES_128.loopBound(length)
        while (offset < limit) { action(offset, FloatVector.SPECIES_128); offset += 4 }
    } else if (bits == 256) {
        val limit = FloatVector.SPECIES_256.loopBound(length)
        while (offset < limit) { action(offset, FloatVector.SPECIES_256); offset += 8 }
    }
    return offset
}

/** Constant offsets unroll the bounded family while sharing prefixes to keep generated code small.
 * Callers have already validated widths 2/4/8/16. Each chain retains its original FMA order.
 */
internal inline fun smallInputOffsets(width: Int, step: (Int) -> Unit) {
    step(0)
    step(1)
    if (width == 2) return
    step(2)
    step(3)
    if (width == 4) return
    step(4)
    step(5)
    step(6)
    step(7)
    if (width == 8) return
    step(8)
    step(9)
    step(10)
    step(11)
    step(12)
    step(13)
    step(14)
    step(15)
}
