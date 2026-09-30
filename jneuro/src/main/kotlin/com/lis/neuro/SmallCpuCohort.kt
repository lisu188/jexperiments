package com.lis.neuro

import jdk.incubator.vector.DoubleVector
import jdk.incubator.vector.FloatVector

/** Structure-of-arrays model lanes: every SIMD lane is an independent model, never a reduction. */
internal class SmallCpuCohort(
    states: Array<NeuroTrainingState>, parameters: Neuro.HyperParameters,
    precision: Neuro.TrainingPrecision, vectorBits: Int = 256
) : AutoCloseable {
    private val bits = smallVectorBits(parameters, vectorBits)
    private val count = states.size
    private val samples: Int
    private val compute: SmallCohortCompute
    private var closed = false
    val info = TrainingDeviceInfo(TrainingBackend.CPU, "CPU model lanes", "jvm-cpu",
        precision.name, "small-cpu-cohort-v1", engine = TrainingEngine.SMALL, simdBits = bits, sigmoid = parameters.sigmoidMode.name)

    init {
        require(states.isNotEmpty()) { "A cohort must contain models" }
        states.forEach(::validateSmallState)
        samples = states[0].samples
        require(states.all { it.topology.contentEquals(states[0].topology) && it.samples == samples }) {
            "Cohort models must share topology and sample count"
        }
        compute = if (precision == Neuro.TrainingPrecision.FP32) SmallFloatCohortCompute(states, parameters, bits)
            else SmallDoubleCohortCompute(states, parameters, bits)
    }

    fun train(orders: Array<Array<IntArray>>, batchSize: Int, online: Boolean,
              active: BooleanArray = BooleanArray(count) { true }): Array<NeuroTrainingState> {
        check(!closed) { "Small CPU cohort is closed" }
        require(batchSize > 0) { "batchSize must be > 0" }
        require(orders.size == count && active.size == count) { "Cohort model count mismatch" }
        val epochs = active.indices.firstOrNull { active[it] }?.let { orders[it].size } ?: 0
        for (model in 0 until count) if (active[model]) {
            require(orders[model].size == epochs) { "Active models must request the same epoch count" }
            validateSmallOrders(orders[model], samples)
        }
        compute.train(orders, batchSize, online, active, epochs)
        return compute.snapshot()
    }

    override fun close() { closed = true }
}

private interface SmallCohortCompute {
    fun train(orders: Array<Array<IntArray>>, batchSize: Int, online: Boolean, active: BooleanArray, epochs: Int)
    fun snapshot(): Array<NeuroTrainingState>
}

internal fun smallStateCopy(state: NeuroTrainingState) = NeuroTrainingState(state.topology.copyOf(),
    Array(state.weights.size) { state.weights[it].copyOf() }, Array(state.biases.size) { state.biases[it].copyOf() },
    Array(state.weightVelocity.size) { state.weightVelocity[it].copyOf() },
    Array(state.biasVelocity.size) { state.biasVelocity[it].copyOf() }, state.inputs.copyOf(), state.targets.copyOf())

private class SmallDoubleCohortCompute(states: Array<NeuroTrainingState>, parameters: Neuro.HyperParameters, private val bits: Int) : SmallCohortCompute {
    private val initial = Array(states.size) { smallStateCopy(states[it]) }
    private val topology = states[0].topology.copyOf()
    private val models = states.size
    private val vector = bits != 0
    private val lanes = if (vector) bits / 64 else 1
    private val stride = ((models + lanes - 1) / lanes) * lanes
    private val activeLanes = BooleanArray(stride)
    private val touched = BooleanArray(models)
    private val weights = pack { it.weights }
    private val biases = pack { it.biases }
    private val velocities = pack { it.weightVelocity }
    private val biasVelocities = pack { it.biasVelocity }
    private val gradients = Array(weights.size) { DoubleArray(weights[it].size) }
    private val biasGradients = Array(biases.size) { DoubleArray(biases[it].size) }
    private val activations = Array(topology.size) { DoubleArray(topology[it] * stride) }
    private val deltas = Array(weights.size) { DoubleArray(topology[it + 1] * stride) }
    private val inputs = Array(models) { smallDoubles(states[it].inputs) }
    private val targets = Array(models) { smallDoubles(states[it].targets) }
    private val gatheredTargets = DoubleArray(stride)
    private val beta = parameters.beta
    private val rate = parameters.learningRate
    private val momentum = parameters.momentum
    private val activation = SmallDoubleActivation(parameters.sigmoidMode, bits, 16 * stride)

    init {
        require(beta.isFinite() && rate.isFinite() && momentum.isFinite()) { "Hyperparameters exceed compute precision" }
        require(arrayOf(weights, biases, velocities, biasVelocities, inputs, targets).all { matrix -> matrix.all { row -> row.all { it.isFinite() } } }) {
            "State exceeds compute precision"
        }
    }

    private fun pack(select: (NeuroTrainingState) -> Array<DoubleArray>): Array<DoubleArray> =
        Array(topology.size - 1) { layer ->
            val size = select(initial[0])[layer].size
            DoubleArray(size * stride) { index ->
                val model = index % stride
                if (model < models) select(initial[model])[layer][index / stride] else 0.0
            }
        }

    override fun train(orders: Array<Array<IntArray>>, batchSize: Int, online: Boolean, active: BooleanArray, epochs: Int) {
        active.copyInto(activeLanes)
        for (model in 0 until models) if (active[model] && epochs > 0) touched[model] = true
        repeat(epochs) { epoch ->
            var start = 0
            while (start < initial[0].samples) {
                val count = if (online) 1 else minOf(batchSize, initial[0].samples - start)
                if (!online) {
                    gradients.forEach { it.fill(0.0) }
                    biasGradients.forEach { it.fill(0.0) }
                }
                for (position in start until start + count) {
                    for (model in 0 until models) if (active[model]) {
                        val sample = orders[model][epoch][position]
                        activations[0][model] = inputs[model][2 * sample]
                        activations[0][stride + model] = inputs[model][2 * sample + 1]
                        gatheredTargets[model] = targets[model][sample]
                    }
                    forward()
                    backward(online)
                    if (online) updateOnline() else accumulate()
                }
                if (!online) updateBatch(count)
                start += count
            }
        }
    }

    private fun forward() {
        for (layer in weights.indices) {
            val source = activations[layer]
            val destination = activations[layer + 1]
            for (output in 0 until topology[layer + 1]) {
                val offset = output * stride
                if (vector) {
                    smallDoubleVectors(stride, bits) { laneOffset, species ->
                        var sum = DoubleVector.fromArray(species, biases[layer], offset + laneOffset)
                        smallInputOffsets(topology[layer]) { input ->
                            val a = DoubleVector.fromArray(species, source, input * stride + laneOffset)
                            val w = DoubleVector.fromArray(species, weights[layer], (output * topology[layer] + input) * stride + laneOffset)
                            sum = a.fma(w, sum)
                        }
                        sum.mul(beta).intoArray(destination, offset + laneOffset)
                    }
                } else for (model in 0 until models) {
                    var sum = biases[layer][offset + model]
                    smallInputOffsets(topology[layer]) { input ->
                        sum = Math.fma(source[input * stride + model],
                            weights[layer][(output * topology[layer] + input) * stride + model], sum)
                    }
                    destination[offset + model] = sum * beta
                }
            }
            activation.apply(destination, destination.size)
        }
    }

    private fun backward(online: Boolean) {
        for (model in 0 until stride) {
            val value = activations.last()[model]
            deltas.last()[model] = (gatheredTargets[model] - value) * beta * value * (1.0 - value)
        }
        for (layer in weights.lastIndex - 1 downTo 0) {
            val width = topology[layer + 1]
            val nextWidth = topology[layer + 2]
            for (input in 0 until width) {
                val offset = input * stride
                if (vector) {
                    smallDoubleVectors(stride, bits) { laneOffset, species ->
                        var sum = if (online) DoubleVector.fromArray(species, deltas[layer + 1], laneOffset)
                            .mul(DoubleVector.fromArray(species, weights[layer + 1], offset + laneOffset)) else DoubleVector.zero(species)
                        for (output in (if (online) 1 else 0) until nextWidth) {
                            val d = DoubleVector.fromArray(species, deltas[layer + 1], output * stride + laneOffset)
                            sum = d.fma(DoubleVector.fromArray(species, weights[layer + 1], (output * width + input) * stride + laneOffset), sum)
                        }
                        val a = DoubleVector.fromArray(species, activations[layer + 1], offset + laneOffset)
                        val complement = DoubleVector.broadcast(species, 1.0).sub(a)
                        val derivative = if (online) sum.mul(a.mul(beta).mul(complement)) else sum.mul(beta).mul(a).mul(complement)
                        derivative.intoArray(deltas[layer], offset + laneOffset)
                    }
                } else for (model in 0 until models) {
                    var sum = if (online) deltas[layer + 1][model] * weights[layer + 1][offset + model] else 0.0
                    for (output in (if (online) 1 else 0) until nextWidth) sum = Math.fma(deltas[layer + 1][output * stride + model],
                        weights[layer + 1][(output * width + input) * stride + model], sum)
                    val a = activations[layer + 1][offset + model]
                    deltas[layer][offset + model] = if (online) sum * (beta * a * (1.0 - a)) else sum * beta * a * (1.0 - a)
                }
            }
        }
    }

    private fun accumulate() {
        for (layer in weights.indices) for (output in 0 until topology[layer + 1]) {
            val deltaOffset = output * stride
            for (model in 0 until stride) biasGradients[layer][deltaOffset + model] += deltas[layer][deltaOffset + model]
            for (input in 0 until topology[layer]) {
                val offset = (output * topology[layer] + input) * stride
                if (vector) {
                    smallDoubleVectors(stride, bits) { laneOffset, species ->
                        val a = DoubleVector.fromArray(species, activations[layer], input * stride + laneOffset)
                        val d = DoubleVector.fromArray(species, deltas[layer], deltaOffset + laneOffset)
                        a.fma(d, DoubleVector.fromArray(species, gradients[layer], offset + laneOffset)).intoArray(gradients[layer], offset + laneOffset)
                    }
                } else for (model in 0 until models) gradients[layer][offset + model] = Math.fma(
                    activations[layer][input * stride + model], deltas[layer][deltaOffset + model], gradients[layer][offset + model])
            }
        }
    }

    private fun updateOnline() {
        for (layer in weights.indices) for (output in 0 until topology[layer + 1]) {
            val deltaOffset = output * stride
            for (input in 0 until topology[layer]) {
                val offset = (output * topology[layer] + input) * stride
                if (vector) {
                    smallDoubleVectors(stride, bits) { laneOffset, species ->
                        val mask = species.loadMask(activeLanes, laneOffset)
                        val scale = DoubleVector.fromArray(species, deltas[layer], deltaOffset + laneOffset).mul(rate)
                        val gradient = scale.mul(DoubleVector.fromArray(species, activations[layer], input * stride + laneOffset))
                        val next = DoubleVector.broadcast(species, momentum).fma(DoubleVector.fromArray(species, velocities[layer], offset + laneOffset), gradient)
                        next.intoArray(velocities[layer], offset + laneOffset, mask)
                        DoubleVector.fromArray(species, weights[layer], offset + laneOffset).add(next).intoArray(weights[layer], offset + laneOffset, mask)
                    }
                } else for (model in 0 until models) if (activeLanes[model]) {
                    val index = offset + model
                    val next = Math.fma(momentum, velocities[layer][index], (rate * deltas[layer][deltaOffset + model]) * activations[layer][input * stride + model])
                    velocities[layer][index] = next
                    weights[layer][index] += next
                }
            }
            for (model in 0 until models) if (activeLanes[model]) {
                val index = deltaOffset + model
                val next = Math.fma(momentum, biasVelocities[layer][index], rate * deltas[layer][index])
                biasVelocities[layer][index] = next
                biases[layer][index] += next
            }
        }
    }

    private fun updateBatch(count: Int) {
        val scale = rate / count
        for (layer in weights.indices) {
            update(weights[layer], velocities[layer], gradients[layer], scale)
            update(biases[layer], biasVelocities[layer], biasGradients[layer], scale)
        }
    }

    private fun update(values: DoubleArray, velocities: DoubleArray, gradient: DoubleArray, scale: Double) {
        if (vector) {
            smallDoubleVectors(values.size, bits) { laneOffset, species ->
                val mask = species.loadMask(activeLanes, laneOffset % stride)
                val next = DoubleVector.broadcast(species, momentum).fma(DoubleVector.fromArray(species, velocities, laneOffset),
                    DoubleVector.fromArray(species, gradient, laneOffset).mul(scale))
                next.intoArray(velocities, laneOffset, mask)
                DoubleVector.fromArray(species, values, laneOffset).add(next).intoArray(values, laneOffset, mask)
            }
        } else for (index in values.indices) if (activeLanes[index % stride]) {
            val next = Math.fma(momentum, velocities[index], scale * gradient[index])
            velocities[index] = next
            values[index] += next
        }
    }

    private fun unpack(values: Array<DoubleArray>, model: Int) = Array(values.size) { layer ->
        DoubleArray(values[layer].size / stride) { values[layer][it * stride + model] }
    }

    override fun snapshot(): Array<NeuroTrainingState> = Array(models) { model ->
        if (!touched[model]) smallStateCopy(initial[model]) else NeuroTrainingState(topology.copyOf(),
            unpack(weights, model), unpack(biases, model), unpack(velocities, model), unpack(biasVelocities, model),
            initial[model].inputs.copyOf(), initial[model].targets.copyOf())
    }
}

private class SmallFloatCohortCompute(states: Array<NeuroTrainingState>, parameters: Neuro.HyperParameters, private val bits: Int) : SmallCohortCompute {
    private val initial = Array(states.size) { smallStateCopy(states[it]) }
    private val topology = states[0].topology.copyOf()
    private val models = states.size
    private val vector = bits != 0
    private val lanes = if (vector) bits / 32 else 1
    private val stride = ((models + lanes - 1) / lanes) * lanes
    private val activeLanes = BooleanArray(stride)
    private val touched = BooleanArray(models)
    private val weights = pack { it.weights }
    private val biases = pack { it.biases }
    private val velocities = pack { it.weightVelocity }
    private val biasVelocities = pack { it.biasVelocity }
    private val gradients = Array(weights.size) { FloatArray(weights[it].size) }
    private val biasGradients = Array(biases.size) { FloatArray(biases[it].size) }
    private val activations = Array(topology.size) { FloatArray(topology[it] * stride) }
    private val deltas = Array(weights.size) { FloatArray(topology[it + 1] * stride) }
    private val inputs = Array(models) { smallFloats(states[it].inputs) }
    private val targets = Array(models) { smallFloats(states[it].targets) }
    private val gatheredTargets = FloatArray(stride)
    private val beta = parameters.beta.toFloat()
    private val rate = parameters.learningRate.toFloat()
    private val momentum = parameters.momentum.toFloat()
    private val activation = SmallFloatActivation(parameters.sigmoidMode, bits, 16 * stride)

    init {
        require(beta.isFinite() && rate.isFinite() && momentum.isFinite()) { "Hyperparameters exceed compute precision" }
        require(arrayOf(weights, biases, velocities, biasVelocities, inputs, targets).all { matrix -> matrix.all { row -> row.all { it.isFinite() } } }) {
            "State exceeds compute precision"
        }
    }

    private fun pack(select: (NeuroTrainingState) -> Array<DoubleArray>): Array<FloatArray> =
        Array(topology.size - 1) { layer ->
            val size = select(initial[0])[layer].size
            FloatArray(size * stride) { index ->
                val model = index % stride
                if (model < models) select(initial[model])[layer][index / stride].toFloat() else 0.0f
            }
        }

    override fun train(orders: Array<Array<IntArray>>, batchSize: Int, online: Boolean, active: BooleanArray, epochs: Int) {
        active.copyInto(activeLanes)
        for (model in 0 until models) if (active[model] && epochs > 0) touched[model] = true
        repeat(epochs) { epoch ->
            var start = 0
            while (start < initial[0].samples) {
                val count = if (online) 1 else minOf(batchSize, initial[0].samples - start)
                if (!online) {
                    gradients.forEach { it.fill(0.0f) }
                    biasGradients.forEach { it.fill(0.0f) }
                }
                for (position in start until start + count) {
                    for (model in 0 until models) if (active[model]) {
                        val sample = orders[model][epoch][position]
                        activations[0][model] = inputs[model][2 * sample]
                        activations[0][stride + model] = inputs[model][2 * sample + 1]
                        gatheredTargets[model] = targets[model][sample]
                    }
                    forward()
                    backward(online)
                    if (online) updateOnline() else accumulate()
                }
                if (!online) updateBatch(count)
                start += count
            }
        }
    }

    private fun forward() {
        for (layer in weights.indices) {
            val source = activations[layer]
            val destination = activations[layer + 1]
            for (output in 0 until topology[layer + 1]) {
                val offset = output * stride
                if (vector) {
                    smallFloatVectors(stride, bits) { laneOffset, species ->
                        var sum = FloatVector.fromArray(species, biases[layer], offset + laneOffset)
                        smallInputOffsets(topology[layer]) { input ->
                            val a = FloatVector.fromArray(species, source, input * stride + laneOffset)
                            val w = FloatVector.fromArray(species, weights[layer], (output * topology[layer] + input) * stride + laneOffset)
                            sum = a.fma(w, sum)
                        }
                        sum.mul(beta).intoArray(destination, offset + laneOffset)
                    }
                } else for (model in 0 until models) {
                    var sum = biases[layer][offset + model]
                    smallInputOffsets(topology[layer]) { input ->
                        sum = Math.fma(source[input * stride + model],
                            weights[layer][(output * topology[layer] + input) * stride + model], sum)
                    }
                    destination[offset + model] = sum * beta
                }
            }
            activation.apply(destination, destination.size)
        }
    }

    private fun backward(online: Boolean) {
        for (model in 0 until stride) {
            val value = activations.last()[model]
            deltas.last()[model] = (gatheredTargets[model] - value) * beta * value * (1.0f - value)
        }
        for (layer in weights.lastIndex - 1 downTo 0) {
            val width = topology[layer + 1]
            val nextWidth = topology[layer + 2]
            for (input in 0 until width) {
                val offset = input * stride
                if (vector) {
                    smallFloatVectors(stride, bits) { laneOffset, species ->
                        var sum = if (online) FloatVector.fromArray(species, deltas[layer + 1], laneOffset)
                            .mul(FloatVector.fromArray(species, weights[layer + 1], offset + laneOffset)) else FloatVector.zero(species)
                        for (output in (if (online) 1 else 0) until nextWidth) {
                            val d = FloatVector.fromArray(species, deltas[layer + 1], output * stride + laneOffset)
                            sum = d.fma(FloatVector.fromArray(species, weights[layer + 1], (output * width + input) * stride + laneOffset), sum)
                        }
                        val a = FloatVector.fromArray(species, activations[layer + 1], offset + laneOffset)
                        val complement = FloatVector.broadcast(species, 1.0f).sub(a)
                        val derivative = if (online) sum.mul(a.mul(beta).mul(complement)) else sum.mul(beta).mul(a).mul(complement)
                        derivative.intoArray(deltas[layer], offset + laneOffset)
                    }
                } else for (model in 0 until models) {
                    var sum = if (online) deltas[layer + 1][model] * weights[layer + 1][offset + model] else 0.0f
                    for (output in (if (online) 1 else 0) until nextWidth) sum = Math.fma(deltas[layer + 1][output * stride + model],
                        weights[layer + 1][(output * width + input) * stride + model], sum)
                    val a = activations[layer + 1][offset + model]
                    deltas[layer][offset + model] = if (online) sum * (beta * a * (1.0f - a)) else sum * beta * a * (1.0f - a)
                }
            }
        }
    }

    private fun accumulate() {
        for (layer in weights.indices) for (output in 0 until topology[layer + 1]) {
            val deltaOffset = output * stride
            for (model in 0 until stride) biasGradients[layer][deltaOffset + model] += deltas[layer][deltaOffset + model]
            for (input in 0 until topology[layer]) {
                val offset = (output * topology[layer] + input) * stride
                if (vector) {
                    smallFloatVectors(stride, bits) { laneOffset, species ->
                        val a = FloatVector.fromArray(species, activations[layer], input * stride + laneOffset)
                        val d = FloatVector.fromArray(species, deltas[layer], deltaOffset + laneOffset)
                        a.fma(d, FloatVector.fromArray(species, gradients[layer], offset + laneOffset)).intoArray(gradients[layer], offset + laneOffset)
                    }
                } else for (model in 0 until models) gradients[layer][offset + model] = Math.fma(
                    activations[layer][input * stride + model], deltas[layer][deltaOffset + model], gradients[layer][offset + model])
            }
        }
    }

    private fun updateOnline() {
        for (layer in weights.indices) for (output in 0 until topology[layer + 1]) {
            val deltaOffset = output * stride
            for (input in 0 until topology[layer]) {
                val offset = (output * topology[layer] + input) * stride
                if (vector) {
                    smallFloatVectors(stride, bits) { laneOffset, species ->
                        val mask = species.loadMask(activeLanes, laneOffset)
                        val scale = FloatVector.fromArray(species, deltas[layer], deltaOffset + laneOffset).mul(rate)
                        val gradient = scale.mul(FloatVector.fromArray(species, activations[layer], input * stride + laneOffset))
                        val next = FloatVector.broadcast(species, momentum).fma(FloatVector.fromArray(species, velocities[layer], offset + laneOffset), gradient)
                        next.intoArray(velocities[layer], offset + laneOffset, mask)
                        FloatVector.fromArray(species, weights[layer], offset + laneOffset).add(next).intoArray(weights[layer], offset + laneOffset, mask)
                    }
                } else for (model in 0 until models) if (activeLanes[model]) {
                    val index = offset + model
                    val next = Math.fma(momentum, velocities[layer][index], (rate * deltas[layer][deltaOffset + model]) * activations[layer][input * stride + model])
                    velocities[layer][index] = next
                    weights[layer][index] += next
                }
            }
            for (model in 0 until models) if (activeLanes[model]) {
                val index = deltaOffset + model
                val next = Math.fma(momentum, biasVelocities[layer][index], rate * deltas[layer][index])
                biasVelocities[layer][index] = next
                biases[layer][index] += next
            }
        }
    }

    private fun updateBatch(count: Int) {
        val scale = rate / count
        for (layer in weights.indices) {
            update(weights[layer], velocities[layer], gradients[layer], scale)
            update(biases[layer], biasVelocities[layer], biasGradients[layer], scale)
        }
    }

    private fun update(values: FloatArray, velocities: FloatArray, gradient: FloatArray, scale: Float) {
        if (vector) {
            smallFloatVectors(values.size, bits) { laneOffset, species ->
                val mask = species.loadMask(activeLanes, laneOffset % stride)
                val next = FloatVector.broadcast(species, momentum).fma(FloatVector.fromArray(species, velocities, laneOffset),
                    FloatVector.fromArray(species, gradient, laneOffset).mul(scale))
                next.intoArray(velocities, laneOffset, mask)
                FloatVector.fromArray(species, values, laneOffset).add(next).intoArray(values, laneOffset, mask)
            }
        } else for (index in values.indices) if (activeLanes[index % stride]) {
            val next = Math.fma(momentum, velocities[index], scale * gradient[index])
            velocities[index] = next
            values[index] += next
        }
    }

    private fun unpack(values: Array<FloatArray>, model: Int) = Array(values.size) { layer ->
        DoubleArray(values[layer].size / stride) { values[layer][it * stride + model].toDouble() }
    }

    override fun snapshot(): Array<NeuroTrainingState> = Array(models) { model ->
        if (!touched[model]) smallStateCopy(initial[model]) else NeuroTrainingState(topology.copyOf(),
            unpack(weights, model), unpack(biases, model), unpack(velocities, model), unpack(biasVelocities, model),
            initial[model].inputs.copyOf(), initial[model].targets.copyOf())
    }
}
