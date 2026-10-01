package com.lis.neuro

/** The deliberately bounded family for the TensorFlow compatibility adapters. */
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

internal fun validateSmallOrders(orders: Array<IntArray>, samples: Int, seen: BooleanArray = BooleanArray(samples)) {
    require(seen.size == samples) { "Permutation workspace does not match dataset" }
    for (order in orders) {
        require(order.size == samples) { "Each epoch order must contain every sample" }
        seen.fill(false)
        for (sample in order) {
            require(sample in 0 until samples && !seen[sample]) { "Epoch order must be a permutation" }
            seen[sample] = true
        }
    }
}

/** Compatibility adapter: all arithmetic is performed by TensorFlow. */
internal class SmallCpuTraining(
    state: NeuroTrainingState,
    parameters: Neuro.HyperParameters,
    precision: Neuro.TrainingPrecision,
    vectorBits: Int = 256
) : SmallTrainingKernel {
    private val samples = state.samples
    private val kernel: SmallTrainingKernel
    private var closed = false
    override val info: TrainingDeviceInfo get() = kernel.info
    init {
        require(vectorBits == 0 || vectorBits == 128 || vectorBits == 256) { "vectorBits must be 0, 128 or 256" }
        validateSmallState(state)
        kernel = TensorFlowMath.trainingKernel(state, parameters, precision, TrainingBackend.CPU, TrainingEngine.SMALL)
    }
    override fun train(orders: Array<IntArray>, batchSize: Int, online: Boolean): NeuroTrainingState {
        check(!closed) { "TensorFlow training kernel is closed" }
        return kernel.train(orders, batchSize, online)
    }
    fun advanceForSearch(orders: Array<IntArray>, batchSize: Int, online: Boolean,
                         mayTrainEpoch: (Int) -> Boolean): SmallCpuAdvance {
        check(!closed) { "TensorFlow training kernel is closed" }
        require(batchSize > 0)
        validateSmallOrders(orders, samples)
        var completed = 0
        var state: NeuroTrainingState? = null
        for (order in orders) {
            if (!mayTrainEpoch(completed)) break
            state = train(arrayOf(order), batchSize, online)
            completed++
        }
        return SmallCpuAdvance(completed, state)
    }
    override fun close() { if (!closed) { closed = true; kernel.close() } }
}

internal data class SmallCpuAdvance(val epochs: Int, val state: NeuroTrainingState?)
