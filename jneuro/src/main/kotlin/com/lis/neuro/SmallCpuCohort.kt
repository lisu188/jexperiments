package com.lis.neuro

/** Compatibility cohort with one independently owned TensorFlow kernel per model. */
internal class SmallCpuCohort(
    states: Array<NeuroTrainingState>, parameters: Neuro.HyperParameters,
    precision: Neuro.TrainingPrecision, vectorBits: Int = 256
) : AutoCloseable {
    private val snapshots = Array(states.size) { smallStateCopy(states[it], shareDataset = true) }
    private val kernels = ArrayList<SmallCpuTraining>()
    private var closed = false
    private var failed = false
    val info: TrainingDeviceInfo get() = kernels.first().info
    init {
        require(states.isNotEmpty()) { "A cohort must contain models" }
        states.forEach(::validateSmallState)
        require(states.all { it.topology.contentEquals(states[0].topology) && it.samples == states[0].samples }) {
            "Cohort models must share topology and sample count"
        }
        try { states.forEach { kernels += SmallCpuTraining(it, parameters, precision, vectorBits) } }
        catch (failure: Throwable) {
            kernels.asReversed().forEach { try { it.close() } catch (cleanup: Throwable) { failure.addSuppressed(cleanup) } }
            throw failure
        }
    }
    fun train(orders: Array<Array<IntArray>>, batchSize: Int, online: Boolean,
              active: BooleanArray = BooleanArray(kernels.size) { true }): Array<NeuroTrainingState> {
        check(!closed && !failed) { "TensorFlow cohort is closed or failed; reopen from committed checkpoints" }
        require(batchSize > 0 && orders.size == kernels.size && active.size == kernels.size)
        val epochs = active.indices.firstOrNull { active[it] }?.let { orders[it].size } ?: 0
        for (index in kernels.indices) if (active[index]) {
            require(orders[index].size == epochs) { "Active models must request the same epoch count" }
            validateSmallOrders(orders[index], snapshots[index].samples)
        }
        val staged = arrayOfNulls<NeuroTrainingState>(kernels.size)
        try {
            for (index in kernels.indices) if (active[index] && epochs > 0)
                staged[index] = kernels[index].train(orders[index], batchSize, online)
        } catch (failure: Throwable) { failed = true; throw failure }
        for (index in kernels.indices) staged[index]?.let { snapshots[index] = it }
        return Array(kernels.size) { smallStateCopy(snapshots[it], shareDataset = true) }
    }
    override fun close() {
        if (closed) return
        closed = true
        var failure: Throwable? = null
        for (kernel in kernels.asReversed()) try { kernel.close() } catch (problem: Throwable) {
            if (failure == null) failure = problem else failure.addSuppressed(problem)
        }
        failure?.let { throw it }
    }
}

internal fun smallStateCopy(state: NeuroTrainingState, shareDataset: Boolean = false) = NeuroTrainingState(state.topology.copyOf(),
    Array(state.weights.size) { state.weights[it].copyOf() }, Array(state.biases.size) { state.biases[it].copyOf() },
    Array(state.weightVelocity.size) { state.weightVelocity[it].copyOf() },
    Array(state.biasVelocity.size) { state.biasVelocity[it].copyOf() },
    if (shareDataset && state.sharedDataset) state.inputs else state.inputs.copyOf(),
    if (shareDataset && state.sharedDataset) state.targets else state.targets.copyOf(), sharedDataset = shareDataset && state.sharedDataset)
