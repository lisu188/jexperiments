package com.lis.neuro

/** Kept injectable so the real allocation/dispatch/lifecycle code runs in CPU-only tests. */
internal interface CudaDriver : AutoCloseable {
    val info: TrainingDeviceInfo
    fun availableMemory(): Long
    fun allocate(bytes: Long): Long
    fun free(pointer: Long)
    fun upload(pointer: Long, values: DoubleArray)
    fun upload(pointer: Long, values: IntArray)
    fun download(pointer: Long, values: DoubleArray)
    fun launch(name: String, workItems: Int, vararg arguments: Any)
    fun synchronize()
}

internal class CudaTraining(private val network: Neuro, private val state: NeuroTrainingState, private val driver: CudaDriver) : AutoCloseable {
    val info: TrainingDeviceInfo get() = driver.info
    private val logId = NeuroLog.id("cuda")
    private val shape = state.topology
    private val hp = network.hyperParameters()
    private val allocations = ArrayList<Long>()
    private val workspace = ArrayList<Long>()
    private var reservation = 0L
    private var capacity = 0
    private var closed = false
    private val weights = LongArray(state.weights.size)
    private val biases = LongArray(weights.size)
    private val velocities = LongArray(weights.size)
    private val biasVelocities = LongArray(weights.size)
    private var inputs = 0L
    private var targets = 0L
    private var order = 0L
    private var batchTargets = 0L
    private var activations = LongArray(shape.size)
    private var deltas = LongArray(weights.size)

    init {
        try {
            val elements = state.weights.sumOf { it.size.toLong() } + state.biases.sumOf { it.size.toLong() }
            val bytes = Math.addExact(Math.multiplyExact(elements, 16L),
                Math.addExact(Math.multiplyExact(state.inputs.size.toLong() + state.targets.size, 8L), state.samples * 4L))
            reserve(bytes)
            inputs = doubles(state.inputs)
            targets = doubles(state.targets)
            order = allocate(maxOf(4L, state.samples * 4L))
            for (layer in weights.indices) {
                weights[layer] = doubles(state.weights[layer])
                biases[layer] = doubles(state.biases[layer])
                velocities[layer] = doubles(state.weightVelocity[layer])
                biasVelocities[layer] = doubles(state.biasVelocity[layer])
            }
        } catch (failure: Throwable) {
            NeuroLog.error("cuda", "training.initialization.failed", failure, "training" to logId, "model" to network.logId)
            try { close() } catch (cleanup: Throwable) { failure.addSuppressed(cleanup) }
            throw failure
        }
    }

    private fun reserve(bytes: Long) {
        CudaMemoryBudget.acquire(info.identity, bytes, driver.availableMemory())
        reservation += bytes
        NeuroLog.debug("cuda", "memory.reserved") { mapOf("training" to logId, "model" to network.logId,
            "session" to network.trainingSessionLogId, "bytes" to bytes, "reservedBytes" to reservation) }
    }

    private fun allocate(bytes: Long, temporary: Boolean = false): Long =
        driver.allocate(maxOf(8, bytes)).also { if (temporary) workspace.add(it) else allocations.add(it) }

    private fun doubles(values: DoubleArray): Long = allocate(values.size * 8L).also {
        if (values.isNotEmpty()) driver.upload(it, values)
    }

    private fun ensureCapacity(requested: Int) {
        if (requested <= capacity) return
        NeuroLog.debug("cuda", "workspace.growing") { mapOf("training" to logId, "oldCapacity" to capacity, "newCapacity" to requested) }
        driver.synchronize()
        // Reserve only the growth; free the old workspace before allocating its replacement.
        val elements = shape.sumOf { it.toLong() } + shape.drop(1).sumOf { it.toLong() } + shape.last()
        reserve(Math.multiplyExact((requested - capacity).toLong(), Math.multiplyExact(elements, 8L)))
        while (workspace.isNotEmpty()) {
            driver.free(workspace.last())
            workspace.removeAt(workspace.lastIndex)
        }
        activations = LongArray(shape.size) { allocate(requested.toLong() * shape[it] * 8L, true) }
        deltas = LongArray(weights.size) { allocate(requested.toLong() * shape[it + 1] * 8L, true) }
        batchTargets = allocate(requested.toLong() * shape.last() * 8L, true)
        capacity = requested
        NeuroLog.debug("cuda", "workspace.ready") { mapOf("training" to logId, "capacity" to capacity, "allocations" to workspace.size) }
    }

    fun epoch(batchSize: Int, online: Boolean): Double {
        check(state.samples > 0) { "no training samples" }
        NeuroLog.trace("cuda", "epoch.submitted") { mapOf("training" to logId, "model" to network.logId,
            "session" to network.trainingSessionLogId, "samples" to state.samples, "batchSize" to batchSize, "online" to online,
            "topology" to shape.joinToString("x")) }
        ensureCapacity(minOf(batchSize, state.samples))
        driver.upload(order, network.deviceTrainingOrder())
        var start = 0
        while (start < state.samples) {
            val count = minOf(batchSize, state.samples - start)
            driver.launch("gather", Math.multiplyExact(count, maxOf(shape.first(), shape.last())),
                inputs, targets, order, activations[0], batchTargets, shape.first(), shape.last(), start, count)
            for (layer in weights.indices) {
                driver.launch("forward", Math.multiplyExact(count, shape[layer + 1]), activations[layer], weights[layer],
                    biases[layer], activations[layer + 1], shape[layer], shape[layer + 1], count, hp.beta, hp.sigmoidMode.ordinal)
            }
            driver.launch("output_delta", Math.multiplyExact(count, shape.last()), activations.last(), batchTargets,
                deltas.last(), shape.last(), count, hp.beta)
            for (layer in weights.lastIndex - 1 downTo 0) {
                driver.launch("hidden_delta", Math.multiplyExact(count, shape[layer + 1]), weights[layer + 1], deltas[layer + 1],
                    activations[layer + 1], deltas[layer], shape[layer + 1], shape[layer + 2], count, hp.beta)
            }
            for (layer in weights.indices) {
                driver.launch("update", Math.addExact(state.weights[layer].size, shape[layer + 1]), activations[layer],
                    deltas[layer], weights[layer], biases[layer], velocities[layer], biasVelocities[layer],
                    shape[layer], shape[layer + 1], count, hp.learningRate, hp.momentum, if (online) 1 else 0)
            }
            start += count
        }
        driver.synchronize()
        for (layer in weights.indices) {
            driver.download(weights[layer], state.weights[layer])
            driver.download(biases[layer], state.biases[layer])
            driver.download(velocities[layer], state.weightVelocity[layer])
            driver.download(biasVelocities[layer], state.biasVelocity[layer])
        }
        val error = network.commitDeviceEpoch(state)
        NeuroLog.debug("cuda", "epoch.published") { mapOf("training" to logId, "model" to network.logId,
            "session" to network.trainingSessionLogId, "error" to error, "samples" to state.samples) }
        return error
    }

    override fun close() {
        if (closed) return
        closed = true
        var failure: Throwable? = null
        fun cleanup(action: () -> Unit) {
            try { action() } catch (exception: Throwable) {
                NeuroLog.error("cuda", "training.cleanup.failed", exception, "training" to logId, "model" to network.logId)
                if (failure == null) failure = exception else failure.addSuppressed(exception)
            }
        }
        cleanup { driver.synchronize() }
        for (pointer in (allocations + workspace).reversed()) cleanup { driver.free(pointer) }
        cleanup { driver.close() }
        CudaMemoryBudget.release(info.identity, reservation)
        NeuroLog.debug("cuda", "memory.released") { mapOf("training" to logId, "bytes" to reservation) }
        failure?.let { throw it }
    }
}

/** No session may exhaust device memory. Admission fails clearly rather than deadlocking growing sessions. */
internal object CudaMemoryBudget {
    private val reserved = HashMap<String, Long>()
    private val limits = HashMap<String, Long>()
    @Synchronized fun acquire(device: String, bytes: Long, free: Long) {
        require(bytes >= 0) { "Negative CUDA allocation." }
        val held = reserved[device] ?: 0L
        val limit = limits.getOrPut(device) { free - free / 5 }
        if (bytes > free - free / 5 || bytes > limit - held) {
            val failure = IllegalStateException("Insufficient CUDA memory; reduce batch size, topology, or search parallelism.")
            NeuroLog.error("cuda", "memory.admission.failed", failure, "device" to device, "requestedBytes" to bytes,
                "freeBytes" to free, "heldBytes" to held, "limitBytes" to limit)
            throw failure
        }
        NeuroLog.debug("cuda", "memory.admitted") { mapOf("device" to device, "requestedBytes" to bytes,
            "freeBytes" to free, "heldBytes" to held, "limitBytes" to limit) }
        reserved[device] = Math.addExact(held, bytes)
    }
    @Synchronized fun release(device: String, bytes: Long) {
        val remaining = (reserved[device] ?: 0L) - bytes
        if (remaining <= 0) { reserved.remove(device); limits.remove(device) } else reserved[device] = remaining
    }
}
