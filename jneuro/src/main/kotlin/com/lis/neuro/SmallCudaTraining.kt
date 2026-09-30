package com.lis.neuro

/** One fused launch per bounded chunk; only the caller may publish its returned state. */
internal class SmallCudaTraining(
    state: NeuroTrainingState,
    parameters: Neuro.HyperParameters,
    precision: Neuro.TrainingPrecision,
    driver: CudaDriver
) : SmallTrainingKernel {
    private val cohort = SmallCudaCohort(arrayOf(state), parameters, precision, driver)
    override val info: TrainingDeviceInfo get() = cohort.info
    override fun train(orders: Array<IntArray>, batchSize: Int, online: Boolean): NeuroTrainingState =
        cohort.train(arrayOf(orders), batchSize, online).single()
    override fun close() = cohort.close()
}

/** Same-shape models share immutable inputs/targets and use independent blocks, orders and optimizer state. */
internal class SmallCudaCohort(
    states: Array<NeuroTrainingState>,
    private val parameters: Neuro.HyperParameters,
    private val precision: Neuro.TrainingPrecision,
    private val driver: CudaDriver
) : AutoCloseable {
    val info = driver.info.copy(precision = precision.name, engine = TrainingEngine.SMALL,
        sigmoid = parameters.sigmoidMode.name, kernelVersion = "small-v2/packed-fp64/" + driver.info.kernelVersion)
    private val allocations = ArrayList<Long>()
    private val topology: IntArray
    private val inputs: DoubleArray
    private val targets: DoubleArray
    private val models: Int
    private val samples: Int
    private val parameterCount: Int
    private var packed: DoubleArray
    private var downloadStaging: DoubleArray
    private var orderStaging = IntArray(0)
    private val seen: IntArray
    private val activeStaging: IntArray
    private var statePointer = 0L
    private var inputPointer = 0L
    private var targetPointer = 0L
    private var topologyPointer = 0L
    private var activePointer = 0L
    private var orderPointer = 0L
    private var orderBytes = 0L
    private var reservation = 0L
    private var closed = false
    private var failed = false

    init {
        try {
            require(states.isNotEmpty()) { "A CUDA cohort needs at least one model." }
            topology = states[0].topology.copyOf()
            require(topology.size in 3..6 && topology.first() == 2 && topology.last() == 1 &&
                topology.drop(1).dropLast(1).all { it == 4 || it == 8 || it == 16 }) { "Unsupported SMALL CUDA topology." }
            parameterCount = (0 until topology.lastIndex).sumOf { (topology[it] + 1) * topology[it + 1] }
            inputs = states[0].inputs.copyOf()
            targets = states[0].targets.copyOf()
            require(inputs.size % 2 == 0 && targets.size == inputs.size / 2) { "Malformed SMALL CUDA dataset." }
            require(inputs.all { it.isFinite() } && targets.all { it.isFinite() }) { "Non-finite SMALL CUDA dataset." }
            samples = targets.size
            models = states.size
            packed = DoubleArray(Math.multiplyExact(models, Math.multiplyExact(parameterCount, 2)))
            downloadStaging = DoubleArray(packed.size)
            seen = IntArray(samples)
            activeStaging = IntArray(models)
            for ((model, state) in states.withIndex()) {
                require(state.topology.contentEquals(topology) && state.inputs.contentEquals(inputs) && state.targets.contentEquals(targets)) {
                    "CUDA cohort models must share topology and immutable dataset values."
                }
                pack(state, model)
            }
            require(packed.all { it.isFinite() }) { "Non-finite SMALL CUDA parameters." }
            reserve(Math.addExact(Math.multiplyExact(packed.size.toLong() + inputs.size + targets.size, 8L),
                Math.multiplyExact(topology.size.toLong() + models, 4L)) + 24L)
            statePointer = allocate(packed.size * 8L).also { driver.upload(it, packed) }
            inputPointer = allocate(inputs.size * 8L).also { if (inputs.isNotEmpty()) driver.upload(it, inputs) }
            targetPointer = allocate(targets.size * 8L).also { if (targets.isNotEmpty()) driver.upload(it, targets) }
            topologyPointer = allocate(topology.size * 4L).also { driver.upload(it, topology) }
            activePointer = allocate(models * 4L)
        } catch (failure: Throwable) {
            try { close() } catch (cleanup: Throwable) { failure.addSuppressed(cleanup) }
            throw failure
        }
    }

    private fun pack(state: NeuroTrainingState, model: Int) {
        require(state.weights.size == topology.size - 1 && state.biases.size == state.weights.size &&
            state.weightVelocity.size == state.weights.size && state.biasVelocity.size == state.weights.size) { "Malformed SMALL CUDA layers." }
        var offset = model * parameterCount * 2
        for (layer in state.weights.indices) {
            require(state.weights[layer].size == topology[layer] * topology[layer + 1] &&
                state.biases[layer].size == topology[layer + 1] && state.weightVelocity[layer].size == state.weights[layer].size &&
                state.biasVelocity[layer].size == state.biases[layer].size) { "Malformed SMALL CUDA parameter buffers." }
            state.weights[layer].copyInto(packed, offset)
            state.weightVelocity[layer].copyInto(packed, offset + parameterCount)
            offset += state.weights[layer].size
            state.biases[layer].copyInto(packed, offset)
            state.biasVelocity[layer].copyInto(packed, offset + parameterCount)
            offset += state.biases[layer].size
        }
    }

    private fun unpack(model: Int): NeuroTrainingState {
        var offset = model * parameterCount * 2
        val weights = Array(topology.size - 1) { DoubleArray(topology[it] * topology[it + 1]) }
        val biases = Array(weights.size) { DoubleArray(topology[it + 1]) }
        val velocities = Array(weights.size) { DoubleArray(weights[it].size) }
        val biasVelocities = Array(weights.size) { DoubleArray(biases[it].size) }
        for (layer in weights.indices) {
            packed.copyInto(weights[layer], 0, offset, offset + weights[layer].size)
            packed.copyInto(velocities[layer], 0, offset + parameterCount, offset + parameterCount + weights[layer].size)
            offset += weights[layer].size
            packed.copyInto(biases[layer], 0, offset, offset + biases[layer].size)
            packed.copyInto(biasVelocities[layer], 0, offset + parameterCount, offset + parameterCount + biases[layer].size)
            offset += biases[layer].size
        }
        return NeuroTrainingState(topology.copyOf(), weights, biases, velocities, biasVelocities, inputs.copyOf(), targets.copyOf())
    }

    private fun reserve(bytes: Long) {
        CudaMemoryBudget.acquire(info.identity, bytes, driver.availableMemory())
        reservation = Math.addExact(reservation, bytes)
    }

    private fun allocate(bytes: Long): Long = driver.allocate(maxOf(8L, bytes)).also {
        check(it != 0L) { "CUDA returned a null allocation." }
        allocations += it
    }

    @Synchronized
    fun train(orders: Array<Array<IntArray>>, batchSize: Int, online: Boolean,
              active: BooleanArray = BooleanArray(models) { true }): Array<NeuroTrainingState> {
        check(!closed && !failed) { "SMALL CUDA kernel is closed or failed." }
        require(batchSize > 0 && (!online || batchSize == 1)) { "Online SMALL CUDA requires batch size one." }
        require(orders.size == models && active.size == models) { "CUDA cohort order/activity count differs from model count." }
        val first = active.indexOfFirst { it }
        if (first < 0) return Array(models, ::unpack)
        check(samples > 0) { "no training samples" }
        val epochs = orders[first].size
        require(epochs in 1..64) { "SMALL CUDA chunks contain 1..64 epochs." }
        val length = Math.multiplyExact(Math.multiplyExact(models, epochs), samples)
        if (orderStaging.size != length) orderStaging = IntArray(length)
        val flat = orderStaging
        if (active.any { !it }) flat.fill(0)
        seen.fill(0)
        var marker = 0
        for (model in 0 until models) if (active[model]) {
            require(orders[model].size == epochs) { "Active CUDA cohort models need the same chunk length." }
            for (epoch in 0 until epochs) {
                val order = orders[model][epoch]
                require(order.size == samples) { "CUDA shuffle length differs from the dataset." }
                marker++
                for (sample in order) {
                    require(sample in 0 until samples && seen[sample] != marker) { "CUDA shuffle must be a permutation." }
                    seen[sample] = marker
                }
                order.copyInto(flat, (model * epochs + epoch) * samples)
            }
        }
        try {
            val bytes = Math.multiplyExact(flat.size.toLong(), 4L)
            if (bytes > orderBytes) {
                driver.synchronize()
                reserve(bytes - orderBytes)
                if (orderPointer != 0L) {
                    driver.free(orderPointer)
                    allocations.remove(orderPointer)
                    orderPointer = 0L
                }
                orderPointer = allocate(bytes)
                orderBytes = bytes
            }
            driver.upload(orderPointer, flat)
            for (model in 0 until models) activeStaging[model] = if (active[model]) 1 else 0
            driver.upload(activePointer, activeStaging)
            driver.launch(if (precision == Neuro.TrainingPrecision.FP64) "small_train_fp64" else "small_train_fp32",
                Math.multiplyExact(models, 128), statePointer, inputPointer, targetPointer, orderPointer, topologyPointer, activePointer,
                models, topology.size - 1, samples, epochs, minOf(batchSize, samples),
                parameters.learningRate, parameters.momentum, parameters.beta, parameters.sigmoidMode.ordinal, if (online) 1 else 0)
            driver.synchronize()
            val result = downloadStaging
            driver.download(statePointer, result)
            check(result.all { it.isFinite() }) { "SMALL CUDA produced non-finite parameters or momentum." }
            downloadStaging = packed
            packed = result
            NeuroLog.debug("cuda", "small.chunk.staged") { mapOf("models" to active.count { it }, "epochs" to epochs,
                "batchSize" to batchSize, "precision" to precision, "packedBytes" to result.size * 8L) }
            return Array(models, ::unpack)
        } catch (failure: Throwable) {
            failed = true
            NeuroLog.error("cuda", "small.chunk.failed", failure, "models" to models, "epochs" to epochs)
            throw failure
        }
    }

    @Synchronized override fun close() {
        if (closed) return
        closed = true
        var failure: Throwable? = null
        fun cleanup(action: () -> Unit) {
            try { action() } catch (problem: Throwable) {
                if (failure == null) failure = problem else failure.addSuppressed(problem)
            }
        }
        cleanup { driver.synchronize() }
        for (pointer in allocations.asReversed()) cleanup { driver.free(pointer) }
        cleanup { driver.close() }
        CudaMemoryBudget.release(info.identity, reservation)
        failure?.let { throw it }
    }
}

/** Explicit owner for warm driver/module/stream leases. Close sessions before closing the service. */
internal class SmallCudaDeviceService(private val maximumDrivers: Int = 1,
                                     private val factory: () -> CudaDriver = { NativeCudaDriver() }) : AutoCloseable {
    private data class Buffer(val pointer: Long, val bytes: Long)
    private class Retained(val driver: CudaDriver) {
        val pool = ArrayList<Buffer>()
        var bytes = 0L

        fun close() {
            var failure: Throwable? = null
            for (buffer in pool.asReversed()) try { driver.free(buffer.pointer) } catch (problem: Throwable) {
                if (failure == null) failure = problem else failure.addSuppressed(problem)
            }
            pool.clear(); bytes = 0
            try { driver.close() } catch (problem: Throwable) {
                if (failure == null) failure = problem else failure.addSuppressed(problem)
            }
            failure?.let { throw it }
        }
    }
    private val idle = ArrayDeque<Retained>()
    private var leased = 0
    private var closed = false
    init { require(maximumDrivers > 0) }

    @Synchronized fun openDriver(): CudaDriver {
        check(!closed) { "SMALL CUDA device service is closed." }
        check(leased < maximumDrivers) { "SMALL CUDA device service has no available lease." }
        val driver = if (idle.isEmpty()) Retained(factory()) else idle.removeFirst()
        leased++
        return Lease(driver)
    }

    private inner class Lease(private val retained: Retained) : CudaDriver {
        private val driver = retained.driver
        private val live = LinkedHashMap<Long, Long>()
        private var released = false
        private var broken = false
        private var pending = false
        override val info: TrainingDeviceInfo get() = driver.info
        private fun <T> call(action: () -> T): T {
            check(!released) { "CUDA device lease is closed." }
            return try { action() } catch (failure: Throwable) { broken = true; throw failure }
        }
        override fun availableMemory(): Long = call { driver.availableMemory() }
        override fun allocate(bytes: Long): Long = call {
            require(bytes > 0) { "CUDA allocation size must be positive" }
            val match = retained.pool.indexOfFirst { it.bytes == bytes }
            val pointer = if (match >= 0) retained.pool.removeAt(match).also { retained.bytes -= it.bytes }.pointer
                else driver.allocate(bytes)
            check(pointer != 0L) { "CUDA returned a null allocation" }
            live[pointer] = bytes
            if (match >= 0) NeuroLog.debug("cuda", "pool.buffer.reused") { mapOf("bytes" to bytes,
                "pooledBytes" to retained.bytes, "pooledBuffers" to retained.pool.size) }
            pointer
        }
        override fun free(pointer: Long): Unit = call {
            val bytes = requireNotNull(live[pointer]) { "CUDA pointer does not belong to this lease" }
            // Reuse must not turn cuMemFree's completion boundary into a use-after-free.
            if (!broken && pending) { driver.synchronize(); pending = false }
            release(Buffer(pointer, bytes))
            live.remove(pointer)
        }
        private fun release(buffer: Buffer) {
            if (!broken && retained.pool.size < 64 && buffer.bytes <= 1024L * 1024 - retained.bytes) {
                retained.pool += buffer
                retained.bytes += buffer.bytes
                NeuroLog.debug("cuda", "pool.buffer.retained") { mapOf("bytes" to buffer.bytes,
                    "pooledBytes" to retained.bytes, "pooledBuffers" to retained.pool.size) }
            } else driver.free(buffer.pointer)
        }
        override fun upload(pointer: Long, values: DoubleArray) = call { driver.upload(pointer, values) }
        override fun upload(pointer: Long, values: IntArray) = call { driver.upload(pointer, values) }
        override fun download(pointer: Long, values: DoubleArray) = call { driver.download(pointer, values) }
        override fun launch(name: String, workItems: Int, vararg arguments: Any) = call {
            driver.launch(name, workItems, *arguments)
            pending = true
        }
        override fun synchronize() = call { driver.synchronize(); pending = false }
        @Synchronized override fun close() {
            if (released) return
            released = true
            var failure: Throwable? = null
            fun cleanup(action: () -> Unit) {
                try { action() } catch (problem: Throwable) {
                    broken = true
                    if (failure == null) failure = problem else failure.addSuppressed(problem)
                }
            }
            if (pending) cleanup { driver.synchronize() }
            for ((pointer, bytes) in live) cleanup { release(Buffer(pointer, bytes)) }
            live.clear()
            synchronized(this@SmallCudaDeviceService) {
                leased--
                if (broken) cleanup { retained.close() } else idle.addLast(retained)
            }
            failure?.let { throw it }
        }
    }

    @Synchronized override fun close() {
        if (closed) return
        check(leased == 0) { "Close all CUDA sessions before closing their device service." }
        closed = true
        var failure: Throwable? = null
        while (idle.isNotEmpty()) {
            try { idle.removeFirst().close() } catch (problem: Throwable) {
                if (failure == null) failure = problem else failure.addSuppressed(problem)
            }
        }
        failure?.let { throw it }
    }
}
