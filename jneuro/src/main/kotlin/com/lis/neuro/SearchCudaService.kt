package com.lis.neuro

import java.util.ArrayDeque
import java.util.concurrent.CompletableFuture
import java.util.concurrent.ExecutionException
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

/** Search scheduling submits without occupying a CPU scoring worker while CUDA runs. */
internal interface AsyncSearchEpochAdvancer {
    fun advanceForSearchAsync(request: TrainingChunkRequest): CompletableFuture<SearchAdvanceResult>
}

/** One explicit search owner, one native thread/stream, and one immutable dataset upload. */
internal class SearchCudaService(
    private val precision: Neuro.TrainingPrecision,
    private val batchSize: Int,
    private val driverFactory: () -> CudaDriver = { NativeCudaDriver() },
    private val maximumModels: Int = 64,
    private val clock: () -> Long = System::nanoTime
) : AutoCloseable {
    init { require(maximumModels in 1..64 && batchSize > 0) }

    private val lock = Any()
    private var nativeThread: Thread? = null
    private val executor = Executors.newSingleThreadExecutor { task ->
        Thread(task, "jneuro-search-cuda").apply { isDaemon = true; nativeThread = this }
    }
    private val sessions = arrayOfNulls<SearchCudaSession>(maximumModels)
    private val requests = ArrayDeque<Pending>()
    private var scheduled = false
    private var closed = false
    private var failure: Throwable? = null
    private var peak = 0
    private var launches = 0L
    private var driver: CudaDriver? = null
    private var dataset: NeuroTrainingState? = null
    private var parameters: Neuro.HyperParameters? = null
    private val allocations = ArrayList<Long>()
    private var reservedBytes = 0L
    private var statesPointer = 0L
    private var inputsPointer = 0L
    private var targetsPointer = 0L
    private var topologyPointer = 0L
    private var metadataPointer = 0L
    private var outputPointer = 0L
    private var ordersPointer = 0L
    private var orderCapacity = 0
    private val metadata = IntArray(maximumModels * 6)
    private var orderStaging = IntArray(0)
    private var outputStaging = DoubleArray(0)

    val residentModels: Int get() = synchronized(lock) { sessions.count { it != null } }
    val peakResidentModels: Int get() = synchronized(lock) { peak }
    val queuedRequests: Int get() = synchronized(lock) { requests.size }
    val batchesLaunched: Long get() = synchronized(lock) { launches }
    internal val configuredBatchSize: Int get() = batchSize

    fun openSession(model: Neuro): SearchCudaSession {
        val session = SearchCudaSession(this, model)
        val slot = synchronized(lock) {
            checkAvailable()
            val index = sessions.indexOfFirst { it == null }
            check(index >= 0) { "CUDA search resident-model limit reached ($maximumModels)." }
            sessions[index] = session
            peak = maxOf(peak, sessions.count { it != null })
            index
        }
        try {
            session.acquire(slot)
            val state = model.withTraining(session) { model.exportTrainingState(shareDataset = true) }
            validateSmallState(state)
            require(state.samples > 0) { "CUDA search requires training samples." }
            require(state.samples <= MAX_ORDER_ELEMENTS / maximumModels) { "CUDA search dataset exceeds the bounded shuffle workspace." }
            val hp = model.hyperParameters()
            if (precision == Neuro.TrainingPrecision.FP32) require(
                listOf(hp.learningRate, hp.momentum, hp.beta).all { it.toFloat().isFinite() }) {
                "Hyperparameters exceed FP32 compute precision."
            }
            onNative {
                synchronized(lock) { checkAvailable() }
                dataset?.let { shared ->
                    require(shared.inputs.contentEquals(state.inputs) && shared.targets.contentEquals(state.targets) &&
                        parameters == hp.copy(seed = 0)) { "CUDA search sessions must share dataset and hyperparameters except seed." }
                }
                try {
                    if (driver == null) initialize(state, hp)
                    val native = checkNotNull(driver)
                    val packed = pack(state)
                    native.upload(statesPointer + slot * STATE_STRIDE * 8L, packed)
                    native.upload(topologyPointer + slot * TOPOLOGY_STRIDE * 4L, state.topology)
                    session.ready(state, native.info.copy(precision = precision.name, engine = TrainingEngine.SMALL,
                        sigmoid = hp.sigmoidMode.name, kernelVersion = "small-search-v3/packed-fp64/" + native.info.kernelVersion))
                } catch (problem: Throwable) {
                    failService(problem)
                    throw problem
                }
            }
            return session
        } catch (problem: Throwable) {
            session.releaseAcquisition()
            synchronized(lock) { sessions[slot] = null }
            throw problem
        }
    }

    private fun initialize(state: NeuroTrainingState, hp: Neuro.HyperParameters) {
        val native = driverFactory()
        driver = native
        dataset = state
        parameters = hp.copy(seed = 0)
        statesPointer = allocate(maximumModels * STATE_STRIDE * 8L)
        topologyPointer = allocate(maximumModels * TOPOLOGY_STRIDE * 4L)
        metadataPointer = allocate(metadata.size * 4L)
        outputPointer = allocate(maximumModels * (STATE_STRIDE + 1) * 8L)
        inputsPointer = allocate(state.inputs.size * 8L).also { native.upload(it, state.inputs) }
        targetsPointer = allocate(state.targets.size * 8L).also { native.upload(it, state.targets) }
    }

    private fun allocate(bytes: Long): Long {
        val native = checkNotNull(driver)
        val size = maxOf(8L, bytes)
        CudaMemoryBudget.acquire(native.info.identity, size, native.availableMemory())
        reservedBytes += size
        val pointer = native.allocate(size)
        check(pointer != 0L) { "CUDA returned a null allocation." }
        allocations += pointer
        return pointer
    }

    private fun checkAvailable() {
        check(!closed) { "CUDA search service is closed." }
        failure?.let { throw IllegalStateException("CUDA search service failed; create a new service from committed checkpoints.", it) }
    }

    internal fun submit(session: SearchCudaSession, request: TrainingChunkRequest): CompletableFuture<SearchAdvanceResult> {
        require(request.targetError == null) { "Search scoring belongs to the CPU evaluator." }
        val pending = Pending(session, request, clock())
        synchronized(lock) {
            checkAvailable()
            session.install(pending)
            requests.addLast(pending)
            if (!scheduled) {
                scheduled = true
                executor.execute(::pump)
            }
        }
        return pending.result
    }

    private fun pump() {
        val batch = synchronized(lock) {
            buildList { while (requests.isNotEmpty() && size < maximumModels) add(requests.removeFirst()) }
        }
        try {
            val ready = batch.filter { pending ->
                try {
                    val reason = termination(pending)
                    if (reason != null) { finish(pending, reason); false } else true
                } catch (problem: Throwable) { reject(pending, problem); false }
            }
            if (ready.isNotEmpty()) compute(ready)
        } catch (problem: Throwable) {
            failService(problem)
            batch.forEach { reject(it, problem) }
        } finally {
            synchronized(lock) {
                if (requests.isNotEmpty() && failure == null && !closed) executor.execute(::pump)
                else scheduled = false
            }
        }
    }

    private fun termination(pending: Pending): TrainingTermination? = when {
        pending.session.closing || pending.result.isCancelled || pending.request.cancelled() -> TrainingTermination.CANCELLED
        pending.completed == pending.limit -> if (pending.completed == pending.request.maxEpochs)
            TrainingTermination.COMPLETED else TrainingTermination.BUDGET
        pending.completed > 0 && clock() - pending.started >= pending.request.maxNanos -> TrainingTermination.BUDGET
        else -> null
    }

    private fun compute(batch: List<Pending>) {
        val native = checkNotNull(driver)
        val data = checkNotNull(dataset)
        val hp = checkNotNull(parameters)
        val samples = data.samples
        val maximumEpochs = maxOf(1, MAX_ORDER_ELEMENTS / batch.size / samples)
        val counts = IntArray(batch.size)
        val orders = arrayOfNulls<Array<IntArray>>(batch.size)
        var orderSize = 0
        var outputSize = 0
        for ((lane, pending) in batch.withIndex()) {
            val session = pending.session
            val remainingNanos = maxOf(1L, pending.request.maxNanos - (clock() - pending.started))
            val adaptive = if (session.nanosPerEpoch == 0L) 1 else (remainingNanos / session.nanosPerEpoch).coerceIn(1, 64).toInt()
            val count = minOf(pending.limit - pending.completed, adaptive, maximumEpochs)
            counts[lane] = count
            orders[lane] = session.model.withTraining(session) { session.model.reserveTrainingOrders(count) }
            val at = lane * 6
            metadata[at] = session.slot * STATE_STRIDE
            metadata[at + 1] = session.slot * TOPOLOGY_STRIDE
            metadata[at + 2] = orderSize
            metadata[at + 3] = session.state.topology.size - 1
            metadata[at + 4] = count
            metadata[at + 5] = outputSize
            orderSize += count * samples
            outputSize += session.parameterCount * 2 + 1
        }
        if (orderStaging.size != orderSize) orderStaging = IntArray(orderSize)
        if (outputStaging.size != outputSize) outputStaging = DoubleArray(outputSize)
        for (lane in batch.indices) {
            var offset = metadata[lane * 6 + 2]
            for (order in checkNotNull(orders[lane])) { order.copyInto(orderStaging, offset); offset += samples }
        }
        if (orderSize > orderCapacity) {
            if (ordersPointer != 0L) {
                native.free(ordersPointer)
                allocations.remove(ordersPointer)
                CudaMemoryBudget.release(native.info.identity, orderCapacity * 4L)
                reservedBytes -= orderCapacity * 4L
                ordersPointer = 0L
            }
            ordersPointer = allocate(orderSize * 4L)
            orderCapacity = orderSize
        }
        val started = clock()
        native.upload(ordersPointer, orderStaging)
        native.upload(metadataPointer, metadata)
        native.launch(if (precision == Neuro.TrainingPrecision.FP64) "search_train_fp64" else "search_train_fp32",
            batch.size * 128, statesPointer, inputsPointer, targetsPointer, ordersPointer, topologyPointer, metadataPointer,
            outputPointer, batch.size, samples, minOf(batchSize, samples), hp.learningRate, hp.momentum, hp.beta,
            hp.sigmoidMode.ordinal, if (batchSize == 1) 1 else 0)
        synchronized(lock) { launches++ }
        native.synchronize()
        native.download(outputPointer, outputStaging)
        // No publication before the entire DMA completes. Numerical failures are
        // classified per lane; a transport failure rejects the whole batch.
        val staged = arrayOfNulls<NeuroTrainingState>(batch.size)
        for ((lane, pending) in batch.withIndex()) {
            try {
                val offset = metadata[lane * 6 + 5]
                check(outputStaging[offset] == 0.0) { "CUDA search model produced non-finite compute state." }
                staged[lane] = unpack(pending.session.state, outputStaging, offset + 1)
                pending.session.model.validateTrainingState(checkNotNull(staged[lane]))
            } catch (problem: Throwable) { staged[lane] = null; reject(pending, problem) }
        }
        val elapsed = maxOf(1L, clock() - started)
        for ((lane, pending) in batch.withIndex()) {
            val state = staged[lane] ?: continue
            val session = pending.session
            session.model.withTraining(session) { session.model.commitTrainingChunk(state, counts[lane], evaluateError = false) }
            session.nanosPerEpoch = maxOf(1L, elapsed / counts[lane])
            pending.completed += counts[lane]
            val reason = try { termination(pending) } catch (problem: Throwable) { reject(pending, problem); continue }
            if (reason != null) finish(pending, reason)
            else synchronized(lock) { requests.addLast(pending) }
        }
    }

    private fun finish(pending: Pending, termination: TrainingTermination) {
        pending.session.finished(pending)
        pending.drained.complete(Unit)
        pending.result.complete(SearchAdvanceResult(pending.completed, null, termination))
    }

    private fun reject(pending: Pending, problem: Throwable) {
        pending.session.failed(problem)
        pending.session.finished(pending)
        pending.drained.complete(Unit)
        pending.result.completeExceptionally(problem)
    }

    private fun failService(problem: Throwable) {
        val waiting = synchronized(lock) {
            if (failure == null) failure = problem
            sessions.filterNotNull().forEach { it.failed(problem) }
            requests.toList().also { requests.clear() }
        }
        waiting.forEach { reject(it, problem) }
    }

    internal fun release(session: SearchCudaSession) {
        onNative { synchronized(lock) { check(sessions[session.slot] === session); sessions[session.slot] = null } }
    }

    private fun <T> onNative(action: () -> T): T = if (Thread.currentThread() === nativeThread) action()
        else awaitSearchCuda(executor.submit<T> { action() })

    override fun close() {
        synchronized(lock) {
            if (closed) return
            check(sessions.all { it == null }) { "Close all CUDA search sessions before closing their service." }
            closed = true
        }
        try {
            onNative {
                val native = driver ?: return@onNative
                var cleanupFailure: Throwable? = null
                fun attempt(action: () -> Unit) {
                    try { action() } catch (problem: Throwable) {
                        val first = cleanupFailure
                        if (first == null) cleanupFailure = problem else first.addSuppressed(problem)
                    }
                }
                attempt { native.synchronize() }
                for (pointer in allocations.asReversed()) attempt { native.free(pointer) }
                allocations.clear()
                attempt { native.close() }
                CudaMemoryBudget.release(native.info.identity, reservedBytes)
                cleanupFailure?.let { throw it }
            }
        } finally {
            executor.shutdown()
            if (Thread.currentThread() !== nativeThread) {
                var interrupted = false
                try {
                    while (!executor.isTerminated) {
                        try { executor.awaitTermination(1, TimeUnit.DAYS) }
                        catch (_: InterruptedException) { interrupted = true }
                    }
                } finally { if (interrupted) Thread.currentThread().interrupt() }
            }
        }
    }

    internal class Pending(val session: SearchCudaSession, val request: TrainingChunkRequest, val started: Long) {
        val result = CompletableFuture<SearchAdvanceResult>()
        val drained = CompletableFuture<Unit>()
        val limit = minOf(64, request.maxEpochs)
        var completed = 0
    }

    companion object {
        private const val STATE_STRIDE = 2 * 881
        private const val TOPOLOGY_STRIDE = 6
        private const val MAX_ORDER_ELEMENTS = 8 * 1024 * 1024
        private fun pack(state: NeuroTrainingState): DoubleArray {
            val count = state.weights.sumOf { it.size } + state.biases.sumOf { it.size }
            val output = DoubleArray(count * 2)
            var offset = 0
            for (layer in state.weights.indices) {
                state.weights[layer].copyInto(output, offset)
                state.weightVelocity[layer].copyInto(output, offset + count)
                offset += state.weights[layer].size
                state.biases[layer].copyInto(output, offset)
                state.biasVelocity[layer].copyInto(output, offset + count)
                offset += state.biases[layer].size
            }
            return output
        }
        private fun unpack(initial: NeuroTrainingState, packed: DoubleArray, start: Int): NeuroTrainingState {
            val count = initial.weights.sumOf { it.size } + initial.biases.sumOf { it.size }
            var offset = start
            val weights = Array(initial.weights.size) { DoubleArray(initial.weights[it].size) }
            val biases = Array(initial.biases.size) { DoubleArray(initial.biases[it].size) }
            val velocity = Array(weights.size) { DoubleArray(weights[it].size) }
            val biasVelocity = Array(biases.size) { DoubleArray(biases[it].size) }
            for (layer in weights.indices) {
                packed.copyInto(weights[layer], 0, offset, offset + weights[layer].size)
                packed.copyInto(velocity[layer], 0, offset + count, offset + count + weights[layer].size)
                offset += weights[layer].size
                packed.copyInto(biases[layer], 0, offset, offset + biases[layer].size)
                packed.copyInto(biasVelocity[layer], 0, offset + count, offset + count + biases[layer].size)
                offset += biases[layer].size
            }
            return NeuroTrainingState(initial.topology, weights, biases, velocity, biasVelocity,
                initial.inputs, initial.targets, sharedDataset = true)
        }
    }
}

/** Public-style synchronous calls exist for strict same-route replay, not CPU search dispatch. */
internal class SearchCudaSession(private val service: SearchCudaService, internal val model: Neuro) :
    NeuroTrainingSession, SearchEpochAdvancer, AsyncSearchEpochAdvancer {
    internal var slot = -1
        private set
    internal lateinit var state: NeuroTrainingState
        private set
    override lateinit var info: TrainingDeviceInfo
        private set
    internal val parameterCount: Int get() = state.weights.sumOf { it.size } + state.biases.sumOf { it.size }
    internal var nanosPerEpoch = 0L
    @Volatile internal var closing = false
        private set
    private var acquired = false
    private var closed = false
    private var failure: Throwable? = null
    private var pending: SearchCudaService.Pending? = null
    override val currentRmse: Double get() = model.statistics().lastTrainingError.let { if (it.isNaN()) model.trainingError() else it }

    internal fun acquire(index: Int) {
        model.acquireTraining(this, NeuroLog.id("search-cuda-session"))
        acquired = true
        slot = index
    }
    internal fun ready(initial: NeuroTrainingState, device: TrainingDeviceInfo) { state = initial; info = device }
    internal fun releaseAcquisition() { if (acquired) { model.releaseTraining(this); acquired = false } }
    @Synchronized private fun checkUsable() {
        check(!closing && !closed) { "CUDA search session is closed." }
        failure?.let { throw IllegalStateException("CUDA search session failed; close and reopen from its committed checkpoint.", it) }
    }
    @Synchronized internal fun install(next: SearchCudaService.Pending) {
        checkUsable()
        check(pending == null) { "Only one CUDA search request may be pending per model." }
        pending = next
    }
    @Synchronized internal fun finished(value: SearchCudaService.Pending) { if (pending === value) pending = null }
    @Synchronized internal fun failed(problem: Throwable) { if (failure == null) failure = problem }
    override fun advanceForSearchAsync(request: TrainingChunkRequest): CompletableFuture<SearchAdvanceResult> = service.submit(this, request)
    override fun advanceForSearch(request: TrainingChunkRequest): SearchAdvanceResult = awaitSearchCuda(advanceForSearchAsync(request))
    override fun trainChunk(request: TrainingChunkRequest): TrainingChunkResult {
        require(request.targetError == null) { "Search session target scoring must run on the CPU evaluator." }
        val result = advanceForSearch(request)
        return TrainingChunkResult(result.committedEpochs, currentRmse, result.termination)
    }
    override fun trainEpoch(): Double { advanceForSearch(TrainingChunkRequest(1)); return currentRmse }
    override fun train(epochs: Int) { checkUsable(); require(epochs >= 0); repeat(epochs) { trainEpoch() } }
    override fun trainUntil(targetError: Double, maxEpochs: Int, checkEvery: Int): Neuro.TrainingResult {
        checkUsable()
        require(targetError.isFinite() && targetError >= 0 && maxEpochs >= 0 && checkEvery > 0)
        var error = currentRmse
        var completed = 0
        if (error <= targetError) model.withTraining(this) { model.recordTrainingError(error) }
        while (completed < maxEpochs && error > targetError) {
            trainEpoch(); completed++
            if (completed % checkEvery == 0 || completed == maxEpochs) error = currentRmse
        }
        return Neuro.TrainingResult(completed, error, error <= targetError)
    }
    override fun trainMiniBatch(epochs: Int, batchSize: Int, parallelism: Int) {
        // A search queue has one optimizer/batch configuration for every resident model.
        require(batchSize > 1 && batchSize == service.configuredBatchSize && parallelism == 1) { "Use the configured search batch through epoch/chunk calls." }
        train(epochs)
    }
    override fun close() {
        val active = synchronized(this) { if (closed) return; closing = true; pending }
        if (active != null) awaitSearchCuda(active.drained)
        synchronized(this) {
            if (closed) return
            closed = true
        }
        try { service.release(this) } finally { releaseAcquisition() }
    }
}

private fun <T> awaitSearchCuda(future: java.util.concurrent.Future<T>): T {
    var interrupted = false
    try {
        while (true) {
            try { return future.get() }
            catch (_: InterruptedException) { interrupted = true }
            catch (problem: ExecutionException) { throw problem.cause ?: problem }
        }
    } finally { if (interrupted) Thread.currentThread().interrupt() }
}
