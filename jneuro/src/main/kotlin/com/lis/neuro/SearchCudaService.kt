package com.lis.neuro

import java.util.ArrayDeque
import java.util.concurrent.CompletableFuture
import java.util.concurrent.ExecutionException
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

/** Search scheduling submits without occupying a CPU scoring worker during numerical work. */
internal interface AsyncSearchEpochAdvancer {
    fun advanceForSearchAsync(request: TrainingChunkRequest): CompletableFuture<SearchAdvanceResult>
}

/** Bounded TensorFlow model admission and one numerical worker; the legacy queue API supports replay. */
internal class SearchCudaService(
    private val precision: Neuro.TrainingPrecision,
    private val batchSize: Int,
    private val kernelFactory: (NeuroTrainingState, Neuro.HyperParameters) -> SmallTrainingKernel = { state, hp ->
        TensorFlowMath.trainingKernel(state, hp, precision, TrainingBackend.CUDA, TrainingEngine.SMALL)
    },
    private val maximumModels: Int = 64,
    private val clock: () -> Long = System::nanoTime
) : AutoCloseable {
    init { require(maximumModels in 1..64 && batchSize > 0) }
    private val lock = Any()
    private var nativeThread: Thread? = null
    private val executor = Executors.newSingleThreadExecutor { task ->
        Thread(task, "jneuro-search-tensorflow").apply { isDaemon = true; nativeThread = this }
    }
    private val sessions = arrayOfNulls<SearchCudaSession>(maximumModels)
    private val requests = ArrayDeque<Pending>()
    private var scheduled = false
    private var closed = false
    private var failure: Throwable? = null
    private var peak = 0
    private var launches = 0L
    private var dataset: NeuroTrainingState? = null
    private var parameters: Neuro.HyperParameters? = null

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
            check(index >= 0) { "TensorFlow search resident-model limit reached ($maximumModels)." }
            sessions[index] = session
            peak = maxOf(peak, sessions.count { it != null })
            index
        }
        try {
            session.acquire(slot)
            val state = model.withTraining(session) { model.exportTrainingState(shareDataset = true) }
            validateSmallState(state)
            require(state.samples > 0) { "TensorFlow search requires training samples." }
            require(state.samples <= MAX_ORDER_ELEMENTS / maximumModels) { "Search dataset exceeds the bounded shuffle workspace." }
            val hp = model.hyperParameters()
            onNative {
                synchronized(lock) { checkAvailable() }
                dataset?.let { shared ->
                    require(shared.inputs.contentEquals(state.inputs) && shared.targets.contentEquals(state.targets) &&
                        parameters == hp.copy(seed = 0)) { "Search sessions must share dataset and hyperparameters except seed." }
                }
                val kernel = kernelFactory(state, hp)
                session.ready(kernel)
                if (dataset == null) { dataset = state; parameters = hp.copy(seed = 0) }
            }
            return session
        } catch (problem: Throwable) {
            session.releaseAcquisition()
            synchronized(lock) { sessions[slot] = null }
            throw problem
        }
    }

    private fun checkAvailable() {
        check(!closed) { "TensorFlow search service is closed." }
        failure?.let { throw IllegalStateException("Search service failed; reopen from committed checkpoints.", it) }
    }

    internal fun submit(session: SearchCudaSession, request: TrainingChunkRequest): CompletableFuture<SearchAdvanceResult> {
        require(request.targetError == null) { "Search scoring belongs to the evaluator." }
        val pending = Pending(session, request, clock())
        synchronized(lock) {
            checkAvailable()
            session.install(pending)
            requests.addLast(pending)
            if (!scheduled) { scheduled = true; executor.execute(::pump) }
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
            val waiting = synchronized(lock) {
                failure = problem
                sessions.filterNotNull().forEach { it.failed(problem) }
                requests.toList().also { requests.clear() }
            }
            (waiting + batch).forEach { reject(it, problem) }
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
        val samples = checkNotNull(dataset).samples
        val maximumEpochs = maxOf(1, MAX_ORDER_ELEMENTS / batch.size / samples)
        synchronized(lock) { launches++ }
        for (pending in batch) {
            val session = pending.session
            try {
                // A prior lane can finish while cancellation arrives; do not start another lane unnecessarily.
                val stopped = termination(pending)
                if (stopped != null) { finish(pending, stopped); continue }
                val remaining = maxOf(1L, pending.request.maxNanos - (clock() - pending.started))
                val adaptive = if (session.nanosPerEpoch == 0L) 1 else (remaining / session.nanosPerEpoch).coerceIn(1, 64).toInt()
                val count = minOf(pending.limit - pending.completed, adaptive, maximumEpochs)
                val orders = session.model.withTraining(session) { session.model.reserveTrainingOrders(count) }
                val started = clock()
                val state = session.kernel.train(orders, batchSize, batchSize == 1)
                // TensorFlow results are copied and validated before publication or shuffle advancement.
                session.model.withTraining(session) { session.model.commitTrainingChunk(state, count, evaluateError = false) }
                session.nanosPerEpoch = maxOf(1L, (clock() - started) / count)
                pending.completed += count
                val reason = termination(pending)
                if (reason != null) finish(pending, reason)
                else synchronized(lock) { requests.addLast(pending) }
            } catch (problem: Throwable) { reject(pending, problem) }
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

    internal fun release(session: SearchCudaSession) {
        onNative {
            try { session.kernel.close() }
            finally { synchronized(lock) { check(sessions[session.slot] === session); sessions[session.slot] = null } }
        }
    }

    private fun <T> onNative(action: () -> T): T = if (Thread.currentThread() === nativeThread) action()
        else awaitSearchCuda(executor.submit<T> { action() })

    override fun close() {
        synchronized(lock) {
            if (closed) return
            check(sessions.all { it == null }) { "Close all search sessions before closing their service." }
            closed = true
        }
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

    internal class Pending(val session: SearchCudaSession, val request: TrainingChunkRequest, val started: Long) {
        val result = CompletableFuture<SearchAdvanceResult>()
        val drained = CompletableFuture<Unit>()
        val limit = minOf(64, request.maxEpochs)
        var completed = 0
    }

    companion object { private const val MAX_ORDER_ELEMENTS = 8 * 1024 * 1024 }
}

/** Public-style synchronous calls exist for strict same-route replay, not CPU search dispatch. */
internal class SearchCudaSession(private val service: SearchCudaService, internal val model: Neuro) :
    NeuroTrainingSession, SearchEpochAdvancer, AsyncSearchEpochAdvancer {
    internal var slot = -1
        private set
    internal lateinit var kernel: SmallTrainingKernel
        private set
    override lateinit var info: TrainingDeviceInfo
        private set
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
    internal fun ready(compute: SmallTrainingKernel) { kernel = compute; info = compute.info }
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
