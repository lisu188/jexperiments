package com.lis.neuro

import java.util.concurrent.Callable
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/** Compatible independent trials. One CPU model per SIMD lane, or one CUDA block per model. */
class NeuroTrainingCohort private constructor(
    private val models: List<Neuro>,
    private val backend: TrainingBackend,
    private val precision: Neuro.TrainingPrecision,
    private val batchSize: Int,
    parallelism: Int,
    private val driverFactory: () -> CudaDriver
) : AutoCloseable {
    private class Group(val indices: IntRange, val info: TrainingDeviceInfo,
                        val train: (Array<Array<IntArray>>, BooleanArray) -> Array<NeuroTrainingState>,
                        val close: () -> Unit)
    private val groups = ArrayList<Group>()
    private var pool: ExecutorService? = null
    private var closed = false
    private var failed = false
    private var nanosPerEpoch = 0L
    val info: TrainingDeviceInfo get() = groups.first().info

    init {
        require(models.isNotEmpty() && models.distinct().size == models.size) { "Cohorts require distinct models" }
        require(batchSize > 0 && parallelism in 1..256)
        require(backend == TrainingBackend.CPU || backend == TrainingBackend.CUDA)
        val first = models.first()
        val parameters = first.hyperParameters()
        val shape = first.topology()
        require(SmallNetworkShape.supports(shape)) { "Unsupported SMALL cohort topology" }
        require(models.all { it.topology().contentEquals(shape) &&
            it.hyperParameters().copy(seed = parameters.seed) == parameters }) { "Cohort models must share topology and hyperparameters except seed" }
        var acquired = 0
        try {
            for (model in models) { model.acquireTraining(this, NeuroLog.id("cohort")); acquired++ }
            val states = models.map { it.exportTrainingState() }
            require(states.all { it.inputs.contentEquals(states.first().inputs) && it.targets.contentEquals(states.first().targets) }) {
                "Cohort models must share immutable dataset values"
            }
            val width = if (backend == TrainingBackend.CUDA) models.size else if (precision == Neuro.TrainingPrecision.FP64) 4 else 8
            for (start in models.indices step width) {
                val end = minOf(models.size, start + width)
                val input = states.subList(start, end).toTypedArray()
                if (backend == TrainingBackend.CUDA) {
                    val kernel = SmallCudaCohort(input, parameters, precision, driverFactory())
                    groups += Group(start until end, kernel.info,
                        { orders, active -> kernel.train(orders, batchSize, batchSize == 1, active) }, kernel::close)
                } else {
                    val kernel = SmallCpuCohort(input, parameters, precision)
                    groups += Group(start until end, kernel.info,
                        { orders, active -> kernel.train(orders, batchSize, batchSize == 1, active) }, kernel::close)
                }
            }
            if (backend == TrainingBackend.CPU && parallelism > 1 && groups.size > 1)
                pool = Executors.newFixedThreadPool(minOf(parallelism, groups.size))
        } catch (failure: Throwable) {
            for (group in groups.asReversed()) try { group.close() } catch (cleanup: Throwable) { failure.addSuppressed(cleanup) }
            repeat(acquired) { models[it].releaseTraining(this) }
            throw failure
        }
    }

    @Synchronized fun trainChunk(request: TrainingChunkRequest,
                                active: BooleanArray = BooleanArray(models.size) { true }): List<TrainingChunkResult> {
        check(!closed && !failed) { "Cohort is closed or failed; reopen from the last committed host states" }
        require(active.size == models.size)
        val enabled = active.copyOf()
        val completed = IntArray(models.size)
        val errors = DoubleArray(models.size) { index -> models[index].statistics().lastTrainingError.let {
            if (it.isNaN()) models[index].trainingError() else it
        } }
        val reasons = Array(models.size) { if (active[it]) TrainingTermination.COMPLETED else TrainingTermination.CANCELLED }
        for (index in models.indices) if (enabled[index] && request.targetError != null && errors[index] <= request.targetError) {
            enabled[index] = false; reasons[index] = TrainingTermination.CONVERGED
            models[index].withTraining(this) { models[index].recordTrainingError(errors[index]) }
        }
        val started = System.nanoTime()
        var progress = 0
        val limit = minOf(64, request.maxEpochs)
        while (progress < limit && enabled.any { it }) {
            if (request.cancelled()) {
                for (index in models.indices) if (enabled[index]) reasons[index] = TrainingTermination.CANCELLED
                break
            }
            val elapsed = System.nanoTime() - started
            if (progress > 0 && elapsed >= request.maxNanos) break
            val adaptive = if (nanosPerEpoch == 0L) 1 else ((request.maxNanos - elapsed) / nanosPerEpoch).coerceIn(1, 64).toInt()
            val scoring = if (request.targetError == null) limit else request.checkEvery - progress % request.checkEvery
            val count = minOf(limit - progress, adaptive, scoring)
            val orders = Array(models.size) { index ->
                if (enabled[index]) models[index].withTraining(this) { models[index].reserveTrainingOrders(count) } else emptyArray()
            }
            val computeStarted = System.nanoTime()
            try {
                val tasks = groups.map { group -> Callable {
                    if (group.indices.none { enabled[it] }) null
                    else group.train(Array(group.indices.count()) { orders[group.indices.first + it] },
                        BooleanArray(group.indices.count()) { enabled[group.indices.first + it] })
                } }
                val states = finishTasks(tasks)
                // No successful sibling model is published until every complete result has passed validation.
                for ((groupIndex, group) in groups.withIndex()) for (index in group.indices) if (enabled[index])
                    models[index].validateTrainingState(checkNotNull(states[groupIndex])[index - group.indices.first])
                for ((groupIndex, group) in groups.withIndex()) for (index in group.indices) if (enabled[index]) {
                    errors[index] = models[index].withTraining(this) {
                        models[index].commitTrainingChunk(checkNotNull(states[groupIndex])[index - group.indices.first], count)
                    }
                    completed[index] += count
                    if (request.targetError != null && ((progress + count) % request.checkEvery == 0 || progress + count == request.maxEpochs) && errors[index] <= request.targetError) {
                        enabled[index] = false; reasons[index] = TrainingTermination.CONVERGED
                    }
                }
                nanosPerEpoch = maxOf(1L, (System.nanoTime() - computeStarted) / count)
                progress += count
            } catch (failure: Throwable) {
                failed = true
                throw failure
            }
        }
        return models.indices.map { index ->
            if (reasons[index] == TrainingTermination.COMPLETED && completed[index] < request.maxEpochs)
                reasons[index] = TrainingTermination.BUDGET
            TrainingChunkResult(completed[index], errors[index], reasons[index])
        }
    }

    private fun finishTasks(tasks: List<Callable<Array<NeuroTrainingState>?>>): List<Array<NeuroTrainingState>?> {
        val executor = pool ?: return tasks.map { it.call() }
        val futures = tasks.map { executor.submit(it) }
        val states = ArrayList<Array<NeuroTrainingState>?>(tasks.size)
        var interrupted = false
        var failure: Throwable? = null
        for (future in futures) {
            while (true) {
                try { states += future.get(); break }
                catch (_: InterruptedException) { interrupted = true }
                catch (problem: java.util.concurrent.ExecutionException) {
                    val cause = problem.cause ?: problem
                    if (failure == null) failure = cause else failure.addSuppressed(cause)
                    states += null; break
                }
            }
        }
        // An interrupted Future is not proof its native work stopped. Join every task before releasing buffers.
        if (interrupted) {
            Thread.currentThread().interrupt()
            val cancellation = InterruptedException("Interrupted cohort retained its last committed host checkpoint")
            if (failure == null) failure = cancellation else failure.addSuppressed(cancellation)
        }
        failure?.let { throw it }
        return states
    }

    @Synchronized override fun close() {
        if (closed) return
        closed = true
        pool?.shutdown()
        var failure: Throwable? = null
        for (group in groups.asReversed()) try { group.close() } catch (problem: Throwable) {
            if (failure == null) failure = problem else failure.addSuppressed(problem)
        }
        for (model in models) model.releaseTraining(this)
        failure?.let { throw it }
    }

    companion object {
        @JvmStatic @JvmOverloads
        fun open(models: List<Neuro>, backend: TrainingBackend = TrainingBackend.CPU,
                 precision: Neuro.TrainingPrecision = Neuro.TrainingPrecision.FP64,
                 batchSize: Int = 1, parallelism: Int = 4): NeuroTrainingCohort =
            openConfigured(models, backend, precision, batchSize, parallelism)

        internal fun openConfigured(models: List<Neuro>, backend: TrainingBackend,
                                    precision: Neuro.TrainingPrecision, batchSize: Int, parallelism: Int,
                                    driverFactory: () -> CudaDriver = { NativeCudaDriver() }): NeuroTrainingCohort =
            NeuroTrainingCohort(java.util.List.copyOf(models), if (backend == TrainingBackend.AUTO) TrainingBackend.CPU else backend,
                precision, batchSize, parallelism, driverFactory)
    }
}
