package com.lis.neuro

import java.util.concurrent.CompletableFuture
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.Executor
import java.util.concurrent.Executors
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

internal typealias AsyncArchitectureEvaluator = (NetworkArchitecture, Long, () -> Boolean,
    (Int, Double) -> Unit, Executor) -> CompletableFuture<ArchitectureTrial>

internal data class ArchitectureDeviceActivity(val residentModels: Int = 0, val peakResidentModels: Int = 0,
    val queuedRequests: Int = 0, val batches: Long = 0)

/** Waiting GPU models consume admission slots, never threads from the bounded CPU scoring pool. */
internal class ArchitectureTrialDispatcher(
    parallelism: Int,
    private val evaluate: (NetworkArchitecture, Long, () -> Boolean, (Int, Double) -> Unit) -> ArchitectureTrial,
    private val evaluateGroup: ((NetworkArchitecture, List<Long>, () -> Boolean, (Int, Int, Double) -> Unit) -> List<ArchitectureTrial>)? = null,
    private val evaluateAsync: AsyncArchitectureEvaluator? = null
) : AutoCloseable {
    data class Finished(val architecture: NetworkArchitecture, val trial: ArchitectureTrial)
    val activity = TrialActivity()
    private val workers = AtomicInteger()
    private val maximumWorkers = AtomicInteger()
    private val ids = AtomicInteger()
    private val pool = Executors.newFixedThreadPool(parallelism) { runnable ->
        Thread(runnable, "jneuro-search-${ids.incrementAndGet()}").apply { isDaemon = true }
    }
    private val executor = Executor { action -> pool.execute {
        maximumWorkers.accumulateAndGet(workers.incrementAndGet(), ::maxOf)
        try { action.run() } finally { workers.decrementAndGet() }
    } }
    private val pending = ConcurrentHashMap.newKeySet<CompletableFuture<List<Finished>>>()
    private val completions = LinkedBlockingQueue<CompletableFuture<List<Finished>>>()
    val activeWorkers: Int get() = workers.get()
    val peakWorkers: Int get() = maximumWorkers.get()

    fun submit(index: Int, architecture: NetworkArchitecture, seeds: List<Long>, cancelled: () -> Boolean) {
        val completion = CompletableFuture<List<Finished>>()
        pending += completion
        completion.whenComplete { _, _ -> pending.remove(completion); completions.add(completion) }
        try {
            if (evaluateAsync != null) {
                check(seeds.size == 1) { "Asynchronous models are admitted independently." }
                val finishActivity = activity.begin(index, architecture, seeds)
                try {
                    evaluateAsync.invoke(architecture, seeds.single(), cancelled,
                        { epoch, best -> activity.update(index, architecture, seeds.single(), epoch, best) }, executor)
                        .whenComplete { trial, failure ->
                            finishActivity()
                            if (failure != null) completion.completeExceptionally(failure)
                            else if (trial.seed != seeds.single()) completion.completeExceptionally(
                                IllegalStateException("Completion does not match submitted seed."))
                            else completion.complete(listOf(Finished(architecture, trial)))
                        }
                } catch (failure: Throwable) { finishActivity(); throw failure }
            } else executor.execute {
                try {
                    val trials = activity.trackMany(index, architecture, seeds) { progress ->
                        if (seeds.size > 1) requireNotNull(evaluateGroup)(architecture, seeds, cancelled, progress)
                        else listOf(evaluate(architecture, seeds.single(), cancelled) { epoch, best -> progress(0, epoch, best) })
                    }
                    check(trials.map { it.seed } == seeds) { "Completion does not match submitted seeds." }
                    completion.complete(trials.map { Finished(architecture, it) })
                } catch (failure: Throwable) { completion.completeExceptionally(failure) }
            }
        } catch (failure: Throwable) { completion.completeExceptionally(failure) }
    }

    fun poll(timeoutMillis: Long = 0): List<Finished>? =
        (if (timeoutMillis == 0L) completions.poll() else completions.poll(timeoutMillis, TimeUnit.MILLISECONDS))?.get()

    override fun close() {
        // Keep the scoring executor available until every GPU completion has released its model.
        var interrupted = Thread.interrupted()
        try {
            while (pending.isNotEmpty()) {
                val future = pending.firstOrNull() ?: break
                try { future.get() }
                catch (_: InterruptedException) { interrupted = true }
                catch (_: java.util.concurrent.ExecutionException) { /* The coordinator reports the original failure. */ }
                catch (_: java.util.concurrent.CancellationException) { /* Cancellation also completes model cleanup. */ }
            }
        } finally {
            if (interrupted) Thread.currentThread().interrupt()
            stopArchitectureWorkers(pool)
        }
    }
}
