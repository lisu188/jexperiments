package com.lis.neuro

import java.util.concurrent.CompletableFuture
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.CountDownLatch
import java.util.concurrent.ExecutionException
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

class ArchitectureTrialDispatcherTest {
    private val architecture = NetworkArchitecture(listOf(4))
    private fun trial(seed: Long) = ArchitectureTrial(seed, ArchitectureTrialState.COMPLETED, 1, 1,
        0.5, 0.5, 0.5, 4, 1, emptyList(), null)

    @Test fun asynchronousModelsLeaveScoringWorkersAvailableAndCloseDrainsBeforeShutdown() {
        val ready = CountDownLatch(4)
        val pending = CopyOnWriteArrayList<CompletableFuture<ArchitectureTrial>>()
        val execute = AtomicReference<java.util.concurrent.Executor>()
        val dispatcher = ArchitectureTrialDispatcher(2, { _, _, _, _ -> error("Synchronous evaluator must not run") },
            evaluateAsync = { _, _, _, _, executor ->
                execute.set(executor)
                CompletableFuture<ArchitectureTrial>().also { future ->
                    pending += future
                    executor.execute { ready.countDown() }
                }
            })
        repeat(4) { dispatcher.submit(it, architecture, listOf(it.toLong()), { false }) }
        assertTrue(ready.await(5, TimeUnit.SECONDS))
        assertEquals(4, dispatcher.activity.snapshot().size)
        val freeWorker = CountDownLatch(1)
        execute.get().execute { freeWorker.countDown() }
        assertTrue(freeWorker.await(5, TimeUnit.SECONDS), "Pending GPU futures must not occupy scoring threads")
        assertTrue(dispatcher.peakWorkers in 1..2)
        val closing = CountDownLatch(1)
        val closed = CountDownLatch(1)
        val interrupted = AtomicBoolean()
        val problem = AtomicReference<Throwable>()
        val closer = Thread {
            closing.countDown()
            try { dispatcher.close(); interrupted.set(Thread.currentThread().isInterrupted) }
            catch (failure: Throwable) { problem.set(failure) }
            finally { closed.countDown() }
        }
        closer.start()
        try {
            assertTrue(closing.await(5, TimeUnit.SECONDS))
            closer.interrupt()
            assertFalse(closed.await(30, TimeUnit.MILLISECONDS), "Closing cannot abandon admitted models")
            pending.forEachIndexed { index, future -> future.complete(trial(index.toLong())) }
            assertTrue(closed.await(5, TimeUnit.SECONDS))
            assertNull(problem.get()); assertTrue(interrupted.get())
            assertTrue(dispatcher.activity.snapshot().isEmpty())
            assertEquals(4, generateSequence { dispatcher.poll() }.sumOf { it.size })
            assertNull(dispatcher.poll(1))
        } finally {
            pending.forEachIndexed { index, future -> future.complete(trial(index.toLong())) }
            closer.join(5000); dispatcher.close()
        }
    }

    @Test fun failedCallbacksAndInconsistentSynchronousGroupsAreReportedWithoutLeakingActivity() {
        val failure = IllegalArgumentException("async fixture")
        ArchitectureTrialDispatcher(1, { _, _, _, _ -> error("unused") }, evaluateAsync = { _, _, _, _, _ ->
            CompletableFuture.failedFuture(failure)
        }).use { dispatcher ->
            dispatcher.submit(0, architecture, listOf(1), { false })
            val raised = assertThrows(ExecutionException::class.java) { dispatcher.poll(5000) }
            assertSame(failure, raised.cause)
            assertTrue(dispatcher.activity.snapshot().isEmpty())
        }
        ArchitectureTrialDispatcher(1, { _, seed, _, progress -> progress(1, 0.5); trial(seed) },
            evaluateGroup = { _, _, _, progress -> progress(0, 1, 0.5); listOf(trial(999)) }).use { dispatcher ->
            dispatcher.submit(0, architecture, listOf(1), { false })
            assertEquals(1, dispatcher.poll(5000)!!.single().trial.epochs)
            dispatcher.submit(1, architecture, listOf(1, 42), { false })
            assertThrows(ExecutionException::class.java) { dispatcher.poll(5000) }
            assertTrue(dispatcher.activity.snapshot().isEmpty())
        }
    }
}
