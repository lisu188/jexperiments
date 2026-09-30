package com.lis.neuro

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertInstanceOf
import org.junit.jupiter.api.Assertions.assertThrows
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import java.util.concurrent.CompletableFuture
import java.util.concurrent.CountDownLatch
import java.util.concurrent.ForkJoinPool
import java.util.concurrent.TimeUnit
import java.util.concurrent.TimeoutException
import java.util.concurrent.atomic.AtomicBoolean

class NeuroCpuBatchTrainerTest {
    @Test fun batchSizeOneTracksOnlineTraining() {
        val online = NeuroTest.prepared(intArrayOf(7, 9, 5, 2), Neuro.Kernel.SCALAR)
        val matrix = NeuroTest.prepared(intArrayOf(7, 9, 5, 2), Neuro.Kernel.SCALAR)
        online.train(5)
        matrix.trainMiniBatch(5, 1, 1, Neuro.BatchBackend.CPU)
        val input = NeuroTest.input(7)
        assertArrayEquals(online.predict(input), matrix.predict(input), 1e-12)
        assertEquals(online.statistics().epochsTrained, matrix.statistics().epochsTrained)
        assertEquals(online.statistics().samplesSeen, matrix.statistics().samplesSeen)
    }

    @Test fun sequentialAndParallelMatrixTrainingAgree() {
        val sequential = NeuroTest.prepared(intArrayOf(33, 65, 17, 3))
        val parallel = NeuroTest.prepared(intArrayOf(33, 65, 17, 3))
        sequential.trainMiniBatch(4, 7, 1, Neuro.BatchBackend.CPU)
        parallel.trainMiniBatch(4, 7, 4, Neuro.BatchBackend.CPU)
        val input = NeuroTest.input(33)
        assertArrayEquals(sequential.predict(input), parallel.predict(input), 1e-12)
        assertEquals(sequential.trainingError(), parallel.trainingError(), 1e-12)
    }

    @Test fun matrixTrainingHandlesPartialAndOversizedBatches() {
        val network = NeuroTest.prepared(intArrayOf(5, 8, 4, 2))
        val before = network.trainingError()
        network.trainMiniBatch(6, 13, 3, Neuro.BatchBackend.CPU)
        assertTrue(network.trainingError() < before)
        assertEquals(6L, network.statistics().epochsTrained)
        assertEquals(192L, network.statistics().samplesSeen)
        val oversized = NeuroTest.prepared(intArrayOf(5, 8, 4, 2))
        oversized.trainMiniBatch(2, 128, 2, Neuro.BatchBackend.CPU)
        assertEquals(64L, oversized.statistics().samplesSeen)
    }

    @Test fun changingBatchCapacityReusesOnlyCurrentSamplesAcrossCalls() {
        val reused = NeuroTest.prepared(intArrayOf(2, 8, 8, 1), Neuro.Kernel.SCALAR)
        val independent = NeuroTest.prepared(intArrayOf(2, 8, 8, 1), Neuro.Kernel.SCALAR)
        // The independent model streams one mini-batch epoch; it has no matrix workspace cache.
        for (batch in listOf(3, 19, 2, 128, 7)) {
            reused.trainMiniBatch(2, batch, 1, Neuro.BatchBackend.CPU)
            independent.trainMiniBatch(2, batch, 1)
            SmallCpuTrainingTest.assertState(independent.exportTrainingState(), reused.exportTrainingState(), 1e-13)
        }
    }

    @Test fun interruptedCallerDrainsEveryWorkerBeforeReturningAndRestoresInterrupt() {
        assertWorkersDrain(interrupt = true)
    }

    @Test fun failedWorkerDoesNotReleaseOwnershipWhileAnotherWorkerStillRuns() {
        assertWorkersDrain(interrupt = false)
    }

    private fun assertWorkersDrain(interrupt: Boolean) {
        val pool = ForkJoinPool(2)
        val started = CountDownLatch(2)
        val release = CountDownLatch(1)
        val finished = CountDownLatch(2)
        val outcome = CompletableFuture<Throwable>()
        val restoredInterrupt = AtomicBoolean()
        val caller = Thread {
            try {
                NeuroCpuBatchTrainer.parallelFor(256, 2, pool) { from, _ ->
                    started.countDown()
                    try {
                        if (!interrupt && from == 0) error("first worker failed")
                        check(release.await(5, TimeUnit.SECONDS)) { "Worker release timed out" }
                        if (!interrupt) error("second worker failed")
                    } finally { finished.countDown() }
                }
                outcome.complete(AssertionError("Training unexpectedly succeeded"))
            } catch (failure: Throwable) {
                restoredInterrupt.set(Thread.currentThread().isInterrupted)
                outcome.complete(failure)
            }
        }
        try {
            caller.start()
            assertTrue(started.await(5, TimeUnit.SECONDS))
            if (interrupt) caller.interrupt()
            assertThrows(TimeoutException::class.java) { outcome.get(100, TimeUnit.MILLISECONDS) }
            release.countDown()
            val failure = assertInstanceOf(IllegalStateException::class.java, outcome.get(5, TimeUnit.SECONDS))
            assertEquals(0L, finished.count)
            if (interrupt) {
                assertTrue(restoredInterrupt.get())
                assertTrue(failure.message!!.contains("interrupted"))
            } else {
                assertFalse(restoredInterrupt.get())
                assertTrue(failure.cause!!.message!!.contains("first worker failed"))
                assertEquals(1, failure.suppressed.size)
                assertTrue(failure.suppressed.single().cause!!.message!!.contains("second worker failed"))
            }
        } finally {
            release.countDown()
            caller.join(5_000)
            pool.shutdown()
            assertFalse(caller.isAlive)
        }
    }
}
