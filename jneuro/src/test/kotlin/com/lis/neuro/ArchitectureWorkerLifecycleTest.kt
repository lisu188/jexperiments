package com.lis.neuro

import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.Timeout

@Timeout(10)
class ArchitectureWorkerLifecycleTest {
    @Test fun interruptedCoordinatorWaitsForWorkerCleanupAndRestoresItsInterrupt() {
        val pool = Executors.newSingleThreadExecutor()
        val started = CountDownLatch(1)
        val stopping = CountDownLatch(1)
        val release = CountDownLatch(1)
        pool.submit {
            started.countDown()
            try { CountDownLatch(1).await() } catch (_: InterruptedException) {
                stopping.countDown()
                release.await()
            }
        }
        assertTrue(started.await(2, TimeUnit.SECONDS))
        val interrupted = AtomicBoolean()
        val failure = AtomicReference<Throwable>()
        val coordinator = Thread {
            try {
                Thread.currentThread().interrupt()
                stopArchitectureWorkers(pool)
                interrupted.set(Thread.currentThread().isInterrupted)
            } catch (problem: Throwable) { failure.set(problem) }
        }
        try {
            coordinator.start()
            assertTrue(stopping.await(2, TimeUnit.SECONDS))
            assertTrue(coordinator.isAlive)
            assertFalse(pool.isTerminated)
            coordinator.interrupt()
        } finally {
            release.countDown()
            coordinator.join(3000)
            pool.shutdownNow()
        }
        assertFalse(coordinator.isAlive)
        assertNull(failure.get())
        assertTrue(interrupted.get())
        assertTrue(pool.isTerminated)
        stopArchitectureWorkers(pool)
        assertThrows(IllegalArgumentException::class.java) { stopArchitectureWorkers(pool, 0) }
    }

    @Test fun stuckWorkerFailsWithinTheBudgetWithoutPretendingResourcesWereReleased() {
        val pool = Executors.newSingleThreadExecutor()
        val started = CountDownLatch(1)
        val release = CountDownLatch(1)
        pool.submit {
            started.countDown()
            while (release.count > 0) try { release.await() } catch (_: InterruptedException) { }
        }
        assertTrue(started.await(2, TimeUnit.SECONDS))
        try {
            val failure = assertThrows(IllegalStateException::class.java) {
                stopArchitectureWorkers(pool, TimeUnit.MILLISECONDS.toNanos(1))
            }
            assertTrue(failure.message!!.contains("active device leases remain protected"))
            assertFalse(pool.isTerminated)
        } finally {
            release.countDown()
            pool.shutdownNow()
            assertTrue(pool.awaitTermination(2, TimeUnit.SECONDS))
        }
    }
}
