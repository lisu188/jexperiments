package com.lis.neuro

import java.util.concurrent.ExecutorService
import java.util.concurrent.TimeUnit

/** Keeps native leases alive until worker cleanup finishes, even if the coordinator was interrupted. */
internal fun stopArchitectureWorkers(pool: ExecutorService, timeoutNanos: Long = TimeUnit.SECONDS.toNanos(5)) {
    require(timeoutNanos > 0) { "Worker shutdown timeout must be positive." }
    pool.shutdownNow()
    var interrupted = Thread.interrupted()
    val started = System.nanoTime()
    try {
        while (!pool.isTerminated) {
            val remaining = timeoutNanos - (System.nanoTime() - started)
            check(remaining > 0) { "Architecture training workers did not stop; their active device leases remain protected." }
            try { pool.awaitTermination(remaining, TimeUnit.NANOSECONDS) }
            catch (_: InterruptedException) { interrupted = true }
        }
    } finally {
        if (interrupted) Thread.currentThread().interrupt()
    }
}
