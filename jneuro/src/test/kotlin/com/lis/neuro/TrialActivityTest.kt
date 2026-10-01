package com.lis.neuro

import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

class TrialActivityTest {
    private val architecture = NetworkArchitecture(listOf(4))

    @Test fun snapshotHandlesLastEntryRemovedBeforeIteration() {
        var iterated = false
        val retiring = object : AbstractCollection<Map.Entry<Int, ArchitectureRunningTrial>>() {
            override val size: Int get() = 1
            override fun iterator(): Iterator<Map.Entry<Int, ArchitectureRunningTrial>> {
                iterated = true
                return emptyList<Map.Entry<Int, ArchitectureRunningTrial>>().iterator()
            }
        }
        // Concurrent views may report one entry and then expose an empty iterator after a worker finishes.
        assertEquals(1, retiring.size)
        assertTrue(snapshotTrialActivity(retiring).isEmpty())
        assertTrue(iterated)
    }

    @Test fun snapshotsRetainSubmissionOrderAndRemainDetachedFromLaterProgress() {
        val activity = TrialActivity()
        val finishLater = activity.begin(20, architecture, listOf(42))
        val finishEarlier = activity.begin(10, architecture, listOf(1))
        activity.update(20, architecture, 42, 7, 0.4)
        val snapshot = activity.snapshot()
        assertEquals(listOf(1L, 42L), snapshot.map { it.seed })
        assertEquals(listOf(0, 7), snapshot.map { it.epoch })
        activity.update(20, architecture, 42, 10, 0.3)
        finishEarlier(); finishLater()
        assertTrue(activity.snapshot().isEmpty())
        assertEquals(2, activity.peak)
        assertEquals(listOf(0, 7), snapshot.map { it.epoch })
        assertEquals(0.4, snapshot.last().bestRmse)
    }

    @Test fun concurrentProgressAndCompletionLeaveNoActivityAndNeverInvalidateSnapshots() {
        val activity = TrialActivity()
        val started = CountDownLatch(1)
        val observed = CountDownLatch(1)
        Executors.newFixedThreadPool(2).use { workers ->
            val writer = workers.submit {
                val finish = activity.begin(0, architecture, listOf(42))
                try {
                    started.countDown()
                    assertTrue(observed.await(5, TimeUnit.SECONDS))
                    activity.update(0, architecture, 42, 1, 0.5)
                } finally { finish() }
                repeat(2_000) { epoch ->
                    val completed = activity.begin(0, architecture, listOf(42))
                    try { activity.update(0, architecture, 42, epoch, 0.5) }
                    finally { completed() }
                }
            }
            val reader = workers.submit {
                assertTrue(started.await(5, TimeUnit.SECONDS))
                val initial = activity.snapshot()
                try {
                    assertEquals(1, initial.size)
                    assertEquals(0, initial.single().epoch)
                } finally { observed.countDown() }
                repeat(2_000) {
                    val snapshot = activity.snapshot()
                    assertTrue(snapshot.size <= 1)
                    snapshot.forEach { trial ->
                        assertEquals(42L, trial.seed)
                        assertEquals(architecture, trial.architecture)
                        assertTrue(trial.epoch >= 0)
                    }
                }
                assertEquals(0, initial.single().epoch)
            }
            try {
                writer.get(5, TimeUnit.SECONDS)
                reader.get(5, TimeUnit.SECONDS)
            } finally { observed.countDown() }
        }
        assertTrue(activity.snapshot().isEmpty())
        assertEquals(1, activity.peak)
    }
}
