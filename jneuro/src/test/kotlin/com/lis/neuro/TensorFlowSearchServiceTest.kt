package com.lis.neuro

import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.Timeout

/** TensorFlow CPU exercises the queue without requiring a GPU on ordinary CI. */
@Timeout(60)
class TensorFlowSearchServiceTest {
    @Test fun heterogeneousModelsShareBoundedAdmissionAndPublishIndependentCheckpoints() {
        val fixture = SearchTensorFlowKernels()
        val shapes = listOf(intArrayOf(2, 4, 1), intArrayOf(2, 8, 4, 1), intArrayOf(2, 4, 8, 4, 1))
        val models = shapes.mapIndexed { index, shape -> model(shape, index.toLong()) }
        TensorFlowSearchService(Neuro.TrainingPrecision.FP64, 3, fixture::open, 3).use { service ->
            val sessions = models.map(service::openSession)
            try {
                assertEquals(3, service.residentModels)
                assertThrows(IllegalStateException::class.java) { service.openSession(model()) }
                assertThrows(IllegalStateException::class.java) { service.close() }
                val futures = sessions.mapIndexed { index, session -> session.advanceForSearchAsync(request(index + 2)) }
                futures.forEachIndexed { index, future ->
                    assertEquals(index + 2, future.get(30, TimeUnit.SECONDS).committedEpochs)
                    val expected = model(shapes[index], index.toLong())
                    expected.newTrainingSession(batchSize = 3, engine = TrainingEngine.SMALL).use { it.train(index + 2) }
                    assertSmallState(expected.exportTrainingState(), models[index].exportTrainingState(), 0.0)
                    assertTrue(models[index].statistics().lastTrainingError.isNaN())
                }
                assertEquals(3, service.peakResidentModels)
                assertTrue(service.batchesLaunched > 0)
                assertTrue(fixture.threads.all { it == "jneuro-search-tensorflow" })
            } finally { sessions.forEach { it.close() } }
            assertEquals(0, service.residentModels)
            service.openSession(model()).close()
        }
        assertEquals(fixture.opened, fixture.closed)
    }

    @Test fun synchronousReplayAndPublicEntrypointsRetainMomentumForBothPrecisions() {
        for (precision in Neuro.TrainingPrecision.entries) {
            val actual = model(); val expected = model()
            val fixture = SearchTensorFlowKernels(precision)
            TensorFlowSearchService(precision, 3, fixture::open).use { service ->
                service.openSession(actual).use { session ->
                    assertEquals(precision.name, session.info.precision)
                    session.train(2); session.trainMiniBatch(2, 3, 1)
                    assertEquals(2, session.trainChunk(request(2)).committedEpochs)
                    assertTrue(session.currentRmse.isFinite())
                    expected.newTrainingSession(precision = precision, batchSize = 3, engine = TrainingEngine.SMALL).use { it.train(6) }
                    assertSmallState(expected.exportTrainingState(), actual.exportTrainingState(), 0.0)
                    assertEquals(0, session.trainUntil(1.0, 9, 3).epochs)
                    assertEquals(3, session.trainUntil(0.0, 3, 2).epochs)
                    assertThrows(IllegalArgumentException::class.java) { session.trainMiniBatch(1, 7, 1) }
                    assertThrows(IllegalArgumentException::class.java) { session.trainMiniBatch(1, 1, 1) }
                    assertThrows(IllegalArgumentException::class.java) { session.trainUntil(Double.NaN, 1, 1) }
                    assertThrows(IllegalArgumentException::class.java) { session.train(-1) }
                }
            }
            actual.trainEpoch()
        }
    }

    @Test fun failedLaneKeepsHostCheckpointAndReopensWithOriginalShuffleWithoutPoisoningSiblings() {
        for (corrupt in listOf(false, true)) {
            val fixture = SearchTensorFlowKernels().apply { if (corrupt) corruptSlots += 0 else failedSlots += 0 }
            val bad = model(); val good = model(seed = 3)
            val before = bad.exportTrainingState()
            TensorFlowSearchService(Neuro.TrainingPrecision.FP64, 3, fixture::open, 2).use { service ->
                val first = service.openSession(bad); val second = service.openSession(good)
                try {
                    assertThrows(Exception::class.java) { first.advanceForSearch(request(2)) }
                    assertSmallState(before, bad.exportTrainingState(), 0.0)
                    assertEquals(0L, bad.statistics().epochsTrained)
                    assertThrows(IllegalStateException::class.java) { first.advanceForSearchAsync(request(1)) }
                    assertEquals(2, second.advanceForSearch(request(2)).committedEpochs)
                } finally { first.close(); second.close() }
                fixture.failedSlots.clear(); fixture.corruptSlots.clear()
                service.openSession(bad).use { assertEquals(2, it.advanceForSearch(request(2)).committedEpochs) }
                val expected = model()
                expected.newTrainingSession(batchSize = 3, engine = TrainingEngine.SMALL).use { it.train(2) }
                assertSmallState(expected.exportTrainingState(), bad.exportTrainingState(), 0.0)
            }
        }
    }

    @Test fun cancellationBudgetAndRequestCapPreserveCommittedPrefixesWithoutScoring() {
        val clock = AtomicLong()
        val fixture = SearchTensorFlowKernels().apply { beforeTrain = { clock.addAndGet(10) } }
        val actual = model()
        TensorFlowSearchService(Neuro.TrainingPrecision.FP64, 1, fixture::open, 1, clock::get).use { service ->
            service.openSession(actual).use { session ->
                assertEquals(TrainingTermination.CANCELLED, session.advanceForSearch(TrainingChunkRequest(8, cancelled = { true })).termination)
                assertEquals(SearchAdvanceResult(0, null, TrainingTermination.COMPLETED), session.advanceForSearch(request(0)))
                assertEquals(SearchAdvanceResult(1, null, TrainingTermination.BUDGET), session.advanceForSearch(TrainingChunkRequest(8, maxNanos = 1)))
                assertEquals(SearchAdvanceResult(64, null, TrainingTermination.BUDGET), session.advanceForSearch(request(65)))
                assertEquals(65L, actual.statistics().epochsTrained)
                assertTrue(actual.statistics().lastTrainingError.isNaN())
                assertThrows(IllegalArgumentException::class.java) { session.advanceForSearch(TrainingChunkRequest(1, targetError = 0.0)) }
                assertThrows(IllegalStateException::class.java) { actual.addTrainingSample(doubleArrayOf(0.0, 0.0), doubleArrayOf(0.0)) }
            }
        }
    }

    @Test fun closeDrainsNativeWorkEvenWhenFutureIsCancelledAndRestoresInterrupt() {
        val entered = CountDownLatch(1); val release = CountDownLatch(1)
        val fixture = SearchTensorFlowKernels().apply { beforeTrain = { entered.countDown(); check(release.await(30, TimeUnit.SECONDS)) } }
        val service = TensorFlowSearchService(Neuro.TrainingPrecision.FP64, 3, fixture::open, 1)
        val actual = model(); val session = service.openSession(actual)
        val result = session.advanceForSearchAsync(request(8))
        assertTrue(entered.await(30, TimeUnit.SECONDS))
        assertThrows(IllegalStateException::class.java) { session.advanceForSearchAsync(request(1)) }
        result.cancel(true)
        val interrupted = AtomicBoolean()
        val closer = Thread { Thread.currentThread().interrupt(); session.close(); interrupted.set(Thread.currentThread().isInterrupted) }
        closer.start()
        try { assertThrows(IllegalStateException::class.java) { service.close() }; assertEquals(0, fixture.closed) }
        finally { release.countDown(); closer.join(30_000); session.close(); service.close() }
        assertFalse(closer.isAlive); assertTrue(interrupted.get())
        assertEquals(1L, actual.statistics().epochsTrained)
        assertThrows(IllegalStateException::class.java) { session.trainEpoch() }
        assertThrows(IllegalStateException::class.java) { service.openSession(model()) }
        service.close(); actual.trainEpoch()
    }

    @Test fun invalidAdmissionFactoryFailureAndCleanupFailureReleaseOwnership() {
        assertThrows(IllegalArgumentException::class.java) { TensorFlowSearchService(Neuro.TrainingPrecision.FP64, 0) }
        assertThrows(IllegalArgumentException::class.java) { TensorFlowSearchService(Neuro.TrainingPrecision.FP64, 1, maximumModels = 65) }
        val actual = model()
        TensorFlowSearchService(Neuro.TrainingPrecision.FP64, 3, { _, _ -> error("factory") }, 1).use { service ->
            assertThrows(IllegalStateException::class.java) { service.openSession(actual) }
            actual.trainEpoch()
        }
        val fixture = SearchTensorFlowKernels()
        TensorFlowSearchService(Neuro.TrainingPrecision.FP64, 3, fixture::open, 2).use { service ->
            service.openSession(model()).use {
                val different = model(samples = 3)
                assertThrows(IllegalArgumentException::class.java) { service.openSession(different) }
                different.trainEpoch()
            }
            val session = service.openSession(actual)
            fixture.failClose = true
            assertThrows(IllegalStateException::class.java) { session.close() }
            session.close(); actual.trainEpoch()
        }
    }

    @Test fun cancellationCallbackFailureFailsOnlyItsSession() {
        val fixture = SearchTensorFlowKernels()
        TensorFlowSearchService(Neuro.TrainingPrecision.FP64, 3, fixture::open, 1).use { service ->
            service.openSession(model()).use { session ->
                assertThrows(IllegalStateException::class.java) { session.advanceForSearch(TrainingChunkRequest(2, cancelled = { error("callback") })) }
                assertThrows(IllegalStateException::class.java) { session.advanceForSearch(request(1)) }
            }
        }
    }

    private fun request(epochs: Int) = TrainingChunkRequest(epochs, maxNanos = Long.MAX_VALUE)
    private fun model(shape: IntArray = intArrayOf(2, 4, 1), seed: Long = 2, samples: Int = 7): Neuro =
        preparedSmall(shape, Neuro.HyperParameters(0.07, 0.21, 1.1, seed, Neuro.Kernel.SCALAR), samples)
}
