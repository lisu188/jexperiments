package com.lis.neuro

import java.util.concurrent.CompletableFuture
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.Timeout

@Timeout(30)
class SearchCudaServiceTest {
    @Test fun sixtyFourHeterogeneousModelsShareOneDriverAndBatchWithoutCpuWorkerLimit() {
        val fake = SearchCudaTestDriver()
        val entered = CountDownLatch(1)
        val release = CountDownLatch(1)
        var first = true
        fake.beforeLaunch = { if (first) { first = false; entered.countDown(); check(release.await(10, TimeUnit.SECONDS)) } }
        val shapes = listOf(intArrayOf(2, 4, 1), intArrayOf(2, 8, 16, 1), intArrayOf(2, 16, 4, 8, 16, 1))
        val models = List(64) { model(shapes[it % shapes.size], it.toLong()) }
        SearchCudaService(Neuro.TrainingPrecision.FP64, 3, { fake }).use { service ->
            val sessions = models.map(service::openSession)
            try {
                val uploads = fake.uploads
                val futures = ArrayList<CompletableFuture<SearchAdvanceResult>>()
                futures += sessions.first().advanceForSearchAsync(request(3))
                assertTrue(entered.await(10, TimeUnit.SECONDS))
                for (session in sessions.drop(1)) futures += session.advanceForSearchAsync(request(3))
                assertEquals(63, service.queuedRequests)
                release.countDown()
                futures.forEach { result ->
                    assertEquals(SearchAdvanceResult(3, null, TrainingTermination.COMPLETED), result.get(10, TimeUnit.SECONDS))
                }
                assertEquals(64, service.peakResidentModels)
                assertEquals(64, service.residentModels)
                assertTrue(fake.launches.any { it.size == 64 })
                assertTrue(fake.launches.any { batch -> batch.map { it[3] }.distinct().size > 1 })
                assertTrue(fake.launches.any { batch -> batch.map { it[4] }.distinct().size > 1 })
                assertEquals(fake.downloads.toLong(), service.batchesLaunched)
                assertEquals(uploads + fake.launches.size * 2, fake.uploads, "Dataset is not re-uploaded between chunks")
                models.forEachIndexed { index, actual ->
                    val expected = model(shapes[index % shapes.size], index.toLong())
                    expected.newTrainingSession(TrainingBackend.CPU, Neuro.TrainingPrecision.FP64, 3, TrainingEngine.SMALL).use { it.train(3) }
                    assertSmallState(expected.exportTrainingState(), actual.exportTrainingState(), 1e-12)
                    assertEquals(3L, actual.statistics().epochsTrained)
                    assertTrue(actual.statistics().lastTrainingError.isNaN(), "GPU thread must not score models")
                }
                assertThrows(IllegalStateException::class.java) { service.openSession(model()) }
                assertThrows(IllegalStateException::class.java) { service.close() }
                assertEquals(setOf("jneuro-search-cuda"), fake.threads)
            } finally { release.countDown(); sessions.forEach { it.close() } }
            assertEquals(0, service.residentModels)
            service.openSession(model()).use { assertTrue(it.info.kernelVersion.startsWith("small-search-v3/")) }
        }
        assertEquals(1, fake.closes)
        assertTrue(fake.memory.isEmpty())
    }

    @Test fun bothPrecisionsModesAndSynchronousReplayRetainFullMomentumState() {
        for (precision in Neuro.TrainingPrecision.entries) for (mode in Neuro.SigmoidMode.entries) {
            val fake = SearchCudaTestDriver()
            val actual = model(mode = mode)
            val expected = model(mode = mode)
            SearchCudaService(precision, 3, { fake }).use { service ->
                service.openSession(actual).use { session ->
                    assertEquals(precision.name, session.info.precision)
                    session.train(2)
                    session.trainMiniBatch(2, 3, 1)
                    val result = session.trainChunk(request(2))
                    assertEquals(2, result.committedEpochs)
                    assertTrue(result.rmse.isFinite())
                    assertEquals(6L, actual.statistics().epochsTrained)
                    expected.newTrainingSession(TrainingBackend.CPU, precision, 3, TrainingEngine.SMALL).use { it.train(6) }
                    assertSmallState(expected.exportTrainingState(), actual.exportTrainingState(), 1e-12)
                    assertEquals(0, session.trainUntil(1.0, 9, 3).epochs)
                    assertEquals(3, session.trainUntil(0.0, 3, 2).epochs)
                    assertThrows(IllegalArgumentException::class.java) { session.trainMiniBatch(1, 7, 1) }
                    assertThrows(IllegalArgumentException::class.java) { session.trainMiniBatch(1, 1, 1) }
                    assertThrows(IllegalArgumentException::class.java) { session.trainUntil(Double.NaN, 1, 1) }
                }
                assertTrue(fake.kernelNames.all { it.endsWith(precision.name.lowercase()) })
                actual.trainEpoch()
            }
        }
    }

    @Test fun numericalAndCorruptLanesFailIndependentlyAndCanResumeWithTheirOriginalShuffle() {
        for (corrupt in listOf(false, true)) {
            val fake = SearchCudaTestDriver().also { if (corrupt) it.corruptSlots += 0 else it.nonfiniteSlots += 0 }
            val bad = model()
            val good = model(seed = 3)
            val before = bad.exportTrainingState()
            SearchCudaService(Neuro.TrainingPrecision.FP64, 3, { fake }, 2).use { service ->
                val first = service.openSession(bad)
                val second = service.openSession(good)
                try {
                    assertThrows(Exception::class.java) { first.advanceForSearch(request(2)) }
                    assertSmallState(before, bad.exportTrainingState(), 0.0)
                    assertEquals(0L, bad.statistics().epochsTrained)
                    assertThrows(IllegalStateException::class.java) { first.advanceForSearchAsync(request(1)) }
                    assertEquals(2, second.advanceForSearch(request(2)).committedEpochs)
                } finally { first.close(); second.close() }
                fake.corruptSlots.clear(); fake.nonfiniteSlots.clear()
                service.openSession(bad).use { assertEquals(2, it.advanceForSearch(request(2)).committedEpochs) }
                val reference = model()
                reference.newTrainingSession(TrainingBackend.CPU, Neuro.TrainingPrecision.FP64, 3, TrainingEngine.SMALL).use { it.train(2) }
                assertSmallState(reference.exportTrainingState(), bad.exportTrainingState(), 1e-12)
            }
        }
    }

    @Test fun transportFailureNeverPublishesPartialDownloadAndPoisonsSharedDriver() {
        for (operation in listOf("launch", "synchronize", "download", "upload")) {
            val fake = SearchCudaTestDriver()
            val models = List(2) { model(seed = it.toLong()) }
            val before = models.map { it.exportTrainingState() }
            SearchCudaService(Neuro.TrainingPrecision.FP64, 3, { fake }, 2).use { service ->
                val sessions = models.map(service::openSession)
                fake.failure = operation
                try {
                    assertThrows(Exception::class.java) { sessions[0].advanceForSearch(request(2)) }
                    assertThrows(IllegalStateException::class.java) { sessions[1].advanceForSearchAsync(request(1)) }
                    assertThrows(IllegalStateException::class.java) { service.openSession(model()) }
                    models.forEachIndexed { index, model -> assertSmallState(before[index], model.exportTrainingState(), 0.0) }
                } finally { fake.failure = ""; sessions.forEach { it.close() } }
            }
            assertTrue(fake.memory.isEmpty())
        }
    }

    @Test fun cancellationBudgetAndRequestCapPreserveCommittedPrefixesWithoutScoring() {
        val clock = AtomicLong()
        val fake = SearchCudaTestDriver().apply { beforeLaunch = { clock.addAndGet(10) } }
        val actual = model()
        SearchCudaService(Neuro.TrainingPrecision.FP64, 1, { fake }, 1, clock::get).use { service ->
            service.openSession(actual).use { session ->
                assertEquals(TrainingTermination.CANCELLED,
                    session.advanceForSearch(TrainingChunkRequest(8, cancelled = { true })).termination)
                assertEquals(0L, actual.statistics().epochsTrained)
                assertEquals(SearchAdvanceResult(0, null, TrainingTermination.COMPLETED), session.advanceForSearch(request(0)))
                val budget = session.advanceForSearch(TrainingChunkRequest(8, maxNanos = 1))
                assertEquals(SearchAdvanceResult(1, null, TrainingTermination.BUDGET), budget)
                val capped = session.advanceForSearch(request(65))
                assertEquals(SearchAdvanceResult(64, null, TrainingTermination.BUDGET), capped)
                assertEquals(65L, actual.statistics().epochsTrained)
                assertTrue(actual.statistics().lastTrainingError.isNaN())
                assertThrows(IllegalArgumentException::class.java) { session.advanceForSearch(TrainingChunkRequest(1, targetError = 0.0)) }
                assertThrows(IllegalStateException::class.java) { actual.addTrainingSample(doubleArrayOf(0.0, 0.0), doubleArrayOf(0.0)) }
            }
        }
    }

    @Test fun closeWaitsForNativeWorkEvenWhenCallerCancelsItsFutureAndRestoresInterrupt() {
        val entered = CountDownLatch(1)
        val release = CountDownLatch(1)
        val fake = SearchCudaTestDriver().apply { beforeLaunch = { entered.countDown(); check(release.await(10, TimeUnit.SECONDS)) } }
        val service = SearchCudaService(Neuro.TrainingPrecision.FP64, 3, { fake }, 1)
        val actual = model()
        val session = service.openSession(actual)
        val result = session.advanceForSearchAsync(request(8))
        assertTrue(entered.await(10, TimeUnit.SECONDS))
        assertThrows(IllegalStateException::class.java) { session.advanceForSearchAsync(request(1)) }
        result.cancel(true)
        val interrupted = AtomicBoolean()
        val closer = Thread { Thread.currentThread().interrupt(); session.close(); interrupted.set(Thread.currentThread().isInterrupted) }
        closer.start()
        try {
            assertThrows(IllegalStateException::class.java) { service.close() }
            assertEquals(0, fake.frees)
        } finally { release.countDown(); closer.join(10_000); session.close(); service.close() }
        assertFalse(closer.isAlive)
        assertTrue(interrupted.get())
        assertEquals(1L, actual.statistics().epochsTrained)
        assertTrue(fake.memory.isEmpty())
        assertThrows(IllegalStateException::class.java) { session.trainEpoch() }
        assertThrows(IllegalStateException::class.java) { service.openSession(model()) }
        service.close()
        actual.trainEpoch()
    }

    @Test fun invalidModelsAndFailedInitializationReleaseOwnershipAndCleanupEveryBuffer() {
        assertThrows(IllegalArgumentException::class.java) { SearchCudaService(Neuro.TrainingPrecision.FP64, 0) }
        assertThrows(IllegalArgumentException::class.java) { SearchCudaService(Neuro.TrainingPrecision.FP64, 1, maximumModels = 65) }
        for (allocation in 1..6) {
            val fake = SearchCudaTestDriver().apply { failAllocation = allocation }
            val actual = model()
            SearchCudaService(Neuro.TrainingPrecision.FP64, 3, { fake }, 1).use { service ->
                assertThrows(Exception::class.java) { service.openSession(actual) }
                actual.trainEpoch()
            }
            assertTrue(fake.memory.isEmpty())
            assertEquals(1, fake.closes)
        }
        val fake = SearchCudaTestDriver()
        SearchCudaService(Neuro.TrainingPrecision.FP64, 3, { fake }, 2).use { service ->
            service.openSession(model()).use {
                val different = model(samples = 3)
                assertThrows(IllegalArgumentException::class.java) { service.openSession(different) }
                different.trainEpoch()
                val other = model().also { network -> network.addTrainingSample(doubleArrayOf(0.1, 0.2), doubleArrayOf(1.0)) }
                assertThrows(IllegalArgumentException::class.java) { service.openSession(other) }
            }
            val closed = service.openSession(model())
            closed.close(); closed.close()
        }
        val cleanup = SearchCudaTestDriver()
        val service = SearchCudaService(Neuro.TrainingPrecision.FP64, 3, { cleanup }, 1)
        service.openSession(model()).close()
        cleanup.failure = "free"
        val problem = assertThrows(IllegalStateException::class.java) { service.close() }
        assertTrue(problem.suppressed.isNotEmpty())
        assertEquals(1, cleanup.closes)
        assertTrue(cleanup.memory.isEmpty())
    }

    private fun request(epochs: Int) = TrainingChunkRequest(epochs, maxNanos = Long.MAX_VALUE)
    private fun model(shape: IntArray = intArrayOf(2, 4, 1), seed: Long = 2, samples: Int = 7,
                      mode: Neuro.SigmoidMode = Neuro.SigmoidMode.EXACT): Neuro =
        preparedSmall(shape, Neuro.HyperParameters(0.07, 0.21, 1.1, seed, Neuro.Kernel.SCALAR, mode), samples)
}
