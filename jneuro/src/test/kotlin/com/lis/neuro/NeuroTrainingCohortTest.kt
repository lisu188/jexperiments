package com.lis.neuro

import com.lis.neuro.SmallCpuTrainingTest.Companion.assertState
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.Timeout
import java.util.concurrent.Callable
import java.util.concurrent.CountDownLatch
import java.util.concurrent.ExecutorService
import java.util.concurrent.Future
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference

class NeuroTrainingCohortTest {
    @Test fun parallelCohortsMatchIndependentSessionsAndReopenWithExactShuffleContinuation() {
        for (precision in Neuro.TrainingPrecision.entries) for (batch in listOf(1, 3)) {
            val expected = models(11)
            val sequential = models(11)
            val parallel = models(11)
            for (epochs in listOf(3, 2)) {
                for (model in expected) model.newTrainingSession(TrainingBackend.CPU, precision, batch, TrainingEngine.SMALL).use {
                    if (batch == 1) it.train(epochs) else it.trainMiniBatch(epochs, batch, 1)
                }
                for ((actual, workers) in listOf(sequential to 1, parallel to 4)) {
                    NeuroTrainingCohort.open(actual, TrainingBackend.AUTO, precision, batch, workers).use { cohort ->
                        assertEquals(TrainingBackend.CPU, cohort.info.backend)
                        assertEquals(TrainingEngine.SMALL, cohort.info.engine)
                        val result = cohort.trainChunk(TrainingChunkRequest(epochs, maxNanos = Long.MAX_VALUE))
                        result.forEach { assertEquals(epochs, it.committedEpochs); assertEquals(TrainingTermination.COMPLETED, it.termination) }
                    }
                    for (index in actual.indices) {
                        assertState(expected[index].exportTrainingState(), actual[index].exportTrainingState())
                        assertEquals(expected[index].statistics(), actual[index].statistics())
                    }
                }
            }
        }
    }

    @Test fun inactiveCancelledConvergedAndBudgetLimitedModelsPublishOnlyCompletedEpochs() {
        val models = models(9)
        NeuroTrainingCohort.open(models).use { cohort ->
            val initial = models.map { it.exportTrainingState() }
            assertThrows(IllegalArgumentException::class.java) { cohort.trainChunk(TrainingChunkRequest(1), booleanArrayOf(true)) }
            val zero = cohort.trainChunk(TrainingChunkRequest(0))
            zero.forEach { assertEquals(TrainingTermination.COMPLETED, it.termination); assertEquals(0, it.committedEpochs) }
            val idle = cohort.trainChunk(TrainingChunkRequest(10), BooleanArray(models.size))
            idle.forEach { assertEquals(TrainingTermination.CANCELLED, it.termination); assertEquals(0, it.committedEpochs) }
            val converged = cohort.trainChunk(TrainingChunkRequest(10, targetError = 2.0))
            converged.forEach { assertEquals(TrainingTermination.CONVERGED, it.termination); assertEquals(0, it.committedEpochs) }
            val cancelled = cohort.trainChunk(TrainingChunkRequest(10, cancelled = { true }))
            cancelled.forEach { assertEquals(TrainingTermination.CANCELLED, it.termination); assertEquals(0, it.committedEpochs) }
            for (index in models.indices) assertState(initial[index], models[index].exportTrainingState())
            val active = BooleanArray(models.size) { it == 0 || it == 8 }
            var checks = 0
            val partial = cohort.trainChunk(TrainingChunkRequest(10, cancelled = { ++checks > 1 }, maxNanos = Long.MAX_VALUE), active)
            for (index in models.indices) {
                assertEquals(if (active[index]) 1 else 0, partial[index].committedEpochs)
                assertEquals(TrainingTermination.CANCELLED, partial[index].termination)
                if (!active[index]) assertState(initial[index], models[index].exportTrainingState())
            }
            val bounded = cohort.trainChunk(TrainingChunkRequest(65, maxNanos = 1L), active)
            for (index in models.indices) if (active[index]) {
                assertEquals(1, bounded[index].committedEpochs)
                assertEquals(TrainingTermination.BUDGET, bounded[index].termination)
            }
            val capped = cohort.trainChunk(TrainingChunkRequest(65, maxNanos = Long.MAX_VALUE), active)
            for (index in models.indices) if (active[index]) {
                assertEquals(64, capped[index].committedEpochs)
                assertEquals(TrainingTermination.BUDGET, capped[index].termination)
            }
        }
    }

    @Test fun convergenceAfterTrainingStopsAtRequestedScoringBoundary() {
        fun converging() = Neuro(intArrayOf(2, 4, 1), Neuro.HyperParameters(0.5, 0.2, 1.0, 42)).also { model ->
            repeat(7) { model.addTrainingSample(doubleArrayOf(0.5, 0.7), doubleArrayOf(0.0)) }
        }
        val reference = converging()
        val initial = reference.trainingError()
        val error = reference.newTrainingSession(TrainingBackend.CPU, engine = TrainingEngine.SMALL).use { it.trainEpoch() }
        assertTrue(error < initial)
        val actual = converging()
        NeuroTrainingCohort.open(listOf(actual)).use { cohort ->
            val result = cohort.trainChunk(TrainingChunkRequest(10, (initial + error) / 2.0, maxNanos = Long.MAX_VALUE)).single()
            assertEquals(1, result.committedEpochs)
            assertEquals(TrainingTermination.CONVERGED, result.termination)
        }
        assertState(reference.exportTrainingState(), actual.exportTrainingState())
    }

    @Test fun trainingDataMutationInvalidatesEachCohortLaneBeforeTargetChecking() {
        for (precision in Neuro.TrainingPrecision.entries) {
            val models = List(2) { Neuro(intArrayOf(2, 4, 1), Neuro.HyperParameters(0.1, 0.2, 1.0, 42)) }
            val input = doubleArrayOf(0.3, 0.7)
            val fittedTarget = models.first().predict(input)[0]
            models.forEach { it.addTrainingSample(input, doubleArrayOf(fittedTarget)) }
            NeuroTrainingCohort.open(models, precision = precision).use {
                it.trainChunk(TrainingChunkRequest(1, maxNanos = Long.MAX_VALUE))
            }
            val oldError = models.first().statistics().lastTrainingError
            models.forEach { it.addTrainingSample(input, doubleArrayOf(fittedTarget + 1.0)) }
            assertTrue(models.all { it.statistics().lastTrainingError.isNaN() })
            val currentError = models.first().trainingError()
            assertTrue(currentError > oldError)
            NeuroTrainingCohort.open(models, precision = precision).use { cohort ->
                val result = cohort.trainChunk(TrainingChunkRequest(1, (oldError + currentError) / 2.0, maxNanos = Long.MAX_VALUE))
                assertTrue(result.all { it.committedEpochs == 1 }, "No lane may converge using the old dataset metric")
            }
            assertTrue(models.all { it.statistics().epochsTrained == 2L && it.statistics().samplesSeen == 3L })
        }
    }

    @Test fun callerListMutationCannotChangeCohortMembershipOrStrandOwnedModels() {
        val original = models(2)
        val supplied = original.toMutableList()
        val replacement = models(1).single()
        NeuroTrainingCohort.open(supplied).use { cohort ->
            supplied.clear()
            supplied += replacement
            val result = cohort.trainChunk(TrainingChunkRequest(1, maxNanos = Long.MAX_VALUE))
            assertEquals(2, result.size)
            assertEquals(original.map { it.trainingError() }, result.map { it.rmse })
            assertTrue(original.all { it.statistics().epochsTrained == 1L })
            assertEquals(0L, replacement.statistics().epochsTrained)
            replacement.newTrainingSession().close()
            supplied.clear()
        }
        original.forEach { it.newTrainingSession().close() }
        replacement.newTrainingSession().close()
    }

    @Test fun invalidConstructionAndPartialOwnershipAcquisitionReleaseEveryAcquiredModel() {
        val models = models(6)
        assertThrows(IllegalArgumentException::class.java) { NeuroTrainingCohort.open(emptyList()) }
        assertThrows(IllegalArgumentException::class.java) { NeuroTrainingCohort.open(listOf(models[0], models[0])) }
        assertThrows(IllegalArgumentException::class.java) { NeuroTrainingCohort.open(models, batchSize = 0) }
        assertThrows(IllegalArgumentException::class.java) { NeuroTrainingCohort.open(models, parallelism = 0) }
        assertThrows(IllegalArgumentException::class.java) { NeuroTrainingCohort.open(listOf(NeuroTest.xor())) }
        assertThrows(IllegalArgumentException::class.java) { NeuroTrainingCohort.open(listOf(models[0], SmallCpuTrainingTest.model(intArrayOf(2, 4, 1)))) }
        val otherParameters = Neuro(intArrayOf(2, 4, 8, 1), models[0].hyperParameters().copy(learningRate = 0.9))
        assertThrows(IllegalArgumentException::class.java) { NeuroTrainingCohort.open(listOf(models[0], otherParameters)) }
        models[1].newTrainingSession().use {
            assertThrows(IllegalStateException::class.java) { NeuroTrainingCohort.open(models) }
            models[0].newTrainingSession().close()
            assertThrows(IllegalStateException::class.java) { models[1].newTrainingSession() }
        }
        val differentData = models(1).single().also { it.addTrainingSample(doubleArrayOf(1.0, 0.0), doubleArrayOf(1.0)) }
        assertThrows(IllegalArgumentException::class.java) { NeuroTrainingCohort.open(listOf(models[0], differentData)) }
        differentData.newTrainingSession().close()
        models[5].backendLayers()[0].weights[0] = Double.MAX_VALUE
        assertThrows(IllegalArgumentException::class.java) { NeuroTrainingCohort.open(models, precision = Neuro.TrainingPrecision.FP32) }
        models.forEach { it.newTrainingSession().close() }
        val cohort = NeuroTrainingCohort.open(models.take(2))
        assertThrows(IllegalStateException::class.java) { models[0].trainEpoch() }
        assertThrows(IllegalStateException::class.java) { models[1].newTrainingSession() }
        cohort.close(); cohort.close()
        assertThrows(IllegalStateException::class.java) { cohort.trainChunk(TrainingChunkRequest(1)) }
        models.take(2).forEach { it.newTrainingSession().close() }
    }

    @Test fun failedWorkerDoesNotPublishAnySiblingStateAndClosingReleasesOwnership() {
        for (workers in listOf(1, 4)) {
            val models = List(5) { index ->
                Neuro(intArrayOf(2, 4, 1), Neuro.HyperParameters(Double.MAX_VALUE, 0.2, 8.0, index.toLong())).also { model ->
                    model.addTrainingSample(doubleArrayOf(0.5, 0.5), doubleArrayOf(10.0))
                    model.backendLayers().forEach { it.weights.fill(0.0); it.biases.fill(0.0) }
                }
            }
            val initial = models.map { it.exportTrainingState() }
            val cohort = NeuroTrainingCohort.open(models, parallelism = workers)
            assertThrows(IllegalStateException::class.java) { cohort.trainChunk(TrainingChunkRequest(3, maxNanos = Long.MAX_VALUE)) }
            assertThrows(IllegalStateException::class.java) { cohort.trainChunk(TrainingChunkRequest(1)) }
            for (index in models.indices) {
                assertState(initial[index], models[index].exportTrainingState())
                assertEquals(0L, models[index].statistics().epochsTrained)
            }
            cohort.close()
            models.forEach { it.newTrainingSession().close() }
        }
    }

    @Test @Timeout(10)
    fun interruptedCoordinatorJoinsEveryWorkerBeforeDiscardingResultsAndReleasingModels() {
        val models = models(2) // Independent TensorFlow kernels, each running on its own worker.
        val initial = models.map { it.exportTrainingState() }
        val cohort = NeuroTrainingCohort.open(models, parallelism = 2)
        val poolField = NeuroTrainingCohort::class.java.getDeclaredField("pool").apply { isAccessible = true }
        val executor = poolField.get(cohort) as ExecutorService
        val started = CountDownLatch(2)
        val gates = List(2) { CountDownLatch(1) }
        val rejoinedFirst = CountDownLatch(1)
        val joinedSecond = CountDownLatch(1)
        val interruptedGet = AtomicBoolean()
        val submitted = AtomicInteger()
        // Gate actual cohort tasks and observe Future.get re-entry without a production test hook.
        poolField.set(cohort, object : ExecutorService by executor {
            override fun <T> submit(task: Callable<T>): Future<T> {
                val index = submitted.getAndIncrement()
                val future = executor.submit(Callable {
                    started.countDown()
                    gates[index].await()
                    task.call()
                })
                return object : Future<T> by future {
                    override fun get(): T {
                        if (index == 0 && interruptedGet.get()) rejoinedFirst.countDown()
                        if (index == 1) joinedSecond.countDown()
                        try { return future.get() }
                        catch (interrupted: InterruptedException) {
                            interruptedGet.set(true)
                            throw interrupted
                        }
                    }
                }
            }
        })
        val failure = AtomicReference<Throwable>()
        val interruptRestored = AtomicBoolean()
        val finished = CountDownLatch(1)
        val coordinator = Thread {
            try {
                cohort.use { it.trainChunk(TrainingChunkRequest(1, maxNanos = Long.MAX_VALUE)) }
            } catch (problem: Throwable) { failure.set(problem) }
            finally {
                interruptRestored.set(Thread.currentThread().isInterrupted)
                finished.countDown()
            }
        }
        try {
            coordinator.start()
            assertTrue(started.await(2, TimeUnit.SECONDS))
            coordinator.interrupt()
            assertTrue(rejoinedFirst.await(2, TimeUnit.SECONDS), "Interrupted join must wait for the first worker again")
            gates[0].countDown()
            assertTrue(joinedSecond.await(2, TimeUnit.SECONDS), "Every worker must finish before ownership is released")
            assertEquals(1L, finished.count)
            models.forEach { assertThrows(IllegalStateException::class.java) { it.newTrainingSession() } }
            gates[1].countDown()
            assertTrue(finished.await(2, TimeUnit.SECONDS))
        } finally {
            gates.forEach { it.countDown() }
            coordinator.join(3000)
            executor.shutdownNow()
        }
        assertFalse(coordinator.isAlive)
        assertInstanceOf(InterruptedException::class.java, failure.get())
        assertTrue(interruptRestored.get())
        for (index in models.indices) {
            assertState(initial[index], models[index].exportTrainingState())
            assertEquals(0L, models[index].statistics().epochsTrained)
            assertEquals(0L, models[index].statistics().samplesSeen)
            models[index].newTrainingSession().close()
        }
    }

    @Test fun deviceServiceSupportsReopenAndRejectsClosingBeforeItsSessions() {
        assertThrows(IllegalArgumentException::class.java) { NeuroTrainingDeviceService(0) }
        val service = NeuroTrainingDeviceService(2)
        val models = models(2)
        service.openSession(models[0], engine = TrainingEngine.SMALL).use {
            assertTrue(it.trainEpoch().isFinite())
            assertThrows(IllegalStateException::class.java) { service.close() }
        }
        service.openSession(models[0]).use { assertTrue(it.trainEpoch().isFinite()) }
        service.openCohort(models).use {
            assertEquals(1, it.trainChunk(TrainingChunkRequest(1)).first().committedEpochs)
            assertThrows(IllegalStateException::class.java) { service.close() }
        }
        models.forEach { it.newTrainingSession().close() }
        service.close(); service.close()
        assertThrows(IllegalStateException::class.java) { service.openSession(models[0]) }
        assertThrows(IllegalStateException::class.java) { service.openCohort(models) }
    }

    @Test fun tensorflowWrapperReleasesEveryModelEvenIfKernelCloseFails() {
        val models = models(2)
        val fixture = SearchTensorFlowKernels()
        val cohort = NeuroTrainingCohort.openConfigured(models, TrainingBackend.CPU, Neuro.TrainingPrecision.FP64, 1, 1, fixture::open)
        val result = cohort.trainChunk(TrainingChunkRequest(2, maxNanos = Long.MAX_VALUE))
        result.forEach { assertEquals(2, it.committedEpochs) }
        fixture.failClose = true
        val failure = assertThrows(IllegalStateException::class.java) { cohort.close() }
        assertEquals(1, failure.suppressed.size)
        assertEquals(2, fixture.closed)
        models.forEach { it.newTrainingSession().close() }
        cohort.close()
    }

    private fun models(count: Int): List<Neuro> = List(count) { index ->
        Neuro(intArrayOf(2, 4, 8, 1), Neuro.HyperParameters(0.1, 0.23, 1.15, index + 19L, Neuro.Kernel.VECTOR)).also { model ->
            repeat(7) { sample -> model.addTrainingSample(doubleArrayOf(sample / 7.0, (sample * 3 % 7) / 7.0), doubleArrayOf((sample % 2).toDouble())) }
        }
    }

}
