package com.lis.neuro

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

class SmallTrainingSessionTest {
    private fun model() = NeuroTest.prepared(intArrayOf(2, 8, 8, 8, 1), Neuro.Kernel.SCALAR)
    private fun session(model: Neuro, batch: Int = 7, clock: () -> Long = { 0L }) =
        SmallTrainingSession(model, batch, { SmallCpuTraining(it, model.hyperParameters(), Neuro.TrainingPrecision.FP64) }, clock)

    private fun same(expected: Neuro, actual: Neuro) {
        val comparison = TensorFlowBenchmarkHarness.validate(expected.exportTrainingState(), actual.exportTrainingState(),
            expected.trainingError(), actual.trainingError(), Neuro.TrainingPrecision.FP64)
        assertTrue(comparison.maximumScaledError <= 1.0, comparison.toString())
        assertEquals(expected.statistics().epochsTrained, actual.statistics().epochsTrained)
        assertEquals(expected.statistics().samplesSeen, actual.statistics().samplesSeen)
    }

    @Test fun boundedChunkPublishesCompleteStateAndReopeningContinuesTheSameSequence() {
        val expected = model(); val actual = model()
        expected.trainMiniBatch(64, 7, 1, Neuro.BatchBackend.CPU)
        session(actual).use {
            val result = it.trainChunk(TrainingChunkRequest(100))
            assertEquals(64, result.committedEpochs)
            assertEquals(TrainingTermination.BUDGET, result.termination)
            assertEquals(actual.trainingError(), result.rmse)
        }
        same(expected, actual)
        expected.trainMiniBatch(5, 7, 1, Neuro.BatchBackend.CPU)
        session(actual).use { assertEquals(5, it.trainChunk(TrainingChunkRequest(5)).committedEpochs) }
        same(expected, actual)
    }

    @Test fun failedChunkRetainsParametersMomentumCountersAndReplayableShuffle() {
        for (nonfinite in listOf(false, true)) {
            val expected = model(); val actual = model()
            val failing = SmallTrainingSession(actual, 7, { state ->
                val cpu = SmallCpuTraining(state, actual.hyperParameters(), Neuro.TrainingPrecision.FP64)
                object : SmallTrainingKernel {
                    override val info = cpu.info
                    override fun train(orders: Array<IntArray>, batchSize: Int, online: Boolean): NeuroTrainingState {
                        val result = cpu.train(orders, batchSize, online)
                        if (!nonfinite) error("Injected launch/download failure")
                        result.biasVelocity.last()[0] = Double.NaN
                        return result
                    }
                }
            }, { 0L })
            // Include a longer reservation than the attempted first launch; rollback must work for CPU continuation too.
            actual.withTraining(failing) { actual.reserveTrainingOrders(64) }
            assertThrows(IllegalStateException::class.java) { failing.trainChunk(TrainingChunkRequest(64)) }
            assertThrows(IllegalStateException::class.java) { failing.trainEpoch() }
            same(expected, actual)
            failing.close(); failing.close()
            expected.trainMiniBatch(9, 7, 1, Neuro.BatchBackend.CPU)
            actual.trainMiniBatch(9, 7, 1, Neuro.BatchBackend.CPU)
            same(expected, actual)
        }
    }

    @Test fun cancellationBudgetConvergenceAndZeroWorkAreExplicit() {
        val actual = model()
        var ticks = 0L
        session(actual, clock = { ticks.also { ticks += 10_000_000 } }).use {
            val cancelled = it.trainChunk(TrainingChunkRequest(10, cancelled = { true }))
            assertEquals(TrainingTermination.CANCELLED, cancelled.termination)
            assertEquals(0, cancelled.committedEpochs)
            assertEquals(0, it.trainChunk(TrainingChunkRequest(0)).committedEpochs)
            assertEquals(TrainingTermination.CONVERGED, it.trainChunk(TrainingChunkRequest(10, targetError = 1.0)).termination)
            val budget = it.trainChunk(TrainingChunkRequest(10, maxNanos = 1))
            assertEquals(1, budget.committedEpochs)
            assertEquals(TrainingTermination.BUDGET, budget.termination)
        }
        session(model()).use { current ->
            var polls = 0
            val result = current.trainChunk(TrainingChunkRequest(5, targetError = 0.0, checkEvery = 2,
                cancelled = { ++polls == 2 }))
            assertEquals(1, result.committedEpochs)
            assertEquals(TrainingTermination.CANCELLED, result.termination)
        }
    }

    @Test fun preservesSingleEpochPublicationAllEntrypointsAndOwnership() {
        val actual = model(); val expected = model()
        val small = session(actual, 1)
        assertThrows(IllegalStateException::class.java) { actual.trainEpoch() }
        expected.trainEpoch(); small.trainEpoch(); same(expected, actual)
        expected.train(3); small.train(3); same(expected, actual)
        val reference = expected.trainUntil(0.0, 5, 2)
        val result = small.trainUntil(0.0, 5, 2)
        assertEquals(reference.epochs, result.epochs)
        same(expected, actual)
        expected.trainMiniBatch(2, 7, 1, Neuro.BatchBackend.CPU); small.trainMiniBatch(2, 7); same(expected, actual)
        small.train(0); small.trainMiniBatch(0, 1)
        assertTrue(small.trainUntil(1.0, 0).converged)
        assertThrows(IllegalArgumentException::class.java) { small.train(-1) }
        assertThrows(IllegalArgumentException::class.java) { small.trainMiniBatch(1, 0) }
        assertThrows(IllegalArgumentException::class.java) { small.trainMiniBatch(1, 1, 0) }
        assertThrows(IllegalArgumentException::class.java) { small.trainUntil(Double.NaN, 1) }
        small.close()
        assertThrows(IllegalStateException::class.java) { small.trainEpoch() }
        assertThrows(IllegalArgumentException::class.java) { TrainingChunkRequest(-1) }
        assertThrows(IllegalArgumentException::class.java) { TrainingChunkRequest(1, targetError = Double.NaN) }
        assertThrows(IllegalArgumentException::class.java) { TrainingChunkRequest(1, checkEvery = 0) }
        assertThrows(IllegalArgumentException::class.java) { TrainingChunkRequest(1, maxNanos = 0) }
    }

    @Test fun trainingDataMutationInvalidatesConvergenceBeforeReopeningEitherEntryPoint() {
        for (precision in Neuro.TrainingPrecision.entries) for (chunk in listOf(false, true)) {
            val model = Neuro(intArrayOf(2, 4, 1), Neuro.HyperParameters(0.1, 0.2, 1.0, 42))
            val input = doubleArrayOf(0.3, 0.7)
            val fittedTarget = model.predict(input)[0]
            model.addTrainingSample(input, doubleArrayOf(fittedTarget))
            model.newTrainingSession(precision = precision, engine = TrainingEngine.SMALL).use { it.trainEpoch() }
            val oldError = model.statistics().lastTrainingError
            model.addTrainingSample(input, doubleArrayOf(fittedTarget + 1.0))
            assertTrue(model.statistics().lastTrainingError.isNaN())
            val currentError = model.trainingError()
            assertTrue(currentError > oldError)
            val target = (oldError + currentError) / 2.0
            model.newTrainingSession(precision = precision, engine = TrainingEngine.SMALL).use { session ->
                assertEquals(currentError, session.currentRmse)
                val epochs = if (chunk) session.trainChunk(TrainingChunkRequest(1, target, maxNanos = Long.MAX_VALUE)).committedEpochs
                    else session.trainUntil(target, 1).epochs
                assertEquals(1, epochs, "The old dataset metric must not report immediate convergence")
            }
            assertEquals(2L, model.statistics().epochsTrained)
            assertEquals(3L, model.statistics().samplesSeen)
        }
    }

    @Test fun engineSelectionIsExplicitAndAutoKeepsSmallCpu() {
        for (backend in listOf(TrainingBackend.CPU, TrainingBackend.AUTO)) for (precision in Neuro.TrainingPrecision.entries) {
            model().newTrainingSession(backend, precision, 7, TrainingEngine.SMALL).use {
                assertEquals(TrainingBackend.CPU, it.info.backend)
                assertEquals(TrainingEngine.SMALL, it.info.engine)
                assertEquals(precision.name, it.info.precision)
                assertTrue(it.trainEpoch().isFinite())
            }
        }
        assertThrows(IllegalArgumentException::class.java) {
            Neuro(intArrayOf(2, 3, 1)).newTrainingSession(engine = TrainingEngine.SMALL)
        }
        val actual = model()
        assertThrows(IllegalStateException::class.java) { SmallTrainingSession(actual, 7, { error("construction") }) }
        actual.newTrainingSession().close()
        val bad = actual.exportTrainingState().copy(topology = intArrayOf(2, 1))
        assertThrows(IllegalArgumentException::class.java) { actual.validateTrainingState(bad) }
    }
}
