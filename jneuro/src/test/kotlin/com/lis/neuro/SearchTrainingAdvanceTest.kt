package com.lis.neuro

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test
import com.lis.neuro.SmallCpuTrainingTest.Companion.assertState

class SearchTrainingAdvanceTest {
    private fun reference(model: Neuro, batch: Int = 1, clock: () -> Long = { 0L }) =
        SmallTrainingSession(model, batch, { TensorFlowMath.trainingKernel(it, model.hyperParameters(), Neuro.TrainingPrecision.FP64, TrainingBackend.CPU) }, clock)

    private fun small(model: Neuro, precision: Neuro.TrainingPrecision, batch: Int = 1, clock: () -> Long = { 0L }) =
        SmallTrainingSession(model, batch, { SmallCpuTraining(it, model.hyperParameters(), precision) }, clock)

    @Test fun referenceSpiralAdvancementPreservesGeneralShapesBothArithmeticPathsAndScoringBoundaries() {
        val samples = NeuroLearningSets.create(NeuroLearningSets.Kind.SPIRAL, 42)
        for (kernel in listOf(Neuro.Kernel.SCALAR, Neuro.Kernel.AUTO)) for (batch in listOf(1, 7)) {
            fun model() = Neuro(intArrayOf(2, 3, 8, 5, 1), Neuro.HyperParameters(0.1, 0.2, 1.0, 42, kernel)).also {
                NeuroLearningSets.addTo(it, samples)
            }
            val expected = model(); val actual = model()
            reference(expected, batch).use { session -> repeat(70) { session.trainEpoch() } }
            reference(actual, batch).use { session ->
                val first = advanceTrainingForSearch(session, TrainingChunkRequest(70, maxNanos = Long.MAX_VALUE))
                assertEquals(64, first.committedEpochs)
                assertEquals(TrainingTermination.BUDGET, first.termination)
                assertNull(first.rmse)
                assertTrue(actual.statistics().lastTrainingError.isNaN(), "Partial progress cannot expose an old score")
                val second = advanceTrainingForSearch(session, TrainingChunkRequest(6, maxNanos = Long.MAX_VALUE))
                assertEquals(6, second.committedEpochs)
                assertEquals(TrainingTermination.COMPLETED, second.termination)
                assertEquals(expected.trainingError(), second.rmse)
                assertEquals(actual.trainingError(), actual.statistics().lastTrainingError)
            }
            assertState(expected.exportTrainingState(), actual.exportTrainingState())
            assertEquals(expected.statistics(), actual.statistics())
            expected.trainEpoch(); actual.trainEpoch()
            assertState(expected.exportTrainingState(), actual.exportTrainingState())
        }
    }

    @Test fun smallAdvancementKeepsFloatStateMomentumAndShuffleAcrossPartialBoundariesAndReopen() {
        for (precision in Neuro.TrainingPrecision.entries) for (batch in listOf(1, 7)) {
            fun model() = NeuroTest.prepared(intArrayOf(2, 8, 4, 1), Neuro.Kernel.VECTOR)
            val expected = model(); val actual = model()
            small(expected, precision, batch).use { session -> repeat(70) { session.trainEpoch() } }
            small(actual, precision, batch).use { session ->
                val first = advanceTrainingForSearch(session, TrainingChunkRequest(70, maxNanos = Long.MAX_VALUE))
                assertEquals(64, first.committedEpochs); assertNull(first.rmse)
                assertTrue(actual.statistics().lastTrainingError.isNaN())
                val second = advanceTrainingForSearch(session, TrainingChunkRequest(6, maxNanos = Long.MAX_VALUE))
                assertEquals(expected.trainingError(), second.rmse)
                assertEquals(TrainingTermination.COMPLETED, second.termination)
            }
            assertState(expected.exportTrainingState(), actual.exportTrainingState())
            small(expected, precision, batch).use { it.trainEpoch() }
            small(actual, precision, batch).use { it.trainEpoch() }
            assertState(expected.exportTrainingState(), actual.exportTrainingState())
            assertEquals(expected.statistics(), actual.statistics())
        }
    }

    @Test fun cancellationBudgetZeroWorkAndClosedOwnershipNeverInventAScore() {
        for (isSmall in listOf(false, true)) {
            val model = NeuroTest.prepared(intArrayOf(2, 4, 1))
            var ticks = 0L
            val clock = { ticks.also { ticks += 10_000_000L } }
            val session: NeuroTrainingSession = if (isSmall) small(model, Neuro.TrainingPrecision.FP64, clock = clock) else reference(model, clock = clock)
            session.use {
                val cancelled = advanceTrainingForSearch(session, TrainingChunkRequest(10, cancelled = { true }))
                assertEquals(SearchAdvanceResult(0, null, TrainingTermination.CANCELLED), cancelled)
                assertEquals(SearchAdvanceResult(0, null, TrainingTermination.COMPLETED),
                    advanceTrainingForSearch(session, TrainingChunkRequest(0)))
                assertThrows(IllegalArgumentException::class.java) {
                    advanceTrainingForSearch(session, TrainingChunkRequest(10, targetError = 1.0))
                }
                val budget = advanceTrainingForSearch(session, TrainingChunkRequest(10, maxNanos = 1))
                assertEquals(SearchAdvanceResult(1, null, TrainingTermination.BUDGET), budget)
                assertEquals(1L, model.statistics().epochsTrained)
                assertThrows(IllegalStateException::class.java) { model.trainEpoch() }
                var polls = 0
                val stopped = advanceTrainingForSearch(session, TrainingChunkRequest(5,
                    cancelled = { ++polls == 2 }))
                assertEquals(TrainingTermination.CANCELLED, stopped.termination)
                assertEquals(1, stopped.committedEpochs)
                assertNull(stopped.rmse)
            }
            assertThrows(IllegalStateException::class.java) { advanceTrainingForSearch(session, TrainingChunkRequest(1)) }
            assertTrue(model.trainEpoch().isFinite())
        }
    }

    @Test fun searchFailuresPoisonComputeOwnershipAndSmallFailureKeepsAtomicHostCheckpoint() {
        val model = NeuroTest.prepared(intArrayOf(2, 4, 1))
        val initial = model.exportTrainingState()
        SmallTrainingSession(model, 1, { state ->
            object : SmallTrainingKernel {
                override val info = TrainingDeviceInfo(TrainingBackend.CPU, "broken", "broken", engine = TrainingEngine.SMALL)
                override fun train(orders: Array<IntArray>, batchSize: Int, online: Boolean): NeuroTrainingState =
                    state.copy(weights = state.weights.map { it.copyOf().also { row -> row[0] = Double.NaN } }.toTypedArray())
            }
        }, { 0L }).use { session ->
            assertThrows(IllegalStateException::class.java) { advanceTrainingForSearch(session, TrainingChunkRequest(5)) }
            assertThrows(IllegalStateException::class.java) { session.trainEpoch() }
            assertState(initial, model.exportTrainingState())
            assertEquals(0L, model.statistics().epochsTrained)
        }
        val broken = Neuro(intArrayOf(2, 4, 1), Neuro.HyperParameters(Double.MAX_VALUE, 0.1, 1.0, 42))
            .addTrainingSample(doubleArrayOf(0.1, 0.9), doubleArrayOf(Double.MAX_VALUE))
        reference(broken).use { session ->
            assertThrows(IllegalStateException::class.java) { advanceTrainingForSearch(session, TrainingChunkRequest(2)) }
            assertThrows(IllegalStateException::class.java) { session.trainEpoch() }
        }
    }

    @Test fun legacyOrInjectedSessionsRetainTheirPublicChunkFallback() {
        val model = NeuroTest.prepared(intArrayOf(2, 3, 1))
        reference(model).use { delegate ->
            val legacy = object : NeuroTrainingSession by delegate {}
            val result = advanceTrainingForSearch(legacy, TrainingChunkRequest(1, maxNanos = Long.MAX_VALUE))
            assertEquals(1, result.committedEpochs)
            assertEquals(model.trainingError(), result.rmse)
            val partial = searchAdvanceFallback(legacy, TrainingChunkRequest(65, maxNanos = Long.MAX_VALUE))
            assertEquals(64, partial.committedEpochs); assertNull(partial.rmse)
            assertThrows(IllegalArgumentException::class.java) {
                searchAdvanceFallback(legacy, TrainingChunkRequest(1, targetError = 1.0))
            }
        }
    }

    @Test fun anOverflowingBoundaryMetricRejectsTheTrialAndPoisonsBothSessionKinds() {
        for (isSmall in listOf(false, true)) {
            val model = Neuro(intArrayOf(2, 4, 1), Neuro.HyperParameters(Double.MIN_VALUE, 0.1, 1.0, 42))
                .addTrainingSample(doubleArrayOf(0.1, 0.9), doubleArrayOf(Double.MAX_VALUE))
            val session: NeuroTrainingSession = if (isSmall) small(model, Neuro.TrainingPrecision.FP64) else reference(model)
            session.use {
                assertThrows(IllegalStateException::class.java) {
                    advanceTrainingForSearch(session, TrainingChunkRequest(1, maxNanos = Long.MAX_VALUE))
                }
                assertEquals(1L, model.statistics().epochsTrained)
                val state = model.exportTrainingState()
                assertTrue(state.weights.all { row -> row.all { it.isFinite() } })
                assertThrows(IllegalStateException::class.java) { session.trainEpoch() }
            }
            assertNull(model.trainingSessionLogId)
        }
    }

    @Test fun privateCpuChunksPollEveryEpochAndRejectNonfiniteStateBeforePublication() {
        for (precision in Neuro.TrainingPrecision.entries) {
            val model = NeuroTest.prepared(intArrayOf(2, 4, 1), Neuro.Kernel.SCALAR)
            val state = model.exportTrainingState()
            val orders = SmallCpuTrainingTest.orders(state.samples, 12)
            SmallCpuTraining(state, model.hyperParameters(), precision).use { kernel ->
                var polls = 0
                val result = kernel.advanceForSearch(orders, 3, false) { completed ->
                    assertEquals(polls++, completed)
                    completed < 4
                }
                assertEquals(4, result.epochs); assertEquals(5, polls)
                SmallCpuTraining(state, model.hyperParameters(), precision).use { expected ->
                    assertState(expected.train(orders.take(4).toTypedArray(), 3, false), checkNotNull(result.state))
                }
                assertEquals(SmallCpuAdvance(0, null), kernel.advanceForSearch(orders, 3, false) { false })
                assertThrows(IllegalArgumentException::class.java) { kernel.advanceForSearch(orders, 0, false) { true } }
            }
            val maximum = if (precision == Neuro.TrainingPrecision.FP32) Float.MAX_VALUE.toDouble() else Double.MAX_VALUE
            val invalid = Neuro(intArrayOf(2, 4, 1), Neuro.HyperParameters(maximum, 0.1, 1.0, 42))
                .addTrainingSample(doubleArrayOf(0.1, 0.9), doubleArrayOf(maximum))
            val before = invalid.exportTrainingState()
            small(invalid, precision).use { session ->
                assertThrows(IllegalStateException::class.java) {
                    advanceTrainingForSearch(session, TrainingChunkRequest(12, maxNanos = Long.MAX_VALUE))
                }
                assertEquals(0L, invalid.statistics().epochsTrained)
                assertState(before, invalid.exportTrainingState())
                assertThrows(IllegalStateException::class.java) { session.trainEpoch() }
            }
        }
    }
}
