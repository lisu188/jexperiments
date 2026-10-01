package com.lis.neuro

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

/** Scheduler fixtures implement a stateful device contract; numerical parity is tested by TensorFlowSearchCohortTest. */
class BatchedPopulationSearchTest {
    private val data = ArchitectureSearchData.fitting(NeuroLearningSets.create(NeuroLearningSets.Kind.XOR, 42))
    private fun config(policy: ArchitectureBudgetPolicy = ArchitectureBudgetPolicy.SUCCESSIVE_HALVING,
                       epochs: Int = 18, trials: Int = 12, models: Int = 6) = ArchitectureSearchConfig(
        minWidth = 1, maxWidth = 8, maxLayers = 2, maxParameters = 256,
        seeds = listOf(1, 42), requiredSuccesses = 2, maxEpochs = epochs, checkEvery = 2,
        initialEpochs = 2, reductionFactor = 3, modelsPerBatch = models, maxTrials = trials,
        execution = ArchitectureExecution.BATCHED, budgetPolicy = policy, targetRmse = 0.0)

    @Test fun allInitialArchitecturesArePublishedBeforeAnyNativeTrainingAndPruningKeepsSeedGroups() {
        val setup = config(trials = 6)
        val harness = DeviceFixture()
        val progress = ArrayList<ArchitectureSearchProgress>()
        val report = BatchedPopulationSearch(data, setup, (1..3).map { NetworkArchitecture(listOf(it)) },
            open = harness::open).search({ update ->
            if (progress.isEmpty()) {
                assertEquals(3, update.generated)
                assertEquals(3, update.running.map { it.architecture }.distinct().size)
                assertEquals(0, harness.advances)
            }
            progress += update
        }, { false })
        assertEquals(3, report.generated)
        assertEquals(1, report.candidates.count { it.valid })
        assertEquals(2, report.candidates.count { candidate -> candidate.trials.all { it.state == ArchitectureTrialState.PRUNED } })
        for (candidate in report.candidates) {
            assertEquals(setOf(1L, 42L), candidate.trials.map { it.seed }.toSet())
            assertEquals(1, candidate.trials.map { it.state }.distinct().size)
            assertEquals(if (candidate.valid) 18 else 2, candidate.trials.first().epochs)
        }
        assertEquals(6, report.modelsPerBatch)
        assertEquals(harness.advances.toLong(), report.nativeTrainingCalls)
        assertTrue(report.aggregateEpochsPerSecond > 0)
        assertTrue(progress.last().running.isEmpty())
        assertEquals(harness.opens, harness.closes)
    }

    @Test fun newlyFreedSlotsExploreBeforeLongPromotedModelsReachTheirEpochCap() {
        val harness = DeviceFixture()
        val progress = ArrayList<ArchitectureSearchProgress>()
        val shapes = (1..6).map { NetworkArchitecture(listOf(it)) }
        val report = BatchedPopulationSearch(data, config(epochs = 54), shapes, open = harness::open)
            .search(progress::add, { false })
        val refilled = progress.first { it.generated > 3 }
        assertTrue(refilled.running.any { it.architecture == shapes.first() && it.epoch == 2 })
        assertTrue(refilled.running.any { it.architecture in shapes.drop(3) && it.epoch == 0 })
        assertTrue(report.candidates.all { it.trials.size == 2 })
        assertTrue(report.candidates.first().trials.all { it.epochs == 54 })
        assertTrue(report.candidates.first().trials.all { it.batchSegments.size > 1 })
        assertEquals(harness.opens, harness.closes)
    }

    @Test fun fullBudgetPreservesScoringBoundariesAndIndependentShuffleAcrossRebinding() {
        val setup = ArchitectureSearchConfig(seeds = listOf(1, 42), requiredSuccesses = 2, maxWidth = 8,
            maxEpochs = 7, checkEvery = 3, initialEpochs = 2, maxTrials = 8, modelsPerBatch = 4,
            execution = ArchitectureExecution.BATCHED, budgetPolicy = ArchitectureBudgetPolicy.FULL)
        val shapes = listOf(NetworkArchitecture(listOf(1)), NetworkArchitecture(listOf(2, 3)),
            NetworkArchitecture(listOf(4)), NetworkArchitecture(listOf(5)))
        val harness = DeviceFixture()
        val report = BatchedPopulationSearch(data, setup, shapes, open = harness::open).search({}, { false })
        assertTrue(report.candidates.all { it.valid })
        for (candidate in report.candidates) for (trial in candidate.trials) {
            assertEquals(7, trial.epochs)
            assertEquals(28L, trial.sampleUpdates)
            assertEquals(listOf(0, 3, 6, 7), trial.history.map { it.epoch })
            assertEquals(7, trial.bestEpoch)
            assertNotNull(trial.snapshot)
            val source = data.newNetwork(candidate.architecture, setup.hyperParameters, trial.seed).exportTrainingState()
            val key = source.weights[0][0].toBits()
            val actual = harness.orders.getValue(key)
            val shuffle = TrainingShuffle(trial.seed xor -7046029254386353131L)
            val expected = shuffle.reserve(data.training.size, 7)
            assertEquals(7, actual.size)
            for (epoch in actual.indices) assertArrayEquals(expected[epoch], actual[epoch])
            assertEquals(7.0, harness.lastState.getValue(key).weightVelocity[0][0], 0.0)
        }
        assertEquals(harness.opens, harness.closes)
    }

    @Test fun repeatedSearchesHaveIdenticalLineagePruningAndScoredHistory() {
        fun run(variableSpeed: Boolean): ArchitectureSearchResult {
            val harness = DeviceFixture(variableSpeed = variableSpeed)
            return BatchedPopulationSearch(data, config(trials = 18), clock = { harness.clockNanos }, open = harness::open).search({}, { false })
        }
        val first = run(false); val second = run(true)
        assertEquals(first.lineage, second.lineage)
        for ((left, right) in first.candidates.zip(second.candidates)) {
            assertEquals(left.architecture, right.architecture)
            assertEquals(left.trials.map { it.state }, right.trials.map { it.state })
            assertEquals(left.trials.map { it.history }, right.trials.map { it.history })
        }
        assertEquals(18, first.candidates.sumOf { it.trials.size })
        assertEquals(first.lineage.size, first.lineage.map { it.architecture }.distinct().size)
        assertTrue(first.lineage.take(3).map { it.architecture }.distinct().size > 1)
    }

    @Test fun unrelatedBracketRebindingNeverScoresAnExistingLaneBetweenItsBoundaries() {
        val setup = ArchitectureSearchConfig(maxLayers = 2, maxWidth = 5, seeds = listOf(1, 42), requiredSuccesses = 2,
            maxEpochs = 18, checkEvery = 5, initialEpochs = 2, maxTrials = 10, modelsPerBatch = 6,
            execution = ArchitectureExecution.BATCHED, budgetPolicy = ArchitectureBudgetPolicy.SUCCESSIVE_HALVING, targetRmse = 0.0)
        val shapes = listOf(listOf(1), listOf(2), listOf(3), listOf(4, 4), listOf(5)).map(::NetworkArchitecture)
        val harness = DeviceFixture(variableSpeed = true)
        val report = BatchedPopulationSearch(data, setup, shapes, clock = { harness.clockNanos }, open = harness::open)
            .search({}, { false })
        val survivor = report.candidates.first().trials.first()
        assertTrue(survivor.batchSegments.any { it.endEpoch == 4 }, "Fixture must rebind between scoring boundaries")
        assertEquals(listOf(0, 2, 5, 6, 10, 15, 18), survivor.history.map { it.epoch })
        assertEquals(18, survivor.bestEpoch)
        assertEquals(harness.opens, harness.closes)
    }

    @Test fun fullSeedGroupTargetSuccessCanFinishOnlyUnderThePruningPolicy() {
        fun run(policy: ArchitectureBudgetPolicy): ArchitectureSearchResult {
            val setup = ArchitectureSearchConfig(seeds = listOf(1, 42), requiredSuccesses = 2, maxWidth = 2,
                maxEpochs = 6, checkEvery = 2, initialEpochs = 2, maxTrials = 2, modelsPerBatch = 2,
                execution = ArchitectureExecution.BATCHED, budgetPolicy = policy, targetRmse = 1.0)
            return BatchedPopulationSearch(data, setup, listOf(NetworkArchitecture(listOf(2))), open = DeviceFixture()::open).search({}, { false })
        }
        assertTrue(run(ArchitectureBudgetPolicy.SUCCESSIVE_HALVING).candidates.single().trials.all { it.epochs == 2 && it.state == ArchitectureTrialState.COMPLETED })
        assertTrue(run(ArchitectureBudgetPolicy.FULL).candidates.single().trials.all { it.epochs == 6 })
    }

    @Test fun oneFailedLaneDoesNotDiscardHealthySiblingArchitecturesOrQualifyItsOwnGroup() {
        val harness = DeviceFixture(failingWidth = 2)
        val report = BatchedPopulationSearch(data, config(trials = 6), (1..3).map { NetworkArchitecture(listOf(it)) },
            open = harness::open).search({}, { false })
        val failed = report.candidates.single { it.architecture.hidden == listOf(2) }
        assertFalse(failed.valid)
        assertTrue(failed.trials.any { it.state == ArchitectureTrialState.FAILED })
        assertTrue(report.candidates.any { it.valid })
        assertEquals(harness.opens, harness.closes)
    }

    @Test fun cancellationDeadlineAndObserverFailureCloseEveryOpenedCohort() {
        val initial = DeviceFixture()
        val cancelled = BatchedPopulationSearch(data, config(), open = initial::open).search({}, { true })
        assertEquals(ArchitectureTermination.CANCELLED, cancelled.termination)
        assertTrue(cancelled.lineage.isEmpty()); assertEquals(0, initial.opens)
        val partial = DeviceFixture()
        val stopped = BatchedPopulationSearch(data, config(epochs = 1_000_000), open = partial::open)
            .search({}, { partial.advances >= 2 })
        assertEquals(ArchitectureTermination.CANCELLED, stopped.termination)
        assertTrue(stopped.candidates.flatMap { it.trials }.all { it.state == ArchitectureTrialState.CANCELLED })
        assertEquals(partial.opens, partial.closes)
        val deadline = DeviceFixture()
        val setup = ArchitectureSearchConfig(maxEpochs = 100, checkEvery = 2, timeLimitSeconds = 1,
            execution = ArchitectureExecution.BATCHED, modelsPerBatch = 10)
        val timed = BatchedPopulationSearch(data, setup, clock = { deadline.advances * 1_000_000_000L }, open = deadline::open)
            .search({}, { false })
        assertEquals(ArchitectureTermination.TIME_LIMIT, timed.termination)
        assertEquals(deadline.opens, deadline.closes)
        val observer = DeviceFixture()
        assertThrows(IllegalStateException::class.java) {
            BatchedPopulationSearch(data, config(), open = observer::open).search({
                if (it.finishedTrials > 0) error("observer failed")
            }, { false })
        }
        assertTrue(observer.opens > 0); assertEquals(observer.opens, observer.closes)
    }

    @Test fun residentMemoryGuardRejectsOversizedPaddingBeforeConstructingModels() {
        val setup = ArchitectureSearchConfig(minLayers = 2, maxLayers = 2, maxWidth = 1024, maxParameters = 2_000_000,
            seeds = listOf(42), requiredSuccesses = 1, maxEpochs = 1, checkEvery = 1, maxTrials = 2, modelsPerBatch = 2,
            execution = ArchitectureExecution.BATCHED)
        val harness = DeviceFixture()
        val result = BatchedPopulationSearch(data, setup,
            listOf(NetworkArchitecture(listOf(1024, 1024)), NetworkArchitecture(listOf(1023, 1024))), open = harness::open)
            .search({}, { false })
        assertEquals(ArchitectureTermination.MEMORY_LIMIT, result.termination)
        assertEquals(0, harness.opens)
        assertTrue(result.candidates.none { it.valid })
    }

    @Test fun tinyFiniteDomainExhaustsWithoutDuplicateProposalsAndRemainderBudgetNeverSplitsSeeds() {
        val setup = ArchitectureSearchConfig(minWidth = 1, maxWidth = 1, maxLayers = 1,
            seeds = listOf(1, 42), requiredSuccesses = 2, maxEpochs = 1, checkEvery = 1, maxTrials = 9,
            modelsPerBatch = 6, execution = ArchitectureExecution.BATCHED)
        val report = BatchedPopulationSearch(data, setup, open = DeviceFixture()::open).search({}, { false })
        assertEquals(ArchitectureTermination.NEIGHBOURHOODS_EXHAUSTED, report.termination)
        assertEquals(1, report.generated); assertEquals(2, report.candidates.single().trials.size)
        val limited = BatchedPopulationSearch(data, config(trials = 7), open = DeviceFixture()::open).search({}, { false })
        assertEquals(ArchitectureTermination.TRIAL_BUDGET, limited.termination)
        assertEquals(6, limited.candidates.sumOf { it.trials.size })
    }

    @Test fun nativeMixedWidthSearchReplaysBothPrecisionsAcrossStreamingRebinds() {
        for (precision in Neuro.TrainingPrecision.entries) {
            val setup = ArchitectureSearchConfig(maxLayers = 1, maxWidth = 4, seeds = listOf(1, 42), requiredSuccesses = 2,
                maxEpochs = 12, checkEvery = 2, initialEpochs = 2, maxTrials = 8, modelsPerBatch = 6,
                batchSize = 3, precision = precision, targetRmse = 0.0,
                hyperParameters = Neuro.HyperParameters(0.07, 0.4, 1.3, 42),
                execution = ArchitectureExecution.BATCHED, budgetPolicy = ArchitectureBudgetPolicy.SUCCESSIVE_HALVING)
            val report = BatchedPopulationSearch(data, setup, (1..4).map { NetworkArchitecture(listOf(it)) }).search({}, { false })
            assertEquals(0, report.numericalFailures)
            val candidate = report.candidates.first { it.valid && it.trials.any { trial -> trial.batchSegments.size > 1 } }
            val trial = candidate.trials.first { it.batchSegments.size > 1 }
            assertTrue(trial.bestEpoch > trial.batchSegments.first().endEpoch)
            val tolerance = if (precision == Neuro.TrainingPrecision.FP32) 2e-6 else 1e-12
            NeuroStudio().use { studio ->
                assertTrue(studio.replayArchitecture(report, candidate, trial))
                assertEquals(trial.bestEpoch, studio.epochs)
                assertArrayEquals(trial.snapshot!!.parameters(), studio.frame().diagnostics.parameters(), tolerance)
                assertEquals(trial.trainingRmseAtBest, studio.frame().diagnostics.error(), tolerance)
            }
        }
    }

    private class DeviceFixture(private val failingWidth: Int = -1, private val variableSpeed: Boolean = false) {
        var opens = 0
        var closes = 0
        var advances = 0
        var clockNanos = 1L
        val orders = HashMap<Long, MutableList<IntArray>>()
        val lastState = HashMap<Long, NeuroTrainingState>()
        fun open(initial: Array<NeuroTrainingState>): TensorFlowSearchCohort {
            opens++
            return object : TensorFlowSearchCohort {
                private val states = Array(initial.size) { smallStateCopy(initial[it], shareDataset = true) }
                private val bestStates = Array(initial.size) { smallStateCopy(initial[it], shareDataset = true) }
                private val epochs = IntArray(initial.size)
                private val bestEpochs = IntArray(initial.size)
                private val best = DoubleArray(initial.size) { Double.POSITIVE_INFINITY }
                private val failed = BooleanArray(initial.size)
                private var closed = false
                override val size = initial.size
                override val paddedTopology = IntArray(initial[0].topology.size) { position -> initial.maxOf { it.topology[position] } }
                override val info = TrainingDeviceInfo(TrainingBackend.CPU, "Fixture", "fixture", kernelVersion = "fixture")
                override fun advance(orders: Array<Array<IntArray>>, active: BooleanArray): BooleanArray {
                    check(!closed); advances++
                    if (variableSpeed) clockNanos += if (states[0].topology.size > 3) 40_000_000L else 1_000_000L
                    for (index in states.indices) if (active[index]) {
                        if (states[index].topology[1] == failingWidth && index % 2 == 0) { failed[index] = true; continue }
                        val key = states[index].weights[0][0].toBits()
                        this@DeviceFixture.orders.getOrPut(key) { arrayListOf() }.addAll(orders[index].map { it.copyOf() })
                        epochs[index] += orders[index].size
                        states[index].weightVelocity[0][0] += orders[index].size
                        lastState[key] = smallStateCopy(states[index])
                    }
                    return failed.copyOf()
                }
                override fun score(active: BooleanArray): TensorFlowCohortMetrics {
                    val values = DoubleArray(size) { 0.8 + states[it].topology[1] * 0.001 - states[it].weightVelocity[0][0] * 0.0001 }
                    for (index in states.indices) if (active[index] && !failed[index] && values[index] < best[index]) {
                        best[index] = values[index]; bestEpochs[index] = epochs[index]
                        bestStates[index] = smallStateCopy(states[index], shareDataset = true)
                    }
                    return TensorFlowCohortMetrics(epochs.copyOf(), values, values.copyOf(), best.copyOf(), bestEpochs.copyOf(), failed.copyOf())
                }
                override fun exportState(lane: Int, best: Boolean) = smallStateCopy(if (best) bestStates[lane] else states[lane], shareDataset = true)
                override fun exportStates(lanes: IntArray, best: Boolean) = Array(lanes.size) { exportState(lanes[it], best) }
                override fun close() { if (!closed) { closes++; closed = true } }
            }
        }
    }
}
