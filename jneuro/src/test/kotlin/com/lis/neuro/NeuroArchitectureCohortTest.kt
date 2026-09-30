package com.lis.neuro

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test
import java.util.concurrent.atomic.AtomicBoolean

class NeuroArchitectureCohortTest {
    private fun data() = ArchitectureSearchData.fitting(NeuroLearningSets.create(NeuroLearningSets.Kind.XOR, 42).toList())
    private fun config(strategy: ArchitectureSearchStrategy = ArchitectureSearchStrategy.EXHAUSTIVE) = ArchitectureSearchConfig(
        strategy = strategy, maxLayers = 1, minWidth = 4, maxWidth = 4, seeds = listOf(1, 42, 123, 999, 2026), requiredSuccesses = 1,
        maxTrials = 5, maxEpochs = 7, checkEvery = 3, parallelism = 3, engine = TrainingEngine.SMALL,
        precision = Neuro.TrainingPrecision.FP32, hyperParameters = Neuro.HyperParameters(0.6, 0.2, 1.0, 42, sigmoidMode = Neuro.SigmoidMode.FAST))

    @Test fun exhaustiveCohortsKeepSeedOrderScoringBoundariesAndIndividualReplay() {
        val settings = config()
        val report = NeuroArchitectureSearch().search(data(), settings)
        val candidate = report.candidates.single()
        assertTrue(candidate.valid)
        assertEquals(settings.seeds, candidate.trials.map { it.seed })
        assertTrue(report.peakParallelTrials in 1..settings.parallelism)
        for (trial in candidate.trials) {
            assertTrue(trial.cohort)
            assertEquals(7, trial.epochs)
            assertEquals(listOf(0, 3, 6, 7), trial.history.map { it.epoch })
            NeuroStudio().use { studio ->
                assertTrue(studio.replayArchitecture(report, candidate, trial))
                assertEquals(trial.deviceInfo, studio.frame().deviceInfo)
                assertArrayEquals(trial.snapshot!!.parameters(), studio.frame().diagnostics.parameters(), 1e-10)
            }
        }
    }

    @Test fun adaptiveCohortsFundWholeSeedGroupsAndUseTheExistingTrialLimit() {
        val settings = ArchitectureSearchConfig(strategy = ArchitectureSearchStrategy.ADAPTIVE,
            maxLayers = 1, maxWidth = 16, seeds = listOf(1, 42), requiredSuccesses = 1,
            maxTrials = 6, maxEpochs = 3, checkEvery = 1, parallelism = 4, engine = TrainingEngine.SMALL)
        val report = NeuroArchitectureSearch().search(data(), settings)
        assertTrue(report.evaluated > 0)
        assertTrue(report.candidates.sumOf { it.trials.size } <= settings.maxTrials)
        assertTrue(report.peakParallelTrials in 1..settings.parallelism)
        assertTrue(report.candidates.all { it.valid && it.trials.map { trial -> trial.seed } == settings.seeds })
        assertTrue(report.candidates.flatMap { it.trials }.all { it.cohort && it.epochs == 3 })
    }

    @Test fun cohortCancellationRetainsIndependentCheckpointsAndNeverStartsWhenAlreadyCancelled() {
        val settings = config()
        val stopped = AtomicBoolean()
        val trials = evaluateArchitectureCohort(data(), settings, NetworkArchitecture(listOf(4)), listOf(1, 42),
            stopped::get, { _, epoch, _ -> if (epoch == 3) stopped.set(true) }, "cancel-test")
        assertTrue(trials.all { it.state == ArchitectureTrialState.CANCELLED && it.epochs == 3 })
        assertTrue(trials.all { it.history.map { point -> point.epoch } == listOf(0, 3) && it.snapshot != null })
        val initial = evaluateArchitectureCohort(data(), settings, NetworkArchitecture(listOf(4)), listOf(1, 42),
            { true }, { _, _, _ -> fail("No progress before opening") }, "cancel-before-open",
            { _, _, _, _, _ -> error("Already-cancelled search must not acquire a cohort") })
        assertTrue(initial.all { it.state == ArchitectureTrialState.CANCELLED && it.epochs == 0 && it.deviceInfo == null })
    }

    @Test fun failedLaneIsMaskedWhileSiblingsFinishAndStartupFailurePreservesItsReason() {
        val settings = config()
        val trials = evaluateArchitectureCohort(data(), settings, NetworkArchitecture(listOf(4)), listOf(1, 42),
            { false }, { lane, _, _ -> if (lane == 0) error("fixture scoring failure") }, "mask-test")
        assertEquals(ArchitectureTrialState.FAILED, trials[0].state)
        assertTrue(trials[0].failure.contains("fixture scoring failure"))
        assertEquals(0, trials[0].epochs)
        assertEquals(ArchitectureTrialState.COMPLETED, trials[1].state)
        assertEquals(7, trials[1].epochs)
        val failed = evaluateArchitectureCohort(data(), settings, NetworkArchitecture(listOf(4)), listOf(1, 42),
            { false }, { _, _, _ -> }, "startup-failure", { _, _, _, _, _ -> error("cohort backend unavailable") })
        assertTrue(failed.all { it.state == ArchitectureTrialState.FAILED && it.failure == "cohort backend unavailable" })
    }

    @Test fun longCohortHistoriesRetainInitialBestAndFinalScoredCheckpoints() {
        val settings = ArchitectureSearchConfig(maxLayers = 1, minWidth = 4, maxWidth = 4, seeds = listOf(1, 42), requiredSuccesses = 1,
            maxEpochs = 140, checkEvery = 1, parallelism = 2, engine = TrainingEngine.SMALL)
        val trials = evaluateArchitectureCohort(data(), settings, NetworkArchitecture(listOf(4)), settings.seeds,
            { false }, { _, _, _ -> }, "history-test")
        for (trial in trials) {
            assertEquals(ArchitectureTrialState.COMPLETED, trial.state)
            assertTrue(trial.history.size <= 128)
            assertEquals(0, trial.history.first().epoch)
            assertEquals(140, trial.history.last().epoch)
            assertTrue(trial.history.any { it.epoch == trial.bestEpoch })
        }
    }
}
