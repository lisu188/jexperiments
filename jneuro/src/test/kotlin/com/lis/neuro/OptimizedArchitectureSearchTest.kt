package com.lis.neuro

import java.util.concurrent.atomic.AtomicBoolean
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

class OptimizedArchitectureSearchTest {
    private fun data() = ArchitectureSearchData.split(NeuroLearningSets.create(NeuroLearningSets.Kind.SPIRAL, 42), 0.2, 42, "Spiral")
    private fun config(execution: ArchitectureExecution, engine: TrainingEngine = TrainingEngine.REFERENCE,
                       precision: Neuro.TrainingPrecision = Neuro.TrainingPrecision.FP64, batch: Int = 1) =
        ArchitectureSearchConfig(minWidth = 4, maxWidth = 8, maxLayers = 2, maxParameters = 256,
            seeds = listOf(1, 42), requiredSuccesses = 2, maxEpochs = 67, checkEvery = 65,
            targetRmse = 0.9, parallelism = 2, maxTrials = 4, strategy = ArchitectureSearchStrategy.EXHAUSTIVE,
            engine = engine, precision = precision, batchSize = batch, execution = execution)

    @Test fun optimizedSpiralPreservesEveryScoringBoundaryBudgetAndReplayAcrossCpuRoutes() {
        val data = data()
        assertEquals(176, data.training.size); assertEquals(44, data.validation.size)
        val manifest = listOf(NetworkArchitecture(listOf(4)), NetworkArchitecture(listOf(8, 4)))
        for (engine in TrainingEngine.entries) for (batch in listOf(1, 3)) {
            val reference = NeuroArchitectureSearch().searchManifest(data, config(ArchitectureExecution.REFERENCE, engine, batch = batch), manifest)
            val progress = ArrayList<ArchitectureSearchProgress>()
            val optimized = NeuroArchitectureSearch().searchManifest(data, config(ArchitectureExecution.OPTIMIZED, engine, batch = batch), manifest, progress::add)
            assertEquals(2, optimized.evaluated)
            assertEquals(0, optimized.numericalFailures)
            assertTrue(optimized.peakWorkers in 1..2)
            assertTrue(optimized.peakResidentModels in 1..2)
            assertTrue(progress.all { it.activeWorkers in 0..2 && it.running.size <= 2 })
            for ((left, right) in reference.candidates.zip(optimized.candidates)) {
                assertEquals(left.architecture, right.architecture)
                for ((expected, actual) in left.trials.zip(right.trials)) {
                    assertEquals(ArchitectureTrialState.COMPLETED, actual.state)
                    assertEquals(ArchitectureExecution.OPTIMIZED, actual.execution)
                    assertEquals(ArchitectureTrialRoute.SESSION, actual.route)
                    assertFalse(actual.cohort)
                    assertEquals(67, actual.epochs)
                    assertEquals(67L * 176, actual.sampleUpdates)
                    assertEquals(listOf(0, 65, 67), actual.history.map { it.epoch })
                    assertEquals(expected.bestEpoch, actual.bestEpoch)
                    assertEquals(expected.bestRmse, actual.bestRmse, 1e-10)
                    assertArrayEquals(expected.snapshot!!.parameters(), actual.snapshot!!.parameters(), 1e-9)
                    assertNotNull(actual.timings)
                    assertTrue(actual.timings!!.openNanos > 0 && actual.timings.trainingNanos > 0 && actual.timings.scoringNanos > 0)
                }
            }
            val candidate = optimized.candidates.first()
            val trial = candidate.trials.first()
            NeuroStudio(StudioConfig(dataset = NeuroLearningSets.Kind.SPIRAL)).use { studio ->
                assertTrue(studio.replayArchitecture(optimized, candidate, trial))
                assertEquals(trial.bestEpoch, studio.epochs)
                assertEquals(data.training, studio.frame().samples)
                assertArrayEquals(trial.snapshot!!.parameters(), studio.frame().diagnostics.parameters(), 1e-9)
            }
        }
    }

    @Test fun optimizedFp32FastAndGeneralReferenceShapesKeepTheirSelectedMathematics() {
        val data = data()
        for (engine in TrainingEngine.entries) {
            val precision = if (engine == TrainingEngine.SMALL) Neuro.TrainingPrecision.FP32 else Neuro.TrainingPrecision.FP64
            val shape = if (engine == TrainingEngine.SMALL) listOf(4, 8) else listOf(3, 6, 5, 2, 7)
            val config = ArchitectureSearchConfig(maxLayers = 5, maxWidth = 8, seeds = listOf(42), requiredSuccesses = 1,
                maxEpochs = 7, checkEvery = 3, parallelism = 1, maxTrials = 1, strategy = ArchitectureSearchStrategy.EXHAUSTIVE,
                engine = engine, precision = precision, hyperParameters = Neuro.HyperParameters(0.6, 0.2, 1.0, 42, sigmoidMode = Neuro.SigmoidMode.FAST),
                execution = ArchitectureExecution.OPTIMIZED)
            val report = NeuroArchitectureSearch().searchManifest(data, config, listOf(NetworkArchitecture(shape)))
            val trial = report.candidates.single().trials.single()
            assertEquals(ArchitectureTrialState.COMPLETED, trial.state)
            assertEquals(listOf(0, 3, 6, 7), trial.history.map { it.epoch })
            assertEquals(precision.name, trial.deviceInfo!!.precision)
            assertEquals("FAST", trial.deviceInfo.sigmoid)
            assertEquals(engine, trial.deviceInfo.engine)
            NeuroStudio(StudioConfig(dataset = NeuroLearningSets.Kind.SPIRAL)).use {
                assertTrue(it.replayArchitecture(report, report.candidates.single(), trial))
            }
        }
    }

    @Test fun manifestIsFrozenValidatedAndOnlyFundsCompleteSeedGroups() {
        val data = data()
        val config = config(ArchitectureExecution.OPTIMIZED)
        val manifest = mutableListOf(NetworkArchitecture(listOf(4)), NetworkArchitecture(listOf(8)))
        val report = NeuroArchitectureSearch().searchManifest(data, config, manifest, { manifest.clear() })
        assertEquals(2, report.generated); assertEquals(2, report.evaluated)
        assertThrows(IllegalArgumentException::class.java) { NeuroArchitectureSearch().searchManifest(data, config, emptyList()) }
        val shape = NetworkArchitecture(listOf(4))
        assertThrows(IllegalArgumentException::class.java) { NeuroArchitectureSearch().searchManifest(data, config, listOf(shape, shape)) }
        assertThrows(IllegalArgumentException::class.java) { NeuroArchitectureSearch().searchManifest(data, config, listOf(NetworkArchitecture(listOf(3)))) }
        assertThrows(IllegalArgumentException::class.java) { NeuroArchitectureSearch().searchManifest(data, ArchitectureSearchConfig(), listOf(shape)) }
        val budget = ArchitectureSearchConfig(minWidth = 4, maxWidth = 8, maxLayers = 1, seeds = listOf(1, 42), requiredSuccesses = 2,
            maxEpochs = 1, checkEvery = 1, maxTrials = 3, strategy = ArchitectureSearchStrategy.EXHAUSTIVE,
            execution = ArchitectureExecution.OPTIMIZED)
        val bounded = NeuroArchitectureSearch().searchManifest(data, budget, listOf(shape, NetworkArchitecture(listOf(8))))
        assertEquals(ArchitectureTermination.TRIAL_BUDGET, bounded.termination)
        assertEquals(1, bounded.evaluated); assertEquals(1, bounded.untested)
        assertEquals(2, bounded.candidates.single().trials.size)
    }

    @Test fun cancellationAndFailureRetainOnlyCommittedWorkAndDoNotQualifyCandidates() {
        val data = data()
        val config = config(ArchitectureExecution.OPTIMIZED)
        val stopped = AtomicBoolean()
        val trial = NeuroArchitectureSearch().evaluate(data, config, NetworkArchitecture(listOf(4)), 42,
            stopped::get, { epoch, _ -> if (epoch == 65) stopped.set(true) })
        assertEquals(ArchitectureTrialState.CANCELLED, trial.state)
        assertEquals(65, trial.epochs); assertEquals(listOf(0, 65), trial.history.map { it.epoch })
        val before = NeuroArchitectureSearch().evaluate(data, config, NetworkArchitecture(listOf(4)), 42, { true }, { _, _ -> fail("Unexpected progress") })
        assertEquals(0, before.epochs); assertNull(before.deviceInfo)
        val failure = IllegalStateException("search fixture failure")
        val failed = NeuroArchitectureSearch { _, _, _, _, _ -> throw failure }
            .searchManifest(data, config, listOf(NetworkArchitecture(listOf(4))))
        assertEquals(2, failed.numericalFailures)
        assertNull(failed.selection.recommended)
        assertTrue(failed.candidates.single().trials.all { it.failure == failure.message && it.epochs == 0 })
        assertEquals(ArchitectureExecution.REFERENCE, ArchitectureSearchConfig().execution)
    }
}
