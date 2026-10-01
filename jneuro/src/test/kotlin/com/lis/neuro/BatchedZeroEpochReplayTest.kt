package com.lis.neuro

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

class BatchedZeroEpochReplayTest {
    @Test fun fp32EpochZeroReplayInstallsRoundedNativeStateWithoutAdvancingCountersOrShuffle() {
        val data = ArchitectureSearchData.fitting(NeuroLearningSets.create(NeuroLearningSets.Kind.XOR, 42))
        val architecture = NetworkArchitecture(listOf(3))
        // The update is representable in FP32 momentum, but too small to improve the initial FP32 prediction.
        val hp = Neuro.HyperParameters(1e-30, 0.4, 1.3, 42)
        val config = ArchitectureSearchConfig(minLayers = 1, maxLayers = 1, minWidth = 3, maxWidth = 3,
            seeds = listOf(42), requiredSuccesses = 1, maxEpochs = 1, checkEvery = 1, maxTrials = 1,
            modelsPerBatch = 2, targetRmse = 0.0, precision = Neuro.TrainingPrecision.FP32,
            execution = ArchitectureExecution.BATCHED, budgetPolicy = ArchitectureBudgetPolicy.FULL,
            hyperParameters = hp)
        val original = data.newNetwork(architecture, hp, 42)
        val initial = original.exportTrainingState()
        val expected = TensorFlowMath.trainingKernel(initial, hp, Neuro.TrainingPrecision.FP32, TrainingBackend.CPU).use {
            it.train(emptyArray(), 1, true)
        }
        assertTrue(initial.weights.indices.any { !initial.weights[it].contentEquals(expected.weights[it]) },
            "Fixture must expose the FP64-to-FP32 initialization rounding")
        val report = BatchedPopulationSearch(data, config, listOf(architecture)).search({}, { false })
        val candidate = report.candidates.single()
        val trial = candidate.trials.single()
        assertEquals(ArchitectureTrialState.COMPLETED, trial.state)
        assertEquals(0, trial.bestEpoch)
        assertEquals(1, trial.epochs)
        val replayed = data.newNetwork(architecture, hp, 42)
        val nextOrder = replayed.reserveTrainingOrders(1).single().copyOf()
        val score = BatchedSearchReplay.replay(replayed, data, config, trial,
            { assertEquals(trial.deviceInfo, it) }, { false })
        assertEquals(trial.bestRmse, score)
        assertSmallState(expected, replayed.exportTrainingState(), 0.0)
        assertEquals(0L, replayed.statistics().epochsTrained)
        assertEquals(0L, replayed.statistics().samplesSeen)
        assertArrayEquals(nextOrder, replayed.reserveTrainingOrders(1).single())

        val afterOne = TensorFlowMath.trainingKernel(expected, hp, Neuro.TrainingPrecision.FP32, TrainingBackend.CPU).use {
            it.train(arrayOf(nextOrder), 1, true)
        }
        val afterOneSnapshot = NeuroXorDiagnostics.Snapshot(1, 0.0, hp.beta, hp.sigmoidMode,
            afterOne.topology, afterOne.weights, afterOne.biases)
        NeuroStudio().use { studio ->
            assertTrue(studio.replayArchitecture(report, candidate, trial))
            assertEquals(0, studio.epochs)
            assertArrayEquals(requireNotNull(trial.snapshot).parameters(), studio.frame().diagnostics.parameters(), 0.0)
            studio.step(1)
            assertEquals(1, studio.advance(1))
            assertEquals(1, studio.epochs)
            assertArrayEquals(afterOneSnapshot.parameters(), studio.frame().diagnostics.parameters(), 0.0,
                "Ordinary training after replay must start from the rounded checkpoint and original next shuffle")
        }
    }
}
