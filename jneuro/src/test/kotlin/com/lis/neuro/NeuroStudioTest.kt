package com.lis.neuro

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

class NeuroStudioTest {
    @Test fun trainsPausesStepsAndPreservesTerminalStates() {
        val studio = NeuroStudio()
        val initial = studio.frame()
        assertEquals(StudioState.READY, initial.state); assertEquals(0, studio.epochs)
        assertFalse(studio.hasWork); assertEquals(0, studio.advance())
        studio.setRunning(true); assertEquals(StudioState.RUNNING, studio.state)
        assertEquals(10, studio.advance(10)); assertEquals(10, studio.epochs)
        studio.setRunning(false); assertEquals(StudioState.PAUSED, studio.state)
        studio.step(10); assertEquals(10, studio.advance(100)); assertEquals(20, studio.epochs)
        assertFalse(studio.hasWork)
        val frame = studio.frame()
        assertEquals(10, frame.beforeEpoch); assertEquals(20, frame.diagnostics.epoch())
        assertEquals(0, initial.diagnostics.epoch())
        assertEquals(listOf(0, 10), frame.checkpoints.map { it.epoch })
        assertEquals(studio.currentError, frame.history.last().error)
        assertEquals(25, frame.history.last().parameterCount); assertEquals(4, frame.history.last().normCount)
        studio.setRunning(true)
        while (studio.hasWork) studio.advance(100)
        assertEquals(StudioState.CONVERGED, studio.state)
        assertEquals(1144, studio.epochs)
        studio.step(1); studio.setRunning(true)
        assertFalse(studio.hasWork); assertEquals(0, studio.advance())
        val final = studio.frame()
        assertEquals(1144, final.checkpoints.last().epoch)
        assertEquals(listOf(0, 10, 50, 100, 250, 500, 1000, 1144), final.checkpoints.map { it.epoch })
    }

    @Test fun resetIsAtomicAcrossDifferentWidthsDepthsAndDatasets() {
        val studio = NeuroStudio()
        studio.step(3); studio.advance(); studio.frame()
        val config = StudioConfig("4,3,2", NeuroLearningSets.Kind.CIRCLE, seed = 17)
        studio.apply(config, true)
        val frame = studio.frame()
        assertEquals(config, studio.activeConfig)
        assertArrayEquals(intArrayOf(2, 4, 3, 2, 1), frame.diagnostics.topology())
        assertEquals(config.description(), "2 → 4 → 3 → 2 → 1")
        assertEquals(0, frame.diagnostics.epoch()); assertEquals(1, frame.history.size)
        assertEquals(1, frame.checkpoints.size); assertTrue(frame.seeds.isEmpty())
        assertEquals(180, frame.samples.size)
        studio.selectHidden(1, 1)
        val secondLayer = studio.frame()
        assertEquals(1, secondLayer.hiddenLayer); assertEquals(1, secondLayer.hiddenStart)
        assertEquals(2, secondLayer.hiddenImages.size)
        assertEquals(0, secondLayer.beforeEpoch)
        studio.step(2); studio.advance(100)
        assertEquals(2, studio.epochs)
    }

    @Test fun emptyAndFinishedRunsNeverKeepPendingWork() {
        val studio = NeuroStudio(StudioConfig(dataset = NeuroLearningSets.Kind.CUSTOM))
        assertEquals(StudioState.EMPTY, studio.state)
        studio.step(10); studio.setRunning(true)
        assertEquals(0, studio.advance()); assertFalse(studio.hasWork)
        assertTrue(studio.frame().diagnostics.error().isNaN()); assertTrue(studio.compareSeeds().isEmpty())
        studio.addSample(0.2, 0.3, 1.0)
        assertEquals(StudioState.READY, studio.state)
        studio.step(1); studio.advance(); assertEquals(1, studio.epochs)
        studio.undoSample(); assertEquals(StudioState.EMPTY, studio.state)
        studio.undoSample(); assertEquals(StudioState.EMPTY, studio.state)
        studio.addSample(0.1, 0.9, 0.0); studio.clearSamples()
        assertTrue(studio.frame().samples.isEmpty())
        studio.apply(StudioConfig("1", maxEpochs = 3, targetError = 1e-12), true)
        assertEquals(3, studio.advance(100)); assertEquals(StudioState.LIMIT_REACHED, studio.state)
        studio.step(10); studio.setRunning(true); assertFalse(studio.hasWork)
        assertEquals(0, studio.advance()); assertEquals(3, studio.frame().checkpoints.last().epoch)
    }

    @Test fun seedStudyIsOnDemandReproducibleAndDoesNotMutateMainModel() {
        val studio = NeuroStudio(StudioConfig("2,2", maxEpochs = 20, targetError = 1e-10))
        val initial = studio.frame()
        assertTrue(initial.seeds.isEmpty())
        val study = studio.compareSeeds(12)
        assertEquals(listOf(1L, 42L, 123L, 999L), study.map { it.seed })
        assertTrue(study.all { it.epochs == 12 && !it.converged })
        assertEquals(0, studio.epochs)
        assertArrayEquals(initial.diagnostics.parameters(), studio.frame().diagnostics.parameters())
        val second = studio.compareSeeds(12)
        assertEquals(study.map { it.error }, second.map { it.error })
        assertTrue(studio.compareSeeds(12) { true }.isEmpty())
        var checks = 0
        assertTrue(studio.compareSeeds(50) { ++checks > 5 }.isEmpty())
        assertEquals(4, studio.frame().seeds.size)
    }

    @Test fun cancellationStopsBeforeAnAdditionalEpochAndErrorsAreRecoverable() {
        val studio = NeuroStudio()
        studio.setRunning(true)
        assertEquals(0, studio.advance(100) { true })
        var checks = 0
        assertTrue(studio.advance(100) { ++checks > 4 } in 1..4)
        studio.fail("Useful failure message")
        assertEquals(StudioState.FAILED, studio.state); assertFalse(studio.hasWork)
        assertEquals("Useful failure message", studio.frame().message)
        studio.apply(StudioConfig())
        assertEquals(StudioState.READY, studio.state); assertEquals("", studio.frame().message)
    }

    @Test fun boundedHistoryRetainsFirstAndLatestSamples() {
        val config = StudioConfig("128,128,128", maxEpochs = 90, targetError = 1e-12)
        val studio = NeuroStudio(config)
        studio.setRunning(true)
        repeat(85) { studio.advance(1); studio.frame() }
        val frame = studio.frame()
        assertTrue(frame.history.size <= 2_000_000 / frame.diagnostics.parameterCount())
        assertEquals(0, frame.history.first().epoch); assertEquals(studio.epochs, frame.history.last().epoch)
        assertTrue(frame.image.width in 8..192)
        val parameters = doubleArrayOf(1.0); val predictions = DoubleArray(4); val norms = doubleArrayOf(2.0)
        val history = StudioHistory(0, 0.5, predictions, parameters, norms)
        parameters[0] = 100.0; predictions[0] = 100.0; norms[0] = 100.0
        assertEquals(1.0, history.parameter(0)); assertEquals(0.0, history.prediction(0)); assertEquals(2.0, history.norm(0))
    }

    @Test fun validatesConfigurationCommandsAndDiffRendering() {
        for (config in listOf<() -> StudioConfig>(
            { StudioConfig("2,") }, { StudioConfig(maxEpochs = 0) }, { StudioConfig(targetError = 0.0) },
            { StudioConfig(targetError = Double.NaN) }, { StudioConfig(learningRate = 0.0) }, { StudioConfig(momentum = 1.0) }))
            assertThrows(IllegalArgumentException::class.java) { config() }
        val studio = NeuroStudio()
        assertThrows(IllegalArgumentException::class.java) { studio.step(0) }
        assertThrows(IllegalArgumentException::class.java) { studio.advance(0) }
        assertThrows(IllegalArgumentException::class.java) { studio.selectHidden(9) }
        assertThrows(IllegalArgumentException::class.java) { studio.selectHidden(0, -1) }
        assertThrows(IllegalArgumentException::class.java) { studio.compareSeeds(0) }
        assertThrows(IllegalArgumentException::class.java) { studio.fail("") }
        assertThrows(IllegalArgumentException::class.java) { studio.addSample(0.5, 0.5, 1.0) }
        assertThrows(IllegalArgumentException::class.java) { studio.undoSample() }
        assertThrows(IllegalArgumentException::class.java) { NeuroStudio.resolution(0, 100) }
        assertThrows(IllegalArgumentException::class.java) { NeuroStudio.renderDifference(DoubleArray(4), DoubleArray(3), 2) }
        val image = NeuroStudio.renderDifference(DoubleArray(4), DoubleArray(4) { 0.2 }, 2)
        assertEquals(NeuroXorDiagnostics.differenceRgb(0.2), image.getRGB(0, 0) and 0xffffff)
    }
}
