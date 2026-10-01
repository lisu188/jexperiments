package com.lis.neuro

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

class NeuroStudioTest {
    @Test fun closingTrainingAllowsExactResumeAndTerminalCloseRejectsNewWork() {
        val config = StudioConfig(hidden = "4", maxEpochs = 10, targetError = 0.0, engine = TrainingEngine.SMALL)
        val expected = NeuroStudio(config).use { studio ->
            studio.step(3)
            while (studio.hasWork) studio.advance(3)
            studio.frame().diagnostics.parameters()
        }
        val sessions = RecordingTrainingSessions()
        for (injected in listOf(false, true)) {
            val studio = if (injected) NeuroStudio(config, openSession = sessions::open) else NeuroStudio(config)
            studio.use {
                studio.step(2)
                while (studio.hasWork) studio.advance(2)
                val before = studio.frame().diagnostics.parameters()
                studio.closeTraining()
                studio.closeTraining()
                assertEquals(2, studio.epochs)
                assertArrayEquals(before, studio.frame().diagnostics.parameters())
                studio.step(1)
                assertEquals(1, studio.advance(1))
                assertArrayEquals(expected, studio.frame().diagnostics.parameters(), 1e-10)
            }
            studio.close()
            studio.closeTraining()
            assertThrows(IllegalStateException::class.java) { studio.step(1) }
            assertThrows(IllegalStateException::class.java) { studio.advance(1) }
            assertThrows(IllegalStateException::class.java) { studio.setRunning(true) }
            assertThrows(IllegalStateException::class.java) { studio.apply(config) }
            assertThrows(IllegalStateException::class.java) { studio.compareSeeds(1) }
        }
        assertEquals(2, sessions.opened.get())
        assertEquals(sessions.opened.get(), sessions.closed.get())
    }

    @Test fun smallEngineChunksRespectStepsMilestonesLimitsAndCancellation() {
        val sessions = RecordingTrainingSessions()
        val config = StudioConfig(hidden = "4", maxEpochs = 70, targetError = 0.0,
            engine = TrainingEngine.SMALL, precision = Neuro.TrainingPrecision.FP32, sigmoid = Neuro.SigmoidMode.FAST)
        NeuroStudio(config, openSession = sessions::open).use { studio ->
            studio.step(70)
            while (studio.epochs < 10) studio.advance(70) { studio.epochs >= 10 }
            assertEquals(10, studio.epochs)
            assertTrue(studio.hasWork)
            while (studio.hasWork) studio.advance(70)
            assertEquals(70, studio.epochs)
            assertEquals(StudioState.LIMIT_REACHED, studio.state)
            assertEquals(listOf(0, 10, 50, 70), studio.frame().checkpoints.map { it.epoch })
            assertTrue(sessions.chunkRequests.all { it.checkEvery == 1 && it.targetError == 0.0 })
            assertTrue(sessions.chunkRequests.first().maxEpochs <= 10)
            assertEquals(TrainingEngine.SMALL, studio.frame().deviceInfo!!.engine)
            assertEquals("FP32", studio.frame().deviceInfo!!.precision)
            assertEquals("FAST", studio.frame().deviceInfo!!.sigmoid)
            assertEquals(listOf(TrainingEngine.SMALL), sessions.engines.toList())
        }
        assertEquals(sessions.opened.get(), sessions.closed.get())
    }

    @Test fun smallStudioStopsAtTheSameTargetEpochAsSingleEpochTraining() {
        val config = StudioConfig(hidden = "4", maxEpochs = 200, targetError = 0.49,
            engine = TrainingEngine.SMALL, precision = Neuro.TrainingPrecision.FP32, sigmoid = Neuro.SigmoidMode.FAST)
        NeuroStudio(config).use { single ->
            while (single.state !in setOf(StudioState.CONVERGED, StudioState.LIMIT_REACHED)) {
                single.step(1); single.advance(1)
            }
            NeuroStudio(config).use { chunked ->
                chunked.setRunning(true)
                while (chunked.hasWork) chunked.advance(64)
                assertEquals(single.epochs, chunked.epochs)
                assertEquals(single.state, chunked.state)
                assertEquals(single.currentError, chunked.currentError, 1e-10)
                assertArrayEquals(single.frame().diagnostics.parameters(), chunked.frame().diagnostics.parameters(), 1e-10)
            }
        }
    }

    @Test fun enginePrecisionAndSigmoidConfigurationRemainExplicit() {
        assertEquals(TrainingEngine.REFERENCE, StudioConfig().engine)
        assertEquals(Neuro.SigmoidMode.EXACT, StudioConfig().sigmoid)
        assertThrows(IllegalArgumentException::class.java) { StudioConfig(hidden = "6", engine = TrainingEngine.SMALL) }
        assertThrows(IllegalArgumentException::class.java) { StudioConfig(hidden = "4,4,4,4,4", engine = TrainingEngine.SMALL) }
        for (backend in TrainingBackend.entries) {
            assertDoesNotThrow { StudioConfig(hidden = "4", backend = backend, engine = TrainingEngine.SMALL, precision = Neuro.TrainingPrecision.FP32) }
        }
        for (backend in TrainingBackend.entries) {
            assertDoesNotThrow { StudioConfig(backend = backend, precision = Neuro.TrainingPrecision.FP32) }
        }
    }

    @Test fun logsRequestedAndEffectiveDeviceAndRunLifecycleWithoutTrainingData() {
        NeuroApplicationLogCapture().use { capture ->
            val sessions = RecordingTrainingSessions()
            val config = StudioConfig(maxEpochs = 2, targetError = 0.0, backend = TrainingBackend.AUTO,
                precision = Neuro.TrainingPrecision.FP32, batchSize = 3)
            NeuroStudio(config, openSession = sessions::open).use { studio ->
                studio.step(2)
                assertEquals(2, studio.advance())
                assertEquals(StudioState.LIMIT_REACHED, studio.state)
            }
            val created = capture.events("studio.run.created").single()
            val runId = capture.fields(created)["runId"]
            assertNotNull(runId)
            val ready = capture.fields(capture.events("studio.backend.ready").single())
            assertEquals(runId, ready["runId"])
            assertNotNull(ready["model"])
            assertNotNull(ready["session"])
            assertEquals("AUTO", ready["requestedBackend"].toString())
            assertEquals("FP32", ready["requestedPrecision"].toString())
            assertEquals("CPU", ready["effectiveBackend"].toString())
            assertEquals("FP32", ready["effectivePrecision"].toString())
            assertEquals("3", ready["batchSize"].toString())
            assertEquals(runId, capture.fields(capture.events("studio.training.completed").single())["runId"])
            assertEquals(runId, capture.fields(capture.events("studio.session.released").single())["runId"])
            assertTrue(capture.events("studio.training.progress").isNotEmpty())
            assertTrue(capture.records.none { record -> capture.fields(record).keys.any { it in setOf("weights", "inputs", "targets", "predictions") } })
        }
    }


    @Test fun failedStudyReportsItsCauseWithoutPublishingSeedResults() {
        NeuroApplicationLogCapture().use { capture ->
            val failure = IllegalStateException("seed backend unavailable")
            NeuroStudio(openSession = { _, _, _, _, _ -> throw failure }).use { studio ->
                assertSame(failure, assertThrows(IllegalStateException::class.java) { studio.compareSeeds(1) })
                assertTrue(studio.frame().seeds.isEmpty())
            }
            assertSame(failure, capture.events("studio.study.failed").single().thrown)
            assertTrue(capture.events("studio.study.completed").isEmpty())
        }
    }

    @Test fun selectedBackendSessionsAreReusedAndReleasedOnResetFailureAndClose() {
        val sessions = RecordingTrainingSessions()
        NeuroStudio(StudioConfig(backend = TrainingBackend.CUDA), openSession = sessions::open).use { studio ->
            assertNull(studio.frame().deviceInfo)
            studio.step(2); studio.advance()
            studio.step(1); studio.advance()
            assertEquals(1, sessions.opened.get())
            assertEquals(3, sessions.epochCalls.get())
            assertEquals(TrainingBackend.CUDA, studio.frame().deviceInfo!!.backend)
            assertEquals("CUDA test fixture", studio.frame().deviceInfo!!.name)
            studio.apply(studio.activeConfig)
            assertEquals(1, sessions.closed.get())
            assertNull(studio.frame().deviceInfo)
            studio.step(1); studio.advance()
            sessions.failEpoch = true
            studio.step(1)
            assertThrows(IllegalStateException::class.java) { studio.advance() }
            studio.fail("Epoch failed")
            assertEquals(2, sessions.closed.get())
            assertEquals(1, studio.frame().diagnostics.epoch())
            sessions.failEpoch = false
            studio.apply(StudioConfig(dataset = NeuroLearningSets.Kind.CUSTOM, backend = TrainingBackend.CUDA))
            studio.addSample(0.2, 0.3, 1.0); studio.step(1); studio.advance()
            studio.addSample(0.4, 0.5, 0.0)
            assertEquals(3, sessions.closed.get())
            studio.step(1); studio.advance()
        }
        assertEquals(sessions.opened.get(), sessions.closed.get())
        assertTrue(sessions.requested.all { it == TrainingBackend.CUDA })
    }

    @Test fun unavailableBackendNeverFallsBackAndSeedStudyClosesCancelledSessions() {
        val sessions = RecordingTrainingSessions().apply { unavailable = true }
        NeuroStudio(StudioConfig(backend = TrainingBackend.CUDA), openSession = sessions::open).use { studio ->
            studio.step(1)
            assertThrows(IllegalStateException::class.java) { studio.advance() }
            assertEquals(listOf(TrainingBackend.CUDA), sessions.requested.toList())
            assertEquals(0, studio.epochs)
            sessions.unavailable = false
            assertEquals(4, studio.compareSeeds(2).size)
            assertEquals(4, sessions.closed.get())
            var polls = 0
            assertTrue(studio.compareSeeds(100) { ++polls > 3 }.isEmpty())
            assertEquals(5, sessions.closed.get())
            assertTrue(sessions.requested.all { it == TrainingBackend.CUDA })
        }
    }

    @Test fun failureAfterCommittedEpochReportsItsActualParametersErrorAndEpoch() {
        for ((backend, batch) in listOf(TrainingBackend.CUDA to 1, TrainingBackend.CUBLAS to 3)) {
            val sessions = RecordingTrainingSessions().apply { failAfterCommit = true }
            lateinit var model: Neuro
            val config = StudioConfig(backend = backend, batchSize = batch, targetError = 0.0)
            NeuroStudio(config, openSession = { network, selected, precision, batchSize, engine ->
                model = network
                sessions.open(network, selected, precision, batchSize, engine)
            }).use { studio ->
                val initial = studio.frame()
                studio.step(1)
                val failure = assertThrows(IllegalStateException::class.java) { studio.advance() }
                assertEquals(1L, model.statistics().epochsTrained)
                assertEquals(4L, model.statistics().samplesSeen)
                val committed = NeuroXorDiagnostics.capture(model, 1, model.trainingError())
                studio.fail(failure.message!!)
                val failed = studio.frame()
                assertEquals(StudioState.FAILED, failed.state)
                assertFalse(studio.hasWork)
                assertEquals(1, studio.epochs)
                assertEquals(1, failed.diagnostics.epoch())
                assertEquals(committed.error(), failed.diagnostics.error(), 0.0)
                assertArrayEquals(committed.parameters(), failed.diagnostics.parameters(), 0.0)
                assertFalse(initial.diagnostics.parameters().contentEquals(failed.diagnostics.parameters()))
                assertEquals(1, failed.history.last().epoch)
                assertTrue(failed.message.contains("cleanup after committed epoch"))
                assertEquals(1, sessions.closed.get())
            }
        }
    }

    @Test fun failedBackendCleanupStillAllowsConfigurationRecovery() {
        val sessions = RecordingTrainingSessions()
        NeuroStudio(StudioConfig(backend = TrainingBackend.CUDA), openSession = sessions::open).use { studio ->
            studio.step(1); studio.advance()
            sessions.failClose = true
            studio.fail("Epoch failed")
            assertEquals(StudioState.FAILED, studio.state)
            assertTrue(studio.frame().message.contains("Epoch failed"))
            assertTrue(studio.frame().message.contains("CUDA fixture cleanup failed"))
            sessions.failClose = false
            studio.apply(StudioConfig())
            studio.step(1); studio.advance()
            assertEquals(TrainingBackend.CPU, studio.frame().deviceInfo!!.backend)
        }
        assertEquals(sessions.opened.get(), sessions.closed.get())
    }

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

    @Test fun configuredMiniBatchesKeepEpochBoundariesAndReachSeedStudies() {
        val sessions = RecordingTrainingSessions()
        val config = StudioConfig("2", maxEpochs = 8, targetError = 0.0,
            backend = TrainingBackend.CUBLAS, precision = Neuro.TrainingPrecision.FP32, batchSize = 3)
        val expected = Neuro(config.topology(), Neuro.HyperParameters(config.learningRate, config.momentum, 1.0, config.seed))
        NeuroLearningSets.addTo(expected, NeuroLearningSets.create(config.dataset, 0xC0FFEE42L).toList())
        NeuroStudio(config, openSession = sessions::open).use { studio ->
            studio.step(4)
            assertEquals(4, studio.advance(10))
            expected.trainMiniBatch(4, 3, 1, Neuro.BatchBackend.CPU, Neuro.TrainingPrecision.FP32)
            assertEquals(4, studio.epochs)
            assertEquals(expected.trainingError(), studio.currentError, 0.0)
            assertArrayEquals(NeuroXorDiagnostics.capture(expected, 4, expected.trainingError()).parameters(),
                studio.frame().diagnostics.parameters(), 0.0)
            assertEquals(TrainingBackend.CUDA, studio.frame().deviceInfo!!.backend)
            assertEquals("FP32", studio.frame().deviceInfo!!.precision)
            studio.setRunning(true)
            var checks = 0
            assertEquals(1, studio.advance(4) { ++checks > 2 })
            assertEquals(5, studio.epochs)
            studio.setRunning(false)
            assertEquals(0, studio.advance())
            val before = studio.frame().diagnostics.parameters()
            assertEquals(4, studio.compareSeeds(2).size)
            assertArrayEquals(before, studio.frame().diagnostics.parameters())
            assertEquals(13, sessions.miniBatches.size)
            assertTrue(sessions.miniBatches.all { it == 3 })
            assertTrue(sessions.configurations.all { it == Triple(TrainingBackend.CUBLAS, Neuro.TrainingPrecision.FP32, 3) })
        }
        assertEquals(sessions.opened.get(), sessions.closed.get())
    }

    @Test fun configuredBatchBackendAdvancesThroughMatrixTrainer() {
        val config = StudioConfig("6", maxEpochs = 20, targetError = 0.0,
            backend = TrainingBackend.AUTO, batchSize = 7,
            precision = Neuro.TrainingPrecision.FP32)
        val studio = NeuroStudio(config)
        val before = studio.currentError
        studio.step(4)
        assertEquals(4, studio.advance(10))
        assertEquals(4, studio.epochs)
        assertEquals(TrainingBackend.AUTO, studio.frame().config.backend)
        assertEquals(7, studio.frame().config.batchSize)
        assertEquals(Neuro.TrainingPrecision.FP32, studio.frame().config.precision)
        assertTrue(studio.currentError.isFinite())
        assertTrue(studio.currentError < before)
    }

    @Test fun validatesConfigurationCommandsAndDiffRendering() {
        for (config in listOf<() -> StudioConfig>(
            { StudioConfig("2,") }, { StudioConfig(maxEpochs = 0) }, { StudioConfig(targetError = -1.0) },
            { StudioConfig(targetError = Double.NaN) }, { StudioConfig(learningRate = 0.0) }, { StudioConfig(momentum = 1.0) },
            { StudioConfig(batchSize = 0) }))
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
