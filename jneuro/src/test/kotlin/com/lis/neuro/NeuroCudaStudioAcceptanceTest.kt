package com.lis.neuro

import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.Timeout

/** Real CUDA integration; no fake device, skipped hardware check or GUI-path claim. */
@Tag("cuda")
@Timeout(120)
class NeuroCudaStudioAcceptanceTest {
    @Test fun studioStepsPausesStudiesAndResetsThroughRealCudaSessions() {
        val config = StudioConfig(hidden = "2", maxEpochs = 8, targetError = 0.0, backend = TrainingBackend.CUDA)
        NeuroStudio(config).use { studio ->
            val initial = studio.frame().diagnostics.parameters()
            studio.step(1)
            assertEquals(1, studio.advance())
            val first = studio.frame()
            val device = requireNotNull(first.deviceInfo)
            assertGpu(device)
            assertEquals(StudioState.PAUSED, first.state)
            assertEquals(1, first.diagnostics.epoch())

            studio.setRunning(true)
            assertEquals(2, studio.advance(2))
            assertEquals(StudioState.RUNNING, studio.state)
            studio.setRunning(false)
            val paused = studio.frame()
            assertEquals(StudioState.PAUSED, paused.state)
            assertEquals(0, studio.advance())
            assertArrayEquals(paused.diagnostics.parameters(), studio.frame().diagnostics.parameters())

            // Closing and reopening must preserve host parameters, momentum and shuffle state.
            studio.close()
            studio.step(1)
            assertEquals(1, studio.advance())
            assertEquals(4, studio.epochs)
            assertEquals(device, studio.frame().deviceInfo)
            val beforeStudy = studio.frame().diagnostics.parameters()
            val study = studio.compareSeeds(2)
            assertEquals(listOf(1L, 42L, 123L, 999L), study.map { it.seed })
            assertTrue(study.all { it.epochs == 2 && it.error.isFinite() })
            assertEquals(4, studio.epochs)
            assertArrayEquals(beforeStudy, studio.frame().diagnostics.parameters())
            studio.step(1)
            assertEquals(0, studio.advance(1) { true })
            assertEquals(1, studio.advance(1))
            assertEquals(device, studio.frame().deviceInfo)

            studio.apply(config)
            assertEquals(StudioState.READY, studio.state)
            assertEquals(0, studio.epochs)
            assertNull(studio.frame().deviceInfo)
            assertArrayEquals(initial, studio.frame().diagnostics.parameters())
            studio.step(1)
            assertEquals(1, studio.advance())
            assertEquals(device, studio.frame().deviceInfo)
            assertArrayEquals(first.diagnostics.parameters(), studio.frame().diagnostics.parameters(), TOLERANCE)
            assertEquals(first.diagnostics.error(), studio.currentError, TOLERANCE)
            println("CUDA Studio acceptance: $device; stepped, paused, reopened, compared four seeds and reset reproducibly")
        }
    }

    @Test fun concurrentGpuArchitectureTrialsRecordTheirDeviceAndReplayTheirScoredCheckpoint() {
        val data = ArchitectureSearchData.fitting(NeuroLearningSets.create(NeuroLearningSets.Kind.XOR, 42), "XOR")
        fun config(backend: TrainingBackend) = ArchitectureSearchConfig(
            strategy = ArchitectureSearchStrategy.EXHAUSTIVE, maxLayers = 1, maxWidth = 2,
            seeds = listOf(1, 42), requiredSuccesses = 1, maxEpochs = 4, checkEvery = 1,
            targetRmse = 0.0, parallelism = 2, maxTrials = 4, backend = backend)
        val initialSessions = CountDownLatch(2)
        val engine = NeuroArchitectureSearch { model, backend ->
            // Both first trials must own real sessions simultaneously before either starts training.
            val session = model.newTrainingSession(backend)
            try {
                initialSessions.countDown()
                check(initialSessions.await(30, TimeUnit.SECONDS)) { "Concurrent CUDA sessions did not initialize." }
                session
            } catch (failure: Throwable) {
                try { session.close() } catch (cleanup: Throwable) { failure.addSuppressed(cleanup) }
                throw failure
            }
        }
        val report = engine.search(data, config(TrainingBackend.CUDA))
        assertEquals(ArchitectureTermination.COMPLETED, report.termination)
        assertEquals(2, report.peakParallelTrials)
        assertEquals(2, report.evaluated)
        assertEquals(0, report.numericalFailures)
        assertTrue(report.candidates.all { it.valid && it.trials.size == 2 })
        val device = requireNotNull(report.candidates.first().trials.first().deviceInfo)
        assertGpu(device)
        val cpu = NeuroArchitectureSearch().search(data, config(TrainingBackend.CPU))
        for (candidate in report.candidates) {
            val reference = cpu.candidates.single { it.architecture == candidate.architecture }
            for (trial in candidate.trials) {
                val expected = reference.trials.single { it.seed == trial.seed }
                assertEquals(ArchitectureTrialState.COMPLETED, trial.state, trial.failure)
                assertEquals(device, trial.deviceInfo)
                assertEquals(4, trial.epochs)
                assertEquals(16L, trial.sampleUpdates)
                assertEquals(expected.bestEpoch, trial.bestEpoch)
                assertEquals(expected.bestRmse, trial.bestRmse, TOLERANCE)
                assertArrayEquals(expected.snapshot!!.parameters(), trial.snapshot!!.parameters(), TOLERANCE)
            }
        }
        val (candidate, trial) = report.candidates.flatMap { candidate -> candidate.trials.map { candidate to it } }
            .maxBy { it.second.bestEpoch }
        assertTrue(trial.bestEpoch > 0, "Replay acceptance must execute trained epochs, not only initial parameters.")
        NeuroStudio(StudioConfig(backend = TrainingBackend.CUDA)).use { studio ->
            assertTrue(studio.replayArchitecture(report, candidate, trial))
            val replay = studio.frame()
            assertEquals(TrainingBackend.CUDA, replay.config.backend)
            assertEquals(device, replay.deviceInfo)
            assertEquals(trial.bestEpoch, replay.diagnostics.epoch())
            assertEquals(trial.trainingRmseAtBest, replay.diagnostics.error(), TOLERANCE)
            assertArrayEquals(trial.snapshot!!.parameters(), replay.diagnostics.parameters(), TOLERANCE)
            assertTrue(replay.replayNote.contains("Search replay"))
            assertTrue(studio.replayArchitecture(report, candidate, trial))
            assertArrayEquals(replay.diagnostics.parameters(), studio.frame().diagnostics.parameters(), TOLERANCE)
            studio.applyArchitecture(report, candidate)
            assertEquals(TrainingBackend.CUDA, studio.activeConfig.backend)
            assertEquals(0, studio.epochs)
            studio.step(1)
            assertEquals(1, studio.advance())
            assertEquals(device, studio.frame().deviceInfo)
        }
        println("CUDA architecture acceptance: $device; four trials, two concurrent sessions, repeatable scored replay")
    }

    private fun assertGpu(info: TrainingDeviceInfo) {
        assertEquals(TrainingBackend.CUDA, info.backend)
        assertEquals("FP64", info.precision)
        assertTrue(info.name.isNotBlank())
        assertTrue(info.identity.isNotBlank())
        assertTrue(info.kernelVersion.isNotBlank())
    }

    companion object { private const val TOLERANCE = 1e-10 }
}
