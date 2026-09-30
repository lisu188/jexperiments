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

            // Releasing and reopening training must preserve parameters, momentum and shuffle state.
            studio.closeTraining()
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
        val engine = NeuroArchitectureSearch { model, backend, precision, batchSize, selectedEngine ->
            // Both first trials must own real sessions simultaneously before either starts training.
            val session = model.newTrainingSession(backend, precision, batchSize, selectedEngine)
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

    @Test fun cublasFp64StudioSearchAndReplayKeepRecordedBatchConfiguration() =
        checkCublasStudioSearchAndReplay(Neuro.TrainingPrecision.FP64)

    @Test fun cublasFp32StudioSearchAndReplayKeepRecordedBatchConfiguration() =
        checkCublasStudioSearchAndReplay(Neuro.TrainingPrecision.FP32)

    private fun checkCublasStudioSearchAndReplay(precision: Neuro.TrainingPrecision) {
        val config = StudioConfig(hidden = "3", dataset = NeuroLearningSets.Kind.AND, maxEpochs = 4,
            targetError = 0.0, backend = TrainingBackend.CUBLAS, precision = precision, batchSize = 3)
        val data = ArchitectureSearchData.fitting(NeuroLearningSets.create(config.dataset, 42), "AND")
        lateinit var device: TrainingDeviceInfo
        NeuroStudio(config).use { studio ->
            val initial = studio.frame().diagnostics.parameters()
            studio.step(1)
            assertEquals(1, studio.advance())
            val first = studio.frame()
            device = requireNotNull(first.deviceInfo)
            assertEquals(TrainingBackend.CUBLAS, device.backend)
            assertEquals(precision.name, device.precision)
            assertTrue(device.name.isNotBlank())
            assertTrue(device.identity.isNotBlank())
            assertTrue(device.kernelVersion.startsWith("cublas-"))
            assertEquals(StudioState.PAUSED, first.state)
            studio.setRunning(true)
            assertEquals(1, studio.advance(1))
            assertEquals(StudioState.RUNNING, studio.state)
            studio.setRunning(false)
            val paused = studio.frame().diagnostics.parameters()
            assertEquals(0, studio.advance())
            assertArrayEquals(paused, studio.frame().diagnostics.parameters())
            studio.step(1)
            assertEquals(0, studio.advance(1) { true })
            studio.closeTraining()
            assertEquals(1, studio.advance(1))
            assertEquals(3, studio.epochs)
            assertEquals(device, studio.frame().deviceInfo)
            studio.apply(config)
            assertEquals(0, studio.epochs)
            assertArrayEquals(initial, studio.frame().diagnostics.parameters())
            studio.step(1)
            assertEquals(1, studio.advance())
            assertEquals(device, studio.frame().deviceInfo)
            assertArrayEquals(first.diagnostics.parameters(), studio.frame().diagnostics.parameters(), TOLERANCE)
        }

        val searchConfig = ArchitectureSearchConfig(strategy = ArchitectureSearchStrategy.EXHAUSTIVE,
            maxLayers = 1, minWidth = 3, maxWidth = 3, seeds = listOf(42, 123), requiredSuccesses = 1,
            maxEpochs = 2, checkEvery = 1, targetRmse = 0.0, parallelism = 2, maxTrials = 2,
            backend = TrainingBackend.CUBLAS, precision = precision, batchSize = 3)
        val initialSessions = CountDownLatch(2)
        val engine = NeuroArchitectureSearch { model, selected, format, batch, selectedEngine ->
            val session = model.newTrainingSession(selected, format, batch, selectedEngine)
            try {
                initialSessions.countDown()
                check(initialSessions.await(30, TimeUnit.SECONDS)) { "Concurrent cuBLAS sessions did not initialize." }
                session
            } catch (failure: Throwable) {
                try { session.close() } catch (cleanup: Throwable) { failure.addSuppressed(cleanup) }
                throw failure
            }
        }
        val report = engine.search(data, searchConfig)
        assertEquals(ArchitectureTermination.COMPLETED, report.termination)
        assertEquals(2, report.peakParallelTrials)
        val candidate = report.candidates.single()
        assertEquals(2, candidate.trials.size)
        for (trial in candidate.trials) {
            assertEquals(ArchitectureTrialState.COMPLETED, trial.state, trial.failure)
            assertEquals(device, trial.deviceInfo)
            assertEquals(2, trial.epochs)
            assertEquals(8L, trial.sampleUpdates)
        }
        val trial = candidate.trials.maxBy { it.bestEpoch }
        assertTrue(trial.bestEpoch > 0, "cuBLAS replay must execute trained epochs.")
        NeuroStudio(config).use { studio ->
            assertTrue(studio.replayArchitecture(report, candidate, trial))
            val replay = studio.frame()
            assertEquals(TrainingBackend.CUBLAS, replay.config.backend)
            assertEquals(precision, replay.config.precision)
            assertEquals(3, replay.config.batchSize)
            assertEquals(device, replay.deviceInfo)
            assertEquals(trial.bestEpoch, replay.diagnostics.epoch())
            assertEquals(trial.bestRmse, replay.diagnostics.error(), TOLERANCE)
            assertArrayEquals(trial.snapshot!!.parameters(), replay.diagnostics.parameters(), TOLERANCE)
            assertTrue(studio.replayArchitecture(report, candidate, trial))
            assertArrayEquals(replay.diagnostics.parameters(), studio.frame().diagnostics.parameters(), TOLERANCE)
            studio.applyArchitecture(report, candidate)
            studio.step(1)
            assertEquals(1, studio.advance())
            assertEquals(device, studio.frame().deviceInfo)
        }
        println("cuBLAS Studio acceptance: $device; batch=3 with partial final batch; pause/reopen/reset; " +
            "two concurrent search trials; strict scored replay")
    }

    @Test fun smallFp64StudioCohortReplayAndContinuationUseRealCuda() =
        checkSmallStudioCohortReplay(Neuro.TrainingPrecision.FP64)

    @Test fun smallFp32StudioCohortReplayAndContinuationUseRealCuda() =
        checkSmallStudioCohortReplay(Neuro.TrainingPrecision.FP32)

    private fun checkSmallStudioCohortReplay(precision: Neuro.TrainingPrecision) {
        val tolerance = if (precision == Neuro.TrainingPrecision.FP64) 1e-9 else 5e-5
        val config = StudioConfig(hidden = "4", dataset = NeuroLearningSets.Kind.AND, maxEpochs = 14,
            targetError = 0.0, backend = TrainingBackend.CUDA, precision = precision,
            engine = TrainingEngine.SMALL, sigmoid = Neuro.SigmoidMode.FAST)
        fun assertSmallCuda(info: TrainingDeviceInfo) {
            assertEquals(TrainingBackend.CUDA, info.backend)
            assertEquals(TrainingEngine.SMALL, info.engine)
            assertEquals(precision.name, info.precision)
            assertEquals("FAST", info.sigmoid)
            assertTrue(info.name.isNotBlank() && info.identity.isNotBlank() && info.kernelVersion.isNotBlank())
        }
        fun model(studio: NeuroStudio): Neuro = NeuroStudio::class.java.getDeclaredField("network").run {
            isAccessible = true
            get(studio) as Neuro
        }
        fun assertModels(expected: Neuro, actual: Neuro) {
            assertSmallState(expected.exportTrainingState(), actual.exportTrainingState(), tolerance)
            assertEquals(expected.statistics().epochsTrained, actual.statistics().epochsTrained)
            assertEquals(expected.statistics().samplesSeen, actual.statistics().samplesSeen)
            assertEquals(expected.statistics().lastTrainingError, actual.statistics().lastTrainingError, tolerance)
            assertEquals(expected.trainingError(), actual.trainingError(), tolerance)
        }
        fun step(studio: NeuroStudio, count: Int) {
            val before = studio.epochs
            studio.step(count)
            while (studio.hasWork) studio.advance(count)
            assertEquals(before + count, studio.epochs)
            assertEquals(StudioState.PAUSED, studio.state)
        }
        NeuroStudio(config).use { gpu ->
            NeuroStudio(config.copy(backend = TrainingBackend.CPU)).use { cpu ->
                step(gpu, 1); step(cpu, 1)
                assertSmallCuda(requireNotNull(gpu.frame().deviceInfo))
                assertModels(model(cpu), model(gpu))
                step(gpu, 10); step(cpu, 10)
                assertEquals(listOf(0, 10), gpu.frame().checkpoints.map { it.epoch })
                assertModels(model(cpu), model(gpu))
                val beforeRelease = model(gpu).exportTrainingState()
                val beforeStatistics = model(gpu).statistics()
                gpu.closeTraining(); cpu.closeTraining()
                assertSmallState(beforeRelease, model(gpu).exportTrainingState(), 0.0)
                assertEquals(beforeStatistics, model(gpu).statistics())
                step(gpu, 1); step(cpu, 1)
                assertSmallCuda(requireNotNull(gpu.frame().deviceInfo))
                assertEquals(12L, model(gpu).statistics().epochsTrained)
                assertEquals(48L, model(gpu).statistics().samplesSeen)
                assertModels(model(cpu), model(gpu))
                gpu.closeTraining(); cpu.closeTraining()

                val data = ArchitectureSearchData.fitting(NeuroLearningSets.create(config.dataset, 42), "AND")
                fun searchConfig(backend: TrainingBackend) = ArchitectureSearchConfig(
                    strategy = ArchitectureSearchStrategy.EXHAUSTIVE, maxLayers = 1, minWidth = 8, maxWidth = 8,
                    seeds = listOf(42, 123), requiredSuccesses = 1, maxTrials = 2, parallelism = 2,
                    maxEpochs = 7, checkEvery = 3, targetRmse = 0.0, backend = backend, precision = precision,
                    engine = TrainingEngine.SMALL,
                    hyperParameters = Neuro.HyperParameters(0.6, 0.2, 1.0, 42, sigmoidMode = Neuro.SigmoidMode.FAST))
                val report = NeuroArchitectureSearch().search(data, searchConfig(TrainingBackend.CUDA))
                val reference = NeuroArchitectureSearch().search(data, searchConfig(TrainingBackend.CPU))
                assertEquals(ArchitectureTermination.COMPLETED, report.termination)
                assertEquals(2, report.peakParallelTrials)
                assertEquals(0, report.numericalFailures)
                val candidate = report.candidates.single()
                val cpuCandidate = reference.candidates.single()
                assertTrue(candidate.valid && cpuCandidate.valid)
                for (trial in candidate.trials) {
                    assertTrue(trial.cohort)
                    assertSmallCuda(requireNotNull(trial.deviceInfo))
                    assertEquals(7, trial.epochs)
                    assertEquals(28L, trial.sampleUpdates)
                    assertEquals(listOf(0, 3, 6, 7), trial.history.map { it.epoch })
                    val expected = cpuCandidate.trials.single { it.seed == trial.seed }
                    assertTrue(expected.cohort)
                    assertEquals(expected.bestEpoch, trial.bestEpoch)
                    assertEquals(expected.bestRmse, trial.bestRmse, tolerance)
                    assertArrayEquals(expected.snapshot!!.parameters(), trial.snapshot!!.parameters(), tolerance)
                }
                val trial = candidate.trials.maxBy { it.bestEpoch }
                val cpuTrial = cpuCandidate.trials.single { it.seed == trial.seed }
                assertTrue(trial.bestEpoch > 0, "SMALL replay must execute trained epochs.")
                assertTrue(gpu.replayArchitecture(report, candidate, trial))
                assertTrue(cpu.replayArchitecture(reference, cpuCandidate, cpuTrial))
                assertEquals(trial.deviceInfo, gpu.frame().deviceInfo)
                assertEquals(trial.bestEpoch, gpu.epochs)
                assertArrayEquals(trial.snapshot!!.parameters(), gpu.frame().diagnostics.parameters(), 1e-10)
                assertModels(model(cpu), model(gpu))

                // The selected checkpoint may equal the search epoch limit. Continue its installed model
                // through the public service, preserving the recorded search budget and replay provenance.
                gpu.closeTraining(); cpu.closeTraining()
                NeuroTrainingDeviceService().use { service ->
                    service.openSession(model(gpu), TrainingBackend.CUDA, precision, engine = TrainingEngine.SMALL).use { resumedGpu ->
                        service.openSession(model(cpu), TrainingBackend.CPU, precision, engine = TrainingEngine.SMALL).use { resumedCpu ->
                            assertSmallCuda(resumedGpu.info)
                            resumedGpu.trainEpoch(); resumedCpu.trainEpoch()
                            assertEquals(trial.bestEpoch.toLong() + 1, model(gpu).statistics().epochsTrained)
                            assertEquals((trial.bestEpoch.toLong() + 1) * 4, model(gpu).statistics().samplesSeen)
                            assertModels(model(cpu), model(gpu))
                        }
                    }
                }
                println("SMALL $precision Studio acceptance: ${trial.deviceInfo}; 1+10 steps, exact reopen, two-seed cohort, scored replay and continuation")
            }
        }
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
