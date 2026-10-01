package com.lis.neuro

import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

/** Tests the real asynchronous scheduler with a CPU driver fixture; hardware acceptance remains separate. */
class OptimizedCudaSearchTest {
    private fun data() = ArchitectureSearchData.split(NeuroLearningSets.create(NeuroLearningSets.Kind.SPIRAL, 42), 0.2, 42, "Spiral")
    private fun config(precision: Neuro.TrainingPrecision = Neuro.TrainingPrecision.FP64, epochs: Int = 7,
                       strategy: ArchitectureSearchStrategy = ArchitectureSearchStrategy.EXHAUSTIVE) =
        ArchitectureSearchConfig(minWidth = 4, maxWidth = 16, maxLayers = 2, maxParameters = 1024,
            seeds = listOf(1, 42), requiredSuccesses = 2, maxEpochs = epochs, checkEvery = 3,
            targetRmse = 0.9, parallelism = 2, maxTrials = 6, strategy = strategy, initialHidden = listOf(4),
            backend = TrainingBackend.CUDA, engine = TrainingEngine.SMALL, precision = precision,
            execution = ArchitectureExecution.OPTIMIZED)

    @Test fun heterogeneousSpiralTrialsUseBoundedWorkersActualQueueProvenanceAndStrictReplay() {
        for (precision in Neuro.TrainingPrecision.entries) {
            val identity = "heterogeneous-search-$precision"
            val driver = SearchCudaTestDriver(identity)
            val engine = NeuroArchitectureSearch(openCudaService = { format, batch -> SearchCudaService(format, batch, { driver }) })
            val snapshots = ArrayList<ArchitectureSearchProgress>()
            val manifest = listOf(NetworkArchitecture(listOf(4)), NetworkArchitecture(listOf(8, 4)), NetworkArchitecture(listOf(16, 8)))
            val report = engine.searchManifest(data(), config(precision), manifest, snapshots::add)
            assertEquals(ArchitectureTermination.COMPLETED, report.termination)
            assertEquals(3, report.evaluated); assertEquals(0, report.numericalFailures)
            assertTrue(report.peakWorkers in 1..2)
            assertTrue(report.peakResidentModels in 1..64)
            assertTrue(report.gpuBatches > 0)
            assertTrue(snapshots.all { it.activeWorkers in 0..2 && it.residentModels in 0..64 && it.queuedGpuRequests in 0..64 })
            assertEquals(1, driver.closes); assertTrue(driver.memory.isEmpty())
            assertTrue(driver.launches.all { it.size in 1..64 })
            assertTrue(driver.threads.all { it.startsWith("jneuro-search-cuda") })
            for (candidate in report.candidates) for (trial in candidate.trials) {
                assertEquals(ArchitectureTrialRoute.CUDA_QUEUE, trial.route)
                assertEquals(ArchitectureExecution.OPTIMIZED, trial.execution)
                assertEquals(7, trial.epochs); assertEquals(7L * 176, trial.sampleUpdates)
                assertEquals(listOf(0, 3, 6, 7), trial.history.map { it.epoch })
                assertEquals(precision.name, trial.deviceInfo!!.precision)
                assertTrue(trial.deviceInfo.kernelVersion.startsWith("small-search-v3/"))
                assertTrue(trial.timings!!.trainingNanos > 0 && trial.timings.scoringNanos > 0 && trial.timings.closeNanos > 0)
            }
            val candidate = report.candidates.first()
            val trial = candidate.trials.first()
            NeuroStudio(StudioConfig(dataset = NeuroLearningSets.Kind.SPIRAL), openSearchCuda = { format, batch ->
                SearchCudaService(format, batch, { SearchCudaTestDriver(identity) }, maximumModels = 1)
            }).use { studio ->
                assertTrue(studio.replayArchitecture(report, candidate, trial))
                assertEquals(trial.bestEpoch, studio.epochs)
                assertEquals(176, studio.frame().samples.size)
                assertArrayEquals(trial.snapshot!!.parameters(), studio.frame().diagnostics.parameters(), 1e-10)
            }
            NeuroStudio(StudioConfig(dataset = NeuroLearningSets.Kind.SPIRAL), openSearchCuda = { format, batch ->
                SearchCudaService(format, batch, { SearchCudaTestDriver("other-device") }, maximumModels = 1)
            }).use { studio ->
                assertThrows(IllegalStateException::class.java) { studio.replayArchitecture(report, candidate, trial) }
                assertEquals(0, studio.epochs); assertEquals(220, studio.frame().samples.size)
            }
        }
    }

    @Test fun adaptiveQueueCompletesFullBudgetsAndCancellationDrainsNativeWork() {
        val driver = SearchCudaTestDriver("adaptive-queue")
        val search = NeuroArchitectureSearch(openCudaService = { format, batch -> SearchCudaService(format, batch, { driver }) })
        val report = search.search(data(), config(strategy = ArchitectureSearchStrategy.ADAPTIVE))
        assertEquals(ArchitectureTermination.TRIAL_BUDGET, report.termination)
        assertEquals(3, report.evaluated); assertEquals(3, report.lineage.size)
        assertTrue(report.candidates.all { it.valid && it.trials.all { trial -> trial.epochs == 7 } })
        assertEquals(1, driver.closes)

        val stopped = AtomicBoolean()
        val cancelledDriver = SearchCudaTestDriver("cancelled-queue").apply { beforeLaunch = { stopped.set(true) } }
        val cancelled = NeuroArchitectureSearch(openCudaService = { format, batch -> SearchCudaService(format, batch, { cancelledDriver }) })
            .searchManifest(data(), config(epochs = 9999), listOf(NetworkArchitecture(listOf(4))), cancelled = stopped::get)
        assertEquals(ArchitectureTermination.CANCELLED, cancelled.termination)
        assertNull(cancelled.selection.recommended)
        assertTrue(cancelled.candidates.single().trials.all { it.state == ArchitectureTrialState.CANCELLED })
        assertTrue(cancelled.candidates.single().trials.sumOf { it.epochs } > 0)
        assertEquals(1, cancelledDriver.closes); assertTrue(cancelledDriver.memory.isEmpty())
    }

    @Test fun originalNativeFailureAndObserverFailureReleaseEveryAdmittedModel() {
        val original = IllegalStateException("primary transfer failure", InterruptedException("native cause"))
        val driver = SearchCudaTestDriver("failed-queue").apply { beforeLaunch = { throw original } }
        val report = NeuroArchitectureSearch(openCudaService = { format, batch -> SearchCudaService(format, batch, { driver }) })
            .searchManifest(data(), config(), listOf(NetworkArchitecture(listOf(4))))
        assertTrue(report.candidates.single().trials.all { it.state == ArchitectureTrialState.FAILED })
        assertTrue(report.candidates.single().trials.any { it.failure == original.message })
        assertNull(report.selection.recommended)
        assertTrue(driver.memory.isEmpty()); assertEquals(1, driver.closes)

        val releaseLaunch = CountDownLatch(1)
        val observerDriver = SearchCudaTestDriver("observer-queue").apply {
            beforeLaunch = { check(releaseLaunch.await(5, TimeUnit.SECONDS)) { "Observer failed to release native work" } }
        }
        val observer = IllegalStateException("progress observer failed")
        val failure = assertThrows(IllegalStateException::class.java) {
            NeuroArchitectureSearch(openCudaService = { format, batch -> SearchCudaService(format, batch, { observerDriver }) })
                .searchManifest(data(), config(epochs = 1000), listOf(NetworkArchitecture(listOf(4))),
                    { if (it.running.isNotEmpty()) { releaseLaunch.countDown(); throw observer } })
        }
        assertSame(observer, failure)
        assertTrue(observerDriver.memory.isEmpty())
        assertTrue(Thread.getAllStackTraces().keys.none { it.isAlive && it.name.startsWith("jneuro-search-") })
    }
}
