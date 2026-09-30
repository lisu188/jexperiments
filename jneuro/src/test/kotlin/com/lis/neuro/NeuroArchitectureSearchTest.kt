package com.lis.neuro

import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

class NeuroArchitectureSearchTest {
    @Test fun searchLogsCorrelateParallelTrialsAndReplayProvenance() {
        NeuroApplicationLogCapture().use { capture ->
            val sessions = RecordingTrainingSessions()
            val config = ArchitectureSearchConfig(strategy = ArchitectureSearchStrategy.EXHAUSTIVE,
                maxLayers = 1, maxWidth = 1, seeds = listOf(42, 43), requiredSuccesses = 1,
                maxEpochs = 3, checkEvery = 1, parallelism = 2, backend = TrainingBackend.CUDA)
            val report = NeuroArchitectureSearch(sessions::open).search(ArchitectureSearchData.fitting(xor()), config)
            val candidate = report.candidates.single()
            val trial = candidate.trials.first()
            assertEquals(report.logId, capture.fields(capture.events("search.started").single())["searchId"])
            assertEquals(report.logId, capture.fields(capture.events("search.completed").single())["searchId"])
            val started = capture.events("search.trial.started")
            val finished = capture.events("search.trial.finished")
            assertEquals(2, started.size)
            assertEquals(2, finished.size)
            assertEquals(candidate.trials.map { it.logId }.toSet(), finished.map { capture.fields(it)["trialId"] }.toSet())
            assertTrue((started + finished).all { capture.fields(it)["searchId"] == report.logId })
            NeuroStudio(openSession = sessions::open).use { studio ->
                sessions.identity = "changed-device"
                assertThrows(IllegalStateException::class.java) { studio.replayArchitecture(report, candidate, trial) }
                assertEquals(1, capture.events("studio.replay.provenance.rejected").size)
                assertTrue(capture.events("studio.replay.completed").isEmpty())
                sessions.identity = trial.deviceInfo!!.identity
                assertTrue(studio.replayArchitecture(report, candidate, trial))
                assertEquals(trial.logId, capture.fields(capture.events("studio.replay.completed").single())["trialId"])
                assertEquals(1, capture.events("studio.replay.provenance.accepted").size)
            }
        }
    }

    @Test fun failedTrialLogsOriginalExceptionAndNeverReportsSuccessfulCompletion() {
        NeuroApplicationLogCapture().use { capture ->
            val failure = IllegalStateException("fixture backend startup failure")
            val config = ArchitectureSearchConfig(strategy = ArchitectureSearchStrategy.EXHAUSTIVE,
                maxLayers = 1, maxWidth = 1, seeds = listOf(42), requiredSuccesses = 1, maxEpochs = 1, checkEvery = 1)
            val report = NeuroArchitectureSearch { _, _, _, _ -> throw failure }
                .search(ArchitectureSearchData.fitting(xor()), config)
            val trial = report.candidates.single().trials.single()
            assertEquals(ArchitectureTrialState.FAILED, trial.state)
            assertSame(failure, capture.events("search.trial.failed").single().thrown)
            val terminal = capture.events("search.trial.finished").single()
            assertEquals("FAILED", capture.fields(terminal)["state"].toString())
        }
    }

    @Test fun adaptiveLoggingMatchesTheAdmittedProposalAndCompletedCandidate() {
        NeuroApplicationLogCapture().use { capture ->
            val config = ArchitectureSearchConfig(maxLayers = 1, maxWidth = 1, seeds = listOf(42),
                requiredSuccesses = 1, maxTrials = 1, maxEpochs = 1, checkEvery = 1)
            val report = NeuroArchitectureSearch().search(ArchitectureSearchData.fitting(xor()), config)
            val proposal = capture.fields(capture.events("search.proposal.admitted").single())
            assertEquals(report.logId, proposal["searchId"])
            assertEquals(report.lineage.single().architecture.toString(), proposal["topology"].toString())
            assertEquals(report.lineage.single().mutation, proposal["mutation"])
            val candidate = capture.fields(capture.events("search.candidate.evaluated").single())
            assertEquals(report.logId, candidate["searchId"])
            assertEquals(report.candidates.single().medianRmse, candidate["medianRmse"])
        }
    }

    @Test fun trialCleanupFailureProducesOnlyOneFailedTerminalEvent() {
        NeuroApplicationLogCapture().use { capture ->
            val sessions = RecordingTrainingSessions().apply { failClose = true }
            val config = ArchitectureSearchConfig(strategy = ArchitectureSearchStrategy.EXHAUSTIVE,
                maxLayers = 1, maxWidth = 1, seeds = listOf(42), requiredSuccesses = 1, maxEpochs = 1, checkEvery = 1)
            val report = NeuroArchitectureSearch(sessions::open).search(ArchitectureSearchData.fitting(xor()), config)
            assertEquals(ArchitectureTrialState.FAILED, report.candidates.single().trials.single().state)
            assertEquals("FAILED", capture.fields(capture.events("search.trial.finished").single())["state"].toString())
            assertEquals(1, capture.events("search.trial.failed").size)
        }
    }

    @Test fun configuredBatchSearchReplaysTheResolvedBackendAndPrecision() {
        for (backend in listOf(TrainingBackend.AUTO, TrainingBackend.CUBLAS)) {
            val sessions = RecordingTrainingSessions()
            val config = ArchitectureSearchConfig(strategy = ArchitectureSearchStrategy.EXHAUSTIVE,
                maxLayers = 1, maxWidth = 1, seeds = listOf(42), requiredSuccesses = 1,
                maxEpochs = 4, checkEvery = 1, backend = backend,
                precision = Neuro.TrainingPrecision.FP32, batchSize = 3)
            val report = NeuroArchitectureSearch(sessions::open).search(ArchitectureSearchData.fitting(xor()), config)
            val candidate = report.candidates.single()
            val trial = candidate.trials.single()
            assertTrue(candidate.valid, trial.failure)
            assertEquals(4, sessions.miniBatches.size)
            assertEquals(Triple(backend, Neuro.TrainingPrecision.FP32, 3), sessions.configurations.single())
            val effective = if (backend == TrainingBackend.AUTO) TrainingBackend.CPU else backend
            val precision = if (effective == TrainingBackend.CPU) Neuro.TrainingPrecision.FP64 else Neuro.TrainingPrecision.FP32
            assertEquals(effective, trial.deviceInfo!!.backend)
            assertEquals(precision.name, trial.deviceInfo.precision)
            NeuroStudio(openSession = sessions::open).use { studio ->
                assertTrue(studio.replayArchitecture(report, candidate, trial))
                assertEquals(effective, studio.activeConfig.backend)
                assertEquals(precision, studio.activeConfig.precision)
                assertEquals(3, studio.activeConfig.batchSize)
                assertEquals(Triple(effective, precision, 3), sessions.configurations.last())
                assertEquals(trial.deviceInfo, studio.frame().deviceInfo)
                assertArrayEquals(trial.snapshot!!.parameters(), studio.frame().diagnostics.parameters(), 1e-10)
            }
            assertEquals(sessions.opened.get(), sessions.closed.get())
            assertTrue(sessions.miniBatches.all { it == 3 })
        }
        assertThrows(IllegalArgumentException::class.java) { ArchitectureSearchConfig(batchSize = 0) }
        assertThrows(IllegalArgumentException::class.java) { ArchitectureSearchConfig(precision = Neuro.TrainingPrecision.FP32) }
    }

    @Test fun selectedBackendIsRecordedAndReplayRequiresTheSameDevice() {
        val sessions = RecordingTrainingSessions()
        val config = ArchitectureSearchConfig(strategy = ArchitectureSearchStrategy.EXHAUSTIVE,
            maxLayers = 1, maxWidth = 1, seeds = listOf(42), requiredSuccesses = 1,
            maxEpochs = 2, checkEvery = 1, backend = TrainingBackend.CUDA)
        val report = NeuroArchitectureSearch(sessions::open).search(ArchitectureSearchData.fitting(xor()), config)
        val candidate = report.candidates.single()
        val trial = candidate.trials.single()
        assertEquals(ArchitectureTrialState.COMPLETED, trial.state)
        assertEquals(TrainingBackend.CUDA, trial.deviceInfo!!.backend)
        assertEquals(1, sessions.closed.get())
        NeuroStudio(openSession = sessions::open).use { studio ->
            assertTrue(studio.replayArchitecture(report, candidate, trial))
            assertEquals(trial.deviceInfo, studio.frame().deviceInfo)
            assertEquals(TrainingBackend.CUDA, studio.activeConfig.backend)
            val replay = studio.frame()
            sessions.identity = "replacement-device"
            assertThrows(IllegalStateException::class.java) { studio.replayArchitecture(report, candidate, trial) }
            assertEquals(replay.config, studio.activeConfig)
            assertArrayEquals(replay.diagnostics.parameters(), studio.frame().diagnostics.parameters())
            sessions.unavailable = true
            assertThrows(IllegalStateException::class.java) { studio.replayArchitecture(report, candidate, trial) }
        }
        assertEquals(sessions.opened.get(), sessions.closed.get())
        assertTrue(sessions.requested.all { it == TrainingBackend.CUDA })
    }

    @Test fun cleanupFailureCountsCommittedEpochButKeepsFailedTrialDisqualified() {
        val sessions = RecordingTrainingSessions().apply { failAfterCommit = true }
        val data = ArchitectureSearchData.fitting(xor())
        val config = ArchitectureSearchConfig(maxEpochs = 4, checkEvery = 1, targetRmse = 1.0,
            seeds = listOf(42), requiredSuccesses = 1, backend = TrainingBackend.CUBLAS, batchSize = 3)
        val architecture = NetworkArchitecture(listOf(2))
        val initialModel = data.newNetwork(architecture, config.hyperParameters, 42)
        val initial = NeuroXorDiagnostics.capture(initialModel, 0, initialModel.trainingError())
        val trial = NeuroArchitectureSearch(sessions::open).evaluate(data, config, architecture, 42, { false }, { _, _ -> })
        assertEquals(ArchitectureTrialState.FAILED, trial.state)
        assertTrue(trial.failure.contains("cleanup after committed epoch"))
        assertEquals(1, trial.epochs)
        assertEquals(4L, trial.sampleUpdates)
        assertEquals(0, trial.bestEpoch)
        assertEquals(listOf(0), trial.history.map { it.epoch })
        assertEquals(initial.error(), trial.bestRmse, 0.0)
        assertEquals(initial.error(), trial.finalRmse, 0.0)
        assertArrayEquals(initial.parameters(), trial.snapshot!!.parameters(), 0.0)
        val candidate = ArchitectureCandidate(architecture, listOf(trial), 1, config.targetRmse)
        assertFalse(candidate.valid)
        assertNull(ArchitectureRanking.select(listOf(candidate), config).recommended)
        assertEquals(1, sessions.closed.get())
    }

    @Test fun backendFailuresRemainFailedTrialsAndReleaseResources() {
        val sessions = RecordingTrainingSessions().apply { unavailable = true }
        val data = ArchitectureSearchData.fitting(xor())
        val config = ArchitectureSearchConfig(maxEpochs = 2, checkEvery = 1, backend = TrainingBackend.CUDA)
        val engine = NeuroArchitectureSearch(sessions::open)
        val missing = engine.evaluate(data, config, NetworkArchitecture(listOf(2)), 42, { false }, { _, _ -> })
        assertEquals(ArchitectureTrialState.FAILED, missing.state)
        assertEquals("CUDA fixture unavailable", missing.failure)
        assertNull(missing.deviceInfo)
        assertEquals(0, missing.epochs)
        sessions.unavailable = false; sessions.failEpoch = true
        val failed = engine.evaluate(data, config, NetworkArchitecture(listOf(2)), 42, { false }, { _, _ -> })
        assertEquals(ArchitectureTrialState.FAILED, failed.state)
        assertEquals(TrainingBackend.CUDA, failed.deviceInfo!!.backend)
        assertEquals("CUDA fixture epoch failed", failed.failure)
        assertEquals(1, sessions.closed.get())
        assertTrue(sessions.requested.all { it == TrainingBackend.CUDA })
    }

    @Test fun enumeratesUniqueParameterOrderedArchitecturesWithoutDiscardingPermutations() {
        val architectures = ArchitectureSearchConfig(strategy = ArchitectureSearchStrategy.EXHAUSTIVE).architectures()
        assertEquals(584, architectures.size)
        assertEquals(architectures.size, architectures.toSet().size)
        assertTrue(architectures.zipWithNext().all { (a, b) -> a.parameters <= b.parameters })
        assertTrue(NetworkArchitecture(listOf(4, 2)) in architectures)
        assertTrue(NetworkArchitecture(listOf(2, 4)) in architectures)
        for (hidden in listOf(listOf(1), listOf(2), listOf(3), listOf(2, 2), listOf(6))) {
            val architecture = NetworkArchitecture(hidden)
            assertEquals(Neuro(architecture.topology()).parameterCount(), architecture.parameters)
        }
        assertEquals(listOf(5, 9, 13, 15, 25), listOf(listOf(1), listOf(2), listOf(3), listOf(2, 2), listOf(6)).map { NetworkArchitecture(it).parameters })
        val hidden = mutableListOf(2, 2)
        val frozen = NetworkArchitecture(hidden); hidden[0] = 100
        val shape = frozen.topology(); shape[1] = 100
        assertEquals(NetworkArchitecture(listOf(2, 2)), frozen)
        assertNotEquals(frozen, "2,2")
        assertNotEquals(frozen, NetworkArchitecture(listOf(2)))
        assertThrows(UnsupportedOperationException::class.java) { (frozen.hidden as MutableList)[0] = 99 }
        val bounded = ArchitectureSearchConfig(strategy = ArchitectureSearchStrategy.EXHAUSTIVE, maxParameters = 9).architectures()
        assertTrue(bounded.all { it.parameters <= 9 })
        assertTrue(bounded.any { it.hidden == listOf(1, 1) })
    }

    @Test fun rejectsUnboundedInvalidAndEmptySearchSpaces() {
        for (factory in listOf<() -> ArchitectureSearchConfig>(
            { ArchitectureSearchConfig(strategy = ArchitectureSearchStrategy.EXHAUSTIVE, minLayers = 0) }, { ArchitectureSearchConfig(strategy = ArchitectureSearchStrategy.EXHAUSTIVE, maxLayers = -1) },
            { ArchitectureSearchConfig(strategy = ArchitectureSearchStrategy.EXHAUSTIVE, minWidth = 0) }, { ArchitectureSearchConfig(strategy = ArchitectureSearchStrategy.EXHAUSTIVE, maxWidth = -1) },
            { ArchitectureSearchConfig(strategy = ArchitectureSearchStrategy.EXHAUSTIVE, minWidth = 5, maxWidth = 4) }, { ArchitectureSearchConfig(strategy = ArchitectureSearchStrategy.EXHAUSTIVE, maxParameters = 0) },
            { ArchitectureSearchConfig(strategy = ArchitectureSearchStrategy.EXHAUSTIVE, seeds = emptyList()) }, { ArchitectureSearchConfig(strategy = ArchitectureSearchStrategy.EXHAUSTIVE, seeds = listOf(1, 1)) },
            { ArchitectureSearchConfig(strategy = ArchitectureSearchStrategy.EXHAUSTIVE, seeds = listOf(2L, 2L)) }, { ArchitectureSearchConfig(strategy = ArchitectureSearchStrategy.EXHAUSTIVE, maxEpochs = 0) },
            { ArchitectureSearchConfig(strategy = ArchitectureSearchStrategy.EXHAUSTIVE, checkEvery = 0) }, { ArchitectureSearchConfig(strategy = ArchitectureSearchStrategy.EXHAUSTIVE, checkEvery = 10_001) },
            { ArchitectureSearchConfig(strategy = ArchitectureSearchStrategy.EXHAUSTIVE, targetRmse = Double.NaN) }, { ArchitectureSearchConfig(strategy = ArchitectureSearchStrategy.EXHAUSTIVE, targetRmse = -1.0) },
            { ArchitectureSearchConfig(strategy = ArchitectureSearchStrategy.EXHAUSTIVE, requiredSuccesses = 0) }, { ArchitectureSearchConfig(strategy = ArchitectureSearchStrategy.EXHAUSTIVE, requiredSuccesses = 6) },
            { ArchitectureSearchConfig(strategy = ArchitectureSearchStrategy.EXHAUSTIVE, nearBestTolerance = -0.1) }, { ArchitectureSearchConfig(strategy = ArchitectureSearchStrategy.EXHAUSTIVE, nearBestTolerance = Double.NaN) },
            { ArchitectureSearchConfig(strategy = ArchitectureSearchStrategy.EXHAUSTIVE, parallelism = 0) }, { ArchitectureSearchConfig(strategy = ArchitectureSearchStrategy.EXHAUSTIVE, maxTrials = 4) },
            { ArchitectureSearchConfig(strategy = ArchitectureSearchStrategy.EXHAUSTIVE, maxTrials = 0) }, { ArchitectureSearchConfig(strategy = ArchitectureSearchStrategy.EXHAUSTIVE, timeLimitSeconds = -1) })) {
            assertThrows(IllegalArgumentException::class.java) { factory() }
        }
        assertThrows(IllegalArgumentException::class.java) { ArchitectureSearchConfig(strategy = ArchitectureSearchStrategy.EXHAUSTIVE, maxParameters = 4).architectures() }
        assertThrows(IllegalArgumentException::class.java) { ArchitectureSearchConfig(strategy = ArchitectureSearchStrategy.EXHAUSTIVE, maxLayers = 4, maxWidth = 12, maxParameters = 100_000).architectures() }
        assertThrows(IllegalArgumentException::class.java) {
            ArchitectureSearchConfig(strategy = ArchitectureSearchStrategy.EXHAUSTIVE, minLayers = 2, maxLayers = 2, minWidth = 32, maxWidth = 80,
                maxParameters = 10_000, seeds = (1L..20L).toList(), maxTrials = 20_000).architectures()
        }
        val input = mutableListOf(1L, 2L)
        val config = ArchitectureSearchConfig(strategy = ArchitectureSearchStrategy.EXHAUSTIVE, seeds = input, requiredSuccesses = 1)
        input.clear()
        assertEquals(listOf(1L, 2L), config.seeds)
    }

    @Test fun freezesDataAndBuildsReproducibleDisjointGroupedValidationPartitions() {
        val samples = NeuroLearningSets.create(NeuroLearningSets.Kind.CIRCLE, 2)
        val source = (samples + samples.take(10)).toMutableList()
        val first = ArchitectureSearchData.split(source, 0.2, 42, "Circle")
        val second = ArchitectureSearchData.split(source, 0.2, 42, "Circle")
        val different = ArchitectureSearchData.split(source, 0.2, 17, "Circle")
        assertEquals(first.training, second.training); assertEquals(first.validation, second.validation)
        assertEquals(first.fingerprint, second.fingerprint); assertNotEquals(first.fingerprint, different.fingerprint)
        assertEquals(64, first.fingerprint.length); assertEquals(42L, first.splitSeed); assertEquals(0.2, first.validationFraction)
        assertEquals(ArchitectureEvaluation.VALIDATION, first.evaluation); assertEquals("Circle", first.label)
        assertTrue(first.training.map { it.x to it.y }.intersect(first.validation.map { it.x to it.y }.toSet()).isEmpty())
        assertEquals(source.size, first.training.size + first.validation.size)
        source.clear(); assertEquals(190, first.training.size + first.validation.size)
        val fitting = ArchitectureSearchData.fitting(samples)
        assertEquals(ArchitectureEvaluation.TRAINING_FIT, fitting.evaluation); assertTrue(fitting.validation.isEmpty())
        assertNotEquals(fitting.fingerprint, first.fingerprint)
        assertThrows(UnsupportedOperationException::class.java) { (first.training as MutableList).clear() }
        val model = first.newNetwork(NetworkArchitecture(listOf(2)), Neuro.HyperParameters.defaults(), 42)
        assertEquals(first.training.size, model.trainingSampleCount()); assertEquals(first.validation.size, model.testSampleCount())
        assertEquals(model.testError(), first.score(model))
        assertThrows(IllegalArgumentException::class.java) { ArchitectureSearchData.fitting(emptyList()) }
        assertThrows(IllegalArgumentException::class.java) { ArchitectureSearchData.split(samples, 0.0) }
        assertThrows(IllegalArgumentException::class.java) { ArchitectureSearchData.split(xor()) }
        val rare = (0..10).map { NeuroLearningSets.Sample(it / 11.0, 0.0, if (it == 0) 1.0 else 0.0) }
        assertThrows(IllegalArgumentException::class.java) { ArchitectureSearchData.split(rare) }
        assertEquals("Training RMSE", ArchitectureEvaluation.TRAINING_FIT.toString())
        assertEquals("Lowest RMSE", ArchitecturePolicy.LOWEST_RMSE.toString())
    }

    @Test fun reliabilityRejectsLuckySeedsAndPoliciesRetainErrorSizeTradeoffs() {
        val lucky = candidate(listOf(1), listOf(0.001, 0.45, 0.5, 0.55, 0.6))
        val reliable = candidate(listOf(2), listOf(0.04, 0.045, 0.047, 0.044, 0.043))
        val accurate = candidate(listOf(4), List(5) { 0.01 })
        val dominated = candidate(listOf(3), List(5) { 0.049 })
        val config = ArchitectureSearchConfig(strategy = ArchitectureSearchStrategy.EXHAUSTIVE)
        val entries = listOf(accurate, lucky, dominated, reliable)
        val selection = ArchitectureRanking.select(entries, config)
        assertSame(reliable, selection.recommended); assertSame(reliable, selection.smallestMeetingTarget)
        assertSame(accurate, selection.bestError)
        assertEquals(listOf(lucky, reliable, accurate), selection.paretoFrontier)
        assertEquals(listOf(reliable, accurate), selection.reliableFrontier)
        assertEquals(1, lucky.successes); assertEquals(0.5, lucky.medianRmse); assertEquals(0.001, lucky.bestRmse); assertEquals(0.6, lucky.worstRmse)
        assertEquals(0.044, reliable.medianRmse); assertEquals(0.044, reliable.representative!!.bestRmse)
        assertSame(accurate, ArchitectureRanking.select(entries, config, ArchitecturePolicy.LOWEST_RMSE).recommended)
        val near = candidate(listOf(2), List(5) { 0.014 })
        assertSame(near, ArchitectureRanking.select(listOf(near, accurate), config, ArchitecturePolicy.SMALLEST_NEAR_BEST).recommended)
        val none = ArchitectureRanking.select(listOf(lucky), config)
        assertNull(none.recommended); assertSame(lucky, none.bestError)
        assertNull(ArchitectureRanking.select(emptyList(), config, ArchitecturePolicy.SMALLEST_NEAR_BEST).recommended)
    }

    @Test fun handlesTiesEvenSeedCountsFailuresAndPartialArchitecturesWithoutOptimism() {
        val sameSize = candidate(listOf(1, 3), List(5) { 0.04 })
        val shallow = candidate(listOf(3), List(5) { 0.04 })
        assertEquals(sameSize.architecture.parameters, shallow.architecture.parameters)
        val config = ArchitectureSearchConfig(strategy = ArchitectureSearchStrategy.EXHAUSTIVE)
        assertSame(shallow, ArchitectureRanking.select(listOf(sameSize, shallow), config).recommended)
        assertEquals(2, ArchitectureRanking.frontier(listOf(sameSize, shallow)).size)
        val even = ArchitectureCandidate(NetworkArchitecture(listOf(2)), listOf(trial(1, 0.02), trial(2, 0.04)), 2, 0.05)
        assertEquals(0.03, even.medianRmse, 1e-15)
        val failed = ArchitectureCandidate(NetworkArchitecture(listOf(2)), listOf(trial(1, 0.01), trial(2, 0.001, ArchitectureTrialState.FAILED)), 2, 0.05)
        assertFalse(failed.valid); assertTrue(failed.fullyEvaluated); assertEquals(Double.POSITIVE_INFINITY, failed.medianRmse)
        assertEquals(Double.POSITIVE_INFINITY, failed.worstRmse)
        val partial = ArchitectureCandidate(NetworkArchitecture(listOf(1)), listOf(trial(1, 0.0001)), 5, 0.05)
        val cancelled = ArchitectureCandidate(NetworkArchitecture(listOf(1)), listOf(trial(1, 0.01, ArchitectureTrialState.CANCELLED)), 1, 0.05)
        assertFalse(partial.valid); assertFalse(cancelled.fullyEvaluated); assertNull(partial.representative)
        assertEquals(listOf(even), ArchitectureRanking.frontier(listOf(even, failed, partial, cancelled)))
        val empty = ArchitectureCandidate(NetworkArchitecture(listOf(1)), emptyList(), 5, 0.05)
        assertEquals(Double.POSITIVE_INFINITY, empty.bestRmse)
    }

    @Test fun realXorSearchUsesFullBudgetsAndFindsSmallReliableNetworks() {
        val data = ArchitectureSearchData.fitting(xor(), "XOR")
        val config = ArchitectureSearchConfig(strategy = ArchitectureSearchStrategy.EXHAUSTIVE, maxLayers = 1, maxWidth = 4, parallelism = 2)
        val progress = ArrayList<ArchitectureSearchProgress>()
        val report = NeuroArchitectureSearch().search(data, config, { progress += it }, { false })
        assertEquals(ArchitectureTermination.COMPLETED, report.termination)
        assertEquals(4, report.generated); assertEquals(4, report.evaluated); assertEquals(0, report.partial); assertEquals(0, report.untested)
        assertEquals(0, report.numericalFailures)
        assertTrue(report.environment.contains("Kotlin")); assertTrue(report.elapsedNanos > 0)
        assertEquals(0, progress.first().finishedTrials); assertTrue(progress.first().candidates.isEmpty())
        assertEquals(20, progress.last().finishedTrials); assertEquals(4, progress.last().fullyEvaluated)
        assertNotNull(report.selection.recommended)
        for (entry in report.candidates) {
            assertTrue(entry.valid)
            for (trial in entry.trials) {
                assertEquals(10_000, trial.epochs)
                assertEquals(40_000, trial.sampleUpdates)
                assertEquals(trial.bestEpoch, trial.snapshot!!.epoch())
                assertEquals(trial.trainingRmseAtBest, trial.snapshot.error())
                assertTrue(trial.history.size <= 128)
                assertEquals(0, trial.history.first().epoch); assertEquals(10_000, trial.history.last().epoch)
                assertEquals(trial.finalRmse, trial.history.last().score)
                assertTrue(trial.history.any { it.epoch == trial.bestEpoch })
                assertTrue(trial.elapsedNanos >= 0)
            }
            println("SEARCH_XOR ${entry.architecture} params=${entry.architecture.parameters} median=${entry.medianRmse} success=${entry.successes}/5")
        }
        assertTrue(report.candidates.first().medianRmse > 0.4)
        assertTrue(report.selection.recommended!!.meetsTarget(4))
    }

    @Test fun sequentialAndParallelSearchesUseIdenticalTrialsAndDeterministicBudgets() {
        val data = ArchitectureSearchData.fitting(xor())
        fun run(parallelism: Int) = NeuroArchitectureSearch().search(data, ArchitectureSearchConfig(strategy = ArchitectureSearchStrategy.EXHAUSTIVE, maxLayers = 2,
            maxWidth = 3, maxEpochs = 100, parallelism = parallelism, maxTrials = 12))
        val sequential = run(1); val parallel = run(3)
        assertEquals(ArchitectureTermination.TRIAL_BUDGET, sequential.termination)
        assertEquals(2, sequential.evaluated); assertEquals(10, sequential.untested)
        assertEquals(sequential.candidates.map { it.architecture }, parallel.candidates.map { it.architecture })
        for ((a, b) in sequential.candidates.zip(parallel.candidates)) {
            assertEquals(a.medianRmse, b.medianRmse, 1e-12)
            for ((t, u) in a.trials.zip(b.trials)) assertArrayEquals(t.snapshot!!.parameters(), u.snapshot!!.parameters(), 1e-12)
        }
        assertNull(sequential.selection.recommended)
    }

    @Test fun cancellationDeadlineAndCallbackFailureCleanUpWorkers() {
        val engine = NeuroArchitectureSearch(); val data = ArchitectureSearchData.fitting(xor())
        val config = ArchitectureSearchConfig(strategy = ArchitectureSearchStrategy.EXHAUSTIVE, maxLayers = 1, maxWidth = 3, maxEpochs = 1_000_000)
        val cancelled = engine.search(data, config, {}, { true })
        assertEquals(ArchitectureTermination.CANCELLED, cancelled.termination); assertEquals(0, cancelled.evaluated)
        val checks = AtomicInteger()
        val partial = engine.search(data, config, {}, { checks.incrementAndGet() > 2 })
        assertEquals(ArchitectureTermination.CANCELLED, partial.termination)
        assertTrue(partial.candidates.any { candidate -> candidate.trials.any { it.state == ArchitectureTrialState.CANCELLED } })
        assertNull(partial.selection.recommended)
        val deadline = engine.search(data, ArchitectureSearchConfig(strategy = ArchitectureSearchStrategy.EXHAUSTIVE, maxLayers = 1, maxWidth = 1, timeLimitSeconds = 1),
            { progress -> if (progress.finishedTrials == 0) Thread.sleep(1050) }, { false })
        assertEquals(ArchitectureTermination.TIME_LIMIT, deadline.termination)
        assertThrows(IllegalStateException::class.java) { engine.search(data, config, { error("Observer failed") }, { false }) }
        assertFalse(Thread.getAllStackTraces().keys.any { it.isAlive && it.name.startsWith("jneuro-search-") })
    }

    @Test fun individualFailuresAndCancelledEpochsAreExplicit() {
        val engine = NeuroArchitectureSearch(); val data = ArchitectureSearchData.fitting(xor())
        val config = ArchitectureSearchConfig(strategy = ArchitectureSearchStrategy.EXHAUSTIVE, maxEpochs = 100)
        val architecture = NetworkArchitecture(listOf(2))
        val failed = engine.evaluate(data, config, architecture, 42, { false }, { _, _ -> throw IllegalStateException("Numeric failure") })
        assertEquals(ArchitectureTrialState.FAILED, failed.state); assertEquals("Numeric failure", failed.failure)
        val noMessage = engine.evaluate(data, config, architecture, 42, { false }, { _, _ -> throw IllegalArgumentException() })
        assertEquals("IllegalArgumentException", noMessage.failure)
        val stop = AtomicBoolean()
        val cancelled = engine.evaluate(data, config, architecture, 42, { stop.get() }, { epoch, _ -> if (epoch == 25) stop.set(true) })
        assertEquals(25, cancelled.epochs); assertEquals(ArchitectureTrialState.CANCELLED, cancelled.state)
        val initial = engine.evaluate(data, config, architecture, 42, { true }, { _, _ -> })
        assertEquals(0, initial.epochs); assertNull(initial.snapshot)
        try {
            val interrupted = engine.evaluate(data, config, architecture, 42, { false }, { _, _ -> throw InterruptedException() })
            assertEquals(ArchitectureTrialState.CANCELLED, interrupted.state); assertTrue(Thread.currentThread().isInterrupted)
        } finally { Thread.interrupted() }
    }

    @Test fun sessionCancellationRetainsResultsButInvalidationRejectsStaleCallbacks() {
        val session = ArchitectureSearchSession()
        session.cancel(); session.invalidate()
        val first = session.begin(); assertTrue(session.isCurrent(first))
        session.cancel(); assertTrue(first.cancelled.get()); assertTrue(session.isCurrent(first))
        val second = session.begin(); assertFalse(session.isCurrent(first)); assertTrue(second.id > first.id)
        session.invalidate(); assertTrue(second.cancelled.get()); assertFalse(session.isCurrent(second))
    }

    @Test fun validationReplayRecreatesScoredPartitionAndApplyingStartsAnUntrainedFullDatasetModel() {
        val studio = NeuroStudio(StudioConfig(dataset = NeuroLearningSets.Kind.CIRCLE))
        val data = studio.searchData(ArchitectureEvaluation.VALIDATION, 0.2, 42)
        val report = NeuroArchitectureSearch().search(data, ArchitectureSearchConfig(strategy = ArchitectureSearchStrategy.EXHAUSTIVE, maxLayers = 1, minWidth = 2, maxWidth = 2, maxEpochs = 125))
        val candidate = report.candidates.single(); val trial = candidate.representative!!
        assertTrue(candidate.valid)
        val original = studio.frame()
        assertFalse(studio.replayArchitecture(report, candidate, trial) { true })
        assertEquals(original.config, studio.activeConfig); assertEquals(0, studio.epochs)
        var checks = 0
        assertFalse(studio.replayArchitecture(report, candidate, trial) { ++checks > 2 })
        assertEquals(0, studio.epochs)
        assertTrue(studio.replayArchitecture(report, candidate, trial))
        val replay = studio.frame()
        assertEquals(trial.bestEpoch, studio.epochs); assertEquals(data.training, replay.samples)
        assertEquals(trial.seed, replay.config.seed); assertEquals(trial.trainingRmseAtBest, studio.currentError, 1e-12)
        assertTrue(replay.replayNote.contains("Validation RMSE"))
        assertArrayEquals(trial.snapshot!!.parameters(), replay.diagnostics.parameters(), 1e-12)
        studio.applyArchitecture(report, candidate)
        assertEquals(0, studio.epochs); assertEquals(180, studio.frame().samples.size); assertEquals("", studio.frame().replayNote)
        assertThrows(IllegalArgumentException::class.java) { studio.applyArchitecture(report, candidate(listOf(8), List(5) { 0.01 })) }
        assertThrows(IllegalArgumentException::class.java) { studio.replayArchitecture(report, candidate, trial(12345, 0.01)) }
    }

    private fun xor() = NeuroLearningSets.create(NeuroLearningSets.Kind.XOR, 42)
    private fun trial(seed: Long, error: Double, state: ArchitectureTrialState = ArchitectureTrialState.COMPLETED) =
        ArchitectureTrial(seed, state, 100, 100, error, error, error, 400, 1, listOf(ArchitectureCheckpoint(100, error, error)), null)
    private fun candidate(hidden: List<Int>, errors: List<Double>) = ArchitectureCandidate(NetworkArchitecture(hidden),
        errors.mapIndexed { index, error -> trial(index.toLong(), error) }, errors.size, 0.05)
}
