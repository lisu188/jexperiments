package com.lis.neuro

import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.Timeout

@Timeout(40)
class ParallelArchitectureSearchTest {
    private val data = ArchitectureSearchData.fitting(NeuroLearningSets.create(NeuroLearningSets.Kind.XOR, 42))

    @Test fun oneSeedActuallyOccupies32DistinctWorkerThreadsWithEliteOffspring() {
        val config = wideConfig(listOf(42), 33)
        val started = ConcurrentHashMap.newKeySet<String>()
        val release = CountDownLatch(1)
        val peakReported = AtomicInteger()
        val root = config.startingArchitecture()
        val report = AdaptiveTrialScheduler(data, config) { architecture, seed, _, progress ->
            if (architecture != root) {
                started += Thread.currentThread().name
                progress(1, 0.2)
                assertTrue(release.await(15, TimeUnit.SECONDS), "Scheduler did not occupy 32 slots with one seed")
            }
            trial(seed, config.maxEpochs)
        }.search({ progress ->
            assertTrue(progress.running.size <= 32)
            if (progress.running.size == 32 && started.size == 32) release.countDown()
            peakReported.accumulateAndGet(progress.running.size, ::maxOf)
            assertAncestry(progress.lineage, progress.candidates, config)
        }, { false })
        assertEquals(32, started.size)
        assertEquals(32, peakReported.get())
        assertEquals(32, report.peakParallelTrials)
        assertEquals(33, report.evaluated)
        assertEquals(ArchitectureTermination.TRIAL_BUDGET, report.termination)
        assertEquals(33, report.lineage.map { it.architecture }.distinct().size)
        assertTrue(report.candidates.all { it.trials.single().epochs == config.maxEpochs })
        assertWorkersStopped()
    }

    @Test fun fiveSeedsFill32SlotsWithoutExceedingTheBudgetOrMixingTrialResults() {
        val config = wideConfig(listOf(1, 42, 123, 999, 2026), 45)
        val release = CountDownLatch(1)
        val started = AtomicInteger()
        val root = config.startingArchitecture()
        val report = AdaptiveTrialScheduler(data, config) { architecture, seed, _, _ ->
            if (architecture != root && started.incrementAndGet() <= 32) {
                assertTrue(release.await(15, TimeUnit.SECONDS), "Seed count still caps parallelism")
            }
            trial(seed, config.maxEpochs)
        }.search({ progress ->
            if (progress.running.size == 32 && started.get() >= 32) release.countDown()
            assertTrue(progress.running.size <= config.parallelism)
        }, { false })
        assertEquals(32, report.peakParallelTrials)
        assertEquals(9, report.evaluated)
        assertEquals(45, report.candidates.sumOf { it.trials.size })
        assertTrue(report.candidates.all { it.trials.map { trial -> trial.seed } == config.seeds })
        assertAncestry(report.lineage, report.candidates, config)
        assertWorkersStopped()
    }

    @Test fun slowOffspringDoesNotBlockOtherArchitectures() {
        val config = ArchitectureSearchConfig(initialHidden = listOf(6), maxWidth = 16,
            parallelism = 2, seeds = listOf(42), requiredSuccesses = 1, maxTrials = 6)
        val root = config.startingArchitecture()
        val childNumber = AtomicInteger()
        val otherCompleted = CountDownLatch(1)
        val report = AdaptiveTrialScheduler(data, config) { architecture, seed, _, _ ->
            if (architecture != root && childNumber.incrementAndGet() == 1) {
                assertTrue(otherCompleted.await(10, TimeUnit.SECONDS), "Waiting for one architecture serialized the search")
            } else if (architecture != root) otherCompleted.countDown()
            trial(seed, 10)
        }.search({}, { false })
        assertEquals(6, report.evaluated)
        assertEquals(2, report.peakParallelTrials)
        assertWorkersStopped()
    }

    @Test fun onlyObservedCompleteElitesBreedWhileOtherOffspringArePending() {
        val config = ArchitectureSearchConfig(initialHidden = listOf(6), maxWidth = 16, policy = ArchitecturePolicy.LOWEST_RMSE)
        val planner = AdaptiveArchitecturePlanner(config)
        val root = planner.propose()!!
        assertNull(planner.propose(), "The initial model is not an elite before evaluation")
        planner.observe(candidate(root.architecture, 0.4))
        val first = planner.propose()!!
        val second = planner.propose()!!
        assertEquals(root.architecture, first.parent)
        assertEquals(root.architecture, second.parent)
        assertThrows(IllegalStateException::class.java) {
            planner.observe(ArchitectureCandidate(first.architecture, emptyList(), 5, 0.05))
        }
        planner.observe(candidate(second.architecture, 0.01))
        planner.observe(candidate(first.architecture, 0.9))
        val next = planner.propose()!!
        assertEquals(second.architecture, next.parent)
        assertEquals(3, next.evaluatedCount)
        assertThrows(IllegalStateException::class.java) { planner.observe(candidate(second.architecture, 0.02)) }
        assertFalse(next.parent == first.architecture, "Pending candidates must never breed")
    }

    @Test fun cancelledParallelSearchDrainsWorkersAndRetainsPartialGroups() {
        val config = wideConfig(listOf(42), 100)
        val stop = AtomicBoolean()
        val peak = AtomicInteger()
        val root = config.startingArchitecture()
        val report = AdaptiveTrialScheduler(data, config) { architecture, seed, cancelled, _ ->
            if (architecture != root) {
                while (!cancelled()) Thread.onSpinWait()
                trial(seed, 0, ArchitectureTrialState.CANCELLED)
            } else trial(seed, 10)
        }.search({ progress ->
            peak.accumulateAndGet(progress.running.size, ::maxOf)
            if (progress.running.size == 32) stop.set(true)
        }, { stop.get() })
        assertEquals(32, peak.get())
        assertEquals(ArchitectureTermination.CANCELLED, report.termination)
        assertEquals(1, report.evaluated)
        assertEquals(32, report.partial)
        assertNull(report.selection.recommended)
        assertWorkersStopped()
    }

    @Test fun observerFailureCancelsEveryOutstandingTrial() {
        val config = wideConfig(listOf(42), 33)
        val root = config.startingArchitecture()
        assertThrows(IllegalStateException::class.java) {
            AdaptiveTrialScheduler(data, config) { architecture, seed, cancelled, _ ->
                if (architecture != root) while (!cancelled()) Thread.onSpinWait()
                trial(seed, 0)
            }.search({ if (it.running.size >= 2) error("observer failed") }, { false })
        }
        assertWorkersStopped()
    }

    @Test fun aOneArchitectureBudgetReportsTheRealLowerConcurrency() {
        val config = wideConfig(listOf(42), 1)
        val result = NeuroArchitectureSearch().search(data, config)
        assertEquals(1, result.evaluated)
        assertEquals(1, result.peakParallelTrials)
        assertEquals(ArchitectureTermination.TRIAL_BUDGET, result.termination)
        assertWorkersStopped()
    }

    @Test fun exhaustiveModeAlsoReportsActualConcurrency() {
        val result = NeuroArchitectureSearch().search(data, ArchitectureSearchConfig(strategy = ArchitectureSearchStrategy.EXHAUSTIVE,
            maxLayers = 1, maxWidth = 2, maxEpochs = 50, checkEvery = 25, seeds = listOf(42), requiredSuccesses = 1, parallelism = 1))
        assertEquals(ArchitectureTermination.COMPLETED, result.termination)
        assertEquals(1, result.peakParallelTrials)
    }

    private fun wideConfig(seeds: List<Long>, trials: Int) = ArchitectureSearchConfig(initialHidden = listOf(4, 5, 6, 7, 8),
        maxLayers = 6, maxWidth = 16, maxParameters = 2048, seeds = seeds, requiredSuccesses = 1,
        parallelism = 32, maxTrials = trials, maxEpochs = 50, checkEvery = 25, maxRestarts = 0)

    private fun trial(seed: Long, epochs: Int, state: ArchitectureTrialState = ArchitectureTrialState.COMPLETED) =
        ArchitectureTrial(seed, state, epochs, epochs, 0.2, 0.2, 0.2, epochs * 4L, 1, emptyList(), null)

    private fun candidate(architecture: NetworkArchitecture, score: Double) = ArchitectureCandidate(architecture,
        List(5) { ArchitectureTrial(it.toLong(), ArchitectureTrialState.COMPLETED, 10, 10, score, score, score, 40, 1, emptyList(), null) }, 5, 0.05)

    private fun assertAncestry(lineage: List<ArchitectureProposal>, candidates: List<ArchitectureCandidate>, config: ArchitectureSearchConfig) {
        for (proposal in lineage) proposal.parent?.let { parent ->
            val observed = candidates.take(proposal.evaluatedCount)
            assertTrue(observed.all { it.fullyEvaluated })
            assertTrue(parent in EliteParentSelection.rank(ArchitectureRanking.select(observed, config), config))
        }
    }

    private fun assertWorkersStopped() {
        val workers = Thread.getAllStackTraces().keys.filter { it.isAlive && it.name.startsWith("jneuro-search-") }
        workers.forEach { it.join(2000) }
        assertFalse(workers.any { it.isAlive })
    }
}
