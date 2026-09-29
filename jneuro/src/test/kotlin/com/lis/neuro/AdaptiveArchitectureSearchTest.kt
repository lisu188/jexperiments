package com.lis.neuro

import java.util.concurrent.atomic.AtomicBoolean
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

class AdaptiveArchitectureSearchTest {
    @Test fun adaptiveIsDefaultAndInitialArchitectureIsCopiedBoundedAndNotEnumerated() {
        val hidden = mutableListOf(6)
        val config = ArchitectureSearchConfig(initialHidden = hidden)
        hidden[0] = 2
        assertEquals(ArchitectureSearchStrategy.ADAPTIVE, config.strategy)
        assertEquals(NetworkArchitecture(listOf(6)), config.startingArchitecture())
        assertEquals(10_000, config.plannedTrials())
        assertThrows(UnsupportedOperationException::class.java) { (config.initialHidden as MutableList)[0] = 3 }
        assertEquals(NetworkArchitecture(listOf(1)), ArchitectureSearchConfig(initialHidden = listOf(128)).startingArchitecture())
        assertEquals(NetworkArchitecture(listOf(1)), ArchitectureSearchConfig(initialHidden = emptyList()).startingArchitecture())
        assertEquals(NetworkArchitecture(listOf(1)), ArchitectureSearchConfig(initialHidden = listOf(8, 8, 8), maxParameters = 9).startingArchitecture())
        assertThrows(IllegalArgumentException::class.java) { ArchitectureSearchConfig(maxParameters = 4).plannedTrials() }
        assertThrows(IllegalArgumentException::class.java) { ArchitectureSearchConfig(restartAfter = 0) }
        assertThrows(IllegalArgumentException::class.java) { ArchitectureSearchConfig(maxRestarts = -1) }
        assertTrue(ArchitectureSearchStrategy.ADAPTIVE.toString().contains("evolve"))
        assertEquals(2920, ArchitectureSearchConfig(strategy = ArchitectureSearchStrategy.EXHAUSTIVE).plannedTrials())
        val huge = ArchitectureSearchConfig(maxLayers = 8, maxWidth = 128, maxParameters = 1_000_000, maxTrials = 5)
        assertEquals(5, huge.plannedTrials())
        assertThrows(IllegalArgumentException::class.java) { huge.architectures() }
    }

    @Test fun mutationsChangeWidthsDepthAndOrderWithoutLeavingConfiguredBounds() {
        val config = ArchitectureSearchConfig(maxWidth = 8)
        val planner = AdaptiveArchitecturePlanner(config)
        val parent = ArchitectureProposal(NetworkArchitecture(listOf(2, 3)), generation = 4)
        val mutations = planner.mutations(parent)
        val shapes = mutations.map { it.architecture.hidden }.toSet()
        for (shape in listOf(listOf(1, 3), listOf(3, 3), listOf(4, 3), listOf(2, 1), listOf(2, 6),
            listOf(3, 2), listOf(3), listOf(2), listOf(1, 2, 3), listOf(2, 3, 3))) assertTrue(shape in shapes, "$shape")
        assertEquals(mutations.size, shapes.size)
        assertTrue(mutations.all { it.parent == parent.architecture && it.generation == 5 && it.mutation.isNotBlank() })
        assertEquals(mutations, AdaptiveArchitecturePlanner(config).mutations(parent))
        assertNotEquals(mutations, AdaptiveArchitecturePlanner(ArchitectureSearchConfig(searchSeed = 19)).mutations(parent))
        val bounded = AdaptiveArchitecturePlanner(ArchitectureSearchConfig(maxLayers = 1, maxWidth = 3, maxParameters = 9))
        assertTrue(bounded.mutations(ArchitectureProposal(NetworkArchitecture(listOf(2)))).all { it.architecture.parameters <= 9 })
        assertTrue(bounded.mutations(ArchitectureProposal(NetworkArchitecture(listOf(1)))).all { it.architecture.hidden.size == 1 })
        assertTrue(AdaptiveArchitecturePlanner(ArchitectureSearchConfig(maxLayers = 1, maxWidth = 1))
            .mutations(ArchitectureProposal(NetworkArchitecture(listOf(1)))).isEmpty())
    }

    @Test fun everyNewLeaderImmediatelyBecomesParentInsteadOfFollowingAFixedList() {
        val planner = AdaptiveArchitecturePlanner(ArchitectureSearchConfig(initialHidden = listOf(3), maxRestarts = 0))
        val first = planner.next()!!
        assertNull(first.parent)
        assertThrows(IllegalStateException::class.java) { planner.next() }
        assertThrows(IllegalStateException::class.java) { planner.observe(candidate(NetworkArchitecture(listOf(2)), 0.4)) }
        assertThrows(IllegalStateException::class.java) {
            planner.observe(ArchitectureCandidate(first.architecture, emptyList(), 5, 0.05))
        }
        planner.observe(candidate(first.architecture, 0.4))
        val second = planner.next()!!
        assertEquals(first.architecture, second.parent)
        planner.observe(candidate(second.architecture, 0.2))
        val third = planner.next()!!
        assertEquals(second.architecture, third.parent)
        assertEquals(2, third.generation)
        planner.observe(candidate(third.architecture, 0.01))
        val fourth = planner.next()!!
        assertEquals(third.architecture, fourth.parent)
        assertEquals(3, fourth.generation)
        val snapshot = planner.lineage
        assertEquals(4, snapshot.size)
        assertThrows(UnsupportedOperationException::class.java) { (snapshot as MutableList).clear() }
    }

    @Test fun differentMeasuredResultsCauseDifferentNextArchitectures() {
        val config = ArchitectureSearchConfig(initialHidden = listOf(1), maxLayers = 1, maxWidth = 8, maxRestarts = 0)
        fun run(childError: Double): ArchitectureProposal {
            val planner = AdaptiveArchitecturePlanner(config)
            val first = planner.next()!!; planner.observe(candidate(first.architecture, 0.3))
            val child = planner.next()!!; planner.observe(candidate(child.architecture, childError))
            return planner.next() ?: ArchitectureProposal(NetworkArchitecture(listOf(128)))
        }
        assertNotEquals(run(0.1), run(0.6))
    }

    @Test fun failedAndDominatedCandidatesAreNotParentsAndPlateausTriggerBoundedRestarts() {
        val config = ArchitectureSearchConfig(maxLayers = 1, maxWidth = 8, initialHidden = listOf(1), restartAfter = 1, maxRestarts = 2)
        val planner = AdaptiveArchitecturePlanner(config)
        val first = planner.next()!!; planner.observe(candidate(first.architecture, 0.3))
        val child = planner.next()!!; planner.observe(candidate(child.architecture, 0.6))
        val restart = planner.next()!!
        assertNull(restart.parent); assertTrue(restart.mutation.contains("restart")); assertEquals(1, planner.restartCount)
        planner.observe(candidate(restart.architecture, 0.9))
        val next = planner.next()!!
        assertNotEquals(child.architecture, next.parent)
        assertNotEquals(restart.architecture, next.parent)
        val failed = AdaptiveArchitecturePlanner(ArchitectureSearchConfig(initialHidden = listOf(4), maxLayers = 1, maxWidth = 8, maxRestarts = 1))
        val initial = failed.next()!!
        failed.observe(candidate(initial.architecture, 0.001, ArchitectureTrialState.FAILED))
        val exploration = failed.next()!!
        assertNull(exploration.parent); assertEquals(listOf(1), exploration.architecture.hidden)
        failed.observe(candidate(exploration.architecture, 0.001, ArchitectureTrialState.FAILED))
        assertNull(failed.next())
    }

    @Test fun everyMutationComesFromAnEligibleParetoParentAndArchitecturesAreNeverRepeated() {
        val config = ArchitectureSearchConfig(maxLayers = 3, maxWidth = 5, maxRestarts = 0)
        val planner = AdaptiveArchitecturePlanner(config)
        val evaluated = ArrayList<ArchitectureCandidate>()
        val seen = HashSet<NetworkArchitecture>()
        repeat(45) { index ->
            val next = planner.next() ?: return@repeat
            assertTrue(seen.add(next.architecture))
            next.parent?.let { parent ->
                val selection = ArchitectureRanking.select(evaluated, config)
                assertTrue(parent in (selection.paretoFrontier + selection.reliableFrontier + listOfNotNull(selection.recommended)).map { it.architecture })
            }
            val error = if (index % 4 == 0) 0.5 else 1.0 / next.architecture.parameters
            val candidate = candidate(next.architecture, error)
            evaluated += candidate; planner.observe(candidate)
        }
        assertTrue(evaluated.size > 5)
        assertEquals(seen.size, planner.lineage.size)
    }

    @Test fun serialAndParallelAdaptiveRunsHaveTheSameLineageFullBudgetsAndRankings() {
        val data = ArchitectureSearchData.fitting(xor())
        fun run(workers: Int) = NeuroArchitectureSearch().search(data, ArchitectureSearchConfig(initialHidden = listOf(2, 2),
            maxLayers = 3, maxWidth = 8, maxTrials = 30, maxEpochs = 100, checkEvery = 25, parallelism = workers))
        val first = run(1); val parallel = run(4)
        assertEquals(ArchitectureTermination.TRIAL_BUDGET, first.termination)
        assertEquals(first.lineage, parallel.lineage)
        assertEquals(6, first.evaluated); assertEquals(0, first.untested)
        for ((left, right) in first.candidates.zip(parallel.candidates)) {
            assertEquals(left.architecture, right.architecture)
            assertEquals(left.medianRmse, right.medianRmse, 1e-12)
            assertEquals(5, left.trials.size)
            for ((a, b) in left.trials.zip(right.trials)) {
                assertEquals(100, a.epochs)
                assertArrayEquals(a.snapshot!!.parameters(), b.snapshot!!.parameters(), 1e-12)
            }
        }
    }

    @Test fun actualAdaptiveXorFindsTheSmallReliableArchitectureWithoutGlobalEnumeration() {
        val progress = ArrayList<ArchitectureSearchProgress>()
        val report = NeuroArchitectureSearch().search(ArchitectureSearchData.fitting(xor()),
            ArchitectureSearchConfig(maxLayers = 1, maxWidth = 4), { progress += it }, { false })
        assertEquals(ArchitectureTermination.NEIGHBOURHOODS_EXHAUSTED, report.termination)
        assertEquals(listOf(2), report.selection.recommended!!.architecture.hidden)
        assertEquals(4, report.selection.recommended.successes)
        assertEquals(4, report.evaluated)
        assertEquals(20, progress.last().finishedTrials)
        assertTrue(report.lineage.drop(1).all { it.parent != null })
        assertTrue(report.candidates.flatMap { it.trials }.all { it.epochs == 10_000 })
        println("ADAPTIVE_XOR " + report.lineage.joinToString { "${it.parent} --${it.mutation}--> ${it.architecture}" })
    }

    @Test fun largeSpacesUseDynamicBudgetsRatherThanTheExhaustiveEnumerationLimit() {
        val config = ArchitectureSearchConfig(maxLayers = 8, maxWidth = 128, maxParameters = 1_000_000,
            initialHidden = listOf(4, 3), maxEpochs = 1, checkEvery = 1, maxTrials = 5)
        val report = NeuroArchitectureSearch().search(ArchitectureSearchData.fitting(xor()), config)
        assertEquals(ArchitectureTermination.TRIAL_BUDGET, report.termination)
        assertEquals(1, report.evaluated); assertEquals(listOf(4, 3), report.candidates.single().architecture.hidden)
        val single = NeuroArchitectureSearch().search(ArchitectureSearchData.fitting(xor()),
            ArchitectureSearchConfig(maxLayers = 1, maxWidth = 1, maxEpochs = 1, checkEvery = 1, maxRestarts = 0))
        assertEquals(ArchitectureTermination.NEIGHBOURHOODS_EXHAUSTED, single.termination)
    }

    @Test fun adaptiveCancellationDeadlineAndObserverFailureDoNotLeakWorkers() {
        val config = ArchitectureSearchConfig(maxEpochs = 1_000_000)
        val engine = NeuroArchitectureSearch(); val data = ArchitectureSearchData.fitting(xor())
        val initial = engine.search(data, config, {}, { true })
        assertEquals(ArchitectureTermination.CANCELLED, initial.termination)
        assertTrue(initial.lineage.isEmpty())
        val stop = AtomicBoolean(false)
        val partial = engine.search(data, config, { if (it.running.isNotEmpty()) stop.set(true) }, { stop.get() })
        assertEquals(ArchitectureTermination.CANCELLED, partial.termination)
        assertNull(partial.selection.recommended)
        assertTrue(partial.partial > 0)
        val deadline = engine.search(data, ArchitectureSearchConfig(timeLimitSeconds = 1),
            { if (it.finishedTrials == 0) Thread.sleep(1050) }, { false })
        assertEquals(ArchitectureTermination.TIME_LIMIT, deadline.termination)
        assertThrows(IllegalStateException::class.java) {
            engine.search(data, config, { if (it.running.isNotEmpty()) error("Observer failed") }, { false })
        }
        assertFalse(Thread.getAllStackTraces().keys.any { it.isAlive && it.name.startsWith("jneuro-search-") })
    }

    @Test fun adaptiveCheckpointStorageIsCheckedBeforeFundingTheNextArchitecture() {
        val config = ArchitectureSearchConfig(minLayers = 8, maxLayers = 8, minWidth = 126, maxWidth = 128,
            maxParameters = 1_000_000, initialHidden = List(8) { 128 }, seeds = (1L..20L).toList(), requiredSuccesses = 1,
            maxTrials = 200, maxEpochs = 1, checkEvery = 1, maxRestarts = 0, parallelism = 2)
        val data = ArchitectureSearchData.fitting(listOf(NeuroLearningSets.Sample(0.2, 0.3, 0.0)))
        val report = NeuroArchitectureSearch().search(data, config)
        assertEquals(ArchitectureTermination.MEMORY_LIMIT, report.termination)
        assertTrue(report.candidates.sumOf { it.architecture.parameters.toLong() * config.seeds.size } <= 8_000_000L)
        assertEquals(1, report.untested)
    }

    private fun xor() = NeuroLearningSets.create(NeuroLearningSets.Kind.XOR, 42)
    private fun candidate(architecture: NetworkArchitecture, error: Double, state: ArchitectureTrialState = ArchitectureTrialState.COMPLETED) =
        ArchitectureCandidate(architecture, List(5) { index ->
            ArchitectureTrial(index.toLong(), state, 100, 100, error, error, error, 400, 1, emptyList(), null)
        }, 5, 0.05)
}
