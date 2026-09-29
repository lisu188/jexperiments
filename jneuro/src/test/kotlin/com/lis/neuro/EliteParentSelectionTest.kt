package com.lis.neuro

import java.util.SplittableRandom
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

class EliteParentSelectionTest {
    @Test fun parentsAreRankedByTheActivePolicyNotTheirInsertionOrder() {
        val candidates = listOf(candidate(1, 0.4), candidate(2, 0.03), candidate(3, 0.027), candidate(4, 0.01),
            candidate(5, 0.8), candidate(6, 0.0001, ArchitectureTrialState.FAILED))
        fun rank(policy: ArchitecturePolicy): List<Int> {
            val config = ArchitectureSearchConfig(policy = policy, nearBestTolerance = 0.018)
            val result = EliteParentSelection.rank(ArchitectureRanking.select(candidates.reversed(), config), config)
            assertEquals(result, EliteParentSelection.rank(ArchitectureRanking.select(candidates, config), config))
            assertThrows(UnsupportedOperationException::class.java) { (result as MutableList).clear() }
            return result.map { it.hidden.single() }
        }
        assertEquals(listOf(2, 3, 4, 1), rank(ArchitecturePolicy.SMALLEST_MEETING_TARGET))
        assertEquals(listOf(3, 4, 2, 1), rank(ArchitecturePolicy.SMALLEST_NEAR_BEST))
        assertEquals(listOf(4, 3, 2, 1), rank(ArchitecturePolicy.LOWEST_RMSE))
        val config = ArchitectureSearchConfig()
        assertTrue(EliteParentSelection.rank(ArchitectureRanking.select(emptyList(), config), config).isEmpty())
    }

    @Test fun reliabilityFrontierSurvivesALuckyLowMedianAndPartialGroupsAreNotParents() {
        val lucky = candidate(NetworkArchitecture(listOf(2)), listOf(0.001, 0.001, 0.001, 0.8, 0.8))
        val reliable = candidate(3, 0.03)
        val accurate = candidate(4, 0.01)
        val partial = candidate(NetworkArchitecture(listOf(1)), listOf(0.00001))
        val candidates = listOf(lucky, reliable, accurate, partial, candidate(5, 0.000001, ArchitectureTrialState.FAILED))
        val config = ArchitectureSearchConfig()
        val selection = ArchitectureRanking.select(candidates, config)
        assertEquals(listOf(3, 4, 2), EliteParentSelection.rank(selection, config).map { it.hidden.single() })
        assertEquals(reliable.architecture, selection.recommended!!.architecture)
        assertFalse(lucky.meetsTarget(config.requiredSuccesses))
        assertFalse(partial.fullyEvaluated)
    }

    @Test fun seededTournamentFavoursHigherFitnessWithoutUsingOnlyOneParent() {
        val ranked = (1..4).map { NetworkArchitecture(listOf(it)) }
        fun draws(seed: Long): List<NetworkArchitecture> {
            val random = SplittableRandom(seed)
            return List(20_000) { EliteParentSelection.choose(ranked, random) }
        }
        val results = draws(42)
        assertEquals(results, draws(42))
        assertNotEquals(results.take(100), draws(43).take(100))
        val counts = results.groupingBy { it }.eachCount()
        assertEquals(ranked.toSet(), counts.keys)
        assertTrue(counts.getValue(ranked[0]) > counts.getValue(ranked[1]))
        assertTrue(counts.getValue(ranked[1]) > counts.getValue(ranked[2]))
        assertTrue(counts.getValue(ranked[0]) > 4 * counts.getValue(ranked[3]))
        assertEquals(ranked[0], EliteParentSelection.choose(listOf(ranked[0]), SplittableRandom(2)))
        assertThrows(IllegalArgumentException::class.java) { EliteParentSelection.choose(emptyList(), SplittableRandom(1)) }
    }

    @Test fun changedScoresChangeTournamentSelectionWithTheSameRandomDraws() {
        val config = ArchitectureSearchConfig(policy = ArchitecturePolicy.LOWEST_RMSE)
        val first = listOf(candidate(1, 0.5), candidate(2, 0.3), candidate(3, 0.1))
        val second = listOf(candidate(1, 0.5), candidate(2, 0.05), candidate(3, 0.1))
        val before = EliteParentSelection.rank(ArchitectureRanking.select(first, config), config)
        val after = EliteParentSelection.rank(ArchitectureRanking.select(second, config), config)
        assertEquals(listOf(3), before.first().hidden)
        assertEquals(listOf(2), after.first().hidden)
        assertFalse(NetworkArchitecture(listOf(3)) in after)
        val left = SplittableRandom(42); val right = SplittableRandom(42)
        assertNotEquals(List(50) { EliteParentSelection.choose(before, left) }, List(50) { EliteParentSelection.choose(after, right) })
    }

    @Test fun demotedParentsCannotLeakThroughPreviouslyGeneratedNeighbourQueues() {
        val root = ArchitectureProposal(NetworkArchitecture(listOf(6)))
        val config = (1L..50L).map { ArchitectureSearchConfig(initialHidden = listOf(6), maxLayers = 1,
            maxWidth = 12, maxRestarts = 5, restartAfter = 1, searchSeed = it) }.first {
            AdaptiveArchitecturePlanner(it).mutations(root).first().architecture.parameters < root.architecture.parameters
        }
        val planner = AdaptiveArchitecturePlanner(config)
        val initial = planner.next()!!
        planner.observe(candidate(initial.architecture, List(5) { 0.5 }))
        val child = planner.next()!!
        assertTrue(child.architecture.parameters < initial.architecture.parameters)
        planner.observe(candidate(child.architecture, List(5) { 0.01 }))
        var count = 0
        while (count < 12) {
            val next = planner.next() ?: break
            assertEquals(child.architecture, next.parent)
            assertNotEquals(initial.architecture, next.parent)
            planner.observe(candidate(next.architecture, List(5) { 0.9 }, ArchitectureTrialState.FAILED))
            count++
        }
        assertTrue(count > 1)
    }

    @Test fun everyOrdinaryAndRestartChildUsesTheCurrentEvaluatedElite() {
        for (policy in ArchitecturePolicy.entries) {
            val config = ArchitectureSearchConfig(initialHidden = listOf(3, 2), maxWidth = 8, maxLayers = 3,
                restartAfter = 1, maxRestarts = 12, policy = policy)
            val planner = AdaptiveArchitecturePlanner(config)
            val completed = ArrayList<ArchitectureCandidate>()
            val parents = HashMap<NetworkArchitecture, ArchitectureProposal>()
            repeat(50) { index ->
                val next = planner.next() ?: return@repeat
                assertNull(parents.put(next.architecture, next))
                if (index > 0) {
                    assertNotNull(next.parent, "No unrelated random restart once an elite exists")
                    val ranked = EliteParentSelection.rank(ArchitectureRanking.select(completed, config), config)
                    assertTrue(next.parent in ranked, "${next.parent} not in $ranked")
                    assertEquals(parents.getValue(next.parent!!).generation + 1, next.generation)
                    assertTrue(next.architecture.hidden.size in config.minLayers..config.maxLayers)
                    assertTrue(next.architecture.hidden.all { it in config.minWidth..config.maxWidth })
                    assertTrue(next.architecture.parameters <= config.maxParameters)
                }
                val score = if (index == 0) 0.1 else if (index % 3 == 0) 0.9 else 1.0 / next.architecture.parameters
                val observed = candidate(next.architecture, List(5) { score })
                completed += observed
                planner.observe(observed)
            }
            assertTrue(completed.size > 5)
            assertTrue(planner.lineage.any { it.mutation.startsWith("Elite restart") })
            assertTrue(planner.restartCount <= config.maxRestarts)
        }
    }

    @Test fun freshBootstrapStopsAsSoonAsAValidParentExists() {
        val config = ArchitectureSearchConfig(initialHidden = listOf(4), maxLayers = 2, maxWidth = 8,
            restartAfter = 1, maxRestarts = 6)
        val planner = AdaptiveArchitecturePlanner(config)
        val initial = planner.next()!!
        planner.observe(candidate(initial.architecture, List(5) { 0.001 }, ArchitectureTrialState.FAILED))
        val bootstrap = planner.next()!!
        assertNull(bootstrap.parent)
        assertTrue(bootstrap.mutation.contains("bootstrap"))
        planner.observe(candidate(bootstrap.architecture, List(5) { 0.1 }))
        val child = planner.next()!!
        assertEquals(bootstrap.architecture, child.parent)
        planner.observe(candidate(child.architecture, List(5) { 0.9 }, ArchitectureTrialState.FAILED))
        val restart = planner.next()!!
        assertEquals(bootstrap.architecture, restart.parent)
        assertTrue(restart.mutation.startsWith("Elite restart"))
    }

    private fun candidate(width: Int, error: Double, state: ArchitectureTrialState = ArchitectureTrialState.COMPLETED) =
        candidate(NetworkArchitecture(listOf(width)), List(5) { error }, state)

    private fun candidate(architecture: NetworkArchitecture, scores: List<Double>, state: ArchitectureTrialState = ArchitectureTrialState.COMPLETED) =
        ArchitectureCandidate(architecture, scores.mapIndexed { index, error ->
            ArchitectureTrial(index.toLong(), state, 100, 100, error, error, error, 400, 1, emptyList(), null)
        }, 5, 0.05)
}
