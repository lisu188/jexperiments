package com.lis.neuro

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

class BatchedSearchReplayTest {
    private val samples = NeuroLearningSets.create(NeuroLearningSets.Kind.SPIRAL, 42)
    private val data = ArchitectureSearchData.split(samples)
    private val config = ArchitectureSearchConfig(maxLayers = 1, maxWidth = 4, seeds = listOf(42),
        requiredSuccesses = 1, maxEpochs = 7, checkEvery = 2, execution = ArchitectureExecution.BATCHED)
    private val shape = NetworkArchitecture(listOf(2))
    private val info = TrainingDeviceInfo(TrainingBackend.CPU, "Replay fixture", "fixture", kernelVersion = "fixture-batched")
    private fun network(source: ArchitectureSearchData = data) = source.newNetwork(shape, config.hyperParameters, 42)
    private fun trial(segments: List<ArchitectureBatchSegment> = listOf(
        ArchitectureBatchSegment(0, 2, listOf(2, 4, 1), 4, 3),
        ArchitectureBatchSegment(2, 7, listOf(2, 3, 1), 2, 1)), bestEpoch: Int = 5) =
        ArchitectureTrial(42, ArchitectureTrialState.COMPLETED, 7, bestEpoch, 0.2, 0.3, 0.2,
            7L * data.training.size, 1, emptyList(), null, deviceInfo = info,
            execution = ArchitectureExecution.BATCHED, route = ArchitectureTrialRoute.TENSOR_BATCH, batchSegments = segments)

    @Test fun replayRetainsShuffleCountersAndGeometryAcrossRebindings() {
        val model = network()
        val fixture = Fixture()
        val devices = arrayListOf<TrainingDeviceInfo>()
        val score = BatchedSearchReplay.replay(model, data, config, trial(), devices::add, { false }, fixture::open)
        assertEquals(0.2, score)
        assertEquals(listOf(4, 2), fixture.capacities)
        assertEquals(listOf(listOf(2, 4, 1), listOf(2, 3, 1)), fixture.padding)
        assertEquals(listOf(3, 3, 1, 1, 1).toSet(), fixture.activeLanes.toSet())
        assertEquals(2, fixture.closed)
        assertEquals(listOf(info, info), devices)
        assertEquals(5L, model.statistics().epochsTrained)
        assertEquals(5L * data.training.size, model.statistics().samplesSeen)
        val shuffle = TrainingShuffle(42L xor -7046029254386353131L)
        val expected = shuffle.reserve(data.training.size, 6).map { it.copyOf() }
        assertEquals(5, fixture.orders.size)
        fixture.orders.forEachIndexed { index, order -> assertArrayEquals(expected[index], order) }
        assertArrayEquals(expected.last(), model.reserveTrainingOrders(1).single())
    }

    @Test fun epochZeroAndTrainingFitUseNativeScoreWithoutAdvancing() {
        val fitting = ArchitectureSearchData.fitting(samples)
        val fixture = Fixture()
        assertEquals(0.3, BatchedSearchReplay.replay(network(fitting), fitting, config, trial(bestEpoch = 0),
            {}, { false }, fixture::open))
        assertTrue(fixture.orders.isEmpty())
        assertEquals(1, fixture.closed)
    }

    @Test fun cancellationBeforeOpenAndAfterCommittedChunkClosesResources() {
        val before = Fixture()
        assertNull(BatchedSearchReplay.replay(network(), data, config, trial(), {}, { true }, before::open))
        assertTrue(before.capacities.isEmpty())
        val during = Fixture()
        val model = network()
        assertNull(BatchedSearchReplay.replay(model, data, config, trial(), {}, { during.orders.isNotEmpty() }, during::open))
        assertEquals(1L, model.statistics().epochsTrained)
        assertEquals(1, during.closed)
    }

    @Test fun malformedOrExcessiveGeometryIsRejectedBeforeNativeAllocation() {
        val fixture = Fixture()
        val invalid = listOf(
            emptyList(),
            listOf(ArchitectureBatchSegment(1, 7, listOf(2, 2, 1), 1, 0)),
            listOf(ArchitectureBatchSegment(0, 4, listOf(2, 2, 1), 1, 0)),
            listOf(ArchitectureBatchSegment(0, 7, listOf(2, 2, 1), 0, 0)),
            listOf(ArchitectureBatchSegment(0, 7, listOf(2, 2, 1), 1, 1)),
            listOf(ArchitectureBatchSegment(0, 7, listOf(2, 1, 1), 1, 0)),
            listOf(ArchitectureBatchSegment(0, 7, listOf(2, 2), 1, 0)),
            listOf(ArchitectureBatchSegment(0, 8, listOf(2, 2, 1), 1, 0)),
            listOf(ArchitectureBatchSegment(0, 7, listOf(2, Int.MAX_VALUE, 1), 1024, 0)))
        for (segments in invalid) assertThrows(IllegalArgumentException::class.java) {
            BatchedSearchReplay.replay(network(), data, config, trial(segments), {}, { false }, fixture::open)
        }
        assertTrue(fixture.capacities.isEmpty())
    }

    @Test fun failedTrainingScoringOrProvenanceNeverSucceeds() {
        for (mode in listOf("training", "scoring", "geometry", "provenance")) {
            val fixture = Fixture(mode)
            val model = network()
            assertThrows(IllegalStateException::class.java) {
                BatchedSearchReplay.replay(model, data, config, trial(), { check(mode != "provenance") }, { false }, fixture::open)
            }
            assertTrue(fixture.closed > 0)
            if (mode != "scoring") assertEquals(0L, model.statistics().epochsTrained)
        }
    }

    /** No numerical training: this fixture verifies replay transactions and native ownership. */
    private inner class Fixture(private val failure: String = "") {
        val capacities = arrayListOf<Int>()
        val padding = arrayListOf<List<Int>>()
        val activeLanes = arrayListOf<Int>()
        val orders = arrayListOf<IntArray>()
        var closed = 0
        fun open(states: Array<NeuroTrainingState>, padded: IntArray): TensorFlowSearchCohort {
            capacities += states.size; padding += padded.toList()
            return object : TensorFlowSearchCohort {
                override val info = this@BatchedSearchReplayTest.info
                override val size = states.size
                override val paddedTopology = if (failure == "geometry") states[0].topology else padded
                private var epochs = 0
                override fun advance(orders: Array<Array<IntArray>>, active: BooleanArray): BooleanArray {
                    val lane = active.indices.single { active[it] }
                    activeLanes += lane
                    assertTrue(orders.indices.filter { !active[it] }.all { orders[it].isEmpty() })
                    this@Fixture.orders.addAll(orders[lane].map { it.copyOf() })
                    epochs += orders[lane].size
                    return BooleanArray(size) { failure == "training" }
                }
                override fun score(active: BooleanArray) = TensorFlowCohortMetrics(IntArray(size) { epochs },
                    DoubleArray(size) { 0.3 }, DoubleArray(size) { 0.2 }, DoubleArray(size) { 0.2 },
                    IntArray(size) { epochs }, BooleanArray(size) { failure == "scoring" })
                override fun exportState(lane: Int, best: Boolean) = states[lane]
                override fun exportStates(lanes: IntArray, best: Boolean) = Array(lanes.size) { states[lanes[it]] }
                override fun close() { closed++ }
            }
        }
    }
}
