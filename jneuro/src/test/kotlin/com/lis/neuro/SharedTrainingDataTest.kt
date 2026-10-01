package com.lis.neuro

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test
import com.lis.neuro.SmallCpuTrainingTest.Companion.assertState

class SharedTrainingDataTest {
    private val shape = intArrayOf(2, 4, 1)
    private val parameters = Neuro.HyperParameters(0.1, 0.2, 1.0, 42)

    @Test fun modelsShareFrozenPackedDataAndPublicMutationDetachesOnlyTheChangedModel() {
        val samples = NeuroLearningSets.create(NeuroLearningSets.Kind.SPIRAL, 42)
        val training = samples.take(176).toMutableList()
        val validation = samples.drop(176).toMutableList()
        val data = Neuro.SharedDatasets.fromSamples(training, validation)
        fun model() = Neuro(shape, parameters).attachSharedDatasets(data)
        val first = model(); val second = model()
        training.clear(); validation.clear()
        assertEquals(176, first.trainingSampleCount()); assertEquals(44, first.testSampleCount())
        assertSame(first.backendTrainingData(), second.backendTrainingData())
        val before = second.testError()
        val input = doubleArrayOf(0.2, 0.8); val target = doubleArrayOf(0.0)
        first.addTrainingSample(input, target).addTestSample(input, target)
        input.fill(100.0); target.fill(100.0)
        assertEquals(177, first.trainingSampleCount()); assertEquals(45, first.testSampleCount())
        assertEquals(176, second.trainingSampleCount()); assertEquals(44, second.testSampleCount())
        assertNotSame(first.backendTrainingData(), second.backendTrainingData())
        assertEquals(before, second.testError())
        assertArrayEquals(doubleArrayOf(0.2, 0.8), first.exportTrainingState().inputs.takeLast(2).toDoubleArray())
        val exported = second.exportTrainingState()
        assertFalse(exported.sharedDataset)
        exported.inputs.fill(Double.NaN); exported.targets.fill(Double.NaN)
        assertTrue(second.trainingError().isFinite())
        val third = model()
        assertSame(second.backendTrainingData(), third.backendTrainingData())
        assertEquals(second.trainingError(), third.trainingError())
    }

    @Test fun sharedDatasetTrainingMatchesOrdinaryConstructionAndRejectsUnsafeAttachment() {
        val training = NeuroLearningSets.create(NeuroLearningSets.Kind.SPIRAL, 42)
        val data = Neuro.SharedDatasets.fromSamples(training, emptyList())
        val shared = Neuro(shape, parameters).attachSharedDatasets(data)
        val ordinary = Neuro(shape, parameters).also { NeuroLearningSets.addTo(it, training) }
        shared.train(5); ordinary.train(5)
        assertState(ordinary.exportTrainingState(), shared.exportTrainingState())
        assertEquals(ordinary.statistics(), shared.statistics())
        assertThrows(IllegalStateException::class.java) { shared.attachSharedDatasets(data) }
        val fresh = Neuro(shape, parameters)
        fresh.newTrainingSession().use {
            assertThrows(IllegalStateException::class.java) { fresh.attachSharedDatasets(data) }
        }
        assertThrows(IllegalArgumentException::class.java) { Neuro(intArrayOf(3, 4, 1)).attachSharedDatasets(data) }
        assertThrows(IllegalArgumentException::class.java) {
            Neuro.SharedDatasets.fromSamples(listOf(NeuroLearningSets.Sample(Double.NaN, 0.0, 0.0)), emptyList())
        }
        assertThrows(IllegalArgumentException::class.java) {
            Neuro.SharedDatasets.fromSamples(emptyList(), listOf(NeuroLearningSets.Sample(0.0, 0.0, Double.POSITIVE_INFINITY)))
        }
    }

    @Test fun singleAndCohortSnapshotsOwnParametersWhileReusingTheirImmutableDataset() {
        for (precision in Neuro.TrainingPrecision.entries) {
            val model = NeuroTest.prepared(shape)
            val state = model.exportTrainingState()
            val firstInput = state.inputs[0]
            SmallCpuTraining(state, model.hyperParameters(), precision).use { kernel ->
                state.inputs[0] = Double.NaN
                val first = kernel.train(emptyArray(), 1, true)
                assertEquals(firstInput, first.inputs[0])
                assertFalse(first.sharedDataset)
                first.weights[0][0] = Double.NaN
                val second = kernel.train(emptyArray(), 1, true)
                assertTrue(second.weights[0][0].isFinite())
                assertNotSame(first.inputs, second.inputs); assertNotSame(first.targets, second.targets)
                first.inputs[0] = Double.NaN
                assertEquals(firstInput, second.inputs[0])
            }
            val cohortState = model.exportTrainingState()
            SmallCpuCohort(arrayOf(cohortState), model.hyperParameters(), precision).use { cohort ->
                cohortState.inputs[0] = Double.NaN
                val first = cohort.train(arrayOf(emptyArray()), 1, true).single()
                val second = cohort.train(arrayOf(emptyArray()), 1, true).single()
                assertEquals(firstInput, first.inputs[0])
                assertNotSame(first.inputs, second.inputs); assertFalse(second.sharedDataset)
                first.inputs[0] = Double.NaN
                assertEquals(firstInput, second.inputs[0])
                first.weights[0][0] = Double.NaN
                assertTrue(second.weights[0][0].isFinite())
            }
            val shared = model.exportTrainingState(shareDataset = true)
            SmallCpuTraining(shared, model.hyperParameters(), precision).use { kernel ->
                assertSame(shared.inputs, kernel.train(emptyArray(), 1, true).inputs)
            }
            SmallCpuCohort(arrayOf(shared), model.hyperParameters(), precision).use { cohort ->
                val snapshot = cohort.train(arrayOf(emptyArray()), 1, true).single()
                assertSame(shared.inputs, snapshot.inputs)
                assertTrue(snapshot.sharedDataset)
            }
            assertThrows(IllegalArgumentException::class.java) {
                validateSmallOrders(emptyArray(), state.samples, BooleanArray(state.samples + 1))
            }
        }
    }

    @Test fun sharedDataGrowthAfterClosingSessionPreservesShuffleContinuationAndSiblingData() {
        val samples = NeuroLearningSets.create(NeuroLearningSets.Kind.XOR, 42)
        val data = Neuro.SharedDatasets.fromSamples(samples, emptyList())
        val model = Neuro(shape, parameters).attachSharedDatasets(data)
        val sibling = Neuro(shape, parameters).attachSharedDatasets(data)
        val expected = Neuro(shape, parameters).also { NeuroLearningSets.addTo(it, samples) }
        model.newTrainingSession(engine = TrainingEngine.SMALL).use { session ->
            repeat(3) { session.trainEpoch() }
            assertThrows(IllegalStateException::class.java) {
                model.addTrainingSample(doubleArrayOf(0.2, 0.8), doubleArrayOf(1.0))
            }
        }
        expected.newTrainingSession(engine = TrainingEngine.SMALL).use { session -> repeat(3) { session.trainEpoch() } }
        for (current in listOf(model, expected)) current.addTrainingSample(doubleArrayOf(0.2, 0.8), doubleArrayOf(1.0))
        assertTrue(model.statistics().lastTrainingError.isNaN())
        assertEquals(4, sibling.trainingSampleCount())
        for (current in listOf(model, expected)) current.newTrainingSession(engine = TrainingEngine.SMALL).use { session ->
            assertEquals(current.trainingError(), session.currentRmse)
            advanceTrainingForSearch(session, TrainingChunkRequest(4, maxNanos = Long.MAX_VALUE))
        }
        assertState(expected.exportTrainingState(), model.exportTrainingState())
        assertEquals(expected.statistics(), model.statistics())
        assertSame(data.packedTraining, sibling.backendTrainingData())
    }
}
