package com.lis.neuro

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

class NeuroCudaBatchBackendTest {
    @Test fun bothPrecisionsDispatchGemmsAndKernelsForFullAndPartialBatches() {
        for (precision in Neuro.TrainingPrecision.entries) {
            CudaNativeTestRuntime().use { native ->
                val model = NeuroTest.prepared(intArrayOf(3, 5, 2), Neuro.Kernel.SCALAR)
                val before = model.exportTrainingState()
                NeuroCudaBatchBackend.train(model, model.backendTrainingData(), 2, 7, precision, native.loader)
                assertEquals(2L, model.statistics().epochsTrained)
                assertEquals(64L, model.statistics().samplesSeen)
                assertTrue(model.statistics().lastTrainingError.isFinite())
                val suffix = if (precision == Neuro.TrainingPrecision.FP32) "Float" else ""
                val gathers = native.launches.filter { it.name == "gatherRows$suffix" }
                assertEquals(20, gathers.size)
                assertEquals(listOf(7, 7, 7, 7, 4), gathers.filterIndexed { index, _ -> index % 2 == 0 }.take(5).map { it.arguments[4] })
                assertTrue(native.gemms.isNotEmpty())
                assertTrue(native.gemms.all { it.single == (precision == Neuro.TrainingPrecision.FP32) })
                assertTrue(native.launches.any { it.name == "applySigmoidDerivative$suffix" })
                assertTrue(native.launches.any { it.name == "momentumUpdate$suffix" })
                for (layer in before.weights.indices) {
                    assertArrayEquals(before.weights[layer], model.backendWeights(layer), if (precision == Neuro.TrainingPrecision.FP32) 1e-7 else 0.0)
                }
                assertTrue(native.allocations.isEmpty())
                assertTrue(native.libraryArenas.all { !it.scope().isAlive })
            }
        }
    }

    @Test fun failedOrNonfiniteDownloadsNeverPublishPartialParameters() {
        for (precision in Neuro.TrainingPrecision.entries) for (nonfinite in listOf(false, true)) {
            CudaNativeTestRuntime().use { native ->
                val model = NeuroTest.prepared(intArrayOf(3, 5, 2), Neuro.Kernel.SCALAR)
                val before = model.exportTrainingState()
                native.failDtoHAt = if (nonfinite) 0 else 2
                native.nonfiniteDtoH = nonfinite
                assertThrows(IllegalStateException::class.java) {
                    NeuroCudaBatchBackend.train(model, model.backendTrainingData(), 1, 7, precision, native.loader)
                }
                assertEquals(0L, model.statistics().epochsTrained)
                for (layer in before.weights.indices) {
                    assertArrayEquals(before.weights[layer], model.backendWeights(layer), 0.0)
                    assertArrayEquals(before.biases[layer], model.backendBiases(layer), 0.0)
                    assertArrayEquals(before.weightVelocity[layer], model.backendWeightVelocity(layer), 0.0)
                }
                assertTrue(native.allocations.isEmpty())
                val expected = NeuroTest.prepared(intArrayOf(3, 5, 2), Neuro.Kernel.SCALAR)
                expected.trainEpoch(); model.trainEpoch()
                assertEquals(expected.trainingError(), model.trainingError(), 0.0)
            }
        }
    }

    @Test fun laterEpochFailureRetainsEarlierEpochAndItsShuffleContinuation() {
        CudaNativeTestRuntime().use { native ->
            val model = NeuroTest.prepared(intArrayOf(3, 5, 2), Neuro.Kernel.SCALAR)
            native.failDtoHAt = 10 // Four buffers per layer: failure is in the second epoch.
            assertThrows(IllegalStateException::class.java) {
                NeuroCudaBatchBackend.train(model, model.backendTrainingData(), 3, 7, Neuro.TrainingPrecision.FP64, native.loader)
            }
            assertEquals(1L, model.statistics().epochsTrained)
            assertEquals(32L, model.statistics().samplesSeen)
            assertTrue(native.allocations.isEmpty())
            val expected = NeuroTest.prepared(intArrayOf(3, 5, 2), Neuro.Kernel.SCALAR)
            expected.backendNextTrainingOrder(32)
            expected.backendCompleteEpoch(32) // Native test kernels intentionally leave parameters unchanged.
            expected.trainEpoch(); model.trainEpoch()
            assertEquals(expected.trainingError(), model.trainingError(), 0.0)
        }
    }

    @Test fun initializationAndDispatchFailuresCloseAllSuccessfulResources() {
        for (operation in listOf("cublasCreate_v2", "nvrtcCreateProgram", "cuModuleLoadData", "cuLaunchKernel", "cublasDgemm_v2")) {
            CudaNativeTestRuntime().use { native ->
                val model = NeuroTest.prepared(intArrayOf(3, 5, 2))
                native.errors[operation] = 7
                assertThrows(IllegalStateException::class.java) {
                    NeuroCudaBatchBackend.train(model, model.backendTrainingData(), 1, 7, Neuro.TrainingPrecision.FP64, native.loader)
                }
                assertTrue(native.allocations.isEmpty(), operation)
                assertTrue(native.libraryArenas.all { !it.scope().isAlive }, operation)
                assertEquals(0L, model.statistics().epochsTrained)
            }
        }
    }

    @Test fun cleanupContinuesAfterOneReleaseFailsAndNoOpTrainingDoesNotLoadCuda() {
        CudaNativeTestRuntime().use { native ->
            val model = NeuroTest.prepared(intArrayOf(3, 2))
            NeuroCudaBatchBackend.train(model, model.backendTrainingData(), 0, 1, Neuro.TrainingPrecision.FP64, native.loader)
            assertTrue(native.calls.isEmpty())
            var releases = 0
            native.beforeCall = { operation ->
                if (operation == "cudaFree") {
                    releases++
                    if (releases == 1) native.errors[operation] = 7 else native.errors.remove(operation)
                }
            }
            assertThrows(IllegalStateException::class.java) {
                NeuroCudaBatchBackend.train(model, model.backendTrainingData(), 1, 64, Neuro.TrainingPrecision.FP64, native.loader)
            }
            assertTrue(releases > 1)
            assertEquals(1, native.allocations.size) // The driver rejected exactly one free; all other frees were attempted.
            assertTrue(native.calls.contains("cublasDestroy_v2"))
            assertTrue(native.calls.contains("cuModuleUnload"))
            assertTrue(native.libraryArenas.all { !it.scope().isAlive })
        }
        val empty = Neuro(intArrayOf(2, 1))
        assertThrows(IllegalStateException::class.java) {
            NeuroCudaBatchBackend.train(empty, empty.backendTrainingData(), 1, 1, Neuro.TrainingPrecision.FP64)
        }
        assertThrows(IllegalArgumentException::class.java) {
            NeuroCudaBatchBackend.train(empty, empty.backendTrainingData(), -1, 1, Neuro.TrainingPrecision.FP64)
        }
    }
}
