package com.lis.neuro

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

class CublasTrainingSessionTest {
    @Test fun sessionRetainsOwnershipAndDispatchesEveryEntrypointWithItsBatchAndPrecision() {
        val model = NeuroTest.prepared(intArrayOf(3, 5, 2))
        val calls = ArrayList<Pair<Int, Int>>()
        val session = CublasTrainingSession(model, Neuro.TrainingPrecision.FP32, 7, ::info) { network, epochs, batch, precision ->
            assertEquals(Neuro.TrainingPrecision.FP32, precision)
            calls += epochs to batch
            network.trainMiniBatch(epochs, batch, 1, Neuro.BatchBackend.CPU)
        }
        assertEquals(TrainingBackend.CUBLAS, session.info.backend)
        assertEquals("FP32", session.info.precision)
        assertThrows(IllegalStateException::class.java) { model.trainEpoch() }
        assertThrows(IllegalStateException::class.java) { model.newTrainingSession() }
        assertTrue(session.trainEpoch().isFinite())
        session.train(2)
        session.trainMiniBatch(2, 64, 8)
        assertEquals(listOf(1 to 7, 2 to 7, 2 to 32), calls)
        assertEquals(5L, model.statistics().epochsTrained)
        val complete = session.trainUntil(1.0, 0)
        assertTrue(complete.converged); assertEquals(0, complete.epochs)
        assertEquals(model.trainingError(), model.statistics().lastTrainingError)
        val incomplete = session.trainUntil(0.0, 5, 2)
        assertFalse(incomplete.converged); assertEquals(5, incomplete.epochs)
        assertEquals(listOf(2 to 7, 2 to 7, 1 to 7), calls.takeLast(3))
        session.close(); session.close()
        assertThrows(IllegalStateException::class.java) { session.trainEpoch() }
        assertTrue(model.trainEpoch().isFinite())
    }

    @Test fun invalidArgumentsDoNotPoisonSessionButNativeFailuresDo() {
        val model = NeuroTest.prepared(intArrayOf(3, 2))
        var fail = false
        CublasTrainingSession(model, Neuro.TrainingPrecision.FP64, 1, ::info) { _, _, _, _ ->
            check(!fail) { "device lost" }
        }.use { session ->
            assertThrows(IllegalArgumentException::class.java) { session.train(-1) }
            assertThrows(IllegalArgumentException::class.java) { session.trainMiniBatch(1, 0) }
            assertThrows(IllegalArgumentException::class.java) { session.trainMiniBatch(1, 1, 0) }
            assertThrows(IllegalArgumentException::class.java) { session.trainUntil(Double.NaN, 1) }
            assertThrows(IllegalArgumentException::class.java) { session.trainUntil(0.0, -1) }
            assertThrows(IllegalArgumentException::class.java) { session.trainUntil(0.0, 1, 0) }
            session.train(0); session.trainMiniBatch(0, 1)
            assertTrue(session.trainEpoch().isFinite())
            fail = true
            assertThrows(IllegalStateException::class.java) { session.trainEpoch() }
            fail = false
            assertThrows(IllegalStateException::class.java) { session.trainEpoch() }
        }
        assertTrue(model.trainEpoch().isFinite())
        val empty = Neuro(intArrayOf(2, 1))
        CublasTrainingSession(empty, Neuro.TrainingPrecision.FP64, 1, ::info).use { session ->
            session.train(0)
            assertThrows(IllegalStateException::class.java) { session.trainUntil(0.0, 1) }
        }
        CublasTrainingSession(empty, Neuro.TrainingPrecision.FP64, 1, ::info).use { session ->
            assertThrows(IllegalStateException::class.java) { session.trainEpoch() }
        }
    }

    @Test fun constructionFailuresReleaseOwnershipAndRejectMisleadingDeviceInformation() {
        val model = NeuroTest.prepared(intArrayOf(3, 2))
        assertThrows(IllegalArgumentException::class.java) { CublasTrainingSession(model, Neuro.TrainingPrecision.FP64, 0, ::info) }
        assertThrows(IllegalStateException::class.java) {
            CublasTrainingSession(model, Neuro.TrainingPrecision.FP64, 1, { error("unavailable") })
        }
        assertThrows(IllegalStateException::class.java) {
            CublasTrainingSession(model, Neuro.TrainingPrecision.FP32, 1, { info(Neuro.TrainingPrecision.FP64) })
        }
        model.newTrainingSession().close()
    }

    @Test fun configuredSelectionReportsActualBackendAndRejectsUnsupportedPrecision() {
        val small = NeuroTest.prepared(intArrayOf(3, 2))
        assertThrows(IllegalArgumentException::class.java) { small.newTrainingSession(TrainingBackend.CUDA, Neuro.TrainingPrecision.FP32) }
        assertThrows(IllegalArgumentException::class.java) { small.newTrainingSession(TrainingBackend.CPU, batchSize = 0) }
        openConfiguredTrainingSession(small, TrainingBackend.AUTO, Neuro.TrainingPrecision.FP32, 32,
            cublasAvailable = { error("Small AUTO workloads must not probe native libraries") }).use { session ->
            assertEquals(TrainingBackend.CPU, session.info.backend)
            assertEquals("FP64", session.info.precision)
            session.trainEpoch()
        }
        val large = NeuroTest.prepared(intArrayOf(128, 256, 128))
        openConfiguredTrainingSession(large, TrainingBackend.AUTO, Neuro.TrainingPrecision.FP32, 32,
            cublasFactory = { network, precision, batch -> CublasTrainingSession(network, precision, batch, ::info) },
            cublasAvailable = { true }).use { session ->
            assertEquals(TrainingBackend.CUBLAS, session.info.backend)
            assertEquals("FP32", session.info.precision)
        }
        openConfiguredTrainingSession(large, TrainingBackend.AUTO, Neuro.TrainingPrecision.FP32, 32,
            cublasAvailable = { false }).use { assertEquals(TrainingBackend.CPU, it.info.backend) }
        openConfiguredTrainingSession(small, TrainingBackend.CUBLAS, Neuro.TrainingPrecision.FP64, 1,
            cublasFactory = { network, precision, batch -> CublasTrainingSession(network, precision, batch, ::info) }).close()
    }

    @Test fun cpuBatchHintPreservesMatrixSemanticsAndDefaultCpuRemainsOnline() {
        val expected = NeuroTest.prepared(intArrayOf(3, 5, 2))
        val actual = NeuroTest.prepared(intArrayOf(3, 5, 2))
        actual.newTrainingSession(TrainingBackend.CPU, Neuro.TrainingPrecision.FP32, 7).use { session ->
            expected.trainMiniBatch(3, 7, 1, Neuro.BatchBackend.CPU); session.train(3)
            assertEquals(expected.trainingError(), actual.trainingError(), 0.0)
            val result = session.trainUntil(0.0, 3, 2)
            expected.trainMiniBatch(3, 7, 1, Neuro.BatchBackend.CPU)
            assertEquals(3, result.epochs)
            assertEquals(expected.trainingError(), result.error, 0.0)
            expected.trainMiniBatch(2, 5, 2, Neuro.BatchBackend.CPU)
            session.trainMiniBatch(2, 5, 2)
            assertEquals(expected.trainingError(), actual.trainingError(), 0.0)
            assertEquals("cpu-matrix-v1", session.info.kernelVersion)
        }
    }

    @Test fun deviceMetadataIncludesTheCompilerArchitectureOptionsAndKernelSource() {
        CudaNativeTestRuntime().use { native ->
            val identity = TrainingDeviceInfo(TrainingBackend.CUDA, "Test GPU", "uuid/driver-13000")
            val actual = cublasDeviceInfo(Neuro.TrainingPrecision.FP32, native.loader) { identity }
            assertEquals(TrainingBackend.CUBLAS, actual.backend)
            assertEquals("Test GPU", actual.name)
            assertEquals("uuid/driver-13000", actual.identity)
            assertEquals("FP32", actual.precision)
            assertTrue(actual.kernelVersion.startsWith("cublas-130002/nvrtc-12.0/compute-8.6/"), actual.kernelVersion)
            assertTrue(actual.kernelVersion.contains("options=--gpu-architecture=compute_86/source-"), actual.kernelVersion)
            assertEquals(64, actual.kernelVersion.substringAfter("/source-").length)
            assertTrue(native.libraryArenas.all { !it.scope().isAlive })

            native.missing += "cublasGetVersion_v2"
            val withoutOptionalVersion = cublasDeviceInfo(Neuro.TrainingPrecision.FP64, native.loader) { identity }
            assertNotEquals(actual.kernelVersion, withoutOptionalVersion.kernelVersion)
            assertTrue(withoutOptionalVersion.kernelVersion.contains("/nvrtc-12.0/"))
            assertTrue(native.libraryArenas.all { !it.scope().isAlive })
        }
    }

    @Test fun metadataFailureClosesLibrariesAndCannotAcquireModelOwnership() {
        CudaNativeTestRuntime().use { native ->
            native.errors["nvrtcVersion"] = 3
            val model = NeuroTest.prepared(intArrayOf(3, 2))
            assertThrows(IllegalStateException::class.java) {
                CublasTrainingSession(model, Neuro.TrainingPrecision.FP64, 7, { precision ->
                    cublasDeviceInfo(precision, native.loader) { info(precision) }
                })
            }
            assertTrue(native.libraryArenas.all { !it.scope().isAlive })
            model.newTrainingSession().close()
            native.errors.clear()
            native.deviceCount = 0
            assertThrows(IllegalStateException::class.java) {
                cublasDeviceInfo(Neuro.TrainingPrecision.FP64, native.loader) { info(Neuro.TrainingPrecision.FP64) }
            }
            assertTrue(native.libraryArenas.all { !it.scope().isAlive })
        }
    }

    private fun info(precision: Neuro.TrainingPrecision) =
        TrainingDeviceInfo(TrainingBackend.CUBLAS, "Test GPU", "test-device", precision.name, "test-kernels")
}
