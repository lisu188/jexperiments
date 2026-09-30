package com.lis.neuro

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertThrows
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class NeuroCudaPlanTest {
    @Test fun planCalculatesResidentMemoryAndWork() {
        val plan = NeuroCudaPlan(intArrayOf(32, 64, 8), 128)
        assertEquals(2, plan.layerCount)
        assertEquals(64, plan.maximumWidth)
        assertEquals(128L * (32 + 64 + 8), plan.activationElements)
        assertEquals(128L * (64 + 8), plan.deltaElements)
        assertEquals(32L * 64 + 64 + 64L * 8 + 8, plan.parameterElements)
        assertEquals(128L * (32L * 64 + 64L * 8) * 6, plan.estimatedFlops)
    }

    @Test fun backendResolutionKeepsExplicitChoicesAndUsesWorkThreshold() {
        val small = intArrayOf(2, 6, 1)
        val large = intArrayOf(512, 1024, 512, 128)
        assertEquals(Neuro.BatchBackend.CPU,
            NeuroCuda.resolveBackend(Neuro.BatchBackend.CPU, large, 1024, true))
        assertEquals(Neuro.BatchBackend.CUDA,
            NeuroCuda.resolveBackend(Neuro.BatchBackend.CUDA, small, 1, false))
        assertEquals(Neuro.BatchBackend.CPU,
            NeuroCuda.resolveBackend(Neuro.BatchBackend.AUTO, large, 1024, false))
        assertEquals(Neuro.BatchBackend.CPU,
            NeuroCuda.resolveBackend(Neuro.BatchBackend.AUTO, small, 1, true))
        assertEquals(Neuro.BatchBackend.CUDA,
            NeuroCuda.resolveBackend(Neuro.BatchBackend.AUTO, large, 1024, true))
    }

    @Test fun invalidPlansFailFastAndStatusIsSafeToQuery() {
        assertThrows(IllegalArgumentException::class.java) { NeuroCudaPlan(intArrayOf(2), 1) }
        assertThrows(IllegalArgumentException::class.java) { NeuroCudaPlan(intArrayOf(2, 0, 1), 1) }
        assertThrows(IllegalArgumentException::class.java) { NeuroCudaPlan(intArrayOf(2, 1), 0) }
        val status = NeuroCuda.status()
        assertTrue(status.description.isNotBlank())
        assertEquals(status.available, NeuroCuda.isAvailable())
        if (!status.available) assertFalse(status.reason.isBlank())
    }
    @Test fun optionalDetectionChecksEveryLibraryAndClosesTheProbe() {
        for (missing in listOf("", "cudaGetDeviceCount", "cublasCreate_v2", "nvrtcCreateProgram", "cuInit")) {
            CudaNativeTestRuntime().use { native ->
                if (missing.isNotEmpty()) native.missing += missing
                val status = NeuroCuda.detect(native.loader)
                assertEquals(missing.isEmpty(), status.available)
                if (status.available) assertTrue(status.description.contains("Adapter test GPU"))
                else assertTrue(status.reason.isNotBlank())
                assertTrue(native.libraryArenas.all { !it.scope().isAlive })
            }
        }
        CudaNativeTestRuntime().use { native ->
            native.deviceCount = 0
            assertFalse(NeuroCuda.detect(native.loader).available)
            native.deviceCount = 1
            native.errors["cudaGetDeviceCount"] = 7
            assertTrue(NeuroCuda.detect(native.loader).reason.contains("cudaGetDeviceCount"))
        }
    }

}
