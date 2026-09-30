package com.lis.neuro

import java.lang.foreign.MemorySegment
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

class NeuroNvrtcKernelsTest {
    @Test fun compilerReturnsPtxAndAlwaysDestroysSuccessfulPrograms() {
        CudaNativeTestRuntime().use { fake ->
            val compiler = NeuroNvrtc.create(fake.loader)
            try {
                assertTrue(compiler.libraryName.contains("JNEURO_NVRTC"))
                assertEquals(12 to 0, compiler.version())
                assertArrayEquals(fake.ptx, compiler.compile("source", "test.cu", 8 to 6))
                assertEquals(listOf("--gpu-architecture=compute_86"), fake.compiledOptions)
                assertEquals(NeuroNvrtc.compilerOptions(8 to 6), fake.compiledOptions)
                assertEquals(1, fake.calls.count { it == "nvrtcDestroyProgram" })
                fake.errors["nvrtcVersion"] = 4
                assertThrows(IllegalStateException::class.java) { compiler.version() }
            } finally { compiler.close(); compiler.close() }
            assertFalse(fake.libraryArenas.single().scope().isAlive)
        }
    }

    @Test fun compileFailuresRetainDiagnosticsAndSuppressDestroyFailure() {
        CudaNativeTestRuntime().use { fake ->
            NeuroNvrtc.create(fake.loader).use { compiler ->
                fake.errors["nvrtcCompileProgram"] = 6
                fake.errors["nvrtcDestroyProgram"] = 4
                val failure = assertThrows(IllegalStateException::class.java) { compiler.compile("broken", "test.cu", 8 to 6) }
                assertTrue(failure.message!!.contains("nvrtcCompileProgram failed with status 6"))
                assertTrue(failure.message!!.contains("test compiler diagnostic"))
                assertEquals(1, failure.suppressed.size)
                assertTrue(failure.suppressed.single().message!!.contains("nvrtcDestroyProgram"))
            }
        }
        CudaNativeTestRuntime().use { fake ->
            NeuroNvrtc.create(fake.loader).use { compiler ->
                fake.errors["nvrtcDestroyProgram"] = 4
                val failure = assertThrows(IllegalStateException::class.java) { compiler.compile("ok", "test.cu", 8 to 6) }
                assertTrue(failure.message!!.contains("nvrtcDestroyProgram"))
            }
        }
    }

    @Test fun missingOrMalformedCompilerLogsDoNotHideTheCompileError() {
        for (mode in listOf("size error", "read error", "empty", "oversized")) {
            CudaNativeTestRuntime().use { fake ->
                fake.errors["nvrtcCompileProgram"] = 6
                when (mode) {
                    "size error" -> fake.errors["nvrtcGetProgramLogSize"] = 4
                    "read error" -> fake.errors["nvrtcGetProgramLog"] = 4
                    "empty" -> fake.logSize = 1
                    "oversized" -> fake.logSize = Int.MAX_VALUE.toLong() + 1
                }
                NeuroNvrtc.create(fake.loader).use { compiler ->
                    val failure = assertThrows(IllegalStateException::class.java) { compiler.compile("broken", "test.cu", 8 to 6) }
                    assertTrue(failure.message!!.contains("nvrtcCompileProgram failed with status 6"))
                    assertEquals(mode.endsWith("error"), failure.message!!.contains("log unavailable"))
                    assertEquals(1, fake.calls.count { it == "nvrtcDestroyProgram" })
                }
            }
        }
    }

    @Test fun programCreationPtxReadAndPtxBoundsErrorsCleanUpCorrectly() {
        for (operation in listOf("nvrtcCreateProgram", "nvrtcGetPTXSize", "nvrtcGetPTX")) {
            CudaNativeTestRuntime().use { fake ->
                fake.errors[operation] = 4
                NeuroNvrtc.create(fake.loader).use { compiler ->
                    val failure = assertThrows(IllegalStateException::class.java) { compiler.compile("source", "test.cu", 8 to 6) }
                    assertTrue(failure.message!!.contains(operation))
                    assertEquals(if (operation == "nvrtcCreateProgram") 0 else 1, fake.calls.count { it == "nvrtcDestroyProgram" })
                }
            }
        }
        for (size in listOf(0L, Int.MAX_VALUE.toLong() + 1)) {
            CudaNativeTestRuntime().use { fake ->
                fake.ptxSize = size
                NeuroNvrtc.create(fake.loader).use { compiler ->
                    assertThrows(IllegalArgumentException::class.java) { compiler.compile("source", "test.cu", 8 to 6) }
                    assertEquals(1, fake.calls.count { it == "nvrtcDestroyProgram" })
                }
            }
        }
    }

    @Test fun allKernelWrappersMarshalCountsScalarsAndPrecision() {
        CudaNativeTestRuntime().use { fake ->
            NeuroNvrtc.create(fake.loader).use { compiler ->
                NeuroCudaDriver.create(fake.loader).use { driver ->
                    val kernels = NeuroCudaKernels(compiler, driver, 8 to 6)
                    val a = MemorySegment.ofAddress(101)
                    val b = MemorySegment.ofAddress(102)
                    val c = MemorySegment.ofAddress(103)
                    try {
                        kernels.gatherRows(a, 3, b, 7, 5, c)
                        kernels.activate(a, b, 5, 3, 1.25, Neuro.SigmoidMode.EXACT)
                        kernels.activate(a, b, 5, 3, 1.25, Neuro.SigmoidMode.FAST)
                        kernels.outputDelta(a, b, c, 15, 1.25)
                        kernels.applyDerivative(a, b, 15, 1.25)
                        kernels.reduceBiasGradient(a, b, 5, 3)
                        kernels.momentumUpdate(a, b, c, 15, 0.2, 0.1)
                        kernels.gatherRowsFloat(a, 3, b, 7, 5, c)
                        kernels.activateFloat(a, b, 5, 3, 1.25f, Neuro.SigmoidMode.EXACT)
                        kernels.activateFloat(a, b, 5, 3, 1.25f, Neuro.SigmoidMode.FAST)
                        kernels.outputDeltaFloat(a, b, c, 15, 1.25f)
                        kernels.applyDerivativeFloat(a, b, 15, 1.25f)
                        kernels.reduceBiasGradientFloat(a, b, 5, 3)
                        kernels.momentumUpdateFloat(a, b, c, 15, 0.2f, 0.1f)
                        assertEquals(12, fake.functionNames.size, "Every function must be resolved once")
                        assertEquals(14, fake.launches.size)
                        assertTrue(fake.launches.all { it.stream == 0L && it.dimensions == listOf(1, 1, 1, 256, 1, 1, 0) })
                        val expected: List<List<Any>> = listOf(
                            listOf<Any>(101L, 3, 102L, 7, 5, 103L), listOf<Any>(101L, 102L, 5, 3, 1.25, 0),
                            listOf<Any>(101L, 102L, 5, 3, 1.25, 1), listOf<Any>(101L, 102L, 103L, 15, 1.25),
                            listOf<Any>(101L, 102L, 15, 1.25), listOf<Any>(101L, 102L, 5, 3),
                            listOf<Any>(101L, 102L, 103L, 15, 0.2, 0.1),
                            listOf<Any>(101L, 3, 102L, 7, 5, 103L), listOf<Any>(101L, 102L, 5, 3, 1.25f, 0),
                            listOf<Any>(101L, 102L, 5, 3, 1.25f, 1), listOf<Any>(101L, 102L, 103L, 15, 1.25f),
                            listOf<Any>(101L, 102L, 15, 1.25f), listOf<Any>(101L, 102L, 5, 3),
                            listOf<Any>(101L, 102L, 103L, 15, 0.2f, 0.1f))
                        assertEquals(expected, fake.launches.map { it.arguments })
                    } finally { kernels.close(); kernels.close() }
                    assertEquals(1, fake.calls.count { it == "cuModuleUnload" })
                }
            }
        }
    }

    @Test fun missingKernelUnloadsPartiallyInitializedModuleAndRetainsPrimaryError() {
        for (cleanupFails in listOf(false, true)) {
            CudaNativeTestRuntime().use { fake ->
                fake.failedFunction = "outputDelta"
                if (cleanupFails) fake.errors["cuModuleUnload"] = 400
                NeuroNvrtc.create(fake.loader).use { compiler ->
                    NeuroCudaDriver.create(fake.loader).use { driver ->
                        val failure = assertThrows(IllegalStateException::class.java) { NeuroCudaKernels(compiler, driver, 8 to 6) }
                        assertTrue(failure.message!!.contains("cuModuleGetFunction"))
                        assertEquals(if (cleanupFails) 1 else 0, failure.suppressed.size)
                        assertEquals(1, fake.calls.count { it == "cuModuleUnload" })
                    }
                }
            }
        }
    }
}
