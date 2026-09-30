package com.lis.neuro

import java.lang.foreign.Arena
import java.lang.foreign.FunctionDescriptor
import java.lang.foreign.MemorySegment
import java.lang.foreign.SymbolLookup
import java.lang.foreign.ValueLayout
import java.lang.invoke.MethodHandles
import java.util.Optional
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

class NeuroNativeAdaptersTest {
    @Test fun runtimeAndBuffersRoundTripAllPrimitiveTypesThroughFfm() {
        CudaNativeTestRuntime().use { fake ->
            val runtime = NeuroCudaRuntime.create(fake.loader)
            try {
                assertTrue(runtime.libraryName.contains("JNEURO_CUDA_RUNTIME"))
                assertEquals(1, runtime.deviceCount())
                runtime.setDevice(2)
                assertEquals(2, runtime.currentDevice())
                assertEquals(8 to 6, runtime.computeCapability(2))
                NeuroCudaBuffer.doubles(runtime, 3).use { buffer ->
                    val values = doubleArrayOf(-0.0, 0.25, 12345.75)
                    buffer.upload(values)
                    assertArrayEquals(values, buffer.downloadDoubles(3), 0.0)
                    buffer.upload(doubleArrayOf())
                    assertEquals(0, buffer.downloadDoubles(0).size)
                }
                NeuroCudaBuffer.floats(runtime, 3).use { buffer ->
                    val values = floatArrayOf(-0.0f, 0.25f, 123.75f)
                    buffer.upload(values)
                    assertArrayEquals(values, buffer.downloadFloats(3), 0.0f)
                    buffer.upload(floatArrayOf())
                    assertEquals(0, buffer.downloadFloats(0).size)
                }
                NeuroCudaBuffer.ints(runtime, 3).use { buffer ->
                    buffer.upload(intArrayOf(2, 0, 1))
                    assertArrayEquals(intArrayOf(2, 0, 1), fake.allocations.getValue(buffer.pointer.address()).toArray(ValueLayout.JAVA_INT))
                    buffer.upload(intArrayOf())
                }
                runtime.release(MemorySegment.NULL)
                runtime.synchronize()
                assertTrue(fake.allocations.isEmpty())
                assertThrows(IllegalArgumentException::class.java) { runtime.setDevice(-1) }
                assertThrows(IllegalArgumentException::class.java) { runtime.allocate(0) }
                Arena.ofConfined().use { arena ->
                    val host = arena.allocate(8)
                    assertThrows(IllegalArgumentException::class.java) { runtime.copyHostToDevice(MemorySegment.NULL, host, 9) }
                    assertThrows(IllegalArgumentException::class.java) { runtime.copyDeviceToHost(host, MemorySegment.NULL, -1) }
                }
            } finally { runtime.close(); runtime.close() }
            assertFalse(fake.libraryArenas.single().scope().isAlive)
        }
    }

    @Test fun runtimeReportsNativeFailures() {
        CudaNativeTestRuntime().use { fake ->
            NeuroCudaRuntime.create(fake.loader).use { runtime ->
                val allocation = runtime.allocate(8)
                Arena.ofConfined().use { arena ->
                    val host = arena.allocate(8)
                    val actions: List<Pair<String, () -> Unit>> = listOf(
                        "cudaGetDeviceCount" to { runtime.deviceCount() },
                        "cudaSetDevice" to { runtime.setDevice(0) },
                        "cudaGetDevice" to { runtime.currentDevice() },
                        "cudaDeviceGetAttribute" to { runtime.computeCapability(0) },
                        "cudaMalloc" to { runtime.allocate(8) },
                        "cudaMemcpy" to { runtime.copyHostToDevice(allocation, host, 8) },
                        "cudaDeviceSynchronize" to { runtime.synchronize() },
                        "cudaFree" to { runtime.release(allocation) })
                    for ((name, action) in actions) {
                        fake.errors[name] = 700
                        val failure = assertThrows(IllegalStateException::class.java) { action() }
                        assertTrue(failure.message!!.contains("700"))
                        fake.errors.remove(name)
                    }
                }
                runtime.release(allocation)
            }
        }
    }

    @Test fun factoriesHandleUnavailableAndIncompleteLibrariesWithoutLeakingArenas() {
        val factories: List<(NeuroLibraryLoader) -> AutoCloseable?> = listOf(
            NeuroCudaRuntime::tryCreate, NeuroCublas::tryCreate, NeuroNvrtc::tryCreate, NeuroCudaDriver::tryCreate)
        for (factory in factories) {
            for (linkFailure in listOf(false, true)) {
                var captured: Arena? = null
                val loader: NeuroLibraryLoader = { arena, _, _, _ ->
                    captured = arena
                    if (linkFailure) throw UnsatisfiedLinkError("test missing dependency")
                    else throw IllegalArgumentException("test missing library")
                }
                assertNull(factory(loader))
                assertFalse(captured!!.scope().isAlive)
            }
        }
        val unavailable: NeuroLibraryLoader = { _, _, _, _ -> throw IllegalArgumentException("unavailable") }
        assertThrows(IllegalStateException::class.java) { NeuroCudaRuntime.create(unavailable) }
        assertThrows(IllegalStateException::class.java) { NeuroCublas.create(unavailable) }
        assertThrows(IllegalStateException::class.java) { NeuroNvrtc.create(unavailable) }
        assertThrows(IllegalStateException::class.java) { NeuroCudaDriver.create(unavailable) }
        CudaNativeTestRuntime().use { fake ->
            fake.errors["cublasCreate_v2"] = 3
            assertNull(NeuroCublas.tryCreate(fake.loader))
            fake.errors["cuInit"] = 100
            assertNull(NeuroCudaDriver.tryCreate(fake.loader))
            fake.missing += "cudaMalloc"
            assertNull(NeuroCudaRuntime.tryCreate(fake.loader))
            assertTrue(fake.libraryArenas.all { !it.scope().isAlive })
        }
    }

    @Test fun cublasPreservesMatrixLayoutsScalarsPrecisionAndVersion() {
        CudaNativeTestRuntime().use { fake ->
            val cublas = NeuroCublas.create(fake.loader)
            val a = MemorySegment.ofAddress(101)
            val b = MemorySegment.ofAddress(102)
            val c = MemorySegment.ofAddress(103)
            try {
                assertTrue(cublas.libraryName.contains("JNEURO_CUBLAS"))
                assertEquals(130002, cublas.version())
                cublas.forward(a, b, c, 7, 3, 5)
                cublas.backward(a, b, c, 7, 3, 5)
                cublas.weightGradient(a, b, c, 7, 3, 5)
                cublas.forwardFloat(a, b, c, 7, 3, 5)
                cublas.backwardFloat(a, b, c, 7, 3, 5)
                cublas.weightGradientFloat(a, b, c, 7, 3, 5)
                assertEquals(6, fake.gemms.size)
                for ((index, call) in fake.gemms.withIndex()) {
                    assertEquals(index >= 3, call.single)
                    assertEquals(1.0, call.alpha)
                    assertEquals(0.0, call.beta)
                    assertEquals(103L, call.c)
                    val expected = when (index % 3) {
                        0 -> listOf(1, 0, 5, 7, 3, 3, 3, 5)
                        1 -> listOf(0, 0, 3, 7, 5, 3, 5, 3)
                        else -> listOf(0, 1, 3, 5, 7, 3, 5, 3)
                    }
                    assertEquals(expected, listOf(call.transposeA, call.transposeB, call.m, call.n, call.k, call.lda, call.ldb, call.ldc))
                    assertEquals(if (index % 3 == 2) 101L else 102L, call.a)
                    assertEquals(if (index % 3 == 2) 102L else 101L, call.b)
                }
                fake.errors["cublasDgemm_v2"] = 13
                assertThrows(IllegalStateException::class.java) { cublas.forward(a, b, c, 7, 3, 5) }
                fake.errors["cublasSgemm_v2"] = 13
                assertThrows(IllegalStateException::class.java) { cublas.forwardFloat(a, b, c, 7, 3, 5) }
                fake.errors["cublasGetVersion_v2"] = 3
                assertThrows(IllegalStateException::class.java) { cublas.version() }
            } finally { cublas.close(); cublas.close() }
            assertEquals(1, fake.calls.count { it == "cublasDestroy_v2" })
            assertFalse(fake.libraryArenas.single().scope().isAlive)
        }
        CudaNativeTestRuntime().use { fake ->
            fake.missing += "cublasGetVersion_v2"
            NeuroCublas.create(fake.loader).use { assertNull(it.version()) }
        }
        CudaNativeTestRuntime().use { fake ->
            val cublas = NeuroCublas.create(fake.loader)
            fake.errors["cublasDestroy_v2"] = 13
            assertThrows(IllegalStateException::class.java) { cublas.close() }
            cublas.close()
            assertFalse(fake.libraryArenas.single().scope().isAlive)
        }
    }

    @Test fun driverTerminatesModuleImagesAndMarshalsAllKernelArgumentTypes() {
        CudaNativeTestRuntime().use { fake ->
            val driver = NeuroCudaDriver.create(fake.loader)
            try {
                assertTrue(driver.libraryName.contains("JNEURO_CUDA_DRIVER"))
                assertEquals("Adapter test GPU", driver.deviceName(0))
                val module = driver.loadModule("raw PTX without terminator".toByteArray())
                assertEquals("raw PTX without terminator", fake.loadedImage)
                fake.functionSignatures["test"] = listOf(ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_FLOAT)
                val function = module.function("test")
                driver.launch(function, Int.MAX_VALUE, NeuroCudaDriver.Argument.Pointer(MemorySegment.ofAddress(123)),
                    NeuroCudaDriver.Argument.IntValue(7), NeuroCudaDriver.Argument.DoubleValue(0.25), NeuroCudaDriver.Argument.FloatValue(0.5f))
                val launch = fake.launches.single()
                assertEquals(listOf(8_388_608, 1, 1, 256, 1, 1, 0), launch.dimensions)
                assertEquals(listOf(123L, 7, 0.25, 0.5f), launch.arguments)
                assertEquals(0L, launch.stream)
                driver.launch(function, 0)
                assertEquals(1, fake.launches.size)
                assertThrows(IllegalArgumentException::class.java) { driver.launch(function, -1) }
                assertThrows(IllegalArgumentException::class.java) { driver.loadModule(byteArrayOf()) }
                module.close(); module.close()
                assertThrows(IllegalStateException::class.java) { module.function("test") }
                assertEquals(1, fake.calls.count { it == "cuModuleUnload" })
            } finally { driver.close(); driver.close() }
        }
    }

    @Test fun driverSurfacesDeviceModuleFunctionAndLaunchErrors() {
        CudaNativeTestRuntime().use { fake ->
            NeuroCudaDriver.create(fake.loader).use { driver ->
                val module = driver.loadModule(fake.ptx)
                val function = module.function("test")
                val actions: List<Pair<String, () -> Unit>> = listOf(
                    "cuDeviceGet" to { driver.deviceName(0) }, "cuDeviceGetName" to { driver.deviceName(0) },
                    "cuModuleLoadData" to { driver.loadModule(fake.ptx) },
                    "cuModuleGetFunction" to { module.function("test") },
                    "cuLaunchKernel" to { driver.launch(function, 1) }, "cuModuleUnload" to { module.close() })
                for ((name, action) in actions) {
                    fake.errors[name] = 500
                    assertThrows(IllegalStateException::class.java) { action() }
                    fake.errors.remove(name)
                }
                module.close()
            }
        }
    }

    @Test fun nativeLibraryRespectsOverridesFallbacksAndRequiredSymbols() {
        CudaNativeTestRuntime().use { fake ->
            Arena.ofConfined().use { arena ->
                val tried = ArrayList<String>()
                val loaded = NeuroNativeLibrary.open(arena, "TEST", listOf("override", "broken", "linked", "valid"), listOf("cuInit"),
                    { "  override  " }, { name, lifetime ->
                        tried += name
                        when (name) {
                            "override" -> SymbolLookup { Optional.empty() }
                            "broken" -> throw IllegalArgumentException("not found")
                            "linked" -> throw UnsatisfiedLinkError("dependency not found")
                            else -> fake.lookup(lifetime)
                        }
                    })
                assertEquals(listOf("override", "broken", "linked", "valid"), tried)
                assertEquals("valid", loaded.name)
                val init = NeuroNativeLibrary.downcall(loaded, "cuInit", FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT))
                assertEquals(0, NeuroNativeLibrary.invokeInt(init, 0))
                assertThrows(IllegalStateException::class.java) {
                    NeuroNativeLibrary.downcall(loaded, "absent", FunctionDescriptor.of(ValueLayout.JAVA_INT))
                }
                val failure = assertThrows(IllegalStateException::class.java) {
                    NeuroNativeLibrary.open(arena, "TEST", listOf("none"), listOf("cuInit"), { null },
                        { _, _ -> throw UnsatisfiedLinkError("missing dependency") })
                }
                assertInstanceOf(UnsatisfiedLinkError::class.java, failure.cause)
                val wrongReturnType = MethodHandles.constant(String::class.java, "not an int")
                assertThrows(IllegalStateException::class.java) { NeuroNativeLibrary.invokeInt(wrongReturnType) }
            }
        }
    }
}
