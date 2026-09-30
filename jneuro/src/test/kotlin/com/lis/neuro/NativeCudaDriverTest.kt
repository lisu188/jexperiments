package com.lis.neuro

import java.lang.foreign.Arena
import java.lang.foreign.FunctionDescriptor
import java.lang.foreign.Linker
import java.lang.foreign.MemoryLayout
import java.lang.foreign.MemorySegment
import java.lang.foreign.SymbolLookup
import java.lang.foreign.ValueLayout
import java.lang.invoke.MethodHandles
import java.lang.invoke.MethodType
import java.security.MessageDigest
import java.util.Optional
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

/** Exercises the production downcalls against native upcall stubs, without a GPU or compiler. */
class NativeCudaDriverTest {
    @Test fun fusedCohortLaunchMarshalsPackedStateAndUsesOneBlockPerModel() {
        Runtime().use { runtime ->
            runtime.kernelArguments = List(6) { ValueLayout.JAVA_LONG } + List(5) { ValueLayout.JAVA_INT } +
                List(3) { ValueLayout.JAVA_DOUBLE } + List(2) { ValueLayout.JAVA_INT }
            runtime.driver().use { driver ->
                for (kernel in listOf("small_train_fp64", "small_train_fp32")) {
                    driver.launch(kernel, 3 * 128, 1L, 2L, 3L, 4L, 5L, 6L,
                        3, 5, 19, 64, 11, 0.11, 0.31, 0.75, 1, 0)
                }
                assertEquals(listOf("small_train_fp64", "small_train_fp32"), runtime.requestedFunctions)
                assertTrue(runtime.launches.all { it.dimensions == listOf(3, 1, 1, 128, 1, 1, 0) })
                assertEquals(listOf(3, 5, 19, 64, 11), runtime.launches.first().arguments.slice(6..10))
                assertEquals(listOf(0.11, 0.31, 0.75), runtime.launches.first().arguments.slice(11..13))
            }
            runtime.assertCallbacksSucceeded()
        }
    }

    @Test fun logsDeviceProvenanceAndBoundedTransferMetadataThroughFfm() {
        NativeLogCapture().use { logs ->
            Runtime().use { runtime ->
                runtime.driver().use { driver ->
                    val buffer = driver.allocate(16)
                    driver.upload(buffer, doubleArrayOf(9876543.25, -1234567.75))
                    driver.download(buffer, DoubleArray(2))
                    driver.synchronize()
                    driver.free(buffer)
                }
                val ready = logs.events("driver.ready").single()
                assertEquals("FFM test GPU", logs.fields(ready)["device"])
                assertEquals("retained-primary", logs.fields(ready)["context"])
                assertEquals(2, logs.events("transfer.submitted").size)
                assertTrue(logs.events("transfer.submitted").all { logs.fields(it)["bytes"] == 16L })
                assertTrue(logs.events("stream.synchronized").isNotEmpty())
                assertEquals(1, logs.events("driver.closed").size)
                assertFalse(logs.records.any { logs.fields(it).toString().contains("9876543.25") })
                runtime.assertCallbacksSucceeded()
            }
        }
    }

    @Test fun discoversDeviceAndRoundTripsNativeBuffers() {
        Runtime().use { runtime ->
            runtime.driver().use { driver ->
                assertEquals(TrainingBackend.CUDA, driver.info.backend)
                assertEquals("FFM test GPU", driver.info.name)
                assertEquals("000102030405060708090a0b0c0d0e0f/driver-13000", driver.info.identity)
                assertEquals("FP64", driver.info.precision)
                val hash = MessageDigest.getInstance("SHA-256").digest(PTX).joinToString("") { "%02x".format(it) }
                assertEquals(hash, driver.info.kernelVersion)
                assertEquals(0, runtime.streamFlags, "Kernel stream must synchronize with legacy-stream HtoD copies")
                assertEquals(String(PTX, Charsets.UTF_8), runtime.loadedPtx)
                assertEquals(64L * 1024 * 1024, driver.availableMemory())

                val pointer = driver.allocate(32)
                val values = doubleArrayOf(-0.0, 0.25, -123.5, Double.MAX_VALUE)
                driver.upload(pointer, values)
                val downloaded = DoubleArray(values.size)
                driver.download(pointer, downloaded)
                assertArrayEquals(values, downloaded, 0.0)
                assertEquals(java.lang.Double.doubleToRawLongBits(-0.0), java.lang.Double.doubleToRawLongBits(downloaded[0]))

                val order = driver.allocate(12)
                driver.upload(order, intArrayOf(3, 1, 2))
                assertArrayEquals(intArrayOf(3, 1, 2), runtime.allocations.getValue(order).toArray(ValueLayout.JAVA_INT))
                driver.synchronize()
                driver.free(order)
                driver.free(pointer)
                assertTrue(runtime.allocations.isEmpty())
            }
            assertEquals(listOf("cuCtxSetCurrent", "cuStreamSynchronize", "cuStreamDestroy_v2", "cuModuleUnload",
                "cuDevicePrimaryCtxRelease_v2"), runtime.calls.takeLast(5))
            runtime.assertCallbacksSucceeded()
        }
    }

    @Test fun marshalsKernelArgumentsAndCachesFunctionLookup() {
        Runtime().use { runtime ->
            runtime.kernelArguments = listOf(ValueLayout.JAVA_LONG, ValueLayout.JAVA_INT, ValueLayout.JAVA_DOUBLE)
            runtime.driver().use { driver ->
                driver.launch("forward", 129, 0x1020304050607080L, 17, -0.125)
                driver.launch("forward", 128, 9L, 3, 0.5)
                assertEquals(1, runtime.calls.count { it == "cuModuleGetFunction" })
                assertEquals(listOf("forward"), runtime.requestedFunctions)
                val launch = runtime.launches.first()
                assertEquals(listOf(2, 1, 1, 128, 1, 1, 0), launch.dimensions)
                assertEquals(listOf(0x1020304050607080L, 17, -0.125), launch.arguments)
                assertEquals(0x2000L, launch.stream)
                assertEquals(0L, launch.extra)
                assertEquals(1, runtime.launches.last().dimensions.first())
                driver.launch("update", 1, 1L, 0, 1.0)
                assertEquals(listOf("forward", "update"), runtime.requestedFunctions)
                assertEquals(2, runtime.calls.count { it == "cuModuleGetFunction" })
            }
            runtime.assertCallbacksSucceeded()
        }
    }

    @Test fun rejectsUnsupportedKernelArgumentBeforeNativeLaunch() {
        Runtime().use { runtime ->
            runtime.driver().use { driver ->
                val failure = assertThrows(IllegalArgumentException::class.java) { driver.launch("forward", 1, "invalid") }
                assertTrue(failure.message!!.contains("java.lang.String"))
                assertFalse("cuLaunchKernel" in runtime.calls)
            }
            runtime.assertCallbacksSucceeded()
        }
    }

    @Test fun rejectsOldDevicesBeforeRetainingAContext() {
        Runtime().use { runtime ->
            runtime.capability = 74
            val failure = assertThrows(IllegalStateException::class.java) { runtime.driver() }
            assertTrue(failure.message!!.contains("compute capability 7.5"))
            assertFalse("cuDevicePrimaryCtxRetain" in runtime.calls)
            runtime.assertCallbacksSucceeded()
        }
    }

    @Test fun reportsMissingDriverAndSymbolsClearly() {
        val missingDriver = assertThrows(IllegalStateException::class.java) {
            NativeCudaDriver(driverLookup = { throw IllegalArgumentException("driver library absent") }, ptxBytes = { PTX })
        }
        assertTrue(missingDriver.message!!.contains("driver library absent"))
        Runtime().use { runtime ->
            runtime.missing += "cuInit"
            val failure = assertThrows(IllegalStateException::class.java) { runtime.driver() }
            assertTrue(failure.message!!.contains("CUDA driver lacks cuInit"))
            assertTrue(runtime.calls.isEmpty())
        }
    }

    @Test fun missingPtxReleasesStreamAndContext() {
        Runtime().use { runtime ->
            val failure = assertThrows(IllegalStateException::class.java) {
                NativeCudaDriver(driverLookup = runtime::lookup, ptxBytes = { null })
            }
            assertTrue(failure.message!!.contains("Packaged CUDA kernels are missing"))
            assertFalse("cuModuleLoadData" in runtime.calls)
            assertTrue("cuStreamDestroy_v2" in runtime.calls)
            assertTrue("cuDevicePrimaryCtxRelease_v2" in runtime.calls)
            runtime.assertCallbacksSucceeded()
        }
    }

    @Test fun setupFailuresReleaseOnlyResourcesAlreadyAcquired() {
        for (step in listOf("cuInit", "cuDeviceGet", "cuDeviceComputeCapability", "cuDeviceGetName", "cuDeviceGetUuid",
            "cuDriverGetVersion", "cuDevicePrimaryCtxRetain", "cuCtxSetCurrent", "cuStreamCreate", "cuModuleLoadData")) {
            Runtime().use { runtime ->
                runtime.errors[step] = 701
                val failure = assertThrows(IllegalStateException::class.java) { runtime.driver() }
                assertTrue(failure.message!!.contains("$step failed (CUDA error 701)"), step)
                val retained = step in listOf("cuCtxSetCurrent", "cuStreamCreate", "cuModuleLoadData")
                assertEquals(retained, "cuDevicePrimaryCtxRelease_v2" in runtime.calls, step)
                assertEquals(step == "cuModuleLoadData", "cuStreamDestroy_v2" in runtime.calls, step)
                assertFalse("cuModuleUnload" in runtime.calls, step)
                if (step == "cuCtxSetCurrent") assertEquals(1, failure.cause!!.suppressed.size)
                runtime.assertCallbacksSucceeded()
            }
        }
    }

    @Test fun reportsAllocationCopyLaunchAndSynchronizationFailures() {
        Runtime().use { runtime ->
            runtime.driver().use { driver ->
                val pointer = driver.allocate(8)
                val operations: List<Pair<String, () -> Unit>> = listOf(
                    "cuMemGetInfo_v2" to { assertTrue(driver.availableMemory() >= 0) },
                    "cuMemAlloc_v2" to { assertTrue(driver.allocate(8) != 0L) },
                    "cuMemcpyHtoD_v2" to { driver.upload(pointer, doubleArrayOf(1.0)) },
                    "cuMemcpyDtoH_v2" to { driver.download(pointer, DoubleArray(1)) },
                    "cuModuleGetFunction" to { driver.launch("missing", 1) },
                    "cuLaunchKernel" to { driver.launch("forward", 1) },
                    "cuStreamSynchronize" to { driver.synchronize() },
                    "cuMemFree_v2" to { driver.free(pointer) }
                )
                for ((name, operation) in operations) {
                    runtime.errors[name] = 2
                    val failure = assertThrows(IllegalStateException::class.java) { operation() }
                    assertTrue(failure.message!!.contains("$name failed (CUDA error 2)"))
                    runtime.errors.remove(name)
                }
                driver.free(pointer)
            }
            runtime.assertCallbacksSucceeded()
        }
    }

    @Test fun closeReleasesRemainingResourcesDespiteFailuresAndIsIdempotent() {
        Runtime().use { runtime ->
            val driver = runtime.driver()
            for (name in listOf("cuCtxSetCurrent", "cuStreamSynchronize", "cuStreamDestroy_v2", "cuModuleUnload",
                "cuDevicePrimaryCtxRelease_v2")) runtime.errors[name] = 999
            val failure = assertThrows(IllegalStateException::class.java) { driver.close() }
            assertTrue(failure.message!!.contains("cuCtxSetCurrent"))
            assertEquals(4, failure.suppressed.size)
            assertEquals(listOf("cuStreamSynchronize", "cuStreamDestroy_v2", "cuModuleUnload", "cuDevicePrimaryCtxRelease_v2"),
                failure.suppressed.map { it.message!!.substringBefore(" failed") })
            val count = runtime.calls.size
            driver.close()
            assertEquals(count, runtime.calls.size)
            runtime.assertCallbacksSucceeded()
        }
    }

    @Test fun successfulCloseIsIdempotent() {
        Runtime().use { runtime ->
            val driver = runtime.driver()
            driver.close()
            val count = runtime.calls.size
            driver.close()
            assertEquals(count, runtime.calls.size)
            assertEquals(1, runtime.calls.count { it == "cuDevicePrimaryCtxRelease_v2" })
            runtime.assertCallbacksSucceeded()
        }
    }

    private data class Launch(val dimensions: List<Int>, val arguments: List<Any>, val stream: Long, val extra: Long)

    private class Callback(private val action: (Array<Any>) -> Int) {
        fun invoke(arguments: Array<Any>): Int = action(arguments)
    }

    private class Runtime : AutoCloseable {
        private val memory = Arena.ofShared()
        private val callbackFailures = ArrayList<Throwable>()
        val calls = ArrayList<String>()
        val errors = HashMap<String, Int>()
        val missing = HashSet<String>()
        val allocations = HashMap<Long, MemorySegment>()
        val requestedFunctions = ArrayList<String>()
        val launches = ArrayList<Launch>()
        var kernelArguments: List<ValueLayout> = emptyList()
        var capability = 75
        var streamFlags = -1
        var loadedPtx = ""
        private var nextPointer = 0x100000L

        fun driver(): NativeCudaDriver = NativeCudaDriver(driverLookup = ::lookup, ptxBytes = { PTX })

        fun lookup(arena: Arena): SymbolLookup {
            val symbols = HashMap<String, MemorySegment>()
            for ((name, layouts) in SIGNATURES) {
                if (name in missing) continue
                val callback = Callback { arguments ->
                    try { call(name, arguments) } catch (failure: Throwable) {
                        // An uncaught exception across an upcall terminates the JVM.
                        callbackFailures += failure
                        9999
                    }
                }
                val descriptor = FunctionDescriptor.of(ValueLayout.JAVA_INT, *layouts.toTypedArray())
                val handle = MethodHandles.lookup().findVirtual(Callback::class.java, "invoke",
                    MethodType.methodType(Int::class.javaPrimitiveType, Array<Any>::class.java))
                    .bindTo(callback).asCollector(Array<Any>::class.java, layouts.size).asType(descriptor.toMethodType())
                symbols[name] = Linker.nativeLinker().upcallStub(handle, descriptor, arena)
            }
            return SymbolLookup { name -> Optional.ofNullable(symbols[name]) }
        }

        private fun call(name: String, arguments: Array<Any>): Int {
            calls += name
            errors[name]?.let { return it }
            fun pointer(index: Int, size: Long) = (arguments[index] as MemorySegment).reinterpret(size)
            fun outputInt(index: Int, value: Int) = pointer(index, 4).set(ValueLayout.JAVA_INT, 0, value)
            fun outputLong(index: Int, value: Long) = pointer(index, 8).set(ValueLayout.JAVA_LONG, 0, value)
            fun outputAddress(index: Int, value: Long) = pointer(index, 8).set(ValueLayout.ADDRESS, 0, MemorySegment.ofAddress(value))
            when (name) {
                "cuDeviceGet" -> outputInt(0, 7)
                "cuDeviceComputeCapability" -> { outputInt(0, capability / 10); outputInt(1, capability % 10) }
                "cuDeviceGetName" -> pointer(0, (arguments[1] as Int).toLong()).setString(0, "FFM test GPU")
                "cuDeviceGetUuid" -> for (index in 0L until 16L) pointer(0, 16).set(ValueLayout.JAVA_BYTE, index, index.toByte())
                "cuDriverGetVersion" -> outputInt(0, 13000)
                "cuDevicePrimaryCtxRetain" -> outputAddress(0, 0x1000L)
                "cuStreamCreate" -> { streamFlags = arguments[1] as Int; outputAddress(0, 0x2000L) }
                "cuModuleLoadData" -> {
                    loadedPtx = pointer(1, PTX.size.toLong() + 1).getString(0)
                    outputAddress(0, 0x3000L)
                }
                "cuMemGetInfo_v2" -> { outputLong(0, 64L * 1024 * 1024); outputLong(1, 128L * 1024 * 1024) }
                "cuMemAlloc_v2" -> {
                    val id = nextPointer++
                    allocations[id] = memory.allocate(arguments[1] as Long, 8)
                    outputLong(0, id)
                }
                "cuMemFree_v2" -> checkNotNull(allocations.remove(arguments[0] as Long))
                "cuMemcpyHtoD_v2" -> {
                    val bytes = arguments[2] as Long
                    MemorySegment.copy(pointer(1, bytes), 0, allocations.getValue(arguments[0] as Long), 0, bytes)
                }
                "cuMemcpyDtoH_v2" -> {
                    val bytes = arguments[2] as Long
                    MemorySegment.copy(allocations.getValue(arguments[1] as Long), 0, pointer(0, bytes), 0, bytes)
                }
                "cuModuleGetFunction" -> {
                    requestedFunctions += pointer(2, 256).getString(0)
                    outputAddress(0, 0x4000L + requestedFunctions.size)
                }
                "cuLaunchKernel" -> {
                    val parameters = pointer(9, kernelArguments.size * 8L)
                    val values = kernelArguments.mapIndexed { index, layout ->
                        val value = parameters.getAtIndex(ValueLayout.ADDRESS, index.toLong()).reinterpret(layout.byteSize())
                        when (layout) {
                            ValueLayout.JAVA_LONG -> value.get(ValueLayout.JAVA_LONG, 0)
                            ValueLayout.JAVA_INT -> value.get(ValueLayout.JAVA_INT, 0)
                            ValueLayout.JAVA_DOUBLE -> value.get(ValueLayout.JAVA_DOUBLE, 0)
                            else -> error("Unexpected test argument layout")
                        }
                    }
                    launches += Launch((1..7).map { arguments[it] as Int }, values,
                        (arguments[8] as MemorySegment).address(), (arguments[10] as MemorySegment).address())
                }
            }
            return 0
        }

        fun assertCallbacksSucceeded() = assertTrue(callbackFailures.isEmpty(), callbackFailures.joinToString { it.toString() })
        override fun close() = memory.close()
    }

    companion object {
        private val PTX = "// test resource, never executed by a CUDA device\n".toByteArray(Charsets.UTF_8)
        private val I = ValueLayout.JAVA_INT
        private val L = ValueLayout.JAVA_LONG
        private val A = ValueLayout.ADDRESS
        private val SIGNATURES: Map<String, List<MemoryLayout>> = mapOf(
            "cuInit" to listOf(I),
            "cuDeviceGet" to listOf(A, I),
            "cuDeviceComputeCapability" to listOf(A, A, I),
            "cuDeviceGetName" to listOf(A, I, I),
            "cuDeviceGetUuid" to listOf(A, I),
            "cuDriverGetVersion" to listOf(A),
            "cuDevicePrimaryCtxRetain" to listOf(A, I),
            "cuCtxSetCurrent" to listOf(A),
            "cuStreamCreate" to listOf(A, I),
            "cuModuleLoadData" to listOf(A, A),
            "cuMemGetInfo_v2" to listOf(A, A),
            "cuMemAlloc_v2" to listOf(A, L),
            "cuMemFree_v2" to listOf(L),
            "cuMemcpyHtoD_v2" to listOf(L, A, L),
            "cuMemcpyDtoH_v2" to listOf(A, L, L),
            "cuModuleGetFunction" to listOf(A, A, A),
            "cuLaunchKernel" to listOf(A, I, I, I, I, I, I, I, A, A, A),
            "cuStreamSynchronize" to listOf(A),
            "cuStreamDestroy_v2" to listOf(A),
            "cuModuleUnload" to listOf(A),
            "cuDevicePrimaryCtxRelease_v2" to listOf(I)
        )
    }
}
