package com.lis.neuro

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

class SmallCudaTrainingTest {
    @Test fun bothPrecisionsUseOneLaunchAndOneDownloadPerChunkWithResidentBuffers() {
        for (precision in Neuro.TrainingPrecision.entries) {
            val state = state()
            val driver = Driver()
            SmallCudaTraining(state, parameters(), precision, driver).use { kernel ->
                assertEquals(TrainingEngine.SMALL, kernel.info.engine)
                assertEquals(precision.name, kernel.info.precision)
                assertTrue(kernel.info.kernelVersion.contains("packed-fp64"))
                val result = kernel.train(orders(3), 11, false)
                assertState(state, result)
                assertEquals(1, driver.downloads)
                val launch = driver.launches.single()
                assertEquals(if (precision == Neuro.TrainingPrecision.FP64) "small_train_fp64" else "small_train_fp32", launch.first)
                assertEquals(128, launch.second)
                assertEquals(listOf(1, 4, 19, 3, 11), driver.arguments!!.slice(6..10))
                val allocations = driver.allocated
                kernel.train(orders(1), 7, false)
                assertEquals(allocations, driver.allocated)
                kernel.train(orders(64), 64, false)
                assertEquals(allocations + 1, driver.allocated)
                kernel.train(orders(1), 1, true)
                assertEquals(1, driver.arguments!!.last())
            }
            assertTrue(driver.memory.isEmpty())
            assertEquals(1, driver.closes)
        }
    }

    @Test fun cohortsShareImmutableDataAndLeaveInactiveModelsUntouched() {
        val first = state()
        val second = state().also { it.weights[0][0] += 0.25 }
        val driver = Driver()
        SmallCudaCohort(arrayOf(first, second), parameters(), Neuro.TrainingPrecision.FP64, driver).use { cohort ->
            val result = cohort.train(arrayOf(orders(2), emptyArray()), 11, false, booleanArrayOf(true, false))
            assertState(first, result[0]); assertState(second, result[1])
            assertEquals(256, driver.launches.single().second)
            val launches = driver.launches.size
            val dormant = cohort.train(arrayOf(emptyArray(), emptyArray()), 1, false, booleanArrayOf(false, false))
            assertState(second, dormant[1])
            assertEquals(launches, driver.launches.size)
            assertThrows(IllegalArgumentException::class.java) { cohort.train(arrayOf(orders(1)), 1, false) }
            assertThrows(IllegalArgumentException::class.java) { cohort.train(arrayOf(orders(1), orders(2)), 1, false) }
            assertThrows(IllegalArgumentException::class.java) { cohort.train(arrayOf(orders(1), orders(1)), 1, false, booleanArrayOf(true)) }
        }
        assertTrue(driver.memory.isEmpty())
        val changed = state().also { it.inputs[0] += 0.25 }
        val rejected = Driver()
        assertThrows(IllegalArgumentException::class.java) { SmallCudaCohort(arrayOf(first, changed), parameters(), Neuro.TrainingPrecision.FP64, rejected) }
        assertEquals(1, rejected.closes)
    }

    @Test fun badOrdersAndArgumentsRejectBeforeDispatchWithoutPoisoningTheKernel() {
        val driver = Driver()
        val kernel = SmallCudaTraining(state(), parameters(), Neuro.TrainingPrecision.FP64, driver)
        for ((order, batch, online) in listOf(
            Triple(emptyArray<IntArray>(), 1, false), Triple(orders(65), 1, false),
            Triple(orders(1), 0, false), Triple(orders(1), 2, true),
            Triple(arrayOf(intArrayOf(0)), 1, false), Triple(arrayOf(IntArray(19)), 1, false),
            Triple(arrayOf(IntArray(19) { if (it == 0) -1 else it }), 1, false))) {
            assertThrows(IllegalArgumentException::class.java) { kernel.train(order, batch, online) }
        }
        assertTrue(driver.launches.isEmpty())
        kernel.train(orders(1), 1, true)
        kernel.close(); kernel.close()
        assertThrows(IllegalStateException::class.java) { kernel.train(orders(1), 1, true) }
    }

    @Test fun failuresNeverReturnPartialStateAndCleanupAttemptsEveryAllocation() {
        for (operation in listOf("launch", "download", "nonfinite", "growth")) {
            val source = state()
            val before = source.weights.map { it.copyOf() }
            val driver = Driver()
            val kernel = SmallCudaTraining(source, parameters(), Neuro.TrainingPrecision.FP64, driver)
            kernel.train(orders(1), 7, false)
            driver.failure = operation
            assertThrows(IllegalStateException::class.java) { kernel.train(orders(2), 7, false) }
            assertThrows(IllegalStateException::class.java) { kernel.train(orders(1), 7, false) }
            source.weights.indices.forEach { assertArrayEquals(before[it], source.weights[it]) }
            driver.failure = ""
            kernel.close()
            assertTrue(driver.memory.isEmpty(), operation)
        }
        val failed = Driver().apply { failAllocation = 3 }
        assertThrows(IllegalStateException::class.java) { SmallCudaTraining(state(), parameters(), Neuro.TrainingPrecision.FP64, failed) }
        assertTrue(failed.memory.isEmpty()); assertEquals(1, failed.closes)
        val cleanup = Driver()
        val kernel = SmallCudaTraining(state(), parameters(), Neuro.TrainingPrecision.FP64, cleanup)
        cleanup.failure = "cleanup"
        val problem = assertThrows(IllegalStateException::class.java) { kernel.close() }
        assertTrue(problem.suppressed.isNotEmpty())
        assertTrue(cleanup.memory.isEmpty()); assertEquals(1, cleanup.closes)
        kernel.close()
    }

    @Test fun constructorsRejectMalformedStateNullAllocationsAndInsufficientMemory() {
        val malformed = listOf(
            state().copy(topology = intArrayOf(2, 3, 1)), state().copy(inputs = doubleArrayOf(1.0)),
            state().copy(targets = doubleArrayOf(1.0)), state().copy(weights = emptyArray()),
            state().also { it.weights[0][0] = Double.NaN }, state().also { it.targets[0] = Double.NaN },
            state().copy(biases = Array(4) { DoubleArray(0) }))
        for (state in malformed) {
            val driver = Driver()
            assertThrows(IllegalArgumentException::class.java) { SmallCudaTraining(state, parameters(), Neuro.TrainingPrecision.FP64, driver) }
            assertEquals(1, driver.closes)
        }
        for (driver in listOf(Driver().apply { nullAllocation = true }, Driver().apply { free = 1L })) {
            assertThrows(IllegalStateException::class.java) { SmallCudaTraining(state(), parameters(), Neuro.TrainingPrecision.FP64, driver) }
            assertEquals(1, driver.closes)
        }
        val empty = state().copy(inputs = doubleArrayOf(), targets = doubleArrayOf())
        SmallCudaTraining(empty, parameters(), Neuro.TrainingPrecision.FP64, Driver()).use {
            assertThrows(IllegalStateException::class.java) { it.train(arrayOf(intArrayOf()), 1, true) }
        }
        val driver = Driver()
        assertThrows(IllegalArgumentException::class.java) { SmallCudaCohort(emptyArray(), parameters(), Neuro.TrainingPrecision.FP64, driver) }
        assertEquals(1, driver.closes)
    }

    @Test fun deviceServiceRetainsHealthyDriversAndDiscardsBrokenLeases() {
        assertThrows(IllegalArgumentException::class.java) { SmallCudaDeviceService(0) }
        val drivers = ArrayList<Driver>()
        val service = SmallCudaDeviceService(1) { Driver().also { drivers += it } }
        val first = service.openDriver()
        val pointer = first.allocate(16)
        first.upload(pointer, doubleArrayOf(1.0, 2.0))
        first.download(pointer, DoubleArray(2))
        first.upload(pointer, intArrayOf(1, 2))
        first.launch("unused", 128, 1)
        first.synchronize(); first.availableMemory(); first.free(pointer)
        assertThrows(IllegalStateException::class.java) { service.openDriver() }
        assertThrows(IllegalStateException::class.java) { service.close() }
        first.close(); first.close()
        assertThrows(IllegalStateException::class.java) { first.allocate(8) }
        service.openDriver().use { assertEquals(first.info, it.info) }
        assertEquals(1, drivers.size); assertEquals(0, drivers[0].closes)
        service.openDriver().use { lease ->
            drivers[0].failure = "launch"
            assertThrows(IllegalStateException::class.java) { lease.launch("unused", 1) }
        }
        assertEquals(1, drivers[0].closes)
        service.openDriver().close()
        assertEquals(2, drivers.size)
        drivers[1].failure = "cleanup"
        assertThrows(IllegalStateException::class.java) { service.close() }
        service.close()
        assertThrows(IllegalStateException::class.java) { service.openDriver() }
    }

    @Test fun deviceServiceReusesExactBuffersAcrossCohortsAndFreesThemWhenClosed() {
        val driver = Driver()
        val service = SmallCudaDeviceService(1) { driver }
        var pointers = emptySet<Long>()
        repeat(2) { round ->
            SmallCudaCohort(arrayOf(state(), state()), parameters(), Neuro.TrainingPrecision.FP64, service.openDriver()).use { cohort ->
                val result = cohort.train(arrayOf(orders(2), orders(2)), 7, false)
                result.forEach { assertState(state(), it) }
                if (round == 0) pointers = driver.memory.keys.toSet() else assertEquals(pointers, driver.memory.keys)
                assertEquals(6, driver.allocated)
            }
            assertEquals(pointers, driver.memory.keys)
            assertEquals(0, driver.frees)
        }
        service.close()
        assertEquals(6, driver.frees)
        assertEquals(1, driver.closes)
        assertTrue(driver.memory.isEmpty())
    }

    @Test fun bufferPoolIsBoundedAndUnreleasedLiveBuffersAreDrained() {
        val driver = Driver()
        val service = SmallCudaDeviceService(1) { driver }
        service.openDriver().use { lease ->
            val pointers = List(66) { lease.allocate(8) }
            pointers.forEach(lease::free)
            assertEquals(2, driver.frees)
            assertEquals(64, driver.memory.size)
            val large = lease.allocate(2L * 1024 * 1024)
            lease.free(large)
            assertEquals(3, driver.frees)
        }
        service.openDriver().use { lease ->
            lease.allocate(8)
            lease.allocate(16)
            lease.launch("pending", 128)
            // close synchronizes and drains these live buffers even without explicit free calls.
        }
        assertTrue(driver.synchronizations > 0)
        assertEquals(64, driver.memory.size)
        service.close()
        assertTrue(driver.memory.isEmpty())

        val bytes = Driver()
        val bounded = SmallCudaDeviceService(1) { bytes }
        bounded.openDriver().use { lease ->
            val first = lease.allocate(700_000)
            val second = lease.allocate(700_000)
            lease.free(first); lease.free(second)
            assertEquals(1, bytes.memory.size)
            assertEquals(1, bytes.frees)
        }
        bounded.close()
        assertTrue(bytes.memory.isEmpty())
    }

    @Test fun brokenPooledDriversDrainEveryBufferAndPreserveCleanupFailures() {
        val driver = Driver()
        val service = SmallCudaDeviceService(1) { driver }
        service.openDriver().use { lease ->
            val buffer = lease.allocate(8)
            lease.free(buffer)
        }
        val lease = service.openDriver()
        lease.allocate(16)
        lease.allocate(24)
        lease.launch("pending", 128)
        driver.failure = "cleanup"
        val failure = assertThrows(IllegalStateException::class.java) { lease.close() }
        assertTrue(failure.suppressed.isNotEmpty())
        assertEquals(3, driver.frees)
        assertEquals(1, driver.closes)
        assertTrue(driver.memory.isEmpty())
        service.close()
    }

    @Test fun poolingSynchronizesBeforeReusingPendingStorageAndRejectsDoubleFree() {
        val driver = Driver()
        val service = SmallCudaDeviceService(1) { driver }
        service.openDriver().use { lease ->
            val first = lease.allocate(16)
            lease.launch("pending", 128)
            lease.free(first)
            assertEquals(1, driver.synchronizations)
            assertEquals(first, lease.allocate(16))
            lease.free(first)
            assertThrows(IllegalArgumentException::class.java) { lease.free(first) }
        }
        assertTrue(driver.memory.isEmpty())
        assertEquals(1, driver.closes)
        service.close()
    }

    @Test fun failedNativeFreeRemainsOwnedUntilLeaseCleanupRetriesIt() {
        val driver = Driver()
        val service = SmallCudaDeviceService(1) { driver }
        val lease = service.openDriver()
        val pooled = lease.allocate(8)
        lease.free(pooled)
        val large = lease.allocate(2L * 1024 * 1024)
        driver.failFreeOnce = true
        assertThrows(IllegalStateException::class.java) { lease.free(large) }
        assertTrue(driver.memory.containsKey(large))
        lease.close()
        assertEquals(3, driver.frees) // initial failed free, retried live allocation, retained pool
        assertEquals(1, driver.closes)
        assertTrue(driver.memory.isEmpty())
        service.close()
    }

    @Test fun chunksReusePrivateHostOrdersActivityAndDoubleBufferedDownloads() {
        val driver = Driver()
        SmallCudaTraining(state(), parameters(), Neuro.TrainingPrecision.FP64, driver).use { kernel ->
            val first = kernel.train(orders(2), 7, false)
            val orderBuffer = driver.intUploads[driver.intUploads.lastIndex - 1]
            val activeBuffer = driver.intUploads.last()
            kernel.train(orders(2), 7, false)
            assertSame(orderBuffer, driver.intUploads[driver.intUploads.lastIndex - 1])
            assertSame(activeBuffer, driver.intUploads.last())
            kernel.train(orders(2), 7, false)
            assertSame(driver.downloadBuffers[0], driver.downloadBuffers[2])
            assertNotSame(driver.downloadBuffers[0], driver.downloadBuffers[1])
            first.weights[0][0] = 99.0
            assertState(state(), kernel.train(orders(2), 7, false))
        }
    }

    private fun parameters() = Neuro.HyperParameters(0.11, 0.31, 0.75, 42, Neuro.Kernel.SCALAR)
    private fun state(): NeuroTrainingState = preparedSmall(intArrayOf(2, 8, 8, 8, 1), parameters()).exportTrainingState()
    private fun orders(epochs: Int): Array<IntArray> = smallOrders(epochs, 19)
    private fun assertState(expected: NeuroTrainingState, actual: NeuroTrainingState) = assertSmallState(expected, actual, 0.0)

    private class Driver : CudaDriver {
        override val info = TrainingDeviceInfo(TrainingBackend.CUDA, "fake GPU", "small-test-driver")
        val memory = LinkedHashMap<Long, Any>()
        val launches = ArrayList<Pair<String, Int>>()
        val intUploads = ArrayList<IntArray>()
        val downloadBuffers = ArrayList<DoubleArray>()
        var arguments: Array<out Any>? = null
        var allocated = 0
        var downloads = 0
        var closes = 0
        var frees = 0
        var synchronizations = 0
        var failure = ""
        var failAllocation = 0
        var failFreeOnce = false
        var nullAllocation = false
        var free = 1_000_000_000L
        override fun availableMemory() = free
        override fun allocate(bytes: Long): Long {
            allocated++
            check(allocated != failAllocation && failure != "growth") { "allocation failed" }
            if (nullAllocation) return 0L
            return allocated.toLong().also { memory[it] = DoubleArray((bytes / 8).toInt()) }
        }
        override fun free(pointer: Long) {
            frees++
            if (failFreeOnce) { failFreeOnce = false; error("first free failed") }
            memory.remove(pointer)
            check(failure != "cleanup") { "free failed" }
        }
        override fun upload(pointer: Long, values: DoubleArray) { memory[pointer] = values.copyOf() }
        override fun upload(pointer: Long, values: IntArray) { intUploads += values; memory[pointer] = values.copyOf() }
        override fun download(pointer: Long, values: DoubleArray) {
            downloads++
            downloadBuffers += values
            (memory.getValue(pointer) as DoubleArray).copyInto(values)
            if (failure == "nonfinite") values[0] = Double.NaN
            if (failure == "download") { values[0] = 123.0; error("partial download failed") }
        }
        override fun launch(name: String, workItems: Int, vararg arguments: Any) {
            check(failure != "launch") { "launch failed" }
            launches += name to workItems; this.arguments = arguments
        }
        override fun synchronize() { synchronizations++; check(failure != "cleanup") { "synchronize failed" } }
        override fun close() { closes++; check(failure != "cleanup") { "close failed" } }
    }
}

internal fun preparedSmall(shape: IntArray, parameters: Neuro.HyperParameters, samples: Int = 19): Neuro = Neuro(shape, parameters).also { model ->
    repeat(samples) { sample -> model.addTrainingSample(doubleArrayOf((sample % 7) / 7.0, (sample % 5) / 5.0), doubleArrayOf((sample % 2).toDouble())) }
}
internal fun smallOrders(epochs: Int, samples: Int): Array<IntArray> = Array(epochs) { epoch ->
    IntArray(samples) { index -> (samples - 1 - index + epoch * 3) % samples }
}
internal fun assertSmallState(expected: NeuroTrainingState, actual: NeuroTrainingState, tolerance: Double) {
    assertArrayEquals(expected.topology, actual.topology)
    val left = expected.weights + expected.biases + expected.weightVelocity + expected.biasVelocity
    val right = actual.weights + actual.biases + actual.weightVelocity + actual.biasVelocity
    for (index in left.indices) assertArrayEquals(left[index], right[index], tolerance, "Parameter/momentum buffer $index")
    assertArrayEquals(expected.inputs, actual.inputs); assertArrayEquals(expected.targets, actual.targets)
}
