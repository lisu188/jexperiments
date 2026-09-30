package com.lis.neuro

import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference
import java.util.concurrent.locks.LockSupport
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

class CudaTrainingTest {
    @Test fun logsReservationAndCommittedEpochWithoutClaimingFailedDownloads() {
        NativeLogCapture().use { logs ->
            val model = NeuroTest.prepared(intArrayOf(3, 5, 2))
            val driver = RecordingCudaDriver()
            openTrainingSession(model, TrainingBackend.CUDA) { driver }.use { session ->
                session.trainMiniBatch(1, 7)
                assertEquals(1, logs.events("epoch.published").size)
                assertEquals(model.logId, logs.fields(logs.events("epoch.published").single())["model"])
                assertTrue(logs.events("memory.admitted").isNotEmpty())
                assertTrue(logs.events("workspace.ready").isNotEmpty())
                driver.nonfiniteDownloads = true
                assertThrows(IllegalStateException::class.java) { session.trainEpoch() }
                assertEquals(1, logs.events("epoch.published").size)
            }
            assertTrue(logs.events("memory.released").isNotEmpty())
            assertTrue(driver.memory.isEmpty())
            val unavailable = RecordingCudaDriver().apply { freeMemory = 1 }
            assertThrows(IllegalStateException::class.java) {
                openTrainingSession(NeuroTest.prepared(intArrayOf(3, 2)), TrainingBackend.CUDA) { unavailable }
            }
            val rejected = logs.events("memory.admission.failed").single()
            assertInstanceOf(IllegalStateException::class.java, rejected.thrown)
            assertEquals(1L, logs.fields(rejected)["freeBytes"])
        }
    }

    @Test fun dispatchesOrderedKernelsAndReusesOrGrowsBoundedWorkspace() {
        val model = NeuroTest.prepared(intArrayOf(3, 5, 2))
        val driver = RecordingCudaDriver()
        val session = openTrainingSession(model, TrainingBackend.CUDA) { driver }
        assertEquals(driver.info, session.info)
        session.trainMiniBatch(1, 7, 4)
        val counts = driver.launches.filter { it.name == "gather" }.map { it.arguments.last() }
        assertEquals(listOf(7, 7, 7, 7, 4), counts)
        assertEquals(listOf("gather", "forward", "forward", "output_delta", "hidden_delta", "update", "update"),
            driver.launches.take(7).map { it.name })
        assertTrue(driver.launches.all { it.workItems > 0 })
        assertEquals(1L, model.statistics().epochsTrained)
        assertEquals(32L, model.statistics().samplesSeen)
        val allocated = driver.allocationCalls
        session.trainMiniBatch(1, 3)
        assertEquals(allocated, driver.allocationCalls)
        session.trainMiniBatch(1, 16)
        assertTrue(driver.allocationCalls > allocated)
        assertTrue(driver.freed.isNotEmpty())
        val updatesBefore = driver.launches.count { it.name == "update" }
        session.trainEpoch()
        assertEquals(updatesBefore + 64, driver.launches.count { it.name == "update" })
        assertTrue(driver.launches.filter { it.name == "update" }.takeLast(64).all { it.arguments.last() == 1 })
        session.close(); session.close()
        assertTrue(driver.memory.isEmpty())
        assertEquals(1, driver.closeCalls)
        assertTrue(driver.synchronizations >= 6)
    }

    @Test fun validatesArgumentsWithoutPoisoningSessionAndPreservesExclusiveOwnership() {
        val model = NeuroTest.prepared(intArrayOf(3, 2))
        val driver = RecordingCudaDriver()
        openTrainingSession(model, TrainingBackend.CUDA) { driver }.use { session ->
            assertThrows(IllegalArgumentException::class.java) { session.train(-1) }
            assertThrows(IllegalArgumentException::class.java) { session.trainMiniBatch(1, 0) }
            assertThrows(IllegalArgumentException::class.java) { session.trainMiniBatch(1, 1, 0) }
            assertThrows(IllegalArgumentException::class.java) { session.trainUntil(-1.0, 1) }
            assertThrows(IllegalArgumentException::class.java) { session.trainUntil(0.0, -1) }
            assertThrows(IllegalArgumentException::class.java) { session.trainUntil(0.0, 1, 0) }
            assertThrows(IllegalStateException::class.java) { model.newTrainingSession() }
            assertThrows(IllegalStateException::class.java) { model.trainEpoch() }
            assertThrows(IllegalStateException::class.java) { model.addTrainingSample(DoubleArray(3), DoubleArray(2)) }
            assertThrows(IllegalStateException::class.java) { model.addTestSample(DoubleArray(3), DoubleArray(2)) }
            session.train(0); session.trainMiniBatch(0, 1)
            assertTrue(driver.launches.isEmpty())
            assertTrue(session.trainUntil(1.0, 0).converged)
            assertFalse(session.trainUntil(0.0, 0).converged)
            val result = session.trainUntil(0.0, 3, 2)
            assertEquals(3, result.epochs)
            assertFalse(result.converged)
            session.train(2)
            assertEquals(5L, model.statistics().epochsTrained)
        }
        assertTrue(model.trainEpoch().isFinite())
    }

    @Test fun initiallyConvergedCudaRunRecordsErrorWithoutClaimingAnEpoch() {
        for (target in listOf(0.0, 1.0)) {
            val expected = NeuroTest.prepared(intArrayOf(3, 2))
            val model = NeuroTest.prepared(intArrayOf(3, 2))
            val driver = RecordingCudaDriver()
            openTrainingSession(model, TrainingBackend.CUDA) { driver }.use { session ->
                assertEquals(expected.trainUntil(target, 0), session.trainUntil(target, 0))
                assertEquals(expected.statistics(), model.statistics())
                assertTrue(driver.launches.isEmpty())
            }
        }
    }

    @Test fun failedOrNonfiniteDownloadsNeverPublishAPartialEpochAndCpuCanResume() {
        for (failure in listOf("download", "nonfinite", "launch")) {
            val expected = NeuroTest.prepared(intArrayOf(3, 5, 2), Neuro.Kernel.SCALAR)
            val model = NeuroTest.prepared(intArrayOf(3, 5, 2), Neuro.Kernel.SCALAR)
            val before = model.exportTrainingState()
            val driver = RecordingCudaDriver().apply {
                failDownloadAt = if (failure == "download") 2 else 0
                nonfiniteDownloads = failure == "nonfinite"
                failLaunch = failure == "launch"
                changeDownloads = true
            }
            val session = openTrainingSession(model, TrainingBackend.CUDA) { driver }
            assertThrows(IllegalStateException::class.java) { session.trainEpoch() }
            assertEquals(0L, model.statistics().epochsTrained)
            for (layer in before.weights.indices) {
                assertArrayEquals(before.weights[layer], model.backendWeights(layer), 0.0)
                assertArrayEquals(before.biases[layer], model.backendBiases(layer), 0.0)
            }
            assertThrows(IllegalStateException::class.java) { session.trainEpoch() }
            session.close()
            assertTrue(driver.memory.isEmpty())
            expected.trainEpoch(); model.trainEpoch()
            for (layer in before.weights.indices) {
                assertArrayEquals(expected.backendWeights(layer), model.backendWeights(layer), 0.0)
                assertArrayEquals(expected.backendBiases(layer), model.backendBiases(layer), 0.0)
            }
        }
    }

    @Test fun datasetChangeAfterFailedEpochBuildsACompleteNewShuffle() {
        val model = NeuroTest.prepared(intArrayOf(3, 2))
        val failing = RecordingCudaDriver().apply { failLaunch = true }
        openTrainingSession(model, TrainingBackend.CUDA) { failing }.use { session ->
            assertThrows(IllegalStateException::class.java) { session.trainEpoch() }
        }
        model.addTrainingSample(DoubleArray(3) { 0.5 }, DoubleArray(2) { 1.0 })
        val recovered = RecordingCudaDriver()
        openTrainingSession(model, TrainingBackend.CUDA) { recovered }.use { session ->
            session.trainEpoch()
            val order = recovered.memory.values.filterIsInstance<IntArray>().single()
            assertEquals((0 until 33).toList(), order.sorted())
            assertEquals(33L, model.statistics().samplesSeen)
        }
    }

    @Test fun initializationFailureReleasesPartialAllocationsAndModelOwnership() {
        for (failure in listOf("memory", "allocation", "factory")) {
            val model = NeuroTest.prepared(intArrayOf(3, 5, 2))
            val driver = RecordingCudaDriver().apply {
                if (failure == "memory") freeMemory = 1
                if (failure == "allocation") failAllocationAt = 5
            }
            assertThrows(IllegalStateException::class.java) {
                openTrainingSession(model, TrainingBackend.CUDA) {
                    check(failure != "factory") { "Driver unavailable" }
                    driver
                }
            }
            assertTrue(driver.memory.isEmpty())
            assertEquals(if (failure == "factory") 0 else 1, driver.closeCalls)
            model.newTrainingSession().use { assertTrue(it.trainEpoch().isFinite()) }
        }
    }

    @Test fun workspaceAllocationAndCleanupFailuresStillReleaseEveryResource() {
        val model = NeuroTest.prepared(intArrayOf(3, 5, 2))
        val driver = RecordingCudaDriver()
        val session = openTrainingSession(model, TrainingBackend.CUDA) { driver }
        driver.failAllocationAt = driver.allocationCalls + 3
        assertThrows(IllegalStateException::class.java) { session.trainMiniBatch(1, 8) }
        driver.failFree = true
        driver.failClose = true
        val failure = assertThrows(IllegalStateException::class.java) { session.close() }
        assertTrue(failure.suppressed.isNotEmpty())
        assertTrue(driver.memory.isEmpty())
        assertEquals(1, driver.closeCalls)
        session.close()
        assertTrue(model.trainEpoch().isFinite())
    }

    @Test fun failedWorkspaceGrowthDoesNotFreeReleasedPointersAgain() {
        val model = NeuroTest.prepared(intArrayOf(3, 5, 2))
        val driver = RecordingCudaDriver()
        val session = openTrainingSession(model, TrainingBackend.CUDA) { driver }
        try {
            session.trainMiniBatch(1, 2)
            driver.failFreeAt = driver.freeCalls + 2
            val failure = assertThrows(IllegalStateException::class.java) { session.trainMiniBatch(1, 8) }
            assertTrue(failure.message!!.contains("selected release failed"))
            assertEquals(1, driver.freed.size)
            val released = driver.freed.single()
            driver.failFreeAt = 0
            session.close()
            assertEquals(1, driver.freeAttempts.count { it == released }, "A released address may already belong to another session")
            assertTrue(driver.memory.isEmpty())
            assertEquals(1L, model.statistics().epochsTrained)
        } finally {
            driver.failFreeAt = 0
            session.close()
        }
    }

    @Test fun queuedOperationCannotReuseDeviceStateAfterAnEpochFailure() {
        val model = NeuroTest.prepared(intArrayOf(3, 5, 2))
        val driver = RecordingCudaDriver()
        val entered = CountDownLatch(1)
        val release = CountDownLatch(1)
        driver.beforeLaunch = {
            entered.countDown()
            check(release.await(10, TimeUnit.SECONDS)) { "Test did not release the failing launch" }
        }
        driver.failLaunch = true
        val session = openTrainingSession(model, TrainingBackend.CUDA) { driver }
        val firstFailure = AtomicReference<Throwable>()
        val queuedFailure = AtomicReference<Throwable>()
        fun attempt(failure: AtomicReference<Throwable>) = Thread {
            try { session.trainEpoch() } catch (exception: Throwable) { failure.set(exception) }
        }.apply { isDaemon = true }
        val first = attempt(firstFailure)
        val queued = attempt(queuedFailure)
        try {
            first.start()
            assertTrue(entered.await(10, TimeUnit.SECONDS), "First operation did not reach the driver")
            queued.start()
            val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10)
            while (queued.state != Thread.State.BLOCKED && queued.isAlive && System.nanoTime() < deadline) {
                LockSupport.parkNanos(TimeUnit.MILLISECONDS.toNanos(1))
            }
            assertEquals(Thread.State.BLOCKED, queued.state, "Second operation must be queued on the model monitor")
            release.countDown()
            first.join(10_000)
            queued.join(10_000)
            assertFalse(first.isAlive)
            assertFalse(queued.isAlive)
            assertInstanceOf(IllegalStateException::class.java, firstFailure.get())
            assertTrue(firstFailure.get().message!!.contains("Test kernel failed"))
            assertInstanceOf(IllegalStateException::class.java, queuedFailure.get())
            assertTrue(queuedFailure.get().message!!.contains("Close this session"))
            assertEquals(1, driver.launchCalls, "The queued operation must not touch failed device state")
            assertEquals(0L, model.statistics().epochsTrained)
        } finally {
            release.countDown()
            first.join(10_000)
            if (queued.state != Thread.State.NEW) queued.join(10_000)
            session.close()
        }
        assertTrue(driver.memory.isEmpty())
    }

    @Test fun emptyDatasetAndMemoryAdmissionFailClearly() {
        val empty = Neuro(intArrayOf(2, 1))
        val driver = RecordingCudaDriver()
        openTrainingSession(empty, TrainingBackend.CUDA) { driver }.use { session ->
            session.train(0)
            assertThrows(IllegalStateException::class.java) { session.trainEpoch() }
        }
        assertTrue(driver.memory.isEmpty())
        assertThrows(IllegalArgumentException::class.java) { CudaMemoryBudget.acquire("negative-test", -1, 100) }
        assertThrows(IllegalStateException::class.java) { CudaMemoryBudget.acquire("small-test", 81, 100) }
        CudaMemoryBudget.acquire("release-test", 10, 100)
        CudaMemoryBudget.acquire("release-test", 10, 100)
        CudaMemoryBudget.release("release-test", 10)
        CudaMemoryBudget.release("release-test", 10)
    }

    @Test fun concurrentReservationsRespectSharedHeadroomAndReleaseTheirLimit() {
        CudaMemoryBudget.acquire("concurrent-test", 60, 100)
        assertThrows(IllegalStateException::class.java) { CudaMemoryBudget.acquire("concurrent-test", 30, 100) }
        CudaMemoryBudget.release("concurrent-test", 20)
        CudaMemoryBudget.acquire("concurrent-test", 30, 100)
        CudaMemoryBudget.release("concurrent-test", 70)
        // A later workload observes a fresh capacity after every previous reservation is released.
        CudaMemoryBudget.acquire("concurrent-test", 160, 200)
        CudaMemoryBudget.release("concurrent-test", 160)
    }

    private class RecordingCudaDriver : CudaDriver {
        override val info = TrainingDeviceInfo(TrainingBackend.CUDA, "Test device", "test-${System.identityHashCode(this)}", "FP64", "test-v1")
        val memory = LinkedHashMap<Long, Any>()
        val freed = ArrayList<Long>()
        val freeAttempts = ArrayList<Long>()
        val launches = ArrayList<Launch>()
        var freeMemory = 1L shl 30
        var allocationCalls = 0
        var freeCalls = 0
        var launchCalls = 0
        var closeCalls = 0
        var synchronizations = 0
        var failAllocationAt = 0
        var failDownloadAt = 0
        var failLaunch = false
        var failFree = false
        var failFreeAt = 0
        var failClose = false
        var changeDownloads = false
        var nonfiniteDownloads = false
        var beforeLaunch: (() -> Unit)? = null
        private var downloads = 0
        override fun availableMemory() = freeMemory
        override fun allocate(bytes: Long): Long {
            allocationCalls++
            check(allocationCalls != failAllocationAt) { "Test allocation failed" }
            assertTrue(bytes > 0)
            return allocationCalls.toLong().also { memory[it] = ByteArray(bytes.toInt()) }
        }
        override fun free(pointer: Long) {
            freeCalls++
            freeAttempts.add(pointer)
            check(freeCalls != failFreeAt) { "Test selected release failed" }
            assertNotNull(memory.remove(pointer), "Allocation freed more than once")
            freed.add(pointer)
            check(!failFree) { "Test release failed" }
        }
        override fun upload(pointer: Long, values: DoubleArray) { assertTrue(memory.containsKey(pointer)); memory[pointer] = values.copyOf() }
        override fun upload(pointer: Long, values: IntArray) { assertTrue(memory.containsKey(pointer)); memory[pointer] = values.copyOf() }
        override fun download(pointer: Long, values: DoubleArray) {
            downloads++
            check(downloads != failDownloadAt) { "Test download failed" }
            (memory.getValue(pointer) as DoubleArray).copyInto(values)
            if (changeDownloads) for (index in values.indices) values[index] += 1
            if (nonfiniteDownloads) values[0] = Double.NaN
        }
        override fun launch(name: String, workItems: Int, vararg arguments: Any) {
            launchCalls++
            beforeLaunch?.invoke()
            check(!failLaunch) { "Test kernel failed" }
            launches.add(Launch(name, workItems, arguments.toList()))
        }
        override fun synchronize() { synchronizations++ }
        override fun close() { closeCalls++; check(!failClose) { "Test context close failed" } }
    }

    private data class Launch(val name: String, val workItems: Int, val arguments: List<Any>)
}
