package com.lis.neuro

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test
import java.util.logging.Handler
import java.util.logging.Level
import java.util.logging.LogRecord
import java.util.logging.Logger

class NeuroTrainingLoggingTest {
    @Test fun disabledProgressAvoidsRunIdsButStillLogsOriginalFailures() = TrainingCapture().use { capture ->
        Logger.getLogger("com.lis.neuro").level = Level.WARNING
        val model = model()
        model.newTrainingSession().use { session ->
            val before = NeuroLog.id("probe").substringAfterLast('-').toLong()
            repeat(3) { session.trainEpoch() }
            val after = NeuroLog.id("probe").substringAfterLast('-').toLong()
            assertEquals(before + 1, after, "Disabled successful progress must not allocate correlated run IDs")
            assertTrue(capture.events("training.completed").isEmpty())
            val failure = assertThrows(IllegalArgumentException::class.java) { session.train(-1) }
            val event = capture.one("training.rejected")
            assertSame(failure, event.thrown)
            assertEquals(0L, event.fields()["completedEpochs"])
            assertEquals(3L, event.fields()["totalEpochs"])
        }
    }

    @Test fun cpuResolutionAndProgressAreCorrelatedAndInfoIsSampled() = TrainingCapture().use { capture ->
        val model = model()
        openConfiguredTrainingSession(model, TrainingBackend.AUTO, Neuro.TrainingPrecision.FP32, 1).use { session ->
            repeat(200) { session.trainEpoch() }
        }
        val resolution = capture.one("session.resolved").fields()
        assertEquals("AUTO", resolution["requestedBackend"])
        assertEquals("CPU", resolution["backend"])
        assertEquals("FP32", resolution["requestedPrecision"])
        assertEquals("FP32", resolution["precision"])
        assertEquals(model.logId, resolution["model"])
        val completed = capture.events("training.completed")
        assertEquals(200, completed.size, "CPU session delegation must not duplicate completion events")
        assertEquals(listOf(1L, 100L, 200L), completed.filter { it.level == Level.INFO }.map { it.fields()["totalEpochs"] })
        assertTrue(completed.all { it.fields()["session"] == resolution["session"] })
        assertTrue(completed.all { it.fields()["completedEpochs"] == 1L })
        assertTrue(completed.all { it.fields()["gpuWorkCompleted"] == false })
        assertEquals(200, completed.map { it.fields()["run"] }.toSet().size)
        assertTrue(completed.all { (it.fields()["durationMs"] as Double) >= 0.0 })
        assertEquals(200L, capture.one("session.closed").fields()["completedEpochs"])
        assertNull(model.trainingSessionLogId)
        assertTrue(capture.records.none { it.fields().keys.any { key -> key in listOf("inputs", "targets", "weights", "biases") } })
    }

    @Test fun searchChunksSamplePublishedBoundariesAtInfoAndKeepEveryChunkAtDebug() {
        for (engine in TrainingEngine.entries) for (level in listOf(Level.INFO, Level.FINE)) {
            TrainingCapture().use { capture ->
                Logger.getLogger("com.lis.neuro").level = level
                val model = NeuroTest.prepared(intArrayOf(2, 4, 1))
                model.train(7)
                capture.records.clear()
                searchSession(model, engine).use { session ->
                    repeat(2) {
                        assertEquals(0, advanceTrainingForSearch(session, TrainingChunkRequest(0)).committedEpochs)
                        assertEquals(0, advanceTrainingForSearch(session,
                            TrainingChunkRequest(25, cancelled = { true })).committedEpochs)
                    }
                    assertTrue(capture.events("training.completed").none { it.level == Level.INFO },
                        "Zero-work search calls must not repeatedly report initial progress")
                    capture.records.clear()
                    val chunks = listOf(25, 64, 12, 64, 30, 25)
                    chunks.forEach { epochs ->
                        assertEquals(epochs, advanceTrainingForSearch(session, TrainingChunkRequest(epochs)).committedEpochs)
                    }
                    val completed = capture.events("training.completed")
                    val info = completed.filter { it.level == Level.INFO }
                    assertEquals(listOf(32L, 108L, 202L), info.map { it.fields()["totalEpochs"] },
                        "Sample the first committed chunk and crossed boundaries, even when totals are not multiples of 100")
                    assertEquals(listOf(25L, 12L, 30L), info.map { it.fields()["completedEpochs"] })
                    assertTrue(completed.all { it.fields()["operation"] == "advanceForSearch" })
                    assertTrue(completed.all { it.fields()["gpuWorkCompleted"] == false })
                    val started = capture.events("training.started")
                    if (level == Level.INFO) {
                        assertTrue(started.isEmpty(), "Search chunk starts belong at DEBUG")
                        assertEquals(3, completed.size)
                    } else {
                        assertEquals(chunks.size, started.size)
                        assertEquals(chunks.size, completed.size)
                        assertTrue(started.all { it.level == Level.FINE })
                        assertEquals(chunks, completed.map { it.fields()["requestedEpochs"] })
                        assertEquals(listOf(32L, 96L, 108L, 172L, 202L, 227L),
                            completed.map { it.fields()["totalEpochs"] })
                        for ((start, end) in started.zip(completed)) {
                            assertEquals(start.fields()["run"], end.fields()["run"])
                            val epochs = end.fields()["completedEpochs"] as Long
                            assertEquals(epochs * model.trainingSampleCount(), end.fields()["samplesProcessed"])
                            assertEquals("COMPLETED", end.fields()["termination"])
                            assertTrue((end.fields()["rmse"] as Double).isFinite())
                        }
                    }
                }
                assertEquals(227L, capture.one("session.closed").fields()["completedEpochs"])
            }
        }
    }

    @Test fun searchCancellationAndFailureDetailsRetainOnlyCommittedProgress() {
        for (engine in TrainingEngine.entries) TrainingCapture().use { capture ->
            Logger.getLogger("com.lis.neuro").level = Level.FINE
            val model = NeuroTest.prepared(intArrayOf(2, 4, 1))
            searchSession(model, engine).use { session ->
                advanceTrainingForSearch(session, TrainingChunkRequest(25))
                var polls = 0
                val result = advanceTrainingForSearch(session, TrainingChunkRequest(25, cancelled = { ++polls > 3 }))
                assertEquals(SearchAdvanceResult(3, null, TrainingTermination.CANCELLED), result)
                val cancelled = capture.events("training.completed").last()
                assertEquals(Level.FINE, cancelled.level)
                assertEquals("CANCELLED", cancelled.fields()["termination"])
                assertEquals(3L, cancelled.fields()["completedEpochs"])
                assertEquals(28L, cancelled.fields()["totalEpochs"])
                assertEquals(3L * model.trainingSampleCount(), cancelled.fields()["samplesProcessed"])
                assertNull(cancelled.fields()["rmse"])
                val problem = IllegalStateException("search cancellation callback failed")
                val thrown = assertThrows(IllegalStateException::class.java) {
                    advanceTrainingForSearch(session, TrainingChunkRequest(25, cancelled = { throw problem }))
                }
                val failed = capture.one("training.failed")
                assertSame(problem, thrown)
                assertSame(problem, failed.thrown)
                assertEquals("advanceForSearch", failed.fields()["operation"])
                assertEquals(25, failed.fields()["requestedEpochs"])
                assertEquals(0L, failed.fields()["completedEpochs"])
                assertEquals(28L, failed.fields()["totalEpochs"])
                assertEquals(0L, failed.fields()["samplesProcessed"])
                assertEquals(false, failed.fields()["gpuWorkCompleted"])
                assertEquals("current-model", failed.fields()["retainedState"])
                assertEquals(2, capture.events("training.completed").size,
                    "A failed chunk must not emit a completion")
            }
        }
    }

    @Test fun publicBulkCallsKeepInfoStartAndCompletionAfterSearchProgress() {
        for (engine in TrainingEngine.entries) TrainingCapture().use { capture ->
            Logger.getLogger("com.lis.neuro").level = Level.INFO
            val model = NeuroTest.prepared(intArrayOf(2, 4, 1))
            searchSession(model, engine).use { session ->
                advanceTrainingForSearch(session, TrainingChunkRequest(25))
                capture.records.clear()
                session.train(2)
                session.trainUntil(1.0, 0)
                assertEquals(listOf("train", "trainUntil"),
                    capture.events("training.started").map { it.fields()["operation"] })
                val completed = capture.events("training.completed")
                assertEquals(listOf("train", "trainUntil"), completed.map { it.fields()["operation"] })
                assertTrue(completed.all { it.level == Level.INFO })
                assertEquals(listOf(2L, 0L), completed.map { it.fields()["completedEpochs"] })
                assertEquals(true, completed.last().fields()["converged"])
            }
        }
    }

    private fun searchSession(model: Neuro, engine: TrainingEngine): NeuroTrainingSession =
        if (engine == TrainingEngine.SMALL) SmallTrainingSession(model, 1,
            { SmallCpuTraining(it, model.hyperParameters(), Neuro.TrainingPrecision.FP64) }, { 0L })
        else SmallTrainingSession(model, 1, { TensorFlowMath.trainingKernel(it, model.hyperParameters(), Neuro.TrainingPrecision.FP64, TrainingBackend.CPU) }, { 0L })

    @Test fun resumedSessionLogsItsFirstCompletedEpochAtInfo() = TrainingCapture().use { capture ->
        val model = model()
        model.train(5)
        model.newTrainingSession().use { it.trainEpoch() }
        val completion = capture.events("training.completed").last()
        assertEquals(Level.INFO, completion.level)
        assertEquals(6L, completion.fields()["totalEpochs"])
    }

    @Test fun directTrainingLogsResultsValidationMatrixResolutionAndInferenceWithoutChangingResults() = TrainingCapture().use { capture ->
        val model = model()
        model.train(2)
        val result = model.trainUntil(1.0, 0)
        assertTrue(result.converged)
        val convergence = capture.events("training.completed").last().fields()
        assertEquals(true, convergence["converged"])
        assertEquals(0L, convergence["completedEpochs"])
        assertEquals(result.error, convergence["rmse"])
        assertEquals(1.0, convergence["targetError"])
        assertEquals(1, convergence["checkEvery"])
        val problem = assertThrows(IllegalArgumentException::class.java) { model.train(-1) }
        assertSame(problem, capture.one("training.rejected").thrown)
        val unsupported = assertThrows(IllegalArgumentException::class.java) {
            model.newTrainingSession(batchSize = 0)
        }
        assertSame(unsupported, capture.one("session.open.rejected").thrown)
        model.trainMiniBatch(2, 4, 1, Neuro.BatchBackend.CPU, Neuro.TrainingPrecision.FP32)
        val matrix = capture.one("training.batch.resolved").fields()
        assertEquals("FP32", matrix["requestedPrecision"])
        assertEquals("FP32", matrix["precision"])
        assertEquals(1, matrix["batchSize"])
        assertEquals(2L, capture.events("training.completed").last().fields()["completedEpochs"])
        val expected = model.predict(doubleArrayOf(0.25, 0.75))
        model.newInferenceSession().predictBatch(doubleArrayOf(0.25, 0.75), 1, DoubleArray(1).also {
            model.newInferenceSession().predictInto(doubleArrayOf(0.25, 0.75), it)
            assertArrayEquals(expected, it, 0.0)
        })
        model.newParallelInferenceSession(2).use { it.predictBatch(doubleArrayOf(0.25, 0.75), 1, DoubleArray(1)) }
        model.toFloatModel()
        assertTrue(capture.events("inference.completed").all { it.level == Level.FINER })
        assertEquals(1, capture.events("inference.parallel.closed").size)
        assertEquals("FP32", capture.one("inference.float.exported").fields()["precision"])
    }

    @Test fun tensorflowProofFollowsHostPublicationAndFailureNeverClaimsSuccess() = TrainingCapture().use { capture ->
        val model = model()
        lateinit var kernel: LoggingTensorFlowKernel
        val session = openConfiguredTrainingSession(model, TrainingBackend.CPU, Neuro.TrainingPrecision.FP64, 1,
            kernelFactory = { state -> LoggingTensorFlowKernel(state, model.hyperParameters()).also { kernel = it } })
        capture.observe = { record ->
            if (record.message == "training.completed")
                assertEquals(1L, model.statistics().epochsTrained, "Host publication must precede the completion event")
        }
        session.trainEpoch()
        val completed = capture.one("training.completed").fields()
        assertEquals(false, completed["gpuWorkCompleted"])
        assertEquals("CPU", completed["backend"])
        assertEquals(kernel.info.kernelVersion, completed["kernel"])
        assertEquals(kernel.info.identity, completed["deviceIdentity"])
        kernel.failure = IllegalStateException("TensorFlow execution failed")
        val problem = assertThrows(IllegalStateException::class.java) { session.trainEpoch() }
        assertSame(kernel.failure, problem)
        val failed = capture.one("training.failed")
        assertSame(problem, failed.thrown)
        assertEquals(false, failed.fields()["gpuWorkCompleted"])
        assertEquals(0L, failed.fields()["completedEpochs"])
        assertEquals(1L, failed.fields()["totalEpochs"])
        assertEquals(1, capture.events("training.completed").size)
        kernel.failure = null
        assertThrows(IllegalStateException::class.java) { session.trainEpoch() }
        session.close()
        assertEquals(true, capture.one("session.closed").fields()["failed"])
    }

    @Test fun rejectedOwnershipAndSessionOpenAndCloseFailuresRetainTheirOriginalErrors() = TrainingCapture().use { capture ->
        val model = model()
        lateinit var kernel: LoggingTensorFlowKernel
        val session = openConfiguredTrainingSession(model, TrainingBackend.CPU, Neuro.TrainingPrecision.FP64, 1,
            kernelFactory = { state -> LoggingTensorFlowKernel(state, model.hyperParameters()).also { kernel = it } })
        assertThrows(IllegalStateException::class.java) { model.trainEpoch() }
        assertThrows(IllegalStateException::class.java) { model.newTrainingSession() }
        assertEquals(listOf("mutate", "acquire"), capture.events("training.ownership.rejected").map { it.fields()["operation"] })
        kernel.failure = IllegalStateException("close failed")
        val failure = assertThrows(IllegalStateException::class.java) { session.close() }
        assertSame(kernel.failure, failure)
        assertSame(failure, capture.one("session.close.failed").thrown)
        assertNull(model.trainingSessionLogId)
        assertTrue(model.trainEpoch().isFinite())
        val expected = IllegalStateException("no numerical runtime")
        assertSame(expected, assertThrows(IllegalStateException::class.java) {
            openConfiguredTrainingSession(model, TrainingBackend.CPU, Neuro.TrainingPrecision.FP64, 1, kernelFactory = { throw expected })
        })
        assertSame(expected, capture.events("session.open.failed").last().thrown)
    }

    @Test fun tensorflowBatchSummaryUsesCommittedStatisticsAndActualPrecision() = TrainingCapture().use { capture ->
        val model = model()
        model.newTrainingSession(precision = Neuro.TrainingPrecision.FP32, batchSize = 7).use { session ->
            session.trainMiniBatch(2, 7, 3)
            session.trainUntil(1.0, 0)
        }
        val batch = capture.events("training.completed").first().fields()
        assertEquals("CPU", batch["backend"])
        assertEquals("FP32", batch["precision"])
        assertEquals(7, batch["requestedBatchSize"])
        assertEquals(1, batch["batchSize"])
        assertEquals(3, batch["requestedParallelism"])
        assertEquals(3, batch["cpuParallelism"])
        assertEquals(2L, batch["completedEpochs"])
        assertEquals(false, batch["gpuWorkCompleted"])
    }

    private fun model() = Neuro(intArrayOf(2, 3, 1)).addTrainingSample(doubleArrayOf(0.25, 0.75), doubleArrayOf(0.5))

    private class TrainingCapture : AutoCloseable {
        val records = ArrayList<LogRecord>()
        var observe: (LogRecord) -> Unit = {}
        private val root = Logger.getLogger("com.lis.neuro")
        private val previousLevel: Level?
        private val previousHandlerLevels: Map<Handler, Level>
        private val handler = object : Handler() {
            override fun publish(record: LogRecord) { records += record; observe(record) }
            override fun flush() {}
            override fun close() {}
        }
        init {
            NeuroLog.initialize()
            previousLevel = root.level
            previousHandlerLevels = root.handlers.associateWith { it.level }
            previousHandlerLevels.keys.forEach { it.level = Level.OFF }
            handler.level = Level.ALL
            root.addHandler(handler)
            root.level = Level.ALL
        }
        fun events(event: String) = records.filter { it.message == event }
        fun one(event: String) = events(event).single()
        override fun close() {
            root.removeHandler(handler)
            root.level = previousLevel
            previousHandlerLevels.forEach { (handler, level) -> handler.level = level }
        }
    }

    private class LoggingTensorFlowKernel(state: NeuroTrainingState, parameters: Neuro.HyperParameters) : SmallTrainingKernel {
        private val delegate = TensorFlowMath.trainingKernel(state, parameters, Neuro.TrainingPrecision.FP64, TrainingBackend.CPU)
        override val info get() = delegate.info
        var failure: Throwable? = null
        override fun train(orders: Array<IntArray>, batchSize: Int, online: Boolean): NeuroTrainingState {
            failure?.let { throw it }
            return delegate.train(orders, batchSize, online)
        }
        override fun close() { delegate.close(); failure?.let { throw it } }
    }

}

@Suppress("UNCHECKED_CAST")
private fun LogRecord.fields() = parameters[0] as Map<String, Any?>
