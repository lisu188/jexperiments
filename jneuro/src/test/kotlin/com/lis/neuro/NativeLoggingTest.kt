package com.lis.neuro

import java.lang.foreign.Arena
import java.util.concurrent.CopyOnWriteArrayList
import java.util.logging.Handler
import java.util.logging.Level
import java.util.logging.LogRecord
import java.util.logging.Logger
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

/** Real FFM downcalls target test upcalls here; these tests make no claim about hardware execution. */
class NativeLoggingTest {
    @Test fun libraryDiscoveryReportsAttemptsAndSelectedLibraryOnlyOnce() {
        NativeLogCapture().use { logs ->
            CudaNativeTestRuntime().use { native ->
                Arena.ofConfined().use { arena ->
                    val setting = NeuroLog.id("TEST_NATIVE")
                    repeat(2) {
                        NeuroNativeLibrary.open(arena, setting, listOf("missing", "selected"), listOf("cuInit"),
                            { null }, { name, lifetime ->
                                if (name == "missing") throw IllegalArgumentException("missing test library")
                                native.lookup(lifetime)
                            })
                    }
                    assertEquals(4, logs.events("library.attempt").size)
                    val selected = logs.events("library.selected").single()
                    assertEquals(Level.INFO, selected.level)
                    assertEquals("selected", logs.fields(selected)["library"])
                    assertEquals(setting, logs.fields(selected)["setting"])
                    assertEquals(2, logs.events("library.rejected").size)
                }
            }
        }
    }

    @Test fun traceDescribesMatrixShapesTransfersAndPublicationWithoutRawValues() {
        NativeLogCapture().use { logs ->
            for (precision in Neuro.TrainingPrecision.entries) {
                CudaNativeTestRuntime().use { native ->
                    val model = NeuroTest.prepared(intArrayOf(3, 5, 2))
                    NeuroCudaBatchBackend.train(model, model.backendTrainingData(), 1, 7, precision, native.loader)
                    assertTrue(native.allocations.isEmpty())
                }
            }
            val gemms = logs.events("gemm.submitted")
            assertEquals(setOf("FP32", "FP64"), gemms.map { logs.fields(it)["precision"] }.toSet())
            assertTrue(gemms.all { it.level == Level.FINER && logs.fields(it)["m"] is Int })
            assertTrue(logs.events("kernel.submitted").any { logs.fields(it)["kernel"] == "gatherRows" })
            assertTrue(logs.events("transfer.submitted").any { logs.fields(it)["direction"] == "device-to-host" })
            assertEquals(2, logs.events("epoch.published").size)
            assertEquals(2, logs.events("compilation.completed").size)
            assertTrue(logs.records.indexOf(logs.events("device.synchronized").first()) <
                logs.records.indexOf(logs.events("epoch.published").first()))
            for (record in logs.records) {
                val fields = logs.fields(record)
                assertFalse(fields.keys.any { it.toString().contains("pointer", ignoreCase = true) })
                assertFalse(fields.values.any { it is DoubleArray || it is FloatArray || it is IntArray })
            }
        }
    }

    @Test fun nativeAndCompilationFailuresCarryTheirCauseAndCannotClaimAnEpoch() {
        NativeLogCapture().use { logs ->
            CudaNativeTestRuntime().use { native ->
                val model = NeuroTest.prepared(intArrayOf(3, 5, 2))
                native.errors["cublasDgemm_v2"] = 13
                val failure = assertThrows(IllegalStateException::class.java) {
                    NeuroCudaBatchBackend.train(model, model.backendTrainingData(), 1, 7, Neuro.TrainingPrecision.FP64, native.loader)
                }
                val status = logs.events("native.failed").single { logs.fields(it)["operation"] == "cublasDgemm_v2" }
                assertSame(failure, status.thrown)
                assertEquals(13, logs.fields(status)["status"])
                assertEquals(Level.SEVERE, status.level)
                assertFalse(logs.fields(status)["hint"].toString().isBlank())
                assertSame(failure, logs.events("batch.training.failed").single().thrown)
                assertTrue(logs.events("epoch.published").isEmpty())
                assertEquals(0L, model.statistics().epochsTrained)
                assertTrue(native.allocations.isEmpty())
            }
            CudaNativeTestRuntime().use { native ->
                native.errors["nvrtcCompileProgram"] = 6
                NeuroNvrtc.create(native.loader).use { compiler ->
                    val failure = assertThrows(IllegalStateException::class.java) { compiler.compile("test", "test.cu", 8 to 6) }
                    assertSame(failure, logs.events("compilation.failed").single().thrown)
                    assertTrue(native.calls.contains("nvrtcDestroyProgram"))
                }
            }
        }
    }

    @Test fun disabledTraceDoesNotEmitPerKernelOrTransferEvents() {
        NativeLogCapture(Level.INFO).use { logs ->
            CudaNativeTestRuntime().use { native ->
                val model = NeuroTest.prepared(intArrayOf(3, 2))
                NeuroCudaBatchBackend.train(model, model.backendTrainingData(), 1, 7, Neuro.TrainingPrecision.FP64, native.loader)
                assertTrue(native.launches.isNotEmpty())
                assertTrue(native.gemms.isNotEmpty())
                assertTrue(logs.events("kernel.submitted").isEmpty())
                assertTrue(logs.events("gemm.submitted").isEmpty())
                assertTrue(logs.events("transfer.submitted").isEmpty())
            }
        }
    }
}

internal class NativeLogCapture(level: Level = Level.ALL) : AutoCloseable {
    val records = CopyOnWriteArrayList<LogRecord>()
    private val logger = Logger.getLogger("com.lis.neuro")
    private val previousLevel: Level?
    private val handler = object : Handler() {
        override fun publish(record: LogRecord) { records += record }
        override fun flush() = Unit
        override fun close() = Unit
    }
    init {
        NeuroLog.initialize()
        previousLevel = logger.level
        logger.level = level
        handler.level = Level.ALL
        logger.addHandler(handler)
    }
    fun events(event: String): List<LogRecord> = records.filter { it.message == event }
    fun fields(record: LogRecord): Map<*, *> = record.parameters.single() as Map<*, *>
    override fun close() { logger.removeHandler(handler); logger.level = previousLevel }
}
