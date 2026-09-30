package com.lis.neuro

import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import java.io.IOException
import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.logging.Handler
import java.util.logging.Level
import java.util.logging.LogRecord
import java.util.logging.Logger

class NeuroLogTest {
    @TempDir lateinit var directory: Path
    private val root = Logger.getLogger("com.lis.neuro")
    private val records = CopyOnWriteArrayList<LogRecord>()
    private val capture = object : Handler() {
        override fun publish(record: LogRecord) { records += record }
        override fun flush() = Unit
        override fun close() = Unit
    }

    @BeforeEach fun prepare() {
        NeuroLog.configure(NeuroLogConfiguration(level = Level.ALL, console = false, file = false))
        root.addHandler(capture)
    }

    @AfterEach fun restore() {
        root.removeHandler(capture)
        NeuroLog.configure(NeuroLogConfiguration(level = Level.WARNING, console = false, file = false))
    }

    @Test fun structuredEventsPreserveScalarTypesAndEscapeUntrustedText() {
        val failure = IllegalStateException("failed\nline", IOException("native cause"))
        failure.addSuppressed(IOException("cleanup failure"))
        NeuroLog.info("training", "session.open", "backend" to "CUDA", "epochs" to 7,
            "rmse" to 0.25, "active" to true, "missing" to null, "note" to "quoted \"text\"\\\r\n\t\u0001 🌍")
        NeuroLog.debug("training", "epoch.completed") { mapOf("value" to Float.NaN) }
        NeuroLog.trace("native", "kernel.launch") { mapOf("bytes" to 32L, "delta" to 0.5f) }
        NeuroLog.warn("training", "epoch.rejected", null, "reason" to "ownership")
        NeuroLog.error("native", "kernel.failed", failure, "nativeStatus" to 1)
        assertEquals(listOf("session.open", "epoch.completed", "kernel.launch", "epoch.rejected", "kernel.failed"), records.map { it.message })
        @Suppress("UNCHECKED_CAST") val data = records.first().parameters[0] as Map<String, Any?>
        assertEquals(7, data["epochs"])
        assertEquals(true, data["active"])
        val lines = records.map { NeuroLogFormatter().format(it) }
        assertTrue(lines.all { it.endsWith("\n") && it.count { char -> char == '\n' } == 1 })
        assertTrue(lines[0].contains("\\\"text\\\"\\\\\\r\\n\\t\\u0001 🌍"))
        assertTrue(lines[1].contains("\"level\":\"DEBUG\""))
        assertTrue(lines[1].contains("\"value\":\"NaN\""))
        assertTrue(lines[2].contains("\"level\":\"TRACE\""))
        assertTrue(lines[3].contains("\"level\":\"WARN\""))
        assertTrue(lines[4].contains("native cause") && lines[4].contains("cleanup failure"))
        assertTrue(lines[4].contains("\"level\":\"ERROR\""))
        assertSame(failure, records.last().thrown)
    }

    @Test fun disabledDetailDoesNotEvaluateSuppliersAndLoggingFailuresDoNotEscape() {
        root.level = Level.INFO
        NeuroLog.debug("training", "disabled") { error("must stay lazy") }
        NeuroLog.trace("training", "disabled") { error("must stay lazy") }
        assertTrue(records.isEmpty())
        root.level = Level.ALL
        assertDoesNotThrow { NeuroLog.debug("training", "broken.fields") { error("bad diagnostic") } }
        val broken = object : Handler() {
            override fun publish(record: LogRecord) { throw IllegalStateException("broken sink") }
            override fun flush() = Unit
            override fun close() = Unit
        }
        root.addHandler(broken)
        try { assertDoesNotThrow { NeuroLog.info("training", "still.trains") } }
        finally { root.removeHandler(broken) }
        assertTrue(records.any { it.message == "still.trains" })
    }

    @Test fun rotatesBoundedFilesWithoutTouchingUnrelatedFilesOrPercentDirectories() {
        val folder = directory.resolve("literal%value")
        Files.createDirectories(folder)
        val unrelated = folder.resolve("keep.txt")
        Files.writeString(unrelated, "preserve")
        NeuroLog.configure(NeuroLogConfiguration(directory = folder, console = false, limitBytes = 1024, fileCount = 2))
        repeat(30) { NeuroLog.info("training", "rotation.sample", "iteration" to it, "detail" to "x".repeat(250)) }
        NeuroLog.shutdown()
        val logs = Files.list(folder).use { stream -> stream.filter { it.fileName.toString().endsWith(".log") }.toList() }
        assertEquals(2, logs.size)
        assertTrue(logs.all { Files.size(it) <= 2048 })
        assertTrue(logs.any { Files.readString(it).contains("\"iteration\":29") })
        assertEquals("preserve", Files.readString(unrelated))
        assertTrue(Files.list(folder).use { stream -> stream.noneMatch { it.fileName.toString().endsWith(".lck") } })
    }

    @Test fun fileFailureFallsBackAndInvalidConfigurationIsActionable() {
        val file = directory.resolve("existing-file")
        Files.writeString(file, "keep")
        assertDoesNotThrow { NeuroLog.configure(NeuroLogConfiguration(directory = file, console = false)) }
        assertTrue(records.any { it.message == "logging.file.unavailable" && it.thrown != null })
        assertTrue(root.handlers.any { it is java.util.logging.ConsoleHandler })
        assertEquals("keep", Files.readString(file))
        val values = mapOf("level" to "invalid", "file" to "maybe", "count" to "0", "limit_bytes" to "999999999", "dir" to "bad\u0000path")
        val config = NeuroLogConfiguration.read({ key -> values[key.removePrefix("jneuro.log.")] }, { null })
        assertEquals(Level.INFO, config.level)
        assertEquals(3, config.fileCount)
        assertEquals(2 * 1024 * 1024, config.limitBytes)
        assertEquals(5, config.warnings.size)
        NeuroLog.configure(config.copy(console = false, file = false))
        assertEquals(5, records.count { it.message == "logging.configuration.invalid" })
    }

    @Test fun environmentOverridesAndOffModeAreRespected() {
        val environment = mapOf("JNEURO_LOG_LEVEL" to "TRACE", "JNEURO_LOG_DIR" to directory.toString(),
            "JNEURO_LOG_FILE" to "false", "JNEURO_LOG_COUNT" to "2", "JNEURO_LOG_LIMIT_BYTES" to "2048")
        val config = NeuroLogConfiguration.read({ null }, environment::get)
        assertEquals(Level.FINER, config.level)
        assertEquals(directory, config.directory)
        assertFalse(config.file)
        assertEquals(2, config.fileCount)
        assertEquals(2048, config.limitBytes)
        assertEquals(Level.FINE, NeuroLogConfiguration.read({ if (it.endsWith("level")) "DEBUG" else null }, environment::get).level)
        for ((name, expected) in listOf("WARN" to Level.WARNING, "ERROR" to Level.SEVERE, "OFF" to Level.OFF)) {
            assertEquals(expected, NeuroLogConfiguration.read({ if (it.endsWith("level")) name else null }, { null }).level)
        }
        NeuroLog.configure(config.copy(level = Level.OFF, console = true, file = true)) {
            error("OFF must not create files")
        }
        NeuroLog.info("training", "disabled")
        assertTrue(records.isEmpty())
    }

    @Test fun concurrentRecordsHaveUniqueIdsAndSnapshotFieldsAreImmutableAndBounded() {
        val ids = java.util.concurrent.ConcurrentHashMap.newKeySet<String>()
        Executors.newFixedThreadPool(4).use { pool ->
            val futures = (0 until 40).map {
                pool.submit {
                    val id = NeuroLog.id("session")
                    ids += id
                    NeuroLog.info("training", "parallel", "session" to id)
                }
            }
            futures.forEach { it.get(5, TimeUnit.SECONDS) }
        }
        assertEquals(40, ids.size)
        assertEquals(40, records.size)
        val map = linkedMapOf<String, Any?>("value" to "before")
        NeuroLog.debug("training", "snapshot") { map }
        map["value"] = "after"
        @Suppress("UNCHECKED_CAST") val captured = records.last().parameters[0] as MutableMap<String, Any?>
        assertEquals("before", captured["value"])
        assertThrows(UnsupportedOperationException::class.java) { captured["value"] = "changed" }
        NeuroLog.info("training", "bounded", "large" to "x".repeat(10000), "infinite" to Double.POSITIVE_INFINITY)
        val formatted = NeuroLogFormatter().format(records.last())
        assertTrue(formatted.length < 4000 && formatted.contains("[truncated]"))
        assertTrue(formatted.contains("\"infinite\":\"Infinity\""))
        assertTrue(NeuroLogFormatter().format(LogRecord(Level.INFO, "external")).contains("external"))
    }

    @Test fun entrypointLifecyclePreservesFailuresAndOtherApplicationsLogging() {
        val global = Logger.getLogger("")
        val handlers = global.handlers.toList()
        val level = global.level
        NeuroLog.application("example") { NeuroLog.info("example", "work.completed") }
        val failure = IllegalStateException("original failure")
        assertSame(failure, assertThrows(IllegalStateException::class.java) {
            NeuroLog.application("failing") { throw failure }
        })
        assertEquals(2, records.count { it.message == "application.started" })
        assertEquals(1, records.count { it.message == "application.completed" })
        assertSame(failure, records.single { it.message == "application.failed" }.thrown)
        assertEquals(level, global.level)
        assertEquals(handlers, global.handlers.toList())
    }
}
