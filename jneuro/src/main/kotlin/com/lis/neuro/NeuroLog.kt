package com.lis.neuro

import java.io.PrintWriter
import java.io.StringWriter
import java.nio.file.Files
import java.nio.file.Path
import java.util.Collections
import java.util.Locale
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import java.util.logging.ConsoleHandler
import java.util.logging.FileHandler
import java.util.logging.Formatter
import java.util.logging.Handler
import java.util.logging.Level
import java.util.logging.LogRecord
import java.util.logging.Logger

/** Application-local JUL configuration; never changes the JVM's root logger. */
internal data class NeuroLogConfiguration(
    val level: Level = Level.INFO,
    val directory: Path = Path.of(System.getProperty("user.home"), ".jneuro", "logs"),
    val console: Boolean = true,
    val file: Boolean = true,
    val limitBytes: Int = 2 * 1024 * 1024,
    val fileCount: Int = 3,
    val warnings: List<String> = emptyList()
) {
    init { require(limitBytes > 0 && fileCount in 1..10) }

    companion object {
        fun read(property: (String) -> String? = System::getProperty,
                 environment: (String) -> String? = System::getenv): NeuroLogConfiguration {
            val defaults = NeuroLogConfiguration()
            val warnings = ArrayList<String>()
            fun value(suffix: String): String? = property("jneuro.log.$suffix")?.trim()?.takeIf { it.isNotEmpty() }
                ?: environment("JNEURO_LOG_" + suffix.uppercase(Locale.ROOT))?.trim()?.takeIf { it.isNotEmpty() }
            fun invalid(key: String) { warnings += "Invalid jneuro.log.$key; using default" }
            val level = when (value("level")?.uppercase(Locale.ROOT)) {
                null, "INFO" -> Level.INFO
                "DEBUG", "FINE" -> Level.FINE
                "TRACE", "FINER", "ALL" -> Level.FINER
                "WARN", "WARNING" -> Level.WARNING
                "ERROR", "SEVERE" -> Level.SEVERE
                "OFF" -> Level.OFF
                else -> { invalid("level"); Level.INFO }
            }
            val directory = try { value("dir")?.let(Path::of) ?: defaults.directory }
                catch (_: IllegalArgumentException) { invalid("dir"); defaults.directory }
            val file = when (value("file")?.lowercase(Locale.ROOT)) {
                null, "true" -> true
                "false" -> false
                else -> { invalid("file"); true }
            }
            fun integer(key: String, default: Int, range: IntRange): Int {
                val raw = value(key) ?: return default
                return raw.toIntOrNull()?.takeIf { it in range } ?: run { invalid(key); default }
            }
            return NeuroLogConfiguration(level, directory, file = file,
                limitBytes = integer("limit_bytes", defaults.limitBytes, 1024..16 * 1024 * 1024),
                fileCount = integer("count", defaults.fileCount, 1..10), warnings = warnings)
        }
    }
}

internal object NeuroLog {
    private val root = Logger.getLogger("com.lis.neuro")
    private val loggers = ConcurrentHashMap<String, Logger>()
    private val ids = AtomicLong()
    private val run = UUID.randomUUID().toString()
    private val reportedFailure = AtomicBoolean()
    private var ownedHandlers = emptyList<Handler>()
    @Volatile private var initialized = false

    fun id(prefix: String): String = "$prefix-${ids.incrementAndGet()}"

    fun application(entrypoint: String, action: () -> Unit) {
        val application = id("application")
        val started = System.nanoTime()
        info("application", "application.started", "application" to application, "entrypoint" to entrypoint,
            "java" to System.getProperty("java.version"), "os" to System.getProperty("os.name"),
            "pid" to ProcessHandle.current().pid())
        try {
            action()
            info("application", "application.completed", "application" to application, "entrypoint" to entrypoint,
                "durationMs" to (System.nanoTime() - started) / 1_000_000.0)
        } catch (failure: Throwable) {
            error("application", "application.failed", failure, "application" to application, "entrypoint" to entrypoint,
                "durationMs" to (System.nanoTime() - started) / 1_000_000.0)
            throw failure
        }
    }

    fun initialize() {
        if (!initialized) synchronized(this) {
            if (!initialized) configure(NeuroLogConfiguration.read())
        }
    }

    @Synchronized
    fun configure(configuration: NeuroLogConfiguration,
                  fileFactory: (NeuroLogConfiguration) -> Handler = ::fileHandler) {
        shutdown()
        root.useParentHandlers = false
        root.level = configuration.level
        val handlers = ArrayList<Handler>()
        if (configuration.console && configuration.level != Level.OFF) handlers += ConsoleHandler()
        var fileFailure: Exception? = null
        if (configuration.file && configuration.level != Level.OFF) {
            try { handlers += fileFactory(configuration) }
            catch (failure: Exception) {
                fileFailure = failure
                if (handlers.isEmpty()) handlers += ConsoleHandler()
            }
        }
        for (handler in handlers) {
            handler.level = Level.ALL
            handler.encoding = "UTF-8"
            handler.formatter = NeuroLogFormatter()
            root.addHandler(handler)
        }
        ownedHandlers = handlers
        initialized = true
        for (warning in configuration.warnings) warn("logging", "logging.configuration.invalid", null, "reason" to warning)
        fileFailure?.let { warn("logging", "logging.file.unavailable", it, "directory" to configuration.directory.toString()) }
    }

    private fun fileHandler(configuration: NeuroLogConfiguration): Handler {
        Files.createDirectories(configuration.directory)
        // FileHandler interprets percent escapes in the entire pattern, including directory names.
        val directory = configuration.directory.toAbsolutePath().toString().replace("%", "%%")
        return FileHandler("$directory/jneuro-%u-%g.log", configuration.limitBytes, configuration.fileCount, true)
    }

    @Synchronized fun shutdown() {
        for (handler in ownedHandlers) {
            root.removeHandler(handler)
            try { handler.close() } catch (failure: Exception) { reportFailure(failure) }
        }
        ownedHandlers = emptyList()
        initialized = false
    }

    fun info(component: String, event: String, vararg fields: Pair<String, Any?>) =
        emit(Level.INFO, component, event, null) { fields.toMap() }
    fun debug(component: String, event: String, fields: () -> Map<String, Any?> = { emptyMap() }) =
        emit(Level.FINE, component, event, null, fields)
    fun trace(component: String, event: String, fields: () -> Map<String, Any?> = { emptyMap() }) =
        emit(Level.FINER, component, event, null, fields)
    fun warn(component: String, event: String, cause: Throwable? = null, vararg fields: Pair<String, Any?>) =
        emit(Level.WARNING, component, event, cause) { fields.toMap() }
    fun error(component: String, event: String, cause: Throwable, vararg fields: Pair<String, Any?>) =
        emit(Level.SEVERE, component, event, cause) { fields.toMap() }

    private fun emit(level: Level, component: String, event: String, cause: Throwable?,
                     fields: () -> Map<String, Any?>) {
        try {
            initialize()
            val logger = loggers.computeIfAbsent(component) { Logger.getLogger("com.lis.neuro.$it") }
            if (!logger.isLoggable(level)) return
            val snapshot = Collections.unmodifiableMap(fields().entries.take(64).associate { (key, value) ->
                bounded(key) to when (value) {
                    null, is Boolean, is Byte, is Short, is Int, is Long -> value
                    is Float -> if (value.isFinite()) value else value.toString()
                    is Double -> if (value.isFinite()) value else value.toString()
                    else -> bounded(value.toString())
                }
            })
            val record = NeuroLogRecord(level, bounded(event), component, run, Thread.currentThread().name, snapshot)
            record.loggerName = logger.name
            record.thrown = cause
            record.parameters = arrayOf(snapshot)
            logger.log(record)
        } catch (failure: Exception) {
            // Diagnostics must not change training, cleanup, or UI control flow.
            reportFailure(failure)
        }
    }

    private fun reportFailure(failure: Exception) {
        if (reportedFailure.compareAndSet(false, true)) {
            System.err.println("JNeuro logging could not write a diagnostic: ${failure.javaClass.simpleName}")
        }
    }

    private fun bounded(value: String): String = if (value.length <= 2048) value else value.take(2048) + "...[truncated]"
}

private class NeuroLogRecord(level: Level, event: String, val component: String, val run: String,
                             val thread: String, val fields: Map<String, Any?>) : LogRecord(level, event)

internal class NeuroLogFormatter : Formatter() {
    override fun format(record: LogRecord): String {
        val detailed = record as? NeuroLogRecord
        val fields = linkedMapOf<String, Any?>(
            "time" to record.instant.toString(), "level" to levelName(record.level),
            "component" to (detailed?.component ?: record.loggerName), "event" to record.message,
            "run" to detailed?.run, "thread" to (detailed?.thread ?: record.longThreadID.toString()))
        val prefix = fields.entries.joinToString(",") { (key, value) -> quoted(key) + ":" + json(value) }
        val data = detailed?.fields.orEmpty().entries.joinToString(",") { (key, value) -> quoted(key) + ":" + json(value) }
        val exception = record.thrown?.let { failure ->
            val writer = StringWriter()
            failure.printStackTrace(PrintWriter(writer))
            val trace = writer.toString()
            ",\"exception\":" + quoted(if (trace.length <= 16384) trace else trace.take(16384) + "...[truncated]")
        }.orEmpty()
        return "{$prefix,\"data\":{$data}$exception}\n"
    }

    private fun levelName(level: Level): String = when {
        level.intValue() >= Level.SEVERE.intValue() -> "ERROR"
        level.intValue() >= Level.WARNING.intValue() -> "WARN"
        level.intValue() >= Level.INFO.intValue() -> "INFO"
        level.intValue() >= Level.FINE.intValue() -> "DEBUG"
        else -> "TRACE"
    }

    private fun json(value: Any?): String = when (value) {
        null -> "null"
        is Number, is Boolean -> value.toString()
        else -> quoted(value.toString())
    }

    private fun quoted(value: String): String = buildString {
        append('"')
        for (character in value) when (character) {
            '"' -> append("\\\"")
            '\\' -> append("\\\\")
            '\n' -> append("\\n")
            '\r' -> append("\\r")
            '\t' -> append("\\t")
            else -> if (character.code < 32) append("\\u%04x".format(Locale.ROOT, character.code)) else append(character)
        }
        append('"')
    }
}
