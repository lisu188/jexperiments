package com.lis.neuro

import java.util.concurrent.CopyOnWriteArrayList
import java.util.logging.Handler
import java.util.logging.Level
import java.util.logging.LogRecord
import java.util.logging.Logger

/** Observe the same structured events delivered to console and file handlers. */
internal class NeuroApplicationLogCapture : AutoCloseable {
    private val logger = Logger.getLogger("com.lis.neuro")
    private val previousLevel: Level?
    val records = CopyOnWriteArrayList<LogRecord>()
    private val handler = object : Handler() {
        override fun publish(record: LogRecord) { records += record }
        override fun flush() = Unit
        override fun close() = Unit
    }
    init {
        NeuroLog.initialize()
        previousLevel = logger.level
        handler.level = Level.ALL
        logger.addHandler(handler)
        logger.level = Level.FINE
    }
    fun events(event: String) = records.filter { it.message == event }
    fun fields(record: LogRecord): Map<*, *> = record.parameters?.firstOrNull() as? Map<*, *> ?: emptyMap<Any, Any>()
    override fun close() {
        logger.removeHandler(handler)
        logger.level = previousLevel
    }
}
