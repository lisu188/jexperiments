package com.lis.neuro

import java.lang.foreign.Arena
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout

internal interface NeuroCudaMemory {
    fun allocate(byteSize: Long): MemorySegment
    fun release(pointer: MemorySegment)
    fun copyHostToDevice(device: MemorySegment, host: MemorySegment, byteSize: Long)
    fun copyDeviceToHost(host: MemorySegment, device: MemorySegment, byteSize: Long)
    fun synchronize()
}

internal class NeuroCudaBuffer private constructor(
    private val memory: NeuroCudaMemory,
    val pointer: MemorySegment,
    val byteSize: Long
) : AutoCloseable {
    private var closed = false
    private val logId = NeuroLog.id("buffer")
    init { NeuroLog.debug("cuda", "buffer.allocated") { mapOf("buffer" to logId, "bytes" to byteSize) } }

    fun upload(values: DoubleArray) {
        ensureOpen()
        NeuroLog.trace("cuda", "buffer.upload") { mapOf("buffer" to logId, "type" to "FP64", "elements" to values.size) }
        val bytes = Math.multiplyExact(values.size.toLong(), java.lang.Double.BYTES.toLong())
        require(bytes <= byteSize) { "double upload exceeds CUDA buffer capacity" }
        Arena.ofConfined().use { arena ->
            val host = arena.allocate(ValueLayout.JAVA_DOUBLE, values.size.toLong())
            MemorySegment.copy(values, 0, host, ValueLayout.JAVA_DOUBLE, 0L, values.size)
            memory.copyHostToDevice(pointer, host, bytes)
        }
    }

    fun upload(values: FloatArray) {
        ensureOpen()
        NeuroLog.trace("cuda", "buffer.upload") { mapOf("buffer" to logId, "type" to "FP32", "elements" to values.size) }
        val bytes = Math.multiplyExact(values.size.toLong(), java.lang.Float.BYTES.toLong())
        require(bytes <= byteSize) { "float upload exceeds CUDA buffer capacity" }
        Arena.ofConfined().use { arena ->
            val host = arena.allocate(ValueLayout.JAVA_FLOAT, values.size.toLong())
            MemorySegment.copy(values, 0, host, ValueLayout.JAVA_FLOAT, 0L, values.size)
            memory.copyHostToDevice(pointer, host, bytes)
        }
    }

    fun upload(values: IntArray) {
        ensureOpen()
        NeuroLog.trace("cuda", "buffer.upload") { mapOf("buffer" to logId, "type" to "INT32", "elements" to values.size) }
        val bytes = Math.multiplyExact(values.size.toLong(), Integer.BYTES.toLong())
        require(bytes <= byteSize) { "int upload exceeds CUDA buffer capacity" }
        Arena.ofConfined().use { arena ->
            val host = arena.allocate(ValueLayout.JAVA_INT, values.size.toLong())
            MemorySegment.copy(values, 0, host, ValueLayout.JAVA_INT, 0L, values.size)
            memory.copyHostToDevice(pointer, host, bytes)
        }
    }

    fun downloadDoubles(elements: Int): DoubleArray {
        ensureOpen()
        NeuroLog.trace("cuda", "buffer.download") { mapOf("buffer" to logId, "type" to "FP64", "elements" to elements) }
        require(elements >= 0)
        val bytes = Math.multiplyExact(elements.toLong(), java.lang.Double.BYTES.toLong())
        require(bytes <= byteSize) { "double download exceeds CUDA buffer capacity" }
        val result = DoubleArray(elements)
        Arena.ofConfined().use { arena ->
            val host = arena.allocate(ValueLayout.JAVA_DOUBLE, elements.toLong())
            memory.copyDeviceToHost(host, pointer, bytes)
            MemorySegment.copy(host, ValueLayout.JAVA_DOUBLE, 0L, result, 0, elements)
        }
        return result
    }

    fun downloadFloats(elements: Int): FloatArray {
        ensureOpen()
        NeuroLog.trace("cuda", "buffer.download") { mapOf("buffer" to logId, "type" to "FP32", "elements" to elements) }
        require(elements >= 0)
        val bytes = Math.multiplyExact(elements.toLong(), java.lang.Float.BYTES.toLong())
        require(bytes <= byteSize) { "float download exceeds CUDA buffer capacity" }
        val result = FloatArray(elements)
        Arena.ofConfined().use { arena ->
            val host = arena.allocate(ValueLayout.JAVA_FLOAT, elements.toLong())
            memory.copyDeviceToHost(host, pointer, bytes)
            MemorySegment.copy(host, ValueLayout.JAVA_FLOAT, 0L, result, 0, elements)
        }
        return result
    }

    override fun close() {
        if (closed) return
        closed = true
        try {
            memory.release(pointer)
            NeuroLog.debug("cuda", "buffer.released") { mapOf("buffer" to logId, "bytes" to byteSize) }
        } catch (failure: Throwable) {
            NeuroLog.error("cuda", "buffer.release.failed", failure, "buffer" to logId, "bytes" to byteSize)
            throw failure
        }
    }

    private fun ensureOpen() = check(!closed) { "CUDA buffer is closed" }

    companion object {
        fun doubles(memory: NeuroCudaMemory, elements: Int): NeuroCudaBuffer {
            require(elements > 0) { "CUDA buffer element count must be positive" }
            val bytes = Math.multiplyExact(elements.toLong(), java.lang.Double.BYTES.toLong())
            return NeuroCudaBuffer(memory, memory.allocate(bytes), bytes)
        }

        fun floats(memory: NeuroCudaMemory, elements: Int): NeuroCudaBuffer {
            require(elements > 0) { "CUDA buffer element count must be positive" }
            val bytes = Math.multiplyExact(elements.toLong(), java.lang.Float.BYTES.toLong())
            return NeuroCudaBuffer(memory, memory.allocate(bytes), bytes)
        }

        fun ints(memory: NeuroCudaMemory, elements: Int): NeuroCudaBuffer {
            require(elements > 0) { "CUDA buffer element count must be positive" }
            val bytes = Math.multiplyExact(elements.toLong(), Integer.BYTES.toLong())
            return NeuroCudaBuffer(memory, memory.allocate(bytes), bytes)
        }
    }
}
