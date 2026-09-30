package com.lis.neuro

import java.lang.foreign.Arena
import java.lang.foreign.FunctionDescriptor
import java.lang.foreign.Linker
import java.lang.foreign.MemoryLayout
import java.lang.foreign.MemorySegment
import java.lang.foreign.SymbolLookup
import java.lang.foreign.ValueLayout
import java.lang.invoke.MethodHandle
import java.security.MessageDigest

/** CUDA Driver API only: the installed NVIDIA driver JITs the packaged PTX. */
internal class NativeCudaDriver(
    driverLookup: ((Arena) -> SymbolLookup)? = null,
    ptxBytes: (() -> ByteArray?)? = null
) : CudaDriver {
    private val logId = NeuroLog.id("driver")
    private val arena = Arena.ofShared()
    private val linker = Linker.nativeLinker()
    private val handles = HashMap<String, MethodHandle>()
    private val functions = HashMap<String, MemorySegment>()
    private val lookup: SymbolLookup
    private var context = MemorySegment.NULL
    private var stream = MemorySegment.NULL
    private var module = MemorySegment.NULL
    private var device = 0
    private var retained = false
    private var closed = false
    override val info: TrainingDeviceInfo

    init {
        try {
            lookup = driverLookup?.invoke(arena) ?: loadDriver()
            checkCall("cuInit", int(0))
            Arena.ofConfined().use { scratch ->
                val out = scratch.allocate(ValueLayout.JAVA_INT)
                checkCall("cuDeviceGet", address(out), int(0))
                device = out.get(ValueLayout.JAVA_INT, 0)
                val major = scratch.allocate(ValueLayout.JAVA_INT)
                val minor = scratch.allocate(ValueLayout.JAVA_INT)
                checkCall("cuDeviceComputeCapability", address(major), address(minor), int(device))
                check(major.get(ValueLayout.JAVA_INT, 0) * 10 + minor.get(ValueLayout.JAVA_INT, 0) >= 75) {
                    "CUDA training requires compute capability 7.5 or newer."
                }
                val name = scratch.allocate(256)
                checkCall("cuDeviceGetName", address(name), int(256), int(device))
                val uuid = scratch.allocate(16)
                checkCall("cuDeviceGetUuid", address(uuid), int(device))
                checkCall("cuDriverGetVersion", address(out))
                val version = out.get(ValueLayout.JAVA_INT, 0)
                val result = scratch.allocate(ValueLayout.ADDRESS)
                checkCall("cuDevicePrimaryCtxRetain", address(result), int(device))
                context = result.get(ValueLayout.ADDRESS, 0)
                retained = true
                current()
                // Pageable HtoD copies may return before DMA finishes. A blocking
                // stream orders launches after those copies on the legacy stream.
                checkCall("cuStreamCreate", address(result), int(0))
                stream = result.get(ValueLayout.ADDRESS, 0)
                val ptx = (if (ptxBytes != null) ptxBytes() else
                    javaClass.getResourceAsStream("/com/lis/neuro/cuda/train.ptx")?.use { it.readAllBytes() })
                    ?: throw IllegalStateException("Packaged CUDA kernels are missing. Rebuild with verified CUDA resources.")
                val bytes = scratch.allocate(ptx.size.toLong() + 1)
                MemorySegment.copy(ptx, 0, bytes, ValueLayout.JAVA_BYTE, 0, ptx.size)
                checkCall("cuModuleLoadData", address(result), address(bytes))
                module = result.get(ValueLayout.ADDRESS, 0)
                val identity = uuid.toArray(ValueLayout.JAVA_BYTE).joinToString("") { "%02x".format(it) }
                val hash = MessageDigest.getInstance("SHA-256").digest(ptx).joinToString("") { "%02x".format(it) }
                info = TrainingDeviceInfo(TrainingBackend.CUDA, name.getString(0), "$identity/driver-$version", "FP64", hash)
                NeuroLog.debug("cuda", "driver.ready") { mapOf("driver" to logId, "device" to info.name,
                    "identity" to info.identity, "precision" to info.precision, "kernel" to hash,
                    "computeMajor" to major.get(ValueLayout.JAVA_INT, 0), "computeMinor" to minor.get(ValueLayout.JAVA_INT, 0),
                    "context" to "retained-primary", "stream" to "blocking") }
            }
        } catch (failure: Throwable) {
            NeuroLog.error("cuda", "driver.initialization.failed", failure, "driver" to logId)
            try { close() } catch (cleanup: Throwable) { failure.addSuppressed(cleanup) }
            throw IllegalStateException("CUDA training is unavailable: ${failure.message}", failure)
        }
    }

    private fun loadDriver(): SymbolLookup {
        val libraries = if (System.getProperty("os.name").startsWith("Windows")) listOf("nvcuda.dll")
            else listOf("libcuda.so.1", "/usr/lib/wsl/lib/libcuda.so.1")
        for (library in libraries) {
            NeuroLog.debug("native", "library.attempt") { mapOf("setting" to "JNEURO_CUDA_DRIVER", "library" to library) }
            try {
                val found = SymbolLookup.libraryLookup(library, arena)
                NeuroNativeLibrary.selected("JNEURO_CUDA_DRIVER", library)
                return found
            } catch (failure: IllegalArgumentException) {
                NeuroLog.debug("native", "library.rejected") { mapOf("library" to library, "reason" to failure.javaClass.simpleName) }
            }
        }
        throw IllegalStateException("NVIDIA CUDA driver was not found. Install a compatible NVIDIA driver or select CPU.")
    }

    private fun current() = checkCall("cuCtxSetCurrent", address(context))

    private fun checkCall(name: String, vararg arguments: Pair<MemoryLayout, Any>) {
        val handle = handles.getOrPut(name) {
            val symbol = lookup.find(name).orElseThrow { IllegalStateException("CUDA driver lacks $name") }
            linker.downcallHandle(symbol, FunctionDescriptor.of(ValueLayout.JAVA_INT, *arguments.map { it.first }.toTypedArray()))
        }
        val result = handle.invokeWithArguments(arguments.map { it.second }) as Int
        NeuroNativeLibrary.checkStatus(result, name, "cuda") {
            "$name failed (CUDA error $result). Check driver compatibility, GPU access, and free device memory."
        }
    }

    override fun availableMemory(): Long {
        current()
        return Arena.ofConfined().use {
            val free = it.allocate(ValueLayout.JAVA_LONG)
            val total = it.allocate(ValueLayout.JAVA_LONG)
            checkCall("cuMemGetInfo_v2", address(free), address(total))
            free.get(ValueLayout.JAVA_LONG, 0).also { available ->
                NeuroLog.debug("cuda", "memory.available") { mapOf("driver" to logId, "freeBytes" to available, "totalBytes" to total.get(ValueLayout.JAVA_LONG, 0)) }
            }
        }
    }

    override fun allocate(bytes: Long): Long {
        current()
        return Arena.ofConfined().use {
            val output = it.allocate(ValueLayout.JAVA_LONG)
            checkCall("cuMemAlloc_v2", address(output), long(bytes))
            output.get(ValueLayout.JAVA_LONG, 0).also {
                NeuroLog.debug("cuda", "allocation.created") { mapOf("driver" to logId, "bytes" to bytes) }
            }
        }
    }

    override fun free(pointer: Long) {
        current()
        checkCall("cuMemFree_v2", long(pointer))
        NeuroLog.debug("cuda", "allocation.released") { mapOf("driver" to logId) }
    }

    override fun upload(pointer: Long, values: DoubleArray) {
        current()
        NeuroLog.trace("cuda", "transfer.submitted") { mapOf("driver" to logId, "direction" to "host-to-device", "type" to "FP64", "elements" to values.size, "bytes" to values.size * 8L) }
        Arena.ofConfined().use {
            val bytes = it.allocate(ValueLayout.JAVA_DOUBLE, values.size.toLong())
            MemorySegment.copy(values, 0, bytes, ValueLayout.JAVA_DOUBLE, 0, values.size)
            checkCall("cuMemcpyHtoD_v2", long(pointer), address(bytes), long(values.size * 8L))
        }
    }

    override fun upload(pointer: Long, values: IntArray) {
        current()
        NeuroLog.trace("cuda", "transfer.submitted") { mapOf("driver" to logId, "direction" to "host-to-device", "type" to "INT32", "elements" to values.size, "bytes" to values.size * 4L) }
        Arena.ofConfined().use {
            val bytes = it.allocate(ValueLayout.JAVA_INT, values.size.toLong())
            MemorySegment.copy(values, 0, bytes, ValueLayout.JAVA_INT, 0, values.size)
            checkCall("cuMemcpyHtoD_v2", long(pointer), address(bytes), long(values.size * 4L))
        }
    }

    override fun download(pointer: Long, values: DoubleArray) {
        current()
        NeuroLog.trace("cuda", "transfer.submitted") { mapOf("driver" to logId, "direction" to "device-to-host", "type" to "FP64", "elements" to values.size, "bytes" to values.size * 8L) }
        Arena.ofConfined().use {
            val bytes = it.allocate(ValueLayout.JAVA_DOUBLE, values.size.toLong())
            checkCall("cuMemcpyDtoH_v2", address(bytes), long(pointer), long(values.size * 8L))
            MemorySegment.copy(bytes, ValueLayout.JAVA_DOUBLE, 0, values, 0, values.size)
        }
    }

    override fun launch(name: String, workItems: Int, vararg arguments: Any) {
        current()
        val function = functions.getOrPut(name) {
            Arena.ofConfined().use {
                val output = it.allocate(ValueLayout.ADDRESS)
                checkCall("cuModuleGetFunction", address(output), address(module), address(it.allocateFrom(name)))
                output.get(ValueLayout.ADDRESS, 0)
            }
        }
        Arena.ofConfined().use { scratch ->
            val params = scratch.allocate(ValueLayout.ADDRESS, arguments.size.toLong())
            for ((index, value) in arguments.withIndex()) {
                val argument = when (value) {
                    is Long -> scratch.allocate(ValueLayout.JAVA_LONG).also { it.set(ValueLayout.JAVA_LONG, 0, value) }
                    is Int -> scratch.allocate(ValueLayout.JAVA_INT).also { it.set(ValueLayout.JAVA_INT, 0, value) }
                    is Double -> scratch.allocate(ValueLayout.JAVA_DOUBLE).also { it.set(ValueLayout.JAVA_DOUBLE, 0, value) }
                    else -> throw IllegalArgumentException("Unsupported CUDA kernel argument: ${value.javaClass.name}")
                }
                params.setAtIndex(ValueLayout.ADDRESS, index.toLong(), argument)
            }
            val blocks = ((workItems.toLong() + 127) / 128).toInt()
            NeuroLog.trace("cuda", "kernel.submitted") { mapOf("driver" to logId, "kernel" to name,
                "workItems" to workItems, "grid" to blocks, "block" to 128, "arguments" to arguments.size, "stream" to "blocking") }
            checkCall("cuLaunchKernel", address(function), int(blocks), int(1), int(1), int(128), int(1), int(1), int(0),
                address(stream), address(params), address(MemorySegment.NULL))
        }
    }

    override fun synchronize() {
        current()
        checkCall("cuStreamSynchronize", address(stream))
        NeuroLog.trace("cuda", "stream.synchronized") { mapOf("driver" to logId, "stream" to "blocking") }
    }

    override fun close() {
        if (closed) return
        closed = true
        var failure: Throwable? = null
        fun cleanup(action: () -> Unit) {
            try { action() } catch (exception: Throwable) {
                NeuroLog.error("cuda", "driver.cleanup.failed", exception, "driver" to logId)
                if (failure == null) failure = exception else failure.addSuppressed(exception)
            }
        }
        if (retained) {
            cleanup { current() }
            if (stream != MemorySegment.NULL) {
                cleanup { checkCall("cuStreamSynchronize", address(stream)) }
                cleanup { checkCall("cuStreamDestroy_v2", address(stream)) }
            }
            if (module != MemorySegment.NULL) cleanup { checkCall("cuModuleUnload", address(module)) }
            cleanup { checkCall("cuDevicePrimaryCtxRelease_v2", int(device)) }
        }
        cleanup { arena.close() }
        failure?.let { throw it }
        NeuroLog.debug("cuda", "driver.closed") { mapOf("driver" to logId) }
    }

    private fun int(value: Int): Pair<MemoryLayout, Any> = ValueLayout.JAVA_INT to value
    private fun long(value: Long): Pair<MemoryLayout, Any> = ValueLayout.JAVA_LONG to value
    private fun address(value: MemorySegment): Pair<MemoryLayout, Any> = ValueLayout.ADDRESS to value
}
