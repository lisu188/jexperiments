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
            }
        } catch (failure: Throwable) {
            try { close() } catch (cleanup: Throwable) { failure.addSuppressed(cleanup) }
            throw IllegalStateException("CUDA training is unavailable: ${failure.message}", failure)
        }
    }

    private fun loadDriver(): SymbolLookup {
        val libraries = if (System.getProperty("os.name").startsWith("Windows")) listOf("nvcuda.dll")
            else listOf("libcuda.so.1", "/usr/lib/wsl/lib/libcuda.so.1")
        for (library in libraries) {
            try { return SymbolLookup.libraryLookup(library, arena) }
            catch (_: IllegalArgumentException) { /* Try the WSL driver location after the normal loader. */ }
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
        check(result == 0) { "$name failed (CUDA error $result). Check driver compatibility, GPU access, and free device memory." }
    }

    override fun availableMemory(): Long {
        current()
        return Arena.ofConfined().use {
            val free = it.allocate(ValueLayout.JAVA_LONG)
            val total = it.allocate(ValueLayout.JAVA_LONG)
            checkCall("cuMemGetInfo_v2", address(free), address(total))
            free.get(ValueLayout.JAVA_LONG, 0)
        }
    }

    override fun allocate(bytes: Long): Long {
        current()
        return Arena.ofConfined().use {
            val output = it.allocate(ValueLayout.JAVA_LONG)
            checkCall("cuMemAlloc_v2", address(output), long(bytes))
            output.get(ValueLayout.JAVA_LONG, 0)
        }
    }

    override fun free(pointer: Long) {
        current()
        checkCall("cuMemFree_v2", long(pointer))
    }

    override fun upload(pointer: Long, values: DoubleArray) {
        current()
        Arena.ofConfined().use {
            val bytes = it.allocate(ValueLayout.JAVA_DOUBLE, values.size.toLong())
            MemorySegment.copy(values, 0, bytes, ValueLayout.JAVA_DOUBLE, 0, values.size)
            checkCall("cuMemcpyHtoD_v2", long(pointer), address(bytes), long(values.size * 8L))
        }
    }

    override fun upload(pointer: Long, values: IntArray) {
        current()
        Arena.ofConfined().use {
            val bytes = it.allocate(ValueLayout.JAVA_INT, values.size.toLong())
            MemorySegment.copy(values, 0, bytes, ValueLayout.JAVA_INT, 0, values.size)
            checkCall("cuMemcpyHtoD_v2", long(pointer), address(bytes), long(values.size * 4L))
        }
    }

    override fun download(pointer: Long, values: DoubleArray) {
        current()
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
            checkCall("cuLaunchKernel", address(function), int(blocks), int(1), int(1), int(128), int(1), int(1), int(0),
                address(stream), address(params), address(MemorySegment.NULL))
        }
    }

    override fun synchronize() {
        current()
        checkCall("cuStreamSynchronize", address(stream))
    }

    override fun close() {
        if (closed) return
        closed = true
        var failure: Throwable? = null
        fun cleanup(action: () -> Unit) {
            try { action() } catch (exception: Throwable) {
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
    }

    private fun int(value: Int): Pair<MemoryLayout, Any> = ValueLayout.JAVA_INT to value
    private fun long(value: Long): Pair<MemoryLayout, Any> = ValueLayout.JAVA_LONG to value
    private fun address(value: MemorySegment): Pair<MemoryLayout, Any> = ValueLayout.ADDRESS to value
}
