package com.lis.neuro

import java.lang.foreign.Arena
import java.lang.foreign.FunctionDescriptor
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout
import java.lang.invoke.MethodHandle

internal class NeuroCudaDriver private constructor(
    private val arena: Arena,
    val libraryName: String,
    private val deviceGetHandle: MethodHandle,
    private val deviceGetNameHandle: MethodHandle,
    private val moduleLoadHandle: MethodHandle,
    private val moduleUnloadHandle: MethodHandle,
    private val moduleFunctionHandle: MethodHandle,
    private val launchHandle: MethodHandle
) : AutoCloseable {
    private var closed = false
    sealed interface Argument {
        data class Pointer(val value: MemorySegment) : Argument
        data class IntValue(val value: Int) : Argument
        data class DoubleValue(val value: Double) : Argument
        data class FloatValue(val value: Float) : Argument
    }

    inner class Module internal constructor(private val handle: MemorySegment) : AutoCloseable {
        private var closed = false
        fun function(name: String): MemorySegment = Arena.ofConfined().use { local ->
            check(!closed) { "CUDA module is closed" }
            val out = local.allocate(ValueLayout.ADDRESS)
            val cName = local.allocateFrom(name)
            checkStatus(NeuroNativeLibrary.invokeInt(moduleFunctionHandle, out, handle, cName), "cuModuleGetFunction")
            out.get(ValueLayout.ADDRESS, 0L)
        }
        override fun close() {
            if (closed) return
            closed = true
            checkStatus(NeuroNativeLibrary.invokeInt(moduleUnloadHandle, handle), "cuModuleUnload")
            NeuroLog.debug("cuda", "module.unloaded") { mapOf("library" to libraryName) }
        }
    }

    fun deviceName(index: Int): String = Arena.ofConfined().use { local ->
        val deviceOut = local.allocate(ValueLayout.JAVA_INT)
        checkStatus(NeuroNativeLibrary.invokeInt(deviceGetHandle, deviceOut, index), "cuDeviceGet")
        val name = local.allocate(256L)
        checkStatus(NeuroNativeLibrary.invokeInt(deviceGetNameHandle, name, 256,
            deviceOut.get(ValueLayout.JAVA_INT, 0L)), "cuDeviceGetName")
        name.getString(0L)
    }

    fun loadModule(ptx: ByteArray): Module = Arena.ofConfined().use { local ->
        require(ptx.isNotEmpty()) { "CUDA module image is empty" }
        val image = local.allocate(ptx.size.toLong() + 1)
        MemorySegment.copy(ptx, 0, image, ValueLayout.JAVA_BYTE, 0L, ptx.size)
        val out = local.allocate(ValueLayout.ADDRESS)
        checkStatus(NeuroNativeLibrary.invokeInt(moduleLoadHandle, out, image), "cuModuleLoadData")
        NeuroLog.debug("cuda", "module.loaded") { mapOf("library" to libraryName, "ptxBytes" to ptx.size) }
        Module(out.get(ValueLayout.ADDRESS, 0L))
    }

    fun launch(function: MemorySegment, elements: Int, vararg arguments: Argument) {
        require(elements >= 0) { "CUDA launch element count must be non-negative" }
        if (elements == 0) return
        val block = 256
        val grid = ((elements.toLong() + block - 1) / block).toInt()
        NeuroLog.trace("cuda", "kernel.grid.submitted") { mapOf("workItems" to elements, "grid" to grid,
            "block" to block, "arguments" to arguments.size, "stream" to "default") }
        Arena.ofConfined().use { local ->
            val params = local.allocate(ValueLayout.ADDRESS, arguments.size.toLong())
            for (index in arguments.indices) {
                val storage = when (val argument = arguments[index]) {
                    is Argument.Pointer -> local.allocateFrom(ValueLayout.ADDRESS, argument.value)
                    is Argument.IntValue -> local.allocateFrom(ValueLayout.JAVA_INT, argument.value)
                    is Argument.DoubleValue -> local.allocateFrom(ValueLayout.JAVA_DOUBLE, argument.value)
                    is Argument.FloatValue -> local.allocateFrom(ValueLayout.JAVA_FLOAT, argument.value)
                }
                params.setAtIndex(ValueLayout.ADDRESS, index.toLong(), storage)
            }
            checkStatus(NeuroNativeLibrary.invokeInt(launchHandle, function, grid, 1, 1, block, 1, 1, 0,
                MemorySegment.NULL, params, MemorySegment.NULL), "cuLaunchKernel")
        }
    }

    override fun close() {
        if (closed) return
        closed = true
        arena.close()
    }

    private fun checkStatus(status: Int, operation: String) {
        NeuroNativeLibrary.checkStatus(status, operation, "cuda") { operation + " failed with CUDA driver status " + status }
    }

    companion object {
        fun tryCreate(loader: NeuroLibraryLoader = NeuroNativeLibrary::open): NeuroCudaDriver? {
            val arena = Arena.ofShared()
            return try {
                val candidates = if (System.getProperty("os.name").lowercase().contains("windows")) {
                    listOf("nvcuda.dll")
                } else {
                    listOf("libcuda.so.1", "libcuda.so", "/usr/lib/wsl/lib/libcuda.so.1")
                }
                val loaded = loader(arena, "JNEURO_CUDA_DRIVER", candidates,
                    listOf("cuInit", "cuDeviceGet", "cuDeviceGetName", "cuModuleLoadData",
                        "cuModuleUnload", "cuModuleGetFunction", "cuLaunchKernel"))
                val init = NeuroNativeLibrary.downcall(loaded, "cuInit",
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT))
                val initStatus = NeuroNativeLibrary.invokeInt(init, 0)
                NeuroNativeLibrary.checkStatus(initStatus, "cuInit", "cuda") { "cuInit failed with status " + initStatus }
                NeuroCudaDriver(
                    arena, loaded.name,
                    NeuroNativeLibrary.downcall(loaded, "cuDeviceGet",
                        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT)),
                    NeuroNativeLibrary.downcall(loaded, "cuDeviceGetName",
                        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT)),
                    NeuroNativeLibrary.downcall(loaded, "cuModuleLoadData",
                        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS)),
                    NeuroNativeLibrary.downcall(loaded, "cuModuleUnload",
                        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS)),
                    NeuroNativeLibrary.downcall(loaded, "cuModuleGetFunction",
                        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS)),
                    NeuroNativeLibrary.downcall(loaded, "cuLaunchKernel",
                        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS,
                            ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS))
                )
            } catch (exception: RuntimeException) {
                NeuroLog.debug("cuda", "adapter.unavailable") { mapOf("adapter" to "NeuroCudaDriver", "reason" to exception.message) }
                arena.close()
                null
            } catch (exception: UnsatisfiedLinkError) {
                NeuroLog.debug("cuda", "adapter.unavailable") { mapOf("adapter" to "NeuroCudaDriver", "reason" to exception.message) }
                arena.close()
                null
            }
        }
        fun create(loader: NeuroLibraryLoader = NeuroNativeLibrary::open): NeuroCudaDriver = tryCreate(loader) ?: throw IllegalStateException(
            "CUDA Driver API is unavailable. Install a compatible NVIDIA driver or set JNEURO_CUDA_DRIVER.")
    }
}
