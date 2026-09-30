package com.lis.neuro

import java.lang.foreign.Arena
import java.lang.foreign.FunctionDescriptor
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout
import java.lang.invoke.MethodHandle

internal class NeuroCudaRuntime private constructor(
    private val arena: Arena,
    val libraryName: String,
    private val getDeviceCountHandle: MethodHandle,
    private val setDeviceHandle: MethodHandle,
    private val getDeviceHandle: MethodHandle,
    private val getAttributeHandle: MethodHandle,
    private val mallocHandle: MethodHandle,
    private val freeHandle: MethodHandle,
    private val memcpyHandle: MethodHandle,
    private val synchronizeHandle: MethodHandle
) : NeuroCudaMemory, AutoCloseable {
    private var closed = false
    fun deviceCount(): Int = Arena.ofConfined().use { local ->
        val out = local.allocate(ValueLayout.JAVA_INT)
        checkStatus(NeuroNativeLibrary.invokeInt(getDeviceCountHandle, out), "cudaGetDeviceCount")
        out.get(ValueLayout.JAVA_INT, 0L)
    }

    fun setDevice(device: Int) {
        require(device >= 0)
        checkStatus(NeuroNativeLibrary.invokeInt(setDeviceHandle, device), "cudaSetDevice")
        NeuroLog.debug("cuda", "runtime.device.selected") { mapOf("deviceIndex" to device, "library" to libraryName) }
    }

    fun currentDevice(): Int = Arena.ofConfined().use { local ->
        val out = local.allocate(ValueLayout.JAVA_INT)
        checkStatus(NeuroNativeLibrary.invokeInt(getDeviceHandle, out), "cudaGetDevice")
        out.get(ValueLayout.JAVA_INT, 0L)
    }

    fun computeCapability(device: Int): Pair<Int, Int> =
        attribute(COMPUTE_CAPABILITY_MAJOR, device) to attribute(COMPUTE_CAPABILITY_MINOR, device)

    override fun allocate(byteSize: Long): MemorySegment {
        require(byteSize > 0L)
        return Arena.ofConfined().use { local ->
            val out = local.allocate(ValueLayout.ADDRESS)
            checkStatus(NeuroNativeLibrary.invokeInt(mallocHandle, out, byteSize), "cudaMalloc")
            NeuroLog.debug("cuda", "allocation.created") { mapOf("bytes" to byteSize, "api" to "runtime") }
            out.get(ValueLayout.ADDRESS, 0L)
        }
    }

    override fun release(pointer: MemorySegment) {
        if (pointer == MemorySegment.NULL) return
        checkStatus(NeuroNativeLibrary.invokeInt(freeHandle, pointer), "cudaFree")
        NeuroLog.debug("cuda", "allocation.released") { mapOf("api" to "runtime") }
    }

    override fun copyHostToDevice(device: MemorySegment, host: MemorySegment, byteSize: Long) {
        require(byteSize >= 0L && byteSize <= host.byteSize())
        if (byteSize == 0L) return
        NeuroLog.trace("cuda", "transfer.submitted") { mapOf("direction" to "host-to-device", "bytes" to byteSize, "api" to "runtime") }
        checkStatus(NeuroNativeLibrary.invokeInt(memcpyHandle, device, host, byteSize, HOST_TO_DEVICE), "cudaMemcpy H2D")
    }

    override fun copyDeviceToHost(host: MemorySegment, device: MemorySegment, byteSize: Long) {
        require(byteSize >= 0L && byteSize <= host.byteSize())
        if (byteSize == 0L) return
        NeuroLog.trace("cuda", "transfer.submitted") { mapOf("direction" to "device-to-host", "bytes" to byteSize, "api" to "runtime") }
        checkStatus(NeuroNativeLibrary.invokeInt(memcpyHandle, host, device, byteSize, DEVICE_TO_HOST), "cudaMemcpy D2H")
    }

    override fun synchronize() {
        checkStatus(NeuroNativeLibrary.invokeInt(synchronizeHandle), "cudaDeviceSynchronize")
        NeuroLog.trace("cuda", "device.synchronized") { mapOf("api" to "runtime") }
    }

    override fun close() {
        if (closed) return
        closed = true
        arena.close()
    }

    private fun attribute(attribute: Int, device: Int): Int = Arena.ofConfined().use { local ->
        val out = local.allocate(ValueLayout.JAVA_INT)
        checkStatus(NeuroNativeLibrary.invokeInt(getAttributeHandle, out, attribute, device), "cudaDeviceGetAttribute")
        out.get(ValueLayout.JAVA_INT, 0L)
    }

    private fun checkStatus(status: Int, operation: String) {
        NeuroNativeLibrary.checkStatus(status, operation, "cuda") { operation + " failed with CUDA status " + status }
    }

    companion object {
        private const val HOST_TO_DEVICE = 1
        private const val DEVICE_TO_HOST = 2
        private const val COMPUTE_CAPABILITY_MAJOR = 75
        private const val COMPUTE_CAPABILITY_MINOR = 76

        fun tryCreate(loader: NeuroLibraryLoader = NeuroNativeLibrary::open): NeuroCudaRuntime? {
            val arena = Arena.ofShared()
            return try {
                val candidates = if (System.getProperty("os.name").lowercase().contains("windows")) {
                    listOf("cudart64_130.dll", "cudart64_13.dll", "cudart64_120.dll", "cudart64_12.dll")
                } else {
                    listOf("libcudart.so", "libcudart.so.13", "libcudart.so.12")
                }
                val required = listOf("cudaGetDeviceCount", "cudaSetDevice", "cudaGetDevice", "cudaDeviceGetAttribute",
                    "cudaMalloc", "cudaFree", "cudaMemcpy", "cudaDeviceSynchronize")
                val loaded = loader(arena, "JNEURO_CUDA_RUNTIME", candidates, required)
                val intAddress = FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS)
                NeuroCudaRuntime(
                    arena, loaded.name,
                    NeuroNativeLibrary.downcall(loaded, "cudaGetDeviceCount", intAddress),
                    NeuroNativeLibrary.downcall(loaded, "cudaSetDevice",
                        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT)),
                    NeuroNativeLibrary.downcall(loaded, "cudaGetDevice", intAddress),
                    NeuroNativeLibrary.downcall(loaded, "cudaDeviceGetAttribute",
                        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT)),
                    NeuroNativeLibrary.downcall(loaded, "cudaMalloc",
                        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG)),
                    NeuroNativeLibrary.downcall(loaded, "cudaFree",
                        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS)),
                    NeuroNativeLibrary.downcall(loaded, "cudaMemcpy",
                        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                            ValueLayout.JAVA_LONG, ValueLayout.JAVA_INT)),
                    NeuroNativeLibrary.downcall(loaded, "cudaDeviceSynchronize",
                        FunctionDescriptor.of(ValueLayout.JAVA_INT))
                )
            } catch (exception: RuntimeException) {
                NeuroLog.debug("cuda", "adapter.unavailable") { mapOf("adapter" to "NeuroCudaRuntime", "reason" to exception.message) }
                arena.close()
                null
            } catch (exception: UnsatisfiedLinkError) {
                NeuroLog.debug("cuda", "adapter.unavailable") { mapOf("adapter" to "NeuroCudaRuntime", "reason" to exception.message) }
                arena.close()
                null
            }
        }

        fun create(loader: NeuroLibraryLoader = NeuroNativeLibrary::open): NeuroCudaRuntime = tryCreate(loader) ?: throw IllegalStateException(
            "CUDA Runtime is unavailable. Install a compatible NVIDIA driver/toolkit or set JNEURO_CUDA_RUNTIME.")
    }
}
