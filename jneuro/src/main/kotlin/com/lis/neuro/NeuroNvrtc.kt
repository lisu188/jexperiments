package com.lis.neuro

import java.lang.foreign.Arena
import java.lang.foreign.FunctionDescriptor
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout
import java.lang.invoke.MethodHandle

internal class NeuroNvrtc private constructor(
    private val arena: Arena,
    val libraryName: String,
    private val createProgramHandle: MethodHandle,
    private val destroyProgramHandle: MethodHandle,
    private val compileProgramHandle: MethodHandle,
    private val getPtxSizeHandle: MethodHandle,
    private val getPtxHandle: MethodHandle,
    private val getLogSizeHandle: MethodHandle,
    private val getLogHandle: MethodHandle
) : AutoCloseable {
    fun compile(source: String, name: String, computeCapability: Pair<Int, Int>): ByteArray = Arena.ofConfined().use { local ->
        val programOut = local.allocate(ValueLayout.ADDRESS)
        val sourceSegment = local.allocateFrom(source)
        val nameSegment = local.allocateFrom(name)
        checkStatus(NeuroNativeLibrary.invokeInt(createProgramHandle, programOut, sourceSegment, nameSegment, 0,
            MemorySegment.NULL, MemorySegment.NULL), "nvrtcCreateProgram")
        val program = programOut.get(ValueLayout.ADDRESS, 0L)
        try {
            val option = local.allocateFrom("--gpu-architecture=compute_" + computeCapability.first + computeCapability.second)
            val options = local.allocate(ValueLayout.ADDRESS)
            options.set(ValueLayout.ADDRESS, 0L, option)
            val compileStatus = NeuroNativeLibrary.invokeInt(compileProgramHandle, program, 1, options)
            if (compileStatus != 0) {
                throw IllegalStateException("nvrtcCompileProgram failed with status " + compileStatus + ": " + log(program, local))
            }
            val sizeOut = local.allocate(ValueLayout.JAVA_LONG)
            checkStatus(NeuroNativeLibrary.invokeInt(getPtxSizeHandle, program, sizeOut), "nvrtcGetPTXSize")
            val size = sizeOut.get(ValueLayout.JAVA_LONG, 0L)
            require(size in 1..Int.MAX_VALUE.toLong()) { "NVRTC produced an invalid PTX size" }
            val ptx = local.allocate(size)
            checkStatus(NeuroNativeLibrary.invokeInt(getPtxHandle, program, ptx), "nvrtcGetPTX")
            ByteArray(size.toInt()).also {
                MemorySegment.copy(ptx, ValueLayout.JAVA_BYTE, 0L, it, 0, it.size)
            }
        } finally {
            checkStatus(NeuroNativeLibrary.invokeInt(destroyProgramHandle, programOut), "nvrtcDestroyProgram")
        }
    }

    private fun log(program: MemorySegment, local: Arena): String {
        val sizeOut = local.allocate(ValueLayout.JAVA_LONG)
        val status = NeuroNativeLibrary.invokeInt(getLogSizeHandle, program, sizeOut)
        if (status != 0) return "log unavailable"
        val size = sizeOut.get(ValueLayout.JAVA_LONG, 0L)
        if (size <= 1L || size > Int.MAX_VALUE) return ""
        val buffer = local.allocate(size)
        if (NeuroNativeLibrary.invokeInt(getLogHandle, program, buffer) != 0) return "log unavailable"
        return buffer.getString(0L)
    }

    override fun close() = arena.close()

    private fun checkStatus(status: Int, operation: String) {
        check(status == 0) { operation + " failed with status " + status }
    }

    companion object {
        fun tryCreate(): NeuroNvrtc? {
            val arena = Arena.ofShared()
            return try {
                val candidates = if (System.getProperty("os.name").lowercase().contains("windows")) {
                    listOf("nvrtc64_130_0.dll", "nvrtc64_130.dll", "nvrtc64_120_0.dll", "nvrtc64_120.dll")
                } else {
                    listOf("libnvrtc.so", "libnvrtc.so.13", "libnvrtc.so.12")
                }
                val loaded = NeuroNativeLibrary.open(arena, "JNEURO_NVRTC", candidates,
                    listOf("nvrtcCreateProgram", "nvrtcDestroyProgram", "nvrtcCompileProgram",
                        "nvrtcGetPTXSize", "nvrtcGetPTX", "nvrtcGetProgramLogSize", "nvrtcGetProgramLog"))
                NeuroNvrtc(
                    arena, loaded.name,
                    NeuroNativeLibrary.downcall(loaded, "nvrtcCreateProgram",
                        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                            ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS)),
                    NeuroNativeLibrary.downcall(loaded, "nvrtcDestroyProgram",
                        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS)),
                    NeuroNativeLibrary.downcall(loaded, "nvrtcCompileProgram",
                        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.ADDRESS)),
                    NeuroNativeLibrary.downcall(loaded, "nvrtcGetPTXSize",
                        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS)),
                    NeuroNativeLibrary.downcall(loaded, "nvrtcGetPTX",
                        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS)),
                    NeuroNativeLibrary.downcall(loaded, "nvrtcGetProgramLogSize",
                        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS)),
                    NeuroNativeLibrary.downcall(loaded, "nvrtcGetProgramLog",
                        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS))
                )
            } catch (exception: RuntimeException) {
                arena.close()
                null
            } catch (exception: UnsatisfiedLinkError) {
                arena.close()
                null
            }
        }

        fun create(): NeuroNvrtc = tryCreate() ?: throw IllegalStateException(
            "NVRTC is unavailable. Install a compatible CUDA Toolkit or set JNEURO_NVRTC.")
    }
}
