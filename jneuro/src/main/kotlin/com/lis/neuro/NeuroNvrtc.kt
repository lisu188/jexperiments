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
    private val getLogHandle: MethodHandle,
    private val versionHandle: MethodHandle
) : AutoCloseable {
    private var closed = false

    fun version(): Pair<Int, Int> = Arena.ofConfined().use { local ->
        val major = local.allocate(ValueLayout.JAVA_INT)
        val minor = local.allocate(ValueLayout.JAVA_INT)
        checkStatus(NeuroNativeLibrary.invokeInt(versionHandle, major, minor), "nvrtcVersion")
        major.get(ValueLayout.JAVA_INT, 0L) to minor.get(ValueLayout.JAVA_INT, 0L)
    }

    fun compile(source: String, name: String, computeCapability: Pair<Int, Int>): ByteArray = Arena.ofConfined().use { local ->
        val compilation = NeuroLog.id("nvrtc")
        val started = System.nanoTime()
        NeuroLog.debug("cuda", "compilation.started") { mapOf("compilation" to compilation, "source" to name,
            "sourceCharacters" to source.length, "options" to compilerOptions(computeCapability).joinToString(","), "library" to libraryName) }
        val programOut = local.allocate(ValueLayout.ADDRESS)
        val sourceSegment = local.allocateFrom(source)
        val nameSegment = local.allocateFrom(name)
        checkStatus(NeuroNativeLibrary.invokeInt(createProgramHandle, programOut, sourceSegment, nameSegment, 0,
            MemorySegment.NULL, MemorySegment.NULL), "nvrtcCreateProgram")
        val program = programOut.get(ValueLayout.ADDRESS, 0L)
        var failure: Throwable? = null
        try {
            val values = compilerOptions(computeCapability)
            val options = local.allocate(ValueLayout.ADDRESS, values.size.toLong())
            for ((index, value) in values.withIndex()) {
                options.setAtIndex(ValueLayout.ADDRESS, index.toLong(), local.allocateFrom(value))
            }
            val compileStatus = NeuroNativeLibrary.invokeInt(compileProgramHandle, program, values.size, options)
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
                NeuroLog.debug("cuda", "compilation.completed") { mapOf("compilation" to compilation,
                    "ptxBytes" to size, "elapsedMs" to (System.nanoTime() - started) / 1_000_000.0) }
            }
        } catch (exception: Throwable) {
            NeuroLog.error("cuda", "compilation.failed", exception, "compilation" to compilation, "source" to name,
                "hint" to "Check NVRTC compatibility with the selected compute capability and the compiler diagnostic.")
            failure = exception
            throw exception
        } finally {
            try {
                checkStatus(NeuroNativeLibrary.invokeInt(destroyProgramHandle, programOut), "nvrtcDestroyProgram")
            } catch (cleanup: Throwable) {
                NeuroLog.error("cuda", "compilation.cleanup.failed", cleanup, "compilation" to compilation)
                if (failure != null) failure.addSuppressed(cleanup) else throw cleanup
            }
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

    override fun close() {
        if (closed) return
        closed = true
        arena.close()
    }

    private fun checkStatus(status: Int, operation: String) {
        NeuroNativeLibrary.checkStatus(status, operation, "cuda") { operation + " failed with status " + status }
    }

    companion object {
        internal fun compilerOptions(computeCapability: Pair<Int, Int>): List<String> =
            listOf("--gpu-architecture=compute_" + computeCapability.first + computeCapability.second)

        fun tryCreate(loader: NeuroLibraryLoader = NeuroNativeLibrary::open): NeuroNvrtc? {
            val arena = Arena.ofShared()
            return try {
                val candidates = if (System.getProperty("os.name").lowercase().contains("windows")) {
                    listOf("nvrtc64_130_0.dll", "nvrtc64_130.dll", "nvrtc64_120_0.dll", "nvrtc64_120.dll")
                } else {
                    listOf("libnvrtc.so", "libnvrtc.so.13", "libnvrtc.so.12")
                }
                val loaded = loader(arena, "JNEURO_NVRTC", candidates,
                    listOf("nvrtcCreateProgram", "nvrtcDestroyProgram", "nvrtcCompileProgram",
                        "nvrtcGetPTXSize", "nvrtcGetPTX", "nvrtcGetProgramLogSize", "nvrtcGetProgramLog", "nvrtcVersion"))
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
                        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS)),
                    NeuroNativeLibrary.downcall(loaded, "nvrtcVersion",
                        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS))
                )
            } catch (exception: RuntimeException) {
                NeuroLog.debug("cuda", "adapter.unavailable") { mapOf("adapter" to "NeuroNvrtc", "reason" to exception.message) }
                arena.close()
                null
            } catch (exception: UnsatisfiedLinkError) {
                NeuroLog.debug("cuda", "adapter.unavailable") { mapOf("adapter" to "NeuroNvrtc", "reason" to exception.message) }
                arena.close()
                null
            }
        }

        fun create(loader: NeuroLibraryLoader = NeuroNativeLibrary::open): NeuroNvrtc = tryCreate(loader) ?: throw IllegalStateException(
            "NVRTC is unavailable. Install a compatible CUDA Toolkit or set JNEURO_NVRTC.")
    }
}
