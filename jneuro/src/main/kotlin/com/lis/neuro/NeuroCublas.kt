package com.lis.neuro

import java.lang.foreign.Arena
import java.lang.foreign.FunctionDescriptor
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout
import java.lang.invoke.MethodHandle

internal class NeuroCublas private constructor(
    private val arena: Arena,
    val libraryName: String,
    private val handle: MemorySegment,
    private val destroyHandle: MethodHandle,
    private val dgemmHandle: MethodHandle,
    private val sgemmHandle: MethodHandle,
    private val versionHandle: MethodHandle?
) : AutoCloseable {
    private var closed = false

    fun version(): Int? = versionHandle?.let { function ->
        Arena.ofConfined().use { local ->
            val out = local.allocate(ValueLayout.JAVA_INT)
            val status = NeuroNativeLibrary.invokeInt(function, handle, out)
            NeuroNativeLibrary.checkStatus(status, "cublasGetVersion_v2", "cublas") { "cublasGetVersion_v2 failed with status " + status }
            out.get(ValueLayout.JAVA_INT, 0L)
        }
    }

    fun forward(input: MemorySegment, weights: MemorySegment, output: MemorySegment,
                batch: Int, inputs: Int, outputs: Int) {
        gemmDouble(TRANSPOSE, NO_TRANSPOSE, outputs, batch, inputs, weights, inputs, input, inputs, output, outputs)
    }

    fun backward(nextDelta: MemorySegment, nextWeights: MemorySegment, currentDelta: MemorySegment,
                 batch: Int, currentWidth: Int, nextWidth: Int) {
        gemmDouble(NO_TRANSPOSE, NO_TRANSPOSE, currentWidth, batch, nextWidth,
            nextWeights, currentWidth, nextDelta, nextWidth, currentDelta, currentWidth)
    }

    fun weightGradient(source: MemorySegment, delta: MemorySegment, gradient: MemorySegment,
                       batch: Int, inputs: Int, outputs: Int) {
        gemmDouble(NO_TRANSPOSE, TRANSPOSE, inputs, outputs, batch, source, inputs, delta, outputs, gradient, inputs)
    }

    fun forwardFloat(input: MemorySegment, weights: MemorySegment, output: MemorySegment,
                     batch: Int, inputs: Int, outputs: Int) {
        gemmFloat(TRANSPOSE, NO_TRANSPOSE, outputs, batch, inputs, weights, inputs, input, inputs, output, outputs)
    }

    fun backwardFloat(nextDelta: MemorySegment, nextWeights: MemorySegment, currentDelta: MemorySegment,
                      batch: Int, currentWidth: Int, nextWidth: Int) {
        gemmFloat(NO_TRANSPOSE, NO_TRANSPOSE, currentWidth, batch, nextWidth,
            nextWeights, currentWidth, nextDelta, nextWidth, currentDelta, currentWidth)
    }

    fun weightGradientFloat(source: MemorySegment, delta: MemorySegment, gradient: MemorySegment,
                            batch: Int, inputs: Int, outputs: Int) {
        gemmFloat(NO_TRANSPOSE, TRANSPOSE, inputs, outputs, batch, source, inputs, delta, outputs, gradient, inputs)
    }

    private fun gemmDouble(transA: Int, transB: Int, m: Int, n: Int, k: Int,
                     a: MemorySegment, lda: Int, b: MemorySegment, ldb: Int,
                     c: MemorySegment, ldc: Int) {
        NeuroLog.trace("cublas", "gemm.submitted") { mapOf("precision" to "FP64", "m" to m, "n" to n, "k" to k,
            "transposeA" to transA, "transposeB" to transB, "lda" to lda, "ldb" to ldb, "ldc" to ldc, "stream" to "default") }
        Arena.ofConfined().use { local ->
            val alpha = local.allocateFrom(ValueLayout.JAVA_DOUBLE, 1.0)
            val beta = local.allocateFrom(ValueLayout.JAVA_DOUBLE, 0.0)
            val status = NeuroNativeLibrary.invokeInt(dgemmHandle, handle, transA, transB, m, n, k,
                alpha, a, lda, b, ldb, beta, c, ldc)
            NeuroNativeLibrary.checkStatus(status, "cublasDgemm_v2", "cublas") { "cublasDgemm_v2 failed with status " + status }
        }
    }

    private fun gemmFloat(transA: Int, transB: Int, m: Int, n: Int, k: Int,
                          a: MemorySegment, lda: Int, b: MemorySegment, ldb: Int,
                          c: MemorySegment, ldc: Int) {
        NeuroLog.trace("cublas", "gemm.submitted") { mapOf("precision" to "FP32", "m" to m, "n" to n, "k" to k,
            "transposeA" to transA, "transposeB" to transB, "lda" to lda, "ldb" to ldb, "ldc" to ldc, "stream" to "default") }
        Arena.ofConfined().use { local ->
            val alpha = local.allocateFrom(ValueLayout.JAVA_FLOAT, 1.0f)
            val beta = local.allocateFrom(ValueLayout.JAVA_FLOAT, 0.0f)
            val status = NeuroNativeLibrary.invokeInt(sgemmHandle, handle, transA, transB, m, n, k,
                alpha, a, lda, b, ldb, beta, c, ldc)
            NeuroNativeLibrary.checkStatus(status, "cublasSgemm_v2", "cublas") { "cublasSgemm_v2 failed with status " + status }
        }
    }

    override fun close() {
        if (closed) return
        closed = true
        try {
            val status = NeuroNativeLibrary.invokeInt(destroyHandle, handle)
            NeuroNativeLibrary.checkStatus(status, "cublasDestroy_v2", "cublas") { "cublasDestroy_v2 failed with status " + status }
        } finally {
            arena.close()
            NeuroLog.debug("cublas", "adapter.closed") { mapOf("library" to libraryName) }
        }
    }

    companion object {
        private const val NO_TRANSPOSE = 0
        private const val TRANSPOSE = 1

        fun tryCreate(loader: NeuroLibraryLoader = NeuroNativeLibrary::open): NeuroCublas? {
            val arena = Arena.ofShared()
            return try {
                val candidates = if (System.getProperty("os.name").lowercase().contains("windows")) {
                    listOf("cublas64_13.dll", "cublas64_12.dll")
                } else {
                    listOf("libcublas.so", "libcublas.so.13", "libcublas.so.12")
                }
                val loaded = loader(arena, "JNEURO_CUBLAS", candidates,
                    listOf("cublasCreate_v2", "cublasDestroy_v2", "cublasDgemm_v2", "cublasSgemm_v2"))
                val create = NeuroNativeLibrary.downcall(loaded, "cublasCreate_v2",
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS))
                val destroy = NeuroNativeLibrary.downcall(loaded, "cublasDestroy_v2",
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS))
                val gemmDescriptor = FunctionDescriptor.of(ValueLayout.JAVA_INT,
                        ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                        ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                        ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT,
                        ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.ADDRESS,
                        ValueLayout.ADDRESS, ValueLayout.JAVA_INT)
                val dgemm = NeuroNativeLibrary.downcall(loaded, "cublasDgemm_v2", gemmDescriptor)
                val sgemm = NeuroNativeLibrary.downcall(loaded, "cublasSgemm_v2", gemmDescriptor)
                val version = if (loaded.lookup.find("cublasGetVersion_v2").isPresent)
                    NeuroNativeLibrary.downcall(loaded, "cublasGetVersion_v2",
                        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS)) else null
                val handle = Arena.ofConfined().use { local ->
                    val out = local.allocate(ValueLayout.ADDRESS)
                    val status = NeuroNativeLibrary.invokeInt(create, out)
                    NeuroNativeLibrary.checkStatus(status, "cublasCreate_v2", "cublas") { "cublasCreate_v2 failed with status " + status }
                    out.get(ValueLayout.ADDRESS, 0L)
                }
                NeuroCublas(arena, loaded.name, handle, destroy, dgemm, sgemm, version)
            } catch (exception: RuntimeException) {
                NeuroLog.debug("cublas", "adapter.unavailable") { mapOf("adapter" to "NeuroCublas", "reason" to exception.message) }
                arena.close()
                null
            } catch (exception: UnsatisfiedLinkError) {
                NeuroLog.debug("cublas", "adapter.unavailable") { mapOf("adapter" to "NeuroCublas", "reason" to exception.message) }
                arena.close()
                null
            }
        }

        fun create(loader: NeuroLibraryLoader = NeuroNativeLibrary::open): NeuroCublas = tryCreate(loader) ?: throw IllegalStateException(
            "cuBLAS is unavailable. Install a compatible CUDA Toolkit or set JNEURO_CUBLAS.")
    }
}
