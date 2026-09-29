package com.lis.neuro

import java.lang.foreign.Arena
import java.lang.foreign.FunctionDescriptor
import java.lang.foreign.Linker
import java.lang.foreign.MemorySegment
import java.lang.foreign.SymbolLookup
import java.lang.foreign.ValueLayout
import java.lang.invoke.MethodHandle
import java.util.Optional

object NeuroNativeBlas {
    private const val ROW_MAJOR = 101
    private const val NO_TRANS = 111
    private val libraries = arrayOf("libopenblas.so.0", "libopenblas.so", "libblas.so.3", "libblas.so", "libmkl_rt.so")

    @JvmStatic fun tryCreate(network: Neuro): Optional<Session> {
        val arena = Arena.ofShared()
        return try {
            val resolved = resolveDgemm(arena)
            Optional.of(Session(network, arena, resolved.first, resolved.second))
        } catch (exception: IllegalArgumentException) {
            arena.close()
            Optional.empty()
        } catch (exception: IllegalStateException) {
            arena.close()
            Optional.empty()
        } catch (exception: IllegalCallerException) {
            arena.close()
            Optional.empty()
        } catch (exception: UnsatisfiedLinkError) {
            arena.close()
            Optional.empty()
        }
    }

    class Session internal constructor(network: Neuro, private val arena: Arena,
                                       private val handle: MethodHandle, private val libraryName: String) : AutoCloseable {
        private val topology = network.topology()
        private val maxWidth = topology.max()
        private val beta = network.hyperParameters().beta
        private val sigmoidMode = network.hyperParameters().sigmoidMode
        private val biases = Array(topology.size - 1) { network.backendBiases(it) }
        private val weights = Array(topology.size - 1) { layer ->
            val inputs = topology[layer]
            val outputs = topology[layer + 1]
            val source = network.backendWeights(layer)
            val transposed = DoubleArray(source.size)
            for (output in 0 until outputs) {
                for (input in 0 until inputs) transposed[input * outputs + output] = source[output * inputs + input]
            }
            arena.allocate(ValueLayout.JAVA_DOUBLE, transposed.size.toLong()).also {
                MemorySegment.copy(transposed, 0, it, ValueLayout.JAVA_DOUBLE, 0L, transposed.size)
            }
        }
        private var firstBuffer = MemorySegment.NULL
        private var secondBuffer = MemorySegment.NULL
        private var capacity = 0

        fun libraryName(): String = libraryName
        fun predictBatch(inputs: DoubleArray, batchSize: Int, outputs: DoubleArray) {
            require(batchSize >= 0) { "batchSize must be >= 0" }
            val inputElements = Math.multiplyExact(batchSize, topology[0])
            val outputElements = Math.multiplyExact(batchSize, topology.last())
            require(inputs.size >= inputElements && outputs.size >= outputElements) { "batch arrays are too small" }
            if (batchSize == 0) return
            ensureCapacity(batchSize)
            MemorySegment.copy(inputs, 0, firstBuffer, ValueLayout.JAVA_DOUBLE, 0L, inputElements)
            var current = firstBuffer
            var next = secondBuffer
            for (layer in weights.indices) {
                dgemm(batchSize, topology[layer + 1], topology[layer], current, weights[layer], next)
                activate(next, biases[layer], batchSize, topology[layer + 1])
                val swap = current
                current = next
                next = swap
            }
            MemorySegment.copy(current, ValueLayout.JAVA_DOUBLE, 0L, outputs, 0, outputElements)
        }
        private fun ensureCapacity(batchSize: Int) {
            if (batchSize <= capacity) return
            capacity = maxOf(batchSize, maxOf(8, capacity * 2))
            val elements = Math.multiplyExact(capacity, maxWidth).toLong()
            firstBuffer = arena.allocate(ValueLayout.JAVA_DOUBLE, elements)
            secondBuffer = arena.allocate(ValueLayout.JAVA_DOUBLE, elements)
        }
        private fun dgemm(rows: Int, columns: Int, inner: Int, left: MemorySegment,
                          right: MemorySegment, destination: MemorySegment) {
            try {
                handle.invoke(ROW_MAJOR, NO_TRANS, NO_TRANS, rows, columns, inner,
                    1.0, left, inner, right, columns, 0.0, destination, columns)
            } catch (throwable: Throwable) {
                throw IllegalStateException("cblas_dgemm failed", throwable)
            }
        }
        private fun activate(values: MemorySegment, layerBiases: DoubleArray, batchSize: Int, width: Int) {
            for (sample in 0 until batchSize) {
                for (output in 0 until width) {
                    val index = (sample * width + output).toLong()
                    val value = values.getAtIndex(ValueLayout.JAVA_DOUBLE, index) + layerBiases[output]
                    values.setAtIndex(ValueLayout.JAVA_DOUBLE, index, Neuro.activate(value * beta, sigmoidMode))
                }
            }
        }
        override fun close() = arena.close()
    }

    private fun resolveDgemm(arena: Arena): Pair<MethodHandle, String> {
        val linker = Linker.nativeLinker()
        var lastFailure: RuntimeException? = null
        for (library in libraries) {
            try {
                val symbol = SymbolLookup.libraryLookup(library, arena).find("cblas_dgemm")
                if (symbol.isPresent) {
                    val descriptor = FunctionDescriptor.ofVoid(
                        ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                        ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                        ValueLayout.JAVA_DOUBLE, ValueLayout.ADDRESS, ValueLayout.JAVA_INT,
                        ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_DOUBLE,
                        ValueLayout.ADDRESS, ValueLayout.JAVA_INT
                    )
                    return linker.downcallHandle(symbol.orElseThrow(), descriptor) to library
                }
            } catch (exception: IllegalArgumentException) {
                lastFailure = exception
            }
        }
        throw IllegalStateException("No CBLAS implementation with cblas_dgemm found", lastFailure)
    }
}
