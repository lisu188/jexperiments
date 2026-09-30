package com.lis.neuro

import java.lang.foreign.Arena
import java.lang.foreign.FunctionDescriptor
import java.lang.foreign.Linker
import java.lang.foreign.MemoryLayout
import java.lang.foreign.MemorySegment
import java.lang.foreign.SymbolLookup
import java.lang.foreign.ValueLayout
import java.lang.invoke.MethodHandles
import java.lang.invoke.MethodType
import java.util.Optional

/** Actual FFM entrypoints backed by host memory; deliberately does not execute GPU kernels. */
internal class CudaNativeTestRuntime : AutoCloseable {
    private val memory = Arena.ofShared()
    val calls = ArrayList<String>()
    val errors = HashMap<String, Int>()
    val missing = HashSet<String>()
    val allocations = LinkedHashMap<Long, MemorySegment>()
    val libraries = ArrayList<String>()
    val libraryArenas = ArrayList<Arena>()
    val callbackFailures = ArrayList<Throwable>()
    val launches = ArrayList<Launch>()
    val gemms = ArrayList<Gemm>()
    val compiledOptions = ArrayList<String>()
    val functionNames = ArrayList<String>()
    val functionSignatures = HashMap<String, List<ValueLayout>>().apply { putAll(KERNEL_SIGNATURES) }
    private val functions = HashMap<Long, String>()
    var deviceCount = 1
    var currentDevice = 0
    var ptx = "// fake PTX for ABI tests only\n\u0000".toByteArray()
    var log = "test compiler diagnostic\u0000".toByteArray()
    var ptxSize: Long? = null
    var logSize: Long? = null
    var failedFunction: String? = null
    var failDtoHAt = 0
    var nonfiniteDtoH = false
    var deviceToHostCopies = 0
    var beforeCall: ((String) -> Unit)? = null
    var loadedImage = ""

    val loader: NeuroLibraryLoader = { arena, environment, _, required ->
        libraries += environment
        libraryArenas += arena
        val lookup = lookup(arena)
        check(required.all { lookup.find(it).isPresent }) { "Missing required test symbol for $environment" }
        NeuroNativeLibrary.Loaded(lookup, "test-$environment")
    }

    fun lookup(arena: Arena): SymbolLookup {
        val symbols = HashMap<String, MemorySegment>()
        for ((name, layouts) in SIGNATURES) {
            if (name in missing) continue
            val callback = Callback { arguments ->
                try { call(name, arguments) } catch (failure: Throwable) {
                    callbackFailures += failure
                    9999
                }
            }
            val descriptor = FunctionDescriptor.of(ValueLayout.JAVA_INT, *layouts.toTypedArray())
            val method = MethodHandles.lookup().findVirtual(Callback::class.java, "invoke",
                MethodType.methodType(Int::class.javaPrimitiveType, Array<Any>::class.java))
                .bindTo(callback).asCollector(Array<Any>::class.java, layouts.size).asType(descriptor.toMethodType())
            symbols[name] = Linker.nativeLinker().upcallStub(method, descriptor, arena)
        }
        return SymbolLookup { name -> Optional.ofNullable(symbols[name]) }
    }

    private fun call(name: String, args: Array<Any>): Int {
        calls += name
        beforeCall?.invoke(name)
        errors[name]?.let { return it }
        fun pointer(index: Int, size: Long) = (args[index] as MemorySegment).reinterpret(size)
        fun integer(index: Int, value: Int) = pointer(index, 4).set(ValueLayout.JAVA_INT, 0, value)
        fun long(index: Int, value: Long) = pointer(index, 8).set(ValueLayout.JAVA_LONG, 0, value)
        fun address(index: Int, value: Long) = pointer(index, 8).set(ValueLayout.ADDRESS, 0, MemorySegment.ofAddress(value))
        when (name) {
            "cudaGetDeviceCount" -> integer(0, deviceCount)
            "cudaSetDevice" -> currentDevice = args[0] as Int
            "cudaGetDevice" -> integer(0, currentDevice)
            "cudaDeviceGetAttribute" -> integer(0, if (args[1] == 75) 8 else 6)
            "cudaMalloc" -> {
                val allocation = memory.allocate(args[1] as Long, 8)
                allocations[allocation.address()] = allocation
                address(0, allocation.address())
            }
            "cudaFree" -> checkNotNull(allocations.remove((args[0] as MemorySegment).address()))
            "cudaMemcpy" -> {
                val size = args[2] as Long
                if (args[3] == 2) {
                    deviceToHostCopies++
                    if (deviceToHostCopies == failDtoHAt) return 700
                }
                MemorySegment.copy(pointer(1, size), 0, pointer(0, size), 0, size)
                if (args[3] == 2 && nonfiniteDtoH && size >= 4) pointer(0, size).fill(0xff.toByte())
            }
            "cublasCreate_v2" -> address(0, 0x1000)
            "cublasGetVersion_v2" -> integer(1, 130002)
            "cublasDgemm_v2", "cublasSgemm_v2" -> {
                val single = name == "cublasSgemm_v2"
                fun scalar(index: Int): Double = if (single) pointer(index, 4).get(ValueLayout.JAVA_FLOAT, 0).toDouble()
                    else pointer(index, 8).get(ValueLayout.JAVA_DOUBLE, 0)
                gemms += Gemm(single, args[1] as Int, args[2] as Int, args[3] as Int, args[4] as Int,
                    args[5] as Int, (args[7] as MemorySegment).address(), args[8] as Int,
                    (args[9] as MemorySegment).address(), args[10] as Int,
                    (args[12] as MemorySegment).address(), args[13] as Int, scalar(6), scalar(11))
            }
            "nvrtcCreateProgram" -> address(0, 0x2000)
            "nvrtcVersion" -> { integer(0, 12); integer(1, 0) }
            "nvrtcCompileProgram" -> {
                val options = pointer(2, (args[1] as Int) * 8L)
                for (index in 0 until (args[1] as Int)) {
                    compiledOptions += options.getAtIndex(ValueLayout.ADDRESS, index.toLong()).reinterpret(256).getString(0)
                }
            }
            "nvrtcGetPTXSize" -> long(1, ptxSize ?: ptx.size.toLong())
            "nvrtcGetPTX" -> MemorySegment.copy(ptx, 0, pointer(1, ptx.size.toLong()), ValueLayout.JAVA_BYTE, 0, ptx.size)
            "nvrtcGetProgramLogSize" -> long(1, logSize ?: log.size.toLong())
            "nvrtcGetProgramLog" -> MemorySegment.copy(log, 0, pointer(1, log.size.toLong()), ValueLayout.JAVA_BYTE, 0, log.size)
            "cuDeviceGet" -> integer(0, args[1] as Int)
            "cuDeviceGetName" -> pointer(0, (args[1] as Int).toLong()).setString(0, "Adapter test GPU")
            "cuModuleLoadData" -> { loadedImage = pointer(1, 4096).getString(0); address(0, 0x3000) }
            "cuModuleGetFunction" -> {
                val function = pointer(2, 256).getString(0)
                if (function == failedFunction) return 500
                functionNames += function
                val id = 0x4000L + functionNames.size
                functions[id] = function
                address(0, id)
            }
            "cuLaunchKernel" -> {
                val function = functions.getValue((args[0] as MemorySegment).address())
                val layouts = functionSignatures[function] ?: emptyList()
                val pointers = pointer(9, layouts.size * 8L)
                val values = layouts.mapIndexed { index, layout ->
                    val value = pointers.getAtIndex(ValueLayout.ADDRESS, index.toLong()).reinterpret(layout.byteSize())
                    when (layout) {
                        ValueLayout.ADDRESS -> value.get(ValueLayout.ADDRESS, 0).address()
                        ValueLayout.JAVA_INT -> value.get(ValueLayout.JAVA_INT, 0)
                        ValueLayout.JAVA_DOUBLE -> value.get(ValueLayout.JAVA_DOUBLE, 0)
                        ValueLayout.JAVA_FLOAT -> value.get(ValueLayout.JAVA_FLOAT, 0)
                        else -> error("Unknown test argument layout")
                    }
                }
                launches += Launch(function, (1..7).map { args[it] as Int }, values, (args[8] as MemorySegment).address())
            }
        }
        return 0
    }

    fun assertHealthy() { check(callbackFailures.isEmpty()) { callbackFailures.joinToString { it.toString() } } }
    override fun close() { memory.close(); assertHealthy() }

    data class Launch(val name: String, val dimensions: List<Int>, val arguments: List<Any>, val stream: Long)
    data class Gemm(val single: Boolean, val transposeA: Int, val transposeB: Int, val m: Int, val n: Int, val k: Int,
                    val a: Long, val lda: Int, val b: Long, val ldb: Int, val c: Long, val ldc: Int,
                    val alpha: Double, val beta: Double)
    private class Callback(private val action: (Array<Any>) -> Int) {
        fun invoke(arguments: Array<Any>): Int = action(arguments)
    }

    companion object {
        private val A = ValueLayout.ADDRESS
        private val I = ValueLayout.JAVA_INT
        private val L = ValueLayout.JAVA_LONG
        private val D = ValueLayout.JAVA_DOUBLE
        private val F = ValueLayout.JAVA_FLOAT
        private val GEMM = listOf(A, I, I, I, I, I, A, A, I, A, I, A, A, I)
        private val SIGNATURES: Map<String, List<MemoryLayout>> = mapOf(
            "cudaGetDeviceCount" to listOf(A), "cudaSetDevice" to listOf(I), "cudaGetDevice" to listOf(A),
            "cudaDeviceGetAttribute" to listOf(A, I, I), "cudaMalloc" to listOf(A, L), "cudaFree" to listOf(A),
            "cudaMemcpy" to listOf(A, A, L, I), "cudaDeviceSynchronize" to emptyList(),
            "cublasCreate_v2" to listOf(A), "cublasDestroy_v2" to listOf(A), "cublasGetVersion_v2" to listOf(A, A),
            "cublasDgemm_v2" to GEMM, "cublasSgemm_v2" to GEMM,
            "nvrtcCreateProgram" to listOf(A, A, A, I, A, A), "nvrtcDestroyProgram" to listOf(A), "nvrtcVersion" to listOf(A, A),
            "nvrtcCompileProgram" to listOf(A, I, A), "nvrtcGetPTXSize" to listOf(A, A),
            "nvrtcGetPTX" to listOf(A, A), "nvrtcGetProgramLogSize" to listOf(A, A), "nvrtcGetProgramLog" to listOf(A, A),
            "cuInit" to listOf(I), "cuDeviceGet" to listOf(A, I), "cuDeviceGetName" to listOf(A, I, I),
            "cuModuleLoadData" to listOf(A, A), "cuModuleUnload" to listOf(A), "cuModuleGetFunction" to listOf(A, A, A),
            "cuLaunchKernel" to listOf(A, I, I, I, I, I, I, I, A, A, A)
        )
        private val KERNEL_SIGNATURES = mapOf(
            "gatherRows" to listOf(A, I, A, I, I, A), "gatherRowsFloat" to listOf(A, I, A, I, I, A),
            "addBiasAndSigmoid" to listOf(A, A, I, I, D, I), "addBiasAndSigmoidFloat" to listOf(A, A, I, I, F, I),
            "outputDelta" to listOf(A, A, A, I, D), "outputDeltaFloat" to listOf(A, A, A, I, F),
            "applySigmoidDerivative" to listOf(A, A, I, D), "applySigmoidDerivativeFloat" to listOf(A, A, I, F),
            "reduceBiasGradient" to listOf(A, A, I, I), "reduceBiasGradientFloat" to listOf(A, A, I, I),
            "momentumUpdate" to listOf(A, A, A, I, D, D), "momentumUpdateFloat" to listOf(A, A, A, I, F, F)
        )
    }
}
