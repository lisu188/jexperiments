package com.lis.neuro

import java.lang.foreign.Arena
import java.lang.foreign.FunctionDescriptor
import java.lang.foreign.MemorySegment
import java.lang.foreign.SymbolLookup
import java.lang.foreign.ValueLayout.ADDRESS
import java.lang.foreign.ValueLayout.JAVA_DOUBLE
import java.lang.foreign.ValueLayout.JAVA_INT
import java.lang.invoke.MethodHandle
import java.nio.file.Files
import java.nio.file.Path
import java.security.MessageDigest

/** Experimental, explicitly loaded fixed-shape FP64 kernel. It is not an application backend. */
internal class NativeSmallTraining(
    state: NeuroTrainingState,
    parameters: Neuro.HyperParameters,
    precision: Neuro.TrainingPrecision = Neuro.TrainingPrecision.FP64,
    loader: (Arena) -> NeuroNativeLibrary.Loaded = ::loadSmallNativeLibrary
) : SmallTrainingKernel {
    private val cohort = NativeSmallCohort(arrayOf(state), parameters, precision, loader = loader)
    override val info get() = cohort.info
    override fun train(orders: Array<IntArray>, batchSize: Int, online: Boolean): NeuroTrainingState =
        cohort.train(arrayOf(orders), batchSize, online).single()
    override fun close() = cohort.close()
}

/** Four independent models per AVX2 vector; remaining models use the native scalar implementation. */
internal class NativeSmallCohort(
    states: Array<NeuroTrainingState>,
    private val parameters: Neuro.HyperParameters,
    precision: Neuro.TrainingPrecision = Neuro.TrainingPrecision.FP64,
    allowSimd: Boolean = true,
    loader: (Arena) -> NeuroNativeLibrary.Loaded = ::loadSmallNativeLibrary
) : AutoCloseable {
    private val arena = Arena.ofShared()
    private val models = states.size
    private val inputs: DoubleArray
    private val targets: DoubleArray
    private val samples: Int
    private val stateBuffer: MemorySegment
    private val inputBuffer: MemorySegment
    private val targetBuffer: MemorySegment
    private val orderBuffer: MemorySegment
    private val packed: DoubleArray
    private val train: MethodHandle
    private val useSimd: Boolean
    private var failed = false
    private var closed = false
    val info: TrainingDeviceInfo

    init {
        try {
            require(states.isNotEmpty()) { "Native SMALL requires at least one model" }
            require(precision == Neuro.TrainingPrecision.FP64) { "Native SMALL experiment supports FP64 only" }
            require(parameters.learningRate.isFinite() && parameters.momentum.isFinite() && parameters.beta.isFinite())
            states.forEach {
                validateSmallState(it)
                require(it.topology.contentEquals(SHAPE)) { "Native SMALL experiment requires 2/8/8/8/1" }
                require(it.inputs.contentEquals(states[0].inputs) && it.targets.contentEquals(states[0].targets)) {
                    "Native cohort models must share the same immutable dataset"
                }
            }
            inputs = states[0].inputs.copyOf()
            targets = states[0].targets.copyOf()
            samples = targets.size
            // A bounded experiment: reserve the largest permitted chunk once, without per-call native allocations.
            val orderBytes = Math.multiplyExact(Math.multiplyExact(models.toLong(), samples.toLong()), 64L * 4)
            require(orderBytes <= 64L * 1024 * 1024) { "Native SMALL order buffer exceeds the experiment's 64 MiB limit" }
            packed = DoubleArray(Math.multiplyExact(models, PARAMETERS * 2))
            for (model in states.indices) {
                var offset = 0
                for (layer in states[model].weights.indices) {
                    for (index in states[model].weights[layer].indices) {
                        packed[offset * models + model] = states[model].weights[layer][index]
                        packed[(offset + PARAMETERS) * models + model] = states[model].weightVelocity[layer][index]
                        offset++
                    }
                    for (index in states[model].biases[layer].indices) {
                        packed[offset * models + model] = states[model].biases[layer][index]
                        packed[(offset + PARAMETERS) * models + model] = states[model].biasVelocity[layer][index]
                        offset++
                    }
                }
            }
            stateBuffer = arena.allocateFrom(JAVA_DOUBLE, *packed)
            inputBuffer = if (inputs.isEmpty()) arena.allocate(8) else arena.allocateFrom(JAVA_DOUBLE, *inputs)
            targetBuffer = if (targets.isEmpty()) arena.allocate(8) else arena.allocateFrom(JAVA_DOUBLE, *targets)
            orderBuffer = arena.allocate(maxOf(4L, orderBytes), 4)
            val library = loader(arena)
            val abi = NeuroNativeLibrary.invokeInt(NeuroNativeLibrary.downcall(library, "jsmall_abi", FunctionDescriptor.of(JAVA_INT)))
            require(abi == 1) { "Unsupported native SMALL ABI $abi" }
            val available = NeuroNativeLibrary.invokeInt(NeuroNativeLibrary.downcall(library, "jsmall_avx2", FunctionDescriptor.of(JAVA_INT))) != 0
            useSimd = allowSimd && parameters.kernel != Neuro.Kernel.SCALAR && available && models >= 4
            train = NeuroNativeLibrary.downcall(library, "jsmall_train", TRAIN_DESCRIPTOR)
            info = TrainingDeviceInfo(TrainingBackend.CPU, if (useSimd) "Native AVX2/FMA experiment" else "Native scalar experiment",
                "native-cpu", precision.name, "native-small-abi1/${library.name}", TrainingEngine.SMALL,
                if (useSimd) 256 else 0, parameters.sigmoidMode.name)
        } catch (failure: Throwable) {
            arena.close()
            throw failure
        }
    }

    @Synchronized
    fun train(orders: Array<Array<IntArray>>, batchSize: Int, online: Boolean): Array<NeuroTrainingState> {
        check(!closed && !failed) { "Native SMALL kernel is closed or failed" }
        require(batchSize > 0)
        require(orders.size == models && orders[0].size in 1..64) { "Each native chunk must contain 1..64 epochs for every model" }
        val epochs = orders[0].size
        orders.forEach {
            require(it.size == epochs) { "Native cohort models must train the same number of epochs" }
            validateSmallOrders(it, samples)
        }
        if (samples == 0) return snapshots()
        try {
            var offset = 0L
            for (model in orders) for (order in model) {
                MemorySegment.copy(order, 0, orderBuffer, JAVA_INT, offset, order.size)
                offset += order.size * 4L
            }
            val status = NeuroNativeLibrary.invokeInt(train, stateBuffer, inputBuffer, targetBuffer, orderBuffer,
                models, samples, epochs, if (online) 1 else batchSize, if (online) 1 else 0,
                parameters.learningRate, parameters.momentum, parameters.beta,
                if (parameters.sigmoidMode == Neuro.SigmoidMode.FAST) 1 else 0, if (useSimd) 1 else 0)
            check(status == 0) { "Native SMALL training failed with status $status (1: arguments, 2: native exception, 3: non-finite state)" }
            MemorySegment.copy(stateBuffer, JAVA_DOUBLE, 0, packed, 0, packed.size)
            check(packed.all { it.isFinite() }) { "Native SMALL returned non-finite state" }
            return snapshots()
        } catch (failure: Throwable) {
            failed = true
            NeuroLog.error("native", "small.training.failed", failure, "models" to models, "epochs" to epochs)
            throw failure
        }
    }

    private fun snapshots(): Array<NeuroTrainingState> = Array(models) { model ->
        var offset = 0
        val weights = Array(4) { DoubleArray(SHAPE[it] * SHAPE[it + 1]) }
        val biases = Array(4) { DoubleArray(SHAPE[it + 1]) }
        val velocity = Array(4) { DoubleArray(weights[it].size) }
        val biasVelocity = Array(4) { DoubleArray(biases[it].size) }
        for (layer in weights.indices) {
            for (index in weights[layer].indices) {
                weights[layer][index] = packed[offset * models + model]
                velocity[layer][index] = packed[(offset + PARAMETERS) * models + model]
                offset++
            }
            for (index in biases[layer].indices) {
                biases[layer][index] = packed[offset * models + model]
                biasVelocity[layer][index] = packed[(offset + PARAMETERS) * models + model]
                offset++
            }
        }
        NeuroTrainingState(SHAPE.copyOf(), weights, biases, velocity, biasVelocity, inputs.copyOf(), targets.copyOf())
    }

    @Synchronized
    override fun close() {
        if (closed) return
        closed = true
        arena.close()
    }

    companion object {
        private const val PARAMETERS = 177
        private val SHAPE = intArrayOf(2, 8, 8, 8, 1)
        internal val TRAIN_DESCRIPTOR = FunctionDescriptor.of(JAVA_INT,
            ADDRESS, ADDRESS, ADDRESS, ADDRESS, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT,
            JAVA_DOUBLE, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_INT, JAVA_INT)
    }
}

internal fun loadSmallNativeLibrary(arena: Arena): NeuroNativeLibrary.Loaded {
    val requested = System.getProperty("jneuro.small.native").orEmpty()
    require(requested.isNotBlank()) { "Set -Djneuro.small.native to the experimental native library's absolute path" }
    val path = Path.of(requested).toAbsolutePath().normalize()
    val hash = MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(path)).joinToString("") { "%02x".format(it) }
    return NeuroNativeLibrary.Loaded(SymbolLookup.libraryLookup(path, arena), hash)
}
