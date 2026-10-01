package com.lis.neuro

import org.tensorflow.Graph
import org.tensorflow.Session
import org.tensorflow.Tensor
import org.tensorflow.TensorFlow
import org.tensorflow.ndarray.Shape
import org.tensorflow.ndarray.buffer.DataBuffers
import org.tensorflow.proto.AttrValue
import org.tensorflow.proto.ConfigProto
import org.tensorflow.proto.DataType
import org.tensorflow.proto.GPUOptions
import org.tensorflow.proto.GraphDef
import org.tensorflow.proto.NodeDef
import org.tensorflow.proto.TensorProto
import org.tensorflow.proto.TensorShapeProto
import org.tensorflow.types.TFloat32
import org.tensorflow.types.TFloat64
import org.tensorflow.types.TInt32

/** The only numerical implementation. Kotlin owns data, ordering and publication, TensorFlow owns math. */
internal object TensorFlowMath {
    private data class Key(val topology: List<Int>, val beta: Double, val sigmoid: Neuro.SigmoidMode,
                           val precision: Neuro.TrainingPrecision, val device: String)
    private class Cached(val graph: NetworkGraph) { var users = 0 }
    private val cache = LinkedHashMap<Key, Cached>(8, 0.75f, true)
    private val primitives by lazy { PrimitiveGraph() }
    private val gpuProbe: Result<Unit> by lazy {
        runCatching {
            val graph = Definition(Neuro.TrainingPrecision.FP32, GPU)
            val input = graph.placeholder("input", longArrayOf(1, 1))
            val output = graph.matmul(input, input)
            graph.open().use { runtime ->
                TFloat32.tensorOf(Shape.of(1, 1), DataBuffers.of(2.0f)).use { value ->
                    runtime.session.runner().feed(input, value).fetch(output).run().use { result ->
                        check((result[0] as TFloat32).getFloat(0, 0) == 4.0f)
                    }
                }
            }
        }
    }
    private const val CPU = "/device:CPU:0"
    private const val GPU = "/device:GPU:0"

    fun isGpuAvailable(): Boolean = gpuProbe.isSuccess
    fun gpuFailure(): String = gpuProbe.exceptionOrNull()?.message ?: ""

    fun info(backend: TrainingBackend, precision: Neuro.TrainingPrecision,
             engine: TrainingEngine = TrainingEngine.REFERENCE,
             sigmoid: Neuro.SigmoidMode = Neuro.SigmoidMode.EXACT): TrainingDeviceInfo {
        val device = device(backend)
        return TrainingDeviceInfo(if (device == CPU) TrainingBackend.CPU else TrainingBackend.CUDA,
            "TensorFlow ${TensorFlow.version()} ${if (device == CPU) "CPU" else "GPU 0"}",
            "tensorflow:$device", precision.name, "tensorflow-${TensorFlow.version()}-dense-v1",
            engine, 0, sigmoid.name)
    }

    private fun device(backend: TrainingBackend): String {
        if (backend == TrainingBackend.CPU || backend == TrainingBackend.AUTO) return CPU
        check(isGpuAvailable()) { "TensorFlow GPU unavailable: ${gpuFailure()}. Use a GPU-enabled Linux runtime or select CPU." }
        return GPU
    }

    fun predict(topology: IntArray, weights: Array<DoubleArray>, biases: Array<DoubleArray>,
                hp: Neuro.HyperParameters, inputs: DoubleArray, batchSize: Int,
                precision: Neuro.TrainingPrecision = Neuro.TrainingPrecision.FP64,
                backend: TrainingBackend = TrainingBackend.CPU): DoubleArray {
        if (batchSize == 0) return DoubleArray(0)
        return withInference(topology, hp, precision, backend) {
            it.infer(weights, biases, inputs, batchSize, false).last()
        }
    }

    fun evaluate(topology: IntArray, weights: Array<DoubleArray>, biases: Array<DoubleArray>,
                 hp: Neuro.HyperParameters, inputs: DoubleArray, batchSize: Int,
                 precision: Neuro.TrainingPrecision = Neuro.TrainingPrecision.FP64,
                 backend: TrainingBackend = TrainingBackend.CPU): Array<DoubleArray> {
        if (batchSize == 0) return Array(topology.size) { DoubleArray(0) }
        return withInference(topology, hp, precision, backend) {
            it.infer(weights, biases, inputs, batchSize, true)
        }
    }

    fun error(topology: IntArray, weights: Array<DoubleArray>, biases: Array<DoubleArray>,
              hp: Neuro.HyperParameters, inputs: DoubleArray, targets: DoubleArray,
              precision: Neuro.TrainingPrecision = Neuro.TrainingPrecision.FP64,
              backend: TrainingBackend = TrainingBackend.CPU): Double {
        if (inputs.isEmpty()) return Double.NaN
        return withInference(topology, hp, precision, backend) { it.error(weights, biases, inputs, targets) }
    }

    fun sigmoid(value: Double, mode: Neuro.SigmoidMode): Double =
        primitives.run(if (mode == Neuro.SigmoidMode.EXACT) "exact" else "fast", doubleArrayOf(value))[0]
    fun contributions(inputs: DoubleArray, weights: DoubleArray): DoubleArray {
        require(inputs.size == weights.size)
        return primitives.run("product", inputs, weights)
    }
    fun sum(values: DoubleArray): Double = primitives.run("sum", values)[0]
    fun squaredNorm(values: DoubleArray): Double = primitives.run("squareSum", values)[0]
    fun norm(values: DoubleArray): Double = primitives.run("norm", values)[0]
    fun xavierLimit(inputs: Int, outputs: Int): Double = primitives.run("xavier", doubleArrayOf((inputs + outputs).toDouble()))[0]

    fun trainingKernel(state: NeuroTrainingState, hp: Neuro.HyperParameters,
                       precision: Neuro.TrainingPrecision, backend: TrainingBackend,
                       engine: TrainingEngine = TrainingEngine.REFERENCE): SmallTrainingKernel =
        TrainingGraph(state, hp, precision, backend, engine)

    private fun <T> withInference(topology: IntArray, hp: Neuro.HyperParameters,
                                  precision: Neuro.TrainingPrecision, backend: TrainingBackend,
                                  operation: (NetworkGraph) -> T): T {
        val key = Key(topology.toList(), hp.beta, hp.sigmoidMode, precision, device(backend))
        val entry = synchronized(cache) {
            val existing = cache[key]
            if (existing != null) existing.also { it.users++ }
            else {
                if (cache.size >= 8) {
                    val idle = cache.entries.firstOrNull { it.value.users == 0 }
                    if (idle != null) { cache.remove(idle.key); idle.value.graph.close() }
                }
                Cached(NetworkGraph(topology.copyOf(), hp, precision, key.device)).also {
                    it.users = 1
                    if (cache.size < 8) cache[key] = it
                }
            }
        }
        try { return operation(entry.graph) }
        finally {
            synchronized(cache) {
                entry.users--
                if (cache[key] !== entry) entry.graph.close()
            }
        }
    }

    /** Used at application shutdown and by lifecycle tests; active calls cannot be invalidated. */
    fun clearInferenceCache() = synchronized(cache) {
        val idle = cache.entries.filter { it.value.users == 0 }
        idle.forEach { (key, value) -> cache.remove(key); value.graph.close() }
    }

    private class NetworkGraph(val topology: IntArray, hp: Neuro.HyperParameters,
                               val precision: Neuro.TrainingPrecision, device: String) : AutoCloseable {
        private val definition = Definition(precision, device)
        private val input = definition.placeholder("input", longArrayOf(-1, topology.first().toLong()))
        private val target = definition.placeholder("target", longArrayOf(-1, topology.last().toLong()))
        private val weights = topology.drop(1).indices.map { layer ->
            definition.placeholder("weight$layer", longArrayOf(topology[layer + 1].toLong(), topology[layer].toLong()))
        }
        private val biases = topology.drop(1).indices.map { layer ->
            definition.placeholder("bias$layer", longArrayOf(topology[layer + 1].toLong()))
        }
        private val activations = definition.forward(input, weights, biases, hp)
        private val rmse = definition.rmse(activations.last(), target)
        private val runtime = definition.open()

        fun infer(w: Array<DoubleArray>, b: Array<DoubleArray>, inputs: DoubleArray,
                  count: Int, intermediate: Boolean): Array<DoubleArray> = feed(w, b, inputs, count) { runner ->
            val fetches = if (intermediate) activations else listOf(activations.last())
            fetches.forEach { runner.fetch(it) }
            runner.run().use { result -> Array(fetches.size) { doubles(result[it]) } }
        }

        fun error(w: Array<DoubleArray>, b: Array<DoubleArray>, inputs: DoubleArray, targets: DoubleArray): Double {
            val count = inputs.size / topology.first()
            return feed(w, b, inputs, count) { runner ->
                tensor(targets, precision, count.toLong(), topology.last().toLong()).use { values ->
                    runner.feed(target, values).fetch(rmse).run().use { doubles(it[0])[0] }
                }
            }
        }

        private fun <T> feed(w: Array<DoubleArray>, b: Array<DoubleArray>, inputs: DoubleArray, count: Int,
                             operation: (Session.Runner) -> T): T {
            val tensors = ArrayList<Tensor>()
            try {
                val runner = runtime.session.runner()
                fun feed(name: String, values: DoubleArray, vararg shape: Long) {
                    val value = tensor(values, precision, *shape); tensors += value; runner.feed(name, value)
                }
                feed(input, inputs.copyOf(count * topology.first()), count.toLong(), topology.first().toLong())
                for (layer in weights.indices) {
                    feed(weights[layer], w[layer], topology[layer + 1].toLong(), topology[layer].toLong())
                    feed(biases[layer], b[layer], topology[layer + 1].toLong())
                }
                return operation(runner)
            } finally { tensors.asReversed().forEach { it.close() } }
        }
        override fun close() = runtime.close()
    }

    private class TrainingGraph(initial: NeuroTrainingState, hp: Neuro.HyperParameters,
                                private val precision: Neuro.TrainingPrecision, backend: TrainingBackend,
                                engine: TrainingEngine) : SmallTrainingKernel {
        override val info = info(backend, precision, engine, hp.sigmoidMode)
        private val source = validate(initial).copy(topology = initial.topology.copyOf(),
            weights = Array(initial.weights.size) { initial.weights[it].copyOf() },
            biases = Array(initial.biases.size) { initial.biases[it].copyOf() },
            weightVelocity = Array(initial.weightVelocity.size) { initial.weightVelocity[it].copyOf() },
            biasVelocity = Array(initial.biasVelocity.size) { initial.biasVelocity[it].copyOf() },
            inputs = if (initial.sharedDataset) initial.inputs else initial.inputs.copyOf(),
            targets = if (initial.sharedDataset) initial.targets else initial.targets.copyOf())
        private val definition = Definition(precision, device(backend))
        private val topology = source.topology
        private val dataset = definition.placeholder("dataset", longArrayOf(-1, topology.first().toLong()))
        private val targets = definition.placeholder("targets", longArrayOf(-1, topology.last().toLong()))
        private val order = definition.placeholder("order", longArrayOf(-1), DataType.DT_INT32)
        private val parameters = arrayOf(source.weights, source.biases, source.weightVelocity, source.biasVelocity)
        private val variables = Array(4) { kind -> Array(source.weights.size) { layer ->
            val shape = if (kind % 2 == 0) longArrayOf(topology[layer + 1].toLong(), topology[layer].toLong())
                        else longArrayOf(topology[layer + 1].toLong())
            definition.variable("p${kind}_$layer", shape)
        } }
        private val runtime: RuntimeGraph
        private val inputTensor: Tensor
        private val targetTensor: Tensor
        private var closed = false
        private var failed = false

        init {
            if (precision == Neuro.TrainingPrecision.FP32) {
                require((parameters.flatMap { it.toList() } + listOf(source.inputs, source.targets,
                    doubleArrayOf(hp.learningRate, hp.momentum, hp.beta))).all { values -> values.all { it.toFloat().isFinite() } }) {
                    "Training state and hyperparameters must be representable in FP32."
                }
            }
            val input = definition.gather(dataset, order)
            val target = definition.gather(targets, order)
            val weights = variables[0].map { definition.op("Identity", it) }
            val biases = variables[1].map { definition.op("Identity", it) }
            val activation = definition.forward(input, weights, biases, hp)
            val deltas = Array(weights.size) { "" }
            for (layer in weights.indices.reversed()) {
                val output = activation[layer + 1]
                val residual = if (layer == weights.lastIndex) definition.op("Sub", target, output)
                    else definition.matmul(deltas[layer + 1], weights[layer + 1])
                val derivative = definition.op("Mul", definition.scalar(hp.beta),
                    definition.op("Mul", output, definition.op("Sub", definition.scalar(1.0), output)))
                deltas[layer] = definition.op("Mul", residual, derivative)
            }
            val count = definition.cast(definition.node("Size", listOf(order), mapOf(
                "T" to Definition.type(DataType.DT_INT32), "out_type" to Definition.type(DataType.DT_INT32))), definition.dtype)
            val next = Array(4) { Array(weights.size) { "" } }
            for (layer in weights.indices) {
                val gradient = definition.op("RealDiv", definition.matmul(deltas[layer], activation[layer], transposeA = true), count)
                val biasGradient = definition.reduce("Mean", deltas[layer], intArrayOf(0))
                for ((kind, change) in listOf(0 to gradient, 1 to biasGradient)) {
                    val velocity = definition.op("AddV2", definition.op("Mul", definition.scalar(hp.momentum), variables[kind + 2][layer]),
                        definition.op("Mul", definition.scalar(hp.learningRate), change))
                    next[kind + 2][layer] = velocity
                    next[kind][layer] = definition.op("AddV2", variables[kind][layer], velocity)
                }
            }
            // Compute every new value before any assign: updates must all use the same old parameters.
            for (kind in next.indices) for (layer in next[kind].indices) {
                next[kind][layer] = definition.node("CheckNumerics", listOf(next[kind][layer]), mapOf(
                    "T" to Definition.type(definition.dtype), "message" to AttrValue.newBuilder()
                        .setS(com.google.protobuf.ByteString.copyFromUtf8("Training produced non-finite parameters or momentum")).build()))
            }
            val barrier = definition.node("NoOp", next.flatMap { it.toList() }.map { "^$it" }, emptyMap())
            val assigns = variables.indices.flatMap { kind -> variables[kind].indices.map { layer ->
                definition.node("Assign", listOf(variables[kind][layer], next[kind][layer], "^$barrier"),
                    mapOf("T" to Definition.type(definition.dtype), "use_locking" to Definition.bool(true)))
            } }
            definition.node("NoOp", assigns.map { "^$it" }, emptyMap(), "train")
            runtime = definition.open()
            var data: Tensor? = null
            var expected: Tensor? = null
            try {
                data = tensor(source.inputs, precision, source.samples.toLong(), topology.first().toLong())
                expected = tensor(source.targets, precision, source.samples.toLong(), topology.last().toLong())
                initialize()
                inputTensor = data; targetTensor = expected
            } catch (failure: Throwable) {
                expected?.close(); data?.close(); runtime.close(); throw failure
            }
        }

        private fun initialize() {
            val values = ArrayList<Tensor>()
            try {
                val runner = runtime.session.runner()
                for (kind in variables.indices) for (layer in variables[kind].indices) {
                    val shape = if (kind % 2 == 0) longArrayOf(topology[layer + 1].toLong(), topology[layer].toLong())
                                else longArrayOf(topology[layer + 1].toLong())
                    val value = tensor(parameters[kind][layer], precision, *shape); values += value
                    runner.feed("initial_${variables[kind][layer]}", value).addTarget("initialize_${variables[kind][layer]}")
                }
                runner.run().close()
            } finally { values.asReversed().forEach { it.close() } }
        }

        @Synchronized override fun train(orders: Array<IntArray>, batchSize: Int, online: Boolean): NeuroTrainingState {
            check(!closed && !failed) { "TensorFlow kernel is closed or failed; reopen from the committed checkpoint." }
            require(batchSize > 0)
            require(orders.isEmpty() || source.samples > 0) { "Training requires samples." }
            require(orders.all { it.size == source.samples && it.toSet().size == source.samples &&
                it.all { sample -> sample in 0 until source.samples } })
            try {
                val batch = if (online) 1 else batchSize
                for (indices in orders) for (start in indices.indices step batch) {
                    val slice = indices.copyOfRange(start, minOf(indices.size, start + batch))
                    TInt32.tensorOf(Shape.of(slice.size.toLong()), DataBuffers.of(slice, false, false)).use { rows ->
                        runtime.session.runner().feed(dataset, inputTensor).feed(targets, targetTensor)
                            .feed(order, rows).addTarget("train").run().close()
                    }
                }
                val runner = runtime.session.runner()
                variables.forEach { layer -> layer.forEach { runner.fetch(it) } }
                return runner.run().use { result ->
                    val layers = source.weights.size
                    val output = Array(4) { kind -> Array(layers) { layer -> doubles(result[kind * layers + layer]) } }
                    check(output.all { values -> values.all { buffer -> buffer.all { it.isFinite() } } }) {
                        "Training produced non-finite parameters or momentum."
                    }
                    source.copy(topology = topology.copyOf(), weights = output[0], biases = output[1],
                        weightVelocity = output[2], biasVelocity = output[3],
                        inputs = if (source.sharedDataset) source.inputs else source.inputs.copyOf(),
                        targets = if (source.sharedDataset) source.targets else source.targets.copyOf())
                }
            } catch (failure: Throwable) {
                failed = true
                throw IllegalStateException("TensorFlow training failed; retained the last committed host checkpoint.", failure)
            }
        }

        @Synchronized override fun close() {
            if (closed) return
            closed = true
            try { runtime.close() } finally { inputTensor.close(); targetTensor.close() }
        }
    }

    private class PrimitiveGraph {
        private val definition = Definition(Neuro.TrainingPrecision.FP64, CPU)
        private val x = definition.placeholder("x", longArrayOf(-1))
        private val y = definition.placeholder("y", longArrayOf(-1))
        private val outputs = mapOf("exact" to definition.activation(x, Neuro.SigmoidMode.EXACT),
            "fast" to definition.activation(x, Neuro.SigmoidMode.FAST),
            "product" to definition.op("Mul", x, y), "sum" to definition.reduce("Sum", x, intArrayOf(0)),
            "squareSum" to definition.reduce("Sum", definition.op("Square", x), intArrayOf(0))).toMutableMap().also {
            it["norm"] = definition.op("Sqrt", checkNotNull(it["squareSum"]))
            it["xavier"] = definition.op("Sqrt", definition.op("RealDiv", definition.scalar(6.0), x))
        }
        private val runtime = definition.open()
        fun run(name: String, values: DoubleArray, other: DoubleArray? = null): DoubleArray =
            tensor(values, Neuro.TrainingPrecision.FP64, values.size.toLong()).use { input ->
                val runner = runtime.session.runner().feed(x, input).fetch(outputs.getValue(name))
                if (other == null) runner.run().use { doubles(it[0]) }
                else tensor(other, Neuro.TrainingPrecision.FP64, other.size.toLong()).use { right ->
                    runner.feed(y, right).run().use { doubles(it[0]) }
                }
            }
    }

    private class RuntimeGraph(val graph: Graph, val session: Session) : AutoCloseable {
        override fun close() { try { session.close() } finally { graph.close() } }
    }

    /** Named TensorFlow operations keep dtype and device decisions inside the numerical boundary. */
    private class Definition(val precision: Neuro.TrainingPrecision, private val device: String) {
        val dtype = if (precision == Neuro.TrainingPrecision.FP64) DataType.DT_DOUBLE else DataType.DT_FLOAT
        private val graph = GraphDef.newBuilder()
        private var sequence = 0
        fun node(op: String, inputs: List<String>, attrs: Map<String, AttrValue>, name: String = "n${sequence++}"): String {
            graph.addNode(NodeDef.newBuilder().setName(name).setOp(op).setDevice(device).addAllInput(inputs).putAllAttr(attrs))
            return name
        }
        fun op(op: String, vararg inputs: String): String = node(op, inputs.toList(), mapOf("T" to type(dtype)))
        fun placeholder(name: String, dims: LongArray, dataType: DataType = dtype) =
            node("Placeholder", emptyList(), mapOf("dtype" to type(dataType), "shape" to shape(dims)), name)
        fun scalar(value: Double): String {
            val tensor = TensorProto.newBuilder().setDtype(dtype).setTensorShape(TensorShapeProto.getDefaultInstance())
            if (dtype == DataType.DT_DOUBLE) tensor.addDoubleVal(value) else tensor.addFloatVal(value.toFloat())
            return node("Const", emptyList(), mapOf("dtype" to type(dtype), "value" to AttrValue.newBuilder().setTensor(tensor).build()))
        }
        fun ints(values: IntArray): String {
            val tensor = TensorProto.newBuilder().setDtype(DataType.DT_INT32)
                .setTensorShape(TensorShapeProto.newBuilder().addDim(TensorShapeProto.Dim.newBuilder().setSize(values.size.toLong())))
                .addAllIntVal(values.toList())
            return node("Const", emptyList(), mapOf("dtype" to type(DataType.DT_INT32), "value" to AttrValue.newBuilder().setTensor(tensor).build()))
        }
        fun matmul(a: String, b: String, transposeA: Boolean = false, transposeB: Boolean = false) =
            node("MatMul", listOf(a, b), mapOf("T" to type(dtype), "transpose_a" to bool(transposeA), "transpose_b" to bool(transposeB)))
        fun reduce(op: String, input: String, axes: IntArray) = node(op, listOf(input, ints(axes)),
            mapOf("T" to type(dtype), "Tidx" to type(DataType.DT_INT32), "keep_dims" to bool(false)))
        fun cast(input: String, destination: DataType, source: DataType = DataType.DT_INT32) = node("Cast", listOf(input),
            mapOf("SrcT" to type(source), "DstT" to type(destination), "Truncate" to bool(false)))
        fun gather(input: String, indices: String): String {
            val axis = node("Const", emptyList(), mapOf("dtype" to type(DataType.DT_INT32),
                "value" to AttrValue.newBuilder().setTensor(TensorProto.newBuilder().setDtype(DataType.DT_INT32)
                    .setTensorShape(TensorShapeProto.getDefaultInstance()).addIntVal(0)).build()))
            return node("GatherV2", listOf(input, indices, axis), mapOf("Tparams" to type(dtype),
                "Tindices" to type(DataType.DT_INT32), "Taxis" to type(DataType.DT_INT32), "batch_dims" to AttrValue.newBuilder().setI(0).build()))
        }
        fun variable(name: String, dims: LongArray): String {
            node("VariableV2", emptyList(), mapOf("dtype" to type(dtype), "shape" to shape(dims),
                "container" to AttrValue.newBuilder().setS(com.google.protobuf.ByteString.EMPTY).build(),
                "shared_name" to AttrValue.newBuilder().setS(com.google.protobuf.ByteString.EMPTY).build()), name)
            val initial = placeholder("initial_$name", dims)
            node("Assign", listOf(name, initial), mapOf("T" to type(dtype), "use_locking" to bool(true)), "initialize_$name")
            return name
        }
        fun forward(input: String, weights: List<String>, biases: List<String>, hp: Neuro.HyperParameters): List<String> {
            val values = arrayListOf(input)
            for (layer in weights.indices) values += activation(op("Mul", scalar(hp.beta),
                op("AddV2", matmul(values.last(), weights[layer], transposeB = true), biases[layer])), hp.sigmoidMode)
            return values
        }
        fun rmse(prediction: String, target: String) = op("Sqrt", reduce("Mean", op("Square", op("Sub", target, prediction)), intArrayOf(0, 1)))
        fun activation(input: String, mode: Neuro.SigmoidMode): String {
            if (mode == Neuro.SigmoidMode.EXACT) return op("Sigmoid", input)
            val negative = op("Neg", op("Abs", input))
            val bounded = op("Maximum", negative, scalar(-745.0))
            val exponent = cast(cast(op("Mul", bounded, scalar(1.4426950408889634)), DataType.DT_INT32, dtype), dtype)
            val remainder = op("Sub", bounded, op("Mul", exponent, scalar(0.6931471805599453)))
            var polynomial = scalar(0.008333333333333333)
            for (coefficient in doubleArrayOf(0.041666666666666664, 0.16666666666666666, 0.5))
                polynomial = op("AddV2", scalar(coefficient), op("Mul", remainder, polynomial))
            polynomial = op("AddV2", op("AddV2", scalar(1.0), remainder), op("Mul", op("Square", remainder), polynomial))
            val exponential = op("Mul", polynomial, op("Pow", scalar(2.0), exponent))
            val underflow = op("LessEqual", negative, scalar(-745.0))
            val exp = node("SelectV2", listOf(underflow, scalar(0.0), exponential), mapOf("T" to type(dtype)))
            val denominator = op("AddV2", scalar(1.0), exp)
            val positive = op("GreaterEqual", input, scalar(0.0))
            return node("SelectV2", listOf(positive, op("RealDiv", scalar(1.0), denominator), op("RealDiv", exp, denominator)), mapOf("T" to type(dtype)))
        }
        fun open(): RuntimeGraph {
            val native = Graph()
            try {
                native.importGraphDef(graph.build())
                val config = ConfigProto.newBuilder().setAllowSoftPlacement(false).setInterOpParallelismThreads(1)
                    .setIntraOpParallelismThreads(1).setIsolateSessionState(true)
                    .setGpuOptions(GPUOptions.newBuilder().setAllowGrowth(true)).build()
                return RuntimeGraph(native, Session(native, false, config))
            } catch (failure: Throwable) { native.close(); throw failure }
        }
        companion object {
            fun type(value: DataType): AttrValue = AttrValue.newBuilder().setType(value).build()
            fun bool(value: Boolean): AttrValue = AttrValue.newBuilder().setB(value).build()
            fun shape(dims: LongArray): AttrValue = AttrValue.newBuilder().setShape(TensorShapeProto.newBuilder()
                .addAllDim(dims.map { TensorShapeProto.Dim.newBuilder().setSize(it).build() })).build()
        }
    }

    private fun tensor(values: DoubleArray, precision: Neuro.TrainingPrecision, vararg dims: Long): Tensor =
        if (precision == Neuro.TrainingPrecision.FP64) TFloat64.tensorOf(Shape.of(*dims), DataBuffers.of(values, false, false))
        else TFloat32.tensorOf(Shape.of(*dims), DataBuffers.of(FloatArray(values.size) { values[it].toFloat() }, false, false))

    private fun validate(state: NeuroTrainingState): NeuroTrainingState {
        require(state.topology.size >= 2 && state.topology.all { it > 0 }) { "Invalid training topology." }
        val layers = state.topology.size - 1
        val buffers = arrayOf(state.weights, state.biases, state.weightVelocity, state.biasVelocity)
        require(buffers.all { it.size == layers }) { "Invalid training layer count." }
        for (layer in 0 until layers) {
            require(state.weights[layer].size.toLong() == state.topology[layer].toLong() * state.topology[layer + 1] &&
                state.weightVelocity[layer].size == state.weights[layer].size &&
                state.biases[layer].size == state.topology[layer + 1] &&
                state.biasVelocity[layer].size == state.biases[layer].size) { "Invalid parameter shape." }
        }
        require(state.inputs.size % state.topology.first() == 0 &&
            state.targets.size.toLong() == state.samples.toLong() * state.topology.last()) { "Invalid dataset shape." }
        require((buffers.flatMap { it.toList() } + listOf(state.inputs, state.targets)).all { values -> values.all { it.isFinite() } }) {
            "Training state and dataset must be finite."
        }
        return state
    }

    private fun doubles(value: Tensor): DoubleArray {
        val size = Math.toIntExact(value.shape().size())
        return when (value) {
            is TFloat64 -> DoubleArray(size).also { value.copyTo(DataBuffers.of(it, false, false)) }
            is TFloat32 -> FloatArray(size).also { value.copyTo(DataBuffers.of(it, false, false)) }.let { data -> DoubleArray(size) { data[it].toDouble() } }
            else -> error("Unexpected TensorFlow result type: ${value.javaClass.name}")
        }
    }
}
