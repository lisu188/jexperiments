package com.lis.neuro

import com.lis.neuro.TensorFlowMath.Definition
import com.lis.neuro.TensorFlowMath.Definition.Companion.bool
import com.lis.neuro.TensorFlowMath.Definition.Companion.type
import org.tensorflow.Tensor
import org.tensorflow.TensorFlow
import org.tensorflow.ndarray.Shape
import org.tensorflow.ndarray.buffer.DataBuffers
import org.tensorflow.proto.AttrValue
import org.tensorflow.proto.DataType
import org.tensorflow.proto.NameAttrList
import org.tensorflow.proto.TensorProto
import org.tensorflow.proto.TensorShapeProto
import org.tensorflow.types.TBool
import org.tensorflow.types.TInt32

/** A model is a leading tensor dimension, never an independent Session or a gradient reduction axis. */
internal class TensorFlowCohortGraph(
    states: Array<NeuroTrainingState>, private val hp: Neuro.HyperParameters,
    private val precision: Neuro.TrainingPrecision, backend: TrainingBackend,
    validationInputs: DoubleArray, validationTargets: DoubleArray, private val batchSize: Int, paddedTopology: IntArray?, intraOpThreads: Int
) : TensorFlowSearchCohort {
    private val source = prepare(states, hp, precision, validationInputs, validationTargets, batchSize, paddedTopology)
    override val size = source.size
    private val caps = (paddedTopology?.copyOf() ?: IntArray(source[0].topology.size) { layer -> source.maxOf { it.topology[layer] } }).also { shape ->
        require(shape.size == source[0].topology.size && shape.first() == source[0].topology.first() && shape.last() == source[0].topology.last() &&
            source.all { state -> shape.indices.all { shape[it] >= state.topology[it] } }) { "Invalid padded cohort topology." }
        require((1 until shape.size).all { size.toLong() * shape[it] * shape[it - 1] <= Int.MAX_VALUE }) { "Padded cohort is too large." }
    }
    override val paddedTopology: IntArray get() = caps.copyOf()
    override val info = TensorFlowMath.info(backend, precision, sigmoid = hp.sigmoidMode)
        .copy(kernelVersion = "tensorflow-${TensorFlow.version()}-batched-v1-t$intraOpThreads")
    private val device = TensorFlowMath.device(backend)
    private val definition = Definition(precision, device)
    private val layers = caps.size - 1
    private val samples = source[0].samples
    private val validationCount = validationInputs.size / caps.first()
    private val parameterShapes = List(4 * layers) { index ->
        val layer = index % layers
        if (index / layers % 2 == 0) longArrayOf(size.toLong(), caps[layer + 1].toLong(), caps[layer].toLong())
        else longArrayOf(size.toLong(), 1, caps[layer + 1].toLong())
    }
    private val variables = List(8 * layers) { index -> definition.variable("p$index", parameterShapes[index % (4 * layers)]) }
    private val bestScore = definition.variable("best_score", longArrayOf(size.toLong()))
    private val dataset = definition.variable("dataset", longArrayOf(samples.toLong(), caps.first().toLong()))
    private val targets = definition.variable("targets", longArrayOf(samples.toLong(), caps.last().toLong()))
    private val validationData = definition.variable("validation", longArrayOf(validationCount.toLong(), caps.first().toLong()))
    private val validationExpected = definition.variable("validation_targets", longArrayOf(validationCount.toLong(), caps.last().toLong()))
    private val orders = definition.placeholder("orders", longArrayOf(size.toLong(), -1), DataType.DT_INT32)
    private val active = definition.placeholder("active", longArrayOf(size.toLong()), DataType.DT_BOOL)
    private val end = definition.placeholder("end", longArrayOf(), DataType.DT_INT32)
    private val exportLanes = definition.placeholder("export_lanes", longArrayOf(-1), DataType.DT_INT32)
    private val exported = variables.map { definition.gather(it, exportLanes) }
    private val failedOutput: String
    private val scoreOutputs: List<String>
    private val runtime: TensorFlowMath.RuntimeGraph
    private val epochs = IntArray(size)
    private val bestEpoch = IntArray(size)
    private val failed = BooleanArray(size)
    private var closed = false
    private var poisoned = false

    init {
        failedOutput = buildTrainingLoop()
        scoreOutputs = buildScoring()
        runtime = definition.open(intraOpThreads, privateThreads = true)
        try { initialize(validationInputs, validationTargets) }
        catch (failure: Throwable) { runtime.close(); throw failure }
    }

    private fun buildTrainingLoop(): String {
        // Inputs 0..6: position, batch, end, X, Y, orders, live lanes. Remaining values are complete optimizer state.
        val shapes = listOf(longArrayOf(), longArrayOf(), longArrayOf(), longArrayOf(samples.toLong(), caps.first().toLong()),
            longArrayOf(samples.toLong(), caps.last().toLong()), longArrayOf(size.toLong(), -1), longArrayOf(size.toLong())) + parameterShapes
        val types = listOf(DataType.DT_INT32, DataType.DT_INT32, DataType.DT_INT32, definition.dtype, definition.dtype,
            DataType.DT_INT32, DataType.DT_BOOL) + List(parameterShapes.size) { definition.dtype }
        fun arguments(graph: Definition) = shapes.indices.map { graph.placeholder("arg$it", shapes[it], types[it]) }
        val condition = Definition(precision, device)
        val conditionArgs = arguments(condition)
        val cond = condition.function("cohort_condition", conditionArgs,
            listOf(condition.intOp("Less", conditionArgs[0], conditionArgs[2])))
        val body = Definition(precision, device)
        val args = arguments(body)
        val count = body.intOp("Minimum", args[1], body.intOp("Sub", body.intScalar(samples),
            body.intOp("FloorMod", args[0], body.intScalar(samples))))
        val rows = body.node("Slice", listOf(args[5], body.pack(body.intScalar(0), args[0]), body.pack(body.intScalar(size), count)),
            mapOf("T" to type(DataType.DT_INT32), "Index" to type(DataType.DT_INT32)))
        val input = body.gather(args[3], rows)
        val target = body.gather(args[4], rows)
        val previous = args.drop(7)
        val next = update(body, previous, input, target, body.cast(count, body.dtype))
        var live = args[6]
        next.forEach { live = body.logical("LogicalAnd", live, body.finiteLanes(it)) }
        val gate = body.reshape(live, intArrayOf(size, 1, 1), DataType.DT_BOOL)
        val selected = next.indices.map { body.select(gate, next[it], previous[it]) }
        val outputs = listOf(body.intOp("AddV2", args[0], count)) + args.subList(1, 6) + live + selected
        val step = body.function("cohort_batch", args, outputs)
        definition.function(cond); definition.function(step)
        val initial = listOf(definition.intScalar(0), definition.intScalar(minOf(batchSize, samples)), end,
            definition.op("Identity", dataset), definition.op("Identity", targets), orders, active) +
            variables.take(4 * layers).map { definition.op("Identity", it) }
        val loop = definition.node("While", initial, mapOf(
            "T" to AttrValue.newBuilder().setList(AttrValue.ListValue.newBuilder().addAllType(types)).build(),
            "cond" to AttrValue.newBuilder().setFunc(NameAttrList.newBuilder().setName(cond.signature.name)).build(),
            "body" to AttrValue.newBuilder().setFunc(NameAttrList.newBuilder().setName(step.signature.name)).build(),
            "parallel_iterations" to AttrValue.newBuilder().setI(1).build(),
            "output_shapes" to AttrValue.newBuilder().setList(AttrValue.ListValue.newBuilder()
                .addAllShape(shapes.map { Definition.shape(it).shape })).build()), "cohort_loop")
        val committed = definition.reshape("$loop:6", intArrayOf(size, 1, 1), DataType.DT_BOOL)
        // A late non-finite update rolls the entire lane back to its chunk-start snapshot, including momentum.
        val assigns = variables.take(4 * layers).indices.map { index ->
            definition.assign(variables[index], definition.select(committed, "$loop:${7 + index}", initial[7 + index]))
        }
        definition.node("NoOp", assigns.map { "^$it" }, emptyMap(), "advance")
        return definition.logical("LogicalAnd", active, definition.logical("LogicalNot", "$loop:6"))
    }

    private fun update(graph: Definition, previous: List<String>, input: String, target: String, count: String): List<String> {
        val activations = forward(graph, input, previous)
        val deltas = Array(layers) { "" }
        for (layer in deltas.indices.reversed()) {
            val output = activations[layer + 1]
            val residual = if (layer == layers - 1) graph.op("Sub", target, output)
                else graph.batchMatmul(deltas[layer + 1], previous[layer + 1])
            val derivative = graph.op("Mul", graph.scalar(hp.beta), graph.op("Mul", output, graph.op("Sub", graph.scalar(1.0), output)))
            deltas[layer] = graph.op("Mul", residual, derivative)
        }
        val result = Array(4 * layers) { "" }
        for (layer in 0 until layers) {
            val gradient = graph.op("RealDiv", graph.batchMatmul(deltas[layer], activations[layer], transposeA = true), count)
            val biasGradient = graph.reduce("Mean", deltas[layer], intArrayOf(1), keepDims = true)
            for ((kind, change) in listOf(0 to gradient, 1 to biasGradient)) {
                val index = kind * layers + layer
                val velocity = graph.op("AddV2", graph.op("Mul", graph.scalar(hp.momentum), previous[index + 2 * layers]),
                    graph.op("Mul", graph.scalar(hp.learningRate), change))
                result[index + 2 * layers] = velocity
                result[index] = graph.op("AddV2", previous[index], velocity)
            }
        }
        return result.toList()
    }

    private fun forward(graph: Definition, input: String, parameters: List<String>): List<String> {
        val values = arrayListOf(input)
        for (layer in 0 until layers) {
            val output = graph.activation(graph.op("Mul", graph.scalar(hp.beta), graph.op("AddV2",
                graph.batchMatmul(values.last(), parameters[layer], transposeB = true), parameters[layers + layer])), hp.sigmoidMode)
            // Sigmoid(0) is 0.5: masking parameters alone would create active padded neurons.
            val mask = DoubleArray(size * caps[layer + 1]) { index ->
                if (index % caps[layer + 1] < source[index / caps[layer + 1]].topology[layer + 1]) 1.0 else 0.0
            }
            values += graph.op("Mul", output, graph.constant(mask, longArrayOf(size.toLong(), 1, caps[layer + 1].toLong())))
        }
        return values
    }

    private fun buildScoring(): List<String> {
        val current = variables.take(4 * layers).map { definition.op("Identity", it) }
        fun error(input: String, target: String): String {
            val prediction = forward(definition, input, current).last()
            return definition.op("Sqrt", definition.reduce("Mean", definition.op("Square", definition.op("Sub", target, prediction)), intArrayOf(1, 2)))
        }
        val training = error(dataset, targets)
        val validation = if (validationCount == 0) definition.op("Identity", training) else error(validationData, validationExpected)
        val finite = definition.logical("LogicalAnd", definition.op("IsFinite", training), definition.op("IsFinite", validation))
        val improved = definition.logical("LogicalAnd", finite, definition.logical("LogicalAnd", active, definition.op("Less", validation, bestScore)))
        val gate = definition.reshape(improved, intArrayOf(size, 1, 1), DataType.DT_BOOL)
        val nextBest = definition.select(improved, validation, definition.op("Identity", bestScore))
        val assignments = current.indices.map { index -> definition.assign(variables[index + 4 * layers],
            definition.select(gate, current[index], definition.op("Identity", variables[index + 4 * layers]))) } + definition.assign(bestScore, nextBest)
        definition.node("NoOp", assignments.map { "^$it" }, emptyMap(), "score")
        return listOf(training, validation, nextBest, improved, finite)
    }

    private fun initialize(validationInputs: DoubleArray, validationTargets: DoubleArray) {
        val tensors = ArrayList<Tensor>()
        try {
            val runner = runtime.session.runner()
            fun feed(name: String, values: DoubleArray, shape: LongArray) {
                val value = TensorFlowMath.tensor(values, precision, *shape); tensors += value
                runner.feed("initial_$name", value).addTarget("initialize_$name")
            }
            for (index in variables.indices) {
                val kind = index / layers % 4
                val layer = index % layers
                val shape = parameterShapes[index % (4 * layers)]
                val perLane = Math.toIntExact(shape.drop(1).fold(1L, Long::times))
                val packed = DoubleArray(Math.multiplyExact(size, perLane))
                source.forEachIndexed { lane, state ->
                    val values = buffers(state)[kind][layer]
                    if (kind % 2 == 1) values.copyInto(packed, lane * perLane)
                    else for (row in 0 until state.topology[layer + 1])
                        values.copyInto(packed, lane * perLane + row * caps[layer], row * state.topology[layer], (row + 1) * state.topology[layer])
                }
                feed(variables[index], packed, shape)
            }
            feed(bestScore, DoubleArray(size) { Double.POSITIVE_INFINITY }, longArrayOf(size.toLong()))
            feed(dataset, source[0].inputs, longArrayOf(samples.toLong(), caps.first().toLong()))
            feed(targets, source[0].targets, longArrayOf(samples.toLong(), caps.last().toLong()))
            feed(validationData, validationInputs, longArrayOf(validationCount.toLong(), caps.first().toLong()))
            feed(validationExpected, validationTargets, longArrayOf(validationCount.toLong(), caps.last().toLong()))
            runner.run().close()
        } finally { tensors.asReversed().forEach { it.close() } }
    }

    @Synchronized override fun advance(orders: Array<Array<IntArray>>, active: BooleanArray): BooleanArray {
        usable()
        require(orders.size == size && active.size == size)
        val enabled = BooleanArray(size) { active[it] && !failed[it] }
        var count = -1
        for (lane in orders.indices) if (enabled[lane]) {
            if (count == -1) count = orders[lane].size
            require(count == orders[lane].size) { "Active cohort lanes must advance equal epoch counts." }
        }
        count = maxOf(0, count)
        require(orders.all { it.size <= 64 }) { "A cohort chunk supports at most 64 epochs." }
        require(orders.indices.all { enabled[it] || orders[it].isEmpty() || orders[it].size == count }) {
            "Inactive lanes require empty orders or the active chunk's epoch count."
        }
        require(count.toLong() * samples * size <= Int.MAX_VALUE) { "Cohort shuffle chunk is too large." }
        val seen = IntArray(samples)
        var stamp = 0
        for (lane in orders) for (order in lane) {
            require(order.size == samples) { "Each epoch requires a complete valid permutation." }
            stamp++
            for (sample in order) {
                require(sample in 0 until samples && seen[sample] != stamp) { "Each epoch requires a complete valid permutation." }
                seen[sample] = stamp
            }
        }
        require(epochs.indices.all { !enabled[it] || epochs[it].toLong() + count <= Int.MAX_VALUE })
        if (count == 0) return failed.copyOf()
        val length = count * samples
        val flattened = IntArray(size * length)
        for (lane in 0 until size) if (enabled[lane]) orders[lane].forEachIndexed { epoch, order ->
            order.copyInto(flattened, lane * length + epoch * samples)
        }
        return nativeCall {
            TInt32.tensorOf(Shape.of(size.toLong(), length.toLong()), DataBuffers.of(flattened, false, false)).use { rows ->
                TInt32.scalarOf(length).use { last -> boolTensor(enabled).use { mask ->
                    runtime.session.runner().feed(this.orders, rows).feed(end, last).feed(this.active, mask)
                        .fetch(failedOutput).addTarget("advance").run().use { result ->
                            val failures = booleans(result[0])
                            for (lane in 0 until size) {
                                failed[lane] = failed[lane] || failures[lane]
                                if (enabled[lane] && !failed[lane]) epochs[lane] += count
                            }
                            failed.copyOf()
                        }
                } }
            }
        }
    }

    @Synchronized override fun score(active: BooleanArray): TensorFlowCohortMetrics {
        usable()
        require(active.size == size)
        return nativeCall { boolTensor(BooleanArray(size) { active[it] && !failed[it] }).use { mask ->
            val runner = runtime.session.runner().feed(this.active, mask).addTarget("score")
            scoreOutputs.forEach { runner.fetch(it) }
            runner.run().use { result ->
                val improved = booleans(result[3]); val finite = booleans(result[4])
                for (lane in 0 until size) {
                    if (improved[lane]) bestEpoch[lane] = epochs[lane]
                    if (active[lane] && !finite[lane]) failed[lane] = true
                }
                TensorFlowCohortMetrics(epochs.copyOf(), TensorFlowMath.doubles(result[0]), TensorFlowMath.doubles(result[1]),
                    TensorFlowMath.doubles(result[2]), bestEpoch.copyOf(), failed.copyOf())
            }
        } }
    }

    override fun exportState(lane: Int, best: Boolean): NeuroTrainingState = exportStates(intArrayOf(lane), best).single()

    @Synchronized override fun exportStates(lanes: IntArray, best: Boolean): Array<NeuroTrainingState> {
        usable()
        require(lanes.all { it in 0 until size })
        if (lanes.isEmpty()) return emptyArray()
        return nativeCall { TInt32.tensorOf(Shape.of(lanes.size.toLong()), DataBuffers.of(lanes, false, false)).use { indices ->
            val runner = runtime.session.runner().feed(exportLanes, indices)
            val offset = if (best) 4 * layers else 0
            repeat(4 * layers) { runner.fetch(exported[offset + it]) }
            runner.run().use { result ->
                val values = Array(4 * layers) { TensorFlowMath.doubles(result[it]) }
                Array(lanes.size) { selection ->
                    val original = source[lanes[selection]]
                    val state = Array(4) { kind -> Array(layers) { layer ->
                        val perLane = if (kind % 2 == 0) caps[layer] * caps[layer + 1] else caps[layer + 1]
                        val buffer = values[kind * layers + layer]
                        if (kind % 2 == 1) buffer.copyOfRange(selection * perLane, selection * perLane + original.topology[layer + 1])
                        else DoubleArray(original.topology[layer] * original.topology[layer + 1]) { index ->
                            buffer[selection * perLane + (index / original.topology[layer]) * caps[layer] + index % original.topology[layer]]
                        }
                    } }
                    original.copy(topology = original.topology.copyOf(), weights = state[0], biases = state[1],
                        weightVelocity = state[2], biasVelocity = state[3])
                }
            }
        } }
    }

    private fun usable() = check(!closed && !poisoned) { "TensorFlow cohort is closed or failed; reopen from a committed checkpoint." }
    private fun <T> nativeCall(action: () -> T): T = try { action() }
    catch (failure: Throwable) { poisoned = true; throw IllegalStateException("TensorFlow cohort execution failed.", failure) }

    @Synchronized override fun close() {
        if (closed) return
        closed = true
        runtime.close()
    }

    private fun Definition.pack(vararg values: String) = node("Pack", values.toList(), mapOf("T" to type(DataType.DT_INT32),
        "N" to AttrValue.newBuilder().setI(values.size.toLong()).build(), "axis" to AttrValue.newBuilder().setI(0).build()))
    private fun Definition.reshape(value: String, dims: IntArray, dataType: DataType = dtype) = node("Reshape", listOf(value, ints(dims)),
        mapOf("T" to type(dataType), "Tshape" to type(DataType.DT_INT32)))
    private fun Definition.batchMatmul(a: String, b: String, transposeA: Boolean = false, transposeB: Boolean = false) =
        node("BatchMatMulV2", listOf(a, b), mapOf("T" to type(dtype), "adj_x" to bool(transposeA), "adj_y" to bool(transposeB)))
    private fun Definition.select(mask: String, yes: String, no: String) = node("SelectV2", listOf(mask, yes, no), mapOf("T" to type(dtype)))
    private fun Definition.logical(op: String, vararg values: String) = node(op, values.toList(), emptyMap())
    private fun Definition.assign(variable: String, value: String) = node("Assign", listOf(variable, value), mapOf("T" to type(dtype), "use_locking" to bool(true)))
    private fun Definition.finiteLanes(value: String) = node("All", listOf(op("IsFinite", value), ints(intArrayOf(1, 2))),
        mapOf("Tidx" to type(DataType.DT_INT32), "keep_dims" to bool(false)))
    private fun Definition.constant(values: DoubleArray, dims: LongArray): String {
        val tensor = TensorProto.newBuilder().setDtype(dtype).setTensorShape(TensorShapeProto.newBuilder()
            .addAllDim(dims.map { TensorShapeProto.Dim.newBuilder().setSize(it).build() }))
        if (dtype == DataType.DT_DOUBLE) tensor.addAllDoubleVal(values.toList()) else tensor.addAllFloatVal(values.map { it.toFloat() })
        return node("Const", emptyList(), mapOf("dtype" to type(dtype), "value" to AttrValue.newBuilder().setTensor(tensor).build()))
    }

    companion object {
        private fun buffers(state: NeuroTrainingState) = arrayOf(state.weights, state.biases, state.weightVelocity, state.biasVelocity)
        private fun boolTensor(values: BooleanArray) = TBool.tensorOf(Shape.of(values.size.toLong()), DataBuffers.of(values, false, false))
        private fun booleans(value: Tensor) = BooleanArray(Math.toIntExact(value.shape().size())).also { (value as TBool).copyTo(DataBuffers.of(it, false, false)) }
        private fun prepare(states: Array<NeuroTrainingState>, hp: Neuro.HyperParameters, precision: Neuro.TrainingPrecision,
                            validationInputs: DoubleArray, validationTargets: DoubleArray, batchSize: Int,
                            paddedTopology: IntArray?): Array<NeuroTrainingState> {
            require(states.isNotEmpty() && states.size <= 1024 && batchSize > 0)
            states.forEach(TensorFlowMath::validate)
            val first = states[0]
            require(first.samples > 0)
            require(states.all { it.topology.size == first.topology.size && it.topology.first() == first.topology.first() &&
                it.topology.last() == first.topology.last() && it.inputs.contentEquals(first.inputs) && it.targets.contentEquals(first.targets) }) {
                "Cohort models must share depth, input/output dimensions and dataset."
            }
            require(validationInputs.size % first.topology.first() == 0 && validationTargets.size.toLong() ==
                validationInputs.size.toLong() / first.topology.first() * first.topology.last())
            val shape = paddedTopology ?: IntArray(first.topology.size) { layer -> states.maxOf { it.topology[layer] } }
            require(shape.size == first.topology.size && shape.first() == first.topology.first() && shape.last() == first.topology.last() &&
                states.all { state -> shape.indices.all { shape[it] >= state.topology[it] } }) { "Invalid padded cohort topology." }
            // Bound the numerical API itself: current/best optimizer state, loop temporaries, scoring activations and 64-epoch orders.
            val budget = 128L * 1024 * 1024
            val bytes = if (precision == Neuro.TrainingPrecision.FP64) 8L else 4L
            var footprint = 0L
            fun reserve(elements: Long, copies: Long = 1, elementBytes: Long = bytes) {
                require(elements <= budget / copies / elementBytes) { "Cohort exceeds the 128 MiB working-set bound." }
                footprint += elements * copies * elementBytes
                require(footprint <= budget) { "Cohort exceeds the 128 MiB working-set bound." }
            }
            for (layer in 1 until shape.size) reserve(shape[layer].toLong() * shape[layer - 1] + shape[layer], states.size * 12L)
            val rows = maxOf(first.samples, validationInputs.size / first.topology.first()).toLong()
            for (width in shape) reserve(rows * width, states.size * 3L)
            reserve(first.inputs.size.toLong() + first.targets.size + validationInputs.size + validationTargets.size)
            reserve(first.samples.toLong(), states.size * 64L, 4L)
            require((validationInputs.asSequence() + validationTargets.asSequence()).all { it.isFinite() })
            if (precision == Neuro.TrainingPrecision.FP32) require((states.asSequence().flatMap { state ->
                (buffers(state).flatMap { it.toList() } + listOf(state.inputs, state.targets)).asSequence().flatMap { it.asSequence() }
            } + validationInputs.asSequence() + validationTargets.asSequence() + sequenceOf(hp.learningRate, hp.momentum, hp.beta)).all { it.toFloat().isFinite() })
            val inputs = first.inputs.copyOf(); val targets = first.targets.copyOf()
            return Array(states.size) { index -> val state = states[index]
                state.copy(topology = state.topology.copyOf(), weights = Array(state.weights.size) { state.weights[it].copyOf() },
                    biases = Array(state.biases.size) { state.biases[it].copyOf() },
                    weightVelocity = Array(state.weightVelocity.size) { state.weightVelocity[it].copyOf() },
                    biasVelocity = Array(state.biasVelocity.size) { state.biasVelocity[it].copyOf() }, inputs = inputs, targets = targets, sharedDataset = true)
            }
        }
    }
}
