package com.lis.neuro

import java.util.SplittableRandom
import java.util.concurrent.ExecutionException
import java.util.concurrent.ForkJoinPool
import java.util.concurrent.Future

class Neuro @JvmOverloads constructor(
    topology: IntArray,
    private val hyperParameters: HyperParameters = HyperParameters.defaults()
) {
    enum class Kernel { AUTO, SCALAR, VECTOR }
    enum class SigmoidMode { EXACT, FAST }

    @JvmRecord
    data class HyperParameters @JvmOverloads constructor(
        val learningRate: Double,
        val momentum: Double,
        val beta: Double,
        val seed: Long,
        val kernel: Kernel = Kernel.AUTO,
        val sigmoidMode: SigmoidMode = SigmoidMode.EXACT
    ) {
        init {
            require(learningRate > 0.0 && learningRate.isFinite()) { "learningRate must be finite and > 0" }
            require(momentum >= 0.0 && momentum < 1.0 && momentum.isFinite()) { "momentum must be finite and in [0, 1)" }
            require(beta > 0.0 && beta.isFinite()) { "beta must be finite and > 0" }
        }

        fun withLearningRate(value: Double) = copy(learningRate = value)
        fun withMomentum(value: Double) = copy(momentum = value)
        fun withBeta(value: Double) = copy(beta = value)
        fun withSeed(value: Long) = copy(seed = value)
        fun withKernel(value: Kernel) = copy(kernel = value)
        fun withSigmoidMode(value: SigmoidMode) = copy(sigmoidMode = value)

        companion object {
            @JvmStatic fun defaults() = HyperParameters(0.5, 0.2, 1.0, DEFAULT_SEED)
        }
    }

    @JvmRecord data class TrainingResult(val epochs: Int, val error: Double, val converged: Boolean)
    @JvmRecord data class Statistics(val epochsTrained: Long, val samplesSeen: Long, val lastTrainingError: Double)

    private class Sample(input: DoubleArray, target: DoubleArray) {
        val input = input.copyOf()
        val target = target.copyOf()
    }

    private class PackedDataset(samples: List<Sample>, val inputSize: Int, val outputSize: Int) {
        val size = samples.size
        val inputs = DoubleArray(size * inputSize)
        val targets = DoubleArray(size * outputSize)
        init {
            for (sample in samples.indices) {
                samples[sample].input.copyInto(inputs, sample * inputSize)
                samples[sample].target.copyInto(targets, sample * outputSize)
            }
        }
    }

    private class Layer(val inputs: Int, val outputs: Int, random: SplittableRandom) {
        val weights = DoubleArray(inputs * outputs)
        val biases = DoubleArray(outputs)
        val weightVelocity = DoubleArray(weights.size)
        val biasVelocity = DoubleArray(outputs)
        init {
            val limit = Math.sqrt(6.0 / (inputs + outputs))
            for (index in weights.indices) weights[index] = random.nextDouble(-limit, limit)
        }
    }

    private class Workspace(topology: IntArray) {
        val activations = Array(topology.size) { if (it == 0) DoubleArray(0) else DoubleArray(topology[it]) }
        val deltas = Array(topology.size - 1) { DoubleArray(topology[it + 1]) }
    }

    private class BatchWorkspace(private val topology: IntArray) {
        val activations = Array(topology.size) { DoubleArray(0) }
        private var capacity = 0
        fun ensureCapacity(batchSize: Int) {
            if (batchSize <= capacity) return
            capacity = maxOf(batchSize, maxOf(8, capacity * 2))
            for (layer in 1 until topology.lastIndex) activations[layer] = DoubleArray(capacity * topology[layer])
        }
    }

    private class GradientBuffer(layers: Array<Layer>) {
        val weights = Array(layers.size) { DoubleArray(layers[it].weights.size) }
        val biases = Array(layers.size) { DoubleArray(layers[it].biases.size) }
        fun clear() {
            for (values in weights) values.fill(0.0)
            for (values in biases) values.fill(0.0)
        }
    }

    private class WorkerState(topology: IntArray, layers: Array<Layer>) {
        val workspace = Workspace(topology)
        val gradient = GradientBuffer(layers)
    }

    inner class InferenceSession internal constructor() {
        private val workspace = Workspace(this@Neuro.topology)
        private val batchWorkspace = BatchWorkspace(this@Neuro.topology)
        fun predictInto(input: DoubleArray, output: DoubleArray) {
            validatePrediction(input, output)
            forward(input, 0, workspace, output, 0)
        }
        fun predictBatch(inputs: DoubleArray, batchSize: Int, outputs: DoubleArray) {
            validateBatch(inputs, batchSize, outputs)
            batchWorkspace.ensureCapacity(batchSize)
            forwardBatch(inputs, batchSize, outputs, batchWorkspace)
        }
        internal fun predictSlice(inputs: DoubleArray, start: Int, end: Int, outputs: DoubleArray) {
            for (sample in start until end) {
                forward(inputs, sample * topology[0], workspace, outputs, sample * topology.last())
            }
        }
    }

    inner class ParallelInferenceSession internal constructor(private val parallelism: Int) : AutoCloseable {
        private val pool: ForkJoinPool
        private val sessions: Array<InferenceSession>
        init {
            require(parallelism > 0) { "parallelism must be > 0" }
            pool = ForkJoinPool(parallelism)
            sessions = Array(parallelism) { newInferenceSession() }
        }
        fun predictBatch(inputs: DoubleArray, batchSize: Int, outputs: DoubleArray) {
            validateBatch(inputs, batchSize, outputs)
            if (batchSize == 0) return
            val workerCount = minOf(parallelism, batchSize)
            val futures = Array<Future<*>>(workerCount) { worker ->
                val start = worker * batchSize / workerCount
                val end = (worker + 1) * batchSize / workerCount
                pool.submit { sessions[worker].predictSlice(inputs, start, end, outputs) }
            }
            await(futures)
        }
        override fun close() = pool.shutdown()
    }

    class FloatModel internal constructor(source: Neuro) {
        private val topology = source.topology.copyOf()
        private val kernel = source.hyperParameters.kernel
        private val sigmoidMode = source.hyperParameters.sigmoidMode
        private val beta = source.beta
        private val weights = Array(source.layers.size) { layer ->
            FloatArray(source.layers[layer].weights.size) { source.layers[layer].weights[it].toFloat() }
        }
        private val biases = Array(source.layers.size) { layer ->
            FloatArray(source.layers[layer].biases.size) { source.layers[layer].biases[it].toFloat() }
        }
        private val workspace = ThreadLocal.withInitial {
            Array(topology.size) { if (it == 0) FloatArray(0) else FloatArray(topology[it]) }
        }
        fun topology(): IntArray = topology.copyOf()
        fun predict(input: FloatArray): FloatArray = FloatArray(topology.last()).also { predictInto(input, it) }
        fun predictInto(input: FloatArray, output: FloatArray) {
            require(input.size == topology[0]) { "input length ${input.size} != expected ${topology[0]}" }
            require(output.size == topology.last()) { "output length ${output.size} != expected ${topology.last()}" }
            forward(input, 0, output, 0, workspace.get())
        }
        fun predictBatch(inputs: FloatArray, batchSize: Int, outputs: FloatArray) {
            require(batchSize >= 0) { "batchSize must be >= 0" }
            require(inputs.size.toLong() >= batchSize.toLong() * topology[0] &&
                outputs.size.toLong() >= batchSize.toLong() * topology.last()) { "batch arrays are too small" }
            val activations = workspace.get()
            for (sample in 0 until batchSize) {
                forward(inputs, sample * topology[0], outputs, sample * topology.last(), activations)
            }
        }
        private fun forward(input: FloatArray, inputOffset: Int, output: FloatArray, outputOffset: Int,
                            activations: Array<FloatArray>) {
            for (layer in weights.indices) {
                val inputs = topology[layer]
                val outputs = topology[layer + 1]
                val source = if (layer == 0) input else activations[layer]
                val sourceOffset = if (layer == 0) inputOffset else 0
                val destination = if (layer == weights.lastIndex) output else activations[layer + 1]
                val destinationOffset = if (layer == weights.lastIndex) outputOffset else 0
                val vector = useFloatVector(kernel, inputs)
                for (out in 0 until outputs) {
                    val offset = out * inputs
                    var sum = biases[layer][out]
                    if (vector) {
                        sum += NeuroVectorOps.dot(source, sourceOffset, weights[layer], offset, inputs)
                    } else {
                        for (index in 0 until inputs) sum = Math.fma(source[sourceOffset + index], weights[layer][offset + index], sum)
                    }
                    destination[destinationOffset + out] = activate(sum * beta, sigmoidMode).toFloat()
                }
            }
        }
    }

    private val topology = validateTopology(topology)
    private val initializationRandom = SplittableRandom(hyperParameters.seed)
    private val layers = Array(this.topology.size - 1) {
        Layer(this.topology[it], this.topology[it + 1], initializationRandom)
    }
    private val trainingSamples = ArrayList<Sample>()
    private val testSamples = ArrayList<Sample>()
    private val trainingWorkspace = Workspace(this.topology)
    private val inferenceSession = ThreadLocal.withInitial { newInferenceSession() }
    private val shuffleRandom = SplittableRandom(hyperParameters.seed xor -7046029254386353131L)
    private val batchGradient = GradientBuffer(layers)
    private val learningRate = hyperParameters.learningRate
    private val momentum = hyperParameters.momentum
    private val beta = hyperParameters.beta
    private var packedTraining: PackedDataset? = null
    private var packedTests: PackedDataset? = null
    private var trainingOrder = IntArray(0)
    private var epochsTrained = 0L
    private var samplesSeen = 0L
    private var lastTrainingError = Double.NaN
    private var trainingOwner: Any? = null
    private var ownerAccess = false
    private var pendingDeviceOrder: IntArray? = null

    constructor(topology: IntArray, momentum: Double, beta: Double, learningRate: Double) :
        this(topology, HyperParameters(learningRate, momentum, beta, DEFAULT_SEED))

    fun topology(): IntArray = topology.copyOf()
    fun hyperParameters(): HyperParameters = hyperParameters
    fun parameterCount(): Int = layers.sumOf { it.weights.size + it.biases.size }
    fun trainingSampleCount(): Int = trainingSamples.size
    fun testSampleCount(): Int = testSamples.size
    fun statistics() = Statistics(epochsTrained, samplesSeen, lastTrainingError)
    @Synchronized fun addTrainingSample(input: DoubleArray, target: DoubleArray): Neuro {
        checkTrainingAccess()
        validateSample(input, target)
        trainingSamples.add(Sample(input, target))
        packedTraining = null
        pendingDeviceOrder = null
        return this
    }
    @Synchronized fun addTestSample(input: DoubleArray, target: DoubleArray): Neuro {
        checkTrainingAccess()
        validateSample(input, target)
        testSamples.add(Sample(input, target))
        packedTests = null
        return this
    }
    fun predict(input: DoubleArray): DoubleArray = DoubleArray(topology.last()).also { predictInto(input, it) }
    fun predictInto(input: DoubleArray, output: DoubleArray) = inferenceSession.get().predictInto(input, output)
    fun newInferenceSession() = InferenceSession()
    fun predictBatch(inputs: DoubleArray, batchSize: Int, outputs: DoubleArray) =
        inferenceSession.get().predictBatch(inputs, batchSize, outputs)
    fun predictBatchParallel(inputs: DoubleArray, batchSize: Int, outputs: DoubleArray, parallelism: Int) {
        newParallelInferenceSession(parallelism).use { it.predictBatch(inputs, batchSize, outputs) }
    }
    fun newParallelInferenceSession(parallelism: Int) = ParallelInferenceSession(parallelism)
    fun toFloatModel() = FloatModel(this)
    internal fun backendWeights(layer: Int) = layers[layer].weights.copyOf()
    internal fun backendBiases(layer: Int) = layers[layer].biases.copyOf()

    @JvmOverloads
    fun newTrainingSession(backend: TrainingBackend = TrainingBackend.CPU): NeuroTrainingSession =
        openTrainingSession(this, backend)

    internal fun exportTrainingState(): NeuroTrainingState {
        val data = trainingData()
        return NeuroTrainingState(topology.copyOf(), Array(layers.size) { layers[it].weights.copyOf() },
            Array(layers.size) { layers[it].biases.copyOf() }, Array(layers.size) { layers[it].weightVelocity.copyOf() },
            Array(layers.size) { layers[it].biasVelocity.copyOf() }, data.inputs.copyOf(), data.targets.copyOf())
    }

    @Synchronized internal fun acquireTraining(owner: Any) {
        check(trainingOwner == null) { "This model already has a training session. Close it before opening another." }
        trainingOwner = owner
    }

    @Synchronized internal fun releaseTraining(owner: Any) {
        check(trainingOwner === owner) { "Training session does not own this model." }
        trainingOwner = null
    }

    @Synchronized internal fun <T> withTraining(owner: Any, action: () -> T): T {
        check(trainingOwner === owner) { "Training session does not own this model." }
        ownerAccess = true
        try { return action() } finally { ownerAccess = false }
    }

    private fun checkTrainingAccess() {
        check(trainingOwner == null || ownerAccess && Thread.holdsLock(this)) {
            "This model belongs to a training session. Use or close that session before modifying it."
        }
    }

    internal fun deviceTrainingOrder(): IntArray {
        checkTrainingAccess()
        requireTrainingSamples()
        pendingDeviceOrder?.let { return it.copyOf() }
        ensureTrainingOrder(trainingSamples.size)
        shuffleTrainingOrder()
        pendingDeviceOrder = trainingOrder.copyOf()
        return trainingOrder.copyOf()
    }

    internal fun recordTrainingError(value: Double) {
        checkTrainingAccess()
        lastTrainingError = value
    }

    internal fun commitDeviceEpoch(state: NeuroTrainingState): Double {
        // Validate every downloaded buffer before publishing any part of the epoch.
        for (values in state.weights + state.biases + state.weightVelocity + state.biasVelocity) {
            check(values.all { it.isFinite() }) { "CUDA training produced non-finite parameters." }
        }
        for (index in layers.indices) {
            state.weights[index].copyInto(layers[index].weights)
            state.biases[index].copyInto(layers[index].biases)
            state.weightVelocity[index].copyInto(layers[index].weightVelocity)
            state.biasVelocity[index].copyInto(layers[index].biasVelocity)
        }
        pendingDeviceOrder = null
        epochsTrained++
        samplesSeen += trainingSamples.size
        lastTrainingError = trainingError()
        return lastTrainingError
    }

    @Synchronized fun trainEpoch(): Double {
        checkTrainingAccess()
        requireTrainingSamples()
        return trainOnlineEpoch(trainingData(), true)
    }
    @Synchronized fun train(epochs: Int) {
        checkTrainingAccess()
        require(epochs >= 0) { "epochs must be >= 0" }
        if (epochs == 0) return
        requireTrainingSamples()
        val data = trainingData()
        repeat(epochs - 1) { trainOnlineEpoch(data, false) }
        trainOnlineEpoch(data, true)
    }
    @Synchronized @JvmOverloads fun trainUntil(targetError: Double, maxEpochs: Int, checkEvery: Int = 1): TrainingResult {
        checkTrainingAccess()
        require(targetError >= 0.0 && targetError.isFinite()) { "targetError must be finite and >= 0" }
        require(maxEpochs >= 0) { "maxEpochs must be >= 0" }
        require(checkEvery > 0) { "checkEvery must be > 0" }
        requireTrainingSamples()
        val data = trainingData()
        var error = error(data, trainingWorkspace)
        if (error <= targetError) {
            lastTrainingError = error
            return TrainingResult(0, error, true)
        }
        val event = NeuroJfr.trainingRun(targetError, maxEpochs)
        var epochs = 0
        while (epochs < maxEpochs && error > targetError) {
            trainOnlineEpoch(data, false)
            epochs++
            if (epochs % checkEvery == 0 || epochs == maxEpochs) {
                error = error(data, trainingWorkspace)
                lastTrainingError = error
            }
        }
        val converged = error <= targetError
        NeuroJfr.commitTrainingRun(event, epochs, error, converged)
        return TrainingResult(epochs, error, converged)
    }
    @Synchronized @JvmOverloads fun trainMiniBatch(epochs: Int, batchSize: Int, parallelism: Int = 1) {
        checkTrainingAccess()
        require(epochs >= 0) { "epochs must be >= 0" }
        require(batchSize > 0) { "batchSize must be > 0" }
        require(parallelism > 0) { "parallelism must be > 0" }
        if (epochs == 0) return
        requireTrainingSamples()
        val data = trainingData()
        val pool = if (parallelism > 1) ForkJoinPool(parallelism) else null
        val workers = if (parallelism > 1) Array(parallelism) { WorkerState(topology, layers) } else emptyArray()
        try {
            repeat(epochs) { trainMiniBatchEpoch(data, batchSize, parallelism, pool, workers) }
        } finally {
            pool?.shutdown()
        }
        lastTrainingError = error(data, trainingWorkspace)
    }
    fun trainingError(): Double = if (trainingSamples.isEmpty()) Double.NaN else error(trainingData(), trainingWorkspace)
    fun testError(): Double = if (testSamples.isEmpty()) Double.NaN else error(testData(), trainingWorkspace)

    private fun trainOnlineEpoch(data: PackedDataset, evaluateError: Boolean): Double {
        ensureTrainingOrder(data.size)
        if (pendingDeviceOrder != null) pendingDeviceOrder = null else shuffleTrainingOrder()
        val event = NeuroJfr.trainingEpoch(epochsTrained + 1, data.size)
        for (sample in trainingOrder) {
            val inputOffset = sample * data.inputSize
            forward(data.inputs, inputOffset, trainingWorkspace, null, 0)
            backpropagate(data.targets, sample * data.outputSize, trainingWorkspace)
            applyOnlineGradient(data.inputs, inputOffset, trainingWorkspace)
        }
        samplesSeen += data.size
        epochsTrained++
        val error = if (evaluateError) error(data, trainingWorkspace) else Double.NaN
        if (evaluateError) lastTrainingError = error
        NeuroJfr.commitTrainingEpoch(event, error)
        return error
    }

    private fun trainMiniBatchEpoch(data: PackedDataset, batchSize: Int, parallelism: Int,
                                    pool: ForkJoinPool?, workers: Array<WorkerState>) {
        ensureTrainingOrder(data.size)
        if (pendingDeviceOrder != null) pendingDeviceOrder = null else shuffleTrainingOrder()
        var start = 0
        while (start < data.size) {
            val end = minOf(data.size, start + batchSize)
            val count = end - start
            batchGradient.clear()
            if (parallelism == 1 || count == 1) {
                for (position in start until end) accumulateSampleGradient(data, trainingOrder[position], trainingWorkspace, batchGradient)
            } else {
                val activeWorkers = minOf(parallelism, count)
                val futures = Array<Future<*>>(activeWorkers) { worker ->
                    val state = workers[worker]
                    state.gradient.clear()
                    val workerStart = start + worker * count / activeWorkers
                    val workerEnd = start + (worker + 1) * count / activeWorkers
                    checkNotNull(pool).submit {
                        for (position in workerStart until workerEnd) {
                            accumulateSampleGradient(data, trainingOrder[position], state.workspace, state.gradient)
                        }
                    }
                }
                await(futures)
                for (worker in 0 until activeWorkers) addGradient(batchGradient, workers[worker].gradient)
            }
            applyBatchGradient(batchGradient, count)
            start = end
        }
        samplesSeen += data.size
        epochsTrained++
    }

    private fun accumulateSampleGradient(data: PackedDataset, sample: Int, workspace: Workspace, gradient: GradientBuffer) {
        val inputOffset = sample * data.inputSize
        forward(data.inputs, inputOffset, workspace, null, 0)
        backpropagate(data.targets, sample * data.outputSize, workspace)
        for (layerIndex in layers.indices) {
            val layer = layers[layerIndex]
            val source = if (layerIndex == 0) data.inputs else workspace.activations[layerIndex]
            val sourceOffset = if (layerIndex == 0) inputOffset else 0
            val delta = workspace.deltas[layerIndex]
            for (output in 0 until layer.outputs) {
                val offset = output * layer.inputs
                val scale = delta[output]
                if (useVector(layer.inputs)) {
                    NeuroVectorOps.addOuterProduct(gradient.weights[layerIndex], offset, source, sourceOffset, layer.inputs, scale)
                } else {
                    for (input in 0 until layer.inputs) {
                        val index = offset + input
                        gradient.weights[layerIndex][index] = Math.fma(source[sourceOffset + input], scale, gradient.weights[layerIndex][index])
                    }
                }
                gradient.biases[layerIndex][output] += scale
            }
        }
    }

    private fun addGradient(destination: GradientBuffer, source: GradientBuffer) {
        for (layer in layers.indices) {
            if (useVector(destination.weights[layer].size)) {
                NeuroVectorOps.add(destination.weights[layer], source.weights[layer], destination.weights[layer].size)
            } else {
                for (index in destination.weights[layer].indices) destination.weights[layer][index] += source.weights[layer][index]
            }
            for (index in destination.biases[layer].indices) destination.biases[layer][index] += source.biases[layer][index]
        }
    }

    private fun applyBatchGradient(gradient: GradientBuffer, sampleCount: Int) {
        val gradientScale = learningRate / sampleCount
        for (layerIndex in layers.indices) {
            val layer = layers[layerIndex]
            for (output in 0 until layer.outputs) {
                val offset = output * layer.inputs
                if (useVector(layer.inputs)) {
                    NeuroVectorOps.update(layer.weights, layer.weightVelocity, offset, gradient.weights[layerIndex], offset,
                        layer.inputs, momentum, gradientScale)
                } else {
                    for (input in 0 until layer.inputs) {
                        val index = offset + input
                        val velocity = Math.fma(momentum, layer.weightVelocity[index], gradientScale * gradient.weights[layerIndex][index])
                        layer.weightVelocity[index] = velocity
                        layer.weights[index] += velocity
                    }
                }
                val velocity = Math.fma(momentum, layer.biasVelocity[output], gradientScale * gradient.biases[layerIndex][output])
                layer.biasVelocity[output] = velocity
                layer.biases[output] += velocity
            }
        }
    }

    private fun forward(input: DoubleArray, inputOffset: Int, workspace: Workspace,
                        externalOutput: DoubleArray?, externalOutputOffset: Int) {
        for (layerIndex in layers.indices) {
            val layer = layers[layerIndex]
            val source = if (layerIndex == 0) input else workspace.activations[layerIndex]
            val sourceOffset = if (layerIndex == 0) inputOffset else 0
            val external = layerIndex == layers.lastIndex && externalOutput != null
            val destination = if (external) externalOutput else workspace.activations[layerIndex + 1]
            val destinationOffset = if (external) externalOutputOffset else 0
            val vector = useVector(layer.inputs)
            for (output in 0 until layer.outputs) {
                val offset = output * layer.inputs
                var sum = layer.biases[output]
                if (vector) {
                    sum += NeuroVectorOps.dot(source, sourceOffset, layer.weights, offset, layer.inputs)
                } else {
                    var index = 0
                    val limit = layer.inputs - (layer.inputs and 3)
                    while (index < limit) {
                        sum = Math.fma(source[sourceOffset + index], layer.weights[offset + index], sum)
                        sum = Math.fma(source[sourceOffset + index + 1], layer.weights[offset + index + 1], sum)
                        sum = Math.fma(source[sourceOffset + index + 2], layer.weights[offset + index + 2], sum)
                        sum = Math.fma(source[sourceOffset + index + 3], layer.weights[offset + index + 3], sum)
                        index += 4
                    }
                    while (index < layer.inputs) {
                        sum = Math.fma(source[sourceOffset + index], layer.weights[offset + index], sum)
                        index++
                    }
                }
                destination[destinationOffset + output] = activate(sum * beta, hyperParameters.sigmoidMode)
            }
        }
    }

    private fun forwardBatch(inputs: DoubleArray, batchSize: Int, outputs: DoubleArray, workspace: BatchWorkspace) {
        for (layerIndex in layers.indices) {
            val layer = layers[layerIndex]
            val source = if (layerIndex == 0) inputs else workspace.activations[layerIndex]
            val destination = if (layerIndex == layers.lastIndex) outputs else workspace.activations[layerIndex + 1]
            val vector = useVector(layer.inputs)
            for (output in 0 until layer.outputs) {
                val offset = output * layer.inputs
                for (sample in 0 until batchSize) {
                    val sourceOffset = sample * layer.inputs
                    var sum = layer.biases[output]
                    if (vector) {
                        sum += NeuroVectorOps.dot(source, sourceOffset, layer.weights, offset, layer.inputs)
                    } else {
                        for (input in 0 until layer.inputs) sum = Math.fma(source[sourceOffset + input], layer.weights[offset + input], sum)
                    }
                    destination[sample * layer.outputs + output] = activate(sum * beta, hyperParameters.sigmoidMode)
                }
            }
        }
    }

    private fun backpropagate(target: DoubleArray, targetOffset: Int, workspace: Workspace) {
        val outputActivation = workspace.activations[topology.lastIndex]
        val outputDelta = workspace.deltas[layers.lastIndex]
        if (useVector(outputDelta.size)) {
            NeuroVectorOps.outputDelta(target, targetOffset, outputActivation, outputDelta, outputDelta.size, beta)
        } else {
            for (output in outputDelta.indices) {
                val activation = outputActivation[output]
                outputDelta[output] = (target[targetOffset + output] - activation) * beta * activation * (1.0 - activation)
            }
        }
        for (layerIndex in layers.lastIndex - 1 downTo 0) {
            val currentDelta = workspace.deltas[layerIndex]
            val currentActivation = workspace.activations[layerIndex + 1]
            val nextLayer = layers[layerIndex + 1]
            val nextDelta = workspace.deltas[layerIndex + 1]
            if (useVector(nextLayer.inputs)) {
                NeuroVectorOps.initScaled(currentDelta, nextLayer.weights, 0, nextLayer.inputs, nextDelta[0])
                for (output in 1 until nextLayer.outputs) {
                    NeuroVectorOps.addScaled(currentDelta, nextLayer.weights, output * nextLayer.inputs, nextLayer.inputs, nextDelta[output])
                }
                NeuroVectorOps.applyDerivative(currentDelta, currentActivation, currentDelta.size, beta)
            } else {
                for (current in 0 until nextLayer.inputs) currentDelta[current] = nextDelta[0] * nextLayer.weights[current]
                for (output in 1 until nextLayer.outputs) {
                    val weightedDelta = nextDelta[output]
                    val offset = output * nextLayer.inputs
                    for (current in 0 until nextLayer.inputs) {
                        currentDelta[current] = Math.fma(weightedDelta, nextLayer.weights[offset + current], currentDelta[current])
                    }
                }
                for (current in currentDelta.indices) {
                    val activation = currentActivation[current]
                    currentDelta[current] *= beta * activation * (1.0 - activation)
                }
            }
        }
    }

    private fun applyOnlineGradient(input: DoubleArray, inputOffset: Int, workspace: Workspace) {
        for (layerIndex in layers.indices) {
            val layer = layers[layerIndex]
            val source = if (layerIndex == 0) input else workspace.activations[layerIndex]
            val sourceOffset = if (layerIndex == 0) inputOffset else 0
            val delta = workspace.deltas[layerIndex]
            val vector = useVector(layer.inputs)
            for (output in 0 until layer.outputs) {
                val offset = output * layer.inputs
                val scale = learningRate * delta[output]
                if (vector) {
                    NeuroVectorOps.update(layer.weights, layer.weightVelocity, offset, source, sourceOffset, layer.inputs, momentum, scale)
                } else {
                    for (index in 0 until layer.inputs) {
                        val weightIndex = offset + index
                        val velocity = Math.fma(momentum, layer.weightVelocity[weightIndex], scale * source[sourceOffset + index])
                        layer.weightVelocity[weightIndex] = velocity
                        layer.weights[weightIndex] += velocity
                    }
                }
                val velocity = Math.fma(momentum, layer.biasVelocity[output], learningRate * delta[output])
                layer.biasVelocity[output] = velocity
                layer.biases[output] += velocity
            }
        }
    }

    private fun error(data: PackedDataset, workspace: Workspace): Double {
        var squaredError = 0.0
        for (sample in 0 until data.size) {
            forward(data.inputs, sample * data.inputSize, workspace, null, 0)
            val output = workspace.activations[topology.lastIndex]
            for (index in 0 until data.outputSize) {
                val difference = data.targets[sample * data.outputSize + index] - output[index]
                squaredError = Math.fma(difference, difference, squaredError)
            }
        }
        return Math.sqrt(squaredError / (data.size.toDouble() * data.outputSize))
    }
    private fun trainingData(): PackedDataset = packedTraining ?: PackedDataset(trainingSamples, topology[0], topology.last()).also { packedTraining = it }
    private fun testData(): PackedDataset = packedTests ?: PackedDataset(testSamples, topology[0], topology.last()).also { packedTests = it }
    private fun useVector(length: Int): Boolean = when (hyperParameters.kernel) {
        Kernel.SCALAR -> false
        Kernel.VECTOR -> length >= NeuroVectorOps.doubleLanes()
        Kernel.AUTO -> length >= NeuroVectorOps.doubleLanes() * 2
    }
    private fun validatePrediction(input: DoubleArray, output: DoubleArray) {
        require(input.size == topology[0]) { "input length ${input.size} != expected ${topology[0]}" }
        require(output.size == topology.last()) { "output length ${output.size} != expected ${topology.last()}" }
    }
    private fun validateBatch(inputs: DoubleArray, batchSize: Int, outputs: DoubleArray) {
        require(batchSize >= 0) { "batchSize must be >= 0" }
        require(inputs.size.toLong() >= batchSize.toLong() * topology[0]) { "inputs array is too small for batchSize" }
        require(outputs.size.toLong() >= batchSize.toLong() * topology.last()) { "outputs array is too small for batchSize" }
    }
    private fun validateSample(input: DoubleArray, target: DoubleArray) {
        require(input.size == topology[0]) { "input length ${input.size} != expected ${topology[0]}" }
        require(target.size == topology.last()) { "target length ${target.size} != expected ${topology.last()}" }
        requireFinite(input, "input")
        requireFinite(target, "target")
    }
    private fun requireTrainingSamples() = check(trainingSamples.isNotEmpty()) { "no training samples" }
    private fun ensureTrainingOrder(size: Int) {
        if (trainingOrder.size != size) trainingOrder = IntArray(size) { it }
    }
    private fun shuffleTrainingOrder() {
        for (index in trainingOrder.lastIndex downTo 1) {
            val other = shuffleRandom.nextInt(index + 1)
            val value = trainingOrder[index]
            trainingOrder[index] = trainingOrder[other]
            trainingOrder[other] = value
        }
    }

    companion object {
        private const val DEFAULT_SEED = 0x5EEDL
        private fun validateTopology(topology: IntArray): IntArray {
            require(topology.size >= 2) { "topology must contain at least input and output layers" }
            require(topology.all { it > 0 }) { "all layer sizes must be positive" }
            NumericInputs.checkAllocation(topology)
            return topology.copyOf()
        }
        private fun requireFinite(values: DoubleArray, name: String) {
            require(values.all { it.isFinite() }) { "$name contains a non-finite value" }
        }
        private fun useFloatVector(kernel: Kernel, length: Int): Boolean = when (kernel) {
            Kernel.SCALAR -> false
            Kernel.VECTOR -> length >= NeuroVectorOps.floatLanes()
            Kernel.AUTO -> length >= NeuroVectorOps.floatLanes() * 2
        }
        internal fun activate(value: Double, mode: SigmoidMode): Double {
            if (mode == SigmoidMode.EXACT) return 1.0 / (1.0 + Math.exp(-value))
            if (value >= 0.0) return 1.0 / (1.0 + fastExpNegative(-value))
            val exp = fastExpNegative(value)
            return exp / (1.0 + exp)
        }
        private fun fastExpNegative(value: Double): Double {
            if (value <= -745.0) return 0.0
            val exponent = (value * 1.4426950408889634).toInt()
            val remainder = value - exponent * 0.6931471805599453
            val square = remainder * remainder
            val polynomial = 1.0 + remainder + square * (0.5 + remainder * (0.16666666666666666 +
                remainder * (0.041666666666666664 + remainder * 0.008333333333333333)))
            return Math.scalb(polynomial, exponent)
        }
        private fun await(futures: Array<Future<*>>) {
            for (future in futures) {
                try {
                    future.get()
                } catch (exception: InterruptedException) {
                    Thread.currentThread().interrupt()
                    throw IllegalStateException("parallel operation interrupted", exception)
                } catch (exception: ExecutionException) {
                    throw IllegalStateException("parallel operation failed", exception.cause)
                }
            }
        }
    }
}
