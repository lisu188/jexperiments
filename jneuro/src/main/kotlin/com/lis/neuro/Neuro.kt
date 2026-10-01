package com.lis.neuro

import java.util.SplittableRandom
import java.util.concurrent.ForkJoinPool

class Neuro @JvmOverloads constructor(
    topology: IntArray,
    private val hyperParameters: HyperParameters = HyperParameters.defaults()
) {
    /** Legacy compatibility hint. TensorFlow selects its compute kernels for every value. */
    enum class Kernel { AUTO, SCALAR, VECTOR }
    enum class SigmoidMode { EXACT, FAST }
    enum class BatchBackend { CPU, GPU, AUTO }
    enum class TrainingPrecision { FP64, FP32 }

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

    internal class Sample(input: DoubleArray, target: DoubleArray) {
        val input = input.copyOf()
        val target = target.copyOf()
    }

    internal class PackedDataset(samples: List<Sample>, val inputSize: Int, val outputSize: Int) {
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

    /** Search-only shared storage. These samples and packed buffers are immutable after construction. */
    internal class SharedDatasets private constructor(
        internal val training: ArrayList<Sample>, internal val validation: ArrayList<Sample>
    ) {
        internal val packedTraining = PackedDataset(training, 2, 1)
        internal val packedValidation = PackedDataset(validation, 2, 1)

        companion object {
            fun fromSamples(training: List<NeuroLearningSets.Sample>, validation: List<NeuroLearningSets.Sample>): SharedDatasets {
                fun pack(samples: List<NeuroLearningSets.Sample>) = ArrayList(samples.map { sample ->
                    require(sample.x.isFinite() && sample.y.isFinite() && sample.target.isFinite()) { "Samples must be finite" }
                    Sample(doubleArrayOf(sample.x, sample.y), doubleArrayOf(sample.target))
                })
                return SharedDatasets(pack(training), pack(validation))
            }
        }
    }

    internal class Layer(val inputs: Int, val outputs: Int, random: SplittableRandom) {
        val weights = DoubleArray(inputs * outputs)
        val biases = DoubleArray(outputs)
        val weightVelocity = DoubleArray(weights.size)
        val biasVelocity = DoubleArray(outputs)
        init {
            val limit = TensorFlowMath.xavierLimit(inputs, outputs)
            for (index in weights.indices) weights[index] = random.nextDouble(-limit, limit)
        }
    }

    inner class InferenceSession internal constructor() {
        private val inferenceId = NeuroLog.id("inference")
        init {
            NeuroLog.debug("inference", "inference.session.opened") { mapOf("model" to logId,
                "inference" to inferenceId, "backend" to "CPU", "precision" to "FP64", "implementation" to "TensorFlow") }
        }
        fun predictInto(input: DoubleArray, output: DoubleArray) {
            validatePrediction(input, output)
            predictTensorFlow(input, 1).copyInto(output)
            NeuroLog.trace("inference", "inference.completed") { mapOf("model" to logId,
                "inference" to inferenceId, "batchSize" to 1, "outputs" to output.size) }
        }
        fun predictBatch(inputs: DoubleArray, batchSize: Int, outputs: DoubleArray) {
            validateBatch(inputs, batchSize, outputs)
            if (batchSize > 0) predictTensorFlow(inputs, batchSize).copyInto(outputs)
            NeuroLog.trace("inference", "inference.completed") { mapOf("model" to logId,
                "inference" to inferenceId, "batchSize" to batchSize, "outputWidth" to topology.last()) }
        }
        internal fun predictSlice(inputs: DoubleArray, start: Int, end: Int, outputs: DoubleArray) {
            val slice = inputs.copyOfRange(start * topology[0], end * topology[0])
            predictTensorFlow(slice, end - start).copyInto(outputs, start * topology.last())
        }
    }

    inner class ParallelInferenceSession internal constructor(private val parallelism: Int) : AutoCloseable {
        private val inferenceId = NeuroLog.id("parallel-inference")
        private val pool: ForkJoinPool
        init {
            require(parallelism > 0) { "parallelism must be > 0" }
            pool = ForkJoinPool(parallelism)
            NeuroLog.debug("inference", "inference.parallel.opened") { mapOf("model" to logId,
                "inference" to inferenceId, "parallelism" to parallelism) }
        }
        fun predictBatch(inputs: DoubleArray, batchSize: Int, outputs: DoubleArray) {
            validateBatch(inputs, batchSize, outputs)
            if (batchSize == 0) return
            NeuroCpuBatchTrainer.parallelFor(batchSize, parallelism, pool) { start, end ->
                inferenceSession.get().predictSlice(inputs, start, end, outputs)
            }
            NeuroLog.trace("inference", "inference.parallel.completed") { mapOf("model" to logId,
                "inference" to inferenceId, "batchSize" to batchSize,
                "workers" to if (batchSize < 256) 1 else minOf(parallelism, batchSize)) }
        }
        override fun close() {
            pool.shutdown()
            NeuroLog.debug("inference", "inference.parallel.closed") { mapOf("model" to logId,
                "inference" to inferenceId) }
        }
    }

    class FloatModel internal constructor(source: Neuro) {
        private val topology = source.topology.copyOf()
        private val parameters = source.hyperParameters
        private val weights = Array(source.layers.size) { source.layers[it].weights.copyOf() }
        private val biases = Array(source.layers.size) { source.layers[it].biases.copyOf() }
        fun topology(): IntArray = topology.copyOf()
        fun predict(input: FloatArray): FloatArray = FloatArray(topology.last()).also { predictInto(input, it) }
        fun predictInto(input: FloatArray, output: FloatArray) {
            require(input.size == topology[0]) { "input length ${input.size} != expected ${topology[0]}" }
            require(output.size == topology.last()) { "output length ${output.size} != expected ${topology.last()}" }
            predictBatch(input, 1, output)
        }
        fun predictBatch(inputs: FloatArray, batchSize: Int, outputs: FloatArray) {
            require(batchSize >= 0) { "batchSize must be >= 0" }
            require(inputs.size.toLong() >= batchSize.toLong() * topology[0] &&
                outputs.size.toLong() >= batchSize.toLong() * topology.last()) { "batch arrays are too small" }
            if (batchSize == 0) return
            val input = DoubleArray(batchSize * topology[0]) { inputs[it].toDouble() }
            val values = TensorFlowMath.predict(topology, weights, biases, parameters, input, batchSize,
                TrainingPrecision.FP32, TrainingBackend.CPU)
            for (index in values.indices) outputs[index] = values[index].toFloat()
        }
    }

    internal val logId = NeuroLog.id("model")
    private val topology = validateTopology(topology)
    private val initializationRandom = SplittableRandom(hyperParameters.seed)
    private val layers = Array(this.topology.size - 1) {
        Layer(this.topology[it], this.topology[it + 1], initializationRandom)
    }
    private var trainingSamples = ArrayList<Sample>()
    private var testSamples = ArrayList<Sample>()
    private var sharedTrainingSamples = false
    private var sharedTestSamples = false
    private val inferenceSession = ThreadLocal.withInitial { newInferenceSession() }
    private val trainingShuffle = TrainingShuffle(hyperParameters.seed xor -7046029254386353131L)
    private val learningRate = hyperParameters.learningRate
    private val momentum = hyperParameters.momentum
    private val beta = hyperParameters.beta
    private var packedTraining: PackedDataset? = null
    private var packedTests: PackedDataset? = null
    private var epochsTrained = 0L
    private var samplesSeen = 0L
    private var lastTrainingError = Double.NaN
    private var trainingOwner: Any? = null
    internal var trainingSessionLogId: String? = null
        private set
    private var ownerAccess = false

    init {
        NeuroLog.debug("model", "model.created") { mapOf("model" to logId,
            "topology" to this.topology.joinToString("x"), "parameters" to parameterCount(),
            "learningRate" to learningRate, "momentum" to momentum, "beta" to beta,
            "seed" to hyperParameters.seed, "kernel" to hyperParameters.kernel,
            "sigmoid" to hyperParameters.sigmoidMode) }
    }

    constructor(topology: IntArray, momentum: Double, beta: Double, learningRate: Double) :
        this(topology, HyperParameters(learningRate, momentum, beta, DEFAULT_SEED))

    fun topology(): IntArray = topology.copyOf()
    fun hyperParameters(): HyperParameters = hyperParameters
    fun parameterCount(): Int = layers.sumOf { it.weights.size + it.biases.size }
    fun trainingSampleCount(): Int = trainingSamples.size
    fun testSampleCount(): Int = testSamples.size
    fun statistics() = Statistics(epochsTrained, samplesSeen, lastTrainingError)
    internal fun trainingEpochCount(): Long = epochsTrained
    internal fun processedSampleCount(): Long = samplesSeen
    internal fun recordedTrainingError(): Double = lastTrainingError

    @Synchronized internal fun attachSharedDatasets(data: SharedDatasets): Neuro {
        checkTrainingAccess()
        check(trainingSamples.isEmpty() && testSamples.isEmpty() && epochsTrained == 0L) { "Shared data requires a fresh model" }
        require(topology.first() == 2 && topology.last() == 1) { "Search data requires topology 2-...-1" }
        trainingSamples = data.training
        testSamples = data.validation
        packedTraining = data.packedTraining
        packedTests = data.packedValidation
        sharedTrainingSamples = true
        sharedTestSamples = true
        lastTrainingError = Double.NaN
        return this
    }

    @Synchronized fun addTrainingSample(input: DoubleArray, target: DoubleArray): Neuro {
        checkTrainingAccess()
        validateSample(input, target)
        if (sharedTrainingSamples) {
            trainingSamples = ArrayList(trainingSamples)
            sharedTrainingSamples = false
        }
        trainingSamples.add(Sample(input, target))
        packedTraining = null
        lastTrainingError = Double.NaN
        trainingShuffle.invalidate()
        NeuroLog.trace("model", "dataset.sample.added") { mapOf("model" to logId,
            "dataset" to "training", "samples" to trainingSamples.size) }
        return this
    }
    @Synchronized fun addTestSample(input: DoubleArray, target: DoubleArray): Neuro {
        checkTrainingAccess()
        validateSample(input, target)
        if (sharedTestSamples) {
            testSamples = ArrayList(testSamples)
            sharedTestSamples = false
        }
        testSamples.add(Sample(input, target))
        packedTests = null
        NeuroLog.trace("model", "dataset.sample.added") { mapOf("model" to logId,
            "dataset" to "test", "samples" to testSamples.size) }
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
    fun toFloatModel() = FloatModel(this).also {
        NeuroLog.debug("inference", "inference.float.exported") { mapOf("model" to logId,
            "precision" to "FP32", "parameters" to parameterCount(), "epoch" to epochsTrained) }
    }
    internal fun backendWeights(layer: Int) = layers[layer].weights.copyOf()
    internal fun backendBiases(layer: Int) = layers[layer].biases.copyOf()

    @JvmOverloads
    fun newTrainingSession(backend: TrainingBackend = TrainingBackend.CPU,
                           precision: TrainingPrecision = TrainingPrecision.FP64,
                           batchSize: Int = 1,
                           engine: TrainingEngine = TrainingEngine.REFERENCE): NeuroTrainingSession =
        openConfiguredTrainingSession(this, backend, precision, batchSize, engine = engine)

    internal fun backendWeightVelocity(layer: Int) = layers[layer].weightVelocity.copyOf()
    internal fun backendBiasVelocity(layer: Int) = layers[layer].biasVelocity.copyOf()
    internal fun backendLayers(): Array<Layer> = layers
    internal fun backendTrainingData(): PackedDataset = trainingData()
    internal fun backendNextTrainingOrder(size: Int): IntArray {
        require(size == trainingSamples.size)
        return deviceTrainingOrder()
    }
    internal fun backendCompleteEpoch(sampleCount: Int) {
        checkTrainingAccess()
        require(sampleCount == trainingSamples.size)
        trainingShuffle.commit(1)
        samplesSeen += sampleCount
        epochsTrained++
    }

    @JvmOverloads internal fun exportTrainingState(shareDataset: Boolean = false): NeuroTrainingState {
        val data = trainingData()
        return NeuroTrainingState(topology.copyOf(), Array(layers.size) { layers[it].weights.copyOf() },
            Array(layers.size) { layers[it].biases.copyOf() }, Array(layers.size) { layers[it].weightVelocity.copyOf() },
            Array(layers.size) { layers[it].biasVelocity.copyOf() },
            if (shareDataset) data.inputs else data.inputs.copyOf(),
            if (shareDataset) data.targets else data.targets.copyOf(), sharedDataset = shareDataset)
    }

    @Synchronized internal fun acquireTraining(owner: Any, sessionId: String? = null) {
        check(trainingOwner == null) {
            logOwnershipRejection("acquire", sessionId)
            "This model already has a training session. Close it before opening another."
        }
        trainingOwner = owner
        trainingSessionLogId = sessionId
    }

    @Synchronized internal fun releaseTraining(owner: Any) {
        check(trainingOwner === owner) {
            logOwnershipRejection("release")
            "Training session does not own this model."
        }
        trainingOwner = null
        trainingSessionLogId = null
    }

    @Synchronized internal fun <T> withTraining(owner: Any, action: () -> T): T {
        check(trainingOwner === owner) {
            logOwnershipRejection("use")
            "Training session does not own this model."
        }
        ownerAccess = true
        try { return action() } finally { ownerAccess = false }
    }

    private fun checkTrainingAccess() {
        check(trainingOwner == null || ownerAccess && Thread.holdsLock(this)) {
            logOwnershipRejection("mutate")
            "This model belongs to a training session. Use or close that session before modifying it."
        }
    }

    private fun logOwnershipRejection(operation: String, requestedSession: String? = null) {
        NeuroLog.warn("training", "training.ownership.rejected", null, "model" to logId,
            "session" to trainingSessionLogId, "requestedSession" to requestedSession, "operation" to operation)
    }

    internal fun deviceTrainingOrder(): IntArray = reserveTrainingOrders(1)[0]

    internal fun reserveTrainingOrders(epochs: Int): Array<IntArray> {
        checkTrainingAccess()
        requireTrainingSamples()
        return trainingShuffle.reserve(trainingSamples.size, epochs)
    }

    internal fun recordTrainingError(value: Double) {
        checkTrainingAccess()
        lastTrainingError = value
    }

    internal fun validateTrainingState(state: NeuroTrainingState) {
        require(state.topology.contentEquals(topology)) { "Training state topology does not match model" }
        val buffers = arrayOf(state.weights, state.biases, state.weightVelocity, state.biasVelocity)
        require(buffers.all { it.size == layers.size }) { "Training state layer count does not match model" }
        for (index in layers.indices) {
            require(state.weights[index].size == layers[index].weights.size &&
                state.weightVelocity[index].size == layers[index].weightVelocity.size &&
                state.biases[index].size == layers[index].biases.size &&
                state.biasVelocity[index].size == layers[index].biasVelocity.size) { "Training state buffer size does not match model" }
        }
        for (values in buffers) for (buffer in values) {
            check(buffer.all { it.isFinite() }) { "Training produced non-finite parameters." }
        }
    }

    internal fun commitDeviceEpoch(state: NeuroTrainingState): Double = commitTrainingChunk(state, 1)

    /** Publishes native precision conversion for a scored, untrained checkpoint without consuming shuffle state. */
    @Synchronized internal fun publishInitialTrainingState(state: NeuroTrainingState) {
        checkTrainingAccess()
        check(epochsTrained == 0L && samplesSeen == 0L) { "Initial state publication requires an untrained model." }
        validateTrainingState(state)
        for (index in layers.indices) {
            state.weights[index].copyInto(layers[index].weights)
            state.biases[index].copyInto(layers[index].biases)
            state.weightVelocity[index].copyInto(layers[index].weightVelocity)
            state.biasVelocity[index].copyInto(layers[index].biasVelocity)
        }
        lastTrainingError = Double.NaN
    }

    @JvmOverloads internal fun commitTrainingChunk(state: NeuroTrainingState, epochs: Int, evaluateError: Boolean = true): Double {
        checkTrainingAccess()
        require(epochs in 1..64)
        validateTrainingState(state)
        // Validate the complete checkpoint before publishing parameters, momentum or progress.
        trainingShuffle.commit(epochs)
        for (index in layers.indices) {
            state.weights[index].copyInto(layers[index].weights)
            state.biases[index].copyInto(layers[index].biases)
            state.weightVelocity[index].copyInto(layers[index].weightVelocity)
            state.biasVelocity[index].copyInto(layers[index].biasVelocity)
        }
        epochsTrained += epochs
        samplesSeen += trainingSamples.size.toLong() * epochs
        lastTrainingError = if (evaluateError) trainingError() else Double.NaN
        return lastTrainingError
    }

    /** Search owns scoring boundaries; TensorFlow owns the same epoch arithmetic used by public training. */
    internal fun advanceCpuSearchEpoch(batchSize: Int) {
        checkTrainingAccess()
        requireTrainingSamples()
        require(batchSize > 0)
        newCpuKernel().use { trainTensorFlowEpoch(it, batchSize, batchSize == 1, false) }
    }

    @Synchronized fun trainEpoch(): Double = logDirectTraining("trainEpoch", 1) {
        checkTrainingAccess()
        requireTrainingSamples()
        newCpuKernel().use { trainTensorFlowEpoch(it, 1, true, true) }
    }

    @Synchronized fun train(epochs: Int): Unit = logDirectTraining("train", epochs) {
        checkTrainingAccess()
        require(epochs >= 0) { "epochs must be >= 0" }
        if (epochs == 0) return@logDirectTraining
        requireTrainingSamples()
        newCpuKernel().use { kernel ->
            repeat(epochs) { epoch -> trainTensorFlowEpoch(kernel, 1, true, epoch == epochs - 1) }
        }
    }

    @Synchronized @JvmOverloads fun trainUntil(targetError: Double, maxEpochs: Int, checkEvery: Int = 1): TrainingResult =
        logDirectTraining("trainUntil", maxEpochs, details = mapOf("targetError" to targetError, "checkEvery" to checkEvery)) {
        checkTrainingAccess()
        require(targetError >= 0.0 && targetError.isFinite()) { "targetError must be finite and >= 0" }
        require(maxEpochs >= 0) { "maxEpochs must be >= 0" }
        require(checkEvery > 0) { "checkEvery must be > 0" }
        requireTrainingSamples()
        var error = trainingError()
        if (error <= targetError) {
            lastTrainingError = error
            return@logDirectTraining TrainingResult(0, error, true)
        }
        val event = NeuroJfr.trainingRun(targetError, maxEpochs)
        var epochs = 0
        if (maxEpochs > 0) newCpuKernel().use { kernel ->
            while (epochs < maxEpochs && error > targetError) {
                val score = (epochs + 1) % checkEvery == 0 || epochs + 1 == maxEpochs
                val current = trainTensorFlowEpoch(kernel, 1, true, score)
                epochs++
                if (score) error = current
            }
        }
        val converged = error <= targetError
        NeuroJfr.commitTrainingRun(event, epochs, error, converged)
        TrainingResult(epochs, error, converged)
    }

    @Synchronized @JvmOverloads fun trainMiniBatch(epochs: Int, batchSize: Int, parallelism: Int = 1): Unit =
        logDirectTraining("trainMiniBatch", epochs, batchSize, parallelism) {
        checkTrainingAccess()
        require(epochs >= 0) { "epochs must be >= 0" }
        require(batchSize > 0) { "batchSize must be > 0" }
        require(parallelism > 0) { "parallelism must be > 0" }
        if (epochs == 0) return@logDirectTraining
        requireTrainingSamples()
        NeuroCpuBatchTrainer.train(this, trainingData(), epochs, batchSize, parallelism)
        lastTrainingError = trainingError()
    }

    /** TensorFlow selects the requested device and precision; parallelism remains a compatibility hint. */
    @Synchronized @JvmOverloads
    fun trainMiniBatch(epochs: Int, batchSize: Int, parallelism: Int, backend: BatchBackend,
                       precision: TrainingPrecision = TrainingPrecision.FP64) {
        checkTrainingAccess()
        require(epochs >= 0) { "epochs must be >= 0" }
        require(batchSize > 0) { "batchSize must be > 0" }
        require(parallelism > 0) { "parallelism must be > 0" }
        if (epochs == 0) return
        requireTrainingSamples()
        val effectiveBatch = minOf(batchSize, trainingSampleCount())
        val requested = when (backend) {
            BatchBackend.CPU -> TrainingBackend.CPU
            BatchBackend.GPU -> TrainingBackend.GPU
            BatchBackend.AUTO -> TrainingBackend.AUTO
        }
        TensorFlowMath.trainingKernel(exportTrainingState(shareDataset = true), hyperParameters,
            precision, requested).use { kernel ->
            NeuroLog.debug("training", "training.batch.resolved") { mapOf("model" to logId,
                "session" to trainingSessionLogId, "requestedBackend" to backend, "backend" to kernel.info.backend,
                "requestedPrecision" to precision, "precision" to kernel.info.precision,
                "requestedBatchSize" to batchSize, "batchSize" to effectiveBatch, "parallelism" to parallelism) }
            logDirectTraining("trainMiniBatch", epochs, effectiveBatch, parallelism, kernel.info) {
                repeat(epochs) { epoch -> trainTensorFlowEpoch(kernel, effectiveBatch, false, epoch == epochs - 1) }
            }
        }
    }

    private fun <T> logDirectTraining(operation: String, epochs: Int, batchSize: Int = 1,
                                      parallelism: Int = 1,
                                      info: TrainingDeviceInfo = TensorFlowMath.info(TrainingBackend.CPU,
                                          TrainingPrecision.FP64, sigmoid = hyperParameters.sigmoidMode),
                                      details: Map<String, Any?> = emptyMap(),
                                      action: () -> T): T =
        if (trainingOwner != null) action() else
            loggedTraining(this, null, info, operation, epochs, batchSize, parallelism, details, action = action)

    fun trainingError(): Double = if (trainingSamples.isEmpty()) Double.NaN else error(trainingData())
    fun testError(): Double = if (testSamples.isEmpty()) Double.NaN else error(testData())

    private fun newCpuKernel() = TensorFlowMath.trainingKernel(exportTrainingState(shareDataset = true),
        hyperParameters, TrainingPrecision.FP64, TrainingBackend.CPU)

    /** Only validated TensorFlow output advances the recoverable host checkpoint and shuffle cursor. */
    internal fun trainTensorFlowEpoch(kernel: SmallTrainingKernel, batchSize: Int, online: Boolean,
                                     evaluateError: Boolean): Double {
        checkTrainingAccess()
        require(batchSize > 0) { "batchSize must be > 0" }
        val orders = reserveTrainingOrders(1)
        val started = System.nanoTime()
        val event = NeuroJfr.trainingEpoch(epochsTrained + 1, trainingSampleCount())
        val state = kernel.train(orders, batchSize, online)
        val error = commitTrainingChunk(state, 1, evaluateError)
        NeuroJfr.commitTrainingEpoch(event, error)
        NeuroLog.debug("training", "training.tensorflow.epoch.completed") { mapOf("model" to logId,
            "session" to trainingSessionLogId, "backend" to kernel.info.backend, "precision" to kernel.info.precision,
            "batchSize" to batchSize, "epoch" to epochsTrained, "samplesSeen" to samplesSeen,
            "rmse" to error, "durationMs" to ((System.nanoTime() - started) / 1_000_000.0)) }
        return error
    }

    private fun predictTensorFlow(inputs: DoubleArray, batchSize: Int): DoubleArray {
        val count = batchSize * topology[0]
        val packed = if (inputs.size == count) inputs else inputs.copyOf(count)
        return TensorFlowMath.predict(topology, Array(layers.size) { layers[it].weights },
            Array(layers.size) { layers[it].biases }, hyperParameters, packed, batchSize)
    }

    private fun error(data: PackedDataset): Double = TensorFlowMath.error(topology,
        Array(layers.size) { layers[it].weights }, Array(layers.size) { layers[it].biases },
        hyperParameters, data.inputs, data.targets)

    private fun trainingData(): PackedDataset = packedTraining ?: PackedDataset(trainingSamples, topology[0], topology.last()).also { packedTraining = it }
    private fun testData(): PackedDataset = packedTests ?: PackedDataset(testSamples, topology[0], topology.last()).also { packedTests = it }
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
        internal fun activate(value: Double, mode: SigmoidMode): Double = TensorFlowMath.sigmoid(value, mode)
    }
}
