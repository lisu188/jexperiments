package com.lis.neuro

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.TreeMap

/** CPU arithmetic stand-in for orchestration tests, not evidence of CUDA execution. */
internal class SearchCudaTestDriver(identity: String = "search-test-gpu") : CudaDriver {
    override val info = TrainingDeviceInfo(TrainingBackend.CUDA, "search test GPU", identity, kernelVersion = "test-ptx")
    val memory = TreeMap<Long, ByteArray>()
    val launches = ArrayList<List<IntArray>>()
    val kernelNames = ArrayList<String>()
    val threads = HashSet<String>()
    val nonfiniteSlots = HashSet<Int>()
    val corruptSlots = HashSet<Int>()
    var beforeLaunch: () -> Unit = {}
    var failure = ""
    var failAllocation = -1
    var allocations = 0
    var uploads = 0
    var downloads = 0
    var closes = 0
    var frees = 0
    var freeBytes = 512L * 1024 * 1024
    private var next = 4096L

    private fun observe() { threads += Thread.currentThread().name }
    private fun fail(operation: String) { if (failure == operation) error("fixture $operation") }
    override fun availableMemory(): Long { observe(); return freeBytes }
    override fun allocate(bytes: Long): Long {
        observe(); allocations++
        if (allocations == failAllocation) error("fixture allocation")
        fail("allocate")
        val pointer = next
        next += bytes + 4096
        memory[pointer] = ByteArray(bytes.toInt())
        return pointer
    }
    override fun free(pointer: Long) {
        observe(); frees++
        check(memory.remove(pointer) != null)
        fail("free")
    }
    private fun buffer(pointer: Long): ByteBuffer {
        val entry = checkNotNull(memory.floorEntry(pointer))
        val offset = (pointer - entry.key).toInt()
        return ByteBuffer.wrap(entry.value, offset, entry.value.size - offset).slice().order(ByteOrder.nativeOrder())
    }
    override fun upload(pointer: Long, values: DoubleArray) {
        observe(); uploads++; fail("upload")
        val target = buffer(pointer); values.forEach { target.putDouble(it) }
    }
    override fun upload(pointer: Long, values: IntArray) {
        observe(); uploads++; fail("upload")
        val target = buffer(pointer); values.forEach { target.putInt(it) }
    }
    override fun download(pointer: Long, values: DoubleArray) {
        observe(); downloads++
        val source = buffer(pointer)
        values.indices.forEach { index ->
            values[index] = source.double
            if (failure == "download" && index == values.size / 2) error("fixture partial download")
        }
    }
    override fun synchronize() { observe(); fail("synchronize") }
    override fun launch(name: String, workItems: Int, vararg arguments: Any) {
        observe(); beforeLaunch(); fail("launch")
        require(name == "search_train_fp64" || name == "search_train_fp32")
        val precision = if (name.endsWith("fp64")) Neuro.TrainingPrecision.FP64 else Neuro.TrainingPrecision.FP32
        val models = arguments[7] as Int
        val samples = arguments[8] as Int
        val batch = arguments[9] as Int
        check(workItems == models * 128)
        val hp = Neuro.HyperParameters(arguments[10] as Double, arguments[11] as Double, arguments[12] as Double, 0L,
            kernel = Neuro.Kernel.SCALAR, sigmoidMode = Neuro.SigmoidMode.entries[arguments[13] as Int])
        val online = arguments[14] == 1
        val input = buffer(arguments[1] as Long)
        val target = buffer(arguments[2] as Long)
        val inputs = DoubleArray(samples * 2) { input.double }
        val targets = DoubleArray(samples) { target.double }
        val metadata = buffer(arguments[5] as Long)
        val requests = List(models) { IntArray(6) { metadata.int } }
        launches += requests
        kernelNames += name
        for (request in requests) {
            val topologyData = buffer((arguments[4] as Long) + request[1] * 4L)
            val topology = IntArray(request[3] + 1) { topologyData.int }
            val weights = Array(topology.size - 1) { DoubleArray(topology[it] * topology[it + 1]) }
            val biases = Array(weights.size) { DoubleArray(topology[it + 1]) }
            val velocity = Array(weights.size) { DoubleArray(weights[it].size) }
            val biasVelocity = Array(biases.size) { DoubleArray(biases[it].size) }
            val stateData = buffer((arguments[0] as Long) + request[0] * 8L)
            for (layer in weights.indices) { weights[layer].indices.forEach { weights[layer][it] = stateData.double }; biases[layer].indices.forEach { biases[layer][it] = stateData.double } }
            for (layer in weights.indices) { velocity[layer].indices.forEach { velocity[layer][it] = stateData.double }; biasVelocity[layer].indices.forEach { biasVelocity[layer][it] = stateData.double } }
            val state = NeuroTrainingState(topology, weights, biases, velocity, biasVelocity, inputs, targets)
            val orderData = buffer((arguments[3] as Long) + request[2] * 4L)
            val orders = Array(request[4]) { IntArray(samples) { orderData.int } }
            val output = buffer((arguments[6] as Long) + request[5] * 8L)
            val slot = request[0] / (2 * 881)
            if (slot in nonfiniteSlots) { output.putDouble(1.0); continue }
            val trained = SmallCpuTraining(state, hp, precision, 0).use { it.train(orders, batch, online) }
            output.putDouble(0.0)
            stateData.position(0)
            val packed = ArrayList<Double>()
            for (layer in weights.indices) { trained.weights[layer].forEach(packed::add); trained.biases[layer].forEach(packed::add) }
            for (layer in weights.indices) { trained.weightVelocity[layer].forEach(packed::add); trained.biasVelocity[layer].forEach(packed::add) }
            packed.forEachIndexed { index, value ->
                stateData.putDouble(value)
                output.putDouble(if (slot in corruptSlots && index == 0) Double.NaN else value)
            }
        }
    }
    override fun close() { observe(); closes++; fail("close") }
}
