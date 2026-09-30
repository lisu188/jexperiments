package com.lis.neuro

import java.util.Collections
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger

/** Exercises backend routing and resource lifetimes without representing hardware acceptance. */
internal class RecordingTrainingSessions {
    val requested = Collections.synchronizedList(ArrayList<TrainingBackend>())
    val configurations = Collections.synchronizedList(ArrayList<Triple<TrainingBackend, Neuro.TrainingPrecision, Int>>())
    val miniBatches = Collections.synchronizedList(ArrayList<Int>())
    val opened = AtomicInteger()
    val closed = AtomicInteger()
    val epochCalls = AtomicInteger()
    @Volatile var unavailable = false
    @Volatile var identity = "fixture-device"
    @Volatile var failEpoch = false
    @Volatile var failClose = false
    @Volatile var failAfterCommit = false

    fun open(model: Neuro, backend: TrainingBackend, precision: Neuro.TrainingPrecision, batchSize: Int): NeuroTrainingSession {
        requested += backend
        configurations += Triple(backend, precision, batchSize)
        check(backend !in setOf(TrainingBackend.CUDA, TrainingBackend.CUBLAS) || !unavailable) { "CUDA fixture unavailable" }
        val delegate = model.newTrainingSession(TrainingBackend.CPU)
        opened.incrementAndGet()
        val device = if (backend in setOf(TrainingBackend.CPU, TrainingBackend.AUTO)) delegate.info else
            TrainingDeviceInfo(backend, "$backend test fixture", identity, precision.name, kernelVersion = "fixture-v1")
        return object : NeuroTrainingSession by delegate {
            private val released = AtomicBoolean()
            override val info = device
            override fun trainEpoch(): Double {
                epochCalls.incrementAndGet()
                check(!failEpoch) { "CUDA fixture epoch failed" }
                val error = delegate.trainEpoch()
                check(!failAfterCommit) { "CUDA fixture cleanup after committed epoch failed" }
                return error
            }
            override fun trainMiniBatch(epochs: Int, batchSize: Int, parallelism: Int) {
                epochCalls.addAndGet(epochs)
                miniBatches += batchSize
                check(!failEpoch) { "CUDA fixture epoch failed" }
                delegate.trainMiniBatch(epochs, batchSize, parallelism)
                check(!failAfterCommit) { "CUDA fixture cleanup after committed epoch failed" }
            }
            override fun close() {
                if (released.compareAndSet(false, true)) {
                    try { delegate.close() } finally { closed.incrementAndGet() }
                    check(!failClose) { "CUDA fixture cleanup failed" }
                }
            }
        }
    }
}
