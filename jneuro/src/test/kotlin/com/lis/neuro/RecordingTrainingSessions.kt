package com.lis.neuro

import java.util.Collections
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger

/** Exercises backend routing and resource lifetimes without representing hardware acceptance. */
internal class RecordingTrainingSessions {
    val requested = Collections.synchronizedList(ArrayList<TrainingBackend>())
    val opened = AtomicInteger()
    val closed = AtomicInteger()
    val epochCalls = AtomicInteger()
    @Volatile var unavailable = false
    @Volatile var identity = "fixture-device"
    @Volatile var failEpoch = false
    @Volatile var failClose = false

    fun open(model: Neuro, backend: TrainingBackend): NeuroTrainingSession {
        requested += backend
        check(backend != TrainingBackend.CUDA || !unavailable) { "CUDA fixture unavailable" }
        val delegate = model.newTrainingSession(TrainingBackend.CPU)
        opened.incrementAndGet()
        val device = if (backend == TrainingBackend.CPU) delegate.info else
            TrainingDeviceInfo(backend, "CUDA test fixture", identity, kernelVersion = "fixture-v1")
        return object : NeuroTrainingSession by delegate {
            private val released = AtomicBoolean()
            override val info = device
            override fun trainEpoch(): Double {
                epochCalls.incrementAndGet()
                check(!failEpoch) { "CUDA fixture epoch failed" }
                return delegate.trainEpoch()
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
