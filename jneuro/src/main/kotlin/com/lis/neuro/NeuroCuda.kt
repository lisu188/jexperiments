package com.lis.neuro

object NeuroCuda {
    data class Status(val available: Boolean, val description: String, val reason: String = "")
    private const val AUTO_FLOP_THRESHOLD = 2_000_000L
    private val detected: Status by lazy { detect() }

    @JvmStatic fun status(): Status = detected
    @JvmStatic fun isAvailable(): Boolean = detected.available
    @JvmStatic fun deviceDescription(): String = detected.description

    internal fun resolveBackend(requested: Neuro.TrainingBackend, topology: IntArray, batchSize: Int): Neuro.TrainingBackend =
        resolveBackend(requested, topology, batchSize, isAvailable())

    internal fun resolveBackend(requested: Neuro.TrainingBackend, topology: IntArray, batchSize: Int,
                                available: Boolean): Neuro.TrainingBackend = when (requested) {
        Neuro.TrainingBackend.CPU -> Neuro.TrainingBackend.CPU
        Neuro.TrainingBackend.CUDA -> Neuro.TrainingBackend.CUDA
        Neuro.TrainingBackend.AUTO -> if (available && NeuroCudaPlan(topology, batchSize).estimatedFlops >= AUTO_FLOP_THRESHOLD) {
            Neuro.TrainingBackend.CUDA
        } else {
            Neuro.TrainingBackend.CPU
        }
    }

    private fun detect(): Status {
        val runtime = NeuroCudaRuntime.tryCreate() ?: return Status(false, "CUDA unavailable", "CUDA Runtime library not found")
        runtime.use {
            return try {
                val count = runtime.deviceCount()
                if (count <= 0) return Status(false, "CUDA unavailable", "No CUDA-capable devices")
                runtime.setDevice(0)
                val cublas = NeuroCublas.tryCreate()
                    ?: return Status(false, "CUDA unavailable", "cuBLAS library not found")
                cublas.use {
                    val nvrtc = NeuroNvrtc.tryCreate()
                        ?: return Status(false, "CUDA unavailable", "NVRTC library not found")
                    nvrtc.use {
                        val driver = NeuroCudaDriver.tryCreate()
                            ?: return Status(false, "CUDA unavailable", "CUDA Driver API not found")
                        driver.use {
                            val capability = runtime.computeCapability(0)
                            Status(true, driver.deviceName(0) + " · compute " + capability.first + "." + capability.second)
                        }
                    }
                }
            } catch (exception: RuntimeException) {
                Status(false, "CUDA unavailable", exception.message ?: exception.javaClass.simpleName)
            }
        }
    }
}
