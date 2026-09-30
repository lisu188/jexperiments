package com.lis.neuro

object NeuroCuda {
    data class Status(val available: Boolean, val description: String, val reason: String = "")
    private const val AUTO_FLOP_THRESHOLD = 2_000_000L
    private val detected: Status by lazy { detect() }

    @JvmStatic fun status(): Status = detected
    @JvmStatic fun isAvailable(): Boolean = detected.available
    @JvmStatic fun deviceDescription(): String = detected.description

    internal fun resolveBackend(requested: Neuro.BatchBackend, topology: IntArray, batchSize: Int): Neuro.BatchBackend =
        resolveBackend(requested, topology, batchSize,
            requested == Neuro.BatchBackend.AUTO &&
                NeuroCudaPlan(topology, batchSize).estimatedFlops >= AUTO_FLOP_THRESHOLD && isAvailable())

    internal fun resolveBackend(requested: Neuro.BatchBackend, topology: IntArray, batchSize: Int,
                                available: Boolean): Neuro.BatchBackend = when (requested) {
        Neuro.BatchBackend.CPU -> Neuro.BatchBackend.CPU
        Neuro.BatchBackend.CUDA -> Neuro.BatchBackend.CUDA
        Neuro.BatchBackend.AUTO -> if (available && NeuroCudaPlan(topology, batchSize).estimatedFlops >= AUTO_FLOP_THRESHOLD) {
            Neuro.BatchBackend.CUDA
        } else {
            Neuro.BatchBackend.CPU
        }
    }

    internal fun detect(loader: NeuroLibraryLoader = NeuroNativeLibrary::open): Status {
        NeuroLog.debug("cuda", "capability.detection.started") { emptyMap() }
        val result = detectLibraries(loader)
        if (result.available) NeuroLog.info("cuda", "capability.detected", "device" to result.description)
        else NeuroLog.debug("cuda", "capability.unavailable") { mapOf("reason" to result.reason) }
        return result
    }

    private fun detectLibraries(loader: NeuroLibraryLoader): Status {
        val runtime = NeuroCudaRuntime.tryCreate(loader) ?: return Status(false, "CUDA unavailable", "CUDA Runtime library not found")
        runtime.use {
            return try {
                val count = runtime.deviceCount()
                if (count <= 0) return Status(false, "CUDA unavailable", "No CUDA-capable devices")
                runtime.setDevice(0)
                val cublas = NeuroCublas.tryCreate(loader)
                    ?: return Status(false, "CUDA unavailable", "cuBLAS library not found")
                cublas.use {
                    val nvrtc = NeuroNvrtc.tryCreate(loader)
                        ?: return Status(false, "CUDA unavailable", "NVRTC library not found")
                    nvrtc.use {
                        val driver = NeuroCudaDriver.tryCreate(loader)
                            ?: return Status(false, "CUDA unavailable", "CUDA Driver API not found")
                        driver.use {
                            val capability = runtime.computeCapability(0)
                            Status(true, driver.deviceName(0) + " · compute " + capability.first + "." + capability.second)
                        }
                    }
                }
            } catch (exception: RuntimeException) {
                NeuroLog.debug("cuda", "capability.detection.failed") { mapOf("reason" to exception.message) }
                Status(false, "CUDA unavailable", exception.message ?: exception.javaClass.simpleName)
            }
        }
    }
}
