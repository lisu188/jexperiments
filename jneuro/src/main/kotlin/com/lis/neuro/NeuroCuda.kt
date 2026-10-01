package com.lis.neuro

/** Legacy GPU capability facade; TensorFlow is the sole numerical runtime. */
object NeuroCuda {
    data class Status(val available: Boolean, val description: String, val reason: String = "")
    private val detected: Status by lazy {
        try {
            if (TensorFlowMath.isGpuAvailable()) Status(true, "TensorFlow GPU")
            else Status(false, "TensorFlow CPU", "TensorFlow did not expose a GPU device")
        } catch (failure: RuntimeException) {
            Status(false, "TensorFlow GPU unavailable", failure.message ?: failure.javaClass.simpleName)
        }
    }
    @JvmStatic fun status(): Status = detected
    @JvmStatic fun isAvailable(): Boolean = detected.available
    @JvmStatic fun deviceDescription(): String = detected.description
    @Suppress("UNUSED_PARAMETER")
    internal fun resolveBackend(requested: Neuro.BatchBackend, topology: IntArray, batchSize: Int,
                                available: Boolean = false): Neuro.BatchBackend =
        if (requested == Neuro.BatchBackend.CUDA) Neuro.BatchBackend.CUDA else Neuro.BatchBackend.CPU
}
