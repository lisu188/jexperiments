package com.lis.neuro

import java.util.concurrent.ExecutionException
import java.util.concurrent.ForkJoinPool
import java.util.concurrent.Future

/** Compatibility entrypoint: TensorFlow owns matrix arithmetic and its intra-operation workers. */
internal object NeuroCpuBatchTrainer {
    fun train(network: Neuro, data: Neuro.PackedDataset, epochs: Int, batchSize: Int, parallelism: Int) {
        require(epochs >= 0 && batchSize > 0 && parallelism > 0)
        if (epochs == 0) return
        require(data.size == network.trainingSampleCount()) { "Dataset does not match the model" }
        TensorFlowMath.trainingKernel(network.exportTrainingState(shareDataset = true), network.hyperParameters(),
            Neuro.TrainingPrecision.FP64, TrainingBackend.CPU).use { kernel ->
            repeat(epochs) { network.trainTensorFlowEpoch(kernel, batchSize, false, false) }
        }
    }

    internal fun parallelFor(size: Int, parallelism: Int, pool: ForkJoinPool?, action: (Int, Int) -> Unit) {
        if (size == 0) return
        if (pool == null || parallelism <= 1 || size < 256) {
            action(0, size)
            return
        }
        val workers = minOf(parallelism, size)
        val futures = Array<Future<*>>(workers) { worker ->
            val from = worker * size / workers
            val to = (worker + 1) * size / workers
            pool.submit { action(from, to) }
        }
        var failure: IllegalStateException? = null
        var interrupted = false
        fun record(problem: IllegalStateException) {
            if (failure == null) failure = problem else failure.addSuppressed(problem)
        }
        // Drain every inference slice before releasing the caller's output arrays,
        // including when another worker fails or the caller is interrupted.
        for (future in futures) {
            while (true) {
                try {
                    future.get()
                    break
                } catch (exception: InterruptedException) {
                    interrupted = true // get() cleared the flag; restore it only after draining.
                    record(IllegalStateException("parallel TensorFlow inference interrupted", exception))
                } catch (exception: ExecutionException) {
                    record(IllegalStateException("parallel TensorFlow inference failed", exception.cause))
                    break
                }
            }
        }
        if (interrupted) Thread.currentThread().interrupt()
        failure?.let { throw it }
    }
}
