package com.lis.neuro

import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.CyclicBarrier
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

object EpochBenchmark {
    private val hp = Neuro.HyperParameters(0.6, 0.2, 1.0, 42)
    private const val warm = 5
    private const val measured = 20
    @Volatile private var blackhole = 0.0

    @JvmStatic fun main(args: Array<String>) {
        val directory = Path.of(args[0]); Files.createDirectories(directory)
        val samples = NeuroLearningSets.create(NeuroLearningSets.Kind.SPIRAL, 42)
        check(samples.size == 220)
        for (shape in listOf(intArrayOf(2, 6, 1), intArrayOf(2, 8, 8, 8, 1))) {
            val label = shape.joinToString("x")
            for (batch in listOf(1, 16, 32)) for (api in listOf("epoch", "chunk")) repeat(3) { round ->
                val network = Neuro(shape, hp).also { NeuroLearningSets.addTo(it, samples) }
                val opened = System.nanoTime()
                val session = network.newTrainingSession(TrainingBackend.CPU, Neuro.TrainingPrecision.FP64, batch)
                val openMs = millis(opened)
                val elapsed = session.use { retained ->
                    check(retained.trainChunk(request(warm)).committedEpochs == warm)
                    val started = System.nanoTime()
                    if (api == "epoch") repeat(measured) { blackhole = retained.trainEpoch() }
                    else {
                        val result = retained.trainChunk(request(measured))
                        check(result.committedEpochs == measured && result.rmse.isFinite())
                        blackhole = result.rmse
                    }
                    millis(started)
                }
                save(directory.resolve("training-$label-b$batch-$api-r$round.json"), network)
                println("{\"kind\":\"training\",\"shape\":\"$label\",\"api\":\"$api\",\"batch\":$batch,\"round\":$round,\"samples\":220,\"warmupEpochs\":$warm,\"epochs\":$measured,\"openMs\":$openMs,\"elapsedMs\":$elapsed,\"epochMs\":${elapsed / measured},\"rmse\":$blackhole}")
            }
            diagnostics(shape, directory)
        }
        concurrentSearch(samples, directory)
        TensorFlowMath.clearInferenceCache()
    }

    private fun request(epochs: Int) = TrainingChunkRequest(epochs, checkEvery = epochs, maxNanos = Long.MAX_VALUE)
    private fun millis(started: Long) = (System.nanoTime() - started) / 1_000_000.0

    private fun diagnostics(shape: IntArray, directory: Path) {
        val hidden = shape.drop(1).dropLast(1).joinToString(",")
        val label = shape.joinToString("x")
        val network = Neuro(shape, hp)
        val started = System.nanoTime()
        repeat(1000) { blackhole = NeuroXorDiagnostics.capture(network, 0, 0.5).weight(0, 0, 0) }
        val snapshotMs = millis(started) / 1000
        NeuroStudio(StudioConfig(hidden = hidden, dataset = NeuroLearningSets.Kind.SPIRAL, targetError = 0.0)).use { studio ->
            repeat(3) { studio.step(1); check(studio.advance(1) == 1); studio.frame() }
            val frameTimes = DoubleArray(20) {
                studio.step(1); check(studio.advance(1) == 1)
                val frameStarted = System.nanoTime()
                val frame = studio.frame()
                blackhole = frame.image.getRGB(0, 0).toDouble()
                millis(frameStarted)
            }
            val cachedStarted = System.nanoTime()
            repeat(1000) { blackhole = studio.frame().diagnostics.error() }
            val cachedMs = millis(cachedStarted) / 1000
            val median = frameTimes.sorted()[frameTimes.size / 2]
            println("{\"kind\":\"diagnostics\",\"shape\":\"$label\",\"snapshotMs\":$snapshotMs,\"freshFrameMedianMs\":$median,\"freshFrameTimesMs\":${frameTimes.joinToString(prefix = "[", postfix = "]")},\"cachedFrameMs\":$cachedMs}")
            Files.writeString(directory.resolve("frame-$label.txt"), "epochs=${studio.frame().diagnostics.epoch()}\nrmse=${studio.frame().diagnostics.error()}\n")
        }
    }

    private fun concurrentSearch(samples: List<NeuroLearningSets.Sample>, directory: Path) {
        val data = ArchitectureSearchData.split(samples, 0.2, 42, "Spiral")
        check(data.training.size == 176 && data.validation.size == 44)
        val seeds = longArrayOf(1, 42, 123, 999, 2026)
        val pool = Executors.newFixedThreadPool(seeds.size)
        try {
            repeat(3) { round ->
                val ready = CyclicBarrier(seeds.size)
                val futures = seeds.map { seed -> pool.submit<Double> {
                    val network = data.newNetwork(NetworkArchitecture(listOf(6)), hp, seed)
                    val elapsed = network.newTrainingSession(TrainingBackend.CPU, Neuro.TrainingPrecision.FP64, 1).use { session ->
                        check(session.trainChunk(request(warm)).committedEpochs == warm)
                        ready.await(30, TimeUnit.SECONDS)
                        val started = System.nanoTime()
                        check(advanceTrainingForSearch(session, request(measured)).committedEpochs == measured)
                        millis(started)
                    }
                    save(directory.resolve("concurrent-seed$seed-r$round.json"), network)
                    elapsed
                } }
                val times = futures.map { it.get(60, TimeUnit.SECONDS) }
                val wall = times.max()
                println("{\"kind\":\"concurrent-search\",\"round\":$round,\"models\":5,\"samplesPerModel\":176,\"epochsPerModel\":$measured,\"batch\":1,\"elapsedMs\":$wall,\"totalEpochsPerSecond\":${measured * seeds.size * 1000.0 / wall},\"perModelTimesMs\":${times.joinToString(prefix = "[", postfix = "]")}}")
            }
        } finally { pool.shutdown(); check(pool.awaitTermination(10, TimeUnit.SECONDS)) }
    }

    private fun save(file: Path, network: Neuro) {
        val state = network.exportTrainingState()
        fun rows(values: Array<DoubleArray>) = values.joinToString(prefix = "[", postfix = "]") { it.joinToString(prefix = "[", postfix = "]") }
        val statistics = network.statistics()
        val order = network.reserveTrainingOrders(1).single()
        val json = "{\"topology\":${state.topology.joinToString(prefix = "[", postfix = "]")},\"weights\":${rows(state.weights)},\"biases\":${rows(state.biases)},\"weightVelocity\":${rows(state.weightVelocity)},\"biasVelocity\":${rows(state.biasVelocity)},\"epochs\":${statistics.epochsTrained},\"samplesSeen\":${statistics.samplesSeen},\"rmse\":${network.trainingError()},\"nextOrder\":${order.joinToString(prefix = "[", postfix = "]")}}"
        Files.writeString(file, json)
    }
}
