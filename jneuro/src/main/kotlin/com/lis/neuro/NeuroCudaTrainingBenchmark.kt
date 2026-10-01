package com.lis.neuro

import java.nio.file.Path

/** Small repeatable comparison by default; opt into the larger matrix explicitly. */
object NeuroCudaTrainingBenchmark {
    @JvmStatic fun main(args: Array<String>) = NeuroLog.application("NeuroCudaTrainingBenchmark") {
        if (args.contentEquals(arrayOf("--help")) || args.contentEquals(arrayOf("-h"))) {
            println(NeuroCudaBenchmarkConfig.usage())
            return@application
        }
        val config = NeuroCudaBenchmarkConfig.parse(args)
        println("Starting CPU/GPU benchmark: profile=${config.profile}, epochs=${config.epochs}, " +
            "warmups=${config.warmups}, repeats=${config.repeats}, CPU parallelism=1, " +
            "batches=${config.batches.joinToString(",")}, backends=${config.backends.joinToString(",")}, " +
            "topology=${config.topology?.joinToString("x") ?: "profile-default"}, samples=${config.sampleCount}")
        try {
            val service = if (config.retainedDevice) NeuroTrainingDeviceService() else null
            val report = try { NeuroCudaBenchmarkHarness.run(config, sessionFactory = { model, engine, batch -> engine.open(model, batch, service) }, onWorkloadCompleted = { workload, completed, total ->
                println("Completed $completed/$total: ${workload.name}, topology=${workload.topology.joinToString("x")}, " +
                    "samples=${workload.samples}, verified ${config.repeats * config.backends.size} measured rounds.")
            }) } finally { service?.close() }
            NeuroCudaBenchmarkReports.write(report, config.output)
            println(NeuroCudaBenchmarkReports.console(report))
            println("Reports: ${config.output.toAbsolutePath()}.json and .csv")
        } catch (failure: NeuroCudaBenchmarkFailure) {
            if (NeuroCudaBenchmarkReports.writeFailure(failure, config.output)) {
                System.err.println("Failure reports: ${config.output.toAbsolutePath()}.json and .csv")
            }
            throw failure
        }
    }
}

internal enum class NeuroCudaBenchmarkBackend(val backend: TrainingBackend, val precision: Neuro.TrainingPrecision,
                                              val engine: TrainingEngine = TrainingEngine.REFERENCE, val vectorBits: Int = 0) {
    CPU(TrainingBackend.CPU, Neuro.TrainingPrecision.FP64),
    CUDA(TrainingBackend.CUDA, Neuro.TrainingPrecision.FP64),
    CUBLAS_FP64(TrainingBackend.CUBLAS, Neuro.TrainingPrecision.FP64),
    CUBLAS_FP32(TrainingBackend.CUBLAS, Neuro.TrainingPrecision.FP32),
    CPU_LEGACY(TrainingBackend.CPU, Neuro.TrainingPrecision.FP64),
    SMALL_SCALAR_FP64(TrainingBackend.CPU, Neuro.TrainingPrecision.FP64, TrainingEngine.SMALL),
    SMALL_SCALAR_FP32(TrainingBackend.CPU, Neuro.TrainingPrecision.FP32, TrainingEngine.SMALL),
    SMALL_128_FP64(TrainingBackend.CPU, Neuro.TrainingPrecision.FP64, TrainingEngine.SMALL, 128),
    SMALL_256_FP64(TrainingBackend.CPU, Neuro.TrainingPrecision.FP64, TrainingEngine.SMALL, 256),
    SMALL_128_FP32(TrainingBackend.CPU, Neuro.TrainingPrecision.FP32, TrainingEngine.SMALL, 128),
    SMALL_256_FP32(TrainingBackend.CPU, Neuro.TrainingPrecision.FP32, TrainingEngine.SMALL, 256),
    SMALL_CUDA_FP64(TrainingBackend.CUDA, Neuro.TrainingPrecision.FP64, TrainingEngine.SMALL),
    SMALL_CUDA_FP32(TrainingBackend.CUDA, Neuro.TrainingPrecision.FP32, TrainingEngine.SMALL),
    NATIVE_FP64(TrainingBackend.CPU, Neuro.TrainingPrecision.FP64, TrainingEngine.SMALL);

    fun open(model: Neuro, batch: Int, service: NeuroTrainingDeviceService? = null): NeuroTrainingSession =
        if (service != null) service.openSession(model, backend, precision, batch, engine)
        else model.newTrainingSession(backend, precision, batch, engine)

}

internal enum class NeuroBenchmarkMode { MINIBATCH, EPOCH, CHUNK }

internal data class NeuroCudaBenchmarkConfig(
    val profile: String = "smoke", val epochs: Int = 2, val warmups: Int = 1, val repeats: Int = 3,
    val batches: List<Int> = listOf(16, 64), val backends: List<NeuroCudaBenchmarkBackend> = listOf(
        NeuroCudaBenchmarkBackend.CPU, NeuroCudaBenchmarkBackend.CUDA, NeuroCudaBenchmarkBackend.CUBLAS_FP64, NeuroCudaBenchmarkBackend.CUBLAS_FP32),
    val output: Path = Path.of("build", "reports", "cuda-benchmark", "benchmark"),
    val topology: List<Int>? = null, val samples: Int? = null,
    val mode: NeuroBenchmarkMode = NeuroBenchmarkMode.MINIBATCH,
    val sigmoid: Neuro.SigmoidMode = Neuro.SigmoidMode.EXACT,
    val retainedDevice: Boolean = false
) {
    val sampleCount: Int get() = samples ?: if (profile == "smoke") 128 else 1024
    init {
        require(profile in listOf("smoke", "matrix")) { "profile must be smoke or matrix" }
        if (topology != null) {
            require(topology.size in 2..16 && topology.all { it in 1..2048 }) {
                "topology must contain 2..16 layer widths in 1..2048, including input and output"
            }
            require(topology.zipWithNext { inputs, outputs -> (inputs.toLong() + 1) * outputs }.sum() <= 2_000_000) {
                "benchmark topology must contain at most 2000000 parameters"
            }
        }
        require(samples == null || samples in 1..8192) { "samples must be 1..8192" }
        require(epochs in 1..1000 && warmups in 0..100 && repeats in 1..100) { "epochs must be 1..1000, warmups 0..100, repeats 1..100" }
        require(batches.isNotEmpty() && batches.all { it in 1..4096 } && batches.distinct().size == batches.size) {
            "batches must contain distinct sizes in 1..4096"
        }
        require(backends.isNotEmpty() && backends.distinct().size == backends.size) { "backends must be nonempty and distinct" }
        require(mode == NeuroBenchmarkMode.MINIBATCH || NeuroCudaBenchmarkBackend.CPU_LEGACY !in backends) {
            "CPU_LEGACY compares bulk mini-batch training; select --mode MINIBATCH."
        }
        require(output.fileName != null && output.toString().isNotBlank()) { "output must be a report file prefix" }
    }
    companion object {
        fun usage() = """
            Usage: cpuGpuBenchmark [options]
              --profile smoke|matrix       Workload presets; default smoke
              --topology 2,8,8,8,1         Override with input, hidden layer(s), output widths
              --samples N                  Dataset size, 1..8192; default smoke=128, matrix=1024
              --epochs N                   1..1000; default 2
              --warmups N                  0..100; default 1, excluded from measurements
              --repeats N                  1..100; default 3
              --batches 16,64              Distinct sizes in 1..4096
              --backends CPU,CUDA,CUBLAS_FP64,CUBLAS_FP32
                         CPU_LEGACY,SMALL_SCALAR_FP64,SMALL_SCALAR_FP32,SMALL_128_FP64,SMALL_256_FP64,
                         SMALL_128_FP32,SMALL_256_FP32,SMALL_CUDA_FP64,SMALL_CUDA_FP32
                         Legacy backend names are compatibility aliases; every entry uses TensorFlow.
              --mode MINIBATCH|EPOCH|CHUNK Publication per call, per UI epoch, or bounded chunks
              --sigmoid EXACT|FAST         Compare matching arithmetic; default EXACT
              --retained-device true|false Reuse the TensorFlow training service between sessions
              --output PATH                Prefix for .json and .csv reports
              --smoke                      Alias for --profile smoke
              --help, -h                   Print this help without training
            Custom topology: 2..16 layers, widths 1..2048, at most 2000000 parameters.
            GPU selection is explicit: missing TensorFlow GPU support fails without CPU fallback.
        """.trimIndent()

        fun parse(args: Array<String>): NeuroCudaBenchmarkConfig {
            var result = NeuroCudaBenchmarkConfig()
            var index = 0
            while (index < args.size) {
                val option = args[index++]
                if (option == "--smoke") { result = result.copy(profile = "smoke"); continue }
                require(index < args.size) { "Missing value for $option" }
                val value = args[index++]
                result = when (option) {
                    "--profile" -> result.copy(profile = value)
                    "--topology" -> result.copy(topology = value.split(',').map { it.trim().toInt() })
                    "--samples" -> result.copy(samples = value.toInt())
                    "--epochs" -> result.copy(epochs = value.toInt())
                    "--warmups" -> result.copy(warmups = value.toInt())
                    "--repeats" -> result.copy(repeats = value.toInt())
                    "--batches" -> result.copy(batches = value.split(',').map { it.trim().toInt() })
                    "--backends" -> result.copy(backends = value.split(',').map { NeuroCudaBenchmarkBackend.valueOf(it.trim().uppercase(java.util.Locale.ROOT)) })
                    "--output" -> result.copy(output = Path.of(value))
                    "--mode" -> result.copy(mode = NeuroBenchmarkMode.valueOf(value.uppercase(java.util.Locale.ROOT)))
                    "--sigmoid" -> result.copy(sigmoid = Neuro.SigmoidMode.valueOf(value.uppercase(java.util.Locale.ROOT)))
                    "--retained-device" -> result.copy(retainedDevice = value.toBooleanStrict())
                    else -> throw IllegalArgumentException("Unknown option $option; use --help to list benchmark options")
                }
            }
            return result
        }
    }
}
