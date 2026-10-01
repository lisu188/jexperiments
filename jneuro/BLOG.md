# JNeuro: A Kotlin Neural-Network Engine and Learning Studio

## Scope and runtime

JNeuro is a small dense feed-forward neural network implemented directly on primitive arrays. The JVM implementation is Kotlin: the training engine, Vector API kernels, optional native BLAS adapter, CUDA training sessions, diagnostics, desktop application and correctness tests. The benchmark source set also contains small handwritten Java JMH and command-line drivers. GPU arithmetic lives in a CUDA C++ source compiled to packaged PTX; a separate C++ AVX2 experiment tests native CPU training. Generated JMH harness classes remain build outputs.

The migration uses Kotlin 2.4.20, the stable release published on 7 September 2026. Kotlin language/API level 2.4 and warnings-as-errors are configured explicitly. The runtime and toolchain remain JDK 27, while JVM bytecode targets 26, the highest target supported by this Kotlin release. The distinction matters: a class-file target is not a promise that incubating JDK 27 APIs will run on JDK 26.

The repository's Gradle 9.8 wrapper is retained rather than changing unrelated modules. Kotlin's published fully-supported Gradle range for 2.4.20 ends at 9.7; compatibility with the existing wrapper is therefore checked by the actual repository workflows, not assumed from that table.

The desktop uses Swing/Java2D and FlatLaf 3.7.2. FlatLaf supplies the component theme, focus handling, and HiDPI-aware widgets without introducing a second native rendering runtime. The numerical engine does not depend on FlatLaf and does not require a display.

## Construction and compatibility

A model is created with a topology and validated hyperparameters:

~~~kotlin
val network = Neuro(
    intArrayOf(2, 6, 1),
    Neuro.HyperParameters.defaults()
        .withLearningRate(0.6)
        .withMomentum(0.2)
        .withSeed(42)
)
~~~

`HyperParameters`, `TrainingResult`, and `Statistics` are Kotlin data classes annotated with `@JvmRecord`. Java callers retain record-style accessors, while Kotlin callers use properties. Existing constructor forms, fluent parameter helpers, inference methods, and training entrypoints remain available. `@JvmOverloads` preserves the shorter constructor and training overloads, and executable objects expose `@JvmStatic main` methods for the existing Gradle tasks.

~~~kotlin
val result = network.trainUntil(0.05, 10_000)
println(result.epochs)
println(result.error)
println(result.converged)
~~~

Topologies, samples, parameter snapshots, and float models own copies of their mutable input arrays. A caller cannot silently alter the model by editing the arrays originally passed to its constructor or training-sample methods. Kotlin non-null contracts also fail fast for invalid Java calls; reflection-based interoperability tests exercise that boundary.

## Data layout and initialization

Each layer owns flat row-major arrays for weights, biases, and their momentum velocities. For output neuron `o` and input `i`, the weight index is `o * inputs + i`. A primitive `DoubleArray` remains a JVM `double[]`, avoiding boxed numeric collections in the hot path.

Initialization is unchanged from the Java baseline. A deterministic `SplittableRandom` uses the configured seed, and the Xavier-style sampling limit is:

~~~kotlin
val limit = Math.sqrt(6.0 / (inputs + outputs))
for (index in weights.indices) {
    weights[index] = random.nextDouble(-limit, limit)
}
~~~

Biases and momentum velocities start at zero. The shuffle generator uses a reproducible seed derived independently from the initialization generator, so changing the batching of visualization refreshes does not change sample order or training results.

Training samples are defensively copied on insertion, then lazily packed into contiguous input and target buffers. Adding another sample invalidates the packed dataset. Training shuffles a reused `IntArray` of sample indices rather than allocating a list every epoch.

## Forward propagation and activation

A dense layer computes a weighted sum plus bias, then applies a sigmoid. Exact mode retains the original numerically stable positive/negative branches. FAST mode retains the optional range-reduced exponential approximation; it is not silently enabled by migration.

~~~kotlin
sum = Math.fma(source[input], layer.weights[offset + input], sum)
destination[output] = activate(sum * beta, sigmoidMode)
~~~

Scalar paths retain the original operation order and unrolling. Vector paths retain explicit `DoubleVector`/`FloatVector` kernels. AUTO still selects vector operations only at the existing width thresholds, and SCALAR/VECTOR remain selectable for verification and benchmarks.

Input values can be read directly from the caller's array. Inference sessions retain activation workspaces, and `predictInto` writes into caller-provided output storage rather than allocating a result per prediction:

~~~kotlin
val session = network.newInferenceSession()
val output = DoubleArray(1)
session.predictInto(doubleArrayOf(0.25, 0.75), output)
~~~

The convenience `predict` still allocates its returned array. Batch methods accept flat row-major buffers; the migration additionally checks buffer-size products without integer overflow.

## Backpropagation and training

The output delta is `(target - activation) * beta * activation * (1 - activation)`. Hidden deltas propagate the weighted next-layer deltas before multiplying by the hidden sigmoid derivative. Weight and bias momentum updates are preserved, including their accumulation order.

~~~kotlin
network.trainEpoch()
network.train(100)
network.trainUntil(0.05, 10_000, checkEvery = 8)
network.trainMiniBatch(10, batchSize = 16, parallelism = 2)
~~~

Online training updates after every shuffled sample. Fixed-epoch training computes its final reported error after the last epoch; sparse checks in `trainUntil` avoid repeatedly evaluating the entire training set when only periodic convergence checks are needed.

The two/three-argument mini-batch API above retains per-sample forward/backpropagation with averaged batch gradients. The explicit `Neuro.BatchBackend` overload described below adds a matrix implementation: it gathers contiguous activation and target rows, evaluates layers across the batch, propagates dense delta matrices, computes weight gradients as the equivalent of D^T × A, reduces bias gradients, and applies one averaged momentum update. Its single-thread and partitioned CPU paths provide a comparison for cuBLAS while preserving the original overload's numerical behavior.

Training and test errors remain root-mean-square error over all samples and output dimensions:

~~~text
RMSE = sqrt(sum((target - prediction)^2) / (sampleCount * outputCount))
~~~

An empty dataset reports `NaN`, not an invented zero error. Training requires samples, a finite target, and a bounded epoch budget. Statistics count actual completed epochs and samples.

## Explicit CUDA training sessions

Training can now run on an NVIDIA GPU through a `NeuroTrainingSession`. The existing `Neuro.train*` methods and a session created without an argument retain CPU execution. A CUDA session owns device weights, biases, momentum velocities and temporary buffers; it is a writer for the existing host model, so diagnostics and ordinary inference can inspect each completed epoch. Java callers can select the backend explicitly:

```java
try (var training = network.newTrainingSession(TrainingBackend.CUDA)) {
    double error = training.trainEpoch();
    System.out.println(training.getInfo());
}
```

`TrainingDeviceInfo` reports the actual backend, device name, identity, precision and kernel version. With the default `REFERENCE` engine, the `CUDA` backend uses FP64; its identity includes the device UUID and driver version, and its kernel version is the packaged PTX SHA-256. Selection is explicit: an unavailable CUDA driver, unsupported device, failed kernel load or exhausted device memory produces an error that can be resolved by selecting CPU or fixing the reported environment. The separate `CUBLAS` backend adds FP32 and matrix multiplication through optional native libraries, as described below. The specialized `SMALL` engine described later also implements driver-only FP32 training.

The native adapter uses the JDK Foreign Function and Memory API against the installed CUDA Driver API. On Windows it loads `nvcuda.dll`; on Linux it tries `libcuda.so.1` and then the WSL driver path. It selects device ordinal zero, requires compute capability 7.5 or newer, retains its primary context and creates a private stream. The installed driver JIT-compiles packaged PTX. Running training does not require a local CUDA toolkit, Python service, extra Java numerical framework or downloading a model. JavaExec tasks enable native access; direct Java launches need `--enable-native-access=ALL-UNNAMED` as well as the module's Vector API flags.

### What executes on the GPU

The host retains the existing seeded shuffle generator and sends the sample order to the device. `gather` copies the next sample or mini-batch from the resident packed dataset. `forward` computes every layer, `output_delta` and `hidden_delta` backpropagate errors, and `update` changes each weight, bias and momentum velocity. A thread owns each reduction and parameter update; floating-point atomic accumulation is unnecessary. Activations use sample-major storage, while weights retain output-major layout.

Online training processes one shuffled sample at a time, preserving the optimizer's update sequence. Mini-batch training computes gradients for a complete batch and divides the update by its actual sample count, including the final short batch. The `parallelism` argument remains relevant to CPU workers; CUDA schedules device work through its stream rather than creating that many host gradient workers. For Java callers, the interface takes all three arguments:

```java
try (var training = network.newTrainingSession(TrainingBackend.CUDA)) {
    training.trainMiniBatch(10, 64, 1);
    System.out.println(network.statistics().samplesSeen());
}
```

Kernels use FP64, explicit fused multiply-add at the corresponding scalar CPU operations and the model's selected EXACT or FAST sigmoid. The build disables implicit FMA contraction and does not enable fast math. Numerical parity is checked with a tolerance, since the device exponential and floating-point implementation need not be bit-identical to a JVM. Hardware acceptance compares parameters and momentum as well as predictions; a visually similar output surface is insufficient evidence of equivalent training.

At the end of each epoch the stream is synchronized and all parameters and velocities are downloaded into private staging arrays. Every downloaded value must be finite before any model parameter is published. A completed commit increments the model's epoch/sample counters and evaluates training RMSE on the CPU. A failed dispatch or partial download leaves the last completed host epoch intact and makes that session unusable for further training. Closing it releases ownership, allowing a new session or CPU continuation from the retained state. The pending shuffle order is retained after a failed epoch so continuation does not silently skip its sample ordering.

The session excludes a second writer and direct model/dataset mutation while it is open. Close the session before changing samples, then open another session to upload the new dataset. `close()` is idempotent; attempting to train through a closed session fails. Workspace capacity grows with the largest requested batch and is reused for smaller batches. A shared per-device admission budget retains 20% headroom from the first session's free-memory snapshot and checks current driver-reported free memory on each reservation. Requests that exceed either remaining limit fail immediately with a diagnostic; growing sessions never wait in a reservation queue. Releasing the final reservation removes the snapshot so a later workload measures capacity again. Parallel searches can therefore fail a trial when simultaneous device demand exceeds that budget.

### Studio, search and replay

The Studio's Training backend, precision and batch-size controls are part of the staged configuration: apply them with **Apply & restart**. CPU, FP64 and batch size one remain the defaults. FP32 is selectable for CUBLAS and AUTO; CPU and driver CUDA execute FP64. Epoch stepping, continuous training and seed comparisons use the applied configuration. Published frames expose the effective device and precision after a session is opened. Initialization or runtime failures enter the existing visible failure state; applying a valid CPU configuration provides an explicit recovery path. Pause and cancellation take effect between complete epochs, so an unusually expensive epoch still determines control latency.

Architecture-search configuration carries the selected backend, precision and batch size into each trial. Trials use independent sessions and retain actual device information with their results. Applying an architecture preserves its configuration. Replay opens the recorded effective backend and precision: an AUTO/FP32 request that resolved to CPU is replayed on CPU/FP64. Replay checks device identity, precision and kernel version before retraining; changing those inputs requires a fresh experiment. Rejected replay retains the scored results and paused main model so the user can inspect or retry it.

### Reproducible kernels and validation

`jneuro/src/main/cuda/train.cu` is the kernel source. CUDA 13.0.2 generates the packaged `train.ptx` for `compute_75`; `train.properties` records source, builder and PTX hashes plus compiler and ABI provenance. `verifyCudaResources` checks those artifacts without a CUDA toolkit. The separate JNeuro CUDA PTX workflow compiles in the pinned `nvidia/cuda:13.0.2-devel-ubuntu24.04` image and compares regenerated output on pull requests. Its manual dispatch produces a small downloadable PTX/provenance artifact, allowing development machines to avoid retaining the compiler image.

```text
python3 jneuro/tools/build_cuda.py verify
./gradlew :jneuro:check :jneuro:jacocoTestReport
./gradlew :jneuro:gpuCheck
```

The first two commands run without a GPU. CPU session tests preserve the prior execution contracts; an injected recording driver exercises actual buffer allocation, dispatch, growth, synchronization, ownership, failure and cleanup paths. These tests do not claim CUDA arithmetic coverage. `gpuCheck` executes the CUDA-tagged suite on a real NVIDIA GPU and fails when the device or driver is unavailable. It checks direct/deep/non-square/wide networks, both sigmoid modes, short mini-batches, sparse convergence checks, session reopening and CPU continuation with momentum. The ordinary module line-coverage gate stays at 90%, and native-window GUI paths remain a separate 90% gate.

GPU execution does not imply a speedup for every experiment. XOR and narrow networks perform little arithmetic per kernel launch; online SGD also launches multiple kernels for every sample, and this version downloads parameters and computes RMSE after every epoch. These costs can dominate the work. Broader networks and mini-batches provide more parallel work, but end-to-end timings must include initialization, transfers and diagnostics. Fusing adjacent stages, retaining parameters across multiple publication intervals, and benchmarking workload-dependent batch sizes are follow-up experiments, not assumed performance results.

## Concurrency contract

Independent inference sessions can be used for concurrent inference when weights are not being modified. Training is not safe concurrently with inference on the same mutable model. Kotlin does not change that contract or pretend to synchronize it implicitly.

The reusable parallel inference session owns a persistent `ForkJoinPool` and one inference workspace per worker. It partitions the batch and joins all tasks before returning. Closing the session shuts down the pool.

~~~kotlin
network.newParallelInferenceSession(2).use { session ->
    session.predictBatch(inputs, batchSize, outputs)
}
~~~

The studio follows a stricter single-owner rule: only its training worker accesses the mutable `Neuro`. The event-dispatch thread sends commands through a queue and renders completed frames. A Swing timer consumes the latest published frame instead of appending an unbounded repaint callback for every training update.

## Float and native inference

`toFloatModel()` captures a separate float parameter snapshot and its own workspaces. It preserves the selected activation mode and uses the float Vector API where configured. Subsequent training of the original model does not alter that snapshot.

`NeuroNativeBlas` preserves the optional FFM CBLAS experiment. It resolves `cblas_dgemm`, transposes and copies parameters into off-heap memory, retains biases and scratch buffers, and performs batched layer evaluation using matrix multiplication followed by activation. The native session is also a snapshot, not a view of a concurrently trained model.

Native BLAS is environment-bound and optional. The current library lookup targets common Linux OpenBLAS/BLAS/MKL names; an unavailable library does not prevent normal Java/Kotlin inference or the desktop UI. The benchmark requires `--enable-native-access=ALL-UNNAMED`, which its Gradle task supplies. Native tests compare outputs, batch resizing, and snapshot independence when a compatible library is installed.

## Optional cuBLAS matrix training

`TrainingBackend.CUBLAS` keeps the dense training algorithm in this project while using cuBLAS GEMM for matrix products. Java FFM resolves the CUDA Runtime, cuBLAS, NVRTC and Driver API dynamically. These libraries are additional runtime requirements for CUBLAS; the driver-only CUDA backend above remains independently usable. A complete matching native installation includes cuBLAS's companion library and NVRTC's builtins. `JNEURO_CUDA_RUNTIME`, `JNEURO_CUBLAS`, `JNEURO_NVRTC` and `JNEURO_CUDA_DRIVER` override their library locations; dependent DLLs/shared libraries must also be discoverable by the operating system. There is no CUDA Maven dependency, and ordinary CPU builds need none of these libraries.

### Selecting an engine and precision

The session API uses the top-level `TrainingBackend` enum. An explicit CUBLAS request fails visibly when its dependencies or device are unavailable. FP64 is the reference precision; FP32 stores device parameters, momentum, activations and gradients as floats and uses `cublasSgemm_v2`. Completed FP32 parameters are converted back into the public double-precision model. This conversion does not restore the precision lost in the GPU arithmetic.

```java
try (var training = network.newTrainingSession(
        TrainingBackend.CUBLAS, Neuro.TrainingPrecision.FP32, 64)) {
    training.train(10); // Ten epochs, batches of at most 64 samples.
    System.out.println(training.getInfo().getPrecision());
}
```

The third factory argument is the default batch size for `trainEpoch`, `train` and `trainUntil`. The `trainMiniBatch` method can override it for a call. Closing any session releases its exclusive writer ownership and keeps completed host state available for CPU continuation.

With `REFERENCE`, AUTO chooses CUBLAS only if estimated work for the effective batch reaches 2,000,000 floating-point operations and the native stack is available. The estimate sums `6 * batch * inputWidth * outputWidth` across layers and clamps the batch to the dataset size. Tiny workloads resolve to CPU before probing any native libraries. This is a deterministic heuristic, not a measured speedup guarantee. The reference CPU engine reports its actual FP64 precision even if AUTO was requested with FP32; `SMALL` has a separate CPU FP32 implementation.

```java
try (var training = network.newTrainingSession(
        TrainingBackend.AUTO, Neuro.TrainingPrecision.FP32, 32)) {
    System.out.println(training.getInfo().getBackend());
    training.trainEpoch();
}
```

The explicit matrix API uses the separate nested `Neuro.BatchBackend` enum: its `CUDA` value selects cuBLAS, `CPU` selects the matrix CPU implementation, and `AUTO` applies the same threshold. Existing two/three-argument `trainMiniBatch` calls retain their original CPU algorithm and numerical behavior.

```java
network.trainMiniBatch(4, 16, 1, Neuro.BatchBackend.CPU);
network.trainMiniBatch(4, 16, 1,
        Neuro.BatchBackend.CUDA, Neuro.TrainingPrecision.FP64);
```

The matrix CPU engine gathers sample-major batches into reusable primitive arrays, computes each layer and its deltas, then accumulates weight and bias gradients before momentum updates. Parallel workers own disjoint output ranges. This provides a CPU comparison for the cuBLAS layout without changing the original online-SGD defaults.

### Resident state and epoch publication

Each cuBLAS training call uploads the packed dataset and model state. Each epoch sends the deterministic shuffled row order. `gatherRows` materializes contiguous batch inputs and targets on the GPU; GEMM computes forward products, hidden deltas and weight gradients. Small CUDA kernels add bias, apply the selected sigmoid and derivative, reduce bias gradients and update momentum. A short final batch scales its update by its actual sample count.

Parameters and workspaces remain resident across epochs within one call. At every completed epoch, all weights, biases and velocities are downloaded into private staging arrays and checked for finite values before publication. The host then advances its epoch/sample counters and computes RMSE. A failed dispatch or partial download retains the last completed host epoch and its pending shuffle order; the failed session must be closed before retrying. All allocated buffers and native handles are attempted during cleanup even if another cleanup operation fails.

This cuBLAS implementation recreates its native handles, NVRTC module and device buffers on each training call. Studio and architecture search call it once per epoch to preserve responsiveness, so setup and compilation can dominate small workloads. The driver CUDA session described earlier retains its resources across calls. CUBLAS currently relies on allocation errors and cleanup rather than the driver backend's shared 80% admission budget; concurrent cuBLAS trials can therefore exhaust available device memory.

### Validation and bounded timing

The native adapters and orchestration remain inside the ordinary 90% JaCoCo gate. CPU-only tests call the real FFM wrappers against controlled native upcall stubs and exercise allocation, dispatch, publication, discovery and failure cleanup. Those stubs do not execute GPU arithmetic. Hardware-required tests separately compare FP64/FP32 parameters, momentum, predictions and continuation on a real device. `:jneuro:gpuCheck` requires both the driver and cuBLAS stacks; `:jneuro:cudaTest` is an alias, and neither treats unavailable hardware as passing acceptance.

```text
./gradlew :jneuro:gpuCheck
./gradlew :jneuro:cpuGpuBenchmark
./gradlew :jneuro:cpuGpuBenchmark --args="--profile matrix --epochs 5 --warmups 2 --repeats 5 --batches 64,256 --output build/reports/cuda-benchmark/matrix"
./gradlew :jneuro:cpuGpuBenchmark --args="--topology 2,8,8,8,1 --samples 128 --epochs 5 --warmups 2 --repeats 5 --batches 16,64 --output build/reports/cuda-benchmark/2-8-8-8-1"
```

`cpuGpuBenchmark` and the retained `cudaBenchmark` task run the same command-line harness. The default `smoke` profile uses 128 samples, a 32/64/32/4 network, two epochs, batches 16 and 64, one warmup and three measured repetitions. The explicit `matrix` profile uses 1024 samples with 128/256/128/32 and 256/512/256/32 networks. Both profiles compare CPU FP64, driver CUDA FP64, cuBLAS FP64 and cuBLAS FP32. These are hardware-bound entrypoints: explicitly requested GPU engines fail if unavailable; the harness never substitutes CPU silently.

Every round starts with the same seed, weights, momentum, dataset and shuffle state. In the default `MINIBATCH` mode, the harness calls `session.trainMiniBatch(epochs, batchSize, 1)` for every engine, with one CPU worker, and rotates the measured backend order each repetition. Warmups exercise the complete path but are excluded from the reported samples. A separate CPU reference runs outside timing, including when `--backends` selects only GPU engines. The `EPOCH` and `CHUNK` modes added with `SMALL` are described below.

The console prints median training-call and total times alongside speedups against the measured CPU median. A speedup above 1 means faster than CPU for that workload. Both JSON and CSV reports retain every measured round, separate session-open/training/close timings, device identity and precision. JSON additionally records complete minimum/median/p95 summaries, JVM/OS, logging level, seed, CUDA resource hashes and benchmark class hash. CSV includes training minimum/median/p95, total median and speedups beside each round. Pass `-PneuroBenchmarkRevision=<revision>` to record source provenance; otherwise that field says `unspecified`. The default report prefix is `build/reports/cuda-benchmark/benchmark`, resolved from the JNeuro module directory when launched by Gradle.

The total interval includes session open, training and close, including native initialization, transfers and diagnostics. In this original reference comparison, both GPU engines publish and compute RMSE on the CPU after every epoch; the matrix CPU engine computes RMSE once at the end of the multi-epoch call. These timings therefore compare application execution paths, including their different diagnostic costs. The interval excludes model/dataset construction and post-run comparison. The cuBLAS implementation creates handles, compiles kernels and allocates buffers inside the training call, so its training column includes those costs. The driver backend performs its retained setup during session open. Comparing both columns makes that difference visible; neither column is an isolated GEMM or kernel throughput measurement.

Each warmup and measured round verifies every weight, bias and momentum value, RMSE, and completed epoch/sample counters against the CPU reference. FP64 uses absolute tolerance `1e-10` plus relative tolerance `1e-8`; FP32 uses `5e-5` plus `2e-3`. Non-finite results or mismatches fail the run. Partial JSON reports carry a failure status and the failing stage/backend. CSV marks any retained rows as failed; a failure before the first measured round produces a header-only CSV. The original error is preserved if writing the report also fails. Tolerances account for precision and reduction ordering; this verifies the tested trajectory, not arbitrary long-run convergence equivalence.

Options accept separate name/value tokens: `--profile smoke|matrix`, `--topology 2,8,8,8,1`, `--samples 128`, `--epochs`, `--warmups`, `--repeats`, `--batches 16,64`, `--backends CPU,CUDA,CUBLAS_FP64,CUBLAS_FP32`, and `--output <prefix>`. `--smoke` remains a compatibility alias. Custom topology lists every layer, including inputs and outputs: `2,8,8,8,1` means two inputs, three hidden layers of eight neurons each and one output. It replaces the profile's shapes; the profile still supplies the sample-count default unless `--samples` overrides it. Custom cases allow 2 through 16 layers, widths 1 through 2048, at most 2,000,000 parameters and 1 through 8192 samples. The console prints startup settings and progress after each verified workload, outside timed intervals.

Larger epoch counts amortize setup differently, and batch size changes the training algorithm's update frequency. Compare engines within the same case, keep other GPU work idle, and retain the report before drawing performance conclusions. Tiny networks often favor CPU because launch, transfer and compilation overhead dominate arithmetic.

A local Windows 11 run on September 30, 2026 used an Intel Core i5-14400F with one CPU worker, an NVIDIA RTX 4060 Ti, driver 616.92 and OpenJDK 27+35. With two warmups and five measured repetitions per engine, median total session times were:

| Network | Samples / epochs / batch | CPU FP64 | CUDA FP64 | cuBLAS FP64 | cuBLAS FP32 |
| --- | --- | --- | --- | --- | --- |
| 2/8/8/8/1 | 128 / 5 / 64 | 0.52 ms | 69.25 ms | 722.91 ms | 735.56 ms |
| 32/64/32/4 | 128 / 2 / 64 | 2.58 ms | 63.37 ms | 626.54 ms | 647.50 ms |
| 128/256/128/32 | 1024 / 5 / 256 | 1535.14 ms | 154.93 ms | 825.81 ms | 843.11 ms |
| 256/512/256/32 | 1024 / 5 / 256 | 7042.04 ms | 683.85 ms | 1350.08 ms | 1367.02 ms |

Across both batches in both profiles and the custom 2/8/8/8/1 case, all 160 measured rounds passed full-state numerical validation. The largest listed case was 10.30 times faster through the driver CUDA backend including setup/cleanup; the tiny cases favored CPU. For 2/8/8/8/1 at batch 64, training alone took 0.495 ms on CPU versus 3.348 ms on CUDA, so even excluding session setup did not favor GPU. This is a bounded local application benchmark with one CPU worker, not a claim about every CPU configuration, sustained kernel throughput, or a general GPU speedup. Repeat the command on the target machine and inspect the distributions rather than treating these medians as a release performance guarantee.

## SMALL: specializing tiny networks without changing the default engine

The reference measurements above make `2/8/8/8/1` a useful optimization target: it has only 177 trainable weights and biases. The specialized family is deliberately finite: two inputs, one through four hidden layers, each independently chosen from widths 4, 8 and 16, and one output. That covers 120 topologies, including mixed widths such as `2/4/16/8/16/1`. Its largest member, `2/16/16/16/16/1`, has 881 parameters. A direct `2/1` network and arbitrary widths continue to use `REFERENCE`.

Engine and device are separate choices. Existing calls default to `TrainingEngine.REFERENCE`, CPU, FP64 and the exact sigmoid. Selecting `SMALL` enables the specialized CPU or driver-CUDA implementation; `SMALL` with AUTO currently resolves to CPU. It does not probe cuBLAS or guess a GPU crossover from network size. An unsupported SMALL topology or a SMALL/CUBLAS combination is rejected explicitly.

```java
// Same public model and dataset, explicit training implementation.
try (var session = model.newTrainingSession(
        TrainingBackend.CPU, Neuro.TrainingPrecision.FP64,
        64, TrainingEngine.SMALL)) {
    session.trainMiniBatch(10, 64, 1);
}
```

The batch hint controls `trainEpoch`, `train` and `trainChunk`: a hint of one uses online SGD, and a larger hint uses averaged mini-batches. Calling `trainMiniBatch(..., 1, ...)` explicitly retains matrix-style batch-one arithmetic. Online SGD and batch-one matrix training have the same mathematical update but different floating-point grouping in hidden deltas and weight scaling; the specialized kernels preserve those two conventions. `train` and `trainUntil` still publish each completed epoch. Multi-epoch publication requires the explicit chunk API.

### CPU training SIMD and precision

Earlier inference sessions vectorized predictions from a model snapshot. `SmallCpuTraining` instead owns mutable training weights, biases, both momentum buffers, activations, deltas and gradients. Its single-model layout keeps each neuron's input weights contiguous and maintains a transposed copy for the forward pass. Vector lanes represent output neurons: broadcast one input, multiply the corresponding contiguous neuron weights and accumulate in the same input order. Backpropagation and optimizer updates use vector operations where the layer width permits them, with scalar tails for the one-neuron output.

The implementation supports 128-bit and 256-bit species and a scalar path. It clamps the selected width to the JVM's preferred species and never requests 512-bit vectors. With FP32 and only width-four hidden layers it chooses 128 bits instead of leaving half of a 256-bit vector idle. Inline dispatch puts static `SPECIES_128` or `SPECIES_256` constants at the intrinsic call sites, giving C2 a constant vector shape. The `simdBits` metadata reports the selected implementation width; confirming machine instructions and establishing a speedup still requires the target JVM's compiled-code and benchmark evidence.

```java
// Read the resolved training path, including precision and SIMD selection.
var device = session.getInfo();
System.out.println(device.getEngine());
System.out.println(device.getPrecision());
System.out.println(device.getSimdBits());
System.out.println(device.getSigmoid());
```

The CPU implementation streams samples through reusable activation/delta/gradient arrays rather than allocating a batch-by-layer activation matrix. A mini-batch still accumulates gradients in sample order and applies momentum once after its actual sample count, including a short final batch. Forward sums and momentum use explicit `Math.fma`; unrelated multiplications retain their original grouping. This limits allocation and setup work without changing the optimizer's update frequency.

FP32 is a real compute-state choice. The CPU kernel stores weights, biases, gradients, momentum, activations and deltas in `FloatArray`, converts inputs and hyperparameters on construction, and widens the completed checkpoint back into the model's public double arrays at publication. Requesting FP32 therefore changes the training trajectory; it is not merely a smaller input buffer. Reopening a CPU FP64 session imports the last rounded checkpoint, including momentum. It cannot recover the precision discarded by FP32 training.

```java
try (var session = model.newTrainingSession(
        TrainingBackend.CPU, Neuro.TrainingPrecision.FP32,
        16, TrainingEngine.SMALL)) {
    session.trainMiniBatch(5, 16, 1);
}
// The model now contains the completed FP32 trajectory, widened to doubles.
```

EXACT remains the default activation. It evaluates `Math.exp` per lane on CPU; vectorizing the surrounding affine transforms does not turn that call into a custom approximate exponential. FAST is an explicit hyperparameter: it reduces the magnitude into a base-two exponent and remainder, evaluates the existing degree-five polynomial, and rescales it. The polynomial arithmetic can be vectorized, but the selected mode is recorded in every training identity and replay. FAST and FP32 need separate paired-seed convergence measurements, including epochs to target and failures to converge, in addition to throughput measurements.

### Chunks, recoverable shuffle state and publication

`TrainingChunkRequest` makes deferred publication visible in the API. A request may train at most 64 epochs in one call, accepts an optional target/error-check interval and cancellation supplier, and has a default 25 ms cooperative time budget.

```java
var request = new TrainingChunkRequest(
    64, null, 1, () -> cancelled.get(), 25_000_000L);
var result = session.trainChunk(request);
int committed = result.getCommittedEpochs();
var reason = result.getTermination();
```

The session initially trains one epoch to estimate the cost. Later launches use the remaining time estimate, the 64-epoch cap, and the next target-check boundary to choose a count. Cancellation is inspected between calls into the compute kernel. A submitted CPU/native/GPU chunk runs to completion before the host can cancel it; the time budget is advisory and can be exceeded by one expensive epoch, launch or driver operation. `COMPLETED`, `CONVERGED`, `CANCELLED` and `BUDGET` distinguish the reasons for returning. Callers advance their visible progress by `committedEpochs`, never by the requested count.

`TrainingShuffle` reserves the ordered sample permutations before computation. The reservation contains the random-generator position associated with each epoch; committing consumes exactly that prefix and makes its generator position recoverable. Order arrays are recycled after commit. A failed chunk leaves its pending orders available, so closing and reopening from the last host checkpoint retries the same samples in the same order. Adding data invalidates obsolete order buffers and resumes from the last committed random-generator position with the new dataset size. Dataset mutation still requires releasing the model's exclusive training owner first.

The compute kernel only returns a private `NeuroTrainingState`. `SmallTrainingSession` validates topology, buffer dimensions and every parameter/momentum value before copying it into the host model, advancing epoch/sample counts and computing the new RMSE. A failure after private computation may have advanced that kernel's internal state, so the session becomes terminal and must be closed. The host retains its last completed checkpoint and remains usable after close. This rule covers native status errors, CUDA dispatch/synchronization failures, partial downloads and non-finite outputs; a submitted kernel is never itself evidence of a published epoch.

### Multiple models per vector and independent trial progress

`SmallCpuCohort` uses a second layout: a structure of arrays where adjacent lanes contain the same parameter from different models. Four FP64 models or eight FP32 models fit a 256-bit group. Each lane has independent weights, momentum and shuffle orders; lanes never reduce into each other's gradients. This avoids having to fill a vector with neurons from a very narrow layer. The last group is padded, and activity masks keep inactive models' public states unchanged.

The public `NeuroTrainingCohort` requires distinct models with identical topology, dataset values and hyperparameters except seed. It acquires every model as a writer before opening compute state. CPU groups can run on a bounded worker pool; CUDA uses one packed cohort on one stream. All completed group results are validated before any enabled sibling model is published. If a CPU task fails or the caller is interrupted, the implementation joins every submitted worker before releasing resources; an interrupted wait is not evidence that a worker stopped using its buffers.

```java
// models have the same shape/data/hyperparameters and independent seeds.
try (var service = new NeuroTrainingDeviceService(1);
     var cohort = service.openCohort(models, TrainingBackend.CPU,
             Neuro.TrainingPrecision.FP64, 64, 4)) {
    boolean[] active = new boolean[models.size()];
    java.util.Arrays.fill(active, true);
    var results = cohort.trainChunk(request, active);
}
```

Each result reports that model's committed epochs and termination reason. Models reaching a target can stop while their siblings continue. Search uses compatible seeds as a cohort while retaining per-seed checkpoints, scores and success/failure status. The search scheduler already supplies bounded parallel workers, so a cohort evaluated on a search worker does not create another nested pool. Architecture candidates still obey the SMALL shape family and the configured search bounds and parameter budget.

### Fused CUDA: one block per model, one packed result per chunk

The SMALL CUDA engine uses the installed driver and the packaged PTX, independently of cuBLAS/NVRTC. `small_train_fp64` and `small_train_fp32` keep one model's parameters and momentum in shared memory during a chunk. The launcher passes `models * 128` work items to the existing 128-thread launch adapter, giving exactly one block per model. A single model therefore occupies one block; launching it on a GPU does not by itself expose device-wide parallelism.

Each block handles the epoch and mini-batch loops internally. Forward activations and deltas are tiled over eight samples; gradient accumulation preserves sample order across tiles. Parameters, velocities and gradients plus the eight-sample activation/delta tile require under 30 KiB of explicit FP64 shared storage for the largest supported shape, below the 48 KiB design bound. Scratch storage does not grow with dataset size or the requested mini-batch. A short final tile and a short final batch use their actual counts. The source preserves the online/matrix arithmetic distinction and uses explicit FMA with compiler contraction disabled for other expressions.

Weights, biases and momentum are packed together. Inputs and targets are uploaded once per session and shared across models in a cohort; topology and state buffers also remain resident. Each chunk uploads its reserved orders and active-model flags, launches once, synchronizes and downloads one complete packed state buffer. The host reuses same-size order staging, activity and permutation-validation arrays, and alternates two packed download buffers. A partial download can therefore damage staging without overwriting the last successful packed result. The host checks the entire result before any model publication. The device order buffer grows only when a larger chunk requires it, with memory admission checked before allocation. A failed or non-finite chunk poisons the compute session and preserves the host checkpoint.

FP32 uses float arithmetic and float shared working state inside the fused kernel. Its global packed state and host transfer ABI remain double precision: values are cast on entry and widened on exit. Inputs and targets also use the common double upload ABI. Thus FP32 currently reduces compute/shared-memory precision, **not** the byte size of persistent global state or the packed download. The device metadata explicitly includes `small-v2/packed-fp64/` and the packaged PTX hash alongside the actual compute precision.

```java
try (var service = new NeuroTrainingDeviceService(1)) {
    try (var session = service.openSession(model, TrainingBackend.CUDA,
            Neuro.TrainingPrecision.FP64, 64, TrainingEngine.SMALL)) {
        session.trainChunk(request);
    }
    // Another session may reuse the released driver/context/module/stream lease.
}
```

`NeuroTrainingDeviceService` is the explicit owner of reusable native driver leases. A lease retains the CUDA primary context, loaded module, cached entrypoints and private stream between sessions. Each retained driver also owns an exact-size buffer pool capped at 1 MiB and 64 allocations. A session release returns eligible storage to that pool; larger/excess allocations are freed. Reopening uploads the new model/dataset into borrowed storage, so pooled capacity does not imply that old tensor values remain a usable model. Live allocations are tracked separately, and pending stream work completes before storage becomes reusable. Session close drains unreleased allocations; a broken lease flushes the entire pool and closes its driver. Service close frees every pooled buffer before releasing native handles.

Concurrent leases have a configured limit and fail immediately when exhausted. Device allocation also uses the existing shared 80% admission budget and current free-memory check. Free-memory reporting remains the driver's physical value, including memory already held by the pool; the active-session estimate is consequently conservative. There is no waiting queue and no process-global hidden context cache. Close every session/cohort before closing its service; cleanup attempts all owned resources and preserves suppressed failures.

### Studio pacing, search scoring and replay

The Studio exposes engine, backend, training precision and sigmoid mode separately, and displays the resolved device identity, engine, precision, SIMD width and kernel provenance. The worker owns a device service, so resetting a model can reuse an available CUDA driver lease. Training remains off the Swing event thread. Command polling while training is paced at 50 ms, and the current Swing refresh timer is 75 ms; these are pacing intervals, not a hard real-time frame-rate promise. User commands can wake the worker earlier.

`NeuroStudio.closeTraining()` releases the active session while preserving the model and device service for a later continuation. `close()` is terminal: it releases both session and service and rejects further training. Pausing, rebuilding data and replay use the resumable operation; window shutdown and `AutoCloseable` ownership use terminal close. Appending training data invalidates both pending shuffle storage and the cached RMSE, so reopening cannot report convergence against the previous dataset's error.

SMALL work within one Studio advancement has a 25 ms cooperative budget and checks queued commands between compute calls. Requests stop at the maximum epoch and the next visible checkpoint milestone. Normal Studio target checking remains per epoch (`checkEvery = 1`), so it deliberately does not skip over a convergence point merely to make a larger GPU launch. Search instead limits chunks to the next configured scoring boundary and retains each seed's best checkpoint; reaching the target is still distinct from completing a full-budget search trial.

Replay records whether a trial used a cohort. A cohort trial reopens the cohort implementation even when replaying just one seed, preserving the arithmetic path instead of switching to a single-model neuron-lane kernel. Replay checks the complete `TrainingDeviceInfo`: backend, device/driver identity, precision, kernel version, engine, SIMD width and sigmoid mode. It advances only to the recorded scoring boundaries and verifies the recovered score before replacing the displayed model. This intentionally rejects incompatible provenance rather than promising bitwise replay after a driver, compiler or arithmetic-mode change.

### Native AVX2/FMA experiment and its retention gate

The optional native experiment targets only `2/8/8/8/1` in FP64. It uses persistent FFM state/data/order buffers and a structure-of-arrays C++ loop with four independent models per AVX2 vector. Remaining models, including a single model, use scalar native arithmetic. One FFM training call processes a 1..64-epoch chunk; copying the completed buffer, constructing a full checkpoint and publishing it are part of the application path being measured. The experiment preallocates orders for 64 epochs and rejects workloads requiring more than 64 MiB for that buffer.

The dispatcher is compiled separately from the AVX2 translation unit. At runtime it checks AVX2 and FMA support; Windows also checks OSXSAVE/AVX support and enabled XState before entering the vector unit. This gives unsupported CPUs a scalar path without executing forced AVX2 instructions in the dispatcher. No AVX-512 target is requested. The Windows workflow builds a DLL with the runner's existing MSVC toolchain and `/fp:strict`, with `/arch:AVX2` restricted to the vector unit; the artifact records compiler flags, source hashes, revision and library hash. The library path is explicit through `jneuro.small.native`, exposed to Gradle tasks as `-PneuroSmallNative=...`.

This is a benchmark candidate, not a new public `TrainingBackend`. Acceptance requires full-state parity first. Retaining it as a production option additionally requires representative end-to-end medians at least 20% better than the optimized JVM path with no p95 regression, including packing, copying and publication. A faster isolated inner loop or a winning best sample does not satisfy that gate. The scalar exact exponential, off-heap copies and small-cohort tails are real costs, and the current experiment does not support the other 119 SMALL shapes or FP32.

### Reproducing the comparisons

CPU-only runs can select an explicit list without touching a CUDA installation. `MINIBATCH` measures the existing bulk mini-batch API, `EPOCH` measures repeated public epoch calls, and `CHUNK` uses bounded deferred-publication requests. `EPOCH` and `CHUNK` use online SGD when batch size is one. Compare the same mode, dataset, batch, precision and sigmoid before interpreting speedups.

```text
./gradlew :jneuro:cpuGpuBenchmark --args="--topology 2,8,8,8,1 --samples 128 --epochs 64 --batches 16,64 --backends CPU,SMALL_SCALAR_FP64,SMALL_128_FP64,SMALL_256_FP64 --mode CHUNK --sigmoid EXACT --warmups 3 --repeats 9 --output build/reports/small-experiments/cpu"
```

The following is hardware-bound. It requires a compatible NVIDIA driver; requested GPU paths fail if unavailable. `--retained-device true` keeps the explicit service alive across benchmark sessions, so measured session-close times exclude final service destruction. Warmups can absorb the first driver/module initialization. Use `false` to measure fresh service ownership and keep the two lifecycle regimes separate in reports.

```text
./gradlew :jneuro:cpuGpuBenchmark --args="--topology 2,8,8,8,1 --samples 128 --epochs 64 --batches 64 --backends CPU,SMALL_CUDA_FP64,SMALL_CUDA_FP32 --mode CHUNK --retained-device true --warmups 3 --repeats 9 --output build/reports/small-experiments/cuda-retained"
```

`SmallTrainingJmhBenchmark.publicEpochs` measures ten public epochs per invocation and reports time per epoch using `@OperationsPerInvocation(10)`. Invocation setup recreates the same seeded trajectory and is excluded from timing, as is session close. The benchmark returns the resulting RMSE. Its scalar, 128-bit, 256-bit, FP32 and reference variants use the same workload description; `-prof gc` exposes allocation costs. JVM warmup and multiple forks matter for the Vector API, and generated-instruction inspection is separate evidence from Java source containing vector calls.

```text
./gradlew :jneuro:jmh --args="com.lis.neuro.SmallTrainingJmhBenchmark.publicEpochs -p engine=REFERENCE_SCALAR,REFERENCE_MATRIX,SMALL_SCALAR,SMALL_128,SMALL_256 -p workload=MINI_BATCH -p sigmoid=EXACT -p topology=2-8-8-8-1 -p samples=128 -p batchSize=64 -wi 3 -i 5 -w 1s -r 1s -f 3 -prof gc -jvmArgsAppend --add-modules=jdk.incubator.vector -rf json -rff build/reports/small-experiments/jmh.json"
```

The separate `smallExperiments` command measures complete cohort session lifetimes, including order reservations, state validation and host publication. It rotates engine order between rounds, checks full parameter/momentum state and epoch/sample counters against independently trained references, and writes per-round CSV plus an environment/provenance file. `REFERENCE_PARALLEL` is a useful baseline: running several scalar/matrix models concurrently can be more effective than packing them into one SIMD group. The CPU-only example below can add `SMALL_CUDA` on a GPU workstation.

```text
./gradlew :jneuro:smallExperiments --args="--mode cohort --models 32 --epochs 64 --samples 128 --batch 64 --parallelism 4 --warmups 3 --repeats 9 --engines REFERENCE_MATRIX,REFERENCE_PARALLEL,SMALL_SEQUENTIAL,SMALL_CPU --output build/reports/small-experiments/cohort.csv"
./gradlew :jneuro:smallExperiments --args="--mode quality --seeds 32 --epochs 10000 --check-every 25 --target 0.05 --output build/reports/small-experiments/quality.csv"
```

The quality mode runs paired seeds across FP64/FP32 and EXACT/FAST, retaining convergence, final RMSE, elapsed time and epochs to the selected target. It is a bounded but longer CPU experiment. Native checks require the explicitly built artifact and fail if it is missing; selecting `NATIVE` adds the native candidate to cohort timings.

```text
./gradlew :jneuro:nativeSmallCheck -PneuroSmallNative=C:/path/to/jneuro-small.dll
./gradlew :jneuro:smallExperiments -PneuroSmallNative=C:/path/to/jneuro-small.dll --args="--mode cohort --models 32 --epochs 64 --samples 128 --batch 64 --parallelism 4 --warmups 3 --repeats 9 --engines SMALL_CPU,NATIVE --output build/reports/small-experiments/native-gate.csv"
```

### SMALL validation and measured results

The ordinary tests exercise all 120 CPU shapes, precision/mode choices, momentum continuation, order reservations, commit/failure boundaries, dataset growth and cohort activity. Injected CUDA drivers and native FFM upcalls assert allocations, argument layouts, launch counts, partial failure and cleanup; they establish orchestration behavior rather than GPU/native arithmetic. Strict CUDA acceptance compares full state against scalar CPU for every supported shape in FP64/FP32 and EXACT/FAST, with additional online, ragged-batch, 64-epoch, cohort and continuation cases. Native acceptance compares its limited shape across scalar, vector and tail-model execution. Missing requested hardware/libraries fail those explicit tasks. Module line coverage and GUI path coverage retain their separate 90% gates.

The [September 2026 measurement record](benchmarks/2026-09-30-small/README.md) retains machine-readable distributions, independent JVM identities, source/artifact hashes, quality outcomes and instruction evidence. The local implementation revision is `b987d7b`; CI measured `7965204`, which adds reporting tools without changing the training kernels. Local checks passed 264 of 265 ordinary tests (one optional native-BLAS test skipped), with **97.0% module line coverage**, plus 16 real CUDA/cuBLAS tests and two actual native-DLL tests. The corrected real-window suite passed 25 tests and covered 127/127 GUI paths at `e0f7de6`. The hardware was an i5-14400F, RTX 4060 Ti and Windows JDK 27. Real GPU acceptance exercised all 120 shapes in both precisions and both sigmoid modes; injected-driver tests remain separate evidence for failure handling.

The three-fork JMH comparison uses `2/8/8/8/1`, 128 samples, batch 64 and EXACT. Each fork has five one-second warmups and five one-second measurements. The table reports **mean microseconds per published epoch**, with JMH's score error, not end-to-end session latency. Both pre-existing CPU trainers were measured.

| Engine | Local i5-14400F µs/epoch | CI EPYC 7763 VM µs/epoch |
|---|---:|---:|
| Reference scalar | 266.439 ± 7.760 | 193.216 ± 4.866 |
| Reference matrix | 261.077 ± 78.562 | 186.070 ± 26.443 |
| SMALL scalar FP64 | 218.376 ± 8.902 | 172.861 ± 43.471 |
| SMALL 128-bit FP64 | 182.567 ± 15.275 | 126.601 ± 2.147 |
| SMALL 256-bit FP64 | 179.358 ± 18.422 | 120.636 ± 1.720 |
| SMALL scalar FP32 | 212.898 ± 6.980 | 155.954 ± 2.585 |
| SMALL 128-bit FP32 | 168.092 ± 17.830 | 143.395 ± 14.930 |
| SMALL 256-bit FP32 | 163.033 ± 8.388 | 128.231 ± 13.671 |

On the CI VM, FP64 256-bit SMALL's observed mean is **1.54× faster** than the strongest reference mean; FP32 256-bit is 1.45×. FP32 is not universally faster: this CI case favors FP64. These results do not establish the 2× CPU stretch target. The VM exposes four logical processors and runs Temurin 27+35 on Windows Server 2025; it is a different host from the local desktop, so their samples are never pooled. The hosted runner is also not a guarantee of exclusive physical CPU scheduling.

JMH invocation setup creates a fresh seeded model/session outside method timing. Each timed method publishes ten epochs and normalizes the result per epoch. Its GC profiler includes invocation-fixture allocation, even though fixture time is excluded; the local SMALL measurements of roughly 13.0–13.5 kB/epoch cannot be described as training-only allocations or an allocation-free public API. The retained workspaces avoid intermediate training arrays, while complete public checkpoints still cost copying and allocation. Explicit chunks amortize those publications. JMH percentiles in the linked record summarize **iteration averages**, not individual-epoch tail latency.

Generated C2 bytes were decoded with `objdump`, independently of the Vector API source. All 142 captured compilations had complete byte ranges. FP64/FP32, 128/256-bit, single-model/cohort forward, backward, gradient and update methods contained the expected packed FMA instructions. FAST's polynomial contained packed multiply/add, while output dots, range reduction and exponent scaling retain intended scalar work. Inspection also found a limitation: FP64 **128-bit cohort** updates contain allocation/helper paths and vector stores around non-deoptimization calls. Default FP64 256-bit and both FP32 widths instead emitted native masked stores without those allocation paths in the captured update methods. Static stack-store counts are not hot-loop spill measurements; the report separates deoptimization-state preservation from other stores.

The native candidate failed every required local workload (nine) and CI workload (seven), including packing, session lifetime and host publication. It remains an **internal benchmark experiment**, with no native application backend added. On CI, native total medians ranged from 1.086× to 4.966× the fastest applicable JVM baseline; none achieved the required ratio of at most 0.8. The cohort comparison allows the configured four-worker JVM baseline while native uses one worker, and records that distinction explicitly. Missing cases or failed parity cannot be removed to manufacture a passing gate.

GPU measurements demonstrate working fused execution, but do not qualify the 10× retained-session GPU target. On the interactive desktop, fused FP64 five-epoch mini-batch fork medians varied from 1.937 to 53.360 ms; CPU chunk medians also varied by nearly 5×. External activity and unmeasured clock/scheduling changes prevent a clean causal speedup claim. A separate 32-model FP32 cohort run observed median total times of 22.747 ms CUDA versus 73.944 ms SIMD CPU, while the FP64 cohort regressed against the parallel reference baseline. The linked report retains these wins **and regressions**, along with p95, rather than promoting the best run. A quiet GPU host with controlled clocks/load is the follow-up measurement.

The 32 paired XOR seeds reached RMSE 0.05 in exactly the same seeds across FP64/FP32 and EXACT/FAST: 6/32 at 10,000 epochs and 32/32 at 100,000, checking every 25 epochs. At the larger budget the median was 16,237.5 epochs in every mode; p95 was 53,150 for EXACT and 53,275 for FAST. Conditional median elapsed times were 60.988, 63.186, 62.156 and 57.092 ms for FP64 EXACT, FP64 FAST, FP32 EXACT and FP32 FAST respectively. The original 26 censored seeds remain visible at the smaller budget. This is quality evidence for this fixture, not a universal convergence or FAST performance guarantee.

The record separates zero-warmup first-session calls, warmed session lifetimes, public epochs, bounded chunks and multi-model throughput. First-session timing excludes JVM launch and model/data construction; the harness has already computed its CPU oracle. It is therefore not a pristine-JVM startup measurement. Process wall times include reference validation and report I/O. The cuBLAS comparison has only one fork and includes its compilation/allocation/transfers, so it is reported separately from retained fused-kernel throughput.

For a bounded CPU/native rerun, dispatch `.github/workflows/jneuro-small-native.yml` with `benchmarks=true`. It builds with existing CI toolchains, verifies the real library, then invokes fresh JVMs directly after Gradle exits. Three forks cover both mini-batch sizes, 64-epoch chunks, 1/4/32-model cohorts and all eight JMH engines. The aggregator requires complete provenance and passing numerical checks. A valid negative native-retention decision is a successful experiment result; malformed evidence, failed tests or incomplete forks still fail the workflow.

## Structured application logging

JNeuro uses the JDK logging API without a new logging dependency. Application entrypoints configure the `com.lis.neuro` namespace and emit JSON lines to the console and rotating UTF-8 files. They do not reconfigure the JVM's global root logger. Interactive launches default to INFO and file output under `${user.home}/.jneuro/logs`; files use `jneuro-%u-%g.log`, where JUL selects a process-safe unique index and rotation generation. The default is three files with an approximate 2 MiB limit each; a final record can cross the rotation threshold. File initialization failure emits `logging.file.unavailable` to the console and leaves the application usable.

Configuration must be set before the first JNeuro operation initializes logging. Java callers can use normal system properties:

```java
System.setProperty("jneuro.log.level", "DEBUG");
System.setProperty("jneuro.log.dir", "./logs");
System.setProperty("jneuro.log.limit_bytes", "2097152");
System.setProperty("jneuro.log.count", "3");
```

Equivalent environment variables are `JNEURO_LOG_LEVEL`, `JNEURO_LOG_DIR`, `JNEURO_LOG_LIMIT_BYTES`, `JNEURO_LOG_COUNT` and `JNEURO_LOG_FILE`. System properties take precedence. Supported levels are INFO, DEBUG, TRACE, WARN, ERROR and OFF; `jneuro.log.file=false` disables the file sink. Limits accept 1024 through 16777216 bytes and counts 1 through 10. Invalid configuration emits a warning and uses the default for that field.

Gradle forwards `-PneuroLogLevel`, `-PneuroLogDir`, `-PneuroLogLimitBytes`, `-PneuroLogCount` and `-PneuroLogFile` to the child JVM. Explicit Gradle properties take precedence over environment variables. Benchmark and verification tasks default to WARN with file logging disabled to reduce measurement noise and disk writes; tests also disable file output. An explicit DEBUG/TRACE benchmark run measures the cost of that logging as well as training, and the report records the selected level.

```text
./gradlew :jneuro:runXorCanvas -PneuroLogLevel=DEBUG
./gradlew :jneuro:cpuGpuBenchmark -PneuroLogLevel=WARN -PneuroLogFile=false
```

Every record includes UTC time, severity, component, event, process run identifier and thread. Structured fields connect model/session IDs with window, run, architecture-search and trial IDs. INFO covers application/window lifecycle, user actions, configuration changes, resolved engines, session ownership and training summaries. Training summaries include epochs, samples, batch size, configured CPU parallelism, precision, device, elapsed time and RMSE. Single-epoch training summaries are sampled at the first session epoch and each hundred-epoch boundary. Internal `advanceForSearch` chunks follow the same bounded INFO policy: starts are DEBUG, and completions are INFO only for the first chunk that publishes work or a chunk crossing a hundred-epoch boundary. A 25-epoch search chunk therefore does not generate an INFO start/completion pair on every call. Zero-work and cancelled-before-start chunks remain DEBUG; cancellation fields report only committed progress, and failures retain their original cause and counters. DEBUG retains every chunk's correlated start/completion, requested and committed epochs, sample counts and termination. Public multi-epoch `train`/`trainUntil` calls retain their INFO summaries, and UI/search terminal states have separate completion events. DEBUG exposes intermediate progress, search scheduling, native library discovery, memory admission, allocations and cleanup. TRACE adds prediction, kernel/GEMM dimensions, transfers and synchronization events. Native addresses and tensor contents are not logged.

Device selection and completed GPU work are distinct events. `session.resolved` reports the requested versus actual backend and precision, while `training.completed` sets `gpuWorkCompleted=true` only after GPU execution has returned and the host model's completed epoch counter has advanced. Native publication follows synchronization and staging validation. A loaded CUDA library alone is therefore insufficient evidence that an epoch ran on the GPU. Failures retain the original exception and stack trace, report retained host progress and do not emit a false completion. Logging failures cannot replace training failures or stop training.

The desktop emits actions such as start, pause, single step, restart, configuration rejection, navigation, replay, export, seed studies and shutdown. Search logs retain trial metadata and terminal success/failure/cancellation. Lazy DEBUG/TRACE fields avoid building detailed payloads when disabled; strings, field counts and serialized exceptions are bounded. TRACE is intentionally verbose and should be used for bounded investigations. Coverage tests exercise rotation, configuration, sink failures, correlation and original-cause preservation; real-window tests assert representative logging paths through ordinary UI input.

## The desktop redesign

The previous full-width toolbar mixed dataset selection, topology editing, training commands, and analysis navigation. It could extend beyond the window, while deeper or wider networks exposed fixed-layout assumptions.

The new studio gives these tasks separate locations. The left sidebar contains the dataset, architecture, seed, and epoch budget. Learning rate, momentum, and target RMSE are under Advanced settings. Editing fields stages a configuration; **Apply & restart** validates and submits it as one command. Invalid inputs produce visible text, not only a colored field or tooltip.

The active configuration is always read from the published model frame. A pending request cannot relabel the old model with a new topology. Applying settings resets the model, parameter history, checkpoints, and seed-study results together.

The header exposes epoch, training RMSE, parameter count, sample count, and a named state. States distinguish Ready, Training, Paused, Target reached, Epoch limit reached, Add training points, and Training failed. Empty or terminal runs disable training/step controls and do not keep spinning on pending work.

Keyboard controls are:

~~~text
Ctrl+Enter   Apply settings and restart
Ctrl+Space   Train / pause
Ctrl+Right   Step one epoch
Ctrl+R       Reset the active configuration
Arrow keys  Move the probe in a focused chart
Enter       Add a point in the Custom learning-set view
~~~

Scrollable analysis views keep plots readable rather than shrinking the complete dashboard into an unusable thumbnail. Narrow windows stack overview cards; parameter and seed views adapt their layouts. The sidebar scrolls independently when its advanced controls exceed the available height.

## Reading the seven views

**Overview** puts the continuous output surface next to the convergence plot. Four reference-input cards show the corner predictions. These coordinates are useful for XOR, but are not described as validation samples for arbitrary datasets. The headline error is explicitly training RMSE.

**Neurons** shows the actual multilayer topology, signed weighted connections, and a probe forwarded through every layer. Large layers show a bounded number of graph nodes with an explicit omitted-node count. The hidden-layer selector and neuron offset page through activation maps rather than silently limiting inspection to the first layer. Only first-layer maps receive straight `z = 0` boundary overlays; later-layer activation surfaces are not incorrectly treated as linear functions of the original inputs.

**Learning set** overlays all current training points. Class 1 uses a circle and class 0 a square, in addition to different colors. Custom datasets support adding points with the selected class, right-clicking for class 0, keyboard entry at the probe, undo, and clear. An edit resets and pauses training so the changed experiment is explicit.

**Step effect** shows the function before and after the last completed update batch and a signed difference map. The epoch labels identify the compared states. Its fixed color scale saturates at an output difference of ±0.25; this is an output-change map, not a gradient visualization. Pausing and stepping isolates a complete shuffled SGD epoch.

**Parameters** selects a real layer and a parameter page. Weights and biases are plotted separately, while the norm chart covers every layer. There are no fixed 25-parameter offsets: offsets and counts come from the captured topology.

**Seeds** is an on-demand experiment rather than work performed on startup and every reset. The same dataset and architecture are trained from seeds 1, 42, 123, and 999, with explicit epoch budgets and convergence outcomes. The main model is paused but not replaced. Applying a new configuration or cancelling invalidates the study. A failed run is not presented as proof that the architecture cannot represent the dataset.

**Timeline** retains epoch zero, exact milestones reached during training, and the final convergence or budget-exhausted state. It preserves the final image even when convergence occurs between scheduled milestones. These are historical images, not editable model checkpoints or a replay control.

## Bounded visualization work

`NeuroStudio` is a headless, unit-testable state machine. Its configurations are immutable values, published histories own their arrays, and the UI reads parameter snapshots rather than mutable model internals.

`BufferedImage` is itself mutable. The safety guarantee is ownership: after an image enters a published frame, the worker never modifies that image again. New images are created for new states. The same rule applies to diagnostic arrays and history records.

The diagnostic renderer reuses a workspace across grid points, instead of allocating a complete probe and several arrays for each pixel. Grid resolution is reduced for large parameter counts to bound visualization cost. History is downsampled within a parameter-dependent memory budget while retaining the initial and most recent sample. Seed studies and timeline lists are bounded as well.

Delta rendering reuses the two output grids rather than evaluating both networks a second time. Repaints, hover inspection, and layer selection do not advance the training RNG or change model results.

## Verification and numerical parity

The migration is checked against the Java code pinned at commit `75f8d619975bca7d2cbac141fc8bfaff2d8ff03d`. CI compiles that source into an isolated directory and loads it through a separate class loader. The source is not copied into the migrated module.

`NeuroMigrationParity` covers five topologies, three seeds, three kernel modes, and two sigmoid modes: 90 scenarios and 450 compared model states. Comparisons include initial inference, fixed-epoch online training, individual epochs, sequential mini-batches, parallel mini-batches, batch inference, float snapshots, errors, and counters. Double comparisons use tolerance `1e-12`; float comparisons use `1e-6`.

The pinned default XOR case reaches the target in 1144 epochs in both implementations:

~~~text
RMSE: 0.04997028695977735
00:   0.045355701043297426
01:   0.9499573164770715
10:   0.9514754011138365
11:   0.055426273796837275
~~~

Cross-kernel/batch tests use a small numerical tolerance rather than requiring bitwise equality from different reduction orders. Same-path deterministic behavior remains tested exactly. The native backend has its own equivalence tests.

JUnit tests also cover malformed topologies including trailing commas, defensive copies, later-layer activation maps, boundary clipping, empty datasets, terminal states, cancellation, bounded history, seed-study isolation, and UI control/worker interaction. The UI smoke test renders all eight implemented Swing views to PNG; these are application renders, not generated design mockups.

The repository's per-module 90% line-coverage rule remains enabled, with no additional production exclusions. The pre-existing environment-bound GUI/benchmark/JFR/native entrypoint exclusions are unchanged. The numerical engine, state machine, dataset generators, topology validation, and diagnostics remain subject to the gate.

## Running and benchmarking

The launch command is unchanged:

~~~text
./gradlew :jneuro:runXorCanvas
~~~

Windows PowerShell:

~~~text
.\gradlew.bat :jneuro:runXorCanvas
~~~

Verification and image rendering:

~~~text
./gradlew :jneuro:check :jneuro:jacocoTestReport
./gradlew :jneuro:renderStudio
./gradlew :jneuro:migrationParity -PneuroJavaBaseline=/path/to/compiled/java/classes
~~~

The renderer works without a graphical desktop and writes the eight application views under `jneuro/build/screenshots`. The normal interactive launcher requires a display. Save PNG exports the currently visible studio panel.

All existing JMH scenarios and task names are retained. Kotlin benchmark classes are open where the JMH harness subclasses them, and their parameter fields use `@JvmField`. Kapt generates the Java harness and benchmark metadata; generated Java is not manually maintained.

~~~text
:jneuro:jmh
:jneuro:jmhSmoke
:jneuro:jmhCompare
:jneuro:jmhBatchCompare
:jneuro:jmhTrainingCompare
:jneuro:nativeBlasBenchmark
:jneuro:performanceMatrix
:jneuro:profileNeuro
~~~

JFR events remain disabled by default and can be enabled through the profiling task. A language migration is not itself a speedup claim: use the retained benchmarks on the same machine to compare throughput, latency, and allocation.

## Design and tooling references

The implementation used the following primary documentation and UX guidance, checked on 29 September 2026:

- [Kotlin releases](https://kotlinlang.org/docs/releases.html): stable compiler version rather than an EAP build.
- [Kotlin compiler options](https://kotlinlang.org/docs/gradle-compiler-options.html) and [Gradle configuration](https://kotlinlang.org/docs/gradle-configure-project.html): explicit language/API/JVM targets and compatibility boundaries.
- [Kotlin-to-Java interoperability](https://kotlinlang.org/docs/java-to-kotlin-interop.html): records, static entrypoints, and overloads.
- [Swing event-dispatch thread](https://docs.oracle.com/javase/tutorial/uiswing/concurrency/dispatch.html): keep long-running computation outside the event thread.
- [FlatLaf](https://www.formdev.com/flatlaf/): standard desktop components, keyboard focus, and HiDPI behavior.
- [NNGroup: Progressive Disclosure](https://www.nngroup.com/articles/progressive-disclosure/): keep essential configuration visible and disclose advanced details separately.
- [W3C: Use of Color](https://www.w3.org/WAI/WCAG22/Understanding/use-of-color.html) and [Contrast Minimum](https://www.w3.org/WAI/WCAG22/Understanding/contrast-minimum.html): supplement color with labels/shapes and use legible text contrast.

These accessibility principles inform the redesign; they are not a claim that a custom desktop visualization has received a complete WCAG or assistive-technology conformance audit.

## Deliberate limitations

JNeuro remains an educational dense network, not a tensor framework. It still uses sigmoid activations, lacks automatic differentiation, softmax/cross-entropy, regularization, persistence, and adaptive optimizers such as Adam. GPU execution is deliberately limited to the optional dense CUDA mini-batch backend rather than becoming a general tensor/autodiff layer. Circle and Spiral can expose the limits of a narrow network and the training budget rather than guaranteeing a low error.

Training-set fit is not generalization evidence. The studio does not silently manufacture a validation set or claim calibrated probabilities. Its role is to expose the actual computation and make controlled architecture experiments easier to run and inspect.


## Automatic architecture search

The **Architecture search** tab, also reachable through **Auto search…** in the architecture sidebar, compares network sizes without changing the active model whenever a better candidate appears. Its initial policy is **Smallest meeting target**: first require the requested median RMSE and observed seed-success count, then minimize trainable parameters. Equal-size candidates are ordered by median error, depth and a stable topology order.

The two other policies are **Smallest near best**, which chooses the smallest fully evaluated model within an absolute RMSE tolerance of the best median, and **Lowest RMSE**, which prioritizes the lowest median error. These policies can be changed after training without rerunning trials. They do not require meeting the target; success counts remain visible so their different semantics are explicit. No weighted sum such as `RMSE + lambda * parameterCount` is used.

The engine is separate from Swing:

~~~kotlin
val data = ArchitectureSearchData.fitting(
    NeuroLearningSets.create(NeuroLearningSets.Kind.XOR, 42),
    "XOR"
)
val result = NeuroArchitectureSearch().search(
    data,
    ArchitectureSearchConfig()
)
val winner = result.selection.recommended
~~~

This API is module-internal, like the Studio model. It deliberately searches the Studio's two-input, one-output dense networks, not arbitrary input/output feature schemas.

### Search space and resource limits

The default **Adaptive · evolve best** strategy searches within 1–3 hidden layers, each 1–8 neurons wide, with at most 256 trainable values. These bounds contain 584 possible architectures, but adaptive search does not enumerate or promise to evaluate them all. Parameter count includes biases:

~~~text
parameters = sum((inputWidth + 1) * outputWidth)
2 → 1 → 1       = 5
2 → 2 → 1       = 9
2 → 3 → 1       = 13
2 → 2 → 2 → 1   = 15
2 → 6 → 1       = 25
~~~

The next candidate is generated from observed results, not taken from a precomputed list. Ordered layer sequences remain distinct: `4,2` is not deduplicated with `2,4`. `NetworkArchitecture` owns an unmodifiable copy of its widths and uses value equality, avoiding the reference equality of Kotlin arrays.

The default budget is 10,000 seed trials; the input has no arbitrary preset maximum. An adaptive run reserves a complete seed group before evaluating the next architecture. It checks retained checkpoint storage before funding each new group and stops with `MEMORY_LIMIT` before exceeding eight million parameter values. The default adaptive strategy does not call `architectures()` and can explore spaces too large for exhaustive enumeration. The optional **Exhaustive · reference** strategy preserves the original grid search: it rejects more than 4,096 candidates or 100,000 visited prefixes and funds a deterministic parameter-ordered prefix of complete seed groups. Checkpoint curves are compacted to at most 128 entries per seed while preserving the initial, final and current best checkpoints. These limits bound stored model state; actual JVM overhead also includes objects, datasets and temporary training workspaces.

Depth/width controls accept positive ordered bounds subject to explicit numeric, parameter and storage safety checks, without the former eight-layer/128-neuron caps. Adaptive mode previews a seed-trial budget and starting architecture, not a fictitious total candidate count. An optional wall-clock limit is a safety stop, not a reproducibility promise: it can leave different subsets evaluated on different machines.

### Adaptive mutation of completed leaders

The Studio supplies its active hidden widths as the initial architecture. If those widths lie outside the selected search bounds (including the direct `2 → 1` baseline when hidden layers are required), the planner explicitly starts from the smallest legal architecture instead. Editing the sidebar alone still does not affect the active search input.

`AdaptiveTrialScheduler` shares the trial pool across architectures as well as seeds. The previous coordinator submitted one architecture and waited for all of its seeds; this limited active work to `min(parallelism, seeds.size)`. One seed therefore used only one worker even with Parallel seed trials set to 32.

The scheduler now reserves a full seed group for each admitted architecture and submits individual trials while a worker slot is available. Once the bootstrap architecture is evaluated, spare slots receive offspring of the current completed elite. Finished groups update the elite immediately; no generation-wide or architecture-wide barrier holds up unrelated work. A slow seed does not block other architectures from progressing.

~~~kotlin
if (trials.size == config.seeds.size) {
    val candidate = candidate(finished.architecture, trials)
    pending.remove(finished.architecture)
    completed += candidate
    if (candidate.fullyEvaluated) planner.observe(candidate)
}
~~~

A partially evaluated architecture is never a parent. `propose()` may issue several children of evaluated parents while other children remain outstanding. An empty neighbourhood while work is pending means wait for feedback, not end the search. The initial single architecture must still finish before any valid parent exists; after that, utilization depends on available unseen elite mutations and the remaining budget. Search parallelism does not parallelize the ordinary Studio Train button.

After each completed seed group, `ArchitectureRanking` recomputes the recommendation, best-error candidate, ordinary Pareto frontier and reliability-filtered frontier. `EliteParentSelection` builds the current mating pool from the two frontiers and the recommendation, using only fully evaluated finite candidates. An architecture dominated out of both frontiers immediately loses eligibility and its cached neighbourhood is discarded. Failed or partial groups never breed.

A new recommendation (or best-error leader before a reliable solution exists) gets the next available mutation opportunity. A newly admitted Pareto candidate also gets one immediate opportunity when the leader is unchanged. Subsequent parents are **selected again by fitness for every offspring**, not rotated through an insertion-order queue. The pool is ordered by the active recommendation policy: reliable target-meeting architectures first and smallest first in the default mode; smallest within tolerance in near-best mode; lowest median RMSE in accuracy mode. Outside the preferred group, median RMSE then parameter count determine order. All comparisons use the completed seed group, not one lucky initialization.

The selector draws two ranks independently with replacement and chooses the better rank. With `n` eligible parents the best rank wins with probability `1 - ((n - 1) / n)^2`; lower-ranked frontier trade-offs retain a chance to breed, preserving diversity without using inferior candidates outside the elite pool. This is a fitness tournament, not a claim to implement NSGA-II or crowding-distance selection. Canonical ordering and the separate search seed make each selection reproducible for a given sequence of observations. With asynchronous evaluation, the observation sequence and therefore the adaptive search trajectory can change with timing or worker count; use one worker for a repeatable proposal sequence. Individual architecture/seed training and replay remain deterministic.

~~~kotlin
val parent = promotedParent?.takeIf { it in available }
    ?: EliteParentSelection.choose(available, random)
return issue(neighbours.getValue(parent).removeFirst())
~~~

`available` contains only current elites with an unseen legal local mutation. A parent whose local neighbours are all tested is omitted from this particular tournament, but remains eligible for a larger exploratory mutation. Neighbour lists are per-parent mutation caches, not a predetermined global search schedule.

The neighbourhood contains bounded, deduplicated mutations:

- Increase/decrease one hidden-layer width by one; also double/halve it for larger steps.
- Insert a hidden layer at any position, using the minimum or adjacent width.
- Remove a hidden layer, or swap adjacent widths.

Every proposal records its **Parent**, **Mutation** and generation. A newly improved child can immediately produce the following generation; it does not wait behind all remaining configurations of the old grid. The UI displays parent and mutation columns, and shows the active lineage above the progress bar.

~~~kotlin
val selection = ArchitectureRanking.select(evaluated.values.toList(), config)
val nextLeader = (selection.recommended ?: selection.bestError)?.architecture
~~~

This is architecture inheritance, **not weight inheritance**. Every proposed network still trains from the same fresh seed list with the same full budget. Carrying optimized parent weights into only some trials would make the existing median/reliability comparison and deterministic replay mean something different. Search changes which architectures are tried, not how a given architecture is scored.

Local mutation order is deterministic from the search seed and parent topology. An issued set prevents reevaluating the same topology. After 12 consecutive results that neither improve the leader nor add a frontier candidate, or after available elite neighbourhoods run out, the planner can perform an **elite-anchored restart**. It selects a current elite with the same fitness tournament, then changes one width to a seeded value anywhere in the configured width interval, inserts a layer with such a width, or removes a layer. Unchanged layers are inherited; Parent and generation are preserved and Mutation starts with `Elite restart`. No unrelated random topology replaces elite selection while a valid parent exists.

The only parentless recovery is bootstrap: if every evaluated candidate has failed, there is no valid parent to select. The minimum-size topology, then bounded random initialization, can supply the first usable parent. Once one valid candidate exists, all subsequent proposals, including restarts, have an elite parent. There are at most four restart rounds by default with at most 256 attempts each. Invalid, over-budget and already-issued children are rejected. There is no exhaustive fallback; failing to find an unseen mutation is not proof of global exhaustion.

The search ends with `NEIGHBOURHOODS_EXHAUSTED` if no more legal unseen children or allowed restart candidates are available. This is not a proof of a global optimum. `TRIAL_BUDGET`, `TIME_LIMIT`, `MEMORY_LIMIT` and `CANCELLED` remain distinct exits. In adaptive progress/results, `generated` counts proposals actually issued; `untested` concerns issued-but-unfunded proposals, **not all unseen topologies inside the bounds**. The UI labels these as proposals and does not claim the full search space was tested. Parameter-space completeness belongs only to the explicitly selected exhaustive reference mode.

Recommendation policy and strategy are frozen while a search is running; after completion the recommendation policy can still be changed to inspect alternative trade-offs. Search seed, restart interval and restart count are available under advanced settings. Cancellation, invalidation of stale sessions and scored-checkpoint replay retain their existing ownership rules.

The adaptive approach is inspired by mutation-based [evolutionary architecture search](https://arxiv.org/abs/1802.01548) and [multi-objective NAS](https://www.jmlr.org/papers/v25/23-1013.html), with fitness tournaments as documented in [DEAP's evolutionary selection operators](https://deap.readthedocs.io/en/master/api/tools.html#deap.tools.selTournament), but is a deliberately small Pareto-guided local search, not an implementation of AmoebaNet, LaMOO or NSGA-II. No speedup or superior final RMSE is assumed merely from using this strategy.

### Full-budget training, checkpoints and seed aggregation

Each architecture uses the same seed list, initially `1,42,123,999,2026`, and the same optimizer settings and dataset partition. The default is 10,000 epochs per seed, with scoring at initialization, every 25 epochs, and the final epoch. The active Studio's learning rate and momentum are captured with the request. Search does not tune those hyperparameters.

The normal Studio stops when its target is reached. Search intentionally does not:

~~~kotlin
while (epoch < config.maxEpochs) {
    if (cancelled()) return result(ArchitectureTrialState.CANCELLED)
    val training = model.trainEpoch()
    epoch++
    check(training.isFinite()) { "Training produced a non-finite RMSE." }
    if (epoch % config.checkEvery == 0 || epoch == config.maxEpochs) checkPoint()
}
~~~

Every completed seed therefore receives its full common epoch budget. A trial retains the best *evaluated checkpoint*, its training RMSE, final score, epoch, elapsed time, sample-update count and diagnostic parameter snapshot. Scores between evaluation checkpoints are not silently treated as measured. The architecture's score is the median of these best checkpoint scores over **all configured seeds**, including seeds that failed to reach the target. It is not the lowest single-seed error. The inspector initially selects the seed nearest the median; its image is a real seed checkpoint, not an averaged model.

The default recommendation requires at least four of five seeds to reach the threshold, as well as a median within the threshold. That is an observed reproducibility screen, not a confidence interval or a proof of an 80% probability of success. Numerical failures are recorded explicitly and exclude that architecture from final recommendations. Cancelled/incomplete seed groups remain provisional and are also excluded.

The Pareto frontier retains every fully evaluated, finite architecture not strictly dominated in parameter count and median RMSE. Equal objective values are retained as ties. A second reliable frontier is computed only among architectures meeting the target and success-count requirement. This prevents an unreliable architecture with a lower median from disqualifying a reliable recommendation merely because it dominates on two objectives. Frontier calculation sorts by parameter count and scans groups rather than doing an all-pairs dominance check on every repaint.

### Training fit versus validation

Boolean truth tables use **Training RMSE**. All four labeled points are used, and there is no claim that the continuous XOR heatmap has a uniquely defined ground truth away from those points.

For larger datasets, **Validation RMSE** creates one fixed split for the complete search:

~~~kotlin
val data = ArchitectureSearchData.split(
    samples,
    validationFraction = 0.2,
    seed = 42,
    label = "Circle"
)
~~~

The split requires at least ten distinct input coordinates and at least two coordinate groups in each observed target stratum. Groups are stratified around target 0.5. Duplicate coordinates, including differently labeled duplicates, always stay in the same partition. This avoids leaking the same input into both training and validation. Sampling is deterministic under the split seed, and the original order within the resulting partitions is preserved. A SHA-256 fingerprint covers the evaluation mode and actual ordered training/validation values.

Only the training partition is added as training samples. Validation samples are attached through `addTestSample` and scored through `testError`; despite the existing method name, they are **selection validation data**, not an untouched final test set. Best checkpoint selection and architecture ranking both consume these validation scores. Their winning error is consequently not an independent estimate of generalization. No holdout is regenerated for different architectures or seeds, and there is no undisclosed final-test feedback loop.

### Cancellation and model ownership

[`ExecutorCompletionService`](https://docs.oracle.com/en/java/javase/25/docs/api/java.base/java/util/concurrent/ExecutorCompletionService.html) delivers completed trials to one coordinator. At most `parallelism` trials are submitted but not yet received. Each newly admitted architecture reserves **all** configured seeds against the trial and eight-million-parameter checkpoint budgets; full-group funding cannot be exceeded by in-flight work. A budget or memory stop prevents new admissions but still completes already funded groups. A cancellation or deadline stops submission and drains active trials cooperatively, leaving unstarted seeds explicitly incomplete.

Each worker owns a fresh `Neuro`; there is no nested training pool and no concurrent weight mutation of one network. Globally unique trial IDs prevent different architectures using the same seed index from overwriting each other's progress. `TrialActivity` counts actual active evaluator calls and records peak overlap in both search strategies. The UI shows `Active trials: N/P`, distinct active architectures and peak utilization, instead of displaying only the first running trial as though it were the whole pool. Allocation and scoring are included in the evaluator lifetime; this is concurrency telemetry, not a CPU-percent or speedup measurement.

Seed results retain configured-seed order. Completed adaptive candidates retain **feedback order**; `lineage` retains **proposal order**, and each proposal records `evaluatedCount`, the length of the completed feedback prefix from which its parent was selected. This allows tests and consumers to reconstruct the actual eligible parent pool even when children finish out of order. A parent demoted after a child's submission does not retroactively invalidate that already-running child. Later proposals use the freshly ranked elite. Progress remains immutable and throttled to approximately ten updates per second, plus initial and final events.

Cancellation is cooperative between epochs. A cancelled search drains its submitted jobs, retains completed candidates and explicitly reports partial and untested counts. The pool is shut down in `finally`, including when a progress callback fails. Cancellation latency is bounded by the current epoch's work, not by a hard millisecond guarantee.

The UI has a search-specific generation token. Cancelling retains the token so the partial result can still be displayed; changing the active configuration or dataset invalidates it. Late callbacks from an invalidated job cannot repopulate results. Tab changes and purely visual inspections do not cancel the search. The active configuration and sample list are checked again when a queued search starts, so settings from an old display frame cannot be combined with a newly applied dataset.

### Inspect, apply and replay

The scatter plot uses parameter count horizontally and the selected RMSE vertically, with a target line and the Pareto frontier. Circles indicate fully evaluated candidates; squares show the current observed median of provisional groups and never enter the final recommendation. The table retains failed groups with explicit status. Clicking a point or a row opens per-seed inspection, including the output surface and training/selection curves. The seed drop-down can inspect every retained checkpoint.

**Apply architecture** starts a fresh model on the active full dataset. Epoch and metrics reset. It does not present a newly initialized network as if it already had the winning score.

**Replay selected run** constructs an independent model with the recorded architecture, seed, optimizer settings and training partition, trains to its best checkpoint epoch, verifies that the recorded selection RMSE is reproduced within `1e-10`, and only then installs it in the Studio. Cancellation before installation leaves the old model intact. Validation examples remain held out on replay. The Studio continues to label its main metric as training RMSE and separately displays the replay's selection metric and held-out count. Reset returns to the regular full-dataset Studio configuration.

The result distinguishes `COMPLETED` (reference mode), `NEIGHBOURHOODS_EXHAUSTED` (adaptive mode), `TRIAL_BUDGET`, `MEMORY_LIMIT`, `TIME_LIMIT` and `CANCELLED`. An absent recommendation is not silently replaced by a target-violating model; `bestError`, evaluated/partial/untested counts and numerical failures remain available.

### Verification and interpretation

Tests cover enumeration, parameter counts, defensive copies, grouped holdouts, deterministic single-thread/multithread results, ties, unreliable lucky seeds, all three policies, numerical failure handling, budgets, deadlines, cancellation, pool cleanup, stale generations and replay parity. Headless Swing tests exercise configuration errors, sorting, selection, policy changes, partial results and the real worker/control integration. The standard screenshot task now includes `search.png`, generated from an actual small architecture sweep.

A repeatable shallow XOR test (widths 1–4, five default seeds, 10,000 epochs each, default optimizer, training-fit scoring) found:

| Hidden widths | Parameters | Median best checkpoint RMSE | Successful seeds at 0.05 |
|---|---:|---:|---:|
| 1 | 5 | 0.40862641 | 0/5 |
| 2 | 9 | 0.01388392 | 4/5 |
| 3 | 13 | 0.01371784 | 5/5 |
| 4 | 17 | 0.01242154 | 5/5 |

The default policy chooses width two in this measured sweep. Increasing the required success count to five would instead require a different eligible candidate. These are results for this protocol, not a mathematical proof that no other initialization or training procedure could change a failure into a success.

Adaptive mutation is now the default; bounded exhaustive search remains an explicitly selected reference mode. Hyperband-style early screening/pruning is intentionally not enabled: eliminating slow-starting candidates would weaken the interpretation of a minimal-network experiment. Exhaustive means every architecture inside the configured, fully funded bounds was tested with the chosen seeds and training protocol—not that all possible weights or optimizers were searched.

Primary references for these boundaries are [Kotlin array equality and defensive copies](https://kotlinlang.org/docs/arrays.html), [ExecutorService lifecycle and interruption](https://docs.oracle.com/en/java/javase/25/docs/api/java.base/java/util/concurrent/ExecutorService.html), and [Cawley and Talbot on model-selection bias](https://www.jmlr.org/papers/v11/cawley10a.html). [Hyperband](https://www.jmlr.org/papers/v18/16-558.html) describes the resource-allocation approach deliberately left for a separate optional fast mode.

### Adaptive search regression checks

Focused tests prove that changing a child's measured RMSE changes the next proposal, that each newly promoted leader is used as a parent immediately, and that dominated or numerically failed candidates are not expanded. Other checks cover width/depth/order mutations, deterministic restarts, deduplication, defensive copies, bounded large-space execution without enumeration, serial trajectory repeatability, per-architecture numerical equality across worker counts, full trial budgets, storage limits, cancellation and observer-failure cleanup. A real shallow XOR adaptive sweep retains the expected two-neuron recommendation under the default five-seed protocol; it is a reproducible local result, not a global-search guarantee.

## Direct 2 → 1 baseline and PR #46 integration

Leaving **Hidden layers** empty (or whitespace-only) now selects the direct two-input,
one-output network with no hidden layer. It has three trainable parameters: two weights
and one bias. Removing the final hidden layer selects this baseline; adding a layer to
an empty editor starts with six neurons. Malformed comma-separated input is still rejected.

This restores the missing baseline from the original configurable-topology Java PR while
preserving the later Kotlin-only implementation, FlatLaf UI and architecture-search tools.
The newer Studio limits remain eight hidden layers and 128 neurons per layer; the earlier
Java visualizer's 64-total-neuron cap is not reintroduced. Search enumeration still uses
its configured positive hidden-layer bounds; the zero-hidden-layer baseline is selected
manually in the Studio rather than silently changing the search space.

~~~kotlin
val baseline = NeuroStudio(StudioConfig(hidden = ""))
val initial = baseline.frame()
check(initial.diagnostics.parameterCount() == 3)
check(initial.hiddenImages.isEmpty())
baseline.step(1)
baseline.advance()
~~~

The output graph, probe, learning curve, parameter/norm plots, update differences,
checkpoints and same-topology seed comparisons still operate. Probe contributions are
now the two input-weight products, not fabricated hidden activations. The Neurons view
shows a direct input/output graph and an explanatory empty state; hidden-layer paging
controls are disabled until a hidden layer is restored. Applying another topology resets
the model, history, snapshots and seed results together, as for all other configurations.

Regression tests cover blank input, layer editing, numerical parity with the actual
forward kernel in both sigmoid modes, snapshot isolation, output/difference maps,
training and seed-study isolation, deep-to-baseline resets, empty custom datasets,
and offscreen rendering of every Studio tab. The per-module coverage threshold is unchanged.


## Harder spatial learning sets

Six additional presets stress repeated decision boundaries, narrow curved regions and disconnected components. They are available in the existing Dataset selector, all Studio views and Architecture search. Circle, Spiral and the boolean generators retain their original samples and behavior; no optimizer, topology default or dependency changes are needed.

| Preset | Structure | Positive-label rule |
|---|---|---|
| Checkerboard 8×8 | 64 alternating cells | The sum of the two cell indices is odd. |
| Concentric rings | Six alternating radial bands inside radius 0.5 | Radius below 0.5 and `floor(12 * radius)` even; exterior is negative. |
| Tight spiral | Four windings out to radius 0.5, continuing into the square's corners | `sin(angle - 16 * PI * radius) >= 0`. |
| Twisted pinwheel | Eight angular sectors, twisted with radius | `sin(4 * angle - 12 * PI * radius) >= 0`. |
| Wave interference | Crossing, warped high-frequency waves | `wave(x, y) * wave(y, x) >= 0`. |
| 16 islands | A 4×4 array of disconnected disks | Distance from the local cell center below 0.075 in input coordinates. |

Radius and angle are measured around `(0.5, 0.5)`. In Concentric rings, the center is positive and successive bands alternate; the outermost band and the exterior are negative. Each island has radius 0.075 and neighboring centers are 0.25 apart, leaving a negative gap. The wave used by the interference preset is:

~~~kotlin
private fun wave(x: Double, y: Double): Double =
    Math.sin(10 * Math.PI * x + 2 * Math.sin(4 * Math.PI * y))
~~~

### Sampling and reproducibility

Every new preset contains exactly 1,024 binary-labeled points, one per cell of a 32×32 spatial grid. Sampling uses seeded jitter inside the central 80% of each cell. This provides spatial coverage without an unbounded rejection loop or duplicate coordinates. Targets are computed from the sampled coordinates; jitter changes positions, not the truth rule. There are no random label flips.

~~~kotlin
val random = SplittableRandom(seed)
val side = 32
val x = (index % side + 0.1 + 0.8 * random.nextDouble()) / side
val y = (index / side + 0.1 + 0.8 * random.nextDouble()) / side
Sample(x, y, if (positive(x, y)) 1.0 else 0.0)
~~~

`NeuroLearningSets.create(kind, seed)` reproduces the same ordered samples for the same seed. Different seeds change positions while preserving geometry. The Studio intentionally keeps its existing fixed dataset seed, `0xC0FFEE42L`; the sidebar's Random seed continues to control network initialization, not the learning set. This keeps architecture and initialization comparisons on identical data.

### Training and architecture search

Select a preset and press **Apply & restart**. The Learning set view overlays the actual targets; its background remains the network's prediction, not a reference solution. Hidden activation maps, parameter plots, checkpoints and seed comparisons use the same training path as before.

Architecture search defaults these larger, non-boolean datasets to **Validation RMSE**. It uses the existing fixed, grouped train/validation split. Duplicate-coordinate protection, model ownership and replay behavior are unchanged. Validation still participates in selection and is not an untouched final test set.

These examples are intentionally not guaranteed to reach RMSE 0.05 with the default six-neuron network. More local boundaries or more disconnected components motivate width/depth experiments; they do not define a universal difficulty ordering over all optimizers and seeds. Research on [spectral bias](https://proceedings.mlr.press/v97/rahaman19a.html) motivates the high-frequency cases, but does not establish a benchmark result for this particular sigmoid engine.

An epoch now processes 1,024 points rather than Circle's 180 or Spiral's 220. Large multi-seed architecture sweeps therefore cost more even before increasing network size. Start with a restricted architecture range and epoch budget, inspect held-out error, and expand deliberately. Do not interpret failure of one small network as proof that a dataset is unlearnable, or a lower raw RMSE on an imbalanced dataset as a universal difficulty ranking.

### Regression coverage

Tests check reproducibility across ordinary and extreme seeds, distinct input coordinates, complete grid coverage, finite binary targets, nondegenerate class counts, all 64 checkerboard cells and all 16 islands. Independent geometric oracles check ring labels, rotated complex-coordinate spiral/pinwheel labels and wave-phase signs. Integration tests train every preset, run validation architecture searches without changing the Studio model, verify the real Dataset selector and its default scoring mode, and render each preset through the implemented Swing Learning set view. Existing regression tests continue to cover the original datasets.


## GUI regression gate, unrestricted numeric editors and all-neuron scrolling

The Neurons tab now displays every neuron in every hidden layer in one vertically scrollable gallery. There is no first-neuron offset or eight-image page selector. A compact connection graph remains an explicitly labelled preview; it is not the complete gallery. Each layer has a heading and each neuron has a uniquely indexed activation card. The direct 2 → 1 baseline has an explanatory empty state.

`NeuronGallery.Layout` calculates layer sections and rows without allocating a Swing component or bitmap for every neuron. Painting requests only the cards intersecting the viewport. A single background renderer evaluates their activation maps from an immutable diagnostic snapshot, in contiguous batches per layer. A bounded latest-work queue and revision checks discard obsolete work after scrolling or model changes. Only visible images are retained; a reset invalidates the previous snapshot. The event thread paints ready images and placeholders, not neural inference. Headless screenshot generation uses the same geometry and a synchronous renderer. Closing the Studio shuts down the renderer as well as the model/search workers.

Numeric configuration editors now use `NumericInputs.spinner(value, step)` with neither a minimum nor a maximum in their `SpinnerNumberModel`. Its formatter preserves numeric precision and accepts scientific notation for floating-point values. Model arrow operations do not wrap at primitive type boundaries. The old 8-layer, 128-neuron, 1,000,000-epoch, learning-rate 10, 20-seed, 32-thread, 20,000-trial and one-day caps are removed from the corresponding validators as well as the controls. Restart controls no longer impose the old 1,000/100 bounds. The time-limit input retains a Long end to end rather than truncating through Int. Epochs per refresh is editable rather than a fixed three-choice preset.

This does not make invalid mathematics or impossible allocations valid. Widths and epoch counts must be positive, momentum remains in [0,1), target RMSE must be finite and non-negative, fractions must be strictly between zero and one, required successes cannot exceed the number of distinct seeds, and a parameter offset must identify an actual parameter. Numeric types, JVM array indexing and available heap still apply. Parameter arithmetic uses Long intermediates before conversion, epoch-step addition and timeout conversion cannot wrap, and model allocation is rejected with a diagnostic when its estimated working memory exceeds half the currently available heap. The exhaustive enumerator and retained-search-checkpoint resource guards remain independent runtime safeguards; removing editor caps is not a promise that astronomical searches can execute. Adaptive search can still use large configured bounds without enumerating them.

GUI verification has two separate dimensions. The usual JaCoCo module line gate remains at 90%; the blanket `NeuroXorCanvas` exclusion has been removed. A separate native-window suite, tagged `gui`, operates the real Studio with `AWT Robot` mouse and keyboard events. Fixture setup and component observations run on the event thread. Tests wait for visible state changes, not assumed training or rendering delays. The virtual desktop uses an explicit Openbox configuration so a machine's personal maximize-all policy cannot invalidate resize tests. Reports include a native screenshot per scenario and extra captures of the first/final neuron, large layers, and search results.

`GUI_PATHS.md` is the reviewed finite inventory of meaningful user-visible paths, including the entry points, branches and state transitions exposed by the controls. This is not a count of every possible loop iteration or arbitrary event sequence. `guiPathCoverage` maps each inventory row to the current JUnit XML result of a native-window test and requires at least 90%. Skipped, missing, failed and non-GUI tests do not count; negative verifier tests exercise these cases. The `guiCheck` task also fails if any executed GUI test fails, regardless of the ratio. GUI JaCoCo line/branch metrics are reported separately, without pretending that line coverage measures user journeys.

~~~text
./gradlew :jneuro:check :jneuro:jacocoTestReport
./gradlew :jneuro:guiCheck
~~~

The GUI task requires a graphical session and Python 3 for the standard-library-only path-report verifier. On Linux CI it runs under Xvfb with Openbox; on a graphical Windows or Linux workstation it uses the current desktop. A missing display is a failure, not a skipped suite. Do not interact with the desktop while native tests are running. CI uploads test reports, path coverage and screenshots even when the GUI job fails. The new policy in the repository's AGENTS.md requires the same 90% documented GUI-path gate for future GUI work.

The harder-dataset regression explicitly selects exhaustive reference mode when it expects `COMPLETED`; adaptive neighbourhood exhaustion is intentionally a different result. Tests must not confuse those termination semantics or weaken that distinction.

Primary API references: [Robot native input and EDT restrictions](https://docs.oracle.com/en/java/javase/26/docs/api/java.desktop/java/awt/Robot.html), [unbounded SpinnerNumberModel bounds](https://docs.oracle.com/en/java/javase/26/docs/api/java.desktop/javax/swing/SpinnerNumberModel.html), and [JaCoCo's independent line and branch counters](https://www.jacoco.org/jacoco/trunk/doc/counters.html).

The GUI path verifier uses Python 3 (`python` on Windows, `python3` elsewhere). Override the executable with `-PguiPython=/path/to/python` when needed. GUI test reports and screenshots are cleared before every run; native tests cannot be fulfilled by an up-to-date or cached result.

### Elite-parent regression verification

Tests assert policy-dependent rank ordering, reliability-frontier preservation, exclusion of failed/partial/dominated groups, deterministic tournament selection and a preference for fitter ranks without collapsing to a single parent. Planner tests reconstruct the elite pool from each completed prefix and verify every ordinary and restart descendant against it; old neighbourhoods cannot leak a demoted parent. The native GUI regression checks that the result table exposes the same valid ancestry. The registered GUI scenario catalog is maintained separately from line coverage.

### Parallel trial regression verification

The concurrency regression holds 32 offspring at a latch until 32 distinct worker threads have entered their evaluator simultaneously, with only one seed configured. A separate five-seed test exercises 32 concurrent slots across several architectures and verifies grouping, exact full-seed budgets and attribution. A slow-offspring test proves other architectures advance without waiting for it. Tests also verify out-of-order feedback, exclusion of partial parents, elite membership at proposal time, cancellation of all 32 workers, observer-failure cleanup, and accurate lower concurrency for a one-architecture budget.

The native GUI scenario GUI-097 enters 32 workers and a single seed through the real controls, starts adaptive search, asserts the live count reaches 32 across different architectures, cancels it, and checks the retained peak. The peak is limited by available independent tasks: bootstrap, an exhausted search space, a narrow budget, and completion of the final few trials can legitimately use fewer workers. A fixed pool does not create additional work merely because its capacity is 32; see the [Executor factory contract](https://docs.oracle.com/en/java/javase/25/docs/api/java.base/java/util/concurrent/Executors.html).

## Architecture search: separate execution from evaluation quality

Architecture search now has an independent **Execution** choice. `REFERENCE` retains the established epoch/cohort evaluator; `OPTIMIZED` removes intermediate scoring and changes how independent trials are scheduled. Strategy still chooses adaptive proposals or an exhaustive grid. Engine still chooses the general reference trainer or the supported SMALL family. Selecting optimized execution does not silently narrow a general-width search, change precision, approximate sigmoid, reduce the seed list, or stop a trial when it first meets the target.

The CPU bottleneck was larger than the arithmetic kernel. A search scores every 25 epochs by default, but calling the ordinary public epoch API also calculated training RMSE after every intervening epoch. A SMALL cohort additionally occupied one admission slot per seed while its CPU evaluator used one worker. Five admitted seeds could therefore leave most of a four-worker pool idle. Optimized CPU execution schedules each model independently. The existing cohort implementation remains available through reference execution; model-lane SIMD is not automatically selected without a demonstrated end-to-end advantage.

Progress snapshots iterate the concurrent activity map into a private list before sorting by submission index. Sorting the live entry collection directly was unsafe: Kotlin's single-element collection-copy shortcut could observe size one, then call `next()` after a worker removed that final entry. Copying through iteration avoids that race while retaining a weakly consistent view of running trials; final trial results still come from the completion queue.

The search-only advance interface makes the scoring contract explicit:

~~~kotlin
internal data class SearchAdvanceResult(
    val committedEpochs: Int,
    val rmse: Double?,
    val termination: TrainingTermination
)
~~~

A nullable RMSE means that no score was computed for this committed prefix. It never means that an earlier epoch's score can be reused. The evaluator requests only the distance to the next scoring boundary, advances in bounded chunks, then evaluates training and validation data. Each trial still runs its complete configured epoch budget unless cancellation, a time limit or a failure interrupts it. Public single-epoch and chunk methods retain their publication behavior.

~~~kotlin
private fun request() = TrainingChunkRequest(
    minOf(config.maxEpochs - epoch, config.checkEvery - epoch % config.checkEvery),
    cancelled = cancelled
)
~~~

Each `ArchitectureSearchData` owns immutable packed training and validation datasets shared by its freshly seeded models. Adding a sample through the public API detaches the affected model's sample list; exported public state remains defensive. SMALL's internal publication copies parameters and momentum while reusing immutable input/target arrays. Permutation validation reuses scratch storage, diagnostic capture performs one defensive parameter copy, and disabled training logs avoid building statistics and field maps. These changes target allocations and scoring overhead while preserving each model's optimizer and shuffle sequence.

### GPU admission, batching and publication

Optimized SMALL CUDA search owns a dedicated device queue. The CPU worker setting bounds CPU scoring work; up to 64 independent models may wait for or execute GPU training. Waiting models do not block the scoring executor. Progress reports distinguish active CPU workers, active/resident models, queued GPU requests and launched batches. Bootstrap still has only the initial architecture's funded seed group, so a 64-model limit does not imply that 64 useful models always exist.

The packaged CUDA ABI adds `search_train_fp64` and `search_train_fp32`. Each block receives its own parameter offset, topology offset, shuffle-order offset, layer count, epoch count and result offset. Immutable dataset buffers are shared. The existing same-topology public cohort entrypoints and atomic cohort contract remain intact. The new entrypoints reuse the same forward/backpropagation/update arithmetic, with one block per model; no native compiler installation is needed on the user's machine.

A numerical failure marks only the affected block and leaves that model's resident checkpoint unchanged. Host code validates returned parameters and momentum before committing shuffle advancement and counters. A driver failure invalidates the affected unpublished batch; replay or reopening starts from the last validated host checkpoint. CUDA requests remain capped at 64 epochs and adapt toward the 25 ms training budget, without crossing the caller's scoring boundary. GPU completion is followed by authoritative CPU scoring.

Trial results record actual execution route, kernel identity, device, precision and SIMD width. Replay uses that recorded route, including a one-model CUDA service for a queued GPU trial. Parallel adaptive search can observe completed parents in a different order after an optimization; that legitimately changes subsequent proposals. Serial reproducibility and shared architecture/seed numerical parity are separate requirements from parallel proposal order.

### Spiral measurement protocol

The command below measures complete searches through the actual scheduler. It includes opening sessions, training, scoring, snapshots, scheduling and cleanup; it is not a microbenchmark of a forward pass. Run separate JVM invocations with order offsets 0, 1 and 2 for warmed forks. Each JSONL record is flushed as it completes, so an interrupted experiment retains its evidence.

~~~text
./gradlew :jneuro:architectureSearchBenchmark --args="--epochs 2000 --workers 1,4,8,16,32 --warmups 1 --repeats 3 --order-offset 0 --output build/reports/search-benchmark/cpu-0.jsonl"
./gradlew :jneuro:architectureSearchBenchmark --args="--epochs 2000 --engine REFERENCE --workers 1,4,8,16,32 --output build/reports/search-benchmark/reference-engine.jsonl"
./gradlew :jneuro:architectureSearchBenchmark --args="--epochs 2000 --backends CUDA --workers 4 --output build/reports/search-benchmark/gpu.jsonl"
~~~

The frozen SMALL manifest includes width 8 at depths one through four and four mixed-width shapes. `--manifest general` exercises general-width CPU regressions. Every shape receives seeds `1,42,123,999,2026`, FP64, exact sigmoid, batch size 1, learning rate 0.6, momentum 0.2 and beta 1. Spiral's fixed 220 points are split with seed 42 into 176 training and 44 validation samples. The separate `JNeuro architecture search benchmark` workflow runs three CPU forks on existing Java/Gradle CI toolchains.

Full-budget quality is a different experiment:

~~~text
./gradlew :jneuro:architectureSearchBenchmark --args="--mode quality --executions OPTIMIZED --workers 32 --epochs 1000000 --trials 60 --search-seeds 42,123 --limit-seconds 7200 --output build/reports/search-benchmark/quality.jsonl"
~~~

This protocol starts from `2→8→8→8→1`, permits one to four SMALL hidden layers with widths 4/8/16 and at most 881 parameters, and funds complete five-seed groups. A reliable winner needs validation RMSE at most **0.01 in at least four of five seeds**. Sixty trials allow twelve fully funded candidates. The wall-time limit applies across the invocation and preserves partial results; it does not convert partial candidates into reliable recommendations. A bounded experiment can finish without meeting the target and cannot establish a global minimum architecture.

Only after selecting a reliable winner does the harness evaluate its representative checkpoint on 218 independent, interleaved Spiral-arm points. Their fractions `(i + 0.5) / 109` lie between the original generator's points. That final RMSE and classification accuracy are reported without feeding them back into selection.

Reports include completed epochs and sample updates, actual worker/model peaks, GPU batch counts, best checkpoint parameters, per-trial route timings and JVM allocated bytes. Route timings overlap across concurrent trials; GPU training phase includes queue wait and publication, and CPU training phase includes any boundary RMSE computed by the advancement API. They are not additive kernel timings. Uninstrumented legacy cohort phases are `null`, not fabricated zero-duration measurements. JVM allocation totals exclude native CUDA allocations. Fixed-work throughput and quality success must be interpreted independently.

The handwritten Java benchmark drivers still use their existing training-state helper signatures. Kotlin default arguments alone do not create Java overloads: `@JvmOverloads` retains the zero-argument state export and two-argument chunk commit while Kotlin search code opts into shared datasets and deferred scoring explicitly. CI compiles the JMH Java source set as well as the Kotlin application and tests.

### Measured search throughput on Spiral

The retained [local measurement report](https://github.com/lisu188/jexperiments/blob/192058a18036c69fae81d43e69a4024b3721ddd3/jneuro/benchmarks/2026-10-01-search/LOCAL.md) separates execution speed from learning quality. Nine sequential JVMs at compute revision `529e7a9` supplied three warmed forks for each CPU engine and CUDA, with nine measured calls per execution mode and configuration. Every call completed the same eight architectures and five seeds: 40 trials, 80,000 epochs and 14,080,000 sample updates. The machine was an Intel Core i5-14400F with 16 logical processors, an RTX 4060 Ti and OpenJDK 27. The existing Studio stayed open; logs were disabled in benchmark JVMs.

| Same-engine comparison | Reference median | Optimized median | Speedup | Reference p95 | Optimized p95 |
|---|---:|---:|---:|---:|---:|
| General CPU, 32 workers | 6.053 s | 4.024 s | 1.504x | 6.990 s | 4.570 s |
| SMALL CPU, 32 workers | 11.521 s | 4.358 s | 2.644x | 36.368 s | 7.672 s |
| SMALL CUDA, 4 scoring workers | 53.335 s | 12.462 s | 4.280x | 63.633 s | 12.937 s |
| SMALL CPU, 1 worker | 15.382 s | 16.625 s | 0.925x | 74.239 s | 68.586 s |

The strongest measured prior CPU baseline was the general engine at 32 workers, making **1.504x** the appropriate overall CPU comparison. CUDA improved its own baseline by **4.280x**, but still took 2.56 times as long as optimized general CPU at four workers, and 3.10 times as long as the fastest measured CPU configuration. SMALL with one worker became 8.1% slower. Ten of eleven matched configurations passed the predeclared complete-work, arithmetic-parity, 10% median-improvement and non-regressing-p95 gates. These results support an explicit optimized option, not an unconditional default switch.

All measured best-parameter snapshots, best/final RMSE values and best epochs matched exactly between reference and optimized execution within each engine/backend. Cross-engine and cross-backend trajectories differed after 2,000 epochs; general CPU reductions, ordered SMALL FMA and backend sigmoid implementations are separate arithmetic modes. These comparisons do not prove identical million-epoch convergence. None of the fixed-work benchmark candidates reached RMSE 0.01.

The third SMALL JVM had substantial late slowdowns, retained in every statistic. Their cause remains unknown; the available point sample does not establish contention, throttling or a JIT failure. Allocation totals also require qualification: SMALL CPU at 32 workers decreased from 148.3 to 20.6 MiB per search, while the faster CUDA queue increased JVM allocation from 168.4 to 226.2 MiB, about 34.3%. These are cumulative JVM allocations, not retained heap or device memory.

The [CI report](https://github.com/lisu188/jexperiments/blob/192058a18036c69fae81d43e69a4024b3721ddd3/jneuro/benchmarks/2026-10-01-search/CI.md) is deliberately inconclusive for throughput qualification: its three initial jobs landed on different CPU models. Their results cannot be pooled into three compatible forks. The workflow now runs the three forks sequentially on one host. The local report retains full reconstructible numerical outcomes and provenance; final-source smoke tests and later full-budget quality runs carry their own revisions and must not be substituted for the measured matrix.


### Full-budget Spiral result: no reliable winner in the bounded run

The two-hour measurement window ended after the fixed-work matrix and separate long-budget searches. These quality invocations used compute revision `7a0a1ab`, SMALL, FP64, exact sigmoid, one million epochs per training seed, and a 60-trial budget. All five seeds must complete before a candidate can qualify; at least four saved best-checkpoint validation RMSE values must be at most 0.01. The implementation never shortened a trial after its first threshold crossing.

| Quality invocation | Full million-epoch trials | Cancelled partial trials | Complete five-seed candidates | Reliable candidates |
|---|---:|---:|---:|---:|
| Local CPU, 8 workers | 42 | 8 | 8 | 0 |
| Local CUDA, 4 scoring workers | 5 | 55 | 1 | 0 |
| Separate CI CPU, 4 workers | 46 | 4 | 8 | 0 |

Every invocation reached its deadline during search seed 42; search seed 123 never started. There were zero failed trials. All completed five-seed candidates had zero threshold successes. No reliable winner was selected, so the independent Spiral test was not run. The local CPU and GPU quality jobs ran concurrently, while CI used a separate EPYC host; their completion counts and elapsed times are not a controlled CPU/GPU throughput comparison and cannot be pooled to complete candidate groups.

CUDA did produce a concrete partial lead: `2→8→8→4→1`, with 137 parameters. Training seed 42 reached validation RMSE **0.00295294** at epoch 101,875; seed 123 reached **0.00931634** at epoch 9,750. However, all five trials were cancelled around 101,000–104,000 epochs, and only two seeds crossed the threshold. The seed-123 score later rose to 0.07548. These observations satisfy neither the four-success requirement nor the full-budget requirement, and establish no smallest reliable architecture. On cancellation, the last reported score belongs to the latest scoring boundary and may precede the final committed epoch.

The [complete quality report](https://github.com/lisu188/jexperiments/blob/192058a18036c69fae81d43e69a4024b3721ddd3/jneuro/benchmarks/2026-10-01-search/QUALITY.md) retains every trial's scores, checkpoint epoch, counters, state, process exit, and source/runtime provenance. Its strict analyzer accepts consistent partial evidence while explicitly leaving completion and reliability false. Keeping those separate prevents a successful JVM exit or CI job from becoming an unsupported learning-quality claim.

The [validation audit](https://github.com/lisu188/jexperiments/blob/192058a18036c69fae81d43e69a4024b3721ddd3/jneuro/benchmarks/2026-10-01-search/VALIDATION.md) records 304 passing CI unit tests and 96.92% line coverage, plus 26 native-control GUI tests covering all 131 documented paths. Final Windows checks passed 303 tests with one optional native-BLAS skip and 96.93% line coverage. Eighteen real CUDA acceptance tests passed at the separately recorded training-kernel revision. The short GUI Spiral fixture verifies controls, budgets, cancellation and replay with a 0.9 target; it is independent of the 0.01 quality experiment. These results support delivery of the explicit optimized execution path while preserving the reference default and the unresolved quality target.
