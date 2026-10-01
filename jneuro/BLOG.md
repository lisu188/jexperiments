# JNeuro: TensorFlow mathematics behind a Kotlin learning studio

## One numerical implementation

JNeuro is a dense sigmoid-network experiment with a Swing learning studio and a reproducible architecture search. Kotlin owns the application: datasets, seeded random streams, shuffle reservations, model ownership, epoch counters, cancellation, checkpoints, ranking, replay, logging and rendering. TensorFlow owns network evaluation, activation functions, backpropagation, gradient reduction, momentum updates, RMSE and numerical neuron diagnostics.

The old scalar, Java Vector API, native AVX2, native BLAS, handwritten CUDA and cuBLAS implementations have been replaced. Historical benchmark reports remain evidence for those earlier revisions; their throughput and device descriptions do not describe the TensorFlow implementation.

The boundary is `TensorFlowMath`. TensorFlow types do not appear in the Swing, architecture-search or public model contracts. This is TensorFlow's JVM API called from Kotlin, rather than a second application framework. KotlinDL also wraps TensorFlow, but the lower-level JVM API lets this experiment preserve its exact weight layout, additive momentum state, activation options and explicit publication boundaries without introducing another model representation.

The official [TensorFlow Java project](https://github.com/tensorflow/java) documents Kotlin as a supported JVM client. JNeuro pins [TensorFlow Java 1.1.0](https://github.com/tensorflow/java/tree/v1.1.0), whose underlying TensorFlow runtime is 2.18.0. This is deliberate: 1.1 is the last release with native Windows CPU binaries. TensorFlow Java 1.2 dropped that platform. The build resolves only the native artifact for the current platform instead of downloading every supported platform.

Kotlin 2.4.20 and JDK 27 remain the application toolchain, with JVM bytecode targeting 26. The repository's existing Gradle wrapper is retained. Swing, Java2D and FlatLaf continue to provide the desktop interface; numerical work needs no display.

## Public model and compatibility

A model still accepts a topology and explicit hyperparameters:

```kotlin
val model = Neuro(
    intArrayOf(2, 8, 8, 8, 1),
    Neuro.HyperParameters(0.5, 0.2, 1.0, 42)
)
NeuroLearningSets.addTo(model, NeuroLearningSets.create(NeuroLearningSets.Kind.XOR, 0))
```

Weights remain flattened in `[output, input]` order. Host snapshots use `DoubleArray` for both precision modes so existing serialization, inspection and replay structures can remain unchanged. In FP32 mode, TensorFlow variables, activations, gradients and momentum are actually float tensors; converting a published float value into a double container does not turn training into FP64.

The public JVM API remains usable from Java, as in the benchmark drivers:

```java
try (NeuroTrainingSession session = network.newTrainingSession(
        TrainingBackend.CPU, precision, 1, TrainingEngine.SMALL)) {
    session.trainEpoch();
}
```

`CPU` explicitly places the graph on `/device:CPU:0`. `CUDA` and the retained `CUBLAS` compatibility value select TensorFlow's GPU device. `AUTO` initially chooses CPU. `REFERENCE` and `SMALL` no longer identify separate numerical implementations; SMALL keeps its supported topology-family validation for existing experiment configurations. Likewise, old SCALAR/VECTOR kernel hints remain source-compatible but TensorFlow selects its own CPU kernels.

Session metadata reports `tensorflow-<runtime-version>-dense-v1`, the actual CPU/GPU device, precision, activation mode and zero application-managed SIMD bits. Zero does not mean TensorFlow uses no SIMD; JNeuro does not inspect or claim the implementation details of TensorFlow's native kernels.

## Graph construction and forward inference

The numerical adapter constructs a TensorFlow graph from named operations. Inference supplies inputs and parameters as tensors, while a retained training graph owns mutable TensorFlow variables. Every dense layer uses TensorFlow matrix multiplication, bias addition, beta scaling and activation:

```kotlin
values += activation(op("Mul", scalar(hp.beta),
    op("AddV2", matmul(values.last(), weights[layer], transposeB = true), biases[layer])), hp.sigmoidMode)
```

The transpose matches the existing output-major host layout. Inputs have shape `[batch, inputs]`, weights `[outputs, inputs]` and activations `[batch, outputs]`. Dynamic leading dimensions allow ragged final mini-batches and differently sized visualization tiles to use the same graph.

Inference graphs are cached by topology, beta, activation, precision and device. The cache retains at most eight idle-or-leased entries. A graph in use cannot be evicted; an overflow request gets a temporary graph that closes after its call. Parameters are fed for each invocation, so different models with the same topology can share immutable graph structure without sharing weights.

This avoids attaching a native session to every model, thread-local inference wrapper or historical snapshot. Tensors and result containers close deterministically after each call. Training sessions own their graph and dataset tensors explicitly and release them on close.

## Backpropagation and optimizer semantics

The existing loss scaling matters. The update is equivalent to half the mean over samples of the sum of squared output errors. Averaging over every output element instead would change the effective learning rate for multi-output networks.

JNeuro expresses backpropagation with TensorFlow tensor operations. Output deltas are `(target - output) * beta * output * (1 - output)`. Hidden deltas multiply the next layer's deltas by its old weights, then apply the same activation derivative. Matrix multiplication produces weight gradients; reduction over the sample axis produces bias gradients.

```kotlin
val gradient = definition.op("RealDiv",
    definition.matmul(deltas[layer], activation[layer], transposeA = true), count)
val biasGradient = definition.reduce("Mean", deltas[layer], intArrayOf(0))
```

Both are averaged by the actual mini-batch size, including a short final batch. TensorFlow then computes the additive momentum update:

```kotlin
val velocity = definition.op("AddV2",
    definition.op("Mul", definition.scalar(hp.momentum), variables[kind + 2][layer]),
    definition.op("Mul", definition.scalar(hp.learningRate), change))
next[kind][layer] = definition.op("AddV2", variables[kind][layer], velocity)
```

The velocity buffer stores the update actually added to the parameter, including learning-rate scaling. It is exported alongside weights and biases, so closing and reopening does not silently reset momentum.

A control-dependency barrier ensures that all new parameters and velocities are computed and checked for finiteness before any assignment begins. Otherwise one layer could read another layer's newly updated weights during the same backpropagation step. TensorFlow `CheckNumerics` failures poison the private training kernel; the caller must reopen from the last committed host checkpoint.

Exact sigmoid remains the default. FAST retains the previous polynomial forward approximation, expressed entirely using TensorFlow operations. Its backward derivative remains the historical sigmoid `beta * a * (1-a)`, rather than differentiating the polynomial. This is why the graph explicitly expresses deltas instead of applying automatic differentiation indiscriminately to both activation modes. Approximate math remains an explicit experiment.

## Training state and transaction boundaries

Kotlin still owns the replayable shuffle stream. A chunk reserves orders without advancing the committed stream. The TensorFlow kernel gathers rows using those indices, performs one graph execution per mini-batch and keeps parameters and momentum in its session between calls.

```kotlin
runtime.session.runner().feed(dataset, inputTensor).feed(targets, targetTensor)
    .feed(order, rows).addTarget("train").run().close()
```

At publication, all parameter and momentum buffers return to the host. The model validates the complete state before committing reserved shuffle steps, copying parameters, advancing counters and publishing RMSE. A failed computation cannot advertise unpublished epochs as completed progress.

Single-epoch APIs retain immediate publication. Chunk APIs remain bounded by the requested epoch count, a maximum of 64 epochs, cancellation checks and an adaptive time budget. Search stops at its scoring/checkpoint boundaries. Cancellation happens between completed numerical calls; it does not interrupt an in-flight native TensorFlow operation or free its buffers prematurely.

This migration does not preserve the old custom CUDA one-launch-per-epoch optimization. TensorFlow receives one mini-batch execution at a time. The retained-session path avoids rebuilding graphs and reinitializing weights, but very small online networks can still be dominated by native call overhead. New throughput measurements must distinguish startup, retained sessions, ordinary epoch publication and whole-search time.

## Architecture search and cohorts

Architecture generation, evaluation policies, seed ordering, bounded worker queues, scoring cadence and candidate ranking remain Kotlin. Each active model owns independent TensorFlow variables and optimizer state. Shared datasets are immutable; independent trials never share mutable parameters.

CPU search uses the existing bounded worker orchestration. TensorFlow sessions are configured with one intra-operation and one inter-operation worker so every search worker does not create another large numerical pool. The GPU-facing queue retains bounded ownership and asynchronous completion, but its computation now uses TensorFlow sessions rather than a heterogeneous custom CUDA kernel.

Full-budget search still runs the requested epoch budget for each trial and records scores at configured boundaries. Convergence reporting, replay and application of selected architectures continue to use detached snapshots. Resetting results clears the ranking cache and references to old checkpoints as well as the visible table.

## Studio diagnostics without a second network

Previously the snapshot renderer implemented its own forward pass. That would leave two mathematical implementations even after training migrated. Snapshot evaluation now calls TensorFlow and can fetch all intermediate activations, enabling hidden-neuron views without recomputing a different network.

Pixel inputs are packed in stable row-major order and evaluated in bounded tiles. Tiles are capped at 4,096 pixels and further reduced for wide hidden layers, keeping fetched intermediate values bounded. The output, difference and hidden-neuron maps therefore avoid one native call per pixel.

Neuron probes use TensorFlow for input-weight products, preactivation sums and parameter norms. Coordinate transforms, color selection, text layout and hit testing remain ordinary Kotlin/Java2D. Keeping that distinction prevents tensor APIs from spreading into application code that does not perform neural-network calculations.

## CPU, GPU and native dependencies

Ordinary Windows and Linux CPU use resolves the matching TensorFlow native artifact automatically:

```text
./gradlew :jneuro:runXorCanvas
```

On Linux x86_64, including WSL, selecting the GPU-enabled artifact is explicit:

```text
./gradlew :jneuro:runXorCanvas -PneuroTensorFlowGpu=true
./gradlew :jneuro:gpuCheck -PneuroTensorFlowGpu=true
```

GPU execution additionally requires a compatible NVIDIA driver, CUDA and cuDNN for the pinned TensorFlow runtime. Follow the versioned TensorFlow installation requirements rather than the removed custom-kernel toolchain. These are hardware-bound entrypoints; the build does not install drivers or system libraries automatically.

The capability probe runs an explicitly placed TensorFlow matrix multiplication with soft placement disabled. Selecting GPU fails when that operation cannot run on the device; it does not silently fall back to CPU. CPU and GPU artifacts for the same platform must not both be placed on the runtime classpath.

`CUBLAS` remains a legacy API value; it does not load JNeuro's former cuBLAS adapter. Similarly, old command-line benchmark variant names are accepted as compatibility aliases, and reports identify the actual TensorFlow runtime and selected device. The current JMH benchmarks compare TensorFlow call patterns, precision and activation choices.

## Validation and reproducibility

The module retains the repository's 90% line-coverage requirement and separate 90% native GUI-path requirement. Numerical tests use analytic expected values, multi-output loss scaling, non-unit beta, momentum continuation, FP32 rounding, ragged batches and explicit order handling. Session tests cover ownership, cancellation, rejected arguments, failed chunks, reopening and transactional publication. Diagnostic tests cover batched ordering, detached snapshots and tile boundaries.

Linux CI runs the full build, JNeuro checks, rendering and native Robot GUI paths under Xvfb. A Windows CPU workflow verifies the platform intentionally retained by the pinned dependency. GPU acceptance remains explicit and requires actual hardware; a green CPU workflow is not evidence that GPU training ran.

Useful focused commands are:

```text
./gradlew :jneuro:check :jneuro:jacocoTestReport
xvfb-run -a ./gradlew :jneuro:guiCheck
./gradlew :blogsite:build
./gradlew clean build
```

The GUI command also needs a window manager as configured in CI. Test reports, GUI coverage, screenshots and precise source revisions distinguish numerical correctness, desktop behavior and performance evidence. Historical custom-engine reports under `benchmarks/` remain associated with their original commits and are not relabeled as TensorFlow measurements.
