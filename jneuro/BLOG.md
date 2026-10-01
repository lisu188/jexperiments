# JNeuro: TensorFlow mathematics behind a Kotlin learning studio

## One numerical implementation

JNeuro is a dense sigmoid-network experiment with a Swing learning studio and a reproducible architecture search. Kotlin owns the application: datasets, seeded random streams, shuffle reservations, model ownership, epoch counters, cancellation, checkpoints, ranking, replay, logging and rendering. TensorFlow owns network evaluation, activation functions, backpropagation, gradient reduction, momentum updates, RMSE and numerical neuron diagnostics.

The old scalar, Java Vector API, native AVX2, native BLAS, handwritten CUDA and cuBLAS implementations have been replaced. Historical benchmark reports remain evidence for those earlier revisions; their throughput and device descriptions do not describe the TensorFlow implementation.

The boundary is `TensorFlowMath`. TensorFlow types do not appear in the Swing, architecture-search or public model contracts. This is TensorFlow's JVM API called from Kotlin, rather than a second application framework. KotlinDL also wraps TensorFlow, but the lower-level JVM API lets this experiment preserve its exact weight layout, additive momentum state, activation options and explicit publication boundaries without introducing another model representation.

The official [TensorFlow Java project](https://github.com/tensorflow/java) documents Kotlin as a supported JVM client. JNeuro pins [TensorFlow Java 1.2.0](https://github.com/tensorflow/java/releases/tag/v1.2.0), whose underlying TensorFlow runtime is 2.21.0. This release no longer publishes native Windows binaries, so Windows users run the application through Linux/WSL. The build resolves only the native artifact for the current platform instead of downloading every supported platform.

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

Session metadata reports `tensorflow-<runtime-version>-dense-loop-v2`, the actual CPU/GPU device, precision, activation mode and zero application-managed SIMD bits. The version distinguishes the functional training loop from the earlier `dense-v1` graph that executed each mini-batch through a separate JVM call. Zero SIMD bits does not mean TensorFlow uses no SIMD; JNeuro does not inspect or claim the implementation details of TensorFlow's native kernels. Device identity is currently the logical TensorFlow placement, such as `GPU:0`, rather than a physical GPU UUID. Replay checks the recorded runtime/settings and reproduced score, but metadata alone does not establish identical physical hardware across machines.

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
    definition.op("Mul", definition.scalar(hp.momentum), previous[kind + 2][layer]),
    definition.op("Mul", definition.scalar(hp.learningRate), change))
next[kind + 2][layer] = velocity
next[kind][layer] = definition.op("AddV2", previous[kind][layer], velocity)
```

The velocity buffer stores the update actually added to the parameter, including learning-rate scaling. It is exported alongside weights and biases, so closing and reopening does not silently reset momentum.

Here `previous` contains the complete parameter and momentum tensors from the preceding loop iteration. The body returns the next values together, so a layer cannot observe another layer's updated weights during the same backpropagation step. Every returned parameter and velocity passes through TensorFlow `CheckNumerics`. A numerical failure poisons the private training kernel; the caller must reopen from the last committed host checkpoint.

Exact sigmoid remains the default. FAST retains the previous polynomial forward approximation, expressed entirely using TensorFlow operations. Its backward derivative remains the historical sigmoid `beta * a * (1-a)`, rather than differentiating the polynomial. This is why the graph explicitly expresses deltas instead of applying automatic differentiation indiscriminately to both activation modes. Approximate math remains an explicit experiment.

## Moving the epoch loop into TensorFlow

Tiny online networks expose the cost of crossing the JVM/native boundary. In the first TensorFlow implementation, 220 samples with batch size one required 220 `Session.run()` calls for an epoch, followed by another call to export the final state. Retaining the graph avoided reconstruction, but each sample still repeated runner setup, feeds, execution and cleanup.

`TrainingGraph.buildTrainingLoop` now constructs condition and body functions with TensorFlow Java's `ConcreteFunction` and `Signature` APIs, imports their definitions into the retained graph, and invokes them with a functional `While`. Temporary graphs and function handles close after their definitions are copied. The loop's graph size depends on network topology; it does not unroll operations for every sample or requested epoch.

The loop carries a cursor, the batch limit, dataset tensors, the flattened order array and all four groups of parameter state: weights, biases and their respective momentum buffers. Its condition compares the cursor with the order length. The body gathers the current mini-batch, executes the existing tensor update and returns the advanced cursor with the complete next state. `parallel_iterations` is explicitly one. Since [TensorFlow declares `CheckNumerics` stateful](https://github.com/tensorflow/tensorflow/blob/v2.21.0/tensorflow/core/ops/array_ops.cc), the operation is `While`, not `StatelessWhile`; this preserves the checks inside the body.

Kotlin still owns the replayable shuffle stream. It reserves complete epoch permutations without advancing the committed stream, then concatenates them without reordering samples:

```kotlin
val flattened = IntArray(orders.size * source.samples)
orders.forEachIndexed { epoch, indices ->
    indices.copyInto(flattened, epoch * source.samples)
}
```

Flattening must preserve mini-batch boundaries too. Five samples with batch size three must produce batches of `3, 2, 3, 2` across two epochs. A naive loop over all ten indices would produce `3, 3, 3, 1` and change both gradients and momentum. The TensorFlow body clips each batch to the remaining samples in its epoch:

```kotlin
val count = body.intOp("Minimum", args[1],
    body.intOp("Sub", samples, body.intOp("FloorMod", position, samples)))
```

Online training supplies a batch limit of one. Larger requested batches are clamped to the dataset size, including `Int.MAX_VALUE`, before creating the scalar tensor. Argument validation also rejects malformed permutations and an order chunk too large for the int32 cursor before execution. An empty order list exports the current state without running an update, including for an empty dataset.

Every nonempty `TrainingGraph.train(orders, batchSize, online)` now uses one `Session.run()` for the complete supplied chunk and its final state export:

```kotlin
repeat(4 * source.weights.size) { runner.fetch("training_loop:${5 + it}") }
runner.feed(dataset, inputTensor).feed(targets, targetTensor)
    .feed(order, rows).feed(batch, batchTensor).addTarget("train").run()
```

The first five loop outputs are cursor, batch limit, inputs, targets and order; the fetched outputs contain only the parameter and momentum state. Graph/session initialization and requested RMSE or visualization evaluations remain separate operations. One JVM execution call therefore describes the training boundary, not a promise of one GPU kernel launch: TensorFlow still schedules the operations inside the loop.

## Training state and transaction boundaries

Persistent TensorFlow variables retain the last completed chunk between calls. Their values enter the loop through `Identity` operations, while intermediate mini-batches carry their state as loop outputs. The graph assigns the final values back only after the whole loop, including every numeric check, succeeds:

```kotlin
definition.node("Assign",
    listOf(variables[kind][layer], "$loop:${5 + kind * layers + layer}"),
    mapOf("T" to Definition.type(definition.dtype),
        "use_locking" to Definition.bool(true)))
```

The `train` target depends on all final assignments. Fetching loop outputs in the same run yields the state used for those assignments without another export call. A failure in a later mini-batch cannot publish the earlier mini-batches of that chunk. Native execution or export failures invalidate the kernel; recovery imports the host checkpoint rather than trusting private native state after an error.

At host publication, the model validates every parameter and momentum buffer before committing reserved shuffle steps, copying parameters, advancing counters and publishing RMSE when requested. Search can defer RMSE until a scoring boundary. A failed computation cannot advertise unpublished epochs as completed progress.

Single-epoch APIs retain immediate publication. Public chunk orchestration remains bounded by the requested epoch count, a maximum of 64 epochs, scoring boundaries and an adaptive time budget. A public `trainChunk` may invoke the numerical kernel more than once as that orchestration measures progress. Ordinary session search explicitly submits one epoch at a time to poll cancellation between epochs; the asynchronous search queue retains its bounded adaptive chunks. These choices determine cancellation granularity even though the mini-batch loop now executes inside TensorFlow.

Cancellation and time budgets are checked between completed numerical calls. They do not interrupt an in-flight loop or free its tensors early. A requested stop can therefore wait for the current submitted chunk to finish and publish; the budget is not a hard native-execution deadline. Closing the asynchronous service still drains outstanding work before releasing the owning session.

## Measured CPU effect

The [TensorFlow epoch-loop report](benchmarks/2026-10-01-tensorflow-loops/README.md) records a local before/after comparison using the same retained-session harness, datasets and seeds. For CPU FP64 with exact sigmoid and online updates over 220 Spiral samples, the ordinary `trainEpoch()` measurements include final host publication and RMSE:

| Topology | Previous median per epoch | Functional loop median | Speedup |
| --- | ---: | ---: | ---: |
| `2 → 6 → 1` | 30.04 ms | 5.88 ms | 5.11× |
| `2 → 8 → 8 → 8 → 1` | 29.15 ms | 8.94 ms | 3.26× |

Five concurrent trials with a 176-training/44-validation split improved from 117.89 to 264.46 aggregate epochs per second, a 2.24× gain. The timed work uses the actual retained-session search advancement route and final training RMSE. It excludes session initialization, validation scoring, snapshots and UI work, so it measures training throughput rather than the elapsed time of a complete architecture sweep.

The comparison used two JVM processes per implementation, reversing baseline/optimized execution order in the second pair. Each process ran three repeats with five warm-up epochs and twenty measured epochs per case. Each timing is the mean over those twenty epochs; the table reports the median of six such observations. These are local measurements: for example, the small-network online timings ranged from 24.01–41.53 ms before and 5.58–7.03 ms after. They are not a throughput guarantee for another machine or workload. All 102 exported comparison checkpoints matched bit for bit, covering 16,044 parameter and momentum values along with RMSE, progress counters and the next shuffle order. Independent analytic tests also cover both precisions, both activation modes, changing batch modes, ragged tails, empty exports and failures after an earlier mini-batch succeeds.

Session construction costs more because it now exports the two function definitions before opening the retained runtime: the median across measured openings rose from 13.43 to 21.95 ms. The benefit belongs to repeated training through that session; the report also records the slower first opening in each process separately. These measurements use CPU execution. GPU operation placement remains explicit, but a faster JVM call pattern does not establish GPU throughput or imply that TensorFlow fuses an entire epoch into one GPU kernel. Default batch size, precision, activation and search budgets are unchanged by this optimization.

## Architecture search: a model dimension in TensorFlow

`ArchitectureExecution.BATCHED` changes the numerical unit of work from a model session to a population tensor. `REFERENCE` and `OPTIMIZED` remain available for comparisons and preserve existing API defaults. Studio starts new searches with batched execution and an explicit **Prune weak trials** budget policy. Selecting **Full budget** retains each admitted trial's requested epoch budget. Neither choice silently changes precision, sigmoid, learning rate, momentum or samples per mini-batch.

The earlier cohort and GPU queue were orchestration mechanisms: they still called separate model sessions. A wide worker pool could therefore multiply scheduling overhead without combining the tiny matrix operations. The new `TensorFlowCohortGraph` uses one leading tensor dimension per independent model. For a layer with padded input width `I`, output width `O`, sample batch `S` and resident model count `M`, the important shapes are:

| Tensor | Shape |
| --- | --- |
| Activations | `[M, S, I]` |
| Weights and weight momentum | `[M, O, I]` |
| Biases and bias momentum | `[M, 1, O]` |
| Output and deltas | `[M, S, O]` |
| Training or validation score | `[M]` |

[TensorFlow's BatchMatMulV2 documentation](https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/batch-mat-mul-v2) defines products of matrix slices with broadcast batch dimensions. That directly fits independent model states and a shared dataset. The [TensorFlow authors' vectorization paper](https://arxiv.org/abs/1903.04243) motivates combining corresponding operations across independent invocations; JNeuro constructs the model axis explicitly through TensorFlow Java rather than relying on a Python vectorization wrapper.

The gradient matrix multiplies over samples, while the bias reduction uses only the sample axis:

```kotlin
val gradient = graph.op("RealDiv",
    graph.batchMatmul(deltas[layer], activations[layer], transposeA = true), count)
val biasGradient = graph.reduce("Mean", deltas[layer], intArrayOf(1), keepDims = true)
```

There is no model-axis reduction. Seeds retain separate weights, biases, momentum, shuffle sequences, epoch counts and stopping state. Forward inference, sigmoid derivatives, gradient accumulation, momentum updates, finite-state checks and RMSE remain TensorFlow operations; Kotlin owns shapes, indices, scheduling and presentation.

### Padding different widths without inventing neurons

Models with the same hidden depth can share a cohort even when their widths differ. The scheduler pads each layer to the widest corresponding layer, splits batches when padding would exceed four times the actual parameter count, and estimates resident tensor storage separately from retained snapshot storage. The 128 MiB admission guard covers padded optimizer copies, activations, datasets and bounded shuffle workspaces; it is not a cap on total TensorFlow or process memory. Native runtime and allocator overhead remain additional costs. Different depths use separate graphs and advance in rounds, so the first admitted topology does not monopolize the search.

Zero weights alone are insufficient padding: exact sigmoid at zero is one half. Forward propagation masks each layer's activations after the sigmoid:

```kotlin
val mask = DoubleArray(size * caps[layer + 1]) { index ->
    if (index % caps[layer + 1] < source[index / caps[layer + 1]].topology[layer + 1]) 1.0 else 0.0
}
values += graph.op("Mul", output,
    graph.constant(mask, longArrayOf(size.toLong(), 1, caps[layer + 1].toLong())))
```

Padded activations and their derivatives therefore contribute zero to following layers and gradients. Export removes padding and preserves the original topology. Tests compare every parameter and momentum buffer with independent TensorFlow sessions across mixed widths, both precisions, both sigmoid modes and ragged sample batches.

### Native state between checkpoints

A retained cohort graph contains current and best parameter/velocity tensors, immutable training and validation data, and best scores. It owns a private TensorFlow worker pool, with one intra-operation worker by default and one inter-operation worker. The recorded kernel version is `tensorflow-<runtime>-batched-v1-t<intraOpThreads>`, so replay can reject a changed numerical configuration. Increasing native threads is an explicit benchmark parameter; it is not coupled to the legacy search-worker spinner. One `advance` call supplies independent shuffle indices and an active-lane mask. Its functional `While` gathers each lane's mini-batch and updates all active models together. One JVM call is **not** a claim of one CUDA kernel launch: TensorFlow still schedules the operations inside that loop.

The active mask selects the complete optimizer state, including momentum. Each mini-batch checks whether every parameter and velocity of a lane remains finite. If any update fails, that lane becomes inactive for the remaining chunk, and final publication selects its original chunk-start state:

```kotlin
val committed = definition.reshape("$loop:6", intArrayOf(size, 1, 1), DataType.DT_BOOL)
val assigns = variables.take(4 * layers).indices.map { index ->
    definition.assign(variables[index],
        definition.select(committed, "$loop:${7 + index}", initial[7 + index]))
}
```

Successful siblings can commit without inheriting a failed lane's state. Host shuffle reservations advance only for successful lanes. A native execution exception poisons that cohort instead of publishing uncertain progress.

Scoring produces vectors of training and validation RMSE. A scoring mask restricts best-checkpoint updates to trials actually due for evaluation; another lane reaching a boundary must not give its siblings extra selection opportunities. Best states stay in TensorFlow until a cohort must be regrouped, a trial finishes, or a selected checkpoint is exported. Bulk export fetches selected lanes together. No per-model prediction session or parameter download is required for ordinary scoring.

### Broad exploration and bounded successive halving

`PopulationArchitectureProposals` starts with the current eligible architecture, the minimum topology, diverse depth/width combinations and seeded exploration. It can admit many proposals before any candidate has finished. Complete seed groups are funded together; `maxTrials` counts admitted architecture/seed pairs rather than only survivors.

The optional pruning policy uses successive-halving stages inspired by [Hyperband](https://keras.io/keras_tuner/api/tuners/hyperband/). The initial budget is clamped to the configured maximum. At a stage boundary, groups compare median validation scores from the same epoch budget, retain approximately one in `reductionFactor` groups, and multiply the survivors' budget by that factor. Ties use stable architecture ordering. A group may finish early when its recorded median and required number of successful seeds meet the requested target.

This implementation uses deterministic streaming brackets, not a claim to reproduce the full Hyperband or ASHA algorithms. Decisions occur after a stable round of numerical batches. Each round advances all live lanes by a deterministic common milestone delta; measured execution time only divides that fixed work into smaller native calls. Faster hardware therefore does not choose a different proposal order. Pruned and completed groups free capacity for new brackets while long-running survivors continue. Regrouping exports current and best state before reopening bounded cohorts, and the benchmark includes that cost. This avoids replacing the original first-candidate barrier with a barrier around an entire population.

A pruned trial has a distinct `PRUNED` terminal state. Its candidate is not fully evaluated and cannot be recommended, applied or replayed as a completed result. Early pruning can discard a slow-starting architecture; **Full budget** remains the appropriate comparison when that tradeoff is undesirable. The policy is distinct from ordinary training convergence and never changes the single-epoch Studio controls.

Native chunks stop at a scoring boundary, a stage boundary or 64 epochs, whichever comes first. An observed per-epoch duration reduces subsequent chunks toward a 50 ms execution budget. One epoch can itself exceed that budget, so it is a responsiveness target rather than a hard deadline. Cancellation is checked between native calls. Ordinary progress publications are throttled to 10 Hz; initial, final and membership-change updates publish immediately. They report actual training calls, active models per batch and aggregate model-epochs per elapsed second.

### Replaying a regrouped population

A replay needs more than the seed. Padding, capacity and lane position can affect a native kernel's arithmetic path. Each trial records consecutive `ArchitectureBatchSegment` entries with committed epoch bounds and the exact tensor geometry used during that interval. Replay recreates those geometries with one active recorded lane and inactive placeholders, carries parameters and momentum between segments, and advances the original shuffle sequence.

```kotlin
val active = BooleanArray(segment.capacity) { it == segment.lane }
val order = model.reserveTrainingOrders(count)
val failed = kernel.advance(
    Array(segment.capacity) { if (active[it]) order else emptyArray() }, active)
check(failed.size == segment.capacity && !failed[segment.lane])
model.commitTrainingChunk(kernel.exportState(segment.lane), count, evaluateError = false)
```

At the best epoch, replay uses the same batched scoring graph. An FP32 best checkpoint at epoch zero explicitly publishes the native rounded parameters and momentum without advancing counters or shuffle; merely returning its native score would leave the original FP64 initialization installed. Recomputing an FP32 or padded checkpoint through ordinary FP64 inference would compare different arithmetic. Device, precision, activation and numerical-kernel provenance must still match exactly, and the reproduced score must pass the existing strict tolerance before Studio installs the model. Geometry is validated for continuity and bounded memory before native allocation.

### Measured whole-search performance

The [population search report](benchmarks/2026-10-01-batched-search/README.md) compares this implementation with the frozen epoch-loop implementation from commit `11ed5d8`, using the same Spiral split, five seeds, FP64, exact sigmoid, online updates and full 50-epoch budgets. Two warmed JVM forks per implementation reverse the execution order in the second pair. Each case has four observations; elapsed time includes initialization, graph opening, scoring, snapshots, regrouping and close.

| Full-budget workload | Previous search median | Batched search median | Improvement |
| --- | ---: | ---: | ---: |
| Five seeds, `2→6→1` | 2,278.526 ms | 710.147 ms | 3.21× |
| Six mixed architectures × five seeds | 13,366.273 ms | 2,605.592 ms | 5.13× |

All 140 compared best checkpoints agreed: 4,220 parameter values differed by at most `7.11e-15`, RMSE differed by at most `1.11e-16`, and completed epochs and selected best epochs matched exactly. The focused numerical tests separately compare complete momentum state. These are local CPU results, not GPU measurements. The report retains ranges and startup observations: the slowest batched first-result observation in the mixed case was 3,962.600 ms versus 3,638.436 ms for the previous implementation, so the throughput improvement is not a guarantee that every first result arrives sooner.

The pruning experiment is separate because it deliberately changes the training budget. Across two search seeds, a 150-epoch cap and 60 admitted trials, full-budget batched search committed 9,000 model-epochs in 15.056/17.349 seconds. Pruning committed 3,875 model-epochs in 8.544/4.060 seconds and pruned 45 of 60 trials. Its best median validation RMSE was slightly worse: `0.463514857` versus `0.463122755`. Neither policy reached the requested `0.05` target. This short experiment demonstrates reduced work and its quality tradeoff; it does not establish time to convergence or certify an optimal Spiral architecture.

An [exploratory cohort matrix](benchmarks/2026-10-01-batched-search/cohort/README.md) separates retained training from graph startup and varies model count and native thread count. It has one JVM and two observations per setting, so it guides further experiments rather than supporting a universal thread recommendation. The default remains one intra-operation thread; actual bucket size and topology matter more than simply increasing the thread count.

## Studio diagnostics without a second network

Previously the snapshot renderer implemented its own forward pass. That would leave two mathematical implementations even after training migrated. Snapshot evaluation now calls TensorFlow and can fetch all intermediate activations, enabling hidden-neuron views without recomputing a different network.

Pixel inputs are packed in stable row-major order and evaluated in bounded tiles. Tiles are capped at 4,096 pixels and further reduced for wide hidden layers, keeping fetched intermediate values bounded. The output, difference and hidden-neuron maps therefore avoid one native call per pixel.

Neuron probes use TensorFlow for input-weight products, preactivation sums and parameter norms. Coordinate transforms, color selection, text layout and hit testing remain ordinary Kotlin/Java2D. Keeping that distinction prevents tensor APIs from spreading into application code that does not perform neural-network calculations.

## CPU, GPU and native dependencies

Linux CPU use, including WSL on Windows, resolves the matching TensorFlow native artifact automatically. Run these commands from a Linux shell with the repository's JDK installed inside Linux. The Studio needs a graphical session, such as WSLg:

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

Linux CI runs the full build, JNeuro checks, rendering and native Robot GUI paths under Xvfb. Native Windows execution is unsupported by the pinned TensorFlow release; use Linux/WSL instead. GPU acceptance remains explicit and requires actual hardware; a green CPU workflow is not evidence that GPU training ran.

Useful focused commands are:

```text
./gradlew :jneuro:check :jneuro:jacocoTestReport
xvfb-run -a ./gradlew :jneuro:guiCheck
./gradlew :blogsite:build
./gradlew clean build
```

The GUI command also needs a window manager as configured in CI. Test reports, GUI coverage, screenshots and precise source revisions distinguish numerical correctness, desktop behavior and performance evidence. Historical custom-engine reports under `benchmarks/` remain associated with their original commits and are not relabeled as TensorFlow measurements.
