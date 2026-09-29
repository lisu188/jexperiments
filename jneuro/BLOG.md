# JNeuro: A Kotlin Neural-Network Engine and Learning Studio

## Scope and runtime

JNeuro is a small dense feed-forward neural network implemented directly on primitive arrays. The complete module is now Kotlin: the training engine, Vector API kernels, optional native BLAS adapter, diagnostics, desktop application, correctness tests, and JMH benchmarks. There is no retained handwritten Java implementation in `jneuro/src`; generated JMH Java harnesses are build outputs only.

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

Mini-batch training accumulates gradients and applies a single averaged update. Parallel workers use private gradient buffers and workspaces. Their gradients are reduced in a fixed worker order rather than racing writes into shared model parameters.

Training and test errors remain root-mean-square error over all samples and output dimensions:

~~~text
RMSE = sqrt(sum((target - prediction)^2) / (sampleCount * outputCount))
~~~

An empty dataset reports `NaN`, not an invented zero error. Training requires samples, a finite target, and a bounded epoch budget. Statistics count actual completed epochs and samples.

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

JUnit tests also cover malformed topologies including trailing commas, defensive copies, later-layer activation maps, boundary clipping, empty datasets, terminal states, cancellation, bounded history, seed-study isolation, and UI control/worker interaction. The UI smoke test renders all seven implemented Swing views to PNG; these are application renders, not generated design mockups.

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

The renderer works without a graphical desktop and writes the seven application views under `jneuro/build/screenshots`. The normal interactive launcher requires a display. Save PNG exports the currently visible studio panel.

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

JNeuro remains an educational dense network, not a tensor framework. It still uses sigmoid activations, lacks automatic differentiation, softmax/cross-entropy, regularization, persistence, GPU execution, and adaptive optimizers such as Adam. Circle and Spiral can expose the limits of a narrow network and the training budget rather than guaranteeing a low error.

Training-set fit is not generalization evidence. The studio does not silently manufacture a validation set or claim calibrated probabilities. Its role is to expose the actual computation and make controlled architecture experiments easier to run and inspect.
