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

JNeuro remains an educational dense network, not a tensor framework. It still uses sigmoid activations, lacks automatic differentiation, softmax/cross-entropy, regularization, persistence, GPU execution, and adaptive optimizers such as Adam. Circle and Spiral can expose the limits of a narrow network and the training budget rather than guaranteeing a low error.

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

The initial budget is 10,000 seed trials; the maximum is 20,000. An adaptive run reserves a complete seed group before evaluating the next architecture. It checks retained checkpoint storage before funding each new group and stops with `MEMORY_LIMIT` before exceeding eight million parameter values. The default adaptive strategy does not call `architectures()` and can explore spaces too large for exhaustive enumeration. The optional **Exhaustive · reference** strategy preserves the original grid search: it rejects more than 4,096 candidates or 100,000 visited prefixes and funds a deterministic parameter-ordered prefix of complete seed groups. Checkpoint curves are compacted to at most 128 entries per seed while preserving the initial, final and current best checkpoints. These limits bound stored model state; actual JVM overhead also includes objects, datasets and temporary training workspaces.

Depth/width controls support the Studio's existing limits of eight hidden layers and 128 neurons per layer, subject to parameter and storage limits. Adaptive mode previews a seed-trial budget and starting architecture, not a fictitious total candidate count. An optional wall-clock limit is a safety stop, not a reproducibility promise: it can leave different subsets evaluated on different machines.

### Adaptive mutation of completed leaders

The Studio supplies its active hidden widths as the initial architecture. If those widths lie outside the selected search bounds (including the direct `2 → 1` baseline when hidden layers are required), the planner explicitly starts from the smallest legal architecture instead. Editing the sidebar alone still does not affect the active search input.

The coordinator evaluates one architecture at a time, parallelizing only its independent seed trials. It waits for the entire seed group before asking the planner for another proposal:

~~~kotlin
val proposal = planner.next()
val candidate = evaluateAllSeeds(proposal.architecture)
planner.observe(candidate)
~~~

This is schematic control flow: `searchAdaptive` owns the completion service and `evaluate` performs each actual training trial. The search engine does not use an epoch-zero score, fastest-finishing seed or partial result as a parent-selection signal. Worker scheduling therefore does not determine the next architecture.

After each completed group, `ArchitectureRanking` computes the recommendation, best-error candidate, ordinary Pareto frontier and reliability-filtered frontier. A new recommendation (or best-error leader before a reliable solution exists) becomes the highest-priority parent. Every newly admitted Pareto candidate also contributes a neighbourhood. Other surviving parents rotate through a queue; candidates dominated out of both frontiers stop generating children. A failed seed group is never a parent.

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

Mutation order is deterministic from the search seed and parent topology. An issued set prevents reevaluating the same topology. After 12 consecutive results that neither improve the leader nor add a frontier candidate, or after available elite neighbourhoods run out, the planner can perform an exploration restart. The first restart tries the minimum-size architecture if it has not been evaluated; subsequent restarts use bounded seeded random sampling. There are at most four restarts by default, each with at most 256 attempts to draw a legal unseen topology. There is no exhaustive fallback. A missed narrow feasible region is possible and is not reported as global exhaustion.

The search ends with `NEIGHBOURHOODS_EXHAUSTED` if no more legal unseen children or allowed restart candidates are available. This is not a proof of a global optimum. `TRIAL_BUDGET`, `TIME_LIMIT`, `MEMORY_LIMIT` and `CANCELLED` remain distinct exits. In adaptive progress/results, `generated` counts proposals actually issued; `untested` concerns issued-but-unfunded proposals, **not all unseen topologies inside the bounds**. The UI labels these as proposals and does not claim the full search space was tested. Parameter-space completeness belongs only to the explicitly selected exhaustive reference mode.

Recommendation policy and strategy are frozen while a search is running; after completion the recommendation policy can still be changed to inspect alternative trade-offs. Search seed, restart interval and restart count are available under advanced settings. Cancellation, invalidation of stale sessions and scored-checkpoint replay retain their existing ownership rules.

The adaptive approach is inspired by mutation-based [evolutionary architecture search](https://arxiv.org/abs/1802.01548) and [multi-objective NAS](https://www.jmlr.org/papers/v25/23-1013.html), but is a deliberately small Pareto-guided local search, not an implementation of AmoebaNet or LaMOO. No speedup or superior final RMSE is assumed merely from using this strategy.

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

`ExecutorCompletionService` manages outstanding seed trials. Adaptive mode submits at most one full seed group (up to 20 trials) and uses a bounded worker count; reference mode retains its original bounded outstanding-trial queue. Each worker owns a fresh `Neuro`; there is no nested parallel training pool and no inference against a network another thread is updating. Seed results are assembled in canonical configured-seed order. Adaptive candidates retain their proposal order; reference candidates retain parameter order. Completion order does not alter ranking or adaptive lineage. Progress contains copied lists and immutable diagnostic snapshots, published at most about ten times per second apart from the initial/final events.

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

Focused tests prove that changing a child's measured RMSE changes the next proposal, that each newly promoted leader is used as a parent immediately, and that dominated or numerically failed candidates are not expanded. Other checks cover width/depth/order mutations, deterministic restarts, deduplication, defensive copies, bounded large-space execution without enumeration, serial/parallel lineage equality, full trial budgets, storage limits, cancellation and observer-failure cleanup. A real shallow XOR adaptive sweep retains the expected two-neuron recommendation under the default five-seed protocol; it is a reproducible local result, not a global-search guarantee.

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
