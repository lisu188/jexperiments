# JNeuro: A Small Feed-Forward Network in Plain Java

## Why this experiment exists

JNeuro is a deliberately small neural-network implementation built directly on primitive Java arrays.

There is no tensor framework, no matrix library, no automatic differentiation, and no optimizer abstraction. Forward propagation, backpropagation, momentum, bias updates, error measurement, initialization, and inference buffers are all visible in normal Java code.

The original implementation was useful as a sketch, but it mixed educational simplicity with several correctness and engineering problems:

- the output-layer delta applied the sigmoid derivative to target-minus-output rather than to the activation,
- neurons had no bias parameters,
- initialization and training order were nondeterministic,
- training-until-error had no maximum epoch limit,
- test error divided by the number of training samples instead of test samples,
- public inference API was effectively missing,
- the model stored weights in nested three-dimensional arrays with poor locality,
- every optimization decision was hard to measure,
- there were no deterministic correctness tests.

The modern version keeps the explicit array-level implementation while making the algorithm mathematically conventional, deterministic, measurable, and reusable.

## Model shape

A network is created from a topology:

~~~java
var network = new Neuro(
        new int[]{2, 6, 1},
        Neuro.HyperParameters.defaults()
                .withLearningRate(0.6)
                .withMomentum(0.2)
                .withSeed(42));
~~~

The topology is defensively copied and validated.

Every adjacent pair of layer sizes creates one internal Layer.

A layer owns:

~~~java
double[] weights;
double[] biases;
double[] weightVelocity;
double[] biasVelocity;
~~~

Weights are flat rather than double[][][].

For an output neuron o and input i:

~~~text
weightIndex = o * inputs + i
~~~

That removes one level of object indirection from the hot loops and gives every neuron's incoming weights one contiguous region.

## Xavier initialization

Weights use a deterministic SplittableRandom seed.

For a layer with fan-in and fan-out:

~~~java
var limit = Math.sqrt(
        6.0 / (inputs + outputs));
~~~

Each weight is sampled uniformly from:

~~~text
[-limit, +limit)
~~~

Biases start at zero.

This is a simple Xavier/Glorot-style initialization suitable for a sigmoid experiment and is substantially better than a fixed Math.random() - 0.5 range for every topology.

The seed is part of HyperParameters, so verification and performance experiments can reproduce exactly the same starting model.

## Correct forward propagation

Input is copied into the first activation buffer.

Each following layer computes:

~~~text
z = bias + sum(input[i] * weight[i])
activation = sigmoid(beta * z)
~~~

The implementation stores one activation array per layer in a reusable Workspace.

The innermost dot product uses Math.fma and a four-element unroll:

~~~java
sum = Math.fma(
        source[inputIndex],
        layer.weights[offset + inputIndex],
        sum);
~~~

The remainder loop handles input counts not divisible by four.

This keeps the implementation explicit while reducing loop overhead and giving the JVM a simple contiguous numeric kernel to optimize.

## Stable sigmoid

The logistic function is implemented in two branches:

~~~java
if (value >= 0.0) {
    return 1.0 / (1.0 + Math.exp(-value));
}

var exp = Math.exp(value);
return exp / (1.0 + exp);
~~~

The negative branch avoids computing exp(-value) for a very large negative number.

That prevents unnecessary overflow while preserving the usual sigmoid result.

## Correct output delta

The original code contained its most important mathematical bug here.

It effectively calculated a derivative from:

~~~text
target - output
~~~

But the helper derivative formula:

~~~text
a * (1 - a)
~~~

expects the sigmoid activation a.

The corrected output delta is:

~~~java
var activation =
        outputActivation[output];

outputDelta[output] =
        (target[output] - activation)
        * sigmoidDerivativeFromActivation(
                activation);
~~~

With configurable sigmoid steepness beta:

~~~java
return beta
        * activation
        * (1.0 - activation);
~~~

That keeps the error term and activation derivative conceptually separate.

## Hidden-layer backpropagation

For each hidden unit, the implementation accumulates the weighted deltas of the next layer:

~~~text
delta_hidden =
    sigmoidDerivative(hiddenActivation)
    * sum(delta_next * weight_to_next)
~~~

The current delta buffer is reused each sample.

There is no allocation of temporary vectors during backpropagation.

The algorithm walks layers from output toward input and uses the already-computed forward activations.

## Biases

Every non-input neuron now has a bias.

The forward pass starts each output accumulator with:

~~~java
var sum = layer.biases[output];
~~~

The backward pass updates bias velocity exactly like weight momentum, except without multiplying by an input activation.

Adding biases is not cosmetic.

Without them, every neuron's decision surface is forced through the origin. Even simple boolean functions become unnecessarily difficult or impossible for small topologies.

## Momentum

Momentum is represented as velocity rather than as a copy of the previous complete weight matrix.

For one weight:

~~~java
var velocity = Math.fma(
        momentum,
        layer.weightVelocity[weightIndex],
        learningRate
                * outputDelta
                * source[input]);

layer.weightVelocity[weightIndex] =
        velocity;

layer.weights[weightIndex] += velocity;
~~~

This is both clearer and cheaper than preserving full previous weights only to subtract them at the next epoch.

Biases use the same rule.

## Deterministic online SGD

Training samples are copied into the model when added:

~~~java
network.addTrainingSample(
        new double[]{0, 1},
        new double[]{1});
~~~

Each epoch creates no shuffled List copy.

Instead JNeuro keeps a reusable int[] containing sample indices and shuffles that primitive array in place with a deterministic SplittableRandom.

The training order therefore changes each epoch while remaining reproducible for a fixed model seed.

Each sample is still applied immediately, so this is online/stochastic gradient descent rather than accumulated mini-batch training.

## Reused workspaces

Training owns one Workspace.

Inference uses:

~~~java
ThreadLocal<Workspace>
~~~

A Workspace contains activation arrays and delta arrays sized exactly for the topology.

This removes per-inference intermediate array allocation.

The normal convenience method:

~~~java
double[] predict(double[] input)
~~~

allocates only the returned output array.

For high-throughput code, callers can provide the destination:

~~~java
network.predictInto(input, output);
~~~

That path performs no result allocation and reuses the calling thread's Workspace.

Concurrent inference is therefore supported across threads.

Training mutates weights and is intentionally not concurrent with training or inference. JNeuro does not pretend to provide a synchronized model-update protocol.

## Training API

One epoch:

~~~java
double error = network.trainEpoch();
~~~

A fixed number of epochs:

~~~java
network.train(100);
~~~

Training to a target with a hard limit:

~~~java
var result =
        network.trainUntil(
                0.05,
                10_000);
~~~

TrainingResult reports:

~~~java
record TrainingResult(
        int epochs,
        double error,
        boolean converged)
~~~

A bad topology or hyperparameter configuration can no longer trap the caller in an unbounded while loop.

## Error metric

Both trainingError() and testError() use one global root-mean-square error:

~~~text
sqrt(
    sum((target - output)^2)
    / (sampleCount * outputCount)
)
~~~

The test-set calculation uses the number of test samples.

That fixes the original test() bug, which divided test error by teachers.size().

An empty dataset returns NaN rather than inventing a meaningful score.

## OR and XOR verification

The deterministic verification harness trains small models on both OR and XOR.

OR verifies that:

~~~text
00 -> 0
01 -> 1
10 -> 1
11 -> 1
~~~

XOR verifies:

~~~text
00 -> 0
01 -> 1
10 -> 1
11 -> 0
~~~

The tests require actual convergence below a fixed RMSE and also check the resulting predictions against low/high thresholds.

This catches algorithm errors that a compilation test or one decreasing loss value would miss.

The verification suite also covers:

- invalid topologies,
- sample dimension validation,
- NaN rejection,
- deterministic seed behavior,
- defensive copies,
- predict versus predictInto equivalence,
- parameter counting including biases,
- test-set error calculation,
- maximum epoch enforcement,
- finite sigmoid output for extreme finite inputs.

## Statistics

The model exposes a lightweight state snapshot:

~~~java
record Statistics(
        long epochsTrained,
        long samplesSeen,
        double lastTrainingError)
~~~

This is useful for experiments and profiling without adding logging to the inner loops.

## Aggressive execution engine

The current engine keeps the simple dense-network API but has several execution paths selected independently from model semantics.

Weights remain flat row-major primitive arrays. Training examples are lazily packed into contiguous input and target buffers, so the hot path does not chase Sample objects. Forward propagation reads the caller input directly and can write the final layer directly into a caller-provided destination.

For repeated inference, callers can retain the workspace explicitly:

~~~java
var session = network.newInferenceSession();
session.predictInto(input, output);
~~~

This avoids both intermediate allocation and the ThreadLocal lookup used by the convenience API.

The AUTO, SCALAR and VECTOR kernel modes allow direct benchmarking and reliable fallback. VECTOR uses the JDK Vector API for dense dot products, hidden-delta propagation, output deltas, momentum updates and mini-batch gradient accumulation. AUTO keeps tiny layers scalar and vectorizes wider layers.

Fixed-epoch training no longer evaluates the entire training set after every epoch. train(n) computes the reported RMSE only after the final epoch. trainUntil retains exact checking by default and also exposes a checkEvery overload for workloads where evaluating the loss every epoch would dominate training time.

Hidden-layer backpropagation initializes its delta buffer from the first next-layer weight row instead of clearing the buffer and then accumulating into it. Sample counters are updated once per epoch instead of once per sample.

## Batch and parallel execution

Inference supports contiguous row-major batches:

~~~java
session.predictBatch(inputs, batchSize, outputs);
~~~

A reusable ParallelInferenceSession partitions a batch deterministically over a persistent ForkJoinPool and retains one workspace per worker.

Training supports mini-batches:

~~~java
network.trainMiniBatch(epochs, batchSize);
network.trainMiniBatch(epochs, batchSize, parallelism);
~~~

Parallel mini-batch workers accumulate private gradient arrays. Their gradients are reduced in a fixed worker order before one momentum update, avoiding concurrent writes to model parameters.

## Fast sigmoid

SigmoidMode.EXACT uses Math.exp.

SigmoidMode.FAST uses range reduction around powers of two plus a fifth-order polynomial approximation of exp. It avoids Math.exp while preserving substantially better numerical accuracy than a low-order direct sigmoid approximation. It remains opt-in because approximation changes floating-point results.

## Float inference

toFloatModel() creates a compact inference snapshot with float weights, biases and workspaces. The float path uses FloatVector when the configured kernel selects SIMD.

This halves parameter and activation storage versus double inference and doubles the preferred SIMD lane count on the same vector width.

## Native BLAS experiment

NeuroNativeBlas is an optional Java Foreign Function and Memory API backend for batched inference.

It dynamically looks for cblas_dgemm in common OpenBLAS, BLAS and MKL library names. Weights are transposed and copied off-heap once when a native session is created. Input/output scratch buffers are reused across calls, and dense layers execute as row-major GEMM operations.

The backend is optional: normal JNeuro has no native library dependency. Running it requires native access for the unnamed module.

## Performance tooling

NeuroInferenceJmhBenchmark compares, in the same fork and on the same topology:

~~~text
legacy copied-buffer scalar kernel
new scalar kernel
Vector API kernel
AUTO kernel
explicit InferenceSession
fast-sigmoid SIMD session
float SIMD inference
~~~

NeuroBatchJmhBenchmark compares vector batch inference, reusable parallel batch inference and float batch inference over several batch sizes.

NeuroTrainingJmhBenchmark compares fixed-epoch scalar training, SIMD training, SIMD mini-batch training and deterministic parallel mini-batch training.

A benchmark-only LegacyNeuroBaseline preserves the previous forward-path structure so the main comparison does not rely on results collected on a different machine.

NeuroNativeBlasBenchmark checks native output against the Java vector backend before timing CBLAS on medium and large batched networks.

Useful tasks are:

~~~text
:jneuro:jmh
:jneuro:jmhCompare
:jneuro:jmhBatchCompare
:jneuro:jmhTrainingCompare
:jneuro:nativeBlasBenchmark
:jneuro:performanceMatrix
:jneuro:profileNeuro
~~~

The module still retains the lightweight NeuroBenchmark and NeuroPerformanceMatrix utilities for fast local regression checks. JMH remains the source for performance comparisons.

## JFR

JNeuro defines disabled-by-default JFR events for individual training epochs and complete trainUntil runs. The profiling task is:

~~~text
./gradlew :jneuro:profileNeuro
~~~

The recording is written below build/jfr/.

## Java 27 build

The module targets Java 27, enables jdk.incubator.vector explicitly and compiles with strict warnings:

~~~text
-Xlint:all,-incubating
-Werror
~~~

The normal API remains pure Java. The native CBLAS benchmark additionally runs with:

~~~text
--enable-native-access=ALL-UNNAMED
~~~

## Remaining limitations

JNeuro remains an educational dense feed-forward network rather than a general machine-learning framework. It still has only sigmoid activations, no softmax or cross-entropy objective, no regularization, no model persistence, no GPU backend, no gradient-check utility and no adaptive optimizer such as Adam.

Those omissions are deliberate boundaries rather than hot-path limitations: the numeric kernels remain visible as ordinary Java code and can be compared directly against SIMD, float, batch and native variants.



## Live XOR learning dashboard

JNeuro includes an AWT Canvas dashboard that exposes how a small 2-6-1 network constructs XOR rather than only showing the final classifier.

The main heatmap keeps the original interpretation: x and y are the two inputs over the range 0 to 1, and pixel brightness is the network output. Black is 0, white is 1, and intermediate activations are gray.

Below it, six smaller heatmaps show the activation surface of every hidden neuron. A red line overlays each neuron's first-layer z = 0 boundary, so movement of its weight vector and bias is visible geometrically while SGD trains.

A live network diagram shows the same snapshot structurally. Connection thickness is proportional to absolute weight magnitude, blue and orange distinguish positive and negative weights, and node brightness is the activation for the point currently selected on the main heatmap. Moving the mouse across the heatmap therefore acts as a forward-pass inspector.

The diagram also shows each hidden neuron's signed contribution to the output pre-activation. This makes it possible to see which hidden features are reinforcing or suppressing the output for a selected point.

A training-history chart records RMSE plus f(0,0), f(0,1), f(1,0), and f(1,1). The separation of the four XOR cases is visible alongside loss convergence instead of being reduced to the latest scalar values.

Pause, single-epoch step, ten-epoch step, resume, and reset controls make the training trajectory inspectable rather than forcing a fixed-speed animation.

Training, output rendering, parameter capture, and hidden-map generation all happen on the same worker thread. The AWT event thread only receives immutable post-update snapshots, so visualization never performs inference against weights while they are being mutated.

Run it with:

~~~text
./gradlew :jneuro:runXorCanvas
~~~
