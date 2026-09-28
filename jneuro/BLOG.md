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

## Performance layout

The main hot-path design choices are:

- flat weights per layer,
- contiguous incoming weights per neuron,
- reusable activation/delta arrays,
- primitive shuffled indices,
- no temporary hidden vectors,
- Math.fma dot products,
- four-way inner-loop unrolling,
- reusable predictInto output,
- no copied previous-weight tensor.

The model still uses one Layer object per edge between topology levels because that keeps dimension metadata explicit without putting object indirection inside individual weight accesses.

## Lightweight benchmark

NeuroBenchmark compares:

~~~text
predict() allocating an output
predictInto() reusing output
100 training epochs
~~~

using a representative 32-64-32-8 network.

The benchmark validates a checksum to keep inference results observable.

It is useful for fast local regression detection.

It is not a replacement for JMH.

## Performance matrix

NeuroPerformanceMatrix runs several topology shapes:

~~~text
tiny    2-6-1
small   32-64-32-8
medium  128-256-128-32
deep    64-128-128-64-32-8
~~~

For each it reports:

- total parameter count,
- predictInto nanoseconds per operation,
- training milliseconds per epoch,
- final training RMSE,
- prediction checksum.

That exposes both width and depth effects rather than reporting one synthetic network size.

## JMH

Two JMH surfaces are included.

NeuroInferenceJmhBenchmark compares:

~~~text
predictAllocating
predictInto
~~~

across small, medium, and deep networks.

NeuroTrainingJmhBenchmark measures one training epoch across small and medium topologies.

The normal JMH configuration uses multiple warmup iterations, measured iterations, and forks.

The CI workflow runs only a bounded prediction smoke benchmark.

Full JMH should be run on a quiet, stable machine when making performance claims.

## JFR

JNeuro defines disabled-by-default JFR events for:

- individual training epochs,
- complete trainUntil runs.

They record duration plus RMSE/convergence metadata.

The profiling task is:

~~~text
./gradlew :jneuro:profileNeuro
~~~

The recording is written below build/jfr/.

That makes it possible to correlate training time with allocation, GC, compilation, CPU sampling, and the topology matrix without putting timing calls inside every numeric loop.

## Java 27 build

The module targets Java 27 and compiles with:

~~~text
-Xlint:all
-Werror
~~~

Available tasks include:

~~~text
:jneuro:runExperiment
:jneuro:verifyExperiment
:jneuro:benchmarkExperiment
:jneuro:performanceMatrix
:jneuro:jmh
:jneuro:jmhSmoke
:jneuro:profileNeuro
~~~

The module-specific GitHub Actions workflow runs deterministic verification, JMH smoke, the performance matrix smoke run, and the full example.

## Remaining limitations

JNeuro remains an educational feed-forward network rather than a general machine-learning system.

It currently has:

- only sigmoid activations,
- only dense fully-connected layers,
- no softmax,
- no cross-entropy objective,
- no mini-batch gradient accumulation,
- no regularization,
- no model persistence,
- no SIMD Vector API kernel,
- no explicit parallelism inside one model,
- no gradient-check utility,
- no adaptive optimizer such as Adam.

Those are useful future experiments only if they preserve the main value of this module: the entire training algorithm remains understandable by reading a few ordinary Java loops.
