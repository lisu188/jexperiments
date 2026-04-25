# A Small Backpropagation Neural Network

## Why this experiment exists

This experiment implements a feed-forward neural network and training loop
using plain Java arrays. There is no machine-learning framework, no tensor
library, and no matrix abstraction. Layers, weights, outputs, errors, previous
weights, and momentum deltas are all stored in primitive arrays and updated by
nested loops.

That makes the module useful for developers who want to see the mechanics of
backpropagation without API indirection. Every index expresses part of the
network topology. `_neuro[layer][neuron][input]` is the weight from an input in
one layer to a neuron in the next. `_o[layer][neuron]` stores activations.
`_e[layer][neuron]` stores error terms. The structure is old-school, but it
makes data movement explicit.

The demo trains a network shaped `{2, 5, 25, 5, 1}` on a small boolean-style
truth table: `00` maps to `0`, and the other three two-input combinations map
to `1`. In other words, the example is closer to an OR function than XOR. The
network is intentionally much larger than needed for that task, which makes it
more about algorithm shape than model minimalism.

## Execution path

The entrypoint constructs the network, adds four training examples, and trains
until the root-mean-square error falls below a threshold:

```java
Neuro neuro = new Neuro(new int[]{2, 5, 25, 5, 1}, 0.2, 1, 0.8);
neuro.add_teacher(new double[]{1, 1}, new double[]{1});
neuro.add_teacher(new double[]{1, 0}, new double[]{1});
neuro.add_teacher(new double[]{0, 1}, new double[]{1});
neuro.add_teacher(new double[]{0, 0}, new double[]{0});
System.out.println(neuro.teach(0.001));
```

The constructor parameters are topology, momentum coefficient `_alfa`, sigmoid
steepness `_beta`, and learning rate `_eta`. The printed value is the number of
training iterations performed before the error condition is met.

Weights are allocated per adjacent layer pair:

```java
_neuro = new double[_nw - 1][][];
_prev = new double[_nw - 1][][];
_diff = new double[_nw - 1][][];
for (int i = 0; i < _nw - 1; i++) {
    _neuro[i] = new double[_str[i + 1]][];
```

The arrays `_prev` and `_diff` support a momentum-style update. The code keeps
the previous weight values so it can add a fraction of the prior change after
the gradient update.

The topology array `_str` is used everywhere instead of deriving dimensions
from the allocated arrays. That makes the intended network shape clear, but it
also means constructor correctness is critical. A mismatch between `_str` and
the arrays would corrupt many loops. In this code, the constructor is the single
place where the structure is created, so later methods can assume consistent
dimensions.

```java
// Outputs are allocated separately from weights so each layer can cache activations.
_o = new double[_nw][];
for (int i = 0; i < _nw; i++) {
    _o[i] = new double[_str[i]];
}
```

That cache is what later lets the backward pass reuse previous-layer outputs
when updating weights.

## Core code walkthrough

The forward pass copies input into the first output layer and then propagates
activations through each following layer:

```java
private void o(double[] t) {
    System.arraycopy(t, 0, _o[0], 0, _str[0]);
    for (int i = 1; i < _nw; i++) {
        for (int j = 0; j < _str[i]; j++) {
            _o[i][j] = 0;
            for (int k = 0; k < _str[i - 1]; k++) {
                _o[i][j] += _o[i - 1][k] * _neuro[i - 1][j][k];
            }
            _o[i][j] = fcn(_o[i][j], _beta);
```

Inside the loops, every neuron accumulates weighted outputs from the previous
layer, then applies the sigmoid.

The activation function is logistic:

```java
private double fcn(double x, double beta) {
    return 1.0 / (1.0 + Math.exp(-beta * x));
}

private double dfcn(double x) {
    return (1.0 - x) * x;
}
```

`dfcn` is the derivative expressed in terms of an activation value, which is a
common optimization for sigmoid networks because the output of the sigmoid is
already available.

There is no bias term in the accumulation. Every neuron computes only a weighted
sum of previous activations. Biases are normally modeled as an additional
constant input or separate weight, and they let activation thresholds shift away
from the origin. Their absence is one reason this code should be read as a
mechanical sketch rather than a reference architecture.

## Important implementation details

Training begins each iteration by recording momentum state, then applies the
gradient update and later adds momentum:

```java
_diff[i][j][k] = _neuro[i][j][k] - _prev[i][j][k];
_prev[i][j][k] = _neuro[i][j][k];

_neuro[j][k][l] += (_eta * _e[j][k] * _o[j][l]) / teachers.size();

_neuro[i][j][k] += _alfa * _diff[i][j][k];
```

The teacher list is shuffled before updates, and the gradient line uses the
current layer's error term, the previous layer's output, the learning rate, and
the number of teachers. This two-phase structure is easy to inspect, but it
also highlights how manual array code can hide conceptual mistakes. The
output-layer error calculation in `e` uses `dfcn(out[i] - _o[_nw - 1][i])`,
while the derivative helper expects a sigmoid activation. A conventional
implementation would keep the target-minus-output factor separate from the
derivative evaluated at the output activation. That makes this module
educational rather than numerically authoritative.

`teach(erms)` keeps training until the average per-teacher RMS error is below
the requested threshold.

```java
// RMS-style error over the output layer.
tmp += Math.pow(out_t[k] - _o[_nw - 1][k], 2);
erms += Math.sqrt(tmp / _str[_nw - 1]);
```

The metric is simple enough to inspect directly, which is useful because the
training loop otherwise has no progress reporting.

## Runtime behavior and caveats

Running the module trains with random initial weights and prints the number of
iterations. Because weights are initialized with `Math.random() - 0.5`, two
runs can take different paths and may converge in different numbers of steps.
There is no maximum-iteration guard, so a bad configuration can run for a long
time.

There are no bias weights. That limits what the network can represent cleanly
and makes the large hidden topology compensate for missing terms. The test
method also divides by `teachers.size()` instead of `tests.size()`, which is a
bug if test data is ever added separately. The class itself has package-private
visibility and most methods are private, so it is written as an executable
experiment rather than as a reusable API.

The main strength of the code is transparency. If you want to discuss the
memory layout and loop structure of a neural network, every update is visible.
If you want reliable training behavior, gradient checking, batching, numeric
stability, or model serialization, those would need to be added.

The training set is also tiny and reused in full every iteration. That makes
the algorithm closer to online or small-batch teaching than to large dataset
training. `Collections.shuffle(teachers)` changes example order, but there is
no train/validation split in `main`, no normalization layer, and no reporting of
intermediate error values.

## Suggested next experiments

Add bias weights and update the forward and backward passes accordingly. Split
the output error into target-minus-output and activation-derivative terms, then
compare convergence. Add a maximum iteration count and deterministic random
seed. Introduce tests for OR and XOR with known tolerances. Finally, replace the
raw three-dimensional arrays with a tiny matrix helper and measure whether the
code becomes clearer or merely more abstract.

Recording the error after each epoch would make those comparisons much easier.
Even a simple CSV output of iteration and error would turn the current single
number into a useful convergence curve.
