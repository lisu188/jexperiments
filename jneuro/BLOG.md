# A Small Backpropagation Neural Network

## Why this experiment exists

This module implements a feed-forward neural network from scratch. It does not
use a machine-learning library. Instead, it stores weights in arrays, computes
layer outputs manually, propagates error terms backward, and updates weights
with learning-rate and momentum-like parameters.

The sample training data in `main` is the OR truth table. The network receives
two inputs and learns to produce one output. The hidden-layer structure is much
larger than the task requires, which makes the example more about the mechanics
of backpropagation than about finding the smallest possible model.

## How it works

`Neuro` is package-private and contains the whole experiment. The constructor
accepts an array of layer sizes plus three tuning parameters named `_alfa`,
`_beta`, and `_eta`. It allocates three-dimensional arrays for weights,
previous weights, and weight differences. It also allocates output arrays and
error arrays for each layer.

The `o` method performs the forward pass. It copies the input into layer zero,
then computes each next layer as a weighted sum followed by a sigmoid function.
The `e` method computes output and hidden-layer error terms. Training happens in
`teach`: teachers are shuffled, forward and error passes are run, weights are
updated, and a previous-difference term is added afterward.

The stopping condition is based on `erms`, the average root mean square error
over the teacher set. `main` trains until that error drops below `0.001` and
prints the number of iterations used.

## What to notice

The useful part of this experiment is that every array dimension is visible. A
weight from layer `j` neuron `k` to previous-layer neuron `l` is stored directly
in `_neuro[j][k][l]`. There are no tensors, graph APIs, or automatic
differentiation steps hiding the math.

That makes the code useful for learning, but it also shows why neural-network
implementations quickly grow supporting abstractions. Manual indexing is easy to
get subtly wrong, and the code has to manage outputs, errors, previous weights,
shuffled training examples, and convergence checks directly.

## Sanity check

The module has no external dependencies and compiles through the shared Gradle
Java configuration. There are no tests. Because initial weights use
`Math.random()`, training duration is nondeterministic.

The static pass found a likely bug in `test()`: it accumulates errors over the
`tests` list but divides by `teachers.size()`. If the method were used with a
different number of test and teacher examples, the result would be misleading.
The output-layer error calculation also deserves review before treating the
network as a reference implementation.

The repository-level Gradle wrapper build compiles this module successfully.

## Suggested next experiments

The best next step would be adding deterministic tests around OR, AND, and XOR
with a fixed random seed. Another useful extension would be separating the
network math from the training data container, making it easier to inspect
weights, run predictions, and compare learning parameters.
