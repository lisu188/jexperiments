# Evolving Arrays toward a Target

## Why this experiment exists

This module is a compact genetic-algorithm style experiment. A population of
`Drone` objects each carries an array of doubles. The goal is another array of
doubles, initialized in `main` as ten values all equal to `1`. Each iteration
removes some candidates, crosses over selected survivors, mutates offspring, and
prints the average distance from the target.

The code is small enough to read in one sitting. That makes it a good place to
inspect the moving parts of evolutionary search without a framework,
visualization layer, or complicated fitness domain.

## How it works

The entrypoint constructs an `Evolution` with an initial population of ten
random drones, a mutation factor of `0.1`, a maximum population of `1000`, a
generation factor of `0.1`, and the all-ones goal vector.

Each `Drone` owns a `double[]`. Its `compare` method returns the average squared
difference from another drone. Smaller values mean a closer match. Its
`crossover` method chooses a random cut point and combines the prefix of one
array with the suffix of another. Its `mutate` method changes a number of
positions proportional to the configured mutation factor.

The `iterate` method performs the evolutionary step. It removes drones based on
their distance from the goal and the current population size, selects candidates
for crossover using another probabilistic filter, creates new mutated children,
and appends them to the field.

## What to notice

The selection pressure is encoded in only a few probability expressions. The
current population size participates in those expressions, so behavior changes
as the field grows or shrinks. This creates a feedback loop between fitness,
population pressure, and reproduction.

The model is intentionally bare. There is no explicit fitness score type, no
generation object, and no separation between immutable parent and mutable child
data. That makes the algorithm easy to follow, but it is best read as an
exploratory sketch rather than a reusable genetic algorithm library.

## Sanity check

The module has no external dependencies and compiles through the shared Gradle
Java configuration. There are no tests. The `main` method runs one million
iterations and prints every iteration, so it is a long-running console
experiment.

The static pass found a couple of demo-grade risks. `mutate` changes the drone's
array in place. The crossover candidate list can become empty if selection
removes too much of the population, which would make
`Random.nextInt(toCrossover.size())` fail. The source also uses the boxed
`new Double(...).intValue()` pattern, which modern Java reports as deprecated.

The repository-level Gradle wrapper build compiles this module successfully.

## Suggested next experiments

A useful next step would be recording best, worst, and average fitness per
generation instead of only the average. Another would be extracting selection,
crossover, and mutation policies into small interfaces so strategies can be
compared.
