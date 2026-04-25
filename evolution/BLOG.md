# Evolving Arrays toward a Target

## Why this experiment exists

This experiment implements a minimal evolutionary search over fixed-length
arrays of doubles. Each candidate, called a `Drone`, contains numeric genes.
The goal is another `Drone` whose data is all ones. Each generation removes
some weak candidates, selects candidates for reproduction, creates crossover
children, mutates them, and appends them to the population.

The code is not a general genetic algorithm framework. It is a compact model of
selection pressure, recombination, mutation, and fitness measurement. That makes
it useful for Java developers who want to reason about how an evolutionary loop
is represented with collections, streams, random generation, and mutable arrays
without introducing a large domain library.

The main technical idea is that population dynamics are emergent from a few
local rules. The comparison function is mean squared distance from the goal.
The closer a candidate is to the goal, the lower its score. Removal and
crossover probabilities depend on that score and on current population size, so
the same candidate can be treated differently as the field grows or shrinks.

## Execution path

The entrypoint constructs one search instance and runs one million iterations:

```java
Evolution droneField = new Evolution(10, 0.1, 1000, 0.1,
        new Drone(new double[]{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));
IntStream.range(0, 1000000).forEach(i -> {
    droneField.iterate();
    System.out.println(droneField.field.size() + " : " + (droneField.field.stream().mapToDouble(drone -> drone.compare(droneField.goal)).average()).orElseGet(() -> 0.0));
});
```

The initial population has ten drones. The target has ten genes. The mutation
factor and generation factor are both `0.1`, and the nominal maximum population
is `1000`. Every iteration prints the population size and average distance from
the goal.

The constructor stores the goal and tuning factors, then fills the field with
`new Drone(RandomGenerator.nextDoubleArray(goal.size()))`. That `goal.size()`
dependency is important: the experiment does not need a separate gene-count
parameter. The target defines the candidate shape.

```java
// The goal length determines every candidate's genome length.
IntStream.range(0, size).forEach(value ->
        field.add(new Drone(RandomGenerator.nextDoubleArray(goal.size()))));
```

That keeps initialization coupled to the objective rather than to a duplicated
configuration value.

## Core code walkthrough

Each iteration begins by removing candidates. A candidate with larger distance
from the goal has a higher removal probability, and the current field size
scales that pressure:

```java
double size = field.size();
field.removeIf(drone -> RandomGenerator.nextDouble() < drone.compare(goal) * size / maxPopulation);
```

The expression is intentionally simple. It does not rank the whole population
or preserve the best individual. It samples each candidate independently. As
population size rises, the same distance score becomes more likely to trigger
removal.

The next loop builds a pool of candidates eligible for crossover with the mirror
image of the removal pressure:
`RandomGenerator.nextDouble() > drone.compare(goal) * field.size() / maxPopulation`.
Better candidates are more likely to enter `toCrossover`, but selection is
still probabilistic. The code then chooses random parents from that pool to
create children.

The number of children is based on the current field size after removal:
`field.size() * genFactor`, truncated to an integer. With the default
generation factor of `0.1`, small populations can produce zero children for an
iteration. Larger populations produce more children and also experience
stronger removal pressure because population size appears in both probability
formulas. The population therefore regulates itself indirectly rather than by a
hard cap.

Crossover uses a single cut point:

```java
int cutPoint = RandomGenerator.nextInt(size() + 1);
System.arraycopy(data, 0, newData, 0, cutPoint);
System.arraycopy(drone.data, cutPoint, newData, cutPoint, size() - cutPoint);
return new Drone(newData);
```

The child receives the prefix from one parent and the suffix from the other.
Because the cut point can be `0` or `size()`, the child may be an exact copy of
one parent before mutation.

## Important implementation details

Mutation modifies the drone in place:

```java
Drone mutate(double factor) {
    IntStream.range(0, Double.valueOf(size() * factor).intValue()).forEach(value -> data[RandomGenerator.nextInt(size())] = RandomGenerator.nextDouble());
    return this;
}
```

In the current pipeline that is safe because mutation is called on the new child
returned by `crossover`. If the same method were called on a parent already in
the field, it would mutate population state directly. That behavior should be
documented because the method returns `this`, not a copy.

Fitness is mean squared distance:

```java
double compare(Drone drone) {
    assert data.length == drone.data.length;
    return IntStream.range(0, data.length).mapToDouble(value ->
            (data[value] - drone.data[value]) * (data[value] - drone.data[value])).average().orElseGet(() -> 0.0);
}
```

Lower is better. The name `compare` is broader than the implementation; it is
really a distance function. Since all genes are generated in `[0, 1)`, and the
goal is all ones, initial scores are bounded and easy to interpret.

Randomness is centralized in a `private static final Random RANDOM = new
SecureRandom()` field with synchronized helper methods. The synchronized wrapper
makes calls thread-safe, although this experiment is single-threaded.
`SecureRandom` also makes runs intentionally non-repeatable unless the
implementation is changed to accept a seeded `Random`.

```java
// Centralized random source; synchronized even though the demo is single-threaded.
static synchronized double[] nextDoubleArray(int size) {
    double[] data = new double[size];
    IntStream.range(0, size).forEach(value -> data[value] = RandomGenerator.nextDouble());
    return data;
}
```

This helper is the only place new candidate genomes are filled with random
values.

Using `SecureRandom` is unusual for an evolutionary toy. It is slower than a
plain pseudo-random generator and has security-oriented behavior the algorithm
does not need. That makes it a good discussion point: randomness quality,
repeatability, and throughput are separate design choices. For experiments,
repeatability is often more valuable than cryptographic unpredictability.

## Runtime behavior and caveats

Running the module can produce a lot of output: one million lines, each with
population size and average distance. There is no termination condition based
on convergence, no charting, and no protection against a population that drifts
into a problematic state.

The main correctness caveat is that `toCrossover` can become empty after
selection. If that happens, `RandomGenerator.nextInt(toCrossover.size())` will
throw. The current parameters may make that unlikely in normal runs, but the
algorithm does not guard against it. There is also no elitism, so the best
candidate can be removed. Mutation may also hit the same gene more than once
because each mutation position is chosen independently.

The use of assertions for length checks is another caveat. Assertions are
disabled by default in most JVM runs, so mismatched drone sizes would not be
reported by those checks unless the process is started with `-ea`.

The console output is also part of the runtime cost. Printing a million lines
can dominate the actual search and make the algorithm appear slower than it is.
If the goal is to evaluate convergence, sampling every hundred or thousand
iterations would provide clearer data and much less I/O noise.

Because the algorithm keeps the whole population in one `ArrayList`, memory use
is easy to understand but tightly coupled to population growth. A separate
generation object would make snapshots and comparisons easier.

## Suggested next experiments

Add deterministic seeding so two runs can be compared. Track the best candidate
and preserve it across generations. Replace probabilistic removal with ranked
tournament selection. Add a convergence stop condition and reduce console I/O.
Finally, extract the fitness function so the same engine can evolve arrays
toward different objective functions, not just an all-ones target.
