# Evolution: A Deterministic Genetic Search over Dense Numeric Genomes

## Why this experiment exists

This experiment implements a compact genetic algorithm over fixed-length arrays of doubles.

The original 2016 version demonstrated selection, single-point crossover, mutation, and distance-to-goal with very little code. That made the mechanics easy to inspect, but several implementation choices obscured the actual algorithmic cost and made results difficult to reproduce:

- randomness came from a synchronized SecureRandom,
- candidates were heap objects wrapping separate double arrays,
- stream pipelines were used inside the hottest numeric loops,
- population size changed indirectly and could collapse into an empty parent pool,
- there was no guaranteed elitism,
- there was no deterministic seed,
- mutation could hit the same gene repeatedly while leaving others untouched,
- population growth and shrinkage made performance hard to compare,
- length checks relied on disabled-by-default assertions,
- the demo printed one million lines,
- there was no convergence API or hard generation limit.

The modern version keeps the same educational goal — evolve numeric genomes toward a target — while turning the implementation into a deterministic, measurable, allocation-conscious genetic search.

## Public shape

A search is created from a goal vector and a configuration:

~~~java
var evolution = new Evolution(
        new double[]{1, 1, 1, 1},
        Evolution.Config.defaults()
                .withPopulationSize(512)
                .withTournamentSize(6)
                .withMutationProbability(0.08)
                .withSeed(42));
~~~

The goal is defensively copied.

The default gene domain is:

~~~text
[0.0, 1.0]
~~~

Custom finite bounds can be supplied with:

~~~java
config.withGeneBounds(-2.0, 2.0)
~~~

The target must lie inside those bounds.

## Fixed-size population

The original code used probabilistic deletion plus a generation factor.

Population size could drift, small populations could generate zero children, and an empty crossover pool could make nextInt(0) fail.

The modern algorithm keeps exactly:

~~~java
config.populationSize()
~~~

candidates every generation.

One elite candidate is copied unchanged into the next generation, and every remaining slot is filled by reproduction.

Fixed population size provides several useful invariants:

- predictable memory use,
- predictable work per generation,
- no empty parent pool,
- no accidental unbounded growth,
- comparable benchmark runs.

## Flat population layout

Candidates are not separate Drone objects.

All genes are stored in one flat primitive array:

~~~java
double[] population;
double[] nextPopulation;
~~~

For candidate c and gene g:

~~~text
index = c * geneCount + g
~~~

This removes:

- one object per candidate,
- one double[] object per candidate,
- pointer chasing through a List<Drone>,
- temporary child objects during crossover.

Two full buffers are allocated once and swapped after each generation.

The next generation is therefore built without allocating one new genome object per child.

## Fitness

Fitness is mean squared error against the target:

~~~text
MSE =
    sum((gene - goal)^2)
    / geneCount
~~~

Lower is better.

The hot kernel is a primitive loop using Math.fma:

~~~java
var difference =
        genome[offset + gene]
        - goal[gene];

sum = Math.fma(
        difference,
        difference,
        sum);
~~~

No streams, boxing, lambdas, or temporary collections are involved.

The method accepts an offset into a flat genome buffer, which lets the same kernel evaluate every candidate without slicing arrays.

## Deterministic randomness

Each Evolution instance owns one SplittableRandom seeded from Config.

~~~java
var random =
        new SplittableRandom(
                config.seed());
~~~

The same:

- goal,
- configuration,
- seed,
- sequence of evolve calls

produces the same population trajectory.

This is critical for correctness tests and performance experiments.

SecureRandom was removed because cryptographic unpredictability provides no benefit to this search while making experiments slower and irreproducible.

## Tournament selection

Parent selection uses a configurable tournament.

The algorithm starts with one random candidate, samples additional candidates, and keeps the one with the lowest fitness.

~~~java
var winner =
        random.nextInt(
                config.populationSize());

for (int competitor = 1;
        competitor < config.tournamentSize();
        competitor++) {
    var candidate =
            random.nextInt(
                    config.populationSize());

    if (fitness[candidate]
            < winnerFitness) {
        winner = candidate;
        winnerFitness =
                fitness[candidate];
    }
}
~~~

Tournament size controls selection pressure without sorting the full population.

That is useful both algorithmically and for performance.

A full ranked sort would add O(P log P) work per generation; tournament selection stays O(P * tournamentSize) across offspring creation.

## Elitism

Candidate zero of the next buffer receives an exact copy of the current best genome.

~~~java
copyGenome(
        population,
        bestIndex,
        nextPopulation,
        0);
~~~

The elite is not mutated.

This guarantees:

~~~text
bestError(generation + 1)
    <= bestError(generation)
~~~

The deterministic verification suite checks this property over hundreds of generations, including configurations with mutationProbability = 1.0.

The old implementation could delete its best candidate.

## Crossover

Every non-elite child receives two tournament-selected parents.

With configurable crossoverProbability, single-point crossover is performed:

~~~java
var cut =
        random.nextInt(
                geneCount + 1);
~~~

The prefix is copied from the first parent and the suffix from the second.

System.arraycopy is used for both segments.

The cut may be zero or geneCount, so exact parent copies remain possible.

If crossover is skipped, the fitter of the two selected parents is copied directly.

No child object is allocated.

## Mutation

Mutation is evaluated independently per gene:

~~~java
if (random.nextDouble()
        < config.mutationProbability()) {
    nextPopulation[offset + gene] =
            randomGene();
}
~~~

The mutation operator is a random reset inside the configured gene bounds.

This differs from the original mutation-count rule, where:

~~~text
floor(geneCount * mutationFactor)
~~~

random positions were selected and the same position could be chosen repeatedly.

Per-gene probability has a clearer interpretation and does not need an intermediate set of positions.

## Generation lifecycle

One evolve() call performs:

1. copy the elite,
2. select two parents for every remaining child,
3. crossover or copy,
4. mutate the child,
5. swap current and next population buffers,
6. evaluate all fitness values,
7. update best and average statistics.

The method returns:

~~~java
record GenerationStats(
        long generation,
        int populationSize,
        int geneCount,
        double bestError,
        double averageError)
~~~

This makes the convergence curve observable without scanning the population from outside.

## Bounded convergence

The old example ran one million generations regardless of progress.

The modern API supports:

~~~java
var result =
        evolution.evolveUntil(
                0.001,
                10_000);
~~~

SearchResult contains:

~~~java
record SearchResult(
        long generations,
        double bestError,
        boolean converged)
~~~

The hard generation limit prevents accidental infinite experiments.

If the initial population already satisfies the target, zero generations are reported.

## Defensive genome access

bestGenome() returns a copy.

~~~java
double[] best =
        evolution.bestGenome();
~~~

For allocation-sensitive code:

~~~java
evolution.copyBestInto(buffer);
~~~

The internal population arrays are never exposed.

The same applies to goal().

This prevents external code from silently corrupting search state.

## Complexity

Let:

~~~text
P = population size
G = gene count
T = tournament size
~~~

Fitness evaluation is:

~~~text
O(P * G)
~~~

Parent selection and offspring construction are:

~~~text
O(P * (T + G))
~~~

Memory is dominated by two population buffers:

~~~text
O(P * G)
~~~

plus:

~~~text
O(P)
~~~

fitness values.

Because the population is fixed, these bounds are stable across generations.

## Deterministic verification

EvolutionVerification covers:

- invalid configurations,
- invalid and non-finite goals,
- custom gene bounds,
- defensive goal copies,
- defensive best-genome copies,
- deterministic reproduction from the same seed,
- fixed population size,
- monotonic best error from elitism,
- gene-bound preservation,
- convergence toward a multi-gene target,
- single-gene convergence,
- generation-limit enforcement,
- the MSE offset kernel.

The convergence tests use fixed seeds.

A regression therefore produces a repeatable failing trajectory instead of a probabilistic test failure.

## Lightweight benchmark

EvolutionBenchmark measures complete generation loops for representative shapes:

~~~text
population 64,   genes 16
population 512,  genes 16
population 512,  genes 128
~~~

Each run constructs a deterministic search, evolves a fixed number of generations, and consumes bestError into a blackhole.

This provides a quick regression signal.

It is not intended to replace JMH.

## Performance matrix

EvolutionPerformanceMatrix explores:

~~~text
population:
64
256
1024
4096

genes:
8
32
128
512
~~~

For each combination it reports:

- nanoseconds per generation,
- nanoseconds per candidate-gene,
- final best error,
- error improvement over the measured window.

The per-candidate-gene number helps distinguish fixed generation overhead from actual genome-processing cost.

## JMH

EvolutionJmhBenchmark measures evolve() using:

~~~text
populationSize = 64, 512, 4096
geneCount      = 16, 128
~~~

The benchmark uses AverageTime in microseconds.

Each measured iteration begins with a fresh deterministic Evolution instance so one JMH iteration does not inherit the convergence state of a previous iteration.

The normal benchmark uses multiple warmup iterations, measured iterations, and forks.

CI runs only a small smoke case.

## JFR

Evolution defines disabled-by-default JFR events for:

- each generation,
- a complete evolveUntil search.

Generation events record:

- generation number,
- population size,
- gene count,
- best MSE,
- average MSE,
- duration.

Search events record:

- target MSE,
- maximum generations,
- actual generations,
- final error,
- convergence,
- duration.

Profiling can be run with:

~~~text
./gradlew :evolution:profileEvolution
~~~

The recording is written under build/jfr/.

This makes allocation, GC, JIT compilation, CPU sampling, and convergence behavior observable in one recording.

## Java 27 build

The module targets Java 27 with:

~~~text
-Xlint:all
-Werror
~~~

Available tasks include:

~~~text
:evolution:runExperiment
:evolution:verifyExperiment
:evolution:benchmarkExperiment
:evolution:performanceMatrix
:evolution:jmh
:evolution:jmhSmoke
:evolution:profileEvolution
~~~

The module-specific GitHub Actions workflow runs:

- deterministic verification,
- JMH smoke,
- a bounded performance matrix,
- the complete example.

The normal check task depends on verifyExperiment.

## What changed from the 2016 experiment

The conceptual mapping is:

~~~text
Drone objects
    ->
flat population buffers

SecureRandom
    ->
seeded SplittableRandom

probabilistic delete/grow
    ->
fixed generation size

ad-hoc parent eligibility list
    ->
tournament selection

no elitism
    ->
one preserved elite

mutation count with duplicate hits
    ->
independent per-gene mutation probability

stream MSE
    ->
primitive Math.fma loop

one-million-line main
    ->
bounded search + compact summary
~~~

The new implementation is intentionally more conventional because that makes the effect of each genetic operator easier to reason about and benchmark.

## Remaining limitations

This remains a compact educational genetic algorithm.

It does not currently provide:

- arbitrary user-defined fitness functions,
- multi-objective optimization,
- rank selection,
- roulette-wheel selection,
- multiple elites,
- uniform crossover,
- Gaussian mutation,
- adaptive mutation rates,
- parallel fitness evaluation,
- island models,
- checkpoint serialization.

Parallel evaluation is deliberately not the first optimization here.

The current target-distance kernel is extremely small, so parallelism can cost more than it saves for typical population sizes.

A useful next experiment would first benchmark much more expensive fitness functions, then add parallel evaluation only when the objective dominates selection and memory traffic.
