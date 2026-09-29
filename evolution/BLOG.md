# Evolution: An Extensible Genetic Search Engine in Plain Java

## Why this experiment exists

This experiment started as a small 2016 program that evolved arrays of doubles toward an all-ones target. The modern implementation keeps the same transparent, low-level character, but now treats target matching as one specialization of a broader genetic search engine.

The current engine supports pluggable scalar fitness, scalar and Vector API MSE, tournament/rank/roulette/truncation selection, four crossover operators, random-reset and Gaussian mutation, geometric mutation skipping, multiple elites, adaptive mutation strength, stagnation restarts, parallel fitness, diversity metrics, island evolution, Pareto-ranked multi-objective evolution, deterministic replay, JFR phases and time-to-target benchmarks.

The implementation remains array-oriented. Genetic operators work directly on flat primitive buffers so the cost of selection, crossover, mutation and fitness stays visible.

## Pluggable scalar fitness

The core objective abstraction is:

~~~java
@FunctionalInterface
public interface EvolutionFitness {
    double evaluate(
            double[] genome,
            int offset,
            int length);
}
~~~

Fitness receives the shared population array plus an offset. The engine does not allocate a temporary genome for every candidate.

A custom sphere objective can therefore be written as:

~~~java
EvolutionFitness sphere =
        (genome, offset, length) -> {
            var sum = 0.0;
            for (int i = 0; i < length; i++) {
                var value = genome[offset + i];
                sum = Math.fma(value, value, sum);
            }
            return sum;
        };
~~~

Lower fitness is always better.

## Goal matching and SIMD

The original use case still has a direct constructor:

~~~java
var evolution =
        new Evolution(
                new double[]{1, 1, 1, 1});
~~~

Goal-backed searches use mean squared error. By default that fitness uses the Java Vector API. Callers can disable the SIMD specialization with Config.withVectorizedGoalFitness(false).

The vector kernel loads candidate and goal vectors, subtracts them, squares lane-wise and reduces the result:

~~~java
var candidate =
        DoubleVector.fromArray(
                SPECIES,
                genome,
                offset + gene);

var desired =
        DoubleVector.fromArray(
                SPECIES,
                target,
                gene);

var difference = candidate.sub(desired);

sum += difference
        .mul(difference)
        .reduceLanes(VectorOperators.ADD);
~~~

A scalar tail handles genomes whose length is not a multiple of the preferred vector species.

The module runs with the incubator module enabled:

~~~text
--add-modules jdk.incubator.vector
~~~

## Flat population and cached fitness

Population memory is two reusable primitive arrays:

~~~java
double[] population;
double[] nextPopulation;
~~~

Candidate c starts at c * geneCount.

Fitness is also double-buffered:

~~~java
double[] fitness;
double[] nextFitness;
~~~

When elites are copied into the next generation, their fitness values are copied too. A generation with P candidates and E elites evaluates only P - E new candidates.

This matters little for cheap MSE and matters a lot when fitness runs a simulation.

## Policy configuration

Selection, crossover and mutation are explicit policy records rather than unrelated fields in one large configuration.

Selection supports:

~~~text
TOURNAMENT
RANK
ROULETTE
TRUNCATION
~~~

Crossover supports:

~~~text
SINGLE_POINT
UNIFORM
ARITHMETIC
BLX_ALPHA
~~~

Mutation supports:

~~~text
RANDOM_RESET
GAUSSIAN
~~~

The outer Config controls population size, elite count, gene bounds, seed, fitness parallelism, SIMD goal fitness and diversity tracking.

## Selection strategies

Tournament selection samples several candidates and retains the lowest fitness. Tournament size directly controls selection pressure.

Rank selection uses the cached ranking and gives linearly decreasing weights from best to worst. It is insensitive to the numeric scale of the fitness values.

Roulette selection converts normalized distance from the current best into inverse weights. It is useful when the distribution itself should affect selection probability.

Truncation selects uniformly from the configured best fraction of the population. It is intentionally aggressive and useful as an experimental extreme.

## Multiple elites

Config exposes eliteCount. The best E candidates are copied unchanged into the next generation before reproduction.

That provides the invariant:

~~~text
best(generation + 1) <= best(generation)
~~~

for the normal scalar search.

Elitism is also a performance optimization because copied elite fitness values are reused.

## Continuous crossover

Single-point crossover remains as the historical baseline.

Uniform crossover chooses one parent independently for every gene.

Arithmetic crossover samples alpha and creates:

~~~text
child = alpha * A + (1 - alpha) * B
~~~

BLX-alpha samples around the parent interval:

~~~text
[min(A,B) - alpha*d,
 max(A,B) + alpha*d]
~~~

where d is the distance between both parent genes. The result is clipped to configured gene bounds.

Arithmetic and BLX-alpha are more natural operators for continuous genomes than treating doubles like a bit string.

## Gaussian mutation

The default mutation is local Gaussian perturbation:

~~~text
x' = clamp(x + N(0, sigma))
~~~

A Box-Muller generator is implemented on top of the per-engine SplittableRandom and caches the second normal sample.

Random reset remains available for exploration-heavy baselines.

## Geometric mutation skipping

At low mutation probability, checking every gene wastes random-number calls.

When geometric skipping is enabled, the engine samples the number of unchanged genes before the next mutation event. The hot path becomes conceptually:

~~~text
skip N
mutate
skip M
mutate
~~~

instead of one probability draw per gene.

The normal per-gene path remains available and is used automatically for high mutation probabilities.

## Adaptive sigma and stagnation restarts

Gaussian mutation has a live mutation sigma.

When the historical best improves, sigma decays toward minSigma. This gradually shifts from exploration toward local refinement.

When the search stagnates for the configured number of generations, sigma is multiplied by stagnationBoost up to maxSigma.

The engine can also reinitialize a configured fraction of the worst non-elite genomes. Elites survive the restart.

Generation statistics expose:

~~~text
mutationSigma
stagnantGenerations
restarts
~~~

so convergence behavior is observable rather than hidden.

## Diversity statistics

GenerationStats now includes best, median, average and worst fitness, fitness standard deviation and average per-gene standard deviation.

Gene diversity is the average standard deviation of each gene across the population.

This distinguishes a population that is still exploring from one that has collapsed into a narrow region without reaching a good solution.

Diversity measurement can be disabled in pure throughput benchmarks because it adds an O(P*G) observation pass.

## Parallel fitness

Config.fitnessParallelism controls optional parallel objective evaluation.

Only fitness is parallelized. Selection, crossover, mutation and RNG remain on the orchestration thread, which preserves deterministic operator order.

When parallelism is greater than one, the population is split into chunks evaluated on a fixed platform-thread pool.

A custom EvolutionFitness must be safe for concurrent calls when this mode is enabled.

For cheap MSE, parallelism is expected to lose. The performance matrix therefore uses a deliberately expensive synthetic fitness when measuring parallel scaling.

## Shared ranking cache

A primitive int[] ranking is refreshed after population evaluation.

The same ranking is reused for:

- multiple elites,
- rank selection,
- truncation selection,
- median fitness,
- stagnation restarts,
- island migration.

This avoids multiple boxed sorting structures.

## Genome injection

External algorithms can offer a candidate through:

~~~java
boolean injectGenome(double[] genome)
~~~

The candidate is evaluated immediately and replaces the current worst candidate only if it is better.

This is the bridge used by island migration and also makes the engine composable with other search techniques.

## Island evolution

IslandEvolution owns several independent Evolution populations with different deterministic seeds.

Islands can evolve sequentially or concurrently.

Every migration interval, the configured number of top genomes from island i are offered to island (i + 1) mod islandCount.

Migration uses injectGenome, so weak migrants cannot degrade the destination population.

The model preserves diversity naturally and parallelizes with very little coordination between generations.

## Pareto multi-objective search

MultiObjectiveFitness writes several objective values for one genome:

~~~java
@FunctionalInterface
public interface MultiObjectiveFitness {
    void evaluate(
            double[] genome,
            int offset,
            int length,
            double[] objectives,
            int objectiveOffset);
}
~~~

ParetoEvolution minimizes every objective.

It constructs pairwise dominance relations, assigns Pareto fronts and calculates crowding distance inside every front.

Parent tournaments compare:

1. lower Pareto rank,
2. higher crowding distance,
3. deterministic random tie breaking.

This is an NSGA-style experiment centered on dominance ranking and objective-space diversity. It is intentionally smaller than a production NSGA-II implementation with parent-plus-offspring environmental selection.

The current non-dominated front can be exported as defensive genome copies.

## Phase-level JFR

JFR events are disabled by default and include:

~~~text
EvolutionGeneration
EvolutionSearch
EvolutionPhase
~~~

Generation events record fitness, diversity, live sigma and restart count.

Phase events separate:

~~~text
reproduction
fitness
statistics
~~~

This makes it possible to see whether a workload is dominated by genetic operators, expensive objective evaluation or observation/restart logic.

## Time-to-target benchmark

EvolutionConvergenceBenchmark runs multiple deterministic seeds and reports:

- success rate,
- p50/p90/p99 generations to target,
- median fitness evaluations,
- p50/p90 wall time.

The benchmark compares:

~~~text
random-reset + single-point
Gaussian + arithmetic
adaptive Gaussian + BLX-alpha
island evolution
~~~

This is more informative than generations per second. A slower generation can still be the better optimizer if it needs far fewer generations or fitness evaluations.

The default benchmark uses 100 seeds. CI uses a tiny smoke sample.

## Expanded performance matrix

The performance matrix contains four groups:

1. population x genome scaling,
2. every selection x crossover combination,
3. expensive-fitness parallelism 1/2/4/8,
4. sequential versus parallel islands.

The scaling matrix still reports normalized nanoseconds per candidate-gene, while operator experiments report both cost and resulting best fitness.

## JMH and GC profiling

JMH now covers:

- complete generations,
- scalar versus Vector API MSE,
- expensive-fitness parallelism.

The dedicated task:

~~~text
./gradlew :evolution:jmhGc
~~~

runs the Evolution JMH suite with the gc profiler so allocation rate and GC behavior can be measured directly.

## Coverage

The module is registered with the repository coverage smoke harness.

EvolutionVerification executes operator variants, generic fitness, SIMD fitness, adaptive restart, parallel evaluation, migration and Pareto ranking under JaCoCo.

Performance/example/JFR reporting classes are narrowly excluded from line-coverage enforcement, while the actual search engines remain subject to the repository-wide 90% module threshold.

## Java 27 tasks

The module targets Java 27 with warnings treated as errors and enables jdk.incubator.vector.

Useful tasks include:

~~~text
:evolution:verifyExperiment
:evolution:benchmarkExperiment
:evolution:performanceMatrix
:evolution:convergenceBenchmark
:evolution:jmh
:evolution:jmhSmoke
:evolution:jmhGc
:evolution:profileEvolution
~~~

## Remaining separate experiments

The engine now covers the enhancement set that naturally belongs in this module. Larger algorithm families are better kept separate: full NSGA-II environmental selection, CMA-ES, differential evolution, distributed islands across processes, checkpoint persistence and GPU/Panama-offloaded objectives.

Keeping those as distinct experiments preserves the main strength of this code: every algorithmic choice can still be understood and benchmarked without a framework hiding the hot path.
