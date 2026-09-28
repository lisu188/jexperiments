package com.lis;

import java.util.Arrays;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class Evolution implements AutoCloseable {
    private static final long DEFAULT_SEED = 0xE701L;
    private static final double IMPROVEMENT_EPSILON = 1.0e-14;

    public enum SelectionType {
        TOURNAMENT,
        RANK,
        ROULETTE,
        TRUNCATION
    }

    public enum CrossoverType {
        SINGLE_POINT,
        UNIFORM,
        ARITHMETIC,
        BLX_ALPHA
    }

    public enum MutationType {
        RANDOM_RESET,
        GAUSSIAN
    }

    public record SelectionPolicy(
            SelectionType type,
            int tournamentSize,
            double truncationFraction) {
        public SelectionPolicy {
            Objects.requireNonNull(type, "type");
            if (tournamentSize < 2) {
                throw new IllegalArgumentException("tournamentSize must be >= 2");
            }
            if (!(truncationFraction > 0.0 && truncationFraction <= 1.0)
                    || !Double.isFinite(truncationFraction)) {
                throw new IllegalArgumentException("truncationFraction must be finite and in (0, 1]");
            }
        }

        public static SelectionPolicy defaults() {
            return new SelectionPolicy(SelectionType.TOURNAMENT, 4, 0.5);
        }

        public SelectionPolicy withType(SelectionType value) {
            return new SelectionPolicy(value, tournamentSize, truncationFraction);
        }

        public SelectionPolicy withTournamentSize(int value) {
            return new SelectionPolicy(type, value, truncationFraction);
        }

        public SelectionPolicy withTruncationFraction(double value) {
            return new SelectionPolicy(type, tournamentSize, value);
        }
    }

    public record CrossoverPolicy(
            CrossoverType type,
            double probability,
            double blxAlpha) {
        public CrossoverPolicy {
            Objects.requireNonNull(type, "type");
            if (probability < 0.0 || probability > 1.0 || !Double.isFinite(probability)) {
                throw new IllegalArgumentException("crossover probability must be finite and in [0, 1]");
            }
            if (blxAlpha < 0.0 || !Double.isFinite(blxAlpha)) {
                throw new IllegalArgumentException("blxAlpha must be finite and >= 0");
            }
        }

        public static CrossoverPolicy defaults() {
            return new CrossoverPolicy(CrossoverType.BLX_ALPHA, 0.9, 0.25);
        }

        public CrossoverPolicy withType(CrossoverType value) {
            return new CrossoverPolicy(value, probability, blxAlpha);
        }

        public CrossoverPolicy withProbability(double value) {
            return new CrossoverPolicy(type, value, blxAlpha);
        }

        public CrossoverPolicy withBlxAlpha(double value) {
            return new CrossoverPolicy(type, probability, value);
        }
    }

    public record MutationPolicy(
            MutationType type,
            double probability,
            double sigma,
            double minSigma,
            double maxSigma,
            double sigmaDecay,
            double stagnationBoost,
            int stagnationGenerations,
            double restartFraction,
            boolean adaptive,
            boolean geometricSkipping) {
        public MutationPolicy {
            Objects.requireNonNull(type, "type");
            if (probability < 0.0 || probability > 1.0 || !Double.isFinite(probability)) {
                throw new IllegalArgumentException("mutation probability must be finite and in [0, 1]");
            }
            if (!(sigma > 0.0) || !Double.isFinite(sigma)) {
                throw new IllegalArgumentException("sigma must be finite and > 0");
            }
            if (!(minSigma > 0.0) || !Double.isFinite(minSigma) || minSigma > sigma) {
                throw new IllegalArgumentException("minSigma must be finite, > 0 and <= sigma");
            }
            if (maxSigma < sigma || !Double.isFinite(maxSigma)) {
                throw new IllegalArgumentException("maxSigma must be finite and >= sigma");
            }
            if (!(sigmaDecay > 0.0 && sigmaDecay <= 1.0) || !Double.isFinite(sigmaDecay)) {
                throw new IllegalArgumentException("sigmaDecay must be finite and in (0, 1]");
            }
            if (!(stagnationBoost >= 1.0) || !Double.isFinite(stagnationBoost)) {
                throw new IllegalArgumentException("stagnationBoost must be finite and >= 1");
            }
            if (stagnationGenerations < 1) {
                throw new IllegalArgumentException("stagnationGenerations must be >= 1");
            }
            if (restartFraction < 0.0 || restartFraction > 1.0 || !Double.isFinite(restartFraction)) {
                throw new IllegalArgumentException("restartFraction must be finite and in [0, 1]");
            }
        }

        public static MutationPolicy defaults() {
            return new MutationPolicy(
                    MutationType.GAUSSIAN,
                    0.05,
                    0.10,
                    0.001,
                    0.50,
                    0.995,
                    1.75,
                    100,
                    0.35,
                    true,
                    true);
        }

        public MutationPolicy withType(MutationType value) {
            return new MutationPolicy(
                    value, probability, sigma, minSigma, maxSigma, sigmaDecay,
                    stagnationBoost, stagnationGenerations, restartFraction, adaptive, geometricSkipping);
        }

        public MutationPolicy withProbability(double value) {
            return new MutationPolicy(
                    type, value, sigma, minSigma, maxSigma, sigmaDecay,
                    stagnationBoost, stagnationGenerations, restartFraction, adaptive, geometricSkipping);
        }

        public MutationPolicy withSigma(double value) {
            var newMin = Math.min(minSigma, value);
            var newMax = Math.max(maxSigma, value);
            return new MutationPolicy(
                    type, probability, value, newMin, newMax, sigmaDecay,
                    stagnationBoost, stagnationGenerations, restartFraction, adaptive, geometricSkipping);
        }

        public MutationPolicy withSigmaBounds(double min, double max) {
            return new MutationPolicy(
                    type, probability, sigma, min, max, sigmaDecay,
                    stagnationBoost, stagnationGenerations, restartFraction, adaptive, geometricSkipping);
        }

        public MutationPolicy withAdaptation(
                boolean enabled,
                double decay,
                double boost,
                int stagnantGenerations,
                double restart) {
            return new MutationPolicy(
                    type, probability, sigma, minSigma, maxSigma, decay,
                    boost, stagnantGenerations, restart, enabled, geometricSkipping);
        }

        public MutationPolicy withGeometricSkipping(boolean enabled) {
            return new MutationPolicy(
                    type, probability, sigma, minSigma, maxSigma, sigmaDecay,
                    stagnationBoost, stagnationGenerations, restartFraction, adaptive, enabled);
        }
    }

    public record Config(
            int populationSize,
            int eliteCount,
            double minGene,
            double maxGene,
            long seed,
            SelectionPolicy selection,
            CrossoverPolicy crossover,
            MutationPolicy mutation,
            int fitnessParallelism,
            boolean vectorizedGoalFitness,
            boolean trackDiversity) {
        public Config {
            if (populationSize < 2) {
                throw new IllegalArgumentException("populationSize must be >= 2");
            }
            if (eliteCount < 1 || eliteCount >= populationSize) {
                throw new IllegalArgumentException("eliteCount must be in [1, populationSize)");
            }
            if (!Double.isFinite(minGene) || !Double.isFinite(maxGene) || !(minGene < maxGene)) {
                throw new IllegalArgumentException("gene bounds must be finite and minGene < maxGene");
            }
            Objects.requireNonNull(selection, "selection");
            Objects.requireNonNull(crossover, "crossover");
            Objects.requireNonNull(mutation, "mutation");
            if (selection.tournamentSize() > populationSize) {
                throw new IllegalArgumentException("tournamentSize must be <= populationSize");
            }
            if (fitnessParallelism < 1) {
                throw new IllegalArgumentException("fitnessParallelism must be >= 1");
            }
        }

        public static Config defaults() {
            return new Config(
                    256,
                    2,
                    0.0,
                    1.0,
                    DEFAULT_SEED,
                    SelectionPolicy.defaults(),
                    CrossoverPolicy.defaults(),
                    MutationPolicy.defaults(),
                    1,
                    true,
                    true);
        }

        public Config withPopulationSize(int value) {
            var elites = Math.min(eliteCount, value - 1);
            var selectionPolicy = selection.withTournamentSize(Math.min(selection.tournamentSize(), value));
            return new Config(
                    value, Math.max(1, elites), minGene, maxGene, seed,
                    selectionPolicy, crossover, mutation, fitnessParallelism,
                    vectorizedGoalFitness, trackDiversity);
        }

        public Config withEliteCount(int value) {
            return new Config(
                    populationSize, value, minGene, maxGene, seed,
                    selection, crossover, mutation, fitnessParallelism,
                    vectorizedGoalFitness, trackDiversity);
        }

        public Config withGeneBounds(double min, double max) {
            return new Config(
                    populationSize, eliteCount, min, max, seed,
                    selection, crossover, mutation, fitnessParallelism,
                    vectorizedGoalFitness, trackDiversity);
        }

        public Config withSeed(long value) {
            return new Config(
                    populationSize, eliteCount, minGene, maxGene, value,
                    selection, crossover, mutation, fitnessParallelism,
                    vectorizedGoalFitness, trackDiversity);
        }

        public Config withSelection(SelectionPolicy value) {
            return new Config(
                    populationSize, eliteCount, minGene, maxGene, seed,
                    value, crossover, mutation, fitnessParallelism,
                    vectorizedGoalFitness, trackDiversity);
        }

        public Config withCrossover(CrossoverPolicy value) {
            return new Config(
                    populationSize, eliteCount, minGene, maxGene, seed,
                    selection, value, mutation, fitnessParallelism,
                    vectorizedGoalFitness, trackDiversity);
        }

        public Config withMutation(MutationPolicy value) {
            return new Config(
                    populationSize, eliteCount, minGene, maxGene, seed,
                    selection, crossover, value, fitnessParallelism,
                    vectorizedGoalFitness, trackDiversity);
        }

        public Config withFitnessParallelism(int value) {
            return new Config(
                    populationSize, eliteCount, minGene, maxGene, seed,
                    selection, crossover, mutation, value,
                    vectorizedGoalFitness, trackDiversity);
        }

        public Config withVectorizedGoalFitness(boolean value) {
            return new Config(
                    populationSize, eliteCount, minGene, maxGene, seed,
                    selection, crossover, mutation, fitnessParallelism,
                    value, trackDiversity);
        }

        public Config withTrackDiversity(boolean value) {
            return new Config(
                    populationSize, eliteCount, minGene, maxGene, seed,
                    selection, crossover, mutation, fitnessParallelism,
                    vectorizedGoalFitness, value);
        }
    }

    public record GenerationStats(
            long generation,
            int populationSize,
            int geneCount,
            double bestError,
            double medianError,
            double averageError,
            double worstError,
            double fitnessStdDev,
            double geneDiversity,
            double mutationSigma,
            long stagnantGenerations,
            long restarts,
            long fitnessEvaluations) {
    }

    public record SearchResult(
            long generations,
            double bestError,
            boolean converged,
            long fitnessEvaluations,
            long restarts) {
    }

    private final double[] goal;
    private final Config config;
    private final int geneCount;
    private final EvolutionFitness fitnessFunction;
    private final SplittableRandom random;
    private final ExecutorService fitnessExecutor;
    private final boolean ownsFitnessExecutor;
    private final int[] ranking;

    private double[] population;
    private double[] nextPopulation;
    private double[] fitness;
    private double[] nextFitness;

    private long generation;
    private int bestIndex;
    private double bestError;
    private double historicalBest;
    private double medianError;
    private double averageError;
    private double worstError;
    private double fitnessStdDev;
    private double geneDiversity;
    private double currentMutationSigma;
    private long stagnantGenerations;
    private long restarts;
    private long fitnessEvaluations;

    private boolean hasGaussianSpare;
    private double gaussianSpare;

    public Evolution(double[] goal) {
        this(goal, Config.defaults());
    }

    public Evolution(double[] goal, Config config) {
        this(
                validateGoal(goal, config).length,
                config.vectorizedGoalFitness()
                        ? EvolutionFitness.vectorMeanSquaredError(goal)
                        : EvolutionFitness.meanSquaredError(goal),
                config,
                null,
                validateGoal(goal, config));
    }

    public Evolution(int geneCount, EvolutionFitness fitnessFunction, Config config) {
        this(geneCount, fitnessFunction, config, null, null);
    }

    public Evolution(
            int geneCount,
            EvolutionFitness fitnessFunction,
            Config config,
            ExecutorService fitnessExecutor) {
        this(geneCount, fitnessFunction, config, fitnessExecutor, null);
    }

    private Evolution(
            int geneCount,
            EvolutionFitness fitnessFunction,
            Config config,
            ExecutorService suppliedExecutor,
            double[] goal) {
        if (geneCount <= 0) {
            throw new IllegalArgumentException("geneCount must be positive");
        }
        this.config = Objects.requireNonNull(config, "config");
        this.fitnessFunction = Objects.requireNonNull(fitnessFunction, "fitnessFunction");
        this.goal = goal == null ? null : goal.clone();
        this.geneCount = geneCount;
        population = new double[config.populationSize() * geneCount];
        nextPopulation = new double[population.length];
        fitness = new double[config.populationSize()];
        nextFitness = new double[fitness.length];
        ranking = new int[config.populationSize()];
        random = new SplittableRandom(config.seed());
        currentMutationSigma = config.mutation().sigma();

        if (config.fitnessParallelism() > 1) {
            if (suppliedExecutor != null) {
                fitnessExecutor = suppliedExecutor;
                ownsFitnessExecutor = false;
            } else {
                fitnessExecutor = Executors.newFixedThreadPool(
                        config.fitnessParallelism(),
                        Thread.ofPlatform().daemon(true).name("evolution-fitness-", 0).factory());
                ownsFitnessExecutor = true;
            }
        } else {
            fitnessExecutor = null;
            ownsFitnessExecutor = false;
        }

        initializePopulation(population, 0, config.populationSize());
        evaluatePopulation(population, fitness, 0);
        refreshStatistics(true);
        historicalBest = bestError;
    }

    public Config config() {
        return config;
    }

    public boolean hasGoal() {
        return goal != null;
    }

    public double[] goal() {
        if (goal == null) {
            throw new IllegalStateException("this Evolution instance uses a custom fitness function");
        }
        return goal.clone();
    }

    public int populationSize() {
        return config.populationSize();
    }

    public int geneCount() {
        return geneCount;
    }

    public long generation() {
        return generation;
    }

    public GenerationStats statistics() {
        return new GenerationStats(
                generation,
                config.populationSize(),
                geneCount,
                bestError,
                medianError,
                averageError,
                worstError,
                fitnessStdDev,
                geneDiversity,
                currentMutationSigma,
                stagnantGenerations,
                restarts,
                fitnessEvaluations);
    }

    public double bestError() {
        return bestError;
    }

    public double averageError() {
        return averageError;
    }

    public double[] bestGenome() {
        var result = new double[geneCount];
        copyBestInto(result);
        return result;
    }

    public void copyBestInto(double[] destination) {
        copyRankedGenomeInto(0, destination);
    }

    public void copyRankedGenomeInto(int rank, double[] destination) {
        Objects.requireNonNull(destination, "destination");
        if (rank < 0 || rank >= config.populationSize()) {
            throw new IllegalArgumentException("rank out of range: " + rank);
        }
        if (destination.length != geneCount) {
            throw new IllegalArgumentException(
                    "destination length " + destination.length + " != expected " + geneCount);
        }
        System.arraycopy(population, ranking[rank] * geneCount, destination, 0, geneCount);
    }

    public boolean injectGenome(double[] genome) {
        validateGenome(genome);
        var candidateFitness = evaluateGenome(genome, 0);
        fitnessEvaluations++;
        var worst = ranking[ranking.length - 1];
        if (candidateFitness >= fitness[worst]) {
            return false;
        }

        System.arraycopy(genome, 0, population, worst * geneCount, geneCount);
        fitness[worst] = candidateFitness;
        refreshStatistics(false);
        updateImprovementState();
        return true;
    }

    public GenerationStats evolve() {
        var event = EvolutionJfr.generation(generation + 1, config.populationSize(), geneCount);

        var reproduction = EvolutionJfr.phase("reproduction", generation + 1);
        copyElites();
        for (int child = config.eliteCount(); child < config.populationSize(); child++) {
            var firstParent = selectParent();
            var secondParent = selectParent();
            reproduce(firstParent, secondParent, child);
            mutate(child);
        }
        EvolutionJfr.commitPhase(reproduction);

        var fitnessPhase = EvolutionJfr.phase("fitness", generation + 1);
        evaluatePopulation(nextPopulation, nextFitness, config.eliteCount());
        EvolutionJfr.commitPhase(fitnessPhase);

        swapGenerations();
        generation++;

        var statisticsPhase = EvolutionJfr.phase("statistics", generation);
        refreshStatistics(false);
        updateImprovementState();
        applyAdaptivePolicy();
        EvolutionJfr.commitPhase(statisticsPhase);

        EvolutionJfr.commitGeneration(event, statistics());
        return statistics();
    }

    public GenerationStats evolve(int generations) {
        if (generations < 0) {
            throw new IllegalArgumentException("generations must be >= 0");
        }
        var stats = statistics();
        for (int index = 0; index < generations; index++) {
            stats = evolve();
        }
        return stats;
    }

    public SearchResult evolveUntil(double targetError, long maxGenerations) {
        if (targetError < 0.0 || !Double.isFinite(targetError)) {
            throw new IllegalArgumentException("targetError must be finite and >= 0");
        }
        if (maxGenerations < 0) {
            throw new IllegalArgumentException("maxGenerations must be >= 0");
        }

        if (bestError <= targetError) {
            return new SearchResult(0, bestError, true, fitnessEvaluations, restarts);
        }

        var event = EvolutionJfr.search(targetError, maxGenerations);
        var evolved = 0L;
        while (evolved < maxGenerations && bestError > targetError) {
            evolve();
            evolved++;
        }

        var converged = bestError <= targetError;
        var result = new SearchResult(evolved, bestError, converged, fitnessEvaluations, restarts);
        EvolutionJfr.commitSearch(event, result);
        return result;
    }

    private void copyElites() {
        for (int elite = 0; elite < config.eliteCount(); elite++) {
            var source = ranking[elite];
            copyGenome(population, source, nextPopulation, elite);
            nextFitness[elite] = fitness[source];
        }
    }

    private int selectParent() {
        return switch (config.selection().type()) {
            case TOURNAMENT -> tournamentSelect();
            case RANK -> rankSelect();
            case ROULETTE -> rouletteSelect();
            case TRUNCATION -> truncationSelect();
        };
    }

    private int tournamentSelect() {
        var winner = random.nextInt(config.populationSize());
        var winnerFitness = fitness[winner];

        for (int competitor = 1; competitor < config.selection().tournamentSize(); competitor++) {
            var candidate = random.nextInt(config.populationSize());
            if (fitness[candidate] < winnerFitness) {
                winner = candidate;
                winnerFitness = fitness[candidate];
            }
        }
        return winner;
    }

    private int rankSelect() {
        var populationSize = config.populationSize();
        var totalWeight = (long) populationSize * (populationSize + 1L) / 2L;
        var target = random.nextLong(totalWeight);
        var cumulative = 0L;
        for (int rank = 0; rank < populationSize; rank++) {
            cumulative += populationSize - rank;
            if (target < cumulative) {
                return ranking[rank];
            }
        }
        return ranking[populationSize - 1];
    }

    private int rouletteSelect() {
        var scale = Math.max(averageError - bestError, 1.0e-15);
        var total = 0.0;
        for (var candidateFitness : fitness) {
            total += 1.0 / (1.0e-9 + Math.max(0.0, candidateFitness - bestError) / scale);
        }

        var target = random.nextDouble() * total;
        var cumulative = 0.0;
        for (int candidate = 0; candidate < fitness.length; candidate++) {
            cumulative += 1.0 / (1.0e-9 + Math.max(0.0, fitness[candidate] - bestError) / scale);
            if (target <= cumulative) {
                return candidate;
            }
        }
        return ranking[0];
    }

    private int truncationSelect() {
        var top = Math.max(
                1,
                (int) Math.ceil(config.populationSize() * config.selection().truncationFraction()));
        return ranking[random.nextInt(top)];
    }

    private void reproduce(int firstParent, int secondParent, int child) {
        if (random.nextDouble() >= config.crossover().probability()) {
            var selected = fitness[firstParent] <= fitness[secondParent] ? firstParent : secondParent;
            copyGenome(population, selected, nextPopulation, child);
            return;
        }

        switch (config.crossover().type()) {
            case SINGLE_POINT -> singlePointCrossover(firstParent, secondParent, child);
            case UNIFORM -> uniformCrossover(firstParent, secondParent, child);
            case ARITHMETIC -> arithmeticCrossover(firstParent, secondParent, child);
            case BLX_ALPHA -> blxAlphaCrossover(firstParent, secondParent, child);
        }
    }

    private void singlePointCrossover(int firstParent, int secondParent, int child) {
        var cut = random.nextInt(geneCount + 1);
        var childOffset = child * geneCount;
        var firstOffset = firstParent * geneCount;
        var secondOffset = secondParent * geneCount;

        System.arraycopy(population, firstOffset, nextPopulation, childOffset, cut);
        System.arraycopy(
                population,
                secondOffset + cut,
                nextPopulation,
                childOffset + cut,
                geneCount - cut);
    }

    private void uniformCrossover(int firstParent, int secondParent, int child) {
        var childOffset = child * geneCount;
        var firstOffset = firstParent * geneCount;
        var secondOffset = secondParent * geneCount;
        for (int gene = 0; gene < geneCount; gene++) {
            nextPopulation[childOffset + gene] =
                    random.nextBoolean() ? population[firstOffset + gene] : population[secondOffset + gene];
        }
    }

    private void arithmeticCrossover(int firstParent, int secondParent, int child) {
        var alpha = random.nextDouble();
        var inverse = 1.0 - alpha;
        var childOffset = child * geneCount;
        var firstOffset = firstParent * geneCount;
        var secondOffset = secondParent * geneCount;
        for (int gene = 0; gene < geneCount; gene++) {
            nextPopulation[childOffset + gene] = Math.fma(
                    alpha,
                    population[firstOffset + gene],
                    inverse * population[secondOffset + gene]);
        }
    }

    private void blxAlphaCrossover(int firstParent, int secondParent, int child) {
        var alpha = config.crossover().blxAlpha();
        var childOffset = child * geneCount;
        var firstOffset = firstParent * geneCount;
        var secondOffset = secondParent * geneCount;

        for (int gene = 0; gene < geneCount; gene++) {
            var first = population[firstOffset + gene];
            var second = population[secondOffset + gene];
            var low = Math.min(first, second);
            var high = Math.max(first, second);
            var interval = high - low;
            var from = low - alpha * interval;
            var to = high + alpha * interval;
            nextPopulation[childOffset + gene] =
                    clamp(Math.fma(random.nextDouble(), to - from, from));
        }
    }

    private void mutate(int child) {
        var probability = config.mutation().probability();
        if (probability == 0.0) {
            return;
        }

        if (probability == 1.0) {
            for (int gene = 0; gene < geneCount; gene++) {
                mutateGene(child, gene);
            }
            return;
        }

        if (config.mutation().geometricSkipping() && probability <= 0.25) {
            var logKeep = Math.log1p(-probability);
            var gene = geometricSkip(logKeep);
            while (gene < geneCount) {
                mutateGene(child, gene);
                gene += 1 + geometricSkip(logKeep);
            }
            return;
        }

        for (int gene = 0; gene < geneCount; gene++) {
            if (random.nextDouble() < probability) {
                mutateGene(child, gene);
            }
        }
    }

    private int geometricSkip(double logKeep) {
        return (int) (Math.log1p(-random.nextDouble()) / logKeep);
    }

    private void mutateGene(int child, int gene) {
        var index = child * geneCount + gene;
        nextPopulation[index] = switch (config.mutation().type()) {
            case RANDOM_RESET -> randomGene();
            case GAUSSIAN -> clamp(nextPopulation[index] + nextGaussian() * currentMutationSigma);
        };
    }

    private void evaluatePopulation(double[] genomes, double[] destination, int fromCandidate) {
        var count = config.populationSize() - fromCandidate;
        if (count <= 0) {
            return;
        }

        if (fitnessExecutor == null || count < config.fitnessParallelism() * 4) {
            evaluateRange(genomes, destination, fromCandidate, config.populationSize());
        } else {
            var parallelism = Math.min(config.fitnessParallelism(), count);
            var chunk = (count + parallelism - 1) / parallelism;
            var futures = new CompletableFuture<?>[parallelism];
            for (int worker = 0; worker < parallelism; worker++) {
                var from = fromCandidate + worker * chunk;
                var to = Math.min(config.populationSize(), from + chunk);
                if (from >= to) {
                    futures[worker] = CompletableFuture.completedFuture(null);
                } else {
                    futures[worker] = CompletableFuture.runAsync(
                            () -> evaluateRange(genomes, destination, from, to),
                            fitnessExecutor);
                }
            }
            CompletableFuture.allOf(futures).join();
        }
        fitnessEvaluations += count;
    }

    private void evaluateRange(double[] genomes, double[] destination, int from, int to) {
        for (int candidate = from; candidate < to; candidate++) {
            var error = evaluateGenome(genomes, candidate * geneCount);
            destination[candidate] = error;
        }
    }

    private double evaluateGenome(double[] genomes, int offset) {
        var value = fitnessFunction.evaluate(genomes, offset, geneCount);
        if (!Double.isFinite(value)) {
            throw new IllegalStateException("fitness must be finite, got " + value);
        }
        return value;
    }

    private void swapGenerations() {
        var previousPopulation = population;
        population = nextPopulation;
        nextPopulation = previousPopulation;

        var previousFitness = fitness;
        fitness = nextFitness;
        nextFitness = previousFitness;
    }

    private void refreshStatistics(boolean initial) {
        for (int candidate = 0; candidate < ranking.length; candidate++) {
            ranking[candidate] = candidate;
        }
        sortRanking(0, ranking.length - 1);

        bestIndex = ranking[0];
        bestError = fitness[bestIndex];
        worstError = fitness[ranking[ranking.length - 1]];

        var sum = 0.0;
        var sumSquares = 0.0;
        for (var value : fitness) {
            sum += value;
            sumSquares = Math.fma(value, value, sumSquares);
        }
        averageError = sum / fitness.length;
        var variance = Math.max(0.0, sumSquares / fitness.length - averageError * averageError);
        fitnessStdDev = Math.sqrt(variance);

        var middle = ranking.length / 2;
        medianError = (ranking.length & 1) == 0
                ? (fitness[ranking[middle - 1]] + fitness[ranking[middle]]) * 0.5
                : fitness[ranking[middle]];

        geneDiversity = config.trackDiversity() ? calculateGeneDiversity() : Double.NaN;
        if (initial) {
            historicalBest = bestError;
        }
    }

    private double calculateGeneDiversity() {
        var populationSize = config.populationSize();
        var totalStdDev = 0.0;
        for (int gene = 0; gene < geneCount; gene++) {
            var sum = 0.0;
            var sumSquares = 0.0;
            for (int candidate = 0; candidate < populationSize; candidate++) {
                var value = population[candidate * geneCount + gene];
                sum += value;
                sumSquares = Math.fma(value, value, sumSquares);
            }
            var mean = sum / populationSize;
            var variance = Math.max(0.0, sumSquares / populationSize - mean * mean);
            totalStdDev += Math.sqrt(variance);
        }
        return totalStdDev / geneCount;
    }

    private void updateImprovementState() {
        var threshold = Math.max(IMPROVEMENT_EPSILON, Math.abs(historicalBest) * 1.0e-12);
        if (bestError < historicalBest - threshold) {
            historicalBest = bestError;
            stagnantGenerations = 0;
            if (config.mutation().adaptive() && config.mutation().type() == MutationType.GAUSSIAN) {
                currentMutationSigma = Math.max(
                        config.mutation().minSigma(),
                        currentMutationSigma * config.mutation().sigmaDecay());
            }
        } else {
            stagnantGenerations++;
        }
    }

    private void applyAdaptivePolicy() {
        if (!config.mutation().adaptive()
                || stagnantGenerations < config.mutation().stagnationGenerations()) {
            return;
        }

        if (config.mutation().type() == MutationType.GAUSSIAN) {
            currentMutationSigma = Math.min(
                    config.mutation().maxSigma(),
                    currentMutationSigma * config.mutation().stagnationBoost());
        }

        restartWorstCandidates();
        stagnantGenerations = 0;
        restarts++;
    }

    private void restartWorstCandidates() {
        var available = config.populationSize() - config.eliteCount();
        var restartCount = Math.min(
                available,
                (int) Math.round(available * config.mutation().restartFraction()));
        if (restartCount <= 0) {
            return;
        }

        for (int index = 0; index < restartCount; index++) {
            var candidate = ranking[ranking.length - 1 - index];
            initializePopulation(population, candidate, candidate + 1);
        }
        evaluatePopulation(population, fitness, 0);
        refreshStatistics(false);
        if (bestError < historicalBest) {
            historicalBest = bestError;
        }
    }

    private void initializePopulation(double[] destination, int fromCandidate, int toCandidate) {
        for (int candidate = fromCandidate; candidate < toCandidate; candidate++) {
            var offset = candidate * geneCount;
            for (int gene = 0; gene < geneCount; gene++) {
                destination[offset + gene] = randomGene();
            }
        }
    }

    private double randomGene() {
        return Math.fma(random.nextDouble(), config.maxGene() - config.minGene(), config.minGene());
    }

    private double nextGaussian() {
        if (hasGaussianSpare) {
            hasGaussianSpare = false;
            return gaussianSpare;
        }

        var first = Math.max(random.nextDouble(), Double.MIN_VALUE);
        var second = random.nextDouble();
        var magnitude = Math.sqrt(-2.0 * Math.log(first));
        var angle = Math.PI * 2.0 * second;
        gaussianSpare = magnitude * Math.sin(angle);
        hasGaussianSpare = true;
        return magnitude * Math.cos(angle);
    }

    private double clamp(double value) {
        return Math.max(config.minGene(), Math.min(config.maxGene(), value));
    }

    private void sortRanking(int low, int high) {
        var left = low;
        var right = high;
        var pivot = fitness[ranking[(low + high) >>> 1]];

        while (left <= right) {
            while (fitness[ranking[left]] < pivot) {
                left++;
            }
            while (fitness[ranking[right]] > pivot) {
                right--;
            }
            if (left <= right) {
                var value = ranking[left];
                ranking[left] = ranking[right];
                ranking[right] = value;
                left++;
                right--;
            }
        }

        if (low < right) {
            sortRanking(low, right);
        }
        if (left < high) {
            sortRanking(left, high);
        }
    }

    private void copyGenome(double[] source, int sourceIndex, double[] destination, int destinationIndex) {
        System.arraycopy(
                source,
                sourceIndex * geneCount,
                destination,
                destinationIndex * geneCount,
                geneCount);
    }

    private void validateGenome(double[] genome) {
        Objects.requireNonNull(genome, "genome");
        if (genome.length != geneCount) {
            throw new IllegalArgumentException("genome length " + genome.length + " != expected " + geneCount);
        }
        for (var value : genome) {
            if (!Double.isFinite(value) || value < config.minGene() || value > config.maxGene()) {
                throw new IllegalArgumentException("genome contains a value outside configured bounds");
            }
        }
    }

    private static double[] validateGoal(double[] goal, Config config) {
        Objects.requireNonNull(config, "config");
        Objects.requireNonNull(goal, "goal");
        if (goal.length == 0) {
            throw new IllegalArgumentException("goal must contain at least one gene");
        }

        var copy = goal.clone();
        for (var value : copy) {
            if (!Double.isFinite(value)) {
                throw new IllegalArgumentException("goal contains a non-finite value");
            }
            if (value < config.minGene() || value > config.maxGene()) {
                throw new IllegalArgumentException(
                        "goal value " + value + " is outside configured gene bounds");
            }
        }
        return copy;
    }

    @Override
    public void close() {
        if (ownsFitnessExecutor) {
            fitnessExecutor.close();
        }
    }
}
