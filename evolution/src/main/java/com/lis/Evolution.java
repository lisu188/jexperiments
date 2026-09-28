package com.lis;

import java.util.Arrays;
import java.util.Objects;
import java.util.SplittableRandom;

public final class Evolution {
    private static final long DEFAULT_SEED = 0xE701L;

    public record Config(
            int populationSize,
            int tournamentSize,
            double mutationProbability,
            double crossoverProbability,
            double minGene,
            double maxGene,
            long seed) {
        public Config {
            if (populationSize < 2) {
                throw new IllegalArgumentException("populationSize must be >= 2");
            }
            if (tournamentSize < 2 || tournamentSize > populationSize) {
                throw new IllegalArgumentException("tournamentSize must be in [2, populationSize]");
            }
            if (mutationProbability < 0.0 || mutationProbability > 1.0
                    || !Double.isFinite(mutationProbability)) {
                throw new IllegalArgumentException("mutationProbability must be finite and in [0, 1]");
            }
            if (crossoverProbability < 0.0 || crossoverProbability > 1.0
                    || !Double.isFinite(crossoverProbability)) {
                throw new IllegalArgumentException("crossoverProbability must be finite and in [0, 1]");
            }
            if (!Double.isFinite(minGene) || !Double.isFinite(maxGene) || !(minGene < maxGene)) {
                throw new IllegalArgumentException("gene bounds must be finite and minGene < maxGene");
            }
        }

        public static Config defaults() {
            return new Config(256, 4, 0.05, 0.9, 0.0, 1.0, DEFAULT_SEED);
        }

        public Config withPopulationSize(int value) {
            var tournament = Math.min(tournamentSize, value);
            return new Config(
                    value,
                    Math.max(2, tournament),
                    mutationProbability,
                    crossoverProbability,
                    minGene,
                    maxGene,
                    seed);
        }

        public Config withTournamentSize(int value) {
            return new Config(
                    populationSize,
                    value,
                    mutationProbability,
                    crossoverProbability,
                    minGene,
                    maxGene,
                    seed);
        }

        public Config withMutationProbability(double value) {
            return new Config(
                    populationSize,
                    tournamentSize,
                    value,
                    crossoverProbability,
                    minGene,
                    maxGene,
                    seed);
        }

        public Config withCrossoverProbability(double value) {
            return new Config(
                    populationSize,
                    tournamentSize,
                    mutationProbability,
                    value,
                    minGene,
                    maxGene,
                    seed);
        }

        public Config withGeneBounds(double min, double max) {
            return new Config(
                    populationSize,
                    tournamentSize,
                    mutationProbability,
                    crossoverProbability,
                    min,
                    max,
                    seed);
        }

        public Config withSeed(long value) {
            return new Config(
                    populationSize,
                    tournamentSize,
                    mutationProbability,
                    crossoverProbability,
                    minGene,
                    maxGene,
                    value);
        }
    }

    public record GenerationStats(
            long generation,
            int populationSize,
            int geneCount,
            double bestError,
            double averageError) {
    }

    public record SearchResult(
            long generations,
            double bestError,
            boolean converged) {
    }

    private final double[] goal;
    private final Config config;
    private final int geneCount;
    private final double[] fitness;
    private final SplittableRandom random;

    private double[] population;
    private double[] nextPopulation;
    private long generation;
    private int bestIndex;
    private double bestError;
    private double averageError;

    public Evolution(double[] goal) {
        this(goal, Config.defaults());
    }

    public Evolution(double[] goal, Config config) {
        this.config = Objects.requireNonNull(config, "config");
        this.goal = validateGoal(goal, config);
        geneCount = this.goal.length;
        population = new double[config.populationSize() * geneCount];
        nextPopulation = new double[population.length];
        fitness = new double[config.populationSize()];
        random = new SplittableRandom(config.seed());

        initializePopulation();
        evaluatePopulation();
    }

    public Config config() {
        return config;
    }

    public double[] goal() {
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
                averageError);
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
        Objects.requireNonNull(destination, "destination");
        if (destination.length != geneCount) {
            throw new IllegalArgumentException(
                    "destination length " + destination.length + " != expected " + geneCount);
        }
        System.arraycopy(population, bestIndex * geneCount, destination, 0, geneCount);
    }

    public GenerationStats evolve() {
        var event = EvolutionJfr.generation(generation + 1, config.populationSize(), geneCount);

        copyGenome(population, bestIndex, nextPopulation, 0);
        for (int child = 1; child < config.populationSize(); child++) {
            var firstParent = tournamentSelect();
            var secondParent = tournamentSelect();
            reproduce(firstParent, secondParent, child);
            mutate(child);
        }

        var previous = population;
        population = nextPopulation;
        nextPopulation = previous;
        generation++;
        evaluatePopulation();

        EvolutionJfr.commitGeneration(event, bestError, averageError);
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
            return new SearchResult(0, bestError, true);
        }

        var event = EvolutionJfr.search(targetError, maxGenerations);
        var evolved = 0L;
        while (evolved < maxGenerations && bestError > targetError) {
            evolve();
            evolved++;
        }

        var converged = bestError <= targetError;
        EvolutionJfr.commitSearch(event, evolved, bestError, converged);
        return new SearchResult(evolved, bestError, converged);
    }

    private void initializePopulation() {
        var min = config.minGene();
        var width = config.maxGene() - min;
        for (int index = 0; index < population.length; index++) {
            population[index] = Math.fma(random.nextDouble(), width, min);
        }
    }

    private void evaluatePopulation() {
        var best = Double.POSITIVE_INFINITY;
        var bestCandidate = -1;
        var sum = 0.0;

        for (int candidate = 0; candidate < config.populationSize(); candidate++) {
            var offset = candidate * geneCount;
            var error = meanSquaredError(population, offset, goal);
            fitness[candidate] = error;
            sum += error;
            if (error < best) {
                best = error;
                bestCandidate = candidate;
            }
        }

        bestIndex = bestCandidate;
        bestError = best;
        averageError = sum / config.populationSize();
    }

    private int tournamentSelect() {
        var winner = random.nextInt(config.populationSize());
        var winnerFitness = fitness[winner];

        for (int competitor = 1; competitor < config.tournamentSize(); competitor++) {
            var candidate = random.nextInt(config.populationSize());
            var candidateFitness = fitness[candidate];
            if (candidateFitness < winnerFitness) {
                winner = candidate;
                winnerFitness = candidateFitness;
            }
        }
        return winner;
    }

    private void reproduce(int firstParent, int secondParent, int child) {
        if (random.nextDouble() >= config.crossoverProbability()) {
            var selected = fitness[firstParent] <= fitness[secondParent] ? firstParent : secondParent;
            copyGenome(population, selected, nextPopulation, child);
            return;
        }

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

    private void mutate(int child) {
        if (config.mutationProbability() == 0.0) {
            return;
        }

        var offset = child * geneCount;
        var min = config.minGene();
        var width = config.maxGene() - min;
        for (int gene = 0; gene < geneCount; gene++) {
            if (random.nextDouble() < config.mutationProbability()) {
                nextPopulation[offset + gene] = Math.fma(random.nextDouble(), width, min);
            }
        }
    }

    static double meanSquaredError(double[] genome, int offset, double[] goal) {
        var sum = 0.0;
        for (int gene = 0; gene < goal.length; gene++) {
            var difference = genome[offset + gene] - goal[gene];
            sum = Math.fma(difference, difference, sum);
        }
        return sum / goal.length;
    }

    private void copyGenome(double[] source, int sourceIndex, double[] destination, int destinationIndex) {
        System.arraycopy(
                source,
                sourceIndex * geneCount,
                destination,
                destinationIndex * geneCount,
                geneCount);
    }

    private static double[] validateGoal(double[] goal, Config config) {
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
}
