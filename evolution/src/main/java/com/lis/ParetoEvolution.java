package com.lis;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

public final class ParetoEvolution {
    public record Statistics(
            long generation,
            int populationSize,
            int objectiveCount,
            int paretoFrontSize,
            int frontCount) {
    }

    private final int geneCount;
    private final int objectiveCount;
    private final Evolution.Config config;
    private final MultiObjectiveFitness fitness;
    private final SplittableRandom random;

    private double[] population;
    private double[] nextPopulation;
    private double[] objectives;
    private double[] nextObjectives;

    private final int[] rank;
    private final double[] crowding;
    private final int[] dominationCount;
    private final boolean[] dominationMatrix;
    private final int[] front;
    private final int[] nextFront;
    private final int[] sortBuffer;

    private long generation;
    private int frontCount;
    private int paretoFrontSize;
    private boolean hasGaussianSpare;
    private double gaussianSpare;

    public ParetoEvolution(
            int geneCount,
            int objectiveCount,
            MultiObjectiveFitness fitness,
            Evolution.Config config) {
        if (geneCount <= 0) {
            throw new IllegalArgumentException("geneCount must be positive");
        }
        if (objectiveCount < 2) {
            throw new IllegalArgumentException("objectiveCount must be >= 2");
        }
        this.geneCount = geneCount;
        this.objectiveCount = objectiveCount;
        this.fitness = Objects.requireNonNull(fitness, "fitness");
        this.config = Objects.requireNonNull(config, "config");
        random = new SplittableRandom(config.seed());

        var populationSize = config.populationSize();
        population = new double[populationSize * geneCount];
        nextPopulation = new double[population.length];
        objectives = new double[populationSize * objectiveCount];
        nextObjectives = new double[objectives.length];
        rank = new int[populationSize];
        crowding = new double[populationSize];
        dominationCount = new int[populationSize];
        dominationMatrix = new boolean[populationSize * populationSize];
        front = new int[populationSize];
        nextFront = new int[populationSize];
        sortBuffer = new int[populationSize];

        initializePopulation();
        evaluatePopulation(population, objectives);
        rankPopulation();
    }

    public Statistics statistics() {
        return new Statistics(
                generation,
                config.populationSize(),
                objectiveCount,
                paretoFrontSize,
                frontCount);
    }

    public Statistics evolve() {
        preserveParetoElites();

        for (int child = config.eliteCount(); child < config.populationSize(); child++) {
            var firstParent = paretoTournament();
            var secondParent = paretoTournament();
            reproduce(firstParent, secondParent, child);
            mutate(child);
        }

        evaluatePopulation(nextPopulation, nextObjectives);
        var oldPopulation = population;
        population = nextPopulation;
        nextPopulation = oldPopulation;

        var oldObjectives = objectives;
        objectives = nextObjectives;
        nextObjectives = oldObjectives;

        generation++;
        rankPopulation();
        return statistics();
    }

    public Statistics evolve(int generations) {
        if (generations < 0) {
            throw new IllegalArgumentException("generations must be >= 0");
        }
        var stats = statistics();
        for (int index = 0; index < generations; index++) {
            stats = evolve();
        }
        return stats;
    }

    public List<double[]> paretoFront() {
        var result = new ArrayList<double[]>(paretoFrontSize);
        for (int candidate = 0; candidate < config.populationSize(); candidate++) {
            if (rank[candidate] == 0) {
                var genome = new double[geneCount];
                System.arraycopy(population, candidate * geneCount, genome, 0, geneCount);
                result.add(genome);
            }
        }
        return List.copyOf(result);
    }

    public double[] objectives(double[] genome) {
        Objects.requireNonNull(genome, "genome");
        if (genome.length != geneCount) {
            throw new IllegalArgumentException("genome length " + genome.length + " != expected " + geneCount);
        }
        var values = new double[objectiveCount];
        fitness.evaluate(genome, 0, geneCount, values, 0);
        validateObjectives(values, 0);
        return values;
    }

    private void preserveParetoElites() {
        var order = sortedByRankAndCrowding();
        for (int elite = 0; elite < config.eliteCount(); elite++) {
            var source = order[elite];
            System.arraycopy(
                    population,
                    source * geneCount,
                    nextPopulation,
                    elite * geneCount,
                    geneCount);
        }
    }

    private int paretoTournament() {
        var winner = random.nextInt(config.populationSize());
        for (int competitor = 1; competitor < config.selection().tournamentSize(); competitor++) {
            var candidate = random.nextInt(config.populationSize());
            if (better(candidate, winner)) {
                winner = candidate;
            }
        }
        return winner;
    }

    private boolean better(int candidate, int current) {
        if (rank[candidate] != rank[current]) {
            return rank[candidate] < rank[current];
        }
        if (Double.compare(crowding[candidate], crowding[current]) != 0) {
            return crowding[candidate] > crowding[current];
        }
        return random.nextBoolean();
    }

    private int[] sortedByRankAndCrowding() {
        for (int index = 0; index < sortBuffer.length; index++) {
            sortBuffer[index] = index;
        }
        quickSortByRankAndCrowding(sortBuffer, 0, sortBuffer.length - 1);
        return sortBuffer;
    }

    private void quickSortByRankAndCrowding(int[] values, int low, int high) {
        var left = low;
        var right = high;
        var pivot = values[(low + high) >>> 1];
        while (left <= right) {
            while (comparePareto(values[left], pivot) < 0) {
                left++;
            }
            while (comparePareto(values[right], pivot) > 0) {
                right--;
            }
            if (left <= right) {
                var value = values[left];
                values[left] = values[right];
                values[right] = value;
                left++;
                right--;
            }
        }
        if (low < right) {
            quickSortByRankAndCrowding(values, low, right);
        }
        if (left < high) {
            quickSortByRankAndCrowding(values, left, high);
        }
    }

    private int comparePareto(int first, int second) {
        var rankCompare = Integer.compare(rank[first], rank[second]);
        if (rankCompare != 0) {
            return rankCompare;
        }
        return -Double.compare(crowding[first], crowding[second]);
    }

    private void rankPopulation() {
        Arrays.fill(dominationCount, 0);
        Arrays.fill(dominationMatrix, false);
        Arrays.fill(crowding, 0.0);

        var populationSize = config.populationSize();
        for (int first = 0; first < populationSize; first++) {
            for (int second = first + 1; second < populationSize; second++) {
                var dominance = dominance(first, second);
                if (dominance < 0) {
                    dominationMatrix[first * populationSize + second] = true;
                    dominationCount[second]++;
                } else if (dominance > 0) {
                    dominationMatrix[second * populationSize + first] = true;
                    dominationCount[first]++;
                }
            }
        }

        var frontSize = 0;
        for (int candidate = 0; candidate < populationSize; candidate++) {
            if (dominationCount[candidate] == 0) {
                rank[candidate] = 0;
                front[frontSize++] = candidate;
            }
        }

        paretoFrontSize = frontSize;
        frontCount = 0;
        var currentRank = 0;

        while (frontSize > 0) {
            frontCount++;
            calculateCrowding(front, frontSize);

            var nextSize = 0;
            for (int index = 0; index < frontSize; index++) {
                var candidate = front[index];
                var row = candidate * populationSize;
                for (int dominated = 0; dominated < populationSize; dominated++) {
                    if (!dominationMatrix[row + dominated]) {
                        continue;
                    }
                    dominationCount[dominated]--;
                    if (dominationCount[dominated] == 0) {
                        rank[dominated] = currentRank + 1;
                        nextFront[nextSize++] = dominated;
                    }
                }
            }

            System.arraycopy(nextFront, 0, front, 0, nextSize);
            frontSize = nextSize;
            currentRank++;
        }
    }

    private int dominance(int first, int second) {
        var firstBetter = false;
        var secondBetter = false;
        var firstOffset = first * objectiveCount;
        var secondOffset = second * objectiveCount;

        for (int objective = 0; objective < objectiveCount; objective++) {
            var left = objectives[firstOffset + objective];
            var right = objectives[secondOffset + objective];
            if (left < right) {
                firstBetter = true;
            } else if (right < left) {
                secondBetter = true;
            }
            if (firstBetter && secondBetter) {
                return 0;
            }
        }

        if (firstBetter == secondBetter) {
            return 0;
        }
        return firstBetter ? -1 : 1;
    }

    private void calculateCrowding(int[] currentFront, int frontSize) {
        if (frontSize == 0) {
            return;
        }
        if (frontSize <= 2) {
            for (int index = 0; index < frontSize; index++) {
                crowding[currentFront[index]] = Double.POSITIVE_INFINITY;
            }
            return;
        }

        for (int index = 0; index < frontSize; index++) {
            crowding[currentFront[index]] = 0.0;
        }

        for (int objective = 0; objective < objectiveCount; objective++) {
            System.arraycopy(currentFront, 0, sortBuffer, 0, frontSize);
            sortByObjective(sortBuffer, 0, frontSize - 1, objective);

            var first = sortBuffer[0];
            var last = sortBuffer[frontSize - 1];
            crowding[first] = Double.POSITIVE_INFINITY;
            crowding[last] = Double.POSITIVE_INFINITY;

            var minimum = objective(first, objective);
            var maximum = objective(last, objective);
            var range = maximum - minimum;
            if (range == 0.0) {
                continue;
            }

            for (int index = 1; index < frontSize - 1; index++) {
                var candidate = sortBuffer[index];
                if (Double.isInfinite(crowding[candidate])) {
                    continue;
                }
                var before = objective(sortBuffer[index - 1], objective);
                var after = objective(sortBuffer[index + 1], objective);
                crowding[candidate] += (after - before) / range;
            }
        }
    }

    private void sortByObjective(int[] values, int low, int high, int objective) {
        var left = low;
        var right = high;
        var pivot = objective(values[(low + high) >>> 1], objective);

        while (left <= right) {
            while (objective(values[left], objective) < pivot) {
                left++;
            }
            while (objective(values[right], objective) > pivot) {
                right--;
            }
            if (left <= right) {
                var value = values[left];
                values[left] = values[right];
                values[right] = value;
                left++;
                right--;
            }
        }

        if (low < right) {
            sortByObjective(values, low, right, objective);
        }
        if (left < high) {
            sortByObjective(values, left, high, objective);
        }
    }

    private double objective(int candidate, int objective) {
        return objectives[candidate * objectiveCount + objective];
    }

    private void reproduce(int firstParent, int secondParent, int child) {
        if (random.nextDouble() >= config.crossover().probability()) {
            var selected = better(firstParent, secondParent) ? firstParent : secondParent;
            copyGenome(selected, child);
            return;
        }

        var firstOffset = firstParent * geneCount;
        var secondOffset = secondParent * geneCount;
        var childOffset = child * geneCount;

        switch (config.crossover().type()) {
            case SINGLE_POINT -> {
                var cut = random.nextInt(geneCount + 1);
                System.arraycopy(population, firstOffset, nextPopulation, childOffset, cut);
                System.arraycopy(
                        population,
                        secondOffset + cut,
                        nextPopulation,
                        childOffset + cut,
                        geneCount - cut);
            }
            case UNIFORM -> {
                for (int gene = 0; gene < geneCount; gene++) {
                    nextPopulation[childOffset + gene] =
                            random.nextBoolean() ? population[firstOffset + gene] : population[secondOffset + gene];
                }
            }
            case ARITHMETIC -> {
                var alpha = random.nextDouble();
                var inverse = 1.0 - alpha;
                for (int gene = 0; gene < geneCount; gene++) {
                    nextPopulation[childOffset + gene] = Math.fma(
                            alpha,
                            population[firstOffset + gene],
                            inverse * population[secondOffset + gene]);
                }
            }
            case BLX_ALPHA -> {
                var alpha = config.crossover().blxAlpha();
                for (int gene = 0; gene < geneCount; gene++) {
                    var first = population[firstOffset + gene];
                    var second = population[secondOffset + gene];
                    var low = Math.min(first, second);
                    var high = Math.max(first, second);
                    var interval = high - low;
                    nextPopulation[childOffset + gene] = clamp(Math.fma(
                            random.nextDouble(),
                            (high + alpha * interval) - (low - alpha * interval),
                            low - alpha * interval));
                }
            }
        }
    }

    private void mutate(int child) {
        var probability = config.mutation().probability();
        if (probability == 0.0) {
            return;
        }

        for (int gene = 0; gene < geneCount; gene++) {
            if (probability == 1.0 || random.nextDouble() < probability) {
                var index = child * geneCount + gene;
                nextPopulation[index] = switch (config.mutation().type()) {
                    case RANDOM_RESET -> randomGene();
                    case GAUSSIAN -> clamp(nextPopulation[index] + nextGaussian() * config.mutation().sigma());
                };
            }
        }
    }

    private void initializePopulation() {
        for (int index = 0; index < population.length; index++) {
            population[index] = randomGene();
        }
    }

    private void evaluatePopulation(double[] genomes, double[] destination) {
        for (int candidate = 0; candidate < config.populationSize(); candidate++) {
            fitness.evaluate(
                    genomes,
                    candidate * geneCount,
                    geneCount,
                    destination,
                    candidate * objectiveCount);
            validateObjectives(destination, candidate * objectiveCount);
        }
    }

    private void validateObjectives(double[] values, int offset) {
        for (int objective = 0; objective < objectiveCount; objective++) {
            if (!Double.isFinite(values[offset + objective])) {
                throw new IllegalStateException("objective " + objective + " must be finite");
            }
        }
    }

    private void copyGenome(int source, int destination) {
        System.arraycopy(
                population,
                source * geneCount,
                nextPopulation,
                destination * geneCount,
                geneCount);
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
}
