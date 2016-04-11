package com.lis;

import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

public class Evolution {
    private final List<Drone> field = new ArrayList<>();
    private final Drone goal;
    private final double mutationFactor;
    private final double genFactor;
    private final double maxPopulation;

    private Evolution(int size, double mutationFactor, double maxPopulation, double genFactor, Drone goal) {
        this.maxPopulation = maxPopulation;
        this.goal = goal;
        this.mutationFactor = mutationFactor;
        this.genFactor = genFactor;
        IntStream.range(0, size).forEach(value -> field.add(new Drone(RandomGenerator.nextDoubleArray(goal.size()))));
    }

    public static void main(String... args) {
        Evolution droneField = new Evolution(10, 0.1, 1000, 0.1, new Drone(new double[]{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));
        IntStream.range(0, 1000000).forEach(i -> {
            droneField.iterate();
            System.out.println(droneField.field.size() + " : " + (droneField.field.stream().mapToDouble(drone -> drone.compare(droneField.goal)).average()).orElseGet(() -> 0.0));
        });
    }

    private void iterate() {
        double size = field.size();
        field.removeIf(drone -> RandomGenerator.nextDouble() < drone.compare(goal) * size / maxPopulation);

        List<Drone> toCrossover = new ArrayList<>();

        field.forEach(drone -> {
            if (RandomGenerator.nextDouble() > drone.compare(goal) * field.size() / maxPopulation) {
                toCrossover.add(drone);
            }
        });

        List<Drone> crossovered = new ArrayList<>();

        IntStream.range(0, new Double(field.size() * genFactor).intValue()).forEach(value ->
                crossovered.add(toCrossover.get(RandomGenerator.nextInt(toCrossover.size()))
                        .crossover(toCrossover.get(RandomGenerator.nextInt(toCrossover.size()))).mutate(mutationFactor)));


        field.addAll(crossovered);
    }

    private static class Drone {
        private final double[] data;

        Drone(double[] data) {
            this.data = data;
        }

        Drone crossover(Drone drone) {
            assert data.length == drone.data.length;
            double[] newData = new double[size()];
            int cutPoint = RandomGenerator.nextInt(size() + 1);
            System.arraycopy(data, 0, newData, 0, cutPoint);
            System.arraycopy(drone.data, cutPoint, newData, cutPoint, size() - cutPoint);
            return new Drone(newData);
        }

        Drone mutate(double factor) {
            IntStream.range(0, Double.valueOf(size() * factor).intValue()).forEach(value -> data[RandomGenerator.nextInt(size())] = RandomGenerator.nextDouble());
            return this;
        }

        double compare(Drone drone) {
            assert data.length == drone.data.length;
            return IntStream.range(0, data.length).mapToDouble(value ->
                    (data[value] - drone.data[value]) * (data[value] - drone.data[value])).average().orElseGet(() -> 0.0);
        }

        int size() {
            return data.length;
        }

    }

    private abstract static class RandomGenerator {
        private static final Random RANDOM = new SecureRandom();

        private RandomGenerator() {
            throw new AssertionError();
        }

        static synchronized int nextInt(int size) {
            return RANDOM.nextInt(size);
        }

        static synchronized double nextDouble() {
            return RANDOM.nextDouble();
        }

        static synchronized double[] nextDoubleArray(int size) {
            double[] data = new double[size];
            IntStream.range(0, size).forEach(value -> data[value] = RandomGenerator.nextDouble());
            return data;
        }
    }
}
