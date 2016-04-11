package com.lis;

import java.security.SecureRandom;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import java.util.stream.IntStream;

public class DroneField {

    private static final Random RANDOM = new SecureRandom();

    private final Set<Drone> field;
    private final Drone goal;
    private final double mutationFactor;
    private final double maxPopulation;

    public DroneField(int size, double mutationFactor, double maxPopulation, Drone goal) {
        this.maxPopulation = maxPopulation;
        this.goal = goal;
        this.mutationFactor = mutationFactor;
        field = new HashSet<>();
        IntStream.range(0, size).forEach(value -> field.add(new Drone(goal.size())));
    }

    public void addDrone(Drone drone) {
        field.add(drone);
    }

    public void crossover() {
        new HashSet<>(field).forEach(drone -> {
            if (RANDOM.nextDouble() > drone.compare(goal) * field.size() / maxPopulation) {
                field.add(drone.duplicate().mutate(mutationFactor));
            }
        });
        field.removeIf(drone -> RANDOM.nextDouble() < drone.compare(goal) * field.size() / maxPopulation);
    }

    public static void main(String... args) {
        DroneField droneField = new DroneField(1, 0.5, 5.0, new Drone(new double[]{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));
        IntStream.range(0, 1000000).forEach(i -> {
            droneField.crossover();
            System.out.println(droneField.field.size() + " : " + (droneField.field.stream().mapToDouble(drone -> drone.compare(droneField.goal)).average()).orElseGet(() -> 0.0));
        });
    }

}
