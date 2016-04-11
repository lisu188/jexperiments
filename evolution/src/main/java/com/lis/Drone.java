package com.lis;

import java.security.SecureRandom;
import java.util.Random;
import java.util.stream.IntStream;

public class Drone {
    private static final Random RANDOM = new SecureRandom();
    private final double[] data;

    public Drone(double[] data) {
        this.data = data;
    }

    public Drone(int size) {
        data = new double[size];
        IntStream.range(0, size).forEach(value -> data[value] = RANDOM.nextDouble());
    }

    public Drone duplicate() {
        double[] newData = new double[data.length];
        IntStream.range(0, data.length).forEach(value ->
        {
            newData[value] = data[value] ;
        });
        return new Drone(newData);
    }

    public Drone mutate(double factor) {
        IntStream.range(0, Double.valueOf(size() * factor).intValue()).forEach(value -> data[RANDOM.nextInt(size())] = RANDOM.nextDouble());
        return this;
    }

    public double compare(Drone drone) {
        assert data.length == drone.data.length;
        return IntStream.range(0, data.length).mapToDouble(value ->
                (data[value] - drone.data[value]) * (data[value] - drone.data[value])).average().getAsDouble();
    }

    public int size() {
        return data.length;
    }

    public double[] getData() {
        return data;
    }

}
