package com.lis;

import org.junit.jupiter.api.Test;

import java.lang.reflect.InvocationTargetException;

import static org.junit.jupiter.api.Assertions.*;

class EvolutionTest {
    @Test
    void droneOperationsAndIterationAreCovered() throws Exception {
        Class<?> droneType = Class.forName("com.lis.Evolution$Drone");
        var droneConstructor = droneType.getDeclaredConstructor(double[].class);
        droneConstructor.setAccessible(true);

        Object goal = droneConstructor.newInstance((Object) new double[]{1.0, 1.0, 1.0, 1.0});
        Object first = droneConstructor.newInstance((Object) new double[]{1.0, 0.0, 1.0, 0.0});
        Object second = droneConstructor.newInstance((Object) new double[]{0.0, 1.0, 0.0, 1.0});

        var size = droneType.getDeclaredMethod("size");
        var compare = droneType.getDeclaredMethod("compare", droneType);
        var crossover = droneType.getDeclaredMethod("crossover", droneType);
        var mutate = droneType.getDeclaredMethod("mutate", double.class);
        for (var method : new java.lang.reflect.Method[]{size, compare, crossover, mutate}) {
            method.setAccessible(true);
        }

        assertEquals(4, size.invoke(first));
        assertEquals(0.5, (double) compare.invoke(first, goal), 0.000001);
        Object child = crossover.invoke(first, second);
        assertEquals(4, size.invoke(child));
        assertSame(child, mutate.invoke(child, 0.5));

        var evolutionConstructor = Evolution.class.getDeclaredConstructor(
                int.class, double.class, double.class, double.class, droneType);
        evolutionConstructor.setAccessible(true);
        Evolution evolution = evolutionConstructor.newInstance(16, 0.25, Double.MAX_VALUE, 0.5, goal);
        var iterate = Evolution.class.getDeclaredMethod("iterate");
        iterate.setAccessible(true);
        iterate.invoke(evolution);

        Class<?> randomType = Class.forName("com.lis.Evolution$RandomGenerator");
        for (String methodName : new String[]{"nextDouble", "nextDoubleArray"}) {
            var method = methodName.equals("nextDouble")
                    ? randomType.getDeclaredMethod(methodName)
                    : randomType.getDeclaredMethod(methodName, int.class);
            method.setAccessible(true);
            Object result = methodName.equals("nextDouble") ? method.invoke(null) : method.invoke(null, 4);
            assertNotNull(result);
        }
        var nextInt = randomType.getDeclaredMethod("nextInt", int.class);
        nextInt.setAccessible(true);
        assertTrue((int) nextInt.invoke(null, 4) >= 0);

        var randomConstructor = randomType.getDeclaredConstructor();
        randomConstructor.setAccessible(true);
        InvocationTargetException failure = assertThrows(InvocationTargetException.class, randomConstructor::newInstance);
        assertInstanceOf(AssertionError.class, failure.getCause());
    }
}
