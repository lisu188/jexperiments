package com.lis.neuro;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class NeuroTest {
    @Test
    void trainingAndEvaluationPathsRun() throws Exception {
        var constructor = Neuro.class.getDeclaredConstructor(
                int[].class, double.class, double.class, double.class);
        constructor.setAccessible(true);
        Neuro neuro = constructor.newInstance(new int[]{2, 3, 1}, 0.2, 1.0, 0.8);

        var addTeacher = Neuro.class.getDeclaredMethod("add_teacher", double[].class, double[].class);
        addTeacher.setAccessible(true);
        addTeacher.invoke(neuro, new double[]{0.0, 0.0}, new double[]{0.0});
        addTeacher.invoke(neuro, new double[]{0.0, 1.0}, new double[]{1.0});
        addTeacher.invoke(neuro, new double[]{1.0, 0.0}, new double[]{1.0});
        addTeacher.invoke(neuro, new double[]{1.0, 1.0}, new double[]{0.0});

        var fcn = Neuro.class.getDeclaredMethod("fcn", double.class, double.class);
        var dfcn = Neuro.class.getDeclaredMethod("dfcn", double.class);
        var output = Neuro.class.getDeclaredMethod("o", double[].class);
        var error = Neuro.class.getDeclaredMethod("e", double[].class, double[].class);
        var teachIterations = Neuro.class.getDeclaredMethod("teach", int.class);
        var teachThresholdStep = Neuro.class.getDeclaredMethod("teach", double.class, int.class);
        var teachThreshold = Neuro.class.getDeclaredMethod("teach", double.class);
        var erms = Neuro.class.getDeclaredMethod("erms");

        for (var method : new java.lang.reflect.Method[]{
                fcn, dfcn, output, error, teachIterations, teachThresholdStep, teachThreshold, erms}) {
            method.setAccessible(true);
        }

        assertEquals(0.5, (double) fcn.invoke(neuro, 0.0, 1.0), 0.000001);
        assertEquals(0.25, (double) dfcn.invoke(neuro, 0.5), 0.000001);
        output.invoke(neuro, (Object) new double[]{1.0, 0.0});
        error.invoke(neuro, new double[]{1.0, 0.0}, new double[]{1.0});
        teachIterations.invoke(neuro, 2);
        assertTrue((double) erms.invoke(neuro) >= 0.0);
        assertEquals(0, teachThresholdStep.invoke(neuro, 10.0, 1));
        assertEquals(0, teachThreshold.invoke(neuro, 10.0));

        neuro.add_test(new double[]{0.0, 0.0}, new double[]{0.0});
        neuro.add_test(new double[]{1.0, 0.0}, new double[]{1.0});
        assertTrue(Double.isFinite(neuro.test()));
    }

    @Test
    void teachingWithoutSamplesIsANoOp() throws Exception {
        var constructor = Neuro.class.getDeclaredConstructor(
                int[].class, double.class, double.class, double.class);
        constructor.setAccessible(true);
        Neuro neuro = constructor.newInstance(new int[]{1, 1}, 0.1, 1.0, 0.1);

        var teach = Neuro.class.getDeclaredMethod("teach", int.class);
        teach.setAccessible(true);
        teach.invoke(neuro, 1);

        assertTrue(Double.isNaN(neuro.test()));
    }
}
