package com.lis;

@FunctionalInterface
public interface MultiObjectiveFitness {
    void evaluate(
            double[] genome,
            int offset,
            int length,
            double[] objectives,
            int objectiveOffset);
}
