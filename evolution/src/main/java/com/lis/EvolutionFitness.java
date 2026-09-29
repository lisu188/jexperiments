package com.lis;

import java.util.Objects;

@FunctionalInterface
public interface EvolutionFitness {
    double evaluate(double[] genome, int offset, int length);

    static EvolutionFitness meanSquaredError(double[] goal) {
        Objects.requireNonNull(goal, "goal");
        if (goal.length == 0) {
            throw new IllegalArgumentException("goal must contain at least one gene");
        }
        var target = goal.clone();
        return (genome, offset, length) -> {
            if (length != target.length) {
                throw new IllegalArgumentException("genome length " + length + " != goal length " + target.length);
            }
            var sum = 0.0;
            for (int gene = 0; gene < length; gene++) {
                var difference = genome[offset + gene] - target[gene];
                sum = Math.fma(difference, difference, sum);
            }
            return sum / length;
        };
    }

    static EvolutionFitness vectorMeanSquaredError(double[] goal) {
        return EvolutionVectorFitness.meanSquaredError(goal);
    }
}
