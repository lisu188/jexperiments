package com.lis;

import java.util.Objects;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;

final class EvolutionVectorFitness {
    private static final jdk.incubator.vector.VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    private EvolutionVectorFitness() {
    }

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
            var vectorLimit = SPECIES.loopBound(length);
            var gene = 0;
            for (; gene < vectorLimit; gene += SPECIES.length()) {
                var candidate = DoubleVector.fromArray(SPECIES, genome, offset + gene);
                var desired = DoubleVector.fromArray(SPECIES, target, gene);
                var difference = candidate.sub(desired);
                sum += difference.mul(difference).reduceLanes(VectorOperators.ADD);
            }
            for (; gene < length; gene++) {
                var difference = genome[offset + gene] - target[gene];
                sum = Math.fma(difference, difference, sum);
            }
            return sum / length;
        };
    }

    static int preferredLaneCount() {
        return SPECIES.length();
    }
}
