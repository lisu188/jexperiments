package com.lis.neuro;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

final class NeuroVectorOps {
    private static final VectorSpecies<Double> DOUBLE_SPECIES = DoubleVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Float> FLOAT_SPECIES = FloatVector.SPECIES_PREFERRED;

    private NeuroVectorOps() {
    }

    static int doubleLanes() {
        return DOUBLE_SPECIES.length();
    }

    static int floatLanes() {
        return FLOAT_SPECIES.length();
    }

    static double dot(double[] left, int leftOffset, double[] right, int rightOffset, int length) {
        var lanes = DOUBLE_SPECIES.length();
        var limit = DOUBLE_SPECIES.loopBound(length);
        var sum = DoubleVector.zero(DOUBLE_SPECIES);
        var i = 0;
        for (; i < limit; i += lanes) {
            var a = DoubleVector.fromArray(DOUBLE_SPECIES, left, leftOffset + i);
            var b = DoubleVector.fromArray(DOUBLE_SPECIES, right, rightOffset + i);
            sum = a.fma(b, sum);
        }
        var scalar = sum.reduceLanes(VectorOperators.ADD);
        for (; i < length; i++) {
            scalar = Math.fma(left[leftOffset + i], right[rightOffset + i], scalar);
        }
        return scalar;
    }

    static void initScaled(double[] destination, double[] source, int sourceOffset, int length, double scale) {
        var lanes = DOUBLE_SPECIES.length();
        var limit = DOUBLE_SPECIES.loopBound(length);
        var i = 0;
        for (; i < limit; i += lanes) {
            DoubleVector.fromArray(DOUBLE_SPECIES, source, sourceOffset + i)
                    .mul(scale)
                    .intoArray(destination, i);
        }
        for (; i < length; i++) {
            destination[i] = source[sourceOffset + i] * scale;
        }
    }

    static void addScaled(double[] destination, double[] source, int sourceOffset, int length, double scale) {
        var lanes = DOUBLE_SPECIES.length();
        var limit = DOUBLE_SPECIES.loopBound(length);
        var i = 0;
        for (; i < limit; i += lanes) {
            var dst = DoubleVector.fromArray(DOUBLE_SPECIES, destination, i);
            var src = DoubleVector.fromArray(DOUBLE_SPECIES, source, sourceOffset + i);
            src.mul(scale).add(dst).intoArray(destination, i);
        }
        for (; i < length; i++) {
            destination[i] = Math.fma(source[sourceOffset + i], scale, destination[i]);
        }
    }

    static void applyDerivative(double[] delta, double[] activation, int length, double beta) {
        var lanes = DOUBLE_SPECIES.length();
        var limit = DOUBLE_SPECIES.loopBound(length);
        var one = DoubleVector.broadcast(DOUBLE_SPECIES, 1.0);
        var betaVector = DoubleVector.broadcast(DOUBLE_SPECIES, beta);
        var i = 0;
        for (; i < limit; i += lanes) {
            var d = DoubleVector.fromArray(DOUBLE_SPECIES, delta, i);
            var a = DoubleVector.fromArray(DOUBLE_SPECIES, activation, i);
            d.mul(a).mul(one.sub(a)).mul(betaVector).intoArray(delta, i);
        }
        for (; i < length; i++) {
            var a = activation[i];
            delta[i] *= beta * a * (1.0 - a);
        }
    }

    static void outputDelta(
            double[] target,
            int targetOffset,
            double[] activation,
            double[] delta,
            int length,
            double beta) {
        var lanes = DOUBLE_SPECIES.length();
        var limit = DOUBLE_SPECIES.loopBound(length);
        var one = DoubleVector.broadcast(DOUBLE_SPECIES, 1.0);
        var betaVector = DoubleVector.broadcast(DOUBLE_SPECIES, beta);
        var i = 0;
        for (; i < limit; i += lanes) {
            var t = DoubleVector.fromArray(DOUBLE_SPECIES, target, targetOffset + i);
            var a = DoubleVector.fromArray(DOUBLE_SPECIES, activation, i);
            t.sub(a).mul(a).mul(one.sub(a)).mul(betaVector).intoArray(delta, i);
        }
        for (; i < length; i++) {
            var a = activation[i];
            delta[i] = (target[targetOffset + i] - a) * beta * a * (1.0 - a);
        }
    }

    static void update(
            double[] weights,
            double[] velocity,
            int weightOffset,
            double[] source,
            int sourceOffset,
            int length,
            double momentum,
            double scale) {
        var lanes = DOUBLE_SPECIES.length();
        var limit = DOUBLE_SPECIES.loopBound(length);
        var momentumVector = DoubleVector.broadcast(DOUBLE_SPECIES, momentum);
        var scaleVector = DoubleVector.broadcast(DOUBLE_SPECIES, scale);
        var i = 0;
        for (; i < limit; i += lanes) {
            var v = DoubleVector.fromArray(DOUBLE_SPECIES, velocity, weightOffset + i);
            var s = DoubleVector.fromArray(DOUBLE_SPECIES, source, sourceOffset + i);
            var nextVelocity = v.mul(momentumVector).add(s.mul(scaleVector));
            var w = DoubleVector.fromArray(DOUBLE_SPECIES, weights, weightOffset + i).add(nextVelocity);
            nextVelocity.intoArray(velocity, weightOffset + i);
            w.intoArray(weights, weightOffset + i);
        }
        for (; i < length; i++) {
            var index = weightOffset + i;
            var nextVelocity = Math.fma(momentum, velocity[index], scale * source[sourceOffset + i]);
            velocity[index] = nextVelocity;
            weights[index] += nextVelocity;
        }
    }

    static void addOuterProduct(
            double[] gradient,
            int gradientOffset,
            double[] source,
            int sourceOffset,
            int length,
            double scale) {
        var lanes = DOUBLE_SPECIES.length();
        var limit = DOUBLE_SPECIES.loopBound(length);
        var i = 0;
        for (; i < limit; i += lanes) {
            var g = DoubleVector.fromArray(DOUBLE_SPECIES, gradient, gradientOffset + i);
            var s = DoubleVector.fromArray(DOUBLE_SPECIES, source, sourceOffset + i);
            s.mul(scale).add(g).intoArray(gradient, gradientOffset + i);
        }
        for (; i < length; i++) {
            gradient[gradientOffset + i] = Math.fma(
                    source[sourceOffset + i],
                    scale,
                    gradient[gradientOffset + i]);
        }
    }

    static void add(double[] destination, double[] source, int length) {
        var lanes = DOUBLE_SPECIES.length();
        var limit = DOUBLE_SPECIES.loopBound(length);
        var i = 0;
        for (; i < limit; i += lanes) {
            DoubleVector.fromArray(DOUBLE_SPECIES, destination, i)
                    .add(DoubleVector.fromArray(DOUBLE_SPECIES, source, i))
                    .intoArray(destination, i);
        }
        for (; i < length; i++) {
            destination[i] += source[i];
        }
    }

    static float dot(float[] left, int leftOffset, float[] right, int rightOffset, int length) {
        var lanes = FLOAT_SPECIES.length();
        var limit = FLOAT_SPECIES.loopBound(length);
        var sum = FloatVector.zero(FLOAT_SPECIES);
        var i = 0;
        for (; i < limit; i += lanes) {
            var a = FloatVector.fromArray(FLOAT_SPECIES, left, leftOffset + i);
            var b = FloatVector.fromArray(FLOAT_SPECIES, right, rightOffset + i);
            sum = a.fma(b, sum);
        }
        var scalar = sum.reduceLanes(VectorOperators.ADD);
        for (; i < length; i++) {
            scalar = Math.fma(left[leftOffset + i], right[rightOffset + i], scalar);
        }
        return scalar;
    }
}
