package com.lis.neuro

import jdk.incubator.vector.DoubleVector
import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorOperators

internal object NeuroVectorOps {
    private val doubles = DoubleVector.SPECIES_PREFERRED
    private val floats = FloatVector.SPECIES_PREFERRED

    fun doubleLanes(): Int = doubles.length()
    fun floatLanes(): Int = floats.length()

    fun dot(left: DoubleArray, leftOffset: Int, right: DoubleArray, rightOffset: Int, length: Int): Double {
        val lanes = doubles.length()
        val limit = doubles.loopBound(length)
        var sum = DoubleVector.zero(doubles)
        var index = 0
        while (index < limit) {
            val a = DoubleVector.fromArray(doubles, left, leftOffset + index)
            val b = DoubleVector.fromArray(doubles, right, rightOffset + index)
            sum = a.fma(b, sum)
            index += lanes
        }
        var scalar = sum.reduceLanes(VectorOperators.ADD)
        while (index < length) {
            scalar = Math.fma(left[leftOffset + index], right[rightOffset + index], scalar)
            index++
        }
        return scalar
    }

    fun initScaled(destination: DoubleArray, source: DoubleArray, sourceOffset: Int, length: Int, scale: Double) {
        val lanes = doubles.length()
        val limit = doubles.loopBound(length)
        var index = 0
        while (index < limit) {
            DoubleVector.fromArray(doubles, source, sourceOffset + index).mul(scale).intoArray(destination, index)
            index += lanes
        }
        while (index < length) {
            destination[index] = source[sourceOffset + index] * scale
            index++
        }
    }

    fun addScaled(destination: DoubleArray, source: DoubleArray, sourceOffset: Int, length: Int, scale: Double) {
        val lanes = doubles.length()
        val limit = doubles.loopBound(length)
        var index = 0
        while (index < limit) {
            val dst = DoubleVector.fromArray(doubles, destination, index)
            val src = DoubleVector.fromArray(doubles, source, sourceOffset + index)
            src.mul(scale).add(dst).intoArray(destination, index)
            index += lanes
        }
        while (index < length) {
            destination[index] = Math.fma(source[sourceOffset + index], scale, destination[index])
            index++
        }
    }

    fun applyDerivative(delta: DoubleArray, activation: DoubleArray, length: Int, beta: Double) {
        val lanes = doubles.length()
        val limit = doubles.loopBound(length)
        val one = DoubleVector.broadcast(doubles, 1.0)
        val betaVector = DoubleVector.broadcast(doubles, beta)
        var index = 0
        while (index < limit) {
            val d = DoubleVector.fromArray(doubles, delta, index)
            val a = DoubleVector.fromArray(doubles, activation, index)
            d.mul(a).mul(one.sub(a)).mul(betaVector).intoArray(delta, index)
            index += lanes
        }
        while (index < length) {
            val a = activation[index]
            delta[index] *= beta * a * (1.0 - a)
            index++
        }
    }

    fun outputDelta(target: DoubleArray, targetOffset: Int, activation: DoubleArray,
                    delta: DoubleArray, length: Int, beta: Double) {
        val lanes = doubles.length()
        val limit = doubles.loopBound(length)
        val one = DoubleVector.broadcast(doubles, 1.0)
        val betaVector = DoubleVector.broadcast(doubles, beta)
        var index = 0
        while (index < limit) {
            val t = DoubleVector.fromArray(doubles, target, targetOffset + index)
            val a = DoubleVector.fromArray(doubles, activation, index)
            t.sub(a).mul(a).mul(one.sub(a)).mul(betaVector).intoArray(delta, index)
            index += lanes
        }
        while (index < length) {
            val a = activation[index]
            delta[index] = (target[targetOffset + index] - a) * beta * a * (1.0 - a)
            index++
        }
    }

    fun update(weights: DoubleArray, velocity: DoubleArray, weightOffset: Int, source: DoubleArray,
               sourceOffset: Int, length: Int, momentum: Double, scale: Double) {
        val lanes = doubles.length()
        val limit = doubles.loopBound(length)
        val momentumVector = DoubleVector.broadcast(doubles, momentum)
        val scaleVector = DoubleVector.broadcast(doubles, scale)
        var index = 0
        while (index < limit) {
            val v = DoubleVector.fromArray(doubles, velocity, weightOffset + index)
            val s = DoubleVector.fromArray(doubles, source, sourceOffset + index)
            val nextVelocity = v.mul(momentumVector).add(s.mul(scaleVector))
            val w = DoubleVector.fromArray(doubles, weights, weightOffset + index).add(nextVelocity)
            nextVelocity.intoArray(velocity, weightOffset + index)
            w.intoArray(weights, weightOffset + index)
            index += lanes
        }
        while (index < length) {
            val offset = weightOffset + index
            val nextVelocity = Math.fma(momentum, velocity[offset], scale * source[sourceOffset + index])
            velocity[offset] = nextVelocity
            weights[offset] += nextVelocity
            index++
        }
    }

    fun addOuterProduct(gradient: DoubleArray, gradientOffset: Int, source: DoubleArray,
                        sourceOffset: Int, length: Int, scale: Double) {
        val lanes = doubles.length()
        val limit = doubles.loopBound(length)
        var index = 0
        while (index < limit) {
            val g = DoubleVector.fromArray(doubles, gradient, gradientOffset + index)
            val s = DoubleVector.fromArray(doubles, source, sourceOffset + index)
            s.mul(scale).add(g).intoArray(gradient, gradientOffset + index)
            index += lanes
        }
        while (index < length) {
            val offset = gradientOffset + index
            gradient[offset] = Math.fma(source[sourceOffset + index], scale, gradient[offset])
            index++
        }
    }

    fun add(destination: DoubleArray, source: DoubleArray, length: Int) {
        val lanes = doubles.length()
        val limit = doubles.loopBound(length)
        var index = 0
        while (index < limit) {
            DoubleVector.fromArray(doubles, destination, index)
                .add(DoubleVector.fromArray(doubles, source, index)).intoArray(destination, index)
            index += lanes
        }
        while (index < length) {
            destination[index] += source[index]
            index++
        }
    }

    fun dot(left: FloatArray, leftOffset: Int, right: FloatArray, rightOffset: Int, length: Int): Float {
        val lanes = floats.length()
        val limit = floats.loopBound(length)
        var sum = FloatVector.zero(floats)
        var index = 0
        while (index < limit) {
            val a = FloatVector.fromArray(floats, left, leftOffset + index)
            val b = FloatVector.fromArray(floats, right, rightOffset + index)
            sum = a.fma(b, sum)
            index += lanes
        }
        var scalar = sum.reduceLanes(VectorOperators.ADD)
        while (index < length) {
            scalar = Math.fma(left[leftOffset + index], right[rightOffset + index], scalar)
            index++
        }
        return scalar
    }
}
