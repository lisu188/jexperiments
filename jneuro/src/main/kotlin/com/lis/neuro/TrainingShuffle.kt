package com.lis.neuro

import java.util.SplittableRandom
import java.util.random.RandomGenerator

/** Bounded, reusable reservations. Only commit advances the recoverable shuffle checkpoint. */
internal class TrainingShuffle(seed: Long) {
    private class Cursor(var seed: Long) : RandomGenerator {
        private val random = SplittableRandom(seed)
        override fun nextLong(): Long { seed += GAMMA; return random.nextLong() }
        override fun nextInt(): Int { seed += GAMMA; return random.nextInt() }
    }
    private data class Pending(val order: IntArray, val seed: Long)
    private var committedSeed = seed
    private var random = Cursor(seed)
    private var order = IntArray(0)
    private val pending = ArrayDeque<Pending>()
    private val recycled = ArrayDeque<IntArray>()

    fun reserve(size: Int, epochs: Int): Array<IntArray> {
        require(size > 0 && epochs in 1..64)
        if (order.size != size) {
            invalidate()
            order = IntArray(size) { it }
        }
        while (pending.size < epochs) {
            for (index in order.lastIndex downTo 1) {
                val other = random.nextInt(index + 1)
                val value = order[index]; order[index] = order[other]; order[other] = value
            }
            val buffer = if (recycled.isEmpty()) IntArray(size) else recycled.removeFirst()
            order.copyInto(buffer)
            pending.addLast(Pending(buffer, random.seed))
        }
        return Array(epochs) { pending[it].order }
    }

    fun commit(epochs: Int) {
        require(epochs in 1..pending.size) { "No shuffle reservation for $epochs epochs" }
        repeat(epochs) {
            val completed = pending.removeFirst()
            committedSeed = completed.seed
            recycled.addLast(completed.order)
        }
    }

    fun invalidate() {
        pending.clear(); recycled.clear()
        order = IntArray(0)
        random = Cursor(committedSeed)
    }

    companion object { private const val GAMMA = -7046029254386353131L }
}
