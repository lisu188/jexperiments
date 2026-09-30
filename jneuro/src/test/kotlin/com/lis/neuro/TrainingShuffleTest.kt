package com.lis.neuro

import java.util.SplittableRandom
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

class TrainingShuffleTest {
    @Test fun reservationsMatchSplittableRandomAndPartialCommitReplaysEveryPendingEpoch() {
        for (size in listOf(1, 2, 3, 17, 128, 257)) {
            val random = SplittableRandom(-17)
            val expected = IntArray(size) { it }
            val actual = TrainingShuffle(-17)
            repeat(3) {
                val orders = actual.reserve(size, 64)
                assertSame(orders[0], actual.reserve(size, 1)[0])
                repeat(64) { epoch ->
                    for (index in expected.lastIndex downTo 1) {
                        val other = random.nextInt(index + 1)
                        val value = expected[index]; expected[index] = expected[other]; expected[other] = value
                    }
                    assertArrayEquals(expected, orders[epoch])
                    actual.commit(1)
                    if (epoch < 63) assertSame(orders[epoch + 1], actual.reserve(size, 1)[0])
                }
            }
        }
    }

    @Test fun datasetMutationRestoresCommittedRandomCursorAndDropsUnpublishedReservations() {
        val expected = TrainingShuffle(431)
        val actual = TrainingShuffle(431)
        for (shuffle in listOf(expected, actual)) { shuffle.reserve(31, 3); shuffle.commit(3) }
        actual.reserve(31, 64)
        expected.invalidate(); actual.invalidate()
        expected.reserve(33, 5).zip(actual.reserve(33, 5)).forEach { (left, right) -> assertArrayEquals(left, right) }
        assertThrows(IllegalArgumentException::class.java) { actual.commit(6) }
        assertThrows(IllegalArgumentException::class.java) { actual.reserve(0, 1) }
        assertThrows(IllegalArgumentException::class.java) { actual.reserve(33, 65) }
        assertThrows(IllegalArgumentException::class.java) { actual.reserve(33, 0) }
    }
}
