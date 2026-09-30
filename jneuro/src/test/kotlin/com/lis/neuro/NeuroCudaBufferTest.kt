package com.lis.neuro

import java.lang.foreign.Arena
import java.lang.foreign.MemorySegment
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertThrows
import org.junit.jupiter.api.Test

class NeuroCudaBufferTest {
    private class FakeMemory : NeuroCudaMemory, AutoCloseable {
        private val arena = Arena.ofShared()
        var allocations = 0
        var releases = 0
        override fun allocate(byteSize: Long): MemorySegment {
            allocations++
            return arena.allocate(byteSize, 8L)
        }
        override fun release(pointer: MemorySegment) {
            require(pointer.byteSize() > 0)
            releases++
        }
        override fun copyHostToDevice(device: MemorySegment, host: MemorySegment, byteSize: Long) =
            MemorySegment.copy(host, 0L, device, 0L, byteSize)
        override fun copyDeviceToHost(host: MemorySegment, device: MemorySegment, byteSize: Long) =
            MemorySegment.copy(device, 0L, host, 0L, byteSize)
        override fun synchronize() = Unit
        override fun close() = arena.close()
    }

    @Test fun doubleBufferRoundTripsAndClosesOnce() {
        FakeMemory().use { memory ->
            val buffer = NeuroCudaBuffer.doubles(memory, 4)
            val values = doubleArrayOf(1.5, -2.0, 3.25, 9.0)
            buffer.upload(values)
            assertArrayEquals(values, buffer.downloadDoubles(4), 0.0)
            buffer.close()
            buffer.close()
            assertEquals(1, memory.allocations)
            assertEquals(1, memory.releases)
            assertThrows(IllegalStateException::class.java) { buffer.downloadDoubles(1) }
        }
    }

    @Test fun floatBufferRoundTripsAndValidatesCapacity() {
        FakeMemory().use { memory ->
            NeuroCudaBuffer.floats(memory, 3).use { buffer ->
                val values = floatArrayOf(1.25f, -3.5f, 7.0f)
                buffer.upload(values)
                assertArrayEquals(values, buffer.downloadFloats(3), 0.0f)
                assertThrows(IllegalArgumentException::class.java) { buffer.upload(FloatArray(4)) }
                assertThrows(IllegalArgumentException::class.java) { buffer.downloadFloats(4) }
            }
        }
    }

    @Test fun intUploadsAndBoundsAreValidated() {
        FakeMemory().use { memory ->
            NeuroCudaBuffer.ints(memory, 3).use { buffer ->
                buffer.upload(intArrayOf(4, 2, 1))
                assertThrows(IllegalArgumentException::class.java) { buffer.upload(IntArray(4)) }
                assertThrows(IllegalArgumentException::class.java) { buffer.downloadDoubles(2) }
            }
            assertThrows(IllegalArgumentException::class.java) { NeuroCudaBuffer.doubles(memory, 0) }
            assertThrows(IllegalArgumentException::class.java) { NeuroCudaBuffer.ints(memory, -1) }
        }
    }
}
