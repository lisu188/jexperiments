package experiments.tesseractviewer

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

class TesseractMathTest {
    @Test
    fun topologyHasExpectedVertexAndEdgeCounts() {
        assertEquals(16, TesseractMath.vertices.size)
        assertEquals(32, TesseractMath.edges.size)
        assertEquals(8, TesseractMath.edges.count { it.dimension == 0 })
        assertEquals(8, TesseractMath.edges.count { it.dimension == 1 })
        assertEquals(8, TesseractMath.edges.count { it.dimension == 2 })
        assertEquals(8, TesseractMath.edges.count { it.dimension == 3 })
    }

    @Test
    fun fourDimensionalRotationPreservesLength() {
        val original = TesseractMath.vertices.first()
        val rotated = TesseractMath.rotate(original, floatArrayOf(0.2f, -0.5f, 0.7f, 0.3f, -0.4f, 0.9f))
        val originalLength = original.x * original.x + original.y * original.y + original.z * original.z + original.w * original.w
        val rotatedLength = rotated.x * rotated.x + rotated.y * rotated.y + rotated.z * rotated.z + rotated.w * rotated.w
        assertTrue(abs(originalLength - rotatedLength) < 0.0001f)
    }

    @Test
    fun orthographicProjectionDropsFourthCoordinate() {
        val point = TesseractMath.Vec4(1f, 2f, 3f, -9f)
        val projected = TesseractMath.project4D(point, false)
        assertEquals(1f, projected[0], 0f)
        assertEquals(2f, projected[1], 0f)
        assertEquals(3f, projected[2], 0f)
    }
}
