package experiments.tesseractviewer

import kotlin.math.cos
import kotlin.math.sin

object TesseractMath {
    data class Vec4(val x: Float, val y: Float, val z: Float, val w: Float)
    data class Edge(val a: Int, val b: Int, val dimension: Int)

    val vertices: List<Vec4> = (0 until 16).map { index ->
        Vec4(
            if (index and 1 == 0) -1f else 1f,
            if (index and 2 == 0) -1f else 1f,
            if (index and 4 == 0) -1f else 1f,
            if (index and 8 == 0) -1f else 1f
        )
    }

    val edges: List<Edge> = buildList {
        for (vertex in 0 until 16) {
            for (dimension in 0..3) {
                val other = vertex xor (1 shl dimension)
                if (vertex < other) add(Edge(vertex, other, dimension))
            }
        }
    }

    fun rotate(vertex: Vec4, rotations: FloatArray): Vec4 {
        require(rotations.size == 6)
        val point = floatArrayOf(vertex.x, vertex.y, vertex.z, vertex.w)
        rotatePlane(point, 0, 1, rotations[0])
        rotatePlane(point, 0, 2, rotations[1])
        rotatePlane(point, 0, 3, rotations[2])
        rotatePlane(point, 1, 2, rotations[3])
        rotatePlane(point, 1, 3, rotations[4])
        rotatePlane(point, 2, 3, rotations[5])
        return Vec4(point[0], point[1], point[2], point[3])
    }

    fun project4D(vertex: Vec4, perspective: Boolean, cameraW: Float = 4f): FloatArray {
        val factor = if (perspective) cameraW / (cameraW - vertex.w) else 1f
        return floatArrayOf(vertex.x * factor, vertex.y * factor, vertex.z * factor)
    }

    private fun rotatePlane(point: FloatArray, a: Int, b: Int, angle: Float) {
        if (angle == 0f) return
        val c = cos(angle)
        val s = sin(angle)
        val va = point[a]
        val vb = point[b]
        point[a] = va * c - vb * s
        point[b] = va * s + vb * c
    }
}
