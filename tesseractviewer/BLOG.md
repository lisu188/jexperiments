# Rendering a Tesseract on Android Without a 3D Engine

A tesseract viewer sounds like a graphics-engine problem, but the actual geometry is tiny: sixteen vertices and thirty-two edges. The interesting part is not polygon throughput. It is keeping three transformations conceptually separate: rotation in four-dimensional space, projection from 4D to 3D, and the ordinary 3D camera used to inspect the projection.

This experiment implements those stages directly in Kotlin and renders the final line segments with Android `Canvas`. There is no Unity scene, no OpenGL mesh, and no asset containing the hypercube.

## Constructing the hypercube

Every tesseract vertex is one combination of four coordinates chosen from `-1` and `1`. Four independent binary choices produce sixteen vertices.

```java
val vertices: List<Vec4> = (0 until 16).map { index ->
    Vec4(
        if (index and 1 == 0) -1f else 1f,
        if (index and 2 == 0) -1f else 1f,
        if (index and 4 == 0) -1f else 1f,
        if (index and 8 == 0) -1f else 1f
    )
}
```

Two vertices share an edge exactly when their bit patterns differ in one position. XOR with one dimension bit therefore finds the adjacent vertex without storing an edge table.

```java
for (vertex in 0 until 16) {
    for (dimension in 0..3) {
        val other = vertex xor (1 shl dimension)
        if (vertex < other) add(Edge(vertex, other, dimension))
    }
}
```

The `vertex < other` condition prevents adding every undirected edge twice. The resulting topology is sixteen vertices and thirty-two edges, with eight edges parallel to each of X, Y, Z, and W.

## Rotation is around a plane

In 3D we commonly describe rotations as being around an axis. In the matrix formulation a rotation acts in a coordinate plane. Four dimensions have six independent coordinate planes: XY, XZ, XW, YZ, YW, and ZW.

The implementation keeps one angle for each plane and applies six planar rotations in a fixed order.

```java
rotatePlane(point, 0, 1, rotations[0])
rotatePlane(point, 0, 2, rotations[1])
rotatePlane(point, 0, 3, rotations[2])
rotatePlane(point, 1, 2, rotations[3])
rotatePlane(point, 1, 3, rotations[4])
rotatePlane(point, 2, 3, rotations[5])
```

A single planar rotation only changes two coordinates.

```java
val va = point[a]
val vb = point[b]
point[a] = va * c - vb * s
point[b] = va * s + vb * c
```

This matters for the UI. Selecting XW, YW, or ZW is not a visual effect that morphs one cube into another. The underlying vertices genuinely move through the fourth spatial coordinate before they are projected.

## Projecting 4D into 3D

Perspective projection along W uses a virtual 4D camera placed at a positive W coordinate. For a camera distance `cameraW`, the scale factor is:

`cameraW / (cameraW - w)`

The same factor is applied to X, Y, and Z.

```java
val factor = if (perspective) cameraW / (cameraW - vertex.w) else 1f
return floatArrayOf(vertex.x * factor, vertex.y * factor, vertex.z * factor)
```

The familiar image of a small cube inside a larger cube is therefore a perspective property. Switching the viewer to orthographic mode removes the W-dependent scaling and makes that distinction visible.

The default `cameraW` is `4`, while a unit tesseract has vertex norm `2`. Rotations preserve that norm, so no rotated vertex can reach the projection singularity at `w = 4`.

## The 3D camera is a separate transform

After 4D projection the viewer has ordinary 3D points. Camera yaw and pitch are deliberately applied after the 4D stage.

```java
val x1 = point.x * yawCos + point.z * yawSin
val z1 = -point.x * yawSin + point.z * yawCos
```

```java
val y2 = point.y * pitchCos - z1 * pitchSin
val z2 = point.y * pitchSin + z1 * pitchCos
```

Keeping these transforms separate avoids a common visualization mistake: making a 3D orbit look like a 4D rotation. The application exposes two explicit interaction modes, `CAMERA` and `4D`, so the difference is observable rather than hidden inside gesture code.

## Mapping the 3D result to the screen

The final stage is a conventional perspective camera in 3D. It only exists to make the projected wireframe easy to inspect.

```java
val perspective3D = cameraDistance / (cameraDistance - point.z)
val scale = min(width.toFloat(), usableHeight) * 0.20f * zoom * perspective3D
```

At this point Android `Canvas` is sufficient. Each edge becomes a `drawLine` call and each vertex a small circle. Edges are sorted by average depth before drawing, which gives the wireframe a stable back-to-front visual order without introducing a depth buffer.

## Dimension-aware rendering

Each edge remembers which coordinate bit changed when it was generated. That gives the renderer dimension information at no extra geometric cost.

```java
linePaint.color = edgeColors[edge.dimension]
linePaint.strokeWidth = if (edge.dimension == 3) dp(3.2f) else dp(2.0f)
```

W-direction edges are thicker, making the fourth-dimensional connections easier to follow during XW, YW, and ZW rotations.

## Gesture model

A one-finger drag has two meanings, selected explicitly by the mode control. In camera mode it changes yaw and pitch.

```java
cameraYaw += dx * 0.008f
cameraPitch = (cameraPitch + dy * 0.008f).coerceIn(-1.45f, 1.45f)
```

In 4D mode the same drag changes the angle of the selected 4D rotation plane.

```java
rotations[selectedPlane] += (dx - dy * 0.35f) * 0.009f
```

Pinch zoom is independent of both modes and only changes the final display scale.

## Why Canvas rather than OpenGL

OpenGL would become useful if the experiment rendered filled and translucent cubic cells, large collections of polytopes, lighting, or GPU-computed projections. None of those are required for a sixteen-vertex wireframe. Canvas keeps the mathematical pipeline readable and leaves almost no graphics infrastructure between the 4D transform and the pixels.

The viewer redraws continuously only while auto-rotation is active. Manual interaction invalidates the view on demand, so an idle screen does not need a permanent render loop.

## Validation

The unit tests verify topology and a core invariant of the math. The generated graph must contain sixteen vertices, thirty-two total edges, and eight edges for each dimension. A composition of all six plane rotations must preserve squared 4D length within floating-point tolerance.

```java
val rotated = TesseractMath.rotate(
    original,
    floatArrayOf(0.2f, -0.5f, 0.7f, 0.3f, -0.4f, 0.9f)
)
```

The CI job builds the Android project independently from the repository's existing JVM Gradle project, runs unit tests and Android lint, assembles a debug APK, and uploads it as `TesseractViewer-debug`.

## Follow-up experiments

The next useful extension would be rendering the eight cubic cells instead of only the thirty-two edges. That would require transparent face ordering or a real depth-buffered renderer and would be the point where moving from Canvas to OpenGL ES becomes technically justified. Another extension is a 3D cross-section mode, where a W hyperplane slices the tesseract and the viewer renders the changing 3D intersection rather than a projection.
