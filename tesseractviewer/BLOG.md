# Rendering a Tesseract on Android Without a 3D Engine

A tesseract viewer sounds like a graphics-engine problem, but the actual geometry is tiny: sixteen vertices and thirty-two edges. The interesting part is not polygon throughput. It is keeping three transformations conceptually separate: rotation in four-dimensional space, projection from 4D to 3D, and the ordinary 3D camera used to inspect the projection.

This experiment implements those stages directly in Kotlin and renders the final line segments with Android `Canvas`. There is no Unity scene, no OpenGL mesh, and no asset containing the hypercube. The UI deliberately stays separate from the renderer: `TesseractView` owns geometry and gestures, while `MainActivity` builds normal Android controls with proper touch targets, focus behavior and system-bar insets.

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

Selecting XW, YW, or ZW is therefore not a decorative morph. The vertices actually move through the fourth spatial coordinate before projection. The UI now surfaces these three W planes first because they are the interactions that best explain what makes the object four-dimensional.

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

## The 3D camera remains a separate transform

After 4D projection the viewer has ordinary 3D points. Camera yaw and pitch are deliberately applied after the 4D stage.

```java
val x1 = point.x * yawCos + point.z * yawSin
val z1 = -point.x * yawSin + point.z * yawCos
```

```java
val y2 = point.y * pitchCos - z1 * pitchSin
val z2 = point.y * pitchSin + z1 * pitchCos
```

Keeping these transforms separate avoids a common visualization mistake: making a 3D orbit look like a 4D rotation. `TesseractView` exposes explicit `FOUR_D` and `CAMERA` modes and the activity presents them as a two-choice segmented control.

## Renderer and controls are separate UI layers

The first viewer prototype drew the header, labels and all buttons directly onto the same `Canvas` as the tesseract. That was compact code, but it meant the controls were only painted rectangles. Individual controls did not exist as Android views, touch targets were only 38 dp high, keyboard focus was unavailable and accessibility services could not reason about the controls independently.

The revised design leaves `Canvas` responsible only for the visualization. State changes are exposed through a small contract:

```java
data class ViewerState(
    val mode: InteractionMode,
    val selectedPlane: Int,
    val perspective: Boolean,
    val autoRotate: Boolean
)
```

```java
fun setRotationPlane(index: Int) {
    require(index in 0..5)
    selectedPlane = index
    mode = InteractionMode.FOUR_D
    updateAccessibilityDescription()
    notifyStateChanged()
    invalidate()
}
```

`MainActivity` then renders that state into real `TextView` controls. They are clickable, focusable, at least 48 dp high and use Android ripple feedback instead of custom hit-testing.

```java
private fun actionChip(label: String, action: () -> Unit): TextView = TextView(this).apply {
    text = label
    gravity = Gravity.CENTER
    minHeight = dp(48)
    isClickable = true
    isFocusable = true
    setOnClickListener { action() }
}
```

This also removes control hit boxes, label alignment and button state logic from the rendering class. The tesseract can evolve independently from the app chrome.

## Showing what the selected plane means

Color is useful only when it conveys structure. Each edge already knows the dimension along which it runs, so the renderer can emphasize the two dimensions belonging to the selected rotation plane and reduce unrelated edges.

```java
val activeDimensions = planeDimensions[selectedPlane]
val dimensionActive = mode == InteractionMode.CAMERA || edge.dimension in activeDimensions
linePaint.alpha = if (dimensionActive) {
    depthAlpha(depth)
} else {
    (depthAlpha(depth) * 0.42f).toInt()
}
```

For XW rotation, X and W edges therefore remain visually dominant while Y and Z recede. The screen also shows a persistent X/Y/Z/W color legend. This is redundant encoding: the meaning is not left to color alone, and the selected plane is repeated in the status chip and gesture hint.

The W-containing planes are ordered first in the control strip:

```java
private val planeOrder = intArrayOf(2, 4, 5, 0, 1, 3)
```

That maps to XW, YW, ZW, followed by XY, XZ, YZ. The mathematics is unchanged; only the information architecture changes.

## Gesture model

A one-finger drag has two meanings, selected explicitly by the interaction control. In camera mode it changes yaw and pitch.

```java
cameraYaw += dx * 0.008f
cameraPitch = (cameraPitch + dy * 0.008f).coerceIn(-1.45f, 1.45f)
```

In 4D mode the same drag changes the angle of the selected rotation plane.

```java
rotations[selectedPlane] += (dx - dy * 0.35f) * 0.009f
```

Pinch zoom is independent of both modes and only changes the final display scale. The current gesture is stated in a live hint below the controls so the user does not have to infer what dragging will do.

Automatic rotation now belongs specifically to 4D mode. Switching to the 3D camera stops it rather than leaving an invisible 4D process running behind a camera-labelled interaction mode.

## Edge-to-edge and adaptive layout

The application targets Android 16 / API 36, where edge-to-edge behavior is mandatory on current Android versions. The root layout reads system-bar and display-cutout insets and pads interactive content away from those regions rather than assuming a fixed status-bar or navigation-bar height.

```java
val bars = insets.getInsets(
    WindowInsets.Type.systemBars() or WindowInsets.Type.displayCutout()
)
view.setPadding(bars.left, bars.top, bars.right, bars.bottom)
```

The same screen also changes composition with orientation. Portrait keeps the visualization above a bottom control surface. Landscape gives the visualization the flexible left side and moves controls into a fixed-width right panel. No separate activity or renderer path is needed.

## Mapping the 3D result to the screen

The final stage is a conventional perspective camera in 3D. It only exists to make the projected wireframe easy to inspect.

```java
val perspective3D = cameraDistance / (cameraDistance - point.z)
val scale = min(width.toFloat(), height.toFloat()) * 0.255f * zoom * perspective3D
```

At this point Android `Canvas` is sufficient. Each edge becomes a `drawLine` call and each vertex a small circle. Edges are sorted by average depth before drawing, which gives the wireframe a stable back-to-front visual order without introducing a depth buffer.

## Why Canvas rather than OpenGL

OpenGL would become useful if the experiment rendered filled and translucent cubic cells, large collections of polytopes, lighting, or GPU-computed projections. None of those are required for a sixteen-vertex wireframe. Canvas keeps the mathematical pipeline readable and leaves almost no graphics infrastructure between the 4D transform and the pixels.

The viewer redraws continuously only while automatic rotation is active. Manual interaction invalidates the view on demand, so an idle screen does not need a permanent render loop.

## Validation, screenshots and release packaging

The unit tests verify topology and a core invariant of the math. The generated graph must contain sixteen vertices, thirty-two total edges, and eight edges for each dimension. A composition of all six plane rotations must preserve squared 4D length within floating-point tolerance.

```java
val rotated = TesseractMath.rotate(
    original,
    floatArrayOf(0.2f, -0.5f, 0.7f, 0.3f, -0.4f, 0.9f)
)
```

CI builds the Android project independently from the repository's existing JVM Gradle project. It runs unit tests, release lint and `assembleRelease`, verifies the APK with `apksigner`, computes SHA-256 and publishes the versioned APK artifact.

A second CI job boots a Pixel emulator, installs the debug APK and captures real screenshots before and after a representative drag gesture. This is intentionally an emulator screenshot rather than a design mock: layout changes can be reviewed against the pixels Android actually renders.

The release build enables R8 minification and resource shrinking and is explicitly non-debuggable. The repository intentionally does not contain a private production keystore. To keep CI artifacts directly installable, the release variant currently uses Android's debug signing configuration. That is suitable for internal distribution and testing, not for Google Play publication or a stable long-term update channel.

## Follow-up experiments

The next useful graphics extension would be rendering the eight cubic cells instead of only the thirty-two edges. That would require transparent face ordering or a real depth-buffered renderer and would be the point where moving from Canvas to OpenGL ES becomes technically justified. Another extension is a 3D cross-section mode, where a W hyperplane slices the tesseract and the viewer renders the changing 3D intersection rather than a projection.
