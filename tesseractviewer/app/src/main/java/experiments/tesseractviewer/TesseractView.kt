package experiments.tesseractviewer

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import android.view.View
import kotlin.math.cos
import kotlin.math.min
import kotlin.math.sin

class TesseractView(context: Context) : View(context) {
    enum class InteractionMode { CAMERA, FOUR_D }

    data class ViewerState(
        val mode: InteractionMode,
        val selectedPlane: Int,
        val perspective: Boolean,
        val autoRotate: Boolean
    )

    private data class Point3(val x: Float, val y: Float, val z: Float)
    private data class ScreenPoint(val x: Float, val y: Float, val depth: Float)

    private val rotations = FloatArray(6)
    private val planeDimensions = arrayOf(
        intArrayOf(0, 1),
        intArrayOf(0, 2),
        intArrayOf(0, 3),
        intArrayOf(1, 2),
        intArrayOf(1, 3),
        intArrayOf(2, 3)
    )
    private val edgeColors = intArrayOf(
        Color.rgb(255, 138, 128),
        Color.rgb(255, 209, 102),
        Color.rgb(102, 217, 239),
        Color.rgb(179, 136, 255)
    )
    private val linePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
    }
    private val vertexPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.rgb(235, 241, 255)
        style = Paint.Style.FILL
    }
    private val scaleDetector = ScaleGestureDetector(context, object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
        override fun onScale(detector: ScaleGestureDetector): Boolean {
            zoom = (zoom * detector.scaleFactor).coerceIn(0.5f, 3.2f)
            invalidate()
            return true
        }
    })

    var onStateChanged: ((ViewerState) -> Unit)? = null

    private var mode = InteractionMode.FOUR_D
    private var selectedPlane = 2
    private var perspective = true
    private var autoRotate = false
    private var cameraYaw = 0.58f
    private var cameraPitch = -0.34f
    private var zoom = 1f
    private var lastX = 0f
    private var lastY = 0f
    private var lastFrameNanos = 0L

    init {
        isFocusable = true
        isClickable = true
        contentDescription = "Interactive tesseract visualization. Drag to rotate in the XW plane. Pinch to zoom."
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        canvas.drawColor(Color.rgb(8, 11, 19))
        updateAnimation()
        drawTesseract(canvas)
        if (autoRotate) postInvalidateOnAnimation()
    }

    fun setInteractionMode(value: InteractionMode) {
        if (mode == value) return
        mode = value
        if (mode == InteractionMode.CAMERA) {
            autoRotate = false
            lastFrameNanos = 0L
        }
        updateAccessibilityDescription()
        notifyStateChanged()
        invalidate()
    }

    fun setRotationPlane(index: Int) {
        require(index in 0..5)
        if (selectedPlane == index && mode == InteractionMode.FOUR_D) return
        selectedPlane = index
        mode = InteractionMode.FOUR_D
        updateAccessibilityDescription()
        notifyStateChanged()
        invalidate()
    }

    fun setPerspective(value: Boolean) {
        if (perspective == value) return
        perspective = value
        notifyStateChanged()
        invalidate()
    }

    fun setAutoRotate(value: Boolean) {
        if (autoRotate == value) return
        autoRotate = value
        lastFrameNanos = 0L
        notifyStateChanged()
        invalidate()
    }

    fun resetView() {
        rotations.fill(0f)
        cameraYaw = 0.58f
        cameraPitch = -0.34f
        zoom = 1f
        selectedPlane = 2
        mode = InteractionMode.FOUR_D
        perspective = true
        autoRotate = false
        lastFrameNanos = 0L
        updateAccessibilityDescription()
        notifyStateChanged()
        invalidate()
    }

    fun state(): ViewerState = ViewerState(mode, selectedPlane, perspective, autoRotate)

    private fun notifyStateChanged() {
        onStateChanged?.invoke(state())
    }

    private fun updateAccessibilityDescription() {
        contentDescription = if (mode == InteractionMode.CAMERA) {
            "Interactive tesseract visualization. Drag to orbit the 3D camera. Pinch to zoom."
        } else {
            "Interactive tesseract visualization. Drag to rotate in the ${planeName(selectedPlane)} plane. Pinch to zoom."
        }
    }

    private fun updateAnimation() {
        val now = System.nanoTime()
        if (autoRotate && lastFrameNanos != 0L) {
            val dt = ((now - lastFrameNanos) / 1_000_000_000f).coerceAtMost(0.05f)
            rotations[selectedPlane] += dt * 0.72f
        }
        lastFrameNanos = now
    }

    private fun drawTesseract(canvas: Canvas) {
        val screenPoints = TesseractMath.vertices.map { vertex ->
            val rotated = TesseractMath.rotate(vertex, rotations)
            val projected = TesseractMath.project4D(rotated, perspective)
            toScreen(rotateCamera(Point3(projected[0], projected[1], projected[2])))
        }

        val sortedEdges = TesseractMath.edges.sortedBy { edge ->
            (screenPoints[edge.a].depth + screenPoints[edge.b].depth) * 0.5f
        }
        val activeDimensions = planeDimensions[selectedPlane]

        for (edge in sortedEdges) {
            val a = screenPoints[edge.a]
            val b = screenPoints[edge.b]
            val dimensionActive = mode == InteractionMode.CAMERA || edge.dimension in activeDimensions
            val depth = (a.depth + b.depth) * 0.5f
            linePaint.color = edgeColors[edge.dimension]
            linePaint.alpha = if (dimensionActive) depthAlpha(depth) else (depthAlpha(depth) * 0.42f).toInt()
            linePaint.strokeWidth = when {
                edge.dimension == 3 && dimensionActive -> dp(3.4f)
                dimensionActive -> dp(2.4f)
                else -> dp(1.5f)
            }
            canvas.drawLine(a.x, a.y, b.x, b.y, linePaint)
        }

        for (point in screenPoints.sortedBy { it.depth }) {
            vertexPaint.alpha = depthAlpha(point.depth)
            canvas.drawCircle(point.x, point.y, dp(3.5f), vertexPaint)
        }
        vertexPaint.alpha = 255
    }

    private fun rotateCamera(point: Point3): Point3 {
        val yawCos = cos(cameraYaw)
        val yawSin = sin(cameraYaw)
        val x1 = point.x * yawCos + point.z * yawSin
        val z1 = -point.x * yawSin + point.z * yawCos
        val pitchCos = cos(cameraPitch)
        val pitchSin = sin(cameraPitch)
        val y2 = point.y * pitchCos - z1 * pitchSin
        val z2 = point.y * pitchSin + z1 * pitchCos
        return Point3(x1, y2, z2)
    }

    private fun toScreen(point: Point3): ScreenPoint {
        val cameraDistance = 6.5f
        val perspective3D = cameraDistance / (cameraDistance - point.z)
        val scale = min(width.toFloat(), height.toFloat()) * 0.255f * zoom * perspective3D
        val centerX = width * 0.5f
        val centerY = height * 0.53f
        return ScreenPoint(centerX + point.x * scale, centerY - point.y * scale, point.z)
    }

    private fun depthAlpha(depth: Float): Int = (180f + (depth + 2f) * 20f).toInt().coerceIn(115, 255)

    override fun onTouchEvent(event: MotionEvent): Boolean {
        scaleDetector.onTouchEvent(event)
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                parent?.requestDisallowInterceptTouchEvent(true)
                lastX = event.x
                lastY = event.y
                return true
            }
            MotionEvent.ACTION_MOVE -> {
                if (event.pointerCount == 1 && !scaleDetector.isInProgress) {
                    val dx = event.x - lastX
                    val dy = event.y - lastY
                    if (mode == InteractionMode.CAMERA) {
                        cameraYaw += dx * 0.008f
                        cameraPitch = (cameraPitch + dy * 0.008f).coerceIn(-1.45f, 1.45f)
                    } else {
                        rotations[selectedPlane] += (dx - dy * 0.35f) * 0.009f
                    }
                    invalidate()
                }
                lastX = event.x
                lastY = event.y
                return true
            }
            MotionEvent.ACTION_UP -> {
                performClick()
                return true
            }
            MotionEvent.ACTION_CANCEL -> return true
        }
        return true
    }

    override fun performClick(): Boolean {
        super.performClick()
        return true
    }

    private fun planeName(index: Int): String = arrayOf("XY", "XZ", "XW", "YZ", "YW", "ZW")[index]

    private fun dp(value: Float): Float = value * resources.displayMetrics.density
}
