package experiments.tesseractviewer

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import android.view.View
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.min
import kotlin.math.sin

class TesseractView(context: Context) : View(context) {
    private enum class InteractionMode { CAMERA, FOUR_D }

    private data class Point3(val x: Float, val y: Float, val z: Float)
    private data class ScreenPoint(val x: Float, val y: Float, val depth: Float)
    private data class Control(val bounds: RectF, val label: String, val active: Boolean, val action: () -> Unit)

    private val rotations = FloatArray(6)
    private val planeNames = arrayOf("XY", "XZ", "XW", "YZ", "YW", "ZW")
    private val edgeColors = intArrayOf(
        Color.rgb(255, 118, 118),
        Color.rgb(112, 224, 166),
        Color.rgb(112, 167, 255),
        Color.rgb(211, 139, 255)
    )
    private val linePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
    }
    private val vertexPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.rgb(228, 237, 255)
        style = Paint.Style.FILL
    }
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.rgb(226, 234, 251)
        typeface = android.graphics.Typeface.create("sans", android.graphics.Typeface.NORMAL)
    }
    private val panelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(235, 12, 18, 34)
        style = Paint.Style.FILL
    }
    private val buttonPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }
    private val controls = mutableListOf<Control>()
    private val scaleDetector = ScaleGestureDetector(context, object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
        override fun onScale(detector: ScaleGestureDetector): Boolean {
            zoom = (zoom * detector.scaleFactor).coerceIn(0.45f, 3.2f)
            invalidate()
            return true
        }
    })

    private var mode = InteractionMode.FOUR_D
    private var selectedPlane = 2
    private var perspective = true
    private var autoRotate = false
    private var cameraYaw = 0.58f
    private var cameraPitch = -0.34f
    private var zoom = 1f
    private var lastX = 0f
    private var lastY = 0f
    private var downX = 0f
    private var downY = 0f
    private var touchingControls = false
    private var lastFrameNanos = 0L
    private var panelTop = 0f

    init {
        isFocusable = true
        contentDescription = "Interactive four-dimensional tesseract viewer"
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        canvas.drawColor(Color.rgb(7, 11, 22))
        updateAnimation()
        drawHeader(canvas)
        drawTesseract(canvas)
        drawControls(canvas)
        if (autoRotate) postInvalidateOnAnimation()
    }

    private fun updateAnimation() {
        val now = System.nanoTime()
        if (autoRotate && lastFrameNanos != 0L) {
            val dt = ((now - lastFrameNanos) / 1_000_000_000f).coerceAtMost(0.05f)
            rotations[selectedPlane] += dt * 0.85f
        }
        lastFrameNanos = now
    }

    private fun drawHeader(canvas: Canvas) {
        textPaint.textSize = dp(22f)
        textPaint.color = Color.rgb(235, 241, 255)
        canvas.drawText("TESSERACT 4D", dp(20f), dp(38f), textPaint)
        textPaint.textSize = dp(12f)
        textPaint.color = Color.rgb(145, 160, 190)
        val interaction = if (mode == InteractionMode.CAMERA) "3D camera" else "4D ${planeNames[selectedPlane]} rotation"
        val projection = if (perspective) "perspective" else "orthographic"
        canvas.drawText("$interaction  •  $projection", dp(20f), dp(58f), textPaint)
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

        for (edge in sortedEdges) {
            val a = screenPoints[edge.a]
            val b = screenPoints[edge.b]
            linePaint.color = edgeColors[edge.dimension]
            linePaint.alpha = depthAlpha((a.depth + b.depth) * 0.5f)
            linePaint.strokeWidth = if (edge.dimension == 3) dp(3.2f) else dp(2.0f)
            canvas.drawLine(a.x, a.y, b.x, b.y, linePaint)
        }

        for (point in screenPoints.sortedBy { it.depth }) {
            vertexPaint.alpha = depthAlpha(point.depth)
            canvas.drawCircle(point.x, point.y, dp(3.7f), vertexPaint)
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
        val usableHeight = panelTop.takeIf { it > 0f } ?: height.toFloat()
        val cameraDistance = 6.5f
        val perspective3D = cameraDistance / (cameraDistance - point.z)
        val scale = min(width.toFloat(), usableHeight) * 0.20f * zoom * perspective3D
        val centerX = width * 0.5f
        val centerY = usableHeight * 0.52f + dp(24f)
        return ScreenPoint(centerX + point.x * scale, centerY - point.y * scale, point.z)
    }

    private fun depthAlpha(depth: Float): Int = (175f + (depth + 2f) * 20f).toInt().coerceIn(105, 255)

    private fun drawControls(canvas: Canvas) {
        val panelHeight = dp(184f)
        panelTop = height - panelHeight
        controls.clear()
        canvas.drawRoundRect(RectF(0f, panelTop, width.toFloat(), height.toFloat()), dp(20f), dp(20f), panelPaint)

        val margin = dp(12f)
        val gap = dp(7f)
        val rowHeight = dp(38f)
        val row1Y = panelTop + dp(12f)
        val row1Labels = arrayOf("CAMERA", "4D", if (autoRotate) "PAUSE" else "AUTO", "RESET")
        val row1Width = (width - margin * 2 - gap * 3) / 4f
        row1Labels.forEachIndexed { index, label ->
            val left = margin + index * (row1Width + gap)
            val rect = RectF(left, row1Y, left + row1Width, row1Y + rowHeight)
            val active = (index == 0 && mode == InteractionMode.CAMERA) || (index == 1 && mode == InteractionMode.FOUR_D) || (index == 2 && autoRotate)
            val action = when (index) {
                0 -> { { mode = InteractionMode.CAMERA } }
                1 -> { { mode = InteractionMode.FOUR_D } }
                2 -> { { autoRotate = !autoRotate; lastFrameNanos = 0L } }
                else -> { { resetView() } }
            }
            addButton(canvas, rect, label, active, action)
        }

        val row2Y = row1Y + rowHeight + dp(12f)
        val planeWidth = (width - margin * 2 - gap * 5) / 6f
        planeNames.forEachIndexed { index, label ->
            val left = margin + index * (planeWidth + gap)
            val rect = RectF(left, row2Y, left + planeWidth, row2Y + rowHeight)
            addButton(canvas, rect, label, mode == InteractionMode.FOUR_D && selectedPlane == index) {
                selectedPlane = index
                mode = InteractionMode.FOUR_D
            }
        }

        val row3Y = row2Y + rowHeight + dp(12f)
        val projectionRect = RectF(margin, row3Y, margin + dp(128f), row3Y + rowHeight)
        addButton(canvas, projectionRect, if (perspective) "PERSPECTIVE" else "ORTHOGRAPHIC", true) {
            perspective = !perspective
        }

        textPaint.textSize = dp(11f)
        textPaint.color = Color.rgb(146, 160, 188)
        val hint = if (mode == InteractionMode.CAMERA) "drag: orbit camera  •  pinch: zoom" else "drag: rotate ${planeNames[selectedPlane]}  •  pinch: zoom"
        canvas.drawText(hint, projectionRect.right + dp(12f), row3Y + dp(24f), textPaint)
    }

    private fun addButton(canvas: Canvas, rect: RectF, label: String, active: Boolean, action: () -> Unit) {
        buttonPaint.color = if (active) Color.rgb(46, 74, 118) else Color.rgb(24, 33, 54)
        canvas.drawRoundRect(rect, dp(10f), dp(10f), buttonPaint)
        textPaint.textSize = if (label.length > 9) dp(10f) else dp(11f)
        textPaint.color = if (active) Color.rgb(238, 245, 255) else Color.rgb(175, 188, 214)
        textPaint.textAlign = Paint.Align.CENTER
        val y = rect.centerY() - (textPaint.ascent() + textPaint.descent()) * 0.5f
        canvas.drawText(label, rect.centerX(), y, textPaint)
        textPaint.textAlign = Paint.Align.LEFT
        controls += Control(rect, label, active, action)
    }

    private fun resetView() {
        rotations.fill(0f)
        cameraYaw = 0.58f
        cameraPitch = -0.34f
        zoom = 1f
        selectedPlane = 2
        mode = InteractionMode.FOUR_D
        perspective = true
        autoRotate = false
        lastFrameNanos = 0L
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        scaleDetector.onTouchEvent(event)
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                parent?.requestDisallowInterceptTouchEvent(true)
                lastX = event.x
                lastY = event.y
                downX = event.x
                downY = event.y
                touchingControls = event.y >= panelTop
                return true
            }
            MotionEvent.ACTION_MOVE -> {
                if (!touchingControls && event.pointerCount == 1 && !scaleDetector.isInProgress) {
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
                if (touchingControls && abs(event.x - downX) < dp(12f) && abs(event.y - downY) < dp(12f)) {
                    controls.firstOrNull { it.bounds.contains(event.x, event.y) }?.action?.invoke()
                    invalidate()
                }
                touchingControls = false
                performClick()
                return true
            }
            MotionEvent.ACTION_CANCEL -> {
                touchingControls = false
                return true
            }
        }
        return true
    }

    override fun performClick(): Boolean {
        super.performClick()
        return true
    }

    private fun dp(value: Float): Float = value * resources.displayMetrics.density
}
