package experiments.tesseractviewer

import android.app.Activity
import android.content.res.ColorStateList
import android.content.res.Configuration
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.graphics.drawable.RippleDrawable
import android.os.Build
import android.os.Bundle
import android.view.Gravity
import android.view.HapticFeedbackConstants
import android.view.View
import android.view.ViewGroup
import android.view.WindowInsets
import android.view.WindowInsetsController
import android.widget.FrameLayout
import android.widget.HorizontalScrollView
import android.widget.LinearLayout
import android.widget.TextView

class MainActivity : Activity() {
    private val planeNames = arrayOf("XY", "XZ", "XW", "YZ", "YW", "ZW")
    private val planeOrder = intArrayOf(2, 4, 5, 0, 1, 3)
    private val planeDescriptions = arrayOf(
        "X ↔ Y · 3D plane",
        "X ↔ Z · 3D plane",
        "X ↔ W · 4D plane",
        "Y ↔ Z · 3D plane",
        "Y ↔ W · 4D plane",
        "Z ↔ W · 4D plane"
    )

    private lateinit var viewer: TesseractView
    private lateinit var mode4D: TextView
    private lateinit var modeCamera: TextView
    private lateinit var projectionButton: TextView
    private lateinit var autoButton: TextView
    private lateinit var statusChip: TextView
    private lateinit var gestureHint: TextView
    private lateinit var planeDescription: TextView
    private val planeButtons = mutableMapOf<Int, TextView>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(buildScreen())
        configureSystemBars()
        viewer.onStateChanged = ::renderState
        renderState(viewer.state())
    }

    private fun buildScreen(): View {
        val landscape = resources.configuration.orientation == Configuration.ORIENTATION_LANDSCAPE
        val root = LinearLayout(this).apply {
            orientation = if (landscape) LinearLayout.HORIZONTAL else LinearLayout.VERTICAL
            setBackgroundColor(COLOR_BACKGROUND)
            setOnApplyWindowInsetsListener { view, insets ->
                val left: Int
                val top: Int
                val right: Int
                val bottom: Int
                if (Build.VERSION.SDK_INT >= 30) {
                    val bars = insets.getInsets(WindowInsets.Type.systemBars() or WindowInsets.Type.displayCutout())
                    left = bars.left
                    top = bars.top
                    right = bars.right
                    bottom = bars.bottom
                } else {
                    @Suppress("DEPRECATION")
                    left = insets.systemWindowInsetLeft
                    @Suppress("DEPRECATION")
                    top = insets.systemWindowInsetTop
                    @Suppress("DEPRECATION")
                    right = insets.systemWindowInsetRight
                    @Suppress("DEPRECATION")
                    bottom = insets.systemWindowInsetBottom
                }
                view.setPadding(left, top, right, bottom)
                insets
            }
        }

        val mainColumn = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(COLOR_BACKGROUND)
            addView(buildHeader(), LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
            addView(buildStage(), LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
        }
        val controls = buildControls()

        if (landscape) {
            root.addView(mainColumn, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, 1f))
            root.addView(controls, LinearLayout.LayoutParams(dp(340), ViewGroup.LayoutParams.MATCH_PARENT))
        } else {
            root.addView(mainColumn, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
            root.addView(controls, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
        }
        root.requestApplyInsets()
        return root
    }

    private fun buildHeader(): View {
        val header = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(20), dp(14), dp(20), dp(10))
        }
        val titleColumn = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
        }
        val title = TextView(this).apply {
            text = "Tesseract"
            setTextColor(COLOR_TEXT)
            textSize = 24f
            typeface = Typeface.create("sans", Typeface.BOLD)
        }
        val subtitle = TextView(this).apply {
            text = "Explore a four-dimensional hypercube"
            setTextColor(COLOR_MUTED)
            textSize = 13f
            setPadding(0, dp(2), 0, 0)
        }
        titleColumn.addView(title)
        titleColumn.addView(subtitle)
        header.addView(titleColumn, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))

        statusChip = TextView(this).apply {
            gravity = Gravity.CENTER
            textSize = 12f
            typeface = Typeface.create("sans", Typeface.BOLD)
            setPadding(dp(12), dp(8), dp(12), dp(8))
            setTextColor(COLOR_ACCENT_TEXT)
            background = roundedRipple(COLOR_ACCENT_SURFACE, 999f, COLOR_ACCENT_OUTLINE)
            importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_NO
        }
        header.addView(statusChip, LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT))
        return header
    }

    private fun buildStage(): View {
        val stage = FrameLayout(this).apply {
            setBackgroundColor(COLOR_BACKGROUND)
        }
        viewer = TesseractView(this)
        stage.addView(viewer, FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT))

        val dimensionLegend = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
            background = roundedRipple(Color.argb(218, 18, 23, 35), 999f, COLOR_OUTLINE)
            setPadding(dp(10), dp(6), dp(10), dp(6))
            contentDescription = "Dimension colors: X coral, Y amber, Z cyan, W violet"
        }
        val dimensionColors = intArrayOf(
            Color.rgb(255, 138, 128),
            Color.rgb(255, 209, 102),
            Color.rgb(102, 217, 239),
            Color.rgb(179, 136, 255)
        )
        arrayOf("X", "Y", "Z", "W").forEachIndexed { index, label ->
            val item = TextView(this).apply {
                text = "● $label"
                textSize = 12f
                setTextColor(dimensionColors[index])
                typeface = Typeface.create("sans", Typeface.BOLD)
                setPadding(dp(5), 0, dp(5), 0)
                importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_NO
            }
            dimensionLegend.addView(item)
        }
        stage.addView(dimensionLegend, FrameLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply {
            gravity = Gravity.BOTTOM or Gravity.CENTER_HORIZONTAL
            bottomMargin = dp(12)
        })
        return stage
    }

    private fun buildControls(): View {
        val panel = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(16), dp(14), dp(16), dp(14))
            background = roundedShape(COLOR_PANEL, 22f, COLOR_OUTLINE)
        }

        val modeLabel = sectionLabel("Interaction")
        panel.addView(modeLabel)

        val modeRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            setPadding(dp(4), dp(4), dp(4), dp(4))
            background = roundedShape(COLOR_SEGMENT_CONTAINER, 14f, COLOR_OUTLINE)
        }
        mode4D = actionChip("4D rotation") {
            viewer.setInteractionMode(TesseractView.InteractionMode.FOUR_D)
        }
        modeCamera = actionChip("3D camera") {
            viewer.setInteractionMode(TesseractView.InteractionMode.CAMERA)
        }
        modeRow.addView(mode4D, LinearLayout.LayoutParams(0, dp(48), 1f).apply { marginEnd = dp(4) })
        modeRow.addView(modeCamera, LinearLayout.LayoutParams(0, dp(48), 1f))
        panel.addView(modeRow, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply {
            topMargin = dp(7)
        })

        val planeHeader = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        planeHeader.addView(sectionLabel("Rotation plane"), LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        planeDescription = TextView(this).apply {
            setTextColor(COLOR_MUTED)
            textSize = 11f
            gravity = Gravity.END
        }
        planeHeader.addView(planeDescription)
        panel.addView(planeHeader, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply {
            topMargin = dp(13)
        })

        val planeScroll = HorizontalScrollView(this).apply {
            isHorizontalScrollBarEnabled = false
            isFillViewport = false
            overScrollMode = View.OVER_SCROLL_NEVER
        }
        val planeRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
        }
        planeOrder.forEachIndexed { position, planeIndex ->
            val button = actionChip(planeNames[planeIndex]) {
                viewer.setRotationPlane(planeIndex)
            }.apply {
                contentDescription = "Rotate in the ${planeNames[planeIndex]} plane"
            }
            planeButtons[planeIndex] = button
            planeRow.addView(button, LinearLayout.LayoutParams(dp(54), dp(48)).apply {
                if (position > 0) marginStart = dp(7)
                if (position == 3) marginStart = dp(16)
            })
        }
        planeScroll.addView(planeRow)
        panel.addView(planeScroll, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(48)).apply {
            topMargin = dp(7)
        })

        val actionRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
        }
        projectionButton = actionChip("Perspective") {
            viewer.setPerspective(!viewer.state().perspective)
        }
        autoButton = actionChip("Auto rotate") {
            viewer.setAutoRotate(!viewer.state().autoRotate)
        }
        val resetButton = actionChip("Reset") {
            viewer.resetView()
        }.apply {
            contentDescription = "Reset view"
        }
        actionRow.addView(projectionButton, LinearLayout.LayoutParams(0, dp(50), 1f).apply { marginEnd = dp(7) })
        actionRow.addView(autoButton, LinearLayout.LayoutParams(0, dp(50), 1f).apply { marginEnd = dp(7) })
        actionRow.addView(resetButton, LinearLayout.LayoutParams(0, dp(50), 0.72f))
        panel.addView(actionRow, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(50)).apply {
            topMargin = dp(13)
        })

        gestureHint = TextView(this).apply {
            setTextColor(COLOR_MUTED)
            textSize = 12f
            gravity = Gravity.CENTER
            setPadding(0, dp(10), 0, 0)
            accessibilityLiveRegion = View.ACCESSIBILITY_LIVE_REGION_POLITE
        }
        panel.addView(gestureHint, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
        return panel
    }

    private fun renderState(state: TesseractView.ViewerState) {
        val fourD = state.mode == TesseractView.InteractionMode.FOUR_D
        styleChip(mode4D, fourD, accent = true)
        styleChip(modeCamera, !fourD, accent = true)
        planeButtons.forEach { (index, button) ->
            styleChip(button, fourD && state.selectedPlane == index, accent = index in intArrayOf(2, 4, 5))
            setSelectionAccessibility(button, fourD && state.selectedPlane == index)
        }

        projectionButton.text = if (state.perspective) "Perspective" else "Orthographic"
        projectionButton.contentDescription = "Projection: ${projectionButton.text}. Tap to switch"
        styleChip(projectionButton, selected = false, accent = false)

        autoButton.text = if (state.autoRotate) "Pause" else "Auto rotate"
        autoButton.isEnabled = fourD
        autoButton.alpha = if (fourD) 1f else 0.45f
        autoButton.contentDescription = when {
            !fourD -> "Automatic rotation is available in 4D rotation mode"
            state.autoRotate -> "Pause automatic rotation"
            else -> "Start automatic rotation"
        }
        styleChip(autoButton, state.autoRotate && fourD, accent = true)

        statusChip.text = if (fourD) "4D · ${planeNames[state.selectedPlane]}" else "3D camera"
        planeDescription.text = if (fourD) planeDescriptions[state.selectedPlane] else "Choose a plane to return to 4D"
        gestureHint.text = if (fourD) {
            "Drag to rotate ${planeNames[state.selectedPlane]}  •  Pinch to zoom"
        } else {
            "Drag to orbit the camera  •  Pinch to zoom"
        }
        setSelectionAccessibility(mode4D, fourD)
        setSelectionAccessibility(modeCamera, !fourD)
    }

    private fun sectionLabel(text: String): TextView = TextView(this).apply {
        this.text = text
        setTextColor(COLOR_TEXT_SECONDARY)
        textSize = 12f
        typeface = Typeface.create("sans", Typeface.BOLD)
    }

    private fun actionChip(label: String, action: () -> Unit): TextView = TextView(this).apply {
        text = label
        gravity = Gravity.CENTER
        textSize = 13f
        typeface = Typeface.create("sans", Typeface.BOLD)
        setTextColor(COLOR_TEXT_SECONDARY)
        minHeight = dp(48)
        isClickable = true
        isFocusable = true
        background = roundedRipple(COLOR_CONTROL, 12f, COLOR_OUTLINE)
        setOnClickListener {
            performHapticFeedback(HapticFeedbackConstants.KEYBOARD_TAP)
            action()
        }
    }

    private fun styleChip(view: TextView, selected: Boolean, accent: Boolean) {
        val fill = when {
            selected && accent -> COLOR_ACCENT_SURFACE
            selected -> COLOR_SELECTED_NEUTRAL
            else -> COLOR_CONTROL
        }
        val stroke = when {
            selected && accent -> COLOR_ACCENT_OUTLINE
            selected -> COLOR_SELECTED_OUTLINE
            else -> COLOR_OUTLINE
        }
        view.setTextColor(if (selected) COLOR_TEXT else COLOR_TEXT_SECONDARY)
        view.background = roundedRipple(fill, 12f, stroke)
        view.isSelected = selected
    }

    private fun setSelectionAccessibility(view: View, selected: Boolean) {
        if (Build.VERSION.SDK_INT >= 30) {
            view.stateDescription = if (selected) "Selected" else "Not selected"
        }
    }

    private fun roundedShape(fill: Int, radiusDp: Float, stroke: Int): GradientDrawable = GradientDrawable().apply {
        shape = GradientDrawable.RECTANGLE
        setColor(fill)
        cornerRadius = dpF(radiusDp)
        setStroke(dp(1), stroke)
    }

    private fun roundedRipple(fill: Int, radiusDp: Float, stroke: Int): RippleDrawable {
        val content = roundedShape(fill, radiusDp, stroke)
        return RippleDrawable(ColorStateList.valueOf(Color.argb(55, 255, 255, 255)), content, null)
    }

    private fun configureSystemBars() {
        window.statusBarColor = Color.TRANSPARENT
        window.navigationBarColor = Color.TRANSPARENT
        if (Build.VERSION.SDK_INT >= 29) {
            window.isNavigationBarContrastEnforced = false
        }
        if (Build.VERSION.SDK_INT >= 30) {
            window.setDecorFitsSystemWindows(false)
            window.decorView.windowInsetsController?.setSystemBarsAppearance(
                0,
                WindowInsetsController.APPEARANCE_LIGHT_STATUS_BARS or WindowInsetsController.APPEARANCE_LIGHT_NAVIGATION_BARS
            )
        }
    }

    private fun dp(value: Int): Int = (value * resources.displayMetrics.density + 0.5f).toInt()
    private fun dpF(value: Float): Float = value * resources.displayMetrics.density

    companion object {
        private val COLOR_BACKGROUND = Color.rgb(8, 11, 19)
        private val COLOR_PANEL = Color.rgb(15, 20, 32)
        private val COLOR_SEGMENT_CONTAINER = Color.rgb(18, 24, 38)
        private val COLOR_CONTROL = Color.rgb(24, 31, 47)
        private val COLOR_SELECTED_NEUTRAL = Color.rgb(38, 48, 68)
        private val COLOR_ACCENT_SURFACE = Color.rgb(35, 55, 87)
        private val COLOR_OUTLINE = Color.rgb(42, 52, 72)
        private val COLOR_SELECTED_OUTLINE = Color.rgb(73, 91, 122)
        private val COLOR_ACCENT_OUTLINE = Color.rgb(83, 126, 184)
        private val COLOR_TEXT = Color.rgb(244, 247, 255)
        private val COLOR_TEXT_SECONDARY = Color.rgb(207, 217, 236)
        private val COLOR_MUTED = Color.rgb(158, 174, 203)
        private val COLOR_ACCENT_TEXT = Color.rgb(195, 221, 255)
    }
}
