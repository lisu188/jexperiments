package com.lis.neuro

import com.formdev.flatlaf.FlatDarkLaf
import java.awt.*
import java.awt.event.*
import java.awt.image.BufferedImage
import java.io.File
import java.util.Locale
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicLong
import javax.imageio.ImageIO
import javax.swing.*
import javax.swing.border.EmptyBorder
import kotlin.math.abs
import kotlin.math.roundToInt

class NeuroXorCanvas private constructor(initial: StudioFrame? = null, private val startWorker: Boolean = true) : AutoCloseable {
    private enum class View(val title: String) {
        OVERVIEW("Overview"), NEURONS("Neurons"), DATA("Learning set"), UPDATE("Step effect"),
        PARAMETERS("Parameters"), SEEDS("Seeds"), TIMELINE("Timeline"), SEARCH("Architecture search")
    }

    private val root = JPanel(BorderLayout(0, 0))
    private val sidebar = JPanel()
    private val tabs = JTabbedPane()
    private val charts = View.entries.filter { it != View.SEARCH }.associateWith { Chart(it) }
    private val hidden = JTextField("6")
    private val dataset = JComboBox(NeuroLearningSets.Kind.entries.toTypedArray())
    private val backend = JComboBox(TrainingBackend.entries.toTypedArray())
    private val deviceStatus = JLabel("CPU · awaiting training").apply { accessibleContext.accessibleName = "Training device" }
    private val seed = JTextField("42")
    private val epochLimit = NumericInputs.spinner(10_000, 1000)
    private val rate = NumericInputs.spinner(0.6, 0.05)
    private val momentum = NumericInputs.spinner(0.2, 0.05)
    private val target = NumericInputs.spinner(0.05, 0.01)
    private val configError = JLabel(" ").apply { accessibleContext.accessibleName = "Configuration error" }
    private val description = JLabel()
    private val status = JLabel("Starting…")
    private val activeTopology = JLabel(" ")
    private val train = JButton("Train")
    private val step = JButton("1 epoch")
    private val stepTen = JButton("10 epochs")
    private val reset = JButton("Reset")
    private val study = JButton("Compare 4 seeds")
    private val cancelStudy = JButton("Cancel study")
    private val neuronGallery = NeuronGallery(synchronous = !startWorker)
    private val parameterLayer = JComboBox<String>().apply { accessibleContext.accessibleName = "Parameter layer" }
    private val parameterPage = NumericInputs.spinner(0, 24).apply { accessibleContext.accessibleName = "First parameter" }
    private val customClass = JComboBox(arrayOf("Class 1 · circle", "Class 0 · square")).apply { accessibleContext.accessibleName = "Point class" }
    private val contextBar = JPanel(FlowLayout(FlowLayout.LEFT, 12, 8))
    private val metrics = List(4) { index -> JLabel("—").apply { name = "metric-$index" } }
    private val metricDetails = List(4) { JLabel(" ") }
    private val commands = LinkedBlockingQueue<(NeuroStudio) -> Unit>()
    private val revision = AtomicLong()
    private val searchSession = ArchitectureSearchSession()
    @Volatile private var searchRunning = false
    private val architectureSearch = ArchitectureSearchPanel(
        { config, evaluation, fraction, splitSeed -> startArchitectureSearch(config, evaluation, fraction, splitSeed) },
        { searchSession.cancel() },
        { result, candidate ->
            post { it.applyArchitecture(result, candidate) }
            tabs.selectedIndex = View.OVERVIEW.ordinal
        },
        { result, candidate, trial -> replayArchitecture(result, candidate, trial) }
    )
    @Volatile private var published: StudioFrame? = initial
    @Volatile private var closing = false
    @Volatile private var studyRunning = false
    @Volatile private var speed = 10
    private var frame: StudioFrame? = initial
    private var shownConfig: StudioConfig? = null
    private var updating = false
    private var hoverX = 0.5
    private var hoverY = 0.5
    private var selectedParameterLayer = 0
    private var selectedParameterStart = 0
    private var thread: Thread? = null
    @Volatile internal var trainingSessionFactory: (Neuro, TrainingBackend) -> NeuroTrainingSession =
        { model, selected -> model.newTrainingSession(selected) }
    private val timer = Timer(75) { refresh() }

    init {
        root.background = BACKGROUND
        sidebar.layout = BoxLayout(sidebar, BoxLayout.Y_AXIS)
        sidebar.background = SURFACE
        sidebar.border = EmptyBorder(24, 20, 20, 20)
        sidebar.minimumSize = Dimension(286, 0)
        buildSidebar()
        val sideScroll = JScrollPane(sidebar).apply {
            border = EmptyBorder(0, 0, 0, 1)
            horizontalScrollBarPolicy = ScrollPaneConstants.HORIZONTAL_SCROLLBAR_NEVER
            verticalScrollBar.unitIncrement = 20
            preferredSize = Dimension(286, 700)
        }
        root.add(sideScroll, BorderLayout.WEST)
        root.add(buildWorkspace(), BorderLayout.CENTER)
        bind("control ENTER", "Apply configuration") { applySettings() }
        bind("control SPACE", "Toggle training") { toggleTraining() }
        bind("control RIGHT", "Single epoch") { if (!searchRunning && !studyRunning) post(false) { it.step(1) } }
        bind("control R", "Reset model") { post { it.apply(it.activeConfig) } }
        dataset.addActionListener { updateDescription() }
        updateDescription()
        if (initial != null) {
            hidden.text = initial.config.hidden
            dataset.selectedItem = initial.config.dataset
            seed.text = initial.config.seed.toString()
            backend.selectedItem = initial.config.backend
            epochLimit.value = initial.config.maxEpochs
            refresh()
        }
        if (startWorker) {
            commands.offer { }
            thread = Thread.ofPlatform().name("jneuro-studio").daemon().start { workerLoop() }
            timer.start()
        }
    }

    private fun buildSidebar() {
        sidebar.add(JLabel("JNeuro").apply { font = uiFont(28, Font.BOLD); foreground = TEXT; alignmentX = 0f })
        sidebar.add(JLabel("NEURAL LEARNING STUDIO").apply { font = uiFont(10, Font.BOLD); foreground = ACCENT; alignmentX = 0f })
        sidebar.add(Box.createVerticalStrut(32))
        section("01  LEARNING SET")
        field("Dataset", dataset)
        description.foreground = MUTED
        description.font = uiFont(12)
        description.alignmentX = 0f
        description.maximumSize = Dimension(242, 88)
        sidebar.add(description)
        sidebar.add(Box.createVerticalStrut(22))
        section("02  ARCHITECTURE")
        field("Hidden layers", hidden)
        hidden.toolTipText = "Comma-separated widths: 2 or 6,4,2. Empty = 2 → 1 baseline. Positive widths; limited only by numeric representation and available heap."
        sidebar.add(note("One width per hidden layer.<br>Example: 4,3 means 2 → 4 → 3 → 1.<br>Leave empty for 2 → 1 (no hidden layer)."))
        val layerButtons = JPanel(GridLayout(1, 2, 8, 0)).apply {
            isOpaque = false; alignmentX = 0f; maximumSize = Dimension(242, 34)
            add(JButton("− Layer").apply { addActionListener { editLayers(false) } })
            add(JButton("+ Layer").apply { addActionListener { editLayers(true) } })
        }
        sidebar.add(Box.createVerticalStrut(8)); sidebar.add(layerButtons)
        sidebar.add(Box.createVerticalStrut(8))
        sidebar.add(JButton("Auto search…").apply {
            alignmentX = 0f; maximumSize = Dimension(242, 34)
            addActionListener { tabs.selectedIndex = View.SEARCH.ordinal }
            toolTipText = "Compare architecture size, RMSE and reliability across seeds."
        })
        sidebar.add(Box.createVerticalStrut(20))
        section("03  TRAINING")
        field("Training backend", backend)
        backend.toolTipText = "CPU is the default. CUDA requires a compatible NVIDIA GPU; failures never fall back to CPU."
        field("Random seed", seed)
        field("Maximum epochs", epochLimit)
        val advanced = JPanel().apply { layout = BoxLayout(this, BoxLayout.Y_AXIS); isOpaque = false; alignmentX = 0f }
        field("Learning rate", rate, advanced)
        field("Momentum", momentum, advanced)
        field("Target RMSE", target, advanced)
        advanced.isVisible = false
        sidebar.add(JCheckBox("Advanced settings").apply {
            isOpaque = false; alignmentX = 0f
            addActionListener { advanced.isVisible = isSelected; sidebar.revalidate() }
        })
        sidebar.add(advanced)
        sidebar.add(Box.createVerticalStrut(16))
        sidebar.add(JButton("Apply & restart").apply {
            alignmentX = 0f; maximumSize = Dimension(242, 40)
            background = ACCENT; foreground = BACKGROUND; font = uiFont(14, Font.BOLD)
            toolTipText = "Apply settings and restart training · Ctrl+Enter"
            addActionListener { applySettings() }
        })
        configError.foreground = ERROR; configError.font = uiFont(12)
        configError.alignmentX = 0f; configError.maximumSize = Dimension(242, 80)
        sidebar.add(Box.createVerticalStrut(8)); sidebar.add(configError)
        sidebar.add(note("Applying settings resets the model, history and checkpoints. Editing fields alone does not change the active run."))
        sidebar.add(Box.createVerticalGlue())
        sidebar.add(Box.createVerticalStrut(24))
        sidebar.add(note("A failed training run is not proof that an architecture cannot solve a dataset. Compare seeds and training budgets."))
    }

    private fun buildWorkspace(): JPanel {
        val workspace = JPanel(BorderLayout(0, 0)).apply { background = BACKGROUND; border = EmptyBorder(20, 24, 12, 24) }
        val top = JPanel().apply { layout = BoxLayout(this, BoxLayout.Y_AXIS); isOpaque = false }
        val heading = JPanel(BorderLayout()).apply {
            isOpaque = false; maximumSize = Dimension(Int.MAX_VALUE, 64)
            add(JPanel(GridLayout(2, 1)).apply {
                isOpaque = false
                add(JLabel("Watch a network learn").apply { font = uiFont(24, Font.BOLD); foreground = TEXT })
                add(activeTopology.apply { font = uiFont(12); foreground = MUTED })
            }, BorderLayout.CENTER)
            add(status.apply {
                font = uiFont(12, Font.BOLD); foreground = ACCENT; isOpaque = true; background = SURFACE
                border = EmptyBorder(10, 14, 10, 14); horizontalAlignment = SwingConstants.CENTER
                accessibleContext.accessibleName = "Training status"
            }, BorderLayout.EAST)
        }
        top.add(heading)
        top.add(deviceStatus.apply { font = uiFont(12); foreground = MUTED; alignmentX = 0f })
        top.add(Box.createVerticalStrut(18))
        val metricRow = JPanel(GridLayout(1, 4, 12, 0)).apply {
            isOpaque = false; maximumSize = Dimension(Int.MAX_VALUE, 90)
            val names = listOf("EPOCH", "TRAINING RMSE", "PARAMETERS", "TRAINING SAMPLES")
            for (i in names.indices) add(JPanel(BorderLayout(0, 6)).apply {
                background = SURFACE; border = EmptyBorder(12, 16, 12, 16)
                add(JLabel(names[i]).apply { font = uiFont(10, Font.BOLD); foreground = MUTED }, BorderLayout.NORTH)
                add(metrics[i].apply { font = uiFont(22, Font.BOLD); foreground = TEXT }, BorderLayout.CENTER)
                add(metricDetails[i].apply { font = uiFont(11); foreground = MUTED }, BorderLayout.SOUTH)
            })
        }
        top.add(metricRow); top.add(Box.createVerticalStrut(12))
        val controls = JPanel(FlowLayout(FlowLayout.LEFT, 8, 8)).apply { isOpaque = false; maximumSize = Dimension(Int.MAX_VALUE, 48) }
        train.background = ACCENT; train.foreground = BACKGROUND; train.preferredSize = Dimension(108, 34)
        train.font = uiFont(13, Font.BOLD); train.toolTipText = "Start or pause · Ctrl+Space"
        train.addActionListener { toggleTraining() }
        step.toolTipText = "Advance exactly one complete epoch · Ctrl+Right"
        step.addActionListener { post(false) { it.step(1) } }
        stepTen.addActionListener { post(false) { it.step(10) } }
        reset.toolTipText = "Reset the active configuration · Ctrl+R"
        reset.addActionListener { post { it.apply(it.activeConfig) } }
        controls.add(train); controls.add(step); controls.add(stepTen); controls.add(reset)
        controls.add(JLabel("   Epochs / refresh").apply { foreground = MUTED })
        controls.add(NumericInputs.spinner(10, 10).apply {
            accessibleContext.accessibleName = "Epochs per refresh"
            (editor as JSpinner.DefaultEditor).textField.accessibleContext.accessibleName = "Epochs per refresh"
            addChangeListener {
                val next = (value as Number).toInt()
                if (next > 0) { speed = next; configError.text = " " }
                else showConfigError("Epochs per refresh must be positive.")
            }
        })
        controls.add(JButton("Save PNG").apply { addActionListener { exportImage() } })
        top.add(JScrollPane(controls).apply {
            border = EmptyBorder(0,0,0,0)
            viewport.background = BACKGROUND
            verticalScrollBarPolicy = ScrollPaneConstants.VERTICAL_SCROLLBAR_NEVER
            maximumSize = Dimension(Int.MAX_VALUE,64)
            preferredSize = Dimension(640,64)
        })
        workspace.add(top, BorderLayout.NORTH)
        val center = JPanel(BorderLayout()).apply { isOpaque = false }
        contextBar.isOpaque = false
        for ((view, chart) in charts) {
            val scroll = JScrollPane(chart).apply {
                border = EmptyBorder(0, 0, 0, 0); viewport.background = BACKGROUND
                horizontalScrollBarPolicy = ScrollPaneConstants.HORIZONTAL_SCROLLBAR_NEVER
                verticalScrollBar.unitIncrement = 28
            }
            tabs.addTab(view.title, scroll)
        }
        tabs.addTab(View.SEARCH.title, JScrollPane(architectureSearch).apply {
            border = EmptyBorder(0, 0, 0, 0)
            verticalScrollBar.unitIncrement = 28
        })
        tabs.addChangeListener { updateContextBar(); charts.values.forEach { it.revalidate(); it.repaint() } }
        tabs.font = uiFont(13)
        tabs.tabLayoutPolicy = JTabbedPane.SCROLL_TAB_LAYOUT
        center.add(tabs, BorderLayout.CENTER)
        center.add(contextBar, BorderLayout.NORTH)
        parameterLayer.addActionListener {
            if (!updating) {
                selectedParameterLayer = parameterLayer.selectedIndex.coerceAtLeast(0); selectedParameterStart = 0
                updateParameterPage(); charts[View.PARAMETERS]?.repaint()
            }
        }
        parameterPage.addChangeListener {
            if (!updating) {
                val next = (parameterPage.value as Number).toInt()
                val current = frame
                val count = current?.diagnostics?.let { it.layerInputCount(selectedParameterLayer).toLong() * it.layerOutputCount(selectedParameterLayer) } ?: 0
                if (next >= 0 && next < count) { selectedParameterStart = next; configError.text = " "; charts[View.PARAMETERS]?.repaint() }
                else showConfigError("Parameter offset is outside the active layer.")
            }
        }
        study.addActionListener { startStudy() }
        cancelStudy.addActionListener { revision.incrementAndGet(); studyRunning = false; refresh() }
        workspace.add(center, BorderLayout.CENTER)
        updateContextBar()
        return workspace
    }

    private fun section(title: String) {
        sidebar.add(JLabel(title).apply { font = uiFont(10, Font.BOLD); foreground = MUTED; alignmentX = 0f })
        sidebar.add(Box.createVerticalStrut(10))
    }

    private fun field(title: String, component: JComponent, targetPanel: JPanel = sidebar) {
        targetPanel.add(JLabel(title).apply { labelFor = component; font = uiFont(12); foreground = TEXT; alignmentX = 0f })
        targetPanel.add(Box.createVerticalStrut(6))
        component.alignmentX = 0f; component.maximumSize = Dimension(242, 34)
        component.preferredSize = Dimension(242, 34); component.accessibleContext.accessibleName = title
        if (component is JSpinner) (component.editor as JSpinner.DefaultEditor).textField.accessibleContext.accessibleName = title
        targetPanel.add(component); targetPanel.add(Box.createVerticalStrut(12))
    }

    private fun note(text: String): JLabel = JLabel("<html><div style='width:174px'>$text</div></html>").apply {
        font = uiFont(12); foreground = MUTED; alignmentX = 0f
    }

    private fun updateDescription() {
        val kind = dataset.selectedItem as NeuroLearningSets.Kind
        description.text = "<html><div style='width:174px'>${kind.description}</div></html>"
    }

    private fun editLayers(add: Boolean) {
        try {
            val values = NeuroTopologyConfig.parseHidden(hidden.text)
            hidden.text = NeuroTopologyConfig.format(if (add) NeuroTopologyConfig.addLayer(values, values.lastOrNull() ?: 6) else NeuroTopologyConfig.removeLayer(values))
            configError.text = " "
        } catch (exception: IllegalArgumentException) { showConfigError(exception.message) }
    }

    private fun applySettings() {
        try {
            epochLimit.commitEdit(); rate.commitEdit(); momentum.commitEdit(); target.commitEdit()
            val config = StudioConfig(NeuroTopologyConfig.format(NeuroTopologyConfig.parseHidden(hidden.text)),
                dataset.selectedItem as NeuroLearningSets.Kind, seed.text.trim().toLong(), (epochLimit.value as Number).toInt(),
                (target.value as Number).toDouble(), (rate.value as Number).toDouble(), (momentum.value as Number).toDouble(),
                backend.selectedItem as TrainingBackend)
            configError.text = " "
            post { it.apply(config, true) }
        } catch (exception: Exception) { showConfigError(exception.message ?: "Check the configuration values.") }
    }

    private fun showConfigError(message: String?) {
        configError.text = "<html><div style='width:174px'>${escape(message ?: "Invalid configuration")}</div></html>"
        configError.accessibleContext.accessibleDescription = message
    }

    private fun toggleTraining() {
        if (searchRunning || studyRunning) return
        val next = frame?.state != StudioState.RUNNING
        post(false) { it.setRunning(next) }
    }

    private fun bind(key: String, name: String, action: () -> Unit) {
        root.getInputMap(JComponent.WHEN_ANCESTOR_OF_FOCUSED_COMPONENT).put(KeyStroke.getKeyStroke(key), name)
        root.actionMap.put(name, object : AbstractAction() { override fun actionPerformed(event: ActionEvent) = action() })
    }

    private fun post(invalidateSearch: Boolean = true, action: (NeuroStudio) -> Unit) {
        if (closing) return
        if (invalidateSearch) {
            searchSession.invalidate(); searchRunning = false; architectureSearch.invalidateResults()
        }
        revision.incrementAndGet()
        commands.offer(action)
    }

    private fun workerLoop() {
        val studio = NeuroStudio(openSession = { model, selected -> trainingSessionFactory(model, selected) })
        try {
            while (!closing) {
                val action = if (studio.hasWork) commands.poll(24, TimeUnit.MILLISECONDS) else commands.take()
                try {
                    action?.invoke(studio)
                    while (true) (commands.poll() ?: break).invoke(studio)
                    if (studio.hasWork && !closing) studio.advance(speed) { closing || commands.isNotEmpty() }
                    if (!closing) published = studio.frame()
                } catch (exception: Exception) {
                    if (closing || exception is InterruptedException) break
                    studio.fail(exception.message ?: exception.javaClass.simpleName)
                    published = studio.frame()
                }
            }
        } catch (_: InterruptedException) { Thread.currentThread().interrupt() }
        finally { studio.close() }
    }

    private fun startArchitectureSearch(config: ArchitectureSearchConfig, evaluation: ArchitectureEvaluation, fraction: Double, splitSeed: Long) {
        val expectedConfig = frame?.config
        val expectedSamples = frame?.samples
        val token = searchSession.begin()
        searchRunning = true
        revision.incrementAndGet()
        commands.offer { studio ->
            if (!searchSession.isCurrent(token)) return@offer
            try {
                studio.setRunning(false)
                studio.close()
                require(studio.activeConfig == expectedConfig && studio.frame().samples == expectedSamples) {
                    "Active configuration changed before search started. Try again on the current run."
                }
                val data = studio.searchData(evaluation, fraction, splitSeed)
                published = studio.frame()
                val report = NeuroArchitectureSearch { model, selected -> trainingSessionFactory(model, selected) }.search(data, config, { progress ->
                    EventQueue.invokeLater {
                        if (!closing && searchSession.isCurrent(token)) architectureSearch.updateProgress(progress)
                    }
                }, { closing || token.cancelled.get() || !searchSession.isCurrent(token) })
                EventQueue.invokeLater {
                    if (!closing && searchSession.isCurrent(token)) {
                        searchRunning = false; architectureSearch.complete(report); refresh()
                    }
                }
            } catch (exception: Exception) {
                EventQueue.invokeLater {
                    if (!closing && searchSession.isCurrent(token)) {
                        searchRunning = false; architectureSearch.failed(exception.message ?: "Search failed"); refresh()
                    }
                }
            }
        }
        refresh()
    }

    private fun replayArchitecture(report: ArchitectureSearchResult, candidate: ArchitectureCandidate, trial: ArchitectureTrial) {
        revision.incrementAndGet()
        val token = searchSession.begin()
        searchRunning = true
        commands.offer { studio ->
            studio.setRunning(false)
            published = studio.frame()
            var replayed = false
            var replayFailure: String? = null
            try {
                replayed = studio.replayArchitecture(report, candidate, trial) { closing || token.cancelled.get() || !searchSession.isCurrent(token) }
            } catch (exception: Exception) {
                if (exception is InterruptedException) Thread.currentThread().interrupt()
                replayFailure = exception.message ?: "Replay failed"
            } finally {
                if (!closing) published = studio.frame()
                EventQueue.invokeLater {
                    if (!closing && searchSession.isCurrent(token)) {
                        searchRunning = false
                        if (replayed) {
                            architectureSearch.invalidateResults()
                            tabs.selectedIndex = View.OVERVIEW.ordinal
                        } else architectureSearch.replayFailed(replayFailure ?: "Replay cancelled")
                        refresh()
                    }
                }
            }
        }
        refresh()
    }

    private fun startStudy() {
        if (studyRunning || searchRunning) return
        studyRunning = true
        val expected = revision.incrementAndGet()
        commands.offer { studio ->
            studio.setRunning(false)
            published = studio.frame()
            try { studio.compareSeeds { closing || expected != revision.get() } }
            finally { studyRunning = false }
        }
        updateContextBar()
    }

    private fun updateParameterPage() {
        val current = frame ?: return
        selectedParameterLayer = selectedParameterLayer.coerceIn(0, current.diagnostics.layerCount() - 1)
        selectedParameterStart = 0
        updating = true
        parameterPage.value = 0
        updating = false
    }

    private fun updateContextBar() {
        contextBar.removeAll()
        when (View.entries[tabs.selectedIndex.coerceAtLeast(0)]) {
            View.NEURONS -> {
                contextBar.add(JLabel("All hidden layers and neurons · scroll to explore; maps render on demand").apply { foreground = MUTED })
            }
            View.PARAMETERS -> {
                contextBar.add(JLabel("Layer")); contextBar.add(parameterLayer)
                contextBar.add(JLabel("First parameter")); contextBar.add(parameterPage)
                contextBar.add(JLabel("24 curves / page").apply { foreground = MUTED })
            }
            View.DATA -> {
                contextBar.add(JLabel("Add point")); contextBar.add(customClass)
                contextBar.add(JButton("Undo point").apply { addActionListener { post { it.undoSample() } }; isEnabled = frame?.config?.dataset == NeuroLearningSets.Kind.CUSTOM })
                contextBar.add(JButton("Clear custom").apply { addActionListener { post { it.clearSamples() } } })
            }
            View.SEEDS -> {
                study.isEnabled = !studyRunning && !searchRunning && frame?.samples?.isNotEmpty() == true
                contextBar.add(study)
                if (studyRunning) { contextBar.add(cancelStudy); contextBar.add(JLabel("Comparing… main run paused").apply { foreground = ACCENT }) }
                else contextBar.add(JLabel("Same dataset and topology; independent initialization").apply { foreground = MUTED })
            }
            else -> contextBar.add(JLabel(when (View.entries[tabs.selectedIndex.coerceAtLeast(0)]) {
                View.UPDATE -> "Before / after one update batch. Pause and step to isolate an epoch."
                View.TIMELINE -> "Saved milestones and final state. These images are history, not replay controls."
                View.SEARCH -> "Evolve promising architectures through mutations. Equal seed budgets; reference grid optional."
                else -> "Grayscale = output 0–1. RMSE = training error, not validation error."
            }).apply { foreground = MUTED; font = uiFont(12) })
        }
        contextBar.revalidate(); contextBar.repaint()
    }

    private fun refresh() {
        if (closing) return
        val next = published ?: return
        val changed = frame !== next
        frame = next
        if (shownConfig != next.config) {
            shownConfig = next.config
            neuronGallery.request(next.diagnostics, emptyList()) {}
            updating = true
            parameterLayer.removeAllItems()
            for (layer in 0 until next.diagnostics.layerCount()) parameterLayer.addItem("${layer + 1}: ${next.diagnostics.layerInputCount(layer)} → ${next.diagnostics.layerOutputCount(layer)}")
            selectedParameterLayer = 0; selectedParameterStart = 0
            updating = false
            updateParameterPage(); updateContextBar()
        }
        architectureSearch.setSource(next.config, next.samples.size)
        status.text = if (searchRunning) "Architecture search…" else if (studyRunning) "Comparing seeds…" else next.state.label
        status.foreground = if (next.state == StudioState.FAILED) ERROR else ACCENT
        deviceStatus.text = next.deviceInfo?.let { "${it.backend} · ${it.name} · ${it.precision}" }
            ?: "${next.config.backend} · ${if (next.state == StudioState.FAILED) "unavailable" else "awaiting training"}"
        deviceStatus.toolTipText = next.deviceInfo?.let { "${it.identity} · kernel ${it.kernelVersion}" }
        activeTopology.text = "${next.config.dataset}   ·   ${next.config.description()}   ·   seed ${next.config.seed}" + if (next.replayNote.isEmpty()) "" else "   ·   ${next.replayNote}"
        metrics[0].text = "%,d".format(Locale.ROOT, next.diagnostics.epoch())
        metricDetails[0].text = "of %,d epochs".format(Locale.ROOT, next.config.maxEpochs)
        metrics[1].text = number(next.diagnostics.error(), 5)
        metricDetails[1].text = "target ${number(next.config.targetError, 3)}"
        metrics[2].text = "%,d".format(Locale.ROOT, next.diagnostics.parameterCount())
        metricDetails[2].text = "${next.diagnostics.hiddenLayerCount()} hidden ${if (next.diagnostics.hiddenLayerCount() == 1) "layer" else "layers"}"
        metrics[3].text = next.samples.size.toString(); metricDetails[3].text = "2 inputs · 1 output"
        train.text = if (next.state == StudioState.RUNNING) "Pause" else "Train"
        val finished = next.state in setOf(StudioState.EMPTY, StudioState.CONVERGED, StudioState.LIMIT_REACHED, StudioState.FAILED)
        train.isEnabled = !finished && !studyRunning && !searchRunning; step.isEnabled = !finished && !studyRunning && !searchRunning; stepTen.isEnabled = !finished && !studyRunning && !searchRunning
        if (next.message.isNotEmpty()) showConfigError(next.message)
        if (View.entries[tabs.selectedIndex.coerceAtLeast(0)] == View.SEEDS) updateContextBar()
        if (changed) charts.values.forEach { it.revalidate(); it.repaint() }
    }

    private fun exportImage() {
        val chooser = JFileChooser().apply { selectedFile = File("jneuro-${View.entries[tabs.selectedIndex].name.lowercase()}.png") }
        if (chooser.showSaveDialog(root) == JFileChooser.APPROVE_OPTION) {
            try { savePanel(chooser.selectedFile) } catch (exception: Exception) { showConfigError(exception.message) }
        }
    }

    private fun savePanel(file: File) {
        val image = BufferedImage(root.width, root.height, BufferedImage.TYPE_INT_RGB)
        val graphics = image.createGraphics()
        try { root.printAll(graphics) } finally { graphics.dispose() }
        check(ImageIO.write(image, "png", file)) { "PNG writer is not available." }
    }

    private inner class Chart(private val view: View) : JPanel(), Scrollable {
        private val interactive = Rectangle()
        init {
            background = BACKGROUND; isFocusable = true
            getAccessibleContext().accessibleName = "${view.title} visualization"
            getAccessibleContext().accessibleDescription = "Network output over x and y between zero and one. Arrow keys move the probe. Enter adds a custom point."
            val mouse = object : MouseAdapter() {
                override fun mouseMoved(event: MouseEvent) {
                    if (!interactive.contains(event.point)) return
                    hoverX = ((event.x - interactive.x).toDouble() / maxOf(1, interactive.width)).coerceIn(0.0, 1.0)
                    hoverY = (1.0 - (event.y - interactive.y).toDouble() / maxOf(1, interactive.height)).coerceIn(0.0, 1.0)
                    repaint()
                }
                override fun mousePressed(event: MouseEvent) {
                    requestFocusInWindow(); mouseMoved(event)
                    if (view == View.DATA && interactive.contains(event.point) && frame?.config?.dataset == NeuroLearningSets.Kind.CUSTOM && event.button in listOf(MouseEvent.BUTTON1, MouseEvent.BUTTON3)) {
                        val x = hoverX; val y = hoverY
                        val label = if (event.button == MouseEvent.BUTTON3 || customClass.selectedIndex == 1) 0.0 else 1.0
                        post { it.addSample(x, y, label) }
                    }
                }
            }
            addMouseMotionListener(mouse); addMouseListener(mouse)
            addKeyListener(object : KeyAdapter() {
                override fun keyPressed(event: KeyEvent) {
                    if (event.isControlDown || event.isAltDown) return
                    when (event.keyCode) {
                        KeyEvent.VK_LEFT -> hoverX = (hoverX - 0.025).coerceAtLeast(0.0)
                        KeyEvent.VK_RIGHT -> hoverX = (hoverX + 0.025).coerceAtMost(1.0)
                        KeyEvent.VK_UP -> hoverY = (hoverY + 0.025).coerceAtMost(1.0)
                        KeyEvent.VK_DOWN -> hoverY = (hoverY - 0.025).coerceAtLeast(0.0)
                        KeyEvent.VK_ENTER -> if (view == View.DATA && frame?.config?.dataset == NeuroLearningSets.Kind.CUSTOM) {
                            val x = hoverX; val y = hoverY; val label = if (customClass.selectedIndex == 0) 1.0 else 0.0
                            post { it.addSample(x, y, label) }
                        }
                    }
                    repaint()
                }
            })
        }
        override fun getPreferredSize(): Dimension {
            val w = if (parent != null) maxOf(640, parent.width) else 1000
            val h = when (view) {
                View.NEURONS -> frame?.let { NeuronGallery.Layout(it.diagnostics, w).height } ?: 450
                View.TIMELINE -> 80 + ((frame?.checkpoints?.size ?: 1) + 3) / 4 * 260
                View.SEEDS -> 760
                View.PARAMETERS -> 800
                View.DATA -> if (w < 860) 1080 else 660
                else -> if (w < 860 && view == View.OVERVIEW) 1200 else 660
            }
            return Dimension(w, h)
        }
        override fun getPreferredScrollableViewportSize() = Dimension(1000, 650)
        override fun getScrollableTracksViewportWidth() = true
        override fun getScrollableTracksViewportHeight() = false
        override fun getScrollableUnitIncrement(visible: Rectangle, orientation: Int, direction: Int) = 28
        override fun getScrollableBlockIncrement(visible: Rectangle, orientation: Int, direction: Int) = maxOf(28, visible.height - 28)

        override fun paintComponent(graphics: Graphics) {
            super.paintComponent(graphics)
            val current = frame ?: return
            val g = graphics.create() as Graphics2D
            try {
                g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
                g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON)
                interactive.setBounds(0, 0, 0, 0)
                when (view) {
                    View.OVERVIEW -> overview(g, current)
                    View.NEURONS -> neurons(g, current)
                    View.DATA -> data(g, current)
                    View.UPDATE -> update(g, current)
                    View.PARAMETERS -> parameters(g, current)
                    View.SEEDS -> seeds(g, current)
                    View.TIMELINE -> timeline(g, current)
                    View.SEARCH -> Unit
                }
                if (hasFocus()) { g.color = ACCENT; g.stroke = BasicStroke(2f); g.drawRect(2, 2, width - 5, height - 5) }
            } finally { g.dispose() }
        }

        private fun overview(g: Graphics2D, current: StudioFrame) {
            val split = width >= 860
            val cardWidth = if (split) (width - 36) / 2 else width - 24
            card(g, 12, 12, cardWidth, 416, "Decision surface", "Continuous model output · black 0, white 1")
            val size = minOf(304, cardWidth - 100)
            val x = 12 + (cardWidth - size) / 2
            plot(g, current.image, x, 90, size, true)
            interactive.setBounds(x, 90, size, size)
            samples(g, current.samples, x, 90, size)
            crosshair(g, x, 90, size)
            val chartX = if (split) cardWidth + 24 else 12
            val chartY = if (split) 12 else 444
            card(g, chartX, chartY, cardWidth, 416, "Learning curve", "RMSE and four reference inputs over completed epochs")
            historyPlot(g, current, chartX + 48, chartY + 84, cardWidth - 74, 246)
            val referenceY = if (split) 448 else 876
            val cellWidth = (width - 60) / 4
            val point = current.history.last()
            val labels = arrayOf("(0, 0)", "(0, 1)", "(1, 0)", "(1, 1)")
            for (i in 0..3) {
                card(g, 12 + i * (cellWidth + 12), referenceY, cellWidth, 112, labels[i], "Reference input")
                text(g, number(point.prediction(i), 4), 28 + i * (cellWidth + 12), referenceY + 89, 24, TEXT, true)
            }
            text(g, "Hover over the surface to probe any input. Inspect activations and signed weights in Neurons.", 14, referenceY + 148, 12, MUTED)
            if (current.state == StudioState.LIMIT_REACHED) text(g, "Epoch budget exhausted — not a proof of insufficient network capacity.", 14, referenceY + 174, 12, WARNING)
        }

        private fun historyPlot(g: Graphics2D, current: StudioFrame, x: Int, y: Int, w: Int, h: Int) {
            axes(g, x, y, w, h, 0.0, 1.0, current.diagnostics.epoch())
            val colors = arrayOf(ACCENT, CLASS_ONE, CLASS_ZERO, Color(207, 169, 255), Color(245, 221, 128))
            for (series in 0..4) {
                g.color = colors[series]
                g.stroke = if (series == 0) BasicStroke(2.6f) else BasicStroke(1.5f, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND, 1f, floatArrayOf(4f + series, 3f), 0f)
                var last: Point? = null
                for (entry in current.history) {
                    val value = if (series == 0) entry.error else entry.prediction(series - 1)
                    if (!value.isFinite()) { last = null; continue }
                    val p = Point(x + (w * entry.epoch.toDouble() / maxOf(1, current.diagnostics.epoch())).roundToInt(), y + (h * (1.0 - value.coerceIn(0.0, 1.0))).roundToInt())
                    last?.let { g.drawLine(it.x, it.y, p.x, p.y) }; last = p
                }
            }
            val legend = arrayOf("RMSE", "00", "01", "10", "11")
            for (i in legend.indices) {
                val lx = x + i * maxOf(45, w / 5)
                g.color = colors[i]; g.fillRect(lx, y + h + 46, 10, 3)
                text(g, legend[i], lx + 15, y + h + 52, 11, MUTED)
            }
        }

        private fun neurons(g: Graphics2D, current: StudioFrame) {
            card(g, 12, 12, width - 24, 332, "Forward pass · ${current.config.description()}", "Graph preview · solid positive / dashed negative · all neuron maps below")
            networkGraph(g, current, 42, 84, width - 84, 195)
            val probe = NeuroXorDiagnostics.probe(current.diagnostics, hoverX, hoverY)
            text(g, "Probe (${number(hoverX, 3)}, ${number(hoverY, 3)}) → ${number(probe.output(), 6)}   |   output z ${number(probe.outputPreActivation(), 4)}", 30, 306, 13, TEXT)
            if (current.diagnostics.hiddenLayerCount() == 0) {
                text(g, "No hidden layers · direct 2 → 1 baseline", 14, 375, 16, TEXT, true)
                text(g, "The output uses two input weights and one bias. There are no hidden activation maps.", 14, 399, 12, MUTED)
                return
            }
            val layout = NeuronGallery.Layout(current.diagnostics, width)
            val clip = g.clipBounds ?: visibleRect
            val cells = layout.visible(clip)
            neuronGallery.request(current.diagnostics, cells) { EventQueue.invokeLater { if (!closing) repaint() } }
            for (section in layout.sections) {
                if (section.top + 58 >= clip.y && section.top.toLong() < clip.y.toLong() + clip.height) {
                    text(g, "H${section.layer + 1} activations · all ${section.neurons} neurons", 14, section.top + 18, 16, TEXT, true)
                    text(g, if (section.layer == 0) "Red line = z = 0 in input space" else "Deeper activations are nonlinear functions of x and y.", 14, section.top + 42, 12, MUTED)
                }
            }
            for (cell in cells) {
                val (x, y, w) = listOf(cell.bounds.x, cell.bounds.y, cell.bounds.width)
                val size = minOf(164, w - 36)
                val px = x + (w - size) / 2
                card(g, x, y, w, 242, "H${cell.layer + 1}.${cell.neuron}", "activation 0–1")
                val image = neuronGallery.image(cell.layer, cell.neuron)
                if (image == null) text(g, "Rendering…", px, y + 94, 12, MUTED)
                else plot(g, image, px, y + 58, size, false)
                if (cell.layer == 0) NeuroXorDiagnostics.boundary(current.diagnostics, cell.neuron)?.let { line ->
                    g.color = WARNING; g.stroke = BasicStroke(1.6f)
                    g.drawLine(px + (line.x1 * size).roundToInt(), y + 58 + ((1 - line.y1) * size).roundToInt(), px + (line.x2 * size).roundToInt(), y + 58 + ((1 - line.y2) * size).roundToInt())
                }
            }
        }

        private fun networkGraph(g: Graphics2D, current: StudioFrame, x: Int, y: Int, w: Int, h: Int) {
            val diagnostics = current.diagnostics
            val shape = diagnostics.topology()
            val probe = NeuroXorDiagnostics.probe(diagnostics, hoverX, hoverY)
            val activations = arrayOf(doubleArrayOf(hoverX, hoverY)) + probe.hiddenLayers() + arrayOf(doubleArrayOf(probe.output()))
            val maximum = maxOf(1e-9, NeuroXorDiagnostics.maxAbsWeight(diagnostics))
            fun px(layer: Int) = x + layer * w / (shape.size - 1)
            fun py(neuron: Int, count: Int) = y + (neuron + 1) * h / (count + 1)
            for (layer in 0 until shape.lastIndex) {
                val ins = minOf(shape[layer], 8); val outs = minOf(shape[layer + 1], 8)
                for (output in 0 until outs) for (input in 0 until ins) {
                    val weight = diagnostics.weight(layer, output, input)
                    val strokeWidth = (0.5 + 2.5 * abs(weight) / maximum).toFloat()
                    g.stroke = if (weight >= 0) BasicStroke(strokeWidth) else BasicStroke(strokeWidth, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND, 1f, floatArrayOf(4f, 3f), 0f)
                    g.color = if (weight >= 0) Color(90, 157, 199, 180) else Color(188, 134, 98, 160)
                    g.drawLine(px(layer), py(input, ins), px(layer + 1), py(output, outs))
                }
            }
            for (layer in shape.indices) {
                val count = minOf(shape[layer], 8)
                for (neuron in 0 until count) {
                    val value = activations[layer][neuron].coerceIn(0.0, 1.0)
                    val gray = (42 + value * 208).roundToInt()
                    g.color = Color(gray, gray, gray); g.fillOval(px(layer) - 8, py(neuron, count) - 8, 16, 16)
                    g.color = ACCENT; g.stroke = BasicStroke(1f); g.drawOval(px(layer) - 8, py(neuron, count) - 8, 16, 16)
                }
                val label = when (layer) { 0 -> "Input"; shape.lastIndex -> "Output"; else -> "H$layer (${shape[layer]})" }
                text(g, label, px(layer) - 22, y - 8, 11, MUTED)
                if (shape[layer] > count) text(g, "+${shape[layer] - count} hidden", px(layer) - 23, y + h + 15, 10, MUTED)
            }
        }

        private fun data(g: Graphics2D, current: StudioFrame) {
            val narrow = width < 860
            card(g, 12, 12, width - 24, if (narrow) 1000 else 586, current.config.dataset.toString(), "Training samples on the learned surface · not validation data")
            val size = if (narrow) 380 else minOf(420, width / 2 - 40)
            val plotX = if (narrow) (width - size) / 2 else 62
            plot(g, current.image, plotX, 104, size, true)
            interactive.setBounds(plotX, 104, size, size)
            samples(g, current.samples, plotX, 104, size); crosshair(g, plotX, 104, size)
            val x = if (narrow) 36 else size + 104
            val base = if (narrow) size + 176 else 110
            text(g, "${current.samples.size} training points", x, base, 18, TEXT, true)
            text(g, "Circle = class 1  ·  square = class 0", x, base + 32, 12, MUTED)
            text(g, "No threshold is applied to the background.", x, base + 58, 12, MUTED)
            val probe = NeuroXorDiagnostics.probe(current.diagnostics, hoverX, hoverY)
            text(g, "x = ${number(hoverX, 3)}", x, base + 120, 14, TEXT)
            text(g, "y = ${number(hoverY, 3)}", x, base + 149, 14, TEXT)
            text(g, "output = ${number(probe.output(), 5)}", x, base + 184, 20, ACCENT, true)
            if (current.config.dataset == NeuroLearningSets.Kind.CUSTOM) {
                text(g, "EDITING CUSTOM DATA", x, base + 246, 11, MUTED, true)
                text(g, "Click: selected class. Right-click: class 0.", x, base + 272, 12, TEXT)
                text(g, "Arrow keys move the probe; Enter adds a point.", x, base + 296, 12, TEXT)
                text(g, "An edit resets and pauses the model.", x, base + 328, 12, WARNING)
            } else {
                text(g, "Select Custom and Apply to edit your own set.", x, base + 270, 12, MUTED)
            }
        }

        private fun update(g: Graphics2D, current: StudioFrame) {
            val cell = (width - 48) / 3
            val size = minOf(292, cell - 50)
            val names = arrayOf("Before · epoch ${current.beforeEpoch}", "After · epoch ${current.diagnostics.epoch()}", "Signed change Δf")
            val images = arrayOf(current.beforeImage, current.image, current.differenceImage)
            for (i in 0..2) {
                val x = 12 + i * (cell + 12)
                card(g, x, 12, cell, size + 135, names[i], if (i == 2) "after − before · same input coordinates" else "output 0–1")
                plot(g, images[i], x + (cell - size) / 2, 91, size, true)
            }
            val y = size + 194
            text(g, "Red increases the output; blue decreases it. Neutral gray means no change.", 14, y, 14, TEXT)
            text(g, "Difference scale is fixed: full color at ±0.25. This is output change, not a gradient map.", 14, y + 28, 12, MUTED)
            text(g, "Pause → 1 epoch isolates one complete shuffled pass over the current dataset.", 14, y + 64, 13, ACCENT)
        }

        private fun parameters(g: Graphics2D, current: StudioFrame) {
            val d = current.diagnostics
            val layer = selectedParameterLayer.coerceIn(0, d.layerCount() - 1)
            val weights = d.layerInputCount(layer) * d.layerOutputCount(layer)
            val biases = d.layerOutputCount(layer)
            val offset = d.parameterOffset(layer)
            val cell = (width - 36) / 2
            parameterChart(g, current, 12, 12, cell, 352, offset, weights, "Layer ${layer + 1} weights")
            parameterChart(g, current, cell + 24, 12, cell, 352, offset + weights, biases, "Layer ${layer + 1} biases")
            card(g, 12, 382, width - 24, 330, "Layer norms", "Solid = weight norm · dashed = bias norm · layers follow paired colors")
            val maximum = current.history.maxOf { entry -> (0 until entry.normCount).maxOf { entry.norm(it) } }.coerceAtLeast(1e-9)
            val x = 66; val y = 458; val w = width - 105; val h = 184
            axes(g, x, y, w, h, 0.0, maximum, d.epoch())
            for (series in 0 until d.layerCount() * 2) {
                g.color = PALETTE[(series / 2) % PALETTE.size]
                g.stroke = if (series % 2 == 0) BasicStroke(2f) else BasicStroke(1.4f, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND, 1f, floatArrayOf(5f, 4f), 0f)
                drawSeries(g, current.history, x, y, w, h, d.epoch(), 0.0, maximum) { it.norm(series) }
            }
        }

        private fun parameterChart(g: Graphics2D, current: StudioFrame, x: Int, y: Int, w: Int, h: Int, offset: Int, total: Int, title: String) {
            val start = selectedParameterStart.coerceAtMost(total - 1)
            val count = minOf(24, total - start)
            card(g, x, y, w, h, title, "Indices $start–${start + count - 1} of $total · history ${current.history.size} samples")
            val maximum = current.history.maxOf { entry -> (0 until count).maxOf { abs(entry.parameter(offset + start + it)) } }.coerceAtLeast(0.01)
            val px = x + 54; val py = y + 80; val pw = w - 80; val ph = h - 137
            axes(g, px, py, pw, ph, -maximum, maximum, current.diagnostics.epoch())
            for (series in 0 until count) {
                g.color = PALETTE[series % PALETTE.size]; g.stroke = BasicStroke(1.4f)
                drawSeries(g, current.history, px, py, pw, ph, current.diagnostics.epoch(), -maximum, maximum) { it.parameter(offset + start + series) }
            }
        }

        private fun seeds(g: Graphics2D, current: StudioFrame) {
            if (current.seeds.isEmpty()) {
                card(g, 12, 12, width - 24, 270, "Same architecture. Different starting points.", "A controlled comparison rather than a promise of convergence")
                text(g, "Run Compare 4 seeds to evaluate 1, 42, 123 and 999.", 34, 112, 17, TEXT, true)
                text(g, "Each run uses ${current.config.description()}, the current dataset and an epoch budget of ${current.config.maxEpochs}.", 34, 151, 13, MUTED)
                text(g, "The main model is paused and never replaced. Cancel or apply new settings to stop the study.", 34, 184, 13, MUTED)
                return
            }
            val cell = (width - 36) / 2
            for ((index, result) in current.seeds.withIndex()) {
                val x = 12 + (index % 2) * (cell + 12); val y = 12 + index / 2 * 362
                card(g, x, y, cell, 344, "Seed ${result.seed}", "${result.epochs} epochs · RMSE ${number(result.error, 5)}")
                val size = minOf(218, cell - 64)
                if (cell < 430) {
                    plot(g, result.image, x + (cell - size) / 2, y + 80, size, false)
                    text(g, if (result.converged) "Target reached" else "Budget exhausted", x + 22, y + 325, 13, if (result.converged) ACCENT else WARNING, true)
                } else {
                    plot(g, result.image, x + 42, y + 80, size, false)
                    val tx = x + 278
                    text(g, if (result.converged) "Target" else "Budget", tx, y + 122, 16, if (result.converged) ACCENT else WARNING, true)
                    text(g, if (result.converged) "reached" else "exhausted", tx, y + 146, 14, TEXT)
                    text(g, "${result.epochs}", tx, y + 203, 22, TEXT, true)
                    text(g, "epochs", tx, y + 228, 12, MUTED)
                }
            }
        }

        private fun timeline(g: Graphics2D, current: StudioFrame) {
            val cols = 4; val cell = (width - 60) / cols
            val size = minOf(170, cell - 28)
            for ((index, checkpoint) in current.checkpoints.withIndex()) {
                val x = 12 + index % cols * (cell + 12); val y = 12 + index / cols * 252
                card(g, x, y, cell, 236, "Epoch ${checkpoint.epoch}", "RMSE ${number(checkpoint.error, 4)}")
                plot(g, checkpoint.image, x + (cell - size) / 2, y + 65, size, false)
            }
        }

        private fun plot(g: Graphics2D, image: BufferedImage, x: Int, y: Int, size: Int, labels: Boolean) {
            g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR)
            g.drawImage(image, x, y, size, size, null)
            g.color = BORDER; g.stroke = BasicStroke(1f); g.drawRect(x, y, size, size)
            if (labels) {
                text(g, "0", x - 3, y + size + 18, 11, MUTED); text(g, "1", x + size - 4, y + size + 18, 11, MUTED)
                text(g, "x", x + size / 2, y + size + 20, 12, MUTED)
                text(g, "1", x - 19, y + 4, 11, MUTED); text(g, "0", x - 19, y + size, 11, MUTED)
                text(g, "y", x - 23, y + size / 2, 12, MUTED)
            }
        }

        private fun samples(g: Graphics2D, samples: List<NeuroLearningSets.Sample>, x: Int, y: Int, size: Int) {
            for (sample in samples) {
                val px = x + (sample.x * (size - 1)).roundToInt(); val py = y + ((1 - sample.y) * (size - 1)).roundToInt()
                g.color = if (sample.target >= 0.5) CLASS_ONE else CLASS_ZERO
                if (sample.target >= 0.5) g.fillOval(px - 4, py - 4, 8, 8) else g.fillRect(px - 4, py - 4, 8, 8)
                g.color = BACKGROUND; g.stroke = BasicStroke(1f)
                if (sample.target >= 0.5) g.drawOval(px - 4, py - 4, 8, 8) else g.drawRect(px - 4, py - 4, 8, 8)
            }
        }

        private fun crosshair(g: Graphics2D, x: Int, y: Int, size: Int) {
            val px = x + (hoverX * size).roundToInt(); val py = y + ((1 - hoverY) * size).roundToInt()
            g.color = Color.BLACK; g.stroke = BasicStroke(3f)
            g.drawLine(px - 7, py, px + 7, py); g.drawLine(px, py - 7, px, py + 7)
            g.color = ACCENT; g.stroke = BasicStroke(1.4f)
            g.drawLine(px - 7, py, px + 7, py); g.drawLine(px, py - 7, px, py + 7)
        }
    }

    private fun showWindow() {
        val window = JFrame("JNeuro · Neural Learning Studio").apply {
            defaultCloseOperation = WindowConstants.DISPOSE_ON_CLOSE
            contentPane = root
            minimumSize = Dimension(900, 640)
            val available = GraphicsEnvironment.getLocalGraphicsEnvironment().maximumWindowBounds
            size = Dimension(minOf(1520, available.width - 32), minOf(1000, available.height - 32))
            setLocationRelativeTo(null)
            addWindowListener(object : WindowAdapter() { override fun windowClosed(event: WindowEvent) = close() })
        }
        window.isVisible = true
    }

    override fun close() {
        closing = true; searchSession.invalidate(); revision.incrementAndGet(); timer.stop(); neuronGallery.close(); thread?.interrupt()
    }

    companion object {
        private val BACKGROUND = Color(13, 20, 29)
        private val SURFACE = Color(23, 35, 47)
        private val BORDER = Color(55, 74, 91)
        private val TEXT = Color(234, 240, 247)
        private val MUTED = Color(166, 185, 203)
        private val ACCENT = Color(126, 224, 192)
        private val CLASS_ONE = Color(115, 200, 255)
        private val CLASS_ZERO = Color(255, 187, 136)
        private val WARNING = Color(255, 179, 150)
        private val ERROR = Color(255, 165, 165)
        private val PALETTE = arrayOf(ACCENT, CLASS_ONE, CLASS_ZERO, Color(207, 169, 255), Color(245, 221, 128), Color(247, 149, 199))
        private fun uiFont(size: Int, style: Int = Font.PLAIN) = Font(Font.SANS_SERIF, style, size)
        private fun number(value: Double, precision: Int): String = if (value.isFinite()) String.format(Locale.ROOT, "%.${precision}f", value) else "—"
        private fun escape(value: String): String = value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        private fun text(g: Graphics2D, value: String, x: Int, y: Int, size: Int, color: Color, bold: Boolean = false) {
            g.color = color; g.font = uiFont(size, if (bold) Font.BOLD else Font.PLAIN); g.drawString(value, x, y)
        }
        private fun card(g: Graphics2D, x: Int, y: Int, w: Int, h: Int, title: String, subtitle: String) {
            g.color = SURFACE; g.fillRoundRect(x, y, w, h, 16, 16)
            g.color = BORDER; g.stroke = BasicStroke(1f); g.drawRoundRect(x, y, w, h, 16, 16)
            val copy = g.create(x + 16, y + 14, maxOf(1, w - 32), 43) as Graphics2D
            try { text(copy, title, 0, 15, 15, TEXT, true); text(copy, subtitle, 0, 36, 11, MUTED) } finally { copy.dispose() }
        }
        private fun axes(g: Graphics2D, x: Int, y: Int, w: Int, h: Int, minimum: Double, maximum: Double, epoch: Int) {
            g.stroke = BasicStroke(1f)
            for (i in 0..4) {
                val py = y + i * h / 4
                g.color = BORDER; g.drawLine(x, py, x + w, py)
                text(g, number(maximum - (maximum - minimum) * i / 4, 2), x - 42, py + 4, 10, MUTED)
            }
            text(g, "0", x, y + h + 19, 11, MUTED)
            text(g, "epoch", x + w / 2 - 18, y + h + 19, 11, MUTED)
            text(g, epoch.toString(), x + w - 34, y + h + 19, 11, MUTED)
        }
        private fun drawSeries(g: Graphics2D, entries: List<StudioHistory>, x: Int, y: Int, w: Int, h: Int, epochs: Int,
                               minimum: Double, maximum: Double, value: (StudioHistory) -> Double) {
            var previous: Point? = null
            for (entry in entries) {
                val p = Point(x + (w * entry.epoch.toDouble() / maxOf(1, epochs)).roundToInt(),
                    y + (h * (1 - (value(entry) - minimum) / (maximum - minimum))).roundToInt())
                previous?.let { g.drawLine(it.x, it.y, p.x, p.y) }; previous = p
            }
        }
        private fun installTheme() {
            FlatDarkLaf.setup()
            UIManager.put("defaultFont", uiFont(13))
            UIManager.put("Panel.background", SURFACE)
            UIManager.put("Label.foreground", TEXT)
            UIManager.put("Component.focusColor", ACCENT)
            UIManager.put("Component.arc", 10)
            UIManager.put("Button.arc", 10)
            UIManager.put("TextComponent.arc", 8)
            UIManager.put("Component.focusWidth", 2)
            UIManager.put("TabbedPane.selectedBackground", SURFACE)
            UIManager.put("TabbedPane.underlineColor", ACCENT)
        }
        private fun layoutAll(component: Container) {
            component.doLayout()
            for (child in component.components) if (child is Container) layoutAll(child)
        }
        @JvmStatic fun main(args: Array<String>) {
            val export = args.firstOrNull { it.startsWith("--screenshots=") }?.substringAfter('=')
            if (export != null) {
                val directory = File(export).apply { mkdirs() }
                val studio = NeuroStudio()
                studio.setRunning(true)
                while (studio.hasWork) studio.advance(100)
                studio.compareSeeds()
                val snapshot = studio.frame()
                val searchPreview = NeuroArchitectureSearch().search(studio.searchData(ArchitectureEvaluation.TRAINING_FIT),
                    ArchitectureSearchConfig(maxLayers = 2, maxWidth = 3, maxEpochs = 2500, maxParameters = 32))
                EventQueue.invokeAndWait {
                    installTheme()
                    NeuroXorCanvas(snapshot, false).use { ui ->
                        ui.architectureSearch.complete(searchPreview)
                        ui.root.setSize(1520, 1060)
                        for (view in View.entries) {
                            ui.tabs.selectedIndex = view.ordinal
                            layoutAll(ui.root)
                            ui.savePanel(File(directory, "${view.name.lowercase()}.png"))
                        }
                    }
                }
                println("Rendered implemented Swing UI to ${directory.absolutePath}")
            } else {
                check(!GraphicsEnvironment.isHeadless()) { "A graphical desktop is required. Use --screenshots=<directory> for headless rendering." }
                EventQueue.invokeLater { installTheme(); NeuroXorCanvas().showWindow() }
            }
        }
    }
}
