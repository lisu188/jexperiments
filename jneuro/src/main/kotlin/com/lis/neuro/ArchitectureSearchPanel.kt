package com.lis.neuro

import java.awt.*
import java.awt.event.MouseAdapter
import java.awt.event.MouseEvent
import java.awt.image.BufferedImage
import java.util.Locale
import javax.swing.*
import javax.swing.border.EmptyBorder
import javax.swing.table.AbstractTableModel
import javax.swing.table.DefaultTableCellRenderer
import kotlin.math.roundToInt

internal class ArchitectureSearchPanel(
    private val startSearch: (ArchitectureSearchConfig, ArchitectureEvaluation, Double, Long) -> Unit,
    private val cancelSearch: () -> Unit,
    private val applyArchitecture: (ArchitectureSearchResult, ArchitectureCandidate) -> Unit,
    private val replayRun: (ArchitectureSearchResult, ArchitectureCandidate, ArchitectureTrial) -> Unit
) : JPanel(BorderLayout(12, 12)) {
    private val minLayers = JSpinner(SpinnerNumberModel(1, 1, 8, 1))
    private val maxLayers = JSpinner(SpinnerNumberModel(3, 1, 8, 1))
    private val minWidth = JSpinner(SpinnerNumberModel(1, 1, 128, 1))
    private val maxWidth = JSpinner(SpinnerNumberModel(8, 1, 128, 1))
    private val parameters = JSpinner(SpinnerNumberModel(256, 1, 1_000_000, 16))
    private val target = JSpinner(SpinnerNumberModel(0.05, 0.000001, 0.999, 0.01))
    private val epochs = JSpinner(SpinnerNumberModel(10_000, 1, 1_000_000, 1000))
    private val checkEvery = JSpinner(SpinnerNumberModel(25, 1, 1_000_000, 1))
    private val seeds = JTextField("1,42,123,999,2026", 17)
    private val successes = JSpinner(SpinnerNumberModel(4, 1, 20, 1))
    private val tolerance = JSpinner(SpinnerNumberModel(0.005, 0.0, 1.0, 0.001))
    private val threads = JSpinner(SpinnerNumberModel(minOf(4, maxOf(1, Runtime.getRuntime().availableProcessors() - 1)), 1, 32, 1))
    private val trials = JSpinner(SpinnerNumberModel(10_000, 1, 20_000, 100))
    private val seconds = JSpinner(SpinnerNumberModel(0, 0, 86_400, 60))
    private val fraction = JSpinner(SpinnerNumberModel(0.2, 0.1, 0.5, 0.05))
    private val splitSeed = JTextField("42", 10)
    private val policy = JComboBox(ArchitecturePolicy.entries.toTypedArray())
    private val evaluation = JComboBox(ArchitectureEvaluation.entries.toTypedArray())
    private val start = JButton("Start search")
    private val cancel = JButton("Cancel search")
    private val apply = JButton("Apply architecture")
    private val replay = JButton("Replay selected run")
    private val inspect = JButton("Inspect result")
    private val summary = JLabel("Search the active dataset. Unapplied sidebar edits are not used.")
    private val sourceLabel = JLabel()
    private val progressBar = JProgressBar()
    private val selectedSeed = JComboBox<String>()
    private val details = JLabel("Select a point or row to inspect a scored checkpoint.")
    private var results: List<ArchitectureCandidate> = emptyList()
    private val tableModel = CandidateTable()
    private val table = JTable(tableModel)
    private val plot = ParetoPlot()
    private val inspector = Inspector()
    private var source = StudioConfig()
    private var sourceSize = 0
    private var config: ArchitectureSearchConfig? = null
    private var scoreMode = ArchitectureEvaluation.TRAINING_FIT
    private var result: ArchitectureSearchResult? = null
    private var selected: NetworkArchitecture? = null
    private var chosenTrial: ArchitectureTrial? = null
    private var surface: BufferedImage? = null
    private var running = false
    private var changing = false

    init {
        background = BACKGROUND
        border = EmptyBorder(14, 10, 10, 10)
        val controls = JPanel().apply { layout = BoxLayout(this, BoxLayout.Y_AXIS); isOpaque = false }
        controls.add(JLabel("Find the smallest network that learns").apply { font = font.deriveFont(Font.BOLD, 20f); foreground = TEXT })
        sourceLabel.foreground = MUTED; controls.add(sourceLabel)
        controls.add(Box.createVerticalStrut(10))
        val primary = JPanel(GridLayout(2, 6, 10, 5)).apply { isOpaque = false }
        for ((name, component) in listOf("Min layers" to minLayers, "Max layers" to maxLayers, "Min width" to minWidth,
            "Max width" to maxWidth, "Max parameters" to parameters, "Target RMSE" to target)) {
            primary.add(field(name, component))
        }
        primary.layout = GridLayout(1, 6, 10, 5)
        controls.add(primary)
        val choices = JPanel(FlowLayout(FlowLayout.LEFT, 8, 8)).apply {
            isOpaque = false
            add(JLabel("Recommend")); add(policy); add(JLabel("Score")); add(evaluation)
            start.background = ACCENT; start.foreground = BACKGROUND; add(start); add(cancel)
        }
        controls.add(choices)
        val advanced = JPanel(GridLayout(3, 4, 12, 8)).apply { isOpaque = false }
        for ((name, component) in listOf("Seeds" to seeds, "Required successes" to successes, "Epochs per seed" to epochs,
            "Check every (epochs)" to checkEvery, "Near-best tolerance" to tolerance, "Parallel trials" to threads,
            "Trial budget" to trials, "Seconds (0 = unlimited)" to seconds, "Validation fraction" to fraction, "Split seed" to splitSeed)) {
            advanced.add(field(name, component))
        }
        advanced.isVisible = false
        controls.add(JCheckBox("Advanced search settings").apply {
            isOpaque = false
            addActionListener { advanced.isVisible = isSelected; revalidate() }
        })
        controls.add(advanced)
        summary.foreground = MUTED
        controls.add(summary)
        progressBar.isStringPainted = true; progressBar.string = "Not started"; controls.add(progressBar)
        for (child in controls.components) if (child is JComponent) child.alignmentX = 0f
        add(controls, BorderLayout.NORTH)

        table.autoCreateRowSorter = true; table.rowHeight = 27; table.intercellSpacing = Dimension(10, 2); table.fillsViewportHeight = true
        table.setSelectionMode(ListSelectionModel.SINGLE_SELECTION)
        table.setDefaultRenderer(Double::class.javaObjectType, object : DefaultTableCellRenderer() {
            override fun setValue(value: Any?) { text = if (value is Double) number(value) else "—"; horizontalAlignment = RIGHT }
        })
        table.getAccessibleContext().accessibleName = "Architecture search results"
        val rows = JScrollPane(table).apply { setColumnHeaderView(table.tableHeader) }
        val upper = SearchSplit(plot, rows, 0.60)
        val detailPanel = JPanel(BorderLayout(0, 8)).apply {
            background = SURFACE; border = EmptyBorder(10, 12, 10, 12)
            val actions = JPanel(FlowLayout(FlowLayout.LEFT, 8, 0)).apply {
                isOpaque = false; add(selectedSeed); add(inspect); add(apply); add(replay)
            }
            add(actions, BorderLayout.NORTH); add(inspector, BorderLayout.CENTER)
            details.foreground = MUTED; add(details, BorderLayout.SOUTH)
        }
        add(SearchSplit(upper, detailPanel, 0.58), BorderLayout.CENTER)
        preferredSize = Dimension(1080, 920)
        minimumSize = Dimension(640, 600)
        start.isEnabled = false; cancel.isEnabled = false; apply.isEnabled = false; replay.isEnabled = false; inspect.isEnabled = false
        start.addActionListener { submit() }
        cancel.addActionListener { cancelSearch(); cancel.isEnabled = false; summary.text = "Stopping between epochs; completed results will be retained." }
        policy.addActionListener { if (!changing) { updateSummary(); plot.repaint() } }
        table.selectionModel.addListSelectionListener { event ->
            if (!event.valueIsAdjusting && !changing && table.selectedRow >= 0) {
                selected = results[table.convertRowIndexToModel(table.selectedRow)].architecture
                updateInspection()
            }
        }
        selectedSeed.addActionListener { if (!changing) inspectSelectedSeed() }
        inspect.addActionListener { inspectSelectedSeed() }
        apply.addActionListener { selectedCandidate()?.let { candidate -> result?.let { applyArchitecture(it, candidate) } } }
        replay.addActionListener { selectedCandidate()?.let { candidate -> result?.let { report -> chosenTrial?.let { replayRun(report, candidate, it) } } } }
        progressBar.getAccessibleContext().accessibleName = "Search progress"
    }

    fun setSource(config: StudioConfig, samples: Int) {
        if (source != config) {
            target.value = config.targetError; epochs.value = config.maxEpochs
            checkEvery.value = minOf(25, config.maxEpochs)
            evaluation.selectedItem = if (config.dataset in BOOLEAN_SETS) ArchitectureEvaluation.TRAINING_FIT else ArchitectureEvaluation.VALIDATION
        }
        source = config; sourceSize = samples
        sourceLabel.text = "Active: ${source.dataset} · $samples samples · learning rate ${source.learningRate} · momentum ${source.momentum}"
        start.isEnabled = !running && samples > 0
    }

    internal fun readConfig(): ArchitectureSearchConfig {
        val spinners = listOf(minLayers, maxLayers, minWidth, maxWidth, parameters, target, epochs, checkEvery, successes, tolerance, threads, trials, seconds, fraction)
        spinners.forEach { it.commitEdit() }
        return ArchitectureSearchConfig(integer(minLayers), integer(maxLayers), integer(minWidth), integer(maxWidth), integer(parameters),
            seeds.text.split(',').map { token -> token.trim().toLongOrNull() ?: throw IllegalArgumentException("Seeds must be comma-separated integers.") },
            integer(epochs), integer(checkEvery), decimal(target), integer(successes), decimal(tolerance), policy.selectedItem as ArchitecturePolicy,
            Neuro.HyperParameters(source.learningRate, source.momentum, 1.0, source.seed), integer(threads), integer(trials), integer(seconds).toLong())
    }

    private fun submit() {
        try {
            val next = readConfig()
            val space = next.architectures()
            val mode = evaluation.selectedItem as ArchitectureEvaluation
            require(mode != ArchitectureEvaluation.VALIDATION || source.dataset !in BOOLEAN_SETS) { "Truth tables use Training RMSE, not validation on four examples." }
            val split = splitSeed.text.trim().toLongOrNull() ?: throw IllegalArgumentException("Split seed must be an integer.")
            started(next, mode)
            summary.text = "${space.size} architectures; up to ${minOf(space.size, next.maxTrials / next.seeds.size) * next.seeds.size} seed trials. Main run paused."
            startSearch(next, mode, decimal(fraction), split)
        } catch (exception: Exception) { failed(exception.message ?: "Check search settings.") }
    }

    fun started(next: ArchitectureSearchConfig, mode: ArchitectureEvaluation) {
        minLayers.value = next.minLayers; maxLayers.value = next.maxLayers
        minWidth.value = next.minWidth; maxWidth.value = next.maxWidth
        parameters.value = next.maxParameters; target.value = next.targetRmse
        epochs.value = next.maxEpochs; checkEvery.value = next.checkEvery
        seeds.text = next.seeds.joinToString(","); successes.value = next.requiredSuccesses
        tolerance.value = next.nearBestTolerance; threads.value = next.parallelism
        trials.value = next.maxTrials; seconds.value = next.timeLimitSeconds.toInt()
        changing = true; policy.selectedItem = next.policy; evaluation.selectedItem = mode; changing = false
        running = true; config = next; scoreMode = mode; result = null; selected = null
        results = emptyList(); chosenTrial = null; surface = null
        tableModel.fireTableDataChanged(); selectedSeed.removeAllItems()
        start.isEnabled = false; cancel.isEnabled = true; apply.isEnabled = false; replay.isEnabled = false; inspect.isEnabled = false
        progressBar.value = 0; progressBar.string = "Starting…"
        details.text = "Inspecting a result never changes the active network."
        plot.repaint(); inspector.repaint()
    }

    fun updateProgress(progress: ArchitectureSearchProgress) {
        if (!running) return
        showCandidates(progress.candidates)
        progressBar.maximum = maxOf(1, progress.plannedTrials); progressBar.value = progress.finishedTrials
        progressBar.string = "${progress.finishedTrials}/${progress.plannedTrials} trials · ${progress.fullyEvaluated}/${progress.generated} architectures evaluated"
        val trial = progress.running.firstOrNull()
        summary.text = if (trial == null) "Collecting results…" else "Training ${trial.architecture} · seed ${trial.seed} · epoch ${trial.epoch} · best ${scoreMode.label} ${number(trial.bestRmse)}"
    }

    fun complete(report: ArchitectureSearchResult) {
        if (config !== report.config) started(report.config, report.data.evaluation)
        running = false; result = report; config = report.config; scoreMode = report.data.evaluation
        start.isEnabled = sourceSize > 0; cancel.isEnabled = false
        showCandidates(report.candidates)
        progressBar.maximum = maxOf(1, report.generated); progressBar.value = report.evaluated
        progressBar.string = "${report.termination} · ${report.evaluated}/${report.generated} evaluated · ${report.partial} partial · ${report.untested} untested"
        if (selected == null) selected = currentSelection()?.recommended?.architecture ?: currentSelection()?.bestError?.architecture
        selectArchitecture(selected)
        updateSummary()
    }

    fun failed(message: String) {
        running = false; start.isEnabled = sourceSize > 0; cancel.isEnabled = false
        summary.text = "Search error: $message"; progressBar.string = "Search failed"
    }

    fun invalidateResults() {
        running = false; result = null; config = null; results = emptyList(); selected = null
        chosenTrial = null; surface = null; tableModel.fireTableDataChanged()
        cancel.isEnabled = false; apply.isEnabled = false; replay.isEnabled = false; inspect.isEnabled = false; start.isEnabled = sourceSize > 0
        summary.text = "Active configuration changed. Start a new search for this dataset."
        progressBar.value = 0; progressBar.string = "Results invalidated"
        selectedSeed.removeAllItems(); details.text = "Select a point or row to inspect a scored checkpoint."
        plot.repaint(); inspector.repaint()
    }

    internal fun selectArchitecture(architecture: NetworkArchitecture?) {
        selected = architecture
        val index = results.indexOfFirst { it.architecture == architecture }
        changing = true
        if (index >= 0) table.setRowSelectionInterval(table.convertRowIndexToView(index), table.convertRowIndexToView(index)) else table.clearSelection()
        changing = false
        updateInspection()
    }

    private fun showCandidates(candidates: List<ArchitectureCandidate>) {
        results = candidates
        changing = true; tableModel.fireTableDataChanged(); changing = false
        selectArchitecture(selected)
        plot.repaint()
    }

    private fun selectedCandidate(): ArchitectureCandidate? = results.firstOrNull { it.architecture == selected }
    private fun currentSelection(): ArchitectureSelection? = config?.let { ArchitectureRanking.select(results, it, policy.selectedItem as ArchitecturePolicy) }

    private fun updateSummary() {
        val report = result ?: return
        val selection = currentSelection() ?: return
        val winner = selection.recommended
        summary.text = if (winner != null) "Recommended: ${winner.architecture} · ${winner.architecture.parameters} parameters · median ${scoreMode.label} ${number(winner.medianRmse)} · ${winner.successes}/${winner.expectedSeeds} successful seeds"
            else if (selection.bestError == null) "No fully evaluated finite candidate. The search was incomplete or every completed architecture had a failed seed."
            else "No evaluated architecture met the target reliably. Best completed RMSE: ${number(selection.bestError.medianRmse)}."
        summary.toolTipText = "${report.environment}; dataset SHA-256 ${report.data.fingerprint}; split seed ${report.data.splitSeed}. Validation is selection data, not an independent test score."
    }

    private fun updateInspection() {
        val candidate = selectedCandidate()
        val oldSeed = chosenTrial?.seed
        changing = true; selectedSeed.removeAllItems()
        candidate?.trials?.forEach { selectedSeed.addItem("Seed ${it.seed} · ${it.state}") }
        val index = candidate?.trials?.indexOfFirst { it.seed == oldSeed }?.takeIf { it >= 0 }
            ?: candidate?.representative?.let { candidate.trials.indexOf(it) }?.takeIf { it >= 0 } ?: 0
        if (selectedSeed.itemCount > 0) selectedSeed.selectedIndex = index
        changing = false
        inspectSelectedSeed()
    }

    private fun inspectSelectedSeed() {
        val candidate = selectedCandidate()
        val trial = candidate?.trials?.getOrNull(selectedSeed.selectedIndex)
        if (chosenTrial !== trial) {
            chosenTrial = trial
            surface = trial?.snapshot?.let { NeuroXorDiagnostics.renderOutputMap(it, NeuroStudio.resolution(it.parameterCount(), 160)) }
        }
        inspect.isEnabled = surface != null
        apply.isEnabled = !running && result != null && candidate?.valid == true
        replay.isEnabled = apply.isEnabled && trial?.state == ArchitectureTrialState.COMPLETED && trial.snapshot != null
        details.text = when {
            candidate == null -> "Select a point or row to inspect a scored checkpoint."
            trial == null -> "This architecture has no completed seed trial yet."
            trial.failure.isNotEmpty() -> "Seed ${trial.seed}: ${trial.failure}"
            else -> "Seed ${trial.seed}: best ${scoreMode.label} ${number(trial.bestRmse)} at epoch ${trial.bestEpoch}; trained ${trial.epochs}. Apply = fresh run; replay = scored training partition."
        }
        inspector.repaint(); plot.repaint()
    }

    private fun field(name: String, component: JComponent): JPanel = JPanel(BorderLayout(0, 4)).apply {
        isOpaque = false
        add(JLabel(name).apply { labelFor = component; foreground = MUTED }, BorderLayout.NORTH)
        component.getAccessibleContext().accessibleName = name
        add(component, BorderLayout.CENTER)
    }

    private inner class CandidateTable : AbstractTableModel() {
        private val columns = arrayOf("Architecture", "Parameters", "Median RMSE", "Worst RMSE", "Successful seeds", "Evaluated seeds", "Status")
        override fun getRowCount() = results.size
        override fun getColumnCount() = columns.size
        override fun getColumnName(column: Int) = columns[column]
        override fun getColumnClass(column: Int): Class<*> = when (column) { 1 -> Int::class.javaObjectType; 2, 3 -> Double::class.javaObjectType; else -> String::class.java }
        override fun getValueAt(row: Int, column: Int): Any = results[row].let { candidate -> when (column) {
            0 -> candidate.architecture.toString(); 1 -> candidate.architecture.parameters; 2 -> candidate.medianRmse; 3 -> candidate.worstRmse
            4 -> "${candidate.successes}/${candidate.expectedSeeds}"; 5 -> "${candidate.trials.size}/${candidate.expectedSeeds}"
            else -> if (!candidate.fullyEvaluated) "Provisional" else if (!candidate.valid) "Failed seed" else if (candidate.meetsTarget(config?.requiredSuccesses ?: 1)) "Target met" else "Above target"
        } }
    }

    private inner class ParetoPlot : JPanel() {
        private var points: List<Pair<Point, NetworkArchitecture>> = emptyList()
        init {
            background = SURFACE; preferredSize = Dimension(900, 300); minimumSize = Dimension(300, 180)
            toolTipText = "Click a point to inspect an architecture. Circles: complete; squares: provisional; diamond: recommended."
            getAccessibleContext().accessibleName = "Parameter count versus median RMSE Pareto plot"
            addMouseListener(object : MouseAdapter() {
                override fun mousePressed(event: MouseEvent) {
                    points.minByOrNull { it.first.distanceSq(event.point) }?.takeIf { it.first.distance(event.point) <= 14 }?.let { selectArchitecture(it.second) }
                }
            })
        }
        override fun paintComponent(graphics: Graphics) {
            super.paintComponent(graphics)
            val g = graphics.create() as Graphics2D
            try {
                smooth(g)
                val x = 66; val y = 35; val w = maxOf(1, width - 95); val h = maxOf(1, height - 85)
                val finite = results.filter { it.observedMedianRmse.isFinite() && (it.valid || !it.fullyEvaluated) }
                val maxP = maxOf(10, finite.maxOfOrNull { it.architecture.parameters } ?: 10)
                val goal = config?.targetRmse ?: decimal(target)
                val maxE = maxOf(0.1, goal, finite.maxOfOrNull { it.observedMedianRmse } ?: 0.6) * 1.1
                drawAxes(g, x, y, w, h, maxP, maxE, "Parameters · circles complete / squares provisional / diamond recommended", scoreMode.label)
                fun point(candidate: ArchitectureCandidate): Point = Point(x + (candidate.architecture.parameters.toDouble() / maxP * w).roundToInt(),
                    y + ((1 - candidate.observedMedianRmse / maxE) * h).roundToInt())
                val selection = currentSelection()
                val frontier = selection?.paretoFrontier.orEmpty()
                g.color = ACCENT; g.stroke = BasicStroke(1.5f)
                frontier.zipWithNext().forEach { (a, b) -> val p = point(a); val q = point(b); g.drawLine(p.x, p.y, q.x, q.y) }
                val goalY = y + ((1 - goal / maxE) * h).roundToInt()
                g.color = MUTED; g.stroke = BasicStroke(1f, BasicStroke.CAP_BUTT, BasicStroke.JOIN_MITER, 10f, floatArrayOf(5f, 5f), 0f)
                g.drawLine(x, goalY, x + w, goalY)
                drawText(g, "target ${number(goal)}", x + w - 120, goalY - 4, MUTED)
                points = finite.map { candidate ->
                    val p = point(candidate)
                    g.color = if (candidate.meetsTarget(config?.requiredSuccesses ?: 1)) ACCENT else BLUE
                    g.stroke = BasicStroke(1.5f)
                    if (candidate.valid) g.fillOval(p.x - 4, p.y - 4, 8, 8) else g.drawRect(p.x - 4, p.y - 4, 8, 8)
                    if (candidate.architecture == selection?.recommended?.architecture) {
                        g.color = ACCENT; g.drawPolygon(intArrayOf(p.x, p.x + 9, p.x, p.x - 9), intArrayOf(p.y - 9, p.y, p.y + 9, p.y), 4)
                    }
                    if (candidate.architecture == selected) { g.color = TEXT; g.drawOval(p.x - 11, p.y - 11, 22, 22) }
                    p to candidate.architecture
                }
                if (finite.isEmpty()) drawText(g, "Completed seed runs will appear here. No architecture is selected from a lucky seed alone.", x + 12, y + h / 2, MUTED)
            } finally { g.dispose() }
        }
    }

    private inner class Inspector : JPanel() {
        init { background = SURFACE; preferredSize = Dimension(900, 210); minimumSize = Dimension(300, 100) }
        override fun paintComponent(graphics: Graphics) {
            super.paintComponent(graphics)
            val g = graphics.create() as Graphics2D
            try {
                smooth(g)
                val trial = chosenTrial
                val image = surface
                if (trial == null || image == null) { drawText(g, "Inspect a seed checkpoint without replacing your active model.", 20, 60, MUTED); return }
                val size = minOf(160, height - 53).coerceAtLeast(40)
                g.drawImage(image, 12, 28, size, size, null)
                drawText(g, "Scored checkpoint · seed ${trial.seed}", 12, 17, TEXT)
                drawText(g, "x: 0 → 1     y: 1 → 0", 12, 30 + size + 15, MUTED)
                val x = maxOf(280, size + 100); val y = 28; val w = maxOf(1, width - x - 20); val h = maxOf(1, height - y - 38)
                val maximum = maxOf(0.1, trial.history.maxOfOrNull { maxOf(it.trainingRmse, it.score) } ?: 1.0) * 1.1
                drawAxes(g, x, y, w, h, maxOf(1, trial.epochs), maximum, "Epoch · solid training / dashed selected score", "RMSE")
                for (scored in listOf(false, true)) {
                    g.color = if (scored) ACCENT else BLUE
                    g.stroke = if (scored) BasicStroke(1.6f, BasicStroke.CAP_BUTT, BasicStroke.JOIN_MITER, 10f, floatArrayOf(6f, 4f), 0f) else BasicStroke(1.5f)
                    var previous: Point? = null
                    for (entry in trial.history) {
                        val p = Point(x + (entry.epoch.toDouble() / maxOf(1, trial.epochs) * w).roundToInt(),
                            y + ((1 - (if (scored) entry.score else entry.trainingRmse) / maximum) * h).roundToInt())
                        previous?.let { g.drawLine(it.x, it.y, p.x, p.y) }; previous = p
                    }
                }
                g.color = TEXT; val bestX = x + (trial.bestEpoch.toDouble() / maxOf(1, trial.epochs) * w).roundToInt()
                g.stroke = BasicStroke(1f); g.drawLine(bestX, y, bestX, y + h)
            } finally { g.dispose() }
        }
    }

    private class SearchSplit(top: Component, bottom: Component, private val ratio: Double) : JSplitPane(VERTICAL_SPLIT, top, bottom) {
        private var lastHeight = -1
        init { resizeWeight = ratio; border = EmptyBorder(0, 0, 0, 0); dividerSize = 7 }
        override fun doLayout() {
            if (height != lastHeight) { lastHeight = height; dividerLocation = (height * ratio).roundToInt() }
            super.doLayout()
        }
    }

    companion object {
        private val BOOLEAN_SETS = setOf(NeuroLearningSets.Kind.XOR, NeuroLearningSets.Kind.AND, NeuroLearningSets.Kind.OR, NeuroLearningSets.Kind.NAND, NeuroLearningSets.Kind.XNOR)
        private val BACKGROUND = Color(13, 20, 29)
        private val SURFACE = Color(23, 35, 47)
        private val TEXT = Color(234, 240, 247)
        private val MUTED = Color(166, 185, 203)
        private val ACCENT = Color(126, 224, 192)
        private val BLUE = Color(115, 200, 255)
        private val BORDER = Color(55, 74, 91)
        private fun integer(spinner: JSpinner) = (spinner.value as Number).toInt()
        private fun decimal(spinner: JSpinner) = (spinner.value as Number).toDouble()
        private fun number(value: Double) = if (value.isFinite()) String.format(Locale.ROOT, "%.5f", value) else "—"
        private fun smooth(g: Graphics2D) {
            g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
            g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON)
        }
        private fun drawText(g: Graphics2D, text: String, x: Int, y: Int, color: Color) {
            g.color = color; g.font = Font(Font.SANS_SERIF, Font.PLAIN, 11); g.drawString(text, x, y)
        }
        private fun drawAxes(g: Graphics2D, x: Int, y: Int, w: Int, h: Int, maxX: Int, maxY: Double, xLabel: String, yLabel: String) {
            drawText(g, yLabel, x - 40, y - 12, TEXT)
            g.stroke = BasicStroke(1f)
            for (i in 0..4) {
                val py = y + h * i / 4
                g.color = BORDER; g.drawLine(x, py, x + w, py)
                drawText(g, "%.2f".format(Locale.ROOT, maxY * (1 - i / 4.0)), x - 38, py + 4, MUTED)
            }
            drawText(g, "0", x, y + h + 16, MUTED); drawText(g, maxX.toString(), x + w - 30, y + h + 16, MUTED)
            drawText(g, xLabel, x + 40, y + h + 30, MUTED)
        }
    }
}
