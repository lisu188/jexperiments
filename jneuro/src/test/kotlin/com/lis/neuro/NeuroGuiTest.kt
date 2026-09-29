package com.lis.neuro

import java.awt.*
import java.awt.datatransfer.StringSelection
import java.awt.event.InputEvent
import java.awt.event.KeyEvent
import java.nio.file.Files
import java.nio.file.Path
import javax.imageio.ImageIO
import javax.swing.*
import javax.swing.text.JTextComponent
import org.junit.jupiter.api.Timeout
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.TestMethodOrder
import org.junit.jupiter.api.MethodOrderer
import org.junit.jupiter.api.TestInfo
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.io.TempDir

@Timeout(value = 120, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
@Tag("gui")
@TestMethodOrder(MethodOrderer.MethodName::class)
class NeuroGuiTest {
    @TempDir lateinit var temporary: Path
    private lateinit var ui: NeuroXorCanvas
    private lateinit var window: JFrame
    private lateinit var robot: Robot
    private val root: JPanel get() = field(ui, "root") as JPanel
    private val tabs: JTabbedPane get() = field(ui, "tabs") as JTabbedPane
    private val current: StudioFrame get() = field(ui, "frame") as StudioFrame
    private val panel: ArchitectureSearchPanel get() = field(ui, "architectureSearch") as ArchitectureSearchPanel

    @BeforeEach fun open() {
        assertFalse(GraphicsEnvironment.isHeadless(), "GUI tests require a desktop/Xvfb; they must never be silently skipped.")
        robot = Robot().apply { autoDelay = 20 }
        edt {
            val theme = NeuroXorCanvas.Companion.javaClass.getDeclaredMethod("installTheme")
            theme.isAccessible = true
            theme.invoke(NeuroXorCanvas.Companion)
            val constructor = NeuroXorCanvas::class.java.getDeclaredConstructor(StudioFrame::class.java, Boolean::class.javaPrimitiveType)
            constructor.isAccessible = true
            ui = constructor.newInstance(null, true)
            ui.javaClass.getDeclaredMethod("showWindow").apply { isAccessible = true }.invoke(ui)
            window = Window.getWindows().filterIsInstance<JFrame>().last { it.isShowing && it.title.contains("JNeuro") }
            window.setSize(1520, 1080)
            window.setLocation(20, 20)
            window.extendedState = JFrame.NORMAL
            window.toFront()
        }
        await("ready visible window") { field(ui, "frame") is StudioFrame && current.state == StudioState.READY && window.isActive }
    }

    @AfterEach fun close(info: TestInfo) {
        if (!::ui.isInitialized) return
        if (window.isShowing) screenshot(info.testMethod.orElseThrow().name)
        edt { Window.getWindows().filter { it.isDisplayable }.forEach { it.dispose() }; ui.close() }
        (field(ui, "thread") as Thread).join(8000)
        assertFalse((field(ui, "thread") as Thread).isAlive, "Window close must terminate the Studio worker.")
        await("gallery and search workers stopped") {
            Thread.getAllStackTraces().keys.none { it.isAlive && (it.name.startsWith("jneuro-search-") || it.name == "jneuro-neuron-gallery") }
        }
    }

    @Test fun opensAndNavigatesEveryView() {
        for (name in listOf("Overview", "Neurons", "Learning set", "Step effect", "Parameters", "Seeds", "Timeline", "Architecture search")) {
            tab(name)
            assertEquals(name, edt { tabs.getTitleAt(tabs.selectedIndex) })
            screenshot("view-${name.replace(' ', '-').lowercase()}")
        }
        click(button("Auto search…"))
        assertEquals("Architecture search", edt { tabs.getTitleAt(tabs.selectedIndex) })
        assertEquals(0, edt { current.diagnostics.epoch() })
    }

    @Test fun validatesConfigurationAndAcceptsUncappedInputs() {
        text(input("Hidden layers"), "2,")
        shortcut(KeyEvent.VK_ENTER)
        await("invalid topology") { errorText().contains("Layer") }
        assertEquals("6", edt { current.config.hidden })
        text(input("Hidden layers"), "129,1,1,1,1,1,1,1,1")
        text(input("Random seed"), "invalid")
        shortcut(KeyEvent.VK_ENTER)
        await("invalid seed") { errorText().contains("invalid") && current.config.hidden == "6" }
        text(input("Random seed"), Long.MAX_VALUE.toString())
        number("Maximum epochs", "1000001")
        advanced()
        number("Learning rate", "25.123456789")
        number("Momentum", "0.9999")
        number("Target RMSE", "0")
        shortcut(KeyEvent.VK_ENTER)
        await("uncapped configuration applied") { current.config.hidden.startsWith("129,") && current.config.maxEpochs == 1_000_001 }
        shortcut(KeyEvent.VK_R)
        await("reset uncapped model") { current.state == StudioState.READY }
        assertEquals(25.123456789, edt { current.config.learningRate }, 0.0)
        assertEquals(0.9999, edt { current.config.momentum }, 0.0)
        assertEquals(Long.MAX_VALUE, edt { current.config.seed })
        assertEquals(9, edt { current.diagnostics.hiddenLayerCount() })
        number("Learning rate", "1e-10")
        shortcut(KeyEvent.VK_ENTER)
        await("precise small rate") { current.config.learningRate == 1e-10 }
        shortcut(KeyEvent.VK_R)
        number("Learning rate", "0")
        shortcut(KeyEvent.VK_ENTER)
        await("semantic rate validation") { errorText().contains("learningRate") }
        number("Learning rate", "0.6")
        number("Momentum", "1")
        shortcut(KeyEvent.VK_ENTER)
        await("semantic momentum validation") { errorText().contains("momentum") }
        edt {
            descendants(root).filterIsInstance<JSpinner>().forEach {
                assertNull((it.model as SpinnerNumberModel).minimum)
                assertNull((it.model as SpinnerNumberModel).maximum)
            }
        }
    }

    @Test fun searchInputsAcceptValuesBeyondOldCaps() {
        tab("Architecture search")
        click(button("Advanced search settings"))
        for ((name, value) in listOf("Max layers" to "9", "Max width" to "129", "Max parameters" to "1000001",
            "Epochs per seed" to "1000001", "Check every (epochs)" to "1000001", "Parallel seed trials" to "33",
            "Trial budget" to "20001", "Seconds (0 = unlimited)" to "2147483648", "Validation fraction" to "0.05",
            "Near-best tolerance" to "2", "Plateau length (architectures)" to "1001", "Max restarts" to "101")) number(name, value, panel)
        text(input("Seeds", panel), (1..21).joinToString(","))
        number("Required successes", "21", panel)
        val config = edt { panel.readConfig() }
        assertEquals(9, config.maxLayers); assertEquals(129, config.maxWidth)
        assertEquals(1_000_001, config.maxParameters); assertEquals(1_000_001, config.maxEpochs)
        assertEquals(33, config.parallelism); assertEquals(20_001, config.maxTrials)
        assertEquals(2_147_483_648L, config.timeLimitSeconds); assertEquals(21, config.seeds.size)
        assertEquals(21, config.requiredSuccesses); assertEquals(101, config.maxRestarts)
        assertEquals(1001, config.restartAfter); assertEquals(2.0, config.nearBestTolerance)
        assertFalse(edt { field(ui, "searchRunning") as Boolean })
    }

    @Test fun trainingButtonsAndKeyboardRespectState() {
        configure("1", 100_000, 0.0)
        click(button("1 epoch"))
        await("one epoch") { current.diagnostics.epoch() == 1 && current.state == StudioState.PAUSED }
        click(button("10 epochs"))
        await("ten epochs") { current.diagnostics.epoch() == 11 }
        shortcut(KeyEvent.VK_R)
        await("keyboard reset") { current.state == StudioState.READY && current.diagnostics.epoch() == 0 }
        click(button("Train"))
        await("training") { current.state == StudioState.RUNNING && current.diagnostics.epoch() > 0 && (field(ui,"train") as JButton).text == "Pause" }
        click(button("Pause"))
        await("paused") { current.state == StudioState.PAUSED && button("Train").isEnabled }
        val paused = edt { current.diagnostics.epoch() }
        robot.delay(200)
        assertEquals(paused, edt { current.diagnostics.epoch() })
        shortcut(KeyEvent.VK_RIGHT)
        await("keyboard single epoch") { current.diagnostics.epoch() == paused + 1 }
        shortcut(KeyEvent.VK_SPACE)
        await("keyboard resume") { current.state == StudioState.RUNNING }
        shortcut(KeyEvent.VK_SPACE)
        await("keyboard pause") { current.state == StudioState.PAUSED }
        click(button("Reset"))
        await("button reset") { current.state == StudioState.READY }
        number("Epochs per refresh", "1234")
        assertEquals(1234, edt { field(ui, "speed") as Int })
        number("Epochs per refresh", "0")
        await("invalid refresh count") { errorText().contains("positive") }
    }

    @Test fun terminalTrainingStatesDisableControls() {
        text(input("Hidden layers"), "2")
        number("Maximum epochs", "2")
        advanced(); number("Target RMSE", "0")
        shortcut(KeyEvent.VK_ENTER)
        await("epoch limit and disabled controls") { current.state == StudioState.LIMIT_REACHED && !button("Train").isEnabled }
        assertFalse(edt { button("1 epoch").isEnabled })
        assertFalse(edt { button("10 epochs").isEnabled })
        tab("Overview"); screenshot("limit-state")
        number("Target RMSE", "2")
        shortcut(KeyEvent.VK_ENTER)
        await("target reached and disabled controls") { current.state == StudioState.CONVERGED && !button("Train").isEnabled }
        assertEquals(0, edt { current.diagnostics.epoch() })
        screenshot("target-state")
    }

    @Test fun everyNeuronIsReachableByScrollingAndBaselineHasNoMaps() {
        configure("16,16,16", 1, 0.0)
        tab("Neurons")
        val scroll = edt { tabs.selectedComponent as JScrollPane }
        val chart = edt { scroll.viewport.view as JComponent }
        val layout = edt { NeuronGallery.Layout(current.diagnostics, chart.width) }
        assertEquals(listOf(16,16,16), layout.sections.map { it.neurons })
        assertEquals(48, layout.sections.sumOf { it.neurons })
        assertTrue(edt { scroll.verticalScrollBar.isVisible })
        wheel(scroll, 120)
        await("last neuron reached and rendered") {
            val last = layout.bounds(layout.sections.last(), 15)
            val gallery = field(ui, "neuronGallery") as NeuronGallery
            scroll.viewport.viewRect.intersects(last) && gallery.image(2,15) != null
        }
        screenshot("neurons-all-bottom")
        assertFalse(edt { descendants(root).any { it is JLabel && it.text == "First neuron" } })
        wheel(scroll, -120)
        await("gallery top") { scroll.verticalScrollBar.value == 0 }
        wheel(scroll, 6)
        await("first layer map") { (field(ui, "neuronGallery") as NeuronGallery).image(0, 0) != null }
        screenshot("neurons-all-top")
        configure("", 1, 0.0)
        await("baseline gallery cleared") { current.diagnostics.hiddenLayerCount() == 0 && (field(ui, "neuronGallery") as NeuronGallery).image(2,15) == null }
        click(button("+ Layer"))
        await("layer added to empty editor") { input("Hidden layers").text == "6" }
        click(button("− Layer"))
        await("last layer removed") { input("Hidden layers").text == "" }
        click(button("− Layer"))
        await("remove empty layer validation") { errorText().contains("no hidden layer") }
        configure("129", 1, 0.0)
        wheel(scroll, 500)
        await("neuron beyond former cap rendered") { (field(ui, "neuronGallery") as NeuronGallery).image(0,128) != null }
        screenshot("neurons-129-bottom")
    }

    @Test fun parameterNavigationAndSmallWindowRemainUsable() {
        configure("33,24", 2, 0.0)
        click(button("1 epoch")); await("parameter history") { current.diagnostics.epoch() == 1 }
        tab("Parameters")
        choose(combo("Parameter layer"), 1)
        number("First parameter", "24")
        await("parameter page") { field(ui,"selectedParameterStart") == 24 && field(ui,"selectedParameterLayer") == 1 }
        number("First parameter", "-1")
        await("parameter index validation") { errorText().contains("offset") }
        assertEquals(24, edt { field(ui,"selectedParameterStart") })
        edt { window.extendedState = JFrame.NORMAL; window.setSize(1000,720) }
        await("window resized") { window.width <= 1020 && window.height <= 740 }
        val scroll = edt { tabs.selectedComponent as JScrollPane }
        wheel(scroll, 80)
        await("parameter scrolling") { scroll.verticalScrollBar.value > 0 }
        screenshot("parameters-small-window")
        edt { window.setSize(1520,1080) }
        tab("Architecture search")
        click(button("Advanced search settings"))
        val searchScroll = edt { tabs.selectedComponent as JScrollPane }
        wheel(searchScroll, 100)
        await("search scrolling") { searchScroll.verticalScrollBar.value > 0 }
    }

    @Test fun customPointsSupportMouseKeyboardUndoAndClear() {
        choose(combo("Dataset"), NeuroLearningSets.Kind.CUSTOM.ordinal)
        shortcut(KeyEvent.VK_ENTER)
        await("empty custom state") { current.state == StudioState.EMPTY && !button("Train").isEnabled }
        tab("Architecture search")
        assertFalse(edt { button("Start search").isEnabled })
        tab("Learning set")
        val chart = edt { (tabs.selectedComponent as JScrollPane).viewport.view as JComponent }
        val plot = edt { field(chart,"interactive") as Rectangle }
        point(chart, plot.x + plot.width / 3, plot.y + plot.height / 3)
        await("left point") { current.samples.size == 1 }
        assertEquals(1.0, edt { current.samples.last().target })
        point(chart, plot.x + plot.width * 2 / 3, plot.y + plot.height * 2 / 3, InputEvent.BUTTON3_DOWN_MASK)
        await("right point") { current.samples.size == 2 }
        assertEquals(0.0, edt { current.samples.last().target })
        choose(combo("Point class"), 1)
        point(chart, plot.x + plot.width / 2, plot.y + plot.height / 2)
        await("selected class point") { current.samples.size == 3 }
        assertEquals(0.0, edt { current.samples.last().target })
        key(KeyEvent.VK_LEFT); key(KeyEvent.VK_UP); key(KeyEvent.VK_RIGHT); key(KeyEvent.VK_DOWN); key(KeyEvent.VK_ENTER)
        await("keyboard point") { current.samples.size == 4 }
        click(button("Undo point")); await("undo") { current.samples.size == 3 }
        click(button("Clear custom")); await("clear") { current.samples.isEmpty() && current.state == StudioState.EMPTY }
    }

    @Test fun seedComparisonCanCompleteAndBeCancelled() {
        configure("2", 50, 0.0)
        tab("Seeds")
        click(button("Compare 4 seeds"))
        await("four seed results") { current.seeds.size == 4 && field(ui,"studyRunning") == false }
        assertEquals(0, edt { current.diagnostics.epoch() })
        screenshot("seeds-completed")
        configure("16", Int.MAX_VALUE, 0.0)
        tab("Seeds"); click(button("Compare 4 seeds"))
        await("study running") { field(ui,"studyRunning") == true && button("Cancel study").isShowing }
        click(button("Cancel study"))
        await("study cancelled") { field(ui,"studyRunning") == false }
        click(button("Reset")); await("model responsive after cancel") { current.state == StudioState.READY }
    }

    @Test fun searchValidationCancellationAndStaleResultsAreHandled() {
        tab("Architecture search")
        number("Max layers", "0", panel)
        click(button("Start search"))
        await("search bounds validation") { (field(panel,"summary") as JLabel).text.contains("bounds") }
        number("Max layers", "1", panel)
        choose(combo("Scoring mode",panel), ArchitectureEvaluation.VALIDATION.ordinal)
        click(button("Start search"))
        await("truth table validation") { (field(panel,"summary") as JLabel).text.contains("Truth tables") }
        choose(combo("Scoring mode",panel), ArchitectureEvaluation.TRAINING_FIT.ordinal)
        searchSettings(Int.MAX_VALUE)
        click(button("Start search"))
        await("search running") { field(ui,"searchRunning") == true }
        click(button("Cancel search"))
        await("search cancelled report") { (field(panel,"result") as ArchitectureSearchResult?)?.termination == ArchitectureTermination.CANCELLED }
        click(button("Start search"))
        await("second search running") { field(ui,"searchRunning") == true }
        click(button("Reset"))
        await("reset invalidates search") { field(ui,"searchRunning") == false && field(panel,"result") == null && current.state == StudioState.READY }
        robot.delay(300)
        assertNull(edt { field(panel,"result") })
    }

    @Test fun searchResultsSupportNativeSelectionInspectionReplayAndApply() {
        val initial = edt { current.diagnostics.parameters() }
        tab("Architecture search"); searchSettings(100)
        click(button("Start search"))
        await("completed sweep") { field(panel,"result") != null }
        val report = edt { field(panel,"result") as ArchitectureSearchResult }
        assertEquals(3, report.evaluated)
        assertArrayEquals(initial, edt { current.diagnostics.parameters() })
        val table = edt { field(panel,"table") as JTable }
        reveal(table)
        point(table, 50, table.rowHeight + table.rowHeight / 2)
        await("table selection") { table.selectedRow >= 0 }
        click(button("Inspect result"))
        assertNotNull(edt { field(panel,"surface") })
        choose(combo("Result seed",panel), 1)
        assertNotNull(edt { field(panel,"chosenTrial") })
        val header = edt { table.tableHeader }
        point(header, 100, header.height / 2)
        assertTrue(edt { table.rowSorter.sortKeys.isNotEmpty() })
        choose(combo("Recommendation policy",panel), ArchitecturePolicy.SMALLEST_NEAR_BEST.ordinal)
        assertTrue(edt { (field(panel,"summary") as JLabel).text.isNotBlank() })
        choose(combo("Recommendation policy",panel), ArchitecturePolicy.LOWEST_RMSE.ordinal)
        val plot = edt { field(panel,"plot") as JComponent }
        reveal(plot)
        val points = edt { field(plot,"points") as List<*> }
        val candidatePoint = (points.first() as Pair<*, *>).first as Point
        point(plot, candidatePoint.x, candidatePoint.y)
        await("scatter selection") { field(panel,"selected") != null }
        screenshot("search-results")
        val trial = edt { field(panel,"chosenTrial") as ArchitectureTrial }
        click(button("Replay selected run"))
        await("replay installation") { current.replayNote.isNotEmpty() && current.diagnostics.epoch() == trial.bestEpoch }
        assertTrue(edt { current.replayNote.contains("Search replay") })
        click(button("Reset")); await("replay reset") { current.replayNote.isEmpty() && current.diagnostics.epoch() == 0 }
        tab("Architecture search"); searchSettings(50)
        choose(combo("Search strategy",panel),ArchitectureSearchStrategy.ADAPTIVE.ordinal)
        click(button("Start search")); await("new sweep") { field(panel,"result") != null }
        val adaptive = edt { field(panel,"result") as ArchitectureSearchResult }
        assertEquals(ArchitectureSearchStrategy.ADAPTIVE, adaptive.config.strategy)
        assertTrue(adaptive.lineage.size > 1)
        for ((index, proposal) in adaptive.lineage.withIndex()) {
            if (index == 0) continue
            val selection = ArchitectureRanking.select(adaptive.candidates.take(proposal.evaluatedCount), adaptive.config)
            assertTrue(proposal.parent in EliteParentSelection.rank(selection, adaptive.config))
        }
        val ancestry = edt { field(panel,"table") as JTable }
        for (row in adaptive.candidates.indices) {
            val proposal = adaptive.lineage.first { it.architecture == adaptive.candidates[row].architecture }
            assertEquals(proposal.parent?.toString() ?: "—", edt { ancestry.model.getValueAt(row, 7) })
        }
        click(button("Apply architecture"))
        await("fresh chosen model") { current.config.maxEpochs == 50 && current.diagnostics.epoch() == 0 && current.replayNote.isEmpty() }
    }

    @Test fun searchDeadlineDoesNotRecommendIncompleteCandidates() {
        tab("Architecture search"); searchSettings(Int.MAX_VALUE)
        number("Seconds (0 = unlimited)","1",panel)
        click(button("Start search"))
        await("deadline report") { (field(panel,"result") as ArchitectureSearchResult?)?.termination == ArchitectureTermination.TIME_LIMIT }
        val result = edt { field(panel,"result") as ArchitectureSearchResult }
        assertNull(result.selection.recommended)
        assertTrue(result.partial > 0)
        assertFalse(edt { button("Apply architecture").isEnabled })
        assertFalse(edt { button("Replay selected run").isEnabled })
        assertTrue(edt { button("Start search").isEnabled })
    }

    @Test fun validationSearchReplaysHeldOutPartition() {
        choose(combo("Dataset"),NeuroLearningSets.Kind.CIRCLE.ordinal)
        number("Maximum epochs","1")
        click(button("Apply & restart"))
        await("circle installed") { current.config.dataset == NeuroLearningSets.Kind.CIRCLE }
        click(button("Reset")); await("circle ready") { current.state == StudioState.READY }
        tab("Architecture search")
        assertEquals(ArchitectureEvaluation.VALIDATION,edt { combo("Scoring mode",panel).selectedItem })
        searchSettings(25)
        choose(combo("Scoring mode",panel),ArchitectureEvaluation.VALIDATION.ordinal)
        click(button("Start search")); await("validation results") { field(panel,"result") != null }
        val report = edt { field(panel,"result") as ArchitectureSearchResult }
        assertEquals(ArchitectureEvaluation.VALIDATION,report.data.evaluation)
        assertTrue(report.data.validation.isNotEmpty())
        click(button("Replay selected run"))
        await("held-out replay") { current.replayNote.contains("Validation RMSE") }
        assertEquals(report.data.training.size,edt { current.samples.size })
        click(button("Reset")); await("replay reset to full dataset") { current.samples.size == 180 && current.replayNote.isEmpty() }
    }

    @Test fun pngExportAndCancelUseTheRealFileDialog() {
        tab("Neurons")
        click(button("Save PNG"))
        await("save dialog") { Window.getWindows().any { it.isShowing && it is JDialog } }
        val chooser = edt { Window.getWindows().filterIsInstance<JDialog>().flatMap { descendants(it) }.filterIsInstance<JFileChooser>().first() }
        val filename = edt { descendants(chooser).filterIsInstance<JTextField>().last { it.isEditable } }
        val output = temporary.resolve("studio-export.png")
        text(filename,output.toString())
        click(edt { descendants(chooser).filterIsInstance<JButton>().first { it.text == "Save" } })
        await("saved PNG") { Files.exists(output) && Files.size(output) > 1000 }
        assertNotNull(ImageIO.read(output.toFile()))
        click(button("Save PNG"))
        await("second save dialog") { Window.getWindows().filterIsInstance<JDialog>().any { it.isShowing } }
        key(KeyEvent.VK_ESCAPE)
        await("cancel closes dialog") { Window.getWindows().filterIsInstance<JDialog>().none { it.isShowing } }
        assertEquals(1L, Files.list(temporary).use { it.count() })
    }

    @Test fun allDatasetsRemainSelectableAndTrainingErrorsRecover() {
        number("Maximum epochs","1")
        for (kind in NeuroLearningSets.Kind.entries) {
            choose(combo("Dataset"), kind.ordinal)
            click(button("Apply & restart"))
            await("dataset $kind") { current.config.dataset == kind }
            tab("Learning set")
            assertEquals(if (kind == NeuroLearningSets.Kind.CUSTOM) 0 else NeuroLearningSets.create(kind,0xC0FFEE42L).size, edt { current.samples.size })
        }
        text(input("Hidden layers"),"2147483647")
        shortcut(KeyEvent.VK_ENTER)
        await("oversized topology reported without crash") { current.state == StudioState.FAILED || errorText().contains("JVM") }
        text(input("Hidden layers"),"6")
        choose(combo("Dataset"),0)
        shortcut(KeyEvent.VK_ENTER)
        await("recovery after rejected allocation") { current.config.hidden == "6" && current.config.dataset == NeuroLearningSets.Kind.XOR && current.state != StudioState.FAILED }
    }

    @Test fun windowCloseStopsTrainingAndSearchWorkers() {
        configure("16,16", Int.MAX_VALUE, 0.0)
        tab("Neurons")
        val scroll = edt { tabs.selectedComponent as JScrollPane }
        wheel(scroll,6)
        tab("Architecture search"); searchSettings(Int.MAX_VALUE)
        click(button("Start search")); await("search before close") { field(ui,"searchRunning") == true }
        robot.keyPress(KeyEvent.VK_ALT); key(KeyEvent.VK_F4); robot.keyRelease(KeyEvent.VK_ALT)
        await("native window close") { !window.isDisplayable }
    }

    @Test fun parallelSettingUsesMultipleArchitecturesWithOneSeedAndReportsUtilization() {
        configure("4,5,6,7,8", 1, 0.0)
        tab("Architecture search"); searchSettings(100_000)
        number("Max layers", "6", panel); number("Max width", "16", panel)
        number("Max parameters", "2048", panel)
        number("Parallel seed trials", "32", panel)
        number("Required successes", "1", panel)
        text(input("Seeds", panel), "42")
        number("Check every (epochs)", "100", panel)
        choose(combo("Search strategy", panel), ArchitectureSearchStrategy.ADAPTIVE.ordinal)
        click(button("Start search"))
        await("32 concurrent search trials") {
            val text = (field(panel, "summary") as JLabel).text
            val active = Regex("Active trials: (\\d+)/32").find(text)?.groupValues?.get(1)?.toIntOrNull() ?: 0
            active == 32
        }
        val summary = edt { (field(panel, "summary") as JLabel).text }
        assertTrue((Regex("(\\d+) architectures").find(summary)?.groupValues?.get(1)?.toInt() ?: 0) > 1)
        assertEquals(32, edt { (field(panel, "config") as ArchitectureSearchConfig).parallelism })
        assertEquals(listOf(42L), edt { (field(panel, "config") as ArchitectureSearchConfig).seeds })
        screenshot("parallel-trials")
        click(button("Cancel search"))
        await("parallel cancellation") { field(panel, "result") != null }
        val report = edt { field(panel, "result") as ArchitectureSearchResult }
        assertEquals(ArchitectureTermination.CANCELLED, report.termination)
        assertEquals(32, report.peakParallelTrials)
        assertTrue(edt { (field(panel, "progressBar") as JProgressBar).string.contains("peak ${report.peakParallelTrials}/32") })
    }

    private fun configure(hidden: String, epochs: Int, target: Double) {
        text(input("Hidden layers"), hidden)
        number("Maximum epochs", epochs.toString())
        advanced(); number("Target RMSE",target.toString())
        shortcut(KeyEvent.VK_ENTER)
        await("configuration installed") { current.config.hidden == hidden && current.config.maxEpochs == epochs && current.config.targetError == target }
        shortcut(KeyEvent.VK_R)
        await("configuration reset") { current.state == StudioState.READY && current.diagnostics.epoch() == 0 }
    }

    private fun searchSettings(epochs: Int) {
        number("Max layers","1",panel); number("Max width","3",panel)
        if (!edt { button("Advanced search settings").isSelected }) click(button("Advanced search settings"))
        number("Epochs per seed",epochs.toString(),panel); number("Check every (epochs)","25",panel)
        number("Target RMSE","0.9",panel)
        choose(combo("Search strategy",panel),ArchitectureSearchStrategy.EXHAUSTIVE.ordinal)
        choose(combo("Scoring mode",panel),ArchitectureEvaluation.TRAINING_FIT.ordinal)
    }

    private fun advanced() {
        val checkbox = button("Advanced settings")
        if (!edt { checkbox.isSelected }) click(checkbox)
    }
    private fun errorText(): String = (field(ui,"configError") as JLabel).text.replace(Regex("<[^>]*>"),"").trim()
    private fun input(label: String, scope: Container = root): JTextComponent = edt {
        descendants(scope).filterIsInstance<JTextComponent>().first { it.accessibleContext?.accessibleName == label }
    }
    private fun combo(label: String, scope: Container = root): JComboBox<*> = edt {
        descendants(scope).filterIsInstance<JComboBox<*>>().first { it.accessibleContext?.accessibleName == label }
    }
    private fun button(label: String): AbstractButton = edt {
        descendants(root).filterIsInstance<AbstractButton>().first { it.text == label }
    }
    private fun number(label: String, value: String, scope: Container = root) {
        val spinner = edt { descendants(scope).filterIsInstance<JSpinner>().first { it.accessibleContext?.accessibleName == label } }
        text(edt { (spinner.editor as JSpinner.DefaultEditor).textField },value)
        key(KeyEvent.VK_TAB)
        await("numeric value $label=$value") { (spinner.value as Number).toDouble() == value.toDouble() }
    }
    private fun choose(combo: JComboBox<*>, index: Int) {
        click(combo); key(KeyEvent.VK_HOME); repeat(index) { key(KeyEvent.VK_DOWN) }; key(KeyEvent.VK_ENTER)
        await("combo selection $index") { combo.selectedIndex == index }
    }
    private fun tab(name: String) {
        val bounds = edt { tabs.getBoundsAt((0 until tabs.tabCount).first { tabs.getTitleAt(it) == name }) }
        point(tabs,bounds.x + bounds.width / 2,bounds.y + bounds.height / 2)
        await("tab $name") { tabs.getTitleAt(tabs.selectedIndex) == name }
        robot.waitForIdle()
    }
    private fun text(component: JTextComponent, value: String) {
        reveal(component); click(component)
        edt { Toolkit.getDefaultToolkit().systemClipboard.setContents(StringSelection(value),null) }
        shortcut(KeyEvent.VK_A); shortcut(KeyEvent.VK_V)
        await("text input") { component.text == value }
    }
    private fun reveal(component: JComponent) {
        edt { component.scrollRectToVisible(Rectangle(0,0,component.width,component.height)) }
        robot.waitForIdle()
    }
    private fun click(component: Component) {
        if (component is JComponent) reveal(component)
        val position = edt { val bounds = (component as? JComponent)?.visibleRect ?: Rectangle(0,0,component.width,component.height)
            val screen = component.locationOnScreen; Point(screen.x + bounds.x + bounds.width/2,screen.y + bounds.y + bounds.height/2) }
        robot.mouseMove(position.x,position.y); robot.mousePress(InputEvent.BUTTON1_DOWN_MASK); robot.mouseRelease(InputEvent.BUTTON1_DOWN_MASK)
        robot.waitForIdle()
    }
    private fun point(component: Component, x: Int, y: Int, mask: Int = InputEvent.BUTTON1_DOWN_MASK) {
        val position = edt { component.locationOnScreen }
        robot.mouseMove(position.x+x,position.y+y); robot.mousePress(mask); robot.mouseRelease(mask); robot.waitForIdle()
    }
    private fun wheel(scroll: JScrollPane, amount: Int) {
        val location = edt { val p = scroll.viewport.locationOnScreen; Point(p.x+scroll.viewport.width/2,p.y+minOf(100,scroll.viewport.height/2)) }
        robot.mouseMove(location.x,location.y); robot.mouseWheel(amount); robot.waitForIdle()
    }
    private fun key(key: Int) { robot.keyPress(key); robot.keyRelease(key); robot.waitForIdle() }
    private fun shortcut(key: Int) { robot.keyPress(KeyEvent.VK_CONTROL); key(key); robot.keyRelease(KeyEvent.VK_CONTROL); robot.waitForIdle() }
    private fun screenshot(name: String) {
        val dir = Path.of(System.getProperty("gui.screenshots", "build/screenshots/gui"))
        Files.createDirectories(dir)
        val bounds = edt { window.bounds }
        ImageIO.write(robot.createScreenCapture(bounds),"png",dir.resolve("$name.png").toFile())
    }
    private fun await(description: String, predicate: () -> Boolean) {
        val deadline = System.nanoTime() + 15_000_000_000L
        while (System.nanoTime() < deadline) {
            if (edt(predicate)) return
            Thread.sleep(25)
        }
        fail<Unit>("Timed out: $description")
    }
    private fun <T> edt(block: () -> T): T {
        if (EventQueue.isDispatchThread()) return block()
        var value: T? = null
        var failure: Throwable? = null
        EventQueue.invokeAndWait { try { value = block() } catch (exception: Throwable) { failure=exception } }
        failure?.let { throw it }
        @Suppress("UNCHECKED_CAST") return value as T
    }
    private fun field(instance: Any, name: String): Any? = instance.javaClass.getDeclaredField(name).run { isAccessible=true; get(instance) }
    private fun descendants(root: Container): List<Component> = root.components.flatMap { listOf(it) + if (it is Container) descendants(it) else emptyList() }
}
