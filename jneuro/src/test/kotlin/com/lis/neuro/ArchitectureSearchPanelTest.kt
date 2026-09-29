package com.lis.neuro

import java.awt.Container
import java.awt.EventQueue
import java.awt.Point
import java.awt.event.MouseEvent
import java.awt.image.BufferedImage
import javax.swing.*
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

class ArchitectureSearchPanelTest {
    @Test fun exposesValidatedSearchSettingsProgressAndCancellation() {
        var starts = 0; var cancels = 0
        EventQueue.invokeAndWait {
            val panel = ArchitectureSearchPanel({ config, evaluation, fraction, split ->
                assertEquals(ArchitectureEvaluation.TRAINING_FIT, evaluation)
                assertEquals(0.2, fraction); assertEquals(42L, split); assertEquals(584, config.architectures().size); starts++
            }, { cancels++ }, { _, _ -> }, { _, _, _ -> })
            panel.setSource(StudioConfig(), 4)
            assertEquals(5, panel.readConfig().seeds.size)
            button(panel, "Start search").doClick()
            assertEquals(1, starts); assertFalse(button(panel, "Start search").isEnabled)
            val config = panel.readConfig()
            panel.updateProgress(ArchitectureSearchProgress(584, 2920, 0, emptyList(),
                listOf(ArchitectureRunningTrial(NetworkArchitecture(listOf(2)), 42, 25, 0.5)), 10))
            assertTrue((field(panel, "summary") as JLabel).text.contains("epoch 25"))
            panel.updateProgress(ArchitectureSearchProgress(584, 2920, 0, emptyList(), emptyList(), 20))
            render(panel, 1150, 850)
            button(panel, "Cancel search").doClick(); assertEquals(1, cancels)
            panel.complete(ArchitectureSearchResult(ArchitectureSearchData.fitting(xor()), config,
                ArchitectureTermination.CANCELLED, 584, emptyList(), 20))
            assertTrue((field(panel, "summary") as JLabel).text.contains("No fully evaluated"))
            (field(panel, "seeds") as JTextField).text = "broken"
            button(panel, "Start search").doClick()
            assertTrue((field(panel, "summary") as JLabel).text.contains("Seeds"))
            (field(panel, "seeds") as JTextField).text = "1,42,123,999,2026"
            (field(panel, "splitSeed") as JTextField).text = "bad"
            button(panel, "Start search").doClick()
            assertTrue((field(panel, "summary") as JLabel).text.contains("Split seed"))
            (field(panel, "splitSeed") as JTextField).text = "42"
            (field(panel, "evaluation") as JComboBox<*>).selectedItem = ArchitectureEvaluation.VALIDATION
            button(panel, "Start search").doClick()
            assertTrue((field(panel, "summary") as JLabel).text.contains("Truth tables"))
            panel.setSource(StudioConfig(dataset = NeuroLearningSets.Kind.CIRCLE, maxEpochs = 100), 180)
            assertEquals(100, panel.readConfig().maxEpochs)
            val advanced = descendants(panel).filterIsInstance<JCheckBox>().single()
            advanced.doClick(); render(panel, 1020, 1100); advanced.doClick()
            panel.invalidateResults(); render(panel, 800, 650)
            assertEquals(0, (field(panel, "table") as JTable).rowCount)
            panel.updateProgress(ArchitectureSearchProgress(1, 5, 0, emptyList(), emptyList(), 0))
            assertEquals(0, (field(panel, "table") as JTable).rowCount)
        }
    }

    @Test fun completedResultsSupportSortingPerSeedInspectionPoliciesAndExplicitApplyReplay() {
        val report = report()
        var applications = 0; var replays = 0
        EventQueue.invokeAndWait {
            val panel = ArchitectureSearchPanel({ _, _, _, _ -> }, {}, { result, candidate ->
                assertSame(report, result); assertTrue(candidate.valid); applications++
            }, { result, candidate, trial ->
                assertSame(report, result); assertTrue(trial in candidate.trials); replays++
            })
            panel.setSource(StudioConfig(), 4)
            panel.complete(report)
            render(panel, 1150, 850)
            assertTrue(button(panel, "Apply architecture").isEnabled)
            assertTrue(button(panel, "Replay selected run").isEnabled)
            val table = field(panel, "table") as JTable
            assertEquals(3, table.rowCount); assertEquals(7, table.columnCount)
            for (row in 0 until table.rowCount) for (column in 0 until table.columnCount) {
                assertNotNull(table.getValueAt(row, column)); assertNotNull(table.getColumnClass(column)); assertNotNull(table.getColumnName(column))
            }
            table.rowSorter.toggleSortOrder(2)
            table.rowSorter.toggleSortOrder(2)
            panel.selectArchitecture(report.candidates.last().architecture)
            assertEquals(report.candidates.last().architecture.toString(), table.getValueAt(table.selectedRow, 0))
            table.setRowSelectionInterval(0, 0)
            (field(panel, "selectedSeed") as JComboBox<*>).selectedIndex = 1
            button(panel, "Inspect result").doClick()
            button(panel, "Apply architecture").doClick()
            button(panel, "Replay selected run").doClick()
            assertEquals(1, applications); assertEquals(1, replays)
            (field(panel, "policy") as JComboBox<*>).selectedItem = ArchitecturePolicy.LOWEST_RMSE
            assertTrue((field(panel, "summary") as JLabel).text.contains("Recommended"))
            (field(panel, "policy") as JComboBox<*>).selectedItem = ArchitecturePolicy.SMALLEST_NEAR_BEST
            render(panel, 800, 650)
            val plot = field(panel, "plot") as JPanel
            @Suppress("UNCHECKED_CAST")
            val points = field(plot, "points") as List<Pair<Point, NetworkArchitecture>>
            val point = points.first().first
            plot.dispatchEvent(MouseEvent(plot, MouseEvent.MOUSE_PRESSED, 0, 0, point.x, point.y, 1, false))
            assertEquals(report.candidates.first().architecture.toString(), table.getValueAt(table.selectedRow, 0))
            plot.dispatchEvent(MouseEvent(plot, MouseEvent.MOUSE_PRESSED, 0, 0, -500, -500, 1, false))
            assertTrue((field(panel, "details") as JLabel).text.contains("trained 100"))
            assertTrue((field(panel, "summary") as JLabel).toolTipText.contains(report.data.fingerprint))
            panel.selectArchitecture(null); render(panel, 1000, 750)
            assertFalse(button(panel, "Apply architecture").isEnabled)
            panel.selectArchitecture(report.candidates.first().architecture)
            panel.invalidateResults()
            assertFalse(button(panel, "Replay selected run").isEnabled)
        }
    }

    @Test fun incompleteFailedAndAboveTargetResultsRemainHonest() {
        val original = report()
        val first = original.candidates.first()
        val partial = ArchitectureCandidate(first.architecture, first.trials.take(1), 5, 0.05)
        val good = first.trials.first()
        val failed = ArchitectureTrial(good.seed, ArchitectureTrialState.FAILED, good.epochs, good.bestEpoch, good.bestRmse,
            good.trainingRmseAtBest, good.finalRmse, good.sampleUpdates, good.elapsedNanos, good.history, good.snapshot, "Non-finite RMSE")
        val badCandidate = ArchitectureCandidate(NetworkArchitecture(listOf(4)), listOf(failed), 1, 0.05)
        val completeAbove = ArchitectureCandidate(original.candidates[1].architecture, original.candidates[1].trials, 5, 1e-12)
        val config = ArchitectureSearchConfig(maxLayers = 1, maxWidth = 4, maxEpochs = 100, targetRmse = 1e-12)
        EventQueue.invokeAndWait {
            val panel = ArchitectureSearchPanel({ _, _, _, _ -> }, {}, { _, _ -> }, { _, _, _ -> })
            panel.setSource(StudioConfig(), 4)
            panel.started(config, ArchitectureEvaluation.TRAINING_FIT)
            panel.updateProgress(ArchitectureSearchProgress(4, 20, 7, listOf(partial, completeAbove, badCandidate), emptyList(), 100))
            panel.selectArchitecture(partial.architecture); render(panel, 1050, 800)
            assertFalse(button(panel, "Apply architecture").isEnabled)
            val table = field(panel, "table") as JTable
            assertEquals("Provisional", table.model.getValueAt(0, 6))
            assertEquals("Above target", table.model.getValueAt(1, 6))
            assertEquals("Failed seed", table.model.getValueAt(2, 6))
            panel.complete(ArchitectureSearchResult(original.data, config, ArchitectureTermination.CANCELLED, 4,
                listOf(partial, completeAbove, badCandidate), 100))
            assertTrue((field(panel, "summary") as JLabel).text.contains("No evaluated architecture"))
            panel.selectArchitecture(badCandidate.architecture)
            assertFalse(button(panel, "Replay selected run").isEnabled)
            assertTrue((field(panel, "details") as JLabel).text.contains("Non-finite"))
            render(panel, 1050, 800)
            panel.failed("Observer failure")
            assertTrue((field(panel, "summary") as JLabel).text.contains("Observer failure"))
            panel.setSource(StudioConfig(dataset = NeuroLearningSets.Kind.CUSTOM), 0)
            assertFalse(button(panel, "Start search").isEnabled)
        }
    }

    private fun report(): ArchitectureSearchResult = NeuroArchitectureSearch().search(ArchitectureSearchData.fitting(xor()),
        ArchitectureSearchConfig(maxLayers = 1, maxWidth = 3, maxEpochs = 100, targetRmse = 0.9))
    private fun xor() = NeuroLearningSets.create(NeuroLearningSets.Kind.XOR, 42)
    private fun field(instance: Any, name: String): Any? = instance.javaClass.getDeclaredField(name).run { isAccessible = true; get(instance) }
    private fun descendants(container: Container): List<java.awt.Component> = container.components.flatMap {
        listOf(it) + if (it is Container) descendants(it) else emptyList()
    }
    private fun button(panel: Container, title: String): JButton = descendants(panel).filterIsInstance<JButton>().first { it.text == title }
    private fun render(panel: JPanel, width: Int, height: Int) {
        panel.setSize(width, height)
        fun layout(container: Container) { container.doLayout(); container.components.filterIsInstance<Container>().forEach { layout(it) } }
        layout(panel)
        val image = BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
        val graphics = image.createGraphics()
        try { panel.printAll(graphics) } finally { graphics.dispose() }
        assertNotEquals(0, image.getRGB(width / 2, height / 2))
    }
}
