package com.lis.neuro

import java.awt.Container
import java.awt.EventQueue
import java.awt.image.BufferedImage
import javax.swing.JButton
import javax.swing.JComboBox
import javax.swing.JPanel
import javax.swing.JSpinner
import javax.swing.JTabbedPane
import javax.swing.JTextField
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

class NeuroBaselineTopologyTest {
    @Test fun emptySpecificationAndLayerEditingRoundTrip() {
        for (text in listOf("", " ", "\t\n")) {
            assertArrayEquals(intArrayOf(), NeuroTopologyConfig.parseHidden(text))
            assertArrayEquals(intArrayOf(2, 1), StudioConfig(hidden = text).topology())
        }
        assertEquals("2 → 1", StudioConfig(hidden = "").description())
        assertArrayEquals(intArrayOf(), NeuroTopologyConfig.removeLayer(intArrayOf(6)))
        assertArrayEquals(intArrayOf(6), NeuroTopologyConfig.addLayer(intArrayOf(), 6))
        assertEquals(8, NeuroTopologyConfig.parseHidden("128,128,128,128,128,128,128,128").size)
        assertThrows(IllegalArgumentException::class.java) { NeuroTopologyConfig.addLayer(intArrayOf(), 0) }
    }

    @Test fun baselineDiagnosticsMatchTrainingAndOwnTheirSnapshots() {
        for (mode in Neuro.SigmoidMode.entries) {
            val network = Neuro(intArrayOf(2, 1), Neuro.HyperParameters(0.6, 0.2, 1.4, 42, Neuro.Kernel.AUTO, mode))
            NeuroLearningSets.addTo(network, NeuroLearningSets.create(NeuroLearningSets.Kind.XOR, 42))
            network.train(3)
            val snapshot = NeuroXorDiagnostics.capture(network, 3, network.trainingError())
            assertEquals(0, snapshot.hiddenCount()); assertEquals(0, snapshot.hiddenLayerCount())
            assertEquals(1, snapshot.layerCount()); assertEquals(3, snapshot.parameterCount())
            assertEquals(0, snapshot.parameterOffset(0))
            for ((x, y) in listOf(0.0 to 0.0, 0.0 to 1.0, 1.0 to 0.0, 1.0 to 1.0, 0.31 to 0.77)) {
                val probe = NeuroXorDiagnostics.probe(snapshot, x, y)
                assertEquals(network.predict(doubleArrayOf(x, y))[0], probe.output(), 1e-12)
                assertTrue(probe.hiddenLayers().isEmpty()); assertTrue(probe.hidden().isEmpty())
                assertArrayEquals(doubleArrayOf(x * snapshot.weight(0, 0, 0), y * snapshot.weight(0, 0, 1)), probe.contributions())
                assertEquals(snapshot.outputBias() + probe.contributions().sum(), probe.outputPreActivation(), 1e-12)
                val original = probe.contributions(); probe.contributions()[0] = 100.0
                assertArrayEquals(original, probe.contributions())
            }
            val before = NeuroXorDiagnostics.probe(snapshot, 0.31, 0.77).output()
            snapshot.topology()[0] = 99; snapshot.parameters()[0] = 99.0
            network.train(20)
            assertEquals(before, NeuroXorDiagnostics.probe(snapshot, 0.31, 0.77).output(), 0.0)
            assertTrue(NeuroXorDiagnostics.renderHiddenMaps(snapshot, 5).isEmpty())
            val image = NeuroXorDiagnostics.renderOutputMap(snapshot, 5)
            assertEquals(NeuroXorGrid.grayRgb(NeuroXorDiagnostics.probe(snapshot, 0.0, 1.0).output()), image.getRGB(0, 0) and 0xffffff)
            val after = NeuroXorDiagnostics.capture(network, 23, network.trainingError())
            assertEquals(5, NeuroXorDiagnostics.renderDifferenceMap(snapshot, after, 5).width)
            assertThrows(IllegalArgumentException::class.java) { NeuroXorDiagnostics.boundary(snapshot, 0) }
            assertThrows(IllegalArgumentException::class.java) { NeuroXorDiagnostics.renderHiddenMaps(snapshot, 5, 0, 0, 1) }
            for (size in listOf(1, 1025))
                assertThrows(IllegalArgumentException::class.java) { NeuroXorDiagnostics.renderHiddenMaps(snapshot, size) }
        }
    }

    @Test fun baselineSupportsStepsSeedStudiesAndAtomicTopologyResets() {
        val config = StudioConfig(hidden = "", maxEpochs = 10, targetError = 1e-12)
        val studio = NeuroStudio(config)
        val initial = studio.frame()
        assertTrue(initial.hiddenImages.isEmpty()); assertEquals(3, initial.history.single().parameterCount)
        assertEquals(2, initial.history.single().normCount)
        assertThrows(IllegalArgumentException::class.java) { studio.selectHidden(0) }
        studio.step(1); assertEquals(1, studio.advance(10))
        val stepped = studio.frame()
        assertEquals(1, stepped.diagnostics.epoch()); assertEquals(0, stepped.beforeEpoch)
        val parameters = stepped.diagnostics.parameters()
        assertEquals(4, studio.compareSeeds(2).size)
        assertArrayEquals(parameters, studio.frame().diagnostics.parameters())
        assertEquals(1, studio.epochs)
        studio.apply(StudioConfig("3,2", targetError = 1e-12))
        studio.selectHidden(1, 1)
        assertEquals(1, studio.frame().hiddenImages.size)
        studio.apply(config)
        val reset = studio.frame()
        assertEquals(0, reset.diagnostics.epoch()); assertEquals(1, reset.history.size)
        assertEquals(1, reset.checkpoints.size); assertTrue(reset.seeds.isEmpty())
        assertTrue(reset.hiddenImages.isEmpty()); assertEquals(0, reset.hiddenLayer)
        assertArrayEquals(initial.diagnostics.parameters(), reset.diagnostics.parameters())
        studio.setRunning(true); assertEquals(10, studio.advance(100))
        assertEquals(StudioState.LIMIT_REACHED, studio.state)
        assertEquals(10, studio.frame().checkpoints.last().epoch)
        studio.apply(config.copy(dataset = NeuroLearningSets.Kind.CUSTOM))
        assertEquals(StudioState.EMPTY, studio.frame().state)
        assertTrue(studio.frame().hiddenImages.isEmpty())
        studio.addSample(0.3, 0.7, 1.0); studio.step(1)
        assertEquals(1, studio.advance()); assertTrue(studio.frame().hiddenImages.isEmpty())
    }

    @Test fun baselineRendersAllViewsAndLayerControlsRecover() {
        EventQueue.invokeAndWait {
            val constructor = NeuroXorCanvas::class.java.getDeclaredConstructor(StudioFrame::class.java, Boolean::class.javaPrimitiveType)
            constructor.isAccessible = true
            val ui = constructor.newInstance(NeuroStudio(StudioConfig(hidden = "")).frame(), false)
            try {
                val root = field(ui, "root") as JPanel
                val tabs = field(ui, "tabs") as JTabbedPane
                assertFalse((field(ui, "hiddenLayer") as JComboBox<*>).isEnabled)
                assertFalse((field(ui, "neuronPage") as JSpinner).isEnabled)
                for (index in 0 until tabs.tabCount) {
                    tabs.selectedIndex = index; root.setSize(1520, 1060); layout(root)
                    val image = BufferedImage(1520, 1060, BufferedImage.TYPE_INT_RGB)
                    val graphics = image.createGraphics()
                    try { root.printAll(graphics) } finally { graphics.dispose() }
                }
                buttons(root).first { it.text == "+ Layer" }.doClick()
                assertEquals("6", (field(ui, "hidden") as JTextField).text)
                buttons(root).first { it.text == "− Layer" }.doClick()
                assertEquals("", (field(ui, "hidden") as JTextField).text)
                val next = NeuroStudio(StudioConfig("3,2")).frame()
                ui.javaClass.getDeclaredField("published").apply { isAccessible = true }.set(ui, next)
                ui.javaClass.getDeclaredMethod("refresh").apply { isAccessible = true }.invoke(ui)
                assertTrue((field(ui, "hiddenLayer") as JComboBox<*>).isEnabled)
                assertTrue((field(ui, "neuronPage") as JSpinner).isEnabled)
            } finally { ui.close() }
        }
    }

    private fun field(instance: Any, name: String): Any? = instance.javaClass.getDeclaredField(name).run { isAccessible = true; get(instance) }
    private fun layout(component: Container) {
        component.doLayout()
        component.components.filterIsInstance<Container>().forEach { layout(it) }
    }
    private fun buttons(component: Container): List<JButton> = component.components.flatMap {
        when (it) { is JButton -> listOf(it); is Container -> buttons(it); else -> emptyList() }
    }
}
