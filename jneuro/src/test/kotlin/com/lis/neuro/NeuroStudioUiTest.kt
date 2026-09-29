package com.lis.neuro

import java.awt.Container
import java.awt.EventQueue
import java.nio.file.Files
import java.nio.file.Path
import javax.imageio.ImageIO
import javax.swing.JButton
import javax.swing.JLabel
import javax.swing.JPanel
import javax.swing.JTextField
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir

class NeuroStudioUiTest {
    @TempDir lateinit var directory: Path

    @Test fun rendersEveryImplementedViewAtDesktopSize() {
        NeuroXorCanvas.main(arrayOf("--screenshots=${directory.toAbsolutePath()}"))
        for (name in listOf("overview", "neurons", "data", "update", "parameters", "seeds", "timeline", "search")) {
            val path = directory.resolve("$name.png")
            assertTrue(Files.size(path) > 10_000, name)
            val image = ImageIO.read(path.toFile())
            assertEquals(1520, image.width); assertEquals(1060, image.height)
            assertNotEquals(image.getRGB(1, 1), image.getRGB(300, 1), "Sidebar and workspace should have distinct surfaces")
        }
    }

    @Test fun controlsDriveWorkerWithoutBlockingTheEventThread() {
        var ui: NeuroXorCanvas? = null
        EventQueue.invokeAndWait {
            val constructor = NeuroXorCanvas::class.java.getDeclaredConstructor(StudioFrame::class.java, Boolean::class.javaPrimitiveType)
            constructor.isAccessible = true
            ui = constructor.newInstance(null, true)
        }
        val studioUi = ui!!
        try {
            await { (field(studioUi, "published") as StudioFrame?)?.state == StudioState.READY }
            EventQueue.invokeAndWait { (field(studioUi, "step") as JButton).doClick() }
            await { (field(studioUi, "published") as StudioFrame).diagnostics.epoch() == 1 }
            EventQueue.invokeAndWait { (field(studioUi, "reset") as JButton).doClick() }
            await { (field(studioUi, "published") as StudioFrame).state == StudioState.READY }
            EventQueue.invokeAndWait {
                (field(studioUi, "hidden") as JTextField).text = "2,"
                buttons(field(studioUi, "root") as JPanel).first { it.text == "Apply & restart" }.doClick()
                assertTrue((field(studioUi, "configError") as JLabel).text.contains("Layer"))
                (field(studioUi, "hidden") as JTextField).text = "2,2"
                (field(studioUi, "seed") as JTextField).text = "123"
                buttons(field(studioUi, "root") as JPanel).first { it.text == "Apply & restart" }.doClick()
            }
            await { (field(studioUi, "published") as StudioFrame).config.hidden == "2,2" }
            val active = field(studioUi, "published") as StudioFrame
            assertArrayEquals(intArrayOf(2, 2, 2, 1), active.diagnostics.topology())
            assertEquals(123L, active.config.seed)
            EventQueue.invokeAndWait { (field(studioUi, "reset") as JButton).doClick() }
            await { (field(studioUi, "published") as StudioFrame).state == StudioState.READY }
        } finally {
            EventQueue.invokeAndWait { studioUi.close() }
            val worker = field(studioUi, "thread") as Thread
            worker.join(3000)
            assertFalse(worker.isAlive, "Closing must stop the worker")
        }
    }

    @Test fun architectureSearchRunsOffEdtAndAppliesOrReplaysOnlyOnRequest() {
        var ui: NeuroXorCanvas? = null
        EventQueue.invokeAndWait {
            val constructor = NeuroXorCanvas::class.java.getDeclaredConstructor(StudioFrame::class.java, Boolean::class.javaPrimitiveType)
            constructor.isAccessible = true
            ui = constructor.newInstance(null, true)
        }
        val studioUi = ui!!
        try {
            await { (field(studioUi, "published") as StudioFrame?)?.state == StudioState.READY }
            awaitEdt { field(studioUi, "frame") != null }
            val panel = field(studioUi, "architectureSearch") as ArchitectureSearchPanel
            val original = field(studioUi, "published") as StudioFrame
            EventQueue.invokeAndWait {
                (field(panel, "maxLayers") as javax.swing.JSpinner).value = 1
                (field(panel, "maxWidth") as javax.swing.JSpinner).value = 3
                (field(panel, "epochs") as javax.swing.JSpinner).value = 150
                (field(panel, "target") as javax.swing.JSpinner).value = 0.9
                buttons(field(studioUi, "root") as JPanel).first { it.text == "Auto search…" }.doClick()
                buttons(panel).first { it.text == "Start search" }.doClick()
                (field(studioUi, "tabs") as javax.swing.JTabbedPane).selectedIndex = 0
            }
            awaitEdt { field(panel, "result") != null }
            val unchanged = field(studioUi, "published") as StudioFrame
            assertArrayEquals(original.diagnostics.parameters(), unchanged.diagnostics.parameters())
            assertEquals(0, unchanged.diagnostics.epoch())
            var report: ArchitectureSearchResult? = null
            EventQueue.invokeAndWait {
                report = field(panel, "result") as ArchitectureSearchResult
                assertEquals(3, report.evaluated)
                buttons(panel).first { it.text == "Replay selected run" }.doClick()
            }
            await { (field(studioUi, "published") as StudioFrame).replayNote.isNotEmpty() }
            val replay = field(studioUi, "published") as StudioFrame
            val winner = report!!.selection.recommended!!
            assertArrayEquals(winner.architecture.topology(), replay.diagnostics.topology())
            assertEquals(winner.representative!!.bestEpoch, replay.diagnostics.epoch())
            EventQueue.invokeAndWait {
                (field(studioUi, "reset") as JButton).doClick()
            }
            await { (field(studioUi, "published") as StudioFrame).diagnostics.epoch() == 0 }
            EventQueue.invokeAndWait {
                (field(panel, "maxLayers") as javax.swing.JSpinner).value = 1
                (field(panel, "maxWidth") as javax.swing.JSpinner).value = 2
                (field(panel, "epochs") as javax.swing.JSpinner).value = 100
                buttons(panel).first { it.text == "Start search" }.doClick()
            }
            awaitEdt { field(panel, "result") != null }
            EventQueue.invokeAndWait { buttons(panel).first { it.text == "Apply architecture" }.doClick() }
            await { (field(studioUi, "published") as StudioFrame).config.maxEpochs == 100 }
            assertEquals(0, (field(studioUi, "published") as StudioFrame).diagnostics.epoch())
            EventQueue.invokeAndWait {
                (field(panel, "epochs") as javax.swing.JSpinner).value = 1_000_000
                buttons(panel).first { it.text == "Start search" }.doClick()
                (field(studioUi, "reset") as JButton).doClick()
            }
            awaitEdt { field(studioUi, "searchRunning") == false }
            EventQueue.invokeAndWait { assertNull(field(panel, "result")) }
        } finally {
            EventQueue.invokeAndWait { studioUi.close() }
            (field(studioUi, "thread") as Thread).join(5000)
            assertFalse((field(studioUi, "thread") as Thread).isAlive)
        }
    }

    private fun awaitEdt(condition: () -> Boolean) {
        await {
            var done = false
            EventQueue.invokeAndWait { done = condition() }
            done
        }
    }

    private fun field(instance: Any, name: String): Any? = instance.javaClass.getDeclaredField(name).run { isAccessible = true; get(instance) }
    private fun buttons(component: Container): List<JButton> = component.components.flatMap {
        when (it) { is JButton -> listOf(it); is Container -> buttons(it); else -> emptyList() }
    }
    private fun await(condition: () -> Boolean) {
        val deadline = System.nanoTime() + 5_000_000_000L
        while (!condition() && System.nanoTime() < deadline) Thread.sleep(10)
        assertTrue(condition(), "Timed out waiting for a published model state")
    }
}
