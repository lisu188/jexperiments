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
        for (name in listOf("overview", "neurons", "data", "update", "parameters", "seeds", "timeline")) {
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
