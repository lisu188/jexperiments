package com.lis.neuro

import java.awt.Container
import java.awt.EventQueue
import java.awt.image.BufferedImage
import javax.swing.JComboBox
import javax.swing.JPanel
import javax.swing.JTabbedPane
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.hypot
import kotlin.math.sin

class NeuroHardLearningSetsTest {
    @Test fun hardDatasetsAreSeededSpatiallyCoveredAndNonDegenerate() {
        for (kind in HARD_KINDS) {
            for (seed in listOf(0L, 1L, 42L, -1L, Long.MIN_VALUE, Long.MAX_VALUE)) {
                val samples = NeuroLearningSets.create(kind, seed)
                assertEquals(1024, samples.size, kind.name)
                assertEquals(samples, NeuroLearningSets.create(kind, seed), kind.name)
                assertEquals(1024, samples.map { it.x to it.y }.toSet().size)
                assertEquals(1024, samples.map { (it.x * 32).toInt() to (it.y * 32).toInt() }.toSet().size)
                assertEquals(setOf(0.0, 1.0), samples.map { it.target }.toSet())
                assertTrue(samples.count { it.target == 1.0 } in 200..824, kind.name)
                assertTrue(samples.all { it.x > 0.0 && it.x < 1.0 && it.y > 0.0 && it.y < 1.0 })
                assertTrue(samples.all { it.x.isFinite() && it.y.isFinite() && it.target.isFinite() })
            }
            assertNotEquals(NeuroLearningSets.create(kind, 1), NeuroLearningSets.create(kind, 2), kind.name)
        }
    }

    @Test fun checkerboardLabelsAlternateInEveryCellAndEveryIslandIsRepresented() {
        val board = NeuroLearningSets.create(NeuroLearningSets.Kind.CHECKERBOARD, 42)
        val cells = board.groupBy { (it.x * 8).toInt() to (it.y * 8).toInt() }
        assertEquals(64, cells.size)
        assertEquals(512, board.count { it.target == 1.0 })
        for ((cell, samples) in cells) {
            assertEquals(16, samples.size)
            assertEquals(setOf(((cell.first + cell.second) % 2).toDouble()), samples.map { it.target }.toSet())
        }
        val islands = NeuroLearningSets.create(NeuroLearningSets.Kind.ISLANDS, 42)
        val regions = islands.groupBy { (it.x * 4).toInt() to (it.y * 4).toInt() }
        assertEquals(16, regions.size)
        for ((cell, samples) in regions) {
            assertEquals(setOf(0.0, 1.0), samples.map { it.target }.toSet())
            val cx = (cell.first + 0.5) / 4
            val cy = (cell.second + 0.5) / 4
            for (sample in samples) {
                val expected = if (hypot(sample.x - cx, sample.y - cy) < 0.075) 1.0 else 0.0
                assertEquals(expected, sample.target)
            }
        }
    }

    @Test fun radialAndTwistingLabelsMatchIndependentGeometricOracles() {
        val rings = NeuroLearningSets.create(NeuroLearningSets.Kind.CONCENTRIC_RINGS, 42)
        for (sample in rings) {
            val radius = hypot(sample.x - 0.5, sample.y - 0.5)
            val positive = (0..2).any { band -> radius >= band / 6.0 && radius < band / 6.0 + 1.0 / 12 }
            assertEquals(if (positive) 1.0 else 0.0, sample.target)
        }
        assertTrue(rings.any { hypot(it.x - 0.5, it.y - 0.5) > 0.5 })
        for (kind in listOf(NeuroLearningSets.Kind.TIGHT_SPIRAL, NeuroLearningSets.Kind.PINWHEEL)) {
            for (sample in NeuroLearningSets.create(kind, 42)) {
                val x = sample.x - 0.5
                val y = sample.y - 0.5
                val r = hypot(x, y)
                val phase: Double
                val real: Double
                val imaginary: Double
                if (kind == NeuroLearningSets.Kind.TIGHT_SPIRAL) {
                    phase = 16 * Math.PI * r
                    real = x
                    imaginary = y
                } else {
                    phase = 12 * Math.PI * r
                    real = x * x * x * x - 6 * x * x * y * y + y * y * y * y
                    imaginary = 4 * x * y * (x * x - y * y)
                }
                val rotated = imaginary * cos(phase) - real * sin(phase)
                assertTrue(abs(rotated) > 1e-14, "Oracle must not rely on a boundary tie")
                assertEquals(if (rotated >= 0) 1.0 else 0.0, sample.target, kind.name)
            }
        }
    }

    @Test fun interferenceLabelsFollowTheSignsOfBothWarpedWavePhases() {
        fun positivePhase(turns: Double): Boolean = turns - floor(turns) < 0.5
        for (sample in NeuroLearningSets.create(NeuroLearningSets.Kind.INTERFERENCE, 42)) {
            val horizontal = 5 * sample.x + sin(4 * Math.PI * sample.y) / Math.PI
            val vertical = 5 * sample.y + sin(4 * Math.PI * sample.x) / Math.PI
            assertEquals(if (positivePhase(horizontal) == positivePhase(vertical)) 1.0 else 0.0, sample.target)
        }
    }

    @Test fun everyHardDatasetTrainsAndSupportsLeakFreeValidationSearch() {
        for (kind in HARD_KINDS) {
            val config = StudioConfig(hidden = "4,3", dataset = kind, maxEpochs = 2)
            val studio = NeuroStudio(config)
            assertEquals(1024, studio.frame().samples.size)
            studio.step(2)
            assertEquals(2, studio.advance(2))
            val frame = studio.frame()
            assertEquals(StudioState.LIMIT_REACHED, frame.state)
            assertTrue(frame.diagnostics.error().isFinite(), kind.name)
            assertEquals(kind, frame.config.dataset)
            val data = studio.searchData(ArchitectureEvaluation.VALIDATION)
            val again = studio.searchData(ArchitectureEvaluation.VALIDATION)
            assertEquals(data.fingerprint, again.fingerprint)
            assertEquals(1024, data.training.size + data.validation.size)
            val training = data.training.map { it.x to it.y }.toSet()
            val validation = data.validation.map { it.x to it.y }.toSet()
            assertTrue(training.intersect(validation).isEmpty())
            assertEquals(setOf(0.0, 1.0), data.validation.map { it.target }.toSet())
            val result = NeuroArchitectureSearch().search(data, ArchitectureSearchConfig(
                maxLayers = 1, maxWidth = 2, seeds = listOf(1L, 42L), maxEpochs = 2,
                checkEvery = 1, requiredSuccesses = 1, parallelism = 2))
            assertEquals(ArchitectureTermination.COMPLETED, result.termination)
            assertEquals(2, result.evaluated)
            assertTrue(result.candidates.all { it.valid && it.medianRmse.isFinite() })
            assertEquals(2, studio.epochs, "Searching must not train the Studio model")
        }
    }

    @Test fun hardExamplesAreAvailableInTheRealStudioAndDefaultToValidationSearch() {
        val constructor = NeuroXorCanvas::class.java.getDeclaredConstructor(StudioFrame::class.java, Boolean::class.javaPrimitiveType)
        constructor.isAccessible = true
        for (kind in HARD_KINDS) {
            val config = StudioConfig(dataset = kind)
            val snapshot = NeuroStudio(config).frame()
            EventQueue.invokeAndWait {
                constructor.newInstance(snapshot, false).use { ui ->
                    val datasets = field(ui, "dataset") as JComboBox<*>
                    assertTrue((0 until datasets.itemCount).map { datasets.getItemAt(it) }.containsAll(HARD_KINDS))
                    assertEquals(kind, datasets.selectedItem)
                    val search = field(ui, "architectureSearch") as ArchitectureSearchPanel
                    assertEquals(ArchitectureEvaluation.VALIDATION, (field(search, "evaluation") as JComboBox<*>).selectedItem)
                    val root = field(ui, "root") as JPanel
                    root.setSize(1520, 1060)
                    val tabs = field(ui, "tabs") as JTabbedPane
                    tabs.selectedIndex = tabs.indexOfTab("Learning set")
                    layout(root)
                    val image = BufferedImage(1520, 1060, BufferedImage.TYPE_INT_RGB)
                    val graphics = image.createGraphics()
                    try { root.printAll(graphics) } finally { graphics.dispose() }
                    assertNotEquals(image.getRGB(1, 1), image.getRGB(300, 1))
                }
            }
        }
    }

    private fun field(instance: Any, name: String): Any = instance.javaClass.getDeclaredField(name).run { isAccessible = true; get(instance) }
    private fun layout(component: Container) {
        component.doLayout()
        for (child in component.components) if (child is Container) layout(child)
    }

    companion object {
        private val HARD_KINDS = listOf(NeuroLearningSets.Kind.CHECKERBOARD, NeuroLearningSets.Kind.CONCENTRIC_RINGS,
            NeuroLearningSets.Kind.TIGHT_SPIRAL, NeuroLearningSets.Kind.PINWHEEL, NeuroLearningSets.Kind.INTERFERENCE,
            NeuroLearningSets.Kind.ISLANDS)
    }
}
