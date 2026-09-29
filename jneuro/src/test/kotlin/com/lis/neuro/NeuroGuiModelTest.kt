package com.lis.neuro

import java.awt.EventQueue
import java.awt.Rectangle
import java.text.ParseException
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import javax.swing.JSpinner
import javax.swing.SpinnerNumberModel
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

class NeuroGuiModelTest {
    @Test fun acceptsUncappedConfigurationButPreservesMathematicalConstraints() {
        assertEquals(9, NeuroTopologyConfig.parseHidden("129,1,1,1,1,1,1,1,1").size)
        assertEquals(129, NeuroTopologyConfig.topology(intArrayOf(129))[1])
        assertEquals(0.0, StudioConfig(maxEpochs = Int.MAX_VALUE, targetError = 0.0, learningRate = 25.0).targetError)
        val config = ArchitectureSearchConfig(maxLayers = 9, maxWidth = 129, maxParameters = 1_000_001,
            seeds = (1L..21).toList(), maxEpochs = Int.MAX_VALUE, checkEvery = Int.MAX_VALUE, parallelism = 33,
            requiredSuccesses = 21, maxTrials = 20_001, timeLimitSeconds = Long.MAX_VALUE, restartAfter = 1001, maxRestarts = 101)
        assertEquals(5, config.startingArchitecture().parameters)
        assertEquals(19_992, config.plannedTrials())
        assertThrows(IllegalArgumentException::class.java) { StudioConfig(momentum = 1.0) }
        assertThrows(IllegalArgumentException::class.java) { StudioConfig(targetError = -0.01) }
        val samples = NeuroLearningSets.create(NeuroLearningSets.Kind.CIRCLE, 123)
        for (fraction in listOf(0.01, 0.8, 0.99)) {
            val data = ArchitectureSearchData.split(samples, fraction)
            assertTrue(data.training.isNotEmpty()); assertTrue(data.validation.isNotEmpty())
        }
        for (fraction in listOf(0.0,1.0,Double.NaN)) assertThrows(IllegalArgumentException::class.java) { ArchitectureSearchData.split(samples,fraction) }
    }

    @Test fun numericEditorsAreUnboundedLosslessAndNeverWrapAtTypeLimits() {
        EventQueue.invokeAndWait {
            for ((value,step) in listOf<Pair<Number,Number>>(10 to 1, 0L to 60L, 0.6 to 0.05)) {
                val spinner = NumericInputs.spinner(value,step)
                val model = spinner.model as SpinnerNumberModel
                assertNull(model.minimum); assertNull(model.maximum)
                val field = (spinner.editor as JSpinner.DefaultEditor).textField
                val next = if (value is Double) "0.000000000123456789" else "1000001"
                field.text = next; spinner.commitEdit()
                assertEquals(next.toDouble(), (spinner.value as Number).toDouble())
                assertNotNull(model.nextValue); assertNotNull(model.previousValue)
                field.text = "nonsense"
                assertThrows(ParseException::class.java) { spinner.commitEdit() }
            }
            val ints = NumericInputs.spinner(Int.MAX_VALUE,1).model as SpinnerNumberModel
            assertNull(ints.nextValue); ints.value = Int.MIN_VALUE; assertNull(ints.previousValue)
            val longs = NumericInputs.spinner(Long.MAX_VALUE,1L).model as SpinnerNumberModel
            assertNull(longs.nextValue); longs.value = Long.MIN_VALUE; assertNull(longs.previousValue)
            val doubles = NumericInputs.spinner(Double.MAX_VALUE,Double.MAX_VALUE).model as SpinnerNumberModel
            assertNull(doubles.nextValue)
            assertThrows(ParseException::class.java) { NumericInputs.parse("2147483648",0) }
            assertThrows(ParseException::class.java) { NumericInputs.parse("9223372036854775808",0L) }
            for (value in listOf("NaN","Infinity","1e999")) assertThrows(ParseException::class.java) { NumericInputs.parse(value,0.0) }
        }
    }

    @Test fun topologyArithmeticAndLargeBudgetsRemainSafe() {
        assertEquals(3, NumericInputs.parameterCount(intArrayOf(2,1)))
        assertThrows(IllegalArgumentException::class.java) { NumericInputs.parameterCount(intArrayOf(2,Int.MAX_VALUE,1)) }
        assertThrows(IllegalArgumentException::class.java) { NumericInputs.parameterCount(intArrayOf(2,0,1)) }
        assertThrows(IllegalArgumentException::class.java) { NumericInputs.checkAllocation(intArrayOf(2,Int.MAX_VALUE,1)) }
        assertThrows(IllegalArgumentException::class.java) { ArchitectureSearchConfig(minLayers = Int.MAX_VALUE, maxLayers = Int.MAX_VALUE, maxWidth = 1).minimumArchitecture() }
        val config = ArchitectureSearchConfig(maxLayers = Int.MAX_VALUE, maxWidth = Int.MAX_VALUE, maxRestarts = 0, maxTrials = 5)
        assertEquals(5, config.plannedTrials())
        val planner = AdaptiveArchitecturePlanner(config)
        assertTrue(planner.mutations(planner.next()!!).all { it.architecture.parameters > 0 })
        val studio = NeuroStudio(StudioConfig(hidden="1", maxEpochs=Int.MAX_VALUE, targetError=0.0))
        studio.step(Int.MAX_VALUE); studio.step(Int.MAX_VALUE)
        assertEquals(2,studio.advance(2)); assertTrue(studio.hasWork)
        val data = ArchitectureSearchData.fitting(NeuroLearningSets.create(NeuroLearningSets.Kind.XOR,42))
        val result = NeuroArchitectureSearch().search(data,ArchitectureSearchConfig(maxLayers=1,maxWidth=1,seeds=listOf(42),requiredSuccesses=1,
            maxEpochs=1,checkEvery=1,maxTrials=1,timeLimitSeconds=Long.MAX_VALUE))
        assertNotEquals(ArchitectureTermination.TIME_LIMIT,result.termination)
    }

    @Test fun galleryIncludesEveryLayerAndNeuronWithoutPaging() {
        val snapshot = NeuroStudio(StudioConfig("16,16,129")).frame().diagnostics
        for (width in listOf(360,640,1200,1800)) {
            val layout = NeuronGallery.Layout(snapshot,width)
            assertEquals(listOf(16,16,129),layout.sections.map { it.neurons })
            val all = layout.visible(Rectangle(0,0,width,layout.height))
            assertEquals(161,all.size)
            assertEquals(161,all.map { it.layer to it.neuron }.toSet().size)
            for (cell in all) assertTrue(cell.bounds.y + cell.bounds.height <= layout.height)
            val last = layout.sections.last()
            assertEquals(128,layout.visible(layout.bounds(last,128)).last().neuron)
            assertTrue(layout.visible(Rectangle(0,0,width,100)).isEmpty())
            assertTrue(layout.visible(Rectangle(0,layout.height+100,width,100)).isEmpty())
        }
        val baseline = NeuroStudio(StudioConfig("")).frame().diagnostics
        assertTrue(NeuronGallery.Layout(baseline,1000).sections.isEmpty())
    }

    @Test fun visibleMapsUseRealActivationsAndInvalidateOnNewSnapshots() {
        val studio = NeuroStudio(StudioConfig("3,2",targetError=0.0))
        val before = studio.frame().diagnostics
        val cells = NeuronGallery.Layout(before,640).visible(Rectangle(0,362,640,1000))
        NeuronGallery(true).use { gallery ->
            var notifications = 0
            gallery.request(before,cells) { notifications++ }
            assertEquals(1,notifications)
            for (cell in cells) {
                val actual = gallery.image(cell.layer,cell.neuron)!!
                val expected = NeuroXorDiagnostics.renderHiddenMaps(before,actual.width,cell.layer,cell.neuron,1)[0]
                assertEquals(expected.getRGB(10,10),actual.getRGB(10,10))
            }
            gallery.request(before,cells) { notifications++ }
            assertEquals(1,notifications)
            gallery.request(before,emptyList()) {}
            assertNull(gallery.image(0,0))
            studio.step(1); studio.advance(1)
            val after=studio.frame().diagnostics
            gallery.request(after,cells) {}
            assertNotNull(gallery.image(1,1))
            gallery.close(); gallery.request(after,cells) { fail<Unit>("Closed gallery must not publish.") }
            assertNull(gallery.image(1,1))
        }
    }

    @Test fun galleryRendersAsynchronouslyAndClosesItsWorker() {
        val snapshot=NeuroStudio(StudioConfig("16,16,16")).frame().diagnostics
        val layout=NeuronGallery.Layout(snapshot,800)
        val cells=layout.visible(layout.bounds(layout.sections.last(),15))
        val ready=CountDownLatch(1)
        NeuronGallery().use { gallery ->
            gallery.request(snapshot,cells) { ready.countDown() }
            assertTrue(ready.await(5,TimeUnit.SECONDS))
            assertNotNull(gallery.image(2,15))
        }
    }
}
