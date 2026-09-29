package com.lis.neuro

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

class NeuroXorDiagnosticsTest {
    @Test fun capturesAllLayersAndMatchesTheActualKernel() {
        for (shape in listOf(intArrayOf(2, 6, 1), intArrayOf(2, 4, 3, 2, 1), intArrayOf(2, 32, 16, 1))) {
            for (mode in Neuro.SigmoidMode.entries) {
                val network = Neuro(shape, Neuro.HyperParameters(0.6, 0.2, 1.4, 42, Neuro.Kernel.AUTO, mode))
                NeuroLearningSets.addTo(network, NeuroLearningSets.create(NeuroLearningSets.Kind.XOR, 42))
                network.train(20)
                val snapshot = NeuroXorDiagnostics.capture(network, 20, network.trainingError())
                val probe = NeuroXorDiagnostics.probe(snapshot, 0.31, 0.77)
                assertArrayEquals(shape, snapshot.topology())
                assertEquals(20, snapshot.epoch()); assertEquals(network.trainingError(), snapshot.error(), 0.0)
                assertEquals(shape.size - 2, snapshot.hiddenLayerCount()); assertEquals(shape[1], snapshot.hiddenCount())
                assertEquals(shape.size - 1, snapshot.layerCount()); assertEquals(network.parameterCount(), snapshot.parameterCount())
                assertEquals(network.predict(doubleArrayOf(0.31, 0.77))[0], probe.output(), 1e-12)
                val hidden = probe.hiddenLayers()
                for (layer in hidden.indices) assertEquals(shape[layer + 1], hidden[layer].size)
                assertArrayEquals(hidden.last(), probe.hidden())
                assertArrayEquals(hidden.first(), probe.hiddenLayer(0))
                assertEquals(snapshot.outputBias() + probe.contributions().sum(), probe.outputPreActivation(), 1e-12)
                var expectedOffset = 0
                val flat = snapshot.parameters()
                for (layer in 0 until snapshot.layerCount()) {
                    assertEquals(expectedOffset, snapshot.parameterOffset(layer))
                    val weights = network.backendWeights(layer); val biases = network.backendBiases(layer)
                    assertEquals(shape[layer], snapshot.layerInputCount(layer)); assertEquals(shape[layer + 1], snapshot.layerOutputCount(layer))
                    for (output in biases.indices) {
                        assertEquals(biases[output], snapshot.bias(layer, output), 0.0)
                        for (input in 0 until shape[layer]) assertEquals(weights[output * shape[layer] + input], snapshot.weight(layer, output, input), 0.0)
                    }
                    assertArrayEquals(weights, flat.copyOfRange(expectedOffset, expectedOffset + weights.size))
                    expectedOffset += weights.size
                    assertArrayEquals(biases, flat.copyOfRange(expectedOffset, expectedOffset + biases.size))
                    expectedOffset += biases.size
                    assertEquals(Math.sqrt(weights.sumOf { it * it }), NeuroXorDiagnostics.weightNorm(snapshot, layer), 1e-12)
                    assertEquals(Math.sqrt(biases.sumOf { it * it }), NeuroXorDiagnostics.biasNorm(snapshot, layer), 1e-12)
                }
                assertEquals((0 until snapshot.layerCount()).maxOf { network.backendWeights(it).maxOf { weight -> kotlin.math.abs(weight) } }, NeuroXorDiagnostics.maxAbsWeight(snapshot))
            }
        }
    }

    @Test fun snapshotsAndProbesOwnTheirArrays() {
        val network = NeuroTest.xor()
        val snapshot = NeuroXorDiagnostics.capture(network, 0, network.trainingError())
        val before = NeuroXorDiagnostics.probe(snapshot, 0.5, 0.5)
        val original = before.hiddenLayers()
        before.hiddenLayers()[0][0] = -100.0
        before.hiddenLayer(0)[0] = -100.0
        before.hidden()[0] = -100.0
        before.contributions()[0] = -100.0
        snapshot.topology()[1] = 99
        snapshot.parameters()[0] = 999.0
        network.train(200)
        val after = NeuroXorDiagnostics.probe(snapshot, 0.5, 0.5)
        assertEquals(before.output(), after.output(), 0.0)
        assertArrayEquals(original[0], after.hiddenLayers()[0])
        assertArrayEquals(before.contributions(), after.contributions())
        assertEquals(6, snapshot.hiddenCount())
        val shape = intArrayOf(2, 1, 1)
        val weights = arrayOf(doubleArrayOf(0.2, 0.1), doubleArrayOf(0.4)); val biases = arrayOf(doubleArrayOf(0.0), doubleArrayOf(0.0))
        val copied = NeuroXorDiagnostics.Snapshot(1, 0.5, 1.0, Neuro.SigmoidMode.EXACT, shape, weights, biases)
        shape[1] = 12; weights[0][0] = 99.0; biases[0][0] = 88.0
        assertEquals(0.2, copied.weight(0, 0, 0), 0.0); assertEquals(0.0, copied.bias(0, 0), 0.0)
    }

    @Test fun rendersPagedHiddenOutputAndDifferenceMaps() {
        val model = Neuro(intArrayOf(2, 16, 4, 1))
        val before = NeuroXorDiagnostics.capture(model, 0, 0.5)
        val first = NeuroXorDiagnostics.renderHiddenMaps(before, 5)
        assertEquals(12, first.size)
        val page = NeuroXorDiagnostics.renderHiddenMaps(before, 5, 1, 1, 3)
        assertEquals(3, page.size)
        val probe = NeuroXorDiagnostics.probe(before, 0.0, 1.0)
        for (neuron in page.indices) assertEquals(NeuroXorGrid.grayRgb(probe.hiddenLayer(1)[neuron + 1]), page[neuron].getRGB(0, 0) and 0xffffff)
        for (image in first) { assertEquals(5, image.width); assertEquals(5, image.height) }
        NeuroLearningSets.addTo(model, NeuroLearningSets.create(NeuroLearningSets.Kind.XOR, 0)); model.train(10)
        val after = NeuroXorDiagnostics.capture(model, 10, model.trainingError())
        val image = NeuroXorDiagnostics.renderOutputMap(after, 6)
        val difference = NeuroXorDiagnostics.renderDifferenceMap(before, after, 6)
        assertEquals(6, image.width); assertEquals(6, difference.height)
        val p = NeuroXorDiagnostics.probe(after, 0.0, 1.0).output()
        assertEquals(NeuroXorGrid.grayRgb(p), image.getRGB(0, 0) and 0xffffff)
        assertEquals(NeuroXorDiagnostics.differenceRgb(p - probe.output()), difference.getRGB(0, 0) and 0xffffff)
        val neutral = NeuroXorDiagnostics.differenceRgb(0.0)
        assertEquals(neutral shr 16 and 255, neutral and 255)
        val positive = NeuroXorDiagnostics.differenceRgb(0.2); val negative = NeuroXorDiagnostics.differenceRgb(-0.2)
        assertTrue((positive shr 16 and 255) > (positive and 255))
        assertTrue((negative and 255) > (negative shr 16 and 255))
        assertEquals(NeuroXorDiagnostics.differenceRgb(1.0), NeuroXorDiagnostics.differenceRgb(10.0))
    }

    @Test fun clipsFirstLayerBoundariesIncludingDegenerateCases() {
        fun snapshot(wx: Double, wy: Double, bias: Double) = NeuroXorDiagnostics.Snapshot(0, 0.5, 1.0, Neuro.SigmoidMode.EXACT,
            intArrayOf(2, 1, 1), arrayOf(doubleArrayOf(wx, wy), doubleArrayOf(1.0)), arrayOf(doubleArrayOf(bias), doubleArrayOf(0.0)))
        for ((wx, wy, bias) in listOf(Triple(1.0, 0.0, -0.5), Triple(0.0, 1.0, -0.5), Triple(1.0, -1.0, 0.0), Triple(1.0, 1.0, -1.0))) {
            val value = snapshot(wx, wy, bias)
            val line = NeuroXorDiagnostics.boundary(value, 0)!!
            for ((x, y) in listOf(line.x1 to line.y1, line.x2 to line.y2)) {
                assertTrue(x in 0.0..1.0 && y in 0.0..1.0)
                assertEquals(0.0, wx * x + wy * y + bias, 1e-9)
            }
        }
        assertNull(NeuroXorDiagnostics.boundary(snapshot(0.0, 0.0, 0.0), 0))
        assertNull(NeuroXorDiagnostics.boundary(snapshot(1.0, 1.0, 4.0), 0))
        assertNull(NeuroXorDiagnostics.boundary(snapshot(1.0, 1.0, 0.0), 0))
    }

    @Test fun validatesDiagnosticArguments() {
        val valid = NeuroXorDiagnostics.capture(NeuroTest.xor(), 0, 0.5)
        for (shape in listOf(intArrayOf(3, 2, 1), intArrayOf(2, 2, 2)))
            assertThrows(IllegalArgumentException::class.java) { NeuroXorDiagnostics.capture(Neuro(shape), 0, 0.5) }
        for (size in listOf(1, 1025)) {
            assertThrows(IllegalArgumentException::class.java) { NeuroXorDiagnostics.renderOutputMap(valid, size) }
            assertThrows(IllegalArgumentException::class.java) { NeuroXorDiagnostics.renderHiddenMaps(valid, size) }
            assertThrows(IllegalArgumentException::class.java) { NeuroXorDiagnostics.renderDifferenceMap(valid, valid, size) }
        }
        assertThrows(IllegalArgumentException::class.java) { NeuroXorDiagnostics.probe(valid, Double.NaN, 0.5) }
        assertThrows(IllegalArgumentException::class.java) { NeuroXorDiagnostics.probe(valid, 0.5, Double.POSITIVE_INFINITY) }
        assertThrows(IllegalArgumentException::class.java) { NeuroXorDiagnostics.differenceRgb(Double.NaN) }
        assertThrows(IllegalArgumentException::class.java) { NeuroXorDiagnostics.renderHiddenMaps(valid, 4, 3, 0, 1) }
        assertThrows(IllegalArgumentException::class.java) { NeuroXorDiagnostics.renderHiddenMaps(valid, 4, 0, -1, 1) }
        assertThrows(IllegalArgumentException::class.java) { NeuroXorDiagnostics.renderHiddenMaps(valid, 4, 0, 0, 10) }
        for (layer in listOf(-1, 3)) {
            assertThrows(IllegalArgumentException::class.java) { NeuroXorDiagnostics.weightNorm(valid, layer) }
            assertThrows(IllegalArgumentException::class.java) { NeuroXorDiagnostics.biasNorm(valid, layer) }
            assertThrows(IllegalArgumentException::class.java) { valid.parameterOffset(layer) }
        }
        assertThrows(IllegalArgumentException::class.java) { NeuroXorDiagnostics.boundary(valid, -1) }
        assertThrows(IllegalArgumentException::class.java) { NeuroXorDiagnostics.boundary(valid, valid.hiddenCount()) }
    }
}
