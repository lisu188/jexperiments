package com.lis.neuro

import java.awt.image.BufferedImage
import java.awt.image.DataBufferInt

internal object NeuroXorGrid {
    fun createInputs(size: Int): DoubleArray {
        require(size >= 2) { "size must be >= 2" }
        val inputs = DoubleArray(Math.multiplyExact(Math.multiplyExact(size, size), 2))
        val scale = 1.0 / (size - 1)
        for (row in 0 until size) {
            for (column in 0 until size) {
                val index = (row * size + column) * 2
                inputs[index] = column * scale
                inputs[index + 1] = 1.0 - row * scale
            }
        }
        return inputs
    }
    fun grayRgb(output: Double): Int {
        require(output.isFinite()) { "output must be finite" }
        val gray = Math.round(output.coerceIn(0.0, 1.0) * 255.0).toInt()
        return (gray shl 16) or (gray shl 8) or gray
    }
    fun render(network: Neuro, size: Int, inputs: DoubleArray, outputs: DoubleArray): BufferedImage {
        require(size >= 2) { "size must be >= 2" }
        val samples = Math.multiplyExact(size, size)
        require(inputs.size == Math.multiplyExact(samples, 2)) { "inputs length must equal size * size * 2" }
        require(outputs.size == samples) { "outputs length must equal size * size" }
        network.predictBatch(inputs, samples, outputs)
        val image = BufferedImage(size, size, BufferedImage.TYPE_INT_RGB)
        val pixels = (image.raster.dataBuffer as DataBufferInt).data
        for (index in pixels.indices) pixels[index] = grayRgb(outputs[index])
        return image
    }
}
