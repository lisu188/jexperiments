package com.lis.neuro

import java.awt.Rectangle
import java.awt.image.BufferedImage
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.ThreadPoolExecutor
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicLong

internal class NeuronGallery(private val synchronous: Boolean = false) : AutoCloseable {
    data class Cell(val layer: Int, val neuron: Int, val bounds: Rectangle)
    data class Section(val layer: Int, val neurons: Int, val top: Int, val rows: Int)
    class Layout(snapshot: NeuroXorDiagnostics.Snapshot, val width: Int) {
        val columns = maxOf(1, (width - 12) / 200)
        val cellWidth = maxOf(180, (width - 12) / columns - 12)
        val sections: List<Section>
        val height: Int
        init {
            var top = 362L
            sections = List(snapshot.hiddenLayerCount()) { layer ->
                val count = snapshot.layerOutputCount(layer)
                val rows = ((count.toLong() + columns - 1) / columns).toInt()
                require(top + 58 + rows.toLong() * ROW_HEIGHT < Int.MAX_VALUE) { "Neuron gallery exceeds the Swing pixel coordinate range." }
                Section(layer, count, top.toInt(), rows).also { top += 58 + rows.toLong() * ROW_HEIGHT }
            }
            height = maxOf(450, top.toInt() + 16)
        }
        fun bounds(section: Section, neuron: Int): Rectangle = Rectangle(
            12 + neuron % columns * (cellWidth + 12), section.top + 58 + neuron / columns * ROW_HEIGHT, cellWidth, 242)
        fun visible(clip: Rectangle): List<Cell> = buildList {
            for (section in sections) {
                val first = maxOf(0, (clip.y - section.top - 58) / ROW_HEIGHT)
                val last = minOf(section.rows - 1, (clip.y.toLong() + clip.height - section.top - 58).div(ROW_HEIGHT).toInt())
                for (row in first..last) for (column in 0 until columns) {
                    val neuron = row.toLong() * columns + column
                    if (neuron < section.neurons) {
                        val bounds = bounds(section, neuron.toInt())
                        if (bounds.intersects(clip)) add(Cell(section.layer, neuron.toInt(), bounds))
                    }
                }
            }
        }
    }

    private val executor = ThreadPoolExecutor(1, 1, 0L, TimeUnit.MILLISECONDS, ArrayBlockingQueue(1),
        { task -> Thread(task, "jneuro-neuron-gallery").apply { isDaemon = true } }, ThreadPoolExecutor.DiscardOldestPolicy())
    private val revision = AtomicLong()
    private var snapshot: NeuroXorDiagnostics.Snapshot? = null
    private val images = LinkedHashMap<Pair<Int, Int>, BufferedImage>()
    private var requested = emptySet<Pair<Int, Int>>()
    @Volatile private var closed = false

    @Synchronized fun image(layer: Int, neuron: Int): BufferedImage? = images[layer to neuron]

    @Synchronized fun request(next: NeuroXorDiagnostics.Snapshot, cells: List<Cell>, ready: () -> Unit) {
        if (closed) return
        if (snapshot !== next) {
            snapshot = next
            revision.incrementAndGet()
            images.clear()
            requested = emptySet()
        }
        val keys = cells.map { it.layer to it.neuron }.toSet()
        images.keys.retainAll(keys)
        if (keys == requested) return
        requested = keys
        val version = revision.incrementAndGet()
        val missing = cells.filter { it.layer to it.neuron !in images }
        val render = Runnable {
            if (closed || revision.get() != version) return@Runnable
            val rendered = LinkedHashMap<Pair<Int, Int>, BufferedImage>()
            for ((layer, group) in missing.groupBy { it.layer }) {
                if (closed || revision.get() != version) return@Runnable
                val first = group.minOf { it.neuron }
                val last = group.maxOf { it.neuron }
                val maps = NeuroXorDiagnostics.renderHiddenMaps(next, NeuroStudio.resolution(next.parameterCount(), 96), layer, first, last - first + 1)
                for (cell in group) rendered[layer to cell.neuron] = maps[cell.neuron - first]
            }
            synchronized(this) {
                if (closed || revision.get() != version) return@Runnable
                images.putAll(rendered)
            }
            ready()
        }
        if (synchronous) render.run() else executor.execute(render)
    }

    @Synchronized override fun close() {
        closed = true
        revision.incrementAndGet()
        requested = emptySet()
        images.clear()
        executor.shutdownNow()
    }

    companion object { const val ROW_HEIGHT = 254 }
}
