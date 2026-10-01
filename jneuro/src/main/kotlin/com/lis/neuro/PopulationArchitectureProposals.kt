package com.lis.neuro

import java.util.ArrayDeque
import java.util.SplittableRandom

/** Deterministic breadth before feedback, then evaluated-parent mutations interleaved with exploration. */
internal class PopulationArchitectureProposals(
    private val config: ArchitectureSearchConfig,
    manifest: List<NetworkArchitecture>? = null
) {
    private val random = SplittableRandom(config.searchSeed)
    private val issued = LinkedHashMap<NetworkArchitecture, ArchitectureProposal>()
    private val frozen = manifest?.let { ArrayDeque(it) }
    private val bootstrap = ArrayDeque<NetworkArchitecture>()
    private val mutations = AdaptiveArchitecturePlanner(config)
    private val neighbourQueues = HashMap<NetworkArchitecture, ArrayDeque<ArchitectureProposal>>()
    private val minimumWidth = config.widthChoices().first()
    private val maximumDepth = minOf(config.supportedMaxLayers.toLong(),
        1L + (config.maxParameters.toLong() - 4L * minimumWidth - 1L) /
            (minimumWidth.toLong() * (minimumWidth + 1L))).coerceAtLeast(config.minLayers.toLong()).toInt()

    init {
        if (frozen == null) {
            bootstrap.add(config.startingArchitecture())
            bootstrap.add(config.minimumArchitecture())
            val depths = listOf(config.minLayers, maximumDepth, config.minLayers + (maximumDepth - config.minLayers) / 2).distinct()
            for (depth in depths) {
                var low = minimumWidth
                var high = config.maxWidth
                while (low < high) {
                    val middle = low + (high.toLong() - low + 1L).div(2).toInt()
                    if (parameters(depth, middle) <= config.maxParameters) low = middle else high = middle - 1
                }
                for (width in listOf(low, minimumWidth, minimumWidth + (low - minimumWidth) / 2).distinct()) {
                    architecture(List(depth) { width })?.let(bootstrap::add)
                }
            }
        }
    }

    fun next(evaluated: List<ArchitectureCandidate>): ArchitectureProposal? {
        frozen?.let { values ->
            if (values.isEmpty()) return null
            return issue(ArchitectureProposal(values.removeFirst(), mutation = "Frozen population manifest"), evaluated.size)
        }
        val parents = EliteParentSelection.rank(ArchitectureRanking.select(evaluated, config), config)
        neighbourQueues.keys.retainAll(parents.toSet())
        if (issued.isNotEmpty() && issued.size % 4 != 0) {
            for (parent in parents) {
                val queue = neighbourQueues.getOrPut(parent) { ArrayDeque(mutations.mutations(issued.getValue(parent))) }
                while (queue.isNotEmpty()) {
                    val proposal = queue.removeFirst()
                    if (proposal.architecture !in issued) return issue(proposal, evaluated.size)
                }
            }
        }
        while (bootstrap.isNotEmpty()) {
            val value = bootstrap.removeFirst()
            if (value !in issued) return issue(ArchitectureProposal(value, mutation = "Diverse population bootstrap"), evaluated.size)
        }
        repeat(512) {
            val depth = random.nextInt(config.minLayers, maximumDepth + 1)
            val widths = if (config.engine == TrainingEngine.SMALL) {
                val choices = config.widthChoices().toList()
                List(depth) { choices[random.nextInt(choices.size)] }
            } else List(depth) { random.nextLong(config.minWidth.toLong(), config.maxWidth.toLong() + 1).toInt() }
            val value = architecture(widths)
            if (value != null && value !in issued)
                return issue(ArchitectureProposal(value, mutation = "Seeded population exploration"), evaluated.size)
        }
        // Small finite domains should terminate by exhaustion, rather than repeatedly selecting issued shapes.
        var remainingAttempts = 4096
        for (depth in config.minLayers..maximumDepth) {
            val choices = config.widthChoices().take(256).toList()
            val indices = IntArray(depth)
            while (remainingAttempts-- > 0) {
                val value = architecture(indices.map { choices[it] })
                if (value != null && value !in issued)
                    return issue(ArchitectureProposal(value, mutation = "Bounded population exploration"), evaluated.size)
                var position = indices.lastIndex
                while (position >= 0 && ++indices[position] == choices.size) { indices[position] = 0; position-- }
                if (position < 0) break
            }
            if (remainingAttempts <= 0) break
        }
        return null
    }

    private fun issue(value: ArchitectureProposal, evaluated: Int): ArchitectureProposal =
        value.copy(evaluatedCount = evaluated).also { check(issued.putIfAbsent(it.architecture, it) == null) }

    private fun architecture(widths: List<Int>): NetworkArchitecture? = try {
        NetworkArchitecture(widths).takeIf(config::acceptsArchitecture)
    } catch (_: IllegalArgumentException) { null }

    private fun parameters(depth: Int, width: Int): Long {
        val perLayer = width.toLong() * (width + 1L)
        if (depth > 1 && perLayer > (Long.MAX_VALUE - 4L * width - 1L) / (depth - 1L)) return Long.MAX_VALUE
        return 4L * width + 1L + (depth - 1L) * perLayer
    }
}
