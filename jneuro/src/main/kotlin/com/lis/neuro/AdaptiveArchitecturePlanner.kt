package com.lis.neuro

import java.util.ArrayDeque
import java.util.SplittableRandom

internal enum class ArchitectureSearchStrategy(val label: String) {
    ADAPTIVE("Adaptive · evolve best"), EXHAUSTIVE("Exhaustive · reference");
    override fun toString() = label
}

internal data class ArchitectureProposal(
    val architecture: NetworkArchitecture,
    val parent: NetworkArchitecture? = null,
    val mutation: String = "Initial architecture",
    val generation: Int = 0
)

internal class AdaptiveArchitecturePlanner(private val config: ArchitectureSearchConfig) {
    private val random = SplittableRandom(config.searchSeed)
    private val issued = LinkedHashMap<NetworkArchitecture, ArchitectureProposal>()
    private val evaluated = LinkedHashMap<NetworkArchitecture, ArchitectureCandidate>()
    private val neighbours = HashMap<NetworkArchitecture, ArrayDeque<ArchitectureProposal>>()
    private val parents = ArrayDeque<NetworkArchitecture>()
    private var elites = emptySet<NetworkArchitecture>()
    private var leader: NetworkArchitecture? = null
    private var stagnant = 0
    private var restarts = 0
    private var awaiting: NetworkArchitecture? = null
    val lineage: List<ArchitectureProposal> get() = java.util.List.copyOf(issued.values)
    val restartCount: Int get() = restarts

    fun next(): ArchitectureProposal? {
        check(awaiting == null) { "Evaluate all seeds before proposing another architecture." }
        if (issued.isEmpty()) return issue(ArchitectureProposal(config.startingArchitecture()))
        if (stagnant >= config.restartAfter && restarts < config.maxRestarts) {
            restart()?.let { return issue(it) }
        }
        while (parents.isNotEmpty()) {
            val parent = parents.removeFirst()
            if (parent !in elites) continue
            val choices = neighbours.getValue(parent)
            while (choices.isNotEmpty()) {
                val proposal = choices.removeFirst()
                if (proposal.architecture in issued) continue
                if (choices.isNotEmpty()) parents.addLast(parent)
                return issue(proposal)
            }
        }
        return restart()?.let { issue(it) }
    }

    fun observe(candidate: ArchitectureCandidate) {
        check(candidate.architecture == awaiting) { "Candidate does not match the outstanding proposal." }
        check(candidate.fullyEvaluated) { "Only completed seed groups can guide the search." }
        awaiting = null
        evaluated[candidate.architecture] = candidate
        val selection = ArchitectureRanking.select(evaluated.values.toList(), config)
        val nextLeader = (selection.recommended ?: selection.bestError)?.architecture
        val nextElites = (selection.paretoFrontier + selection.reliableFrontier + listOfNotNull(selection.recommended))
            .map { it.architecture }.toSet()
        val promoted = nextElites - elites
        stagnant = if (promoted.isNotEmpty() || nextLeader != leader) 0 else stagnant + 1
        parents.removeIf { it !in nextElites }
        for (architecture in promoted.sortedWith(ArchitectureSearchConfig.ARCHITECTURE_ORDER).asReversed()) {
            neighbours.getOrPut(architecture) { ArrayDeque(mutations(issued.getValue(architecture))) }
            parents.remove(architecture)
            parents.addFirst(architecture)
        }
        if (nextLeader != null && nextLeader != leader) {
            parents.remove(nextLeader)
            parents.addFirst(nextLeader)
        }
        elites = nextElites
        leader = nextLeader
    }

    internal fun mutations(parent: ArchitectureProposal): List<ArchitectureProposal> {
        val shape = parent.architecture.hidden
        val result = LinkedHashMap<NetworkArchitecture, ArchitectureProposal>()
        fun add(widths: List<Int>, operation: String) {
            if (widths.size !in config.minLayers..config.maxLayers || widths.any { it !in config.minWidth..config.maxWidth }) return
            val architecture = try { NetworkArchitecture(widths) } catch (_: IllegalArgumentException) { return }
            if (architecture == parent.architecture || architecture.parameters > config.maxParameters) return
            result.putIfAbsent(architecture, ArchitectureProposal(architecture, parent.architecture, operation, parent.generation + 1))
        }
        for (layer in shape.indices) {
            for (width in listOf(shape[layer] - 1, (shape[layer].toLong() + 1).coerceAtMost(config.maxWidth.toLong()).toInt(), maxOf(config.minWidth, shape[layer] / 2),
                minOf(config.maxWidth, (shape[layer].toLong() * 2).coerceAtMost(config.maxWidth.toLong()).toInt())).distinct()) {
                val widths = shape.toMutableList(); widths[layer] = width
                add(widths, "H${layer + 1}: ${shape[layer]} → $width")
            }
            if (shape.size > config.minLayers) add(shape.filterIndexed { index, _ -> index != layer }, "Remove H${layer + 1}")
            if (layer < shape.lastIndex) {
                val widths = shape.toMutableList()
                widths[layer] = shape[layer + 1]; widths[layer + 1] = shape[layer]
                add(widths, "Swap H${layer + 1}/H${layer + 2}")
            }
        }
        if (shape.size < config.maxLayers) for (position in 0..shape.size) {
            for (width in listOf(config.minWidth, shape[minOf(position, shape.lastIndex)]).distinct()) {
                val widths = shape.toMutableList(); widths.add(position, width)
                add(widths, "Insert H${position + 1} ($width)")
            }
        }
        val values = result.values.toMutableList()
        val mutationRandom = SplittableRandom(config.searchSeed xor shape.fold(17L) { value, width -> value * 131 + width })
        for (index in values.lastIndex downTo 1) {
            val other = mutationRandom.nextInt(index + 1)
            val value = values[index]; values[index] = values[other]; values[other] = value
        }
        return values
    }

    private fun issue(proposal: ArchitectureProposal): ArchitectureProposal {
        check(issued.putIfAbsent(proposal.architecture, proposal) == null) { "Architecture was already evaluated." }
        awaiting = proposal.architecture
        return proposal
    }

    private fun restart(): ArchitectureProposal? {
        while (restarts < config.maxRestarts) {
            restarts++
            stagnant = 0
            val minimum = config.minimumArchitecture()
            if (minimum !in issued) return ArchitectureProposal(minimum, mutation = "Minimum-size exploration")
            repeat(256) {
                val depth = random.nextLong(config.minLayers.toLong(), config.maxLayers.toLong() + 1).toInt()
                if (depth.toLong() * 2 + 3 > config.maxParameters) return@repeat
                val architecture = try { NetworkArchitecture(List(depth) { random.nextLong(config.minWidth.toLong(), config.maxWidth.toLong() + 1).toInt() }) } catch (_: IllegalArgumentException) { return@repeat }
                if (architecture.parameters <= config.maxParameters && architecture !in issued) {
                    return ArchitectureProposal(architecture, mutation = "Exploration restart $restarts")
                }
            }
        }
        return null
    }
}
