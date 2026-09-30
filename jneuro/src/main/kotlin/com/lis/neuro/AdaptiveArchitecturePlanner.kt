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
    val generation: Int = 0,
    val evaluatedCount: Int = 0
)

internal object EliteParentSelection {
    fun rank(selection: ArchitectureSelection, config: ArchitectureSearchConfig): List<NetworkArchitecture> {
        val candidates = (selection.paretoFrontier + selection.reliableFrontier + listOfNotNull(selection.recommended))
            .filter { it.valid }.distinctBy { it.architecture }
        val bySize = compareBy<ArchitectureCandidate> { it.architecture.parameters }.thenBy { it.medianRmse }
            .thenComparator { a, b -> ArchitectureSearchConfig.ARCHITECTURE_ORDER.compare(a.architecture, b.architecture) }
        val byError = compareBy<ArchitectureCandidate> { it.medianRmse }.then(bySize)
        val threshold = (selection.bestError?.medianRmse ?: Double.POSITIVE_INFINITY) + config.nearBestTolerance
        val preferred: (ArchitectureCandidate) -> Boolean = when (config.policy) {
            ArchitecturePolicy.SMALLEST_MEETING_TARGET -> { candidate -> candidate.meetsTarget(config.requiredSuccesses) }
            ArchitecturePolicy.SMALLEST_NEAR_BEST -> { candidate -> candidate.medianRmse <= threshold }
            ArchitecturePolicy.LOWEST_RMSE -> { _ -> false }
        }
        val ranked = candidates.filter(preferred).sortedWith(bySize) + candidates.filterNot(preferred).sortedWith(byError)
        return java.util.List.copyOf(ranked.map { it.architecture })
    }

    fun choose(ranked: List<NetworkArchitecture>, random: SplittableRandom): NetworkArchitecture {
        require(ranked.isNotEmpty()) { "An evaluated elite parent is required." }
        return ranked[minOf(random.nextInt(ranked.size), random.nextInt(ranked.size))]
    }
}

internal class AdaptiveArchitecturePlanner(private val config: ArchitectureSearchConfig) {
    private val random = SplittableRandom(config.searchSeed)
    private val issued = LinkedHashMap<NetworkArchitecture, ArchitectureProposal>()
    private val evaluated = LinkedHashMap<NetworkArchitecture, ArchitectureCandidate>()
    private val neighbours = HashMap<NetworkArchitecture, ArrayDeque<ArchitectureProposal>>()
    private var elites = emptyList<NetworkArchitecture>()
    private var promotedParent: NetworkArchitecture? = null
    private var leader: NetworkArchitecture? = null
    private var stagnant = 0
    private var restarts = 0
    private val awaiting = HashSet<NetworkArchitecture>()
    val lineage: List<ArchitectureProposal> get() = java.util.List.copyOf(issued.values)
    val restartCount: Int get() = restarts

    fun next(): ArchitectureProposal? {
        check(awaiting.isEmpty()) { "Evaluate all seeds before proposing another architecture." }
        return propose()
    }

    fun propose(): ArchitectureProposal? {
        if (elites.isEmpty() && awaiting.isNotEmpty()) return null
        if (issued.isEmpty()) return issue(ArchitectureProposal(config.startingArchitecture()))
        if (stagnant >= config.restartAfter && restarts < config.maxRestarts) {
            restart()?.let { return issue(it) }
        }
        val available = elites.filter { architecture ->
            val choices = neighbours.getOrPut(architecture) { ArrayDeque(mutations(issued.getValue(architecture))) }
            while (choices.isNotEmpty() && choices.first.architecture in issued) choices.removeFirst()
            choices.isNotEmpty()
        }
        if (available.isNotEmpty()) {
            val parent = promotedParent?.takeIf { it in available } ?: EliteParentSelection.choose(available, random)
            promotedParent = null
            return issue(neighbours.getValue(parent).removeFirst())
        }
        return if (awaiting.isEmpty()) restart()?.let { issue(it) } else null
    }

    fun observe(candidate: ArchitectureCandidate) {
        check(candidate.architecture in awaiting) { "Candidate does not match the outstanding proposal." }
        check(candidate.fullyEvaluated) { "Only completed seed groups can guide the search." }
        awaiting.remove(candidate.architecture)
        evaluated[candidate.architecture] = candidate
        val selection = ArchitectureRanking.select(evaluated.values.toList(), config)
        val nextLeader = (selection.recommended ?: selection.bestError)?.architecture
        val nextElites = EliteParentSelection.rank(selection, config)
        val promoted = nextElites.toSet() - elites.toSet()
        stagnant = if (promoted.isNotEmpty() || nextLeader != leader) 0 else stagnant + 1
        promotedParent = when {
            nextLeader != leader -> nextLeader
            promotedParent == nextLeader -> promotedParent
            candidate.architecture in promoted -> candidate.architecture
            else -> promotedParent?.takeIf { it in nextElites }
        }
        neighbours.keys.retainAll(nextElites.toSet())
        elites = nextElites
        leader = nextLeader
    }

    internal fun mutations(parent: ArchitectureProposal): List<ArchitectureProposal> {
        val shape = parent.architecture.hidden
        val result = LinkedHashMap<NetworkArchitecture, ArchitectureProposal>()
        fun add(widths: List<Int>, operation: String) {
            if (widths.size !in config.minLayers..config.supportedMaxLayers || widths.any { it !in config.minWidth..config.maxWidth }) return
            val architecture = try { NetworkArchitecture(widths) } catch (_: IllegalArgumentException) { return }
            if (architecture == parent.architecture || !config.acceptsArchitecture(architecture)) return
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
        if (shape.size < config.supportedMaxLayers) for (position in 0..shape.size) {
            for (width in listOf(config.widthChoices().first(), shape[minOf(position, shape.lastIndex)]).distinct()) {
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
        val recorded = proposal.copy(evaluatedCount = evaluated.size)
        check(issued.putIfAbsent(recorded.architecture, recorded) == null) { "Architecture was already evaluated." }
        awaiting.add(recorded.architecture)
        return recorded
    }

    private fun restart(): ArchitectureProposal? {
        while (restarts < config.maxRestarts) {
            restarts++
            stagnant = 0
            if (elites.isNotEmpty()) {
                repeat(256) {
                    val parent = issued.getValue(EliteParentSelection.choose(elites, random))
                    eliteRestart(parent)?.let { return it }
                }
                continue
            }
            val minimum = config.minimumArchitecture()
            if (minimum !in issued) return ArchitectureProposal(minimum, mutation = "Minimum-size bootstrap")
            repeat(256) {
                val depth = random.nextLong(config.minLayers.toLong(), config.supportedMaxLayers.toLong() + 1).toInt()
                if (depth.toLong() * 2 + 3 > config.maxParameters) return@repeat
                val architecture = try { NetworkArchitecture(List(depth) { randomWidth() }) } catch (_: IllegalArgumentException) { return@repeat }
                if (config.acceptsArchitecture(architecture) && architecture !in issued) {
                    return ArchitectureProposal(architecture, mutation = "Bootstrap restart $restarts (no valid parent)")
                }
            }
        }
        return null
    }

    private fun eliteRestart(parent: ArchitectureProposal): ArchitectureProposal? {
        val widths = parent.architecture.hidden.toMutableList()
        val operation = when (random.nextInt(3)) {
            0 -> {
                val layer = random.nextInt(widths.size)
                val previous = widths[layer]
                widths[layer] = randomWidth()
                "H${layer + 1}: $previous → ${widths[layer]}"
            }
            1 -> {
                if (widths.size >= config.supportedMaxLayers) return null
                val layer = random.nextInt(widths.size + 1)
                val width = randomWidth()
                widths.add(layer, width)
                "Insert H${layer + 1} ($width)"
            }
            else -> {
                if (widths.size <= config.minLayers) return null
                val layer = random.nextInt(widths.size)
                widths.removeAt(layer)
                "Remove H${layer + 1}"
            }
        }
        val architecture = try { NetworkArchitecture(widths) } catch (_: IllegalArgumentException) { return null }
        if (!config.acceptsArchitecture(architecture) || architecture in issued) return null
        return ArchitectureProposal(architecture, parent.architecture, "Elite restart $restarts: $operation", parent.generation + 1)
    }

    private fun randomWidth(): Int = if (config.engine == TrainingEngine.SMALL) {
        val widths = config.widthChoices().toList()
        widths[random.nextInt(widths.size)]
    } else random.nextLong(config.minWidth.toLong(), config.maxWidth.toLong() + 1).toInt()
}
