package com.lis.neuro

import java.util.ArrayDeque
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.ExecutorCompletionService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger

internal class TrialActivity {
    private val active = ConcurrentHashMap<Int, ArchitectureRunningTrial>()
    private val count = AtomicInteger()
    private val maximum = AtomicInteger()
    val peak: Int get() = maximum.get()
    fun snapshot(): List<ArchitectureRunningTrial> = active.entries.sortedBy { it.key }.map { it.value }

    fun <T> track(index: Int, architecture: NetworkArchitecture, seed: Long,
                  action: ((Int, Double) -> Unit) -> T): T {
        maximum.accumulateAndGet(count.incrementAndGet(), ::maxOf)
        try {
            active[index] = ArchitectureRunningTrial(architecture, seed, 0, Double.POSITIVE_INFINITY)
            return action { epoch, best -> active[index] = ArchitectureRunningTrial(architecture, seed, epoch, best) }
        } finally {
            active.remove(index)
            count.decrementAndGet()
        }
    }
}

internal class AdaptiveTrialScheduler(
    private val data: ArchitectureSearchData,
    private val config: ArchitectureSearchConfig,
    private val searchId: String? = null,
    private val evaluate: (NetworkArchitecture, Long, () -> Boolean, (Int, Double) -> Unit) -> ArchitectureTrial
) {
    fun search(onProgress: (ArchitectureSearchProgress) -> Unit, cancelled: () -> Boolean): ArchitectureSearchResult {
        val plannedTrials = config.plannedTrials()
        val planner = AdaptiveArchitecturePlanner(config)
        val start = System.nanoTime()
        val stopping = AtomicBoolean(false)
        val activity = TrialActivity()
        val workerId = AtomicInteger()
        val pool = Executors.newFixedThreadPool(config.parallelism) { runnable ->
            Thread(runnable, "jneuro-search-${workerId.incrementAndGet()}").apply { isDaemon = true }
        }
        data class Finished(val architecture: NetworkArchitecture, val trial: ArchitectureTrial)
        val completions = ExecutorCompletionService<Finished>(pool)
        val completed = ArrayList<ArchitectureCandidate>()
        val pending = LinkedHashMap<NetworkArchitecture, MutableMap<Long, ArchitectureTrial>>()
        val ready = ArrayDeque<Pair<NetworkArchitecture, Long>>()
        var fundedTrials = 0
        var submitted = 0
        var received = 0
        var retainedParameters = 0L
        var lastPublish = Long.MIN_VALUE
        var admissionEnd: ArchitectureTermination? = null
        var stopReason: ArchitectureTermination? = null

        fun candidate(architecture: NetworkArchitecture, trials: Map<Long, ArchitectureTrial>) =
            ArchitectureCandidate(architecture, config.seeds.mapNotNull { trials[it] }, config.seeds.size, config.targetRmse)
        fun candidates(): List<ArchitectureCandidate> = completed + pending.map { (architecture, trials) -> candidate(architecture, trials) }
        fun publish(force: Boolean = false) {
            val now = System.nanoTime()
            if (force || lastPublish == Long.MIN_VALUE || now - lastPublish >= 100_000_000L) {
                onProgress(ArchitectureSearchProgress(planner.lineage.size, plannedTrials, received, candidates(),
                    activity.snapshot(), now - start, planner.lineage, activity.peak))
                lastPublish = now
            }
        }
        fun checkStop() {
            if (stopReason != null) return
            if (cancelled() || Thread.currentThread().isInterrupted) stopReason = ArchitectureTermination.CANCELLED
            else if (config.timeLimitSeconds > 0 && (System.nanoTime() - start) / 1_000_000_000L >= config.timeLimitSeconds) {
                stopReason = ArchitectureTermination.TIME_LIMIT
            }
            if (stopReason != null) stopping.set(true)
        }
        fun admit(): Boolean {
            if (admissionEnd != null) return false
            if (fundedTrials.toLong() + config.seeds.size > plannedTrials) {
                admissionEnd = ArchitectureTermination.TRIAL_BUDGET
                return false
            }
            val proposal = planner.propose() ?: return false
            val architecture = proposal.architecture
            val storage = architecture.parameters.toLong() * config.seeds.size
            if (retainedParameters + storage > 8_000_000L) {
                admissionEnd = ArchitectureTermination.MEMORY_LIMIT
                return false
            }
            NeuroLog.debug("search", "search.proposal.admitted") { mapOf("searchId" to searchId,
                "topology" to architecture, "parent" to proposal.parent, "mutation" to proposal.mutation,
                "generation" to proposal.generation, "fundedTrials" to fundedTrials + config.seeds.size) }
            retainedParameters += storage
            fundedTrials += config.seeds.size
            pending[architecture] = LinkedHashMap()
            config.seeds.forEach { ready.addLast(architecture to it) }
            return true
        }
        fun receive(finished: Finished) {
            val trials = pending.getValue(finished.architecture)
            check(trials.putIfAbsent(finished.trial.seed, finished.trial) == null) { "Duplicate trial completion." }
            received++
            if (trials.size == config.seeds.size) {
                val candidate = candidate(finished.architecture, trials)
                pending.remove(finished.architecture)
                completed += candidate
                NeuroLog.debug("search", "search.candidate.evaluated") { mapOf("searchId" to searchId,
                    "topology" to candidate.architecture, "valid" to candidate.valid,
                    "medianRmse" to candidate.medianRmse, "successfulSeeds" to candidate.successes) }
                if (candidate.fullyEvaluated) planner.observe(candidate)
            }
        }
        try {
            publish(true)
            while (true) {
                checkStop()
                while (true) receive((completions.poll() ?: break).get())
                while (!stopping.get() && submitted - received < config.parallelism) {
                    checkStop()
                    if (stopping.get()) break
                    if (ready.isEmpty() && !admit()) break
                    val (architecture, seed) = ready.removeFirst()
                    val index = submitted++
                    completions.submit {
                        activity.track(index, architecture, seed) { progress ->
                            Finished(architecture, evaluate(architecture, seed,
                                { stopping.get() || Thread.currentThread().isInterrupted }, progress))
                        }
                    }
                }
                publish()
                if (received == submitted) break
                completions.poll(50, TimeUnit.MILLISECONDS)?.let { receive(it.get()) }
            }
            checkStop()
            publish(true)
            val end = stopReason ?: admissionEnd ?: ArchitectureTermination.NEIGHBOURHOODS_EXHAUSTED
            return ArchitectureSearchResult(data, config, end, planner.lineage.size, candidates(),
                System.nanoTime() - start, planner.lineage, activity.peak)
        } finally {
            stopping.set(true)
            pool.shutdownNow()
            try { pool.awaitTermination(5, TimeUnit.SECONDS) } catch (_: InterruptedException) { Thread.currentThread().interrupt() }
        }
    }
}
