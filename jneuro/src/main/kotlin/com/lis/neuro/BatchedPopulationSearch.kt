package com.lis.neuro

/** Bounded streaming brackets share TensorFlow executions; pruning never splits an architecture's seed group. */
internal class BatchedPopulationSearch(
    private val data: ArchitectureSearchData,
    private val config: ArchitectureSearchConfig,
    manifest: List<NetworkArchitecture>? = null,
    private val clock: () -> Long = System::nanoTime,
    private val searchId: String? = null,
    private val open: (Array<NeuroTrainingState>) -> TensorFlowSearchCohort = { states ->
        TensorFlowMath.searchCohort(states, config.hyperParameters, config.precision, config.backend,
            DoubleArray(data.validation.size * 2) { index -> data.validation[index / 2].let { if (index % 2 == 0) it.x else it.y } },
            DoubleArray(data.validation.size) { data.validation[it].target }, config.batchSize)
    }
) {
    private class Group(val proposal: ArchitectureProposal, seeds: List<Long>, val admitted: Long) {
        val lanes = seeds.map { Lane(this, it) }
        lateinit var bracket: Bracket
        val terminal: Boolean get() = lanes.all { it.state != null }
        var candidate: ArchitectureCandidate? = null
    }
    private class Bracket(val groups: MutableList<Group>, var rung: Int)
    private class Lane(val group: Group, val seed: Long) {
        val shuffle = TrainingShuffle(seed xor -7046029254386353131L)
        val id = NeuroLog.id("trial")
        var epoch = 0
        var nativeOffset = 0
        var best = Double.POSITIVE_INFINITY
        var bestEpoch = 0
        var trainingAtBest = Double.POSITIVE_INFINITY
        var finalScore = Double.POSITIVE_INFINITY
        var snapshot: NeuroXorDiagnostics.Snapshot? = null
        var exportBest = false
        var current: NeuroTrainingState? = null
        var state: ArchitectureTrialState? = null
        var failure = ""
        var info: TrainingDeviceInfo? = null
        var finishedLogged = false
        val history = ArrayList<ArchitectureCheckpoint>()
        val segments = ArrayList<ArchitectureBatchSegment>()
    }
    private class Bucket(val lanes: List<Lane>, val kernel: TensorFlowSearchCohort) {
        var nanosPerEpoch = 0L
    }

    private val frozen = manifest ?: if (config.strategy == ArchitectureSearchStrategy.EXHAUSTIVE) config.architectures() else null
    private val proposals = PopulationArchitectureProposals(config, frozen)
    private val groups = ArrayList<Group>()
    private val brackets = ArrayList<Bracket>()
    private val buckets = ArrayList<Bucket>()
    private var retainedParameters = 0L
    private var admissionEnd: ArchitectureTermination? = null
    private var stop: ArchitectureTermination? = null
    private var nativeCalls = 0L
    private var aggregateEpochs = 0L
    private var maximumBatch = 0
    private var peakResidents = 0
    private var started = 0L
    private var lastPublish = Long.MIN_VALUE
    private var membershipChanged = false
    private val capacity = maxOf(1, config.modelsPerBatch / config.seeds.size)

    fun search(onProgress: (ArchitectureSearchProgress) -> Unit, cancelled: () -> Boolean): ArchitectureSearchResult {
        started = clock()
        fun checkStop() {
            if (stop == null) {
                if (cancelled() || Thread.currentThread().isInterrupted) stop = ArchitectureTermination.CANCELLED
                else if (config.timeLimitSeconds > 0 && (clock() - started) / 1_000_000_000L >= config.timeLimitSeconds)
                    stop = ArchitectureTermination.TIME_LIMIT
            }
        }
        fun publish(force: Boolean = false) {
            val now = clock()
            if (!force && lastPublish != Long.MIN_VALUE && now - lastPublish < 100_000_000L) return
            val active = groups.flatMap { group -> group.lanes.filter { it.state == null }.map {
                ArchitectureRunningTrial(group.proposal.architecture, it.seed, it.epoch, it.best)
            } }
            val planned = minOf(config.maxTrials / config.seeds.size, frozen?.size ?: Int.MAX_VALUE) * config.seeds.size
            onProgress(ArchitectureSearchProgress(groups.size, planned,
                groups.sumOf { it.lanes.count { lane -> lane.state != null } }, candidates(), active, now - started,
                groups.map { it.proposal }, maximumBatch, if (buckets.isEmpty()) 0 else 1,
                if (peakResidents == 0) 0 else 1, buckets.sumOf { it.lanes.size },
                nativeTrainingCalls = nativeCalls, modelsPerBatch = maximumBatch,
                aggregateEpochsPerSecond = rate(now)))
            lastPublish = now
        }
        var problem: Throwable? = null
        try {
            checkStop()
            if (stop == null) admit()
            publish(true)
            while (groups.any { !it.terminal }) {
                checkStop()
                if (stop != null) break
                if (buckets.isEmpty()) rebuild { checkStop(); stop == null }
                checkStop()
                if (stop != null) break
                val active = groups.flatMap { it.lanes }.filter { it.state == null && it.epoch < it.group.bracket.rung }
                val step = active.minOfOrNull { minOf(it.group.bracket.rung - it.epoch, config.checkEvery - it.epoch % config.checkEvery, 64) } ?: 0
                val targets = active.associateWith { it.epoch + step }
                for (bucket in buckets) {
                    while (bucket.lanes.any { it.state == null && it.epoch < (targets[it] ?: it.epoch) }) {
                        checkStop()
                        if (stop != null) break
                        advance(bucket, targets)
                        publish()
                    }
                    if (stop != null) break
                }
                checkStop()
                if (stop != null) break
                if (settleBrackets()) {
                    releaseBuckets(retainCurrent = true)
                    admit()
                    publish(true)
                }
                if (buckets.isEmpty() && groups.none { !it.terminal }) break
            }
            checkStop()
            if (stop != null) groups.flatMap { it.lanes }.filter { it.state == null }.forEach {
                it.state = ArchitectureTrialState.CANCELLED
            }
            releaseBuckets(retainCurrent = false)
            publish(true)
            return ArchitectureSearchResult(data, config, stop ?: admissionEnd ?: ArchitectureTermination.COMPLETED,
                groups.size, candidates(), clock() - started, groups.map { it.proposal }, maximumBatch,
                if (peakResidents == 0) 0 else 1, peakResidents, nativeTrainingCalls = nativeCalls,
                modelsPerBatch = maximumBatch, aggregateEpochsPerSecond = rate(clock()))
        } catch (failure: Throwable) { problem = failure; throw failure }
        finally {
            for (bucket in buckets.asReversed()) try { bucket.kernel.close() } catch (failure: Throwable) {
                if (problem == null) problem = failure else problem.addSuppressed(failure)
            }
            buckets.clear()
        }
    }

    private fun admit() {
        if (admissionEnd != null) return
        val added = ArrayList<Group>()
        val evaluated = candidates().filter { it.fullyEvaluated }
        val available = capacity - groups.count { !it.terminal }
        repeat(available) {
            if ((groups.size.toLong() + 1) * config.seeds.size > config.maxTrials) {
                admissionEnd = if (frozen != null && groups.size == frozen.size) ArchitectureTermination.COMPLETED
                    else ArchitectureTermination.TRIAL_BUDGET
                return@repeat
            }
            if (admissionEnd != null) return@repeat
            val proposal = proposals.next(evaluated)
            if (proposal == null) {
                admissionEnd = if (frozen == null) ArchitectureTermination.NEIGHBOURHOODS_EXHAUSTED else ArchitectureTermination.COMPLETED
                return@repeat
            }
            val parameters = proposal.architecture.parameters.toLong() * config.seeds.size
            if (retainedParameters + parameters > MAX_RETAINED_PARAMETERS) {
                admissionEnd = ArchitectureTermination.MEMORY_LIMIT
                return@repeat
            }
            retainedParameters += parameters
            Group(proposal, config.seeds, clock()).also { group ->
                groups += group; added += group
                NeuroLog.debug("search", "search.proposal.admitted") { mapOf("searchId" to searchId,
                    "topology" to proposal.architecture, "parent" to proposal.parent, "mutation" to proposal.mutation,
                    "budgetPolicy" to config.budgetPolicy, "modelsPerBatch" to config.modelsPerBatch) }
                for (lane in group.lanes) NeuroLog.info("search", "search.trial.started", "searchId" to searchId,
                    "trialId" to lane.id, "topology" to proposal.architecture, "seed" to lane.seed,
                    "execution" to config.execution, "route" to ArchitectureTrialRoute.TENSOR_BATCH,
                    "budgetPolicy" to config.budgetPolicy, "modelsPerBatch" to config.modelsPerBatch)
            }
        }
        if (added.isNotEmpty()) {
            val first = if (config.budgetPolicy == ArchitectureBudgetPolicy.FULL) config.maxEpochs
                else minOf(config.initialEpochs, config.maxEpochs)
            val bracket = Bracket(added, first)
            added.forEach { it.bracket = bracket }
            brackets += bracket
        }
    }

    private fun rebuild(continuing: () -> Boolean) {
        val alive = groups.flatMap { it.lanes }.filter { it.state == null }
        val packed = ArrayList<List<Lane>>()
        for (depth in alive.groupBy { it.group.proposal.architecture.hidden.size }.values) {
            var pending = arrayListOf<Lane>()
            for (lane in depth) {
                val proposed = pending + lane
                val padded = paddedParameters(proposed)
                val actual = proposed.sumOf { it.group.proposal.architecture.parameters.toLong() }
                if (pending.isNotEmpty() && (proposed.size > config.modelsPerBatch || padded > actual * 4)) {
                    packed += pending; pending = arrayListOf()
                }
                pending += lane
            }
            if (pending.isNotEmpty()) packed += pending
        }
        // Include multiple native parameter/velocity/best/loop buffers and bounded shuffle workspaces.
        val bytes = packed.sumOf(::residentBytes)
        if (bytes > MAX_RESIDENT_BYTES) { stop = ArchitectureTermination.MEMORY_LIMIT; return }
        for (lanes in packed) {
            if (!continuing()) break
            try {
                val states = Array(lanes.size) { index -> lanes[index].current ?: data.newNetwork(
                    lanes[index].group.proposal.architecture, config.hyperParameters, lanes[index].seed).exportTrainingState(shareDataset = true) }
                val kernel = open(states)
                val bucket = Bucket(lanes, kernel)
                buckets += bucket
                check(kernel.size == lanes.size)
                for ((index, lane) in lanes.withIndex()) {
                    lane.current = null; lane.info = kernel.info; lane.nativeOffset = lane.epoch
                    lane.segments += ArchitectureBatchSegment(lane.epoch, lane.epoch, kernel.paddedTopology.toList(), lanes.size, index)
                    NeuroLog.info("search", "search.trial.backend.ready", "searchId" to searchId, "trialId" to lane.id,
                        "effectiveBackend" to kernel.info.backend, "effectivePrecision" to kernel.info.precision,
                        "kernelVersion" to kernel.info.kernelVersion, "deviceIdentity" to kernel.info.identity,
                        "epoch" to lane.epoch, "modelsPerBatch" to lanes.size, "paddedTopology" to kernel.paddedTopology.toList())
                }
                peakResidents = maxOf(peakResidents, buckets.sumOf { it.lanes.size })
                val initial = BooleanArray(lanes.size) { lanes[it].history.isEmpty() }
                if (initial.any { it }) score(bucket, initial)
            } catch (failure: Exception) { lanes.forEach { fail(it, failure) } }
        }
    }

    private fun advance(bucket: Bucket, targets: Map<Lane, Int>) {
        val active = BooleanArray(bucket.lanes.size) { index -> bucket.lanes[index].let {
            it.state == null && it.epoch < (targets[it] ?: it.epoch)
        } }
        if (active.none { it }) return
        val boundary = bucket.lanes.indices.filter { active[it] }.minOf { index -> bucket.lanes[index].let {
            minOf(targets.getValue(it) - it.epoch, it.group.bracket.rung - it.epoch, config.checkEvery - it.epoch % config.checkEvery)
        } }
        val byTime = if (bucket.nanosPerEpoch == 0L) 1 else (50_000_000L / bucket.nanosPerEpoch).coerceIn(1, 64).toInt()
        val byOrders = maxOf(1, 1_048_576 / bucket.lanes.size / data.training.size)
        val count = minOf(boundary, byTime, byOrders, 64)
        val orders = Array(bucket.lanes.size) { index ->
            if (active[index]) bucket.lanes[index].shuffle.reserve(data.training.size, count) else emptyArray()
        }
        val start = clock()
        try {
            nativeCalls++
            maximumBatch = maxOf(maximumBatch, active.count { it })
            val failed = bucket.kernel.advance(orders, active)
            check(failed.size == bucket.lanes.size)
            for ((index, lane) in bucket.lanes.withIndex()) if (active[index]) {
                if (failed[index]) fail(lane, IllegalStateException("TensorFlow lane produced non-finite training state"))
                else {
                    lane.shuffle.commit(count); lane.epoch += count; aggregateEpochs += count
                    lane.segments[lane.segments.lastIndex] = lane.segments.last().copy(endEpoch = lane.epoch)
                }
            }
            bucket.nanosPerEpoch = maxOf(1L, (clock() - start) / count)
            val due = BooleanArray(bucket.lanes.size) { index -> bucket.lanes[index].let {
                active[index] && it.state == null && (it.epoch % config.checkEvery == 0 || it.epoch == it.group.bracket.rung)
            } }
            if (due.any { it }) score(bucket, due)
        } catch (failure: Exception) { bucket.lanes.filter { it.state == null }.forEach { fail(it, failure) } }
    }

    private fun score(bucket: Bucket, due: BooleanArray) {
        val result = bucket.kernel.score(due)
        for ((index, lane) in bucket.lanes.withIndex()) if (due[index] && lane.state == null) {
            if (result.failed[index]) { fail(lane, IllegalStateException("TensorFlow lane scoring failed")); continue }
            check(lane.epoch == lane.nativeOffset + result.epochs[index]) { "TensorFlow epoch checkpoint does not match reserved orders" }
            val training = result.trainingRmse[index]
            val value = if (data.evaluation == ArchitectureEvaluation.TRAINING_FIT) training else result.validationRmse[index]
            if (!training.isFinite() || !value.isFinite()) { fail(lane, IllegalStateException("TensorFlow produced non-finite RMSE")); continue }
            lane.finalScore = value
            if (result.bestScore[index] < lane.best) {
                lane.best = result.bestScore[index]; lane.bestEpoch = lane.nativeOffset + result.bestEpoch[index]
                lane.trainingAtBest = training; lane.exportBest = true
            }
            if (lane.history.lastOrNull()?.epoch != lane.epoch) lane.history += ArchitectureCheckpoint(lane.epoch, training, value)
            NeuroLog.debug("search", "search.trial.checkpoint") { mapOf("searchId" to searchId, "trialId" to lane.id,
                "epoch" to lane.epoch, "trainingRmse" to training, "score" to value, "bestRmse" to lane.best) }
            if (lane.history.size > 128) {
                val keep = lane.history.filterIndexed { position, item -> position % 2 == 0 || position == lane.history.lastIndex || item.epoch == lane.bestEpoch }
                lane.history.clear(); lane.history.addAll(keep)
            }
        }
    }

    private fun settleBrackets(): Boolean {
        var changed = membershipChanged
        membershipChanged = false
        for (bracket in brackets) {
            for (group in bracket.groups) if (!group.terminal) {
                val failed = group.lanes.any { it.state == ArchitectureTrialState.FAILED }
                val reached = group.lanes.all { it.epoch >= bracket.rung }
                if (failed) {
                    group.lanes.filter { it.state == null }.forEach { it.state = ArchitectureTrialState.PRUNED }
                    changed = true
                } else if (reached && (bracket.rung == config.maxEpochs ||
                    config.budgetPolicy == ArchitectureBudgetPolicy.SUCCESSIVE_HALVING && successful(group))) {
                    group.lanes.forEach { it.state = ArchitectureTrialState.COMPLETED }; changed = true
                }
            }
            val remaining = bracket.groups.filter { !it.terminal }
            if (remaining.isNotEmpty() && remaining.all { group -> group.lanes.all { it.epoch >= bracket.rung } }) {
                check(config.budgetPolicy == ArchitectureBudgetPolicy.SUCCESSIVE_HALVING)
                val ranked = remaining.sortedWith(compareBy<Group> { median(it.lanes.map { lane -> lane.finalScore }) }
                    .thenBy { it.proposal.architecture.parameters }
                    .thenComparator { a, b -> ArchitectureSearchConfig.ARCHITECTURE_ORDER.compare(a.proposal.architecture, b.proposal.architecture) })
                val keep = maxOf(1, (remaining.size + config.reductionFactor - 1) / config.reductionFactor)
                for (group in ranked.drop(keep)) group.lanes.forEach { it.state = ArchitectureTrialState.PRUNED }
                bracket.rung = minOf(config.maxEpochs.toLong(), bracket.rung.toLong() * config.reductionFactor).toInt()
                if (ranked.size > keep) changed = true
            }
        }
        brackets.removeAll { it.groups.all { group -> group.terminal } }
        return changed
    }

    private fun successful(group: Group): Boolean = group.lanes.count { it.best <= config.targetRmse } >= config.requiredSuccesses &&
        median(group.lanes.map { it.best }) <= config.targetRmse

    private fun releaseBuckets(retainCurrent: Boolean) {
        var failure: Throwable? = null
        for (bucket in buckets) {
            try {
                val improved = bucket.lanes.indices.filter { bucket.lanes[it].exportBest }.toIntArray()
                if (improved.isNotEmpty()) {
                    val states = bucket.kernel.exportStates(improved, best = true)
                    for ((position, index) in improved.withIndex()) {
                        val lane = bucket.lanes[index]; val state = states[position]
                        lane.snapshot = NeuroXorDiagnostics.Snapshot(lane.bestEpoch, lane.trainingAtBest, config.hyperParameters.beta,
                            config.hyperParameters.sigmoidMode, state.topology, state.weights, state.biases)
                        lane.exportBest = false
                    }
                }
                if (retainCurrent) {
                    val live = bucket.lanes.indices.filter { bucket.lanes[it].state == null }.toIntArray()
                    if (live.isNotEmpty()) {
                        val states = bucket.kernel.exportStates(live)
                        live.forEachIndexed { position, index -> bucket.lanes[index].current = states[position] }
                    }
                }
            } catch (problem: Exception) {
                bucket.lanes.filter { it.state == null || it.state == ArchitectureTrialState.COMPLETED }.forEach { fail(it, problem) }
            }
            finally { try { bucket.kernel.close() } catch (problem: Throwable) {
                if (failure == null) failure = problem else failure.addSuppressed(problem)
            } }
        }
        buckets.clear()
        for (lane in groups.flatMap { it.lanes }) if (lane.state != null && !lane.finishedLogged) {
            NeuroLog.info("search", "search.trial.finished", "searchId" to searchId, "trialId" to lane.id,
                "state" to lane.state, "epoch" to lane.epoch, "bestEpoch" to lane.bestEpoch, "bestRmse" to lane.best,
                "execution" to config.execution, "budgetPolicy" to config.budgetPolicy,
                "nativeTrainingCalls" to nativeCalls, "modelsPerBatch" to maximumBatch)
            lane.finishedLogged = true
        }
        failure?.let { throw it }
    }

    private fun candidates(): List<ArchitectureCandidate> = groups.map { group ->
        group.candidate ?: ArchitectureCandidate(group.proposal.architecture, group.lanes.mapNotNull { lane -> lane.state?.let { state ->
            ArchitectureTrial(lane.seed, state, lane.epoch, lane.bestEpoch, lane.best, lane.trainingAtBest, lane.finalScore,
                lane.epoch.toLong() * data.training.size, clock() - group.admitted, lane.history, lane.snapshot, lane.failure,
                lane.info, lane.id, cohort = true, execution = ArchitectureExecution.BATCHED,
                route = ArchitectureTrialRoute.TENSOR_BATCH, batchSegments = lane.segments)
        } }, config.seeds.size, config.targetRmse).also {
            if (group.terminal && group.lanes.all { lane -> lane.finishedLogged }) group.candidate = it
        }
    }

    private fun fail(lane: Lane, failure: Exception) {
        membershipChanged = true
        if (failure is InterruptedException) Thread.currentThread().interrupt()
        lane.state = if (failure is InterruptedException) ArchitectureTrialState.CANCELLED else ArchitectureTrialState.FAILED
        lane.failure = failure.message ?: failure.javaClass.simpleName
        NeuroLog.error("search", "search.trial.failed", failure, "searchId" to searchId, "trialId" to lane.id,
            "epoch" to lane.epoch, "execution" to config.execution)
    }

    private fun paddedParameters(lanes: List<Lane>): Long {
        val shapes = lanes.map { it.group.proposal.architecture.topology() }
        val widths = IntArray(shapes.first().size) { position -> shapes.maxOf { it[position] } }
        var result = 0L
        for (layer in 1..widths.lastIndex) {
            val value = widths[layer].toLong() * (widths[layer - 1] + 1L)
            if (value > MAX_RESIDENT_BYTES || result > MAX_RESIDENT_BYTES - value) return MAX_RESIDENT_BYTES + 1
            result += value
        }
        return result * lanes.size
    }

    private fun residentBytes(lanes: List<Lane>): Long {
        val shapes = lanes.map { it.group.proposal.architecture.topology() }
        val widthSum = shapes.first().indices.sumOf { position -> shapes.maxOf { it[position] }.toLong() }
        val bytes = if (config.precision == Neuro.TrainingPrecision.FP64) 8L else 4L
        val parameters = paddedParameters(lanes)
        if (parameters > MAX_RESIDENT_BYTES / 12 / bytes) return MAX_RESIDENT_BYTES + 1
        val activationRows = maxOf(data.training.size, data.validation.size).toLong() * lanes.size * 3
        if (widthSum > MAX_RESIDENT_BYTES / activationRows / bytes) return MAX_RESIDENT_BYTES + 1
        return parameters * 12 * bytes + widthSum * activationRows * bytes +
            (data.training.size + data.validation.size).toLong() * 3 * bytes +
            data.training.size.toLong() * lanes.size * 64 * 4
    }

    private fun rate(now: Long): Double = if (now <= started) 0.0 else aggregateEpochs * 1_000_000_000.0 / (now - started)
    private fun median(values: List<Double>): Double = values.sorted().let {
        if (it.size % 2 == 1) it[it.size / 2] else (it[it.size / 2 - 1] + it[it.size / 2]) / 2
    }

    companion object {
        private const val MAX_RETAINED_PARAMETERS = 8_000_000L
        private const val MAX_RESIDENT_BYTES = 128L * 1024 * 1024
    }
}
