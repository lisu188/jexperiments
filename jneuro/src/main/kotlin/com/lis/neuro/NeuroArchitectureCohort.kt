package com.lis.neuro

/** Scores compatible seeds independently while one bounded search worker owns the packed cohort. */
internal fun evaluateArchitectureCohort(
    data: ArchitectureSearchData, config: ArchitectureSearchConfig, architecture: NetworkArchitecture, seeds: List<Long>,
    cancelled: () -> Boolean, progress: (Int, Int, Double) -> Unit, searchId: String?,
    openCohort: (List<Neuro>, TrainingBackend, Neuro.TrainingPrecision, Int, Int) -> NeuroTrainingCohort = NeuroTrainingCohort::open
): List<ArchitectureTrial> {
    val started = System.nanoTime()
    val records = seeds.map { CohortTrialRecord(it) }
    var device: TrainingDeviceInfo? = null
    fun checkpoint(index: Int) {
        val record = records[index]
        val model = checkNotNull(record.model)
        val training = model.trainingError()
        record.finalScore = if (data.evaluation == ArchitectureEvaluation.TRAINING_FIT) training else data.score(model)
        check(training.isFinite() && record.finalScore.isFinite()) { "Training produced a non-finite RMSE." }
        if (record.finalScore < record.best) {
            record.best = record.finalScore
            record.bestEpoch = record.epoch
            record.trainingAtBest = training
            record.snapshot = NeuroXorDiagnostics.capture(model, record.epoch, training)
        }
        record.history += ArchitectureCheckpoint(record.epoch, training, record.finalScore)
        if (record.history.size > 128) {
            val retained = record.history.filterIndexed { position, point ->
                position == 0 || position % 2 == 0 || position == record.history.lastIndex || point.epoch == record.bestEpoch
            }
            record.history.clear(); record.history.addAll(retained)
        }
        progress(index, record.epoch, record.best)
        NeuroLog.debug("search", "search.trial.checkpoint") { mapOf("searchId" to searchId, "trialId" to record.id,
            "cohort" to true, "epoch" to record.epoch, "trainingRmse" to training, "score" to record.finalScore) }
    }
    fun fail(record: CohortTrialRecord, failure: Exception) {
        record.epoch = record.model?.statistics()?.epochsTrained?.toInt() ?: record.epoch
        record.state = if (failure is InterruptedException) ArchitectureTrialState.CANCELLED else ArchitectureTrialState.FAILED
        record.failure = if (failure is InterruptedException) "" else failure.message ?: failure.javaClass.simpleName
        NeuroLog.error("search", "search.trial.failed", failure, "searchId" to searchId, "trialId" to record.id,
            "cohort" to true, "epoch" to record.epoch)
    }
    records.forEach { record ->
        NeuroLog.info("search", "search.trial.started", "searchId" to searchId, "trialId" to record.id,
            "topology" to architecture, "seed" to record.seed, "engine" to config.engine, "cohort" to true,
            "requestedBackend" to config.backend, "requestedPrecision" to config.precision, "batchSize" to config.batchSize)
    }
    try {
        if (cancelled()) records.forEach { it.state = ArchitectureTrialState.CANCELLED }
        else {
            records.forEach { it.model = data.newNetwork(architecture, config.hyperParameters, it.seed) }
            // Search already owns a bounded executor. One vector group runs on this worker, with no nested pool.
            openCohort(records.map { checkNotNull(it.model) }, config.backend, config.precision, config.batchSize, 1).use { cohort ->
                device = cohort.info
                val active = BooleanArray(records.size) { true }
                for (index in records.indices) {
                    val record = records[index]
                    NeuroLog.info("search", "search.trial.backend.ready", "searchId" to searchId, "trialId" to record.id,
                        "model" to record.model?.logId, "session" to record.model?.trainingSessionLogId,
                        "effectiveBackend" to cohort.info.backend, "effectivePrecision" to cohort.info.precision,
                        "engine" to cohort.info.engine, "simdBits" to cohort.info.simdBits, "sigmoid" to cohort.info.sigmoid,
                        "device" to cohort.info.name, "deviceIdentity" to cohort.info.identity, "kernelVersion" to cohort.info.kernelVersion)
                    try { checkpoint(index) } catch (failure: Exception) { fail(record, failure); active[index] = false }
                }
                while (active.any { it }) {
                    if (cancelled()) {
                        for (index in records.indices) if (active[index]) records[index].state = ArchitectureTrialState.CANCELLED
                        break
                    }
                    val boundary = records.indices.filter { active[it] }.minOf { index ->
                        val epoch = records[index].epoch
                        minOf(config.maxEpochs - epoch, config.checkEvery - epoch % config.checkEvery)
                    }
                    val results = cohort.trainChunk(TrainingChunkRequest(boundary, checkEvery = boundary, cancelled = cancelled), active)
                    for (index in records.indices) if (active[index]) {
                        val record = records[index]
                        val result = results[index]
                        try {
                            check(result.committedEpochs in 0..boundary) { "Cohort crossed a search scoring boundary." }
                            record.epoch += result.committedEpochs
                            check(result.rmse.isFinite()) { "Training produced a non-finite RMSE." }
                            if (result.termination == TrainingTermination.CANCELLED) {
                                record.state = ArchitectureTrialState.CANCELLED; active[index] = false
                            } else {
                                if (result.committedEpochs > 0 && (record.epoch % config.checkEvery == 0 || record.epoch == config.maxEpochs)) checkpoint(index)
                                if (record.epoch == config.maxEpochs) {
                                    record.state = ArchitectureTrialState.COMPLETED; active[index] = false
                                }
                            }
                        } catch (failure: Exception) { fail(record, failure); active[index] = false }
                    }
                }
            }
        }
    } catch (failure: Exception) {
        if (failure is InterruptedException) Thread.currentThread().interrupt()
        records.forEach { fail(it, failure) }
    }
    return records.map { record ->
        NeuroLog.info("search", "search.trial.finished", "searchId" to searchId, "trialId" to record.id,
            "cohort" to true, "state" to record.state, "epochs" to record.epoch, "bestEpoch" to record.bestEpoch, "bestRmse" to record.best)
        ArchitectureTrial(record.seed, record.state, record.epoch, record.bestEpoch, record.best, record.trainingAtBest,
            record.finalScore, record.epoch.toLong() * data.training.size, System.nanoTime() - started, record.history,
            record.snapshot, record.failure, device, record.id, cohort = true)
    }
}

private class CohortTrialRecord(val seed: Long) {
    val id = NeuroLog.id("trial")
    var model: Neuro? = null
    var state = ArchitectureTrialState.CANCELLED
    var epoch = 0
    var bestEpoch = 0
    var best = Double.POSITIVE_INFINITY
    var trainingAtBest = Double.POSITIVE_INFINITY
    var finalScore = Double.POSITIVE_INFINITY
    var snapshot: NeuroXorDiagnostics.Snapshot? = null
    var failure = ""
    val history = ArrayList<ArchitectureCheckpoint>()
}
