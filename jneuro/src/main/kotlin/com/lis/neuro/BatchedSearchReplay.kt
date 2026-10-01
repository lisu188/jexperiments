package com.lis.neuro

/** Recreates the tensor geometry that produced a checkpoint, including each resident-batch rebind. */
internal object BatchedSearchReplay {
    fun replay(model: Neuro, data: ArchitectureSearchData, config: ArchitectureSearchConfig,
               trial: ArchitectureTrial, verifyDevice: (TrainingDeviceInfo) -> Unit,
               cancelled: () -> Boolean,
               open: (Array<NeuroTrainingState>, IntArray) -> TensorFlowSearchCohort = { states, padded ->
                   val device = requireNotNull(trial.deviceInfo)
                   TensorFlowMath.searchCohort(states, config.hyperParameters,
                       Neuro.TrainingPrecision.valueOf(device.precision), device.backend,
                       DoubleArray(data.validation.size * 2) { i -> data.validation[i / 2].let { if (i % 2 == 0) it.x else it.y } },
                       DoubleArray(data.validation.size) { data.validation[it].target }, config.batchSize, padded)
               }): Double? {
        require(trial.bestEpoch in 0..trial.epochs && trial.batchSegments.isNotEmpty()) {
            "Batched replay requires recorded training geometry."
        }
        val topology = model.exportTrainingState(shareDataset = true).topology
        var covered = 0
        for (segment in trial.batchSegments) {
            require(segment.startEpoch == covered && segment.endEpoch >= covered && segment.endEpoch <= trial.epochs) {
                "Batched replay segments must cover consecutive committed epochs."
            }
            require(segment.capacity in 1..1024 && segment.lane in 0 until segment.capacity &&
                segment.paddedTopology.size == topology.size &&
                segment.paddedTopology.first() == topology.first() && segment.paddedTopology.last() == topology.last() &&
                topology.indices.all { segment.paddedTopology[it] >= topology[it] }) { "Invalid recorded tensor geometry." }
            val limit = 128L * 1024 * 1024 / 64 / segment.capacity
            var parameters = 0L
            for ((input, output) in segment.paddedTopology.zipWithNext()) {
                val added = (input.toLong() + 1) * output
                require(added <= limit - parameters) { "Recorded tensor geometry exceeds the replay memory limit." }
                parameters += added
            }
            covered = segment.endEpoch
        }
        require(covered >= trial.bestEpoch) { "Batched replay is missing the scored checkpoint." }
        var replayed = 0
        for (segment in trial.batchSegments) {
            if (cancelled()) return null
            val state = model.exportTrainingState(shareDataset = true)
            // Inactive lanes retain the original batch shape but never advance or contribute to reductions.
            open(Array(segment.capacity) { state }, segment.paddedTopology.toIntArray()).use { kernel ->
                verifyDevice(kernel.info)
                check(kernel.size == segment.capacity && kernel.paddedTopology.contentEquals(segment.paddedTopology.toIntArray())) {
                    "Replay did not recreate the recorded tensor geometry."
                }
                val active = BooleanArray(segment.capacity) { it == segment.lane }
                val end = minOf(segment.endEpoch, trial.bestEpoch)
                var nanosPerEpoch = 0L
                while (replayed < end) {
                    if (cancelled()) return null
                    val byTime = if (nanosPerEpoch == 0L) 1 else (50_000_000L / nanosPerEpoch).coerceIn(1, 64).toInt()
                    val count = minOf(64, byTime, end - replayed, config.checkEvery - replayed % config.checkEvery)
                    val order = model.reserveTrainingOrders(count)
                    val start = System.nanoTime()
                    val failed = kernel.advance(Array(segment.capacity) { if (active[it]) order else emptyArray() }, active)
                    check(failed.size == segment.capacity && !failed[segment.lane]) { "Batched replay produced non-finite training state." }
                    model.commitTrainingChunk(kernel.exportState(segment.lane), count, evaluateError = false)
                    replayed += count
                    nanosPerEpoch = maxOf(1L, (System.nanoTime() - start) / count)
                }
                if (replayed == trial.bestEpoch) {
                    val metrics = kernel.score(active)
                    check(!metrics.failed[segment.lane]) { "Batched replay scoring failed." }
                    if (replayed == 0) model.publishInitialTrainingState(kernel.exportState(segment.lane))
                    model.recordTrainingError(metrics.trainingRmse[segment.lane])
                    return if (data.evaluation == ArchitectureEvaluation.TRAINING_FIT) metrics.trainingRmse[segment.lane]
                        else metrics.validationRmse[segment.lane]
                }
            }
        }
        error("Batched replay did not reach the scored checkpoint.")
    }
}
