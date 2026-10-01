package com.lis.neuro

import java.util.concurrent.CompletableFuture
import java.util.concurrent.Executor

/** Search-only advancement preserves scoring boundaries without evaluating RMSE after every epoch. */
internal class OptimizedArchitectureTrial(
    private val data: ArchitectureSearchData,
    private val config: ArchitectureSearchConfig,
    private val architecture: NetworkArchitecture,
    private val seed: Long,
    private val cancelled: () -> Boolean,
    private val progress: (Int, Double) -> Unit,
    private val searchId: String?
) {
    private val trialId = NeuroLog.id("trial")
    private val started = System.nanoTime()
    private var model: Neuro? = null
    private var device: TrainingDeviceInfo? = null
    private var epoch = 0
    private var bestEpoch = 0
    private var best = Double.POSITIVE_INFINITY
    private var trainingAtBest = Double.POSITIVE_INFINITY
    private var finalScore = Double.POSITIVE_INFINITY
    private var snapshot: NeuroXorDiagnostics.Snapshot? = null
    private val history = ArrayList<ArchitectureCheckpoint>()
    private var route = ArchitectureTrialRoute.SESSION
    private var openNanos = 0L
    private var trainingNanos = 0L
    private var scoringNanos = 0L
    private var snapshotNanos = 0L
    private var closeNanos = 0L

    private fun create(): Neuro {
        NeuroLog.info("search", "search.trial.started", "searchId" to searchId, "trialId" to trialId,
            "topology" to architecture, "seed" to seed, "execution" to config.execution, "route" to route,
            "requestedBackend" to config.backend, "requestedPrecision" to config.precision, "batchSize" to config.batchSize)
        return data.newNetwork(architecture, config.hyperParameters, seed).also { model = it }
    }

    private fun opened(info: TrainingDeviceInfo) {
        device = info
        NeuroLog.info("search", "search.trial.backend.ready", "searchId" to searchId, "trialId" to trialId,
            "model" to model?.logId, "session" to model?.trainingSessionLogId, "execution" to config.execution,
            "route" to route, "effectiveBackend" to info.backend, "effectivePrecision" to info.precision,
            "engine" to info.engine, "simdBits" to info.simdBits, "sigmoid" to info.sigmoid,
            "device" to info.name, "deviceIdentity" to info.identity, "kernelVersion" to info.kernelVersion)
    }

    private fun checkpoint(rmse: Double? = null) {
        val network = checkNotNull(model)
        val scoringStarted = System.nanoTime()
        val training = rmse ?: network.trainingError()
        finalScore = if (data.evaluation == ArchitectureEvaluation.TRAINING_FIT) training else data.score(network)
        scoringNanos += System.nanoTime() - scoringStarted
        check(training.isFinite() && finalScore.isFinite()) { "Training produced a non-finite RMSE." }
        if (finalScore < best) {
            best = finalScore; bestEpoch = epoch; trainingAtBest = training
            val snapshotStarted = System.nanoTime()
            snapshot = NeuroXorDiagnostics.capture(network, epoch, training)
            snapshotNanos += System.nanoTime() - snapshotStarted
        }
        history += ArchitectureCheckpoint(epoch, training, finalScore)
        if (history.size > 128) {
            val retained = history.filterIndexed { index, point ->
                index == 0 || index % 2 == 0 || index == history.lastIndex || point.epoch == bestEpoch
            }
            history.clear(); history.addAll(retained)
        }
        NeuroLog.debug("search", "search.trial.checkpoint") { mapOf("searchId" to searchId, "trialId" to trialId,
            "epoch" to epoch, "trainingRmse" to training, "score" to finalScore, "bestRmse" to best) }
        progress(epoch, best)
    }

    private fun request() = TrainingChunkRequest(minOf(config.maxEpochs - epoch,
        config.checkEvery - epoch % config.checkEvery), cancelled = cancelled)

    private fun advanced(request: TrainingChunkRequest, result: SearchAdvanceResult): Boolean {
        check(result.committedEpochs in 0..minOf(64, request.maxEpochs)) { "Training crossed a search scoring boundary." }
        epoch += result.committedEpochs
        if (result.termination == TrainingTermination.CANCELLED) return false
        check(result.committedEpochs > 0) { "Search advancement made no progress." }
        if (epoch % config.checkEvery == 0 || epoch == config.maxEpochs) checkpoint(result.rmse)
        return true
    }

    private fun finished(state: ArchitectureTrialState, failure: Throwable? = null): ArchitectureTrial {
        epoch = model?.statistics()?.epochsTrained?.toInt() ?: epoch
        if (failure != null) NeuroLog.error("search", "search.trial.failed", failure,
            "searchId" to searchId, "trialId" to trialId, "epoch" to epoch, "route" to route)
        NeuroLog.info("search", "search.trial.finished", "searchId" to searchId, "trialId" to trialId,
            "state" to state, "epoch" to epoch, "bestEpoch" to bestEpoch, "bestRmse" to best,
            "execution" to config.execution, "route" to route)
        return ArchitectureTrial(seed, state, epoch, bestEpoch, best, trainingAtBest, finalScore,
            epoch.toLong() * data.training.size, System.nanoTime() - started, history, snapshot,
            if (state == ArchitectureTrialState.FAILED) failure?.message ?: failure?.javaClass?.simpleName ?: "Training failed" else "",
            device, trialId, execution = config.execution, route = route,
            timings = if (device == null) null else ArchitectureTrialTimings(openNanos, trainingNanos, scoringNanos, snapshotNanos, closeNanos))
    }

    private fun close(session: NeuroTrainingSession) {
        val started = System.nanoTime()
        try { session.close() } finally { closeNanos += System.nanoTime() - started }
    }

    private inline fun <T> useSession(session: NeuroTrainingSession, action: () -> T): T {
        var problem: Throwable? = null
        try { return action() }
        catch (failure: Throwable) { problem = failure; throw failure }
        finally {
            try { close(session) } catch (cleanup: Throwable) {
                if (problem == null) throw cleanup else problem.addSuppressed(cleanup)
            }
        }
    }

    fun evaluate(open: (Neuro, TrainingBackend, Neuro.TrainingPrecision, Int, TrainingEngine) -> NeuroTrainingSession): ArchitectureTrial {
        try {
            if (cancelled()) return finished(ArchitectureTrialState.CANCELLED)
            val openStarted = System.nanoTime()
            val network = create()
            val session = open(network, config.backend, config.precision, config.batchSize, config.engine)
            openNanos += System.nanoTime() - openStarted
            val state = useSession(session) {
                opened(session.info); checkpoint()
                while (epoch < config.maxEpochs) {
                    if (cancelled()) return@useSession ArchitectureTrialState.CANCELLED
                    val request = request()
                    val trainingStarted = System.nanoTime()
                    val result = try { advanceTrainingForSearch(session, request) }
                        finally { trainingNanos += System.nanoTime() - trainingStarted }
                    if (!advanced(request, result)) return@useSession ArchitectureTrialState.CANCELLED
                }
                ArchitectureTrialState.COMPLETED
            }
            return finished(state)
        } catch (failure: Exception) {
            if (failure is InterruptedException) Thread.currentThread().interrupt()
            return finished(if (failure is InterruptedException) ArchitectureTrialState.CANCELLED else ArchitectureTrialState.FAILED, failure)
        }
    }

    fun evaluateAsync(service: SearchCudaService, executor: Executor): CompletableFuture<ArchitectureTrial> {
        route = ArchitectureTrialRoute.CUDA_QUEUE
        val completion = CompletableFuture<ArchitectureTrial>()
        var session: SearchCudaSession? = null
        fun finish(state: ArchitectureTrialState, problem: Throwable? = null) {
            var failure = problem
            try { session?.let(::close) } catch (cleanup: Throwable) {
                if (failure == null) failure = cleanup else failure.addSuppressed(cleanup)
            }
            if (failure != null && failure !is Exception) completion.completeExceptionally(failure)
            else completion.complete(finished(if (failure == null) state else if (failure is InterruptedException)
                ArchitectureTrialState.CANCELLED else ArchitectureTrialState.FAILED, failure))
        }
        fun dispatch(action: () -> Unit) {
            try { executor.execute { try { action() } catch (failure: Throwable) { finish(ArchitectureTrialState.FAILED, failure) } } }
            catch (failure: Throwable) { finish(ArchitectureTrialState.FAILED, failure) }
        }
        fun advance() {
            if (cancelled()) { finish(ArchitectureTrialState.CANCELLED); return }
            if (epoch == config.maxEpochs) { finish(ArchitectureTrialState.COMPLETED); return }
            val request = request()
            val trainingStarted = System.nanoTime()
            checkNotNull(session).advanceForSearchAsync(request).whenComplete { result, failure ->
                val elapsed = System.nanoTime() - trainingStarted
                dispatch {
                    trainingNanos += elapsed
                    if (failure != null) {
                        val cause = if (failure is java.util.concurrent.CompletionException || failure is java.util.concurrent.ExecutionException)
                            failure.cause ?: failure else failure
                        finish(ArchitectureTrialState.FAILED, cause)
                    }
                    else if (!advanced(request, result)) finish(ArchitectureTrialState.CANCELLED)
                    else advance()
                }
            }
        }
        dispatch {
            if (cancelled()) finish(ArchitectureTrialState.CANCELLED)
            else {
                val openStarted = System.nanoTime()
                session = service.openSession(create())
                openNanos += System.nanoTime() - openStarted
                opened(checkNotNull(session).info); checkpoint(); advance()
            }
        }
        return completion
    }
}
