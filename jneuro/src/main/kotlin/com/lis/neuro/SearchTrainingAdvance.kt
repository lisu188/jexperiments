package com.lis.neuro

/** Search owns scoring boundaries; a partial advance need not evaluate the training dataset. */
internal data class SearchAdvanceResult(
    val committedEpochs: Int,
    val rmse: Double?,
    val termination: TrainingTermination
)

internal interface SearchEpochAdvancer {
    fun advanceForSearch(request: TrainingChunkRequest): SearchAdvanceResult
}

/** Advance at most 64 epochs while retaining the caller's complete scoring-boundary request. */
internal fun advanceTrainingForSearch(session: NeuroTrainingSession, request: TrainingChunkRequest): SearchAdvanceResult {
    require(request.targetError == null) { "Search must complete its full trial budget without target stopping" }
    return if (session is SearchEpochAdvancer) session.advanceForSearch(request)
    else searchAdvanceFallback(session, request)
}

internal fun searchAdvanceFallback(session: NeuroTrainingSession, request: TrainingChunkRequest): SearchAdvanceResult {
    require(request.targetError == null) { "Search must complete its full trial budget without target stopping" }
    val result = session.trainChunk(request)
    return SearchAdvanceResult(result.committedEpochs,
        result.rmse.takeIf { result.committedEpochs > 0 && result.committedEpochs == request.maxEpochs }, result.termination)
}
