package com.lis.neuro

/** Resident independent models; each leading tensor lane owns parameters, momentum and its best checkpoint. */
internal interface TensorFlowSearchCohort : AutoCloseable {
    val info: TrainingDeviceInfo
    val size: Int
    val paddedTopology: IntArray
    /** All active lanes advance the same complete epochs. A failed lane rolls back the whole chunk. */
    fun advance(orders: Array<Array<IntArray>>, active: BooleanArray = BooleanArray(size) { true }): BooleanArray
    /** Scores all lanes and retains improved best states on the device. No parameter arrays are returned. */
    fun score(active: BooleanArray = BooleanArray(size) { true }): TensorFlowCohortMetrics
    fun exportState(lane: Int, best: Boolean = false): NeuroTrainingState
    fun exportStates(lanes: IntArray, best: Boolean = false): Array<NeuroTrainingState>
}

internal data class TensorFlowCohortMetrics(
    val epochs: IntArray,
    val trainingRmse: DoubleArray,
    val validationRmse: DoubleArray,
    val bestScore: DoubleArray,
    val bestEpoch: IntArray,
    val failed: BooleanArray
)
