package com.lis.neuro

object CohortBenchmark {
    private val hp = Neuro.HyperParameters(0.6, 0.2, 1.0, 42)
    @JvmStatic fun main(args: Array<String>) {
        val rounds = args.firstOrNull()?.toInt() ?: 2
        val data = ArchitectureSearchData.split(NeuroLearningSets.create(NeuroLearningSets.Kind.SPIRAL, 42), 0.2, 42, "Spiral")
        val validationX = data.validation.flatMap { listOf(it.x, it.y) }.toDoubleArray()
        val validationY = data.validation.map { it.target }.toDoubleArray()
        for (family in listOf("shallow", "deep", "padded")) for (models in intArrayOf(5, 32, 128, 512)) {
            val startedInit = System.nanoTime()
            val states = Array(models) { lane ->
                val hidden = when(family) { "shallow" -> listOf(6); "deep" -> listOf(8,8,8); else -> listOf(1+lane%8, 1+(lane*3)%8, 1+(lane*5)%8) }
                data.newNetwork(NetworkArchitecture(hidden), hp, lane.toLong()+1).exportTrainingState(shareDataset=true)
            }
            val initialMs = (System.nanoTime()-startedInit)/1e6
            val orders = Array(models) { lane -> Array(5) { epoch -> IntArray(states[0].samples) { (it+lane+epoch)%states[0].samples } } }
            for (threads in intArrayOf(1,2,4)) repeat(rounds) { round ->
                val open = System.nanoTime()
                TensorFlowMath.searchCohort(states,hp,Neuro.TrainingPrecision.FP64,TrainingBackend.CPU,validationX,validationY,1,intraOpThreads=threads).use { cohort ->
                    val openMs=(System.nanoTime()-open)/1e6
                    check(cohort.advance(Array(models){orders[it].copyOfRange(0,2)}).none { it })
                    cohort.score()
                    val started=System.nanoTime()
                    check(cohort.advance(orders).none { it })
                    val metric=cohort.score()
                    check(metric.trainingRmse.all { it.isFinite() })
                    val ms=(System.nanoTime()-started)/1e6
                    println("{\"family\":\"$family\",\"models\":$models,\"threads\":$threads,\"round\":$round,\"initialMs\":$initialMs,\"openMs\":$openMs,\"elapsedMs\":$ms,\"epochsPerSecond\":${models*5*1000/ms},\"rmse0\":${metric.validationRmse[0]}}")
                }
            }
        }
        TensorFlowMath.clearInferenceCache()
    }
}
