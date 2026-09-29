package com.lis.neuro

import jdk.jfr.Category
import jdk.jfr.Enabled
import jdk.jfr.Event
import jdk.jfr.EventType
import jdk.jfr.Label
import jdk.jfr.Name

internal object NeuroJfr {
    private val trainingEpochType = EventType.getEventType(TrainingEpochEvent::class.java)
    private val trainingRunType = EventType.getEventType(TrainingRunEvent::class.java)

    fun trainingEpoch(epoch: Long, samples: Int): TrainingEpochEvent? {
        if (!trainingEpochType.isEnabled) return null
        return TrainingEpochEvent().apply {
            this.epoch = epoch
            this.samples = samples
            begin()
        }
    }
    fun commitTrainingEpoch(event: TrainingEpochEvent?, error: Double) {
        event?.apply {
            this.error = error
            end()
            if (shouldCommit()) commit()
        }
    }
    fun trainingRun(targetError: Double, maxEpochs: Int): TrainingRunEvent? {
        if (!trainingRunType.isEnabled) return null
        return TrainingRunEvent().apply {
            this.targetError = targetError
            this.maxEpochs = maxEpochs
            begin()
        }
    }
    fun commitTrainingRun(event: TrainingRunEvent?, epochs: Int, error: Double, converged: Boolean) {
        event?.apply {
            this.epochs = epochs
            this.error = error
            this.converged = converged
            end()
            if (shouldCommit()) commit()
        }
    }
    @Name("experiments.NeuroTrainingEpoch")
    @Label("Neuro training epoch")
    @Category("JExperiments", "JNeuro")
    @Enabled(false)
    class TrainingEpochEvent : Event() {
        @JvmField @Label("Epoch") var epoch = 0L
        @JvmField @Label("Samples") var samples = 0
        @JvmField @Label("RMSE") var error = 0.0
    }
    @Name("experiments.NeuroTrainingRun")
    @Label("Neuro training run")
    @Category("JExperiments", "JNeuro")
    @Enabled(false)
    class TrainingRunEvent : Event() {
        @JvmField @Label("Target RMSE") var targetError = 0.0
        @JvmField @Label("Maximum epochs") var maxEpochs = 0
        @JvmField @Label("Epochs") var epochs = 0
        @JvmField @Label("Final RMSE") var error = 0.0
        @JvmField @Label("Converged") var converged = false
    }
}
