package com.lis.neuro;

import jdk.jfr.Category;
import jdk.jfr.Enabled;
import jdk.jfr.Event;
import jdk.jfr.EventType;
import jdk.jfr.Label;
import jdk.jfr.Name;

final class NeuroJfr {
    private static final EventType TRAINING_EPOCH = EventType.getEventType(TrainingEpochEvent.class);
    private static final EventType TRAINING_RUN = EventType.getEventType(TrainingRunEvent.class);

    private NeuroJfr() {
    }

    static TrainingEpochEvent trainingEpoch(long epoch, int samples) {
        if (!TRAINING_EPOCH.isEnabled()) {
            return null;
        }
        var event = new TrainingEpochEvent();
        event.epoch = epoch;
        event.samples = samples;
        event.begin();
        return event;
    }

    static void commitTrainingEpoch(TrainingEpochEvent event, double error) {
        if (event != null) {
            event.error = error;
            event.end();
            if (event.shouldCommit()) {
                event.commit();
            }
        }
    }

    static TrainingRunEvent trainingRun(double targetError, int maxEpochs) {
        if (!TRAINING_RUN.isEnabled()) {
            return null;
        }
        var event = new TrainingRunEvent();
        event.targetError = targetError;
        event.maxEpochs = maxEpochs;
        event.begin();
        return event;
    }

    static void commitTrainingRun(
            TrainingRunEvent event,
            int epochs,
            double error,
            boolean converged) {
        if (event != null) {
            event.epochs = epochs;
            event.error = error;
            event.converged = converged;
            event.end();
            if (event.shouldCommit()) {
                event.commit();
            }
        }
    }

    @Name("experiments.NeuroTrainingEpoch")
    @Label("Neuro training epoch")
    @Category({"JExperiments", "JNeuro"})
    @Enabled(false)
    static final class TrainingEpochEvent extends Event {
        @Label("Epoch") long epoch;
        @Label("Samples") int samples;
        @Label("RMSE") double error;
    }

    @Name("experiments.NeuroTrainingRun")
    @Label("Neuro training run")
    @Category({"JExperiments", "JNeuro"})
    @Enabled(false)
    static final class TrainingRunEvent extends Event {
        @Label("Target RMSE") double targetError;
        @Label("Maximum epochs") int maxEpochs;
        @Label("Epochs") int epochs;
        @Label("Final RMSE") double error;
        @Label("Converged") boolean converged;
    }
}
