package com.lis;

import jdk.jfr.Category;
import jdk.jfr.Enabled;
import jdk.jfr.Event;
import jdk.jfr.EventType;
import jdk.jfr.Label;
import jdk.jfr.Name;

final class EvolutionJfr {
    private static final EventType GENERATION = EventType.getEventType(GenerationEvent.class);
    private static final EventType SEARCH = EventType.getEventType(SearchEvent.class);

    private EvolutionJfr() {
    }

    static GenerationEvent generation(long generation, int populationSize, int geneCount) {
        if (!GENERATION.isEnabled()) {
            return null;
        }
        var event = new GenerationEvent();
        event.generation = generation;
        event.populationSize = populationSize;
        event.geneCount = geneCount;
        event.begin();
        return event;
    }

    static void commitGeneration(GenerationEvent event, double bestError, double averageError) {
        if (event != null) {
            event.bestError = bestError;
            event.averageError = averageError;
            event.end();
            if (event.shouldCommit()) {
                event.commit();
            }
        }
    }

    static SearchEvent search(double targetError, long maxGenerations) {
        if (!SEARCH.isEnabled()) {
            return null;
        }
        var event = new SearchEvent();
        event.targetError = targetError;
        event.maxGenerations = maxGenerations;
        event.begin();
        return event;
    }

    static void commitSearch(
            SearchEvent event,
            long generations,
            double bestError,
            boolean converged) {
        if (event != null) {
            event.generations = generations;
            event.bestError = bestError;
            event.converged = converged;
            event.end();
            if (event.shouldCommit()) {
                event.commit();
            }
        }
    }

    @Name("experiments.EvolutionGeneration")
    @Label("Evolution generation")
    @Category({"JExperiments", "Evolution"})
    @Enabled(false)
    static final class GenerationEvent extends Event {
        @Label("Generation") long generation;
        @Label("Population") int populationSize;
        @Label("Genes") int geneCount;
        @Label("Best MSE") double bestError;
        @Label("Average MSE") double averageError;
    }

    @Name("experiments.EvolutionSearch")
    @Label("Evolution search")
    @Category({"JExperiments", "Evolution"})
    @Enabled(false)
    static final class SearchEvent extends Event {
        @Label("Target MSE") double targetError;
        @Label("Maximum generations") long maxGenerations;
        @Label("Generations") long generations;
        @Label("Best MSE") double bestError;
        @Label("Converged") boolean converged;
    }
}
