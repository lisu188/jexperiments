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
    private static final EventType PHASE = EventType.getEventType(PhaseEvent.class);

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

    static void commitGeneration(GenerationEvent event, Evolution.GenerationStats stats) {
        if (event != null) {
            event.bestError = stats.bestError();
            event.averageError = stats.averageError();
            event.diversity = stats.geneDiversity();
            event.sigma = stats.mutationSigma();
            event.restarts = stats.restarts();
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

    static void commitSearch(SearchEvent event, Evolution.SearchResult result) {
        if (event != null) {
            event.generations = result.generations();
            event.bestError = result.bestError();
            event.converged = result.converged();
            event.fitnessEvaluations = result.fitnessEvaluations();
            event.restarts = result.restarts();
            event.end();
            if (event.shouldCommit()) {
                event.commit();
            }
        }
    }

    static PhaseEvent phase(String phase, long generation) {
        if (!PHASE.isEnabled()) {
            return null;
        }
        var event = new PhaseEvent();
        event.phase = phase;
        event.generation = generation;
        event.begin();
        return event;
    }

    static void commitPhase(PhaseEvent event) {
        if (event != null) {
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
        @Label("Best fitness") double bestError;
        @Label("Average fitness") double averageError;
        @Label("Gene diversity") double diversity;
        @Label("Mutation sigma") double sigma;
        @Label("Restarts") long restarts;
    }

    @Name("experiments.EvolutionSearch")
    @Label("Evolution search")
    @Category({"JExperiments", "Evolution"})
    @Enabled(false)
    static final class SearchEvent extends Event {
        @Label("Target fitness") double targetError;
        @Label("Maximum generations") long maxGenerations;
        @Label("Generations") long generations;
        @Label("Best fitness") double bestError;
        @Label("Fitness evaluations") long fitnessEvaluations;
        @Label("Restarts") long restarts;
        @Label("Converged") boolean converged;
    }

    @Name("experiments.EvolutionPhase")
    @Label("Evolution phase")
    @Category({"JExperiments", "Evolution"})
    @Enabled(false)
    static final class PhaseEvent extends Event {
        @Label("Phase") String phase;
        @Label("Generation") long generation;
    }
}
