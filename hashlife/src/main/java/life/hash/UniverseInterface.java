package life.hash;

/**
 * The interface to which all of our Life interfaces will conform,
 * allowing us to use a uniform driver program.
 */
interface UniverseInterface {
    /**
     * Set a single bit; can only do this before running, and once
     * we've started running cannot change.
     */
    void setBit(int x, int y);

    /**
     * Run a step.
     */
    void runStep();

    /**
     * Display the current statistics about the current generation.
     * This should include the generation count and the population.
     */
    String stats();
}
