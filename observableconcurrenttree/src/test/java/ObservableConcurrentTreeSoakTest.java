import static org.junit.jupiter.api.Assertions.assertTrue;

import java.time.Duration;
import java.util.ArrayList;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.LongAdder;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("soak")
class ObservableConcurrentTreeSoakTest {
    @Test
    void longRunningVirtualThreadMutationAndReadSoak() throws Exception {
        int seconds = integerProperty("tree.soak.seconds", 60);
        int workers = integerProperty("tree.soak.workers", 2_000);
        int initialNodes = integerProperty("tree.soak.initialNodes", 100_000);

        var tree = new ObservableConcurrentTree<Integer, Integer>(0, 0);
        for (int i = 1; i <= initialNodes; i++) {
            tree.add(0, i, i);
        }

        var nextId = new AtomicInteger(initialNodes + 1);
        var operations = new LongAdder();
        long deadline = System.nanoTime() + Duration.ofSeconds(seconds).toNanos();
        long begin = System.nanoTime();

        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            var futures = new ArrayList<java.util.concurrent.Future<?>>(workers);
            for (int worker = 0; worker < workers; worker++) {
                int workerId = worker;
                futures.add(executor.submit(() -> {
                    var random = ThreadLocalRandom.current();
                    while (System.nanoTime() < deadline) {
                        switch (workerId % 10) {
                            case 0 -> {
                                int id = nextId.getAndIncrement();
                                tree.add(0, id, id);
                                tree.removeSubtree(id);
                            }
                            case 1, 2 -> {
                                int id = 1 + random.nextInt(initialNodes);
                                tree.update(id, random.nextInt());
                            }
                            case 3 -> tree.snapshot();
                            case 4 -> tree.stream().filter(entry -> (entry.id() & 1023) == 0).count();
                            case 5 -> tree.childrenStream(0).limit(32).count();
                            case 6 -> {
                                try {
                                    tree.toJson();
                                } catch (com.fasterxml.jackson.core.JsonProcessingException exception) {
                                    throw new AssertionError(exception);
                                }
                            }
                            default -> {
                                int id = random.nextInt(initialNodes + 1);
                                tree.contains(id);
                                tree.getRootId();
                                tree.getVersion();
                                tree.size();
                            }
                        }
                        operations.increment();
                    }
                }));
            }
            for (var future : futures) {
                future.get();
            }
        }

        double elapsed = (System.nanoTime() - begin) / 1_000_000_000.0;
        System.out.printf(
                "SOAK seconds=%d workers=%,d initialNodes=%,d operations=%,d throughput=%,.0f ops/s%n",
                seconds, workers, initialNodes, operations.sum(), operations.sum() / elapsed);
        assertTrue(operations.sum() > 0);
        TreeTestSupport.assertValid(tree);
    }

    private static int integerProperty(String name, int fallback) {
        return Integer.parseInt(System.getProperty(name, Integer.toString(fallback)));
    }
}
