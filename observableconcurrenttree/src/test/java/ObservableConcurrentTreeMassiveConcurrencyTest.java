import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("massive")
class ObservableConcurrentTreeMassiveConcurrencyTest {
    @Test
    void oneHundredThousandStyleParallelUniqueAdds() throws Exception {
        int tasks = integerProperty("tree.massive.addTasks", 100_000);
        var tree = new ObservableConcurrentTree<Integer, Integer>(0, 0);
        var start = new CountDownLatch(1);
        long begin = System.nanoTime();

        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            var futures = new ArrayList<java.util.concurrent.Future<?>>(tasks);
            for (int id = 1; id <= tasks; id++) {
                int nodeId = id;
                futures.add(executor.submit(() -> {
                    await(start);
                    tree.add(0, nodeId, nodeId);
                }));
            }
            start.countDown();
            for (var future : futures) {
                future.get();
            }
        }

        double seconds = (System.nanoTime() - begin) / 1_000_000_000.0;
        System.out.printf("MASSIVE unique-add tasks=%,d time=%.3fs throughput=%,.0f ops/s%n",
                tasks, seconds, tasks / seconds);
        assertEquals(tasks + 1, tree.size());
        assertEquals(tasks + 1L, tree.getVersion());
        TreeTestSupport.assertValid(tree);
    }

    @Test
    void highContentionDuplicateAddHasExactlyOneWinner() throws Exception {
        int tasks = Math.max(1_000, integerProperty("tree.massive.addTasks", 100_000) / 10);
        var tree = new ObservableConcurrentTree<Integer, Integer>(0, 0);
        var winners = new AtomicInteger();
        var rejected = new AtomicInteger();
        var start = new CountDownLatch(1);

        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            var futures = new ArrayList<java.util.concurrent.Future<?>>(tasks);
            for (int i = 0; i < tasks; i++) {
                futures.add(executor.submit(() -> {
                    await(start);
                    try {
                        tree.add(0, 1, 1);
                        winners.incrementAndGet();
                    } catch (IllegalArgumentException expected) {
                        rejected.incrementAndGet();
                    }
                }));
            }
            start.countDown();
            for (var future : futures) {
                future.get();
            }
        }

        assertEquals(1, winners.get());
        assertEquals(tasks - 1, rejected.get());
        assertEquals(2, tree.size());
        TreeTestSupport.assertValid(tree);
    }

    @Test
    void massiveMixedVirtualReadersAndWriters() throws Exception {
        int readerTasks = integerProperty("tree.massive.readerTasks", 5_000);
        int writerTasks = integerProperty("tree.massive.writerTasks", 500);
        int opsPerWriter = integerProperty("tree.massive.opsPerWriter", 40);
        var tree = new ObservableConcurrentTree<Integer, Integer>(0, 0);
        var start = new CountDownLatch(1);
        var readOps = new java.util.concurrent.atomic.LongAdder();
        var writeOps = new java.util.concurrent.atomic.LongAdder();
        long begin = System.nanoTime();

        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            var futures = new ArrayList<java.util.concurrent.Future<?>>(readerTasks + writerTasks);
            for (int writer = 0; writer < writerTasks; writer++) {
                int writerId = writer;
                futures.add(executor.submit(() -> {
                    await(start);
                    int base = 1 + writerId * opsPerWriter;
                    for (int i = 0; i < opsPerWriter; i++) {
                        int id = base + i;
                        tree.add(0, id, id);
                        if ((i & 7) == 0) {
                            tree.update(id, -id);
                        }
                        writeOps.increment();
                    }
                }));
            }
            for (int reader = 0; reader < readerTasks; reader++) {
                int readerId = reader;
                futures.add(executor.submit(() -> {
                    await(start);
                    long local = 0;
                    for (int i = 0; i < 32; i++) {
                        local += tree.size();
                        local += tree.getVersion();
                        tree.contains((readerId + i) % Math.max(1, writerTasks * opsPerWriter));
                        tree.getRoot();
                        if ((i & 15) == 0) {
                            local += tree.snapshot().entries().size();
                        }
                        readOps.increment();
                    }
                    if (local == Long.MIN_VALUE) {
                        throw new AssertionError();
                    }
                }));
            }

            start.countDown();
            for (var future : futures) {
                future.get();
            }
        }

        double seconds = (System.nanoTime() - begin) / 1_000_000_000.0;
        System.out.printf(
                "MASSIVE mixed readers=%,d writers=%,d writes=%,d reads=%,d time=%.3fs%n",
                readerTasks, writerTasks, writeOps.sum(), readOps.sum(), seconds);
        assertEquals(1 + writerTasks * opsPerWriter, tree.size());
        TreeTestSupport.assertValid(tree);
    }

    @Test
    void loadStateAtomicityUnderMassiveReaderFanout() throws Exception {
        int readerTasks = integerProperty("tree.massive.loadReaders", 250);
        var stateA = starState(1_000, 10_000);
        var stateB = starState(1_500, 20_000);
        var tree = new ObservableConcurrentTree<Integer, Integer>();
        tree.loadState(stateA);
        var start = new CountDownLatch(1);

        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            var writer = executor.submit(() -> {
                await(start);
                for (int i = 0; i < 200; i++) {
                    tree.loadState((i & 1) == 0 ? stateB : stateA);
                }
            });
            var readers = new ArrayList<java.util.concurrent.Future<?>>(readerTasks);
            for (int r = 0; r < readerTasks; r++) {
                readers.add(executor.submit(() -> {
                    await(start);
                    for (int i = 0; i < 50; i++) {
                        int size = tree.snapshot().entries().size();
                        assertTrue(size == 1_001 || size == 1_501, "partial loadState size=" + size);
                    }
                }));
            }
            start.countDown();
            writer.get();
            for (var reader : readers) {
                reader.get();
            }
        }
        TreeTestSupport.assertValid(tree);
    }

    @Test
    void massiveMovePingPongWithSnapshotReaders() throws Exception {
        int readerTasks = Math.max(100, integerProperty("tree.massive.readerTasks", 5_000) / 10);
        int moverTasks = Math.max(16, integerProperty("tree.massive.writerTasks", 500) / 10);
        var tree = new ObservableConcurrentTree<Integer, Integer>(0, 0);
        tree.add(0, 1, 1);
        tree.add(0, 2, 2);
        tree.add(1, 3, 3);
        var start = new CountDownLatch(1);

        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            var futures = new ArrayList<java.util.concurrent.Future<?>>(readerTasks + moverTasks);
            for (int mover = 0; mover < moverTasks; mover++) {
                int parity = mover;
                futures.add(executor.submit(() -> {
                    await(start);
                    for (int i = 0; i < 2_000; i++) {
                        tree.move(3, ((i + parity) & 1) == 0 ? 1 : 2);
                    }
                }));
            }
            for (int reader = 0; reader < readerTasks; reader++) {
                futures.add(executor.submit(() -> {
                    await(start);
                    for (int i = 0; i < 200; i++) {
                        var snapshot = tree.snapshot();
                        var x = snapshot.entries().stream()
                                .filter(entry -> entry.id() == 3)
                                .findFirst()
                                .orElseThrow();
                        assertTrue(x.parentId() == 1 || x.parentId() == 2);
                        long owners = snapshot.entries().stream()
                                .filter(entry -> entry.id() == 1 || entry.id() == 2)
                                .filter(entry -> entry.children().contains(3))
                                .count();
                        assertEquals(1, owners);
                    }
                }));
            }
            start.countDown();
            for (var future : futures) {
                future.get();
            }
        }
        TreeTestSupport.assertValid(tree);
    }

    @Test
    void observerRegistrationChurnDuringWritesRemainsSafe() throws Exception {
        int churnTasks = Math.max(32, integerProperty("tree.massive.writerTasks", 500) / 5);
        int writerTasks = Math.max(32, integerProperty("tree.massive.writerTasks", 500) / 5);
        var tree = new ObservableConcurrentTree<Integer, Integer>(0, 0);
        var observed = new java.util.concurrent.atomic.LongAdder();
        var start = new CountDownLatch(1);

        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            var futures = new ArrayList<java.util.concurrent.Future<?>>(churnTasks + writerTasks);
            for (int i = 0; i < churnTasks; i++) {
                futures.add(executor.submit(() -> {
                    await(start);
                    ObservableConcurrentTree.Observer<Integer, Integer> observer =
                            change -> observed.increment();
                    for (int round = 0; round < 1_000; round++) {
                        tree.addObserver(observer);
                        if ((round & 3) == 0) {
                            tree.getRoot();
                        }
                        tree.removeObserver(observer);
                    }
                }));
            }
            for (int writer = 0; writer < writerTasks; writer++) {
                int writerId = writer;
                futures.add(executor.submit(() -> {
                    await(start);
                    int base = 1_000_000 + writerId * 500;
                    for (int j = 0; j < 500; j++) {
                        int id = base + j;
                        tree.add(0, id, id);
                    }
                }));
            }
            start.countDown();
            for (var future : futures) {
                future.get();
            }
        }

        assertEquals(1 + writerTasks * 500, tree.size());
        TreeTestSupport.assertValid(tree);
        System.out.printf("MASSIVE observer-churn tasks=%d writers=%d observedCallbacks=%,d%n",
                churnTasks, writerTasks, observed.sum());
    }

    @Test
    void jsonStormWhileWritersMutateProducesOnlyValidTrees() throws Exception {
        int serializers = Math.max(32, integerProperty("tree.massive.loadReaders", 250) / 2);
        int writers = Math.max(8, integerProperty("tree.massive.writerTasks", 500) / 20);
        var tree = new ObservableConcurrentTree<Integer, Integer>(0, 0);
        for (int i = 1; i <= 2_000; i++) {
            tree.add(0, i, i);
        }
        var start = new CountDownLatch(1);

        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            var futures = new ArrayList<java.util.concurrent.Future<?>>(serializers + writers);
            for (int writer = 0; writer < writers; writer++) {
                int writerId = writer;
                futures.add(executor.submit(() -> {
                    await(start);
                    for (int i = 0; i < 500; i++) {
                        int id = 1 + Math.floorMod(writerId * 997 + i, 2_000);
                        tree.update(id, writerId + i);
                    }
                }));
            }
            for (int serializer = 0; serializer < serializers; serializer++) {
                futures.add(executor.submit(() -> {
                    await(start);
                    for (int i = 0; i < 30; i++) {
                        try {
                            var restored = ObservableConcurrentTree.fromJson(
                                    tree.toJson(), Integer.class, Integer.class);
                            assertEquals(2_001, restored.size());
                            TreeTestSupport.assertValid(restored);
                        } catch (Exception exception) {
                            throw new AssertionError(exception);
                        }
                    }
                }));
            }
            start.countDown();
            for (var future : futures) {
                future.get();
            }
        }
        TreeTestSupport.assertValid(tree);
    }

    private static List<ObservableConcurrentTree.NodeState<Integer, Integer>> starState(int children, int base) {
        var state = new ArrayList<ObservableConcurrentTree.NodeState<Integer, Integer>>(children + 1);
        state.add(new ObservableConcurrentTree.NodeState<>(0, null, 0));
        for (int i = 1; i <= children; i++) {
            state.add(new ObservableConcurrentTree.NodeState<>(base + i, 0, i));
        }
        return state;
    }

    private static int integerProperty(String name, int fallback) {
        return Integer.parseInt(System.getProperty(name, Integer.toString(fallback)));
    }

    private static void await(CountDownLatch latch) {
        try {
            latch.await();
        } catch (InterruptedException exception) {
            Thread.currentThread().interrupt();
            throw new AssertionError(exception);
        }
    }
}
