import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.Executors;

import org.junit.jupiter.api.Test;

class ObservableConcurrentTreeConcurrencyTest {
    @Test
    void moveAndSnapshotAreAtomic() throws Exception {
        var tree = new ObservableConcurrentTree<Integer, Integer>(0, 0);
        tree.add(0, 1, 1);
        tree.add(0, 2, 2);
        tree.add(1, 3, 3);

        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            var start = new CountDownLatch(1);
            var futures = new ArrayList<java.util.concurrent.Future<?>>();

            for (int mover = 0; mover < 4; mover++) {
                var parity = mover;
                futures.add(executor.submit(() -> {
                    await(start);
                    for (int i = 0; i < 10_000; i++) {
                        tree.move(3, ((i + parity) & 1) == 0 ? 1 : 2);
                    }
                }));
            }
            for (int reader = 0; reader < 8; reader++) {
                futures.add(executor.submit(() -> {
                    await(start);
                    for (int i = 0; i < 5_000; i++) {
                        assertMovedNodeSnapshot(tree.snapshot());
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
    void loadStateAndSnapshotNeverMixGenerations() throws Exception {
        var stateA = starState(500, 10_000);
        var stateB = starState(750, 20_000);
        var tree = new ObservableConcurrentTree<Integer, Integer>();
        tree.loadState(stateA);

        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            var start = new CountDownLatch(1);
            var writer = executor.submit(() -> {
                await(start);
                for (int i = 0; i < 500; i++) {
                    tree.loadState((i & 1) == 0 ? stateB : stateA);
                }
            });

            var readers = new ArrayList<java.util.concurrent.Future<?>>();
            for (int r = 0; r < 32; r++) {
                readers.add(executor.submit(() -> {
                    await(start);
                    for (int i = 0; i < 500; i++) {
                        var snapshot = tree.snapshot();
                        var size = snapshot.entries().size();
                        assertTrue(size == 501 || size == 751, "mixed generation size=" + size);
                        int minChild = size == 501 ? 10_001 : 20_001;
                        int maxChild = size == 501 ? 10_500 : 20_750;
                        for (int p = 1; p < snapshot.entries().size(); p++) {
                            int id = snapshot.entries().get(p).id();
                            assertTrue(id >= minChild && id <= maxChild,
                                    "mixed generation id=" + id + " size=" + size);
                        }
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
    void removeVersusSubtreeStreamHasOnlyLinearizableOutcomes() throws Exception {
        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            for (int round = 0; round < 2_000; round++) {
                var tree = new ObservableConcurrentTree<Integer, Integer>(0, 0);
                tree.add(0, 1, 1);
                tree.add(1, 2, 2);
                tree.add(1, 3, 3);
                var barrier = new CyclicBarrier(2);

                var remove = executor.submit(() -> {
                    await(barrier);
                    tree.removeSubtree(1);
                });
                var read = executor.submit(() -> {
                    await(barrier);
                    try {
                        return tree.subtreeStream(1)
                                .map(ObservableConcurrentTree.Entry::id)
                                .toList();
                    } catch (IllegalArgumentException removedFirst) {
                        return List.<Integer>of();
                    }
                });

                remove.get();
                var result = read.get();
                assertTrue(result.isEmpty() || result.equals(List.of(1, 2, 3)),
                        "partial subtree observed: " + result);
                TreeTestSupport.assertValid(tree);
            }
        }
    }

    private static void assertMovedNodeSnapshot(
            ObservableConcurrentTree.Snapshot<Integer, Integer> snapshot) {
        var byId = new HashMap<Integer, ObservableConcurrentTree.Entry<Integer, Integer>>();
        for (var entry : snapshot.entries()) {
            byId.put(entry.id(), entry);
        }
        var x = byId.get(3);
        assertTrue(x.parentId() == 1 || x.parentId() == 2);
        int owners = 0;
        if (byId.get(1).children().contains(3)) {
            owners++;
            assertEquals(1, x.parentId());
        }
        if (byId.get(2).children().contains(3)) {
            owners++;
            assertEquals(2, x.parentId());
        }
        assertEquals(1, owners);
    }

    private static List<ObservableConcurrentTree.NodeState<Integer, Integer>> starState(int children, int base) {
        var state = new ArrayList<ObservableConcurrentTree.NodeState<Integer, Integer>>(children + 1);
        state.add(new ObservableConcurrentTree.NodeState<>(0, null, 0));
        for (int i = 1; i <= children; i++) {
            state.add(new ObservableConcurrentTree.NodeState<>(base + i, 0, i));
        }
        return state;
    }

    private static void await(CountDownLatch latch) {
        try {
            latch.await();
        } catch (InterruptedException exception) {
            Thread.currentThread().interrupt();
            throw new AssertionError(exception);
        }
    }

    private static void await(CyclicBarrier barrier) {
        try {
            barrier.await();
        } catch (Exception exception) {
            throw new AssertionError(exception);
        }
    }
}
