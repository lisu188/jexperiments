import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

public final class ObservableConcurrentTreeVerification {
    public static void main(String[] args) throws Exception {
        verifyCoreOperations();
        verifyObserverSemantics();
        verifyBulkLoadValidation();
        verifySealedEvents();
        verifyStreamApi();
        verifyJsonSerialization();
        verifyConcurrentAccess();
        System.out.println("ObservableConcurrentTree verification passed");
    }

    private static void verifyCoreOperations() {
        var tree = new ObservableConcurrentTree<Integer, String>(0, "root");
        tree.add(0, 1, "one");
        tree.add(0, 2, "two");
        tree.add(1, 3, "three");
        require(tree.size() == 4, "size after adds");
        require(tree.getParentId(3) == 1, "parent before move");
        tree.move(3, 2);
        require(tree.getParentId(3) == 2, "parent after move");
        var version = tree.getVersion();
        tree.move(3, 2);
        require(tree.getVersion() == version, "same-parent move must be a no-op");
        tree.update(3, "THREE");
        require("THREE".equals(tree.get(3).value()), "update");
        require(tree.snapshot().entries().size() == 4, "snapshot size");
        require(tree.depthFirst().size() == 4, "dfs size");
        require(tree.removeSubtree(2).equals(List.of(2, 3)), "subtree removal order");
        require(tree.size() == 2, "size after removal");
    }

    private static void verifyObserverSemantics() {
        var tree = new ObservableConcurrentTree<Integer, Integer>(0, 0);
        var calls = new AtomicInteger();
        ObservableConcurrentTree.Observer<Integer, Integer> observer = change -> calls.incrementAndGet();
        tree.addObserver(observer);
        tree.add(0, 1, 1);
        tree.update(1, 2);
        tree.removeObserver(observer);
        tree.add(0, 2, 2);
        require(calls.get() == 2, "observer registration/removal");

        tree.addObserver(change -> {
            throw new IllegalStateException("expected");
        });
        try {
            tree.add(0, 3, 3);
            throw new AssertionError("observer failure not propagated");
        } catch (IllegalStateException expected) {
            require(tree.contains(3), "mutation must commit before observer failure");
        }
    }

    private static void verifyBulkLoadValidation() {
        var tree = new ObservableConcurrentTree<Integer, Integer>();
        tree.loadState(List.of(
                new ObservableConcurrentTree.NodeState<>(0, null, 0),
                new ObservableConcurrentTree.NodeState<>(1, 0, 1),
                new ObservableConcurrentTree.NodeState<>(2, 1, 2)));
        require(tree.size() == 3, "bulk load size");
        require(tree.getParentId(2) == 1, "bulk load parent");

        expectIllegalArgument(() -> tree.loadState(List.of(
                new ObservableConcurrentTree.NodeState<>(0, null, 0),
                new ObservableConcurrentTree.NodeState<>(1, 2, 1),
                new ObservableConcurrentTree.NodeState<>(2, 1, 2))));
    }

    private static void verifySealedEvents() {
        var tree = new ObservableConcurrentTree<Integer, String>(0, "root");
        tree.apply(ObservableConcurrentTree.TreeEvent.add(0, 1, "one"));
        tree.apply(ObservableConcurrentTree.TreeEvent.update(1, "ONE"));
        tree.apply(ObservableConcurrentTree.TreeEvent.add(0, 2, "two"));
        tree.apply(ObservableConcurrentTree.TreeEvent.move(1, 2));
        require(tree.getParentId(1) == 2, "sealed move event");
        tree.apply(ObservableConcurrentTree.TreeEvent.remove(1));
        require(!tree.contains(1), "sealed remove event");
    }

    private static void verifyStreamApi() {
        var tree = new ObservableConcurrentTree<Integer, String>(0, "root");
        tree.add(0, 1, "one");
        tree.add(0, 2, "two");
        tree.add(1, 3, "three");

        require(
                tree.stream()
                        .map(ObservableConcurrentTree.Entry::id)
                        .toList()
                        .equals(List.of(0, 1, 3, 2)),
                "whole-tree stream order");

        require(
                tree.childrenStream(0)
                        .map(ObservableConcurrentTree.Entry::id)
                        .toList()
                        .equals(List.of(1, 2)),
                "children stream order");

        require(
                tree.subtreeStream(1)
                        .map(ObservableConcurrentTree.Entry::id)
                        .toList()
                        .equals(List.of(1, 3)),
                "subtree stream order");

        var detached = tree.subtreeStream(1);
        tree.add(1, 4, "four");

        require(
                detached.map(ObservableConcurrentTree.Entry::id)
                        .toList()
                        .equals(List.of(1, 3)),
                "subtree stream detached from later writes");
    }

    private static void verifyJsonSerialization() throws Exception {
        var tree = new ObservableConcurrentTree<Integer, String>(0, "root");
        tree.add(0, 1, "one");
        var version = tree.getVersion();

        var json = tree.toJson();
        var restored = ObservableConcurrentTree.fromJson(
                json,
                Integer.class,
                String.class);

        require(restored.size() == 2, "JSON size");
        require(restored.getVersion() == version, "JSON version");
        require("one".equals(restored.get(1).value()), "JSON value");
        restored.add(0, 2, "two");
        require(restored.contains(2), "restored tree remains mutable");
    }

    private static void verifyConcurrentAccess() throws Exception {
        var tree = new ObservableConcurrentTree<Integer, Integer>(0, 0);
        var start = new CountDownLatch(1);
        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            var writerA = executor.submit(() -> writeRange(tree, start, 1, 5_000));
            var writerB = executor.submit(() -> writeRange(tree, start, 5_001, 10_000));
            var readerA = executor.submit(() -> readLoop(tree, start));
            var readerB = executor.submit(() -> readLoop(tree, start));
            start.countDown();
            writerA.get();
            writerB.get();
            readerA.get();
            readerB.get();
        }
        require(tree.size() == 10_001, "concurrent final size");
        require(tree.snapshot().entries().size() == 10_001, "concurrent snapshot size");
    }

    private static void writeRange(
            ObservableConcurrentTree<Integer, Integer> tree,
            CountDownLatch start,
            int first,
            int last) {
        await(start);
        for (int i = first; i <= last; i++) {
            tree.add(0, i, i);
        }
    }

    private static void readLoop(ObservableConcurrentTree<Integer, Integer> tree, CountDownLatch start) {
        await(start);
        for (int i = 0; i < 2_000; i++) {
            tree.getRootId();
            tree.getVersion();
            tree.size();
            tree.contains(i);
            tree.getRoot();
        }
    }

    private static void await(CountDownLatch latch) {
        try {
            latch.await();
        } catch (InterruptedException exception) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException(exception);
        }
    }

    private static void expectIllegalArgument(Runnable operation) {
        try {
            operation.run();
            throw new AssertionError("Expected IllegalArgumentException");
        } catch (IllegalArgumentException expected) {
        }
    }

    private static void require(boolean condition, String label) {
        if (!condition) {
            throw new AssertionError("Failed: " + label);
        }
    }
}
