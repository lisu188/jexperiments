import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

public final class ObservableConcurrentTreeExample {

    public static void main(String[] args) throws Exception {
        constructorAndInitialization();
        directMutationsAndReads();
        eventDrivenMutations();
        bulkStateLoading();
        serializationRoundTrip();
        concurrentReadersAndWriters();
        observerFailureSemantics();
        validationFailures();
        clearAndObserverRemoval();
    }

    private static void constructorAndInitialization() {
        section("1. Constructors and initialization");

        ObservableConcurrentTree<String, String> emptyTree =
                new ObservableConcurrentTree<String, String>();

        System.out.println("emptyTree.isEmpty() = " + emptyTree.isEmpty());
        System.out.println("emptyTree.size() = " + emptyTree.size());
        System.out.println("emptyTree.getRootId() = " + emptyTree.getRootId());
        System.out.println("emptyTree.getRoot() = " + emptyTree.getRoot());

        emptyTree.initialize("root", "Root");
        printEntry("initialized root", emptyTree.getRoot());

        ObservableConcurrentTree<String, String> initializedTree =
                new ObservableConcurrentTree<String, String>("root", "Root from constructor");

        printEntry("constructor root", initializedTree.getRoot());
        System.out.println("constructor tree version = " + initializedTree.getVersion());
    }

    private static void directMutationsAndReads() {
        section("2. Direct mutations, observers, and read API");

        ObservableConcurrentTree<String, String> tree =
                new ObservableConcurrentTree<String, String>();

        ObservableConcurrentTree.Observer<String, String> observer =
                new ObservableConcurrentTree.Observer<String, String>() {
                    @Override
                    public void onChange(ObservableConcurrentTree.Change<String, String> change) {
                        printChange(change);
                    }
                };

        tree.addObserver(observer);
        tree.initialize("root", "Infrastructure");

        tree.add("root", "dc-1", "Datacenter 1");
        tree.add("root", "dc-2", "Datacenter 2");
        tree.add("dc-1", "server-1", "Server 1");
        tree.add("dc-1", "server-2", "Server 2");

        tree.update("server-1", "Server 1 - updated");

        System.out.println("contains(server-1) = " + tree.contains("server-1"));
        System.out.println("contains(missing) = " + tree.contains("missing"));
        System.out.println("contains(null) = " + tree.contains(null));
        System.out.println("size = " + tree.size());
        System.out.println("isEmpty = " + tree.isEmpty());
        System.out.println("rootId = " + tree.getRootId());
        System.out.println("version = " + tree.getVersion());

        printEntry("get(server-1)", tree.get("server-1"));
        printEntry("getRoot()", tree.getRoot());

        System.out.println("parent of server-1 = " + tree.getParentId("server-1"));

        System.out.println("children of dc-1:");
        for (ObservableConcurrentTree.Entry<String, String> child : tree.getChildren("dc-1")) {
            printEntry("  child", child);
        }

        System.out.println("depth-first traversal:");
        for (ObservableConcurrentTree.Entry<String, String> entry : tree.depthFirst()) {
            printEntry("  dfs", entry);
        }

        ObservableConcurrentTree.Snapshot<String, String> beforeMove = tree.snapshot();
        printSnapshot("snapshot before move", beforeMove);

        tree.move("server-2", "dc-2");
        System.out.println("parent of server-2 after move = " + tree.getParentId("server-2"));
        long versionAfterMove = tree.getVersion();
        tree.move("server-2", "dc-2");
        System.out.println("same-parent move is a no-op = " + (tree.getVersion() == versionAfterMove));

        List<String> removed = tree.removeSubtree("dc-1");
        System.out.println("removeSubtree(dc-1) removed = " + removed);
        System.out.println("size after subtree removal = " + tree.size());

        printSnapshot("snapshot after removal", tree.snapshot());

        tree.removeObserver(observer);
        tree.add("dc-2", "server-3", "No observer output should follow this add");
        System.out.println("observer removed; server-3 added");
    }

    private static void eventDrivenMutations() {
        section("3. TreeEvent factories, getters, and apply()");

        ObservableConcurrentTree<String, String> tree =
                new ObservableConcurrentTree<String, String>("root", "Root");

        ObservableConcurrentTree.TreeEvent<String, String> addParent =
                ObservableConcurrentTree.TreeEvent.add("root", "parent", "Parent");
        printEvent(addParent);
        tree.apply(addParent);

        ObservableConcurrentTree.TreeEvent<String, String> addChild =
                ObservableConcurrentTree.TreeEvent.add("parent", "child", "Child");
        printEvent(addChild);
        tree.apply(addChild);

        ObservableConcurrentTree.TreeEvent<String, String> updateChild =
                ObservableConcurrentTree.TreeEvent.update("child", "Child updated through event");
        printEvent(updateChild);
        tree.apply(updateChild);

        ObservableConcurrentTree.TreeEvent<String, String> moveChild =
                ObservableConcurrentTree.TreeEvent.move("child", "root");
        printEvent(moveChild);
        tree.apply(moveChild);

        ObservableConcurrentTree.TreeEvent<String, String> removeParent =
                ObservableConcurrentTree.TreeEvent.remove("parent");
        printEvent(removeParent);
        tree.apply(removeParent);

        printSnapshot("state after event processing", tree.snapshot());
    }

    private static void bulkStateLoading() {
        section("4. NodeState and atomic loadState()");

        ObservableConcurrentTree.NodeState<String, String> root =
                new ObservableConcurrentTree.NodeState<String, String>("root", null, "Loaded root");
        ObservableConcurrentTree.NodeState<String, String> branchA =
                new ObservableConcurrentTree.NodeState<String, String>("a", "root", "Branch A");
        ObservableConcurrentTree.NodeState<String, String> branchB =
                new ObservableConcurrentTree.NodeState<String, String>("b", "root", "Branch B");
        ObservableConcurrentTree.NodeState<String, String> leaf =
                new ObservableConcurrentTree.NodeState<String, String>("leaf", "a", "Leaf");

        System.out.println(
                "NodeState getters: id=" + leaf.getId()
                        + ", parentId=" + leaf.getParentId()
                        + ", value=" + leaf.getValue());

        ObservableConcurrentTree<String, String> tree =
                new ObservableConcurrentTree<String, String>();

        tree.addObserver(new ObservableConcurrentTree.Observer<String, String>() {
            @Override
            public void onChange(ObservableConcurrentTree.Change<String, String> change) {
                printChange(change);
            }
        });

        tree.loadState(Arrays.asList(root, branchA, branchB, leaf));

        printSnapshot("loaded state", tree.snapshot());
    }

    private static void serializationRoundTrip() throws Exception {
        section("5. Serialization and deserialization");

        ObservableConcurrentTree<String, String> tree =
                new ObservableConcurrentTree<String, String>("root", "Serializable root");

        tree.add("root", "child", "Serializable child");

        byte[] bytes = tree.toByteArray();
        System.out.println("serialized bytes = " + bytes.length);

        ObservableConcurrentTree<String, String> restored =
                ObservableConcurrentTree.fromByteArray(bytes);

        System.out.println("restored version = " + restored.getVersion());
        System.out.println("restored size = " + restored.size());
        printSnapshot("restored snapshot", restored.snapshot());

        restored.addObserver(new ObservableConcurrentTree.Observer<String, String>() {
            @Override
            public void onChange(ObservableConcurrentTree.Change<String, String> change) {
                System.out.println("observer registered after deserialization:");
                printChange(change);
            }
        });

        restored.update("child", "Observers are transient and must be registered again");
    }

    private static void concurrentReadersAndWriters() throws Exception {
        section("6. Concurrent readers and writers");

        final ObservableConcurrentTree<String, String> tree =
                new ObservableConcurrentTree<String, String>("root", "Concurrent root");

        final AtomicInteger observedChanges = new AtomicInteger();
        tree.addObserver(new ObservableConcurrentTree.Observer<String, String>() {
            @Override
            public void onChange(ObservableConcurrentTree.Change<String, String> change) {
                observedChanges.incrementAndGet();
            }
        });

        ExecutorService executor = Executors.newFixedThreadPool(4);
        CountDownLatch start = new CountDownLatch(1);
        List<Future<?>> futures = new ArrayList<Future<?>>();

        for (int writer = 0; writer < 2; writer++) {
            final int writerId = writer;
            futures.add(executor.submit(new Runnable() {
                @Override
                public void run() {
                    await(start);
                    for (int i = 0; i < 5; i++) {
                        tree.add(
                                "root",
                                "writer-" + writerId + "-node-" + i,
                                "value-" + writerId + "-" + i);
                    }
                }
            }));
        }

        for (int reader = 0; reader < 2; reader++) {
            futures.add(executor.submit(new Runnable() {
                @Override
                public void run() {
                    await(start);
                    for (int i = 0; i < 20; i++) {
                        tree.contains("root");
                        tree.getRoot();
                        tree.snapshot();
                        tree.depthFirst();
                    }
                }
            }));
        }

        start.countDown();

        for (Future<?> future : futures) {
            future.get();
        }

        executor.shutdown();
        if (!executor.awaitTermination(5, TimeUnit.SECONDS)) {
            throw new IllegalStateException("Concurrent example did not terminate");
        }

        System.out.println("concurrent final size = " + tree.size());
        System.out.println("concurrent observer notifications = " + observedChanges.get());
        printSnapshot("concurrent final snapshot", tree.snapshot());
    }

    private static void observerFailureSemantics() {
        section("7. Observer failure happens after mutation commit");

        final ObservableConcurrentTree<String, String> tree =
                new ObservableConcurrentTree<String, String>("root", "Root");

        tree.addObserver(new ObservableConcurrentTree.Observer<String, String>() {
            @Override
            public void onChange(ObservableConcurrentTree.Change<String, String> change) {
                throw new IllegalStateException("observer failed intentionally");
            }
        });

        try {
            tree.add("root", "committed", "Mutation commits before observer callback");
            throw new AssertionError("Expected observer failure");
        } catch (IllegalStateException expected) {
            System.out.println("observer exception = " + expected.getMessage());
        }

        System.out.println("node exists despite observer exception = " + tree.contains("committed"));
        printEntry("committed node", tree.get("committed"));
    }

    private static void validationFailures() {
        section("8. Validation and rejected operations");

        ObservableConcurrentTree<String, String> tree =
                new ObservableConcurrentTree<String, String>("root", "Root");

        tree.add("root", "a", "A");
        tree.add("a", "b", "B");

        expectFailure("duplicate node id", new Runnable() {
            @Override
            public void run() {
                tree.add("root", "a", "Duplicate");
            }
        });

        expectFailure("missing parent", new Runnable() {
            @Override
            public void run() {
                tree.add("missing", "x", "X");
            }
        });

        expectFailure("moving root", new Runnable() {
            @Override
            public void run() {
                tree.move("root", "a");
            }
        });

        expectFailure("self-parent move", new Runnable() {
            @Override
            public void run() {
                tree.move("a", "a");
            }
        });

        expectFailure("cycle-producing move", new Runnable() {
            @Override
            public void run() {
                tree.move("a", "b");
            }
        });

        expectFailure("second initialize", new Runnable() {
            @Override
            public void run() {
                tree.initialize("new-root", "New root");
            }
        });

        expectFailure("invalid loaded state with two roots", new Runnable() {
            @Override
            public void run() {
                tree.loadState(Arrays.asList(
                        new ObservableConcurrentTree.NodeState<String, String>("root-1", null, "Root 1"),
                        new ObservableConcurrentTree.NodeState<String, String>("root-2", null, "Root 2")));
            }
        });
    }

    private static void clearAndObserverRemoval() {
        section("9. clear(), empty-state reads, and root subtree removal");

        ObservableConcurrentTree<String, String> tree =
                new ObservableConcurrentTree<String, String>("root", "Root");

        tree.add("root", "child", "Child");

        List<String> allRemoved = tree.removeSubtree("root");
        System.out.println("removing root subtree removed = " + allRemoved);
        System.out.println("rootId after root removal = " + tree.getRootId());
        System.out.println("isEmpty after root removal = " + tree.isEmpty());
        System.out.println("depthFirst on empty tree = " + tree.depthFirst());

        tree.initialize("new-root", "New Root");
        tree.add("new-root", "child", "Child");
        tree.clear();

        System.out.println("after clear: rootId = " + tree.getRootId());
        System.out.println("after clear: size = " + tree.size());
        System.out.println("after clear: isEmpty = " + tree.isEmpty());

        tree.clear();
        System.out.println("clearing an already empty tree is a no-op");
    }

    private static void printEntry(
            String label,
            ObservableConcurrentTree.Entry<String, String> entry) {
        if (entry == null) {
            System.out.println(label + " = null");
            return;
        }

        System.out.println(
                label
                        + " = {id=" + entry.getId()
                        + ", parentId=" + entry.getParentId()
                        + ", value=" + entry.getValue()
                        + ", children=" + entry.getChildren()
                        + "}");
    }

    private static void printSnapshot(
            String label,
            ObservableConcurrentTree.Snapshot<String, String> snapshot) {
        System.out.println(
                label
                        + ": rootId=" + snapshot.getRootId()
                        + ", version=" + snapshot.getVersion()
                        + ", entries=" + snapshot.getEntries().size());

        for (ObservableConcurrentTree.Entry<String, String> entry : snapshot.getEntries()) {
            printEntry("  snapshot entry", entry);
        }
    }

    private static void printChange(
            ObservableConcurrentTree.Change<String, String> change) {
        System.out.println(
                "change"
                        + " type=" + change.getType()
                        + " version=" + change.getVersion()
                        + " nodeId=" + change.getNodeId()
                        + " oldParentId=" + change.getOldParentId()
                        + " newParentId=" + change.getNewParentId()
                        + " oldValue=" + change.getOldValue()
                        + " newValue=" + change.getNewValue()
                        + " affectedNodeIds=" + change.getAffectedNodeIds());
    }

    private static void printEvent(
            ObservableConcurrentTree.TreeEvent<String, String> event) {
        System.out.println(
                "event"
                        + " type=" + event.getType()
                        + " nodeId=" + event.getNodeId()
                        + " parentId=" + event.getParentId()
                        + " value=" + event.getValue());
    }

    private static void expectFailure(String label, Runnable action) {
        try {
            action.run();
            throw new AssertionError("Expected failure for: " + label);
        } catch (IllegalArgumentException | IllegalStateException expected) {
            System.out.println(label + " rejected: " + expected.getMessage());
        }
    }

    private static void await(CountDownLatch latch) {
        try {
            latch.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Interrupted while waiting to start", e);
        }
    }

    private static void section(String title) {
        System.out.println();
        System.out.println("=== " + title + " ===");
    }
}
