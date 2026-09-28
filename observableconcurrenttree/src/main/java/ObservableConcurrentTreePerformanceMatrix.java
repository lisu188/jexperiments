import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.LongAdder;

public final class ObservableConcurrentTreePerformanceMatrix {
    private static volatile long blackhole;

    private enum Topology {
        STAR,
        CHAIN,
        BALANCED_8,
        RANDOM
    }

    public static void main(String[] args) throws Exception {
        int nodes = args.length > 0 ? Integer.parseInt(args[0]) : 100_000;
        int repetitions = args.length > 1 ? Integer.parseInt(args[1]) : 3;

        System.out.printf("ObservableConcurrentTree performance matrix: nodes=%,d reps=%d%n", nodes, repetitions);
        for (var topology : Topology.values()) {
            benchmarkTopology(topology, nodes, repetitions);
        }
        benchmarkWideMoves(Math.min(nodes, 100_000), repetitions);
        benchmarkRemovals(Math.min(nodes, 100_000), repetitions);
        benchmarkObservers(Math.min(nodes, 50_000), repetitions);
        benchmarkReadWriteRatios(Math.min(nodes, 100_000));
        benchmarkThreadModels(Math.min(nodes, 100_000));
        benchmarkMemory(Math.min(nodes, 250_000));
        System.out.println("blackhole=" + blackhole);
    }

    private static void benchmarkTopology(Topology topology, int nodes, int repetitions) throws Exception {
        System.out.println();
        System.out.println("TOPOLOGY " + topology);

        var build = samples(repetitions, () -> {
            var tree = build(topology, nodes);
            blackhole += tree.size();
        });
        print("build", build);

        print("dfs-cold", coldTraversalSamples(topology, nodes, repetitions, false));
        print("snapshot-cold", coldTraversalSamples(topology, nodes, repetitions, true));

        var tree = build(topology, nodes);
        warm(tree);

        print("scalar-5m", samples(repetitions, () -> {
            long value = 0;
            for (int i = 0; i < 5_000_000; i++) {
                value += tree.size();
                value += tree.getVersion();
                var root = tree.getRootId();
                if (root != null) {
                    value += root;
                }
            }
            blackhole += value;
        }));

        print("get-1m", samples(repetitions, () -> {
            long value = 0;
            int mask = nodes - 1;
            for (int i = 0; i < 1_000_000; i++) {
                int id = nodes == 0 ? 0 : Math.floorMod(i * 0x9E3779B9, nodes);
                var entry = tree.get(id);
                if (entry != null) {
                    value += entry.id();
                }
            }
            blackhole += value + mask;
        }));

        int traversalReps = nodes >= 200_000 ? 3 : 10;
        print("dfs-hot", samples(repetitions, () -> {
            long value = 0;
            for (int i = 0; i < traversalReps; i++) {
                value += tree.depthFirst().size();
            }
            blackhole += value;
        }));

        print("snapshot-hot", samples(repetitions, () -> {
            long value = 0;
            for (int i = 0; i < traversalReps; i++) {
                value += tree.snapshot().entries().size();
            }
            blackhole += value;
        }));

        print("stream-count", samples(repetitions, () -> {
            long value = 0;
            for (int i = 0; i < traversalReps; i++) {
                value += tree.stream().count();
            }
            blackhole += value;
        }));

        print("stream-filter", samples(repetitions, () -> {
            long value = 0;
            for (int i = 0; i < traversalReps; i++) {
                value += tree.stream().filter(entry -> (entry.id() & 15) == 0).count();
            }
            blackhole += value;
        }));

        var state = tree.depthFirst().stream()
                .map(entry -> new ObservableConcurrentTree.NodeState<>(
                        entry.id(), entry.parentId(), entry.value()))
                .toList();
        print("load-state", samples(repetitions, () -> {
            var loaded = new ObservableConcurrentTree<Integer, Integer>();
            loaded.loadState(state);
            blackhole += loaded.size();
        }));

        if (nodes <= 100_000) {
            print("json-encode", samples(repetitions, () -> {
                var json = tree.toJson();
                blackhole += json.length();
            }));
            var json = tree.toJson();
            print("json-decode", samples(repetitions, () -> {
                var decoded = ObservableConcurrentTree.fromJson(json, Integer.class, Integer.class);
                blackhole += decoded.size();
            }));
        }
    }

    private static void benchmarkWideMoves(int children, int repetitions) throws Exception {
        System.out.println();
        System.out.printf("WIDE MOVE children=%,d%n", children);
        var tree = new ObservableConcurrentTree<Integer, Integer>(0, 0);
        tree.add(0, -1, -1);
        tree.add(0, -2, -2);
        for (int i = 1; i <= children; i++) {
            tree.add(-1, i, i);
        }

        int first = 1;
        int middle = Math.max(1, children / 2);
        int last = Math.max(1, children);
        for (int id : new int[] {first, middle, last}) {
            print("move-child-" + id, samples(repetitions, () -> {
                tree.move(id, -2);
                tree.move(id, -1);
                blackhole += tree.getParentId(id);
            }));
        }
    }

    private static void benchmarkRemovals(int nodes, int repetitions) throws Exception {
        System.out.println();
        System.out.printf("REMOVE SUBTREE chainNodes=%,d%n", nodes);
        for (int percent : new int[] {1, 10, 50, 100}) {
            var values = new double[repetitions];
            for (int r = 0; r < repetitions; r++) {
                var tree = build(Topology.CHAIN, nodes);
                int subtree = Math.max(1, nodes * percent / 100);
                int removeId = Math.max(0, nodes - subtree);
                long start = System.nanoTime();
                var removed = tree.removeSubtree(removeId);
                values[r] = (System.nanoTime() - start) / 1_000_000.0;
                blackhole += removed.size();
            }
            java.util.Arrays.sort(values);
            print("remove-" + percent + "%", values);
        }
    }

    private static void benchmarkObservers(int operations, int repetitions) throws Exception {
        System.out.println();
        System.out.printf("OBSERVERS operations=%,d%n", operations);
        for (int observers : new int[] {0, 1, 10, 100}) {
            print("observers-" + observers, samples(repetitions, () -> {
                var tree = new ObservableConcurrentTree<Integer, Integer>(0, 0);
                for (int i = 0; i < observers; i++) {
                    tree.addObserver(change -> blackhole += change.version());
                }
                for (int i = 1; i <= operations; i++) {
                    tree.add(0, i, i);
                }
                blackhole += tree.size();
            }));
        }
    }

    private static void benchmarkReadWriteRatios(int nodes) throws Exception {
        System.out.println();
        System.out.printf("READ/WRITE CONTENTION nodes=%,d%n", nodes);
        var tree = build(Topology.BALANCED_8, nodes);
        for (int writePercent : new int[] {0, 1, 5, 10, 25, 50, 100}) {
            int tasks = 256;
            int operationsPerTask = 2_000;
            var start = new CountDownLatch(1);
            var operations = new LongAdder();
            long begin = System.nanoTime();
            try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
                var futures = new ArrayList<java.util.concurrent.Future<?>>(tasks);
                for (int t = 0; t < tasks; t++) {
                    futures.add(executor.submit(() -> {
                        await(start);
                        var random = ThreadLocalRandom.current();
                        long local = 0;
                        for (int i = 0; i < operationsPerTask; i++) {
                            int id = random.nextInt(nodes);
                            if (random.nextInt(100) < writePercent) {
                                tree.update(id, random.nextInt());
                            } else {
                                var entry = tree.get(id);
                                if (entry != null) {
                                    local += entry.value();
                                }
                            }
                            operations.increment();
                        }
                        blackhole += local;
                    }));
                }
                start.countDown();
                for (var future : futures) {
                    future.get();
                }
            }
            double seconds = (System.nanoTime() - begin) / 1_000_000_000.0;
            System.out.printf("rw write=%3d%% tasks=%d ops=%,d time=%.3fs throughput=%,.0f ops/s%n",
                    writePercent, tasks, operations.sum(), seconds, operations.sum() / seconds);
        }
    }

    private static double[] coldTraversalSamples(
            Topology topology,
            int nodes,
            int repetitions,
            boolean snapshot) {
        var values = new double[repetitions];
        for (int i = 0; i < repetitions; i++) {
            var tree = build(topology, nodes);
            long start = System.nanoTime();
            int size = snapshot ? tree.snapshot().entries().size() : tree.depthFirst().size();
            values[i] = (System.nanoTime() - start) / 1_000_000.0;
            blackhole += size;
        }
        java.util.Arrays.sort(values);
        return values;
    }

    private static void benchmarkThreadModels(int nodes) throws Exception {
        System.out.println();
        System.out.printf("THREAD MODEL 95/5 read/write nodes=%,d%n", nodes);
        int totalOperations = 500_000;

        for (int threads : new int[] {1, 2, 4, 8, 16, 32, 64}) {
            var tree = build(Topology.BALANCED_8, nodes);
            try (var executor = Executors.newFixedThreadPool(threads)) {
                double seconds = runContention(tree, executor, threads,
                        Math.max(1, totalOperations / threads), 5);
                System.out.printf("platform threads=%3d throughput=%,.0f ops/s%n",
                        threads, totalOperations / seconds);
            }
        }

        for (int tasks : new int[] {100, 1_000, 5_000}) {
            var tree = build(Topology.BALANCED_8, nodes);
            try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
                int opsPerTask = Math.max(1, totalOperations / tasks);
                double seconds = runContention(tree, executor, tasks, opsPerTask, 5);
                int actual = tasks * opsPerTask;
                System.out.printf("virtual  tasks=%5d throughput=%,.0f ops/s%n",
                        tasks, actual / seconds);
            }
        }
    }

    private static double runContention(
            ObservableConcurrentTree<Integer, Integer> tree,
            ExecutorService executor,
            int tasks,
            int operationsPerTask,
            int writePercent) throws Exception {
        var start = new CountDownLatch(1);
        var futures = new ArrayList<java.util.concurrent.Future<?>>(tasks);
        long begin = System.nanoTime();
        for (int t = 0; t < tasks; t++) {
            futures.add(executor.submit(() -> {
                await(start);
                var random = ThreadLocalRandom.current();
                long local = 0;
                for (int i = 0; i < operationsPerTask; i++) {
                    int id = random.nextInt(tree.size());
                    if (random.nextInt(100) < writePercent) {
                        tree.update(id, random.nextInt());
                    } else {
                        var entry = tree.get(id);
                        if (entry != null) {
                            local += entry.value();
                        }
                    }
                }
                blackhole += local;
            }));
        }
        start.countDown();
        for (var future : futures) {
            future.get();
        }
        return (System.nanoTime() - begin) / 1_000_000_000.0;
    }

    private static void benchmarkMemory(int nodes) throws InterruptedException {
        System.out.println();
        System.out.printf("APPROX MEMORY nodes=%,d%n", nodes);
        forceGc();
        long before = usedMemory();
        var tree = build(Topology.BALANCED_8, nodes);
        forceGc();
        long afterTree = usedMemory();
        tree.snapshot();
        forceGc();
        long afterSnapshot = usedMemory();
        long treeBytes = Math.max(0, afterTree - before);
        long cacheBytes = Math.max(0, afterSnapshot - afterTree);
        System.out.printf("tree retained≈%,d bytes (≈%.1f bytes/node)%n",
                treeBytes, nodes == 0 ? 0.0 : (double) treeBytes / nodes);
        System.out.printf("Entry cache increment≈%,d bytes (≈%.1f bytes/node)%n",
                cacheBytes, nodes == 0 ? 0.0 : (double) cacheBytes / nodes);
        blackhole += tree.size();
    }

    private static long usedMemory() {
        var runtime = Runtime.getRuntime();
        return runtime.totalMemory() - runtime.freeMemory();
    }

    private static void forceGc() throws InterruptedException {
        System.gc();
        Thread.sleep(100);
    }

    private static ObservableConcurrentTree<Integer, Integer> build(Topology topology, int nodes) {
        if (nodes <= 0) {
            return new ObservableConcurrentTree<>();
        }
        var tree = new ObservableConcurrentTree<Integer, Integer>(0, 0);
        var random = new Random(0x51A7E);
        for (int id = 1; id < nodes; id++) {
            int parent = switch (topology) {
                case STAR -> 0;
                case CHAIN -> id - 1;
                case BALANCED_8 -> (id - 1) / 8;
                case RANDOM -> random.nextInt(id);
            };
            tree.add(parent, id, id);
        }
        return tree;
    }

    private static void warm(ObservableConcurrentTree<Integer, Integer> tree) {
        tree.snapshot();
        tree.depthFirst();
        tree.stream().count();
    }

    private static double[] samples(int repetitions, ThrowingRunnable operation) throws Exception {
        operation.run();
        var values = new double[repetitions];
        for (int i = 0; i < repetitions; i++) {
            long start = System.nanoTime();
            operation.run();
            values[i] = (System.nanoTime() - start) / 1_000_000.0;
        }
        java.util.Arrays.sort(values);
        return values;
    }

    private static void print(String name, double[] samples) {
        double median = samples[samples.length / 2];
        double min = samples[0];
        double max = samples[samples.length - 1];
        System.out.printf("%-22s median=%10.3f ms min=%10.3f max=%10.3f%n", name, median, min, max);
    }

    private static void await(CountDownLatch latch) {
        try {
            latch.await();
        } catch (InterruptedException exception) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException(exception);
        }
    }

    @FunctionalInterface
    private interface ThrowingRunnable {
        void run() throws Exception;
    }
}
