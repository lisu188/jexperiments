import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.Map;

public final class ObservableConcurrentTreeBenchmark {
    private static final int DEFAULT_NODE_COUNT = 25_000;
    private static final int DEFAULT_REPETITIONS = 5;
    private static volatile long blackhole;

    public static void main(String[] args) throws Exception {
        var nodeCount = args.length > 0 ? Integer.parseInt(args[0]) : DEFAULT_NODE_COUNT;
        var repetitions = args.length > 1 ? Integer.parseInt(args[1]) : DEFAULT_REPETITIONS;

        for (int i = 0; i < 2; i++) {
            run(nodeCount, false);
        }

        var samples = new LinkedHashMap<String, double[]>();
        for (int i = 0; i < repetitions; i++) {
            var result = run(nodeCount, true);
            var index = i;
            result.forEach((name, value) ->
                    samples.computeIfAbsent(name, ignored -> new double[repetitions])[index] = value);
        }

        System.out.printf(
                "ObservableConcurrentTree benchmark: %,d nodes, %d measured runs%n",
                nodeCount,
                repetitions);
        for (var entry : samples.entrySet()) {
            var values = entry.getValue().clone();
            java.util.Arrays.sort(values);
            var median = values[values.length / 2];
            System.out.printf("%-10s %8.3f ms median%n", entry.getKey(), median);
        }
        System.out.println("blackhole=" + blackhole);
    }

    private static Map<String, Double> run(int nodeCount, boolean measured) throws Exception {
        var result = new LinkedHashMap<String, Double>();

        result.put("build", millis(() -> {
            var tree = new ObservableConcurrentTree<Integer, Integer>(0, 0);
            for (int i = 1; i <= nodeCount; i++) {
                tree.add(0, i, i);
            }
            blackhole += tree.size();
        }));

        var tree = new ObservableConcurrentTree<Integer, Integer>(0, 0);
        for (int i = 1; i <= nodeCount; i++) {
            tree.add(0, i, i);
        }

        result.put("scalar", millis(() -> {
            long value = 0;
            for (int i = 0; i < 5_000_000; i++) {
                value += tree.size();
                value += tree.getVersion();
                value += tree.getRootId();
            }
            blackhole += value;
        }));

        result.put("children", millis(() -> {
            long value = 0;
            for (int i = 0; i < 40; i++) {
                value += tree.getChildren(0).size();
            }
            blackhole += value;
        }));

        result.put("dfs", millis(() -> {
            long value = 0;
            for (int i = 0; i < 40; i++) {
                value += tree.depthFirst().size();
            }
            blackhole += value;
        }));

        result.put("snapshot", millis(() -> {
            long value = 0;
            for (int i = 0; i < 40; i++) {
                value += tree.snapshot().getEntries().size();
            }
            blackhole += value;
        }));

        var state = new ArrayList<ObservableConcurrentTree.NodeState<Integer, Integer>>(nodeCount + 1);
        state.add(new ObservableConcurrentTree.NodeState<>(0, null, 0));
        for (int i = 1; i <= nodeCount; i++) {
            state.add(new ObservableConcurrentTree.NodeState<>(i, 0, i));
        }

        result.put("load", millis(() -> {
            var loaded = new ObservableConcurrentTree<Integer, Integer>();
            for (int i = 0; i < 20; i++) {
                loaded.loadState(state);
            }
            blackhole += loaded.size();
        }));

        result.put("remove", millis(() -> {
            var removable = new ObservableConcurrentTree<Integer, Integer>(0, 0);
            for (int i = 1; i <= nodeCount; i++) {
                removable.add(0, i, i);
            }
            blackhole += removable.removeSubtree(0).size();
        }));

        if (!measured) {
            result.clear();
        }
        return result;
    }

    private static double millis(ThrowingRunnable runnable) throws Exception {
        var start = System.nanoTime();
        runnable.run();
        return (System.nanoTime() - start) / 1_000_000.0;
    }

    @FunctionalInterface
    private interface ThrowingRunnable {
        void run() throws Exception;
    }
}
