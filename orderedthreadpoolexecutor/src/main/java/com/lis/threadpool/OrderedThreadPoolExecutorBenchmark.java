package com.lis.threadpool;

import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.TreeMap;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;

public final class OrderedThreadPoolExecutorBenchmark {
    private static final int DEFAULT_TASK_COUNT = 50_000;
    private static final int DEFAULT_REPETITIONS = 5;
    private static volatile long blackhole;

    public static void main(String[] args) throws Exception {
        var taskCount = args.length > 0 ? Integer.parseInt(args[0]) : DEFAULT_TASK_COUNT;
        var repetitions = args.length > 1 ? Integer.parseInt(args[1]) : DEFAULT_REPETITIONS;
        var reverseOrder = reverseOrder(taskCount);
        var shuffledOrder = shuffledOrder(taskCount, 0x5eedL);

        for (int i = 0; i < 2; i++) {
            run(taskCount, reverseOrder, shuffledOrder, false);
        }

        var samples = new LinkedHashMap<String, double[]>();
        for (int i = 0; i < repetitions; i++) {
            var result = run(taskCount, reverseOrder, shuffledOrder, true);
            var index = i;
            result.forEach((name, value) ->
                    samples.computeIfAbsent(name, ignored -> new double[repetitions])[index] = value);
        }

        System.out.printf(
                "OrderedThreadPoolExecutor benchmark: %,d completions, %d measured runs%n",
                taskCount,
                repetitions);
        for (var entry : samples.entrySet()) {
            var values = entry.getValue().clone();
            Arrays.sort(values);
            var median = values[values.length / 2];
            System.out.printf("%-24s %9.3f ms median%n", entry.getKey(), median);
        }
        System.out.println("blackhole=" + blackhole);
    }

    private static Map<String, Double> run(
            int taskCount,
            int[] reverseOrder,
            int[] shuffledOrder,
            boolean measured) throws Exception {
        var result = new LinkedHashMap<String, Double>();
        result.put("tree-map-reverse", millis(() -> blackhole += drainTreeMap(reverseOrder)));
        result.put("hash-map-reverse", millis(() -> blackhole += drainHashMap(reverseOrder)));
        result.put("tree-map-shuffled", millis(() -> blackhole += drainTreeMap(shuffledOrder)));
        result.put("hash-map-shuffled", millis(() -> blackhole += drainHashMap(shuffledOrder)));
        result.put("executor-throughput", millis(() -> blackhole += runExecutor(taskCount)));

        if (!measured) {
            result.clear();
        }
        return result;
    }

    private static long drainTreeMap(int[] completionOrder) {
        var buffer = new TreeMap<Long, Integer>();
        long next = 0;
        long sum = 0;
        for (var sequence : completionOrder) {
            buffer.put((long) sequence, sequence);
            while (!buffer.isEmpty() && buffer.firstKey() == next) {
                sum += buffer.remove(next);
                next++;
            }
        }
        if (!buffer.isEmpty() || next != completionOrder.length) {
            throw new AssertionError("TreeMap reorder buffer did not drain");
        }
        return sum;
    }

    private static long drainHashMap(int[] completionOrder) {
        var buffer = new HashMap<Long, Integer>();
        long next = 0;
        long sum = 0;
        for (var sequence : completionOrder) {
            buffer.put((long) sequence, sequence);
            Integer value;
            while ((value = buffer.remove(next)) != null) {
                sum += value;
                next++;
            }
        }
        if (!buffer.isEmpty() || next != completionOrder.length) {
            throw new AssertionError("HashMap reorder buffer did not drain");
        }
        return sum;
    }

    private static long runExecutor(int taskCount) throws Exception {
        var output = new LinkedBlockingQueue<Integer>();
        var parallelism = Math.max(2, Math.min(Runtime.getRuntime().availableProcessors(), 16));
        try (var workers = Executors.newFixedThreadPool(parallelism);
             var executor = new OrderedThreadPoolExecutor<Integer>(output, workers)) {
            for (int i = 0; i < taskCount; i++) {
                var value = i;
                executor.process(() -> value);
            }

            long sum = 0;
            for (int expected = 0; expected < taskCount; expected++) {
                var actual = output.take();
                if (actual != expected) {
                    throw new AssertionError("Unexpected publication order: " + actual + " != " + expected);
                }
                sum += actual;
            }
            return sum;
        }
    }

    private static int[] reverseOrder(int size) {
        var order = new int[size];
        for (int i = 0; i < size; i++) {
            order[i] = size - i - 1;
        }
        return order;
    }

    private static int[] shuffledOrder(int size, long seed) {
        var order = new int[size];
        for (int i = 0; i < size; i++) {
            order[i] = i;
        }
        var random = new SplittableRandom(seed);
        for (int i = size - 1; i > 0; i--) {
            var swapWith = random.nextInt(i + 1);
            var value = order[i];
            order[i] = order[swapWith];
            order[swapWith] = value;
        }
        return order;
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
