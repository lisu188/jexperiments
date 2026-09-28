package com.lis.threadpool;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.LinkedTransferQueue;
import java.util.concurrent.SynchronousQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.LockSupport;

public final class OrderedThreadPoolExecutorPerformanceMatrix {
    private record TimedValue(long submittedNanos) {}
    private record Result(double throughput, double p50Micros, double p99Micros, double p999Micros, long maxBuffered) {}
    private record Scenario(String worker, String queue, int producers, long taskCostNanos, int failureBasisPoints) {}

    public static void main(String[] args) throws Exception {
        var tasks = args.length > 0 ? Integer.parseInt(args[0]) : 20_000;
        var repetitions = args.length > 1 ? Integer.parseInt(args[1]) : 3;
        var scenarios = new LinkedHashMap<String, Scenario>();
        for (var worker : new String[]{"fixed", "virtual"}) {
            for (var queue : new String[]{"array-1", "array-256", "linked", "transfer", "synchronous"}) {
                scenarios.put(worker + "/" + queue + "/p1", new Scenario(worker, queue, 1, 0, 0));
            }
            for (var producers : new int[]{4, 16, 64}) {
                scenarios.put(worker + "/linked/p" + producers, new Scenario(worker, "linked", producers, 0, 0));
            }
        }
        scenarios.put("fixed/linked/p16/task10us", new Scenario("fixed", "linked", 16, 10_000, 0));
        scenarios.put("virtual/linked/p16/task100us", new Scenario("virtual", "linked", 16, 100_000, 0));
        scenarios.put("virtual/linked/p16/failure1pct", new Scenario("virtual", "linked", 16, 0, 100));

        System.out.printf("OrderedThreadPoolExecutor performance matrix: %,d tasks, %d repetitions%n", tasks, repetitions);
        for (var entry : scenarios.entrySet()) {
            var samples = new Result[repetitions];
            for (int i = 0; i < repetitions; i++) {
                samples[i] = run(entry.getValue(), tasks);
            }
            print(entry.getKey(), median(samples));
        }
    }

    private static Result run(Scenario scenario, int tasks) throws Exception {
        var output = queue(scenario.queue());
        try (var workers = workers(scenario.worker());
             var executor = new OrderedThreadPoolExecutor<TimedValue>(output, workers);
             var producers = Executors.newVirtualThreadPerTaskExecutor()) {
            var expectedSuccesses = tasks - expectedFailures(tasks, scenario.failureBasisPoints());
            var latencies = new long[expectedSuccesses];
            var consumed = new AtomicInteger();
            var consumer = Thread.ofPlatform().daemon(true).start(() -> {
                try {
                    while (consumed.get() < expectedSuccesses) {
                        var value = output.take();
                        var index = consumed.getAndIncrement();
                        latencies[index] = System.nanoTime() - value.submittedNanos();
                    }
                } catch (InterruptedException interruption) {
                    Thread.currentThread().interrupt();
                }
            });

            var perProducer = (tasks + scenario.producers() - 1) / scenario.producers();
            var start = System.nanoTime();
            var producerTasks = new java.util.ArrayList<java.util.concurrent.Future<?>>();
            for (int producer = 0; producer < scenario.producers(); producer++) {
                var from = producer * perProducer;
                var to = Math.min(tasks, from + perProducer);
                if (from >= to) {
                    continue;
                }
                producerTasks.add(producers.submit(() -> {
                    for (int id = from; id < to; id++) {
                        var taskId = id;
                        var submitted = System.nanoTime();
                        executor.executeOrdered(() -> {
                            if (scenario.taskCostNanos() != 0) {
                                LockSupport.parkNanos(scenario.taskCostNanos());
                            }
                            if (scenario.failureBasisPoints() != 0
                                    && taskId % 10_000 < scenario.failureBasisPoints()) {
                                throw new IllegalStateException("synthetic");
                            }
                            return new TimedValue(submitted);
                        });
                    }
                }));
            }
            for (var producerTask : producerTasks) {
                producerTask.get();
            }
            executor.shutdown();
            if (!executor.awaitTermination(java.time.Duration.ofMinutes(2))) {
                throw new AssertionError("executor did not terminate");
            }
            var elapsed = System.nanoTime() - start;
            consumer.join(TimeUnit.MINUTES.toMillis(2));
            if (consumer.isAlive()) {
                consumer.interrupt();
                throw new AssertionError("consumer did not terminate");
            }

            var samples = Arrays.copyOf(latencies, consumed.get());
            Arrays.sort(samples);
            return new Result(
                    tasks / (elapsed / 1_000_000_000.0),
                    percentile(samples, 0.50) / 1_000.0,
                    percentile(samples, 0.99) / 1_000.0,
                    percentile(samples, 0.999) / 1_000.0,
                    executor.statistics().maxBuffered());
        }
    }

    private static int expectedFailures(int tasks, int basisPoints) {
        if (basisPoints == 0) {
            return 0;
        }
        var full = tasks / 10_000;
        var remainder = tasks % 10_000;
        return full * basisPoints + Math.min(remainder, basisPoints);
    }

    private static BlockingQueue<TimedValue> queue(String name) {
        return switch (name) {
            case "array-1" -> new ArrayBlockingQueue<>(1);
            case "array-256" -> new ArrayBlockingQueue<>(256);
            case "linked" -> new LinkedBlockingQueue<>();
            case "transfer" -> new LinkedTransferQueue<>();
            case "synchronous" -> new SynchronousQueue<>();
            default -> throw new IllegalArgumentException(name);
        };
    }

    private static ExecutorService workers(String name) {
        return switch (name) {
            case "fixed" -> Executors.newFixedThreadPool(
                    Math.max(2, Math.min(Runtime.getRuntime().availableProcessors(), 16)));
            case "virtual" -> Executors.newVirtualThreadPerTaskExecutor();
            default -> throw new IllegalArgumentException(name);
        };
    }

    private static Result median(Result[] values) {
        var throughput = Arrays.stream(values).mapToDouble(Result::throughput).sorted().toArray();
        var p50 = Arrays.stream(values).mapToDouble(Result::p50Micros).sorted().toArray();
        var p99 = Arrays.stream(values).mapToDouble(Result::p99Micros).sorted().toArray();
        var p999 = Arrays.stream(values).mapToDouble(Result::p999Micros).sorted().toArray();
        var max = Arrays.stream(values).mapToLong(Result::maxBuffered).sorted().toArray();
        var middle = values.length / 2;
        return new Result(throughput[middle], p50[middle], p99[middle], p999[middle], max[middle]);
    }

    private static long percentile(long[] values, double percentile) {
        if (values.length == 0) {
            return 0;
        }
        var index = Math.min(values.length - 1, (int) Math.ceil(values.length * percentile) - 1);
        return values[index];
    }

    private static void print(String name, Result result) {
        System.out.printf(
                "%-34s %12.0f ops/s  p50=%9.2f us  p99=%9.2f us  p99.9=%9.2f us  maxBuffered=%d%n",
                name, result.throughput(), result.p50Micros(), result.p99Micros(), result.p999Micros(), result.maxBuffered());
    }
}
