package com.lis.distributed.thread.pool;

import com.lis.distributed.thread.pool.client.ThreadPoolClient;
import com.lis.distributed.thread.pool.server.ThreadPoolServer;

import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class DistributedThreadPoolPerformanceMatrix {
    private record Scenario(String workers, int batchSize, boolean tcpNoDelay, int concurrency) {
    }

    private record Result(double throughput, double p50Micros, double p99Micros, double p999Micros, long batches) {
    }

    public static void main(String[] args) throws Exception {
        var requests = args.length > 0 ? Integer.parseInt(args[0]) : 5_000;
        var repetitions = args.length > 1 ? Integer.parseInt(args[1]) : 3;
        var scenarios = new LinkedHashMap<String, Scenario>();
        for (var workers : new String[]{"fixed", "virtual"}) {
            for (var batch : new int[]{1, 16, 64}) {
                for (var concurrency : new int[]{1, 8, 64}) {
                    var name = workers + "/batch" + batch + "/c" + concurrency;
                    scenarios.put(name, new Scenario(workers, batch, true, concurrency));
                }
            }
        }
        scenarios.put("virtual/batch16/c8/nodelay-off", new Scenario("virtual", 16, false, 8));

        System.out.printf("DistributedThreadPool performance matrix: %,d requests, %d repetitions%n", requests, repetitions);
        for (var entry : scenarios.entrySet()) {
            var samples = new Result[repetitions];
            for (int i = 0; i < repetitions; i++) {
                samples[i] = run(entry.getValue(), requests);
            }
            print(entry.getKey(), median(samples));
        }
    }

    private static Result run(Scenario scenario, int requests) throws Exception {
        var options = SocketAccessor.Options.defaults()
                .withWriterBatchSize(scenario.batchSize())
                .withTcpNoDelay(scenario.tcpNoDelay());
        try (var serverWorkers = workers(scenario.workers());
             var clientWorkers = workers(scenario.workers());
             var server = new ThreadPoolServer(0, serverWorkers, options).start();
             var client = new ThreadPoolClient("127.0.0.1", server.port(), clientWorkers, options)) {
            client.awaitClientId(Duration.ofSeconds(5));
            var latencies = new long[requests];
            var start = System.nanoTime();
            var next = 0;
            var active = new ArrayList<CompletableFuture<Integer>>(scenario.concurrency());
            while (next < requests || !active.isEmpty()) {
                while (next < requests && active.size() < scenario.concurrency()) {
                    var index = next++;
                    var submitted = System.nanoTime();
                    var future = client.callOnServer(() -> index);
                    future.whenComplete((value, failure) -> latencies[index] = System.nanoTime() - submitted);
                    active.add(future);
                }
                for (var iterator = active.iterator(); iterator.hasNext();) {
                    var future = iterator.next();
                    if (future.isDone()) {
                        future.join();
                        iterator.remove();
                    }
                }
                if (!active.isEmpty()) {
                    Thread.onSpinWait();
                }
            }
            var elapsed = System.nanoTime() - start;
            Arrays.sort(latencies);
            var stats = client.statistics();
            return new Result(
                    requests / (elapsed / 1_000_000_000.0),
                    percentile(latencies, 0.50) / 1_000.0,
                    percentile(latencies, 0.99) / 1_000.0,
                    percentile(latencies, 0.999) / 1_000.0,
                    stats.writerBatches());
        }
    }

    private static ExecutorService workers(String model) {
        return switch (model) {
            case "fixed" -> Executors.newFixedThreadPool(
                    Math.max(2, Math.min(Runtime.getRuntime().availableProcessors(), 16)));
            case "virtual" -> Executors.newVirtualThreadPerTaskExecutor();
            default -> throw new IllegalArgumentException(model);
        };
    }

    private static long percentile(long[] sorted, double p) {
        var index = Math.min(sorted.length - 1, Math.max(0, (int) Math.ceil(sorted.length * p) - 1));
        return sorted[index];
    }

    private static Result median(Result[] values) {
        var throughput = Arrays.stream(values).mapToDouble(Result::throughput).sorted().toArray();
        var p50 = Arrays.stream(values).mapToDouble(Result::p50Micros).sorted().toArray();
        var p99 = Arrays.stream(values).mapToDouble(Result::p99Micros).sorted().toArray();
        var p999 = Arrays.stream(values).mapToDouble(Result::p999Micros).sorted().toArray();
        var batches = Arrays.stream(values).mapToLong(Result::batches).sorted().toArray();
        var middle = values.length / 2;
        return new Result(throughput[middle], p50[middle], p99[middle], p999[middle], batches[middle]);
    }

    private static void print(String name, Result result) {
        System.out.printf(
                "%-34s %10.0f req/s p50=%8.2f us p99=%8.2f us p99.9=%8.2f us batches=%d%n",
                name, result.throughput(), result.p50Micros(), result.p99Micros(), result.p999Micros(), result.batches());
    }
}
