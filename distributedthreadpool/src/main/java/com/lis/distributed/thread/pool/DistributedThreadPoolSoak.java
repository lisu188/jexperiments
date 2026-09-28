package com.lis.distributed.thread.pool;

import com.lis.distributed.thread.pool.client.ThreadPoolClient;
import com.lis.distributed.thread.pool.server.ThreadPoolServer;

import java.time.Duration;
import java.util.SplittableRandom;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.LongAdder;

public final class DistributedThreadPoolSoak {
    public static void main(String[] args) throws Exception {
        var seconds = args.length > 0 ? Integer.parseInt(args[0]) : 60;
        var producers = args.length > 1 ? Integer.parseInt(args[1]) : 16;
        var successes = new LongAdder();
        var failures = new LongAdder();

        try (var server = new ThreadPoolServer(0).start();
             var client = new ThreadPoolClient("127.0.0.1", server.port());
             var producerPool = Executors.newVirtualThreadPerTaskExecutor()) {
            client.awaitClientId(Duration.ofSeconds(5));
            var deadline = System.nanoTime() + Duration.ofSeconds(seconds).toNanos();
            var tasks = new java.util.ArrayList<java.util.concurrent.Future<?>>();
            for (int producer = 0; producer < producers; producer++) {
                var seed = producer + 1L;
                tasks.add(producerPool.submit(() -> {
                    var random = new SplittableRandom(seed);
                    while (System.nanoTime() < deadline) {
                        var value = random.nextInt(10_000);
                        try {
                            var result = client.callOnServer(() -> {
                                if (value == 0) {
                                    throw new IllegalStateException("synthetic");
                                }
                                return value;
                            }).get(30, TimeUnit.SECONDS);
                            if (result != value) {
                                throw new AssertionError("wrong result");
                            }
                            successes.increment();
                        } catch (java.util.concurrent.ExecutionException expected) {
                            failures.increment();
                        } catch (java.util.concurrent.TimeoutException timeout) {
                            throw new java.util.concurrent.CompletionException(timeout);
                        } catch (InterruptedException interruption) {
                            Thread.currentThread().interrupt();
                            return;
                        }
                    }
                }));
            }
            for (var task : tasks) {
                task.get();
            }
            var stats = client.statistics();
            if (stats.pendingRequests() != 0 || stats.writeFailures() != 0) {
                throw new AssertionError("transport invariants failed: " + stats);
            }
            System.out.println("successes=" + successes.sum() + " failures=" + failures.sum() + " transport=" + stats);
        }
    }
}
