package com.lis.distributed.thread.pool;

import com.lis.distributed.thread.pool.client.ThreadPoolClient;
import com.lis.distributed.thread.pool.server.ThreadPoolServer;

import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.CompletableFuture;

public final class DistributedThreadPoolBenchmark {
    private static final int DEFAULT_REQUESTS = 10_000;
    private static final int DEFAULT_REPETITIONS = 5;

    public static void main(String[] args) throws Exception {
        var requests = args.length > 0 ? Integer.parseInt(args[0]) : DEFAULT_REQUESTS;
        var repetitions = args.length > 1 ? Integer.parseInt(args[1]) : DEFAULT_REPETITIONS;
        var sequential = new double[repetitions];
        var pipelined = new double[repetitions];

        for (int i = 0; i < repetitions; i++) {
            sequential[i] = runSequential(Math.min(requests, 2_000));
            pipelined[i] = runPipelined(requests);
        }
        Arrays.sort(sequential);
        Arrays.sort(pipelined);
        System.out.printf("DistributedThreadPool benchmark: %,d requests%n", requests);
        System.out.printf("sequential RTT median: %.3f us%n", sequential[repetitions / 2]);
        System.out.printf("pipelined throughput median: %.0f req/s%n", pipelined[repetitions / 2]);
    }

    private static double runSequential(int requests) throws Exception {
        try (var server = new ThreadPoolServer(0).start();
             var client = new ThreadPoolClient("127.0.0.1", server.port())) {
            client.awaitClientId(Duration.ofSeconds(5));
            var start = System.nanoTime();
            for (int i = 0; i < requests; i++) {
                if (client.callOnServer(() -> 1).join() != 1) {
                    throw new AssertionError();
                }
            }
            return (System.nanoTime() - start) / 1_000.0 / requests;
        }
    }

    private static double runPipelined(int requests) throws Exception {
        try (var server = new ThreadPoolServer(0).start();
             var client = new ThreadPoolClient("127.0.0.1", server.port())) {
            client.awaitClientId(Duration.ofSeconds(5));
            var futures = new ArrayList<CompletableFuture<Integer>>(requests);
            var start = System.nanoTime();
            for (int i = 0; i < requests; i++) {
                futures.add(client.callOnServer(() -> 1));
            }
            CompletableFuture.allOf(futures.toArray(CompletableFuture[]::new)).join();
            var seconds = (System.nanoTime() - start) / 1_000_000_000.0;
            return requests / seconds;
        }
    }
}
