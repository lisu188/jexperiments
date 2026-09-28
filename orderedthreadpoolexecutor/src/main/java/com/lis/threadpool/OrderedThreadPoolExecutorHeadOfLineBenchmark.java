package com.lis.threadpool;

import java.time.Duration;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

public final class OrderedThreadPoolExecutorHeadOfLineBenchmark {
    private static final long[] DELAYS_NANOS = {0, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000};

    public static void main(String[] args) throws Exception {
        var tasks = args.length > 0 ? Integer.parseInt(args[0]) : 100_000;
        System.out.printf("Head-of-line benchmark: %,d tasks%n", tasks);
        for (var delay : DELAYS_NANOS) {
            run(tasks, delay);
        }
    }

    private static void run(int tasks, long delayNanos) throws Exception {
        var output = new LinkedBlockingQueue<Integer>();
        var firstGate = new CountDownLatch(1);
        var laterDone = new CountDownLatch(tasks - 1);
        try (var workers = Executors.newVirtualThreadPerTaskExecutor();
             var executor = new OrderedThreadPoolExecutor<Integer>(output, workers)) {
            var start = System.nanoTime();
            executor.executeOrdered(() -> {
                await(firstGate);
                return 0;
            });
            for (int i = 1; i < tasks; i++) {
                var value = i;
                executor.executeOrdered(() -> {
                    laterDone.countDown();
                    return value;
                });
            }
            laterDone.await();
            var beforeRelease = executor.statistics();
            if (delayNanos != 0) {
                TimeUnit.NANOSECONDS.sleep(delayNanos);
            }
            var release = System.nanoTime();
            firstGate.countDown();
            for (int expected = 0; expected < tasks; expected++) {
                var actual = output.take();
                if (actual != expected) {
                    throw new AssertionError(actual + " != " + expected);
                }
            }
            executor.shutdown();
            if (!executor.awaitTermination(Duration.ofMinutes(2))) {
                throw new AssertionError("termination timeout");
            }
            var drained = System.nanoTime();
            System.out.printf(
                    "delay=%9.3f ms buffered=%8d maxBuffered=%8d releaseToDrain=%9.3f ms total=%9.3f ms%n",
                    delayNanos / 1_000_000.0,
                    beforeRelease.buffered(),
                    executor.statistics().maxBuffered(),
                    (drained - release) / 1_000_000.0,
                    (drained - start) / 1_000_000.0);
        }
    }

    private static void await(CountDownLatch latch) {
        try {
            latch.await();
        } catch (InterruptedException interruption) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException(interruption);
        }
    }
}
