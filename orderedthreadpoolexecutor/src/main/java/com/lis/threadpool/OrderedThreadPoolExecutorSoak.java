package com.lis.threadpool;

import java.time.Duration;
import java.util.SplittableRandom;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.LongAdder;
import java.util.concurrent.locks.LockSupport;

public final class OrderedThreadPoolExecutorSoak {
    public static void main(String[] args) throws Exception {
        var seconds = args.length > 0 ? Integer.parseInt(args[0]) : 60;
        var producerCount = args.length > 1 ? Integer.parseInt(args[1]) : 8;
        var output = new LinkedBlockingQueue<Long>();
        var options = OrderedThreadPoolExecutor.Options.defaults().withMaxInFlight(100_000);
        var consumed = new LongAdder();

        try (var workers = Executors.newVirtualThreadPerTaskExecutor();
             var producers = Executors.newVirtualThreadPerTaskExecutor();
             var executor = new OrderedThreadPoolExecutor<Long>(output, workers, options)) {
            var consumer = Thread.ofPlatform().daemon(true).start(() -> {
                try {
                    while (!executor.isTerminated() || !output.isEmpty()) {
                        if (output.poll(100, TimeUnit.MILLISECONDS) != null) {
                            consumed.increment();
                        }
                    }
                } catch (InterruptedException interruption) {
                    Thread.currentThread().interrupt();
                }
            });

            var deadline = System.nanoTime() + Duration.ofSeconds(seconds).toNanos();
            var producerTasks = new java.util.ArrayList<java.util.concurrent.Future<?>>();
            for (int producer = 0; producer < producerCount; producer++) {
                var seed = producer + 1L;
                producerTasks.add(producers.submit(() -> {
                    var random = new SplittableRandom(seed);
                    while (System.nanoTime() < deadline) {
                        var dice = random.nextInt(10_000);
                        executor.executeOrdered(() -> {
                            if (dice == 0) {
                                throw new IllegalStateException("synthetic");
                            }
                            if (dice < 100) {
                                LockSupport.parkNanos((dice + 1L) * 1_000L);
                            }
                            return (long) dice;
                        });
                    }
                }));
            }
            for (var task : producerTasks) {
                task.get();
            }
            executor.shutdown();
            if (!executor.awaitTermination(Duration.ofMinutes(5))) {
                throw new AssertionError("executor did not terminate");
            }
            consumer.join(TimeUnit.MINUTES.toMillis(5));
            if (consumer.isAlive()) {
                consumer.interrupt();
                throw new AssertionError("consumer did not terminate");
            }

            var stats = executor.statistics();
            if (stats.submitted() != stats.completed()
                    || stats.nextSequence() != stats.submitted()
                    || stats.buffered() != 0
                    || stats.published() + stats.failed() != stats.submitted()
                    || consumed.sum() != stats.published()) {
                throw new AssertionError("soak invariants failed: " + stats + ", consumed=" + consumed.sum());
            }
            System.out.println("soak statistics = " + stats);
        }
    }
}
