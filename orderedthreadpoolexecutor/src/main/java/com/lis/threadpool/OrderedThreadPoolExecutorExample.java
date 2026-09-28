package com.lis.threadpool;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;

public final class OrderedThreadPoolExecutorExample {
    public static void main(String[] args) throws Exception {
        reverseCompletionStillPublishesInOrder();
        failuresDoNotBlockLaterResults();
        boundedQueueAppliesBackpressure();
        printStatistics();
    }

    private static void reverseCompletionStillPublishesInOrder() throws Exception {
        section("1. Reverse completion, ordered publication");
        var taskCount = 6;
        var output = new ArrayBlockingQueue<Integer>(taskCount);
        var gates = new ArrayList<CountDownLatch>(taskCount);
        var completed = new ArrayList<CountDownLatch>(taskCount);
        var started = new CountDownLatch(taskCount);

        try (var workers = Executors.newVirtualThreadPerTaskExecutor();
             var executor = new OrderedThreadPoolExecutor<Integer>(output, workers)) {
            for (int i = 0; i < taskCount; i++) {
                gates.add(new CountDownLatch(1));
                completed.add(new CountDownLatch(1));
            }

            var futures = new ArrayList<CompletableFuture<Integer>>(taskCount);
            for (int i = 0; i < taskCount; i++) {
                var value = i;
                futures.add(executor.process(() -> {
                    started.countDown();
                    await(gates.get(value));
                    completed.get(value).countDown();
                    return value;
                }));
            }

            started.await();
            for (int i = taskCount - 1; i >= 0; i--) {
                gates.get(i).countDown();
                completed.get(i).await();
            }

            var published = new ArrayList<Integer>(taskCount);
            for (int i = 0; i < taskCount; i++) {
                published.add(output.take());
            }
            futures.forEach(CompletableFuture::join);

            System.out.println("worker completion order = reverse submission order");
            System.out.println("published order = " + published);
        }
    }

    private static void failuresDoNotBlockLaterResults() throws Exception {
        section("2. Failure is an ordered terminal slot, not a permanent gap");
        var output = new ArrayBlockingQueue<Integer>(3);

        try (var workers = Executors.newVirtualThreadPerTaskExecutor();
             var executor = new OrderedThreadPoolExecutor<Integer>(output, workers)) {
            var first = executor.process(() -> 10);
            var failed = executor.process(() -> {
                throw new IllegalStateException("expected failure");
            });
            var third = executor.process(() -> 30);

            System.out.println("published successes = " + List.of(output.take(), output.take()));
            System.out.println("first future = " + first.join());
            try {
                failed.join();
            } catch (RuntimeException failure) {
                System.out.println("failed future cause = " + failure.getCause().getMessage());
            }
            System.out.println("third future = " + third.join());
        }
    }

    private static void boundedQueueAppliesBackpressure() throws Exception {
        section("3. BlockingQueue capacity applies publication backpressure");
        var output = new ArrayBlockingQueue<Integer>(1);

        try (var workers = Executors.newVirtualThreadPerTaskExecutor();
             var executor = new OrderedThreadPoolExecutor<Integer>(output, workers)) {
            var first = executor.process(() -> 1);
            var second = executor.process(() -> 2);

            System.out.println("first published = " + output.take());
            System.out.println("first future = " + first.join());
            System.out.println("second published after capacity is released = " + output.take());
            System.out.println("second future = " + second.join());
        }
    }

    private static void printStatistics() throws Exception {
        section("4. Statistics and lifecycle");
        var output = new ArrayBlockingQueue<Integer>(4);

        try (var workers = Executors.newVirtualThreadPerTaskExecutor();
             var executor = new OrderedThreadPoolExecutor<Integer>(output, workers)) {
            var first = executor.process(() -> 1);
            var second = executor.process(() -> 2);
            first.join();
            second.join();
            System.out.println("statistics = " + executor.statistics());
            executor.shutdown();
            System.out.println("isShutdown = " + executor.isShutdown());
        }
    }

    private static void await(CountDownLatch latch) {
        try {
            latch.await();
        } catch (InterruptedException interruption) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("task interrupted", interruption);
        }
    }

    private static void section(String title) {
        System.out.println();
        System.out.println("=== " + title + " ===");
    }
}
