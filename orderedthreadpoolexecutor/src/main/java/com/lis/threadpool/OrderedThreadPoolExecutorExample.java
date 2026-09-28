package com.lis.threadpool;

import java.util.ArrayList;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public final class OrderedThreadPoolExecutorExample {
    public static void main(String[] args) throws Exception {
        reverseCompletion();
        failureGap();
        fireAndForget();
        callbackIsolation();
        boundedInFlight();
    }

    private static void reverseCompletion() throws Exception {
        section("Reverse worker completion, ordered publication");
        var output = new ArrayBlockingQueue<Integer>(8);
        var gates = new ArrayList<CountDownLatch>();
        var completed = new ArrayList<CountDownLatch>();
        try (var workers = Executors.newVirtualThreadPerTaskExecutor();
             var executor = new OrderedThreadPoolExecutor<Integer>(output, workers)) {
            for (int i = 0; i < 8; i++) {
                gates.add(new CountDownLatch(1));
                completed.add(new CountDownLatch(1));
            }
            for (int i = 0; i < 8; i++) {
                var value = i;
                executor.process(() -> {
                    await(gates.get(value));
                    completed.get(value).countDown();
                    return value;
                });
            }
            for (int i = 7; i >= 0; i--) {
                gates.get(i).countDown();
                completed.get(i).await();
            }
            for (int i = 0; i < 8; i++) {
                System.out.print(output.take() + (i == 7 ? "\n" : " "));
            }
            System.out.println("statistics = " + executor.statistics());
        }
    }

    private static void failureGap() throws Exception {
        section("Failure consumes its sequence slot");
        var output = new ArrayBlockingQueue<Integer>(2);
        try (var workers = Executors.newVirtualThreadPerTaskExecutor();
             var executor = new OrderedThreadPoolExecutor<Integer>(output, workers)) {
            var first = executor.process(() -> 1);
            var failed = executor.process(() -> {
                throw new IllegalStateException("expected failure");
            });
            var third = executor.process(() -> 3);
            System.out.println("queue = " + output.take() + ", " + output.take());
            System.out.println("first = " + first.join());
            try {
                failed.join();
            } catch (RuntimeException expected) {
                System.out.println("middle future failed = " + expected.getCause());
            }
            System.out.println("third = " + third.join());
        }
    }

    private static void fireAndForget() throws Exception {
        section("Fire-and-forget avoids CompletableFuture allocation");
        var output = new ArrayBlockingQueue<Integer>(4);
        try (var workers = Executors.newVirtualThreadPerTaskExecutor();
             var executor = new OrderedThreadPoolExecutor<Integer>(output, workers)) {
            for (int i = 0; i < 4; i++) {
                var value = i;
                System.out.println("sequence = " + executor.executeOrdered(() -> value));
            }
            while (!output.isEmpty() || executor.statistics().published() < 4) {
                var value = output.poll(1, TimeUnit.SECONDS);
                if (value != null) {
                    System.out.println("published = " + value);
                }
            }
        }
    }

    private static void callbackIsolation() throws Exception {
        section("Slow future callback does not block publication");
        var output = new ArrayBlockingQueue<Integer>(4);
        var taskGate = new CountDownLatch(1);
        var callbackEntered = new CountDownLatch(1);
        var callbackGate = new CountDownLatch(1);
        try (var workers = Executors.newVirtualThreadPerTaskExecutor();
             var executor = new OrderedThreadPoolExecutor<Integer>(output, workers)) {
            var first = executor.process(() -> {
                await(taskGate);
                return 1;
            });
            first.thenRun(() -> {
                callbackEntered.countDown();
                await(callbackGate);
            });
            var second = executor.process(() -> 2);
            taskGate.countDown();
            callbackEntered.await();
            System.out.println("published while callback is blocked = " + output.take() + ", " + output.take());
            callbackGate.countDown();
            System.out.println("second future = " + second.get(5, TimeUnit.SECONDS));
        }
    }

    private static void boundedInFlight() throws Exception {
        section("maxInFlight bounds outstanding ordered work");
        var options = OrderedThreadPoolExecutor.Options.defaults().withMaxInFlight(1024);
        var output = new ArrayBlockingQueue<Integer>(4);
        try (var workers = Executors.newVirtualThreadPerTaskExecutor();
             var executor = new OrderedThreadPoolExecutor<Integer>(output, workers, options)) {
            executor.executeOrdered(() -> 1);
            executor.executeOrdered(() -> 2);
            System.out.println("configured maxInFlight = " + options.maxInFlight());
            System.out.println("outputs = " + output.take() + ", " + output.take());
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

    private static void section(String title) {
        System.out.println();
        System.out.println("=== " + title + " ===");
    }
}
