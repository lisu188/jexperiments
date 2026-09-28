package com.lis.threadpool;

import java.time.Duration;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.RejectedExecutionException;
import java.util.function.BooleanSupplier;

public final class OrderedThreadPoolExecutorVerification {
    private record Item(int producer, int ordinal) {
    }

    public static void main(String[] args) throws Exception {
        verifyReverseCompletionOrder();
        verifyFailureDoesNotCreateGap();
        verifyConcurrentProducers();
        verifyBoundedQueueBackpressure();
        verifyShutdownAndTermination();
        verifyNullResultFailsWithoutBlocking();
        System.out.println("OrderedThreadPoolExecutor verification passed");
    }

    private static void verifyReverseCompletionOrder() throws Exception {
        var taskCount = 12;
        var output = new ArrayBlockingQueue<Integer>(taskCount);
        var gates = new ArrayList<CountDownLatch>(taskCount);
        var computations = new ArrayList<CountDownLatch>(taskCount);
        var started = new CountDownLatch(taskCount);

        try (var workers = Executors.newVirtualThreadPerTaskExecutor();
             var executor = new OrderedThreadPoolExecutor<Integer>(output, workers)) {
            for (int i = 0; i < taskCount; i++) {
                gates.add(new CountDownLatch(1));
                computations.add(new CountDownLatch(1));
            }

            var futures = new ArrayList<CompletableFuture<Integer>>(taskCount);
            for (int i = 0; i < taskCount; i++) {
                var value = i;
                futures.add(executor.process(() -> {
                    started.countDown();
                    await(gates.get(value));
                    computations.get(value).countDown();
                    return value;
                }));
            }

            started.await();
            for (int i = taskCount - 1; i >= 0; i--) {
                gates.get(i).countDown();
                computations.get(i).await();
            }

            var published = new ArrayList<Integer>(taskCount);
            for (int i = 0; i < taskCount; i++) {
                published.add(output.take());
            }
            futures.forEach(CompletableFuture::join);

            var expected = new ArrayList<Integer>(taskCount);
            for (int i = 0; i < taskCount; i++) {
                expected.add(i);
            }
            require(published.equals(expected), "reverse completion must preserve submission order");

            var stats = executor.statistics();
            require(stats.submitted() == taskCount, "reverse submitted count");
            require(stats.completed() == taskCount, "reverse completed count");
            require(stats.published() == taskCount, "reverse published count");
            require(stats.failed() == 0, "reverse failure count");
            require(stats.buffered() == 0, "reverse buffer drained");
            require(stats.nextSequence() == taskCount, "reverse next sequence");
        }
    }

    private static void verifyFailureDoesNotCreateGap() throws Exception {
        var output = new ArrayBlockingQueue<Integer>(3);

        try (var workers = Executors.newVirtualThreadPerTaskExecutor();
             var executor = new OrderedThreadPoolExecutor<Integer>(output, workers)) {
            var first = executor.process(() -> 10);
            var failed = executor.process(() -> {
                throw new IllegalStateException("expected");
            });
            var third = executor.process(() -> 30);

            require(first.join() == 10, "first future");
            expectFailure(failed, IllegalStateException.class);
            require(third.join() == 30, "later future must not be blocked by failure");
            require(List.of(output.take(), output.take()).equals(List.of(10, 30)),
                    "failed slot must be skipped in output");

            var stats = executor.statistics();
            require(stats.submitted() == 3, "failure submitted count");
            require(stats.completed() == 3, "failure completed count");
            require(stats.published() == 2, "failure published count");
            require(stats.failed() == 1, "failure count");
            require(stats.buffered() == 0, "failure buffer drained");
            require(stats.nextSequence() == 3, "failure next sequence");
        }
    }

    private static void verifyConcurrentProducers() throws Exception {
        var producerCount = 6;
        var itemsPerProducer = 200;
        var total = producerCount * itemsPerProducer;
        var output = new ArrayBlockingQueue<Item>(total);

        try (var workers = Executors.newVirtualThreadPerTaskExecutor();
             var executor = new OrderedThreadPoolExecutor<Item>(output, workers);
             var producers = Executors.newVirtualThreadPerTaskExecutor()) {
            var producerTasks = new ArrayList<java.util.concurrent.Future<?>>(producerCount);
            for (int producer = 0; producer < producerCount; producer++) {
                var producerId = producer;
                producerTasks.add(producers.submit(() -> {
                    for (int ordinal = 0; ordinal < itemsPerProducer; ordinal++) {
                        var item = new Item(producerId, ordinal);
                        executor.process(() -> item);
                    }
                }));
            }
            for (var producerTask : producerTasks) {
                producerTask.get();
            }

            var nextOrdinal = new HashMap<Integer, Integer>();
            for (int i = 0; i < total; i++) {
                var item = output.take();
                var expected = nextOrdinal.getOrDefault(item.producer(), 0);
                require(item.ordinal() == expected, "per-producer submission order");
                nextOrdinal.put(item.producer(), expected + 1);
            }

            for (int producer = 0; producer < producerCount; producer++) {
                require(nextOrdinal.getOrDefault(producer, 0) == itemsPerProducer, "producer item count");
            }
        }
    }

    private static void verifyBoundedQueueBackpressure() throws Exception {
        var output = new ArrayBlockingQueue<Integer>(1);

        try (var workers = Executors.newVirtualThreadPerTaskExecutor();
             var executor = new OrderedThreadPoolExecutor<Integer>(output, workers)) {
            var first = executor.process(() -> 1);
            var second = executor.process(() -> 2);

            waitUntil(first::isDone, Duration.ofSeconds(5), "first publication");
            require(!second.isDone(), "second future must wait while output queue is full");
            require(output.take() == 1, "first bounded output");
            require(second.join() == 2, "second future after backpressure release");
            require(output.take() == 2, "second bounded output");
        }
    }

    private static void verifyShutdownAndTermination() throws Exception {
        var output = new ArrayBlockingQueue<Integer>(2);
        var gate = new CountDownLatch(1);

        try (var workers = Executors.newVirtualThreadPerTaskExecutor();
             var executor = new OrderedThreadPoolExecutor<Integer>(output, workers)) {
            var future = executor.process(() -> {
                await(gate);
                return 1;
            });

            executor.shutdown();
            require(executor.isShutdown(), "shutdown state");
            expectRejected(() -> executor.process(() -> 2));
            require(!executor.awaitTermination(Duration.ofMillis(25)), "running task must delay termination");

            gate.countDown();
            require(future.join() == 1, "in-flight task completes after shutdown");
            require(output.take() == 1, "in-flight result published after shutdown");
            require(executor.awaitTermination(Duration.ofSeconds(5)), "executor termination");
            require(executor.isTerminated(), "terminated state");
        }
    }

    private static void verifyNullResultFailsWithoutBlocking() throws Exception {
        var output = new ArrayBlockingQueue<Integer>(2);

        try (var workers = Executors.newVirtualThreadPerTaskExecutor();
             var executor = new OrderedThreadPoolExecutor<Integer>(output, workers)) {
            var nullFuture = executor.process(() -> null);
            var later = executor.process(() -> 2);

            expectFailure(nullFuture, NullPointerException.class);
            require(later.join() == 2, "null result must not block later result");
            require(output.take() == 2, "later result after null failure");
        }
    }

    private static void expectFailure(CompletableFuture<?> future, Class<? extends Throwable> expectedType) {
        try {
            future.join();
            throw new AssertionError("expected future failure");
        } catch (CompletionException failure) {
            require(expectedType.isInstance(failure.getCause()), "unexpected future failure type");
        }
    }

    private static void expectRejected(Runnable operation) {
        try {
            operation.run();
            throw new AssertionError("expected rejection");
        } catch (RejectedExecutionException expected) {
        }
    }

    private static void waitUntil(BooleanSupplier condition, Duration timeout, String description) {
        var deadline = System.nanoTime() + timeout.toNanos();
        while (!condition.getAsBoolean()) {
            if (System.nanoTime() - deadline >= 0) {
                throw new AssertionError("Timed out waiting for " + description);
            }
            Thread.onSpinWait();
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

    private static void require(boolean condition, String message) {
        if (!condition) {
            throw new AssertionError(message);
        }
    }
}
