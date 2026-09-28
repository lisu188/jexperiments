package com.lis.threadpool;

import java.time.Duration;
import java.util.ArrayList;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.TimeUnit;

public final class OrderedThreadPoolExecutorVerification {
    public static void main(String[] args) throws Exception {
        reverseCompletionStillPublishesInOrder();
        failureDoesNotCreateGap();
        fireAndForgetAvoidsFuturePath();
        asyncFutureCallbackCannotBlockPublisher();
        boundedInFlightAppliesUpstreamBackpressure();
        shutdownRejectsAndDrainsAcceptedWork();
        concurrentProducersPreserveUniqueSequenceNumbers();
        segmentedBufferHandlesSparseSegments();
        System.out.println("OrderedThreadPoolExecutor verification passed");
    }

    private static void reverseCompletionStillPublishesInOrder() throws Exception {
        var count = 32;
        var output = new ArrayBlockingQueue<Integer>(count);
        var gates = new ArrayList<CountDownLatch>(count);
        var done = new ArrayList<CountDownLatch>(count);
        var started = new CountDownLatch(count);
        try (var workers = Executors.newVirtualThreadPerTaskExecutor();
             var executor = new OrderedThreadPoolExecutor<Integer>(output, workers)) {
            var futures = new ArrayList<CompletableFuture<Integer>>(count);
            for (int i = 0; i < count; i++) {
                gates.add(new CountDownLatch(1));
                done.add(new CountDownLatch(1));
            }
            for (int i = 0; i < count; i++) {
                var value = i;
                futures.add(executor.process(() -> {
                    started.countDown();
                    await(gates.get(value));
                    done.get(value).countDown();
                    return value;
                }));
            }
            started.await();
            for (int i = count - 1; i >= 0; i--) {
                gates.get(i).countDown();
                done.get(i).await();
            }
            for (int i = 0; i < count; i++) {
                require(output.take() == i, "reverse publication order");
            }
            futures.forEach(CompletableFuture::join);
            var stats = executor.statistics();
            require(stats.maxBuffered() >= count - 1, "reverse workload should create large reorder window");
            require(stats.buffered() == 0, "buffer must drain");
        }
    }

    private static void failureDoesNotCreateGap() throws Exception {
        var output = new ArrayBlockingQueue<Integer>(2);
        try (var workers = Executors.newVirtualThreadPerTaskExecutor();
             var executor = new OrderedThreadPoolExecutor<Integer>(output, workers)) {
            var first = executor.process(() -> 1);
            var failed = executor.process(() -> {
                throw new IllegalStateException("expected");
            });
            var third = executor.process(() -> 3);
            require(first.join() == 1, "first future");
            try {
                failed.join();
                throw new AssertionError("failure expected");
            } catch (CompletionException expected) {
                require(expected.getCause() instanceof IllegalStateException, "failure cause");
            }
            require(third.join() == 3, "third future");
            require(output.take() == 1, "first output");
            require(output.take() == 3, "third output");
        }
    }

    private static void fireAndForgetAvoidsFuturePath() throws Exception {
        var output = new ArrayBlockingQueue<Integer>(64);
        try (var workers = Executors.newVirtualThreadPerTaskExecutor();
             var executor = new OrderedThreadPoolExecutor<Integer>(output, workers)) {
            for (int i = 0; i < 64; i++) {
                var value = i;
                require(executor.executeOrdered(() -> value) == i, "sequence return");
            }
            for (int i = 0; i < 64; i++) {
                require(output.take() == i, "fire-and-forget order");
            }
        }
    }

    private static void asyncFutureCallbackCannotBlockPublisher() throws Exception {
        var output = new ArrayBlockingQueue<Integer>(4);
        var callbackEntered = new CountDownLatch(1);
        var releaseCallback = new CountDownLatch(1);
        var firstGate = new CountDownLatch(1);
        try (var workers = Executors.newVirtualThreadPerTaskExecutor();
             var executor = new OrderedThreadPoolExecutor<Integer>(output, workers)) {
            var first = executor.process(() -> {
                await(firstGate);
                return 1;
            });
            first.thenRun(() -> {
                callbackEntered.countDown();
                await(releaseCallback);
            });
            var second = executor.process(() -> 2);
            firstGate.countDown();
            callbackEntered.await();
            require(output.poll(5, TimeUnit.SECONDS) == 1, "first published");
            require(output.poll(5, TimeUnit.SECONDS) == 2, "publisher must not wait for user callback");
            require(!second.isDone(), "future notifier remains ordered behind blocked callback");
            releaseCallback.countDown();
            require(second.get(5, TimeUnit.SECONDS) == 2, "second future eventually completes");
        }
    }

    private static void boundedInFlightAppliesUpstreamBackpressure() throws Exception {
        var output = new ArrayBlockingQueue<Integer>(8);
        var gate = new CountDownLatch(1);
        var options = OrderedThreadPoolExecutor.Options.defaults().withMaxInFlight(2);
        try (var workers = Executors.newVirtualThreadPerTaskExecutor();
             var executor = new OrderedThreadPoolExecutor<Integer>(output, workers, options);
             var submitter = Executors.newVirtualThreadPerTaskExecutor()) {
            executor.executeOrdered(() -> {
                await(gate);
                return 1;
            });
            executor.executeOrdered(() -> 2);
            var thirdStarted = new CountDownLatch(1);
            var thirdReturned = new CountDownLatch(1);
            submitter.submit(() -> {
                thirdStarted.countDown();
                executor.executeOrdered(() -> 3);
                thirdReturned.countDown();
            });
            thirdStarted.await();
            require(!thirdReturned.await(50, TimeUnit.MILLISECONDS), "third submission should be backpressured");
            gate.countDown();
            require(output.take() == 1, "bounded first output");
            require(thirdReturned.await(5, TimeUnit.SECONDS), "third submission should resume after terminal slot drains");
            require(output.take() == 2, "bounded second output");
            require(output.take() == 3, "bounded third output");
        }
    }

    private static void shutdownRejectsAndDrainsAcceptedWork() throws Exception {
        var output = new ArrayBlockingQueue<Integer>(2);
        var gate = new CountDownLatch(1);
        try (var workers = Executors.newVirtualThreadPerTaskExecutor();
             var executor = new OrderedThreadPoolExecutor<Integer>(output, workers)) {
            var accepted = executor.process(() -> {
                await(gate);
                return 1;
            });
            executor.shutdown();
            try {
                executor.process(() -> 2);
                throw new AssertionError("rejection expected");
            } catch (RejectedExecutionException expected) {
            }
            require(!executor.awaitTermination(Duration.ofMillis(20)), "accepted worker still running");
            gate.countDown();
            require(output.take() == 1, "accepted task output");
            require(accepted.get(5, TimeUnit.SECONDS) == 1, "accepted future");
            require(executor.awaitTermination(Duration.ofSeconds(5)), "termination");
        }
    }

    private static void concurrentProducersPreserveUniqueSequenceNumbers() throws Exception {
        var producerCount = 16;
        var perProducer = 250;
        var total = producerCount * perProducer;
        var output = new LinkedBlockingQueue<Integer>();
        var sequenceSeen = new boolean[total];
        var sequenceLock = new Object();
        try (var workers = Executors.newVirtualThreadPerTaskExecutor();
             var producers = Executors.newVirtualThreadPerTaskExecutor();
             var executor = new OrderedThreadPoolExecutor<Integer>(output, workers)) {
            var done = new CountDownLatch(producerCount);
            for (int producer = 0; producer < producerCount; producer++) {
                producers.submit(() -> {
                    for (int i = 0; i < perProducer; i++) {
                        var value = i;
                        var sequence = executor.executeOrdered(() -> value);
                        synchronized (sequenceLock) {
                            require(sequence >= 0 && sequence < total, "sequence range");
                            require(!sequenceSeen[(int) sequence], "duplicate sequence");
                            sequenceSeen[(int) sequence] = true;
                        }
                    }
                    done.countDown();
                });
            }
            done.await();
            for (int i = 0; i < total; i++) {
                output.take();
            }
            for (var seen : sequenceSeen) {
                require(seen, "all sequence numbers assigned exactly once");
            }
        }
    }

    private static void segmentedBufferHandlesSparseSegments() {
        var buffer = new OrderedCompletionBuffer<String>(16);
        buffer.put(0, "zero");
        buffer.put(31, "thirty-one");
        buffer.put(1_000_000, "far");
        require("zero".equals(buffer.remove(0)), "segment zero");
        require(buffer.remove(1) == null, "missing slot");
        require("thirty-one".equals(buffer.remove(31)), "second segment");
        require("far".equals(buffer.remove(1_000_000)), "far segment");
        require(buffer.isEmpty(), "segmented buffer empty");
    }

    private static void await(CountDownLatch latch) {
        try {
            latch.await();
        } catch (InterruptedException interruption) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException(interruption);
        }
    }

    private static void require(boolean condition, String message) {
        if (!condition) {
            throw new AssertionError(message);
        }
    }
}
