package com.lis.threadpool;

import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.util.concurrent.CompletionException;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.*;

class OrderedThreadPoolExecutorTest {
    @Test
    void optionsValidationAndFluentCopiesAreCovered() {
        var defaults = OrderedThreadPoolExecutor.Options.defaults();

        assertEquals(0, defaults.maxInFlight());
        assertEquals(256, defaults.publisherBatchSize());
        assertEquals(64, defaults.publisherSpinCount());
        assertEquals(1024, defaults.segmentSize());
        assertTrue(defaults.asyncFutureCompletion());
        assertTrue(defaults.virtualPublisher());
        assertTrue(defaults.virtualFutureNotifier());

        assertThrows(IllegalArgumentException.class,
                () -> new OrderedThreadPoolExecutor.Options(-1, 1, 0, 16, true, true, true));
        assertThrows(IllegalArgumentException.class,
                () -> new OrderedThreadPoolExecutor.Options(0, 0, 0, 16, true, true, true));
        assertThrows(IllegalArgumentException.class,
                () -> new OrderedThreadPoolExecutor.Options(0, 1, -1, 16, true, true, true));
        assertThrows(IllegalArgumentException.class,
                () -> new OrderedThreadPoolExecutor.Options(0, 1, 0, 8, true, true, true));
        assertThrows(IllegalArgumentException.class,
                () -> new OrderedThreadPoolExecutor.Options(0, 1, 0, 24, true, true, true));

        var changed = defaults
                .withMaxInFlight(7)
                .withPublisherBatchSize(3)
                .withPublisherSpinCount(0)
                .withSegmentSize(32)
                .withAsyncFutureCompletion(false)
                .withVirtualPublisher(false)
                .withVirtualFutureNotifier(false);

        assertEquals(7, changed.maxInFlight());
        assertEquals(3, changed.publisherBatchSize());
        assertEquals(0, changed.publisherSpinCount());
        assertEquals(32, changed.segmentSize());
        assertFalse(changed.asyncFutureCompletion());
        assertFalse(changed.virtualPublisher());
        assertFalse(changed.virtualFutureNotifier());
    }

    @Test
    void completionBufferCoversSparseResizeDuplicateRemovalAndCompaction() {
        assertThrows(IllegalArgumentException.class, () -> new OrderedCompletionBuffer<>(8));
        assertThrows(IllegalArgumentException.class, () -> new OrderedCompletionBuffer<>(24));

        var buffer = new OrderedCompletionBuffer<String>(16);
        assertTrue(buffer.isEmpty());
        assertEquals(0, buffer.size());
        assertNull(buffer.remove(123));
        assertNull(buffer.removeAny());
        assertThrows(NullPointerException.class, () -> buffer.put(0, null));

        buffer.put(0, "zero");
        buffer.put(1, "one");
        assertThrows(IllegalStateException.class, () -> buffer.put(0, "duplicate"));
        assertEquals(2, buffer.size());
        assertEquals("zero", buffer.remove(0));
        assertFalse(buffer.isEmpty());
        assertEquals("one", buffer.remove(1));
        assertTrue(buffer.isEmpty());

        for (int segment = 0; segment < 40; segment++) {
            long sequence = (long) segment * 16;
            buffer.put(sequence, "v" + segment);
        }
        assertEquals(40, buffer.size());

        for (int segment = 0; segment < 25; segment++) {
            long sequence = (long) segment * 16;
            assertEquals("v" + segment, buffer.remove(sequence));
        }
        assertEquals(15, buffer.size());

        var removed = 0;
        while (buffer.removeAny() != null) {
            removed++;
        }
        assertEquals(15, removed);
        assertTrue(buffer.isEmpty());

        buffer.put(1_000_000, "far");
        assertNull(buffer.remove(1_000_001));
        assertEquals("far", buffer.remove(1_000_000));
        assertTrue(buffer.isEmpty());
    }

    @Test
    void synchronousFutureCompletionCoversSuccessNullFailureAndShutdown() throws Exception {
        var output = new LinkedBlockingQueue<Integer>();

        try (var workers = Executors.newSingleThreadExecutor()) {
            assertThrows(NullPointerException.class,
                    () -> new OrderedThreadPoolExecutor<Integer>(null, workers));
            assertThrows(NullPointerException.class,
                    () -> new OrderedThreadPoolExecutor<Integer>(output, null));
            assertThrows(NullPointerException.class,
                    () -> new OrderedThreadPoolExecutor<Integer>(output, workers, null));
        }
        var options = OrderedThreadPoolExecutor.Options.defaults()
                .withAsyncFutureCompletion(false)
                .withVirtualPublisher(false)
                .withPublisherSpinCount(0);

        try (var workers = Executors.newFixedThreadPool(2);
             var executor = new OrderedThreadPoolExecutor<Integer>(output, workers, options)) {
            assertFalse(executor.isShutdown());
            assertFalse(executor.isTerminated());
            assertFalse(executor.awaitTermination(Duration.ZERO));
            assertThrows(NullPointerException.class, () -> executor.awaitTermination(null));
            assertThrows(IllegalArgumentException.class,
                    () -> executor.awaitTermination(Duration.ofMillis(-1)));
            assertThrows(NullPointerException.class, () -> executor.process(null));
            assertThrows(NullPointerException.class, () -> executor.executeOrdered(null));

            var success = executor.process(() -> 11);
            var returnedNull = executor.process(() -> null);
            var failed = executor.process(() -> {
                throw new IllegalStateException("boom");
            });
            var fireAndForgetSequence = executor.executeOrdered(() -> 22);

            assertEquals(3, fireAndForgetSequence);
            assertEquals(11, success.get(5, TimeUnit.SECONDS));
            assertThrows(CompletionException.class, returnedNull::join);
            var failure = assertThrows(CompletionException.class, failed::join);
            assertInstanceOf(IllegalStateException.class, failure.getCause());
            assertEquals(11, output.poll(5, TimeUnit.SECONDS));
            assertEquals(22, output.poll(5, TimeUnit.SECONDS));

            var stats = executor.statistics();
            assertEquals(4, stats.submitted());
            assertEquals(4, stats.completed());
            assertEquals(2, stats.published());
            assertEquals(2, stats.failed());
            assertEquals(0, stats.buffered());
            assertEquals(4, stats.nextSequence());

            executor.shutdown();
            executor.shutdown();
            assertTrue(executor.isShutdown());
            assertThrows(RejectedExecutionException.class, () -> executor.process(() -> 33));
            assertTrue(executor.awaitTermination(Duration.ofSeconds(5)));
            assertTrue(executor.isTerminated());
        }
    }

    @Test
    void asynchronousPlatformNotifierCompletesFuture() throws Exception {
        var output = new LinkedBlockingQueue<Integer>();
        var options = OrderedThreadPoolExecutor.Options.defaults()
                .withVirtualPublisher(false)
                .withVirtualFutureNotifier(false)
                .withPublisherBatchSize(1)
                .withPublisherSpinCount(0);

        try (var workers = Executors.newSingleThreadExecutor();
             var executor = new OrderedThreadPoolExecutor<Integer>(output, workers, options)) {
            var future = executor.process(() -> 7);
            assertEquals(7, future.get(5, TimeUnit.SECONDS));
            assertEquals(7, output.poll(5, TimeUnit.SECONDS));
        }
    }

    @Test
    void rejectedWorkerSubmissionIsAccountedAndExecutorStillTerminates() throws Exception {
        var output = new LinkedBlockingQueue<Integer>();
        var workers = Executors.newSingleThreadExecutor();
        workers.shutdown();

        try (var executor = new OrderedThreadPoolExecutor<Integer>(
                output,
                workers,
                OrderedThreadPoolExecutor.Options.defaults().withAsyncFutureCompletion(false))) {
            assertThrows(RejectedExecutionException.class, () -> executor.process(() -> 1));
            executor.shutdown();
            assertTrue(executor.awaitTermination(Duration.ofSeconds(5)));

            var stats = executor.statistics();
            assertEquals(1, stats.submitted());
            assertEquals(1, stats.completed());
            assertEquals(1, stats.failed());
            assertEquals(0, stats.inFlight());
        }
    }

    @Test
    void interruptedBackpressureRestoresInterruptAndStillSubmits() throws Exception {
        var output = new LinkedBlockingQueue<Integer>();
        var gate = new CountDownLatch(1);
        var started = new CountDownLatch(1);
        var finished = new CountDownLatch(1);
        var interruptRestored = new AtomicBoolean();
        var options = OrderedThreadPoolExecutor.Options.defaults()
                .withMaxInFlight(1)
                .withVirtualPublisher(false)
                .withPublisherSpinCount(0);

        try (var workers = Executors.newFixedThreadPool(2);
             var executor = new OrderedThreadPoolExecutor<Integer>(output, workers, options)) {
            executor.executeOrdered(() -> {
                try {
                    gate.await();
                } catch (InterruptedException failure) {
                    Thread.currentThread().interrupt();
                    throw new IllegalStateException(failure);
                }
                return 1;
            });

            var submitter = Thread.ofPlatform().start(() -> {
                started.countDown();
                executor.executeOrdered(() -> 2);
                interruptRestored.set(Thread.currentThread().isInterrupted());
                Thread.interrupted();
                finished.countDown();
            });

            assertTrue(started.await(5, TimeUnit.SECONDS));
            submitter.interrupt();
            Thread.sleep(20);
            gate.countDown();

            assertTrue(finished.await(5, TimeUnit.SECONDS));
            submitter.join();
            assertTrue(interruptRestored.get());
            assertEquals(1, output.poll(5, TimeUnit.SECONDS));
            assertEquals(2, output.poll(5, TimeUnit.SECONDS));
        }
    }

    @Test
    void publisherAndFutureNotifierRecoverFromInterrupts() throws Exception {
        var output = new LinkedBlockingQueue<Integer>();
        var options = OrderedThreadPoolExecutor.Options.defaults()
                .withVirtualPublisher(false)
                .withVirtualFutureNotifier(false)
                .withPublisherSpinCount(0);

        try (var workers = Executors.newSingleThreadExecutor();
             var executor = new OrderedThreadPoolExecutor<Integer>(output, workers, options)) {
            var first = executor.process(() -> 1);
            assertEquals(1, first.get(5, TimeUnit.SECONDS));
            assertEquals(1, output.poll(5, TimeUnit.SECONDS));

            var publisherField = OrderedThreadPoolExecutor.class.getDeclaredField("publisherThread");
            publisherField.setAccessible(true);
            ((Thread) publisherField.get(executor)).interrupt();

            var notifierField = OrderedThreadPoolExecutor.class.getDeclaredField("futureNotifierThread");
            notifierField.setAccessible(true);
            var notifier = (Thread) notifierField.get(executor);
            assertNotNull(notifier);
            notifier.interrupt();

            var second = executor.process(() -> 2);
            assertEquals(2, second.get(5, TimeUnit.SECONDS));
            assertEquals(2, output.poll(5, TimeUnit.SECONDS));
        }
    }

    @Test
    void publicationFailureCompletesFutureExceptionallyAndDoesNotCreateGap() throws Exception {
        var output = new LinkedBlockingQueue<Integer>() {
            @Override
            public void put(Integer value) {
                throw new IllegalStateException("publication failed");
            }
        };

        try (var workers = Executors.newSingleThreadExecutor();
             var executor = new OrderedThreadPoolExecutor<Integer>(
                     output,
                     workers,
                     OrderedThreadPoolExecutor.Options.defaults().withAsyncFutureCompletion(false))) {
            var future = executor.process(() -> 1);
            var failure = assertThrows(CompletionException.class, future::join);
            assertInstanceOf(IllegalStateException.class, failure.getCause());

            executor.shutdown();
            assertTrue(executor.awaitTermination(Duration.ofSeconds(5)));
            assertEquals(1, executor.statistics().failed());
        }
    }
}
