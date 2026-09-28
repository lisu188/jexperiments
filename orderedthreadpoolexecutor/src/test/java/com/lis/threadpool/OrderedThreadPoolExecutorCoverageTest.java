package com.lis.threadpool;

import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.CompletionException;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

class OrderedThreadPoolExecutorCoverageTest {
    @Test
    void optionsValidateAndAllWithersPreserveConfiguration() {
        var defaults = OrderedThreadPoolExecutor.Options.defaults();
        assertEquals(0, defaults.maxInFlight());
        assertEquals(256, defaults.publisherBatchSize());
        assertEquals(64, defaults.publisherSpinCount());
        assertEquals(1024, defaults.segmentSize());
        assertTrue(defaults.asyncFutureCompletion());
        assertTrue(defaults.virtualPublisher());
        assertTrue(defaults.virtualFutureNotifier());

        var configured = defaults
                .withMaxInFlight(3)
                .withPublisherBatchSize(4)
                .withPublisherSpinCount(0)
                .withSegmentSize(16)
                .withAsyncFutureCompletion(false)
                .withVirtualPublisher(false)
                .withVirtualFutureNotifier(false);

        assertEquals(3, configured.maxInFlight());
        assertEquals(4, configured.publisherBatchSize());
        assertEquals(0, configured.publisherSpinCount());
        assertEquals(16, configured.segmentSize());
        assertFalse(configured.asyncFutureCompletion());
        assertFalse(configured.virtualPublisher());
        assertFalse(configured.virtualFutureNotifier());

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
    }

    @Test
    void segmentedBufferCoversResizeRemovalAndValidationPaths() {
        assertThrows(IllegalArgumentException.class, () -> new OrderedCompletionBuffer<>(8));
        assertThrows(IllegalArgumentException.class, () -> new OrderedCompletionBuffer<>(24));

        var buffer = new OrderedCompletionBuffer<String>(16);
        assertTrue(buffer.isEmpty());
        assertNull(buffer.remove(0));
        assertNull(buffer.removeAny());
        assertThrows(NullPointerException.class, () -> buffer.put(0, null));

        buffer.put(0, "zero");
        assertThrows(IllegalStateException.class, () -> buffer.put(0, "duplicate"));
        assertNull(buffer.remove(1));

        for (int segment = 1; segment <= 20; segment++) {
            buffer.put((long) segment * 16, "s" + segment);
        }
        assertEquals(21, buffer.size());

        for (int segment = 0; segment <= 15; segment++) {
            assertNotNull(buffer.remove((long) segment * 16));
        }
        assertEquals(5, buffer.size());

        var removed = 0;
        while (buffer.removeAny() != null) {
            removed++;
        }
        assertEquals(5, removed);
        assertEquals(0, buffer.size());
        assertTrue(buffer.isEmpty());
    }

    @Test
    void synchronousPlatformConfigurationCoversSuccessFailureAndLifecycle() throws Exception {
        var output = new LinkedBlockingQueue<Integer>();
        var options = OrderedThreadPoolExecutor.Options.defaults()
                .withPublisherBatchSize(2)
                .withPublisherSpinCount(0)
                .withSegmentSize(16)
                .withAsyncFutureCompletion(false)
                .withVirtualPublisher(false)
                .withVirtualFutureNotifier(false);

        try (var workers = Executors.newFixedThreadPool(2);
             var executor = new OrderedThreadPoolExecutor<Integer>(output, workers, options)) {
            assertFalse(executor.isShutdown());
            assertFalse(executor.isTerminated());
            assertFalse(executor.awaitTermination(Duration.ofMillis(1)));
            assertThrows(NullPointerException.class, () -> executor.awaitTermination(null));
            assertThrows(IllegalArgumentException.class,
                    () -> executor.awaitTermination(Duration.ofMillis(-1)));
            assertThrows(NullPointerException.class, () -> executor.process(null));
            assertThrows(NullPointerException.class, () -> executor.executeOrdered(null));

            var success = executor.process(() -> 7);
            var failure = executor.process(() -> {
                throw new IllegalStateException("boom");
            });
            var nullResult = executor.process(() -> null);
            assertEquals(3, executor.executeOrdered(() -> 9));

            assertEquals(7, success.get(5, TimeUnit.SECONDS));
            assertEquals(7, output.poll(5, TimeUnit.SECONDS));
            assertEquals(9, output.poll(5, TimeUnit.SECONDS));
            assertThrows(CompletionException.class, failure::join);
            assertThrows(CompletionException.class, nullResult::join);

            var stats = executor.statistics();
            assertEquals(4, stats.submitted());
            assertEquals(4, stats.completed());
            assertEquals(2, stats.published());
            assertEquals(2, stats.failed());

            executor.shutdown();
            executor.shutdown();
            assertTrue(executor.isShutdown());
            assertThrows(RejectedExecutionException.class, () -> executor.process(() -> 11));
            assertTrue(executor.awaitTermination(Duration.ofSeconds(5)));
        }
    }

    @Test
    void rejectedWorkerExecutionIsPublishedAsFailure() throws Exception {
        var output = new LinkedBlockingQueue<Integer>();
        var workers = Executors.newSingleThreadExecutor();
        workers.shutdown();

        try (var executor = new OrderedThreadPoolExecutor<Integer>(output, workers)) {
            assertThrows(RejectedExecutionException.class, () -> executor.process(() -> 1));
            executor.shutdown();
            assertTrue(executor.awaitTermination(Duration.ofSeconds(5)));
            assertEquals(1, executor.statistics().failed());
        }
    }

    @Test
    void publisherRuntimeFailureCompletesFutureExceptionally() throws Exception {
        var output = new LinkedBlockingQueue<Integer>() {
            @Override
            public void put(Integer value) {
                throw new IllegalStateException("publication failed");
            }
        };

        try (var workers = Executors.newVirtualThreadPerTaskExecutor();
             var executor = new OrderedThreadPoolExecutor<Integer>(output, workers)) {
            var future = executor.process(() -> 1);
            var failure = assertThrows(java.util.concurrent.ExecutionException.class,
                    () -> future.get(5, TimeUnit.SECONDS));
            assertInstanceOf(IllegalStateException.class, failure.getCause());
            assertEquals(1, executor.statistics().failed());
        }
    }

    @Test
    void interruptingBlockedPublisherTurnsPublicationIntoFailure() throws Exception {
        var output = new ArrayBlockingQueue<Integer>(1);
        output.put(-1);

        try (var workers = Executors.newVirtualThreadPerTaskExecutor();
             var executor = new OrderedThreadPoolExecutor<Integer>(output, workers)) {
            var future = executor.process(() -> 1);
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
            while (executor.statistics().completed() == 0 && System.nanoTime() < deadline) {
                Thread.onSpinWait();
            }

            var field = OrderedThreadPoolExecutor.class.getDeclaredField("publisherThread");
            field.setAccessible(true);
            ((Thread) field.get(executor)).interrupt();

            var failure = assertThrows(java.util.concurrent.ExecutionException.class,
                    () -> future.get(5, TimeUnit.SECONDS));
            assertInstanceOf(InterruptedException.class, failure.getCause());
            assertEquals(-1, output.take());
        }
    }

    @Test
    void constructorRejectsNullCollaborators() {
        var queue = new LinkedBlockingQueue<Integer>();
        try (var workers = Executors.newSingleThreadExecutor()) {
            assertThrows(NullPointerException.class,
                    () -> new OrderedThreadPoolExecutor<Integer>(null, workers));
            assertThrows(NullPointerException.class,
                    () -> new OrderedThreadPoolExecutor<Integer>(queue, null));
            assertThrows(NullPointerException.class,
                    () -> new OrderedThreadPoolExecutor<Integer>(queue, workers, null));
        }
    }
}
