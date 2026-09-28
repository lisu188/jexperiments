package com.lis.threadpool;

import java.time.Duration;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Supplier;

public final class OrderedThreadPoolExecutor<T> implements AutoCloseable {
    public record Statistics(
            long submitted,
            long completed,
            long published,
            long failed,
            long buffered,
            long nextSequence) {
    }

    private sealed interface Completion<T> permits Success, Failure {
        long sequence();

        CompletableFuture<T> future();
    }

    private record Success<T>(long sequence, T value, CompletableFuture<T> future)
            implements Completion<T> {
        private Success {
            Objects.requireNonNull(value, "value");
            Objects.requireNonNull(future, "future");
        }
    }

    private record Failure<T>(long sequence, Throwable cause, CompletableFuture<T> future)
            implements Completion<T> {
        private Failure {
            Objects.requireNonNull(cause, "cause");
            Objects.requireNonNull(future, "future");
        }
    }

    private final BlockingQueue<T> outputQueue;
    private final ExecutorService workerExecutor;
    private final ExecutorService publisherExecutor;
    private final Map<Long, Completion<T>> completionBuffer = new HashMap<>();
    private final AtomicLong nextAssignedSequence = new AtomicLong();
    private final AtomicLong completedCount = new AtomicLong();
    private final AtomicLong publishedCount = new AtomicLong();
    private final AtomicLong failedCount = new AtomicLong();
    private final AtomicLong bufferedCount = new AtomicLong();
    private final AtomicLong nextPublishSequence = new AtomicLong();
    private final Object lifecycleMonitor = new Object();

    private boolean accepting = true;
    private int inFlightWorkers;

    public OrderedThreadPoolExecutor(BlockingQueue<T> outputQueue, ExecutorService workerExecutor) {
        this.outputQueue = Objects.requireNonNull(outputQueue, "outputQueue");
        this.workerExecutor = Objects.requireNonNull(workerExecutor, "workerExecutor");
        this.publisherExecutor = Executors.newSingleThreadExecutor(
                Thread.ofVirtual().name("ordered-result-publisher-", 0).factory());
    }

    public CompletableFuture<T> process(Supplier<? extends T> task) {
        Objects.requireNonNull(task, "task");

        var future = new CompletableFuture<T>();
        final long sequence;
        synchronized (lifecycleMonitor) {
            if (!accepting) {
                throw new RejectedExecutionException("OrderedThreadPoolExecutor is shut down");
            }
            sequence = nextAssignedSequence.getAndIncrement();
            inFlightWorkers++;
        }

        try {
            workerExecutor.execute(() -> executeTask(sequence, task, future));
        } catch (RuntimeException | Error failure) {
            completedCount.incrementAndGet();
            try {
                enqueueCompletion(new Failure<>(sequence, failure, future));
            } finally {
                workerFinished();
            }
            throw failure;
        }

        return future;
    }

    public Statistics statistics() {
        return new Statistics(
                nextAssignedSequence.get(),
                completedCount.get(),
                publishedCount.get(),
                failedCount.get(),
                bufferedCount.get(),
                nextPublishSequence.get());
    }

    public void shutdown() {
        synchronized (lifecycleMonitor) {
            if (!accepting) {
                return;
            }
            accepting = false;
            if (inFlightWorkers == 0) {
                publisherExecutor.shutdown();
            }
            lifecycleMonitor.notifyAll();
        }
    }

    public boolean isShutdown() {
        synchronized (lifecycleMonitor) {
            return !accepting;
        }
    }

    public boolean isTerminated() {
        return isShutdown() && publisherExecutor.isTerminated();
    }

    public boolean awaitTermination(Duration timeout) throws InterruptedException {
        Objects.requireNonNull(timeout, "timeout");
        if (timeout.isNegative()) {
            throw new IllegalArgumentException("timeout must not be negative");
        }

        var timeoutNanos = timeout.toNanos();
        var start = System.nanoTime();

        synchronized (lifecycleMonitor) {
            while (inFlightWorkers != 0) {
                if (timeoutNanos <= 0) {
                    return false;
                }
                TimeUnit.NANOSECONDS.timedWait(lifecycleMonitor, timeoutNanos);
                timeoutNanos = remainingNanos(timeout, start);
            }
            if (accepting) {
                return false;
            }
            publisherExecutor.shutdown();
        }

        timeoutNanos = remainingNanos(timeout, start);
        if (timeoutNanos <= 0) {
            return publisherExecutor.isTerminated();
        }
        return publisherExecutor.awaitTermination(timeoutNanos, TimeUnit.NANOSECONDS);
    }

    @Override
    public void close() {
        shutdown();
        var interrupted = false;
        synchronized (lifecycleMonitor) {
            while (inFlightWorkers != 0) {
                try {
                    lifecycleMonitor.wait();
                } catch (InterruptedException interruption) {
                    interrupted = true;
                }
            }
            publisherExecutor.shutdown();
        }
        while (!publisherExecutor.isTerminated()) {
            try {
                publisherExecutor.awaitTermination(1, TimeUnit.DAYS);
            } catch (InterruptedException interruption) {
                interrupted = true;
            }
        }
        if (interrupted) {
            Thread.currentThread().interrupt();
        }
    }

    private void executeTask(long sequence, Supplier<? extends T> task, CompletableFuture<T> future) {
        var completion = evaluate(sequence, task, future);
        completedCount.incrementAndGet();
        try {
            enqueueCompletion(completion);
        } finally {
            workerFinished();
        }
    }

    private Completion<T> evaluate(long sequence, Supplier<? extends T> task, CompletableFuture<T> future) {
        try {
            var value = task.get();
            if (value == null) {
                return new Failure<>(sequence, new NullPointerException("task returned null"), future);
            }
            return new Success<>(sequence, value, future);
        } catch (Throwable failure) {
            return new Failure<>(sequence, failure, future);
        }
    }

    private void enqueueCompletion(Completion<T> completion) {
        try {
            publisherExecutor.execute(() -> acceptCompletion(completion));
        } catch (RejectedExecutionException rejection) {
            failedCount.incrementAndGet();
            completion.future().completeExceptionally(rejection);
            throw rejection;
        }
    }

    private void acceptCompletion(Completion<T> completion) {
        var previous = completionBuffer.put(completion.sequence(), completion);
        if (previous != null) {
            throw new IllegalStateException("Duplicate completion sequence: " + completion.sequence());
        }
        bufferedCount.incrementAndGet();
        drainAvailable();
    }

    private void drainAvailable() {
        var interrupted = false;
        while (true) {
            var expectedSequence = nextPublishSequence.get();
            var completion = completionBuffer.remove(expectedSequence);
            if (completion == null) {
                break;
            }
            bufferedCount.decrementAndGet();

            switch (completion) {
                case Success<T>(var sequence, var value, var future) -> {
                    try {
                        outputQueue.put(value);
                        publishedCount.incrementAndGet();
                        future.complete(value);
                    } catch (InterruptedException interruption) {
                        interrupted = true;
                        failedCount.incrementAndGet();
                        future.completeExceptionally(interruption);
                    } catch (RuntimeException publicationFailure) {
                        failedCount.incrementAndGet();
                        future.completeExceptionally(publicationFailure);
                    }
                }
                case Failure<T>(var sequence, var cause, var future) -> {
                    failedCount.incrementAndGet();
                    future.completeExceptionally(cause);
                }
            }

            nextPublishSequence.incrementAndGet();
        }

        if (interrupted) {
            Thread.currentThread().interrupt();
        }
    }

    private void workerFinished() {
        synchronized (lifecycleMonitor) {
            inFlightWorkers--;
            if (inFlightWorkers < 0) {
                throw new IllegalStateException("inFlightWorkers became negative");
            }
            if (!accepting && inFlightWorkers == 0) {
                publisherExecutor.shutdown();
            }
            lifecycleMonitor.notifyAll();
        }
    }

    private static long remainingNanos(Duration timeout, long startNanos) {
        return timeout.toNanos() - (System.nanoTime() - startNanos);
    }
}
