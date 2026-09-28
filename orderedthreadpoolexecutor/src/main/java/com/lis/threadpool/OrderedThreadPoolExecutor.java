package com.lis.threadpool;

import java.time.Duration;
import java.util.Objects;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.LinkedTransferQueue;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.Semaphore;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.LongAdder;
import java.util.function.Supplier;

public final class OrderedThreadPoolExecutor<T> implements AutoCloseable {
    private static final long CLOSED_MASK = Long.MIN_VALUE;
    private static final long COUNT_MASK = Long.MAX_VALUE;
    private static final Object STOP = new Object();

    public record Options(
            int maxInFlight,
            int publisherBatchSize,
            int publisherSpinCount,
            int segmentSize,
            boolean asyncFutureCompletion,
            boolean virtualPublisher,
            boolean virtualFutureNotifier) {
        public Options {
            if (maxInFlight < 0) {
                throw new IllegalArgumentException("maxInFlight must be >= 0");
            }
            if (publisherBatchSize < 1) {
                throw new IllegalArgumentException("publisherBatchSize must be >= 1");
            }
            if (publisherSpinCount < 0) {
                throw new IllegalArgumentException("publisherSpinCount must be >= 0");
            }
            if (segmentSize < 16 || Integer.bitCount(segmentSize) != 1) {
                throw new IllegalArgumentException("segmentSize must be a power of two >= 16");
            }
        }

        public static Options defaults() {
            return new Options(0, 256, 64, 1024, true, true, true);
        }

        public Options withMaxInFlight(int value) {
            return new Options(
                    value,
                    publisherBatchSize,
                    publisherSpinCount,
                    segmentSize,
                    asyncFutureCompletion,
                    virtualPublisher,
                    virtualFutureNotifier);
        }

        public Options withPublisherBatchSize(int value) {
            return new Options(
                    maxInFlight,
                    value,
                    publisherSpinCount,
                    segmentSize,
                    asyncFutureCompletion,
                    virtualPublisher,
                    virtualFutureNotifier);
        }

        public Options withPublisherSpinCount(int value) {
            return new Options(
                    maxInFlight,
                    publisherBatchSize,
                    value,
                    segmentSize,
                    asyncFutureCompletion,
                    virtualPublisher,
                    virtualFutureNotifier);
        }

        public Options withSegmentSize(int value) {
            return new Options(
                    maxInFlight,
                    publisherBatchSize,
                    publisherSpinCount,
                    value,
                    asyncFutureCompletion,
                    virtualPublisher,
                    virtualFutureNotifier);
        }

        public Options withAsyncFutureCompletion(boolean value) {
            return new Options(
                    maxInFlight,
                    publisherBatchSize,
                    publisherSpinCount,
                    segmentSize,
                    value,
                    virtualPublisher,
                    virtualFutureNotifier);
        }

        public Options withVirtualPublisher(boolean value) {
            return new Options(
                    maxInFlight,
                    publisherBatchSize,
                    publisherSpinCount,
                    segmentSize,
                    asyncFutureCompletion,
                    value,
                    virtualFutureNotifier);
        }

        public Options withVirtualFutureNotifier(boolean value) {
            return new Options(
                    maxInFlight,
                    publisherBatchSize,
                    publisherSpinCount,
                    segmentSize,
                    asyncFutureCompletion,
                    virtualPublisher,
                    value);
        }
    }

    public record Statistics(
            long submitted,
            long completed,
            long published,
            long failed,
            long buffered,
            long maxBuffered,
            long inboxDepth,
            long inFlight,
            long nextSequence) {
    }

    private final BlockingQueue<T> outputQueue;
    private final ExecutorService workerExecutor;
    private final Options options;
    private final LinkedTransferQueue<Object> completionInbox = new LinkedTransferQueue<>();
    private final LinkedTransferQueue<Object> futureNotificationInbox = new LinkedTransferQueue<>();
    private final OrderedCompletionBuffer<Submission> completionBuffer;
    private final AtomicLong nextAssignedSequence = new AtomicLong();
    private final LongAdder completedCount = new LongAdder();
    private final AtomicLong lifecycleState = new AtomicLong();
    private final AtomicBoolean stopEnqueued = new AtomicBoolean();
    private final Semaphore inFlightLimit;
    private final CountDownLatch publisherTerminated = new CountDownLatch(1);
    private final CountDownLatch futureNotifierTerminated = new CountDownLatch(1);
    private final Thread publisherThread;
    private final AtomicBoolean futureNotifierStarted = new AtomicBoolean();
    private volatile Thread futureNotifierThread;

    private volatile long publishedCount;
    private volatile long failedCount;
    private volatile long bufferedCount;
    private volatile long maxBufferedCount;
    private volatile long nextPublishSequence;
    private boolean publisherInterrupted;

    public OrderedThreadPoolExecutor(BlockingQueue<T> outputQueue, ExecutorService workerExecutor) {
        this(outputQueue, workerExecutor, Options.defaults());
    }

    public OrderedThreadPoolExecutor(
            BlockingQueue<T> outputQueue,
            ExecutorService workerExecutor,
            Options options) {
        this.outputQueue = Objects.requireNonNull(outputQueue, "outputQueue");
        this.workerExecutor = Objects.requireNonNull(workerExecutor, "workerExecutor");
        this.options = Objects.requireNonNull(options, "options");
        completionBuffer = new OrderedCompletionBuffer<>(options.segmentSize());
        inFlightLimit = options.maxInFlight() == 0 ? null : new Semaphore(options.maxInFlight());
        publisherThread = newThread(
                options.virtualPublisher(),
                "ordered-result-publisher",
                this::publisherLoop);
        if (!options.asyncFutureCompletion()) {
            futureNotifierTerminated.countDown();
        }
        publisherThread.start();
    }

    public CompletableFuture<T> process(Supplier<? extends T> task) {
        Objects.requireNonNull(task, "task");
        var future = new CompletableFuture<T>();
        submit(task, future);
        return future;
    }

    public long executeOrdered(Supplier<? extends T> task) {
        Objects.requireNonNull(task, "task");
        return submit(task, null);
    }

    public Statistics statistics() {
        var lifecycle = lifecycleState.get();
        return new Statistics(
                nextAssignedSequence.get(),
                completedCount.sum(),
                publishedCount,
                failedCount,
                bufferedCount,
                maxBufferedCount,
                completionInbox.size(),
                lifecycle & COUNT_MASK,
                nextPublishSequence);
    }

    public void shutdown() {
        while (true) {
            var state = lifecycleState.get();
            if ((state & CLOSED_MASK) != 0) {
                return;
            }
            var closed = state | CLOSED_MASK;
            if (lifecycleState.compareAndSet(state, closed)) {
                if ((state & COUNT_MASK) == 0) {
                    enqueueStop();
                }
                return;
            }
        }
    }

    public boolean isShutdown() {
        return (lifecycleState.get() & CLOSED_MASK) != 0;
    }

    public boolean isTerminated() {
        return publisherTerminated.getCount() == 0 && futureNotifierTerminated.getCount() == 0;
    }

    public boolean awaitTermination(Duration timeout) throws InterruptedException {
        Objects.requireNonNull(timeout, "timeout");
        if (timeout.isNegative()) {
            throw new IllegalArgumentException("timeout must not be negative");
        }
        if (!isShutdown()) {
            return false;
        }

        var deadline = System.nanoTime() + timeout.toNanos();
        if (!awaitUntil(publisherTerminated, deadline)) {
            return false;
        }
        return awaitUntil(futureNotifierTerminated, deadline);
    }

    @Override
    public void close() {
        shutdown();
        var interrupted = false;
        while (true) {
            try {
                publisherTerminated.await();
                break;
            } catch (InterruptedException interruption) {
                interrupted = true;
            }
        }
        while (true) {
            try {
                futureNotifierTerminated.await();
                break;
            } catch (InterruptedException interruption) {
                interrupted = true;
            }
        }
        if (interrupted) {
            Thread.currentThread().interrupt();
        }
    }

    private long submit(Supplier<? extends T> task, CompletableFuture<T> future) {
        acquireInFlightPermit();
        var reserved = false;
        try {
            reserveWorker();
            reserved = true;
            var sequence = nextAssignedSequence.getAndIncrement();
            var submission = new Submission(sequence, task, future);
            OrderedThreadPoolJfr.submission(sequence, future != null);
            try {
                workerExecutor.execute(submission);
            } catch (RuntimeException | Error failure) {
                submission.fail(failure);
                completedCount.increment();
                completionInbox.offer(submission);
                workerFinished();
                throw failure;
            }
            return sequence;
        } catch (RuntimeException | Error failure) {
            if (!reserved) {
                releaseInFlightPermit();
            }
            throw failure;
        }
    }

    private void executeSubmission(Submission submission) {
        try {
            var value = submission.task.get();
            if (value == null) {
                submission.fail(new NullPointerException("task returned null"));
            } else {
                submission.succeed(value);
            }
        } catch (Throwable failure) {
            submission.fail(failure);
        } finally {
            submission.task = null;
            completedCount.increment();
            OrderedThreadPoolJfr.completion(submission.sequence, submission.failure != null);
            completionInbox.offer(submission);
            workerFinished();
        }
    }

    private void publisherLoop() {
        var interrupted = false;
        var stopSeen = false;
        try {
            while (true) {
                Object event;
                try {
                    event = stopSeen ? completionInbox.poll() : nextPublisherEvent();
                } catch (InterruptedException interruption) {
                    interrupted = true;
                    continue;
                }

                if (event == null) {
                    drainAvailable();
                    if (stopSeen) {
                        finishOrFailStoppedPublisher();
                        break;
                    }
                    continue;
                }

                if (event == STOP) {
                    stopSeen = true;
                } else {
                    acceptCompletion(castSubmission(event));
                }

                var batch = 1;
                while (batch < options.publisherBatchSize()) {
                    var additional = completionInbox.poll();
                    if (additional == null) {
                        break;
                    }
                    if (additional == STOP) {
                        stopSeen = true;
                    } else {
                        acceptCompletion(castSubmission(additional));
                    }
                    batch++;
                }

                drainAvailable();
                if (stopSeen && completionInbox.isEmpty()) {
                    finishOrFailStoppedPublisher();
                    break;
                }
            }
        } finally {
            if (options.asyncFutureCompletion()) {
                if (futureNotifierStarted.get()) {
                    futureNotificationInbox.offer(STOP);
                } else {
                    futureNotifierTerminated.countDown();
                }
            }
            publisherTerminated.countDown();
            if (interrupted || publisherInterrupted) {
                Thread.currentThread().interrupt();
            }
        }
    }

    private Object nextPublisherEvent() throws InterruptedException {
        for (int i = 0; i < options.publisherSpinCount(); i++) {
            var event = completionInbox.poll();
            if (event != null) {
                return event;
            }
            Thread.onSpinWait();
        }
        return completionInbox.take();
    }

    private void putUninterruptibly(T value) {
        while (true) {
            try {
                outputQueue.put(value);
                return;
            } catch (InterruptedException interruption) {
                publisherInterrupted = true;
            }
        }
    }

    private void acceptCompletion(Submission submission) {
        completionBuffer.put(submission.sequence, submission);
        bufferedCount++;
        if (bufferedCount > maxBufferedCount) {
            maxBufferedCount = bufferedCount;
            OrderedThreadPoolJfr.highWatermark(bufferedCount, nextPublishSequence);
        }
    }

    private void drainAvailable() {
        while (true) {
            var submission = completionBuffer.remove(nextPublishSequence);
            if (submission == null) {
                if (bufferedCount != 0) {
                    OrderedThreadPoolJfr.headOfLine(nextPublishSequence, bufferedCount);
                }
                return;
            }
            bufferedCount--;

            if (submission.failure == null) {
                try {
                    var blocked = OrderedThreadPoolJfr.publisherBlocked(submission.sequence);
                    if (blocked != null) {
                        blocked.begin();
                    }
                    putUninterruptibly(submission.value);
                    OrderedThreadPoolJfr.commitPublisherBlocked(blocked);
                    publishedCount++;
                    OrderedThreadPoolJfr.publication(submission.sequence, false);
                } catch (RuntimeException publicationFailure) {
                    submission.fail(publicationFailure);
                    failedCount++;
                    OrderedThreadPoolJfr.publication(submission.sequence, true);
                }
            } else {
                failedCount++;
                OrderedThreadPoolJfr.publication(submission.sequence, true);
            }

            nextPublishSequence++;
            releaseInFlightPermit();
            notifyFuture(submission);
        }
    }

    private void notifyFuture(Submission submission) {
        if (submission.future == null) {
            submission.clearTerminalState();
            return;
        }
        if (options.asyncFutureCompletion()) {
            ensureFutureNotifierStarted();
            futureNotificationInbox.offer(submission);
        } else {
            completeFuture(submission);
        }
    }

    private void ensureFutureNotifierStarted() {
        if (!futureNotifierStarted.compareAndSet(false, true)) {
            return;
        }
        var thread = newThread(
                options.virtualFutureNotifier(),
                "ordered-future-notifier",
                this::futureNotifierLoop);
        futureNotifierThread = thread;
        thread.start();
    }

    private void futureNotifierLoop() {
        var interrupted = false;
        try {
            while (true) {
                Object event;
                try {
                    event = futureNotificationInbox.take();
                } catch (InterruptedException interruption) {
                    interrupted = true;
                    continue;
                }
                if (event == STOP) {
                    break;
                }
                completeFuture(castSubmission(event));
            }
        } finally {
            futureNotifierTerminated.countDown();
            if (interrupted) {
                Thread.currentThread().interrupt();
            }
        }
    }

    private void completeFuture(Submission submission) {
        var future = submission.future;
        if (submission.failure == null) {
            future.complete(submission.value);
        } else {
            future.completeExceptionally(submission.failure);
        }
        submission.clearTerminalState();
    }

    private void finishOrFailStoppedPublisher() {
        if (completionBuffer.isEmpty()) {
            return;
        }
        var missingSequence = nextPublishSequence;
        var failure = new IllegalStateException(
                "Publisher stopped with unresolved sequence " + missingSequence
                        + " and " + completionBuffer.size() + " buffered completions");
        Submission submission;
        while ((submission = completionBuffer.removeAny()) != null) {
            bufferedCount--;
            failedCount++;
            submission.fail(failure);
            releaseInFlightPermit();
            notifyFuture(submission);
        }
    }

    private void reserveWorker() {
        while (true) {
            var state = lifecycleState.get();
            if ((state & CLOSED_MASK) != 0) {
                throw new RejectedExecutionException("OrderedThreadPoolExecutor is shut down");
            }
            var count = state & COUNT_MASK;
            if (count == COUNT_MASK) {
                throw new RejectedExecutionException("Too many in-flight submissions");
            }
            if (lifecycleState.compareAndSet(state, state + 1)) {
                return;
            }
        }
    }

    private void workerFinished() {
        while (true) {
            var state = lifecycleState.get();
            var count = state & COUNT_MASK;
            if (count == 0) {
                throw new IllegalStateException("in-flight worker count underflow");
            }
            var next = state - 1;
            if (lifecycleState.compareAndSet(state, next)) {
                if ((next & CLOSED_MASK) != 0 && (next & COUNT_MASK) == 0) {
                    enqueueStop();
                }
                return;
            }
        }
    }

    private void enqueueStop() {
        if (stopEnqueued.compareAndSet(false, true)) {
            completionInbox.offer(STOP);
        }
    }

    private void acquireInFlightPermit() {
        if (inFlightLimit == null) {
            return;
        }
        var interrupted = false;
        try {
            while (true) {
                if (isShutdown()) {
                    throw new RejectedExecutionException("OrderedThreadPoolExecutor is shut down");
                }
                try {
                    if (inFlightLimit.tryAcquire(1, TimeUnit.MILLISECONDS)) {
                        return;
                    }
                } catch (InterruptedException interruption) {
                    interrupted = true;
                }
            }
        } finally {
            if (interrupted) {
                Thread.currentThread().interrupt();
            }
        }
    }

    private void releaseInFlightPermit() {
        if (inFlightLimit != null) {
            inFlightLimit.release();
        }
    }

    @SuppressWarnings("unchecked")
    private Submission castSubmission(Object value) {
        return (Submission) value;
    }

    private static boolean awaitUntil(CountDownLatch latch, long deadlineNanos) throws InterruptedException {
        if (latch.getCount() == 0) {
            return true;
        }
        var remaining = deadlineNanos - System.nanoTime();
        return remaining > 0 && latch.await(remaining, TimeUnit.NANOSECONDS);
    }

    private static Thread newThread(boolean virtual, String name, Runnable task) {
        ThreadFactory factory = virtual
                ? Thread.ofVirtual().name(name + "-", 0).factory()
                : Thread.ofPlatform().daemon(true).name(name + "-", 0).factory();
        return factory.newThread(task);
    }

    private final class Submission implements Runnable {
        private final long sequence;
        private Supplier<? extends T> task;
        private final CompletableFuture<T> future;
        private T value;
        private Throwable failure;

        private Submission(long sequence, Supplier<? extends T> task, CompletableFuture<T> future) {
            this.sequence = sequence;
            this.task = task;
            this.future = future;
        }

        @Override
        public void run() {
            executeSubmission(this);
        }

        private void succeed(T result) {
            value = result;
            failure = null;
        }

        private void fail(Throwable cause) {
            value = null;
            failure = Objects.requireNonNull(cause, "cause");
        }

        private void clearTerminalState() {
            value = null;
            failure = null;
        }
    }
}
