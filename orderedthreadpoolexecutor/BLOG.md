# Ordered Parallel Execution Without Losing Submission Order

## Why this experiment exists

OrderedThreadPoolExecutor separates parallel computation from ordered delivery. Work is submitted to a caller-owned ExecutorService and may finish in any order, but successful values reach a caller-owned BlockingQueue in submission order.

This is useful for chunked file processing, media pipelines, protocol framing, batched enrichment, and other fan-out/fan-in workloads where expensive work can run concurrently while downstream consumers still need stable ordering.

The contract is deliberately narrow:

- each accepted submission receives a monotonically increasing sequence number,
- workers execute concurrently,
- every sequence becomes a terminal success or failure,
- one internal publisher owns the reorder buffer,
- only a contiguous terminal prefix can advance,
- successful values are published to BlockingQueue,
- failures complete their per-submission future exceptionally and consume their sequence slot without publishing a value.

The distinction between completion order and publication order is the core of the experiment.

## What was wrong with the old version

The original implementation assigned an integer id, ran Supplier work on a worker pool, and serialized publication through a single finalizer executor.

Its reorder logic used TreeMap:

~~~java
_results.put(id, result);
Integer next = _results.firstKey();
while (next == _nextId) {
    _nextId++;
    _queue.put(_results.remove(next));
    next = _results.isEmpty() ? -1 : _results.firstKey();
}
~~~

The ownership idea was sound: only one thread mutated ordering state. The implementation still had important problems.

If a Supplier threw, no result was inserted for that id. Every later completion could then remain buffered forever behind a sequence number that would never exist.

The internal finalizer executor was never shut down. process returned no handle for failures. Interruption policy was implicit. The demo relied on random sleeps. TreeMap also paid O(log n) costs even though the algorithm only needs an exact lookup of the next expected sequence.

The refactor preserves serialized publication but fixes those semantics.

## Modern public API

The main type is now:

~~~java
public final class OrderedThreadPoolExecutor<T>
        implements AutoCloseable
~~~

The caller supplies both the output queue and the worker executor:

~~~java
public OrderedThreadPoolExecutor(
        BlockingQueue<T> outputQueue,
        ExecutorService workerExecutor)
~~~

The worker executor remains caller-owned. OrderedThreadPoolExecutor never shuts it down.

Submissions now return CompletableFuture:

~~~java
public CompletableFuture<T> process(
        Supplier<? extends T> task)
~~~

A successful future completes only after the value has been published in order. A failed future completes exceptionally when its sequence slot reaches the ordered head.

This means a fast task submitted later can finish computation while its future is still incomplete because an earlier sequence is unresolved. That is intentional head-of-line behavior, not a scheduler bug.

## Every accepted sequence has a terminal state

The main correctness change is that an accepted sequence cannot silently disappear.

Terminal worker outcomes are modeled with a sealed hierarchy:

~~~java
private sealed interface Completion<T>
        permits Success, Failure {
    long sequence();
    CompletableFuture<T> future();
}
~~~

Success and failure are records:

~~~java
private record Success<T>(
        long sequence,
        T value,
        CompletableFuture<T> future)
        implements Completion<T> {
}

private record Failure<T>(
        long sequence,
        Throwable cause,
        CompletableFuture<T> future)
        implements Completion<T> {
}
~~~

This avoids a status enum combined with nullable fields.

Task evaluation always creates one of those outcomes:

~~~java
private Completion<T> evaluate(
        long sequence,
        Supplier<? extends T> task,
        CompletableFuture<T> future) {
    try {
        var value = task.get();
        if (value == null) {
            return new Failure<>(
                    sequence,
                    new NullPointerException("task returned null"),
                    future);
        }
        return new Success<>(sequence, value, future);
    } catch (Throwable failure) {
        return new Failure<>(sequence, failure, future);
    }
}
~~~

Null is treated as failure because BlockingQueue does not accept null elements. Converting it to a terminal Failure prevents a publication-time NullPointerException from creating a permanent ordering gap.

If the caller-owned worker executor rejects a submission after a sequence was assigned, process still propagates the rejection to its caller, but the sequence is internally completed as Failure so later accepted work cannot become permanently blocked behind it.

## Why HashMap replaces TreeMap

The old code repeatedly asked TreeMap for firstKey. The actual algorithm does not need the smallest buffered key.

It already knows the only key that matters: nextPublishSequence.

The new reorder buffer is therefore:

~~~java
private final Map<Long, Completion<T>> completionBuffer =
        new HashMap<>();
~~~

The ordered publisher removes the exact next sequence:

~~~java
var expectedSequence = nextPublishSequence.get();
var completion =
        completionBuffer.remove(expectedSequence);
if (completion == null) {
    break;
}
~~~

After processing that slot, nextPublishSequence increments and the lookup repeats.

Expected reorder-buffer insertion and removal therefore become O(1) rather than O(log n).

The benchmark harness keeps a TreeMap version of the old reorder algorithm and compares it with the HashMap exact-key algorithm under reverse and deterministically shuffled completion orders. It also measures end-to-end OrderedThreadPoolExecutor throughput.

This is a lightweight benchmark harness rather than JMH, so small differences should not be treated as universal JVM conclusions. Its purpose is regression detection and validating large design decisions.

## Single-owner publication

Workers never mutate completionBuffer.

Every worker posts its terminal Completion to one internal publisher:

~~~java
publisherExecutor.execute(
        () -> acceptCompletion(completion));
~~~

The publisher uses a virtual-thread factory:

~~~java
this.publisherExecutor =
        Executors.newSingleThreadExecutor(
                Thread.ofVirtual()
                        .name(
                                "ordered-result-publisher-",
                                0)
                        .factory());
~~~

There is still exactly one publisher at a time. The virtual thread is appropriate because BlockingQueue.put may park under downstream backpressure.

Because only that serialized publisher touches the mutable buffer, the map does not need to be concurrent.

## Ordered draining

A completion is buffered first:

~~~java
private void acceptCompletion(
        Completion<T> completion) {
    var previous =
            completionBuffer.put(
                    completion.sequence(),
                    completion);
    if (previous != null) {
        throw new IllegalStateException(
                "Duplicate completion sequence: "
                        + completion.sequence());
    }
    bufferedCount.incrementAndGet();
    drainAvailable();
}
~~~

drainAvailable repeatedly processes a contiguous terminal prefix.

Successful publication happens before its future completes:

~~~java
outputQueue.put(value);
publishedCount.incrementAndGet();
future.complete(value);
~~~

Failure publishes no queue element:

~~~java
failedCount.incrementAndGet();
future.completeExceptionally(cause);
~~~

Both outcomes advance the sequence.

For three submissions:

1. success A
2. failure B
3. success C

the output queue contains A followed by C. B is visible through its failed CompletableFuture, but it no longer blocks C forever.

That is the central correctness improvement over the original implementation.

## Backpressure remains visible

BlockingQueue capacity is intentionally part of the behavior.

With queue capacity one, sequence 0 can publish and fill the queue. Sequence 1 then blocks in put until the consumer removes sequence 0.

Because publication is serialized, later results cannot overtake the blocked sequence.

This also means graceful close can wait indefinitely if the queue is full and no consumer is draining it. That is not hidden by the executor. Applications needing bounded shutdown time would require a different publication policy such as timed offer, fail-fast, dropping, or explicit abort.

## Lifecycle and ownership

The class now exposes:

~~~java
public void shutdown()

public boolean isShutdown()

public boolean isTerminated()

public boolean awaitTermination(Duration timeout)
        throws InterruptedException
~~~

shutdown stops new submissions immediately.

Already accepted worker tasks are allowed to finish. The publisher is shut down only after every in-flight worker has handed off its Completion.

~~~java
if (!accepting && inFlightWorkers == 0) {
    publisherExecutor.shutdown();
}
~~~

This avoids racing shutdown against workers that still need to enqueue terminal outcomes.

close performs graceful cleanup of the internally owned publisher. It intentionally leaves workerExecutor alone because that executor belongs to the caller.

close remembers interruption while waiting, finishes owned-resource cleanup, then restores the caller thread's interrupt flag. awaitTermination remains the interruptible timeout-oriented API.

## Statistics

The executor exposes a record:

~~~java
public record Statistics(
        long submitted,
        long completed,
        long published,
        long failed,
        long buffered,
        long nextSequence) {
}
~~~

The counters make head-of-line behavior visible:

- submitted: sequence numbers assigned,
- completed: worker computations that reached terminal state,
- published: values successfully placed on the output queue,
- failed: ordered terminal failures,
- buffered: completions waiting for earlier slots,
- nextSequence: next sequence expected by the publisher.

These are independent atomics. statistics is therefore an observational snapshot rather than a transactionally consistent state image.

After a fully drained workload useful invariants are:

~~~text
completed == submitted
published + failed == nextSequence
buffered == 0
nextSequence == submitted
~~~

The verification harness checks these invariants.

## Deterministic examples

The old main method used ThreadLocalRandom sleep to make out-of-order completion likely.

The new OrderedThreadPoolExecutorExample uses CountDownLatch gates instead. Tasks are deliberately released in reverse order, forcing computation to finish backwards while publication must still produce:

~~~text
[0, 1, 2, 3, 4, 5]
~~~

The example also demonstrates:

- a failed middle task followed by a later success,
- bounded-queue backpressure,
- statistics,
- explicit shutdown.

The example is illustrative. Correctness is handled by a separate verification harness.

## Verification harness

OrderedThreadPoolExecutorVerification runs deterministic checks without adding a testing framework dependency.

It verifies:

- reverse worker completion with ordered output,
- failure without a permanent sequence gap,
- concurrent producer threads,
- per-producer submission order,
- bounded queue backpressure,
- rejection after shutdown,
- in-flight completion during shutdown,
- timeout-based termination,
- null-result failure,
- quiescent statistics invariants.

The concurrent producer test intentionally does not require one fixed global order between producer threads. Their global interleaving depends on which thread acquires the sequence-assignment monitor first.

What must remain true is that submissions made sequentially by the same producer cannot be reversed in the final stream.

## Benchmark harness

OrderedThreadPoolExecutorBenchmark measures two things.

First, it isolates reorder-buffer cost by comparing:

~~~java
TreeMap<Long, Integer>
~~~

with:

~~~java
HashMap<Long, Integer>
~~~

for reverse and deterministically shuffled completion sequences.

Second, it runs an end-to-end executor workload using a fixed worker pool, consumes every result, and verifies exact publication order while measuring the whole submission/publication path.

The harness performs warmup passes and reports medians.

For serious performance claims, JMH and allocation profiling would still be required.

## Java 27 configuration

The module now has the same modernized build shape as ObservableConcurrentTree.

It targets Java 27 and enables:

~~~text
-Xlint:all
-Werror
~~~

The module defines:

~~~text
:orderedthreadpoolexecutor:runExperiment
:orderedthreadpoolexecutor:verifyExperiment
:orderedthreadpoolexecutor:benchmarkExperiment
~~~

The normal check task depends on verifyExperiment.

## Complexity and remaining limits

Let b be the number of out-of-order terminal completions currently buffered.

Sequence assignment is O(1). HashMap insertion and exact next-sequence lookup are expected O(1). Across n terminal outcomes, reorder bookkeeping is expected O(n), excluding worker computation and BlockingQueue behavior.

Memory use is O(b). A slow early task can therefore cause many later completions to accumulate.

The refactored component is safer, but it is still an experiment rather than a complete ExecutorService replacement.

Important limitations remain:

- cancelling CompletableFuture does not cancel worker execution,
- the reorder buffer is unbounded,
- a permanently stuck worker can still block every later result,
- graceful close can block on downstream queue backpressure,
- the output queue carries only successful values while failures use futures,
- statistics are observational,
- worker executor lifecycle remains caller-owned.

The implementation catches task Throwable values so an accepted sequence always gets a terminal outcome. Applications that need special fatal-Error propagation semantics should define that policy explicitly instead of assuming this experiment's future-oriented behavior is appropriate.

Useful next experiments include a bounded reorder buffer, cancellation as another sealed terminal state, configurable publication policy, an ordered success/failure result envelope, and JMH measurement of the virtual publisher and reorder-buffer choices.

The invariant that should survive all of those changes is simple: once a sequence number is accepted, it must not silently disappear.
