# Preserving Result Order from Parallel Work

## Why this experiment exists

This experiment implements a small ordered executor facade. Tasks are submitted
with an implicit sequence number, run in parallel on a supplied executor, and
publish their results to a blocking queue only when all earlier results have
also arrived. The goal is not to maximize throughput. The goal is to preserve
input order while still allowing the expensive part of each task to execute
concurrently.

That pattern appears in log processing, media pipelines, chunked file work, and
network fan-out where downstream consumers expect ordered output. A naive
parallel implementation often emits results as soon as they complete, which
turns scheduling noise into observable order changes. This experiment separates
execution order from publication order with a `TreeMap` and a single finalizer
executor.

For experienced Java developers, the useful part is the coordination boundary:
worker threads compute results, but only one thread mutates the result buffer
and advances the next expected id. That keeps the ordering logic simple without
locking around every worker completion.

## Execution path

The `OrderedThreadPool<T>` constructor accepts a `BlockingQueue<T>` for ordered
results and an `ExecutorService` for parallel task execution. The caller owns
both pieces. The class does not create the worker pool and does not decide
where ordered results go. This keeps the experiment focused on the ordering
algorithm rather than lifecycle management.

```java
// The caller supplies both the output queue and the worker executor.
public OrderedThreadPool(BlockingQueue<T> queue, ExecutorService executor) {
    _queue = queue;
    _executor = executor;
}
```

That dependency shape makes the class easy to embed in different queue and
executor experiments.

The demo `main` builds an integer pipeline:

```java
BlockingQueue<Integer> integers = new ArrayBlockingQueue<>(1000);
OrderedThreadPool<Integer> objectOrderedThreadPool = new OrderedThreadPool<>(integers, Executors.newCachedThreadPool());
for (int i = 0; i < tasks; i++) {
    objectOrderedThreadPool.process(new TemporaryTask(i));
}
```

Each `TemporaryTask` sleeps for a random short interval and then returns its
input number. Without ordering, those integers would complete in nondeterminic
order. The queue should still produce `0, 1, 2, ...` to the consumer.

The experiment uses `ArrayBlockingQueue` rather than an unbounded queue in the
demo. That matters because publication can back up if the consumer does not
drain results. The ordered publisher is therefore not just coordinating result
order; it is also subject to downstream backpressure through `BlockingQueue`.

## Core code walkthrough

Task submission assigns a monotonic id before handing work to the executor:

```java
public void process(Supplier<T> task) {
    int id = _counter.getAndIncrement();
    _executor.execute(() -> {
        T result = task.get();
        _finalizer.execute(new Runnable() {
```

The id is captured by the worker closure. The task itself is just a
`Supplier<T>`, so the experiment does not expose futures, cancellation handles,
or checked exceptions. Once `task.get()` returns, the worker schedules a small
publication step on `_finalizer`.

The finalizer owns the ordered buffer:

```java
_results.put(id, result);
Integer next = _results.firstKey();
while (next == _nextId) {
    _nextId++;
    _queue.put(_results.remove(next));
    next = _results.isEmpty() ? -1 : _results.firstKey();
}
```

`_results` can receive completions out of order because it is a `TreeMap`.
Only the contiguous prefix starting at `_nextId` is published to the queue.
If task 7 finishes before task 6, result 7 waits in the map. As soon as result
6 arrives, the loop can publish 6 and then continue publishing 7 and any later
contiguous results already buffered.

The temporary task intentionally introduces completion jitter by sleeping for
`ThreadLocalRandom.current().nextInt(100)` milliseconds before returning its
integer. That makes the order-preservation behavior observable even on a fast
local machine.

```java
// Random sleep makes completion order differ from submission order.
Thread.sleep(ThreadLocalRandom.current().nextInt(100));
return _i;
```

Without that jitter, a local run might accidentally look ordered even if the
publication algorithm were wrong.

## Important implementation details

The key field layout is small:

```java
private final AtomicInteger _counter = new AtomicInteger();
private final BlockingQueue<T> _queue;
private final ExecutorService _executor;
private final SortedMap<Integer, T> _results = new TreeMap<>();
private int _nextId = 0;
private final Executor _finalizer = Executors.newSingleThreadExecutor();
```

`AtomicInteger` protects id assignment across calling threads. `_results` and
`_nextId` are not thread-safe by themselves, but they are only accessed from the
single finalizer executor. This is a common and useful trick: instead of
guarding state with locks, serialize all mutations through one executor.

The `TreeMap` choice makes the smallest completed id cheap to find with
`firstKey()`. A hash map plus repeated lookup of `_nextId` would also work for
this exact algorithm, but the sorted map makes the buffer's ordering role
obvious when reading the code. It also generalizes to diagnostics such as
reporting the lowest and highest buffered ids.

The validation loop consumes exactly the expected number of results:

```java
for (int i = 0; i < tasks; i++) {
    if (!Objects.equals(i, integers.take())) {
        System.exit(1);
    }
}
```

The process exits with status `1` on the first ordering failure. It is a blunt
test harness, but it proves the invariant the experiment cares about.

## Runtime behavior and caveats

Running the module with no arguments submits 1000 tasks. Passing one argument
changes the task count. The program does not print progress on success; it only
exits. A mismatch terminates the JVM with `System.exit(1)`.

There are several lifecycle caveats. The worker executor is supplied by the
caller but not shut down by the demo. The finalizer executor is created inside
`OrderedThreadPool` and also is not shut down. That is acceptable for a sketch,
but a reusable component would implement `AutoCloseable` or expose an explicit
shutdown method.

Failure handling is also intentionally thin. If a supplier throws, no result is
published for that id, and later results wait forever behind the gap. If
`_queue.put` blocks because the consumer is slow, the single finalizer thread
blocks too, which stops publication of all later results. The design therefore
assumes that consumers keep draining the queue and tasks normally return
values.

There is a subtle interruption issue as well. `TemporaryTask` converts
interruption into `RuntimeException`, and the finalizer converts interruption
from `_queue.put` into `RuntimeException`. Neither path restores the interrupt
status. That is common in quick demos, but production executor code should make
interruption policy explicit.

## Suggested next experiments

Return a handle from `process` that can represent success, failure, or
cancellation for each id. Add a shutdown method for both worker and finalizer
executors. Replace the external result queue with a callback and compare
backpressure behavior. Add tests that submit from multiple producer threads.
Finally, introduce a bounded reorder buffer and decide what should happen when
a missing early result blocks too many later completions.

A second useful direction is observability. Expose the number of buffered
out-of-order results, the next expected id, and the highest assigned id. Those
metrics would make head-of-line blocking visible without changing the core
ordering algorithm. They would also help distinguish slow workers from a slow
consumer, which currently look similar from outside the class.
Logging those counters under a stress test would make the queueing behavior
much easier to explain during review and failure analysis.

Another extension would let callers choose between blocking publication and
dropping or failing when the output queue is full. That decision is part of the
ordering contract.
