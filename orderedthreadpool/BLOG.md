# Preserving Result Order from Parallel Work

## Why this experiment exists

This experiment solves a common concurrency problem: tasks may finish out of
order, but consumers sometimes need results in submission order. The
`OrderedThreadPool` accepts suppliers, runs them through an executor, and emits
completed values to a blocking queue only when all earlier values have also
arrived.

The module is small, but the idea is practical. Parallelism and ordering often
pull in opposite directions. This code demonstrates one way to let work happen
concurrently while keeping the output stream deterministic.

## How it works

Each submitted task receives a monotonically increasing id from an
`AtomicInteger`. The supplier runs on the provided executor. When it finishes,
the result is passed to a single-thread finalizer executor.

The finalizer owns the ordering state. It stores completed results in a
`TreeMap<Integer, T>` keyed by task id. It then checks the smallest available
key. While that key equals `_nextId`, the finalizer removes the result, puts it
onto the output queue, and increments `_nextId`.

The `main` method acts like a smoke test. It submits a configurable number of
temporary tasks, each sleeping for a random short duration and returning its
original id. Then it consumes the same number of queue entries and exits with
status `1` if any result arrives out of order.

## What to notice

The useful design choice is the single-thread finalizer. The worker executor can
run many suppliers at the same time, but all updates to `_results` and `_nextId`
happen through one executor. That avoids explicit locking around the ordered
buffer while still keeping worker execution parallel.

The tradeoff is head-of-line blocking. If task `7` finishes slowly, results for
`8`, `9`, and later tasks can accumulate in the `TreeMap` but cannot be emitted.
That is not a bug; it is the cost of preserving submission order.

## Sanity check

The module has no external dependencies and compiles through the shared Gradle
Java configuration. There are no formal unit tests, but `main` is a simple
runtime check for ordering.

The static pass found that the executors are not shut down after `main` finishes
consuming results. In a larger application, lifecycle management would need to
be explicit. The output queue can also apply backpressure: if consumers stop
reading, the finalizer can block on `_queue.put(...)`.

The repository-level Gradle wrapper build compiles this module successfully.

## Suggested next experiments

A useful next step would be adding a `close` or `shutdown` method and making the
class implement `AutoCloseable`. Another extension would report task failures
through the ordered stream, because preserving order becomes more nuanced when
some tasks throw exceptions.
