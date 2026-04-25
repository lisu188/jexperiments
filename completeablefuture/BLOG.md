# Chaining CompletableFuture Work on an Explicit Executor

## Why this experiment exists

This module explores a small but important part of `CompletableFuture`: how a
chain of asynchronous transformations moves through an executor. The program
starts with a completed integer value, attaches several `thenApplyAsync` stages,
and prints both the current thread name and the value each stage receives.

It is deliberately minimal. There is no business logic, no fan-out, and no
exception recovery. The point is to make scheduling behavior easy to see. Each
stage increments the value and the final `join()` waits until the asynchronous
chain has finished.

## How it works

The entrypoint is the top-level `Main` class. It creates a fixed thread pool
with ten workers and stores it in a static `ExecutorService`. The transformation
function prints the current thread, prints the incoming integer, and returns the
integer plus one.

The future chain starts with `CompletableFuture.completedFuture(0)`. The code
then reassigns the future five times:

```java
f = f.thenApplyAsync(fn, EXECUTOR);
```

Each new stage depends on the previous stage's result, but the stage itself is
scheduled on the explicit executor. The stages are logically ordered by data
dependency even though the executor has several available threads. The final
`join()` blocks the main thread until the last stage completes, and the executor
is shut down afterward.

## What to notice

The experiment separates dependency ordering from thread selection. A chain of
`thenApplyAsync` calls does not make the transformations independent. Stage two
still needs the value from stage one. The `Async` variant changes where the
continuation is scheduled.

Using an explicit executor is also worth noticing. Without it, `thenApplyAsync`
uses the common fork-join pool. That may be fine for examples, but real
applications often need tighter control over thread counts, queueing, and
shutdown.

The folder name is `completeablefuture`, while the Java API is correctly named
`CompletableFuture`.

## Sanity check

The module has no external dependencies and compiles through the shared Gradle
Java configuration. There are no tests. The program should terminate cleanly
because it calls `EXECUTOR.shutdown()` after `join()`.

The repository-level Gradle wrapper build compiles this module successfully.

## Suggested next experiments

The next useful step would be adding `thenCombine` or `allOf` examples to
contrast dependent and independent asynchronous work. Another extension would be
adding a failing stage and comparing `exceptionally`, `handle`, and
`whenComplete`.
