# Chaining CompletableFuture Work on an Explicit Executor

## Why this experiment exists

This experiment studies a narrow but important corner of `CompletableFuture`:
what actually happens when several asynchronous stages are chained on a
specific executor. The code is intentionally small. It starts with a completed
integer value, applies the same function five times with `thenApplyAsync`, and
prints both the current thread name and the value seen by each stage.

For experienced Java developers, the interesting part is the distinction
between asynchronous execution and parallel execution. A chain of dependent
future stages does not become five independent tasks just because the executor
has ten worker threads. Each stage depends on the previous stage's result, so
the pipeline is ordered by data dependency. The executor decides where each
stage runs, but the `CompletableFuture` graph decides when a stage is eligible
to run.

The module is a useful reminder that `CompletableFuture` is not only a callback
API. It is also a graph of completion dependencies. Choosing `thenApply`,
`thenApplyAsync`, or `thenApplyAsync(..., executor)` changes the scheduling
contract without changing the logical shape of the computation.

## Execution path

The entrypoint is the default-package `Main` class. It creates one shared
executor and then builds a future chain from an already completed root:

```java
private static final ExecutorService EXECUTOR = Executors.newFixedThreadPool(10);
```

That line is deliberately explicit. The code does not use the common
fork-join pool, so all asynchronous stages are scheduled onto this fixed pool.
The pool size is larger than the number of stages, making it easy to see that
available threads alone do not make dependent stages run concurrently.

The function used by every stage has a visible side effect:

```java
Function<Integer, Integer> fn = (x) -> {
    System.out.println(Thread.currentThread().getName());
    System.out.println(x);
    return x + 1;
};
```

Each invocation prints the worker thread and the input integer, then returns
the incremented value. Since the next stage receives the prior result, the
values should appear as `0`, `1`, `2`, `3`, and `4` if the chain completes
normally.

## Core code walkthrough

The root future is already complete:

```java
CompletableFuture<Integer> f = CompletableFuture.completedFuture(0);
```

That means the first asynchronous stage can be submitted immediately. There is
no external producer, no blocking I/O, and no race to complete the root future.
The rest of the program is only about how dependent continuations are scheduled.

The pipeline is built by reassigning `f` to the future returned by each stage:

```java
f = f.thenApplyAsync(fn, EXECUTOR);
f = f.thenApplyAsync(fn, EXECUTOR);
f = f.thenApplyAsync(fn, EXECUTOR);
f = f.thenApplyAsync(fn, EXECUTOR);
f = f.thenApplyAsync(fn, EXECUTOR);
```

This is easy to misread. The code is not attaching five independent consumers
to the same root. Every line replaces `f` with the newly returned future, so
the next line attaches to the previous stage. The result is a linear chain:
stage 2 cannot start until stage 1 completes, stage 3 cannot start until stage
2 completes, and so on.

The final synchronization is also minimal:

```java
f.join();
EXECUTOR.shutdown();
```

`join()` waits for the last future in the chain. Waiting for the last stage is
enough because every earlier stage must already have completed for the last
stage to complete. The executor is then shut down so the fixed-pool threads do
not keep the JVM alive.

Reassignment is a small source-level choice with a large semantic effect. If
the code had kept the original root future and called `root.thenApplyAsync(...)`
five times, it would have created five sibling stages that could all consume
the same value `0`. This experiment instead builds a chain whose intermediate
values are meaningful. That is the difference between fan-out and pipeline
composition, and `CompletableFuture` supports both with nearly identical
syntax.

## Important implementation details

Using `thenApplyAsync(fn, EXECUTOR)` instead of `thenApply(fn)` makes the
scheduling boundary visible. Without `Async`, a continuation may run in the
thread that completes the previous stage. With `Async` and an explicit
executor, the continuation is queued for the executor. That matters when a
continuation is CPU-heavy, blocking, or should be isolated from the thread that
completed the upstream future.

The fixed thread pool does not guarantee a different thread for every stage.
The output may show one worker thread handling several stages, or several
workers participating over the life of the chain. Both outcomes are compatible
with the contract. The dependency chain guarantees value order, not worker
affinity.

```java
// These prints reveal scheduling, not business logic.
System.out.println(Thread.currentThread().getName());
System.out.println(x);
return x + 1;
```

The thread name shows where the continuation ran, while the integer proves that
the logical value still moves through the stages in order.

The root future is already complete before any asynchronous stage is attached.
That removes one variable from the experiment: there is no race between
producer completion and continuation registration. In production systems,
stages are often attached before the upstream work finishes, and cancellation
or timeout policies may be attached in parallel. Here, the only scheduling
question is which executor thread runs each continuation after its dependency
is satisfied.

Another subtle point is exception propagation. The demo function never throws,
but if it did, the current stage would complete exceptionally and downstream
stages created with `thenApplyAsync` would be skipped. A production chain would
usually add `exceptionally`, `handle`, or `whenComplete` to make failure
behavior explicit. This experiment keeps those branches absent so the scheduling
path remains visible.

## Runtime behavior and caveats

Running the module prints five worker-thread names and five integer values,
then exits after `shutdown()`. The exact thread names are runtime-dependent
because they come from `Executors.newFixedThreadPool`. The integer sequence is
the stable observation.

The program is safe to run, but it is not a throughput benchmark. There is no
timing, no warmup, no backpressure, and no comparison against synchronous
composition. The function performs console I/O, which dominates the tiny amount
of arithmetic and can distort any attempt to infer scheduler cost.

`join()` also wraps failures in unchecked `CompletionException`, unlike `get`,
which exposes checked `ExecutionException` and `InterruptedException`. The demo
uses `join()` because it wants a compact main method and no checked exception
plumbing. A service boundary might prefer `get` or explicit completion handlers
so interruption and failure reporting are part of the design.

```java
// Waiting for the tail waits for the whole dependent chain.
f.join();
EXECUTOR.shutdown();
```

That shutdown path is enough here because the last future cannot complete until
every earlier stage has produced its value.

The default-package `Main` class is also a caveat for reuse. It is convenient
for a small experiment, but a real library or application module would use a
named package and a lifecycle strategy for the executor. The static executor is
fine here because the process has one entrypoint and shuts it down explicitly.

## Suggested next experiments

Attach five branches to the same root future and compare that output with the
current linear chain. Replace `thenApplyAsync` with `thenApply` to observe which
thread executes continuations when the root is already complete. Add one stage
that throws and trace how exceptional completion moves through the remaining
chain. Finally, compare a CPU-bound function on a fixed pool with the common
pool to make executor selection visible under load.
