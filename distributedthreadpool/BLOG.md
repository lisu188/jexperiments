# Distributed Thread Pool: Symmetric RPC over Java Object Streams

## Why this experiment exists

This experiment explores the mechanics behind remote execution without hiding them behind HTTP, gRPC, JSON, or a framework runtime.

Two JVM peers are connected by a TCP socket. A caller serializes a Java lambda implementing a serializable functional interface, the remote peer executes it, and a correlated response travels back over the same connection.

The original version from 2015 intentionally collapsed transport, dispatch, result correlation, callbacks, and lifecycle into a handful of classes. That made the idea visible, but it also exposed several real distributed-systems problems:

- responses were implemented as nested remote callbacks,
- client registration was incomplete,
- blocking requests could wait forever,
- Java object streams were not explicitly flushed,
- one executor serialized both inbound execution and outbound writes,
- exceptions were not returned as protocol results,
- sockets and executors leaked,
- there was no backpressure or transport observability,
- ObjectOutputStream could retain cross-message object references indefinitely,
- user CompletableFuture callbacks could block the socket reader,
- the server had no deterministic test or performance harness.

The modern implementation keeps the educational premise — serializable Java code crossing a socket — while separating the transport responsibilities explicitly.

It is still a trusted-peer experiment. Native Java deserialization of arbitrary network input is unsafe for untrusted networks.

## Protocol model

The transport now sends a small sealed protocol instead of treating every serialized consumer as an implicit protocol message.

The wire model contains four message variants:

~~~java
sealed interface WireMessage extends Serializable
        permits Invocation, Command, Response, Registration
~~~

A normal remote function call is an Invocation:

~~~java
record Invocation(
        long requestId,
        SerializableSupplier<?> task)
        implements WireMessage
~~~

A context-aware one-way operation is represented as a correlated Command:

~~~java
record Command(
        long requestId,
        TaskMessage<?> task)
        implements WireMessage
~~~

Both Invocation and Command receive a Response.

~~~java
record Response(
        long requestId,
        Object value,
        RemoteFailure failure)
        implements WireMessage
~~~

Registration is explicit:

~~~java
record Registration(int clientId)
        implements WireMessage
~~~

This removes the old requirement that a server task send another serialized callback to the client just to complete a synchronous request.

A request is now one outbound message plus one response message.

## Correlation without CountDownLatch maps

The old DataRepository allocated one CountDownLatch and one separate value map entry for each blocking request.

The new repository stores CompletableFuture values directly:

~~~java
private final AtomicLong nextId = new AtomicLong();

private final ConcurrentHashMap<
        Long,
        CompletableFuture<Object>> pending =
        new ConcurrentHashMap<>();
~~~

Registering a request creates one future and one correlation id.

A response removes the future from the table and completes it.

Failures complete the same future exceptionally.

Connection shutdown fails every still-pending request.

The repository therefore has a single lifecycle for each request instead of separate latch and value structures.

## Public async API

Client calls are asynchronous by default:

~~~java
public <T> CompletableFuture<T> callOnServer(
        SerializableSupplier<T> target)
~~~

A timeout-oriented blocking convenience overload is also available:

~~~java
public <T> T callOnServer(
        SerializableSupplier<T> target,
        Duration timeout)
        throws Exception
~~~

The server exposes the symmetric API for invoking work on an already connected client:

~~~java
public <T> CompletableFuture<T> callOnClient(
        int clientId,
        SerializableSupplier<T> target)
~~~

This symmetry is important.

The client and server now use the same transport semantics in both directions.

## Explicit remote failures

Arbitrary Throwable graphs are not serialized back to the caller.

Instead the remote peer converts failures to a stable record containing:

- remote exception class name,
- message,
- rendered remote stack trace.

~~~java
record RemoteFailure(
        String type,
        String message,
        String stackTrace)
        implements Serializable
~~~

The caller receives RemoteExecutionException.

That keeps the wire error model independent from the serializability of a particular Throwable subclass.

A result that is not Serializable is also converted into a remote failure before the transport attempts to write it.

## Reader and writer separation

The original SocketAccesor used one single-thread executor for both received work and outbound writes.

That created unnecessary coupling.

Slow execution could delay writes, and slow writes could delay local work.

The new SocketAccessor has separate responsibilities:

~~~text
socket input
    ↓
reader thread
    ↓
protocol dispatch
    ↓
invocation executor

producers
    ↓
MPSC write queue
    ↓
writer thread
    ↓
socket output
~~~

The reader blocks in ObjectInputStream.readObject().

The writer owns ObjectOutputStream exclusively.

Remote task execution runs on a separate executor supplied by ThreadPoolClient or ThreadPoolServer.

By default those invocation executors use virtual threads.

## Batched MPSC writer

Outbound messages are placed into a LinkedTransferQueue.

~~~java
writes.offer(outbound);
~~~

Only the dedicated writer thread touches ObjectOutputStream.

It takes one message, polls up to writerBatchSize additional messages, serializes the batch, resets the object stream handle table, then flushes once.

~~~java
for (int i = 0; i < count; i++) {
    out.writeObject(batch[i].message());
}

out.reset();
out.flush();
~~~

This has several consequences.

Writes cannot interleave and corrupt ObjectOutputStream.

Multiple concurrent producers do not synchronize on the stream.

Flush overhead can be amortized across bursts.

ObjectOutputStream does not retain an ever-growing identity table across unrelated RPC messages.

The writer reuses one array for batch collection instead of allocating a new collection for every flush.

The default batch size is 64 and is configurable.

## Buffered object streams

Both directions use buffered socket streams.

~~~java
new BufferedOutputStream(
        socket.getOutputStream(),
        options.ioBufferBytes())
~~~

and:

~~~java
new BufferedInputStream(
        socket.getInputStream(),
        options.ioBufferBytes())
~~~

The default is 64 KiB.

ObjectOutputStream is created first and its stream header is flushed before ObjectInputStream is created.

Both peers follow the same construction order, avoiding the classic object-stream header deadlock.

## TCP_NODELAY

Low-latency RPC behavior and batched throughput prefer different network policies.

Transport options therefore expose TCP_NODELAY.

~~~java
socket.setTcpNoDelay(options.tcpNoDelay());
~~~

It defaults to true because this experiment primarily measures request/response latency.

The performance matrix includes a TCP_NODELAY-disabled scenario so the effect can be measured rather than assumed.

## Virtual versus platform transport threads

Reader and writer thread models are independently configurable.

~~~java
Options.defaults()
        .withVirtualReader(true)
        .withVirtualWriter(true)
~~~

Blocking socket I/O is a natural virtual-thread workload.

The option remains explicit because one reader and one writer per connection are not a high-cardinality workload by themselves.

The performance harness can therefore compare thread models rather than treating virtual threads as automatically faster.

## Callback isolation

Completing a CompletableFuture may run non-async dependent actions inline on the completing thread.

Completing remote futures directly from the socket reader would therefore allow arbitrary caller code to stall all further inbound network processing.

Responses are instead handed to a separate virtual-thread-per-task response executor.

~~~java
responseExecutor.execute(() -> {
    if (response.failure() == null) {
        repository.complete(
                response.requestId(),
                response.value());
    } else {
        repository.fail(
                response.requestId(),
                response.failure().toException());
    }
});
~~~

A slow user continuation can occupy its own virtual thread while the reader continues decoding later responses.

The verification harness contains a blocked-callback test specifically for this property.

## Explicit client registration

Server connections now receive monotonically assigned client ids.

The server registers the connection before its transport begins serving calls, then sends:

~~~java
new Registration(clientId)
~~~

The client exposes:

~~~java
CompletableFuture<Integer> clientId()
~~~

and:

~~~java
int awaitClientId(Duration timeout)
~~~

Server-to-client calls therefore have a complete routing path instead of relying on the commented-out registration code in the old experiment.

## Connection lifecycle

SocketAccessor implements AutoCloseable.

Closing a connection:

1. stops accepting new outbound messages,
2. fails pending correlated requests,
3. wakes the writer,
4. allows already queued writes to drain when possible,
5. closes the socket,
6. unblocks the reader,
7. waits for reader and writer termination,
8. closes the response completion executor.

Unexpected EOF, socket errors, deserialization failures, and writer errors fail the connection and propagate failure to pending request futures.

ThreadPoolClient and ThreadPoolServer also implement AutoCloseable.

The server closes all registered client connections and its accept socket.

Default internally-created invocation executors are owned and closed by their client or server.

Executors supplied by a caller remain caller-owned.

## Remote context commands

Some experiments need access to the remote ThreadPoolServer or ThreadPoolClient object rather than a context-free supplier.

TaskMessage remains available for that purpose.

~~~java
client.executeOnServer(server -> {
    if (server.clientCount() == 0) {
        throw new IllegalStateException();
    }
});
~~~

The command is correlated like a normal RPC call, so a remote exception completes the returned future exceptionally.

This is safer to reason about than the original raw SerializableConsumer transport because the protocol can distinguish a command from a response or registration frame.

## Transport statistics

Each connection exposes a Statistics record:

~~~java
public record Statistics(
        long sent,
        long received,
        long writeFailures,
        long pendingWrites,
        long writerBatches,
        int pendingRequests)
~~~

These counters make batching, backlog, leaks, and transport failure visible during stress runs.

The verification and soak harnesses assert that pendingRequests returns to zero.

## JFR observability

DistributedThreadPool defines disabled-by-default JFR events for:

- outbound protocol messages,
- inbound protocol messages,
- remote execution duration and failure state,
- writer batch size.

The profiling task is:

~~~text
./gradlew :distributedthreadpool:profileDistributedPool
~~~

The recording is written under build/jfr/.

This makes it possible to correlate RPC latency with GC, socket stalls, virtual-thread behavior, task execution, and writer batching without permanently enabling verbose logging.

## Deterministic verification

DistributedThreadPoolVerification runs everything on loopback sockets and covers:

- client registration,
- basic client-to-server request/response,
- hundreds of concurrent pipelined requests,
- remote exception propagation,
- rejection of non-serializable results,
- server-to-client calls,
- context-aware commands in both directions,
- slow CompletableFuture callback isolation,
- multiple-client routing,
- pending-future failure when a client closes.

No fixed external port is required.

The server binds to port zero and exposes the selected local port.

## Lightweight benchmark

DistributedThreadPoolBenchmark reports two basic metrics.

Sequential RTT measures one complete request/response at a time.

Pipelined throughput submits many requests before joining them.

The benchmark validates response values while measuring the transport.

It is useful as a fast regression harness but not a substitute for JMH.

## Performance matrix

DistributedThreadPoolPerformanceMatrix measures:

- fixed versus virtual invocation executors,
- writer batch sizes 1, 16, and 64,
- request concurrency 1, 8, and 64,
- TCP_NODELAY on versus off,
- throughput,
- p50 latency,
- p99 latency,
- p99.9 latency,
- observed writer batch count.

The matrix uses the real loopback socket protocol.

It therefore includes serialization, queueing, context switching, TCP, remote execution, response routing, and future completion.

## JMH

DistributedThreadPoolJmhBenchmark creates a real loopback server/client pair and measures complete round trips.

Parameters include:

~~~text
batchSize = 1, 16, 64
tcpNoDelay = true, false
workerModel = fixed, virtual
~~~

The benchmark uses SampleTime because tail latency is more important for RPC than one average throughput number.

For contention testing, JMH thread count can be increased externally.

## jcstress

The module includes jcstress races for the correlation repository.

RepositoryCompleteCloseStress races request completion against repository shutdown and verifies that the request ends either successfully or exceptionally, never leaked.

RepositoryRequestIdStress races two request registrations and verifies that ids remain unique.

These tests target the shared-memory coordination layer separately from loopback network tests.

## Soak test

DistributedThreadPoolSoak runs multiple virtual-thread producers against one loopback connection for a configurable duration.

Requests include deterministic synthetic failures.

At the end it checks:

- no pending correlated requests remain,
- no transport write failures occurred,
- all producer tasks completed.

This is aimed at lifecycle leaks and long-run queue growth rather than microbenchmark timing.

## Java 27 tasks

The module targets Java 27 with:

~~~text
-Xlint:all
-Werror
~~~

Available tasks include:

~~~text
:distributedthreadpool:runExperiment
:distributedthreadpool:verifyExperiment
:distributedthreadpool:benchmarkExperiment
:distributedthreadpool:performanceMatrix
:distributedthreadpool:soakExperiment
:distributedthreadpool:jmh
:distributedthreadpool:jmhSmoke
:distributedthreadpool:jcstress
:distributedthreadpool:profileDistributedPool
~~~

The module-specific GitHub Actions workflow runs verification, JMH smoke, jcstress sanity mode, and a small performance matrix.

## Security boundary

This experiment still deserializes Java objects from a network peer and then executes received serializable code.

That is dangerous.

Do not expose this transport to untrusted clients or an untrusted network.

Transport Options accept an ObjectInputFilter:

~~~java
Options.defaults()
        .withInputFilter(filter)
~~~

No restrictive default filter is installed because generic serialized lambdas and arbitrary serializable result types make a universal allow-list impossible.

Production systems should use an explicit schema, authenticated peers, authorization, bounded message sizes, and a serialization format that does not instantiate arbitrary Java object graphs.

The absence of those controls is not a minor deployment detail.

It is the main reason this remains an experiment rather than an RPC library.

## Remaining limitations

Strict request deadlines are not propagated to the remote peer.

Cancelling a CompletableFuture does not cancel already-running remote code.

There is no transport-level flow-control limit on pending RPCs.

One connection still has one serialized ObjectOutputStream writer, so all messages share a single TCP ordering domain.

A permanently blocked remote task can still consume remote executor resources indefinitely.

Java serialized lambdas require compatible capturing classes and implementation methods on both peers.

Cross-version protocol compatibility remains tied to Java serialization compatibility.

Those are useful directions for further experiments, but the current design now makes them explicit instead of hiding them inside callback chains.
