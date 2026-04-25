# A Distributed Thread Pool over Object Streams

## Why this experiment exists

This experiment asks a provocative question: how little code is needed to make
one JVM ask another JVM to run a piece of Java logic? The answer here is a
small distributed thread-pool sketch built on sockets, `ObjectInputStream`,
`ObjectOutputStream`, serializable functional interfaces, and callbacks that can
cross the client/server boundary.

It is not a production RPC framework. That is precisely why it is interesting.
The module exposes the raw mechanics that frameworks usually hide: both peers
must share compatible classes, lambdas must be serializable, socket streams must
agree on object serialization protocol, responses need correlation ids, and a
remote callback is just another serialized command sent in the opposite
direction. The code is compact enough that those tradeoffs are easy to see.

For experienced Java developers, the main value is architectural. Java
serialization makes remote execution look deceptively local. A
`SerializableSupplier<T>` can be shipped to another process, executed there, and
the result can be returned through a callback. But that convenience comes with
security, compatibility, lifecycle, and failure-mode costs that are visible in
this experiment.

## Execution path

The server entrypoint calls
`new ThreadPoolServer(55555).start().await()`. `start()` submits an accept loop
to a cached executor. For every accepted socket it creates a
`ServerConnectionThread`, which is a specialization of the shared socket
accessor. The server also stores connected clients in a concurrent map using
generated integer ids.

The client entrypoint connects to that port and periodically asks the server to
execute a supplier:

```java
ThreadPoolClient threadPoolClient = new ThreadPoolClient("127.0.0.1",
        55555);
for (int i = 0; i < 100; i++) {
    Thread.sleep(1000);
    threadPoolClient.callOnServer(() -> {
        return Numbers.getId();
    }, System.out::println);
}
```

The supplier runs where the receiving side executes it. The callback prints the
result after the answer comes back. In a single-machine demo this looks simple;
across real machines it assumes the same bytecode is available on both sides.

```java
// Accept clients on a fixed local demo port.
public static void main(String[] args) throws InterruptedException {
    new ThreadPoolServer(55555).start().await();
}
```

That one line hides the accept loop, client registration, and long-running
executor lifecycle that the rest of the module expands.

## Core code walkthrough

The shared transport loop is `SocketAccesor`. It owns the object streams and a
single-thread executor for running received work:

```java
Object readObject = in.readObject();
if (readObject instanceof SerializableConsumer) {
    executor.submit((Callable<Void>) () -> {
        ((SerializableConsumer<T>) readObject).accept(context);
        return null;
    });
}
```

This is the core of the experiment. The received object is not decoded into a
custom protocol message. If it is a `SerializableConsumer`, it is invoked with
the local context. On the server side that context is `ThreadPoolServer`; on the
client side it is `ThreadPoolClient`.

That choice makes the transport extremely flexible. It also makes the transport
almost impossible to reason about statically. A message can call any public
method reachable from the context object, capture values from the sender, and
throw any checked exception declared by the functional interface. In effect,
the protocol is "execute this Java closure", not a fixed set of named commands.

Sending work is symmetric. `postMessage` submits a write task to the same
single-thread executor and calls `out.writeObject(message)`. That serializes
outbound writes and inbound callback execution through one executor. It keeps
the demo small, although it also means slow local work can delay socket writes.

```java
// The "protocol" is literally a serialized Java consumer.
public void postMessage(SerializableConsumer<U> message) throws Exception {
    executor.submit((Callable<Void>) () -> {
        out.writeObject(message);
        return null;
    });
}
```

This is the point where the transport boundary disappears into Java object
serialization.

The server's synchronous request form uses a repository as a correlation table:

```java
int msgId = repository.lock();
callOnClient(clientid, (ThreadPoolClient context) -> {
    T t = target.get();
    context.callOnServer((ThreadPoolServer ctx) -> {
        ctx.repository.setValue(msgId, t);
    });
});
return (T) repository.getValue(msgId);
```

`lock()` creates a latch keyed by id. The remote side runs the supplier and then
sends a callback that stores the value under the same id. `getValue` blocks
until that callback arrives.

## Important implementation details

The functional interfaces are the glue that makes lambdas eligible for object
serialization:

```java
@FunctionalInterface
public interface SerializableSupplier<T> extends Serializable {
    T get() throws Exception;
}
```

The repository uses `CountDownLatch` to turn asynchronous callbacks into
blocking results:

```java
public Object getValue(int msgId) throws Exception {
    locks.get(msgId).await();
    locks.remove(msgId);
    return values.remove(msgId);
}
```

That is clear and effective in a toy setup. It is also a place where production
code would need timeouts, cancellation, error propagation, cleanup after broken
connections, and defensive handling for missing ids.

`FuncUtils.bind` is another important piece. It captures one argument and
returns a serializable function or consumer that accepts the remaining argument.
That is what lets the client embed its local id while still building a command
whose runtime parameter is the remote context.

```java
// Bind one argument locally; receive the remote context later.
public static <T, U, R> SerializableFunction<U, R> bind(
        SerializableBiFunction<T, U, R> func, T arg) {
    return (x) -> {
        return func.apply(arg, x);
    };
}
```

The helper is small, but it is what makes callbacks feel like partially applied
remote commands.

The code uses generic type parameters throughout the public methods, but the
transport erases most of that safety at runtime. Casts such as
`(SerializableConsumer<T>) readObject` and `(T) repository.getValue(msgId)` are
unchecked. That is typical when object serialization is used as a generic
message bus: the compiler can check the local call site, but it cannot prove
that the remote peer sent the expected object type.

## Runtime behavior and caveats

The most important caveat is security. Deserializing arbitrary objects from a
socket and executing them as `SerializableConsumer` is unsafe unless both peers
and the network are fully trusted. Modern Java systems generally avoid native
Java serialization for remote boundaries or use strict serialization filters.

There are also correctness caveats. `ObjectOutputStream.writeObject` is not
followed by an explicit `flush()`, so delivery can depend on stream buffering.
The server assigns ids, but `ServerConnectionThread` contains a commented-out
block that would send the id to the client, so the client's `id` field remains
at its default unless set elsewhere. A blocking `getValue` can wait forever if
the remote side fails. Connections are accepted forever, executors are not shut
down, and exceptions from submitted tasks are not surfaced to the caller.

Despite those limits, the experiment usefully demonstrates a core RPC pattern:
ship a command, execute it against a local context, optionally ship a callback,
and correlate a response. Seeing that pattern without HTTP, JSON, or framework
abstractions makes the hidden costs easier to discuss.

It also demonstrates why production RPC systems usually separate transport,
authorization, serialization, dispatch, and result handling. This module
collapses those concerns into a few classes, which is excellent for learning
and risky for reuse. The missing boundaries are not accidental polish items;
they are the hard parts of distributed execution.

One practical reading strategy is to trace one request in both directions:
client supplier to server context, server callback to client context, then
repository latch release. That path explains nearly every moving part in the
module.

## Suggested next experiments

Add a small message envelope with `id`, `type`, and `payload` fields instead of
sending raw consumers. Add timeouts and exceptional responses to
`DataRepository`. Flush object streams after writes and close sockets on
failure. Replace Java serialization with a narrow command protocol. Finally,
make client registration explicit so the server-assigned id is delivered before
the client sends its first callback-bearing request.
