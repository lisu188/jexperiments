# A Distributed Thread Pool over Object Streams

## Why this experiment exists

This experiment sketches a distributed execution model using Java sockets,
object streams, and serializable functional interfaces. The server accepts
client connections. Both sides can send work to the other side by serializing a
function-like object and executing it against a local context.

It is not a production distributed executor. It is a research note: what happens
if lambdas, serialization, sockets, callbacks, and blocking result retrieval are
combined into a tiny RPC-like thread pool?

## How it works

The server entrypoint is `ThreadPoolServer`. It listens on port `55555`, accepts
connections, wraps each socket in a `ServerConnectionThread`, and stores clients
by generated id. The client entrypoint is `ThreadPoolClient`, which connects to
`127.0.0.1:55555` and periodically asks the server to execute a supplier that
returns a generated number.

`SocketAccesor` is the shared transport primitive. It owns an
`ObjectOutputStream`, an `ObjectInputStream`, and a single-thread executor. Its
read loop waits for serialized objects, checks whether they are
`SerializableConsumer` instances, and invokes them with the local context.

The `func` package contains serializable versions of common Java functional
interfaces plus a small `FuncUtils.bind` helper. The bind helper lets a piece of
code capture an argument, such as a client id, before it is sent to the other
side.

For request-response calls, `DataRepository` allocates an id, stores a
`CountDownLatch`, and lets the caller wait until the remote side posts a value
back under the same id.

## What to notice

The central idea is that "work" is represented as data moving over a socket.
Instead of defining a fixed protocol with command names, the experiment sends
serialized behavior. That keeps the code compact and expressive, but it brings
the usual tradeoffs of Java serialization: tight class coupling, fragile
compatibility, and a large trust boundary.

The code also shows how callbacks are layered on top of the same primitive. A
client can ask the server to compute something and then ask the server context
to call back to the client with the result. The same pattern appears in the
server-to-client direction.

## Sanity check

The module has no external dependencies and compiles through the shared Gradle
Java configuration. There are no tests. Runtime behavior depends on starting the
server before the client.

The static pass found an important limitation: `ServerConnectionThread` contains
a FIXME around assigning the generated server-side client id back to the client.
Because that code is commented out, `ThreadPoolClient.id` remains its default
value unless another path sets it. The transport also uses infinite read loops
and Java object deserialization without authentication or versioning.

The repository-level Gradle wrapper build compiles this module successfully.

## Suggested next experiments

The most useful next step would be making client registration explicit and
tested, so request routing does not depend on a default id. After that, a small
message envelope with a type, request id, payload, and error channel would make
the protocol easier to reason about.
