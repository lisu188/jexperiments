# Coordinating Subscribers with Java Flow

## Why this experiment exists

This experiment explores the `java.util.concurrent.Flow` API that arrived with
Java 9. The API is small, but it introduces the core pieces of reactive-streams
style programming: a publisher emits items, subscribers receive them, and a
subscription controls demand.

The module uses `SubmissionPublisher`, the JDK's built-in publisher
implementation. It publishes a short sequence of load readings to two
subscribers. One subscriber asks for three items at a time and processes them
immediately. The other asks for two items at a time and sleeps briefly on each
item, making the publisher's lag estimate visible.

## How it works

The entrypoint is `com.lis.flow.FlowPublisherExperiment`. It creates a fixed
two-thread executor and a `SubmissionPublisher<MetricReading>` with a small
buffer size of two. The small buffer keeps the demo compact while still making
subscriber demand observable.

Each subscriber implements `Flow.Subscriber<MetricReading>`. In
`onSubscribe`, it saves the subscription and calls `request(...)` for its first
window of items. In `onNext`, it records the reading, prints a running average,
and requests another window whenever the current one is exhausted.

The publisher is used in a try-with-resources block. Closing it signals
completion to subscribers, and a `CountDownLatch` keeps `main` alive until both
subscribers receive `onComplete`.

## What to notice

The important idea is that subscriber demand is explicit. The publisher can
have data ready, but a subscriber controls how many items it is prepared to
receive. That makes `request(...)` part of the protocol, not just an
implementation detail.

The experiment also shows that `SubmissionPublisher.submit(...)` returns an
estimated maximum lag across subscribers. With one slower subscriber and a small
buffer, that value helps show how a slow consumer affects the overall stream.

## Sanity check

The module has no external dependencies and compiles through the shared Gradle
Java configuration. There are no formal tests. Running the program is safe: it
publishes a fixed list of six in-memory readings, waits for both subscribers to
complete, and shuts down its executor.

The repository-level Gradle wrapper build compiles this module successfully.

## Suggested next experiments

A useful follow-up would be adding a subscriber that cancels after a fixed
number of readings. Another extension would compare `submit(...)` with
`offer(...)`, where dropped-item handling can be made explicit.
