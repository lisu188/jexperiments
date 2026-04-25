# Coordinating Subscribers with Java Flow

## Why this experiment exists

This experiment explores the `java.util.concurrent.Flow` API through the JDK's
`SubmissionPublisher`. The code is small, but it exercises the essential
reactive-streams protocol: a publisher has data, subscribers receive signals,
and each subscription controls demand. That last part is the point. Data does
not move merely because it exists. It moves when a subscriber asks for it.

The module publishes six in-memory load readings to two subscribers. One
subscriber requests three readings at a time and processes them immediately.
The other requests two readings at a time and sleeps briefly for every item.
With a small publisher buffer, this makes demand windows and lag visible without
introducing networking, external queues, or a third-party reactive library.

For experienced Java developers, the useful lesson is where the JDK Flow API is
low-level. It gives you the interfaces and a basic publisher implementation,
but it does not give the operators, schedulers, diagnostics, or test harnesses
that larger reactive libraries provide. This experiment keeps the protocol in
plain sight.

## Execution path

The entrypoint is `com.lis.flow.FlowPublisherExperiment`. It creates a fixed
list of six `MetricReading` values named `load`, with values from `0.38` to
`0.52`. There is no live sensor or random input. A fixed sequence makes the
behavior repeatable enough to reason about, while still giving each subscriber
enough items to cross at least one request-window boundary.

The publisher is configured with a small executor and buffer:

```java
ExecutorService executor = Executors.newFixedThreadPool(2);
CountDownLatch completed = new CountDownLatch(2);

try (SubmissionPublisher<MetricReading> publisher = new SubmissionPublisher<>(executor, 2)) {
    publisher.subscribe(new WindowedAverageSubscriber("fast", completed, 3, 0));
    publisher.subscribe(new WindowedAverageSubscriber("slow", completed, 2, 75));
```

The executor provides asynchronous delivery. The buffer size of two keeps the
demo sensitive to subscriber lag. The latch is not part of Flow; it is only
there to keep `main` alive until both subscribers observe completion.

## Core code walkthrough

Publishing is a simple loop:

```java
for (MetricReading reading : readings) {
    int estimatedLag = publisher.submit(reading);
    System.out.println("published " + reading + " with max lag " + estimatedLag);
}
```

`submit` enqueues the item for subscribers and returns an estimated maximum lag.
That value is not a precise queue size contract, but it is useful for a demo:
the slow subscriber and small buffer can make the publisher report visible lag.

The try-with-resources block is also part of the protocol story. Exiting the
block calls `SubmissionPublisher.close()`, which stops new submissions and
eventually delivers `onComplete` to current subscribers after submitted items
have been handled. The latch that follows is therefore waiting for terminal
signals, not for the `submit` loop itself.

The subscriber begins by storing the subscription and requesting its first
window:

```java
public void onSubscribe(Flow.Subscription subscription) {
    this.subscription = subscription;
    System.out.println(name + " subscribed and requested " + requestWindow);
    subscription.request(requestWindow);
}
```

This is the handshake that makes Flow different from a plain listener list. The
publisher calls `onSubscribe`, but the subscriber decides the initial demand.
If it never calls `request`, it should not receive normal `onNext` items.

The windowing logic lives in `onNext`:

```java
received++;
receivedInWindow++;
total += item.value;

if (receivedInWindow == requestWindow) {
    receivedInWindow = 0;
    System.out.println(name + " requested " + requestWindow + " more");
    subscription.request(requestWindow);
}
```

Each subscriber counts items in the current window. Once the window is
exhausted, it requests another window of the same size. This is a manual,
visible form of demand management.

Because each subscriber has its own `Flow.Subscription`, demand is independent.
The fast subscriber can request three more while the slow subscriber is still
processing its first window. The publisher has to respect both subscribers, so
the slow subscriber can still affect buffering and lag even though it does not
directly block the fast subscriber's callback code.

## Important implementation details

The `WindowedAverageSubscriber` also simulates slow processing:

```java
if (processingDelayMillis > 0) {
    try {
        Thread.sleep(processingDelayMillis);
    } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        subscription.cancel();
        onError(e);
        return;
    }
}
```

The sleep is not a recommendation for production reactive code. It is a
controlled way to make one subscriber slower than the other. The interruption
path cancels the subscription and reports the error, which is important because
Flow subscribers should not keep accepting data after their processing thread
has been interrupted.

Completion is handled by the normal terminal signal. `onComplete` prints how
many readings the subscriber received and calls `completed.countDown()`.
Closing the `SubmissionPublisher` after the try-with-resources block causes
completion to be sent asynchronously, and the latch bridges that signal back to
`main`.

```java
// Terminal signal: no more onNext calls should follow this callback.
public void onComplete() {
    System.out.println(name + " completed after " + received + " readings");
    completed.countDown();
}
```

That callback is the only place the latch is decremented for the success path,
so the main thread waits for both subscribers to reach the terminal state.

## Runtime behavior and caveats

Running the module publishes six readings, prints subscriber averages, and exits
after both subscribers complete. It is safe to run because all input is
in-memory and the executor is shut down in a `finally` block. If completion does
not arrive within five seconds, the program throws an `IllegalStateException`.

```java
if (!completed.await(5, TimeUnit.SECONDS)) {
    throw new IllegalStateException("Subscribers did not complete in time");
}
```

The timeout is a small but important guard: a broken demand cycle fails visibly
instead of leaving the process waiting forever.

The caveat is that `SubmissionPublisher` is a demonstration-grade primitive,
not a full reactive application framework. It does not provide transformations
like `map` or `buffer`, it does not model domain-specific retry behavior, and
its lag estimate is an implementation signal rather than business logic. The
program also performs console output from asynchronous callbacks, so line order
can vary even when demand rules are followed.

Another important boundary: Flow backpressure is a protocol between publisher
and subscriber. It does not automatically protect external systems. If a real
publisher is reading from a socket, database, or message broker, the code that
bridges that source into Flow must still decide what to do when demand is low.

The subscriber implementation also assumes callbacks are serialized per
subscription, as required by the Flow protocol. That lets it update `received`,
`receivedInWindow`, and `total` without additional synchronization. If similar
state were updated from arbitrary executor tasks outside the subscription
callback contract, the code would need locking or atomic state.

The demo also avoids shared mutable state between subscribers. That keeps the
fast and slow observers independent, which is useful when interpreting the
interleaved console output.

## Suggested next experiments

Add a subscriber that cancels after three readings and confirm that it receives
no terminal completion signal after cancellation. Replace `submit` with
`offer` and implement a dropped-item handler. Add a transformation stage by
subscribing a custom processor between publisher and final subscribers. Finally,
record timestamps in `MetricReading` to distinguish publisher lag from
subscriber processing latency.

Another useful variation would make the publisher finite but slower, for
example by submitting readings from a scheduled executor. That would separate
producer pacing from subscriber demand and make it easier to observe how the
same subscribers behave when items arrive gradually instead of being submitted
in one tight loop. That variant would also make completion timing easier to
trace in logs.
