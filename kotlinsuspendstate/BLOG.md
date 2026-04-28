# Kotlin Suspend Functions as JVM State Machines

A `suspend` function is callable from Kotlin like a regular function, but the JVM method is an `Object`-returning function that accepts a `Continuation`. This experiment builds a suspend point with only `kotlin.coroutines`, drives it from Java, and inspects the generated state-machine class.

## Why This Matters

Java developers who call suspend APIs need to understand the raw `Continuation` ABI. Kotlin developers need to understand why suspend functions have generated classes, labels, and `Object` return values. Library authors need to know that this ABI is real binary surface even when no coroutine library is used.

The demo is safe and local. It captures an in-memory continuation, resumes it once, and prints deterministic output. It does not use external coroutine libraries, threads, timers, network calls, hardware, or file writes.

## Execution Path

The suspend function stores its continuation instead of resuming immediately:

```kotlin
suspend fun delayedDouble(seed: Int): String {
    val resumed = suspendCoroutine<Int> { continuation ->
        println("kotlin-suspend-point seed=$seed")
        savedContinuation = continuation
    }
    return "delayedDouble=${seed + resumed}"
}
```

Java calls the compiler-generated method directly:

```java
return SuspendStateExperimentKt.delayedDouble(seed, new Continuation<String>() {
    @Override
    public CoroutineContext getContext() {
        return EmptyCoroutineContext.INSTANCE;
    }

    @Override
    public void resumeWith(Object result) {
        System.out.println("java-continuation-result=" + result);
    }
});
```

The main function proves that the first Java call returns the suspension marker and the later resume completes the Java continuation:

```text
kotlin-suspend-point seed=10
java-call-return=COROUTINE_SUSPENDED
java-continuation-result=delayedDouble=15
```

## What To Run

```bash
./gradlew --gradle-user-home /tmp/jexperiments-gradle-home --no-daemon :kotlinsuspendstate:runExperiment
./gradlew --gradle-user-home /tmp/jexperiments-gradle-home --no-daemon :kotlinsuspendstate:classes
javap -classpath kotlinsuspendstate/build/classes/kotlin/main:kotlinsuspendstate/build/classes/java/main -v -p -c experiments.kotlin.suspendstate.SuspendStateExperimentKt
javap -classpath kotlinsuspendstate/build/classes/kotlin/main:kotlinsuspendstate/build/classes/java/main -v -p -c 'experiments.kotlin.suspendstate.SuspendStateExperimentKt$delayedDouble$1'
```

## Bytecode Landmarks

The suspend function descriptor is:

```text
(ILkotlin/coroutines/Continuation;)Ljava/lang/Object;
```

`javap` shows a generated continuation class named:

```text
SuspendStateExperimentKt$delayedDouble$1
```

In the function body, inspect the `label` field, the `I$0` field that stores the `seed` across suspension, and the `tableswitch` over labels `0` and `1`. The method reads `IntrinsicsKt.getCOROUTINE_SUSPENDED()` and compares the result of `SafeContinuation.getOrThrow()` with that marker.

The resume helper calls:

```text
kotlin.Result."constructor-impl"
kotlin.coroutines.Continuation.resumeWith(Object)
```

That is the raw JVM shape behind Kotlin's `continuation.resume(value)` helper.

## Interop Caveats

Java can call the generated suspend method, but it must provide a valid `Continuation` and handle `COROUTINE_SUSPENDED`. The result type is erased to `Object`; generic information survives only in the `Signature` attribute.

The generated continuation class and state fields are compiler output. Their shape is useful for inspection, but code should not depend on their exact names. Kotlin metadata records the suspend declaration, but Java reflection over methods only sees the lowered JVM method.

## Follow-Up Experiments

Add a second suspension point and inspect the larger label switch. Add a thrown exception after resumption and observe `Result.Failure` delivery. Add `startCoroutine` from Kotlin and compare its wrapper continuation with the direct Java call.
