# Kotlin Inline, Noinline, Crossinline, and Reified Bytecode

Kotlin `inline` functions are a source-level promise to copy selected code into call sites, but not every lambda disappears. This experiment puts `inline`, `noinline`, `crossinline`, and `reified` in one compact demo and inspects what becomes bytecode in the callee and at the call site.

## Why This Matters

Java developers reading Kotlin libraries often see generated helper methods and lambda classes that do not match source intuition. Kotlin developers need to know when inline removes allocation and when `noinline` or returned `Runnable` wrappers keep objects. Library authors need to understand that inline function bodies become part of callers and therefore affect binary compatibility differently from ordinary method bodies.

The demo is safe and deterministic. It allocates small lambdas and a `Runnable`, prints strings, and performs no network, hardware, or destructive work.

## Execution Path

The inline function has one `noinline` lambda, one `crossinline` lambda, and a reified type parameter:

```kotlin
inline fun <reified T> renderValue(
    value: Any,
    prefix: String,
    noinline deferred: (String) -> String,
    crossinline immediate: (T) -> String
): String {
    val typed = if (value is T) immediate(value) else "not-${T::class.java.simpleName}"
    val savedForLater = listOf(deferred).single()
    return "${T::class.java.simpleName}:$typed:${savedForLater(prefix)}"
}
```

The call site captures a local value:

```kotlin
val captured = "capture"
val rendered = renderValue<String>(
    value = "kotlin",
    prefix = "prefix",
    deferred = { text -> "$text-$captured" },
    immediate = { text -> text.uppercase() + "-$captured" }
)
```

The `crossinline` example returns a `Runnable`, so a wrapper object remains:

```kotlin
inline fun runLater(crossinline block: () -> String): Runnable =
    Runnable { println("crossinline-runnable=${block()}") }
```

## What To Run

```bash
./gradlew --gradle-user-home /tmp/jexperiments-gradle-home --no-daemon :kotlininlinecapture:runExperiment
./gradlew --gradle-user-home /tmp/jexperiments-gradle-home --no-daemon :kotlininlinecapture:classes
javap -classpath kotlininlinecapture/build/classes/kotlin/main -v -p -c experiments.kotlin.inlinecapture.InlineCaptureExperimentKt
javap -classpath kotlininlinecapture/build/classes/kotlin/main -v -p -c 'experiments.kotlin.inlinecapture.InlineCaptureExperimentKt$main$$inlined$runLater$1'
```

## Bytecode Landmarks

The compiled `renderValue` method still exists for callable references and non-inlined uses. Inside it, `javap` shows `Intrinsics.reifiedOperationMarker`, because a standalone generic JVM method cannot really perform `T::class` without call-site substitution.

In `main`, the reified `String` checks are copied into the call site. The `immediate` lambda body is inlined into the call-site flow, while the `noinline` lambda is represented as a `Function1` allocation. Kotlin 2.3 emits `invokedynamic` for the noinline `Function1` in this build.

The `runLater` function has two visible shapes:

```text
InlineCaptureExperimentKt$runLater$1
InlineCaptureExperimentKt$main$$inlined$runLater$1
```

The first is the generic wrapper used by the function body. The second is the call-site-specific wrapper generated after inlining the `crossinline` block into a returned `Runnable`.

## Interop Caveats

Java cannot use reified type parameters because reification is a Kotlin call-site rewrite. Java can call the erased helper method, but the reified operations in that helper are not meaningful for normal Java use.

Inline functions are source-compatible conveniences with ABI consequences. Changing an inline body changes future callers after recompilation, but already compiled callers keep the old copied bytecode until they are rebuilt.

## Follow-Up Experiments

Compile with a different Kotlin lambda generation setting and compare `invokedynamic` with class-based lambdas. Add a public inline function in one module and call it from another module to show copied bytecode across project boundaries. Add a non-local return example and compare why `crossinline` rejects it.
