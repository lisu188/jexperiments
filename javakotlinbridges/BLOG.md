# Java and Kotlin Bridges at API Edges

Kotlin and Java agree on JVM descriptors, but they bring different source type systems. This experiment combines generic overrides, variance, platform types, `@JvmName`, `@JvmSynthetic`, and Java callers to show where the compiler emits bridges, metadata, renamed methods, and null-check boundaries.

## Why This Matters

Java developers need to understand Kotlin bridge methods and renamed top-level functions when calling Kotlin libraries. Kotlin developers need to know where Java platform types bypass source nullability. Library authors need to design overloads and generic APIs with erasure, bridge generation, and Java source names in mind.

The demo is safe. It constructs in-memory values and prints deterministic strings.

## Execution Path

The generic override forces a bridge:

```kotlin
open class KotlinBase<T> {
    open fun identity(value: T): T = value
}

class StringKotlinBase : KotlinBase<String>() {
    override fun identity(value: String): String = value.uppercase()
}
```

Top-level overloads that would erase to the same descriptor use `@JvmName`:

```kotlin
@JvmName("joinStrings")
fun join(values: List<String>): String = values.joinToString(separator = ":")

@JvmName("joinInts")
fun join(values: List<Int>): String = values.joinToString(separator = ":")
```

The Java caller touches raw generics and renamed APIs:

```java
@SuppressWarnings({"rawtypes", "unchecked"})
public static String call() {
    StringKotlinBase typed = new StringKotlinBase();
    KotlinBase raw = typed;
    Object bridge = raw.identity("bridge");
    OverloadEdges edges = new OverloadEdges();
    return typed.identity("java")
            + "|" + bridge
            + "|" + BridgeExperimentKt.joinStrings(Arrays.asList("x", "y"))
            + "|" + edges.acceptNullableName(null)
            + "|" + edges.accept(7);
}
```

## What To Run

```bash
./gradlew --gradle-user-home /tmp/jexperiments-gradle-home --no-daemon :javakotlinbridges:runExperiment
./gradlew --gradle-user-home /tmp/jexperiments-gradle-home --no-daemon :javakotlinbridges:classes
javap -classpath javakotlinbridges/build/classes/kotlin/main:javakotlinbridges/build/classes/java/main -v -p -c experiments.kotlin.bridges.StringKotlinBase
javap -classpath javakotlinbridges/build/classes/kotlin/main:javakotlinbridges/build/classes/java/main -v -p -c experiments.kotlin.bridges.BridgeExperimentKt
```

## Bytecode Landmarks

`StringKotlinBase` has the real override:

```text
identity(String): String
```

It also has the erased bridge:

```text
identity(Object): Object
flags: ACC_PUBLIC, ACC_BRIDGE, ACC_SYNTHETIC
```

The bridge casts to `String` and invokes the typed method.

`BridgeExperimentKt` exposes:

```text
joinStrings(List): String
joinInts(List): String
kotlinOnlySecret(): String
```

`joinStrings` and `joinInts` have erased `List` descriptors plus generic `Signature` attributes. `kotlinOnlySecret` is `ACC_SYNTHETIC` because of `@JvmSynthetic`, so Java source does not see it even though bytecode tools do.

`platformLength(JavaPlatformProvider)` calls the Java `maybeText()` method and then `String.length`. The Kotlin source treats that Java return as a platform type; the Java method descriptor itself does not encode Kotlin nullability.

## Interop Caveats

Raw Java calls can route through bridges and defer type errors to casts. `@JvmName` is useful for erasure conflicts, but it creates Java names that differ from Kotlin source names. `@JvmSynthetic` hides APIs from Java source, not from reflection or bytecode.

Platform types are not a guarantee. If `JavaPlatformProvider.maybeText()` returned `null`, Kotlin source would allow the call path but the runtime would fail when dereferencing the value.

## Follow-Up Experiments

Add Java nullability annotations and inspect Kotlin's generated checks. Add a covariant Java override and compare bridge generation from `javac`. Add binary-compatible and binary-incompatible `@JvmName` changes and inspect old callers.
