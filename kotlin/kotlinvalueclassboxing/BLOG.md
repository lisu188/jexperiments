# Kotlin Value Class Boxing Across Kotlin and Java

`@JvmInline` value classes erase to their underlying value in many Kotlin-to-Kotlin calls, but they still need wrapper methods and boxes at JVM boundaries. This experiment shows `constructor-impl`, `box-impl`, `unbox-impl`, mangled member names, nullable value-class handling, and collection boxing.

## Why This Matters

Java developers need to know why a nice Kotlin type may expose methods with names that Java source cannot call. Kotlin developers need to know when value classes are represented as the underlying type and when wrappers appear. Library authors need to treat mangled names and nullable value-class behavior as part of the ABI story.

The demo is safe. It creates in-memory values, reflects on generated methods, and prints deterministic output. It performs no network, hardware, or destructive work.

## Execution Path

The value class wraps a `String`:

```kotlin
@JvmInline
value class CustomerId(val raw: String) {
    fun display(): String = "customer:$raw"

    override fun toString(): String = "CustomerId($raw)"
}
```

The registry uses direct, nullable, and collection forms:

```kotlin
class CustomerRegistry {
    fun accept(id: CustomerId): String = "direct=${id.raw}"

    fun acceptNullable(id: CustomerId?): String = "nullable=${id?.raw ?: "missing"}"

    fun collect(ids: List<CustomerId>): String = ids.joinToString(separator = "|") { it.raw }
}
```

Java cannot spell `box-impl` as a source method, so the helper uses reflection:

```java
Method box = idClass.getDeclaredMethod("box-impl", String.class);
Method unbox = idClass.getDeclaredMethod("unbox-impl");
Object boxed = box.invoke(null, raw);
Object unboxed = unbox.invoke(boxed);
```

## What To Run

```bash
./gradlew --gradle-user-home /tmp/jexperiments-gradle-home --no-daemon :kotlinvalueclassboxing:runExperiment
./gradlew --gradle-user-home /tmp/jexperiments-gradle-home --no-daemon :kotlinvalueclassboxing:classes
javap -classpath kotlinvalueclassboxing/build/classes/kotlin/main:kotlinvalueclassboxing/build/classes/java/main -v -p -c experiments.kotlin.valueclassboxing.CustomerId
javap -classpath kotlinvalueclassboxing/build/classes/kotlin/main:kotlinvalueclassboxing/build/classes/java/main -v -p -c experiments.kotlin.valueclassboxing.CustomerRegistry
```

## Bytecode Landmarks

`CustomerId` contains generated static helpers:

```text
display-impl(String)
toString-impl(String)
hashCode-impl(String)
equals-impl(String, Object)
constructor-impl(String)
box-impl(String)
unbox-impl()
```

`box-impl` and `unbox-impl` are synthetic. The private wrapper constructor stores the underlying `String`.

`CustomerRegistry` shows mangled names for value-class parameters:

```text
accept-1OISdRc(String)
acceptNullable-0gCk3lE(String)
```

The descriptors use `String`, not `CustomerId`, for those direct calls. The collection method keeps a normal erased `List` descriptor plus a `Signature` attribute mentioning `List<CustomerId>`, and collection elements are boxed because generics need references.

## Interop Caveats

Java source cannot call mangled methods containing hyphens. Public Kotlin APIs that need Java callers should provide explicit Java-friendly wrappers or avoid value-class parameters at that boundary.

Nullable value classes need a representation that can carry `null`; with a reference underlying type here, the nullable method still uses `String`, but behavior differs for primitive-backed value classes. Reflection can see synthetic helpers, but relying on their names is compiler-sensitive.

## Follow-Up Experiments

Repeat the module with an `Int`-backed value class and compare nullable boxing. Add `@JvmName` wrappers for Java callers. Put `CustomerId` in a generic map and inspect casts at read sites.
