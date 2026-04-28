# Kotlin Sealed Hierarchies and Exhaustive `when`

Kotlin sealed hierarchies let the compiler prove a `when` is exhaustive, but the JVM still sees ordinary classes, singleton objects, `instanceof` checks, and enum switch mapping arrays. This experiment compares a sealed-interface `when` with an enum `when`.

## Why This Matters

Java developers need to understand that Kotlin sealed exhaustiveness is not the same as Java sealed bytecode when targeting Java 9. Kotlin developers need to know how source exhaustiveness lowers into runtime tests. Library authors need to be careful when evolving sealed hierarchies because new subtypes affect exhaustive callers.

The demo is safe. It creates three in-memory events and prints deterministic descriptions.

## Execution Path

The hierarchy is sealed in Kotlin source:

```kotlin
sealed interface Event {
    val id: Int
}

data class Started(override val id: Int, val name: String) : Event
data class Stopped(override val id: Int, val code: Int) : Event

object Heartbeat : Event {
    override val id: Int = -1
}
```

The `when` over `Event` is exhaustive without an `else`:

```kotlin
fun describe(event: Event): String =
    when (event) {
        is Started -> "started:${event.id}:${event.name}"
        is Stopped -> "stopped:${event.id}:${event.code}"
        Heartbeat -> "heartbeat"
    }
```

The enum `when` is a separate lowering target:

```kotlin
fun severityCode(severity: Severity): Int =
    when (severity) {
        Severity.LOW -> 1
        Severity.MEDIUM -> 2
        Severity.HIGH -> 3
    }
```

## What To Run

```bash
./gradlew --gradle-user-home /tmp/jexperiments-gradle-home --no-daemon :kotlinsealedwhen:runExperiment
./gradlew --gradle-user-home /tmp/jexperiments-gradle-home --no-daemon :kotlinsealedwhen:classes
javap -classpath kotlinsealedwhen/build/classes/kotlin/main -v -p -c experiments.kotlin.sealedwhen.SealedWhenExperimentKt
javap -classpath kotlinsealedwhen/build/classes/kotlin/main -v -p -c 'experiments.kotlin.sealedwhen.SealedWhenExperimentKt$WhenMappings'
```

## Bytecode Landmarks

`describe(Event)` uses an `instanceof` chain for `Started` and `Stopped`, then compares with `Heartbeat.INSTANCE`. Even though the source has no `else`, `javap` shows a defensive `NoWhenBranchMatchedException` path.

`severityCode(Severity)` uses a generated mapping class:

```text
SealedWhenExperimentKt$WhenMappings.$EnumSwitchMapping$0
```

The function reads `severity.ordinal()`, indexes the mapping array, and then uses a `tableswitch`. This is different from the sealed hierarchy path.

## Interop Caveats

With JVM target 9, Kotlin sealed types are enforced by Kotlin metadata and compiler rules, not by Java 17 `PermittedSubclasses` attributes. Java code can still see ordinary classes and interfaces.

Adding a new sealed subtype can force Kotlin callers to update exhaustive `when` expressions. Existing compiled callers keep their old bytecode until rebuilt, and the defensive `NoWhenBranchMatchedException` path is one reason this matters.

## Follow-Up Experiments

Compile a similar hierarchy with JVM target 17 and inspect Java sealed attributes. Add a sealed class with nested subclasses and compare metadata. Add a `when` over strings or ints and compare `lookupswitch` and `tableswitch`.
