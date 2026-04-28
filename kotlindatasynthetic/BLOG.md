# Kotlin Data Class Synthetic Members

Kotlin data classes generate a compact API surface in source and a much larger class file. This experiment inspects `componentN`, `copy`, `copy$default`, default constructors, `equals`, `hashCode`, `toString`, and Java usability for a small data class.

## Why This Matters

Java developers using Kotlin data classes need to know which methods are real Java-callable methods and which helpers are synthetic. Kotlin developers need to understand how `copy(version = 2)` becomes a method call plus default masks. Library authors need to know that changing data-class constructor parameters changes generated method names, descriptors, and binary compatibility.

The demo is safe and deterministic. It constructs values, calls generated methods, reflects on `copy$default`, and prints strings.

## Execution Path

The data class has two required properties and one defaulted property:

```kotlin
data class BuildCoordinate(
    val group: String,
    val artifact: String,
    val version: Int = 1
) {
    fun gav(): String = "$group:$artifact:$version"
}
```

Kotlin source uses named arguments and `copy`:

```kotlin
val coordinate = BuildCoordinate(group = "experiments", artifact = "data")
val copy = coordinate.copy(version = 2)
```

The Java helper uses the public constructor and public `copy`, then reflects on `copy$default`:

```java
Method copyDefault = BuildCoordinate.class.getDeclaredMethod(
        "copy$default",
        BuildCoordinate.class,
        String.class,
        String.class,
        int.class,
        int.class,
        Object.class);
Object defaulted = copyDefault.invoke(null, coordinate, null, null, 3, 0b001 | 0b010, null);
```

## What To Run

```bash
./gradlew --gradle-user-home /tmp/jexperiments-gradle-home --no-daemon :kotlindatasynthetic:runExperiment
./gradlew --gradle-user-home /tmp/jexperiments-gradle-home --no-daemon :kotlindatasynthetic:classes
javap -classpath kotlindatasynthetic/build/classes/kotlin/main:kotlindatasynthetic/build/classes/java/main -v -p -c experiments.kotlin.datasynthetic.BuildCoordinate
```

## Bytecode Landmarks

`BuildCoordinate` contains:

```text
component1()
component2()
component3()
copy(String, String, int)
copy$default(BuildCoordinate, String, String, int, int, Object)
equals(Object)
hashCode()
toString()
```

Because `version` has a default, `javap` also shows a synthetic constructor:

```text
(Ljava/lang/String;Ljava/lang/String;IILkotlin/jvm/internal/DefaultConstructorMarker;)V
```

`copy$default` has `ACC_SYNTHETIC`. The mask uses bit `1` for `group`, bit `2` for `artifact`, and bit `4` for `version`. The Java helper passes `null` for the first two values and sets bits `1` and `2`, so the copied object keeps those original fields while accepting the supplied version.

## Interop Caveats

Java can call `componentN`, `copy`, getters, and the primary constructor. Java cannot use named arguments, and it should not call `copy$default` except for inspection because the mask protocol is compiler ABI.

Adding, removing, or reordering primary-constructor properties changes `componentN` numbering and `copy` descriptors. That is a source-level data-model change and also a binary compatibility change.

## Follow-Up Experiments

Add a body property outside the primary constructor and prove it is excluded from `copy` and `componentN`. Add a nullable property and inspect null-check placement. Compare a regular class with manually written `equals` and `hashCode`.
