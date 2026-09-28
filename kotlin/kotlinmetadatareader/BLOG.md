# Reading `kotlin.Metadata` Without a Metadata Library

Every Kotlin class file carries a `kotlin.Metadata` annotation that helps Kotlin tools reconstruct declarations beyond raw JVM descriptors. This experiment reads that annotation reflectively from compiled classes and explains what the values can and cannot tell you without a parser such as `kotlin-metadata-jvm`.

## Why This Matters

Java developers often encounter Kotlin classes through reflection and need to know why Java reflection alone misses Kotlin details. Kotlin developers need to understand that metadata is a compiler contract for tools, not a replacement for bytecode. Library authors need to preserve metadata carefully because serializers, reflection frameworks, and documentation tools often depend on it.

The demo is safe. It reads annotations from classes already loaded in the JVM and prints deterministic values. It performs no network, hardware, file writes, or destructive work.

## Execution Path

The Kotlin targets include a data class, a sealed class, nested subclasses, and a top-level function:

```kotlin
data class MetadataRecord(
    val name: String,
    val flags: Int = 7
)

sealed class MetadataShape {
    data class Point(val x: Int, val y: Int) : MetadataShape()

    object Empty : MetadataShape()
}

fun metadataTopLevel(record: MetadataRecord): String = "${record.name}:${record.flags}"
```

The Java reader uses only reflection and the annotation type from the Kotlin standard library:

```java
Metadata metadata = type.getAnnotation(Metadata.class);
System.out.println(type.getSimpleName()
        + ": kind=" + metadata.k()
        + " mv=" + Arrays.toString(metadata.mv())
        + " d1=" + metadata.d1().length
        + " d2.head=" + head(metadata.d2())
        + " xi=" + metadata.xi()
        + " pn=" + metadata.pn());
```

The output shows class metadata kind `1` and file-facade metadata kind `2`:

```text
MetadataRecord: kind=1 mv=[2, 3, 0] d1=1 d2.head=Lexperiments/kotlin/metadatareader/MetadataRecord; xi=48 pn=
MetadataTargetsKt: kind=2 mv=[2, 3, 0] d1=1 d2.head=metadataTopLevel xi=48 pn=
```

## What To Run

```bash
./gradlew --gradle-user-home /tmp/jexperiments-gradle-home --no-daemon :kotlinmetadatareader:runExperiment
./gradlew --gradle-user-home /tmp/jexperiments-gradle-home --no-daemon :kotlinmetadatareader:classes
javap -classpath kotlinmetadatareader/build/classes/kotlin/main:kotlinmetadatareader/build/classes/java/main -v -p -c experiments.kotlin.metadatareader.MetadataRecord
javap -classpath kotlinmetadatareader/build/classes/kotlin/main:kotlinmetadatareader/build/classes/java/main -v -p -c experiments.kotlin.metadatareader.MetadataTargetsKt
```

## Bytecode Landmarks

`javap -v` on `MetadataRecord` shows `RuntimeVisibleAnnotations` with:

```text
kotlin.Metadata(
  mv=[2,3,0]
  k=1
  xi=48
  d1=[...]
  d2=[...]
)
```

The same class also has normal data-class bytecode such as `copy$default`, a default constructor with `DefaultConstructorMarker`, and generated `componentN` methods. Metadata complements those class-file facts; it does not replace them.

For `MetadataTargetsKt`, `k=2` marks a file facade. The `d2` array includes names such as `metadataTopLevel`, but the strings are encoded for Kotlin tooling, not designed as a stable human-readable API.

## Interop Caveats

Java reflection can read annotation methods such as `k()`, `mv()`, `d1()`, and `d2()`, but it cannot decode Kotlin declarations by itself. A real tool should use a metadata parser that understands the metadata version.

Metadata is compiler-version-sensitive. This build prints `mv=[2, 3, 0]` with Kotlin 2.3.0. Older tools may not understand newer metadata. Conversely, stripping metadata can make Kotlin classes look like plain JVM classes to Kotlin-aware frameworks.

## Follow-Up Experiments

Add `kotlin-metadata-jvm` and decode declarations into readable signatures. Strip the annotation from a copied class file and observe reflection-framework behavior. Compare metadata for a value class, a suspend function, and a multifile facade.
