# Kotlin Properties, Fields, Accessors, and Callable References

Kotlin property syntax can produce fields, getters, setters, constants, static bridges, and callable-reference objects. This experiment compares `lateinit`, `const val`, `@JvmField`, `@JvmStatic`, backing fields, computed accessors, and property references in one small module.

## Why This Matters

Java developers need to know whether a Kotlin property is a field or an accessor method. Kotlin developers need to know when property references allocate helper objects and when `lateinit` adds runtime checks. Library authors need to choose `@JvmField` and `@JvmStatic` deliberately because they change Java source shape and binary surface.

The demo is safe. It mutates only an in-memory object and prints deterministic strings.

## Execution Path

The class mixes backing fields and accessors:

```kotlin
class PropertyBox(initial: String) {
    @JvmField
    var directJvmField: String = "direct-$initial"

    var mutable: String = initial

    val computed: String
        get() = "computed-${mutable.length}"

    lateinit var late: String
}
```

The companion exposes constants and Java-friendly statics:

```kotlin
companion object {
    const val CONST_NAME: String = "CONST"

    @JvmField
    val exposedField: String = "exposed"

    @JvmStatic
    fun staticAccessor(): String = "static-$exposedField"
}
```

The main function uses callable references:

```kotlin
val mutableRef = PropertyBox::mutable
mutableRef.set(box, "abcd")
val lateRef = PropertyBox::late
```

Java reads fields and accessors side by side:

```java
return PropertyBox.CONST_NAME
        + "|" + PropertyBox.exposedField
        + "|" + PropertyBox.staticAccessor()
        + "|" + box.getMutable()
        + "|" + box.directJvmField;
```

## What To Run

```bash
./gradlew --gradle-user-home /tmp/jexperiments-gradle-home --no-daemon :kotlinpropertyrefs:runExperiment
./gradlew --gradle-user-home /tmp/jexperiments-gradle-home --no-daemon :kotlinpropertyrefs:classes
javap -classpath kotlinpropertyrefs/build/classes/kotlin/main:kotlinpropertyrefs/build/classes/java/main -v -p -c experiments.kotlin.propertyrefs.PropertyBox
javap -classpath kotlinpropertyrefs/build/classes/kotlin/main:kotlinpropertyrefs/build/classes/java/main -v -p -c experiments.kotlin.propertyrefs.PropertyRefsExperimentKt
```

## Bytecode Landmarks

`PropertyBox` contains a public field for `directJvmField`, private storage for `mutable`, and public `getMutable`/`setMutable` accessors. `computed` has `getComputed` but no field. `late` has a public field plus generated accessors that enforce initialization checks.

The companion produces:

```text
public static final String CONST_NAME
public static final String exposedField
public static final String staticAccessor()
public static final PropertyBox$Companion Companion
```

The callable references in `main` are singleton helper classes:

```text
PropertyRefsExperimentKt$main$mutableRef$1
PropertyRefsExperimentKt$main$lateRef$1
```

The bytecode invokes `KMutableProperty1.set(Object, Object)` and `KMutableProperty1.get(Object)`, so the property-reference path is not the same as a direct getter/setter call.

## Interop Caveats

`const val` is a compile-time constant; Java callers may inline it. Changing its value may require Java callers to recompile. `@JvmField` removes accessor use for Java and exposes a field directly. `@JvmStatic` adds a static bridge but keeps the companion model for Kotlin.

Callable references need Kotlin reflection interfaces from the standard library. They are useful but not free, and they expose property names in metadata-sensitive ways.

## Follow-Up Experiments

Compare `const val` changes with already compiled Java callers. Add `@get:JvmName` and inspect accessor names. Add a delegated property and compare its reference path with the direct property reference path.
