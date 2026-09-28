# Kotlin Fun Interfaces, Java SAMs, and LambdaMetafactory

Kotlin `fun interface` declarations are JVM single-abstract-method types, so both Kotlin and Java can produce implementations with lambdas and method references. This experiment compares Kotlin and Java output for the same SAM target and records where `invokedynamic` and generated classes appear.

## Why This Matters

Java developers need to know Kotlin fun interfaces are ordinary SAM interfaces at the JVM level. Kotlin developers need to know that lambda and method-reference lowering can differ between source forms. Library authors need to inspect SAM APIs because generated call sites, bootstrap methods, and Java source usability affect public API ergonomics.

The demo is safe. It creates in-memory SAM objects and prints deterministic arithmetic results.

## Execution Path

The Kotlin SAM type is minimal:

```kotlin
fun interface KotlinOperation {
    fun apply(value: Int): Int
}
```

Kotlin creates one lambda and one method-reference implementation:

```kotlin
fun double(value: Int): Int = value * 2

fun kotlinLambda(): KotlinOperation = KotlinOperation { value -> value + 1 }

fun kotlinMethodReference(): KotlinOperation = KotlinOperation(::double)
```

Java creates equivalent shapes:

```java
public static KotlinOperation javaLambda() {
    return value -> value + 10;
}

public static KotlinOperation javaMethodReference() {
    return JavaSamSources::triple;
}
```

## What To Run

```bash
./gradlew --gradle-user-home /tmp/jexperiments-gradle-home --no-daemon :kotlinsamfuninterface:runExperiment
./gradlew --gradle-user-home /tmp/jexperiments-gradle-home --no-daemon :kotlinsamfuninterface:classes
javap -classpath kotlinsamfuninterface/build/classes/kotlin/main:kotlinsamfuninterface/build/classes/java/main -v -p -c experiments.kotlin.samfuninterface.SamFunInterfaceExperimentKt
javap -classpath kotlinsamfuninterface/build/classes/kotlin/main:kotlinsamfuninterface/build/classes/java/main -v -p -c experiments.kotlin.samfuninterface.JavaSamSources
```

## Bytecode Landmarks

In this Kotlin 2.3.0 build, `kotlinLambda()` uses:

```text
invokedynamic apply()KotlinOperation
BootstrapMethods -> LambdaMetafactory.metafactory
```

The implementation handle points at the synthetic static method:

```text
kotlinLambda$lambda$0(int)
```

The Kotlin method reference is different here: `kotlinMethodReference()` returns the singleton helper class:

```text
SamFunInterfaceExperimentKt$kotlinMethodReference$1.INSTANCE
```

Java emits `invokedynamic` for both `javaLambda()` and `javaMethodReference()`. The bootstrap implementation handles point at `lambda$javaLambda$0(int)` and `triple(int)`.

## Interop Caveats

Kotlin fun interfaces are Java-callable, but Kotlin nullability and metadata are not part of the SAM method descriptor. Java can pass `null` where Kotlin source would reject it unless runtime checks are present at the receiving boundary.

Lambda lowering is compiler-version-sensitive. The durable facts are the SAM interface descriptor and the bootstrap/helper artifacts that `javap` shows for the compiler version used here.

## Follow-Up Experiments

Add a capturing Java lambda and compare bootstrap arguments. Add a Kotlin capturing method reference and inspect whether it stays a singleton. Add an overloaded Java SAM API and study Kotlin overload resolution.
