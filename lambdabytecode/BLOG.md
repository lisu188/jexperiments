# Comparing Lambda Bytecode Shapes

## Why this experiment exists

This experiment compares several Java 8 lambda and method-reference source
forms that all do nearly the same thing: iterate over a fixed list of integers
and print or construct something for each value. The runtime behavior is small,
but the bytecode shapes are different. That makes the module a useful starting
point for inspecting `invokedynamic`, bootstrap methods, synthetic helper
methods, and the way `javac` represents lambda bodies.

The source does not include a bytecode disassembler. Instead, it provides a
set of tiny classes that can be compiled and inspected with `javap -c -v`.
Because each class isolates one source form, the generated classfile differences
are easier to compare than they would be in a larger application.

For experienced Java developers, the key lesson is that a lambda expression is
not an anonymous inner class in modern Java. The compiler emits an
`invokedynamic` call site whose bootstrap points at `LambdaMetafactory`. The
exact method handle used by that call site depends on whether the source is an
expression lambda, a static method reference, an instance method reference, or a
constructor reference.

## Execution path

All examples share the same constants:

```java
public static final List<Integer> INTEGERS = Arrays.asList(1, 2, 3, 4, 5);

public static void println(Integer x) {
    System.out.println(x);
}
```

The list supplies deterministic input. The static `println` method gives the
method-reference examples a target that is owned by application code rather
than `java.io.PrintStream`.

The expression-lambda example is the most explicit source form:

```java
public static void main(String[] args) {
    LambdaConstant.INTEGERS.stream().forEach(x -> {
        System.out.println(x);
    });
}
```

Compiling this shape usually creates a private synthetic method for the lambda
body and an `invokedynamic` instruction that adapts it to the `Consumer`
functional interface expected by `forEach`.

That synthetic method is one reason expression lambdas are useful to inspect
first. The source body has no standalone name, so the compiler has to invent an
implementation target. When reading `javap -v` output, compare the bootstrap
method's implementation handle with the generated private method. That link is
the bridge from source lambda body to runtime functional object.

## Core code walkthrough

The static method reference delegates to application code:

```java
public static void main(String[] args) {
    LambdaConstant.INTEGERS.stream().forEach(LambdaConstant::println);
}
```

Here the compiler does not need to synthesize a body equivalent to
`System.out.println(x)` inside the class containing `main`. It can point the
bootstrap method at `LambdaConstant.println(Integer)`. The resulting call site
still produces a `Consumer<Integer>`, but the implementation method is already
named in source.

The member method reference uses a receiver object:

```java
public static void main(String[] args) {
    LambdaConstant.INTEGERS.stream().forEach(System.out::println);
}
```

This shape is useful because `System.out` is a field read that produces the
receiver for the referenced instance method. In bytecode inspection, look for
where that receiver is loaded and how the generated functional object captures
or binds it.

The constructor reference points at `new`:

```java
public static void main(String[] args) {
    LambdaConstant.INTEGERS.stream().forEach(LambdaConstant.Constructor::new);
}
```

The target class is intentionally tiny: `LambdaConstant.Constructor` has a
constructor that accepts `Integer x` and calls `System.out.println(x)`. The
source reads like a factory, but `forEach` still receives a `Consumer`. The
constructed object is ignored after each call, so the side effect is in the
constructor.

```java
// Constructor reference target: allocation is the side effect carrier.
public static class Constructor {
    public Constructor(Integer x) {
        System.out.println(x);
    }
}
```

When inspecting bytecode, this target shows up as a constructor method handle
rather than a synthetic lambda body.

This is also a reminder that method-reference syntax does not imply a specific
functional interface. The target type comes from context. In this module the
context is `Stream.forEach`, so every example becomes something compatible with
`Consumer<? super Integer>`. Put the same method reference in a different
assignment context and the compiler may choose a different descriptor or reject
it.

## Important implementation details

The module has five small classes rather than one class with five methods. That
keeps each generated classfile focused. When you run `javap -c -v` on
`ExpressionLambda`, the constant pool and bootstrap method table describe only
that expression-lambda case. Running the same command on
`StaticMethodReferenceLambda` or `ConstructorReferenceLambda` lets you compare
targets without filtering through unrelated examples.

All examples use `stream().forEach(...)`, not a plain enhanced `for` loop. That
choice forces the source form to become an implementation of a functional
interface. The stream itself is not the interesting part; `Consumer<Integer>` is
the interesting target type.

The constant list is also deliberately shared. If every class declared its own
input, bytecode differences would include unrelated class-initialization noise.
By centralizing the list and helper methods in `LambdaConstant`, each example's
main method is mostly about the lambda shape and the method reference target.

```java
// Shared input keeps each example focused on lambda shape.
public static final List<Integer> INTEGERS = Arrays.asList(1, 2, 3, 4, 5);
```

That constant is boring at runtime and useful for bytecode comparison.

The examples also avoid captured local variables. Capturing would add another
dimension to the bytecode because values would need to be passed into the
factory call site. These classes focus on non-capturing or receiver-bound forms
first, which is the right baseline before comparing closure state.

## Runtime behavior and caveats

Running any of the main classes prints the numbers 1 through 5, although the
constructor-reference class does so through constructor side effects. The
runtime output is therefore not enough to understand the experiment. The real
inspection step is bytecode analysis after compilation.

The exact bytecode can vary by JDK version. The high-level shape, especially
the use of `invokedynamic` for lambdas, is stable for modern Java, but constant
pool ordering, synthetic method names, and bootstrap details are compiler
outputs rather than source-level API. If you compare results across JDKs, keep
the compiler version in the notes.

Another caveat is that these examples are too small to say anything meaningful
about performance. Method references are not automatically faster than
expression lambdas, and allocation behavior can depend on capture state and JVM
optimization. Treat this module as a classfile inspection fixture, not a
benchmark.

When inspecting output, pay attention to three places: the `main` bytecode, the
`BootstrapMethods` attribute, and any synthetic methods emitted by the compiler.
The source difference may be one token, but the classfile difference can move
from a generated helper method to a direct method handle or constructor handle.

## Suggested next experiments

Add a lambda that captures a local variable and compare its bootstrap arguments
with the current non-capturing expression lambda. Add an unbound instance
method reference such as `String::length`. Compile with two JDK versions and
diff the `javap -v` output. Finally, add a tiny test that runs `javap` during
the build and stores selected bytecode observations as documentation rather
than relying on manual inspection.

It would also be useful to add one overloaded method-reference example. That
would force the target functional interface to participate in overload
resolution, making the connection between source context and generated
descriptor even clearer. Pairing that with the compiler error for an ambiguous
reference would make the type-inference boundary visible too.
It is a small change with useful bytecode evidence.
