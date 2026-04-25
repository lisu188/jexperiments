# Comparing Lambda Bytecode Shapes

## Why this experiment exists

This module is a small set of Java lambda and method-reference examples. Each
class iterates over the same list of integers and prints or consumes the values
through a different source-level shape: an expression lambda, a static method
reference, an instance method reference, and a constructor reference.

The source code is intentionally tiny because the interesting part is not what
the program prints. The interesting part is what the Java compiler emits for
each form. This module is a setup for compiling the examples and inspecting the
resulting bytecode with tools such as `javap`.

## How it works

`LambdaConstant` provides the shared data and helpers. It exposes
`INTEGERS = Arrays.asList(1, 2, 3, 4, 5)`, a static `println(Integer)` method,
and a nested `Constructor` class whose constructor prints the integer it
receives.

The example classes are deliberately parallel:

```java
LambdaConstant.INTEGERS.stream().forEach(x -> {
    System.out.println(x);
});
```

```java
LambdaConstant.INTEGERS.stream().forEach(LambdaConstant::println);
```

```java
LambdaConstant.INTEGERS.stream().forEach(System.out::println);
```

```java
LambdaConstant.INTEGERS.stream().forEach(LambdaConstant.Constructor::new);
```

Each version is a valid `Consumer<Integer>` for `forEach`, but each reaches that
consumer through a different expression form.

## What to notice

The module is especially useful when paired with bytecode inspection. Java 8 and
newer lambda expressions are not compiled as anonymous inner classes in the old
Java 7 style. They are normally represented through `invokedynamic` and
bootstrap methods. Method references and constructor references provide
different metadata to that bootstrap machinery while keeping source code
compact.

Because the examples are so small, differences in constant-pool entries,
synthetic methods, and bootstrap arguments are easier to spot. That is the real
purpose of the module: reduce source-level noise until bytecode-level structure
becomes visible.

## Sanity check

The module has no external dependencies and compiles through the shared Gradle
Java configuration. There are no tests. Each class has its own `main` method and
should produce the same visible output: the numbers one through five printed to
standard output.

The repository-level Gradle wrapper build compiles this module successfully.

## Suggested next experiments

A useful addition would be a checked-in script or documentation snippet showing
the exact `javap -c -v` command for each class. Another extension would compare
capturing and non-capturing lambdas, because captured variables change the
runtime shape and make the generated call sites more interesting.
