# Generating a Class with BCEL

## Why this experiment exists

This experiment is a compact tour of Apache BCEL, the Byte Code Engineering
Library. Instead of writing a Java source file and asking `javac` to produce a
class, the code builds a JVM class file directly through BCEL's object model.
The generated class is named `HelloWorld`, and its `main` method prompts for a
name, reads a line from standard input, and prints a greeting.

That makes the module useful as a small laboratory for thinking about what the
Java compiler usually hides. Local variables, stack operations, constructor
calls, exception handlers, and method signatures all have to be assembled
explicitly. The result is not a new language or framework. It is a hands-on look
at JVM bytecode generation.

## How it works

The entrypoint is `com.company.Main`. It creates a `ClassGen` for a public
`HelloWorld` class extending `java.lang.Object`, then builds a constant pool, an
`InstructionList`, and a `MethodGen` for `main(String[] argv)`.

The method body is assembled instruction by instruction. The code creates a
`BufferedReader` around `System.in`, stores it in a local variable, initializes a
`name` variable, prints a prompt, calls `readLine`, and stores the result. It
also adds an exception handler for `IOException`. After the try block, it builds
a greeting with `StringBuffer` and emits a `PrintStream.println` call.

The final steps are bytecode housekeeping: compute the max stack, add the method
to the class, add an empty constructor, and dump `HelloWorld.class` to disk.

## What to notice

The interesting part is how explicit the boundary between Java-the-language and
Java-the-virtual-machine becomes. A local variable is allocated with
`addLocalVariable`, assigned a slot index, and given a start instruction handle.
A `new BufferedReader(...)` expression becomes object allocation, `DUP`,
constructor argument setup, and `INVOKESPECIAL`.

The experiment also shows why bytecode libraries can feel verbose. They are not
just APIs for writing Java in another syntax. They model details that source
code intentionally abstracts away. That verbosity is valuable for
instrumentation, code generation, class transformation, and JVM education.

## Sanity check

The Gradle build keeps the original BCEL dependency, `org.apache.bcel:bcel:5.2`,
and compiles the module with Java 9 source and target compatibility. There are
no tests. Running the program writes `HelloWorld.class` to the current working
directory, and `.class` files are ignored by the repository.

The repository-level Gradle wrapper build compiles this module successfully.

## Suggested next experiments

A natural follow-up would be generating a second method and invoking it from
`main`, because that would make constant-pool references and method descriptors
more visible. Another useful extension would be reading an existing class,
modifying one method, and writing the transformed class back out.
