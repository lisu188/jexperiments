# Generating a Class with BCEL

## Why this experiment exists

This experiment is a compact but revealing tour of Apache BCEL, the Byte Code
Engineering Library. Instead of asking `javac` to translate Java source into a
class file, the program constructs a JVM class directly from BCEL's model of
classes, constant pools, instructions, local variables, exception tables, and
method metadata. The generated class is named `HelloWorld`, and its `main`
method asks for a name on standard input before printing a greeting.

For experienced Java developers, the value is not the greeting. The value is
seeing which details Java source normally erases from view. An expression such
as `new BufferedReader(new InputStreamReader(System.in))` becomes allocation,
stack duplication, field access, constructor invocation, and local-variable
storage. A `try` block is not syntax anymore; it is a pair of instruction
handles plus an exception-table entry. Even a simple `println` requires a
constant-pool reference, a receiver on the operand stack, and a descriptor that
matches the selected JVM instruction.

This makes the module a good laboratory for bytecode generation, build-time
instrumentation, and classfile education. It is intentionally small enough to
read in one sitting, but it touches the pieces that real code generators have
to manage carefully.

## Execution path

The entrypoint is `com.company.Main`. Running it does not execute the generated
program. It writes `HelloWorld.class` into the current working directory. You
then inspect that file with tools such as `javap` or run it as a normal class if
the generated file is on the classpath.

The generator starts by declaring the class and the `main` method:

```java
ClassGen cg = new ClassGen("HelloWorld", "java.lang.Object",
        "<generated>", Constants.ACC_PUBLIC | Constants.ACC_SUPER,
        null);
ConstantPoolGen cp = cg.getConstantPool(); // cg creates constant pool
InstructionList il = new InstructionList();

MethodGen mg = new MethodGen(Constants.ACC_STATIC | Constants.ACC_PUBLIC,
        Type.VOID,
        new Type[]{
                new ArrayType(Type.STRING, 1)},
        new String[]{"argv"},
        "main", "HelloWorld",
        il, cp);
```

`ClassGen` owns the evolving class definition. `ConstantPoolGen` tracks strings,
method references, field references, and type descriptors as instructions are
created. `InstructionList` is the mutable instruction stream for the method
body. The file name `"<generated>"` is debug metadata, not a Java source file.
The `MethodGen` constructor says: create `public static void main(String[] argv)` in the
generated `HelloWorld` class, using the instruction list and constant pool
already being assembled. From that point on, every appended instruction becomes
part of this method.

## Core code walkthrough

The first real operation constructs the input reader. BCEL does not have a
single instruction for nested constructor expressions, so the code spells out
the stack choreography:

```java
il.append(factory.createNew("java.io.BufferedReader"));
il.append(InstructionConstants.DUP);
il.append(factory.createNew("java.io.InputStreamReader"));
il.append(InstructionConstants.DUP);
il.append(factory.createFieldAccess("java.lang.System", "in", i_stream,
        Constants.GETSTATIC));
```

The two `DUP` instructions preserve object references for the following
constructor calls. `GETSTATIC` pushes `System.in`. After the nested
`INVOKESPECIAL` calls finish, a fully constructed `BufferedReader` remains on
the operand stack and can be stored into a local variable.

Local variables are also explicit:

```java
LocalVariableGen lg = mg.addLocalVariable("in",
        new ObjectType("java.io.BufferedReader"), null, null);
int in = lg.getIndex();
lg.setStart(il.append(new ASTORE(in)));
```

The variable name is metadata. The slot index is what the bytecode uses.
`ASTORE` consumes the reader reference from the stack and writes it to that
slot. The later `ALOAD(in)` instruction reloads it before calling `readLine`.

Exception handling is attached to instruction handles rather than lexical
braces:

```java
GOTO g = new GOTO(null);
InstructionHandle try_end = il.append(g);

InstructionHandle handler = il.append(InstructionConstants.RETURN);
mg.addExceptionHandler(try_start, try_end, handler, new ObjectType("java.io.IOException"));
```

The generated method returns immediately on `IOException`. The normal path uses
the `GOTO` to skip over that handler and continue with greeting output.

## Important implementation details

The final output uses the pre-Java-5 `StringBuffer` style that BCEL examples
often use. It is verbose, but it demonstrates receiver construction and virtual
dispatch clearly:

```java
il.append(factory.createNew(Type.STRINGBUFFER));
il.append(InstructionConstants.DUP);
il.append(new PUSH(cp, "Hello, "));
il.append(factory.createInvoke("java.lang.StringBuffer", "<init>",
        Type.VOID, new Type[]{Type.STRING},
        Constants.INVOKESPECIAL));
```

After construction, the generator appends the `name` local, calls `append`, then
calls `toString`, leaving the final greeting as the argument for
`PrintStream.println`.

The cleanup section matters as much as the instruction list. The code calls
`mg.setMaxStack()`, `cg.addMethod(mg.getMethod())`, `il.dispose()`, and
`cg.addEmptyConstructor(Constants.ACC_PUBLIC)`. `setMaxStack()` asks BCEL to
compute the required stack depth instead of making the author count it manually.
`addMethod` freezes the method into the class. `dispose` releases instruction
handles, and `addEmptyConstructor` gives the generated class a public
no-argument constructor even though the demo only uses `main`.

```java
// Compute verifier metadata before freezing the generated method.
mg.setMaxStack();
cg.addMethod(mg.getMethod());
il.dispose(); // Allow instruction handles to be reused
cg.addEmptyConstructor(Constants.ACC_PUBLIC);
```

The method is not part of the class until `addMethod` is called. The stack
calculation is also deliberately late because the instruction list must exist
before BCEL can compute a safe operand-stack depth.

Descriptors are another detail worth noticing. BCEL APIs require the caller to
state return types and argument types for every method invocation. That means a
bad descriptor is not a Java compile error in the generated class's source; it
is a classfile construction error or a verifier/runtime problem later. The demo
uses `Type.STRING`, `Type.VOID`, `Type.NO_ARGS`, `ObjectType`, and `ArrayType`
so the mapping between Java concepts and JVM descriptors is visible without
requiring the reader to decode raw strings such as `([Ljava/lang/String;)V`.

Instruction handles also give branch targets identity. The `GOTO` is created
with a null target first because the destination instruction does not exist yet.
Only after the handler is appended can the normal continuation be appended and
installed as the target. That pattern appears frequently in code generators:
emit a placeholder, keep the handle, then patch control flow after both sides
of the branch are known.

```java
// Materialize the class and write it as a file for external inspection.
JavaClass javaClass = cg.getJavaClass();
javaClass.dump("HelloWorld.class");
```

There is no custom class loader in the experiment. The artifact is a class file
on disk, which makes it easy to inspect with JVM tools before executing it.

## Runtime behavior and caveats

The module depends on the old `org.apache.bcel:bcel:5.2` artifact. It compiles
under the repository's shared Java configuration, but the generated class is an
artifact of running `com.company.Main`, not an output of Gradle compilation.
Because `javaClass.dump("HelloWorld.class")` writes to the process working
directory, repeated runs overwrite the same file.

The generated bytecode is intentionally simple. It does not preserve source
line numbers, it treats `IOException` by returning silently, and it assumes the
reader construction and `readLine` call are the only interesting failures.
There is no class loading step, no verification step in the program itself, and
no automated assertion that the generated `HelloWorld.class` behaves as
expected. Use `javap -c -v HelloWorld.class` to inspect the exact constant-pool
entries and instruction sequence.

Because the generator writes a class with only the members it explicitly adds,
the shape of the class can differ from what a modern Java compiler would emit
for equivalent source. For example, the greeting uses `StringBuffer` rather than
`invokedynamic` string concatenation, and the exception handler is broader than
a source-level reader-only catch might be. Those differences are useful when
reading the output: they remind you that BCEL is not imitating `javac`; it is
building exactly the classfile you ask it to build.

The experiment is still useful because it exposes real constraints: stack
order, descriptor accuracy, local-variable slots, branch targets, and exception
table boundaries. Any larger bytecode generator has to solve the same problems,
usually with additional layers for symbol management and diagnostics.

## Suggested next experiments

Generate a second method and call it from `main`, then inspect how BCEL records
the owner, name, and descriptor for that call. Add source line metadata and
compare debugger behavior with the current file. Replace the silent
`IOException` handler with generated code that prints the exception message.
Finally, read an existing class, modify one method, and write it back out; that
would move the experiment from class generation into bytecode transformation.
