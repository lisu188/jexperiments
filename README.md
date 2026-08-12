# JVM Experiments

A long-running laboratory for small, focused **Java and Kotlin/JVM experiments**. The repository is intentionally a collection of isolated probes rather than a single application: each module exists to answer a concrete implementation, concurrency, bytecode, language-runtime or framework question with executable code.

## Current themes

### JVM and Kotlin internals

- Java/Kotlin bridge generation
- Kotlin default arguments, delegation and property references
- SAM/fun-interface behavior
- sealed `when` compilation
- suspend-function state machines
- value-class boxing
- inline capture behavior
- Kotlin metadata inspection
- Java lambda bytecode
- Apache BCEL experiments

### Concurrency and execution

- `CompletableFuture`
- ordered and distributed thread-pool experiments
- `Flow.Publisher`

### Algorithms and simulation

- HashLife
- evolutionary experiments
- small neural-network experiments

### Framework and platform probes

- Spring Boot experiment blog site
- SOAP / embedded Tomcat experiments
- Raspberry Pi controller code
- Slick2D/game experiments

The Gradle build currently includes more than twenty independent modules; see `settings.gradle` for the authoritative list.

## Build

```bash
./gradlew build
```

Several Kotlin/JVM bytecode-focused modules expose a `runExperiment` task. For example:

```bash
./gradlew :kotlinvalueclassboxing:runExperiment
./gradlew :kotlinsuspendstate:runExperiment
```

## Why this repository is public

This repository is an **engineering notebook in executable form**. Some modules are historical and intentionally use old APIs; others are recent probes updated to current Java/Kotlin tooling. It should be read as evidence of investigation and learning across the JVM rather than as one production system with a single support lifecycle.

For larger end-to-end projects, see `winrisk`, `fall-of-nouraajd`, `clash-disassembly` and `clife` on my profile.
