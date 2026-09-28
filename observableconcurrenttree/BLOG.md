# ObservableConcurrentTree: a Java 27 concurrent tree optimized for read-heavy workloads

## Why this experiment exists

`ObservableConcurrentTree` is a mutable rooted tree that combines several concerns that are easy to implement separately and harder to keep coherent together:

- concurrent readers and writers,
- stable insertion order for children,
- subtree moves and removals,
- immutable read views,
- observers notified after a committed mutation,
- event-driven mutation,
- atomic full-state replacement,
- snapshots and depth-first traversal,
- JSON persistence.

The first reconstructed implementation deliberately favored obvious correctness. Every structural operation used a single `ReentrantReadWriteLock`, nodes referred to parents and children by id, children lived in a `LinkedHashSet`, every read rebuilt immutable `Entry` objects, traversals repeatedly looked nodes up in a `HashMap`, and even unobserved mutations allocated `Change` objects.

That version was useful as a reference implementation, but those choices leave substantial performance on the table. This revision keeps the public behavior while changing the internal representation around the actual hot paths.

The experiment now targets Java 27 and Gradle 9.8.0. Java 27 is the current Java SE feature release; Java 25 remains the current LTS release. Gradle 9.8 added official Java 27 runtime and toolchain support.

## Running it

The example program still demonstrates the full public surface:

```bash
./gradlew :observableconcurrenttree:runExperiment
```

Deterministic verification is wired into the module's `check` task and can also be run directly:

```bash
./gradlew :observableconcurrenttree:verifyExperiment
```

The exploratory benchmark harness is separate:

```bash
./gradlew :observableconcurrenttree:benchmarkExperiment
```

It accepts optional node-count and repetition arguments when invoked directly as a Java main class. It intentionally has no JMH dependency because this repository treats experiments as small self-contained probes. The results below should therefore be read as comparative engineering measurements, not publication-grade microbenchmark numbers.

## Java 27 build policy

Only this experiment is forced to Java 27. Historical modules retain their existing lower bytecode targets.

```groovy
project(':observableconcurrenttree') {
    java {
        toolchain {
            languageVersion = JavaLanguageVersion.of(27)
        }
        sourceCompatibility = JavaVersion.VERSION_27
        targetCompatibility = JavaVersion.VERSION_27
    }

    tasks.withType(JavaCompile).configureEach {
        options.release = 27
        options.compilerArgs += ['-Xlint:all', '-Werror']
    }
}
```

The explicit toolchain keeps the runtime choice local to this module. `-Xlint:all -Werror` turns compiler warnings into build failures so serialization, raw-type, and other accidental regressions do not silently accumulate.

The Gradle wrapper is 9.8.0 because earlier wrapper versions did not officially support running on Java 27.

## Core representation: ids at the boundary, object references inside

The most important change is the internal graph.

The original representation stored a parent id and child ids:

```java
private K parentId;
private final LinkedHashSet<K> children;
```

That is convenient for serialization and debugging, but every traversal step requires another map lookup. A DFS over `n` nodes performs repeated hashing even though the tree already knows the exact adjacent nodes.

The optimized node stores direct references:

```java
private final K id;
private Node<K, V> parent;
private V value;
private final ArrayList<Node<K, V>> children = new ArrayList<>();
```

The public API still speaks in ids. Internally, topology navigation is pointer chasing rather than map lookup.

The `HashMap<K, Node<K,V>>` remains because point lookup by id is still a primary operation. The map is now an index over the graph rather than the graph itself.

## Why ArrayList replaced LinkedHashSet

A tree invariant already guarantees that each node has exactly one parent. The implementation also rejects duplicate node ids globally before a node is linked. Under those constraints, a per-parent `Set` is doing duplicate-detection work that the global map has already done.

```java
if (nodes.containsKey(nodeId)) {
    throw new IllegalArgumentException("Node already exists: " + nodeId);
}
```

An `ArrayList<Node<K,V>>` gives a substantially denser child representation, cheap append, cache-friendly indexed reverse traversal, and no hash-table allocation for every non-leaf node.

There is a trade-off: removing or moving one child from a very wide parent is linear in that parent's child count rather than expected constant time. The workload measured here is read-heavy and traversal-heavy, so the memory locality and traversal savings dominate. A workload dominated by random moves among parents with millions of direct children would deserve a different representation.

## Cached immutable Entry objects

The public `Entry` view is immutable. That makes it safe to cache.

```java
private transient volatile Entry<K, V> entry;
```

A node's external representation changes only when one of three things happens:

- its value changes,
- its parent changes,
- its direct child list changes.

Writers invalidate the affected cache entries while holding the write lock.

```java
node.value = value;
node.entry = null;
```

A move invalidates the moved node plus both parents:

```java
oldParent.children.remove(node);
oldParent.entry = null;
newParent.children.add(node);
newParent.entry = null;
node.parent = newParent;
node.entry = null;
```

Reads reuse the cached object:

```java
var cached = node.entry;
if (cached != null) {
    return cached;
}
```

The first read after a relevant mutation constructs the immutable child-id list and `Entry`; subsequent reads, snapshots, and traversals reuse it.

This removes the largest allocation source in the original `snapshot()` and `depthFirst()` paths.

## Traversal without per-node temporary lists

The original traversal copied every node's child ids to a temporary `ArrayList` before pushing them onto the stack in reverse order.

The optimized representation already has random-access children, so it pushes direct references in reverse index order:

```java
private static <K, V> void pushChildrenReverse(
        Node<K, V> node,
        ArrayDeque<Node<K, V>> stack) {
    var children = node.children;
    for (int i = children.size() - 1; i >= 0; i--) {
        stack.push(children.get(i));
    }
}
```

Insertion order therefore remains observable while traversal avoids one temporary collection per visited node.

The same helper is shared by DFS, snapshots, subtree-id collection, subtree removal, and reachability validation.

## Lock-free scalar reads

`rootId`, `version`, and `size` are maintained as volatile scalar state:

```java
private volatile K rootId;
private volatile long version;
private volatile int size;
```

Those values are committed under the tree's write lock, but reading a single scalar does not require acquiring the read lock.

```java
public K getRootId() {
    return rootId;
}

public long getVersion() {
    return version;
}

public int size() {
    return size;
}
```

This is deliberately narrow. Operations that dereference the mutable node graph still acquire the read lock. The implementation does not pretend a `HashMap` or `ArrayList` is safe for lock-free structural reads.

This distinction accounts for the large scalar-read benchmark improvement without weakening tree consistency.

## One structural lock, cached Lock handles

A single `ReentrantReadWriteLock` still guards graph consistency.

```java
private transient ReentrantReadWriteLock treeLock;
private transient Lock readLock;
private transient Lock writeLock;
```

The read and write lock handles are cached during initialization rather than repeatedly obtained from the enclosing lock object.

A more complicated design could use a `StampedLock`, striped locks, or per-node locking. Those alternatives can improve particular workloads, but they make atomic subtree moves, state replacement, observer semantics, and serialization materially harder to reason about. This version optimizes representation and allocation first, where the measured gains are larger and the consistency model remains straightforward.

## Observers: copy-on-write without CopyOnWriteArrayList overhead

Observer registration is expected to be rare compared with mutations. Notification must iterate a stable listener set without holding the tree lock.

The implementation therefore stores a volatile array:

```java
private transient volatile Observer<K, V>[] observers;
```

Registration and removal synchronize only on a dedicated observer mutex and publish a new array. Mutation paths capture the current array while holding the tree write lock.

Most importantly, an unobserved mutation no longer allocates a `Change` object:

```java
listeners = observers;
if (listeners.length != 0) {
    change = new Change<>(...);
}
```

Notification still occurs after the structural lock is released. If an observer throws, the mutation remains committed, all remaining observers are still called, and the first runtime failure is rethrown afterward.

This preserves the original post-commit observer contract while reducing the common no-observer mutation path.

## Atomic state replacement

Bulk loading builds an entirely new graph before touching live state.

```java
var rebuilt = rebuild(state);
```

The rebuild pre-sizes its map:

```java
var rebuilt = HashMap.<K, Node<K, V>>newHashMap(state.size());
```

It validates duplicate ids, roots, missing parents, self-parenting, and disconnected/cyclic topology. Because every non-root node has exactly one parent, any cycle must be disconnected from the unique root. Counting nodes reachable from the root is therefore sufficient to reject cycles and disconnected components without allocating a separate visited `HashSet`.

Once validation succeeds, the write-side commit is a reference swap:

```java
nodes = rebuilt.nodes();
rootId = root == null ? null : root.id;
size = nodes.size();
var newVersion = ++version;
```

Readers see either the old graph or the new graph while holding the structural lock. The write-lock hold time no longer includes rebuilding or copying every new node into the live map.

## Fast clear and root removal

`clear()` is now O(1) when there are no observers. The only reason to traverse the old graph during clear is to populate `affectedNodeIds` for a `Change` event.

```java
List<K> removed = listeners.length == 0
        ? List.of()
        : collectSubtreeIds(nodes.get(oldRootId));
```

Root-subtree removal still has to produce the public list of removed ids, but it can clear the node index in one operation after collecting that list rather than deleting each map entry individually.

## Records and sealed events

The externally visible state carriers are now Java records rather than boilerplate classes.

```java
public record NodeState<K, V>(K id, K parentId, V value) {
    public NodeState {
        Objects.requireNonNull(id, "id");
    }
}

public record Entry<K, V>(K id, K parentId, V value, List<K> children) {
    public Entry {
        Objects.requireNonNull(id, "id");
        children = List.copyOf(children);
    }
}
```

`Snapshot` and `Change` follow the same pattern. The compact constructors preserve the invariants that previously lived in private constructors while record components provide the canonical `id()`, `value()`, `entries()`, and similar accessors.

Events are a sealed algebraic data type instead of an enum plus one container with nullable fields.

```java
public sealed interface TreeEvent<K, V> permits Add, Update, Move, Remove {
    K nodeId();
}

public record Add<K, V>(K parentId, K nodeId, V value)
        implements TreeEvent<K, V> {}

public record Update<K, V>(K nodeId, V value)
        implements TreeEvent<K, V> {}

public record Move<K, V>(K nodeId, K parentId)
        implements TreeEvent<K, V> {}

public record Remove<K, V>(K nodeId)
        implements TreeEvent<K, V> {}
```

Each event can now only contain fields valid for that operation. `apply` uses an exhaustive record-pattern switch, so adding another permitted event makes the compiler force the dispatch code to be updated.

```java
switch (event) {
    case Add<K, V>(var parentId, var nodeId, var value) ->
            add(parentId, nodeId, value);
    case Update<K, V>(var nodeId, var value) ->
            update(nodeId, value);
    case Move<K, V>(var nodeId, var parentId) ->
            move(nodeId, parentId);
    case Remove<K, V>(var nodeId) ->
            removeSubtree(nodeId);
}
```

## Generic model without Serializable bounds

The tree is now a normal generic data structure:

```java
public final class ObservableConcurrentTree<K, V>
```

Neither keys nor values need to implement `Serializable`. Persistence is no longer part of the generic type contract, so callers can store records, immutable domain objects, or other application types without inheriting a marker interface solely for the tree.

This also removes Java native object serialization from the internal node graph. The in-memory representation is free to evolve without coupling persistence compatibility to private implementation fields.

## JSON persistence

Persistence is explicit JSON state built from the public `NodeState` records. The tree graph itself, locks, observer registrations, and cached `Entry` objects are not serialized.

```java
var json = tree.toJson();

var restored = ObservableConcurrentTree.fromJson(
        json,
        String.class,
        MyValue.class);
```

The serialized document contains the tree version and a depth-first list of node states. Deserialization rebuilds and validates a fresh graph before restoring the saved version.

Jackson is used as the JSON codec. Convenience methods use a shared default `ObjectMapper`, while overloads accept a caller-supplied mapper for applications that need custom modules or configuration.

```java
public String toJson(ObjectMapper mapper) throws JsonProcessingException

public static <K, V> ObservableConcurrentTree<K, V> fromJson(
        ObjectMapper mapper,
        String json,
        Class<K> keyType,
        Class<V> valueType)
```

The default API deliberately requires explicit key and value classes on deserialization. Jackson polymorphic default typing is not enabled, so the JSON format does not embed arbitrary runtime class metadata.

## Java 27 example code

The concurrency example now uses virtual threads:

```java
try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
    futures.add(executor.submit(() -> {
        await(start);
        for (int i = 0; i < 5; i++) {
            tree.add("root", "writer-" + writerId + "-node-" + i, "value-" + writerId + "-" + i);
        }
    }));
}
```

The code also uses modern type inference, lambdas, method references, switch expressions/statements with arrow labels, pattern matching for `instanceof`, `List.of`, `@Serial`, and `Objects.requireNonNull`.

None of those syntax changes is counted as a performance optimization. They are separate from the representation and allocation changes.

## Verification

`ObservableConcurrentTreeVerification` is dependency-free and runs as part of this module's Gradle `check` task. It covers:

- add, update, move, no-op move, subtree removal,
- parent, snapshot, and DFS correctness,
- observer registration and removal,
- observer failure after commit,
- valid and invalid bulk state loading,
- JSON serialization/deserialization,
- lock reinitialization after deserialization,
- concurrent readers and two concurrent writers using virtual threads.

This is intentionally separate from the human-readable example program.

## Exploratory benchmark results

The benchmark below compares the implementation that was on `main` before this optimization with the final optimized implementation.

Environment for these numbers:

- OpenJDK 21.0.11,
- Linux container,
- `-Xms1g -Xmx1g`,
- 25,000 direct children under one root,
- two warm-up passes per process,
- five measured process runs,
- median reported,
- identical `TreeBench` workload for baseline and optimized code.

These are comparative local measurements, not JMH results and not Java 27 numbers.

| Operation | Baseline median | Optimized median | Speedup |
| --- | ---: | ---: | ---: |
| build 25k nodes | 3.716 ms | 2.718 ms | 1.37× |
| 5M scalar read loops | 91.555 ms | 4.299 ms | 21.30× |
| 40 × getChildren(root) | 26.472 ms | 4.318 ms | 6.13× |
| 40 × depthFirst() | 42.989 ms | 10.177 ms | 4.22× |
| 40 × snapshot() | 46.085 ms | 10.299 ms | 4.47× |
| 20 × loadState(25k) | 56.471 ms | 21.569 ms | 2.62× |
| build + remove root subtree | 5.575 ms | 4.092 ms | 1.36× |

The largest scalar gain comes from removing lock acquisition from `size()`, `getVersion()`, and `getRootId()`. The largest collection-read gains come from cached immutable entries, direct node references, and eliminating temporary per-node child-list copies.

A separate observed-mutation probe with one observer also improved rather than regressed, because listener snapshotting is a volatile array read and observer registration is off the mutation hot path.

## What this optimization does not claim

It does not make arbitrary user-supplied keys or values thread-safe. Mutable keys remain unsafe because mutating fields involved in `equals` or `hashCode` after insertion breaks any normal hash-based index.

It does not make a multi-call read transaction atomic. For example, calling `size()` and then `snapshot()` can observe different committed versions if a writer runs between those calls. Use one `Snapshot` when a coherent compound read is required.

It does not guarantee superior performance for every topology. An `ArrayList` is intentionally chosen for compact ordered children. Extremely wide parents with move-heavy workloads may prefer a linked/indexed structure despite its higher memory cost.

It does not replace JMH. The included benchmark is a reproducible experiment harness that makes large regressions visible and supports quick iteration without external dependencies.

## Further experiments

The next useful optimization would be workload-driven rather than automatic:

- compare `ReentrantReadWriteLock` with `StampedLock` under controlled read/write ratios,
- measure deep-chain trees separately from wide trees,
- quantify retained memory per node with JOL or a profiler,
- add a move-heavy benchmark for very wide parents,
- consider an explicit stable JSON schema/versioning strategy instead of Java native serialization,
- evaluate immutable/persistent snapshots when snapshot frequency dominates all other operations.

Those changes should only be accepted with a benchmark demonstrating that the additional complexity pays for itself.
