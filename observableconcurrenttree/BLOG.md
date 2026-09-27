# Observable Concurrent Trees with Snapshot Reads and Change Events

## Why this experiment exists

This experiment reconstructs an `ObservableConcurrentTree` abstraction from an older architecture note. The interesting combination is not any one feature in isolation. The class combines a mutable rooted tree, concurrent access, immutable read views, observer notifications, bulk state replacement, event application, subtree movement, and Java serialization in one small implementation.

The design deliberately uses ordinary JDK concurrency primitives rather than a specialized concurrent collection. A `ReentrantReadWriteLock` protects structural state, while a `CopyOnWriteArrayList` stores observers. Mutations happen under the write lock, reads happen under the read lock, and notifications are emitted only after the structural lock has been released.

For experienced Java developers, the useful questions are about boundaries: what state is protected by the tree lock, what is copied before returning to callers, which behavior is atomic, and what happens when observer callbacks fail after a mutation has already committed.

## Running the complete example

The module includes `ObservableConcurrentTreeExample`, a runnable walkthrough of the complete public API. It can be launched with:

```bash
./gradlew :observableconcurrenttree:runExperiment
```

The example intentionally uses small string ids and values so the topology changes are easy to follow in console output. It covers both construction styles:

```java
ObservableConcurrentTree<String, String> emptyTree =
        new ObservableConcurrentTree<String, String>();
emptyTree.initialize("root", "Root");

ObservableConcurrentTree<String, String> initializedTree =
        new ObservableConcurrentTree<String, String>("root", "Root from constructor");
```

It registers an observer, performs direct mutations, then removes the observer again:

```java
ObservableConcurrentTree.Observer<String, String> observer =
        change -> System.out.println(change.getType() + " " + change.getNodeId());

tree.addObserver(observer);
tree.add("root", "dc-1", "Datacenter 1");
tree.update("dc-1", "Datacenter 1 updated");
tree.removeObserver(observer);
```

The runnable class exercises all read views as well: `contains`, `get`, `getRootId`, `getRoot`, `getParentId`, `getChildren`, `depthFirst`, `snapshot`, `getVersion`, `size`, and `isEmpty`.

Event-driven usage is shown separately so it is clear that the event API is only another entrypoint into the same mutation logic:

```java
tree.apply(ObservableConcurrentTree.TreeEvent.add("root", "parent", "Parent"));
tree.apply(ObservableConcurrentTree.TreeEvent.add("parent", "child", "Child"));
tree.apply(ObservableConcurrentTree.TreeEvent.update("child", "Updated"));
tree.apply(ObservableConcurrentTree.TreeEvent.move("child", "root"));
tree.apply(ObservableConcurrentTree.TreeEvent.remove("parent"));
```

The example also demonstrates atomic full-state loading:

```java
tree.loadState(Arrays.asList(
        new ObservableConcurrentTree.NodeState<String, String>("root", null, "Root"),
        new ObservableConcurrentTree.NodeState<String, String>("a", "root", "A"),
        new ObservableConcurrentTree.NodeState<String, String>("leaf", "a", "Leaf")));
```

Serialization is exercised as an actual round-trip:

```java
byte[] bytes = tree.toByteArray();
ObservableConcurrentTree<String, String> restored =
        ObservableConcurrentTree.fromByteArray(bytes);
```

Finally, the program deliberately attempts representative invalid operations: duplicate ids, missing parents, moving the root, self-parenting, introducing a cycle, initializing twice, and loading multiple roots. Those examples document the class's rejection behavior alongside the successful paths rather than leaving validation semantics implicit.

## Data model

The tree is generic in both node id and value:

```java
public final class ObservableConcurrentTree<K extends Serializable, V extends Serializable>
        implements Serializable {
```

Both generic types are required to be `Serializable` because the tree supports round-trip Java object serialization. Internally, nodes are mutable and never exposed directly:

```java
private static final class Node<K extends Serializable, V extends Serializable> implements Serializable {
    private final K id;
    private K parentId;
    private V value;
    private final LinkedHashSet<K> children;
}
```

The `LinkedHashSet` prevents duplicate child ids while preserving insertion order. Externally, callers receive immutable `Entry` snapshots instead of references to internal nodes. This separation is important because returning the internal child set would let callers mutate the structure without acquiring the tree lock.

The central state is intentionally small:

```java
private final Map<K, Node<K, V>> nodes = new HashMap<K, Node<K, V>>();
private K rootId;
private long version;
private transient ReentrantReadWriteLock lock;
private transient CopyOnWriteArrayList<Observer<K, V>> observers;
```

`nodes`, `rootId`, and `version` are serialized. The lock and observer list are transient because neither represents persistent tree data.

## Concurrency model

Every structural mutation acquires the write lock. `add`, for example, validates both ids, resolves the parent, links the new node into the parent, increments the version, and builds the corresponding change event while the write lock is still held.

```java
lock.writeLock().lock();
try {
    Node<K, V> parent = nodes.get(parentId);
    Node<K, V> node = new Node<K, V>(nodeId, parentId, value);
    nodes.put(nodeId, node);
    parent.children.add(nodeId);
    long newVersion = ++version;
} finally {
    lock.writeLock().unlock();
}
```

Read operations use the read lock. Multiple readers can therefore proceed concurrently while still seeing a structurally consistent state.

```java
lock.readLock().lock();
try {
    Node<K, V> node = nodes.get(nodeId);
    return node == null ? null : toEntry(node);
} finally {
    lock.readLock().unlock();
}
```

The returned `Entry` copies the child ids into a new list and wraps the copy with `Collections.unmodifiableList`. A caller never observes the mutable set stored inside a node.

The class does not attempt lock-free updates. Its goal is a clear consistency model for a tree whose updates may touch multiple objects at once. Moving one node requires changing the old parent, the new parent, and the node's own parent id. A single write lock makes that compound mutation atomic relative to other tree operations.

## Observable mutations

Observers implement one callback:

```java
public interface Observer<K extends Serializable, V extends Serializable> {
    void onChange(Change<K, V> change);
}
```

Each mutation creates a `Change` containing a type, version, affected node, old and new parents, old and new values, and a list of affected node ids where appropriate.

The observer collection is a `CopyOnWriteArrayList`:

```java
private transient CopyOnWriteArrayList<Observer<K, V>> observers;
```

That choice favors cheap, stable iteration during notifications over cheap listener registration. It fits a common observer workload where subscriptions are relatively rare and events are more frequent.

Notifications happen after the tree lock is released:

```java
notifyObservers(change);
```

This avoids invoking arbitrary user code while holding the structural lock. An observer is therefore free to call back into read methods without deadlocking on the same mutation boundary.

There is an important semantic consequence. The mutation has already committed before observer callbacks run. If an observer throws, `notifyObservers` records the first runtime failure, continues notifying the remaining observers, and rethrows the first failure afterward. Callers must not interpret an observer exception as a transaction rollback.

## Adding, updating, moving, and removing

Adding a node requires an existing parent and a previously unused id:

```java
if (nodes.containsKey(nodeId)) {
    throw new IllegalArgumentException("Node already exists: " + nodeId);
}
Node<K, V> parent = nodes.get(parentId);
if (parent == null) {
    throw new IllegalArgumentException("Parent does not exist: " + parentId);
}
```

Updating replaces only the stored value. The node's topology does not change, but the version still advances and observers receive an `UPDATED` event.

Moving a subtree has stricter invariants. The root cannot be moved, a node cannot become its own parent, and the proposed parent cannot be below the node being moved.

```java
if (node.parentId == null) {
    throw new IllegalArgumentException("Root node cannot be moved");
}
if (nodeId.equals(newParentId)) {
    throw new IllegalArgumentException("Node cannot be its own parent");
}
ensureNotDescendant(nodeId, newParentId);
```

Cycle rejection walks upward from the candidate parent through parent links until it reaches the root. If the node being moved appears on that path, the move would introduce a cycle and is rejected.

Removing a node removes its complete subtree. The implementation first performs an iterative depth-first collection of ids, unlinks the subtree root from its parent, and then removes every collected node from the index.

```java
removed = collectSubtreeIds(nodeId);
for (K id : removed) {
    nodes.remove(id);
}
```

Using an iterative traversal avoids consuming the Java call stack for deep trees.

## Bulk state loading

`loadState` accepts flat `NodeState` records containing `id`, `parentId`, and `value`. Rebuilding happens before the live tree acquires its write lock.

```java
RebuiltTree<K, V> rebuilt = rebuild(state);
```

The rebuild validates the complete candidate state first. It rejects duplicate ids, multiple roots, missing roots, self-parenting nodes, missing parents, cycles, and disconnected components.

Only after the candidate has passed validation does the method acquire the write lock and replace the live state:

```java
nodes.clear();
nodes.putAll(rebuilt.nodes);
rootId = rebuilt.rootId;
long newVersion = ++version;
```

This makes replacement atomic from the perspective of concurrent tree readers. They see either the previous state or the fully rebuilt new state, not a partially assembled tree.

Validation outside the lock also reduces write-lock hold time for large imports. The trade-off is temporary memory usage because the old tree and rebuilt tree coexist until the swap completes.

## Event application

The nested `TreeEvent` type is a small command representation for four mutations:

```java
public enum EventType {
    ADD,
    UPDATE,
    MOVE,
    REMOVE
}
```

Factory methods construct valid event shapes, and `apply` routes each event to the normal public mutation method.

```java
switch (event.getType()) {
    case ADD:
        add(event.getParentId(), event.getNodeId(), event.getValue());
        return;
    case UPDATE:
        update(event.getNodeId(), event.getValue());
        return;
    case MOVE:
        move(event.getNodeId(), event.getParentId());
        return;
    case REMOVE:
        removeSubtree(event.getNodeId());
        return;
}
```

The event layer therefore does not bypass validation, versioning, locking, or observer delivery. There is one mutation path rather than a separate implementation for event-driven updates.

## Snapshots and traversal

`depthFirst` returns an immutable list of copied `Entry` objects in depth-first order. Children are pushed onto an explicit stack in reverse order so iteration preserves the insertion ordering held by each `LinkedHashSet`.

`snapshot` captures the same traversal plus the root id and current version:

```java
return new Snapshot<K, V>(rootId, version, entries);
```

Because the complete snapshot is constructed while holding the read lock, the entries and version refer to one consistent structural state. Once returned, the snapshot is detached from future tree changes.

This is useful when a consumer needs to perform expensive processing without holding a tree lock. It can take a snapshot quickly and work from the immutable copy afterward.

## Serialization behavior

The tree can serialize itself to a byte array using normal Java object serialization:

```java
ObjectOutputStream out = new ObjectOutputStream(bytes);
out.writeObject(this);
```

The custom `writeObject` acquires the read lock around `defaultWriteObject` so serialization cannot race with a structural mutation.

```java
lock.readLock().lock();
try {
    out.defaultWriteObject();
} finally {
    lock.readLock().unlock();
}
```

On deserialization, `readObject` recreates transient concurrency state:

```java
in.defaultReadObject();
initializeTransients();
```

Observers are intentionally not serialized. A restored tree starts with a fresh empty observer list. That avoids serializing callback object graphs and makes observer registration a runtime concern rather than persisted state.

## Version semantics

`version` increments once for every successful state-changing operation: initialization, add, update, move, subtree removal, clear, and state load.

A no-op move to the node's current parent returns without incrementing the version or notifying observers. Calling `clear` on an already empty tree behaves the same way.

The version is not a globally unique revision id. It is a monotonic counter inside one serialized tree lineage. A deserialized instance continues with the serialized counter value.

## Runtime behavior and caveats

This module is a library-style experiment and has no external I/O, networking, persistence service, or background executor. The class is safe to instantiate and exercise in-process.

The strongest guarantee is structural consistency for operations performed through this class. It does not make the objects stored as keys or values thread-safe. Mutable `K` instances are especially dangerous because changing fields that participate in `equals` or `hashCode` after insertion can corrupt normal `HashMap` lookup behavior.

Observer callbacks are synchronous with the calling thread after the tree mutation is unlocked. A slow observer therefore increases mutation latency even though it does not hold the tree lock. If observers require isolation, a future experiment could publish changes through an executor or `Flow.Publisher` instead.

Observer failures are also post-commit failures. If a callback throws, the mutation remains visible. Production code would normally document this explicitly or separate mutation results from notification delivery errors.

Java native serialization is included because the reconstructed design called for serializable state, not because native serialization is recommended as a long-term wire format. A production system would normally use an explicit versioned format.

Finally, this implementation has one global read-write lock. That keeps invariants simple but means unrelated branches cannot be mutated concurrently. A higher-throughput variant could explore immutable persistent nodes, striped locks, copy-on-write roots, or transactional path locking.

## Suggested next experiments

Add deterministic multithreaded tests that coordinate readers and writers with barriers rather than sleeps. Measure how the global write lock behaves for wide trees and deeply nested trees. Compare `CopyOnWriteArrayList` observers with an asynchronous event queue. Add optimistic reads using `StampedLock` and determine whether the added complexity improves real read-heavy workloads.

Another useful extension is conditional mutation by expected version. A method such as `updateIfVersion` could provide a small optimistic-concurrency primitive for clients that build a change from a prior snapshot.

A final direction would replace Java serialization with a stable tree snapshot format and test schema migration. That would separate persisted topology from implementation details such as internal node classes and collection choices.
