import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serial;
import java.io.Serializable;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

public final class ObservableConcurrentTree<K extends Serializable, V extends Serializable> implements Serializable {
    @Serial
    private static final long serialVersionUID = 1L;

    private static final Observer<?, ?>[] NO_OBSERVERS = new Observer<?, ?>[0];

    public enum ChangeType {
        INITIALIZED,
        ADDED,
        UPDATED,
        MOVED,
        REMOVED,
        CLEARED,
        STATE_LOADED
    }

    public enum EventType {
        ADD,
        UPDATE,
        MOVE,
        REMOVE
    }

    @FunctionalInterface
    public interface Observer<K extends Serializable, V extends Serializable> {
        void onChange(Change<K, V> change);
    }

    public static final class NodeState<K extends Serializable, V extends Serializable> implements Serializable {
        @Serial
        private static final long serialVersionUID = 1L;

        private final K id;
        private final K parentId;
        private final V value;

        public NodeState(K id, K parentId, V value) {
            this.id = Objects.requireNonNull(id, "id");
            this.parentId = parentId;
            this.value = value;
        }

        public K getId() {
            return id;
        }

        public K getParentId() {
            return parentId;
        }

        public V getValue() {
            return value;
        }
    }

    @SuppressWarnings("serial")
    public static final class Entry<K extends Serializable, V extends Serializable> implements Serializable {
        @Serial
        private static final long serialVersionUID = 1L;

        private final K id;
        private final K parentId;
        private final V value;
        private final List<K> children;

        private Entry(K id, K parentId, V value, List<K> children) {
            this.id = id;
            this.parentId = parentId;
            this.value = value;
            this.children = children;
        }

        public K getId() {
            return id;
        }

        public K getParentId() {
            return parentId;
        }

        public V getValue() {
            return value;
        }

        public List<K> getChildren() {
            return children;
        }
    }

    @SuppressWarnings("serial")
    public static final class Snapshot<K extends Serializable, V extends Serializable> implements Serializable {
        @Serial
        private static final long serialVersionUID = 1L;

        private final K rootId;
        private final long version;
        private final List<Entry<K, V>> entries;

        private Snapshot(K rootId, long version, List<Entry<K, V>> entries) {
            this.rootId = rootId;
            this.version = version;
            this.entries = entries;
        }

        public K getRootId() {
            return rootId;
        }

        public long getVersion() {
            return version;
        }

        public List<Entry<K, V>> getEntries() {
            return entries;
        }
    }

    @SuppressWarnings("serial")
    public static final class Change<K extends Serializable, V extends Serializable> implements Serializable {
        @Serial
        private static final long serialVersionUID = 1L;

        private final ChangeType type;
        private final long version;
        private final K nodeId;
        private final K oldParentId;
        private final K newParentId;
        private final V oldValue;
        private final V newValue;
        private final List<K> affectedNodeIds;

        private Change(
                ChangeType type,
                long version,
                K nodeId,
                K oldParentId,
                K newParentId,
                V oldValue,
                V newValue,
                List<K> affectedNodeIds) {
            this.type = type;
            this.version = version;
            this.nodeId = nodeId;
            this.oldParentId = oldParentId;
            this.newParentId = newParentId;
            this.oldValue = oldValue;
            this.newValue = newValue;
            this.affectedNodeIds = affectedNodeIds;
        }

        public ChangeType getType() {
            return type;
        }

        public long getVersion() {
            return version;
        }

        public K getNodeId() {
            return nodeId;
        }

        public K getOldParentId() {
            return oldParentId;
        }

        public K getNewParentId() {
            return newParentId;
        }

        public V getOldValue() {
            return oldValue;
        }

        public V getNewValue() {
            return newValue;
        }

        public List<K> getAffectedNodeIds() {
            return affectedNodeIds;
        }
    }

    public static final class TreeEvent<K extends Serializable, V extends Serializable> implements Serializable {
        @Serial
        private static final long serialVersionUID = 1L;

        private final EventType type;
        private final K nodeId;
        private final K parentId;
        private final V value;

        private TreeEvent(EventType type, K nodeId, K parentId, V value) {
            this.type = Objects.requireNonNull(type, "type");
            this.nodeId = Objects.requireNonNull(nodeId, "nodeId");
            this.parentId = parentId;
            this.value = value;
        }

        public static <K extends Serializable, V extends Serializable> TreeEvent<K, V> add(K parentId, K nodeId, V value) {
            return new TreeEvent<>(EventType.ADD, nodeId, Objects.requireNonNull(parentId, "parentId"), value);
        }

        public static <K extends Serializable, V extends Serializable> TreeEvent<K, V> update(K nodeId, V value) {
            return new TreeEvent<>(EventType.UPDATE, nodeId, null, value);
        }

        public static <K extends Serializable, V extends Serializable> TreeEvent<K, V> move(K nodeId, K newParentId) {
            return new TreeEvent<>(EventType.MOVE, nodeId, Objects.requireNonNull(newParentId, "newParentId"), null);
        }

        public static <K extends Serializable, V extends Serializable> TreeEvent<K, V> remove(K nodeId) {
            return new TreeEvent<>(EventType.REMOVE, nodeId, null, null);
        }

        public EventType getType() {
            return type;
        }

        public K getNodeId() {
            return nodeId;
        }

        public K getParentId() {
            return parentId;
        }

        public V getValue() {
            return value;
        }
    }

    private static final class Node<K extends Serializable, V extends Serializable> implements Serializable {
        @Serial
        private static final long serialVersionUID = 1L;

        private final K id;
        private Node<K, V> parent;
        private V value;
        private final ArrayList<Node<K, V>> children = new ArrayList<>();
        private transient volatile List<K> childIds;

        private Node(K id, V value) {
            this.id = id;
            this.value = value;
        }
    }

    private record RebuiltTree<K extends Serializable, V extends Serializable>(
            HashMap<K, Node<K, V>> nodes,
            Node<K, V> root) {
    }

    private HashMap<K, Node<K, V>> nodes = new HashMap<>();
    private volatile K rootId;
    private volatile long version;
    private volatile int size;

    private transient ReentrantReadWriteLock treeLock;
    private transient Lock readLock;
    private transient Lock writeLock;
    private transient Object observerLock;
    private transient volatile Observer<K, V>[] observers;

    public ObservableConcurrentTree() {
        initializeTransients();
    }

    public ObservableConcurrentTree(K rootId, V rootValue) {
        initializeTransients();
        initialize(rootId, rootValue);
    }

    public void addObserver(Observer<K, V> observer) {
        Objects.requireNonNull(observer, "observer");
        synchronized (observerLock) {
            var current = observers;
            for (var existing : current) {
                if (Objects.equals(existing, observer)) {
                    return;
                }
            }
            var updated = Arrays.copyOf(current, current.length + 1);
            updated[current.length] = observer;
            observers = updated;
        }
    }

    public void removeObserver(Observer<K, V> observer) {
        if (observer == null) {
            return;
        }
        synchronized (observerLock) {
            var current = observers;
            for (int i = 0; i < current.length; i++) {
                if (!Objects.equals(current[i], observer)) {
                    continue;
                }
                if (current.length == 1) {
                    observers = emptyObservers();
                    return;
                }
                var updated = Arrays.copyOf(current, current.length - 1);
                System.arraycopy(current, i + 1, updated, i, current.length - i - 1);
                observers = updated;
                return;
            }
        }
    }

    public void initialize(K newRootId, V value) {
        Objects.requireNonNull(newRootId, "rootId");

        Observer<K, V>[] listeners;
        Change<K, V> change = null;
        writeLock.lock();
        try {
            if (!nodes.isEmpty()) {
                throw new IllegalStateException("Tree is already initialized");
            }
            var root = new Node<K, V>(newRootId, value);
            nodes.put(newRootId, root);
            rootId = newRootId;
            size = 1;
            var newVersion = ++version;
            listeners = observers;
            if (listeners.length != 0) {
                change = new Change<>(
                        ChangeType.INITIALIZED,
                        newVersion,
                        newRootId,
                        null,
                        null,
                        null,
                        value,
                        List.of(newRootId));
            }
        } finally {
            writeLock.unlock();
        }
        notifyObservers(listeners, change);
    }

    public void add(K parentId, K nodeId, V value) {
        Objects.requireNonNull(parentId, "parentId");
        Objects.requireNonNull(nodeId, "nodeId");

        Observer<K, V>[] listeners;
        Change<K, V> change = null;
        writeLock.lock();
        try {
            if (nodes.containsKey(nodeId)) {
                throw new IllegalArgumentException("Node already exists: " + nodeId);
            }
            var parent = nodes.get(parentId);
            if (parent == null) {
                throw new IllegalArgumentException("Parent does not exist: " + parentId);
            }
            var node = new Node<K, V>(nodeId, value);
            node.parent = parent;
            nodes.put(nodeId, node);
            parent.children.add(node);
            parent.childIds = null;
            size++;
            var newVersion = ++version;
            listeners = observers;
            if (listeners.length != 0) {
                change = new Change<>(
                        ChangeType.ADDED,
                        newVersion,
                        nodeId,
                        null,
                        parentId,
                        null,
                        value,
                        List.of(nodeId));
            }
        } finally {
            writeLock.unlock();
        }
        notifyObservers(listeners, change);
    }

    public void update(K nodeId, V value) {
        Objects.requireNonNull(nodeId, "nodeId");

        Observer<K, V>[] listeners;
        Change<K, V> change = null;
        writeLock.lock();
        try {
            var node = requireNode(nodeId);
            var oldValue = node.value;
            node.value = value;
            var newVersion = ++version;
            listeners = observers;
            if (listeners.length != 0) {
                var parentId = node.parent == null ? null : node.parent.id;
                change = new Change<>(
                        ChangeType.UPDATED,
                        newVersion,
                        nodeId,
                        parentId,
                        parentId,
                        oldValue,
                        value,
                        List.of(nodeId));
            }
        } finally {
            writeLock.unlock();
        }
        notifyObservers(listeners, change);
    }

    public void move(K nodeId, K newParentId) {
        Objects.requireNonNull(nodeId, "nodeId");
        Objects.requireNonNull(newParentId, "newParentId");

        Observer<K, V>[] listeners;
        Change<K, V> change = null;
        writeLock.lock();
        try {
            var node = requireNode(nodeId);
            var newParent = requireNode(newParentId);
            var oldParent = node.parent;
            if (oldParent == null) {
                throw new IllegalArgumentException("Root node cannot be moved");
            }
            if (node == newParent) {
                throw new IllegalArgumentException("Node cannot be its own parent");
            }
            if (oldParent == newParent) {
                return;
            }
            ensureNotDescendant(node, newParent);
            oldParent.children.remove(node);
            oldParent.childIds = null;
            newParent.children.add(node);
            newParent.childIds = null;
            node.parent = newParent;
            var newVersion = ++version;
            listeners = observers;
            if (listeners.length != 0) {
                change = new Change<>(
                        ChangeType.MOVED,
                        newVersion,
                        nodeId,
                        oldParent.id,
                        newParentId,
                        node.value,
                        node.value,
                        List.of(nodeId));
            }
        } finally {
            writeLock.unlock();
        }
        notifyObservers(listeners, change);
    }

    public List<K> removeSubtree(K nodeId) {
        Objects.requireNonNull(nodeId, "nodeId");

        Observer<K, V>[] listeners;
        Change<K, V> change = null;
        List<K> removedView;
        writeLock.lock();
        try {
            var node = requireNode(nodeId);
            var oldParent = node.parent;
            var oldParentId = oldParent == null ? null : oldParent.id;
            var oldValue = node.value;
            if (oldParent != null) {
                oldParent.children.remove(node);
                oldParent.childIds = null;
            }
            var removed = removeSubtreeNodes(node);
            size -= removed.size();
            if (node.id.equals(rootId)) {
                rootId = null;
            }
            removedView = Collections.unmodifiableList(removed);
            var newVersion = ++version;
            listeners = observers;
            if (listeners.length != 0) {
                change = new Change<>(
                        ChangeType.REMOVED,
                        newVersion,
                        nodeId,
                        oldParentId,
                        null,
                        oldValue,
                        null,
                        removedView);
            }
        } finally {
            writeLock.unlock();
        }
        notifyObservers(listeners, change);
        return removedView;
    }

    public void clear() {
        Observer<K, V>[] listeners;
        Change<K, V> change = null;
        writeLock.lock();
        try {
            if (nodes.isEmpty()) {
                return;
            }
            listeners = observers;
            var oldRootId = rootId;
            List<K> removed = listeners.length == 0 ? List.of() : collectSubtreeIds(nodes.get(oldRootId));
            nodes.clear();
            rootId = null;
            size = 0;
            var newVersion = ++version;
            if (listeners.length != 0) {
                change = new Change<>(
                        ChangeType.CLEARED,
                        newVersion,
                        oldRootId,
                        null,
                        null,
                        null,
                        null,
                        Collections.unmodifiableList(removed));
            }
        } finally {
            writeLock.unlock();
        }
        notifyObservers(listeners, change);
    }

    public void loadState(Collection<NodeState<K, V>> state) {
        Objects.requireNonNull(state, "state");
        var rebuilt = rebuild(state);

        Observer<K, V>[] listeners;
        Change<K, V> change = null;
        writeLock.lock();
        try {
            nodes = rebuilt.nodes();
            var root = rebuilt.root();
            rootId = root == null ? null : root.id;
            size = nodes.size();
            var newVersion = ++version;
            listeners = observers;
            if (listeners.length != 0) {
                var affected = root == null ? List.<K>of() : Collections.unmodifiableList(collectSubtreeIds(root));
                change = new Change<>(
                        ChangeType.STATE_LOADED,
                        newVersion,
                        rootId,
                        null,
                        null,
                        null,
                        root == null ? null : root.value,
                        affected);
            }
        } finally {
            writeLock.unlock();
        }
        notifyObservers(listeners, change);
    }

    public void apply(TreeEvent<K, V> event) {
        Objects.requireNonNull(event, "event");
        switch (event.getType()) {
            case ADD -> add(event.getParentId(), event.getNodeId(), event.getValue());
            case UPDATE -> update(event.getNodeId(), event.getValue());
            case MOVE -> move(event.getNodeId(), event.getParentId());
            case REMOVE -> removeSubtree(event.getNodeId());
        }
    }

    public boolean contains(K nodeId) {
        if (nodeId == null) {
            return false;
        }
        readLock.lock();
        try {
            return nodes.containsKey(nodeId);
        } finally {
            readLock.unlock();
        }
    }

    public Entry<K, V> get(K nodeId) {
        Objects.requireNonNull(nodeId, "nodeId");
        readLock.lock();
        try {
            var node = nodes.get(nodeId);
            return node == null ? null : toEntry(node);
        } finally {
            readLock.unlock();
        }
    }

    public K getRootId() {
        return rootId;
    }

    public Entry<K, V> getRoot() {
        readLock.lock();
        try {
            var currentRootId = rootId;
            if (currentRootId == null) {
                return null;
            }
            return toEntry(nodes.get(currentRootId));
        } finally {
            readLock.unlock();
        }
    }

    public K getParentId(K nodeId) {
        Objects.requireNonNull(nodeId, "nodeId");
        readLock.lock();
        try {
            var parent = requireNode(nodeId).parent;
            return parent == null ? null : parent.id;
        } finally {
            readLock.unlock();
        }
    }

    public List<Entry<K, V>> getChildren(K nodeId) {
        Objects.requireNonNull(nodeId, "nodeId");
        readLock.lock();
        try {
            var children = requireNode(nodeId).children;
            if (children.isEmpty()) {
                return List.of();
            }
            var result = new ArrayList<Entry<K, V>>(children.size());
            for (var child : children) {
                result.add(toEntry(child));
            }
            return Collections.unmodifiableList(result);
        } finally {
            readLock.unlock();
        }
    }

    public List<Entry<K, V>> depthFirst() {
        readLock.lock();
        try {
            var currentRootId = rootId;
            if (currentRootId == null) {
                return List.of();
            }
            var result = new ArrayList<Entry<K, V>>(size);
            appendDepthFirst(nodes.get(currentRootId), result);
            return Collections.unmodifiableList(result);
        } finally {
            readLock.unlock();
        }
    }

    public Snapshot<K, V> snapshot() {
        readLock.lock();
        try {
            var entries = new ArrayList<Entry<K, V>>(size);
            var currentRootId = rootId;
            if (currentRootId != null) {
                appendDepthFirst(nodes.get(currentRootId), entries);
            }
            return new Snapshot<>(currentRootId, version, Collections.unmodifiableList(entries));
        } finally {
            readLock.unlock();
        }
    }

    public long getVersion() {
        return version;
    }

    public int size() {
        return size;
    }

    public boolean isEmpty() {
        return size == 0;
    }

    public byte[] toByteArray() throws IOException {
        try (var bytes = new ByteArrayOutputStream(); var out = new ObjectOutputStream(bytes)) {
            out.writeObject(this);
            out.flush();
            return bytes.toByteArray();
        }
    }

    @SuppressWarnings("unchecked")
    public static <K extends Serializable, V extends Serializable> ObservableConcurrentTree<K, V> fromByteArray(byte[] bytes)
            throws IOException, ClassNotFoundException {
        Objects.requireNonNull(bytes, "bytes");
        try (var in = new ObjectInputStream(new ByteArrayInputStream(bytes))) {
            var value = in.readObject();
            if (!(value instanceof ObservableConcurrentTree<?, ?> tree)) {
                throw new IOException("Serialized value is not an ObservableConcurrentTree");
            }
            return (ObservableConcurrentTree<K, V>) tree;
        }
    }

    private void ensureNotDescendant(Node<K, V> node, Node<K, V> candidateParent) {
        for (var current = candidateParent; current != null; current = current.parent) {
            if (current == node) {
                throw new IllegalArgumentException("Move would create a cycle");
            }
        }
    }

    private ArrayList<K> removeSubtreeNodes(Node<K, V> node) {
        var removed = new ArrayList<K>();
        var stack = new ArrayDeque<Node<K, V>>();
        stack.push(node);
        while (!stack.isEmpty()) {
            var current = stack.pop();
            nodes.remove(current.id);
            removed.add(current.id);
            pushChildrenReverse(current, stack);
        }
        return removed;
    }

    private ArrayList<K> collectSubtreeIds(Node<K, V> root) {
        var result = new ArrayList<K>();
        var stack = new ArrayDeque<Node<K, V>>();
        stack.push(root);
        while (!stack.isEmpty()) {
            var current = stack.pop();
            result.add(current.id);
            pushChildrenReverse(current, stack);
        }
        return result;
    }

    private void appendDepthFirst(Node<K, V> root, ArrayList<Entry<K, V>> target) {
        var stack = new ArrayDeque<Node<K, V>>();
        stack.push(root);
        while (!stack.isEmpty()) {
            var current = stack.pop();
            target.add(toEntry(current));
            pushChildrenReverse(current, stack);
        }
    }

    private static <K extends Serializable, V extends Serializable> void pushChildrenReverse(
            Node<K, V> node,
            ArrayDeque<Node<K, V>> stack) {
        var children = node.children;
        for (int i = children.size() - 1; i >= 0; i--) {
            stack.push(children.get(i));
        }
    }

    private Entry<K, V> toEntry(Node<K, V> node) {
        var parent = node.parent;
        return new Entry<>(node.id, parent == null ? null : parent.id, node.value, childIds(node));
    }

    private List<K> childIds(Node<K, V> node) {
        var children = node.children;
        if (children.isEmpty()) {
            return List.of();
        }
        var cached = node.childIds;
        if (cached != null) {
            return cached;
        }
        var ids = new ArrayList<K>(children.size());
        for (var child : children) {
            ids.add(child.id);
        }
        cached = Collections.unmodifiableList(ids);
        node.childIds = cached;
        return cached;
    }

    private Node<K, V> requireNode(K nodeId) {
        var node = nodes.get(nodeId);
        if (node == null) {
            throw new IllegalArgumentException("Node does not exist: " + nodeId);
        }
        return node;
    }

    private RebuiltTree<K, V> rebuild(Collection<NodeState<K, V>> state) {
        if (state.isEmpty()) {
            return new RebuiltTree<>(new HashMap<>(), null);
        }

        var rebuilt = HashMap.<K, Node<K, V>>newHashMap(state.size());
        Node<K, V> root = null;

        for (var item : state) {
            Objects.requireNonNull(item, "state contains null");
            var id = item.getId();
            if (rebuilt.putIfAbsent(id, new Node<>(id, item.getValue())) != null) {
                throw new IllegalArgumentException("Duplicate node id: " + id);
            }
            if (item.getParentId() == null) {
                if (root != null) {
                    throw new IllegalArgumentException("State contains more than one root");
                }
                root = rebuilt.get(id);
            }
        }

        if (root == null) {
            throw new IllegalArgumentException("State has no root");
        }

        for (var item : state) {
            var parentId = item.getParentId();
            if (parentId == null) {
                continue;
            }
            if (item.getId().equals(parentId)) {
                throw new IllegalArgumentException("Node cannot be its own parent: " + item.getId());
            }
            var parent = rebuilt.get(parentId);
            if (parent == null) {
                throw new IllegalArgumentException("Missing parent " + parentId + " for node " + item.getId());
            }
            var node = rebuilt.get(item.getId());
            node.parent = parent;
            parent.children.add(node);
        }

        if (countReachable(root) != rebuilt.size()) {
            throw new IllegalArgumentException("State contains disconnected nodes or a cycle");
        }

        return new RebuiltTree<>(rebuilt, root);
    }

    private static <K extends Serializable, V extends Serializable> int countReachable(Node<K, V> root) {
        var count = 0;
        var stack = new ArrayDeque<Node<K, V>>();
        stack.push(root);
        while (!stack.isEmpty()) {
            var current = stack.pop();
            count++;
            pushChildrenReverse(current, stack);
        }
        return count;
    }

    private void notifyObservers(Observer<K, V>[] listeners, Change<K, V> change) {
        if (listeners.length == 0) {
            return;
        }
        RuntimeException firstFailure = null;
        for (var observer : listeners) {
            try {
                observer.onChange(change);
            } catch (RuntimeException exception) {
                if (firstFailure == null) {
                    firstFailure = exception;
                }
            }
        }
        if (firstFailure != null) {
            throw firstFailure;
        }
    }

    @SuppressWarnings("unchecked")
    private Observer<K, V>[] emptyObservers() {
        return (Observer<K, V>[]) NO_OBSERVERS;
    }

    private void initializeTransients() {
        treeLock = new ReentrantReadWriteLock();
        readLock = treeLock.readLock();
        writeLock = treeLock.writeLock();
        observerLock = new Object();
        observers = emptyObservers();
        size = nodes.size();
    }

    @Serial
    private void writeObject(ObjectOutputStream out) throws IOException {
        readLock.lock();
        try {
            out.defaultWriteObject();
        } finally {
            readLock.unlock();
        }
    }

    @Serial
    private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();
        initializeTransients();
    }
}
