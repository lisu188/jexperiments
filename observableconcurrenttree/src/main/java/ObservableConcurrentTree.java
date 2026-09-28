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
import java.util.stream.Stream;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;

public final class ObservableConcurrentTree<K, V> {

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


    @FunctionalInterface
    public interface Observer<K, V> {
        void onChange(Change<K, V> change);
    }

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

    public record Snapshot<K, V>(K rootId, long version, List<Entry<K, V>> entries) {
        public Snapshot {
            entries = List.copyOf(entries);
        }
    }

    public record Change<K, V>(
            ChangeType type,
            long version,
            K nodeId,
            K oldParentId,
            K newParentId,
            V oldValue,
            V newValue,
            List<K> affectedNodeIds) {
        public Change {
            Objects.requireNonNull(type, "type");
            affectedNodeIds = List.copyOf(affectedNodeIds);
        }
    }

    public sealed interface TreeEvent<K, V> permits Add, Update, Move, Remove {
        K nodeId();

        static <K, V> Add<K, V> add(K parentId, K nodeId, V value) {
            return new Add<>(parentId, nodeId, value);
        }

        static <K, V> Update<K, V> update(K nodeId, V value) {
            return new Update<>(nodeId, value);
        }

        static <K, V> Move<K, V> move(K nodeId, K newParentId) {
            return new Move<>(nodeId, newParentId);
        }

        static <K, V> Remove<K, V> remove(K nodeId) {
            return new Remove<>(nodeId);
        }
    }

    public record Add<K, V>(K parentId, K nodeId, V value) implements TreeEvent<K, V> {
        public Add {
            Objects.requireNonNull(parentId, "parentId");
            Objects.requireNonNull(nodeId, "nodeId");
        }
    }

    public record Update<K, V>(K nodeId, V value) implements TreeEvent<K, V> {
        public Update {
            Objects.requireNonNull(nodeId, "nodeId");
        }
    }

    public record Move<K, V>(K nodeId, K parentId) implements TreeEvent<K, V> {
        public Move {
            Objects.requireNonNull(nodeId, "nodeId");
            Objects.requireNonNull(parentId, "parentId");
        }
    }

    public record Remove<K, V>(K nodeId) implements TreeEvent<K, V> {
        public Remove {
            Objects.requireNonNull(nodeId, "nodeId");
        }
    }

    private static final class Node<K, V> {

        private final K id;
        private Node<K, V> parent;
        private V value;
        private ArrayList<Node<K, V>> children;
        private transient volatile Entry<K, V> entry;

        private Node(K id, V value) {
            this(id, value, 0);
        }

        private Node(K id, V value, int expectedChildren) {
            this.id = id;
            this.value = value;
            this.children = expectedChildren == 0 ? null : new ArrayList<>(expectedChildren);
        }
    }

    private record RebuiltTree<K, V>(
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
            addChild(parent, node);
            parent.entry = null;
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
            node.entry = null;
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
            removeChild(oldParent, node);
            oldParent.entry = null;
            addChild(newParent, node);
            newParent.entry = null;
            node.parent = newParent;
            node.entry = null;
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
                removeChild(oldParent, node);
                oldParent.entry = null;
            }
            ArrayList<K> removed;
            if (node.id.equals(rootId)) {
                removed = collectSubtreeIds(node);
                nodes.clear();
                rootId = null;
                size = 0;
            } else {
                removed = removeSubtreeNodes(node);
                size -= removed.size();
            }
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
                        removed);
                removedView = change.affectedNodeIds();
            } else {
                removedView = List.copyOf(removed);
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
                        removed);
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
                var affected = root == null ? List.<K>of() : collectSubtreeIds(root);
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
        switch (event) {
            case Add<K, V>(var parentId, var nodeId, var value) -> add(parentId, nodeId, value);
            case Update<K, V>(var nodeId, var value) -> update(nodeId, value);
            case Move<K, V>(var nodeId, var parentId) -> move(nodeId, parentId);
            case Remove<K, V>(var nodeId) -> removeSubtree(nodeId);
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
            if (children == null) {
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

    public Stream<Entry<K, V>> stream() {
        return snapshot().entries().stream();
    }

    public Stream<Entry<K, V>> childrenStream(K nodeId) {
        return getChildren(nodeId).stream();
    }

    public Stream<Entry<K, V>> subtreeStream(K nodeId) {
        Objects.requireNonNull(nodeId, "nodeId");
        readLock.lock();
        try {
            var entries = new ArrayList<Entry<K, V>>();
            appendDepthFirst(requireNode(nodeId), entries);
            return entries.stream();
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
            return new Snapshot<>(currentRootId, version, entries);
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

    private static final ObjectMapper DEFAULT_JSON_MAPPER = new ObjectMapper();

    private record JsonState<K, V>(long version, List<NodeState<K, V>> nodes) {
        private JsonState {
            if (version < 0) {
                throw new IllegalArgumentException("version must be non-negative");
            }
            nodes = List.copyOf(nodes);
        }
    }

    public String toJson() throws JsonProcessingException {
        return toJson(DEFAULT_JSON_MAPPER);
    }

    public String toJson(ObjectMapper mapper) throws JsonProcessingException {
        Objects.requireNonNull(mapper, "mapper");
        return mapper.writeValueAsString(jsonState());
    }

    public static <K, V> ObservableConcurrentTree<K, V> fromJson(
            String json,
            Class<K> keyType,
            Class<V> valueType) throws JsonProcessingException {
        return fromJson(DEFAULT_JSON_MAPPER, json, keyType, valueType);
    }

    public static <K, V> ObservableConcurrentTree<K, V> fromJson(
            ObjectMapper mapper,
            String json,
            Class<K> keyType,
            Class<V> valueType) throws JsonProcessingException {
        Objects.requireNonNull(mapper, "mapper");
        Objects.requireNonNull(json, "json");
        Objects.requireNonNull(keyType, "keyType");
        Objects.requireNonNull(valueType, "valueType");

        var type = mapper.getTypeFactory().constructParametricType(JsonState.class, keyType, valueType);
        @SuppressWarnings("unchecked")
        var state = (JsonState<K, V>) mapper.readValue(json, type);

        var tree = new ObservableConcurrentTree<K, V>();
        tree.restoreState(state.nodes(), state.version());
        return tree;
    }

    private JsonState<K, V> jsonState() {
        readLock.lock();
        try {
            var state = new ArrayList<NodeState<K, V>>(size);
            var currentRootId = rootId;
            if (currentRootId != null) {
                appendStateDepthFirst(nodes.get(currentRootId), state);
            }
            return new JsonState<>(version, state);
        } finally {
            readLock.unlock();
        }
    }

    private void restoreState(Collection<NodeState<K, V>> state, long restoredVersion) {
        if (restoredVersion < 0) {
            throw new IllegalArgumentException("version must be non-negative");
        }
        var rebuilt = rebuild(state);
        writeLock.lock();
        try {
            nodes = rebuilt.nodes();
            var root = rebuilt.root();
            rootId = root == null ? null : root.id;
            size = nodes.size();
            version = restoredVersion;
        } finally {
            writeLock.unlock();
        }
    }

    private static <K, V> void addChild(Node<K, V> parent, Node<K, V> child) {
        var children = parent.children;
        if (children == null) {
            children = new ArrayList<>(4);
            parent.children = children;
        }
        children.add(child);
    }

    private static <K, V> void removeChild(Node<K, V> parent, Node<K, V> child) {
        var children = parent.children;
        if (children == null || !children.remove(child)) {
            throw new IllegalStateException("Parent/child relationship is inconsistent");
        }
        if (children.isEmpty()) {
            parent.children = null;
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

    private void appendStateDepthFirst(Node<K, V> root, ArrayList<NodeState<K, V>> target) {
        var stack = new ArrayDeque<Node<K, V>>();
        stack.push(root);
        while (!stack.isEmpty()) {
            var current = stack.pop();
            target.add(new NodeState<>(
                    current.id,
                    current.parent == null ? null : current.parent.id,
                    current.value));
            pushChildrenReverse(current, stack);
        }
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

    private static <K, V> void pushChildrenReverse(
            Node<K, V> node,
            ArrayDeque<Node<K, V>> stack) {
        var children = node.children;
        if (children == null) {
            return;
        }
        for (int i = children.size() - 1; i >= 0; i--) {
            stack.push(children.get(i));
        }
    }

    private Entry<K, V> toEntry(Node<K, V> node) {
        var cached = node.entry;
        if (cached != null) {
            return cached;
        }

        var parent = node.parent;
        var children = node.children;
        List<K> childIds;
        if (children == null) {
            childIds = List.of();
        } else {
            var ids = new ArrayList<K>(children.size());
            for (var child : children) {
                ids.add(child.id);
            }
            childIds = ids;
        }

        cached = new Entry<>(node.id, parent == null ? null : parent.id, node.value, childIds);
        node.entry = cached;
        return cached;
    }

    private Node<K, V> requireNode(K nodeId) {
        var node = nodes.get(nodeId);
        if (node == null) {
            throw new IllegalArgumentException("Node does not exist: " + nodeId);
        }
        return node;
    }

    private static <K, V> boolean shouldPreSizeChildren(Collection<NodeState<K, V>> state) {
        if (state.size() < 64) {
            return false;
        }

        var sampleCounts = new HashMap<K, Integer>();
        var sampled = 0;
        for (var item : state) {
            Objects.requireNonNull(item, "state contains null");
            var parentId = item.parentId();
            if (parentId != null && sampleCounts.merge(parentId, 1, Integer::sum) >= 32) {
                return true;
            }
            if (++sampled == 1_024) {
                break;
            }
        }
        return false;
    }

    private RebuiltTree<K, V> rebuild(Collection<NodeState<K, V>> state) {
        if (state.isEmpty()) {
            return new RebuiltTree<>(new HashMap<>(), null);
        }

        HashMap<K, Integer> childCounts = null;
        if (shouldPreSizeChildren(state)) {
            childCounts = new HashMap<>();
            for (var item : state) {
                Objects.requireNonNull(item, "state contains null");
                var parentId = item.parentId();
                if (parentId != null) {
                    childCounts.merge(parentId, 1, Integer::sum);
                }
            }
        }

        var rebuilt = HashMap.<K, Node<K, V>>newHashMap(state.size());
        Node<K, V> root = null;

        for (var item : state) {
            Objects.requireNonNull(item, "state contains null");
            var id = item.id();
            var expectedChildren = childCounts == null ? 0 : childCounts.getOrDefault(id, 0);
            if (rebuilt.putIfAbsent(id, new Node<>(id, item.value(), expectedChildren)) != null) {
                throw new IllegalArgumentException("Duplicate node id: " + id);
            }
            if (item.parentId() == null) {
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
            var parentId = item.parentId();
            if (parentId == null) {
                continue;
            }
            if (item.id().equals(parentId)) {
                throw new IllegalArgumentException("Node cannot be its own parent: " + item.id());
            }
            var parent = rebuilt.get(parentId);
            if (parent == null) {
                throw new IllegalArgumentException("Missing parent " + parentId + " for node " + item.id());
            }
            var node = rebuilt.get(item.id());
            node.parent = parent;
            addChild(parent, node);
        }

        if (countReachable(root) != rebuilt.size()) {
            throw new IllegalArgumentException("State contains disconnected nodes or a cycle");
        }

        return new RebuiltTree<>(rebuilt, root);
    }

    private static <K, V> int countReachable(Node<K, V> root) {
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

}
