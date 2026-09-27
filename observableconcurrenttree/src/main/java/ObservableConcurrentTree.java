import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.locks.ReentrantReadWriteLock;

public final class ObservableConcurrentTree<K extends Serializable, V extends Serializable> implements Serializable {
    private static final long serialVersionUID = 1L;

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

    public interface Observer<K extends Serializable, V extends Serializable> {
        void onChange(Change<K, V> change);
    }

    public static final class NodeState<K extends Serializable, V extends Serializable> implements Serializable {
        private static final long serialVersionUID = 1L;

        private final K id;
        private final K parentId;
        private final V value;

        public NodeState(K id, K parentId, V value) {
            if (id == null) {
                throw new NullPointerException("id");
            }
            this.id = id;
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

    public static final class Entry<K extends Serializable, V extends Serializable> implements Serializable {
        private static final long serialVersionUID = 1L;

        private final K id;
        private final K parentId;
        private final V value;
        private final List<K> children;

        private Entry(K id, K parentId, V value, List<K> children) {
            this.id = id;
            this.parentId = parentId;
            this.value = value;
            this.children = Collections.unmodifiableList(new ArrayList<K>(children));
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

    public static final class Snapshot<K extends Serializable, V extends Serializable> implements Serializable {
        private static final long serialVersionUID = 1L;

        private final K rootId;
        private final long version;
        private final List<Entry<K, V>> entries;

        private Snapshot(K rootId, long version, List<Entry<K, V>> entries) {
            this.rootId = rootId;
            this.version = version;
            this.entries = Collections.unmodifiableList(new ArrayList<Entry<K, V>>(entries));
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

    public static final class Change<K extends Serializable, V extends Serializable> implements Serializable {
        private static final long serialVersionUID = 1L;

        private final ChangeType type;
        private final long version;
        private final K nodeId;
        private final K oldParentId;
        private final K newParentId;
        private final V oldValue;
        private final V newValue;
        private final List<K> affectedNodeIds;

        private Change(ChangeType type,
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
            this.affectedNodeIds = Collections.unmodifiableList(new ArrayList<K>(affectedNodeIds));
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
        private static final long serialVersionUID = 1L;

        private final EventType type;
        private final K nodeId;
        private final K parentId;
        private final V value;

        private TreeEvent(EventType type, K nodeId, K parentId, V value) {
            if (type == null) {
                throw new NullPointerException("type");
            }
            if (nodeId == null) {
                throw new NullPointerException("nodeId");
            }
            this.type = type;
            this.nodeId = nodeId;
            this.parentId = parentId;
            this.value = value;
        }

        public static <K extends Serializable, V extends Serializable> TreeEvent<K, V> add(K parentId, K nodeId, V value) {
            if (parentId == null) {
                throw new NullPointerException("parentId");
            }
            return new TreeEvent<K, V>(EventType.ADD, nodeId, parentId, value);
        }

        public static <K extends Serializable, V extends Serializable> TreeEvent<K, V> update(K nodeId, V value) {
            return new TreeEvent<K, V>(EventType.UPDATE, nodeId, null, value);
        }

        public static <K extends Serializable, V extends Serializable> TreeEvent<K, V> move(K nodeId, K newParentId) {
            if (newParentId == null) {
                throw new NullPointerException("newParentId");
            }
            return new TreeEvent<K, V>(EventType.MOVE, nodeId, newParentId, null);
        }

        public static <K extends Serializable, V extends Serializable> TreeEvent<K, V> remove(K nodeId) {
            return new TreeEvent<K, V>(EventType.REMOVE, nodeId, null, null);
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
        private static final long serialVersionUID = 1L;

        private final K id;
        private K parentId;
        private V value;
        private final LinkedHashSet<K> children;

        private Node(K id, K parentId, V value) {
            this.id = id;
            this.parentId = parentId;
            this.value = value;
            this.children = new LinkedHashSet<K>();
        }
    }

    private final Map<K, Node<K, V>> nodes = new HashMap<K, Node<K, V>>();
    private K rootId;
    private long version;
    private transient ReentrantReadWriteLock lock;
    private transient CopyOnWriteArrayList<Observer<K, V>> observers;

    public ObservableConcurrentTree() {
        initializeTransients();
    }

    public ObservableConcurrentTree(K rootId, V rootValue) {
        initializeTransients();
        initialize(rootId, rootValue);
    }

    public void addObserver(Observer<K, V> observer) {
        if (observer == null) {
            throw new NullPointerException("observer");
        }
        observers.addIfAbsent(observer);
    }

    public void removeObserver(Observer<K, V> observer) {
        if (observer != null) {
            observers.remove(observer);
        }
    }

    public void initialize(K newRootId, V value) {
        if (newRootId == null) {
            throw new NullPointerException("rootId");
        }

        Change<K, V> change;
        lock.writeLock().lock();
        try {
            if (!nodes.isEmpty()) {
                throw new IllegalStateException("Tree is already initialized");
            }
            nodes.put(newRootId, new Node<K, V>(newRootId, null, value));
            rootId = newRootId;
            long newVersion = ++version;
            change = new Change<K, V>(
                    ChangeType.INITIALIZED,
                    newVersion,
                    newRootId,
                    null,
                    null,
                    null,
                    value,
                    Collections.singletonList(newRootId));
        } finally {
            lock.writeLock().unlock();
        }
        notifyObservers(change);
    }

    public void add(K parentId, K nodeId, V value) {
        requireId(parentId, "parentId");
        requireId(nodeId, "nodeId");

        Change<K, V> change;
        lock.writeLock().lock();
        try {
            if (nodes.containsKey(nodeId)) {
                throw new IllegalArgumentException("Node already exists: " + nodeId);
            }
            Node<K, V> parent = nodes.get(parentId);
            if (parent == null) {
                throw new IllegalArgumentException("Parent does not exist: " + parentId);
            }
            Node<K, V> node = new Node<K, V>(nodeId, parentId, value);
            nodes.put(nodeId, node);
            parent.children.add(nodeId);
            long newVersion = ++version;
            change = new Change<K, V>(
                    ChangeType.ADDED,
                    newVersion,
                    nodeId,
                    null,
                    parentId,
                    null,
                    value,
                    Collections.singletonList(nodeId));
        } finally {
            lock.writeLock().unlock();
        }
        notifyObservers(change);
    }

    public void update(K nodeId, V value) {
        requireId(nodeId, "nodeId");

        Change<K, V> change;
        lock.writeLock().lock();
        try {
            Node<K, V> node = requireNode(nodeId);
            V oldValue = node.value;
            node.value = value;
            long newVersion = ++version;
            change = new Change<K, V>(
                    ChangeType.UPDATED,
                    newVersion,
                    nodeId,
                    node.parentId,
                    node.parentId,
                    oldValue,
                    value,
                    Collections.singletonList(nodeId));
        } finally {
            lock.writeLock().unlock();
        }
        notifyObservers(change);
    }

    public void move(K nodeId, K newParentId) {
        requireId(nodeId, "nodeId");
        requireId(newParentId, "newParentId");

        Change<K, V> change;
        lock.writeLock().lock();
        try {
            Node<K, V> node = requireNode(nodeId);
            Node<K, V> newParent = requireNode(newParentId);
            if (node.parentId == null) {
                throw new IllegalArgumentException("Root node cannot be moved");
            }
            if (nodeId.equals(newParentId)) {
                throw new IllegalArgumentException("Node cannot be its own parent");
            }
            ensureNotDescendant(nodeId, newParentId);
            K oldParentId = node.parentId;
            if (oldParentId.equals(newParentId)) {
                return;
            }
            Node<K, V> oldParent = nodes.get(oldParentId);
            oldParent.children.remove(nodeId);
            newParent.children.add(nodeId);
            node.parentId = newParentId;
            long newVersion = ++version;
            change = new Change<K, V>(
                    ChangeType.MOVED,
                    newVersion,
                    nodeId,
                    oldParentId,
                    newParentId,
                    node.value,
                    node.value,
                    Collections.singletonList(nodeId));
        } finally {
            lock.writeLock().unlock();
        }
        notifyObservers(change);
    }

    public List<K> removeSubtree(K nodeId) {
        requireId(nodeId, "nodeId");

        Change<K, V> change;
        List<K> removed;
        lock.writeLock().lock();
        try {
            Node<K, V> node = requireNode(nodeId);
            K oldParentId = node.parentId;
            V oldValue = node.value;
            removed = collectSubtreeIds(nodeId);
            if (oldParentId != null) {
                Node<K, V> parent = nodes.get(oldParentId);
                if (parent != null) {
                    parent.children.remove(nodeId);
                }
            }
            for (K id : removed) {
                nodes.remove(id);
            }
            if (nodeId.equals(rootId)) {
                rootId = null;
            }
            long newVersion = ++version;
            change = new Change<K, V>(
                    ChangeType.REMOVED,
                    newVersion,
                    nodeId,
                    oldParentId,
                    null,
                    oldValue,
                    null,
                    removed);
        } finally {
            lock.writeLock().unlock();
        }
        notifyObservers(change);
        return Collections.unmodifiableList(new ArrayList<K>(removed));
    }

    public void clear() {
        Change<K, V> change;
        lock.writeLock().lock();
        try {
            if (nodes.isEmpty()) {
                return;
            }
            List<K> removed = snapshotIdsDepthFirst();
            nodes.clear();
            K oldRoot = rootId;
            rootId = null;
            long newVersion = ++version;
            change = new Change<K, V>(
                    ChangeType.CLEARED,
                    newVersion,
                    oldRoot,
                    null,
                    null,
                    null,
                    null,
                    removed);
        } finally {
            lock.writeLock().unlock();
        }
        notifyObservers(change);
    }

    public void loadState(Collection<NodeState<K, V>> state) {
        if (state == null) {
            throw new NullPointerException("state");
        }

        RebuiltTree<K, V> rebuilt = rebuild(state);
        Change<K, V> change;
        lock.writeLock().lock();
        try {
            nodes.clear();
            nodes.putAll(rebuilt.nodes);
            rootId = rebuilt.rootId;
            long newVersion = ++version;
            List<K> affected = snapshotIdsDepthFirst();
            change = new Change<K, V>(
                    ChangeType.STATE_LOADED,
                    newVersion,
                    rootId,
                    null,
                    null,
                    null,
                    rootId == null ? null : nodes.get(rootId).value,
                    affected);
        } finally {
            lock.writeLock().unlock();
        }
        notifyObservers(change);
    }

    public void apply(TreeEvent<K, V> event) {
        if (event == null) {
            throw new NullPointerException("event");
        }
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
            default:
                throw new IllegalArgumentException("Unsupported event type: " + event.getType());
        }
    }

    public boolean contains(K nodeId) {
        if (nodeId == null) {
            return false;
        }
        lock.readLock().lock();
        try {
            return nodes.containsKey(nodeId);
        } finally {
            lock.readLock().unlock();
        }
    }

    public Entry<K, V> get(K nodeId) {
        requireId(nodeId, "nodeId");
        lock.readLock().lock();
        try {
            Node<K, V> node = nodes.get(nodeId);
            return node == null ? null : toEntry(node);
        } finally {
            lock.readLock().unlock();
        }
    }

    public K getRootId() {
        lock.readLock().lock();
        try {
            return rootId;
        } finally {
            lock.readLock().unlock();
        }
    }

    public Entry<K, V> getRoot() {
        lock.readLock().lock();
        try {
            if (rootId == null) {
                return null;
            }
            return toEntry(nodes.get(rootId));
        } finally {
            lock.readLock().unlock();
        }
    }

    public K getParentId(K nodeId) {
        requireId(nodeId, "nodeId");
        lock.readLock().lock();
        try {
            return requireNode(nodeId).parentId;
        } finally {
            lock.readLock().unlock();
        }
    }

    public List<Entry<K, V>> getChildren(K nodeId) {
        requireId(nodeId, "nodeId");
        lock.readLock().lock();
        try {
            Node<K, V> node = requireNode(nodeId);
            List<Entry<K, V>> result = new ArrayList<Entry<K, V>>(node.children.size());
            for (K childId : node.children) {
                result.add(toEntry(nodes.get(childId)));
            }
            return Collections.unmodifiableList(result);
        } finally {
            lock.readLock().unlock();
        }
    }

    public List<Entry<K, V>> depthFirst() {
        lock.readLock().lock();
        try {
            if (rootId == null) {
                return Collections.emptyList();
            }
            List<Entry<K, V>> result = new ArrayList<Entry<K, V>>(nodes.size());
            Deque<K> stack = new ArrayDeque<K>();
            stack.push(rootId);
            while (!stack.isEmpty()) {
                K id = stack.pop();
                Node<K, V> node = nodes.get(id);
                result.add(toEntry(node));
                List<K> children = new ArrayList<K>(node.children);
                for (int i = children.size() - 1; i >= 0; i--) {
                    stack.push(children.get(i));
                }
            }
            return Collections.unmodifiableList(result);
        } finally {
            lock.readLock().unlock();
        }
    }

    public Snapshot<K, V> snapshot() {
        lock.readLock().lock();
        try {
            List<Entry<K, V>> entries = new ArrayList<Entry<K, V>>(nodes.size());
            if (rootId != null) {
                Deque<K> stack = new ArrayDeque<K>();
                stack.push(rootId);
                while (!stack.isEmpty()) {
                    K id = stack.pop();
                    Node<K, V> node = nodes.get(id);
                    entries.add(toEntry(node));
                    List<K> children = new ArrayList<K>(node.children);
                    for (int i = children.size() - 1; i >= 0; i--) {
                        stack.push(children.get(i));
                    }
                }
            }
            return new Snapshot<K, V>(rootId, version, entries);
        } finally {
            lock.readLock().unlock();
        }
    }

    public long getVersion() {
        lock.readLock().lock();
        try {
            return version;
        } finally {
            lock.readLock().unlock();
        }
    }

    public int size() {
        lock.readLock().lock();
        try {
            return nodes.size();
        } finally {
            lock.readLock().unlock();
        }
    }

    public boolean isEmpty() {
        return size() == 0;
    }

    public byte[] toByteArray() throws IOException {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        ObjectOutputStream out = new ObjectOutputStream(bytes);
        try {
            out.writeObject(this);
        } finally {
            out.close();
        }
        return bytes.toByteArray();
    }

    @SuppressWarnings("unchecked")
    public static <K extends Serializable, V extends Serializable> ObservableConcurrentTree<K, V> fromByteArray(byte[] bytes)
            throws IOException, ClassNotFoundException {
        if (bytes == null) {
            throw new NullPointerException("bytes");
        }
        ObjectInputStream in = new ObjectInputStream(new ByteArrayInputStream(bytes));
        try {
            Object value = in.readObject();
            if (!(value instanceof ObservableConcurrentTree)) {
                throw new IOException("Serialized value is not an ObservableConcurrentTree");
            }
            return (ObservableConcurrentTree<K, V>) value;
        } finally {
            in.close();
        }
    }

    private void ensureNotDescendant(K nodeId, K candidateParentId) {
        K current = candidateParentId;
        while (current != null) {
            if (nodeId.equals(current)) {
                throw new IllegalArgumentException("Move would create a cycle");
            }
            Node<K, V> node = nodes.get(current);
            current = node == null ? null : node.parentId;
        }
    }

    private List<K> collectSubtreeIds(K nodeId) {
        List<K> result = new ArrayList<K>();
        Deque<K> stack = new ArrayDeque<K>();
        stack.push(nodeId);
        while (!stack.isEmpty()) {
            K id = stack.pop();
            Node<K, V> node = nodes.get(id);
            if (node == null) {
                continue;
            }
            result.add(id);
            List<K> children = new ArrayList<K>(node.children);
            for (int i = children.size() - 1; i >= 0; i--) {
                stack.push(children.get(i));
            }
        }
        return result;
    }

    private List<K> snapshotIdsDepthFirst() {
        if (rootId == null) {
            return Collections.emptyList();
        }
        return collectSubtreeIds(rootId);
    }

    private Entry<K, V> toEntry(Node<K, V> node) {
        return new Entry<K, V>(node.id, node.parentId, node.value, new ArrayList<K>(node.children));
    }

    private Node<K, V> requireNode(K nodeId) {
        Node<K, V> node = nodes.get(nodeId);
        if (node == null) {
            throw new IllegalArgumentException("Node does not exist: " + nodeId);
        }
        return node;
    }

    private static void requireId(Object id, String name) {
        if (id == null) {
            throw new NullPointerException(name);
        }
    }

    private static final class RebuiltTree<K extends Serializable, V extends Serializable> {
        private final Map<K, Node<K, V>> nodes;
        private final K rootId;

        private RebuiltTree(Map<K, Node<K, V>> nodes, K rootId) {
            this.nodes = nodes;
            this.rootId = rootId;
        }
    }

    private RebuiltTree<K, V> rebuild(Collection<NodeState<K, V>> state) {
        Map<K, Node<K, V>> rebuilt = new HashMap<K, Node<K, V>>();
        List<NodeState<K, V>> ordered = new ArrayList<NodeState<K, V>>(state.size());
        K rebuiltRootId = null;

        for (NodeState<K, V> item : state) {
            if (item == null) {
                throw new NullPointerException("state contains null");
            }
            K id = item.getId();
            if (rebuilt.containsKey(id)) {
                throw new IllegalArgumentException("Duplicate node id: " + id);
            }
            if (item.getParentId() == null) {
                if (rebuiltRootId != null) {
                    throw new IllegalArgumentException("State contains more than one root");
                }
                rebuiltRootId = id;
            }
            rebuilt.put(id, new Node<K, V>(id, item.getParentId(), item.getValue()));
            ordered.add(item);
        }

        if (!rebuilt.isEmpty() && rebuiltRootId == null) {
            throw new IllegalArgumentException("State has no root");
        }

        for (NodeState<K, V> item : ordered) {
            K parentId = item.getParentId();
            if (parentId == null) {
                continue;
            }
            if (item.getId().equals(parentId)) {
                throw new IllegalArgumentException("Node cannot be its own parent: " + item.getId());
            }
            Node<K, V> parent = rebuilt.get(parentId);
            if (parent == null) {
                throw new IllegalArgumentException("Missing parent " + parentId + " for node " + item.getId());
            }
            parent.children.add(item.getId());
        }

        if (rebuiltRootId != null) {
            Set<K> visited = new HashSet<K>();
            Deque<K> stack = new ArrayDeque<K>();
            stack.push(rebuiltRootId);
            while (!stack.isEmpty()) {
                K id = stack.pop();
                if (!visited.add(id)) {
                    throw new IllegalArgumentException("State contains a cycle at node: " + id);
                }
                Node<K, V> node = rebuilt.get(id);
                for (K childId : node.children) {
                    stack.push(childId);
                }
            }
            if (visited.size() != rebuilt.size()) {
                throw new IllegalArgumentException("State contains disconnected nodes or a cycle");
            }
        }

        return new RebuiltTree<K, V>(rebuilt, rebuiltRootId);
    }

    private void notifyObservers(Change<K, V> change) {
        RuntimeException firstFailure = null;
        for (Observer<K, V> observer : observers) {
            try {
                observer.onChange(change);
            } catch (RuntimeException e) {
                if (firstFailure == null) {
                    firstFailure = e;
                }
            }
        }
        if (firstFailure != null) {
            throw firstFailure;
        }
    }

    private void initializeTransients() {
        lock = new ReentrantReadWriteLock();
        observers = new CopyOnWriteArrayList<Observer<K, V>>();
    }

    private void writeObject(ObjectOutputStream out) throws IOException {
        lock.readLock().lock();
        try {
            out.defaultWriteObject();
        } finally {
            lock.readLock().unlock();
        }
    }

    private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();
        initializeTransients();
    }
}
