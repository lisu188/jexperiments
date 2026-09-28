import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayDeque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

final class TreeTestSupport {
    private TreeTestSupport() {
    }

    static <K, V> void assertValid(ObservableConcurrentTree<K, V> tree) {
        var snapshot = tree.snapshot();
        assertEquals(tree.size(), snapshot.entries().size(), "snapshot size must match tree size");

        if (snapshot.entries().isEmpty()) {
            assertNull(snapshot.rootId(), "empty tree must have null root");
            assertTrue(tree.isEmpty(), "empty snapshot must imply empty tree");
            return;
        }

        assertNotNull(snapshot.rootId(), "non-empty tree must have a root");
        var byId = new HashMap<K, ObservableConcurrentTree.Entry<K, V>>();
        for (var entry : snapshot.entries()) {
            assertNull(byId.put(entry.id(), entry), "duplicate entry id: " + entry.id());
        }

        assertTrue(byId.containsKey(snapshot.rootId()), "root must exist in snapshot");
        assertNull(byId.get(snapshot.rootId()).parentId(), "root parent must be null");

        var childOwners = new HashMap<K, K>();
        for (var entry : snapshot.entries()) {
            for (var childId : entry.children()) {
                assertTrue(byId.containsKey(childId), "child must exist: " + childId);
                var previous = childOwners.put(childId, entry.id());
                assertNull(previous, "child has more than one parent: " + childId);
                assertEquals(entry.id(), byId.get(childId).parentId(), "parent/child link mismatch");
            }
            if (!entry.id().equals(snapshot.rootId())) {
                assertNotNull(entry.parentId(), "non-root parent must be non-null");
                assertTrue(byId.containsKey(entry.parentId()), "parent must exist: " + entry.parentId());
            }
        }

        assertEquals(snapshot.entries().size() - 1, childOwners.size(), "every non-root must be owned once");

        Set<K> reachable = new HashSet<>();
        var stack = new ArrayDeque<K>();
        stack.push(snapshot.rootId());
        while (!stack.isEmpty()) {
            var id = stack.pop();
            assertTrue(reachable.add(id), "cycle detected at " + id);
            var children = byId.get(id).children();
            for (int i = children.size() - 1; i >= 0; i--) {
                stack.push(children.get(i));
            }
        }
        assertEquals(byId.keySet(), reachable, "all nodes must be reachable from root");
    }

    static ObservableConcurrentTree<Integer, Integer> star(int children) {
        var tree = new ObservableConcurrentTree<Integer, Integer>(0, 0);
        for (int i = 1; i <= children; i++) {
            tree.add(0, i, i);
        }
        return tree;
    }

    static ObservableConcurrentTree<Integer, Integer> chain(int nodes) {
        var tree = new ObservableConcurrentTree<Integer, Integer>(0, 0);
        for (int i = 1; i < nodes; i++) {
            tree.add(i - 1, i, i);
        }
        return tree;
    }

    static List<Integer> ids(ObservableConcurrentTree<Integer, ?> tree) {
        return tree.depthFirst().stream().map(ObservableConcurrentTree.Entry::id).toList();
    }
}
