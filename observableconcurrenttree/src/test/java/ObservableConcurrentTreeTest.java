import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class ObservableConcurrentTreeTest {
    @Test
    void supportsCompleteMutationLifecycle() {
        var tree = new ObservableConcurrentTree<Integer, String>();
        assertTrue(tree.isEmpty());

        tree.initialize(0, "root");
        tree.add(0, 1, "one");
        tree.add(0, 2, "two");
        tree.add(1, 3, "three");

        assertEquals(List.of(0, 1, 3, 2), TreeTestSupport.ids(tree));
        assertEquals(1, tree.getParentId(3));

        tree.update(3, "THREE");
        assertEquals("THREE", tree.get(3).value());

        tree.move(3, 2);
        assertEquals(2, tree.getParentId(3));
        assertEquals(List.of(0, 1, 2, 3), TreeTestSupport.ids(tree));

        assertEquals(List.of(2, 3), tree.removeSubtree(2));
        assertEquals(List.of(0, 1), TreeTestSupport.ids(tree));

        tree.clear();
        assertTrue(tree.isEmpty());
        TreeTestSupport.assertValid(tree);
    }

    @Test
    void sameParentMoveIsARealNoOp() {
        var tree = TreeTestSupport.star(2);
        var version = tree.getVersion();
        var snapshot = tree.snapshot();

        tree.move(1, 0);

        assertEquals(version, tree.getVersion());
        assertEquals(snapshot, tree.snapshot());
    }

    @Test
    void rejectedMutationsLeaveStateAndVersionUnchanged() {
        var tree = new ObservableConcurrentTree<Integer, Integer>(0, 0);
        tree.add(0, 1, 1);
        tree.add(1, 2, 2);

        assertUnchanged(tree, () -> tree.add(0, 1, 999));
        assertUnchanged(tree, () -> tree.add(99, 3, 3));
        assertUnchanged(tree, () -> tree.move(0, 1));
        assertUnchanged(tree, () -> tree.move(1, 1));
        assertUnchanged(tree, () -> tree.move(1, 2));
        assertUnchanged(tree, () -> tree.removeSubtree(99));
        assertUnchanged(tree, () -> tree.update(99, 1));

        TreeTestSupport.assertValid(tree);
    }

    @Test
    void entryCacheIsStableUntilRelevantMutation() {
        var tree = new ObservableConcurrentTree<Integer, String>(0, "root");
        tree.add(0, 1, "one");

        var first = tree.get(1);
        var second = tree.get(1);
        assertSame(first, second);

        tree.update(1, "ONE");
        var updated = tree.get(1);
        assertNotSame(first, updated);
        assertEquals("ONE", updated.value());
    }

    @Test
    void recordViewsAreImmutable() {
        var tree = TreeTestSupport.star(2);

        var root = tree.getRoot();
        assertThrows(UnsupportedOperationException.class, () -> root.children().add(3));
        assertThrows(UnsupportedOperationException.class, () -> tree.snapshot().entries().clear());
    }

    @Test
    void nullAndMissingIdContractsAreConsistent() {
        var tree = new ObservableConcurrentTree<Integer, Integer>(0, 0);
        assertFalse(tree.contains(null));
        assertThrows(NullPointerException.class, () -> tree.get(null));
        assertThrows(NullPointerException.class, () -> tree.getChildren(null));
        assertThrows(NullPointerException.class, () -> tree.add(null, 1, 1));
        assertThrows(NullPointerException.class, () -> tree.add(0, null, 1));
        assertThrows(IllegalArgumentException.class, () -> tree.getParentId(99));
    }

    private static void assertUnchanged(ObservableConcurrentTree<Integer, Integer> tree, Runnable operation) {
        var snapshot = tree.snapshot();
        var version = tree.getVersion();
        assertThrows(RuntimeException.class, operation::run);
        assertEquals(version, tree.getVersion());
        assertEquals(snapshot, tree.snapshot());
    }
}
