import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import org.junit.jupiter.api.Test;

class ObservableConcurrentTreeObserverTest {
    @Test
    void emitsPreciseChangePayloads() {
        var tree = new ObservableConcurrentTree<Integer, String>();
        var changes = new ArrayList<ObservableConcurrentTree.Change<Integer, String>>();
        tree.addObserver(changes::add);

        tree.initialize(0, "root");
        tree.add(0, 1, "one");
        tree.update(1, "ONE");
        tree.add(0, 2, "two");
        tree.move(1, 2);
        tree.removeSubtree(2);
        tree.clear();
        tree.initialize(10, "again");
        tree.clear();

        assertEquals(List.of(
                ObservableConcurrentTree.ChangeType.INITIALIZED,
                ObservableConcurrentTree.ChangeType.ADDED,
                ObservableConcurrentTree.ChangeType.UPDATED,
                ObservableConcurrentTree.ChangeType.ADDED,
                ObservableConcurrentTree.ChangeType.MOVED,
                ObservableConcurrentTree.ChangeType.REMOVED,
                ObservableConcurrentTree.ChangeType.CLEARED,
                ObservableConcurrentTree.ChangeType.INITIALIZED,
                ObservableConcurrentTree.ChangeType.CLEARED),
                changes.stream().map(ObservableConcurrentTree.Change::type).toList());

        assertEquals(0, changes.get(0).nodeId());
        assertEquals(0, changes.get(1).newParentId());
        assertEquals("one", changes.get(2).oldValue());
        assertEquals("ONE", changes.get(2).newValue());
        assertEquals(List.of(2, 1), changes.get(5).affectedNodeIds());
    }

    @Test
    void duplicateObserverIsRegisteredOnceAndRemovalIsIdempotent() {
        var tree = new ObservableConcurrentTree<Integer, Integer>(0, 0);
        var calls = new AtomicInteger();
        ObservableConcurrentTree.Observer<Integer, Integer> observer = ignored -> calls.incrementAndGet();

        tree.addObserver(observer);
        tree.addObserver(observer);
        tree.add(0, 1, 1);
        assertEquals(1, calls.get());

        tree.removeObserver(observer);
        tree.removeObserver(observer);
        tree.removeObserver(null);
        tree.add(0, 2, 2);
        assertEquals(1, calls.get());
    }

    @Test
    void allObserversRunEvenWhenOneThrowsAndMutationRemainsCommitted() {
        var tree = new ObservableConcurrentTree<Integer, Integer>(0, 0);
        var calls = new AtomicInteger();
        tree.addObserver(change -> {
            calls.incrementAndGet();
            throw new IllegalStateException("first");
        });
        tree.addObserver(change -> calls.incrementAndGet());
        tree.addObserver(change -> {
            calls.incrementAndGet();
            throw new IllegalArgumentException("third");
        });

        var failure = assertThrows(IllegalStateException.class, () -> tree.add(0, 1, 1));
        assertEquals("first", failure.getMessage());
        assertEquals(3, calls.get());
        assertTrue(tree.contains(1));
    }

    @Test
    void observerMayReenterReadApiWithoutDeadlock() {
        var tree = new ObservableConcurrentTree<Integer, Integer>(0, 0);
        tree.addObserver(change -> {
            tree.snapshot();
            tree.stream().count();
            tree.getRoot();
            if (change.nodeId() != null && tree.contains(change.nodeId())) {
                tree.get(change.nodeId());
            }
        });

        tree.add(0, 1, 1);
        tree.update(1, 2);
        TreeTestSupport.assertValid(tree);
    }
}
