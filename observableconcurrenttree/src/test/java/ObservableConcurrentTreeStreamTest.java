import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.List;

import org.junit.jupiter.api.Test;

class ObservableConcurrentTreeStreamTest {
    @Test
    void wholeTreeStreamUsesDepthFirstSnapshotOrder() {
        var tree = new ObservableConcurrentTree<Integer, Integer>(0, 0);
        tree.add(0, 1, 1);
        tree.add(0, 2, 2);
        tree.add(1, 3, 3);

        assertEquals(List.of(0, 1, 3, 2),
                tree.stream().map(ObservableConcurrentTree.Entry::id).toList());
    }

    @Test
    void childrenAndSubtreeStreamsPreserveOrder() {
        var tree = new ObservableConcurrentTree<Integer, Integer>(0, 0);
        tree.add(0, 1, 1);
        tree.add(0, 2, 2);
        tree.add(1, 3, 3);

        assertEquals(List.of(1, 2),
                tree.childrenStream(0).map(ObservableConcurrentTree.Entry::id).toList());
        assertEquals(List.of(1, 3),
                tree.subtreeStream(1).map(ObservableConcurrentTree.Entry::id).toList());
    }

    @Test
    void streamsAreDetachedFromLaterMutations() {
        var tree = new ObservableConcurrentTree<Integer, Integer>(0, 0);
        tree.add(0, 1, 1);

        var whole = tree.stream();
        var subtree = tree.subtreeStream(0);
        var children = tree.childrenStream(0);

        tree.add(0, 2, 2);
        tree.update(1, 10);

        assertEquals(List.of(0, 1), whole.map(ObservableConcurrentTree.Entry::id).toList());
        assertEquals(List.of(0, 1), subtree.map(ObservableConcurrentTree.Entry::id).toList());
        assertEquals(List.of(1), children.map(ObservableConcurrentTree.Entry::id).toList());
    }

    @Test
    void missingSubtreeFailsAtStreamCreationRatherThanTerminalOperation() {
        var tree = new ObservableConcurrentTree<Integer, Integer>(0, 0);
        assertThrows(IllegalArgumentException.class, () -> tree.subtreeStream(99));
    }
}
