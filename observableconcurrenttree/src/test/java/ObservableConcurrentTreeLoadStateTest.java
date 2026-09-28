import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.List;

import org.junit.jupiter.api.Test;

class ObservableConcurrentTreeLoadStateTest {
    @Test
    void loadsStateIndependentlyOfInputOrder() {
        var tree = new ObservableConcurrentTree<Integer, String>();
        tree.loadState(List.of(
                new ObservableConcurrentTree.NodeState<>(3, 1, "three"),
                new ObservableConcurrentTree.NodeState<>(1, 0, "one"),
                new ObservableConcurrentTree.NodeState<>(0, null, "root"),
                new ObservableConcurrentTree.NodeState<>(2, 0, "two")));

        assertEquals(List.of(0, 1, 3, 2), TreeTestSupport.ids(tree));
        TreeTestSupport.assertValid(tree);
    }

    @Test
    void emptyStateClearsTreeAtomically() {
        var tree = TreeTestSupport.star(10);
        tree.loadState(List.of());
        assertEquals(0, tree.size());
        TreeTestSupport.assertValid(tree);
    }

    @Test
    void invalidStatesNeverReplaceExistingTree() {
        var tree = TreeTestSupport.star(3);
        assertRejectedUnchanged(tree, List.of(
                new ObservableConcurrentTree.NodeState<>(0, null, 0),
                new ObservableConcurrentTree.NodeState<>(0, null, 1)));
        assertRejectedUnchanged(tree, List.of(
                new ObservableConcurrentTree.NodeState<>(1, null, 1),
                new ObservableConcurrentTree.NodeState<>(2, null, 2)));
        assertRejectedUnchanged(tree, List.of(
                new ObservableConcurrentTree.NodeState<>(1, 2, 1),
                new ObservableConcurrentTree.NodeState<>(2, 1, 2)));
        assertRejectedUnchanged(tree, List.of(
                new ObservableConcurrentTree.NodeState<>(0, null, 0),
                new ObservableConcurrentTree.NodeState<>(1, 99, 1)));
        assertRejectedUnchanged(tree, List.of(
                new ObservableConcurrentTree.NodeState<>(0, null, 0),
                new ObservableConcurrentTree.NodeState<>(1, 1, 1)));
    }

    private static void assertRejectedUnchanged(
            ObservableConcurrentTree<Integer, Integer> tree,
            List<ObservableConcurrentTree.NodeState<Integer, Integer>> state) {
        var before = tree.snapshot();
        var version = tree.getVersion();
        assertThrows(RuntimeException.class, () -> tree.loadState(state));
        assertEquals(before, tree.snapshot());
        assertEquals(version, tree.getVersion());
    }
}
