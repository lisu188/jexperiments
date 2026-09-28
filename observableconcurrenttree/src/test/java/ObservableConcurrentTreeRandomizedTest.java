import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.junit.jupiter.api.Test;

class ObservableConcurrentTreeRandomizedTest {
    private static final int SEEDS = 12;
    private static final int STEPS = 5_000;

    @Test
    void randomizedOperationsMatchReferenceModel() throws Exception {
        for (long seed = 1; seed <= SEEDS; seed++) {
            runSeed(seed * 0x9E3779B97F4A7C15L);
        }
    }

    private static void runSeed(long seed) throws Exception {
        var random = new Random(seed);
        var tree = new ObservableConcurrentTree<Integer, Integer>(0, 0);
        var reference = new ReferenceTree();
        var nextId = 1;

        for (int step = 0; step < STEPS; step++) {
            int op = random.nextInt(100);
            if (op < 35) {
                var parent = reference.randomId(random);
                int id = nextId++;
                int value = random.nextInt();
                tree.add(parent, id, value);
                reference.add(parent, id, value);
            } else if (op < 55) {
                var id = reference.randomId(random);
                int value = random.nextInt();
                tree.update(id, value);
                reference.update(id, value);
            } else if (op < 70 && reference.size() > 1) {
                var node = reference.randomNonRootId(random);
                var target = reference.randomMoveTarget(random, node);
                if (target != null) {
                    tree.move(node, target);
                    reference.move(node, target);
                }
            } else if (op < 82 && reference.size() > 1) {
                var node = reference.randomNonRootId(random);
                assertEquals(reference.removeSubtree(node), tree.removeSubtree(node),
                        "remove order mismatch seed=" + seed + " step=" + step);
            } else if (op < 90) {
                var state = reference.nodeStates();
                tree.loadState(state);
                reference.loaded();
            } else if (op < 95) {
                tree = ObservableConcurrentTree.fromJson(tree.toJson(), Integer.class, Integer.class);
            } else {
                tree.clear();
                reference.clear();
                tree.initialize(0, 0);
                reference.initialize();
            }

            if ((step & 31) == 0) {
                assertModel(seed, step, reference, tree);
            }
        }
        assertModel(seed, STEPS, reference, tree);
    }

    private static void assertModel(
            long seed,
            int step,
            ReferenceTree reference,
            ObservableConcurrentTree<Integer, Integer> tree) {
        assertEquals(reference.version, tree.getVersion(),
                () -> "version mismatch seed=" + seed + " step=" + step);
        assertEquals(reference.size(), tree.size(),
                () -> "size mismatch seed=" + seed + " step=" + step);
        assertEquals(reference.rootId, tree.getRootId(),
                () -> "root mismatch seed=" + seed + " step=" + step);

        var actual = tree.depthFirst().stream()
                .map(entry -> new View(entry.id(), entry.parentId(), entry.value(), entry.children()))
                .toList();
        assertEquals(reference.views(), actual,
                () -> "topology mismatch seed=" + seed + " step=" + step);
        TreeTestSupport.assertValid(tree);
    }

    private record View(int id, Integer parentId, int value, List<Integer> children) {
        private View {
            children = List.copyOf(children);
        }
    }

    private static final class ReferenceTree {
        private final Map<Integer, RefNode> nodes = new HashMap<>();
        private Integer rootId;
        private long version;

        private ReferenceTree() {
            initialize();
        }

        private void initialize() {
            var root = new RefNode(0, 0);
            nodes.put(0, root);
            rootId = 0;
            version++;
        }

        private void clear() {
            if (!nodes.isEmpty()) {
                nodes.clear();
                rootId = null;
                version++;
            }
        }

        private int size() {
            return nodes.size();
        }

        private void add(int parentId, int id, int value) {
            var parent = nodes.get(parentId);
            var node = new RefNode(id, value);
            node.parent = parent;
            nodes.put(id, node);
            parent.children.add(node);
            version++;
        }

        private void update(int id, int value) {
            nodes.get(id).value = value;
            version++;
        }

        private void move(int id, int newParentId) {
            var node = nodes.get(id);
            var newParent = nodes.get(newParentId);
            if (node.parent == newParent) {
                return;
            }
            node.parent.children.remove(node);
            newParent.children.add(node);
            node.parent = newParent;
            version++;
        }

        private List<Integer> removeSubtree(int id) {
            var node = nodes.get(id);
            node.parent.children.remove(node);
            var removed = collectIds(node);
            for (var removedId : removed) {
                nodes.remove(removedId);
            }
            version++;
            return removed;
        }

        private void loaded() {
            version++;
        }

        private int randomId(Random random) {
            return randomElement(random, new ArrayList<>(nodes.keySet()));
        }

        private int randomNonRootId(Random random) {
            var ids = new ArrayList<>(nodes.keySet());
            ids.remove(rootId);
            return randomElement(random, ids);
        }

        private Integer randomMoveTarget(Random random, int nodeId) {
            var candidates = new ArrayList<Integer>();
            for (var candidate : nodes.keySet()) {
                if (candidate != nodeId && !isDescendant(nodeId, candidate)) {
                    candidates.add(candidate);
                }
            }
            return candidates.isEmpty() ? null : randomElement(random, candidates);
        }

        private boolean isDescendant(int nodeId, int candidateId) {
            for (var current = nodes.get(candidateId); current != null; current = current.parent) {
                if (current.id == nodeId) {
                    return true;
                }
            }
            return false;
        }

        private ArrayList<ObservableConcurrentTree.NodeState<Integer, Integer>> nodeStates() {
            var result = new ArrayList<ObservableConcurrentTree.NodeState<Integer, Integer>>();
            for (var view : views()) {
                result.add(new ObservableConcurrentTree.NodeState<>(view.id(), view.parentId(), view.value()));
            }
            return result;
        }

        private List<View> views() {
            if (rootId == null) {
                return List.of();
            }
            var result = new ArrayList<View>(nodes.size());
            var stack = new ArrayDeque<RefNode>();
            stack.push(nodes.get(rootId));
            while (!stack.isEmpty()) {
                var node = stack.pop();
                result.add(new View(
                        node.id,
                        node.parent == null ? null : node.parent.id,
                        node.value,
                        node.children.stream().map(child -> child.id).toList()));
                for (int i = node.children.size() - 1; i >= 0; i--) {
                    stack.push(node.children.get(i));
                }
            }
            return result;
        }

        private static ArrayList<Integer> collectIds(RefNode root) {
            var result = new ArrayList<Integer>();
            var stack = new ArrayDeque<RefNode>();
            stack.push(root);
            while (!stack.isEmpty()) {
                var node = stack.pop();
                result.add(node.id);
                for (int i = node.children.size() - 1; i >= 0; i--) {
                    stack.push(node.children.get(i));
                }
            }
            return result;
        }

        private static int randomElement(Random random, List<Integer> values) {
            assertTrue(!values.isEmpty());
            return values.get(random.nextInt(values.size()));
        }

        private static final class RefNode {
            private final int id;
            private int value;
            private RefNode parent;
            private final ArrayList<RefNode> children = new ArrayList<>();

            private RefNode(int id, int value) {
                this.id = id;
                this.value = value;
            }
        }
    }
}
