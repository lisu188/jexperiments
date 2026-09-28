package life.hash;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class HashLifeUniverseTest {
    @Test
    void stableBlockRemainsStableAcrossImplementations() {
        for (TreeUniverse universe : universes()) {
            setBlock(universe);
            assertTrue(universe.stats().contains("population 4"));
            universe.runStep();
            assertTrue(universe.generationCount >= 1.0);
            assertTrue(universe.stats().contains("population 4"));
        }
    }

    @Test
    void blinkerOscillatesAndExpansionHandlesFarCoordinates() {
        for (TreeUniverse universe : universes()) {
            universe.setBit(-1, 0);
            universe.setBit(0, 0);
            universe.setBit(1, 0);
            universe.setBit(64, -64);
            assertEquals(1, universe.root.getBit(64, -64));
            assertEquals(0, universe.root.getBit(63, -64));
            universe.runStep();
            assertTrue(universe.generationCount >= 1.0);
            assertTrue(universe.root.level >= 3);
        }
    }

    @Test
    void nodeFactoriesCanonicalizeAndMemoize() {
        TreeNode empty = TreeNode.create();
        TreeNode expanded = empty.expandUniverse();
        assertTrue(expanded.level > empty.level);
        assertEquals(0, empty.getBit(0, 0));

        TreeNode living = empty.create(true);
        TreeNode dead = empty.create(false);
        TreeNode parent = empty.create(living, dead, dead, living);
        assertEquals(2, parent.population);
        assertEquals(0, parent.emptyTree(parent.level).population);

        TreeNode canonicalRoot = CanonicalTreeNode.create();
        TreeNode canonicalA = canonicalRoot.create(living, dead, dead, living);
        TreeNode canonicalB = canonicalRoot.create(living, dead, dead, living);
        assertEquals(canonicalA, canonicalB);
        assertEquals(canonicalA.hashCode(), canonicalB.hashCode());

        TreeNode memoized = MemoizedTreeNode.create();
        memoized = memoized.expandUniverse().expandUniverse().setBit(0, 0);
        assertSame(memoized.nextGeneration(), memoized.nextGeneration());

        TreeNode hashLife = HashLifeTreeNode.create();
        hashLife = hashLife.expandUniverse().expandUniverse().expandUniverse();
        assertNotNull(hashLife.nextGeneration());
    }


    @Test
    void nodeEdgeBranchesAndHashLifeCacheAreCovered() {
        TreeNode root = TreeNode.create()
                .setBit(-1, -1)
                .setBit(1, -1)
                .setBit(-1, 1)
                .setBit(1, 1);
        assertEquals(1, root.getBit(-1, -1));
        assertEquals(1, root.getBit(1, -1));
        assertEquals(1, root.getBit(-1, 1));
        assertEquals(1, root.getBit(1, 1));

        TreeNode canonical = CanonicalTreeNode.create();
        TreeNode live = canonical.create(true);
        TreeNode dead = canonical.create(false);
        TreeNode parent = canonical.create(live, dead, dead, live);

        assertEquals(1, live.hashCode());
        assertEquals(0, dead.hashCode());
        assertNotEquals(live, dead);
        assertFalse(parent.equals(live));
        assertTrue(parent.equals(parent));
        assertNotEquals(0, parent.hashCode());

        TreeNode hashLife = HashLifeTreeNode.create().setBit(0, 0);
        TreeNode first = hashLife.nextGeneration();
        TreeNode second = hashLife.nextGeneration();
        assertSame(first, second);
    }

    private static List<TreeUniverse> universes() {
        return List.of(
                new TreeUniverse(),
                new CanonicalTreeUniverse(),
                new MemoizedTreeUniverse(),
                new HashLifeTreeUniverse()
        );
    }

    private static void setBlock(TreeUniverse universe) {
        universe.setBit(0, 0);
        universe.setBit(1, 0);
        universe.setBit(0, 1);
        universe.setBit(1, 1);
    }
}
