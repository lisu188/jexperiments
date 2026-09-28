package life.hash;

import java.util.HashMap;

/**
 * CanonicalTreeNode extends TreeNode to canonicalize the nodes.
 * We use a HashMap so we can return the canonicalized node.
 */
class CanonicalTreeNode extends TreeNode {
    /**
     * Our canonicalization hashset.
     */
    private static final HashMap<CanonicalTreeNode, CanonicalTreeNode> hashMap = new HashMap<>();

    /**
     * Provide constructors.  The rest of the code manages the factory
     * interface mechanism used by TreeNode.  We use intern() in all
     * three create() functions to guarantee that all new nodes are
     * canonicalized.
     */
    CanonicalTreeNode(boolean alive) {
        super(alive);
    }

    CanonicalTreeNode(TreeNode nw, TreeNode ne, TreeNode sw, TreeNode se) {
        super(nw, ne, sw, se);
    }

    static TreeNode create() {
        return new CanonicalTreeNode(false).emptyTree(3);
    }

    /**
     * We need to provide a hashCode() and an equals() method to be
     * able to hash these objects.
     */
    public int hashCode() {
        int classHash = System.identityHashCode(getClass());
        if (level == 0)
            return 31 * classHash + (int) population;
        return 31 * classHash +
                System.identityHashCode(nw) +
                11 * System.identityHashCode(ne) +
                101 * System.identityHashCode(sw) +
                1007 * System.identityHashCode(se);
    }

    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null || getClass() != o.getClass())
            return false;
        TreeNode t = (TreeNode) o;
        if (level != t.level)
            return false;
        if (level == 0)
            return alive == t.alive;
        return nw == t.nw && ne == t.ne && sw == t.sw && se == t.se;
    }

    /**
     * Given a node, return the canonical one if it exists, or make it
     * the canonical one.
     */
    TreeNode intern() {
        TreeNode canon = hashMap.get(this);
        if (canon != null)
            return canon;

        hashMap.put(this, this);
        return this;
    }

    /**
     * We override the three create functions.
     */
    TreeNode create(boolean living) {
        return new CanonicalTreeNode(living).intern();
    }

    TreeNode create(TreeNode nw, TreeNode ne, TreeNode sw, TreeNode se) {
        return new CanonicalTreeNode(nw, ne, sw, se).intern();
    }
}
