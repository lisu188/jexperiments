# Exploring HashLife and Quadtree Universes

## Why this experiment exists

This experiment implements several versions of Conway's Game of Life on top of
immutable quadtrees. It starts with a plain tree universe, adds canonicalization,
adds memoization, and finishes with a HashLife-style node that can advance large
regions by large time steps. The code is adapted as a compact study of how data
structure choices change the cost of cellular automata.

For experienced Java developers, the interesting part is not the Life rules
themselves. It is the representation. A rectangular grid updates every cell at
every generation. A quadtree stores empty space compactly, shares equal
subtrees, and can cache future results for repeated patterns. HashLife pushes
that further by recursively computing the center of a region multiple
generations ahead. When patterns contain large empty areas or repeated
structure, this changes the problem dramatically.

The module is also a study in object identity. Canonical nodes make structurally
equal regions share the same object instance. Memoized nodes attach future
results to those canonical instances. Those two ideas together are what make
the recursive algorithm practical.

## Execution path

The driver reads an RLE Life pattern from standard input and picks a universe
implementation by class name with `Class.forName(classToUse).newInstance()`.
This reflective entrypoint lets the same input pattern run against
`life.hash.TreeUniverse`, `life.hash.CanonicalTreeUniverse`,
`life.hash.MemoizedTreeUniverse`, or `life.hash.HashLifeTreeUniverse`. The
driver then loops forever, printing stats and running steps.

```java
// The implementation class is selected at runtime for easy comparison.
String classToUse = args[0];
univ = (UniverseInterface) Class.forName(classToUse).newInstance();
readPattern();
```

The reflection is old-style Java, but it keeps the benchmark harness decoupled
from any single universe implementation.

The RLE reader handles the subset needed by the experiment. `b` advances over
dead cells, `o` calls `univ.setBit(x++, y)` for live cells, `$` moves to a new
row, and digits build repeat counts. The parser is intentionally direct so the
universe behavior remains the main topic.

## Core code walkthrough

Every tree node stores four children, its level, a leaf value, and cached
population:

```java
final TreeNode nw, ne, sw, se; // our children
final int level;           // distance to root
final boolean alive;       // if leaf node, are we alive or dead?
final double population;   // we cache the population here
```

Internal nodes are immutable. Updating a bit returns a new path of nodes from
the changed leaf back to the root, reusing all unchanged branches. Population
is computed when the node is constructed, which makes later emptiness checks
cheap.

The universe expands until the requested coordinate fits:

```java
while (true) {
    int maxCoordinate = 1 << (root.level - 1);
    if (-maxCoordinate <= x && x <= maxCoordinate - 1 &&
            -maxCoordinate <= y && y <= maxCoordinate - 1)
        break;
    root = root.expandUniverse();
}
```

This gives the universe an unbounded feel without allocating an infinite grid.
The root grows outward as needed, keeping the existing pattern centered.

At the bottom of the recursive simulation, a level-2 node can be advanced with
bit operations:

```java
private TreeNode oneGen(int bitmask) {
    if (bitmask == 0)
        return create(false);
    int self = (bitmask >> 5) & 1;
    bitmask &= 0x757;
```

The bitmask encodes the cell neighborhood. Counting the remaining bits applies
the Life rule: a cell lives when it has three neighbors, or when it is already
alive and has two neighbors.

The recursive `nextGeneration` method returns the center of the current node,
not another node of the same size. That is why the universe expands before each
step: enough empty border must exist so the center result still contains every
cell that can be affected by the current live population. This shrinking-center
property is central to quadtree Life implementations.

## Important implementation details

Canonicalization interns structurally equal nodes:

```java
TreeNode intern() {
    TreeNode canon = hashMap.get(this);
    if (canon != null)
        return canon;

    hashMap.put(this, this);
    return this;
}
```

The custom `hashCode` for internal nodes uses child identity, and `equals`
compares child identity as well. That works because children are themselves
canonical. The result is a directed acyclic graph of shared subtrees rather
than a pile of duplicate tree objects.

Memoization adds a `result` field to each canonical node. Its `nextGeneration`
method computes `super.nextGeneration()` only when `result == null`. Once a
canonical node has a computed next generation, every future reference to that
same node can reuse the result. HashLife builds on that by asking for larger
time jumps:

```java
// Cache the result on the canonical node itself.
TreeNode nextGeneration() {
    if (result == null)
        result = super.nextGeneration();
    return result;
}
```

```java
TreeNode n00 = nw.nextGeneration(),
        n01 = horizontalForward(nw, ne),
        n02 = ne.nextGeneration(),
        n10 = verticalForward(nw, sw),
        n11 = centerForward();
```

The full method builds nine advanced subnodes, combines them into four larger
advanced subnodes, and returns the advanced center. The recursive structure is
the algorithm.

`HashLifeTreeUniverse` has stricter expansion checks than a naive grid would
need because large time jumps require a quiet border for more than the next
single generation. The conditions involving `root.nw.se.se.population` and
similar paths are checking whether live cells are too close to the area that
will be discarded when the advanced center is returned.

## Runtime behavior and caveats

The driver warns in source comments that it runs forever. That is accurate. It
prints stats, advances, and repeats until the process is stopped. For HashLife,
`HashLifeTreeUniverse.runStep` computes `Math.pow(2.0, root.level - 2)`, assigns
`root = root.nextGeneration()`, and increments `generationCount` by that step
size, so one loop iteration can represent many Life generations.

```java
// HashLife advances by a power-of-two generation count.
double stepSize = Math.pow(2.0, root.level - 2);
root = root.nextGeneration();
generationCount += stepSize;
```

That annotated step is the visible runtime difference from the plain
one-generation tree universe.

There are practical limits. The canonicalization map is static and unbounded.
Long runs over complex patterns can consume memory. Generation count and
population are stored as `double`, which avoids some integer overflow pressure
but is not an exact arbitrary-precision model. The RLE parser is deliberately
small and not a full format implementation. The reflective `newInstance` call
is also old-style Java and assumes accessible no-argument constructors.

Despite those caveats, the module captures the core HashLife insight well:
immutable structure plus canonical sharing plus cached future states can turn
repeated spatial patterns into reusable computation.

The implementation is single-threaded. That keeps canonicalization and result
caching simple because the static map and node result fields do not need
concurrent access control. Parallelizing this style of HashLife would require
careful coordination around interning and memoized result publication.

The tree classes are package-private except for the universe entrypoints, so the
module presents universes as the public surface and keeps node mechanics local
to the package. That matches the experiment: clients set bits and run steps;
only the implementation needs to know how quadrants are shared.
That boundary keeps experimentation focused.

## Suggested next experiments

Add command-line controls for maximum steps and stats frequency so runs do not
have to be interrupted manually. Track canonical map size and cache hit rate.
Replace the static intern map with a bounded or weak-reference strategy and
measure the tradeoff. Add a complete RLE parser test suite. Finally, compare
the same pattern across plain, canonical, memoized, and HashLife universes with
timing and memory measurements.
