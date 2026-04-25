# Exploring HashLife and Quadtree Universes

## Why this experiment exists

This module explores implementations of Conway's Game of Life using tree-shaped
universes. The ordinary Life update rule is simple, but large sparse patterns
quickly make naive grids expensive. HashLife and related quadtree approaches
attack that problem by representing space hierarchically and reusing repeated
substructures.

The experiment includes multiple universe and node variants: plain tree nodes,
canonical nodes, memoized nodes, and HashLife nodes. A small driver reads a Life
pattern in RLE format from standard input, instantiates the requested universe
implementation by class name, and runs it indefinitely.

## How it works

`Driver` is the launcher. It expects one argument: the class name of a
`UniverseInterface` implementation. It reads a subset of RLE, using `b` for
blank cells, `o` for live cells, `$` for line breaks, repeat counts, and `!` to
end the pattern. As live cells are parsed, the driver calls `setBit(x, y)`.

`TreeUniverse` stores the current state in a root `TreeNode`. When a live cell
falls outside the current root's coordinate range, the root expands. Before each
step, the universe expands until the border is empty enough for the next
generation to fit safely, then replaces the root with `root.nextGeneration()`.

`HashLifeTreeUniverse` extends that idea. Its root starts as a
`HashLifeTreeNode`, and each step advances by a power-of-two generation count
based on the root level. That is the core HashLife promise: for suitable
patterns, the algorithm can jump through time rather than simulate every single
generation.

## What to notice

The design is built around tree transformations and structural sharing.
Expanding the universe returns a larger root. Setting a bit returns a new root.
Computing the next generation returns another tree. Variants such as canonical
and memoized nodes can then focus on reusing equivalent subtrees or caching
expensive next-generation computations.

The driver also acts as a comparison harness. Because it loads the universe
implementation by class name, the same input pattern can be tried against
different internal representations.

## Sanity check

The module has no external dependencies and compiles through the shared Gradle
Java configuration. There are no tests. The driver is intentionally unbounded:
after reading a pattern, it prints stats and simulates forever until stopped.

The RLE parser supports only the subset described in the code comments. The
driver uses reflective `Class.forName(...).newInstance()`, an older style that
newer Java code would usually replace with an explicit constructor lookup.

The repository-level Gradle wrapper build compiles this module successfully.

## Suggested next experiments

The most useful follow-up would be adding a few small RLE fixtures and a bounded
driver mode that runs a fixed number of steps. That would make it possible to
compare the universe variants automatically. Another good extension would be
exporting population, root level, cache hits, and generation count as structured
metrics.
