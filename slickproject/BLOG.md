# Moving a Wizard around a Slick2D Tile Map

## Why this experiment exists

This experiment is a small Slick2D game prototype. It opens a window, loads a
Tiled map, animates a wizard sprite, and blocks movement based on tile
properties. The code is focused on the basic loop every 2D tile game needs:
load assets, read input, update position, check collisions, and render.

The module is not trying to be a full game. It is a concentrated test of how
Slick2D's `BasicGame`, `AppGameContainer`, animation support, and Tiled map
loader fit together.

## How it works

`WizardGame` extends `BasicGame`. The `main` method creates an
`AppGameContainer`, sets a 500 by 400 display mode, and starts the game.

During `init`, the game loads two-frame animations for each movement direction,
loads `data/grassmap.tmx`, and builds a boolean collision grid. The collision
grid is derived from the first tile layer: if a tile's `blocked` property is
`true`, the corresponding map coordinate is marked as blocked.

During `update`, the game reads arrow-key state. Pressing a key changes the
current animation and attempts to move the sprite by `delta * 0.1f`. Before
committing movement, the code checks whether the relevant edge of the sprite
would enter a blocked tile. During `render`, the map is drawn first and the
current sprite animation is drawn at the player position.

## What to notice

The experiment demonstrates a clean first version of tile collision. Collision
data lives in the map, not hardcoded in Java. The game reads tile properties
once during initialization and then uses a fast boolean lookup while updating
movement.

It also shows the shape of a frame-based game loop. Input is sampled on each
update, movement is scaled by `delta`, and rendering is separated from state
updates. Even though the game is small, those boundaries are the same ones used
in larger 2D games.

## Sanity check

The Gradle build keeps the original `org.slick2d:slick2d-core:1.0.2`
dependency and treats the Tiled map under `src/main/java/data` as a runtime
resource. There are no tests, and the entrypoint opens a GUI window.

The static pass found an asset issue: the repository includes
`data/grassmap.tmx`, but the Java code references sprite images such as
`data/wmg1_bk1.png`, `data/wmg1_fr1.png`, and related files that are not present
in the tracked file list. Running the game as-is may fail during asset loading
unless those files are supplied externally. The collision check also assumes
coordinates stay within the map bounds.

The repository-level Gradle wrapper build compiles this module successfully.

## Suggested next experiments

A practical next step would be adding the missing sprite assets or replacing
them with generated placeholder sprites. Another useful extension would be
clamping movement to map bounds and showing debug overlays for blocked tiles and
the player's collision point.
